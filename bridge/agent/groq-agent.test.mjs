import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import { z } from 'zod'
import { createAgentSession } from './agent-router.mjs'
import { createGroqAgentSession } from './groq-agent.mjs'
import { defineLocalTool, defineLocalToolServer } from '../tools/registry.mjs'
import { displayServer } from '../panels.mjs'
import { uiServer } from '../ui.mjs'
import { visionServer } from '../vision.mjs'
import { chromeServer } from '../chrome.mjs'
import { exportFunctionTools } from '../tools/schema-export.mjs'
import { GROQ_SYSTEM_PROMPT } from './system-prompts.mjs'

const reply = (message, { ok = true, status = 200, headers = {} } = {}) => ({
  ok,
  status,
  headers: new Headers(headers),
  json: async () => ({ choices: [{ message }] }),
})

const errorReply = (status, payload, headers = {}) => ({
  ok: false,
  status,
  headers: new Headers(headers),
  json: async () => payload,
})

const prompt = (text = 'hello') =>
  (async function* messages() {
    yield { type: 'user', message: { role: 'user', content: text } }
  })()

const prompts = (...texts) =>
  (async function* messages() {
    for (const text of texts) {
      yield { type: 'user', message: { role: 'user', content: text } }
    }
  })()

const server = (tools) =>
  defineLocalToolServer({ name: 'local', version: '1.0.0', tools })

async function collect(session) {
  const events = []
  for await (const event of session) events.push(event)
  return events
}

test('routes groq and normalizes ready, final text, and completion events', async () => {
  const requests = []
  const fetchImpl = async (_url, init) => {
    requests.push(init)
    return reply({ role: 'assistant', content: 'Good evening.' })
  }
  const events = await collect(
    createAgentSession({
      provider: 'groq',
      prompt: prompt(),
      systemPrompt: 'You are JARVIS.',
      localToolServers: [server([])],
      apiKey: 'test-key',
      effort: 'medium',
      fetchImpl,
    }),
  )

  assert.deepEqual(events, [
    { type: 'ready', servers: ['local'] },
    { type: 'text_delta', text: 'Good evening.' },
    { type: 'turn_complete', text: 'Good evening.', costUsd: null },
  ])
  const body = JSON.parse(requests[0].body)
  assert.equal(body.model, 'openai/gpt-oss-20b')
  assert.equal(body.reasoning_effort, 'medium')
  assert.deepEqual(body.messages, [
    { role: 'system', content: 'You are JARVIS.' },
    { role: 'user', content: 'hello' },
  ])
  assert.equal(body.parallel_tool_calls, false)
})

test('executes a tool call, returns its result, and continues to final text', async () => {
  const bodies = []
  const responses = [
    reply({
      role: 'assistant',
      content: null,
      tool_calls: [
        {
          id: 'call-1',
          type: 'function',
          function: { name: 'lookup', arguments: '{"value":"arc"}' },
        },
      ],
    }),
    reply({ role: 'assistant', content: 'The result is reactor.' }),
  ]
  const fetchImpl = async (_url, init) => {
    bodies.push(JSON.parse(init.body))
    return responses.shift()
  }
  const local = server([
    defineLocalTool({
      name: 'lookup',
      description: 'Look something up.',
      inputSchema: { value: z.string() },
      access: 'read',
      execute: async ({ value }) => ({
        content: [{ type: 'text', text: `${value}: reactor` }],
      }),
    }),
  ])

  const events = await collect(
    createGroqAgentSession({
      prompt: prompt('find it'),
      systemPrompt: 'system',
      localToolServers: [local],
      apiKey: 'test-key',
      fetchImpl,
    }),
  )

  assert.deepEqual(events.slice(1), [
    { type: 'tool_start', id: 'call-1', name: 'mcp__local__lookup' },
    { type: 'tool_result', id: 'call-1', failed: false },
    { type: 'text_delta', text: 'The result is reactor.' },
    { type: 'turn_complete', text: 'The result is reactor.', costUsd: null },
  ])
  assert.equal(bodies[0].tools[0].function.name, 'lookup')
  assert.equal(bodies[1].messages.at(-1).role, 'tool')
  assert.deepEqual(JSON.parse(bodies[1].messages.at(-1).content), {
    ok: true,
    content: 'arc: reactor',
  })
})

test('shared executor blocks writes and reports the failed result to Groq', async () => {
  let executed = false
  const requests = []
  const fetchImpl = async (_url, init) => {
    requests.push(JSON.parse(init.body))
    return requests.length === 1
      ? reply({
          role: 'assistant',
          tool_calls: [
            {
              id: 'write-1',
              type: 'function',
              function: { name: 'change', arguments: '{}' },
            },
          ],
        })
      : reply({ role: 'assistant', content: 'Write access is disabled.' })
  }
  const local = server([
    defineLocalTool({
      name: 'change',
      description: 'Change state.',
      inputSchema: {},
      access: 'write',
      execute: async () => {
        executed = true
        return { content: [{ type: 'text', text: 'changed' }] }
      },
    }),
  ])

  const events = await collect(
    createGroqAgentSession({
      prompt: prompt(),
      localToolServers: [local],
      allowWrites: false,
      apiKey: 'test-key',
      fetchImpl,
    }),
  )

  assert.equal(executed, false)
  assert.deepEqual(events[2], {
    type: 'tool_result',
    id: 'write-1',
    failed: true,
  })
  assert.match(requests[1].messages.at(-1).content, /read-only mode/)
})

test('handles unknown tools, invalid arguments, and failed local tools', async () => {
  const calls = [
    ['missing', '{}'],
    ['fails', '{bad json'],
    ['fails', '{}'],
  ]
  const requests = []
  const fetchImpl = async (_url, init) => {
    requests.push(JSON.parse(init.body))
    if (calls.length) {
      const [name, args] = calls.shift()
      return reply({
        role: 'assistant',
        tool_calls: [
          {
            id: `call-${requests.length}`,
            type: 'function',
            function: { name, arguments: args },
          },
        ],
      })
    }
    return reply({ role: 'assistant', content: 'Recovered.' })
  }
  const local = server([
    defineLocalTool({
      name: 'fails',
      description: 'Returns an expected failure.',
      inputSchema: {},
      access: 'read',
      execute: async () => ({
        isError: true,
        content: [{ type: 'text', text: 'Expected failure.' }],
      }),
    }),
  ])

  const events = await collect(
    createGroqAgentSession({
      prompt: prompt(),
      localToolServers: [local],
      apiKey: 'test-key',
      fetchImpl,
    }),
  )

  assert.deepEqual(
    events.filter((event) => event.type === 'tool_result').map((event) => event.failed),
    [true, true, true],
  )
  assert.match(requests[1].messages.at(-1).content, /Unknown local tool/)
  assert.match(requests[2].messages.at(-1).content, /Invalid JSON arguments/)
  assert.match(requests[3].messages.at(-1).content, /Expected failure/)
  assert.equal(events.at(-1).type, 'turn_complete')
})

test('rolls back an unsupported image turn before the next user request', async () => {
  const requests = []
  const fetchImpl = async (_url, init) => {
    requests.push(JSON.parse(init.body))
    return requests.length === 1
      ? reply({
          role: 'assistant',
          tool_calls: [
            {
              id: 'image-1',
              type: 'function',
              function: { name: 'camera', arguments: '{}' },
            },
          ],
        })
      : reply({ role: 'assistant', content: 'Second turn is valid.' })
  }
  const local = server([
    defineLocalTool({
      name: 'camera',
      description: 'Get a frame.',
      inputSchema: {},
      access: 'read',
      execute: async () => ({
        content: [{ type: 'image', data: 'secret-image-data', mimeType: 'image/jpeg' }],
      }),
    }),
  ])

  const events = await collect(
    createGroqAgentSession({
      prompt: prompts('show me', 'continue'),
      systemPrompt: 'system',
      localToolServers: [local],
      apiKey: 'test-key',
      fetchImpl,
    }),
  )

  assert.deepEqual(events.find((event) => event.type === 'tool_result'), {
    type: 'tool_result',
    id: 'image-1',
    failed: true,
  })
  assert.equal(
    events.find((event) => event.type === 'turn_error')?.code,
    'groq_unsupported_tool_result',
  )
  assert.equal(events.at(-1).type, 'turn_complete')
  assert.deepEqual(requests[1].messages, [
    { role: 'system', content: 'system' },
    { role: 'user', content: 'continue' },
  ])
  assert.doesNotMatch(JSON.stringify({ events, requests }), /secret-image-data/)
})

test('preserves an executed write tool pair when the following request fails', async () => {
  let writeCount = 0
  const requests = []
  const fetchImpl = async (_url, init) => {
    requests.push(JSON.parse(init.body))
    if (requests.length === 1) {
      return reply({
        role: 'assistant',
        content: null,
        tool_calls: [
          {
            id: 'write-1',
            type: 'function',
            function: { name: 'change', arguments: '{"value":"updated"}' },
          },
        ],
      })
    }
    if (requests.length === 2) {
      return reply({}, { ok: false, status: 503 })
    }
    return reply({ role: 'assistant', content: 'Continued without repeating the write.' })
  }
  const local = server([
    defineLocalTool({
      name: 'change',
      description: 'Change state.',
      inputSchema: { value: z.string() },
      access: 'write',
      execute: async ({ value }) => {
        writeCount += 1
        return { content: [{ type: 'text', text: `changed to ${value}` }] }
      },
    }),
  ])

  const events = await collect(
    createGroqAgentSession({
      prompt: prompts('change it', 'continue'),
      localToolServers: [local],
      allowWrites: true,
      apiKey: 'test-key',
      fetchImpl,
    }),
  )

  assert.equal(writeCount, 1)
  assert.equal(events.find((event) => event.code === 'groq_http_503')?.type, 'turn_error')
  assert.equal(events.at(-1).type, 'turn_complete')
  assert.deepEqual(requests[2].messages.slice(0, 4), [
    { role: 'user', content: 'change it' },
    {
      role: 'assistant',
      content: null,
      tool_calls: [
        {
          id: 'write-1',
          type: 'function',
          function: { name: 'change', arguments: '{"value":"updated"}' },
        },
      ],
    },
    {
      role: 'tool',
      tool_call_id: 'write-1',
      name: 'change',
      content: JSON.stringify({ ok: true, content: 'changed to updated' }),
    },
    { role: 'user', content: 'continue' },
  ])
  const retainedHistory = requests[2].messages.slice(0, -1)
  const toolCallIds = retainedHistory.flatMap((message) =>
    (message.tool_calls ?? []).map((call) => call.id),
  )
  const toolResultIds = retainedHistory
    .filter((message) => message.role === 'tool')
    .map((message) => message.tool_call_id)
  assert.deepEqual(toolCallIds, toolResultIds)
})

test('rejects unsupported Groq reasoning effort values clearly', () => {
  assert.throws(
    () =>
      createGroqAgentSession({
        prompt: prompt(),
        apiKey: 'test-key',
        effort: 'max',
      }),
    /Groq reasoning effort must be low, medium, or high; received "max"/,
  )
})

test('normalizes HTTP and network errors without logging or leaking secrets', async (t) => {
  const secret = 'gsk_super_secret_value'
  const logged = []
  t.mock.method(console, 'error', (...args) => logged.push(args))

  const httpEvents = await collect(
    createGroqAgentSession({
      prompt: prompt(),
      apiKey: secret,
      fetchImpl: async () => reply({}, { ok: false, status: 401 }),
    }),
  )
  const networkEvents = await collect(
    createGroqAgentSession({
      prompt: prompt(),
      apiKey: secret,
      fetchImpl: async () => {
        throw new Error(`socket failed with ${secret}`)
      },
    }),
  )

  assert.equal(httpEvents.at(-1).code, 'groq_http_401')
  assert.equal(networkEvents.at(-1).code, 'groq_network_error')
  assert.doesNotMatch(JSON.stringify([httpEvents, networkEvents]), /gsk_super_secret/)
  assert.equal(logged.length, 0)
})

test('captures only allowlisted rate-limit headers on HTTP 429', async () => {
  const secret = 'gsk_super_secret_value'
  let bodyRead = false
  const responseBody = `upstream body containing ${secret}`
  const events = await collect(
    createGroqAgentSession({
      prompt: prompt(),
      apiKey: secret,
      fetchImpl: async () => ({
        ...reply({}, {
          ok: false,
          status: 429,
          headers: {
            'retry-after': '12',
            'x-ratelimit-limit-tokens': '8000',
            'x-ratelimit-remaining-tokens': '0',
            'x-ratelimit-reset-tokens': '11.4s',
            'x-ratelimit-limit-requests': '30',
            'x-ratelimit-remaining-requests': '29',
            'x-ratelimit-reset-requests': '2s',
            'x-request-id': 'not-allowlisted',
          },
        }),
        text: async () => {
          bodyRead = true
          return responseBody
        },
      }),
    }),
  )

  assert.equal(events.at(-1).code, 'groq_http_429')
  assert.equal(events.at(-1).message, 'Groq rejected the request (HTTP 429).')
  assert.deepEqual(JSON.parse(events.at(-1).details.replace(/^rate-limit /, '')), {
    'retry-after': '12',
    'x-ratelimit-limit-tokens': '8000',
    'x-ratelimit-remaining-tokens': '0',
    'x-ratelimit-reset-tokens': '11.4s',
    'x-ratelimit-limit-requests': '30',
    'x-ratelimit-remaining-requests': '29',
    'x-ratelimit-reset-requests': '2s',
  })
  assert.equal(bodyRead, false)
  assert.doesNotMatch(JSON.stringify(events), /gsk_super_secret|upstream body|x-request-id/)
})

test('logs only allowlisted HTTP 400 fields and content-free request metadata', async (t) => {
  const secret = 'gsk_do_not_log_this'
  const attempted = `{"secret":"${secret}","theme":"amber"}`
  const logs = []
  t.mock.method(console, 'log', (...args) => logs.push(args.join(' ')))
  const events = await collect(createGroqAgentSession({
    prompt: prompt(`private prompt ${secret}`),
    systemPrompt: `private system ${secret}`,
    localToolServers: [capabilityServer('jarvis_ui', [defineLocalTool({
      name: 'ui_theme', description: `private schema ${secret}`, inputSchema: {},
      access: 'local-ui', execute: async () => ({ content: [{ type: 'text', text: 'ok' }] }),
    })])],
    apiKey: secret,
    fetchImpl: async () => errorReply(400, {
      error: {
        type: 'invalid_request_error',
        code: 'tool_use_failed',
        message: 'Invalid tool call generated',
        failed_generation: {
          reason: 'Tool call arguments are not valid JSON',
          tool_call_id: 'call_safe_id',
          attempted_arguments: attempted,
          arbitrary_secret: secret,
        },
        arbitrary_upstream: secret,
      },
      messages: [secret],
      tools: [secret],
    }),
  }))

  assert.equal(logs.length, 1)
  assert.match(logs[0], /^\[jarvis\] groq request: initial-loader tools=1 bytes=\d+$/)
  assert.deepEqual(events.at(-1), {
    type: 'turn_error',
    code: 'groq_http_400',
    message: 'Groq rejected the request (HTTP 400).',
    details:
      'request initial-loader tools=1 bytes=' + logs[0].match(/bytes=(\d+)/)[1] +
      ' bad-request {"type":"invalid_request_error","code":"tool_use_failed","message":"Invalid tool call generated","failed_generation":{"reason":"Tool call arguments are not valid JSON","tool_call_id":"call_safe_id"}}',
  })
  assert.doesNotMatch(JSON.stringify({ logs, events }), /gsk_do_not|attempted_arguments|amber|arbitrary|private prompt|private schema|private system/)
})

test('labels loader, loaded-tools, and post-tool-result Groq rounds', async (t) => {
  const logs = []
  t.mock.method(console, 'log', (...args) => logs.push(args.join(' ')))
  const responses = [
    reply({ role: 'assistant', tool_calls: [toolCall('load-ui', 'jarvis_load_tools', { groups: ['ui'] })] }),
    reply({ role: 'assistant', tool_calls: [toolCall('theme-ui', 'ui_theme', { hue: 12 })] }),
    reply({ role: 'assistant', content: 'Done.' }),
  ]
  await collect(createGroqAgentSession({
    prompt: prompt('change the theme'),
    localToolServers: [capabilityServer('jarvis_ui', [defineLocalTool({
      name: 'ui_theme', description: 'Change theme.', inputSchema: { hue: z.number() },
      access: 'local-ui', execute: async () => ({ content: [{ type: 'text', text: 'changed' }] }),
    })])],
    apiKey: 'test-key',
    fetchImpl: async () => responses.shift(),
  }))

  assert.deepEqual(logs.map((line) => line.match(/request: (\S+) /)[1]), [
    'initial-loader', 'loaded-tools', 'post-tool-result',
  ])
  assert.ok(logs.every((line) => /tools=\d+ bytes=\d+$/.test(line)))
})

test('interrupt aborts the in-flight request and emits one generic error', async () => {
  let started
  const requestStarted = new Promise((resolve) => { started = resolve })
  const session = createGroqAgentSession({
    prompt: prompt(),
    apiKey: 'test-key',
    fetchImpl: async (_url, init) => {
      started()
      return new Promise((_resolve, reject) => {
        init.signal.addEventListener('abort', () => {
          const error = new Error('request aborted')
          error.name = 'AbortError'
          reject(error)
        }, { once: true })
      })
    },
  })
  const collecting = collect(session)
  await requestStarted
  session.interrupt()
  const events = await collecting

  assert.deepEqual(events.filter((event) => event.type === 'turn_error'), [{
    type: 'turn_error',
    code: 'groq_interrupted',
    message: 'The Groq turn was interrupted.',
  }])
})

test('requires a key without echoing environment contents', () => {
  assert.throws(
    () => createGroqAgentSession({ prompt: prompt(), apiKey: '' }),
    /^Error: GROQ_API_KEY is required/,
  )
})

const capabilityServer = (name, tools) =>
  defineLocalToolServer({ name, version: '1.0.0', tools })

const toolCall = (id, name, args = {}) => ({
  id,
  type: 'function',
  function: { name, arguments: JSON.stringify(args) },
})

test('production capability servers begin with only the small loader schema', async () => {
  const requests = []
  const display = capabilityServer('jarvis', [
    defineLocalTool({
      name: 'display',
      description: 'x'.repeat(10_000),
      inputSchema: {},
      access: 'local-ui',
      execute: async () => ({ content: [{ type: 'text', text: 'shown' }] }),
    }),
  ])
  await collect(createGroqAgentSession({
    prompt: prompt('hello'),
    systemPrompt: 'You are JARVIS.',
    localToolServers: [display],
    apiKey: 'test-key',
    fetchImpl: async (_url, init) => {
      requests.push(JSON.parse(init.body))
      return reply({ role: 'assistant', content: 'Hello.' })
    },
  }))

  assert.deepEqual(requests[0].tools.map((tool) => tool.function.name), [
    'jarvis_load_tools',
  ])
  assert.doesNotMatch(JSON.stringify(requests[0]), /xxxxxxxxxx/)
  assert.ok(Buffer.byteLength(requests[0].body ?? JSON.stringify(requests[0]), 'utf8') < 2_000)
})

test('real production display plus chrome schemas stay below the safe cap', async () => {
  const requests = []
  const responses = [
    reply({ role: 'assistant', tool_calls: [toolCall('load-real', 'jarvis_load_tools', { groups: ['display', 'chrome'] })] }),
    reply({ role: 'assistant', content: 'Ready.' }),
  ]
  await collect(createGroqAgentSession({
    prompt: prompt('browse and show it'),
    systemPrompt: 'You are JARVIS.',
    localToolServers: [
      displayServer(() => {}, () => {}),
      uiServer(() => {}),
      visionServer(async () => ({})),
      chromeServer({ allowWrites: false }),
    ],
    apiKey: 'test-key',
    fetchImpl: async (_url, init) => {
      requests.push(JSON.parse(init.body))
      return responses.shift()
    },
  }))
  assert.ok(Buffer.byteLength(JSON.stringify(requests[0].tools), 'utf8') < 1_000)
  assert.ok(Buffer.byteLength(JSON.stringify(requests[1].tools), 'utf8') < 22_000)
  assert.equal(requests[1].tools.some((tool) => tool.function.name === 'ui_theme'), false)
  assert.equal(requests[1].tools.some((tool) => tool.function.name === 'look'), false)
})

test('does not expose external MCP configuration to Groq', async () => {
  const requests = []
  await collect(createGroqAgentSession({
    prompt: prompt('hello'),
    localToolServers: [],
    mcpServers: {
      private_service: { url: 'https://example.invalid', authorization: 'secret' },
    },
    apiKey: 'test-key',
    fetchImpl: async (_url, init) => {
      requests.push(init.body)
      return reply({ role: 'assistant', content: 'Hello.' })
    },
  }))
  assert.doesNotMatch(requests[0], /private_service|example\.invalid|secret/)
})

test('loads chrome and executes its real tool with valid paired history', async () => {
  const requests = []
  const responses = [
    reply({ role: 'assistant', tool_calls: [toolCall('load-1', 'jarvis_load_tools', { groups: ['chrome'] })] }),
    reply({ role: 'assistant', tool_calls: [toolCall('read-1', 'chrome_read_page')] }),
    reply({ role: 'assistant', content: 'Page read.' }),
  ]
  const chrome = capabilityServer('jarvis_chrome', [
    defineLocalTool({
      name: 'chrome_read_page',
      description: 'Read the current page.',
      inputSchema: {},
      access: 'read',
      execute: async () => ({ content: [{ type: 'text', text: 'page' }] }),
    }),
  ])
  const events = await collect(createGroqAgentSession({
    prompt: prompt('read the page'),
    localToolServers: [chrome],
    apiKey: 'test-key',
    fetchImpl: async (_url, init) => {
      requests.push(JSON.parse(init.body))
      return responses.shift()
    },
  }))

  assert.deepEqual(requests[1].tools.map((tool) => tool.function.name), [
    'jarvis_load_tools',
    'chrome_read_page',
  ])
  assert.equal(events.some((event) => event.name === 'jarvis_load_tools'), false)
  assert.equal(events.find((event) => event.type === 'tool_start').name, 'mcp__jarvis_chrome__chrome_read_page')
  const history = requests[2].messages
  assert.deepEqual(
    history.flatMap((message) => (message.tool_calls ?? []).map((call) => call.id)),
    history.filter((message) => message.role === 'tool').map((message) => message.tool_call_id),
  )
})

test('loads display and vision schemas without emitting loader HUD events', async () => {
  const requests = []
  const responses = [
    reply({ role: 'assistant', tool_calls: [toolCall('load-1', 'jarvis_load_tools', { groups: ['display', 'vision'] })] }),
    reply({ role: 'assistant', content: 'Ready.' }),
  ]
  const basic = (serverName, name, access = 'read') =>
    capabilityServer(serverName, [defineLocalTool({
      name,
      description: name,
      inputSchema: {},
      access,
      execute: async () => ({ content: [{ type: 'text', text: 'ok' }] }),
    })])
  const events = await collect(createGroqAgentSession({
    prompt: prompt('show what you see'),
    localToolServers: [basic('jarvis', 'display', 'local-ui'), basic('jarvis_eyes', 'look')],
    apiKey: 'test-key',
    fetchImpl: async (_url, init) => {
      requests.push(JSON.parse(init.body))
      return responses.shift()
    },
  }))
  assert.deepEqual(requests[1].tools.map((tool) => tool.function.name), [
    'jarvis_load_tools', 'display', 'look',
  ])
  assert.equal(events.some((event) => event.type === 'tool_start'), false)
})

test('write blocking and image-result safety still apply after capability loading', async () => {
  let wrote = false
  const chrome = capabilityServer('jarvis_chrome', [
    defineLocalTool({
      name: 'chrome_click', description: 'Click.', inputSchema: {}, access: 'write',
      execute: async () => { wrote = true; return { content: [{ type: 'text', text: 'clicked' }] } },
    }),
  ])
  const vision = capabilityServer('jarvis_eyes', [
    defineLocalTool({
      name: 'look', description: 'Look.', inputSchema: {}, access: 'read',
      execute: async () => ({ content: [{ type: 'image', data: 'private-image', mimeType: 'image/jpeg' }] }),
    }),
  ])
  const responses = [
    reply({ role: 'assistant', tool_calls: [toolCall('load-c', 'jarvis_load_tools', { groups: ['chrome'] })] }),
    reply({ role: 'assistant', tool_calls: [toolCall('write-1', 'chrome_click')] }),
    reply({ role: 'assistant', content: 'Blocked.' }),
    reply({ role: 'assistant', tool_calls: [toolCall('load-v', 'jarvis_load_tools', { groups: ['vision'] })] }),
    reply({ role: 'assistant', tool_calls: [toolCall('image-1', 'look')] }),
  ]
  const events = await collect(createGroqAgentSession({
    prompt: prompts('click', 'look'),
    localToolServers: [chrome, vision],
    apiKey: 'test-key',
    fetchImpl: async () => responses.shift(),
  }))
  assert.equal(wrote, false)
  assert.equal(events.find((event) => event.id === 'write-1' && event.type === 'tool_result').failed, true)
  assert.equal(events.find((event) => event.type === 'turn_error').code, 'groq_unsupported_tool_result')
  assert.doesNotMatch(JSON.stringify(events), /private-image/)
})

test('capability state resets per turn and oversized combinations are rejected', async () => {
  const requests = []
  const huge = (serverName, name) => capabilityServer(serverName, [
    defineLocalTool({
      name, description: 'x'.repeat(12_000), inputSchema: {}, access: 'read',
      execute: async () => ({ content: [{ type: 'text', text: 'ok' }] }),
    }),
  ])
  const responses = [
    reply({ role: 'assistant', tool_calls: [toolCall('load-1', 'jarvis_load_tools', { groups: ['display', 'ui'] })] }),
    reply({ role: 'assistant', content: 'Use a smaller set.' }),
    reply({ role: 'assistant', content: 'Fresh turn.' }),
  ]
  await collect(createGroqAgentSession({
    prompt: prompts('load both', 'hello'),
    localToolServers: [huge('jarvis', 'display'), huge('jarvis_ui', 'ui_theme')],
    apiKey: 'test-key',
    fetchImpl: async (_url, init) => {
      requests.push(JSON.parse(init.body))
      return responses.shift()
    },
  }))
  assert.match(requests[1].messages.at(-1).content, /22000-byte safe limit/)
  assert.deepEqual(requests[1].tools.map((tool) => tool.function.name), ['jarvis_load_tools'])
  assert.deepEqual(requests[2].tools.map((tool) => tool.function.name), ['jarvis_load_tools'])
})

test('compact prompt and loader materially reduce a representative UI tool loop', async () => {
  const source = readFileSync(new URL('../server.mjs', import.meta.url), 'utf8')
    .replaceAll('\r\n', '\n')
  const prefix = 'const SYSTEM_PROMPT = `'
  const suffix = '`\n\n/**\n * ElevenLabs credentials'
  const start = source.indexOf(prefix)
  const end = source.indexOf(suffix, start + prefix.length)
  assert.ok(start >= 0 && end > start, 'full Claude prompt remains in server.mjs')
  const fullClaudePrompt = source.slice(start + prefix.length, end)

  const servers = [
    displayServer(() => {}, () => {}),
    uiServer(() => {}),
    visionServer(async () => ({})),
    chromeServer({ allowWrites: false }),
  ]
  const allToolSchemas = servers.flatMap(exportFunctionTools)
  const requests = []
  const responses = [
    reply({ role: 'assistant', tool_calls: [
      toolCall('load-ui', 'jarvis_load_tools', { groups: ['ui'] }),
    ] }),
    reply({ role: 'assistant', tool_calls: [
      toolCall('theme-ui', 'ui_theme', { hue: 12 }),
    ] }),
    reply({ role: 'assistant', content: 'The interface is amber, sir.' }),
  ]

  await collect(createGroqAgentSession({
    prompt: prompt('Make the interface amber.'),
    systemPrompt: GROQ_SYSTEM_PROMPT,
    localToolServers: servers,
    allowWrites: false,
    apiKey: 'test-key',
    fetchImpl: async (_url, init) => {
      requests.push(JSON.parse(init.body))
      return responses.shift()
    },
  }))

  assert.equal(requests.length, 3)
  assert.deepEqual(requests[0].tools.map((tool) => tool.function.name), [
    'jarvis_load_tools',
  ])
  const bytes = (value) => Buffer.byteLength(JSON.stringify(value), 'utf8')
  const legacyRequests = requests.map((request) => ({
    ...request,
    messages: request.messages.map((message) =>
      message.role === 'system'
        ? { ...message, content: fullClaudePrompt }
        : message,
    ),
    tools: allToolSchemas,
  }))
  const initialAfter = bytes(requests[0])
  const initialBefore = bytes(legacyRequests[0])
  const cumulativeAfter = requests.reduce((sum, request) => sum + bytes(request), 0)
  const cumulativeBefore = legacyRequests.reduce(
    (sum, request) => sum + bytes(request),
    0,
  )

  assert.ok(initialAfter < initialBefore * 0.5)
  assert.ok(cumulativeAfter < cumulativeBefore * 0.6)
})
