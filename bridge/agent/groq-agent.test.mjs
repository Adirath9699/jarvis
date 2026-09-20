import assert from 'node:assert/strict'
import test from 'node:test'
import { z } from 'zod'
import { createAgentSession } from './agent-router.mjs'
import { createGroqAgentSession } from './groq-agent.mjs'
import { defineLocalTool, defineLocalToolServer } from '../tools/registry.mjs'

const reply = (message, { ok = true, status = 200 } = {}) => ({
  ok,
  status,
  json: async () => ({ choices: [{ message }] }),
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

test('requires a key without echoing environment contents', () => {
  assert.throws(
    () => createGroqAgentSession({ prompt: prompt(), apiKey: '' }),
    /^Error: GROQ_API_KEY is required/,
  )
})
