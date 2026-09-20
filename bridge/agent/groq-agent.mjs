import { createLocalToolExecutor } from '../tools/registry.mjs'
import { exportFunctionTools } from '../tools/schema-export.mjs'

const GROQ_CHAT_COMPLETIONS_URL =
  'https://api.groq.com/openai/v1/chat/completions'
const DEFAULT_GROQ_MODEL = 'openai/gpt-oss-20b'
const DEFAULT_MAX_TOOL_ROUNDS = 24
const TOOL_LOADER_NAME = 'jarvis_load_tools'
const MAX_LOADED_TOOL_BYTES = 22_000
const RATE_LIMIT_HEADERS = Object.freeze([
  'retry-after',
  'x-ratelimit-limit-tokens',
  'x-ratelimit-remaining-tokens',
  'x-ratelimit-reset-tokens',
  'x-ratelimit-limit-requests',
  'x-ratelimit-remaining-requests',
  'x-ratelimit-reset-requests',
])

const CAPABILITY_SERVERS = Object.freeze({
  display: 'jarvis',
  ui: 'jarvis_ui',
  vision: 'jarvis_eyes',
  chrome: 'jarvis_chrome',
})

const TOOL_LOADER = {
  type: 'function',
  function: {
    name: TOOL_LOADER_NAME,
    description:
      'Load only the JARVIS capability needed for this turn: display shows panels/blades, ui changes the HUD, vision uses the camera, and chrome controls the user browser.',
    parameters: {
      type: 'object',
      properties: {
        groups: {
          type: 'array',
          items: { type: 'string', enum: Object.keys(CAPABILITY_SERVERS) },
          minItems: 1,
          uniqueItems: true,
          description: 'Capability groups needed to complete the current turn.',
        },
      },
      required: ['groups'],
      additionalProperties: false,
    },
  },
}

const byteLength = (value) => Buffer.byteLength(JSON.stringify(value), 'utf8')

function badRequestDetails(payload) {
  const upstream = payload?.error
  if (!upstream || typeof upstream !== 'object' || Array.isArray(upstream)) return null
  const details = {}
  for (const field of ['type', 'code', 'message']) {
    if (typeof upstream[field] === 'string') details[field] = upstream[field]
  }
  const failed = upstream.failed_generation
  if (failed && typeof failed === 'object' && !Array.isArray(failed)) {
    const allowed = {}
    for (const field of ['reason', 'tool_call_id']) {
      if (typeof failed[field] === 'string') allowed[field] = failed[field]
    }
    if (Object.keys(allowed).length > 0) details.failed_generation = allowed
  }
  return Object.keys(details).length > 0 ? details : null
}

const textResult = (text, isError = false) => ({
  ...(isError ? { isError: true } : {}),
  content: [{ type: 'text', text }],
})

function userText(message) {
  const content = message?.message?.content ?? message?.content
  if (typeof content === 'string') return content
  if (!Array.isArray(content)) return String(content ?? '')
  return content
    .filter((block) => block?.type === 'text')
    .map((block) => block.text)
    .join('\n')
}

function toolResultText(result) {
  const content = result.content
    .filter((block) => block?.type === 'text')
    .map((block) => block.text)
    .join('\n')
  return JSON.stringify({
    ok: result.isError !== true,
    content: content || (result.isError ? 'Tool failed.' : 'Tool completed.'),
  })
}

function hasUnsupportedContent(result) {
  return result.content.some((block) => block?.type !== 'text')
}

function providerError(error) {
  if (error?.name === 'AbortError') {
    return {
      type: 'turn_error',
      code: 'groq_interrupted',
      message: 'The Groq turn was interrupted.',
    }
  }
  if (Number.isInteger(error?.status)) {
    return {
      type: 'turn_error',
      code: `groq_http_${error.status}`,
      message:
        error.status === 401 || error.status === 403
          ? 'Groq authentication failed. Check GROQ_API_KEY.'
          : `Groq rejected the request (HTTP ${error.status}).`,
      ...(error.status === 429 && error.rateLimit
        ? { details: `rate-limit ${JSON.stringify(error.rateLimit)}` }
        : {}),
      ...(error.status === 400
        ? {
            details: [
              `request ${error.requestPhase} tools=${error.toolCount} bytes=${error.requestBytes}`,
              ...(error.badRequest
                ? [`bad-request ${JSON.stringify(error.badRequest)}`]
                : []),
            ].join(' '),
          }
        : {}),
    }
  }
  return {
    type: 'turn_error',
    code: 'groq_network_error',
    message: 'Groq could not be reached. Check the network connection and try again.',
  }
}

async function requestCompletion({
  fetchImpl,
  apiKey,
  model,
  effort,
  messages,
  tools,
  requestPhase,
  signal,
}) {
  const requestBody = JSON.stringify({
    model,
    ...(effort ? { reasoning_effort: effort } : {}),
    messages,
    tools,
    tool_choice: 'auto',
    parallel_tool_calls: false,
  })
  const requestMeta = {
    requestPhase,
    toolCount: tools.length,
    requestBytes: Buffer.byteLength(requestBody, 'utf8'),
  }
  console.log(
    `[jarvis] groq request: ${requestMeta.requestPhase} ` +
      `tools=${requestMeta.toolCount} bytes=${requestMeta.requestBytes}`,
  )
  const response = await fetchImpl(GROQ_CHAT_COMPLETIONS_URL, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: requestBody,
    signal,
  })

  if (!response.ok) {
    const error = new Error(`Groq request failed with HTTP ${response.status}`)
    error.status = response.status
    Object.assign(error, requestMeta)
    if (response.status === 400) {
      try {
        error.badRequest = badRequestDetails(await response.json())
      } catch {
        // A non-JSON or unreadable body is intentionally not logged.
      }
    }
    if (response.status === 429) {
      const rateLimit = {}
      for (const name of RATE_LIMIT_HEADERS) {
        const value = response.headers?.get?.(name)
        if (value != null && value !== '') rateLimit[name] = value
      }
      if (Object.keys(rateLimit).length > 0) error.rateLimit = rateLimit
    }
    throw error
  }

  const payload = await response.json()
  const message = payload?.choices?.[0]?.message
  if (!message || typeof message !== 'object') {
    const error = new Error('Groq returned no assistant message')
    error.status = 502
    throw error
  }
  return message
}

/**
 * OpenAI-compatible Groq adapter for JARVIS-local tools.
 *
 * External MCP servers are intentionally not sent to Groq. They remain a
 * Claude-only capability until JARVIS has a provider-neutral remote-MCP policy.
 */
export function createGroqAgentSession({
  prompt,
  systemPrompt,
  localToolServers = [],
  allowWrites = false,
  apiKey = process.env.GROQ_API_KEY,
  model = process.env.GROQ_MODEL || DEFAULT_GROQ_MODEL,
  effort,
  fetchImpl = globalThis.fetch,
  maxTurns = DEFAULT_MAX_TOOL_ROUNDS,
}) {
  if (!apiKey) {
    throw new Error(
      'GROQ_API_KEY is required when JARVIS_PROVIDER=groq. No key is sent to the browser.',
    )
  }
  if (typeof fetchImpl !== 'function') {
    throw new Error('This Node runtime does not provide fetch for the Groq adapter.')
  }
  if (effort != null && !['low', 'medium', 'high'].includes(effort)) {
    throw new Error(
      `Groq reasoning effort must be low, medium, or high; received ${JSON.stringify(effort)}.`,
    )
  }

  const executor = createLocalToolExecutor(localToolServers, { allowWrites })
  const ownerByTool = new Map()
  for (const server of localToolServers) {
    for (const tool of server.tools) {
      if (ownerByTool.has(tool.name)) {
        throw new Error(`Groq local tool names must be unique: ${tool.name}`)
      }
      ownerByTool.set(tool.name, server.name)
    }
  }
  const serverByName = new Map(localToolServers.map((server) => [server.name, server]))
  const exportedByGroup = new Map()
  for (const [group, serverName] of Object.entries(CAPABILITY_SERVERS)) {
    const server = serverByName.get(serverName)
    if (server) exportedByGroup.set(group, exportFunctionTools(server))
  }
  const usesCapabilityLoader = exportedByGroup.size > 0
  const legacyTools = localToolServers.flatMap(exportFunctionTools)
  let controller = null
  let closed = false

  const session = {
    interrupt() {
      controller?.abort()
    },
    close() {
      closed = true
      controller?.abort()
    },
    async *[Symbol.asyncIterator]() {
      yield { type: 'ready', servers: localToolServers.map((server) => server.name) }
      const messages = systemPrompt
        ? [{ role: 'system', content: systemPrompt }]
        : []

      for await (const input of prompt) {
        if (closed) return
        controller = new AbortController()
        const turnStart = messages.length
        let turnSucceeded = false
        let hasCommittedToolResult = false
        const loadedGroups = new Set()
        let turnTools = usesCapabilityLoader ? [TOOL_LOADER] : legacyTools
        let requestPhase = usesCapabilityLoader ? 'initial-loader' : 'initial-tools'
        messages.push({ role: 'user', content: userText(input) })

        try {
          let completed = false
          for (let round = 0; round < maxTurns; round += 1) {
            const assistant = await requestCompletion({
              fetchImpl,
              apiKey,
              model,
              effort,
              messages,
              tools: turnTools,
              requestPhase,
              signal: controller.signal,
            })
            const toolCalls = Array.isArray(assistant.tool_calls)
              ? assistant.tool_calls
              : []

            if (toolCalls.length === 0) {
              const text = typeof assistant.content === 'string' ? assistant.content : ''
              messages.push({ role: 'assistant', content: text })
              if (text) yield { type: 'text_delta', text }
              turnSucceeded = true
              yield { type: 'turn_complete', text, costUsd: null }
              completed = true
              break
            }

            const assistantHistory = {
              role: 'assistant',
              content: assistant.content ?? null,
              tool_calls: toolCalls,
            }
            messages.push(assistantHistory)
            let committedToolCalls = 0

            for (const call of toolCalls) {
              const id = call?.id
              const name = call?.function?.name
              if (name === TOOL_LOADER_NAME) {
                let args
                try {
                  args = JSON.parse(call?.function?.arguments ?? '{}')
                } catch {
                  args = null
                }
                const requested = Array.isArray(args?.groups)
                  ? [...new Set(args.groups)]
                  : []
                const invalid = requested.filter(
                  (group) => !exportedByGroup.has(group),
                )
                const proposed = new Set([...loadedGroups, ...requested])
                const proposedTools = [
                  TOOL_LOADER,
                  ...[...proposed].flatMap(
                    (group) => exportedByGroup.get(group) ?? [],
                  ),
                ]
                const tooLarge = byteLength(proposedTools) > MAX_LOADED_TOOL_BYTES
                const accepted =
                  requested.length > 0 && invalid.length === 0 && !tooLarge
                const content = accepted
                  ? `Loaded JARVIS capabilities for this turn: ${requested.join(', ')}.`
                  : invalid.length
                    ? `Unknown or unavailable JARVIS capability: ${invalid.join(', ')}.`
                    : tooLarge
                      ? `Capability request rejected: the combined tool schemas exceed the ${MAX_LOADED_TOOL_BYTES}-byte safe limit. Load a smaller set.`
                      : 'Capability request rejected: groups must be a non-empty array.'
                if (accepted) {
                  for (const group of requested) loadedGroups.add(group)
                  turnTools = proposedTools
                  requestPhase = 'loaded-tools'
                }
                messages.push({
                  role: 'tool',
                  tool_call_id: id,
                  name,
                  content: JSON.stringify({ ok: accepted, content }),
                })
                committedToolCalls += 1
                continue
              }
              const serverName = ownerByTool.get(name)
              const loadedServerNames = new Set(
                [...loadedGroups].map((group) => CAPABILITY_SERVERS[group]),
              )
              const activeServerName =
                !usesCapabilityLoader || loadedServerNames.has(serverName)
                ? serverName
                : null
              const eventName = activeServerName
                ? `mcp__${serverName}__${name}`
                : name
              yield { type: 'tool_start', id, name: eventName }

              let args
              try {
                args = JSON.parse(call?.function?.arguments ?? '{}')
              } catch {
                args = null
              }
              const result =
                args && activeServerName
                  ? await executor.execute(activeServerName, name, args)
                  : textResult(
                      activeServerName
                        ? `Invalid JSON arguments for local tool: ${name}`
                        : `Unknown local tool (or capability not loaded): ${String(name)}`,
                      true,
                    )

              if (hasUnsupportedContent(result)) {
                assistantHistory.tool_calls = toolCalls.slice(0, committedToolCalls)
                yield { type: 'tool_result', id, failed: true }
                yield {
                  type: 'turn_error',
                  code: 'groq_unsupported_tool_result',
                  message:
                    `The local tool ${String(name)} returned image or multimodal content ` +
                    'that this Groq checkpoint cannot safely send back to the model.',
                }
                completed = true
                break
              }

              yield { type: 'tool_result', id, failed: result.isError === true }
              messages.push({
                role: 'tool',
                tool_call_id: id,
                name,
                content: toolResultText(result),
              })
              committedToolCalls += 1
              hasCommittedToolResult = true
              requestPhase = 'post-tool-result'
            }
            if (completed) break
          }

          if (!completed) {
            yield {
              type: 'turn_error',
              code: 'groq_max_turns',
              message: 'The Groq tool loop reached its turn limit.',
            }
          }
        } catch (error) {
          yield providerError(error)
        } finally {
          if (!turnSucceeded && !hasCommittedToolResult) messages.splice(turnStart)
          controller = null
        }
      }
    },
  }

  return session
}
