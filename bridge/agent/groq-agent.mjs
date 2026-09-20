import { createLocalToolExecutor } from '../tools/registry.mjs'
import { exportFunctionTools } from '../tools/schema-export.mjs'

const GROQ_CHAT_COMPLETIONS_URL =
  'https://api.groq.com/openai/v1/chat/completions'
const DEFAULT_GROQ_MODEL = 'openai/gpt-oss-20b'
const DEFAULT_MAX_TOOL_ROUNDS = 24

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
  signal,
}) {
  const response = await fetchImpl(GROQ_CHAT_COMPLETIONS_URL, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model,
      ...(effort ? { reasoning_effort: effort } : {}),
      messages,
      tools,
      tool_choice: 'auto',
      parallel_tool_calls: false,
    }),
    signal,
  })

  if (!response.ok) {
    const error = new Error(`Groq request failed with HTTP ${response.status}`)
    error.status = response.status
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
  const tools = localToolServers.flatMap(exportFunctionTools)
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
              tools,
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
              const serverName = ownerByTool.get(name)
              const eventName = serverName
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
                args && serverName
                  ? await executor.execute(serverName, name, args)
                  : textResult(
                      serverName
                        ? `Invalid JSON arguments for local tool: ${name}`
                        : `Unknown local tool: ${String(name)}`,
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
