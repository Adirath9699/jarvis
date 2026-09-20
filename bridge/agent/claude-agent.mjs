import { query } from '@anthropic-ai/claude-agent-sdk'
import { createClaudeToolServer } from './claude-tools.mjs'
import { createLocalToolExecutor } from '../tools/registry.mjs'

const RESULT_FAILURES = {
  error_during_execution: 'The turn failed part way through.',
  error_max_turns: 'The turn ran too long and was stopped.',
  error_max_budget_usd: 'The budget for this turn ran out.',
  error_max_structured_output_retries: 'The answer could not be assembled.',
  default: 'The turn ended without an answer.',
}

/**
 * Translate one Claude SDK message into provider-neutral JARVIS events.
 * An array accommodates SDK messages that contain several tool blocks.
 */
export function normalizeClaudeMessage(message) {
  switch (message.type) {
    case 'stream_event': {
      const event = message.event
      if (
        event?.type === 'content_block_delta' &&
        event.delta?.type === 'text_delta' &&
        event.delta.text
      ) {
        return [{ type: 'text_delta', text: event.delta.text }]
      }
      if (
        event?.type === 'content_block_start' &&
        event.content_block?.type === 'tool_use'
      ) {
        return [
          {
            type: 'tool_start',
            id: event.content_block.id,
            name: event.content_block.name,
          },
        ]
      }
      return []
    }

    case 'assistant':
      return (message.content ?? message.message?.content ?? [])
        .filter((block) => block.type === 'tool_use')
        .map((block) => ({
          type: 'tool_start',
          id: block.id,
          name: block.name,
        }))

    case 'user': {
      const blocks = message.message?.content
      if (!Array.isArray(blocks)) return []
      return blocks
        .filter((block) => block?.type === 'tool_result')
        .map((block) => ({
          type: 'tool_result',
          id: block.tool_use_id,
          failed: block.is_error === true,
        }))
    }

    case 'result':
      return message.subtype === 'success'
        ? [
            {
              type: 'turn_complete',
              text: message.result ?? '',
              costUsd: message.total_cost_usd ?? null,
            },
          ]
        : [
            {
              type: 'turn_error',
              message:
                RESULT_FAILURES[message.subtype] ?? RESULT_FAILURES.default,
              code: message.subtype,
              details: message.errors,
            },
          ]

    case 'system':
      if (message.subtype !== 'init') return []
      return [
        {
          type: 'ready',
          servers: (message.mcp_servers ?? [])
            .filter(
              (server) =>
                server.status !== 'needs-auth' && server.status !== 'failed',
            )
            .map((server) => server.name),
        },
      ]

    default:
      return []
  }
}

function neutralClaudeSession(session) {
  return new Proxy(session, {
    get(target, property) {
      if (property === Symbol.asyncIterator) {
        return async function* iterateNeutralEvents() {
          for await (const message of target) {
            yield* normalizeClaudeMessage(message)
          }
        }
      }

      const value = Reflect.get(target, property, target)
      return typeof value === 'function' ? value.bind(target) : value
    },
  })
}

/**
 * First provider boundary for JARVIS's future multi-provider architecture.
 *
 * Keep Claude Agent SDK session creation here so the bridge can eventually
 * select another provider without owning that provider's construction API.
 * The returned iterable exposes only the neutral JARVIS event contract while
 * preserving the SDK session's lifecycle methods such as interrupt and close.
 */
export function createClaudeAgentSession({
  prompt,
  systemPrompt,
  mcpServers,
  localToolServers = [],
  allowWrites = false,
  cwd,
  model,
  effort,
  maxTurns,
  settingSources,
  permissionMode,
  includePartialMessages,
  canUseTool,
}) {
  const localToolExecutor = createLocalToolExecutor(localToolServers, {
    allowWrites,
  })
  const session = query({
    prompt,
    options: {
      systemPrompt,
      mcpServers: {
        ...mcpServers,
        ...Object.fromEntries(
          localToolServers.map((server) => [
            server.name,
            createClaudeToolServer(server, localToolExecutor),
          ]),
        ),
      },
      cwd,
      model,
      effort,
      maxTurns,
      settingSources,
      permissionMode,
      includePartialMessages,
      canUseTool,
    },
  })

  return neutralClaudeSession(session)
}
