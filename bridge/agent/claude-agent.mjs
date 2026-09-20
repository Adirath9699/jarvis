import { query } from '@anthropic-ai/claude-agent-sdk'

/**
 * First provider boundary for JARVIS's future multi-provider architecture.
 *
 * Keep Claude Agent SDK session creation here so the bridge can eventually
 * select another provider without owning that provider's construction API.
 * The returned value is deliberately the SDK's original async iterable; event
 * translation remains in server.mjs until a later, explicitly scoped refactor.
 */
export function createClaudeAgentSession({
  prompt,
  systemPrompt,
  mcpServers,
  cwd,
  model,
  effort,
  maxTurns,
  settingSources,
  permissionMode,
  includePartialMessages,
  canUseTool,
}) {
  return query({
    prompt,
    options: {
      systemPrompt,
      mcpServers,
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
}
