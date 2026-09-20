import { createClaudeAgentSession } from './claude-agent.mjs'

const PROVIDERS = new Map([['claude', createClaudeAgentSession]])

/**
 * Provider-neutral session factory. Every adapter yields the small JARVIS event
 * contract, keeping server.mjs independent of provider APIs and event formats.
 */
export function createAgentSession({ provider, ...sessionOptions }) {
  const createSession = PROVIDERS.get(provider)
  if (!createSession) {
    throw new Error(`Unsupported agent provider: ${String(provider)}`)
  }

  return createSession(sessionOptions)
}
