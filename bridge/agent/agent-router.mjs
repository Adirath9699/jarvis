import { createClaudeAgentSession } from './claude-agent.mjs'

const PROVIDERS = new Map([['claude', createClaudeAgentSession]])

/**
 * Provider-neutral session factory. Provider adapters return their native
 * async iterable unchanged so server.mjs stays independent of construction.
 */
export function createAgentSession({ provider, ...sessionOptions }) {
  const createSession = PROVIDERS.get(provider)
  if (!createSession) {
    throw new Error(`Unsupported agent provider: ${String(provider)}`)
  }

  return createSession(sessionOptions)
}
