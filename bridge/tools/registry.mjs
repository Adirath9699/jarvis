import { z } from 'zod'

/** Provider-neutral local tool and server descriptors. */
export const defineLocalTool = (definition) => definition

export const defineLocalToolServer = (definition) => definition

const ACCESS_LEVELS = new Set(['read', 'local-ui', 'write'])

const textResult = (text, isError = false) => ({
  ...(isError ? { isError: true } : {}),
  content: [{ type: 'text', text }],
})

function validationMessage(error) {
  const issues = error?.issues
  if (!Array.isArray(issues) || issues.length === 0) return 'Invalid tool input.'

  const details = issues
    .map((issue) => {
      const path = issue.path?.length ? issue.path.join('.') : 'input'
      return `${path}: ${issue.message}`
    })
    .join('; ')
  return `Invalid tool input: ${details}`
}

function normalizeResult(result) {
  if (!result || typeof result !== 'object' || !Array.isArray(result.content)) {
    return textResult('The local tool returned an invalid result.', true)
  }

  return {
    ...result,
    ...(result.isError === true ? { isError: true } : {}),
    content: result.content,
  }
}

/**
 * Build the provider-independent execution boundary for JARVIS-local tools.
 * Schemas remain raw Zod shapes on the descriptors; they become objects only
 * here, at the point where untrusted provider arguments enter a handler.
 */
export function createLocalToolExecutor(localToolServers, { allowWrites = false } = {}) {
  const servers = new Set()
  const tools = new Map()

  for (const server of localToolServers) {
    if (servers.has(server.name)) {
      throw new Error(`Duplicate local tool server: ${server.name}`)
    }
    servers.add(server.name)

    for (const definition of server.tools) {
      const key = `${server.name}\0${definition.name}`
      if (tools.has(key)) {
        throw new Error(`Duplicate local tool: ${server.name}/${definition.name}`)
      }
      if (!ACCESS_LEVELS.has(definition.access)) {
        throw new Error(
          `Invalid access metadata for local tool ${server.name}/${definition.name}`,
        )
      }
      tools.set(key, {
        ...definition,
        serverName: server.name,
        schema: z.object(definition.inputSchema),
      })
    }
  }

  const lookup = (serverName, toolName) =>
    tools.get(`${serverName}\0${toolName}`) ?? null

  const isAuthorized = (definition) =>
    definition.access !== 'write' || allowWrites

  const execute = async (serverName, toolName, input) => {
    const definition = lookup(serverName, toolName)
    if (!definition) {
      return textResult(`Unknown local tool: ${serverName}/${toolName}`, true)
    }
    if (!isAuthorized(definition)) {
      return textResult(
        'Blocked: JARVIS is running in read-only mode and cannot take actions that change anything.',
        true,
      )
    }

    const parsed = definition.schema.safeParse(input)
    if (!parsed.success) return textResult(validationMessage(parsed.error), true)

    try {
      return normalizeResult(await definition.execute(parsed.data))
    } catch (error) {
      console.error(
        `[jarvis] local tool failed: ${serverName}/${toolName}`,
        error,
      )
      return textResult('Local tool failed unexpectedly.', true)
    }
  }

  return { execute, isAuthorized, lookup }
}
