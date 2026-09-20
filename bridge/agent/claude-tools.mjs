import { createSdkMcpServer, tool } from '@anthropic-ai/claude-agent-sdk'

/** Translate one neutral local tool server into a Claude SDK MCP server. */
export function createClaudeToolServer(server, executor) {
  return createSdkMcpServer({
    name: server.name,
    version: server.version,
    instructions: server.instructions,
    alwaysLoad: server.alwaysLoad,
    tools: server.tools.map((definition) =>
      tool(
        definition.name,
        definition.description,
        definition.inputSchema,
        (args) => executor.execute(server.name, definition.name, args),
      ),
    ),
  })
}
