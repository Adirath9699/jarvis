import { z } from 'zod'

function assertNoTransforms(value, seen = new Set()) {
  if (!value || typeof value !== 'object' || seen.has(value)) return
  seen.add(value)

  if (value._zod?.def) {
    if (value._zod.def.type === 'transform') {
      throw new Error('Transforms cannot be represented in JSON Schema')
    }
    assertNoTransforms(value._zod.def, seen)
    return
  }

  for (const child of Object.values(value)) assertNoTransforms(child, seen)
}

/**
 * Convert a neutral tool's canonical raw Zod shape into the JSON Schema used
 * by OpenAI-style function/tool definitions. Runtime validation continues to
 * use the original shape in createLocalToolExecutor().
 */
export function exportToolInputSchema(rawShape) {
  const objectSchema = z.object(rawShape)
  assertNoTransforms(objectSchema)

  const schema = z.toJSONSchema(objectSchema, {
    io: 'input',
    target: 'draft-7',
    unrepresentable: 'throw',
  })

  // Function parameters are already known to be JSON Schema, so providers do
  // not need (and do not consistently accept) the draft declaration.
  const { $schema: _draft, ...parameters } = schema
  return parameters
}

/** Export the provider-neutral tools on a server as OpenAI-style functions. */
export function exportFunctionTools(server) {
  return server.tools.map((definition) => ({
    type: 'function',
    function: {
      name: definition.name,
      description: definition.description,
      parameters: exportToolInputSchema(definition.inputSchema),
    },
  }))
}
