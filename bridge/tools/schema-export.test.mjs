import assert from 'node:assert/strict'
import test from 'node:test'
import { z } from 'zod'
import { chromeServer } from '../chrome.mjs'
import { displayServer } from '../panels.mjs'
import { uiServer } from '../ui.mjs'
import { visionServer } from '../vision.mjs'
import { exportFunctionTools, exportToolInputSchema } from './schema-export.mjs'

test('exports required, optional, default, and coercion input semantics', () => {
  const schema = exportToolInputSchema({
    required: z.string(),
    optional: z.boolean().optional(),
    defaulted: z.coerce.number().default(3),
  })

  assert.deepEqual(schema, {
    type: 'object',
    properties: {
      required: { type: 'string' },
      optional: { type: 'boolean' },
      defaulted: { default: 3, type: 'number' },
    },
    required: ['required'],
  })
})

test('exports enums, unions, arrays, objects, and primitive types', () => {
  const schema = exportToolInputSchema({
    mode: z.enum(['fast', 'safe']),
    value: z.union([z.string(), z.number(), z.boolean()]),
    points: z.array(z.number()),
    options: z.object({ label: z.string(), enabled: z.boolean() }),
  })

  assert.deepEqual(schema.properties.mode, {
    type: 'string',
    enum: ['fast', 'safe'],
  })
  assert.deepEqual(schema.properties.value, {
    anyOf: [{ type: 'string' }, { type: 'number' }, { type: 'boolean' }],
  })
  assert.deepEqual(schema.properties.points, {
    type: 'array',
    items: { type: 'number' },
  })
  assert.deepEqual(schema.properties.options, {
    type: 'object',
    properties: {
      label: { type: 'string' },
      enabled: { type: 'boolean' },
    },
    required: ['label', 'enabled'],
  })
  assert.deepEqual(schema.required, ['mode', 'value', 'points', 'options'])
})

test('preserves field and tool descriptions in function definitions', () => {
  const [tool] = exportFunctionTools({
    tools: [
      {
        name: 'inspect',
        description: 'Inspect a thing.',
        inputSchema: {
          query: z.string().describe('What to inspect.'),
        },
      },
    ],
  })

  assert.deepEqual(tool, {
    type: 'function',
    function: {
      name: 'inspect',
      description: 'Inspect a thing.',
      parameters: {
        type: 'object',
        properties: {
          query: { type: 'string', description: 'What to inspect.' },
        },
        required: ['query'],
      },
    },
  })
})

test('exports catch fallbacks as optional provider inputs', () => {
  const schema = exportToolInputSchema({
    direction: z.enum(['up', 'down']).catch('down'),
    amount: z.union([z.number(), z.string()]).optional().catch(undefined),
  })

  assert.deepEqual(schema.properties.direction, {
    default: 'down',
    type: 'string',
    enum: ['up', 'down'],
  })
  assert.deepEqual(schema.properties.amount, {
    anyOf: [{ type: 'number' }, { type: 'string' }],
  })
  assert.equal(schema.required, undefined)
})

test('fails loudly for Zod constructs that JSON Schema cannot represent', () => {
  assert.throws(
    () => exportToolInputSchema({ createdAt: z.date() }),
    /Date cannot be represented in JSON Schema/,
  )
  assert.throws(
    () => exportToolInputSchema({ transformed: z.string().transform((value) => value.length) }),
    /Transforms cannot be represented in JSON Schema/,
  )
})

test('exports every current neutral local tool schema', () => {
  const noop = () => {}
  const readOnlyChrome = chromeServer({ allowWrites: false })
  const writeEnabledChrome = chromeServer({ allowWrites: true })
  const readOnlyChromeTools = exportFunctionTools(readOnlyChrome)
  const writeEnabledChromeTools = exportFunctionTools(writeEnabledChrome)
  const writeToolNames = [
    'chrome_click',
    'chrome_type',
    'chrome_key',
    'chrome_form_input',
    'chrome_new_tab',
    'chrome_close_tab',
  ]

  assert.equal(readOnlyChromeTools.length, 10)
  assert.equal(writeEnabledChromeTools.length, 16)
  assert.deepEqual(
    writeEnabledChromeTools
      .map((tool) => tool.function.name)
      .filter((name) => !readOnlyChromeTools.some((tool) => tool.function.name === name)),
    writeToolNames,
  )

  const servers = [
    writeEnabledChrome,
    displayServer(noop, noop),
    uiServer(noop),
    visionServer(noop),
  ]

  for (const server of servers) {
    const exported = exportFunctionTools(server)
    assert.equal(exported.length, server.tools.length)
    for (const tool of exported) {
      assert.equal(tool.type, 'function')
      assert.equal(tool.function.parameters.type, 'object')
      assert.ok(tool.function.name.length > 0)
    }
  }
})
