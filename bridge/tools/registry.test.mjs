import assert from 'node:assert/strict'
import test from 'node:test'
import { z } from 'zod'
import {
  createLocalToolExecutor,
  defineLocalTool,
  defineLocalToolServer,
} from './registry.mjs'

const result = (text) => ({ content: [{ type: 'text', text }] })

const server = (tools) =>
  defineLocalToolServer({ name: 'test', version: '1.0.0', tools })

test('looks up, parses, and executes a local tool', async () => {
  let received
  const executor = createLocalToolExecutor([
    server([
      defineLocalTool({
        name: 'read',
        description: 'read',
        inputSchema: { value: z.string(), count: z.coerce.number().default(1) },
        access: 'read',
        execute: async (args) => {
          received = args
          return result('done')
        },
      }),
    ]),
  ])

  assert.equal(executor.lookup('test', 'read')?.access, 'read')
  assert.deepEqual(await executor.execute('test', 'read', { value: 'x' }), result('done'))
  assert.deepEqual(received, { value: 'x', count: 1 })
})

test('allows read and local-ui but denies write without write access', async () => {
  const called = []
  const tools = ['read', 'local-ui', 'write'].map((access) =>
    defineLocalTool({
      name: access,
      description: access,
      inputSchema: {},
      access,
      execute: async () => {
        called.push(access)
        return result(access)
      },
    }),
  )
  const executor = createLocalToolExecutor([server(tools)])

  assert.equal((await executor.execute('test', 'read', {})).isError, undefined)
  assert.equal((await executor.execute('test', 'local-ui', {})).isError, undefined)
  assert.equal((await executor.execute('test', 'write', {})).isError, true)
  assert.deepEqual(called, ['read', 'local-ui'])
})

test('allows write tools only when write access is enabled', async () => {
  const executor = createLocalToolExecutor(
    [
      server([
        defineLocalTool({
          name: 'write',
          description: 'write',
          inputSchema: {},
          access: 'write',
          execute: async () => result('written'),
        }),
      ]),
    ],
    { allowWrites: true },
  )

  assert.deepEqual(await executor.execute('test', 'write', {}), result('written'))
})

test('normalizes lookup, validation, and invalid result failures', async () => {
  const executor = createLocalToolExecutor([
    server([
      defineLocalTool({
        name: 'validates',
        description: 'validates',
        inputSchema: { value: z.string() },
        access: 'read',
        execute: async () => result('valid'),
      }),
      defineLocalTool({
        name: 'invalid-result',
        description: 'invalid result',
        inputSchema: {},
        access: 'read',
        execute: async () => undefined,
      }),
    ]),
  ])

  const missing = await executor.execute('test', 'missing', {})
  const invalid = await executor.execute('test', 'validates', { value: 1 })
  const invalidResult = await executor.execute('test', 'invalid-result', {})

  for (const failure of [missing, invalid, invalidResult]) {
    assert.equal(failure.isError, true)
    assert.equal(failure.content[0].type, 'text')
    assert.ok(failure.content[0].text.length > 0)
  }
  assert.match(invalid.content[0].text, /value/)
})

test('normalizes thrown exceptions without returning sensitive details', async (t) => {
  const sensitive = 'secret-token at C:\\private\\host.txt'
  const logged = []
  t.mock.method(console, 'error', (...args) => logged.push(args))
  const executor = createLocalToolExecutor([
    server([
      defineLocalTool({
        name: 'throws',
        description: 'throws',
        inputSchema: {},
        access: 'read',
        execute: async () => {
          throw new Error(sensitive)
        },
      }),
    ]),
  ])

  const failure = await executor.execute('test', 'throws', {})
  assert.deepEqual(failure, {
    isError: true,
    content: [{ type: 'text', text: 'Local tool failed unexpectedly.' }],
  })
  assert.doesNotMatch(JSON.stringify(failure), /secret-token|private|host\.txt/)
  assert.equal(logged.length, 1)
  assert.match(logged[0][0], /test\/throws/)
  assert.equal(logged[0][1].message, sensitive)
})

test('preserves explicit errors and image content blocks', async () => {
  const explicit = {
    isError: true,
    content: [{ type: 'text', text: 'Expected tool failure.' }],
  }
  const image = {
    content: [
      { type: 'text', text: 'Camera frame:' },
      { type: 'image', data: 'base64-data', mimeType: 'image/jpeg' },
    ],
  }
  const executor = createLocalToolExecutor([
    server([
      defineLocalTool({
        name: 'explicit-error',
        description: 'explicit error',
        inputSchema: {},
        access: 'read',
        execute: async () => explicit,
      }),
      defineLocalTool({
        name: 'image',
        description: 'image',
        inputSchema: {},
        access: 'read',
        execute: async () => image,
      }),
    ]),
  ])

  assert.deepEqual(await executor.execute('test', 'explicit-error', {}), explicit)
  assert.deepEqual(await executor.execute('test', 'image', {}), image)
})

test('rejects duplicate local server names', () => {
  assert.throws(
    () => createLocalToolExecutor([server([]), server([])]),
    /Duplicate local tool server: test/,
  )
})

test('rejects duplicate tools within one server', () => {
  const tool = defineLocalTool({
    name: 'same',
    description: 'same',
    inputSchema: {},
    access: 'read',
    execute: async () => result('ok'),
  })
  assert.throws(
    () => createLocalToolExecutor([server([tool, tool])]),
    /Duplicate local tool: test\/same/,
  )
})

test('rejects unknown access metadata at construction', () => {
  const tool = defineLocalTool({
    name: 'bad',
    description: 'bad',
    inputSchema: {},
    access: 'network',
    execute: async () => result('ok'),
  })
  assert.throws(
    () => createLocalToolExecutor([server([tool])]),
    /Invalid access metadata/,
  )
})
