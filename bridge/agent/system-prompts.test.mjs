import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

import {
  GROQ_SYSTEM_PROMPT,
  systemPromptForProvider,
} from './system-prompts.mjs'

test('Claude keeps the original full system prompt by identity', () => {
  const source = readFileSync(new URL('../server.mjs', import.meta.url), 'utf8')
    .replaceAll('\r\n', '\n')
  const originalClaudePrompt = source.slice(
    source.indexOf('const SYSTEM_PROMPT = `') + 'const SYSTEM_PROMPT = `'.length,
    source.indexOf('`\n\n/**\n * ElevenLabs credentials'),
  )
  assert.ok(originalClaudePrompt.length > 8_000)
  assert.equal(
    systemPromptForProvider('claude', originalClaudePrompt),
    originalClaudePrompt,
  )
})

test('Groq receives the dedicated compact system prompt', () => {
  const originalClaudePrompt = 'x'.repeat(8_900)
  assert.equal(systemPromptForProvider('groq', originalClaudePrompt), GROQ_SYSTEM_PROMPT)
  assert.ok(Buffer.byteLength(GROQ_SYSTEM_PROMPT, 'utf8') < 2_500)
  assert.match(GROQ_SYSTEM_PROMPT, /jarvis_load_tools/)
  assert.match(GROQ_SYSTEM_PROMPT, /smallest necessary capability group/)
  assert.match(GROQ_SYSTEM_PROMPT, /Ordinary conversation needs no tools/)
})
