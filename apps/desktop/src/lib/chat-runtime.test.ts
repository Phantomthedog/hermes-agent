import { describe, expect, it } from 'vitest'

import { coerceThinkingText, parseCommandDispatch } from './chat-runtime'

describe('coerceThinkingText', () => {
  it('strips streaming status prefixes from thinking deltas', () => {
    expect(coerceThinkingText("◉_◉ processing... checking the user's request")).toBe("checking the user's request")
    expect(coerceThinkingText('(¬‿¬) analyzing... reading the file')).toBe('reading the file')
  })

  it('drops empty thinking rewrite placeholder text', () => {
    expect(
      coerceThinkingText(
        "◉_◉ processing... I don't see any current rewritten thinking or next thinking to process. Could you provide the thinking content you'd like me to rewrite?"
      )
    ).toBe('')
  })
})

describe('parseCommandDispatch', () => {
  it('returns null for non-object input', () => {
    expect(parseCommandDispatch(null)).toBeNull()
    expect(parseCommandDispatch('string')).toBeNull()
    expect(parseCommandDispatch(42)).toBeNull()
    expect(parseCommandDispatch(undefined)).toBeNull()
  })

  it('parses exec type', () => {
    const result = parseCommandDispatch({ type: 'exec', output: 'done' })
    expect(result).toEqual({ type: 'exec', output: 'done' })
  })

  it('parses plugin type', () => {
    const result = parseCommandDispatch({ type: 'plugin', output: 'plugin output' })
    expect(result).toEqual({ type: 'plugin', output: 'plugin output' })
  })

  it('parses exec/plugin with missing output', () => {
    expect(parseCommandDispatch({ type: 'exec' })).toEqual({ type: 'exec', output: undefined })
    expect(parseCommandDispatch({ type: 'plugin' })).toEqual({ type: 'plugin', output: undefined })
  })

  it('parses alias type', () => {
    const result = parseCommandDispatch({ type: 'alias', target: 'help' })
    expect(result).toEqual({ type: 'alias', target: 'help' })
  })

  it('returns null for alias without target', () => {
    expect(parseCommandDispatch({ type: 'alias' })).toBeNull()
  })

  it('parses skill type', () => {
    const result = parseCommandDispatch({ type: 'skill', name: 'my-skill', message: 'Use this skill' })
    expect(result).toEqual({ type: 'skill', name: 'my-skill', message: 'Use this skill', notice: undefined })
  })

  it('parses skill type with optional notice', () => {
    const result = parseCommandDispatch({ type: 'skill', name: 'my-skill', message: 'Do X', notice: 'loading...' })
    expect(result).toEqual({ type: 'skill', name: 'my-skill', message: 'Do X', notice: 'loading...' })
  })

  it('parses skill type with missing message (still valid)', () => {
    const result = parseCommandDispatch({ type: 'skill', name: 'my-skill' })
    expect(result).toEqual({ type: 'skill', name: 'my-skill', message: undefined, notice: undefined })
  })

  it('returns null for skill without name', () => {
    expect(parseCommandDispatch({ type: 'skill', message: 'test' })).toBeNull()
  })

  it('parses send type', () => {
    const result = parseCommandDispatch({ type: 'send', message: 'queue this' })
    expect(result).toEqual({ type: 'send', message: 'queue this', notice: undefined })
  })

  it('parses send type with optional notice', () => {
    const result = parseCommandDispatch({ type: 'send', message: 'retry message', notice: 'retrying...' })
    expect(result).toEqual({ type: 'send', message: 'retry message', notice: 'retrying...' })
  })

  it('returns null for send without message', () => {
    expect(parseCommandDispatch({ type: 'send' })).toBeNull()
  })

  it('parses prefill type', () => {
    const result = parseCommandDispatch({ type: 'prefill', message: 'edit this text' })
    expect(result).toEqual({ type: 'prefill', message: 'edit this text', notice: undefined })
  })

  it('parses prefill type with optional notice', () => {
    const result = parseCommandDispatch({ type: 'prefill', message: 'edited text', notice: 'undid 1 turn' })
    expect(result).toEqual({ type: 'prefill', message: 'edited text', notice: 'undid 1 turn' })
  })

  it('returns null for prefill without message', () => {
    expect(parseCommandDispatch({ type: 'prefill' })).toBeNull()
  })

  it('returns null for unknown type', () => {
    expect(parseCommandDispatch({ type: 'unknown' })).toBeNull()
  })
})
