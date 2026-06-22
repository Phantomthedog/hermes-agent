import { describe, expect, it } from 'vitest'

import { nextPendingScrollDelta } from './ScrollBox.js'

describe('nextPendingScrollDelta', () => {
  it('accumulates same-direction scroll deltas', () => {
    expect(nextPendingScrollDelta(4, 3)).toBe(7)
    expect(nextPendingScrollDelta(-4, -3)).toBe(-7)
  })

  it('keeps fractional no-op deltas from changing pending scroll', () => {
    expect(nextPendingScrollDelta(4, 0.4)).toBe(4)
    expect(nextPendingScrollDelta(undefined, 0.4)).toBeUndefined()
  })

  it('replaces queued momentum when the user reverses direction', () => {
    expect(nextPendingScrollDelta(9, -1)).toBe(-1)
    expect(nextPendingScrollDelta(-9, 1)).toBe(1)
  })
})
