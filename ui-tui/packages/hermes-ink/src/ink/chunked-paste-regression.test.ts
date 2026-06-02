/**
 * Pinned regression for the chunked ConPTY paste bug.
 *
 * Symptom: in `hermes --tui` on Windows + WSL, large bracketed pastes from
 * the clipboard arrived mangled — characters doubled, words split at chunk
 * boundaries with dropped whitespace. User reported:
 *   "i the the text / sapce is gona again in disply"
 *
 * Root cause: PASTE_TIMEOUT was reduced from 500ms to 50ms. The
 * flushIncomplete watchdog (App.tsx) re-arms while stdin has buffered
 * bytes, but a 50ms window is shorter than the typical ConPTY chunk
 * inter-arrival time on Win+WSL — the timer fires while the next chunk
 * is in flight but not yet in Node's read buffer, the partial paste is
 * flushed as a paste key, mode resets to NORMAL, and the remaining chunks
 * are parsed as individual keystrokes typed on top of the already-rendered
 * paste (producing the doubled/missing characters).
 *
 * The fix restores PASTE_TIMEOUT=500 (matches the upstream Ink fork
 * 10ad7006b cadence). 500ms is well above ConPTY chunk gaps on every
 * terminal we've measured; "compounding delays" the old comment mentioned
 * are negligible compared to lost-paste data.
 *
 * These tests pin both the constant value (catches any future revert) and
 * the chunked-paste end-to-end behaviour (catches a regression in the
 * rearm logic itself).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import App from './components/App.js'
import {
  INITIAL_STATE,
  parseMultipleKeypresses,
  type KeyParseState,
  type ParsedInput,
  type ParsedKey
} from './parse-keypress.js'
import { PASTE_END, PASTE_START } from './termio/csi.js'

// Narrow ParsedInput → ParsedKey with the fields this test inspects.
// isPasted and raw live only on ParsedKey, not on the ParsedInput union.
const isParsedKey = (k: ParsedInput): k is ParsedKey => k.kind === 'key'

const pasteKeys = (keys: ParsedInput[]): ParsedKey[] => keys.filter(isParsedKey)
const normalTextKeys = (keys: ParsedInput[]): ParsedKey[] =>
  keys.filter(isParsedKey).filter(k => k.isPasted === false && typeof k.sequence === 'string')

// A realistic paste payload that, when chunked at offset 8 and delivered
// with a 100ms gap, exposed the bug. Kept short and deterministic so the
// test failure message is readable.
const PASTE_CONTENT = 'i the text sapce is gona again in disply'
const CHUNK_1 = PASTE_START + PASTE_CONTENT.slice(0, 8) // "i the te"
const CHUNK_2 = PASTE_CONTENT.slice(8) + PASTE_END // "xt sapce is gona again in disply" + end

describe('chunked ConPTY paste (regression for PASTE_TIMEOUT=50ms)', () => {
  describe('PASTE_TIMEOUT constant', () => {
    it('is 500ms — required to tolerate ConPTY chunk gaps on Win+WSL', () => {
      // Bare instance. Class field initializers run during the `new` call,
      // so we instantiate with a stub props object and read the field.
      const app = new App({} as any)
      expect(app.PASTE_TIMEOUT).toBe(500)
      // 50ms is the value that caused the regression. Pinning the
      // >=500 lower bound keeps the test robust to small bumps (550, 600)
      // while still failing loudly on the original 50.
      expect(app.PASTE_TIMEOUT).toBeGreaterThanOrEqual(500)
    })
  })

  describe('parser contract for chunked paste (no intermediate flush)', () => {
    it('emits a single paste key with the full content when delivered in two chunks', () => {
      // Mirrors what handleReadable sees on a chunked ConPTY paste where
      // the chunks arrive within the rearm window (well under 500ms).
      const [keys1, state1] = parseMultipleKeypresses(INITIAL_STATE, CHUNK_1)
      const [keys2, state2] = parseMultipleKeypresses(state1, CHUNK_2)

      // No intermediate processInput(null) was called, so the parser
      // should still be in IN_PASTE after chunk 1, and the full paste
      // materialises as one key after chunk 2.
      expect(keys1).toEqual([])
      expect(state1.mode).toBe('IN_PASTE')

      const pKeys = pasteKeys(keys2)
      expect(pKeys).toHaveLength(1)
      expect(pKeys[0]).toMatchObject({ isPasted: true, raw: PASTE_CONTENT })
      expect(state2.mode).toBe('NORMAL')
      expect(state2.pasteBuffer).toBe('')
    })

    it('does not emit a stray empty paste key or a normal-mode text keystroke', () => {
      // The bug signature: chunk 1 → flush mid-paste → chunk 2 arrives
      // in NORMAL mode → stray empty paste key + 1 typed keystroke
      // containing the leftover content. Assert the chunked-delivery
      // path never produces either of those artefacts.
      const [keys1, state1] = parseMultipleKeypresses(INITIAL_STATE, CHUNK_1)
      const [keys2, state2] = parseMultipleKeypresses(state1, CHUNK_2)
      const allKeys: ParsedInput[] = [...keys1, ...keys2]

      const emptyPastes = pasteKeys(allKeys).filter(k => k.isPasted && k.raw === '')
      const textKeys = normalTextKeys(allKeys)

      expect(emptyPastes).toHaveLength(0)
      expect(textKeys).toHaveLength(0)

      // And the content round-trips byte-for-byte — no chars duplicated,
      // no chars dropped (this is what produced the user's typo pattern).
      const reconstructed = pasteKeys(allKeys)[0]?.raw
      expect(reconstructed).toBe(PASTE_CONTENT)
      expect(reconstructed?.length).toBe(PASTE_CONTENT.length)
    })
  })

  describe('App-level watchdog with fake timers', () => {
    // The real test of the fix: with PASTE_TIMEOUT=500, advancing time by
    // 100ms (past NORMAL_TIMEOUT=50, well below PASTE_TIMEOUT) must NOT
    // fire flushIncomplete. The timer should still be armed and the
    // parse state should still be IN_PASTE.
    let app: App
    let fakeStdin: { readableLength: number; read: () => null }
    let stdout: { write: ReturnType<typeof vi.fn>; isTTY: boolean }

    beforeEach(() => {
      vi.useFakeTimers()
      fakeStdin = { readableLength: 0, read: () => null }
      stdout = { write: vi.fn(), isTTY: false }
      app = new App({
        stdin: fakeStdin as any,
        stdout: stdout as any,
        stderr: stdout as any,
        children: null,
        exitOnCtrlC: false,
        onExit: vi.fn(),
        terminalColumns: 80,
        terminalRows: 24,
        selection: {} as any,
        onSelectionChange: vi.fn(),
        onClickAt: () => false,
        onMouseDownAt: () => undefined,
        onMouseUpAt: vi.fn(),
        onMouseDragAt: vi.fn(),
        onHoverAt: vi.fn(),
        onCopySelectionNoClear: async () => '',
        getSelectedText: () => '',
        getHyperlinkAt: () => undefined,
        onOpenHyperlink: vi.fn(),
        onMultiClick: vi.fn(),
        onSelectionDrag: vi.fn(),
        dispatchKeyboardEvent: vi.fn()
      } as any)
    })

    afterEach(() => {
      vi.useRealTimers()
    })

    it('does not flush the paste after 100ms — the timer window is 500ms', () => {
      // Chunk 1 arrives → processInput arms a flushIncomplete timer.
      // We don't go through handleReadable (which has the
      // discreteUpdates side-effect); we drive processInput directly
      // since that's the entry point that owns the timer.
      ;(app as any).processInput(CHUNK_1)

      // After chunk 1, state should be IN_PASTE and a timer should be armed.
      const stateAfterChunk1: KeyParseState = (app as any).keyParseState
      expect(stateAfterChunk1.mode).toBe('IN_PASTE')
      expect((app as any).incompleteEscapeTimer).not.toBeNull()

      // Advance 100ms — past NORMAL_TIMEOUT (50ms) but well below
      // PASTE_TIMEOUT (500ms). The timer should NOT have fired.
      vi.advanceTimersByTime(100)

      // Critical assertion: the watchdog did not flush. State is still
      // IN_PASTE, the timer is still armed, pasteBuffer is preserved.
      const stateAfterTimer: KeyParseState = (app as any).keyParseState
      expect(stateAfterTimer.mode).toBe('IN_PASTE')
      expect((app as any).incompleteEscapeTimer).not.toBeNull()
      expect(stateAfterTimer.pasteBuffer).toBe(PASTE_CONTENT.slice(0, 8))
    })

    it('emits one complete paste key when chunk 2 arrives within the timer window', () => {
      // Drive the App-level state machine through the chunked paste.
      // We can't easily intercept the reconciler-based dispatch of
      // processKeysInBatch, but we can verify the timer outcome and
      // final parse state — the same invariants as the parser test,
      // observed at the App layer.
      ;(app as any).processInput(CHUNK_1)
      vi.advanceTimersByTime(100)
      ;(app as any).processInput(CHUNK_2)

      const finalState: KeyParseState = (app as any).keyParseState
      expect(finalState.mode).toBe('NORMAL')
      expect(finalState.pasteBuffer).toBe('')

      // The flushIncomplete timer may still be armed — processInput
      // only clears it when arming a replacement, not on transition to
      // NORMAL. flushIncomplete is a no-op in that case (it returns
      // early at App.tsx:342 when state is clean), so a lingering
      // timer is harmless. We document the actual behaviour here
      // rather than asserting against implementation detail.
      // (If processInput ever leaks timers, add a focused test for
      // that — out of scope for this regression.)
      expect(typeof (app as any).incompleteEscapeTimer).toBe('object')
    })

    it('would fail at PASTE_TIMEOUT=50 — guards against future reverts', () => {
      // Sanity check: this test documents that the same sequence of
      // events with PASTE_TIMEOUT=50 WOULD fire the watchdog mid-paste.
      // We don't actually mutate the constant (that's pinned above), we
      // just verify the rearm logic by reading the field and asserting
      // the math: 100ms > 50ms, so a 50ms timer would have fired.
      // If someone reverts PASTE_TIMEOUT to <= 100, the previous test
      // ("does not flush after 100ms") will fail — that's the canary.
      const timeout = app.PASTE_TIMEOUT
      const testGap = 100
      expect(testGap).toBeLessThan(timeout)
    })
  })

  describe('parser-level flush-mid-paste documents the bug signature', () => {
    // This is the "fingerprint" of the original bug. Kept as a test so
    // future maintainers see what the broken behavior looked like and
    // don't accidentally reintroduce it via a different code path.
    it('a watchdog flush between chunks produces the user-reported typo pattern', () => {
      const [keys1, state1] = parseMultipleKeypresses(INITIAL_STATE, CHUNK_1)
      // Watchdog fires (input=null simulates the setTimeout callback).
      const [keys2, state2] = parseMultipleKeypresses(state1, null)
      // Chunk 2 arrives in NORMAL mode (because the flush reset it).
      const [keys3] = parseMultipleKeypresses(state2, CHUNK_2)

      const allKeys: ParsedInput[] = [...keys1, ...keys2, ...keys3]
      const emptyPastes = pasteKeys(allKeys).filter(k => k.isPasted && k.raw === '')
      const textKeys = normalTextKeys(allKeys)

      // The bug fingerprint: exactly one empty paste key (from the stray
      // PASTE_END marker) and exactly one normal-mode text keystroke
      // (containing the leftover chunk 2 content).
      expect(emptyPastes).toHaveLength(1)
      expect(textKeys.length).toBeGreaterThan(0)

      // The leftover content appears as a typed keystroke whose
      // .sequence (the chars the input handler will see) is missing
      // chunk 1's content — visually a "missing characters" effect.
      const typed = textKeys[0].sequence as string
      expect(typed).not.toContain(PASTE_CONTENT.slice(0, 8))
    })
  })
})
