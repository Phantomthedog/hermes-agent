import { beforeEach, describe, expect, it } from 'vitest'

import { $backgroundStatusBySession, $statusItemsBySession, dismissBackgroundProcess, reconcileBackgroundProcesses } from './composer-status'
import { setSessionTodos } from './todos'
import { setWorkingSessionIds } from './session'

const SID = 'sess-1'

const running = (id: string, command = `cmd ${id}`) => ({ command, session_id: id, status: 'running' })

const exited = (id: string, exit_code = 0, command = `cmd ${id}`) => ({
  command,
  exit_code,
  session_id: id,
  status: 'exited'
})

const items = () => $backgroundStatusBySession.get()[SID] ?? []

describe('reconcileBackgroundProcesses', () => {
  beforeEach(() => {
    $backgroundStatusBySession.set({})
  })

  it('maps registry entries to status items', () => {
    reconcileBackgroundProcesses(SID, [running('a'), exited('b', 0), exited('c', 1)])

    expect(items().map(i => [i.id, i.state])).toEqual([
      ['a', 'running'],
      ['b', 'done'],
      ['c', 'failed']
    ])
    expect(items()[2]!.exitCode).toBe(1)
  })

  it('keeps row order stable when a process flips state or the snapshot reorders', () => {
    reconcileBackgroundProcesses(SID, [running('a'), running('b')])
    // Snapshot arrives reordered AND `a` has exited — rows must not move.
    reconcileBackgroundProcesses(SID, [running('b'), exited('a', 0)])

    expect(items().map(i => [i.id, i.state])).toEqual([
      ['a', 'done'],
      ['b', 'running']
    ])
  })

  it('appends new processes after existing rows', () => {
    reconcileBackgroundProcesses(SID, [running('a')])
    reconcileBackgroundProcesses(SID, [running('b'), running('a')])

    expect(items().map(i => i.id)).toEqual(['a', 'b'])
  })

  it('preserves object identity for unchanged rows (memo stability)', () => {
    reconcileBackgroundProcesses(SID, [running('a'), running('b')])
    const [a1] = items()

    reconcileBackgroundProcesses(SID, [running('a'), exited('b', 0)])
    const [a2, b2] = items()

    expect(a2).toBe(a1)
    expect(b2!.state).toBe('done')
  })

  it('is a no-op store write when nothing changed', () => {
    reconcileBackgroundProcesses(SID, [running('a')])
    const before = $backgroundStatusBySession.get()

    reconcileBackgroundProcesses(SID, [running('a')])

    expect($backgroundStatusBySession.get()).toBe(before)
  })

  it('never resurrects a dismissed process while the registry still reports it', () => {
    reconcileBackgroundProcesses(SID, [exited('a', 0), running('b')])
    dismissBackgroundProcess(SID, 'a')

    reconcileBackgroundProcesses(SID, [exited('a', 0), running('b')])

    expect(items().map(i => i.id)).toEqual(['b'])
  })

  it('forgets a dismissal once the registry prunes the process', () => {
    reconcileBackgroundProcesses(SID, [exited('a', 0)])
    dismissBackgroundProcess(SID, 'a')

    // Registry pruned it…
    reconcileBackgroundProcesses(SID, [])
    // …so a future process reusing the id (new spawn) shows again.
    reconcileBackgroundProcesses(SID, [running('a')])

    expect(items().map(i => i.id)).toEqual(['a'])
  })

  it('drops the session key entirely when the last row goes away', () => {
    reconcileBackgroundProcesses(SID, [running('a')])
    reconcileBackgroundProcesses(SID, [])
    expect($backgroundStatusBySession.get()).toEqual({})
  })
})

describe('$statusItemsBySession todo isLive', () => {
  const SID = 'sess-todo-live'

  beforeEach(() => {
    $backgroundStatusBySession.set({})
    setSessionTodos(SID, [])
    setWorkingSessionIds([])
  })

  it('marks todo items as live when session is in workingSessionIds', () => {
    setWorkingSessionIds([SID])
    setSessionTodos(SID, [
      { content: 'Step one', id: 's1', status: 'in_progress' },
      { content: 'Step two', id: 's2', status: 'pending' }
    ])

    const items = $statusItemsBySession.get()[SID] ?? []
    expect(items).toHaveLength(2)
    expect(items.every(i => i.isLive === true)).toBe(true)
  })

  it('marks todo items as archived when session is NOT in workingSessionIds', () => {
    setWorkingSessionIds([])
    setSessionTodos(SID, [
      { content: 'Step one', id: 's1', status: 'completed' },
      { content: 'Step two', id: 's2', status: 'in_progress' }
    ])

    const items = $statusItemsBySession.get()[SID] ?? []
    expect(items).toHaveLength(2)
    expect(items.every(i => i.isLive === false)).toBe(true)
  })

  it('transitions isLive from true to false when session stops working', () => {
    setWorkingSessionIds([SID])
    setSessionTodos(SID, [
      { content: 'Work', id: 'w1', status: 'in_progress' }
    ])

    const itemsBefore = $statusItemsBySession.get()[SID] ?? []
    expect(itemsBefore[0]!.isLive).toBe(true)

    // Session finishes
    setWorkingSessionIds([])

    const itemsAfter = $statusItemsBySession.get()[SID] ?? []
    expect(itemsAfter[0]!.isLive).toBe(false)
  })

  it('non-todo items do not have isLive set', () => {
    setWorkingSessionIds([SID])
    reconcileBackgroundProcesses(SID, [running('bg-1')])

    const items = $statusItemsBySession.get()[SID] ?? []
    const bgItem = items.find(i => i.type === 'background')
    expect(bgItem).toBeDefined()
    expect(bgItem!.isLive).toBeUndefined()
  })
})
