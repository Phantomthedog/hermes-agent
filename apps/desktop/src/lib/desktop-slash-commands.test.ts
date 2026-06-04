import { describe, expect, it } from 'vitest'

import {
  desktopSkinSlashCompletions,
  desktopSlashDescription,
  desktopSlashUnavailableMessage,
  filterDesktopCommandsCatalog,
  isDesktopSlashCommand,
  isDesktopSlashSuggestion
} from './desktop-slash-commands'

describe('desktop slash command curation', () => {
  it('keeps core desktop chat commands in suggestions', () => {
    expect(isDesktopSlashSuggestion('/new')).toBe(true)
    expect(isDesktopSlashSuggestion('/branch')).toBe(true)
    expect(isDesktopSlashSuggestion('/skin')).toBe(true)
    expect(isDesktopSlashSuggestion('/usage')).toBe(true)
    expect(isDesktopSlashSuggestion('/yolo')).toBe(true)
    expect(isDesktopSlashCommand('/yolo')).toBe(true)
  })

  it('surfaces skill and quick commands (extensions) in suggestions and lets them run', () => {
    expect(isDesktopSlashSuggestion('/my-skill')).toBe(true)
    expect(isDesktopSlashSuggestion('/gif-search')).toBe(true)
    expect(isDesktopSlashCommand('/my-skill')).toBe(true)
  })

  it('hides terminal, messaging, and dedicated-UI commands from suggestions', () => {
    expect(isDesktopSlashSuggestion('/clear')).toBe(false)
    expect(isDesktopSlashSuggestion('/compact')).toBe(false)
    expect(isDesktopSlashSuggestion('/redraw')).toBe(false)
    expect(isDesktopSlashSuggestion('/approve')).toBe(false)
    expect(isDesktopSlashSuggestion('/model')).toBe(false)
    expect(isDesktopSlashSuggestion('/skills')).toBe(false)
    expect(isDesktopSlashSuggestion('/voice')).toBe(false)
    expect(isDesktopSlashSuggestion('/curator')).toBe(false)
  })

  it('allows aliases to execute without cluttering the popover', () => {
    expect(isDesktopSlashSuggestion('/reset')).toBe(false)
    expect(isDesktopSlashCommand('/reset')).toBe(true)
  })

  it('filters built-in catalog noise but keeps skill / quick-command extensions', () => {
    const filtered = filterDesktopCommandsCatalog({
      categories: [
        {
          name: 'Session',
          pairs: [
            ['/new', 'Start a new session'],
            ['/clear', 'Clear terminal screen']
          ]
        },
        {
          name: 'User commands',
          pairs: [['/ship-it', 'Run release checklist']]
        }
      ],
      pairs: [
        ['/new', 'Start a new session'],
        ['/model', 'Switch model'],
        ['/ship-it', 'Run release checklist']
      ],
      skill_count: 2
    })

    expect(filtered.categories).toEqual([
      { name: 'Session', pairs: [['/new', 'Start a new desktop chat']] },
      { name: 'User commands', pairs: [['/ship-it', 'Run release checklist']] }
    ])
    expect(filtered.pairs).toEqual([
      ['/new', 'Start a new desktop chat'],
      ['/ship-it', 'Run release checklist']
    ])
    expect(filtered.skill_count).toBe(2)
  })

  it('keeps skills in catalog when typed prefix matches', () => {
    const filtered = filterDesktopCommandsCatalog({
      categories: [
        { name: 'Skills', pairs: [['/android-tv-adb', 'Android TV ADB'], ['/cloudflare-dns', 'Cloudflare DNS']] }
      ],
      pairs: [
        ['/android-tv-adb', 'Android TV ADB management'],
        ['/cloudflare-dns', 'Cloudflare DNS records'],
        ['/new', 'Start a new session']
      ],
      skill_count: 2
    })

    // Skills are always in the filtered catalog regardless of prefix
    // (prefix filtering is done client-side in use-slash-completions)
    expect(filtered.pairs!.find(([cmd]) => cmd === '/android-tv-adb')).toBeDefined()
    expect(filtered.pairs!.find(([cmd]) => cmd === '/cloudflare-dns')).toBeDefined()
    expect(filtered.pairs!.find(([cmd]) => cmd === '/new')).toBeDefined()
  })

  it('client-side prefix matching finds skills by partial name', () => {
    const catalog = filterDesktopCommandsCatalog({
      pairs: [
        ['/android-tv-adb', 'Full lifecycle ADB management'],
        ['/cloudflare-dns', 'DNS records via Cloudflare'],
        ['/docker-management', 'Docker containers'],
        ['/new', 'New chat session'],
        ['/help', 'Show help']
      ],
      skill_count: 3
    })

    const pairs = catalog.pairs ?? []

    // Simulating what use-slash-completions does for typed query
    const q = 'and'
    const matches = pairs.filter(
      ([cmd]) =>
        cmd.toLowerCase().startsWith(`/${q}`) ||
        cmd.toLowerCase().slice(1).startsWith(q)
    )
    expect(matches.map(([c]) => c)).toEqual(['/android-tv-adb'])

    const q2 = 'clo'
    const matches2 = pairs.filter(
      ([cmd]) =>
        cmd.toLowerCase().startsWith(`/${q2}`) ||
        cmd.toLowerCase().slice(1).startsWith(q2)
    )
    expect(matches2.map(([c]) => c)).toEqual(['/cloudflare-dns'])

    const q3 = 'doc'
    const matches3 = pairs.filter(
      ([cmd]) =>
        cmd.toLowerCase().startsWith(`/${q3}`) ||
        cmd.toLowerCase().slice(1).startsWith(q3)
    )
    expect(matches3.map(([c]) => c)).toEqual(['/docker-management'])
  })

  it('client-side prefix matching returns empty for non-matching prefix', () => {
    const catalog = filterDesktopCommandsCatalog({
      pairs: [
        ['/android-tv-adb', 'ADB'],
        ['/cloudflare-dns', 'DNS']
      ],
      skill_count: 2
    })

    const pairs = catalog.pairs ?? []
    const q = 'zzz'
    const matches = pairs.filter(
      ([cmd]) =>
        cmd.toLowerCase().startsWith(`/${q}`) ||
        cmd.toLowerCase().slice(1).startsWith(q)
    )
    expect(matches).toEqual([])
  })

  it('uses desktop-specific labels for commands with different UI behavior', () => {
    expect(desktopSlashDescription('/branch', 'Branch the current session')).toBe(
      'Branch the latest message into a new chat'
    )
    expect(desktopSlashDescription('/skin', 'Show or change the display skin/theme')).toBe(
      'Switch desktop theme or cycle to the next one'
    )
  })

  it('builds /skin completions from desktop themes', () => {
    const completions = desktopSkinSlashCompletions(
      [
        { name: 'mono', label: 'Mono', description: 'Clean grayscale' },
        { name: 'midnight', label: 'Midnight', description: 'Deep blue' },
        { name: 'slate', label: 'Slate', description: 'Cool slate blue' }
      ],
      'mono',
      'm'
    )

    expect(completions).toEqual([
      {
        text: '/skin mono',
        display: '/skin mono',
        meta: 'Mono (current) - Clean grayscale'
      },
      {
        text: '/skin midnight',
        display: '/skin midnight',
        meta: 'Midnight - Deep blue'
      }
    ])
  })

  it('explains known commands that desktop owns elsewhere', () => {
    expect(desktopSlashUnavailableMessage('/model sonnet')).toContain('model picker')
    expect(desktopSlashUnavailableMessage('/skills')).toContain('desktop sidebar')
    expect(desktopSlashUnavailableMessage('/clear')).toContain('terminal interface')
  })

  it('allows skill commands like /plan or /xhs-image to pass through isDesktopSlashCommand', () => {
    // Skill commands not in DESKTOP_COMMANDS, DESKTOP_ALIASES, or BLOCKED_COMMANDS
    expect(isDesktopSlashCommand('/plan')).toBe(true)
    expect(isDesktopSlashCommand('/xhs-image')).toBe(true)
    expect(isDesktopSlashCommand('/hermes-agent-dev')).toBe(true)
    expect(isDesktopSlashCommand('/writing-plans')).toBe(true)
  })

  it('includes skill commands from commands.catalog in the slash palette', () => {
    // Skills from commands.catalog are now included in the merged
    // results alongside complete.slash entries.
    expect(isDesktopSlashSuggestion('/plan')).toBe(true)
    expect(isDesktopSlashSuggestion('/xhs-image')).toBe(true)
    expect(isDesktopSlashSuggestion('/hermes-agent-dev')).toBe(true)
  })

  it('still blocks known desktop-unavailable commands for skill commands that shadow a block', () => {
    // If a skill happened to share a name with a blocked command,
    // blocked-list precedence should win
    expect(isDesktopSlashCommand('/model')).toBe(false)
    expect(isDesktopSlashCommand('/skills')).toBe(false)
    expect(isDesktopSlashCommand('/clear')).toBe(false)
    expect(isDesktopSlashCommand('/quit')).toBe(false)
  })
})
