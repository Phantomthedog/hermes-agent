import type { Unstable_TriggerAdapter, Unstable_TriggerItem } from '@assistant-ui/core'
import { useCallback } from 'react'

import type { HermesGateway } from '@/hermes'
import {
  type CommandsCatalogLike,
  desktopSlashDescription,
  filterDesktopCommandsCatalog,
  isDesktopSlashSuggestion
} from '@/lib/desktop-slash-commands'

import type { CompletionEntry, CompletionPayload } from './use-live-completion-adapter'
import { useLiveCompletionAdapter } from './use-live-completion-adapter'

interface SlashItemMetadata extends Record<string, string> {
  command: string
  display: string
  meta: string
  rawText: string
}

function textValue(value: unknown, fallback = ''): string {
  if (typeof value === 'string') {
    return value
  }

  if (Array.isArray(value)) {
    return value
      .map(part => (Array.isArray(part) ? String(part[1] ?? '') : typeof part === 'string' ? part : ''))
      .join('')
      .trim()
  }

  return fallback
}

function commandText(value: string): string {
  return value.startsWith('/') ? value : `/${value}`
}

/** Live `/` completions backed by the gateway's `commands.catalog` +
 * `complete.slash` RPC. Skills and quick_commands are always sourced from
 * `commands.catalog` (complete.slash dropped them in PR #38467). */
export function useSlashCompletions(options: { gateway: HermesGateway | null }): {
  adapter: Unstable_TriggerAdapter
  loading: boolean
} {
  const { gateway } = options
  const enabled = Boolean(gateway)

  const fetcher = useCallback(
    async (query: string): Promise<CompletionPayload> => {
      if (!gateway) {
        return { items: [], query }
      }

      const text = `/${query}`

      try {
        // Always fetch the catalog — it is the source of truth for skills and
        // quick_commands.  For typed queries we also call complete.slash and
        // merge its results (deduplicated by command text).
        const catalog = filterDesktopCommandsCatalog(
          await gateway.request<CommandsCatalogLike>('commands.catalog')
        )

        if (!query) {
          // Empty "/" palette: return everything the catalog offers.
          const items = (catalog.pairs ?? []).map(([command, meta]) => ({
            text: command,
            display: command,
            meta
          }))
          return { items, query }
        }

        // Typed query: fetch catalog + complete.slash in parallel.
        const slashResult = gateway
          .request<{ items?: CompletionEntry[] }>('complete.slash', { text })
          .catch(() => null)

        // Build items from catalog, filtered client-side by the typed prefix.
        const q = query.toLowerCase()
        const catalogItems: CompletionEntry[] = (catalog.pairs ?? [])
          .filter(
            ([cmd]) =>
              cmd.toLowerCase().startsWith(`/${q}`) ||
              cmd.toLowerCase().slice(1).startsWith(q)
          )
          .map(([command, meta]) => ({
            text: command,
            display: command,
            meta: desktopSlashDescription(command, meta)
          }))

        // Merge complete.slash results, deduplicating by command text.
        const slashItems: CompletionEntry[] = (
          (await slashResult)?.items ?? []
        )
          .filter(item => isDesktopSlashSuggestion(item.text))
          .map(item => ({
            ...item,
            meta: desktopSlashDescription(item.text, textValue(item.meta))
          }))

        const seen = new Set(catalogItems.map(i => i.text.toLowerCase()))
        for (const item of slashItems) {
          if (!seen.has(item.text.toLowerCase())) {
            catalogItems.push(item)
            seen.add(item.text.toLowerCase())
          }
        }

        return { items: catalogItems, query }
      } catch {
        return { items: [], query }
      }
    },
    [gateway]
  )

  const toItem = useCallback((entry: CompletionEntry, index: number): Unstable_TriggerItem => {
    const command = commandText(entry.text)
    const display = textValue(entry.display, commandText(entry.text))
    const meta = textValue(entry.meta)

    const metadata: SlashItemMetadata = {
      command,
      display,
      meta,
      // Provide rawText so hermesDirectiveFormatter.serialize uses the
      // direct-insertion path instead of the legacy @type:id fallback.
      // Without this, the item.id (which includes a "|index" suffix for
      // trigger-adapter uniqueness) leaks into the serialized chip text
      // and the submitted command.
      rawText: command
    }

    return {
      id: `${entry.text}|${index}`,
      type: 'slash',
      label: display.startsWith('/') ? display.slice(1) : display,
      ...(meta ? { description: meta } : {}),
      metadata
    }
  }, [])

  return useLiveCompletionAdapter({ enabled, fetcher, toItem })
}
