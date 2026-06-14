import { Ansi, Box, NoSelect, Text } from '@hermes/ink'
import { memo, useState } from 'react'

import { TERMUX_TUI_MODE } from '../config/env.js'
import { LONG_MSG } from '../config/limits.js'
import { hasLeadGap } from '../domain/blockLayout.js'
import { sectionMode } from '../domain/details.js'
import { userDisplay } from '../domain/messages.js'
import { ROLE } from '../domain/roles.js'
import { transcriptBodyWidth, transcriptGutterWidth } from '../lib/inputMetrics.js'
import {
  boundedLiveRenderText,
  compactPreview,
  hasAnsi,
  isPasteBackedText,
  sanitizeAnsiForRender,
  stripAnsi
} from '../lib/text.js'
import type { Theme } from '../theme.js'
import type { ActiveTool, DetailsMode, Msg, SectionVisibility } from '../types.js'

import { Md } from './markdown.js'
import { StreamingMd } from './streamingMarkdown.js'
import { ToolTrail } from './thinking.js'
import { TodoPanel } from './todoPanel.js'

// Collapse threshold for long system messages (system prompt etc.)
const SYSTEM_COLLAPSE_CHARS = 400

export const MessageLine = memo(function MessageLine({
  cols,
  compact,
  detailsMode = 'collapsed',
  detailsModeCommandOverride = false,
  isStreaming = false,
  msg,
  onLcmExpand,
  prev,
  sections,
  t,
  tools = []
}: MessageLineProps) {
  // Per-section overrides win over the global mode, so resolve each section
  // we might consume here once and gate visibility on the *content-bearing*
  // sections only — never on the global mode.  A `trail` message feeds Tool
  // calls + Activity; an assistant message with thinking/tools metadata
  // feeds Thinking + Tool calls.  Gating on every section would let
  // `thinking` (expanded by default) keep an empty wrapper alive when only
  // `tools` is hidden — exactly the empty-Box bug Copilot caught.
  const thinkingMode = sectionMode('thinking', detailsMode, sections, detailsModeCommandOverride)
  const toolsMode = sectionMode('tools', detailsMode, sections, detailsModeCommandOverride)
  const activityMode = sectionMode('activity', detailsMode, sections, detailsModeCommandOverride)
  const thinking = msg.thinking?.trim() ?? ''

  // One blank line above this block iff it opens a new visual group relative
  // to the block directly above it (`prev`) — the flex-grouping rule. Applied
  // intrinsically on each *rendered* element (not via an outer wrapper) so a
  // block that renders nothing — e.g. a tool trail hidden by /details — emits
  // no floating gap. Streaming-safe: the gap is derived from the stable
  // predecessor, never this block's own live content. See domain/blockLayout.
  const leadGap = hasLeadGap(prev, msg)

  // Collapse toggle for long system messages
  const systemIsLong = msg.role === 'system' && msg.text.length > SYSTEM_COLLAPSE_CHARS
  const [systemOpen, setSystemOpen] = useState(false)

  // LCM summary expanded-content cache (component state, NOT in history)
  const [lcmExpanded, setLcmExpanded] = useState(false)
  const [lcmLoading, setLcmLoading] = useState(false)
  const [lcmError, setLcmError] = useState<string | undefined>()
  const [lcmExpandedMessages, setLcmExpandedMessages] = useState<Msg[]>([])
  const [lcmHasMore, setLcmHasMore] = useState(false)

  if (msg.kind === 'trail' && msg.todos?.length) {
    return (
      <TodoPanel
        defaultCollapsed={msg.todoCollapsedByDefault}
        incomplete={msg.todoIncomplete}
        t={t}
        todos={msg.todos}
      />
    )
  }

  if (msg.kind === 'trail' && (msg.tools?.length || tools.length || thinking)) {
    return thinkingMode !== 'hidden' || toolsMode !== 'hidden' || activityMode !== 'hidden' ? (
      <Box flexDirection="column" marginTop={leadGap ? 1 : 0}>
        <ToolTrail
          commandOverride={detailsModeCommandOverride}
          detailsMode={detailsMode}
          reasoning={thinking}
          reasoningTokens={msg.thinkingTokens}
          sections={sections}
          t={t}
          tools={tools}
          toolTokens={msg.toolTokens}
          trail={msg.tools ?? []}
        />
      </Box>
    ) : null
  }

  // A trail with no reasoning, tools, or todos to show (e.g. the finalDetails
  // segment message.complete appends carrying only a token tally) has nothing
  // to draw — render nothing instead of an empty gutter row. blockRenders()
  // agrees, so it also stays transparent to grouping and never opens a gap.
  if (msg.kind === 'trail') {
    return null
  }

  if (msg.role === 'tool') {
    const maxChars = Math.max(24, cols - 14)
    const stripped = hasAnsi(msg.text) ? stripAnsi(msg.text) : msg.text
    const safeAnsi = hasAnsi(msg.text) ? sanitizeAnsiForRender(msg.text) : msg.text
    const preview = compactPreview(stripped, maxChars) || '(empty tool result)'

    return (
      <Box alignSelf="flex-start" borderColor={t.color.muted} borderStyle="round" marginLeft={3} paddingX={1}>
        {hasAnsi(msg.text) ? (
          <Text wrap="truncate-end">
            <Ansi>{safeAnsi}</Ansi>
          </Text>
        ) : (
          <Text color={t.color.muted} wrap="truncate-end">
            {preview}
          </Text>
        )}
      </Box>
    )
  }

  const { body, glyph, prefix } = ROLE[msg.role](t)
  const gutterWidth = transcriptGutterWidth(msg.role, t.brand.prompt)

  const showDetails =
    (toolsMode !== 'hidden' && Boolean(msg.tools?.length)) || (thinkingMode !== 'hidden' && Boolean(thinking))

  const showResponseSeparator = shouldShowResponseSeparator(msg, showDetails)

  const content = (() => {
    if (msg.kind === 'slash') {
      return <Text color={t.color.muted}>{msg.text}</Text>
    }

    // ── LCM summary block: collapsed/expandable ──
    if (msg.lcmSummary) {
      const handleExpand = async () => {
        if (lcmLoading) return
        if (lcmExpanded) {
          setLcmExpanded(false)
          return
        }
        if (!onLcmExpand) return
        setLcmLoading(true)
        setLcmError(undefined)
        try {
          // Expand the first node (most relevant summary)
          const nodeId = msg.lcmSummary!.nodeIds[0]
          const result = await onLcmExpand(nodeId)
          if ('error' in result) {
            setLcmError(result.error)
          } else {
            setLcmExpandedMessages(result.messages)
            setLcmHasMore(result.hasMore)
            setLcmExpanded(true)
          }
        } catch (e: any) { // eslint-disable-line @typescript-eslint/no-explicit-any
          setLcmError(String(e?.message ?? e))
        } finally {
          setLcmLoading(false)
        }
      }

      // Build preview: first 2 non-empty, non-marker lines of the summary
      const summaryLines = msg.text
        .split('\n')
        .filter(l => l.trim() && !l.trim().startsWith('[Recent') && !l.trim().startsWith('[Session') && !l.trim().startsWith('[Depth') && !l.trim().startsWith('[Expand'))
        .slice(0, 2)
      const previewText = summaryLines.join('\n')
      const depthLabel = msg.lcmSummary.depthLabels[0] ?? 'Summary'
      const nodeCount = msg.lcmSummary.nodeIds.length

      return (
        <Box flexDirection="column" marginTop={leadGap ? 1 : 0}>
          <Box onClick={handleExpand}>
            <Text color={t.color.accent}>{lcmExpanded ? '▾ ' : '▸ '}</Text>
            <Text color={t.color.accent} bold>{`[LCM: ${depthLabel} — ${nodeCount} node${nodeCount > 1 ? 's' : ''} compressed]`}</Text>
            <Text color={t.color.muted}>{'  '}</Text>
            <Text color={t.color.muted} dimColor>
              {lcmLoading ? '⠋ loading…' : lcmExpanded ? 'click to collapse' : 'click to expand'}
            </Text>
          </Box>
          {!lcmExpanded && previewText && (
            <Box marginLeft={2}>
              <Text color={t.color.muted} wrap="truncate-end">{previewText.slice(0, 200)}</Text>
            </Box>
          )}
          {lcmError && (
            <Box marginLeft={2}>
              <Text color="red">{`⚠ ${lcmError}`}</Text>
            </Box>
          )}
          {lcmExpanded && lcmExpandedMessages.length > 0 && (
            <Box flexDirection="column" marginLeft={2} borderStyle="round" borderColor={t.color.muted} paddingX={1}>
              {lcmExpandedMessages.map((em, i) => (
                <Text key={i} color={em.role === 'assistant' ? t.color.text : t.color.muted} wrap="wrap">
                  {em.text}
                </Text>
              ))}
              {lcmHasMore && (
                <Text color={t.color.muted} dimColor>{'… (more available)'}</Text>
              )}
            </Box>
          )}
        </Box>
      )
    }

    // ── Collapsible long system message (system prompt, AGENTS.md, etc.) ──
    // MUST come before the hasAnsi check — system messages from the backend
    // contain Rich markup escape codes that would otherwise hit <Ansi> full render.
    if (systemIsLong) {
      const firstLine = (msg.text.split('\n')[0] ?? '').trim().slice(0, 120) || '(system message)'

      return (
        <Box flexDirection="column">
          <Box onClick={() => setSystemOpen(v => !v)}>
            <Text color={t.color.accent}>{systemOpen ? '▾ ' : '▸ '}</Text>
            <Text color={t.color.muted}>{firstLine}</Text>
            <Text color={t.color.muted} dimColor>
              {' — '}
              {msg.text.length.toLocaleString()} chars
            </Text>
          </Box>
          {systemOpen && <Ansi>{sanitizeAnsiForRender(msg.text)}</Ansi>}
        </Box>
      )
    }

    if (msg.role !== 'user' && hasAnsi(msg.text)) {
      return <Ansi>{sanitizeAnsiForRender(msg.text)}</Ansi>
    }

    if (msg.role === 'assistant') {
      const bodyWidth = transcriptBodyWidth(cols, msg.role, t.brand.prompt, TERMUX_TUI_MODE)

      return isStreaming ? (
        // Incremental markdown: split at the last stable block boundary so
        // only the in-flight tail re-tokenizes per delta. See
        // streamingMarkdown.tsx for the cost model.
        <StreamingMd cols={bodyWidth} compact={compact} t={t} text={boundedLiveRenderText(msg.text)} />
      ) : (
        <Md cols={bodyWidth} compact={compact} t={t} text={msg.text} />
      )
    }

    if (msg.role === 'user' && msg.text.length > LONG_MSG && isPasteBackedText(msg.text)) {
      const [head, ...rest] = userDisplay(msg.text).split('[long message]')

      return (
        <Text color={body}>
          {head}
          <Text color={t.color.muted} dimColor>
            [long message]
          </Text>
          {rest.join('')}
        </Text>
      )
    }

    return <Text {...(body ? { color: body } : {})}>{msg.text}</Text>
  })()

  // Diff segments (emitted by pushInlineDiffSegment between narration
  // segments) keep a blank line on both sides so the patch doesn't butt up
  // against the prose around it.
  const isDiffSegment = msg.kind === 'diff'

  return (
    <Box
      flexDirection="column"
      marginBottom={msg.role === 'user' || isDiffSegment ? 1 : 0}
      marginTop={msg.role === 'user' || msg.kind === 'slash' || isDiffSegment || leadGap ? 1 : 0}
    >
      {showDetails && (
        <Box flexDirection="column" marginBottom={1}>
          <ToolTrail
            commandOverride={detailsModeCommandOverride}
            detailsMode={detailsMode}
            reasoning={thinking}
            reasoningTokens={msg.thinkingTokens}
            sections={sections}
            t={t}
            toolTokens={msg.toolTokens}
            trail={msg.tools}
          />
        </Box>
      )}

      {showResponseSeparator && (
        <Box marginBottom={1}>
          <NoSelect flexShrink={0} fromLeftEdge width={gutterWidth}>
            <Text color={t.color.border}>└─ </Text>
          </NoSelect>
          <Text color={t.color.muted} dim>
            Response
          </Text>
        </Box>
      )}

      <Box>
        <NoSelect flexShrink={0} fromLeftEdge width={gutterWidth}>
          <Text bold={msg.role === 'user'} color={prefix}>
            {glyph}{' '}
          </Text>
        </NoSelect>

        <Box width={transcriptBodyWidth(cols, msg.role, t.brand.prompt, TERMUX_TUI_MODE)}>{content}</Box>
      </Box>
    </Box>
  )
})

export const shouldShowResponseSeparator = (msg: Msg, showDetails: boolean): boolean =>
  msg.role === 'assistant' && showDetails && /\S/.test(msg.text)

interface MessageLineProps {
  cols: number
  compact?: boolean
  detailsMode?: DetailsMode
  detailsModeCommandOverride?: boolean
  isStreaming?: boolean
  msg: Msg
  // The block rendered directly above this one. Drives the group-boundary
  // lead gap (see domain/blockLayout.ts::hasLeadGap). Undefined at the top of
  // the transcript or when spacing is irrelevant.
  onLcmExpand?: (nodeId: number) => Promise<{ messages: Msg[]; hasMore: boolean } | { error: string }>
  prev?: Msg
  sections?: SectionVisibility
  t: Theme
  tools?: ActiveTool[]
}
