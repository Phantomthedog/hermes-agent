import { describe, expect, it, vi } from 'vitest'

import {
  checkClipboardReadResult,
  isUsableClipboardText,
  readClipboardText,
  writeClipboardText
} from '../lib/clipboard.js'

const EXPECTED_OPTS = {
  encoding: 'utf8',
  maxBuffer: 64 * 1024 * 1024,
  timeout: 5000,
  windowsHide: true
}

describe('readClipboardText', () => {
  it('reads text from pbpaste on macOS', async () => {
    const run = vi.fn().mockResolvedValue({ stdout: 'hello world\n' })

    await expect(readClipboardText('darwin', run)).resolves.toBe('hello world\n')
    expect(run).toHaveBeenCalledWith(
      'pbpaste',
      [],
      expect.objectContaining(EXPECTED_OPTS)
    )
  })

  it('reads text from PowerShell on Windows', async () => {
    const run = vi.fn().mockResolvedValue({ stdout: 'from windows\r\n' })

    await expect(readClipboardText('win32', run)).resolves.toBe('from windows\r\n')
    expect(run).toHaveBeenCalledWith(
      'powershell',
      ['-NoProfile', '-NonInteractive', '-Command', 'Get-Clipboard -Raw'],
      expect.objectContaining(EXPECTED_OPTS)
    )
  })

  it('tries powershell.exe first on WSL', async () => {
    const run = vi.fn().mockResolvedValue({ stdout: 'from wsl\n' })

    await expect(readClipboardText('linux', run, { WSL_INTEROP: '/tmp/socket' } as NodeJS.ProcessEnv)).resolves.toBe(
      'from wsl\n'
    )
    expect(run).toHaveBeenCalledWith(
      'powershell.exe',
      ['-NoProfile', '-NonInteractive', '-Command', 'Get-Clipboard -Raw'],
      expect.objectContaining(EXPECTED_OPTS)
    )
  })

  it('uses wl-paste on Wayland Linux', async () => {
    const run = vi.fn().mockResolvedValue({ stdout: 'from wayland\n' })

    await expect(readClipboardText('linux', run, { WAYLAND_DISPLAY: 'wayland-1' } as NodeJS.ProcessEnv)).resolves.toBe(
      'from wayland\n'
    )
    expect(run).toHaveBeenCalledWith(
      'wl-paste',
      ['--type', 'text'],
      expect.objectContaining(EXPECTED_OPTS)
    )
  })

  it('falls back to xclip on Linux when wl-paste fails', async () => {
    const run = vi
      .fn()
      .mockRejectedValueOnce(new Error('wl-paste missing'))
      .mockResolvedValueOnce({ stdout: 'from xclip\n' })

    await expect(readClipboardText('linux', run, { WAYLAND_DISPLAY: 'wayland-1' } as NodeJS.ProcessEnv)).resolves.toBe(
      'from xclip\n'
    )
    expect(run).toHaveBeenNthCalledWith(
      1,
      'wl-paste',
      ['--type', 'text'],
      expect.objectContaining(EXPECTED_OPTS)
    )
    expect(run).toHaveBeenNthCalledWith(
      2,
      'xclip',
      ['-selection', 'clipboard', '-out'],
      expect.objectContaining(EXPECTED_OPTS)
    )
  })

  it('returns null when every clipboard backend fails', async () => {
    const run = vi.fn().mockRejectedValue(new Error('clipboard failed'))

    await expect(
      readClipboardText('linux', run, { WAYLAND_DISPLAY: 'wayland-1' } as NodeJS.ProcessEnv)
    ).resolves.toBeNull()
  })

  it('populates hardErrors on timeout (killed=true)', async () => {
    const run = vi.fn().mockRejectedValue({ killed: true, message: 'process timed out', code: 'ETIMEOUT' })
    const errors: string[] = []

    const result = await readClipboardText('linux', run, { WSL_INTEROP: '/tmp/socket' } as NodeJS.ProcessEnv, errors)

    expect(result).toBeNull()
    expect(errors.length).toBeGreaterThan(0)
    expect(errors[0]).toContain('powershell.exe')
    expect(errors[0]).toContain('process timed out')
  })

  it('populates hardErrors on maxBuffer exceeded', async () => {
    const run = vi.fn().mockRejectedValue({ killed: true, code: 'ERR_CHILD_PROCESS_STDIO_MAXBUFFER', message: 'maxBuffer exceeded' })
    const errors: string[] = []

    const result = await readClipboardText('linux', run, { WSL_INTEROP: '/tmp/socket' } as NodeJS.ProcessEnv, errors)

    expect(result).toBeNull()
    expect(errors.length).toBeGreaterThan(0)
    expect(errors[0]).toContain('powershell.exe')
    expect(errors[0]).toContain('maxBuffer exceeded')
  })

  it('does not populate hardErrors on soft failures (command not found)', async () => {
    const run = vi.fn().mockRejectedValue({ code: 'ENOENT', message: 'command not found' })
    const errors: string[] = []

    // On Linux without WSL/WAYLAND, only xclip/xsel are tried
    const result = await readClipboardText('linux', run, {}, errors)

    expect(result).toBeNull()
    expect(errors).toEqual([])
  })

  it('does not populate hardErrors on normal successful read', async () => {
    const run = vi.fn().mockResolvedValue({ stdout: 'hello\n' })
    const errors: string[] = []

    const result = await readClipboardText('linux', run, { WSL_INTEROP: '/tmp/socket' } as NodeJS.ProcessEnv, errors)

    expect(result).toBe('hello\n')
    expect(errors).toEqual([])
  })

  it('falls through backends after a hard error (still logs)', async () => {
    // First backend (powershell.exe) has a hard error, second (xclip) has a soft error
    const run = vi
      .fn()
      .mockRejectedValueOnce({ killed: true, message: 'timeout', code: 'ETIMEOUT' })
      .mockRejectedValueOnce({ code: 'ENOENT', message: 'not found' })
    const errors: string[] = []

    const result = await readClipboardText('linux', run, { WSL_INTEROP: '/tmp/socket' } as NodeJS.ProcessEnv, errors)

    expect(result).toBeNull()
    expect(errors.length).toBe(1) // Only the hard error from powershell.exe
    expect(errors[0]).toContain('powershell.exe')
  })
})

describe('checkClipboardReadResult', () => {
  it('returns canPaste:true for usable text regardless of hardErrors', () => {
    expect(checkClipboardReadResult('hello', [])).toEqual({ canPaste: true })
    expect(checkClipboardReadResult('hello', ['something failed'])).toEqual({ canPaste: true })
  })

  it('returns canPaste:false + message when null text and hardErrors exist', () => {
    const result = checkClipboardReadResult(null, ['[powershell.exe] process failed'])
    expect(result).toEqual({
      canPaste: false,
      message: expect.stringContaining('Could not read text from clipboard')
    })
  })

  it('returns canPaste:false with no message when null text and no hardErrors', () => {
    const result = checkClipboardReadResult(null, [])
    expect(result).toEqual({ canPaste: false })
  })

  it('returns canPaste:false with no message for empty string and no hardErrors', () => {
    const result = checkClipboardReadResult('', [])
    expect(result).toEqual({ canPaste: false })
  })

  it('prefers hardErrors over empty text — shows message even when text is present but unusable', () => {
    // Binary-like text with null bytes is not usable
    const binaryText = 'PNG\x00\x01\x02\x03IHDR'
    const result = checkClipboardReadResult(binaryText, ['[powershell.exe] process failed'])
    expect(result).toEqual({
      canPaste: false,
      message: expect.stringContaining('Could not read text from clipboard')
    })
  })
})

describe('isUsableClipboardText', () => {
  it('accepts normal text', () => {
    expect(isUsableClipboardText('hello world\n')).toBe(true)
  })

  it('rejects empty or whitespace-only content', () => {
    expect(isUsableClipboardText('')).toBe(false)
    expect(isUsableClipboardText('  \n\t')).toBe(false)
  })

  it('rejects binary-looking clipboard payloads', () => {
    expect(isUsableClipboardText('PNG\x00\x01\x02\x03IHDR')).toBe(false)
    expect(isUsableClipboardText('TIFF\ufffd\ufffd\ufffdmetadata')).toBe(false)
  })
})

describe('writeClipboardText', () => {
  it('does nothing off macOS when no tools are available', async () => {
    const child = {
      once: vi.fn((event: string, cb: (code?: number) => void) => {
        if (event === 'close') {
          cb(1) // non-zero exit = failure
        }

        return child
      }),
      stdin: { end: vi.fn() }
    }

    const start = vi.fn().mockReturnValue(child)

    // Linux with no WAYLAND_DISPLAY / no WSL_INTEROP — falls through xclip then xsel, both fail
    await expect(writeClipboardText('hello', 'linux', start, {})).resolves.toBe(false)
  })

  it('writes text to pbcopy on macOS', async () => {
    const stdin = { end: vi.fn() }

    const child = {
      once: vi.fn((event: string, cb: (code?: number) => void) => {
        if (event === 'close') {
          cb(0)
        }

        return child
      }),
      stdin
    }

    const start = vi.fn().mockReturnValue(child)

    await expect(writeClipboardText('hello world', 'darwin', start as any)).resolves.toBe(true)
    expect(start).toHaveBeenCalledWith(
      'pbcopy',
      [],
      expect.objectContaining({ stdio: ['pipe', 'ignore', 'ignore'], windowsHide: true })
    )
    expect(stdin.end).toHaveBeenCalledWith('hello world')
  })

  it('returns false when pbcopy fails', async () => {
    const child = {
      once: vi.fn((event: string, cb: () => void) => {
        if (event === 'error') {
          cb()
        }

        return child
      }),
      stdin: { end: vi.fn() }
    }

    const start = vi.fn().mockReturnValue(child)

    await expect(writeClipboardText('hello world', 'darwin', start as any)).resolves.toBe(false)
  })

  it('uses wl-copy on Wayland Linux', async () => {
    const stdin = { end: vi.fn() }

    const child = {
      once: vi.fn((event: string, cb: (code?: number) => void) => {
        if (event === 'close') {
          cb(0)
        }

        return child
      }),
      stdin
    }

    const start = vi.fn().mockReturnValue(child)

    await expect(
      writeClipboardText('wayland text', 'linux', start as any, { WAYLAND_DISPLAY: 'wayland-1' })
    ).resolves.toBe(true)
    expect(start).toHaveBeenCalledWith(
      'wl-copy',
      ['--type', 'text/plain'],
      expect.objectContaining({ stdio: ['pipe', 'ignore', 'ignore'], windowsHide: true })
    )
    expect(stdin.end).toHaveBeenCalledWith('wayland text')
  })

  it('falls back to xclip when wl-copy fails on Wayland', async () => {
    let callCount = 0
    const stdin = { end: vi.fn() }

    const child = {
      once: vi.fn((event: string, cb: (code?: number) => void) => {
        if (event === 'close') {
          callCount++
          // wl-copy fails, xclip succeeds
          cb(callCount === 1 ? 1 : 0)
        }

        return child
      }),
      stdin
    }

    const start = vi.fn().mockReturnValue(child)

    await expect(
      writeClipboardText('x11 text', 'linux', start as any, { WAYLAND_DISPLAY: 'wayland-1' })
    ).resolves.toBe(true)
    expect(start).toHaveBeenNthCalledWith(
      1,
      'wl-copy',
      ['--type', 'text/plain'],
      expect.anything()
    )
    expect(start).toHaveBeenNthCalledWith(
      2,
      'xclip',
      ['-selection', 'clipboard', '-in'],
      expect.anything()
    )
  })

  it('falls back to xsel when both wl-copy and xclip fail', async () => {
    let callCount = 0
    const stdin = { end: vi.fn() }

    const child = {
      once: vi.fn((event: string, cb: (code?: number) => void) => {
        if (event === 'close') {
          callCount++
          cb(callCount < 3 ? 1 : 0) // first two fail, third (xsel) succeeds
        }

        return child
      }),
      stdin
    }

    const start = vi.fn().mockReturnValue(child)

    await expect(
      writeClipboardText('xsel text', 'linux', start as any, { WAYLAND_DISPLAY: 'wayland-1' })
    ).resolves.toBe(true)
    expect(start).toHaveBeenNthCalledWith(3, 'xsel', ['--clipboard', '--input'], expect.anything())
  })

  it('uses PowerShell on WSL2 when WSL_DISTRO_NAME is set', async () => {
    const stdin = { end: vi.fn() }

    const child = {
      once: vi.fn((event: string, cb: (code?: number) => void) => {
        if (event === 'close') {
          cb(0)
        }

        return child
      }),
      stdin
    }

    const start = vi.fn().mockReturnValue(child)

    await expect(writeClipboardText('wsl text', 'linux', start as any, { WSL_DISTRO_NAME: 'Ubuntu' })).resolves.toBe(true)
    expect(start).toHaveBeenCalledWith(
      'powershell.exe',
      expect.arrayContaining(['-NoProfile', '-NonInteractive']),
      expect.anything()
    )
    // PowerShell uses base64-encoded UTF-8 via command argument, not stdin
    expect(stdin.end).not.toHaveBeenCalled()
    const calledArgs = start.mock.calls[0][1] as string[]
    const commandIdx = calledArgs.indexOf('-Command')
    expect(commandIdx).toBeGreaterThan(-1)
    const script = calledArgs[commandIdx + 1]
    expect(script).toContain('FromBase64String')
    expect(script).toContain(Buffer.from('wsl text', 'utf8').toString('base64'))
  })

  it('prefers the Windows clipboard path over wl-copy inside WSLg', async () => {
    const stdin = { end: vi.fn() }

    const child = {
      once: vi.fn((event: string, cb: (code?: number) => void) => {
        if (event === 'close') {
          cb(0)
        }

        return child
      }),
      stdin
    }

    const start = vi.fn().mockReturnValue(child)

    await expect(
      writeClipboardText('wslg text', 'linux', start as any, {
        WAYLAND_DISPLAY: 'wayland-0',
        WSL_DISTRO_NAME: 'Ubuntu'
      })
    ).resolves.toBe(true)
    expect(start).toHaveBeenNthCalledWith(
      1,
      'powershell.exe',
      expect.arrayContaining(['-NoProfile', '-NonInteractive']),
      expect.anything()
    )
    // PowerShell uses base64-encoded UTF-8 via command argument, not stdin
    expect(stdin.end).not.toHaveBeenCalled()
    const calledArgs = start.mock.calls[0][1] as string[]
    const commandIdx = calledArgs.indexOf('-Command')
    const script = calledArgs[commandIdx + 1]
    expect(script).toContain('FromBase64String')
    expect(script).toContain(Buffer.from('wslg text', 'utf8').toString('base64'))
  })

  it('uses PowerShell on Windows', async () => {
    const stdin = { end: vi.fn() }

    const child = {
      once: vi.fn((event: string, cb: (code?: number) => void) => {
        if (event === 'close') {
          cb(0)
        }

        return child
      }),
      stdin
    }

    const start = vi.fn().mockReturnValue(child)

    await expect(writeClipboardText('windows text', 'win32', start as any)).resolves.toBe(true)
    expect(start).toHaveBeenCalledWith(
      'powershell',
      expect.arrayContaining(['-NoProfile', '-NonInteractive']),
      expect.anything()
    )
    // PowerShell uses base64-encoded UTF-8 via command argument, not stdin
    expect(stdin.end).not.toHaveBeenCalled()
  })

  it('preserves CJK text via base64 encoding in PowerShell on WSL', async () => {
    const stdin = { end: vi.fn() }

    const child = {
      once: vi.fn((event: string, cb: (code?: number) => void) => {
        if (event === 'close') {
          cb(0)
        }

        return child
      }),
      stdin
    }

    const start = vi.fn().mockReturnValue(child)
    const cjkText = '你好世界，测试中文 🎉'

    await expect(writeClipboardText(cjkText, 'linux', start as any, { WSL_INTEROP: '/tmp/socket' })).resolves.toBe(true)
    const calledArgs = start.mock.calls[0][1] as string[]
    const commandIdx = calledArgs.indexOf('-Command')
    const script = calledArgs[commandIdx + 1]
    expect(script).toContain(Buffer.from(cjkText, 'utf8').toString('base64'))
    expect(script).toContain('UTF8.GetString')
  })
})
