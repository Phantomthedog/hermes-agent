const fs = require('node:fs')
const http = require('node:http')
const path = require('node:path')
const { spawn } = require('node:child_process')

const DEFAULT_CONTROL_BASE_URL = 'http://127.0.0.1:10011'
const DEFAULT_TIMEOUT_MS = 3000
const WINDOWS_CURL_CANDIDATES = [
  '/mnt/c/Windows/System32/curl.exe',
  '/mnt/c/Windows/Sysnative/curl.exe'
]

const WINDOWS_USER_PROFILE_PREFIX = '/mnt/c/Users/'
const RELATIVE_TOKEN_FILE = 'AppData/Local/WhisperDictate/control_token'

function normalizeControlBaseUrl(raw) {
  const value = String(raw || '').trim()
  if (!value) return null
  return value.replace(/\/+$/, '')
}

function findWindowsCurl(fileExists = fs.existsSync) {
  for (const candidate of WINDOWS_CURL_CANDIDATES) {
    try {
      if (fileExists(candidate)) return candidate
    } catch {
      // Keep checking the remaining candidates.
    }
  }
  return null
}

function resolveTokenFromEnv(env = process.env) {
  return (env.HERMES_DICTATE_CONTROL_TOKEN || env.WHISPER_DICTATE_CONTROL_TOKEN || '').trim() || null
}

function resolveTokenFromFile(fileExists = fs.existsSync, readFile = fs.readFileSync) {
  // Known token file paths: Windows user dirs under /mnt/c/Users/
  // Also check a direct path if we can determine the user
  const candidates = []

  // Scan known Windows user directories
  try {
    const entries = fs.readdirSync(WINDOWS_USER_PROFILE_PREFIX, { withFileTypes: true })
    for (const entry of entries) {
      if (entry.isDirectory() && !entry.name.startsWith('.')) {
        const tokenPath = path.join(WINDOWS_USER_PROFILE_PREFIX, entry.name, RELATIVE_TOKEN_FILE)
        candidates.push(tokenPath)
      }
    }
  } catch {
    // Can't scan /mnt/c/Users/ — fall through
  }

  // Read first existing file
  for (const candidate of candidates) {
    try {
      if (fileExists(candidate)) {
        const content = readFile(candidate, 'utf-8').trim()
        if (content) return content
      }
    } catch {
      // Try next candidate
    }
  }
  return null
}

function resolveControlToken({ env = process.env, fileExists = fs.existsSync, readFile = fs.readFileSync } = {}) {
  const fromEnv = resolveTokenFromEnv(env)
  if (fromEnv) return fromEnv
  try {
    return resolveTokenFromFile(fileExists, readFile)
  } catch {
    return null
  }
}

function resolveDictationControlTransport({
  env = process.env,
  fileExists = fs.existsSync,
  readFile = fs.readFileSync,
  isWsl = false
} = {}) {

  const token = resolveControlToken({ env, fileExists, readFile })

  const overrideUrl = normalizeControlBaseUrl(env.HERMES_DICTATE_CONTROL_URL)
  if (overrideUrl) {
    return {
      baseUrl: overrideUrl,
      reason: 'env-override',
      transport: 'http',
      token
    }
  }

  if (isWsl) {
    const windowsCurl = findWindowsCurl(fileExists)
    if (windowsCurl) {
      return {
        baseUrl: DEFAULT_CONTROL_BASE_URL,
        binary: windowsCurl,
        reason: 'wsl-windows-curl',
        transport: 'windows-curl',
        token
      }
    }
  }

  return {
    baseUrl: DEFAULT_CONTROL_BASE_URL,
    reason: 'localhost-http',
    transport: 'http',
    token
  }
}

function buildActionUrl(baseUrl, action) {
  if (!/^[a-z]+$/i.test(String(action || ''))) {
    throw new Error(`Invalid dictation action: ${action}`)
  }
  const normalizedBaseUrl = normalizeControlBaseUrl(baseUrl) || DEFAULT_CONTROL_BASE_URL
  return `${normalizedBaseUrl}/${action}`
}

function settleOnce(resolve, value) {
  resolve(value)
  return true
}

function sendHttpPost(url, {
  httpRequest = http.request,
  log = () => {},
  timeoutMs = DEFAULT_TIMEOUT_MS,
  token = null
} = {}) {
  return new Promise(resolve => {
    let settled = false
    let timedOut = false

    const settle = value => {
      if (settled) return
      settled = settleOnce(resolve, value)
    }

    const headers = {}
    if (token) {
      headers['Authorization'] = `Bearer ${token}`
    }

    const req = httpRequest(url, { method: 'POST', headers, timeout: timeoutMs }, res => {
      let body = ''
      res.on('data', chunk => { body += chunk })
      res.on('end', () => {
        log(`[HERMES_DICTATE] control response ${res.statusCode}: ${body}`)
        settle({ body, ok: res.statusCode >= 200 && res.statusCode < 300, statusCode: res.statusCode })
      })
    })

    req.on('error', err => {
      if (timedOut) return
      const message = err?.message || String(err)
      log(`[HERMES_DICTATE] control error: ${message}`)
      settle({ error: message, ok: false })
    })
    req.on('timeout', () => {
      timedOut = true
      log('[HERMES_DICTATE] control request timed out')
      try {
        req.destroy()
      } catch {}
      settle({ error: 'timeout', ok: false })
    })
    req.end()
  })
}

function sendWindowsCurlPost(url, {
  binary,
  log = () => {},
  spawnFn = spawn,
  timeoutMs = DEFAULT_TIMEOUT_MS,
  token = null
} = {}) {
  return new Promise(resolve => {
    if (!binary) {
      log('[HERMES_DICTATE] control error: Windows curl transport requested with no binary')
      resolve({ error: 'missing-windows-curl', ok: false })
      return
    }

    const maxTimeSeconds = Math.max(1, Math.ceil(timeoutMs / 1000))
    const args = ['-sS', '--max-time', String(maxTimeSeconds), '-X', 'POST']
    if (token) {
      args.push('-H', `Authorization: Bearer ${token}`)
    }
    args.push(url)
    const child = spawnFn(binary, args, {
      windowsHide: true
    })

    let body = ''
    let stderr = ''
    let settled = false

    const settle = value => {
      if (settled) return
      settled = settleOnce(resolve, value)
    }

    const timeout = setTimeout(() => {
      log('[HERMES_DICTATE] control request timed out')
      try {
        child.kill()
      } catch {}
      settle({ error: 'timeout', ok: false })
    }, timeoutMs + 500)

    child.stdout?.on('data', chunk => { body += chunk })
    child.stderr?.on('data', chunk => { stderr += chunk })
    child.on('error', err => {
      clearTimeout(timeout)
      const message = err?.message || String(err)
      log(`[HERMES_DICTATE] control error: ${message}`)
      settle({ error: message, ok: false })
    })
    child.on('close', code => {
      clearTimeout(timeout)
      const trimmedBody = body.trim()
      const trimmedStderr = stderr.trim()
      if (code === 0) {
        log(`[HERMES_DICTATE] control response curl-exe: ${trimmedBody}`)
        settle({ body: trimmedBody, ok: true, statusCode: 0 })
        return
      }
      const message = trimmedStderr || `curl.exe exited ${code}`
      log(`[HERMES_DICTATE] control error: ${message}`)
      settle({ error: message, ok: false, statusCode: code })
    })
  })
}

function createDictationControlClient({
  env = process.env,
  fileExists = fs.existsSync,
  httpRequest = http.request,
  isWsl = false,
  log = () => {},
  spawnFn = spawn,
  timeoutMs = DEFAULT_TIMEOUT_MS
} = {}) {
  const endpoint = resolveDictationControlTransport({ env, fileExists, isWsl })

  function send(action) {
    const url = buildActionUrl(endpoint.baseUrl, action)
    const suffix = endpoint.transport === 'windows-curl' ? ` via ${endpoint.binary}` : ''
    const tokenInfo = endpoint.token ? ' (with auth token)' : ''
    log(`[HERMES_DICTATE] control request POST ${url}${suffix}${tokenInfo}`)

    if (endpoint.transport === 'windows-curl') {
      return sendWindowsCurlPost(url, {
        binary: endpoint.binary,
        log,
        spawnFn,
        timeoutMs,
        token: endpoint.token
      })
    }

    return sendHttpPost(url, {
      httpRequest,
      log,
      timeoutMs,
      token: endpoint.token
    })
  }

  return { endpoint, send }
}

module.exports = {
  DEFAULT_CONTROL_BASE_URL,
  DEFAULT_TIMEOUT_MS,
  WINDOWS_CURL_CANDIDATES,
  buildActionUrl,
  createDictationControlClient,
  findWindowsCurl,
  normalizeControlBaseUrl,
  resolveControlToken,
  resolveDictationControlTransport,
  sendHttpPost,
  sendWindowsCurlPost
}
