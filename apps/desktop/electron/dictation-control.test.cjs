const assert = require('node:assert/strict')
const { EventEmitter } = require('node:events')
const test = require('node:test')

const {
  DEFAULT_CONTROL_BASE_URL,
  buildActionUrl,
  createDictationControlClient,
  findWindowsCurl,
  normalizeControlBaseUrl,
  resolveDictationControlTransport
} = require('./dictation-control.cjs')

test('normalizeControlBaseUrl trims whitespace and trailing slashes', () => {
  assert.equal(normalizeControlBaseUrl(' http://127.0.0.1:10011/// '), 'http://127.0.0.1:10011')
  assert.equal(normalizeControlBaseUrl(''), null)
})

test('findWindowsCurl prefers System32 curl.exe under WSL mounts', () => {
  const seen = []
  const result = findWindowsCurl(path => {
    seen.push(path)
    return path === '/mnt/c/Windows/System32/curl.exe'
  })

  assert.equal(result, '/mnt/c/Windows/System32/curl.exe')
  assert.deepEqual(seen, ['/mnt/c/Windows/System32/curl.exe'])
})

test('WSL uses Windows curl.exe to reach the Windows loopback control API', () => {
  const endpoint = resolveDictationControlTransport({
    env: {},
    fileExists: path => path === '/mnt/c/Windows/System32/curl.exe',
    isWsl: true
  })

  assert.deepEqual(endpoint, {
    baseUrl: DEFAULT_CONTROL_BASE_URL,
    binary: '/mnt/c/Windows/System32/curl.exe',
    reason: 'wsl-windows-curl',
    transport: 'windows-curl'
  })
})

test('WSL transport does not derive control URL from public DNS nameservers', () => {
  const endpoint = resolveDictationControlTransport({
    env: {},
    fileExists: () => true,
    isWsl: true
  })

  assert.equal(endpoint.baseUrl, 'http://127.0.0.1:10011')
  assert.notEqual(endpoint.baseUrl, 'http://1.1.1.1:10011')
})

test('env override wins and uses direct HTTP transport', () => {
  const endpoint = resolveDictationControlTransport({
    env: { HERMES_DICTATE_CONTROL_URL: 'http://192.0.2.10:10011/' },
    fileExists: () => true,
    isWsl: true
  })

  assert.deepEqual(endpoint, {
    baseUrl: 'http://192.0.2.10:10011',
    reason: 'env-override',
    transport: 'http'
  })
})

test('native desktop uses direct localhost HTTP', () => {
  const endpoint = resolveDictationControlTransport({ env: {}, fileExists: () => false, isWsl: false })

  assert.deepEqual(endpoint, {
    baseUrl: DEFAULT_CONTROL_BASE_URL,
    reason: 'localhost-http',
    transport: 'http'
  })
})

test('buildActionUrl appends validated action', () => {
  assert.equal(buildActionUrl('http://127.0.0.1:10011/', 'start'), 'http://127.0.0.1:10011/start')
  assert.throws(() => buildActionUrl('http://127.0.0.1:10011', '../stop'), /Invalid dictation action/)
})

test('client send uses Windows curl transport on WSL', async () => {
  const calls = []
  const logs = []
  const spawnFn = (command, args, options) => {
    calls.push({ args, command, options })
    const child = new EventEmitter()
    child.stdout = new EventEmitter()
    child.stderr = new EventEmitter()
    child.kill = () => {}
    process.nextTick(() => {
      child.stdout.emit('data', Buffer.from('{"ok": true, "status": "started"}'))
      child.emit('close', 0)
    })
    return child
  }

  const client = createDictationControlClient({
    env: {},
    fileExists: path => path === '/mnt/c/Windows/System32/curl.exe',
    isWsl: true,
    log: message => logs.push(message),
    spawnFn,
    timeoutMs: 1000
  })

  const result = await client.send('start')

  assert.equal(result.ok, true)
  assert.equal(calls.length, 1)
  assert.equal(calls[0].command, '/mnt/c/Windows/System32/curl.exe')
  assert.deepEqual(calls[0].args, [
    '-sS',
    '--max-time',
    '1',
    '-X',
    'POST',
    'http://127.0.0.1:10011/start'
  ])
  assert.equal(calls[0].options.windowsHide, true)
  assert.match(logs[0], /POST http:\/\/127\.0\.0\.1:10011\/start via \/mnt\/c\/Windows\/System32\/curl\.exe/)
})
