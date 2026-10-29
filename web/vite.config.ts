import vue from '@vitejs/plugin-vue'
import { execSync } from 'node:child_process'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { defineConfig, type ProxyOptions } from 'vite'

// In dev, Vite serves the app and proxies everything else to a running ComfyUI.
// In prod, ComfyUI serves web/dist itself at /zentrainer/ (see engine/zentrainer/routes.py).
const comfy = process.env.COMFY_URL ?? 'http://127.0.0.1:8188'
// ComfyUI refuses a request whose Origin isn't the address it was reached at (its CSRF guard); the
// proxy reaches it at its own (changeOrigin), so the browser's Origin, Vite's, is set to that too.
// Without it every POST from the dev server (saving, starting a run, a mask) came back 403. Only
// the dev server's own pages get that: another site's Origin goes through as it is, and is refused
const toComfy: ProxyOptions = {
  target: comfy,
  changeOrigin: true,
  configure: (proxy) =>
    proxy.on('proxyReq', (proxyReq, req) => {
      const origin = req.headers.origin
      if (origin && req.headers.host && new URL(origin).host === req.headers.host)
        proxyReq.setHeader('origin', new URL(comfy).origin)
    }),
}

// What this build is, for Settings > Versions: web/dist isn't in git, so a tab can be an older
// build than the engine it talks to. No git (a downloaded copy): just the version.
const here = fileURLToPath(new URL('.', import.meta.url))
const git = (args: string) => {
  try {
    return execSync(`git ${args}`, { cwd: here, stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim()
  } catch {
    return ''
  }
}
const engineInit = readFileSync(new URL('../engine/zentrainer/__init__.py', import.meta.url), 'utf8')
const build = {
  version: /__version__ = "([^"]+)"/.exec(engineInit)?.[1] ?? '',
  commit: git('rev-parse HEAD'),
  describe: git('describe --tags --always'),
  // only the web app's own files count: an engine edit doesn't change this build
  dirty: !!git('status --porcelain --untracked-files=no -- .'),
  built: Date.now(),
}

export default defineConfig({
  base: '/zentrainer/',
  plugins: [vue()],
  define: { __ZT_BUILD__: JSON.stringify(build) },
  // zenkit-ui is linked from the ZenKit checkout, which has its own vue: one copy only
  resolve: { alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) }, dedupe: ['vue'] },
  build: { target: 'es2022' },
  server: {
    // engine/catalog/families.json is imported from outside web/
    fs: { allow: ['..'] },
    proxy: {
      '/zentrainer/api': toComfy,
      '/zenkit': toComfy,
      '/object_info': toComfy,
      '/settings': toComfy,
      '/internal': toComfy,
      '/prompt': toComfy,
      '/view': toComfy,
      '/upload': toComfy,
      '/templates': toComfy,
      '/ws': { target: comfy.replace(/^http/, 'ws'), ws: true },
    },
  },
})
