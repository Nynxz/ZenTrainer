/// <reference types="vite/client" />
declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  const component: DefineComponent<object, object, unknown>
  export default component
}

/** this web build: the ZenTrainer version and commit it was built from (vite.config.ts) */
declare const __ZT_BUILD__: { version: string; commit: string; describe: string; dirty: boolean; built: number }
