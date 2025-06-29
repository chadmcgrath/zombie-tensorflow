import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  base: './',
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
  },
  server: {
    port: 3001,
    strictPort: false,
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ['./test/setup.js']
  },
})
