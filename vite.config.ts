import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

const rootDir = path.dirname(fileURLToPath(import.meta.url));

export const workspaceAliases = {
  '@drtechcares/smart-biometric-core': path.resolve(rootDir, 'packages/smart-biometric-core/src/index.ts'),
  '@drtechcares/face-client': path.resolve(rootDir, 'packages/face-client/src/index.ts'),
  '@drtechcares/pose-engine': path.resolve(rootDir, 'packages/pose-engine/src/index.ts'),
  '@drtechcares/kiosk-input': path.resolve(rootDir, 'packages/kiosk-input/src/index.ts'),
  '@drtechcares/clinical-i18n': path.resolve(rootDir, 'packages/clinical-i18n/src/index.ts'),
  '@drtechcares/api-client': path.resolve(rootDir, 'packages/api-client/src/index.ts'),
  '@apps/kiosk': path.resolve(rootDir, 'apps/kiosk/src/index.ts'),
  '@apps/pt': path.resolve(rootDir, 'apps/pt/src/index.ts'),
  '@apps/admin': path.resolve(rootDir, 'apps/admin/src/index.ts'),
  '@apps/api': path.resolve(rootDir, 'apps/api/src/index.ts'),
};

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: workspaceAliases,
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8787',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    chunkSizeWarningLimit: 1500,
  },
});

