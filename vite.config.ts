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
  esbuild: {
    legalComments: 'none',
    drop: ['debugger'],
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
    sourcemap: false,
    minify: 'esbuild',
    chunkSizeWarningLimit: 1500,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes('node_modules')) return undefined;
          const normalized = id.replace(/\\/g, '/');
          if (normalized.includes('/node_modules/react-dom/') || normalized.includes('/node_modules/react/')) {
            return 'vendor-react';
          }
          if (normalized.includes('/node_modules/framer-motion/')) {
            return 'vendor-motion';
          }
          if (normalized.includes('/node_modules/@mediapipe/tasks-vision/')) {
            return 'vendor-mediapipe';
          }
          if (normalized.includes('/node_modules/@supabase/')) {
            return 'vendor-supabase';
          }
          if (normalized.includes('/node_modules/lucide-react/')) {
            return 'vendor-icons';
          }
          return undefined;
        },
      },
    },
  },
});

