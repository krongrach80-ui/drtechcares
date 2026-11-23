import { defineConfig } from 'vitest/config';
import { workspaceAliases } from './vite.config';

export default defineConfig({
  resolve: {
    alias: workspaceAliases,
  },
  test: {
    environment: 'node',
    include: [
      'packages/*/src/**/*.test.ts',
      'apps/*/src/**/*.test.ts',
      'src/**/*.test.ts',
      'src/**/*.test.tsx',
    ],
    globals: true,
  },
});

