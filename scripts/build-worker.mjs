import { mkdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const outDir = path.join(root, 'dist-worker');

await mkdir(outDir, { recursive: true });

await build({
  entryPoints: [path.join(root, 'scripts', 'worker.js')],
  outfile: path.join(outDir, 'worker.js'),
  bundle: true,
  format: 'esm',
  platform: 'neutral',
  target: 'es2022',
  mainFields: ['module', 'main'],
  alias: {
    '@drtechcares/smart-biometric-core': path.join(root, 'packages/smart-biometric-core/src/index.ts'),
    '@drtechcares/api-client': path.join(root, 'packages/api-client/src/index.ts'),
    '@drtechcares/pose-engine': path.join(root, 'packages/pose-engine/src/index.ts'),
    '@drtechcares/clinical-i18n': path.join(root, 'packages/clinical-i18n/src/index.ts'),
  },
});

console.log(`[build-worker] Bundled Cloudflare Worker -> ${path.join(outDir, 'worker.js')}`);

