import { existsSync } from 'node:fs';
import { readdir, rm, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const MAX_CF_ASSET_BYTES = 24 * 1024 * 1024;
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

const targetDirs = [
  { dir: path.join(root, 'dist', 'models'), extensions: ['.onnx'] },
  { dir: path.join(root, 'dist', 'assets'), extensions: ['.wasm', '.onnx'] },
];

for (const { dir, extensions } of targetDirs) {
  if (!existsSync(dir)) continue;
  const entries = await readdir(dir, { withFileTypes: true });
  for (const entry of entries) {
    if (!entry.isFile()) continue;
    if (!extensions.some((ext) => entry.name.endsWith(ext))) continue;

    const fullPath = path.join(dir, entry.name);
    const info = await stat(fullPath);
    if (info.size > MAX_CF_ASSET_BYTES) {
      await rm(fullPath, { force: true });
      const statusPath = path.join(dir, `${entry.name}.status.json`);
      await writeFile(
        statusPath,
        JSON.stringify(
          {
            stripped: true,
            fileName: entry.name,
            sizeBytes: info.size,
            fallbackEngine: 'hybrid-512d',
            envOverride: entry.name.endsWith('.wasm')
              ? 'VITE_ORT_WASM_PATHS'
              : 'VITE_ARCFACE_MODEL_URL',
          },
          null,
          2,
        ),
        'utf-8',
      );
      console.log(
        `[strip-models] Stripped oversized asset (>24MiB) from ${path.relative(root, dir)}: ${entry.name} (wrote ${path.basename(statusPath)})`,
      );
    }
  }
}
