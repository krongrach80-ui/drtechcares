import { existsSync } from 'node:fs';
import { readdir, readFile, rm, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const MAX_CF_ASSET_BYTES = 24 * 1024 * 1024;
const CHUNK_BYTES = 16 * 1024 * 1024;
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
      const data = await readFile(fullPath);
      const parts = [];
      for (let offset = 0, idx = 0; offset < data.length; offset += CHUNK_BYTES, idx += 1) {
        const partName = `${entry.name}.part${String(idx).padStart(2, '0')}`;
        const chunk = data.subarray(offset, Math.min(offset + CHUNK_BYTES, data.length));
        await writeFile(path.join(dir, partName), chunk);
        parts.push(partName);
      }

      await rm(fullPath, { force: true });
      const statusPath = path.join(dir, `${entry.name}.status.json`);
      await writeFile(
        statusPath,
        JSON.stringify(
          {
            chunked: true,
            fileName: entry.name,
            sizeBytes: info.size,
            parts,
            contentType: entry.name.endsWith('.wasm')
              ? 'application/wasm'
              : 'application/octet-stream',
          },
          null,
          2,
        ),
        'utf-8',
      );
      console.log(
        `[strip-models] Chunked oversized asset (>24MiB) in ${path.relative(root, dir)}: ${entry.name} -> ${parts.length} parts (${path.basename(statusPath)})`,
      );
    }
  }
}
