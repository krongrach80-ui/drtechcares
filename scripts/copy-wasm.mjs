import { existsSync } from 'node:fs';
import { cp, copyFile, mkdir, rm } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const nodeWasmSrc = path.join(root, 'node_modules', '@mediapipe', 'tasks-vision', 'wasm');
const mediapipeWasmDest = path.join(root, 'public', 'mediapipe', 'wasm');
const modelsWasmDest = path.join(root, 'public', 'models', 'wasm');
const modelsDest = path.join(root, 'public', 'models');

await mkdir(modelsDest, { recursive: true });

if (existsSync(nodeWasmSrc)) {
  for (const dest of [mediapipeWasmDest, modelsWasmDest]) {
    await rm(dest, { recursive: true, force: true });
    await mkdir(dest, { recursive: true });
    await cp(nodeWasmSrc, dest, { recursive: true });
  }
  console.log(`[copy-wasm] Copied MediaPipe WASM files to ${mediapipeWasmDest} and ${modelsWasmDest}`);
} else {
  console.warn('[copy-wasm] node_modules/@mediapipe/tasks-vision/wasm not found yet.');
}

const envCacheDir = process.env.DRTECH_MODEL_CACHE_DIR?.trim();

const candidateAssets = [
  {
    name: 'w600k_r50.onnx',
    sources: [
      ...(envCacheDir ? [path.join(envCacheDir, 'w600k_r50.onnx')] : []),
      'C:\\Users\\Administrator\\drtech-kiosk\\public\\models\\w600k_r50.onnx',
    ],
    dest: path.join(modelsDest, 'w600k_r50.onnx'),
  },
  {
    name: 'det_10g.onnx',
    sources: [
      ...(envCacheDir ? [path.join(envCacheDir, 'det_10g.onnx')] : []),
      'C:\\Users\\Administrator\\drtech-kiosk\\public\\models\\det_10g.onnx',
    ],
    dest: path.join(modelsDest, 'det_10g.onnx'),
  },
  {
    name: 'pose_landmarker_full.task',
    sources: [
      ...(envCacheDir ? [path.join(envCacheDir, 'pose_landmarker_full.task')] : []),
      'C:\\Users\\Administrator\\drtech-kiosk\\public\\models\\pose_landmarker_full.task',
    ],
    dest: path.join(modelsDest, 'pose_landmarker_full.task'),
  },
  {
    name: 'pose_landmarker.task',
    sources: [
      ...(envCacheDir ? [path.join(envCacheDir, 'pose_landmarker.task'), path.join(envCacheDir, 'pose_landmarker_full.task')] : []),
      'C:\\Users\\Administrator\\drtech-kiosk\\public\\models\\pose_landmarker.task',
      'C:\\Users\\Administrator\\drtech-kiosk\\public\\models\\pose_landmarker_full.task',
    ],
    dest: path.join(modelsDest, 'pose_landmarker.task'),
  },
  {
    name: 'face_landmarker.task',
    sources: [
      ...(envCacheDir ? [path.join(envCacheDir, 'face_landmarker.task')] : []),
      'C:\\Users\\Administrator\\touchless-face-login\\frontend\\public\\mediapipe\\face_landmarker.task',
    ],
    dest: path.join(modelsDest, 'face_landmarker.task'),
  },
  {
    name: 'hand_landmarker.task',
    sources: [
      ...(envCacheDir ? [path.join(envCacheDir, 'hand_landmarker.task')] : []),
      'C:\\Users\\Administrator\\drtech-kiosk\\public\\models\\hand_landmarker.task',
    ],
    dest: path.join(modelsDest, 'hand_landmarker.task'),
  },
];

const present = [];
const missing = [];

for (const asset of candidateAssets) {
  if (!existsSync(asset.dest)) {
    const matchedSrc = asset.sources.find((src) => existsSync(src));
    if (matchedSrc) {
      await copyFile(matchedSrc, asset.dest);
      console.log(`[copy-wasm] Provisioned local model asset: ${asset.name}`);
    }
  }
  if (existsSync(asset.dest)) {
    present.push(asset.name);
  } else {
    missing.push(asset.name);
  }
}

console.log(`[copy-wasm] Model inventory ready (${present.length}/${candidateAssets.length}): ${present.join(', ')}`);
if (missing.length > 0) {
  console.warn(
    `[copy-wasm] Optional missing model files in public/models (${missing.join(', ')}). Set DRTECH_MODEL_CACHE_DIR or VITE_ARCFACE_MODEL_URL if needed.`,
  );
}
