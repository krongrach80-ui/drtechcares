import * as ort from 'onnxruntime-web';
import {
  CANONICAL_HUMAN_FACE_BASELINE_512,
  FACE_EMBEDDING_DIM,
  l2Normalize,
} from '@drtechcares/smart-biometric-core';
import { ARCFACE_INPUT_SIZE } from './align';

const configuredUrl: unknown =
  typeof import.meta !== 'undefined' && import.meta.env
    ? import.meta.env.VITE_ARCFACE_MODEL_URL
    : undefined;

const configuredWasmPaths: unknown =
  typeof import.meta !== 'undefined' && import.meta.env
    ? import.meta.env.VITE_ORT_WASM_PATHS
    : undefined;

if (typeof configuredWasmPaths === 'string' && configuredWasmPaths.length > 0) {
  ort.env.wasm.wasmPaths = configuredWasmPaths;
}

export const ARCFACE_MODEL_URL: string =
  typeof configuredUrl === 'string' && configuredUrl.length > 0
    ? configuredUrl
    : '/models/w600k_r50.onnx';

const heavyOnnxEnabled: boolean =
  typeof import.meta !== 'undefined' &&
  Boolean(import.meta.env) &&
  import.meta.env.VITE_ENABLE_HEAVY_ARCFACE_ONNX === 'true';

let arcfaceSessionPromise: Promise<ort.InferenceSession> | null = null;
let arcfaceUnavailable = !heavyOnnxEnabled;

export function getArcFaceSession(): Promise<ort.InferenceSession> {
  if (arcfaceUnavailable) {
    return Promise.reject(new Error(`ArcFace model unavailable at ${ARCFACE_MODEL_URL}`));
  }
  if (!arcfaceSessionPromise) {
    arcfaceSessionPromise = (async () => {
      if (typeof window !== 'undefined' && typeof fetch === 'function') {
        try {
          const headRes = await fetch(ARCFACE_MODEL_URL, { method: 'HEAD' });
          const contentType = headRes.headers.get('content-type') ?? '';
          if (headRes.status !== 405 && (!headRes.ok || contentType.includes('text/html'))) {
            arcfaceUnavailable = true;
            throw new Error(
              `ArcFace model unavailable at ${ARCFACE_MODEL_URL} (status=${headRes.status}, content-type=${contentType || 'unknown'})`,
            );
          }
        } catch (err: unknown) {
          if (err instanceof Error && err.message.startsWith('ArcFace model unavailable')) {
            throw err;
          }
        }
      }
      return ort.InferenceSession.create(ARCFACE_MODEL_URL, {
        executionProviders: ['wasm'],
      });
    })().catch((err: unknown) => {
      arcfaceUnavailable = true;
      arcfaceSessionPromise = null;
      throw err;
    });
  }
  return arcfaceSessionPromise;
}

export function toArcFaceNchwTensor(image: { width: number; height: number; data: Uint8ClampedArray | Uint8Array }): Float32Array {
  const { width, height, data } = image;
  const plane = width * height;
  const out = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i += 1) {
    const p = i * 4;
    out[i] = (data[p] - 127.5) / 128.0;
    out[plane + i] = (data[p + 1] - 127.5) / 128.0;
    out[2 * plane + i] = (data[p + 2] - 127.5) / 128.0;
  }
  return out;
}

/**
 * Extracts a 512-D L2-normalized embedding from a 112x112 aligned face crop
 * using the browser-hosted ArcFace ONNX model (w600k_r50.onnx).
 */
export async function extractArcFaceOnnxEmbedding(
  image: { width: number; height: number; data: Uint8ClampedArray | Uint8Array },
): Promise<number[]> {
  if (image.width !== ARCFACE_INPUT_SIZE || image.height !== ARCFACE_INPUT_SIZE) {
    throw new RangeError(`Expected ${ARCFACE_INPUT_SIZE}x${ARCFACE_INPUT_SIZE} image, got ${image.width}x${image.height}`);
  }
  const session = await getArcFaceSession();
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  if (!inputName || !outputName) {
    throw new Error('Invalid ArcFace ONNX model signature');
  }
  const tensor = new ort.Tensor('float32', toArcFaceNchwTensor(image), [
    1,
    3,
    ARCFACE_INPUT_SIZE,
    ARCFACE_INPUT_SIZE,
  ]);
  const results = await session.run({ [inputName]: tensor });
  const output = results[outputName] as ort.Tensor | undefined;
  if (!output) {
    throw new Error('ArcFace ONNX inference returned empty output');
  }
  const arr = Array.from(output.data as Float32Array);
  if (arr.length !== FACE_EMBEDDING_DIM) {
    throw new Error(`Expected ${FACE_EMBEDDING_DIM}-D output, got ${arr.length}`);
  }
  return l2Normalize(arr);
}

export interface Landmark3D {
  x: number;
  y: number;
  z?: number;
}

/**
 * Computes Hybrid 512-D Descriptor:
 * - 0..127: 2D DCT low-frequency appearance descriptors from 112x112 aligned face
 * - 128..255: 16-Patch gradient & local binary pattern (LBP) descriptors
 * - 256..511: 3D craniofacial geometric ratios from MediaPipe FaceLandmarker
 */
export function extractHybrid512Descriptor(
  image: { width: number; height: number; data: Uint8ClampedArray | Uint8Array },
  landmarks3D: readonly Landmark3D[],
): number[] {
  const { width, height, data } = image;
  const vec = new Array<number>(FACE_EMBEDDING_DIM).fill(0);

  // Downsample grayscale to 16x16 grid with 2x2 cell averaging for noise-resistant DCT (0..127)
  const grid = 16;
  const gray16 = new Float32Array(grid * grid);
  const cellW = Math.max(1, width / grid);
  const cellH = Math.max(1, height / grid);
  const sampleLum = (px: number, py: number): number => {
    const idx = (py * width + px) * 4;
    return 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
  };

  for (let gy = 0; gy < grid; gy += 1) {
    for (let gx = 0; gx < grid; gx += 1) {
      const x0 = Math.min(width - 1, Math.floor((gx + 0.25) * cellW));
      const x1 = Math.min(width - 1, Math.floor((gx + 0.75) * cellW));
      const y0 = Math.min(height - 1, Math.floor((gy + 0.25) * cellH));
      const y1 = Math.min(height - 1, Math.floor((gy + 0.75) * cellH));
      gray16[gy * grid + gx] =
        (sampleLum(x0, y0) + sampleLum(x1, y0) + sampleLum(x0, y1) + sampleLum(x1, y1)) / (4 * 255.0);
    }
  }

  // 0..127: 2D DCT coefficients (11x12 low-frequency block excluding DC)
  let dctIdx = 0;
  for (let u = 0; u < 12 && dctIdx < 128; u += 1) {
    for (let v = 0; v < 12 && dctIdx < 128; v += 1) {
      if (u === 0 && v === 0) continue;
      let sum = 0;
      for (let y = 0; y < grid; y += 1) {
        for (let x = 0; x < grid; x += 1) {
          sum +=
            gray16[y * grid + x] *
            Math.cos(((2 * x + 1) * u * Math.PI) / (2 * grid)) *
            Math.cos(((2 * y + 1) * v * Math.PI) / (2 * grid));
        }
      }
      vec[dctIdx] = sum / grid;
      dctIdx += 1;
    }
  }

  // 128..255: 16-Patch (4x4 patches x 8 gradient/LBP bins)
  const patchW = Math.max(2, Math.floor(width / 4));
  const patchH = Math.max(2, Math.floor(height / 4));
  for (let py = 0; py < 4; py += 1) {
    for (let px = 0; px < 4; px += 1) {
      const patchOffset = 128 + (py * 4 + px) * 8;
      const cx = Math.min(width - 2, Math.max(1, px * patchW + Math.floor(patchW / 2)));
      const cy = Math.min(height - 2, Math.max(1, py * patchH + Math.floor(patchH / 2)));

      const lum = (x: number, y: number): number => {
        const p = (y * width + x) * 4;
        return 0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2];
      };

      const center = lum(cx, cy);
      const gxVal = (lum(cx + 1, cy) - lum(cx - 1, cy)) / 255.0;
      const gyVal = (lum(cx, cy + 1) - lum(cx, cy - 1)) / 255.0;
      vec[patchOffset] = gxVal;
      vec[patchOffset + 1] = gyVal;
      vec[patchOffset + 2] = Math.hypot(gxVal, gyVal);
      vec[patchOffset + 3] = lum(cx - 1, cy - 1) > center ? 0.1 : -0.1;
      vec[patchOffset + 4] = lum(cx + 1, cy - 1) > center ? 0.1 : -0.1;
      vec[patchOffset + 5] = lum(cx + 1, cy + 1) > center ? 0.1 : -0.1;
      vec[patchOffset + 6] = lum(cx - 1, cy + 1) > center ? 0.1 : -0.1;
      vec[patchOffset + 7] = center / 255.0 - 0.5;
    }
  }

  // 256..511: High-detail 3D Craniofacial geometric ratios anchored by inter-ocular distance & key anatomical landmarks
  if (landmarks3D.length >= 10) {
    const pLeft = landmarks3D[33] ?? landmarks3D[0];
    const pRight = landmarks3D[263] ?? landmarks3D[1];
    const nose = landmarks3D[1] ?? landmarks3D[2];
    const iod = Math.max(
      1e-4,
      Math.hypot(pRight.x - pLeft.x, pRight.y - pLeft.y, (pRight.z ?? 0) - (pLeft.z ?? 0)),
    );
    const keyAnchors = [
      landmarks3D[33] ?? nose,
      landmarks3D[133] ?? nose,
      landmarks3D[362] ?? nose,
      landmarks3D[263] ?? nose,
      landmarks3D[4] ?? nose,
      landmarks3D[168] ?? nose,
      landmarks3D[61] ?? nose,
      landmarks3D[291] ?? nose,
      landmarks3D[152] ?? nose,
      landmarks3D[234] ?? nose,
      landmarks3D[454] ?? nose,
    ];

    const anchorPairs = 256;
    for (let k = 0; k < anchorPairs; k += 1) {
      const lmIdx = (k * 7) % landmarks3D.length;
      const lm = landmarks3D[lmIdx];
      const ref = keyAnchors[k % keyAnchors.length];
      const dx = (lm.x - nose.x) / iod;
      const dy = (lm.y - nose.y) / iod;
      const dz = ((lm.z ?? 0) - (nose.z ?? 0)) / iod;
      const dist = Math.hypot(dx, dy, dz);
      const refDist = Math.hypot(lm.x - ref.x, lm.y - ref.y, (lm.z ?? 0) - (ref.z ?? 0)) / iod;
      // Combine with common-mode canonical baseline structure so whitening highlights true deviation
      vec[256 + k] =
        0.65 * CANONICAL_HUMAN_FACE_BASELINE_512[256 + k] +
        0.35 * (dist * 0.04 + refDist * 0.03 + (dx - dy) * 0.02);
    }
  } else {
    for (let k = 256; k < FACE_EMBEDDING_DIM; k += 1) {
      vec[k] = CANONICAL_HUMAN_FACE_BASELINE_512[k];
    }
  }

  return l2Normalize(vec);
}

/**
 * Attempts ArcFace ONNX inference first, falling back seamlessly to Hybrid 512-D Descriptor
 * when ONNX weights are stripped (e.g. Cloudflare Workers edge fallback without R2 URL).
 */
export async function extractFaceEmbeddingWithFallback(
  image: { width: number; height: number; data: Uint8ClampedArray | Uint8Array },
  landmarks3D: readonly Landmark3D[],
): Promise<{ vector: number[]; engine: 'arcface-onnx' | 'hybrid-512d' }> {
  if (arcfaceUnavailable) {
    return { vector: extractHybrid512Descriptor(image, landmarks3D), engine: 'hybrid-512d' };
  }
  try {
    const vector = await extractArcFaceOnnxEmbedding(image);
    return { vector, engine: 'arcface-onnx' };
  } catch {
    arcfaceUnavailable = true;
    const vector = extractHybrid512Descriptor(image, landmarks3D);
    return { vector, engine: 'hybrid-512d' };
  }
}
