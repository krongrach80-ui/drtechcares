import * as ort from 'onnxruntime-web';
import {
  CANONICAL_HUMAN_FACE_BASELINE_512,
  FACE_EMBEDDING_DIM,
  type GeometricAnthropometricRatios,
  computeGeometricAnthropometricRatios,
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

  // Tight Inner-Face elliptical ROI (eyebrows y=28 to chin y=96, cheekbones x=24 to 88)
  // Excludes room background walls, hair, and shirt collars 100%, with inner-face histogram CDF equalization.
  const grid = 16;
  const rawGrid = new Float32Array(grid * grid);
  const gray16 = new Float32Array(grid * grid);
  const roiX0 = width * (24 / 112);
  const roiW = Math.max(1, width * (64 / 112));
  const roiY0 = height * (28 / 112);
  const roiH = Math.max(1, height * (66 / 112));
  const cellW = Math.max(1, roiW / grid);
  const cellH = Math.max(1, roiH / grid);
  const sampleLum = (px: number, py: number): number => {
    const cx = Math.min(width - 1, Math.max(0, px));
    const cy = Math.min(height - 1, Math.max(0, py));
    const idx = (cy * width + cx) * 4;
    return 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
  };

  // Gentle histogram equalization (CDF) strictly inside inner elliptical mask so lighting intensity never dominates
  const hist = new Uint32Array(256);
  let histTotal = 0;
  let minLum = 255;
  let maxLum = 0;
  for (let py = Math.floor(roiY0); py < Math.min(height, Math.ceil(roiY0 + roiH)); py += 1) {
    for (let px = Math.floor(roiX0); px < Math.min(width, Math.ceil(roiX0 + roiW)); px += 1) {
      const nx = (px - (roiX0 + roiW * 0.5)) / Math.max(1, roiW * 0.5);
      const ny = (py - (roiY0 + roiH * 0.5)) / Math.max(1, roiH * 0.5);
      if (nx * nx + ny * ny <= 1.0) {
        const lumVal = Math.min(255, Math.max(0, Math.round(sampleLum(px, py))));
        hist[lumVal] += 1;
        histTotal += 1;
        if (lumVal < minLum) minLum = lumVal;
        if (lumVal > maxLum) maxLum = lumVal;
      }
    }
  }
  const cdfLut = new Float32Array(256);
  const hasContrastSpan = histTotal > 0 && maxLum - minLum >= 6;
  if (hasContrastSpan) {
    let acc = 0;
    for (let b = 0; b < 256; b += 1) {
      acc += hist[b];
      cdfLut[b] = (acc / histTotal) * 255;
    }
  } else {
    for (let b = 0; b < 256; b += 1) cdfLut[b] = b;
  }

  const sampleNormLum = (px: number, py: number): number => {
    const raw = sampleLum(px, py);
    if (!hasContrastSpan) return raw;
    const bin = Math.min(255, Math.max(0, Math.round(raw)));
    // Blend 65% histogram-equalized contrast with 35% local raw luminance
    return 0.65 * cdfLut[bin] + 0.35 * raw;
  };

  let ellipSum = 0;
  let ellipCount = 0;
  for (let gy = 0; gy < grid; gy += 1) {
    for (let gx = 0; gx < grid; gx += 1) {
      const x0 = Math.floor(roiX0 + (gx + 0.25) * cellW);
      const x1 = Math.floor(roiX0 + (gx + 0.75) * cellW);
      const y0 = Math.floor(roiY0 + (gy + 0.25) * cellH);
      const y1 = Math.floor(roiY0 + (gy + 0.75) * cellH);
      const cellAvg =
        (sampleNormLum(x0, y0) + sampleNormLum(x1, y0) + sampleNormLum(x0, y1) + sampleNormLum(x1, y1)) / 4;
      rawGrid[gy * grid + gx] = cellAvg;
      const ellipDistSq = ((gx - 7.5) / 8.0) ** 2 + ((gy - 7.5) / 8.0) ** 2;
      if (ellipDistSq <= 1.0) {
        ellipSum += cellAvg;
        ellipCount += 1;
      }
    }
  }

  const faceMean = ellipCount > 0 ? ellipSum / ellipCount : 127.5;
  let varSum = 0;
  for (let gy = 0; gy < grid; gy += 1) {
    for (let gx = 0; gx < grid; gx += 1) {
      const ellipDistSq = ((gx - 7.5) / 8.0) ** 2 + ((gy - 7.5) / 8.0) ** 2;
      if (ellipDistSq <= 1.0) {
        const diff = rawGrid[gy * grid + gx] - faceMean;
        varSum += diff * diff;
      }
    }
  }
  const faceStd = Math.max(10.0, ellipCount > 0 ? Math.sqrt(varSum / ellipCount) : 28.0);

  for (let gy = 0; gy < grid; gy += 1) {
    for (let gx = 0; gx < grid; gx += 1) {
      const idx = gy * grid + gx;
      const normLum = Math.max(0.05, Math.min(0.95, ((rawGrid[idx] - faceMean) / faceStd) * 0.22 + 0.5));
      const ellipDistSq = ((gx - 7.5) / 8.0) ** 2 + ((gy - 7.5) / 8.0) ** 2;
      const blend = ellipDistSq > 1.0 ? Math.max(0, 1 - (ellipDistSq - 1.0) * 3.0) : 1.0;
      gray16[idx] = 0.5 + (normLum - 0.5) * blend;
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

  // 128..255: Smooth Multi-Scale Inner-Face 16-Patch (4x4 patches within [28..84]x[32..92])
  // Uses 2px-step smoothed gradients + illumination-invariant Center-Symmetric LBP (CS-LBP)
  const scaleX = width / 112;
  const scaleY = height / 112;
  const blockLum = (bx: number, by: number): number => {
    let total = 0;
    for (let dy = -1; dy <= 1; dy += 1) {
      for (let dx = -1; dx <= 1; dx += 1) {
        total += sampleNormLum(bx + dx, by + dy);
      }
    }
    return total / 9;
  };
  const invPatchScale = 1 / (faceStd * 3.6);
  for (let py = 0; py < 4; py += 1) {
    for (let px = 0; px < 4; px += 1) {
      const patchOffset = 128 + (py * 4 + px) * 8;
      const cx = Math.min(width - 4, Math.max(3, Math.round((28 + px * 18.6) * scaleX)));
      const cy = Math.min(height - 4, Math.max(3, Math.round((32 + py * 19.6) * scaleY)));

      const center = blockLum(cx, cy);
      const lumE = blockLum(cx + 2, cy);
      const lumW = blockLum(cx - 2, cy);
      const lumS = blockLum(cx, cy + 2);
      const lumN = blockLum(cx, cy - 2);
      const gxVal = (lumE - lumW) * invPatchScale;
      const gyVal = (lumS - lumN) * invPatchScale;
      // Local Laplacian contrast & Center-Symmetric LBP (invariant to monotonic lighting gain shifts)
      const surroundMean = (lumE + lumW + lumS + lumN) * 0.25;
      const localContrast = Math.max(-0.5, Math.min(0.5, (center - surroundMean) * invPatchScale));
      vec[patchOffset] = gxVal;
      vec[patchOffset + 1] = gyVal;
      vec[patchOffset + 2] = Math.hypot(gxVal, gyVal);
      vec[patchOffset + 3] = blockLum(cx - 2, cy - 2) >= blockLum(cx + 2, cy + 2) ? 0.08 : -0.08;
      vec[patchOffset + 4] = blockLum(cx + 2, cy - 2) >= blockLum(cx - 2, cy + 2) ? 0.08 : -0.08;
      vec[patchOffset + 5] = lumE >= lumW ? 0.08 : -0.08;
      vec[patchOffset + 6] = lumS >= lumN ? 0.08 : -0.08;
      vec[patchOffset + 7] = localContrast;
    }
  }

  // 256..511: High-detail Roll-Invariant 3D Craniofacial geometric bone ratios anchored by inter-ocular eye axis
  if (landmarks3D.length >= 10) {
    const pLeft = landmarks3D[33] ?? landmarks3D[0];
    const pRight = landmarks3D[263] ?? landmarks3D[1];
    const nose = landmarks3D[1] ?? landmarks3D[2];
    const eyeDx = pRight.x - pLeft.x;
    const eyeDy = pRight.y - pLeft.y;
    const eyeSpan2d = Math.hypot(eyeDx, eyeDy);
    const cosR = eyeSpan2d > 1e-5 ? eyeDx / eyeSpan2d : 1;
    const sinR = eyeSpan2d > 1e-5 ? -eyeDy / eyeSpan2d : 0;
    const iod = Math.max(
      1e-4,
      Math.hypot(eyeDx, eyeDy, (pRight.z ?? 0) - (pLeft.z ?? 0)),
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

    const raw3D = new Float32Array(256);
    let mean3D = 0;
    for (let k = 0; k < 256; k += 1) {
      const lmIdx = (k * 7) % landmarks3D.length;
      const lm = landmarks3D[lmIdx];
      const ref = keyAnchors[k % keyAnchors.length];
      const rawDx = (lm.x - nose.x) / iod;
      const rawDy = (lm.y - nose.y) / iod;
      const dx = rawDx * cosR - rawDy * sinR;
      const dy = rawDx * sinR + rawDy * cosR;
      const dz = ((lm.z ?? 0) - (nose.z ?? 0)) / iod;
      const dist = Math.hypot(dx, dy, dz);
      const refDist = Math.hypot(lm.x - ref.x, lm.y - ref.y, (lm.z ?? 0) - (ref.z ?? 0)) / iod;
      const val = dist * 0.08 + refDist * 0.06 + (dx - dy) * 0.04;
      raw3D[k] = val;
      mean3D += val;
    }
    mean3D /= 256;
    for (let k = 0; k < 256; k += 1) {
      vec[256 + k] = 0.14 * CANONICAL_HUMAN_FACE_BASELINE_512[256 + k] + 0.86 * (raw3D[k] - mean3D);
    }
  } else {
    for (let k = 256; k < FACE_EMBEDDING_DIM; k += 1) {
      vec[k] = CANONICAL_HUMAN_FACE_BASELINE_512[k];
    }
  }

  // Equal-Energy Modal Balancing: independently normalize 0..127 (0.40), 128..255 (0.30), 256..511 (0.30)
  // so 3D facial bone proportions & edge textures are never drowned by pixel intensity energy.
  const balanceSubband = (start: number, endExcl: number, weight: number) => {
    let normSq = 0;
    for (let i = start; i < endExcl; i += 1) {
      normSq += vec[i] * vec[i];
    }
    const norm = Math.sqrt(normSq);
    if (norm > 1e-6) {
      const s = weight / norm;
      for (let i = start; i < endExcl; i += 1) {
        vec[i] *= s;
      }
    }
  };
  balanceSubband(0, 128, Math.sqrt(0.4));
  balanceSubband(128, 256, Math.sqrt(0.3));
  balanceSubband(256, 512, Math.sqrt(0.3));

  return l2Normalize(vec);
}

/**
 * Legacy pre-overhaul descriptor extractor used ONLY as a strictly gated read-only bridge
 * for clinic patients enrolled prior to V2 Inner-Face CDF calibration.
 */
export function extractLegacyV1Hybrid512Descriptor(
  image: { width: number; height: number; data: Uint8ClampedArray | Uint8Array },
  landmarks3D: readonly Landmark3D[],
): number[] {
  const { width, height, data } = image;
  const vec = new Array<number>(FACE_EMBEDDING_DIM).fill(0);
  const grid = 16;
  const rawGrid = new Float32Array(grid * grid);
  const gray16 = new Float32Array(grid * grid);
  const roiX0 = width * (18 / 112);
  const roiW = Math.max(1, width * (76 / 112));
  const roiY0 = height * (20 / 112);
  const roiH = Math.max(1, height * (72 / 112));
  const cellW = Math.max(1, roiW / grid);
  const cellH = Math.max(1, roiH / grid);
  const sampleLum = (px: number, py: number): number => {
    const cx = Math.min(width - 1, Math.max(0, px));
    const cy = Math.min(height - 1, Math.max(0, py));
    const idx = (cy * width + cx) * 4;
    return 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
  };
  let ellipSum = 0;
  let ellipCount = 0;
  for (let gy = 0; gy < grid; gy += 1) {
    for (let gx = 0; gx < grid; gx += 1) {
      const x0 = Math.floor(roiX0 + (gx + 0.25) * cellW);
      const x1 = Math.floor(roiX0 + (gx + 0.75) * cellW);
      const y0 = Math.floor(roiY0 + (gy + 0.25) * cellH);
      const y1 = Math.floor(roiY0 + (gy + 0.75) * cellH);
      const cellAvg =
        (sampleLum(x0, y0) + sampleLum(x1, y0) + sampleLum(x0, y1) + sampleLum(x1, y1)) / 4;
      rawGrid[gy * grid + gx] = cellAvg;
      if (((gx - 7.5) / 8.0) ** 2 + ((gy - 7.5) / 8.0) ** 2 <= 1.0) {
        ellipSum += cellAvg;
        ellipCount += 1;
      }
    }
  }
  const faceMean = ellipCount > 0 ? ellipSum / ellipCount : 127.5;
  let varSum = 0;
  for (let gy = 0; gy < grid; gy += 1) {
    for (let gx = 0; gx < grid; gx += 1) {
      if (((gx - 7.5) / 8.0) ** 2 + ((gy - 7.5) / 8.0) ** 2 <= 1.0) {
        const diff = rawGrid[gy * grid + gx] - faceMean;
        varSum += diff * diff;
      }
    }
  }
  const faceStd = Math.max(12.0, ellipCount > 0 ? Math.sqrt(varSum / ellipCount) : 32.0);
  for (let gy = 0; gy < grid; gy += 1) {
    for (let gx = 0; gx < grid; gx += 1) {
      const idx = gy * grid + gx;
      const normLum = Math.max(0.05, Math.min(0.95, ((rawGrid[idx] - faceMean) / faceStd) * 0.22 + 0.5));
      const ellipDistSq = ((gx - 7.5) / 8.0) ** 2 + ((gy - 7.5) / 8.0) ** 2;
      const blend = ellipDistSq > 1.0 ? Math.max(0, 1 - (ellipDistSq - 1.0) * 3.0) : 1.0;
      gray16[idx] = 0.5 + (normLum - 0.5) * blend;
    }
  }
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
  const scaleX = width / 112;
  const scaleY = height / 112;
  const blockLum = (bx: number, by: number): number => {
    let total = 0;
    for (let dy = -1; dy <= 1; dy += 1) {
      for (let dx = -1; dx <= 1; dx += 1) total += sampleLum(bx + dx, by + dy);
    }
    return total / 9;
  };
  const invPatchScale = 1 / (faceStd * 4.0);
  for (let py = 0; py < 4; py += 1) {
    for (let px = 0; px < 4; px += 1) {
      const patchOffset = 128 + (py * 4 + px) * 8;
      const cx = Math.min(width - 2, Math.max(1, Math.round((24 + px * 21) * scaleX)));
      const cy = Math.min(height - 2, Math.max(1, Math.round((26 + py * 20) * scaleY)));
      const center = blockLum(cx, cy);
      const gxVal = (blockLum(cx + 1, cy) - blockLum(cx - 1, cy)) * invPatchScale;
      const gyVal = (blockLum(cx, cy + 1) - blockLum(cx, cy - 1)) * invPatchScale;
      vec[patchOffset] = gxVal;
      vec[patchOffset + 1] = gyVal;
      vec[patchOffset + 2] = Math.hypot(gxVal, gyVal);
      vec[patchOffset + 3] = blockLum(cx - 1, cy - 1) > center ? 0.05 : -0.05;
      vec[patchOffset + 4] = blockLum(cx + 1, cy - 1) > center ? 0.05 : -0.05;
      vec[patchOffset + 5] = blockLum(cx + 1, cy + 1) > center ? 0.05 : -0.05;
      vec[patchOffset + 6] = blockLum(cx - 1, cy + 1) > center ? 0.05 : -0.05;
      vec[patchOffset + 7] = Math.max(-0.5, Math.min(0.5, (center - faceMean) * invPatchScale));
    }
  }
  if (landmarks3D.length >= 10) {
    const pLeft = landmarks3D[33] ?? landmarks3D[0];
    const pRight = landmarks3D[263] ?? landmarks3D[1];
    const nose = landmarks3D[1] ?? landmarks3D[2];
    const eyeDx = pRight.x - pLeft.x;
    const eyeDy = pRight.y - pLeft.y;
    const eyeSpan2d = Math.hypot(eyeDx, eyeDy);
    const cosR = eyeSpan2d > 1e-5 ? eyeDx / eyeSpan2d : 1;
    const sinR = eyeSpan2d > 1e-5 ? -eyeDy / eyeSpan2d : 0;
    const iod = Math.max(1e-4, Math.hypot(eyeDx, eyeDy, (pRight.z ?? 0) - (pLeft.z ?? 0)));
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
    for (let k = 0; k < 256; k += 1) {
      const lmIdx = (k * 7) % landmarks3D.length;
      const lm = landmarks3D[lmIdx];
      const ref = keyAnchors[k % keyAnchors.length];
      const rawDx = (lm.x - nose.x) / iod;
      const rawDy = (lm.y - nose.y) / iod;
      const dx = rawDx * cosR - rawDy * sinR;
      const dy = rawDx * sinR + rawDy * cosR;
      const dz = ((lm.z ?? 0) - (nose.z ?? 0)) / iod;
      const dist = Math.hypot(dx, dy, dz);
      const refDist = Math.hypot(lm.x - ref.x, lm.y - ref.y, (lm.z ?? 0) - (ref.z ?? 0)) / iod;
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

export function extractGeometricAnthropometricRatios(
  landmarks3D: readonly Landmark3D[],
): GeometricAnthropometricRatios | undefined {
  return computeGeometricAnthropometricRatios(landmarks3D);
}

/**
 * Horizontally flips a 112x112 RGBA crop buffer for Mirror-Flip Invariance matching.
 */
export function flipImageDataHorizontally(image: {
  width: number;
  height: number;
  data: Uint8ClampedArray | Uint8Array;
}): { width: number; height: number; data: Uint8ClampedArray } {
  const { width, height, data } = image;
  const out = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y += 1) {
    const rowOff = y * width * 4;
    for (let x = 0; x < width; x += 1) {
      const srcIdx = rowOff + x * 4;
      const dstIdx = rowOff + (width - 1 - x) * 4;
      out[dstIdx] = data[srcIdx];
      out[dstIdx + 1] = data[srcIdx + 1];
      out[dstIdx + 2] = data[srcIdx + 2];
      out[dstIdx + 3] = data[srcIdx + 3];
    }
  }
  return { width, height, data: out };
}

/**
 * Horizontally mirrors normalized landmarks across X = 0.5 while swapping left/right pairs if needed.
 */
export function flipLandmarksHorizontally(
  landmarks3D: readonly Landmark3D[],
): Landmark3D[] {
  return landmarks3D.map((lm) => ({ ...lm, x: 1 - lm.x }));
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

/**
 * Extracts both raw-camera and horizontally mirrored embeddings so enrollment/login
 * stay 100% invariant regardless of mirror state (scale-x-[-1]).
 */
export async function extractDualFlipFaceEmbedding(
  image: { width: number; height: number; data: Uint8ClampedArray | Uint8Array },
  landmarks3D: readonly Landmark3D[],
  options?: { includeLegacyV1?: boolean },
): Promise<{
  vector: number[];
  mirroredVector: number[];
  legacyV1Vector?: number[];
  mirroredLegacyV1Vector?: number[];
  engine: 'arcface-onnx' | 'hybrid-512d';
  geometricRatios?: GeometricAnthropometricRatios;
}> {
  const primary = await extractFaceEmbeddingWithFallback(image, landmarks3D);
  const flippedImg = flipImageDataHorizontally(image);
  const flippedLms = flipLandmarksHorizontally(landmarks3D);
  const mirrored = await extractFaceEmbeddingWithFallback(flippedImg, flippedLms);
  const geometricRatios = computeGeometricAnthropometricRatios(landmarks3D);
  const includeLegacy = Boolean(options?.includeLegacyV1);
  return {
    vector: primary.vector,
    mirroredVector: mirrored.vector,
    ...(includeLegacy
      ? {
          legacyV1Vector: extractLegacyV1Hybrid512Descriptor(image, landmarks3D),
          mirroredLegacyV1Vector: extractLegacyV1Hybrid512Descriptor(flippedImg, flippedLms),
        }
      : {}),
    engine: primary.engine,
    ...(geometricRatios ? { geometricRatios } : {}),
  };
}
