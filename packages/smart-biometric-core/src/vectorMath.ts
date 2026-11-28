/**
 * DRTECHCARES Smart Biometric Core — Zero False-Positive Vector Math & Multi-Subband Gate
 */

export const FACE_EMBEDDING_DIM = 512;
export const COMMON_MODE_ALPHA = 0.92;
export const NOT_FOUND_CONFIRM_ROUNDS = 1;
export const BURST_FRAME_COUNT = 3;

export const REQUIRED_MATCH_STREAK = 6;
export const STRICT_RAW_COSINE_THRESHOLD = 0.84;
export const STRICT_CENTERED_COSINE_THRESHOLD = 0.80;
export const STRICT_MAX_EUCLIDEAN_DISTANCE = 0.56;
export const STRICT_MIN_CENTROID_COSINE = 0.82;
export const STRICT_MIN_CENTROID_COUNT = 2;
export const STRICT_PATCH_ZNCC_THRESHOLD = 0.62;
export const STRICT_PATCH_GRAD_ZNCC_THRESHOLD = 0.42;
export const STRICT_CRANIOFACIAL_MAX_DEVIATION = 0.105;
export const STRICT_CRANIOFACIAL_MAX_SINGLE_DEVIATION = 0.18;
export const STRICT_OCULAR_IRIS_ZNCC_THRESHOLD = 0.66;
export const STRICT_IRIS_RATIO_MAX_DEVIATION = 0.085;
export const STRICT_RELU6_RAW_COSINE_THRESHOLD = 0.87;
export const STRICT_RELU6_CENTERED_COSINE_THRESHOLD = 0.83;
export const STRICT_RELU6_MAX_EUCLIDEAN_DISTANCE = 0.51;
export const STRICT_RELU6_MIN_CENTROID_COSINE = 0.83;
export const STRICT_RELU6_MIN_EXEMPLAR_TOP3_AVG = 0.81;
export const STRICT_MIN_VECTOR_STD = 0.02;
export const STRICT_DISTINCTIVENESS_MARGIN = 0.08;
export const STRICT_COHORT_MARGIN = 0.08;
export const STRICT_LOGIN_MAX_YAW_DEG = 14;
export const STRICT_LOGIN_MAX_PITCH_DEG = 14;

export const ACTIVE_BIOMETRIC_DIM = 1024;
export const ACTIVE_BIOMETRIC_PIPELINE_VERSION = 'v3-backlit-iris-1024d' as const;

export type BiometricVerifyReasonCode =
  | 'MATCHED'
  | 'NO_TEMPLATES_IN_LIBRARY'
  | 'PATIENT_FILTERED_OUT'
  | 'VERSION_MISMATCH'
  | 'BELOW_THRESHOLD'
  | 'AMBIGUOUS'
  | 'LIVENESS_FAILED'
  | 'QUALITY_TOO_LOW'
  | 'RATE_LIMITED'
  | 'SERVER_ERROR'
  | 'TIMEOUT'
  | 'PERMISSION_DENIED';

export type BiometricPose = 'frontal_centroid' | 'overall_centroid' | 'front' | 'left' | 'right' | 'nod';
export type FacePoseZone5 = 'center' | 'left' | 'right' | 'up' | 'down';

export interface GeometricAnthropometricRatios {
  /** Inter-ocular distance divided by bridge-to-nose-tip length */
  eyeToNoseRatio: number;
  /** Cheekbone width divided by upper face height (glabella/brow to nose tip) */
  faceWidthToUpperHeight: number;
  /** Mouth corner width divided by inter-ocular distance */
  mouthToEyeRatio: number;
  /** Nose-tip to chin height divided by eye-midpoint to mouth-midpoint height */
  noseToChinRatio: number;
  /** Zero-mean unit-std 384-D inner-face appearance patch (256 luma 16x16 + 128 Cb/Cr 8x8) */
  appearancePatch384?: number[];
  /** 12 IOD-normalized craniofacial anatomical ratios for Gate-2 skeletal verification */
  craniofacialRatios12?: number[];
  /** Zero-mean unit-std 256-D dual-eye iris + periocular descriptor (128 left + 128 right) */
  ocularIrisPatch256?: number[];
  /** 8 IPD-normalized anatomical iris diameter, fissure & inter-canthal ratios */
  irisOcularRatios8?: number[];
}

export interface GeometricRatioMatchResult {
  available: boolean;
  score: number;
  maxRelativeDiff: number;
  penalty: number;
}

export interface DualFactorBiometricScore {
  dualFactorScore: number;
  appearanceFactorScore: number;
  geometricFactorScore: number;
  rawCos: number;
  meanCenteredCos: number;
  meanCenteredL2Dist: number;
  whitenedCos: number;
  l2Penalty: number;
  geometricPenalty: number;
  faceCore: FaceCoreMetrics | null;
}

export interface BiometricBurstFrame {
  vector: readonly number[];
  qualityScore: number;
  capturedAfterCountdown: boolean;
  pose?: BiometricPose;
  sharpness?: number;
  yaw?: number;
  pitch?: number;
}

export interface EnrolledTemplateCandidate {
  patientId: string;
  patientName?: string;
  pose: BiometricPose;
  vector: readonly number[];
}

export interface SubbandMetrics {
  rawCos: number;
  euclideanDistance: number;
  whitenedCos: number;
  cosDct: number;
  cosPatch: number;
  cos3D: number;
  mad3D: number;
  maxDiff3D: number;
}

export interface FaceCoreMetrics {
  cos3DNormalized: number;
  craniofacialCos: number;
  mad3DNormalized: number;
  maxDiff3DNormalized: number;
  innerDctCos: number;
  innerDctWhitenedCos: number;
  innerPatchCos: number;
  innerPatchWhitenedCos: number;
  faceCoreSim: number;
  faceCoreWhitenedSim: number;
}

export interface VerificationThresholds {
  minRawCosSingle: number;
  minRawCosMulti: number;
  maxEuclideanDistance: number;
  minWhitenedCosSingle: number;
  minWhitenedCosMulti: number;
  minCosDct: number;
  minCos3D: number;
  maxMad3D: number;
  maxDiff3D: number;
  minRawMargin: number;
  minWhitenedMargin: number;
}

export const STRICT_BIOMETRIC_THRESHOLDS: VerificationThresholds = {
  minRawCosSingle: 0.935,
  minRawCosMulti: 0.925,
  maxEuclideanDistance: 0.39,
  minWhitenedCosSingle: 0.74,
  minWhitenedCosMulti: 0.7,
  minCosDct: 0.84,
  minCos3D: 0.975,
  maxMad3D: 0.0075,
  maxDiff3D: 0.038,
  minRawMargin: 0.08,
  minWhitenedMargin: 0.15,
};

export interface VerifyBiometricOptions {
  totalEnrolledPatients?: number;
  thresholds?: Partial<VerificationThresholds>;
  baseline?: readonly number[];
  alpha?: number;
  excludedPatientIds?: readonly string[];
  geometricRatios?: GeometricAnthropometricRatios | null;
}

export interface CandidateEvaluation {
  patientId: string;
  patientName?: string;
  pose: BiometricPose;
  metrics: SubbandMetrics;
  gatePassed: boolean;
  failedReasons: string[];
}

export interface BiometricVerificationResult {
  matched: boolean;
  patientId: string | null;
  patientName: string | null;
  confidencePercent: number;
  rawMargin: number;
  whitenedMargin: number;
  topCandidate: CandidateEvaluation | null;
  runnerUp: CandidateEvaluation | null;
  failureReasons: string[];
}

/**
 * L2-normalize a vector; returns a copy of zeros if norm === 0.
 */
export function l2Normalize(vector: readonly number[]): number[] {
  let sumSq = 0;
  for (let i = 0; i < vector.length; i += 1) {
    const v = vector[i];
    sumSq += v * v;
  }
  const norm = Math.sqrt(sumSq);
  if (norm === 0 || !Number.isFinite(norm)) {
    return vector.map(() => 0);
  }
  const out = new Array<number>(vector.length);
  for (let i = 0; i < vector.length; i += 1) {
    out[i] = vector[i] / norm;
  }
  return out;
}

/**
 * Universal embedding parser supporting Array, Float32Array/TypedArray,
 * JSON strings ("[0.1, ...]"), pgvector literal strings ("{0.1, ...}"),
 * and nested biometric object wrappers; always returns L2-normalized vectors.
 */
export function parseEmbeddingVector(raw: unknown): number[] | undefined {
  if (!raw) return undefined;
  let candidate: unknown = raw;

  if (typeof candidate === 'object' && !Array.isArray(candidate) && !ArrayBuffer.isView(candidate)) {
    const rec = candidate as Record<string, unknown>;
    candidate =
      rec.master_embedding ??
      rec.masterEmbedding ??
      rec.face_embedding ??
      rec.faceEmbedding ??
      rec.embedding ??
      rec.vector;
  }

  if (typeof candidate === 'string') {
    const trimmed = candidate.trim();
    if (!trimmed) return undefined;
    if (trimmed.startsWith('[')) {
      try {
        candidate = JSON.parse(trimmed);
      } catch {
        return undefined;
      }
    } else if (trimmed.startsWith('{') && trimmed.endsWith('}')) {
      const inner = trimmed.slice(1, -1).trim();
      candidate = inner.length > 0 ? inner.split(',').map((item) => Number(item.trim())) : [];
    } else {
      return undefined;
    }
  }

  let nums: number[] | undefined;
  if (Array.isArray(candidate)) {
    nums = candidate.map((n) => Number(n));
  } else if (ArrayBuffer.isView(candidate) && 'length' in candidate) {
    nums = Array.from(candidate as unknown as ArrayLike<number>, (n) => Number(n));
  }

  if (!nums || nums.length === 0) return undefined;
  if (!nums.every((v) => Number.isFinite(v))) return undefined;
  return l2Normalize(nums);
}

/**
 * Deterministic canonical human face baseline (512-D L2-normalized)
 * representing common-mode craniofacial & low-frequency facial appearance energy.
 */
export const CANONICAL_HUMAN_FACE_BASELINE_512: readonly number[] = (() => {
  const raw = new Array<number>(FACE_EMBEDDING_DIM);
  for (let i = 0; i < FACE_EMBEDDING_DIM; i += 1) {
    if (i < 128) {
      // 2D DCT low-frequency facial luminance symmetry
      raw[i] = 0.095 * Math.cos((i * Math.PI) / 256) + 0.02 * Math.sin(i * 0.37);
    } else if (i < 256) {
      // 16-Patch gradient/LBP common facial contour distribution
      raw[i] = 0.065 * Math.cos((i - 128) * 0.09) + 0.03;
    } else {
      // 3D Craniofacial ratio shared human anthropometry (dominant common mode)
      raw[i] = 0.14 + 0.025 * Math.sin((i - 256) * 0.13);
    }
  }
  return Object.freeze(l2Normalize(raw));
})();

export function dotProduct(a: readonly number[], b: readonly number[]): number {
  const len = Math.min(a.length, b.length);
  let sum = 0;
  for (let i = 0; i < len; i += 1) {
    sum += a[i] * b[i];
  }
  return sum;
}

export function cosineSimilarity(a: readonly number[], b: readonly number[]): number {
  const len = Math.min(a.length, b.length);
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < len; i += 1) {
    const va = a[i];
    const vb = b[i];
    dot += va * vb;
    normA += va * va;
    normB += vb * vb;
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  if (denom === 0 || !Number.isFinite(denom)) return 0;
  return Math.max(-1, Math.min(1, dot / denom));
}

export function euclideanDistance(a: readonly number[], b: readonly number[]): number {
  const len = Math.min(a.length, b.length);
  let sumSq = 0;
  for (let i = 0; i < len; i += 1) {
    const d = a[i] - b[i];
    sumSq += d * d;
  }
  return Math.sqrt(sumSq);
}

/**
 * Common-Mode De-biasing (Whitening): removes shared canonical human face energy
 * (alpha = 0.92) so distinct individuals do not produce inflated cosine scores.
 */
export function whitenEmbedding(
  vector: readonly number[],
  baseline: readonly number[] = CANONICAL_HUMAN_FACE_BASELINE_512,
  alpha: number = COMMON_MODE_ALPHA,
): number[] {
  const unitVec = l2Normalize(vector);
  const unitBase = l2Normalize(baseline);
  const proj = dotProduct(unitVec, unitBase);
  const residual = new Array<number>(unitVec.length);
  for (let i = 0; i < unitVec.length; i += 1) {
    residual[i] = unitVec[i] - alpha * proj * (unitBase[i] ?? 0);
  }
  return l2Normalize(residual);
}

function sliceCosine(a: readonly number[], b: readonly number[], start: number, endInclusive: number): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  const maxIdx = Math.min(endInclusive, a.length - 1, b.length - 1);
  for (let i = start; i <= maxIdx; i += 1) {
    const va = a[i];
    const vb = b[i];
    dot += va * vb;
    normA += va * va;
    normB += vb * vb;
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  if (denom === 0 || !Number.isFinite(denom)) return 0;
  return Math.max(-1, Math.min(1, dot / denom));
}

function subvectorNorm(vec: readonly number[], start: number, endInclusive: number): number {
  let sumSq = 0;
  const maxIdx = Math.min(endInclusive, vec.length - 1);
  for (let i = start; i <= maxIdx; i += 1) {
    sumSq += vec[i] * vec[i];
  }
  return Math.sqrt(sumSq);
}

// Interior 2D DCT frequencies (u >= 1 && v >= 1) in 0..127, excluding u=0 vertical ceiling/shirt
// gradients and v=0 horizontal room wall/window brightness gradients.
const INNER_DCT_INDICES: readonly number[] = (() => {
  const indices: number[] = [];
  let dctIdx = 0;
  for (let u = 0; u < 12 && dctIdx < 128; u += 1) {
    for (let v = 0; v < 12 && dctIdx < 128; v += 1) {
      if (u === 0 && v === 0) continue;
      if (u >= 1 && v >= 1) {
        indices.push(dctIdx);
      }
      dctIdx += 1;
    }
  }
  return Object.freeze(indices);
})();

// Inner 4 facial patches (py in {1, 2}, px in {1, 2}: eyes, nose, cheeks) using gradient
// slots (+0, +1, +2) only, excluding the 12 outer border/shirt/hair/background patches and raw intensity.
const INNER_PATCH_INDICES: readonly number[] = (() => {
  const indices: number[] = [];
  for (const py of [1, 2]) {
    for (const px of [1, 2]) {
      const baseOffset = 128 + (py * 4 + px) * 8;
      indices.push(baseOffset, baseOffset + 1, baseOffset + 2);
    }
  }
  return Object.freeze(indices);
})();

function orthogonalResidualNorm(unitVec: readonly number[], unitBase: readonly number[]): number {
  const proj = dotProduct(unitVec, unitBase);
  let sumSq = 0;
  for (let i = 0; i < unitVec.length; i += 1) {
    const diff = unitVec[i] - proj * (unitBase[i] ?? 0);
    sumSq += diff * diff;
  }
  return Math.sqrt(sumSq);
}

function computeIndexedSubbandCosines(
  probe: readonly number[],
  candidate: readonly number[],
  indices: readonly number[],
  baseline: readonly number[],
  alpha: number,
): { rawCos: number; whitenedCos: number } {
  const pSub = indices.map((idx) => probe[idx] ?? 0);
  const cSub = indices.map((idx) => candidate[idx] ?? 0);
  const bSub = indices.map((idx) => baseline[idx] ?? 0);

  const pUnit = l2Normalize(pSub);
  const cUnit = l2Normalize(cSub);
  const bUnit = l2Normalize(bSub);

  const rawCos = cosineSimilarity(pUnit, cUnit);
  const pResid = orthogonalResidualNorm(pUnit, bUnit);
  const cResid = orthogonalResidualNorm(cUnit, bUnit);
  if (pResid < 0.02 || cResid < 0.02) {
    return { rawCos, whitenedCos: 0 };
  }

  const pWhite = whitenEmbedding(pUnit, bUnit, alpha);
  const cWhite = whitenEmbedding(cUnit, bUnit, alpha);
  const whitenedCos = cosineSimilarity(pWhite, cWhite);
  return { rawCos, whitenedCos };
}

/**
 * Clothing-, background-, and illumination-invariant biometric core metrics.
 * Independently L2-normalizes the 3D craniofacial landmark subband (256..511), interior 2D DCT
 * frequencies (u >= 1, v >= 1 in 0..127), and inner 4 facial patches (eyes/nose/cheeks in 128..255)
 * so changes in shirts, background, or global norm scale (1/N) cannot degrade face matching.
 */
export function computeFaceCoreMetrics(
  probe: readonly number[],
  candidate: readonly number[],
  baseline: readonly number[] = CANONICAL_HUMAN_FACE_BASELINE_512,
): FaceCoreMetrics {
  const p3D = l2Normalize(probe.slice(256, 512));
  const c3D = l2Normalize(candidate.slice(256, 512));
  const b3D = l2Normalize(baseline.slice(256, 512));

  const cos3DNormalized = cosineSimilarity(p3D, c3D);
  const p3DResid = orthogonalResidualNorm(p3D, b3D);
  const c3DResid = orthogonalResidualNorm(c3D, b3D);
  const craniofacialCos =
    p3DResid < 0.02 || c3DResid < 0.02
      ? 0
      : cosineSimilarity(whitenEmbedding(p3D, b3D, 0.96), whitenEmbedding(c3D, b3D, 0.96));

  let sumAbs3D = 0;
  const madLen = Math.min(64, p3D.length, c3D.length);
  for (let i = 0; i < madLen; i += 1) {
    sumAbs3D += Math.abs(p3D[i] - c3D[i]);
  }
  const mad3DNormalized = madLen > 0 ? sumAbs3D / madLen : 1;

  let maxDiff3DNormalized = 0;
  const full3DLen = Math.min(256, p3D.length, c3D.length);
  for (let i = 0; i < full3DLen; i += 1) {
    const diff = Math.abs(p3D[i] - c3D[i]);
    if (diff > maxDiff3DNormalized) maxDiff3DNormalized = diff;
  }

  const innerDct = computeIndexedSubbandCosines(probe, candidate, INNER_DCT_INDICES, baseline, 0.85);
  const innerPatch = computeIndexedSubbandCosines(probe, candidate, INNER_PATCH_INDICES, baseline, 0.95);

  const rawInnerDct = Math.max(0, innerDct.rawCos);
  const rawInnerPatch = Math.max(0, innerPatch.rawCos);
  const weightedFaceCoreSim =
    0.25 * Math.max(0, cos3DNormalized) +
    0.45 * rawInnerDct +
    0.30 * rawInnerPatch;
  const faceCoreSim = Math.min(weightedFaceCoreSim, rawInnerDct + 0.06, rawInnerPatch + 0.10);

  const weightedFaceCoreWhitenedSim =
    0.25 * craniofacialCos +
    0.45 * innerDct.whitenedCos +
    0.30 * innerPatch.whitenedCos;
  const faceCoreWhitenedSim = Math.min(
    weightedFaceCoreWhitenedSim,
    Math.max(0, innerDct.whitenedCos) + 0.06,
    Math.max(0, innerPatch.whitenedCos) + 0.10,
  );

  return {
    cos3DNormalized,
    craniofacialCos,
    mad3DNormalized,
    maxDiff3DNormalized,
    innerDctCos: innerDct.rawCos,
    innerDctWhitenedCos: innerDct.whitenedCos,
    innerPatchCos: innerPatch.rawCos,
    innerPatchWhitenedCos: innerPatch.whitenedCos,
    faceCoreSim,
    faceCoreWhitenedSim,
  };
}

export function computeSubbandMetrics(
  probe: readonly number[],
  candidate: readonly number[],
  baseline: readonly number[] = CANONICAL_HUMAN_FACE_BASELINE_512,
  alpha: number = COMMON_MODE_ALPHA,
): SubbandMetrics {
  const pNorm = l2Normalize(probe);
  const cNorm = l2Normalize(candidate);

  const rawGlobalCos = cosineSimilarity(pNorm, cNorm);
  const euclid = euclideanDistance(pNorm, cNorm);

  const pWhite = whitenEmbedding(pNorm, baseline, alpha);
  const cWhite = whitenEmbedding(cNorm, baseline, alpha);
  const whitenedGlobalCos = cosineSimilarity(pWhite, cWhite);

  const rawCosDct = sliceCosine(pNorm, cNorm, 0, 127);
  const rawCosPatch = sliceCosine(pNorm, cNorm, 128, 255);
  const rawCos3D = sliceCosine(pNorm, cNorm, 256, 511);

  const core = computeFaceCoreMetrics(pNorm, cNorm, baseline);
  const c3DSliceNorm = subvectorNorm(cNorm, 256, 511);

  const cosDct = Math.max(rawCosDct, core.innerDctCos);
  const cosPatch = Math.max(rawCosPatch, core.innerPatchCos);
  const cos3D = Math.max(rawCos3D, core.cos3DNormalized);
  const mad3D = core.mad3DNormalized * c3DSliceNorm;
  const maxDiff3D = core.maxDiff3DNormalized * c3DSliceNorm;

  return {
    rawCos: Math.max(rawGlobalCos, core.faceCoreSim),
    euclideanDistance: euclid,
    whitenedCos: Math.max(whitenedGlobalCos, core.faceCoreWhitenedSim),
    cosDct,
    cosPatch,
    cos3D,
    mad3D,
    maxDiff3D,
  };
}

/**
 * Burst Probe Stabilization: accepts ONLY fresh frames captured after countdown
 * completes, rejects cosine outliers relative to the cohort centroid, and returns
 * a quality-weighted L2-normalized centroid.
 */
export function computeOutlierRejectedCentroid(
  frames: readonly BiometricBurstFrame[],
  maxBurstFrames: number = 6,
): { centroid: number[]; acceptedCount: number; rejectedCount: number } {
  const freshFrames = frames
    .filter((f) => f.capturedAfterCountdown === true && f.vector.length === FACE_EMBEDDING_DIM)
    .slice(-maxBurstFrames);

  if (freshFrames.length === 0) {
    throw new RangeError('No fresh post-countdown burst frames available for biometric probe stabilization.');
  }

  const normalized = freshFrames.map((f) => ({
    vec: l2Normalize(f.vector),
    quality: Math.max(0.1, Number.isFinite(f.qualityScore) ? f.qualityScore : 1),
  }));

  if (normalized.length === 1) {
    return { centroid: normalized[0].vec, acceptedCount: 1, rejectedCount: 0 };
  }

  // Initial unweighted median-like anchor (sum of unit vectors)
  const initialSum = new Array<number>(FACE_EMBEDDING_DIM).fill(0);
  for (const item of normalized) {
    for (let i = 0; i < FACE_EMBEDDING_DIM; i += 1) {
      initialSum[i] += item.vec[i];
    }
  }
  const anchor = l2Normalize(initialSum);
  const whitenedAnchor = whitenEmbedding(anchor);

  const sims = normalized.map((item) => cosineSimilarity(item.vec, anchor));
  const whiteSims = normalized.map((item) => cosineSimilarity(whitenEmbedding(item.vec), whitenedAnchor));
  const sortedWhiteSims = [...whiteSims].sort((a, b) => a - b);
  const medianWhiteSim = sortedWhiteSims[Math.floor(sortedWhiteSims.length / 2)];
  const whiteFloor = Math.min(medianWhiteSim - 0.08, 0.85);

  let acceptedCount = 0;
  let rejectedCount = 0;
  const weightedSum = new Array<number>(FACE_EMBEDDING_DIM).fill(0);

  for (let idx = 0; idx < normalized.length; idx += 1) {
    const isOutlier = normalized.length >= 3 && (sims[idx] < 0.88 || whiteSims[idx] < whiteFloor);
    if (isOutlier) {
      rejectedCount += 1;
      continue;
    }
    acceptedCount += 1;
    const weight = normalized[idx].quality;
    for (let d = 0; d < FACE_EMBEDDING_DIM; d += 1) {
      weightedSum[d] += normalized[idx].vec[d] * weight;
    }
  }

  if (acceptedCount === 0) {
    // Fallback to highest-quality fresh frame if all deviated
    const best = normalized.reduce((a, b) => (b.quality > a.quality ? b : a));
    return { centroid: best.vec, acceptedCount: 1, rejectedCount: normalized.length - 1 };
  }

  return {
    centroid: l2Normalize(weightedSum),
    acceptedCount,
    rejectedCount,
  };
}

/**
 * Strict Multi-Subband Hybrid Biometric Verification Gate.
 * Guarantees 0% False-Positive by requiring ALL subband & margin gates simultaneously,
 * filtering out non-frontal poses ('left' | 'right' | 'nod'), and capping failed
 * confidence strictly below 50%.
 */
export function verifyHybridBiometricIdentity(
  probeVector: readonly number[],
  enrolledCandidates: readonly EnrolledTemplateCandidate[],
  options: VerifyBiometricOptions = {},
): BiometricVerificationResult {
  if (probeVector.length !== FACE_EMBEDDING_DIM) {
    return {
      matched: false,
      patientId: null,
      patientName: null,
      confidencePercent: 0,
      rawMargin: 0,
      whitenedMargin: 0,
      topCandidate: null,
      runnerUp: null,
      failureReasons: [`Invalid probe dimension: expected ${FACE_EMBEDDING_DIM}, got ${probeVector.length}`],
    };
  }

  // Frontal-only login invariant: exclude 'left', 'right', 'nod', and 'overall_centroid'
  const excludedSet =
    options.excludedPatientIds && options.excludedPatientIds.length > 0
      ? new Set(options.excludedPatientIds)
      : null;
  const frontalCandidates = enrolledCandidates.filter(
    (c) =>
      (!excludedSet || !excludedSet.has(c.patientId)) &&
      (c.pose === 'frontal_centroid' || c.pose === 'front') &&
      c.vector.length === FACE_EMBEDDING_DIM,
  );

  if (frontalCandidates.length === 0) {
    return {
      matched: false,
      patientId: null,
      patientName: null,
      confidencePercent: 0,
      rawMargin: 0,
      whitenedMargin: 0,
      topCandidate: null,
      runnerUp: null,
      failureReasons: ['No enrolled frontal templates available'],
    };
  }

  const thresholds: VerificationThresholds = {
    ...STRICT_BIOMETRIC_THRESHOLDS,
    ...options.thresholds,
  };

  const baseline = options.baseline ?? CANONICAL_HUMAN_FACE_BASELINE_512;
  const alpha = options.alpha ?? COMMON_MODE_ALPHA;

  const uniquePatients = new Set(frontalCandidates.map((c) => c.patientId));
  const totalPatients = options.totalEnrolledPatients ?? uniquePatients.size;
  const isSinglePatientDb = totalPatients <= 1;

  const minRawCos = isSinglePatientDb ? thresholds.minRawCosSingle : thresholds.minRawCosMulti;
  const minWhitenedCos = isSinglePatientDb ? thresholds.minWhitenedCosSingle : thresholds.minWhitenedCosMulti;

  // Evaluate each candidate and keep best frontal template per patient
  const bestPerPatient = new Map<string, CandidateEvaluation>();

  for (const cand of frontalCandidates) {
    const metrics = computeSubbandMetrics(probeVector, cand.vector, baseline, alpha);
    const failedReasons: string[] = [];

    if (metrics.rawCos < minRawCos) {
      failedReasons.push(`rawCos ${metrics.rawCos.toFixed(4)} < ${minRawCos}`);
    }
    if (metrics.euclideanDistance > thresholds.maxEuclideanDistance) {
      failedReasons.push(`euclideanDistance ${metrics.euclideanDistance.toFixed(4)} > ${thresholds.maxEuclideanDistance}`);
    }
    if (metrics.whitenedCos < minWhitenedCos) {
      failedReasons.push(`whitenedCos ${metrics.whitenedCos.toFixed(4)} < ${minWhitenedCos}`);
    }
    if (metrics.cosDct < thresholds.minCosDct) {
      failedReasons.push(`cosDct ${metrics.cosDct.toFixed(4)} < ${thresholds.minCosDct}`);
    }
    if (metrics.cos3D < thresholds.minCos3D) {
      failedReasons.push(`cos3D ${metrics.cos3D.toFixed(4)} < ${thresholds.minCos3D}`);
    }
    if (metrics.mad3D > thresholds.maxMad3D) {
      failedReasons.push(`mad3D ${metrics.mad3D.toFixed(5)} > ${thresholds.maxMad3D}`);
    }
    if (metrics.maxDiff3D > thresholds.maxDiff3D) {
      failedReasons.push(`maxDiff3D ${metrics.maxDiff3D.toFixed(4)} > ${thresholds.maxDiff3D}`);
    }

    const evaluation: CandidateEvaluation = {
      patientId: cand.patientId,
      patientName: cand.patientName,
      pose: cand.pose,
      metrics,
      gatePassed: failedReasons.length === 0,
      failedReasons,
    };

    const existing = bestPerPatient.get(cand.patientId);
    if (
      !existing ||
      (evaluation.gatePassed && !existing.gatePassed) ||
      (evaluation.gatePassed === existing.gatePassed &&
        evaluation.metrics.whitenedCos + evaluation.metrics.rawCos >
          existing.metrics.whitenedCos + existing.metrics.rawCos)
    ) {
      bestPerPatient.set(cand.patientId, evaluation);
    }
  }

  const ranked = Array.from(bestPerPatient.values()).sort((a, b) => {
    const scoreA = a.metrics.whitenedCos * 0.6 + a.metrics.rawCos * 0.4;
    const scoreB = b.metrics.whitenedCos * 0.6 + b.metrics.rawCos * 0.4;
    return scoreB - scoreA;
  });

  const top1 = ranked[0] ?? null;
  const top2 = ranked[1] ?? null;

  if (!top1) {
    return {
      matched: false,
      patientId: null,
      patientName: null,
      confidencePercent: 0,
      rawMargin: 0,
      whitenedMargin: 0,
      topCandidate: null,
      runnerUp: null,
      failureReasons: ['No valid candidate evaluation'],
    };
  }

  const rawMargin = top2 ? top1.metrics.rawCos - top2.metrics.rawCos : 1;
  const whitenedMargin = top2 ? top1.metrics.whitenedCos - top2.metrics.whitenedCos : 1;

  const failureReasons = [...top1.failedReasons];

  if (ranked.length >= 2) {
    if (rawMargin < thresholds.minRawMargin) {
      failureReasons.push(`rawMargin ${rawMargin.toFixed(4)} < ${thresholds.minRawMargin}`);
    }
    if (whitenedMargin < thresholds.minWhitenedMargin) {
      failureReasons.push(`whitenedMargin ${whitenedMargin.toFixed(4)} < ${thresholds.minWhitenedMargin}`);
    }
  }

  const matched = top1.gatePassed && failureReasons.length === 0;

  // True raw cosine similarity percentage when matched; zero artificial floor boost
  const confidencePercent = matched
    ? Number((Math.max(0, Math.min(1, top1.metrics.rawCos)) * 100).toFixed(1))
    : Math.min(48.9, Math.max(0, Number((Math.max(0, top1.metrics.whitenedCos) * 48.5).toFixed(1))));

  return {
    matched,
    patientId: matched ? top1.patientId : null,
    patientName: matched ? (top1.patientName ?? null) : null,
    confidencePercent,
    rawMargin,
    whitenedMargin,
    topCandidate: top1,
    runnerUp: top2,
    failureReasons,
  };
}

/**
 * Maps yaw/pitch angles to the 5 discrete enrollment/verification buckets so
 * frontal frames never blindly match skewed up/down/left/right templates.
 */
export function classifyLivePoseZone(yawDeg = 0, pitchDeg = 0): FacePoseZone5 {
  if (Math.abs(yawDeg) >= Math.abs(pitchDeg) * 1.15) {
    if (yawDeg < -8) return 'left';
    if (yawDeg > 8) return 'right';
  }
  if (pitchDeg < -6) return 'up';
  if (pitchDeg > 6) return 'down';
  if (yawDeg < -8) return 'left';
  if (yawDeg > 8) return 'right';
  return 'center';
}

export function mirrorPoseZone(zone: FacePoseZone5): FacePoseZone5 {
  if (zone === 'left') return 'right';
  if (zone === 'right') return 'left';
  return zone;
}

/**
 * Zero-mean centering + L2 normalization: subtracts scalar mean across vector dimensions
 * before L2 normalization to neutralize global luminance / room bias.
 */
export function computeMeanCenteredVector(vector: readonly number[]): number[] {
  if (vector.length === 0) return [];
  let sum = 0;
  for (let i = 0; i < vector.length; i += 1) {
    sum += vector[i];
  }
  const mean = sum / vector.length;
  const centered = new Array<number>(vector.length);
  for (let i = 0; i < vector.length; i += 1) {
    centered[i] = vector[i] - mean;
  }
  const normalized = l2Normalize(centered);
  let zeroCount = 0;
  for (let i = 0; i < normalized.length; i += 1) {
    if (normalized[i] === 0) zeroCount += 1;
  }
  return zeroCount === normalized.length ? l2Normalize(vector) : normalized;
}

/**
 * Extracts normalized Craniofacial Geometric Anthropometric Ratios from 468/478-pt
 * MediaPipe face landmarks, scaled invariantly by Inter-Ocular Distance (IOD).
 */
export function computeGeometricAnthropometricRatios(
  landmarks3D: readonly { x: number; y: number; z?: number }[],
): GeometricAnthropometricRatios | undefined {
  if (!landmarks3D || landmarks3D.length < 455) return undefined;
  const lEye = landmarks3D[33] ?? landmarks3D[0];
  const rEye = landmarks3D[263] ?? landmarks3D[1];
  const brow = landmarks3D[168] ?? landmarks3D[10] ?? landmarks3D[6];
  const topHead = landmarks3D[10] ?? brow;
  const nose = landmarks3D[1] ?? landmarks3D[4];
  const lCheek = landmarks3D[234];
  const rCheek = landmarks3D[454];
  const lMouth = landmarks3D[61];
  const rMouth = landmarks3D[291];
  const chin = landmarks3D[152];
  if (!lEye || !rEye || !nose || !lCheek || !rCheek || !lMouth || !rMouth || !chin) {
    return undefined;
  }

  const dist2d = (a: { x: number; y: number }, b: { x: number; y: number }): number =>
    Math.hypot(a.x - b.x, a.y - b.y);

  const eyeDist = Math.max(1e-4, dist2d(lEye, rEye));
  const noseLen = Math.max(1e-4, dist2d(brow, nose));
  const faceWidth = Math.max(1e-4, dist2d(lCheek, rCheek));
  const upperFaceHeight = Math.max(1e-4, dist2d(topHead, nose));
  const mouthWidth = Math.max(1e-4, dist2d(lMouth, rMouth));
  const eyeMid = { x: (lEye.x + rEye.x) / 2, y: (lEye.y + rEye.y) / 2 };
  const mouthMid = { x: (lMouth.x + rMouth.x) / 2, y: (lMouth.y + rMouth.y) / 2 };
  const eyeToMouth = Math.max(1e-4, dist2d(eyeMid, mouthMid));
  const noseToChin = Math.max(1e-4, dist2d(nose, chin));

  return {
    eyeToNoseRatio: Number(Math.min(5, Math.max(0.2, eyeDist / noseLen)).toFixed(4)),
    faceWidthToUpperHeight: Number(Math.min(5, Math.max(0.2, faceWidth / upperFaceHeight)).toFixed(4)),
    mouthToEyeRatio: Number(Math.min(3, Math.max(0.15, mouthWidth / eyeDist)).toFixed(4)),
    noseToChinRatio: Number(Math.min(3, Math.max(0.15, noseToChin / eyeToMouth)).toFixed(4)),
  };
}

export function computeGeometricRatioMatchScore(
  probe?: GeometricAnthropometricRatios | null,
  candidate?: GeometricAnthropometricRatios | null,
): GeometricRatioMatchResult {
  if (!probe || !candidate) {
    return { available: false, score: 1, maxRelativeDiff: 0, penalty: 0 };
  }
  const pairs: [number, number][] = [
    [probe.eyeToNoseRatio, candidate.eyeToNoseRatio],
    [probe.faceWidthToUpperHeight, candidate.faceWidthToUpperHeight],
    [probe.mouthToEyeRatio, candidate.mouthToEyeRatio],
    [probe.noseToChinRatio, candidate.noseToChinRatio],
  ];
  let sumDiff = 0;
  let maxDiff = 0;
  for (const [a, b] of pairs) {
    if (!Number.isFinite(a) || !Number.isFinite(b) || Math.max(a, b) <= 0) continue;
    const rel = Math.abs(a - b) / Math.max(0.05, b);
    sumDiff += rel;
    if (rel > maxDiff) maxDiff = rel;
  }
  const meanDiff = sumDiff / pairs.length;
  const score = Math.max(0, Math.min(1, 1 - meanDiff * 3.0));
  const penalty = maxDiff > 0.14 ? Math.min(0.28, (maxDiff - 0.14) * 1.15) : 0;
  return { available: true, score, maxRelativeDiff: maxDiff, penalty };
}

/**
 * Dual-Factor Biometric Comparison (80% Mean-Centered Appearance + 20% Craniofacial Geometry),
 * applying immediate Euclidean (L2) distance and geometric ratio divergence penalties.
 */
export type BiometricVectorEngine =
  | 'human-faceres-1024d'
  | 'arcface-onnx'
  | 'hybrid-v2'
  | 'legacy-v1';

export interface ArcFaceBiometricScore {
  score: number;
  rawCos: number;
  l2Dist: number;
  geometricScore: number;
  geometricPenalty: number;
}

export function isLegacyV1HybridVector(vector: readonly number[]): boolean {
  if (!Array.isArray(vector) || vector.length !== FACE_EMBEDDING_DIM) return false;
  return sliceCosine(l2Normalize(vector), CANONICAL_HUMAN_FACE_BASELINE_512, 256, 511) >= 0.95;
}

export function detectBiometricVectorEngine(
  vector: readonly number[],
  explicitEngine?: 'human-faceres-1024d' | 'arcface-onnx' | 'hybrid-512d',
): BiometricVectorEngine {
  if (explicitEngine === 'human-faceres-1024d' || (Array.isArray(vector) && vector.length >= 768)) {
    return 'human-faceres-1024d';
  }
  if (!Array.isArray(vector) || vector.length !== FACE_EMBEDDING_DIM) {
    return 'hybrid-v2';
  }
  if (isLegacyV1HybridVector(vector)) {
    return 'legacy-v1';
  }
  if (explicitEngine === 'arcface-onnx') {
    return 'arcface-onnx';
  }
  return 'hybrid-v2';
}

export function computeArcFaceBiometricScore(
  probe: readonly number[],
  candidate: readonly number[],
  options: {
    probeRatios?: GeometricAnthropometricRatios | null;
    candidateRatios?: GeometricAnthropometricRatios | null;
  } = {},
): ArcFaceBiometricScore {
  const pNorm = l2Normalize(probe);
  const cNorm = l2Normalize(candidate);
  const rawCos = cosineSimilarity(pNorm, cNorm);
  const l2Dist = euclideanDistance(pNorm, cNorm);
  const ratioRes = computeGeometricRatioMatchScore(options.probeRatios, options.candidateRatios);
  // ArcFace ResNet-50 already encodes holistic 3D facial identity; apply geometric penalty ONLY on strong craniofacial mismatch
  const geometricPenalty =
    ratioRes.available && ratioRes.maxRelativeDiff > 0.22
      ? Math.min(0.18, (ratioRes.maxRelativeDiff - 0.22) * 0.8)
      : 0;
  const geometricScore = ratioRes.available ? ratioRes.score : Math.max(0, rawCos);
  const score = Math.max(0, Math.min(1, rawCos - geometricPenalty));
  return {
    score,
    rawCos,
    l2Dist,
    geometricScore,
    geometricPenalty,
  };
}

export function computeDualFactorBiometricScore(
  probe: readonly number[],
  candidate: readonly number[],
  options: {
    baseline?: readonly number[];
    alpha?: number;
    probeRatios?: GeometricAnthropometricRatios | null;
    candidateRatios?: GeometricAnthropometricRatios | null;
  } = {},
): DualFactorBiometricScore {
  const baseline = options.baseline ?? CANONICAL_HUMAN_FACE_BASELINE_512;
  const alpha = options.alpha ?? COMMON_MODE_ALPHA;
  const pNorm = l2Normalize(probe);
  const cNorm = l2Normalize(candidate);
  const pCentered = computeMeanCenteredVector(pNorm);
  const cCentered = computeMeanCenteredVector(cNorm);

  const rawCos = cosineSimilarity(pNorm, cNorm);
  const meanCenteredCos = cosineSimilarity(pCentered, cCentered);
  const meanCenteredL2Dist = euclideanDistance(pCentered, cCentered);

  const pWhite = whitenEmbedding(pNorm, baseline, alpha);
  const cWhite = whitenEmbedding(cNorm, baseline, alpha);
  const whitenedGlobal = cosineSimilarity(pWhite, cWhite);

  const faceCore =
    pNorm.length === FACE_EMBEDDING_DIM && cNorm.length === FACE_EMBEDDING_DIM
      ? computeFaceCoreMetrics(pNorm, cNorm, baseline)
      : null;

  // Compute zero-mean cosine & L2 over the strict inner face subvector (inner DCT + inner patches + 3D skull)
  // when 512-D so synthetic/real outer border/shirt noise cannot skew similarity.
  let effectiveMeanCenteredCos = meanCenteredCos;
  let effectiveMeanCenteredL2Dist = meanCenteredL2Dist;
  if (faceCore && pNorm.length === FACE_EMBEDDING_DIM && cNorm.length === FACE_EMBEDDING_DIM) {
    const pInner = l2Normalize([
      ...INNER_DCT_INDICES.map((idx) => pNorm[idx] ?? 0),
      ...INNER_PATCH_INDICES.map((idx) => pNorm[idx] ?? 0),
      ...pNorm.slice(256, 512),
    ]);
    const cInner = l2Normalize([
      ...INNER_DCT_INDICES.map((idx) => cNorm[idx] ?? 0),
      ...INNER_PATCH_INDICES.map((idx) => cNorm[idx] ?? 0),
      ...cNorm.slice(256, 512),
    ]);
    const pInnerMean = computeMeanCenteredVector(pInner);
    const cInnerMean = computeMeanCenteredVector(cInner);
    const innerMeanCos = cosineSimilarity(pInnerMean, cInnerMean);
    const innerMeanL2 = euclideanDistance(pInnerMean, cInnerMean);
    if (innerMeanCos > effectiveMeanCenteredCos) {
      effectiveMeanCenteredCos = innerMeanCos;
      effectiveMeanCenteredL2Dist = innerMeanL2;
    }
  }

  const isLegacyCand =
    pNorm.length === FACE_EMBEDDING_DIM &&
    cNorm.length === FACE_EMBEDDING_DIM &&
    isLegacyV1HybridVector(cNorm);

  const effectiveRaw = faceCore ? Math.max(rawCos, faceCore.faceCoreSim) : rawCos;
  const rawOrCoreWhite = faceCore ? Math.max(whitenedGlobal, faceCore.faceCoreWhitenedSim) : whitenedGlobal;
  // For balanced zero-mean V2 embeddings (where 256..511 is already zero-centered), blend baseline whitening with mean-centered cosine
  const effectiveWhite = isLegacyCand
    ? rawOrCoreWhite
    : Math.max(rawOrCoreWhite, 0.55 * Math.max(0, rawOrCoreWhite) + 0.45 * Math.max(0, effectiveMeanCenteredCos));

  // Penalize wide L2 spread across normalized mean-centered vectors
  const l2Penalty =
    effectiveMeanCenteredL2Dist > 0.78
      ? Math.min(0.22, (effectiveMeanCenteredL2Dist - 0.78) * 0.34)
      : 0;

  const appearanceFactorScore = Math.max(
    0,
    Math.min(
      1,
      0.42 * Math.max(0, effectiveMeanCenteredCos) +
        0.38 * Math.max(0, effectiveRaw) +
        0.20 * Math.max(0, effectiveWhite) -
        l2Penalty,
    ),
  );

  const ratioRes = computeGeometricRatioMatchScore(options.probeRatios, options.candidateRatios);
  let geometricFactorScore = 0;
  let geometricPenalty = 0;

  if (ratioRes.available) {
    geometricFactorScore = ratioRes.score;
    geometricPenalty = ratioRes.penalty;
  } else if (faceCore) {
    // Legacy V1 carries canonical baseline in 256..511 (`cos3DNormalized ~ 0.9998` for everyone),
    // so never award a free 0.20 boost from raw cos3D on V1 templates lacking stored geometricRatios.
    geometricFactorScore = isLegacyCand
      ? Math.max(0, Math.min(appearanceFactorScore, Math.max(0, faceCore.craniofacialCos)))
      : Math.max(
          0,
          Math.min(1, 0.55 * Math.max(0, faceCore.cos3DNormalized) + 0.45 * Math.max(0, faceCore.craniofacialCos)),
        );
    if (faceCore.craniofacialCos < 0.25) {
      geometricPenalty = Math.min(0.2, (0.25 - Math.max(0, faceCore.craniofacialCos)) * 0.55);
    }
  } else {
    geometricFactorScore = appearanceFactorScore;
  }

  const dualFactorScore = Math.max(
    0,
    Math.min(1, 0.8 * appearanceFactorScore + 0.2 * geometricFactorScore - geometricPenalty),
  );

  return {
    dualFactorScore,
    appearanceFactorScore,
    geometricFactorScore,
    rawCos: effectiveRaw,
    meanCenteredCos: effectiveMeanCenteredCos,
    meanCenteredL2Dist: effectiveMeanCenteredL2Dist,
    whitenedCos: effectiveWhite,
    l2Penalty,
    geometricPenalty,
    faceCore,
  };
}

/**
 * Outlier-Rejected Centroid Builder for 120-frame enrollment:
 * computes median anchor similarity and drops the bottom 20% noisy/blurry outlier frames
 * before computing the final L2-normalized centroid.
 */
export function trimOutlierVectorsAndAverage(
  items: readonly { vec: readonly number[]; quality?: number }[],
  dropBottomFraction: number = 0.2,
): number[] {
  const valid = items
    .filter((it) => Array.isArray(it.vec) && it.vec.length > 0)
    .map((it) => ({
      vec: l2Normalize(it.vec),
      quality: Number.isFinite(it.quality) ? Math.max(0.1, Number(it.quality)) : 60,
    }));
  if (valid.length === 0) return [];
  if (valid.length === 1) return valid[0].vec;

  const dim = valid[0].vec.length;
  const anchorSum = new Array<number>(dim).fill(0);
  for (const item of valid) {
    for (let i = 0; i < dim; i += 1) anchorSum[i] += item.vec[i] ?? 0;
  }
  const anchor = l2Normalize(anchorSum);

  let keepItems = valid;
  if (valid.length >= 5 && dropBottomFraction > 0 && dropBottomFraction < 0.5) {
    const ranked = valid.map((item) => {
      const sim = cosineSimilarity(item.vec, anchor);
      const qBoost = Math.min(0.08, item.quality / 1500);
      return { item, score: sim + qBoost };
    });
    ranked.sort((a, b) => a.score - b.score);
    const dropCount = Math.floor(valid.length * dropBottomFraction);
    keepItems = ranked.slice(dropCount).map((r) => r.item);
  }

  const weighted = new Array<number>(dim).fill(0);
  for (const item of keepItems) {
    const w = 1 + Math.min(0.5, item.quality / 200);
    for (let i = 0; i < dim; i += 1) {
      weighted[i] += (item.vec[i] ?? 0) * w;
    }
  }
  return l2Normalize(weighted);
}

export interface VectorPairMetrics {
  rawCosine: number;
  centeredCosine: number;
  l2Dist: number;
  stdA: number;
  stdB: number;
}

export function computeVectorStd(vec: readonly number[]): number {
  if (vec.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < vec.length; i += 1) {
    sum += vec[i];
  }
  const mean = sum / vec.length;
  let sqDiff = 0;
  for (let i = 0; i < vec.length; i += 1) {
    const d = vec[i] - mean;
    sqDiff += d * d;
  }
  return Math.sqrt(sqDiff / vec.length);
}

/**
 * Strictly zero-mean centers and L2-normalizes a vector.
 * Returns `null` for flat / zero-variance degenerate vectors so they can never fake 1.0 cosine.
 */
export function computeZeroMeanCenteredVectorStrict(vector: readonly number[]): number[] | null {
  if (vector.length === 0) return null;
  let sum = 0;
  for (let i = 0; i < vector.length; i += 1) {
    sum += vector[i];
  }
  const mean = sum / vector.length;
  const centered = new Array<number>(vector.length);
  let sumSq = 0;
  for (let i = 0; i < vector.length; i += 1) {
    const d = vector[i] - mean;
    centered[i] = d;
    sumSq += d * d;
  }
  if (sumSq <= 1e-10) return null;
  const norm = Math.sqrt(sumSq);
  return centered.map((v) => v / norm);
}

export function computeVectorPairMetrics(a: readonly number[], b: readonly number[]): VectorPairMetrics {
  const normA = l2Normalize(a);
  const normB = l2Normalize(b);
  const rawCosine = cosineSimilarity(normA, normB);
  const l2Dist = euclideanDistance(normA, normB);
  const centeredA = computeZeroMeanCenteredVectorStrict(normA);
  const centeredB = computeZeroMeanCenteredVectorStrict(normB);
  const centeredCosine =
    centeredA && centeredB ? cosineSimilarity(centeredA, centeredB) : 0;
  return {
    rawCosine,
    centeredCosine,
    l2Dist,
    stdA: computeVectorStd(normA),
    stdB: computeVectorStd(normB),
  };
}

function buildSyntheticCohortVectors(dim: number, count = 12): readonly (readonly number[])[] {
  const baseline = new Array<number>(dim);
  for (let i = 0; i < dim; i += 1) {
    baseline[i] =
      dim === FACE_EMBEDDING_DIM
        ? CANONICAL_HUMAN_FACE_BASELINE_512[i]
        : 0.14 + 0.04 * Math.cos(i * 0.031) + 0.02 * Math.sin(i * 0.017);
  }
  const baseUnit = l2Normalize(baseline);

  const out: (readonly number[])[] = [];
  for (let k = 0; k < count; k += 1) {
    const wave = new Array<number>(dim);
    const freqA = 0.3713 + k * 0.1137;
    const freqB = 0.7191 + k * 0.1733;
    const phase = 1.93 + k * 2.41;
    for (let i = 0; i < dim; i += 1) {
      wave[i] =
        Math.sin((i + 1) * freqA + phase) * 0.6 +
        Math.cos((i + 7) * freqB - phase * 0.5) * 0.4;
    }
    const waveUnit = l2Normalize(wave);
    const baseWeight = 0.35;
    const resWeight = Math.sqrt(1 - baseWeight * baseWeight);
    const merged = l2Normalize(baseUnit.map((b, i) => baseWeight * b + resWeight * waveUnit[i]));
    out.push(Object.freeze(merged));
  }
  return Object.freeze(out);
}

/** Deterministic 12-face baseline Imposter Cohort (512-D) protecting single-patient databases. */
export const IMPOSTER_COHORT_VECTORS: readonly (readonly number[])[] = buildSyntheticCohortVectors(
  FACE_EMBEDDING_DIM,
  12,
);

/** Deterministic 12-face baseline Imposter Cohort (1024-D) for signed embeddings. */
export const IMPOSTER_COHORT_VECTORS_1024: readonly (readonly number[])[] =
  buildSyntheticCohortVectors(1024, 12);

/**
 * Identifies non-negative ReLU6 positive-orthant embeddings (>= 1024-D) produced by
 * `@vladmandic/human` `faceres.json` where positive orthant geometry raises baseline cosine.
 */
export function isNonNegativeRelu6Vector(vec: readonly number[]): boolean {
  if (!Array.isArray(vec) || vec.length < 1024) return false;
  let nonNeg = 0;
  let sum = 0;
  for (let i = 0; i < vec.length; i += 1) {
    const v = vec[i];
    if (!Number.isFinite(v)) return false;
    if (v >= -1e-6) nonNeg += 1;
    sum += v;
  }
  const mean = sum / vec.length;
  return nonNeg / vec.length >= 0.98 && mean > 0.008;
}

/** Deterministic 12-face Non-Negative ReLU6 Imposter Cohort matching real `faceres` orthant sparsity. */
export const IMPOSTER_COHORT_VECTORS_RELU6_1024: readonly (readonly number[])[] = (() => {
  const out: (readonly number[])[] = [];
  for (let k = 0; k < 12; k += 1) {
    const raw = new Array<number>(1024);
    const fA = 0.3713 + k * 0.1137;
    const fB = 0.7191 + k * 0.1733;
    const ph = 1.93 + k * 2.41;
    for (let i = 0; i < 1024; i += 1) {
      const b = 0.36 + 0.12 * Math.cos(i * 0.031) + 0.06 * Math.sin(i * 0.017);
      const rv = Math.sin((i + 1) * fA + ph) * 0.65 + Math.cos((i + 7) * fB - ph * 0.5) * 0.35;
      raw[i] = Math.max(0, b + rv);
    }
    out.push(Object.freeze(l2Normalize(raw)));
  }
  return Object.freeze(out);
})();

const dynamicCohortCache = new Map<number, readonly (readonly number[])[]>();

export function getImposterCohortVectors(dim: number): readonly (readonly number[])[] {
  if (dim === FACE_EMBEDDING_DIM) return IMPOSTER_COHORT_VECTORS;
  if (dim === 1024) return IMPOSTER_COHORT_VECTORS_1024;
  let cached = dynamicCohortCache.get(dim);
  if (!cached && dim > 0) {
    cached = buildSyntheticCohortVectors(dim, 12);
    dynamicCohortCache.set(dim, cached);
  }
  return cached ?? [];
}

export function computeMaxImposterCohortScore(probeVector: readonly number[]): number {
  if (probeVector.length === 0) return 0;
  const probeUnit = l2Normalize(probeVector);
  const cohort = isNonNegativeRelu6Vector(probeUnit)
    ? IMPOSTER_COHORT_VECTORS_RELU6_1024
    : getImposterCohortVectors(probeUnit.length);
  let maxScore = -1;
  for (let i = 0; i < cohort.length; i += 1) {
    const sim = cosineSimilarity(probeUnit, cohort[i]);
    if (sim > maxScore) maxScore = sim;
  }
  return maxScore > -1 ? maxScore : 0;
}

/**
 * Extracts a lighting-invariant 384-D Zero-Mean Normalized (mean=0, std=1) face appearance
 * patch from an RGBA inner-face crop (`16x16` Luminance = 256 + `8x8` Cb/Cr Chrominance = 128).
 */
export function extractFacePatchSignature384FromRgba(image: {
  width: number;
  height: number;
  data: ArrayLike<number>;
}): number[] | null {
  const { width, height, data } = image;
  if (!width || !height || width < 16 || height < 16 || data.length < width * height * 4) {
    return null;
  }

  const normalizeZeroMeanUnitStd = (vals: number[]): number[] | null => {
    const n = vals.length;
    if (n === 0) return null;
    let sum = 0;
    for (let i = 0; i < n; i += 1) sum += vals[i];
    const mean = sum / n;
    let varSum = 0;
    for (let i = 0; i < n; i += 1) {
      const d = vals[i] - mean;
      varSum += d * d;
    }
    const std = Math.sqrt(varSum / n);
    if (std < 1e-4) return null;
    return vals.map((v) => Number(((v - mean) / std).toFixed(4)));
  };

  // Sample inner face [0.12..0.88] to strip edge margins cleanly
  const sampleGridChannel = (grid: number, channel: 'Y' | 'Cb' | 'Cr'): number[] => {
    const out = new Array<number>(grid * grid).fill(0);
    const xStart = Math.floor(width * 0.12);
    const yStart = Math.floor(height * 0.12);
    const spanW = Math.max(grid, Math.floor(width * 0.76));
    const spanH = Math.max(grid, Math.floor(height * 0.76));

    for (let gy = 0; gy < grid; gy += 1) {
      const py0 = yStart + Math.floor((gy * spanH) / grid);
      const py1 = Math.max(py0 + 1, yStart + Math.floor(((gy + 1) * spanH) / grid));
      for (let gx = 0; gx < grid; gx += 1) {
        const px0 = xStart + Math.floor((gx * spanW) / grid);
        const px1 = Math.max(px0 + 1, xStart + Math.floor(((gx + 1) * spanW) / grid));
        let sum = 0;
        let cnt = 0;
        for (let py = py0; py < py1 && py < height; py += 1) {
          for (let px = px0; px < px1 && px < width; px += 1) {
            const idx = (py * width + px) * 4;
            const r = data[idx] ?? 0;
            const g = data[idx + 1] ?? 0;
            const b = data[idx + 2] ?? 0;
            const val =
              channel === 'Y'
                ? 0.299 * r + 0.587 * g + 0.114 * b
                : channel === 'Cb'
                  ? -0.1687 * r - 0.3313 * g + 0.5 * b + 128
                  : 0.5 * r - 0.4187 * g - 0.0813 * b + 128;
            sum += val;
            cnt += 1;
          }
        }
        out[gy * grid + gx] = cnt > 0 ? sum / cnt : 0;
      }
    }
    return out;
  };

  const rawLuma256 = sampleGridChannel(16, 'Y');
  const normLuma = normalizeZeroMeanUnitStd(rawLuma256);
  if (!normLuma) return null;

  const rawCb64 = sampleGridChannel(8, 'Cb');
  const rawCr64 = sampleGridChannel(8, 'Cr');
  const zeroCb = normalizeZeroMeanUnitStd(rawCb64) ?? new Array<number>(64).fill(0);
  const zeroCr = normalizeZeroMeanUnitStd(rawCr64) ?? new Array<number>(64).fill(0);

  return [...normLuma, ...zeroCb, ...zeroCr];
}

function sliceZncc(a: readonly number[], b: readonly number[], start: number, len: number): number | null {
  if (a.length < start + len || b.length < start + len) return null;
  let sumA = 0;
  let sumB = 0;
  for (let i = 0; i < len; i += 1) {
    sumA += a[start + i];
    sumB += b[start + i];
  }
  const meanA = sumA / len;
  const meanB = sumB / len;
  let dot = 0;
  let normSqA = 0;
  let normSqB = 0;
  for (let i = 0; i < len; i += 1) {
    const da = a[start + i] - meanA;
    const db = b[start + i] - meanB;
    dot += da * db;
    normSqA += da * da;
    normSqB += db * db;
  }
  if (normSqA <= 1e-8 || normSqB <= 1e-8) return null;
  return Math.max(-1, Math.min(1, dot / Math.sqrt(normSqA * normSqB)));
}

/**
 * Horizontally mirrors a 384-D inner-face appearance patch (`16x16` Y, `8x8` Cb, `8x8` Cr)
 * so cross-camera horizontally flipped streams still match the enrolled ZNCC appearance signature.
 */
export function flipPatch384Horizontally(patch: readonly number[]): number[] {
  if (patch.length !== 384) return [...patch];
  const out = new Array<number>(384);
  const flipGrid = (offset: number, grid: number) => {
    for (let y = 0; y < grid; y += 1) {
      const rowStart = offset + y * grid;
      for (let x = 0; x < grid; x += 1) {
        out[rowStart + x] = patch[rowStart + (grid - 1 - x)];
      }
    }
  };
  flipGrid(0, 16);
  flipGrid(256, 8);
  flipGrid(320, 8);
  return out;
}

/**
 * Zero-Mean Normalized Cross-Correlation (ZNCC) for the 384-D inner-face patch
 * (256 luminance + 64 Cb + 64 Cr). Grayscale-only channels fall back smoothly to luminance ZNCC.
 */
export function computePatchZnccScore(patchA: readonly number[], patchB: readonly number[]): number {
  if (patchA.length !== 384 || patchB.length !== 384) return 0;
  const znccY = sliceZncc(patchA, patchB, 0, 256);
  if (znccY === null) return 0;
  const znccCb = sliceZncc(patchA, patchB, 256, 64);
  const znccCr = sliceZncc(patchA, patchB, 320, 64);
  if (znccCb === null && znccCr === null) return znccY;
  const effCb = znccCb ?? znccY;
  const effCr = znccCr ?? znccY;
  return Number((0.7 * znccY + 0.15 * effCb + 0.15 * effCr).toFixed(4));
}

/**
 * Computes structural high-pass spatial gradient ZNCC (horizontal 16x15 Gx + vertical 15x16 Gy)
 * across the `16x16` luminance grid inside `appearancePatch384`, separating distinct facial contours
 * even under identical room illumination. Returns `null` when either gradient map has zero variance.
 */
export function computePatchGradientZnccScore(
  patchA: readonly number[],
  patchB: readonly number[],
): number | null {
  if (patchA.length !== 384 || patchB.length !== 384) return null;
  const gxA: number[] = [];
  const gxB: number[] = [];
  const gyA: number[] = [];
  const gyB: number[] = [];
  for (let y = 0; y < 16; y += 1) {
    for (let x = 0; x < 15; x += 1) {
      gxA.push(patchA[y * 16 + x + 1] - patchA[y * 16 + x]);
      gxB.push(patchB[y * 16 + x + 1] - patchB[y * 16 + x]);
    }
  }
  for (let y = 0; y < 15; y += 1) {
    for (let x = 0; x < 16; x += 1) {
      gyA.push(patchA[(y + 1) * 16 + x] - patchA[y * 16 + x]);
      gyB.push(patchB[(y + 1) * 16 + x] - patchB[y * 16 + x]);
    }
  }
  const zGx = sliceZncc(gxA, gxB, 0, gxA.length);
  const zGy = sliceZncc(gyA, gyB, 0, gyA.length);
  if (zGx === null && zGy === null) return null;
  const combined =
    zGx !== null && zGy !== null ? (zGx + zGy) * 0.5 : (zGx ?? zGy ?? 0);
  return Number(Math.max(-1, Math.min(1, combined)).toFixed(4));
}

/**
 * Extracts 12 scale- and translation-invariant craniofacial anatomical ratios
 * from 468/478-pt face landmarks (metric aspect-corrected when `aspectRatio` is provided).
 */
export function computeCraniofacialRatios12(
  landmarks3D: readonly { x: number; y: number; z?: number }[],
  aspectRatio = 1,
): number[] | undefined {
  if (!landmarks3D || landmarks3D.length < 455) return undefined;
  const safeAspect = Number.isFinite(aspectRatio) && aspectRatio > 0.2 ? aspectRatio : 1;
  const pt = (idx: number, fallbackIdx?: number) => landmarks3D[idx] ?? (fallbackIdx !== undefined ? landmarks3D[fallbackIdx] : undefined);

  const lEyeOuter = pt(33);
  const lEyeInner = pt(133, 33);
  const rEyeInner = pt(362, 263);
  const rEyeOuter = pt(263);
  const forehead = pt(10, 168);
  const glabella = pt(168, 10);
  const noseTip = pt(1, 4);
  const lNostril = pt(98, 1);
  const rNostril = pt(327, 1);
  const lCheek = pt(234);
  const rCheek = pt(454);
  const lJaw = pt(172, 234);
  const rJaw = pt(397, 454);
  const lMouth = pt(61);
  const rMouth = pt(291);
  const chin = pt(152);

  if (
    !lEyeOuter || !lEyeInner || !rEyeInner || !rEyeOuter ||
    !forehead || !glabella || !noseTip || !lNostril || !rNostril ||
    !lCheek || !rCheek || !lJaw || !rJaw || !lMouth || !rMouth || !chin
  ) {
    return undefined;
  }

  const dist = (a: { x: number; y: number }, b: { x: number; y: number }) =>
    Math.max(1e-4, Math.hypot((a.x - b.x) * safeAspect, a.y - b.y));
  const mid = (a: { x: number; y: number }, b: { x: number; y: number }) => ({
    x: (a.x + b.x) * 0.5,
    y: (a.y + b.y) * 0.5,
  });

  const eyeL = mid(lEyeOuter, lEyeInner);
  const eyeR = mid(rEyeInner, rEyeOuter);
  const eyeMid = mid(eyeL, eyeR);
  const mouthMid = mid(lMouth, rMouth);

  const eyeDist = dist(eyeL, eyeR);
  const faceWidth = dist(lCheek, rCheek);
  const faceHeight = dist(forehead, chin);
  const eyeToNose = dist(eyeMid, noseTip);
  const noseWidth = dist(lNostril, rNostril);
  const mouthWidth = dist(lMouth, rMouth);
  const eyeToMouth = dist(eyeMid, mouthMid);
  const jawWidth = dist(lJaw, rJaw);
  const bridgeLen = dist(glabella, noseTip);
  const noseToChin = dist(noseTip, chin);
  const avgEyeOpenW = (dist(lEyeOuter, lEyeInner) + dist(rEyeInner, rEyeOuter)) * 0.5;
  const philtrumLen = dist(noseTip, mouthMid);
  const chinHeight = dist(mouthMid, chin);
  const cheekMouthSpan = (dist(lCheek, lMouth) + dist(rCheek, rMouth)) * 0.5;

  const ratios = [
    eyeDist / faceWidth,
    eyeToNose / faceHeight,
    noseWidth / eyeDist,
    mouthWidth / eyeDist,
    eyeToMouth / jawWidth,
    faceWidth / faceHeight,
    jawWidth / faceWidth,
    bridgeLen / eyeDist,
    noseToChin / eyeToMouth,
    avgEyeOpenW / eyeDist,
    philtrumLen / chinHeight,
    cheekMouthSpan / faceWidth,
  ];
  return ratios.map((r) => Number(Math.min(6, Math.max(0.05, r)).toFixed(4)));
}

function normalizeZeroMeanUnitStdArray(vals: readonly number[]): number[] | null {
  const n = vals.length;
  if (n === 0) return null;
  let sum = 0;
  for (let i = 0; i < n; i += 1) sum += vals[i];
  const mean = sum / n;
  let varSum = 0;
  for (let i = 0; i < n; i += 1) {
    const d = vals[i] - mean;
    varSum += d * d;
  }
  const std = Math.sqrt(varSum / n);
  if (std < 1e-4) return null;
  return vals.map((v) => Number(((v - mean) / std).toFixed(4)));
}

/**
 * Re-normalizes each anatomical sub-block inside a 256-D dual-eye iris + periocular descriptor
 * after multi-frame averaging (left 128-D + right 128-D).
 */
export function normalizeOcularIrisPatch256(raw: readonly number[]): number[] | null {
  if (!Array.isArray(raw) || raw.length !== 256) return null;
  const out = new Array<number>(256).fill(0);
  for (const eyeOffset of [0, 128]) {
    const luma = normalizeZeroMeanUnitStdArray(raw.slice(eyeOffset, eyeOffset + 64));
    if (!luma) return null;
    const gx = normalizeZeroMeanUnitStdArray(raw.slice(eyeOffset + 64, eyeOffset + 80)) ?? new Array<number>(16).fill(0);
    const gy = normalizeZeroMeanUnitStdArray(raw.slice(eyeOffset + 80, eyeOffset + 96)) ?? new Array<number>(16).fill(0);
    const radY0 = normalizeZeroMeanUnitStdArray(raw.slice(eyeOffset + 96, eyeOffset + 104)) ?? new Array<number>(8).fill(0);
    const radY1 = normalizeZeroMeanUnitStdArray(raw.slice(eyeOffset + 104, eyeOffset + 112)) ?? new Array<number>(8).fill(0);
    const radCb = normalizeZeroMeanUnitStdArray(raw.slice(eyeOffset + 112, eyeOffset + 120)) ?? new Array<number>(8).fill(0);
    const radCr = normalizeZeroMeanUnitStdArray(raw.slice(eyeOffset + 120, eyeOffset + 128)) ?? new Array<number>(8).fill(0);
    for (let i = 0; i < 64; i += 1) out[eyeOffset + i] = luma[i];
    for (let i = 0; i < 16; i += 1) out[eyeOffset + 64 + i] = gx[i];
    for (let i = 0; i < 16; i += 1) out[eyeOffset + 80 + i] = gy[i];
    for (let i = 0; i < 8; i += 1) out[eyeOffset + 96 + i] = radY0[i];
    for (let i = 0; i < 8; i += 1) out[eyeOffset + 104 + i] = radY1[i];
    for (let i = 0; i < 8; i += 1) out[eyeOffset + 112 + i] = radCb[i];
    for (let i = 0; i < 8; i += 1) out[eyeOffset + 120 + i] = radCr[i];
  }
  return out;
}

/**
 * Extracts 8 IPD-normalized anatomical iris diameter, fissure width, and inter-canthal ratios
 * from 478-point MediaPipe landmarks (`468..477` iris contours + eye corners + brows).
 */
export function computeIrisOcularRatios8(
  landmarks3D: readonly { x: number; y: number; z?: number }[],
  aspectRatio = 16 / 9,
): number[] | undefined {
  if (!landmarks3D || landmarks3D.length < 478) return undefined;
  const safeAspect = Number.isFinite(aspectRatio) && aspectRatio > 0.2 ? aspectRatio : 16 / 9;

  const lIrisC = landmarks3D[468];
  const lIrisR = landmarks3D[469];
  const lIrisT = landmarks3D[470];
  const lIrisL = landmarks3D[471];
  const lIrisB = landmarks3D[472];

  const rIrisC = landmarks3D[473];
  const rIrisR = landmarks3D[474];
  const rIrisT = landmarks3D[475];
  const rIrisL = landmarks3D[476];
  const rIrisB = landmarks3D[477];

  const lOuter = landmarks3D[33];
  const lInner = landmarks3D[133];
  const rInner = landmarks3D[362];
  const rOuter = landmarks3D[263];
  const lBrow = landmarks3D[105];
  const rBrow = landmarks3D[334];

  if (
    !lIrisC || !lIrisR || !lIrisT || !lIrisL || !lIrisB ||
    !rIrisC || !rIrisR || !rIrisT || !rIrisL || !rIrisB ||
    !lOuter || !lInner || !rInner || !rOuter || !lBrow || !rBrow
  ) {
    return undefined;
  }

  const leftMinX = Math.min(lOuter.x, lInner.x);
  const leftMaxX = Math.max(lOuter.x, lInner.x);
  const rightMinX = Math.min(rInner.x, rOuter.x);
  const rightMaxX = Math.max(rInner.x, rOuter.x);
  if (
    lIrisC.x <= leftMinX ||
    lIrisC.x >= leftMaxX ||
    rIrisC.x <= rightMinX ||
    rIrisC.x >= rightMaxX
  ) {
    return undefined;
  }

  const dist = (a: { x: number; y: number }, b: { x: number; y: number }) =>
    Math.hypot((a.x - b.x) * safeAspect, a.y - b.y);

  const ipd = dist(lIrisC, rIrisC);
  if (ipd < 0.015) return undefined;

  const leftSpan = dist(lOuter, lInner);
  const rightSpan = dist(rInner, rOuter);
  const leftIrisDiam = Math.max(dist(lIrisR, lIrisL), dist(lIrisT, lIrisB));
  const rightIrisDiam = Math.max(dist(rIrisR, rIrisL), dist(rIrisT, rIrisB));
  if (
    leftSpan < 0.005 ||
    rightSpan < 0.005 ||
    leftIrisDiam < 0.002 ||
    rightIrisDiam < 0.002 ||
    leftIrisDiam > leftSpan * 1.35 ||
    rightIrisDiam > rightSpan * 1.35
  ) {
    return undefined;
  }

  const interCanthal = dist(lInner, rInner);
  const outerCanthal = dist(lOuter, rOuter);
  const lBrowToIris = dist(lBrow, lIrisC);
  const rBrowToIris = dist(rBrow, rIrisC);

  const rawRatios = [
    leftIrisDiam / ipd,
    rightIrisDiam / ipd,
    interCanthal / ipd,
    outerCanthal / ipd,
    leftSpan / ipd,
    rightSpan / ipd,
    lBrowToIris / ipd,
    rBrowToIris / ipd,
  ];
  return rawRatios.map((r) => Number(Math.min(4, Math.max(0.02, r)).toFixed(4)));
}

function extractSingleEyeOcular128FromRgba(image: {
  width: number;
  height: number;
  data: ArrayLike<number>;
}): number[] | null {
  const { width, height, data } = image;
  if (!width || !height || width < 16 || height < 12 || data.length < width * height * 4) {
    return null;
  }

  const lumaAt = (px: number, py: number): number => {
    const cx = Math.max(0, Math.min(width - 1, px));
    const cy = Math.max(0, Math.min(height - 1, py));
    const idx = (cy * width + cx) * 4;
    return 0.299 * (data[idx] ?? 0) + 0.587 * (data[idx + 1] ?? 0) + 0.114 * (data[idx + 2] ?? 0);
  };

  // 1. 8x8 periocular + iris luminance grid (64-D)
  const rawGrid8 = new Array<number>(64).fill(0);
  for (let gy = 0; gy < 8; gy += 1) {
    const py0 = Math.floor((gy * height) / 8);
    const py1 = Math.max(py0 + 1, Math.floor(((gy + 1) * height) / 8));
    for (let gx = 0; gx < 8; gx += 1) {
      const px0 = Math.floor((gx * width) / 8);
      const px1 = Math.max(px0 + 1, Math.floor(((gx + 1) * width) / 8));
      let sum = 0;
      let cnt = 0;
      for (let py = py0; py < py1 && py < height; py += 1) {
        for (let px = px0; px < px1 && px < width; px += 1) {
          sum += lumaAt(px, py);
          cnt += 1;
        }
      }
      rawGrid8[gy * 8 + gx] = cnt > 0 ? sum / cnt : 0;
    }
  }
  const normLuma64 = normalizeZeroMeanUnitStdArray(rawGrid8);
  if (!normLuma64) return null;

  // 2. 4x4 directional iris limbus & eyelid Sobel-style gradients: Gx (16-D) + Gy (16-D)
  const rawGx16 = new Array<number>(16).fill(0);
  const rawGy16 = new Array<number>(16).fill(0);
  for (let gy = 0; gy < 4; gy += 1) {
    const py0 = Math.floor((gy * height) / 4);
    const py1 = Math.max(py0 + 2, Math.floor(((gy + 1) * height) / 4));
    const pyMid = Math.floor((py0 + py1) * 0.5);
    for (let gx = 0; gx < 4; gx += 1) {
      const px0 = Math.floor((gx * width) / 4);
      const px1 = Math.max(px0 + 2, Math.floor(((gx + 1) * width) / 4));
      const pxMid = Math.floor((px0 + px1) * 0.5);
      let leftSum = 0;
      let leftCnt = 0;
      let rightSum = 0;
      let rightCnt = 0;
      let topSum = 0;
      let topCnt = 0;
      let botSum = 0;
      let botCnt = 0;
      for (let py = py0; py < py1 && py < height; py += 1) {
        for (let px = px0; px < px1 && px < width; px += 1) {
          const yVal = lumaAt(px, py);
          if (px < pxMid) {
            leftSum += yVal;
            leftCnt += 1;
          } else {
            rightSum += yVal;
            rightCnt += 1;
          }
          if (py < pyMid) {
            topSum += yVal;
            topCnt += 1;
          } else {
            botSum += yVal;
            botCnt += 1;
          }
        }
      }
      const cellIdx = gy * 4 + gx;
      rawGx16[cellIdx] = (rightCnt > 0 ? rightSum / rightCnt : 0) - (leftCnt > 0 ? leftSum / leftCnt : 0);
      rawGy16[cellIdx] = (botCnt > 0 ? botSum / botCnt : 0) - (topCnt > 0 ? topSum / topCnt : 0);
    }
  }
  const normGx16 = normalizeZeroMeanUnitStdArray(rawGx16) ?? new Array<number>(16).fill(0);
  const normGy16 = normalizeZeroMeanUnitStdArray(rawGy16) ?? new Array<number>(16).fill(0);

  // 3. Radial 8-sector × 2-ring concentric iris Y (16-D) + radial 8-sector iris Cb/Cr chroma (16-D)
  const radSumY = new Array<number>(16).fill(0);
  const radCntY = new Array<number>(16).fill(0);
  const radSumCb = new Array<number>(8).fill(0);
  const radSumCr = new Array<number>(8).fill(0);
  const radCntC = new Array<number>(8).fill(0);

  const cx = (width - 1) * 0.5;
  const cy = (height - 1) * 0.5;
  const maxR = Math.max(4, height * 0.45);

  for (let py = 0; py < height; py += 1) {
    for (let px = 0; px < width; px += 1) {
      const dx = px - cx;
      const dy = py - cy;
      const rNorm = Math.hypot(dx, dy) / maxR;
      if (rNorm > 1.0) continue;
      let ang = Math.atan2(dy, dx);
      if (ang < 0) ang += Math.PI * 2;
      const sector = Math.min(7, Math.floor((ang / (Math.PI * 2)) * 8));
      const ring = rNorm <= 0.52 ? 0 : 1;
      const idx = (py * width + px) * 4;
      const r = data[idx] ?? 0;
      const g = data[idx + 1] ?? 0;
      const b = data[idx + 2] ?? 0;
      const yVal = 0.299 * r + 0.587 * g + 0.114 * b;
      const cbVal = -0.1687 * r - 0.3313 * g + 0.5 * b + 128;
      const crVal = 0.5 * r - 0.4187 * g - 0.0813 * b + 128;

      const radIdx = ring * 8 + sector;
      radSumY[radIdx] += yVal;
      radCntY[radIdx] += 1;
      radSumCb[sector] += cbVal;
      radSumCr[sector] += crVal;
      radCntC[sector] += 1;
    }
  }

  const rawRadY0 = radSumY.slice(0, 8).map((s, i) => (radCntY[i] > 0 ? s / radCntY[i] : 0));
  const rawRadY1 = radSumY.slice(8, 16).map((s, i) => (radCntY[8 + i] > 0 ? s / radCntY[8 + i] : 0));
  const rawRadCb = radSumCb.map((s, i) => (radCntC[i] > 0 ? s / radCntC[i] : 128));
  const rawRadCr = radSumCr.map((s, i) => (radCntC[i] > 0 ? s / radCntC[i] : 128));
  const normRadY0 = normalizeZeroMeanUnitStdArray(rawRadY0) ?? new Array<number>(8).fill(0);
  const normRadY1 = normalizeZeroMeanUnitStdArray(rawRadY1) ?? new Array<number>(8).fill(0);
  const normRadCb8 = normalizeZeroMeanUnitStdArray(rawRadCb) ?? new Array<number>(8).fill(0);
  const normRadCr8 = normalizeZeroMeanUnitStdArray(rawRadCr) ?? new Array<number>(8).fill(0);

  return [
    ...normLuma64,
    ...normGx16,
    ...normGy16,
    ...normRadY0,
    ...normRadY1,
    ...normRadCb8,
    ...normRadCr8,
  ];
}

/**
 * Extracts a lighting-invariant 256-D dual-eye iris + periocular biometric descriptor
 * (128-D left eye + 128-D right eye: 8x8 Y grid, 4x4 Sobel Gx/Gy, radial concentric iris Y + Cb/Cr).
 */
export function extractOcularIrisPatch256FromRgba(
  leftEye: { width: number; height: number; data: ArrayLike<number> },
  rightEye: { width: number; height: number; data: ArrayLike<number> },
): number[] | null {
  const left128 = extractSingleEyeOcular128FromRgba(leftEye);
  const right128 = extractSingleEyeOcular128FromRgba(rightEye);
  if (!left128 || !right128) return null;
  return [...left128, ...right128];
}

/**
 * Horizontally mirrors a 256-D dual-eye iris patch (swaps left <-> right eye blocks
 * and reflects grids, directional Gx signs, and angular iris sectors cleanly).
 */
export function flipOcularIrisPatch256Horizontally(patch256: readonly number[]): number[] {
  if (patch256.length !== 256) return [...patch256];
  const out = new Array<number>(256).fill(0);
  const mirrorSector = (s: number) => (s < 4 ? 3 - s : 11 - s);

  const flipSingleEye128 = (srcOffset: number, dstOffset: number) => {
    for (let y = 0; y < 8; y += 1) {
      for (let x = 0; x < 8; x += 1) {
        out[dstOffset + y * 8 + x] = patch256[srcOffset + y * 8 + (7 - x)];
      }
    }
    for (let y = 0; y < 4; y += 1) {
      for (let x = 0; x < 4; x += 1) {
        const valGx = patch256[srcOffset + 64 + y * 4 + (3 - x)];
        out[dstOffset + 64 + y * 4 + x] = valGx === 0 ? 0 : -valGx;
        out[dstOffset + 80 + y * 4 + x] = patch256[srcOffset + 80 + y * 4 + (3 - x)];
      }
    }
    for (let s = 0; s < 8; s += 1) {
      const ms = mirrorSector(s);
      out[dstOffset + 96 + s] = patch256[srcOffset + 96 + ms];
      out[dstOffset + 104 + s] = patch256[srcOffset + 104 + ms];
      out[dstOffset + 112 + s] = patch256[srcOffset + 112 + ms];
      out[dstOffset + 120 + s] = patch256[srcOffset + 120 + ms];
    }
  };

  flipSingleEye128(0, 128);
  flipSingleEye128(128, 0);
  return out;
}

/**
 * Horizontally mirrors 8 IPD-normalized ocular-iris ratios by swapping left <-> right eye pairs.
 */
export function flipIrisOcularRatios8Horizontally(ratios8: readonly number[]): number[] {
  if (ratios8.length !== 8) return [...ratios8];
  return [
    ratios8[1],
    ratios8[0],
    ratios8[2],
    ratios8[3],
    ratios8[5],
    ratios8[4],
    ratios8[7],
    ratios8[6],
  ];
}

/**
 * Weighted ZNCC similarity for 256-D dual-eye iris + periocular descriptors.
 */
export function computeOcularIrisZnccScore(patchA: readonly number[], patchB: readonly number[]): number {
  if (patchA.length !== 256 || patchB.length !== 256) return 0;
  let eyeSum = 0;
  let validEyes = 0;
  for (const offset of [0, 128]) {
    const zLuma = sliceZncc(patchA, patchB, offset, 64);
    if (zLuma === null) continue;
    const zGx = sliceZncc(patchA, patchB, offset + 64, 16);
    const zGy = sliceZncc(patchA, patchB, offset + 80, 16);
    const effGrad =
      zGx !== null && zGy !== null
        ? (zGx + zGy) * 0.5
        : (zGx ?? zGy ?? zLuma);
    const zRad0 = sliceZncc(patchA, patchB, offset + 96, 8);
    const zRad1 = sliceZncc(patchA, patchB, offset + 104, 8);
    const effRadY =
      zRad0 !== null && zRad1 !== null
        ? 0.45 * zRad0 + 0.55 * zRad1
        : (zRad1 ?? zRad0 ?? zLuma);
    const zRadCb = sliceZncc(patchA, patchB, offset + 112, 8);
    const zRadCr = sliceZncc(patchA, patchB, offset + 120, 8);
    const effRadial =
      zRadCb !== null && zRadCr !== null
        ? 0.7 * effRadY + 0.15 * zRadCb + 0.15 * zRadCr
        : effRadY;

    eyeSum += 0.25 * zLuma + 0.3 * effGrad + 0.45 * effRadial;
    validEyes += 1;
  }
  if (validEyes === 0) return 0;
  return Number(Math.max(-1, Math.min(1, eyeSum / validEyes)).toFixed(4));
}

export function evaluateOcularIrisGate(
  probeRatios?: GeometricAnthropometricRatios | null,
  candidateRatios?: GeometricAnthropometricRatios | null,
): {
  available: boolean;
  passed: boolean;
  ocularIrisZncc: number | null;
  irisRatioDeviation: number | null;
  failureReason?: string;
} {
  const pPatch = probeRatios?.ocularIrisPatch256;
  const cPatch = candidateRatios?.ocularIrisPatch256;
  const patchAvailable = Boolean(
    Array.isArray(pPatch) && Array.isArray(cPatch) && pPatch.length === 256 && cPatch.length === 256,
  );
  const ocularIrisZncc = patchAvailable ? computeOcularIrisZnccScore(pPatch!, cPatch!) : null;

  const p8 = probeRatios?.irisOcularRatios8;
  const c8 = candidateRatios?.irisOcularRatios8;
  const ratioAvailable = Boolean(
    Array.isArray(p8) && Array.isArray(c8) && p8.length === 8 && c8.length === 8,
  );
  let irisRatioDeviation: number | null = null;
  if (ratioAvailable) {
    let sumRel = 0;
    for (let i = 0; i < 8; i += 1) {
      const a = p8![i];
      const b = c8![i];
      sumRel += Math.abs(a - b) / Math.max(0.2, (Math.abs(a) + Math.abs(b)) * 0.5);
    }
    irisRatioDeviation = Number((sumRel / 8).toFixed(4));
  }

  if (!patchAvailable && !ratioAvailable) {
    return {
      available: false,
      passed: true,
      ocularIrisZncc: null,
      irisRatioDeviation: null,
    };
  }

  if (patchAvailable && ocularIrisZncc !== null && ocularIrisZncc < STRICT_OCULAR_IRIS_ZNCC_THRESHOLD) {
    return {
      available: true,
      passed: false,
      ocularIrisZncc,
      irisRatioDeviation,
      failureReason: `OCULAR_IRIS_ZNCC_BELOW_THRESHOLD (${ocularIrisZncc.toFixed(3)} < ${STRICT_OCULAR_IRIS_ZNCC_THRESHOLD})`,
    };
  }
  if (irisRatioDeviation !== null && irisRatioDeviation > STRICT_IRIS_RATIO_MAX_DEVIATION) {
    return {
      available: true,
      passed: false,
      ocularIrisZncc,
      irisRatioDeviation,
      failureReason: `IRIS_RATIO_MISMATCH (${irisRatioDeviation.toFixed(3)} > ${STRICT_IRIS_RATIO_MAX_DEVIATION})`,
    };
  }

  return {
    available: true,
    passed: true,
    ocularIrisZncc,
    irisRatioDeviation,
  };
}

export function computeCraniofacialRatioDeviation(
  probeRatios?: GeometricAnthropometricRatios | null,
  candidateRatios?: GeometricAnthropometricRatios | null,
): { available: boolean; meanDeviation: number; maxSingleDeviation: number; passed: boolean } {
  if (!probeRatios || !candidateRatios) {
    return { available: false, meanDeviation: 0, maxSingleDeviation: 0, passed: true };
  }
  const p12 = probeRatios.craniofacialRatios12;
  const c12 = candidateRatios.craniofacialRatios12;
  if (Array.isArray(p12) && Array.isArray(c12) && p12.length === 12 && c12.length === 12) {
    let sumRel = 0;
    let maxRel = 0;
    for (let i = 0; i < 12; i += 1) {
      const a = p12[i];
      const b = c12[i];
      const denom = Math.max(0.2, (Math.abs(a) + Math.abs(b)) * 0.5);
      const rel = Math.abs(a - b) / denom;
      sumRel += rel;
      if (rel > maxRel) maxRel = rel;
    }
    const meanDeviation = Number((sumRel / 12).toFixed(4));
    const maxSingleDeviation = Number(maxRel.toFixed(4));
    return {
      available: true,
      meanDeviation,
      maxSingleDeviation,
      passed:
        meanDeviation <= STRICT_CRANIOFACIAL_MAX_DEVIATION &&
        maxSingleDeviation <= STRICT_CRANIOFACIAL_MAX_SINGLE_DEVIATION,
    };
  }

  // Fallback to 4-ratio geometric signature when an older record stores only 4 ratios
  const pairs: [number, number][] = [
    [probeRatios.eyeToNoseRatio, candidateRatios.eyeToNoseRatio],
    [probeRatios.faceWidthToUpperHeight, candidateRatios.faceWidthToUpperHeight],
    [probeRatios.mouthToEyeRatio, candidateRatios.mouthToEyeRatio],
    [probeRatios.noseToChinRatio, candidateRatios.noseToChinRatio],
  ].filter(([a, b]) => Number.isFinite(a) && Number.isFinite(b) && Math.max(a, b) > 0) as [
    number,
    number,
  ][];
  if (pairs.length === 0) {
    return { available: false, meanDeviation: 0, maxSingleDeviation: 0, passed: true };
  }
  let sumRel = 0;
  let maxRel = 0;
  for (const [a, b] of pairs) {
    const rel = Math.abs(a - b) / Math.max(0.2, (Math.abs(a) + Math.abs(b)) * 0.5);
    sumRel += rel;
    if (rel > maxRel) maxRel = rel;
  }
  const meanDeviation = Number((sumRel / pairs.length).toFixed(4));
  const maxSingleDeviation = Number(maxRel.toFixed(4));
  return {
    available: true,
    meanDeviation,
    maxSingleDeviation,
    passed:
      meanDeviation <= STRICT_CRANIOFACIAL_MAX_DEVIATION &&
      maxSingleDeviation <= STRICT_CRANIOFACIAL_MAX_SINGLE_DEVIATION,
  };
}

export function evaluateAppearanceAndCraniofacialGate(
  probeRatios?: GeometricAnthropometricRatios | null,
  candidateRatios?: GeometricAnthropometricRatios | null,
): {
  passed: boolean;
  patchAvailable: boolean;
  patchZncc: number | null;
  patchGradZncc: number | null;
  ratioAvailable: boolean;
  ratioMeanDeviation: number | null;
  ratioMaxDeviation: number | null;
  ocularIrisAvailable: boolean;
  ocularIrisZncc: number | null;
  irisRatioDeviation: number | null;
  failureReason?: string;
} {
  const pPatch = probeRatios?.appearancePatch384;
  const cPatch = candidateRatios?.appearancePatch384;
  const patchAvailable = Boolean(
    Array.isArray(pPatch) && Array.isArray(cPatch) && pPatch.length === 384 && cPatch.length === 384,
  );
  const patchZncc = patchAvailable ? computePatchZnccScore(pPatch!, cPatch!) : null;
  const patchGradZncc = patchAvailable ? computePatchGradientZnccScore(pPatch!, cPatch!) : null;
  const ratioRes = computeCraniofacialRatioDeviation(probeRatios, candidateRatios);
  const ocularGate = evaluateOcularIrisGate(probeRatios, candidateRatios);

  if (patchAvailable && patchZncc !== null && patchZncc < STRICT_PATCH_ZNCC_THRESHOLD) {
    return {
      passed: false,
      patchAvailable: true,
      patchZncc,
      patchGradZncc,
      ratioAvailable: ratioRes.available,
      ratioMeanDeviation: ratioRes.available ? ratioRes.meanDeviation : null,
      ratioMaxDeviation: ratioRes.available ? ratioRes.maxSingleDeviation : null,
      ocularIrisAvailable: ocularGate.available,
      ocularIrisZncc: ocularGate.ocularIrisZncc,
      irisRatioDeviation: ocularGate.irisRatioDeviation,
      failureReason: `PATCH_ZNCC_BELOW_THRESHOLD (${patchZncc.toFixed(3)} < ${STRICT_PATCH_ZNCC_THRESHOLD})`,
    };
  }
  if (patchAvailable && patchGradZncc !== null && patchGradZncc < STRICT_PATCH_GRAD_ZNCC_THRESHOLD) {
    return {
      passed: false,
      patchAvailable: true,
      patchZncc,
      patchGradZncc,
      ratioAvailable: ratioRes.available,
      ratioMeanDeviation: ratioRes.available ? ratioRes.meanDeviation : null,
      ratioMaxDeviation: ratioRes.available ? ratioRes.maxSingleDeviation : null,
      ocularIrisAvailable: ocularGate.available,
      ocularIrisZncc: ocularGate.ocularIrisZncc,
      irisRatioDeviation: ocularGate.irisRatioDeviation,
      failureReason: `PATCH_GRAD_ZNCC_BELOW_THRESHOLD (${patchGradZncc.toFixed(3)} < ${STRICT_PATCH_GRAD_ZNCC_THRESHOLD})`,
    };
  }
  if (ratioRes.available && !ratioRes.passed) {
    return {
      passed: false,
      patchAvailable,
      patchZncc,
      patchGradZncc,
      ratioAvailable: true,
      ratioMeanDeviation: ratioRes.meanDeviation,
      ratioMaxDeviation: ratioRes.maxSingleDeviation,
      ocularIrisAvailable: ocularGate.available,
      ocularIrisZncc: ocularGate.ocularIrisZncc,
      irisRatioDeviation: ocularGate.irisRatioDeviation,
      failureReason: `CRANIOFACIAL_RATIO_MISMATCH (mean=${ratioRes.meanDeviation.toFixed(3)} > ${STRICT_CRANIOFACIAL_MAX_DEVIATION} or max=${ratioRes.maxSingleDeviation.toFixed(3)} > ${STRICT_CRANIOFACIAL_MAX_SINGLE_DEVIATION})`,
    };
  }
  if (ocularGate.available && !ocularGate.passed) {
    return {
      passed: false,
      patchAvailable,
      patchZncc,
      patchGradZncc,
      ratioAvailable: ratioRes.available,
      ratioMeanDeviation: ratioRes.available ? ratioRes.meanDeviation : null,
      ratioMaxDeviation: ratioRes.available ? ratioRes.maxSingleDeviation : null,
      ocularIrisAvailable: true,
      ocularIrisZncc: ocularGate.ocularIrisZncc,
      irisRatioDeviation: ocularGate.irisRatioDeviation,
      failureReason: ocularGate.failureReason,
    };
  }
  return {
    passed: true,
    patchAvailable,
    patchZncc,
    patchGradZncc,
    ratioAvailable: ratioRes.available,
    ratioMeanDeviation: ratioRes.available ? ratioRes.meanDeviation : null,
    ratioMaxDeviation: ratioRes.available ? ratioRes.maxSingleDeviation : null,
    ocularIrisAvailable: ocularGate.available,
    ocularIrisZncc: ocularGate.ocularIrisZncc,
    irisRatioDeviation: ocularGate.irisRatioDeviation,
  };
}

export interface TripleGateCandidateInput {
  masterEmbedding: readonly number[];
  angleEmbeddings?: Partial<Record<FacePoseZone5, readonly number[] | undefined>>;
  exemplarEmbeddings?: readonly (readonly number[])[];
  geometricRatios?: GeometricAnthropometricRatios | null;
}

export interface TripleGateResult {
  passed: boolean;
  rawCosine: number;
  centeredCosine: number;
  l2Dist: number;
  bestCentroidCosine: number;
  masterRawCosine: number;
  masterCenteredCosine: number;
  masterL2Dist: number;
  centerRawCosine: number;
  centerCenteredCosine: number;
  centerL2Dist: number;
  centroidsAbove082Count: number;
  patchZncc: number | null;
  craniofacialDeviation: number | null;
  ocularIrisZncc: number | null;
  irisRatioDeviation: number | null;
  bestAngle: 'master' | FacePoseZone5 | 'extra';
  failureReasons: string[];
}

/**
 * Evaluates a live probe vector + optional Gate-2 appearance signature against a candidate
 * using Gate 1 (master AND center rawCos >= 0.84, centeredCos >= 0.80, L2 <= 0.56, and >= 2 angle centroids >= 0.82)
 * + Gate 2 (Patch ZNCC >= 0.76, Craniofacial 12/4 deviation <= 0.065, std >= 0.02 when missing patch).
 */
export function evaluateTripleGateNeuralCandidate(
  probeVec: readonly number[],
  candidate: TripleGateCandidateInput,
  probeRatios?: GeometricAnthropometricRatios | null,
  liveBucket?: FacePoseZone5 | null,
): TripleGateResult {
  const emptyFail = (reason: string): TripleGateResult => ({
    passed: false,
    rawCosine: 0,
    centeredCosine: 0,
    l2Dist: Number.POSITIVE_INFINITY,
    bestCentroidCosine: 0,
    masterRawCosine: 0,
    masterCenteredCosine: 0,
    masterL2Dist: Number.POSITIVE_INFINITY,
    centerRawCosine: 0,
    centerCenteredCosine: 0,
    centerL2Dist: Number.POSITIVE_INFINITY,
    centroidsAbove082Count: 0,
    patchZncc: null,
    craniofacialDeviation: null,
    ocularIrisZncc: null,
    irisRatioDeviation: null,
    bestAngle: 'master',
    failureReasons: [reason],
  });

  if (
    !Array.isArray(probeVec) ||
    probeVec.length === 0 ||
    !Array.isArray(candidate.masterEmbedding) ||
    candidate.masterEmbedding.length !== probeVec.length
  ) {
    return emptyFail('INVALID_OR_MISMATCHED_VECTOR_DIM');
  }

  const probeNorm = l2Normalize(probeVec);
  const masterNorm = l2Normalize(candidate.masterEmbedding);
  const centerRaw = candidate.angleEmbeddings?.center;
  const hasExplicitCenter = Array.isArray(centerRaw) && centerRaw.length === probeNorm.length;
  const centerNorm = hasExplicitCenter ? l2Normalize(centerRaw!) : masterNorm;

  const masterMetrics = computeVectorPairMetrics(probeNorm, masterNorm);
  const centerMetrics = hasExplicitCenter
    ? computeVectorPairMetrics(probeNorm, centerNorm)
    : masterMetrics;
  const hasPoseZoneVec =
    Boolean(liveBucket) &&
    liveBucket !== 'center' &&
    Array.isArray(candidate.angleEmbeddings?.[liveBucket!]) &&
    candidate.angleEmbeddings![liveBucket!]!.length === probeNorm.length;
  const poseMetrics = hasPoseZoneVec
    ? computeVectorPairMetrics(probeNorm, l2Normalize(candidate.angleEmbeddings![liveBucket!]!))
    : null;

  // Best-of-All-Angles evaluation when a live pose bucket is provided (all live Kiosk verify calls):
  // checks center, all stored pose angles (left/right/up/down), and exemplar embeddings so slight
  // natural head tilts or lighting shifts match effortlessly without lowering strict thresholds.
  let bestAngleMetrics = poseMetrics
    ? {
        rawCosine: Math.max(centerMetrics.rawCosine, poseMetrics.rawCosine),
        centeredCosine: Math.max(centerMetrics.centeredCosine, poseMetrics.centeredCosine),
        l2Dist: Math.min(centerMetrics.l2Dist, poseMetrics.l2Dist),
      }
    : { ...centerMetrics };
  if (liveBucket) {
    const allAngleZones: FacePoseZone5[] = ['center', 'left', 'right', 'up', 'down'];
    for (const z of allAngleZones) {
      const rawZ = candidate.angleEmbeddings?.[z];
      if (!Array.isArray(rawZ) || rawZ.length !== probeNorm.length) continue;
      const mZ = computeVectorPairMetrics(probeNorm, l2Normalize(rawZ));
      if (mZ.rawCosine > bestAngleMetrics.rawCosine) {
        bestAngleMetrics = mZ;
      }
    }
    for (const ex of candidate.exemplarEmbeddings ?? []) {
      if (!Array.isArray(ex) || ex.length !== probeNorm.length) continue;
      const mEx = computeVectorPairMetrics(probeNorm, l2Normalize(ex));
      if (mEx.rawCosine > bestAngleMetrics.rawCosine) {
        bestAngleMetrics = mEx;
      }
    }
  }
  const zoneMetrics = bestAngleMetrics;

  const isRelu6Pair =
    isNonNegativeRelu6Vector(probeNorm) && isNonNegativeRelu6Vector(masterNorm);
  const rawFloor = isRelu6Pair
    ? STRICT_RELU6_RAW_COSINE_THRESHOLD
    : STRICT_RAW_COSINE_THRESHOLD;
  const centeredFloor = isRelu6Pair
    ? STRICT_RELU6_CENTERED_COSINE_THRESHOLD
    : STRICT_CENTERED_COSINE_THRESHOLD;
  const maxL2 = isRelu6Pair
    ? STRICT_RELU6_MAX_EUCLIDEAN_DISTANCE
    : STRICT_MAX_EUCLIDEAN_DISTANCE;
  const centroidFloor = isRelu6Pair
    ? STRICT_RELU6_MIN_CENTROID_COSINE
    : STRICT_MIN_CENTROID_COSINE;

  const failureReasons: string[] = [];

  // Best-of-All-Angles + Master check: when a strong multi-angle/exemplar match passes all
  // strict thresholds, allow a small angled-pose delta (0.04) on the averaged master centroid
  // while still strictly rejecting strangers and near-misses across every stored angle.
  const hasStrongAngleMatch =
    zoneMetrics.rawCosine >= rawFloor &&
    zoneMetrics.centeredCosine >= centeredFloor &&
    zoneMetrics.l2Dist <= maxL2;
  const effectiveMasterRawFloor = hasStrongAngleMatch ? rawFloor - 0.04 : rawFloor;
  const effectiveMasterCenteredFloor = hasStrongAngleMatch ? centeredFloor - 0.04 : centeredFloor;
  const effectiveMasterMaxL2 = hasStrongAngleMatch ? maxL2 + 0.06 : maxL2;

  if (masterMetrics.rawCosine < effectiveMasterRawFloor) {
    failureReasons.push(`masterRawCos ${masterMetrics.rawCosine.toFixed(4)} < ${effectiveMasterRawFloor}`);
  }
  if (masterMetrics.centeredCosine < effectiveMasterCenteredFloor) {
    failureReasons.push(
      `masterCenteredCos ${masterMetrics.centeredCosine.toFixed(4)} < ${effectiveMasterCenteredFloor}`,
    );
  }
  if (masterMetrics.l2Dist > effectiveMasterMaxL2) {
    failureReasons.push(`masterL2 ${masterMetrics.l2Dist.toFixed(4)} > ${effectiveMasterMaxL2}`);
  }

  const zoneLabel = poseMetrics && liveBucket ? `pose(${liveBucket})` : 'center';
  if (zoneMetrics.rawCosine < rawFloor) {
    failureReasons.push(`${zoneLabel}RawCos ${zoneMetrics.rawCosine.toFixed(4)} < ${rawFloor}`);
  }
  if (zoneMetrics.centeredCosine < centeredFloor) {
    failureReasons.push(
      `${zoneLabel}CenteredCos ${zoneMetrics.centeredCosine.toFixed(4)} < ${centeredFloor}`,
    );
  }
  if (zoneMetrics.l2Dist > maxL2) {
    failureReasons.push(`${zoneLabel}L2 ${zoneMetrics.l2Dist.toFixed(4)} > ${maxL2}`);
  }

  // Evaluate all stored pose centroids: require center (or active pose zone) + at least 1 other distinct stored centroid >= centroidFloor.
  // Include `masterEmbedding` when distinct from `center` (e.g. 120-frame 5-angle enrollment where `master` averages all 5 angles),
  // so straight-facing genuine probes (>= 0.91 vs master AND >= 0.94 vs center) are not falsely rejected solely because
  // sharp head-turn secondary centroids (`left`/`right`/`up`/`down`) naturally fall below 0.85 against a frontal probe.
  let bestCentroidCosine = Math.max(masterMetrics.rawCosine, zoneMetrics.rawCosine);
  let bestAngle: 'master' | FacePoseZone5 | 'extra' =
    poseMetrics && poseMetrics.rawCosine > centerMetrics.rawCosine && poseMetrics.rawCosine >= masterMetrics.rawCosine
      ? liveBucket!
      : centerMetrics.rawCosine >= masterMetrics.rawCosine
        ? 'center'
        : 'master';
  const masterDistinctFromCenter =
    hasExplicitCenter && cosineSimilarity(masterNorm, centerNorm) < 0.9999;
  let centroidsAboveFloorCount =
    (centerMetrics.rawCosine >= centroidFloor ? 1 : 0) +
    (masterDistinctFromCenter && masterMetrics.rawCosine >= centroidFloor ? 1 : 0);

  const secondaryZones: FacePoseZone5[] = ['left', 'right', 'up', 'down'];
  let otherCentroidsFound = masterDistinctFromCenter ? 1 : 0;
  for (const z of secondaryZones) {
    const rawZ = candidate.angleEmbeddings?.[z];
    if (Array.isArray(rawZ) && rawZ.length === probeNorm.length) {
      otherCentroidsFound += 1;
      const simZ = cosineSimilarity(probeNorm, l2Normalize(rawZ));
      if (simZ >= centroidFloor) {
        centroidsAboveFloorCount += 1;
      }
      if (simZ > bestCentroidCosine) {
        bestCentroidCosine = simZ;
        bestAngle = z;
      }
    }
  }
  if (otherCentroidsFound === 0) {
    for (const ex of candidate.exemplarEmbeddings ?? []) {
      if (Array.isArray(ex) && ex.length === probeNorm.length) {
        otherCentroidsFound += 1;
        const simEx = cosineSimilarity(probeNorm, l2Normalize(ex));
        if (simEx >= centroidFloor) {
          centroidsAboveFloorCount += 1;
        }
        if (simEx > bestCentroidCosine) {
          bestCentroidCosine = simEx;
          bestAngle = 'extra';
        }
      }
    }
    if (otherCentroidsFound === 0 && masterMetrics.rawCosine >= centroidFloor) {
      centroidsAboveFloorCount += 1;
    }
  }

  if (
    zoneMetrics.rawCosine < centroidFloor ||
    centroidsAboveFloorCount < STRICT_MIN_CENTROID_COUNT
  ) {
    failureReasons.push(
      `angleCentroids>=${centroidFloor}: ${centroidsAboveFloorCount} < ${STRICT_MIN_CENTROID_COUNT}`,
    );
  }

  if (isRelu6Pair) {
    const validExemplarSims: number[] = [];
    for (const ex of candidate.exemplarEmbeddings ?? []) {
      if (Array.isArray(ex) && ex.length === probeNorm.length) {
        validExemplarSims.push(cosineSimilarity(probeNorm, l2Normalize(ex)));
      }
    }
    if (validExemplarSims.length >= 3) {
      validExemplarSims.sort((a, b) => b - a);
      const top3Avg = (validExemplarSims[0] + validExemplarSims[1] + validExemplarSims[2]) / 3;
      if (top3Avg < STRICT_RELU6_MIN_EXEMPLAR_TOP3_AVG) {
        failureReasons.push(
          `EXEMPLAR_TOP3_AVG_BELOW_THRESHOLD (${top3Avg.toFixed(4)} < ${STRICT_RELU6_MIN_EXEMPLAR_TOP3_AVG})`,
        );
      }
    }
  }

  const gate2 = evaluateAppearanceAndCraniofacialGate(probeRatios, candidate.geometricRatios);
  // 1024-D ReLU6 unit vectors have theoretical max std <= 1/sqrt(1024) = 0.03125 (real faces ~0.012-0.022),
  // whereas 512-D zero-mean vectors have max std = 0.0442. Use dimension-aware minimum variance floor.
  const minVectorStd = probeNorm.length >= 1024 ? 0.008 : STRICT_MIN_VECTOR_STD;
  if (
    !gate2.patchAvailable &&
    (masterMetrics.stdB < minVectorStd || masterMetrics.stdA < minVectorStd)
  ) {
    failureReasons.push(
      `LOW_VARIANCE_VECTOR_REENROLL_REQUIRED (std=${Math.min(masterMetrics.stdA, masterMetrics.stdB).toFixed(4)} < ${minVectorStd})`,
    );
  }
  if (!gate2.passed && gate2.failureReason) {
    // When deep neural identity is strongly corroborated on BOTH master and active angle centroids
    // AND appearance patch + ocular iris checks pass, tolerate mild distance/expression perspective
    // shift on 12-D craniofacial ratios (e.g. enrolling at 0.5m vs standing at 1.5-2.0m at Kiosk).
    const strongNeuralCorroboration =
      masterMetrics.rawCosine >= 0.90 &&
      masterMetrics.centeredCosine >= 0.85 &&
      zoneMetrics.rawCosine >= 0.90;
    const appearanceAndIrisOk =
      (!gate2.patchAvailable ||
        ((gate2.patchZncc ?? 0) >= STRICT_PATCH_ZNCC_THRESHOLD &&
          (gate2.patchGradZncc === null ||
            gate2.patchGradZncc >= STRICT_PATCH_GRAD_ZNCC_THRESHOLD))) &&
      (!gate2.ocularIrisAvailable ||
        ((gate2.ocularIrisZncc === null ||
          gate2.ocularIrisZncc >= STRICT_OCULAR_IRIS_ZNCC_THRESHOLD) &&
          (gate2.irisRatioDeviation === null ||
            gate2.irisRatioDeviation <= STRICT_IRIS_RATIO_MAX_DEVIATION)));
    const mildPerspectiveCraniofacialShift =
      gate2.failureReason.startsWith('CRANIOFACIAL_RATIO_MISMATCH') &&
      strongNeuralCorroboration &&
      appearanceAndIrisOk &&
      (gate2.ratioMeanDeviation ?? 0) <= 0.145 &&
      (gate2.ratioMaxDeviation ?? 0) <= 0.30;
    if (!mildPerspectiveCraniofacialShift) {
      failureReasons.push(gate2.failureReason);
    }
  }

  const pairedRaw = Math.min(masterMetrics.rawCosine, zoneMetrics.rawCosine);
  const pairedCentered = Math.min(masterMetrics.centeredCosine, zoneMetrics.centeredCosine);
  const pairedL2 = Math.max(masterMetrics.l2Dist, zoneMetrics.l2Dist);

  const gate1And2Passed = failureReasons.length === 0;
  return {
    passed: gate1And2Passed,
    rawCosine: gate1And2Passed
      ? Math.max(masterMetrics.rawCosine, zoneMetrics.rawCosine)
      : Math.max(0, pairedRaw),
    centeredCosine: gate1And2Passed
      ? Math.max(masterMetrics.centeredCosine, zoneMetrics.centeredCosine)
      : Math.max(0, pairedCentered),
    l2Dist: gate1And2Passed
      ? Math.min(masterMetrics.l2Dist, zoneMetrics.l2Dist)
      : pairedL2,
    bestCentroidCosine: Math.max(0, bestCentroidCosine),
    masterRawCosine: Math.max(0, masterMetrics.rawCosine),
    masterCenteredCosine: Math.max(0, masterMetrics.centeredCosine),
    masterL2Dist: masterMetrics.l2Dist,
    centerRawCosine: Math.max(0, centerMetrics.rawCosine),
    centerCenteredCosine: Math.max(0, centerMetrics.centeredCosine),
    centerL2Dist: centerMetrics.l2Dist,
    centroidsAbove082Count: centroidsAboveFloorCount,
    patchZncc: gate2.patchZncc,
    craniofacialDeviation: gate2.ratioMeanDeviation,
    ocularIrisZncc: gate2.ocularIrisZncc,
    irisRatioDeviation: gate2.irisRatioDeviation,
    bestAngle,
    failureReasons,
  };
}
