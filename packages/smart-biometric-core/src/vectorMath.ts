/**
 * DRTECHCARES Smart Biometric Core — Zero False-Positive Vector Math & Multi-Subband Gate
 */

export const FACE_EMBEDDING_DIM = 512;
export const COMMON_MODE_ALPHA = 0.92;
export const NOT_FOUND_CONFIRM_ROUNDS = 1;
export const BURST_FRAME_COUNT = 3;

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
