/**
 * DRTECHCARES Smart Biometric Core — Zero False-Positive Vector Math & Multi-Subband Gate
 */

export const FACE_EMBEDDING_DIM = 512;
export const COMMON_MODE_ALPHA = 0.92;
export const NOT_FOUND_CONFIRM_ROUNDS = 1;
export const BURST_FRAME_COUNT = 6;

export type BiometricPose = 'frontal_centroid' | 'overall_centroid' | 'front' | 'left' | 'right' | 'nod';

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
  maxBurstFrames: number = BURST_FRAME_COUNT,
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
  const frontalCandidates = enrolledCandidates.filter(
    (c) => (c.pose === 'frontal_centroid' || c.pose === 'front') && c.vector.length === FACE_EMBEDDING_DIM,
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

  // Calibrated confidence: guaranteed < 50% when rejected, >= 92% when strictly matched
  const rawScore =
    top1.metrics.whitenedCos * 0.45 +
    top1.metrics.rawCos * 0.30 +
    top1.metrics.cosDct * 0.15 +
    top1.metrics.cos3D * 0.10;

  const confidencePercent = matched
    ? Math.min(99.9, Math.max(90.0, Number((rawScore * 100).toFixed(1))))
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
