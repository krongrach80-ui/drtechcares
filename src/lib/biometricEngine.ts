import type Human from '@vladmandic/human';
import type { Config, FaceResult } from '@vladmandic/human';
import {
  DrTechCaresApiClient,
  type ActivePatientRosterItem,
  type AngleEmbeddings5D,
  type PatientBiometricIndexEntry,
  type PatientRecord,
} from '@drtechcares/api-client';
import { notifyBiometricIndexChanged } from './faceMatcher';
import {
  REQUIRED_MATCH_STREAK,
  STRICT_CENTERED_COSINE_THRESHOLD,
  STRICT_COHORT_MARGIN,
  STRICT_DISTINCTIVENESS_MARGIN,
  STRICT_MAX_EUCLIDEAN_DISTANCE,
  STRICT_MIN_CENTROID_COSINE,
  STRICT_RAW_COSINE_THRESHOLD,
  computeMaxImposterCohortScore,
  evaluateTripleGateNeuralCandidate,
  extractFacePatchSignature384FromRgba,
  type GeometricAnthropometricRatios,
} from '@drtechcares/smart-biometric-core';

export { REQUIRED_MATCH_STREAK };

export const BIOMETRIC_VAULT_STORAGE_KEY = 'drtech_biometric_vault_v3';
export const BIOMETRIC_ENGINE_ID = 'human-faceres-1024d' as const;
export const BIOMETRIC_MIN_EMBEDDING_DIM = 128;
export const BIOMETRIC_EXPECTED_DIM = 1024;
export const BIOMETRIC_MATCH_THRESHOLD = STRICT_RAW_COSINE_THRESHOLD;
export const BIOMETRIC_CENTERED_COSINE_THRESHOLD = STRICT_CENTERED_COSINE_THRESHOLD;
export const BIOMETRIC_FINAL_SCORE_THRESHOLD = STRICT_RAW_COSINE_THRESHOLD;
export const BIOMETRIC_EXEMPLAR_FLOOR_THRESHOLD = STRICT_MIN_CENTROID_COSINE;
export const BIOMETRIC_MAX_EUCLIDEAN_DISTANCE = STRICT_MAX_EUCLIDEAN_DISTANCE;
export const BIOMETRIC_HUMAN_SIM_THRESHOLD = STRICT_RAW_COSINE_THRESHOLD;
export const BIOMETRIC_MARGIN_THRESHOLD = STRICT_DISTINCTIVENESS_MARGIN;
export const BIOMETRIC_REQUIRED_STREAK = REQUIRED_MATCH_STREAK;
export const BIOMETRIC_STREAK_MIN_FRAME_COSINE = STRICT_RAW_COSINE_THRESHOLD;
export const BIOMETRIC_MAX_SCORE_VARIANCE = 0.08;
export const BIOMETRIC_EXEMPLAR_TARGET_COUNT = 15;

export const HUMAN_BIOMETRIC_CONFIG: Partial<Config> = {
  modelBasePath: '/models/human/',
  backend: 'webgl',
  async: true,
  warmup: 'face',
  cacheModels: true,
  cacheSensitivity: 0,
  skipAllowed: false,
  deallocate: false,
  debug: false,
  filter: {
    enabled: false,
    equalization: false,
    return: false,
  },
  face: {
    enabled: true,
    detector: {
      rotation: true,
      maxDetected: 1,
      minConfidence: 0.45,
      minSize: 16,
      skipFrames: 0,
      skipTime: 0,
      return: false,
    },
    mesh: { enabled: true },
    iris: { enabled: false },
    description: {
      enabled: true,
      skipFrames: 0,
      skipTime: 0,
      minConfidence: 0.1,
    },
    emotion: { enabled: false },
    antispoof: { enabled: false },
    liveness: { enabled: false },
  },
  body: { enabled: false },
  hand: { enabled: false },
  object: { enabled: false },
  gesture: { enabled: false },
  segmentation: { enabled: false },
};

export interface BiometricProfile {
  patientId: string;
  hn: string;
  fullName: string;
  diagnosis?: string;
  assignedPtId?: string;
  nextAppointmentDate?: string;
  avatarUrl?: string;
  engine: typeof BIOMETRIC_ENGINE_ID;
  dimension: number;
  master_embedding: number[];
  angle_embeddings: AngleEmbeddings5D;
  exemplar_embeddings: number[][];
  geometricRatios?: GeometricAnthropometricRatios;
  framesCaptured: number;
  qualityScore: number;
  updatedAt: string;
}

export interface EnrolledBiometricCandidate {
  patientId: string;
  hn: string;
  fullName: string;
  diagnosis: string;
  assignedPtId: string;
  nextAppointmentDate: string;
  avatarUrl?: string;
  framesCaptured: number;
  qualityScore: number;
  biometricEngine?: PatientRecord['biometricEngine'];
  dimension: number;
  master_embedding: number[];
  angle_embeddings: AngleEmbeddings5D;
  exemplar_embeddings: number[][];
  geometricRatios?: GeometricAnthropometricRatios;
  sourceEntry: PatientBiometricIndexEntry;
}

export interface LoadedBiometricVaultResult {
  candidates: EnrolledBiometricCandidate[];
  compatibleCandidates: EnrolledBiometricCandidate[];
  totalAccounts: number;
  validAccounts: number;
  legacyAccountsCount: number;
  syncedFromSupabase: boolean;
  supabaseSyncedAccounts: number;
  syncError?: string;
}

export interface CanonicalFaceExtraction {
  embedding: number[] | null;
  embeddingDim: number;
  faceCount: number;
  score: number;
  box: { x: number; y: number; width: number; height: number } | null;
  yawDeg: number;
  pitchDeg: number;
  rollDeg: number;
  distanceM: number;
  landmarks: { x: number; y: number; z?: number }[];
}

export interface NeuralSampleInput {
  bucket: keyof AngleEmbeddings5D;
  sharpness: number;
  yawDeg: number;
  pitchDeg: number;
  embedding: number[];
  geometricRatios?: GeometricAnthropometricRatios;
}

export interface PatientMatchScoreResult {
  candidate: EnrolledBiometricCandidate;
  bestSim: number;
  centeredSimilarity: number;
  centroidSim: number;
  exemplarSim: number;
  humanSim: number;
  finalScore: number;
  minEuclideanDist: number;
  consensusTop3Avg: number;
  vectorsAbove073: number;
  uniqueVectorCount: number;
  patchZncc: number | null;
  craniofacialDeviation: number | null;
  passesMultiExemplarGate: boolean;
  tripleGateFailureReasons: string[];
  bestSource: string;
}

export interface Live1toNMatchEvaluation {
  matched: boolean;
  topCandidate: EnrolledBiometricCandidate | null;
  runnerUpCandidate: EnrolledBiometricCandidate | null;
  topSimilarity: number;
  topCenteredSimilarity: number;
  topCentroidSimilarity: number;
  topExemplarSimilarity: number;
  topHumanSimilarity: number;
  topFinalScore: number;
  topEuclideanDistance: number;
  topConsensusTop3Avg: number;
  topPatchZncc: number | null;
  topCraniofacialDeviation: number | null;
  cohortMargin: number;
  runnerUpSimilarity: number;
  runnerUpFinalScore: number;
  margin: number;
  confidencePercent: number;
  bestSource: string;
  reason:
    | 'OK'
    | 'NO_CANDIDATES'
    | 'EXCLUDED_ONLY'
    | 'BELOW_THRESHOLD'
    | 'AMBIGUOUS_CLOSE_CANDIDATES'
    | 'IMPOSTER_COHORT_REJECTED'
    | 'LOW_VARIANCE_REENROLL_REQUIRED';
  validAccounts: number;
  totalAccounts: number;
}

export interface LiveStreakEvaluation {
  confirmed: boolean;
  streak: number;
  requiredStreak: number;
  candidate: EnrolledBiometricCandidate | null;
  runnerUp: EnrolledBiometricCandidate | null;
  similarity: number;
  finalScore: number;
  runnerUpSimilarity: number;
  margin: number;
  confidencePercent: number;
  scoreVariance: number;
  bestSource: string;
  reason: Live1toNMatchEvaluation['reason'] | 'SPOOF_BLOCKED' | 'LIVENESS_PENDING' | 'HIGH_SCORE_VARIANCE';
}

export type BiometricEngineLifecycleStatus = 'idle' | 'loading' | 'ready' | 'error';

/**
 * Pure L2-normalization (unit vector) — NO Zero-Mean Centering and NO post-norm mutation.
 */
export function normalizeVector(vec: ArrayLike<number>): number[] {
  let norm = 0;
  const arr = Array.from(vec, (v) => (Number.isFinite(v) ? Number(v) : 0));
  for (let i = 0; i < arr.length; i += 1) {
    norm += arr[i] * arr[i];
  }
  norm = Math.sqrt(norm) || 1;
  return arr.map((v) => v / norm);
}

export function isValidNeuralEmbedding(
  vec: unknown,
  minDim: number = BIOMETRIC_MIN_EMBEDDING_DIM,
): vec is number[] {
  if (!Array.isArray(vec) || vec.length < minDim) return false;
  let sumSq = 0;
  for (let i = 0; i < vec.length; i += 1) {
    const val = vec[i];
    if (typeof val !== 'number' || !Number.isFinite(val)) return false;
    sumSq += val * val;
  }
  return sumSq > 1e-8;
}

export function cosineSimilarityNormalized(a: readonly number[], b: readonly number[]): number {
  if (a.length === 0 || b.length === 0 || a.length !== b.length) return 0;
  let dot = 0;
  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
  }
  return Math.max(-1, Math.min(1, dot));
}

export function euclideanDistanceNormalized(a: readonly number[], b: readonly number[]): number {
  if (a.length === 0 || b.length === 0 || a.length !== b.length) return Number.POSITIVE_INFINITY;
  let sumSq = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = a[i] - b[i];
    sumSq += d * d;
  }
  return Math.sqrt(sumSq);
}

/**
 * Pure raw L2-normalized cosine similarity without any curve inflation or baseline offset boosting.
 */
export function computeHumanDescriptorSimilarity(
  a: readonly number[],
  b: readonly number[],
  _humanOverride?: Pick<Human, 'match'> | null,
): number {
  if (a.length === 0 || b.length === 0 || a.length !== b.length) return 0;
  const normA = normalizeVector(a);
  const normB = normalizeVector(b);
  return Math.max(0, cosineSimilarityNormalized(normA, normB));
}

let sharedHumanInstance: Human | null = null;
let sharedHumanPromise: Promise<Human> | null = null;
let engineStatus: BiometricEngineLifecycleStatus = 'idle';
let detectRunQueue: Promise<unknown> = Promise.resolve();

export function getBiometricEngineStatus(): BiometricEngineLifecycleStatus {
  return engineStatus;
}

export function getSharedHumanInstanceSync(): Human | null {
  return sharedHumanInstance;
}

export async function getSharedHuman(): Promise<Human> {
  if (typeof window === 'undefined') {
    throw new Error('Human biometric engine requires a browser environment.');
  }
  if (sharedHumanInstance && engineStatus === 'ready') {
    return sharedHumanInstance;
  }
  if (!sharedHumanPromise) {
    engineStatus = 'loading';
    sharedHumanPromise = (async () => {
      const { default: HumanCtor } = await import('@vladmandic/human');
      let instance: Human | null = null;
      for (const backend of ['webgl', 'cpu'] as const) {
        try {
          const candidate = new HumanCtor({
            ...HUMAN_BIOMETRIC_CONFIG,
            backend,
          });
          await candidate.load();
          await candidate.warmup();
          instance = candidate;
          break;
        } catch {
          instance = null;
        }
      }
      if (!instance) {
        throw new Error('Failed to load @vladmandic/human face recognition models.');
      }
      sharedHumanInstance = instance;
      engineStatus = 'ready';
      return instance;
    })().catch((err) => {
      engineStatus = 'error';
      sharedHumanPromise = null;
      throw err;
    });
  }
  return sharedHumanPromise;
}

export async function warmupBiometricEngine(): Promise<boolean> {
  if (typeof window === 'undefined') return false;
  try {
    await getSharedHuman();
    return true;
  } catch {
    return false;
  }
}

function pickPrimaryHumanFace(faces: readonly FaceResult[]): FaceResult | null {
  if (!faces || faces.length === 0) return null;
  let best = faces[0];
  let bestScore = -1;
  for (const f of faces) {
    const area = Array.isArray(f.boxRaw) ? (f.boxRaw[2] ?? 0) * (f.boxRaw[3] ?? 0) : 0;
    const conf = typeof f.boxScore === 'number' ? f.boxScore : typeof f.score === 'number' ? f.score : 0.5;
    const rank = area * (0.4 + conf);
    if (rank > bestScore) {
      bestScore = rank;
      best = f;
    }
  }
  return best ?? null;
}

/**
 * Extracts a canonical L2-normalized neural face embedding ONLY from `@vladmandic/human` `faceres`.
 * Returns `embedding: null` when no real neural descriptor (`>= 128` dims) is present — ZERO fake fallbacks.
 */
export async function extractCanonicalFaceDescriptor(
  input: HTMLVideoElement | HTMLCanvasElement | OffscreenCanvas | ImageData,
): Promise<CanonicalFaceExtraction> {
  const human = await getSharedHuman();
  const task = detectRunQueue.then(() => human.detect(input));
  detectRunQueue = task.catch(() => {});
  const result = await task;

  const faces = Array.isArray(result.face) ? result.face : [];
  const primary = pickPrimaryHumanFace(faces);
  if (!primary) {
    return {
      embedding: null,
      embeddingDim: 0,
      faceCount: 0,
      score: 0,
      box: null,
      yawDeg: 0,
      pitchDeg: 0,
      rollDeg: 0,
      distanceM: 0,
      landmarks: [],
    };
  }

  const rawEmb = Array.isArray(primary.embedding)
    ? primary.embedding
    : primary.embedding && typeof (primary.embedding as ArrayLike<number>).length === 'number'
      ? Array.from(primary.embedding as ArrayLike<number>)
      : null;

  const embedding =
    rawEmb && isValidNeuralEmbedding(rawEmb) ? normalizeVector(rawEmb) : null;

  const rawBox = Array.isArray(primary.boxRaw) && primary.boxRaw.length >= 4
    ? {
        x: Math.max(0, Math.min(1, primary.boxRaw[0])),
        y: Math.max(0, Math.min(1, primary.boxRaw[1])),
        width: Math.max(0.01, Math.min(1, primary.boxRaw[2])),
        height: Math.max(0.01, Math.min(1, primary.boxRaw[3])),
      }
    : null;

  const radToDeg = (rad: number | undefined): number =>
    typeof rad === 'number' && Number.isFinite(rad) ? (rad * 180) / Math.PI : 0;

  const landmarks = Array.isArray(primary.meshRaw)
    ? primary.meshRaw.map((pt) => ({
        x: pt[0] ?? 0.5,
        y: pt[1] ?? 0.5,
        z: pt[2] ?? 0,
      }))
    : [];

  return {
    embedding,
    embeddingDim: embedding?.length ?? 0,
    faceCount: faces.length,
    score: typeof primary.score === 'number' ? primary.score : primary.boxScore ?? 0,
    box: rawBox,
    yawDeg: radToDeg(primary.rotation?.angle?.yaw),
    pitchDeg: radToDeg(primary.rotation?.angle?.pitch),
    rollDeg: radToDeg(primary.rotation?.angle?.roll),
    distanceM: typeof primary.distance === 'number' ? primary.distance : 0,
    landmarks,
  };
}

export const TARGET_CANONICAL_CROP_SIZE = 224;
export const TARGET_MEAN_LUMINANCE = 128;

/**
 * Illumination-Invariant Luminance Normalization (`Mean = 128`, contrast-stabilized)
 * so dark/backlit/clinic lighting produce consistent pixel distributions for deep neural extraction.
 */
export function normalizeImageDataLuminance(
  imageData: ImageData,
  targetMean: number = TARGET_MEAN_LUMINANCE,
  targetStd: number = 42,
): ImageData {
  const { data } = imageData;
  const pixelCount = data.length >>> 2;
  if (pixelCount === 0) return imageData;

  let sumY = 0;
  let validPixels = 0;
  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] === 0) continue;
    const y = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    sumY += y;
    validPixels += 1;
  }
  if (validPixels === 0) return imageData;
  const meanY = sumY / validPixels;

  let varY = 0;
  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] === 0) continue;
    const y = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    const d = y - meanY;
    varY += d * d;
  }
  const stdY = Math.sqrt(varY / validPixels);
  const gain = stdY > 4 ? Math.max(0.55, Math.min(2.3, targetStd / stdY)) : 1;

  const chromaGain = Math.max(0.85, Math.min(1.15, gain));
  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] === 0) continue;
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const y = 0.299 * r + 0.587 * g + 0.114 * b;
    const targetY = targetMean + (y - meanY) * gain;
    data[i] = Math.max(0, Math.min(255, Math.round(targetY + (r - y) * chromaGain)));
    data[i + 1] = Math.max(0, Math.min(255, Math.round(targetY + (g - y) * chromaGain)));
    data[i + 2] = Math.max(0, Math.min(255, Math.round(targetY + (b - y) * chromaGain)));
  }
  return imageData;
}

function applyCanonicalLuminanceToCanvas(canvas: HTMLCanvasElement): void {
  const ctx = canvas.getContext('2d', { alpha: false, willReadFrequently: true });
  if (!ctx) return;
  try {
    const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
    normalizeImageDataLuminance(img, TARGET_MEAN_LUMINANCE);
    ctx.putImageData(img, 0, 0);
  } catch {
    // Ignore tainted canvas edge cases gracefully
  }
}

/**
 * Extracts a lighting-invariant 384-D Zero-Mean Normalized (mean=0, std=1) appearance patch
 * (16x16 inner-face luminance = 256 + 8x8 Cb/Cr chrominance = 128) from a canonical inner-face crop.
 */
export function extractFacePatchSignature384(
  source: HTMLCanvasElement | ImageData | null | undefined,
): number[] | null {
  if (!source) return null;
  if ('data' in source && 'width' in source && 'height' in source && !('getContext' in source)) {
    return extractFacePatchSignature384FromRgba(source);
  }
  if ('getContext' in source && typeof (source as HTMLCanvasElement).getContext === 'function') {
    const canvas = source as HTMLCanvasElement;
    if (canvas.width < 16 || canvas.height < 16) return null;
    try {
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (!ctx) return null;
      const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
      return extractFacePatchSignature384FromRgba(img);
    } catch {
      return null;
    }
  }
  return null;
}

/**
 * Captures a race-free, illumination-invariant inner-face snapshot (`224x224`, Mean Luminance `128`):
 * - When `landmarks` (FaceMesh/Human) are provided: levels eye-roll to 0 deg and tightly crops
 *   eyebrows-to-chin and cheekbone-to-cheekbone without wall/background noise.
 * - When only `faceBox` is provided: tightens onto inner face at `224x224` with Mean Luminance `128`.
 * - Falls back to full scaled frame (`448px`) only when neither box nor landmarks are known yet.
 */
export function captureCanonicalFaceSnapshot(
  video: HTMLVideoElement,
  faceBox?: { x: number; y: number; width: number; height: number } | null,
  targetCanvas?: HTMLCanvasElement | null,
  landmarks?: readonly { x: number; y: number; z?: number }[] | null,
): HTMLCanvasElement | null {
  if (typeof document === 'undefined' || !video || video.videoWidth <= 0 || video.videoHeight <= 0) {
    return null;
  }
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const canvas = targetCanvas || document.createElement('canvas');

  if (landmarks && landmarks.length >= 264) {
    const lEyeOuter = landmarks[33];
    const lEyeInner = landmarks[133];
    const rEyeInner = landmarks[362];
    const rEyeOuter = landmarks[263];
    const browForehead = landmarks[10];
    const chin = landmarks[152];
    const leftCheek = landmarks[234];
    const rightCheek = landmarks[454];
    if (lEyeOuter && lEyeInner && rEyeInner && rEyeOuter && browForehead && chin && leftCheek && rightCheek) {
      const lx = ((lEyeOuter.x + lEyeInner.x) * 0.5) * vw;
      const ly = ((lEyeOuter.y + lEyeInner.y) * 0.5) * vh;
      const rx = ((rEyeInner.x + rEyeOuter.x) * 0.5) * vw;
      const ry = ((rEyeInner.y + rEyeOuter.y) * 0.5) * vh;
      const eyeDist = Math.hypot(rx - lx, ry - ly);

      if (eyeDist >= 12) {
        const rollRad = Math.atan2(ry - ly, rx - lx);
        const xs = [leftCheek.x * vw, rightCheek.x * vw, lx, rx];
        const ys = [browForehead.y * vh, chin.y * vh, ly, ry];
        const minX = Math.min(...xs);
        const maxX = Math.max(...xs);
        const minY = Math.min(...ys);
        const maxY = Math.max(...ys);
        // Include moderate head contour padding (1.32x) so @vladmandic/human BlazeFace reliably detects turned yaw/pitch faces
        const faceSpan = Math.max(42, Math.max((maxX - minX) * 1.32, (maxY - minY) * 1.32));
        const cx = (minX + maxX) * 0.5;
        const cy = minY + (maxY - minY) * 0.48;

        if (canvas.width !== TARGET_CANONICAL_CROP_SIZE || canvas.height !== TARGET_CANONICAL_CROP_SIZE) {
          canvas.width = TARGET_CANONICAL_CROP_SIZE;
          canvas.height = TARGET_CANONICAL_CROP_SIZE;
        }
        const ctx = canvas.getContext('2d', { alpha: false, willReadFrequently: true });
        if (!ctx) return null;
        const scale = TARGET_CANONICAL_CROP_SIZE / faceSpan;
        ctx.save();
        ctx.fillStyle = '#808080';
        ctx.fillRect(0, 0, TARGET_CANONICAL_CROP_SIZE, TARGET_CANONICAL_CROP_SIZE);
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        ctx.translate(TARGET_CANONICAL_CROP_SIZE * 0.5, TARGET_CANONICAL_CROP_SIZE * 0.5);
        ctx.rotate(-rollRad);
        ctx.scale(scale, scale);
        ctx.translate(-cx, -cy);
        ctx.drawImage(video, 0, 0, vw, vh);
        ctx.restore();
        applyCanonicalLuminanceToCanvas(canvas);
        return canvas;
      }
    }
  }

  if (faceBox && faceBox.height > 0 && faceBox.width > 0) {
    const cxPx = (faceBox.x + faceBox.width * 0.5) * vw;
    const cyPx = (faceBox.y + faceBox.height * 0.48) * vh;
    // Tight 1:1 pixel-square inner-face crop (avoiding 16:9 aspect distortion into 224x224)
    const padScale = faceBox.height < 0.18 ? 1.45 : 1.18;
    const sidePx = Math.max(
      32,
      Math.min(Math.min(vw, vh), Math.round(Math.max(faceBox.width * vw, faceBox.height * vh) * padScale)),
    );
    const sx = Math.max(0, Math.min(vw - sidePx, Math.round(cxPx - sidePx * 0.5)));
    const sy = Math.max(0, Math.min(vh - sidePx, Math.round(cyPx - sidePx * 0.5)));
    const sw = Math.max(32, Math.min(vw - sx, sidePx));
    const sh = Math.max(32, Math.min(vh - sy, sidePx));

    if (canvas.width !== TARGET_CANONICAL_CROP_SIZE || canvas.height !== TARGET_CANONICAL_CROP_SIZE) {
      canvas.width = TARGET_CANONICAL_CROP_SIZE;
      canvas.height = TARGET_CANONICAL_CROP_SIZE;
    }
    const ctx = canvas.getContext('2d', { alpha: false, willReadFrequently: true });
    if (!ctx) return null;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(video, sx, sy, sw, sh, 0, 0, TARGET_CANONICAL_CROP_SIZE, TARGET_CANONICAL_CROP_SIZE);
    applyCanonicalLuminanceToCanvas(canvas);
    return canvas;
  }

  const maxW = 448;
  const scale = vw > maxW ? maxW / vw : 1;
  const outW = Math.max(160, Math.round(vw * scale));
  const outH = Math.max(120, Math.round(vh * scale));
  if (canvas.width !== outW || canvas.height !== outH) {
    canvas.width = outW;
    canvas.height = outH;
  }
  const ctx = canvas.getContext('2d', { alpha: false, willReadFrequently: true });
  if (!ctx) return null;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(video, 0, 0, outW, outH);
  return canvas;
}

function qualityWeightedCentroid(samples: readonly NeuralSampleInput[], dim: number): number[] {
  const sum = new Array<number>(dim).fill(0);
  for (const s of samples) {
    const w = 1 + Math.min(1, Math.max(0, s.sharpness) / 150);
    for (let i = 0; i < dim; i += 1) {
      sum[i] += s.embedding[i] * w;
    }
  }
  return normalizeVector(sum);
}

/**
 * Builds a complete `BiometricProfile` (master + 5 angle centroids + Top 10 sharp KNN exemplars)
 * strictly from verified neural descriptor samples with uniform L2-normalization.
 */
export function buildBiometricProfileFromSamples(
  rawSamples: readonly NeuralSampleInput[],
  meta: {
    patientId?: string;
    hn?: string;
    fullName?: string;
    qualityScore?: number;
    avatarUrl?: string;
    geometricRatios?: GeometricAnthropometricRatios;
  } = {},
): BiometricProfile | null {
  const valid = rawSamples.filter((s) => isValidNeuralEmbedding(s.embedding));
  if (valid.length === 0) return null;

  // Enforce consistent dimension matching majority/last frame
  const dimCounts = new Map<number, number>();
  for (const s of valid) {
    dimCounts.set(s.embedding.length, (dimCounts.get(s.embedding.length) ?? 0) + 1);
  }
  let targetDim = valid[0].embedding.length;
  let maxCount = 0;
  for (const [dim, count] of dimCounts.entries()) {
    if (count > maxCount) {
      maxCount = count;
      targetDim = dim;
    }
  }

  const normalizedSamples: NeuralSampleInput[] = valid
    .filter((s) => s.embedding.length === targetDim)
    .map((s) => ({
      ...s,
      embedding: normalizeVector(s.embedding),
    }));
  if (normalizedSamples.length === 0) return null;

  const master_embedding = qualityWeightedCentroid(normalizedSamples, targetDim);
  const bucketCentroid = (bucket: keyof AngleEmbeddings5D): number[] => {
    const items = normalizedSamples.filter((s) => s.bucket === bucket);
    return items.length > 0 ? qualityWeightedCentroid(items, targetDim) : master_embedding;
  };
  const angle_embeddings: AngleEmbeddings5D = {
    center: bucketCentroid('center'),
    left: bucketCentroid('left'),
    right: bucketCentroid('right'),
    up: bucketCentroid('up'),
    down: bucketCentroid('down'),
  };

  // 15 sharpest distinct frames distributed evenly across all 5 pose angles (3 per bucket + sharpest fill)
  const bucketOrder: (keyof AngleEmbeddings5D)[] = ['center', 'left', 'right', 'up', 'down'];
  const selectedExemplars: number[][] = [];
  const usedVectors = new Set<string>();
  const pushDistinctExemplar = (vec: number[]): boolean => {
    const sig = `${vec[0]?.toFixed(4)}|${vec[1]?.toFixed(4)}|${vec[Math.floor(vec.length / 2)]?.toFixed(4)}|${vec[vec.length - 1]?.toFixed(4)}`;
    if (usedVectors.has(sig)) return false;
    usedVectors.add(sig);
    selectedExemplars.push(vec);
    return true;
  };

  for (const bucket of bucketOrder) {
    const rankedBucket = normalizedSamples
      .filter((s) => s.bucket === bucket)
      .sort((a, b) => b.sharpness - a.sharpness);
    let pickedForBucket = 0;
    for (const s of rankedBucket) {
      if (pickedForBucket >= 3 || selectedExemplars.length >= BIOMETRIC_EXEMPLAR_TARGET_COUNT) break;
      if (pushDistinctExemplar(s.embedding)) pickedForBucket += 1;
    }
  }

  // Fill remaining slots up to 15 from sharpest frames overall (prioritizing frontal/mild angles)
  const allRanked = [...normalizedSamples].sort((a, b) => {
    const aFrontal = a.bucket === 'center' ? 1 : 0;
    const bFrontal = b.bucket === 'center' ? 1 : 0;
    if (aFrontal !== bFrontal) return bFrontal - aFrontal;
    return b.sharpness - a.sharpness;
  });
  for (const s of allRanked) {
    if (selectedExemplars.length >= BIOMETRIC_EXEMPLAR_TARGET_COUNT) break;
    pushDistinctExemplar(s.embedding);
  }
  if (selectedExemplars.length === 0) {
    selectedExemplars.push(master_embedding);
  }

  // Merge zero-mean appearance patch & 12 craniofacial anatomical ratios into geometricRatios when collected on samples
  let resolvedGeometricRatios = meta.geometricRatios ? { ...meta.geometricRatios } : undefined;
  const centerSamplePatches = normalizedSamples
    .filter((s) => s.bucket === 'center' && Array.isArray(s.geometricRatios?.appearancePatch384) && s.geometricRatios!.appearancePatch384!.length === 384)
    .map((s) => s.geometricRatios!.appearancePatch384!);
  const fallbackPatches = normalizedSamples
    .filter((s) => Array.isArray(s.geometricRatios?.appearancePatch384) && s.geometricRatios!.appearancePatch384!.length === 384)
    .map((s) => s.geometricRatios!.appearancePatch384!);
  const patchPool = centerSamplePatches.length > 0 ? centerSamplePatches : fallbackPatches;
  if (patchPool.length > 0 && !resolvedGeometricRatios?.appearancePatch384) {
    const avgPatch = new Array<number>(384).fill(0);
    for (const p of patchPool) {
      for (let i = 0; i < 384; i += 1) avgPatch[i] += p[i];
    }
    for (let i = 0; i < 384; i += 1) avgPatch[i] /= patchPool.length;
    const normBlock = (arr: number[], start: number, len: number) => {
      let sum = 0;
      for (let i = 0; i < len; i += 1) sum += arr[start + i];
      const mean = sum / len;
      let varSum = 0;
      for (let i = 0; i < len; i += 1) {
        const d = arr[start + i] - mean;
        varSum += d * d;
      }
      const std = Math.sqrt(varSum / len);
      if (std > 1e-4) {
        for (let i = 0; i < len; i += 1) {
          arr[start + i] = Number(((arr[start + i] - mean) / std).toFixed(4));
        }
      }
    };
    normBlock(avgPatch, 0, 256);
    normBlock(avgPatch, 256, 64);
    normBlock(avgPatch, 320, 64);
    resolvedGeometricRatios = {
      ...(resolvedGeometricRatios ?? {
        eyeToNoseRatio: 1,
        faceWidthToUpperHeight: 1,
        mouthToEyeRatio: 1,
        noseToChinRatio: 1,
      }),
      appearancePatch384: avgPatch,
    };
  }

  const centerCranio = normalizedSamples
    .filter(
      (s) =>
        s.bucket === 'center' &&
        Array.isArray(s.geometricRatios?.craniofacialRatios12) &&
        s.geometricRatios!.craniofacialRatios12!.length === 12,
    )
    .map((s) => s.geometricRatios!.craniofacialRatios12!);
  const fallbackCranio = normalizedSamples
    .filter(
      (s) =>
        Array.isArray(s.geometricRatios?.craniofacialRatios12) &&
        s.geometricRatios!.craniofacialRatios12!.length === 12,
    )
    .map((s) => s.geometricRatios!.craniofacialRatios12!);
  const cranioPool = centerCranio.length > 0 ? centerCranio : fallbackCranio;
  if (cranioPool.length > 0 && !resolvedGeometricRatios?.craniofacialRatios12) {
    const avgCranio = new Array<number>(12).fill(0);
    for (const c of cranioPool) {
      for (let i = 0; i < 12; i += 1) avgCranio[i] += c[i];
    }
    for (let i = 0; i < 12; i += 1) {
      avgCranio[i] = Number((avgCranio[i] / cranioPool.length).toFixed(4));
    }
    resolvedGeometricRatios = {
      ...(resolvedGeometricRatios ?? {
        eyeToNoseRatio: 1,
        faceWidthToUpperHeight: 1,
        mouthToEyeRatio: 1,
        noseToChinRatio: 1,
      }),
      craniofacialRatios12: avgCranio,
    };
  }

  return {
    patientId: meta.patientId ?? '',
    hn: meta.hn ?? '',
    fullName: meta.fullName ?? '',
    engine: BIOMETRIC_ENGINE_ID,
    dimension: targetDim,
    master_embedding,
    angle_embeddings,
    exemplar_embeddings: selectedExemplars,
    ...(meta.avatarUrl ? { avatarUrl: meta.avatarUrl } : {}),
    ...(resolvedGeometricRatios ? { geometricRatios: resolvedGeometricRatios } : {}),
    framesCaptured: normalizedSamples.length,
    qualityScore: meta.qualityScore ?? 0.96,
    updatedAt: new Date().toISOString(),
  };
}

export function readLocalBiometricVault(): BiometricProfile[] {
  if (typeof globalThis.localStorage === 'undefined') return [];
  try {
    const raw = globalThis.localStorage.getItem(BIOMETRIC_VAULT_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (item): item is BiometricProfile =>
        Boolean(item) &&
        typeof item === 'object' &&
        typeof (item as BiometricProfile).patientId === 'string' &&
        isValidNeuralEmbedding((item as BiometricProfile).master_embedding),
    );
  } catch {
    return [];
  }
}

export function saveProfileToLocalBiometricVault(profile: BiometricProfile): void {
  if (typeof globalThis.localStorage === 'undefined') return;
  if (!profile.patientId || !isValidNeuralEmbedding(profile.master_embedding)) return;
  try {
    const normalizedMaster = normalizeVector(profile.master_embedding);
    const dim = normalizedMaster.length;
    const cleanAngle = (v: unknown): number[] =>
      isValidNeuralEmbedding(v) && v.length === dim ? normalizeVector(v) : normalizedMaster;
    const cleanProfile: BiometricProfile = {
      ...profile,
      engine: BIOMETRIC_ENGINE_ID,
      dimension: dim,
      master_embedding: normalizedMaster,
      angle_embeddings: {
        center: cleanAngle(profile.angle_embeddings?.center),
        left: cleanAngle(profile.angle_embeddings?.left),
        right: cleanAngle(profile.angle_embeddings?.right),
        up: cleanAngle(profile.angle_embeddings?.up),
        down: cleanAngle(profile.angle_embeddings?.down),
      },
      exemplar_embeddings: (profile.exemplar_embeddings ?? [])
        .filter((v) => isValidNeuralEmbedding(v) && v.length === dim)
        .map((v) => normalizeVector(v))
        .slice(0, BIOMETRIC_EXEMPLAR_TARGET_COUNT),
      updatedAt: profile.updatedAt || new Date().toISOString(),
    };
    const existing = readLocalBiometricVault();
    const filtered = existing.filter(
      (item) =>
        item.patientId !== cleanProfile.patientId &&
        (!cleanProfile.hn || item.hn !== cleanProfile.hn),
    );
    filtered.unshift(cleanProfile);
    globalThis.localStorage.setItem(
      BIOMETRIC_VAULT_STORAGE_KEY,
      JSON.stringify(filtered.slice(0, 100)),
    );

    // Mirror biometric vectors directly into drtech_patients so Kiosk reads from BOTH storage layers reliably
    try {
      const rawPatients = globalThis.localStorage.getItem('drtech_patients');
      const parsedPatients = rawPatients ? (JSON.parse(rawPatients) as unknown) : [];
      if (Array.isArray(parsedPatients)) {
        const pidLo = cleanProfile.patientId.trim().toLowerCase();
        const hnLo = (cleanProfile.hn ?? '').trim().toLowerCase();
        let updatedAny = false;
        const nextPatients = parsedPatients.map((rec) => {
          if (!rec || typeof rec !== 'object') return rec;
          const r = rec as Record<string, unknown>;
          const rId = String(r.id ?? '').trim().toLowerCase();
          const rHn = String(r.hn ?? '').trim().toLowerCase();
          if ((pidLo && rId === pidLo) || (hnLo && rHn === hnLo)) {
            updatedAny = true;
            return {
              ...r,
              faceBiometricStatus: 'VERIFIED',
              faceFramesCount: cleanProfile.framesCaptured,
              biometricEngine: BIOMETRIC_ENGINE_ID,
              masterEmbedding: cleanProfile.master_embedding,
              master_embedding: cleanProfile.master_embedding,
              angleEmbeddings: cleanProfile.angle_embeddings,
              angle_embeddings: cleanProfile.angle_embeddings,
              faceEmbedding: cleanProfile.master_embedding,
              face_embedding: cleanProfile.master_embedding,
              exemplarEmbeddings: cleanProfile.exemplar_embeddings,
              ...(cleanProfile.geometricRatios ? { geometricRatios: cleanProfile.geometricRatios } : {}),
              ...(cleanProfile.avatarUrl ? { faceImageUrl: cleanProfile.avatarUrl } : {}),
            };
          }
          return rec;
        });
        if (!updatedAny && cleanProfile.patientId) {
          nextPatients.unshift({
            id: cleanProfile.patientId,
            hn: cleanProfile.hn,
            fullName: cleanProfile.fullName,
            age: 45,
            phone: '',
            diagnosis: cleanProfile.diagnosis ?? 'General Rehabilitation',
            assignedPtId: cleanProfile.assignedPtId ?? 'user-pt-1',
            delegatedToPtId: null,
            delegationNote: null,
            nextAppointmentDate: cleanProfile.nextAppointmentDate ?? '',
            registrationStatus: 'active',
            enrolledAt: cleanProfile.updatedAt,
            faceBiometricStatus: 'VERIFIED',
            faceFramesCount: cleanProfile.framesCaptured,
            biometricEngine: BIOMETRIC_ENGINE_ID,
            masterEmbedding: cleanProfile.master_embedding,
            master_embedding: cleanProfile.master_embedding,
            angleEmbeddings: cleanProfile.angle_embeddings,
            angle_embeddings: cleanProfile.angle_embeddings,
            faceEmbedding: cleanProfile.master_embedding,
            face_embedding: cleanProfile.master_embedding,
            exemplarEmbeddings: cleanProfile.exemplar_embeddings,
            ...(cleanProfile.geometricRatios ? { geometricRatios: cleanProfile.geometricRatios } : {}),
            ...(cleanProfile.avatarUrl ? { faceImageUrl: cleanProfile.avatarUrl } : {}),
          });
        }
        globalThis.localStorage.setItem('drtech_patients', JSON.stringify(nextPatients.slice(0, 120)));
      }
    } catch {
      // Ignore secondary mirror errors
    }
  } catch {
    // Ignore storage quota errors
  }
}

export function removeProfileFromLocalBiometricVault(patientIdOrHn: string): void {
  if (typeof globalThis.localStorage === 'undefined' || !patientIdOrHn) return;
  try {
    const existing = readLocalBiometricVault();
    const next = existing.filter(
      (item) => item.patientId !== patientIdOrHn && item.hn !== patientIdOrHn,
    );
    if (next.length === 0) {
      globalThis.localStorage.removeItem(BIOMETRIC_VAULT_STORAGE_KEY);
    } else {
      globalThis.localStorage.setItem(BIOMETRIC_VAULT_STORAGE_KEY, JSON.stringify(next));
    }
  } catch {
    // Ignore storage errors
  }
}

/**
 * Replaces the cached local vault strictly from Supabase-verified biometric records
 * so stale or deleted accounts never persist across sessions as phantom profiles.
 */
export function replaceLocalBiometricVaultFromSupabase(
  candidates: readonly EnrolledBiometricCandidate[],
): void {
  if (typeof globalThis.localStorage === 'undefined') return;
  try {
    const cleanProfiles: BiometricProfile[] = [];
    for (const c of candidates) {
      if (!isValidNeuralEmbedding(c.master_embedding)) continue;
      cleanProfiles.push({
        patientId: c.patientId,
        hn: c.hn,
        fullName: c.fullName,
        diagnosis: c.diagnosis,
        assignedPtId: c.assignedPtId,
        nextAppointmentDate: c.nextAppointmentDate,
        avatarUrl: c.avatarUrl,
        engine: BIOMETRIC_ENGINE_ID,
        dimension: c.dimension,
        master_embedding: c.master_embedding,
        angle_embeddings: c.angle_embeddings,
        exemplar_embeddings: c.exemplar_embeddings.slice(0, BIOMETRIC_EXEMPLAR_TARGET_COUNT),
        ...(c.geometricRatios ? { geometricRatios: c.geometricRatios } : {}),
        framesCaptured: c.framesCaptured,
        qualityScore: c.qualityScore,
        updatedAt: new Date().toISOString(),
      });
    }
    globalThis.localStorage.setItem(
      BIOMETRIC_VAULT_STORAGE_KEY,
      JSON.stringify(cleanProfiles.slice(0, 150)),
    );
  } catch {
    // Ignore storage quota errors
  }
}

function extractMissingPostgrestColumn(err: { code?: string; message?: string } | null | undefined): string | null {
  if (!err?.message) return null;
  const msg = err.message;
  const m =
    msg.match(/Could not find the ['"]?([a-zA-Z0-9_]+)['"]? column/i) ||
    msg.match(/column\s+["']?([a-zA-Z0-9_]+)["']?\s+(?:of relation|does not exist)/i);
  return m?.[1] ?? null;
}

const PROTECTED_VECTOR_COLUMNS = new Set([
  'biometric_data',
  'master_embedding',
  'angle_embeddings',
  'exemplar_embeddings',
  'embedding_dim',
]);

async function executeSupabaseWriteWithSchemaCompat(
  run: (payload: Record<string, unknown>) => PromiseLike<{ data?: unknown; error: { code?: string; message?: string } | null }>,
  initialPayload: Record<string, unknown>,
): Promise<{ data?: unknown; error: { code?: string; message?: string } | null }> {
  const payload = { ...initialPayload };
  for (let attempt = 0; attempt < 6; attempt += 1) {
    const res = await run(payload);
    if (!res.error) return res;
    const missingCol = extractMissingPostgrestColumn(res.error);
    if (!missingCol || PROTECTED_VECTOR_COLUMNS.has(missingCol) || !(missingCol in payload)) {
      return res;
    }
    delete payload[missingCol];
  }
  return run(payload);
}

/**
 * Persists a newly enrolled or re-enrolled `BiometricProfile` to Supabase Cloud (`patients` + `patient_biometrics`)
 * with strict **Read-After-Write `SELECT` Verification (`embedding_dim >= 128`)**.
 * Never swallows Supabase errors or writes fake vectors.
 */
export async function syncPatientBiometricEverywhere(input: {
  apiClient: DrTechCaresApiClient;
  patient: PatientRecord;
  masterEmbedding: number[];
  angleEmbeddings: AngleEmbeddings5D;
  exemplarEmbeddings?: number[][];
  geometricRatios?: GeometricAnthropometricRatios;
  framesCaptured?: number;
  qualityScore?: number;
  requireSupabase?: boolean;
}): Promise<BiometricProfile | null> {
  if (!isValidNeuralEmbedding(input.masterEmbedding)) {
    throw new Error('ไม่พบเวกเตอร์ใบหน้าจริง (Neural Embedding >= 128 มิติ) ปฏิเสธการบันทึกข้อมูลจำลอง');
  }
  const masterNorm = normalizeVector(input.masterEmbedding);
  const dim = masterNorm.length;
  const normAngle = (v: unknown): number[] =>
    isValidNeuralEmbedding(v) && v.length === dim ? normalizeVector(v) : masterNorm;
  const angleNorm: AngleEmbeddings5D = {
    center: normAngle(input.angleEmbeddings?.center),
    left: normAngle(input.angleEmbeddings?.left),
    right: normAngle(input.angleEmbeddings?.right),
    up: normAngle(input.angleEmbeddings?.up),
    down: normAngle(input.angleEmbeddings?.down),
  };
  const exemplars = (input.exemplarEmbeddings ?? [])
    .filter((v) => isValidNeuralEmbedding(v) && v.length === dim)
    .map((v) => normalizeVector(v))
    .slice(0, BIOMETRIC_EXEMPLAR_TARGET_COUNT);

  const profile: BiometricProfile = {
    patientId: input.patient.id,
    hn: input.patient.hn,
    fullName: input.patient.fullName,
    diagnosis: input.patient.diagnosis,
    assignedPtId: input.patient.assignedPtId,
    nextAppointmentDate: input.patient.nextAppointmentDate,
    avatarUrl: input.patient.faceImageUrl,
    engine: BIOMETRIC_ENGINE_ID,
    dimension: dim,
    master_embedding: masterNorm,
    angle_embeddings: angleNorm,
    exemplar_embeddings: exemplars.length > 0 ? exemplars : [masterNorm],
    ...(input.geometricRatios ? { geometricRatios: input.geometricRatios } : {}),
    framesCaptured: input.framesCaptured ?? 120,
    qualityScore: input.qualityScore ?? 0.96,
    updatedAt: new Date().toISOString(),
  };

  const supabase = input.apiClient.getSupabaseClient();
  if (!supabase && input.requireSupabase) {
    throw new Error('ไม่ได้เชื่อมต่อฐานข้อมูล Supabase กรุณาตรวจสอบ VITE_SUPABASE_URL และ VITE_SUPABASE_ANON_KEY');
  }

  const UUID_STRICT_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
  const safeAssignedPtId =
    input.patient.assignedPtId && UUID_STRICT_RE.test(input.patient.assignedPtId.trim())
      ? input.patient.assignedPtId.trim()
      : null;
  const dbExemplars = profile.exemplar_embeddings.slice(0, 9);

  // Stop browser -> direct Supabase Postgres writes unless explicitly required by a strict test (`requireSupabase: true`).
  // Normal UI enrollment persists through Worker API (/api/biometric/enroll & /api/patients & /api/biometrics) + localStorage.
  if (supabase && input.requireSupabase) {
    try {
      const blob = {
        engine: BIOMETRIC_ENGINE_ID,
        dimension: dim,
        master_embedding: masterNorm,
        angle_embeddings: angleNorm,
        exemplar_embeddings: dbExemplars,
        ...(profile.geometricRatios ? { geometric_ratios: profile.geometricRatios } : {}),
        frames_captured: profile.framesCaptured,
        updated_at: profile.updatedAt,
      };

      const existingPat = await supabase
        .from('patients')
        .select('id, history')
        .eq('id', input.patient.id)
        .maybeSingle();
      if (existingPat.error) {
        throw new Error(`Supabase ตรวจสอบข้อมูลผู้ป่วยไม่สำเร็จ: ${existingPat.error.message}`);
      }

      const baseHistory =
        existingPat.data && existingPat.data.history && typeof existingPat.data.history === 'object'
          ? (existingPat.data.history as Record<string, unknown>)
          : {};
      const enrichedHistory = {
        ...baseHistory,
        biometricEngine: BIOMETRIC_ENGINE_ID,
        masterEmbedding: masterNorm,
        angleEmbeddings: angleNorm,
        exemplarEmbeddings: dbExemplars,
        ...(profile.geometricRatios ? { geometricRatios: profile.geometricRatios } : {}),
        biometricUpdatedAt: profile.updatedAt,
      };

      const patientBioPatch: Record<string, unknown> = {
        face_biometric_status: 'VERIFIED',
        face_frames_count: profile.framesCaptured,
        face_embedding: blob,
        biometric_data: blob,
        biometric_updated_at: profile.updatedAt,
        history: enrichedHistory,
        updated_at: profile.updatedAt,
        ...(input.patient.faceImageUrl ? { face_image_url: input.patient.faceImageUrl } : {}),
      };

      if (existingPat.data?.id) {
        const patRes = await executeSupabaseWriteWithSchemaCompat(
          (payload) => supabase.from('patients').update(payload).eq('id', input.patient.id).select('id'),
          patientBioPatch,
        );
        if (patRes.error) {
          throw new Error(`Supabase บันทึก patients ไม่สำเร็จ: ${patRes.error.message}`);
        }
      } else {
        const patInsertRes = await executeSupabaseWriteWithSchemaCompat(
          (payload) => supabase.from('patients').upsert(payload, { onConflict: 'id' }).select('id'),
          {
            id: input.patient.id,
            hn: input.patient.hn,
            full_name: input.patient.fullName,
            age:
              typeof input.patient.age === 'number' && Number.isFinite(input.patient.age)
                ? input.patient.age
                : 45,
            phone: input.patient.phone ?? '',
            diagnosis: input.patient.diagnosis || 'กำลังประเมินอาการ',
            registration_status: input.patient.registrationStatus || 'active',
            assigned_pt_id: safeAssignedPtId,
            next_appointment_date: input.patient.nextAppointmentDate || null,
            ...patientBioPatch,
          },
        );
        if (patInsertRes.error) {
          throw new Error(`Supabase สร้างระเบียน patients ไม่สำเร็จ: ${patInsertRes.error.message}`);
        }
      }

      const bioPayload: Record<string, unknown> = {
        patient_id: input.patient.id,
        hn: input.patient.hn,
        master_embedding: masterNorm,
        angle_embeddings: angleNorm,
        exemplar_embeddings: dbExemplars,
        ...(profile.geometricRatios ? { geometric_ratios: profile.geometricRatios } : {}),
        biometric_engine: BIOMETRIC_ENGINE_ID,
        model_version: BIOMETRIC_ENGINE_ID,
        embedding_dim: dim,
        frame_count: profile.framesCaptured,
        frames_captured: profile.framesCaptured,
        quality_score: profile.qualityScore,
        avatar_url: input.patient.faceImageUrl ?? null,
        updated_at: profile.updatedAt,
      };

      const upRes = await executeSupabaseWriteWithSchemaCompat(
        (payload) => supabase.from('patient_biometrics').upsert(payload, { onConflict: 'patient_id' }),
        bioPayload,
      );
      if (upRes.error) {
        throw new Error(`Supabase บันทึก patient_biometrics ไม่สำเร็จ: ${upRes.error.message}`);
      }

      // Strict Read-After-Write Verification
      const [verifyPat, verifyBio] = await Promise.all([
        supabase.from('patients').select('*').eq('id', input.patient.id).maybeSingle(),
        supabase.from('patient_biometrics').select('*').eq('patient_id', input.patient.id).maybeSingle(),
      ]);
      if (verifyPat.error && verifyBio.error) {
        throw new Error(
          `Read-After-Write Verification ล้มเหลว: ${verifyBio.error.message || verifyPat.error.message}`,
        );
      }

      const bioRow = (verifyBio.data ?? null) as Record<string, unknown> | null;
      const patRow = (verifyPat.data ?? null) as Record<string, unknown> | null;
      const verifiedBioVec = parseVectorLoose(bioRow?.master_embedding);
      const verifiedPatVec =
        parseVectorLoose(patRow?.biometric_data) ?? parseVectorLoose(patRow?.face_embedding);
      const reportedDim =
        typeof bioRow?.embedding_dim === 'number'
          ? bioRow.embedding_dim
          : verifiedBioVec?.length ?? verifiedPatVec?.length ?? 0;

      if (
        reportedDim < BIOMETRIC_MIN_EMBEDDING_DIM ||
        (!verifiedBioVec && !verifiedPatVec) ||
        ((verifiedBioVec?.length ?? 0) < BIOMETRIC_MIN_EMBEDDING_DIM &&
          (verifiedPatVec?.length ?? 0) < BIOMETRIC_MIN_EMBEDDING_DIM)
      ) {
        removeProfileFromLocalBiometricVault(input.patient.id);
        throw new Error(
          `ไม่สามารถยืนยันข้อมูลใบหน้าใน Supabase หลังการบันทึก (Read-After-Write embedding_dim=${reportedDim})`,
        );
      }
    } catch (err) {
      if (input.requireSupabase) {
        removeProfileFromLocalBiometricVault(input.patient.id);
        throw err;
      }
    }
  }

  saveProfileToLocalBiometricVault(profile);
  notifyBiometricIndexChanged();
  return profile;
}

function parseVectorLoose(raw: unknown): number[] | undefined {
  let candidate = raw;
  if (candidate && typeof candidate === 'object' && !Array.isArray(candidate) && !ArrayBuffer.isView(candidate)) {
    const obj = candidate as Record<string, unknown>;
    candidate = obj.master_embedding ?? obj.masterEmbedding ?? obj.embedding ?? obj.vector;
  }
  if (typeof candidate === 'string' && candidate.trim().startsWith('[')) {
    try {
      candidate = JSON.parse(candidate);
    } catch {
      return undefined;
    }
  }
  if (!Array.isArray(candidate) || candidate.length < BIOMETRIC_MIN_EMBEDDING_DIM) return undefined;
  const nums = candidate.map(Number);
  if (!nums.every(Number.isFinite)) return undefined;
  return normalizeVector(nums);
}

function buildCandidateFromSource(
  patientId: string,
  hn: string,
  fullName: string,
  masterVec: number[],
  angles: Partial<AngleEmbeddings5D> | undefined,
  exemplarsRaw: readonly unknown[] | undefined,
  meta: {
    diagnosis?: string;
    assignedPtId?: string;
    nextAppointmentDate?: string;
    avatarUrl?: string;
    framesCaptured?: number;
    qualityScore?: number;
    biometricEngine?: PatientRecord['biometricEngine'];
    geometricRatios?: GeometricAnthropometricRatios;
    sourceEntry?: PatientBiometricIndexEntry;
  },
): EnrolledBiometricCandidate {
  const master_embedding = normalizeVector(masterVec);
  const dim = master_embedding.length;
  const normAngle = (v: unknown): number[] =>
    isValidNeuralEmbedding(v) && v.length === dim ? normalizeVector(v) : master_embedding;
  const angle_embeddings: AngleEmbeddings5D = {
    center: normAngle(angles?.center),
    left: normAngle(angles?.left),
    right: normAngle(angles?.right),
    up: normAngle(angles?.up),
    down: normAngle(angles?.down),
  };
  const exemplar_embeddings = (exemplarsRaw ?? [])
    .map((ex) => (isValidNeuralEmbedding(ex) && ex.length === dim ? normalizeVector(ex) : null))
    .filter((v): v is number[] => v !== null);

  const sourceEntry: PatientBiometricIndexEntry = meta.sourceEntry ?? {
    patientId,
    hn,
    fullName,
    diagnosis: meta.diagnosis ?? '',
    assignedPtId: meta.assignedPtId ?? '',
    nextAppointmentDate: meta.nextAppointmentDate ?? '',
    avatarUrl: meta.avatarUrl,
    framesCaptured: meta.framesCaptured ?? 120,
    qualityScore: meta.qualityScore ?? 0.96,
    masterEmbedding: master_embedding,
    angleEmbeddings: angle_embeddings,
    ...(exemplar_embeddings.length > 0 ? { exemplarEmbeddings: exemplar_embeddings } : {}),
    biometricEngine:
      meta.biometricEngine ?? (dim >= 768 ? BIOMETRIC_ENGINE_ID : undefined),
    ...(meta.geometricRatios ? { geometricRatios: meta.geometricRatios } : {}),
    todayRoutine: null,
    rehabSchedules: [],
  };

  return {
    patientId,
    hn,
    fullName,
    diagnosis: meta.diagnosis ?? '',
    assignedPtId: meta.assignedPtId ?? '',
    nextAppointmentDate: meta.nextAppointmentDate ?? '',
    avatarUrl: meta.avatarUrl,
    framesCaptured: meta.framesCaptured ?? 120,
    qualityScore: meta.qualityScore ?? 0.96,
    biometricEngine: meta.biometricEngine ?? (dim >= 768 ? BIOMETRIC_ENGINE_ID : undefined),
    dimension: dim,
    master_embedding,
    angle_embeddings,
    exemplar_embeddings,
    ...(meta.geometricRatios ? { geometricRatios: meta.geometricRatios } : {}),
    sourceEntry,
  };
}

/**
 * Strictly READ-ONLY Kiosk Biometric Loader:
 * 1. Fetches live active patients & deleted tombstones (`GET /api/biometrics/index` + `GET /api/patients` when authenticated + Supabase).
 * 2. Immediately garbage-collects any orphaned/deleted vectors in `localStorage` (`drtech_biometric_vault_v3`, `drtech_patients`, `drtech_face_templates`).
 * 3. NEVER pushes local vault entries to Supabase, NEVER resurrects orphan `patient_biometrics` rows without an active row in `patients`.
 */
export async function loadEnrolledPatientsWithBiometrics(
  apiClient: DrTechCaresApiClient,
  expectedDim: number = BIOMETRIC_EXPECTED_DIM,
): Promise<LoadedBiometricVaultResult> {
  const byPatientId = new Map<string, EnrolledBiometricCandidate>();
  const deletedLowerSet = DrTechCaresApiClient.readDeletedTombstones
    ? DrTechCaresApiClient.readDeletedTombstones()
    : new Set<string>();

  let authoritativeActivePatients: ActivePatientRosterItem[] | null = null;
  let serverIndexEntries: PatientBiometricIndexEntry[] = [];

  const applyCandidate = (cand: EnrolledBiometricCandidate, preferIfExpectedDim = false) => {
    const pidLo = cand.patientId.trim().toLowerCase();
    const hnLo = cand.hn.trim().toLowerCase();
    if (deletedLowerSet.has(pidLo) || (hnLo && deletedLowerSet.has(hnLo))) return;

    const prev = byPatientId.get(cand.patientId);
    if (!prev) {
      byPatientId.set(cand.patientId, cand);
      return;
    }
    const prevMatches = prev.dimension === expectedDim;
    const nextMatches = cand.dimension === expectedDim;
    const pickBestRatios = (
      a?: GeometricAnthropometricRatios,
      b?: GeometricAnthropometricRatios,
    ): GeometricAnthropometricRatios | undefined => {
      if (a?.appearancePatch384?.length) return a;
      if (b?.appearancePatch384?.length) return b;
      return a ?? b;
    };
    const mergedRatios = pickBestRatios(
      cand.geometricRatios ?? cand.sourceEntry?.geometricRatios,
      prev.geometricRatios ?? prev.sourceEntry?.geometricRatios,
    );
    if ((!prevMatches && nextMatches) || (preferIfExpectedDim && nextMatches)) {
      byPatientId.set(cand.patientId, {
        ...cand,
        ...(mergedRatios ? { geometricRatios: mergedRatios } : {}),
        exemplar_embeddings:
          cand.exemplar_embeddings.length > 0
            ? cand.exemplar_embeddings
            : prev.dimension === cand.dimension
              ? prev.exemplar_embeddings
              : [],
        sourceEntry: {
          ...prev.sourceEntry,
          ...cand.sourceEntry,
          ...(mergedRatios ? { geometricRatios: mergedRatios } : {}),
          todayRoutine: prev.sourceEntry.todayRoutine ?? cand.sourceEntry.todayRoutine,
          todayExercises: prev.sourceEntry.todayExercises ?? cand.sourceEntry.todayExercises,
          rehabSchedules:
            (prev.sourceEntry.rehabSchedules ?? []).length > 0
              ? prev.sourceEntry.rehabSchedules
              : cand.sourceEntry.rehabSchedules,
          completedToday: prev.sourceEntry.completedToday ?? cand.sourceEntry.completedToday,
          hasAppointmentToday:
            prev.sourceEntry.hasAppointmentToday ?? cand.sourceEntry.hasAppointmentToday,
        },
      });
    } else if (prevMatches && nextMatches) {
      const nextExemplars =
        cand.exemplar_embeddings.length > prev.exemplar_embeddings.length
          ? cand.exemplar_embeddings
          : prev.exemplar_embeddings;
      byPatientId.set(cand.patientId, {
        ...prev,
        ...(mergedRatios ? { geometricRatios: mergedRatios } : {}),
        exemplar_embeddings: nextExemplars,
        sourceEntry: {
          ...prev.sourceEntry,
          ...(mergedRatios ? { geometricRatios: mergedRatios } : {}),
          exemplarEmbeddings: nextExemplars,
          todayRoutine: prev.sourceEntry.todayRoutine ?? cand.sourceEntry.todayRoutine,
          todayExercises: prev.sourceEntry.todayExercises ?? cand.sourceEntry.todayExercises,
          rehabSchedules:
            (prev.sourceEntry.rehabSchedules ?? []).length > 0
              ? prev.sourceEntry.rehabSchedules
              : cand.sourceEntry.rehabSchedules,
          completedToday: prev.sourceEntry.completedToday ?? cand.sourceEntry.completedToday,
          hasAppointmentToday:
            prev.sourceEntry.hasAppointmentToday ?? cand.sourceEntry.hasAppointmentToday,
        },
      });
    }
  };

  // 1. Primary Cloud API biometric index + authoritative active roster & tombstones
  try {
    if (typeof apiClient.getBiometricIndexWithRoster === 'function') {
      const rosterRes = await apiClient.getBiometricIndexWithRoster();
      serverIndexEntries = rosterRes.index ?? [];
      for (const delId of rosterRes.deletedIds ?? []) {
        const clean = String(delId ?? '').trim().toLowerCase();
        if (clean) deletedLowerSet.add(clean);
      }
      if (Array.isArray(rosterRes.activePatients)) {
        authoritativeActivePatients = rosterRes.activePatients;
      }
    } else {
      serverIndexEntries = await apiClient.getBiometricIndex();
    }

    if (typeof apiClient.getAuthToken === 'function' && apiClient.getAuthToken() && typeof apiClient.listPatients === 'function') {
      try {
        const liveList = await apiClient.listPatients();
        if (Array.isArray(liveList)) {
          authoritativeActivePatients = liveList.map((p) => ({
            id: p.id,
            hn: p.hn,
            fullName: p.fullName,
            registrationStatus: p.registrationStatus ?? 'active',
          }));
          // Also merge any biometric embeddings attached directly on /api/patients records
          for (const p of liveList as unknown as Record<string, unknown>[]) {
            const pid = String(p.id ?? '');
            const hn = String(p.hn ?? '');
            const pidLo = pid.trim().toLowerCase();
            const hnLo = hn.trim().toLowerCase();
            if (!pid || deletedLowerSet.has(pidLo) || (hnLo && deletedLowerSet.has(hnLo))) continue;
            const bioObj = (p.biometric && typeof p.biometric === 'object' ? p.biometric : null) as Record<string, unknown> | null;
            const templatesArr = Array.isArray(p.face_templates) ? (p.face_templates as unknown[]) : null;
            const masterVec =
              parseVectorLoose(p.masterEmbedding) ??
              parseVectorLoose(p.master_embedding) ??
              parseVectorLoose(p.faceEmbedding) ??
              parseVectorLoose(p.face_embedding) ??
              parseVectorLoose(bioObj?.master_embedding) ??
              parseVectorLoose(bioObj?.embeddings) ??
              parseVectorLoose(p.biometric_data) ??
              parseVectorLoose(templatesArr?.[0]);
            if (!masterVec) continue;
            const angles =
              (p.angleEmbeddings as Partial<AngleEmbeddings5D> | undefined) ??
              (p.angle_embeddings as Partial<AngleEmbeddings5D> | undefined) ??
              (bioObj?.angle_embeddings as Partial<AngleEmbeddings5D> | undefined);
            applyCandidate(
              buildCandidateFromSource(
                pid,
                hn,
                String(p.fullName ?? p.full_name ?? ''),
                masterVec,
                angles,
                Array.isArray(p.exemplarEmbeddings) ? (p.exemplarEmbeddings as unknown[]) : undefined,
                {
                  diagnosis: String(p.diagnosis ?? ''),
                  assignedPtId: String(p.assignedPtId ?? p.assigned_pt_id ?? ''),
                  nextAppointmentDate: String(p.nextAppointmentDate ?? p.next_appointment_date ?? ''),
                  avatarUrl: typeof p.faceImageUrl === 'string' ? p.faceImageUrl : undefined,
                  framesCaptured: typeof p.faceFramesCount === 'number' ? p.faceFramesCount : 120,
                  biometricEngine: p.biometricEngine as PatientRecord['biometricEngine'],
                  geometricRatios: (p.geometricRatios ?? p.geometric_ratios) as
                    | GeometricAnthropometricRatios
                    | undefined,
                },
              ),
            );
          }
        }
      } catch {
        // Keep unauthenticated roster from /api/biometrics/index if /api/patients isn't authorized
      }
    }

    for (const entry of serverIndexEntries) {
      if (!isValidNeuralEmbedding(entry.masterEmbedding)) continue;
      applyCandidate(
        buildCandidateFromSource(
          entry.patientId,
          entry.hn,
          entry.fullName,
          entry.masterEmbedding,
          entry.angleEmbeddings,
          entry.exemplarEmbeddings,
          {
            diagnosis: entry.diagnosis,
            assignedPtId: entry.assignedPtId,
            nextAppointmentDate: entry.nextAppointmentDate,
            avatarUrl: entry.avatarUrl,
            framesCaptured: entry.framesCaptured,
            qualityScore: entry.qualityScore,
            biometricEngine: entry.biometricEngine,
            geometricRatios: entry.geometricRatios,
            sourceEntry: entry,
          },
        ),
      );
    }
  } catch {
    // Continue cleanly to localStorage fallback if Cloud API failed
  }

  // 2. Direct Supabase fallback ONLY in non-browser test runners when Cloud index is empty
  // (Never open direct browser -> Postgres connections in Chrome / Kiosk runtime!)
  let syncedFromSupabase = false;
  let supabaseSyncedAccounts = 0;
  let syncError: string | undefined;
  const isBrowserRuntime = typeof document !== 'undefined';
  const supabase = apiClient.getSupabaseClient();
  if (supabase && !isBrowserRuntime && serverIndexEntries.length === 0) {
    try {
      let patientRows: Record<string, unknown>[] = [];
      const bioByPid = new Map<string, Record<string, unknown>>();

      const joinRes = await supabase.from('patients').select('*, patient_biometrics(*)');
      if (!joinRes.error && Array.isArray(joinRes.data)) {
        syncedFromSupabase = true;
        for (const r of joinRes.data as Record<string, unknown>[]) {
          patientRows.push(r);
          const embeddedBioRaw = r.patient_biometrics;
          const embeddedBio = Array.isArray(embeddedBioRaw)
            ? (embeddedBioRaw[0] as Record<string, unknown> | undefined)
            : embeddedBioRaw && typeof embeddedBioRaw === 'object'
              ? (embeddedBioRaw as Record<string, unknown>)
              : undefined;
          if (embeddedBio) {
            bioByPid.set(String(embeddedBio.patient_id ?? r.id), embeddedBio);
          }
        }
      } else {
        const [patRes, bioRes] = await Promise.all([
          supabase.from('patients').select('*'),
          supabase.from('patient_biometrics').select('*'),
        ]);
        if (!bioRes.error && Array.isArray(bioRes.data)) {
          for (const row of bioRes.data as Record<string, unknown>[]) {
            bioByPid.set(String(row.patient_id), row);
          }
        }
        if (!patRes.error && Array.isArray(patRes.data)) {
          syncedFromSupabase = true;
          patientRows = patRes.data as Record<string, unknown>[];
        } else if (patRes.error) {
          syncError = patRes.error.message;
        }
      }

      if (syncedFromSupabase) {
        const supabaseCandidates: EnrolledBiometricCandidate[] = [];
        const supabaseActiveRoster: ActivePatientRosterItem[] = [];
        const prevCloudByPid = new Map(byPatientId.entries());
        byPatientId.clear();

        for (const r of patientRows) {
          const pid = String(r.id);
          const pidLo = pid.trim().toLowerCase();
          const hn = String(r.hn ?? '');
          const hnLo = hn.trim().toLowerCase();
          const status = String(r.registration_status ?? 'active').toLowerCase();
          if (
            status === 'deleted' ||
            status === 'rejected' ||
            status === 'inactive' ||
            deletedLowerSet.has(pidLo) ||
            (hnLo && deletedLowerSet.has(hnLo))
          ) {
            deletedLowerSet.add(pidLo);
            if (hnLo) deletedLowerSet.add(hnLo);
            continue;
          }

          supabaseActiveRoster.push({
            id: pid,
            hn,
            fullName: String(r.full_name ?? ''),
            registrationStatus: status || 'active',
          });

          const bio = bioByPid.get(pid);
          const history = (r.history && typeof r.history === 'object' ? r.history : {}) as Record<string, unknown>;
          const bioDataObj = (r.biometric_data && typeof r.biometric_data === 'object' && !Array.isArray(r.biometric_data)
            ? r.biometric_data
            : {}) as Record<string, unknown>;
          const faceBlob = (r.face_embedding && typeof r.face_embedding === 'object' && !Array.isArray(r.face_embedding)
            ? r.face_embedding
            : {}) as Record<string, unknown>;

          const master =
            parseVectorLoose(bio?.master_embedding) ??
            parseVectorLoose(r.biometric_data) ??
            parseVectorLoose(r.face_embedding) ??
            parseVectorLoose(history.masterEmbedding);
          if (!master) continue;

          const rawAngles =
            (bio?.angle_embeddings as Partial<AngleEmbeddings5D> | undefined) ??
            (bioDataObj.angle_embeddings as Partial<AngleEmbeddings5D> | undefined) ??
            (faceBlob.angle_embeddings as Partial<AngleEmbeddings5D> | undefined) ??
            (history.angleEmbeddings as Partial<AngleEmbeddings5D> | undefined);
          const rawExemplars = Array.isArray(bio?.exemplar_embeddings)
            ? (bio.exemplar_embeddings as unknown[])
            : Array.isArray(bioDataObj.exemplar_embeddings)
              ? (bioDataObj.exemplar_embeddings as unknown[])
              : Array.isArray(faceBlob.exemplar_embeddings)
                ? (faceBlob.exemplar_embeddings as unknown[])
                : Array.isArray(history.exemplarEmbeddings)
                  ? (history.exemplarEmbeddings as unknown[])
                  : undefined;

          const prevCloud = prevCloudByPid.get(pid);
          const cand = buildCandidateFromSource(
            pid,
            String(r.hn ?? bio?.hn ?? ''),
            String(r.full_name ?? ''),
            master,
            rawAngles,
            rawExemplars,
            {
              diagnosis: String(r.diagnosis ?? ''),
              assignedPtId: String(r.assigned_pt_id ?? ''),
              nextAppointmentDate: String(r.next_appointment_date ?? ''),
              avatarUrl:
                typeof bio?.avatar_url === 'string' && bio.avatar_url
                  ? bio.avatar_url
                  : typeof r.face_image_url === 'string'
                    ? r.face_image_url
                    : undefined,
              framesCaptured:
                typeof bio?.frame_count === 'number'
                  ? bio.frame_count
                  : typeof bio?.frames_captured === 'number'
                    ? bio.frames_captured
                    : typeof r.face_frames_count === 'number'
                      ? r.face_frames_count
                      : 120,
              qualityScore: typeof bio?.quality_score === 'number' ? bio.quality_score : 0.96,
              biometricEngine:
                master.length >= 768
                  ? BIOMETRIC_ENGINE_ID
                  : (history.biometricEngine as PatientRecord['biometricEngine']),
              geometricRatios:
                history.geometricRatios && typeof history.geometricRatios === 'object'
                  ? (history.geometricRatios as GeometricAnthropometricRatios)
                  : undefined,
            },
          );
          if (prevCloud) {
            cand.sourceEntry = {
              ...cand.sourceEntry,
              todayRoutine: prevCloud.sourceEntry.todayRoutine ?? cand.sourceEntry.todayRoutine,
              todayExercises: prevCloud.sourceEntry.todayExercises ?? cand.sourceEntry.todayExercises,
              rehabSchedules:
                (prevCloud.sourceEntry.rehabSchedules ?? []).length > 0
                  ? prevCloud.sourceEntry.rehabSchedules
                  : cand.sourceEntry.rehabSchedules,
              completedToday: prevCloud.sourceEntry.completedToday ?? cand.sourceEntry.completedToday,
              hasAppointmentToday:
                prevCloud.sourceEntry.hasAppointmentToday ?? cand.sourceEntry.hasAppointmentToday,
            };
          }
          supabaseCandidates.push(cand);
          applyCandidate(cand, true);
        }

        authoritativeActivePatients = supabaseActiveRoster;
        supabaseSyncedAccounts = supabaseCandidates.length;
        replaceLocalBiometricVaultFromSupabase(supabaseCandidates);
        if (supabaseCandidates.length === 0) {
          DrTechCaresApiClient.clearRecentEnrolledEntries?.();
        }
      }
    } catch (err) {
      syncError = err instanceof Error ? err.message : String(err);
    }
  }

  // Garbage-collect localStorage caches strictly against the live active patients & tombstones
  if (typeof DrTechCaresApiClient.purgeOrphanedLocalStorageRecords === 'function') {
    DrTechCaresApiClient.purgeOrphanedLocalStorageRecords(
      authoritativeActivePatients,
      Array.from(deletedLowerSet),
    );
  }

  // Remove any explicitly tombstoned records from in-memory candidate map
  for (const [pid, cand] of Array.from(byPatientId.entries())) {
    const pidLo = pid.trim().toLowerCase();
    const hnLo = cand.hn.trim().toLowerCase();
    if (deletedLowerSet.has(pidLo) || (hnLo && deletedLowerSet.has(hnLo))) {
      byPatientId.delete(pid);
    }
  }

  // Always merge active, non-deleted local candidates from drtech_biometric_vault_v3 AND drtech_patients
  // so newly enrolled patients (e.g. during Supabase HTTP 544 outage) are immediately available at Kiosk.
  const vaultEntries = readLocalBiometricVault();
  for (const v of vaultEntries) {
    const pidLo = v.patientId.trim().toLowerCase();
    const hnLo = v.hn.trim().toLowerCase();
    if (deletedLowerSet.has(pidLo) || (hnLo && deletedLowerSet.has(hnLo))) continue;
    if (!isValidNeuralEmbedding(v.master_embedding)) continue;
    applyCandidate(
      buildCandidateFromSource(
        v.patientId,
        v.hn,
        v.fullName,
        v.master_embedding,
        v.angle_embeddings,
        v.exemplar_embeddings,
        {
          diagnosis: v.diagnosis,
          assignedPtId: v.assignedPtId,
          nextAppointmentDate: v.nextAppointmentDate,
          avatarUrl: v.avatarUrl,
          framesCaptured: v.framesCaptured,
          qualityScore: v.qualityScore,
          biometricEngine: BIOMETRIC_ENGINE_ID,
          geometricRatios: v.geometricRatios,
        },
      ),
      true,
    );
  }

  if (typeof globalThis.localStorage !== 'undefined') {
    try {
      const rawPatients = globalThis.localStorage.getItem('drtech_patients');
      const parsedPatients = rawPatients ? (JSON.parse(rawPatients) as unknown) : [];
      if (Array.isArray(parsedPatients)) {
        for (const rec of parsedPatients) {
          if (!rec || typeof rec !== 'object') continue;
          const p = rec as Record<string, unknown>;
          const pid = String(p.id ?? p.patientId ?? '').trim();
          const hn = String(p.hn ?? '').trim();
          const pidLo = pid.toLowerCase();
          const hnLo = hn.toLowerCase();
          const status = String(p.registrationStatus ?? p.registration_status ?? 'active').toLowerCase();
          if (
            !pid ||
            status === 'deleted' ||
            status === 'rejected' ||
            status === 'inactive' ||
            deletedLowerSet.has(pidLo) ||
            (hnLo && deletedLowerSet.has(hnLo))
          ) {
            continue;
          }
          const bioObj = (
            p.biometric && typeof p.biometric === 'object' ? p.biometric : null
          ) as Record<string, unknown> | null;
          const templatesArr = Array.isArray(p.face_templates)
            ? (p.face_templates as unknown[])
            : null;
          const master =
            parseVectorLoose(p.masterEmbedding) ??
            parseVectorLoose(p.master_embedding) ??
            parseVectorLoose(p.faceEmbedding) ??
            parseVectorLoose(p.face_embedding) ??
            parseVectorLoose(bioObj?.master_embedding) ??
            parseVectorLoose(bioObj?.embeddings) ??
            parseVectorLoose(p.biometric_data) ??
            parseVectorLoose(templatesArr?.[0]);
          if (!master) continue;
          const rawAngles =
            (p.angleEmbeddings as Partial<AngleEmbeddings5D> | undefined) ??
            (p.angle_embeddings as Partial<AngleEmbeddings5D> | undefined) ??
            (bioObj?.angle_embeddings as Partial<AngleEmbeddings5D> | undefined);
          applyCandidate(
            buildCandidateFromSource(
              pid,
              hn,
              String(p.fullName ?? p.full_name ?? ''),
              master,
              rawAngles,
              Array.isArray(p.exemplarEmbeddings) ? (p.exemplarEmbeddings as unknown[]) : undefined,
              {
                diagnosis: String(p.diagnosis ?? ''),
                assignedPtId: String(p.assignedPtId ?? p.assigned_pt_id ?? ''),
                nextAppointmentDate: String(p.nextAppointmentDate ?? p.next_appointment_date ?? ''),
                avatarUrl: typeof p.faceImageUrl === 'string' ? p.faceImageUrl : undefined,
                framesCaptured: typeof p.faceFramesCount === 'number' ? p.faceFramesCount : 120,
                biometricEngine: p.biometricEngine as PatientRecord['biometricEngine'],
                geometricRatios: (p.geometricRatios as PatientRecord['geometricRatios']) ?? undefined,
              },
            ),
            true,
          );
        }
      }
    } catch {
      // Ignore malformed localStorage entries
    }
  }

  const allCandidates = Array.from(byPatientId.values());
  const compatibleCandidates = allCandidates.filter((c) => c.dimension === expectedDim);
  const legacyAccountsCount = allCandidates.length - compatibleCandidates.length;

  return {
    candidates: allCandidates,
    compatibleCandidates,
    totalAccounts: allCandidates.length,
    validAccounts: compatibleCandidates.length,
    legacyAccountsCount,
    syncedFromSupabase,
    supabaseSyncedAccounts: syncedFromSupabase
      ? supabaseSyncedAccounts
      : compatibleCandidates.length,
    ...(syncError ? { syncError } : {}),
  };
}

/**
 * Explicit alias for Kiosk live Supabase synchronization (`fetchLiveBiometricsFromSupabase`).
 */
export async function fetchLiveBiometricsFromSupabase(
  apiClient: DrTechCaresApiClient,
  expectedDim: number = BIOMETRIC_EXPECTED_DIM,
): Promise<LoadedBiometricVaultResult> {
  return loadEnrolledPatientsWithBiometrics(apiClient, expectedDim);
}

/**
 * Evaluates a live L2-normalized embedding strictly under Triple-Gate Zero False-Accept Verification:
 * - Gate 1: Both master_embedding & center satisfy rawCos >= 0.84, centeredCos >= 0.80, L2 <= 0.56,
 *   plus >= 2 angle centroids >= 0.82.
 * - Gate 2: Patch ZNCC >= 0.76 & craniofacial deviation <= 0.065 (+ std >= 0.02 when missing patch).
 */
export function computePatientMatchScore(
  liveEmbedding: readonly number[],
  candidate: EnrolledBiometricCandidate,
  _humanOverride?: Pick<Human, 'match'> | null,
  liveBucket?: keyof AngleEmbeddings5D,
  probeRatios?: GeometricAnthropometricRatios | null,
): PatientMatchScoreResult {
  const liveNorm = normalizeVector(liveEmbedding);
  const expectedDim = liveNorm.length;

  const uniqueStored: { vec: number[]; source: string }[] = [];
  const pushUnique = (raw: unknown, source: string) => {
    if (!Array.isArray(raw) || raw.length !== expectedDim) return;
    const normV = normalizeVector(raw);
    if (!uniqueStored.some((item) => cosineSimilarityNormalized(item.vec, normV) >= 0.9999)) {
      uniqueStored.push({ vec: normV, source });
    }
  };

  pushUnique(candidate.master_embedding, 'master');
  const angleKeys: (keyof AngleEmbeddings5D)[] = liveBucket
    ? [liveBucket, ...(['center', 'left', 'right', 'up', 'down'] as const).filter((k) => k !== liveBucket)]
    : ['center', 'left', 'right', 'up', 'down'];
  for (const k of angleKeys) {
    pushUnique(candidate.angle_embeddings?.[k], k);
  }
  candidate.exemplar_embeddings.forEach((ex, idx) => {
    pushUnique(ex, `exemplar-${idx + 1}`);
  });

  if (uniqueStored.length === 0) {
    return {
      candidate,
      bestSim: 0,
      centeredSimilarity: 0,
      centroidSim: 0,
      exemplarSim: 0,
      humanSim: 0,
      finalScore: 0,
      minEuclideanDist: Number.POSITIVE_INFINITY,
      consensusTop3Avg: 0,
      vectorsAbove073: 0,
      uniqueVectorCount: 0,
      patchZncc: null,
      craniofacialDeviation: null,
      passesMultiExemplarGate: false,
      tripleGateFailureReasons: ['NO_STORED_VECTORS'],
      bestSource: 'none',
    };
  }

  const tg = evaluateTripleGateNeuralCandidate(
    liveNorm,
    {
      masterEmbedding: candidate.master_embedding,
      angleEmbeddings: candidate.angle_embeddings,
      exemplarEmbeddings: candidate.exemplar_embeddings,
      geometricRatios: candidate.geometricRatios ?? candidate.sourceEntry?.geometricRatios ?? null,
    },
    probeRatios,
  );

  const scoredVecs = uniqueStored
    .map((item) => ({
      cos: cosineSimilarityNormalized(liveNorm, item.vec),
      dist: euclideanDistanceNormalized(liveNorm, item.vec),
      source: item.source,
    }))
    .sort((a, b) => b.cos - a.cos);

  const k = Math.min(3, scoredVecs.length);
  let topSum = 0;
  for (let i = 0; i < k; i += 1) {
    topSum += Math.max(0, scoredVecs[i].cos);
  }
  const consensusTop3Avg = topSum / k;
  const vectorsAbove073 = scoredVecs.filter((s) => s.cos >= BIOMETRIC_EXEMPLAR_FLOOR_THRESHOLD).length;

  return {
    candidate,
    bestSim: tg.rawCosine,
    centeredSimilarity: tg.centeredCosine,
    centroidSim: tg.bestCentroidCosine,
    exemplarSim: consensusTop3Avg,
    humanSim: tg.rawCosine,
    finalScore: tg.rawCosine,
    minEuclideanDist: tg.l2Dist,
    consensusTop3Avg,
    vectorsAbove073,
    uniqueVectorCount: scoredVecs.length,
    patchZncc: tg.patchZncc,
    craniofacialDeviation: tg.craniofacialDeviation,
    passesMultiExemplarGate: tg.passed,
    tripleGateFailureReasons: tg.failureReasons,
    bestSource: tg.bestAngle,
  };
}

function normalizeNameForIdentity(name: string): string {
  return name.trim().toLowerCase().replace(/\s+/g, ' ');
}

/**
 * Zero False-Accept Open-Set 1:N Neural Face Verification (Unified across 1-patient & N-patient DBs):
 * - Gate 1: Both master_embedding & center satisfy rawCos >= 0.84, centeredCos >= 0.80, L2 <= 0.56,
 *   plus >= 2 angle centroids >= 0.82.
 * - Gate 2: Patch ZNCC >= 0.76 & craniofacial deviation <= 0.065 (or std >= 0.02 when missing patch).
 * - Gate 3: Strict margin >= 0.08 over Rank #2 AND strict margin > 0.08 over 12 Imposter Cohort templates.
 */
export function evaluateLiveMatch1toN(
  liveEmbedding: readonly number[],
  candidates: readonly EnrolledBiometricCandidate[],
  excludedIds: readonly string[] = [],
  humanOverride?: Pick<Human, 'match'> | null,
  liveBucket?: keyof AngleEmbeddings5D,
  probeRatios?: GeometricAnthropometricRatios | null,
): Live1toNMatchEvaluation {
  const excludedSet = new Set(excludedIds);
  const dim = liveEmbedding.length;
  const dimensionCompatible = candidates.filter((c) => c.dimension === dim);
  const eligible = dimensionCompatible.filter(
    (c) => !excludedSet.has(c.patientId) && !excludedSet.has(c.hn),
  );

  if (dimensionCompatible.length === 0) {
    return {
      matched: false,
      topCandidate: null,
      runnerUpCandidate: null,
      topSimilarity: 0,
      topCenteredSimilarity: 0,
      topCentroidSimilarity: 0,
      topExemplarSimilarity: 0,
      topHumanSimilarity: 0,
      topFinalScore: 0,
      topEuclideanDistance: Number.POSITIVE_INFINITY,
      topConsensusTop3Avg: 0,
      topPatchZncc: null,
      topCraniofacialDeviation: null,
      cohortMargin: 0,
      runnerUpSimilarity: 0,
      runnerUpFinalScore: 0,
      margin: 0,
      confidencePercent: 0,
      bestSource: 'none',
      reason: 'NO_CANDIDATES',
      validAccounts: 0,
      totalAccounts: candidates.length,
    };
  }

  if (eligible.length === 0) {
    return {
      matched: false,
      topCandidate: null,
      runnerUpCandidate: null,
      topSimilarity: 0,
      topCenteredSimilarity: 0,
      topCentroidSimilarity: 0,
      topExemplarSimilarity: 0,
      topHumanSimilarity: 0,
      topFinalScore: 0,
      topEuclideanDistance: Number.POSITIVE_INFINITY,
      topConsensusTop3Avg: 0,
      topPatchZncc: null,
      topCraniofacialDeviation: null,
      cohortMargin: 0,
      runnerUpSimilarity: 0,
      runnerUpFinalScore: 0,
      margin: 0,
      confidencePercent: 0,
      bestSource: 'none',
      reason: 'EXCLUDED_ONLY',
      validAccounts: dimensionCompatible.length,
      totalAccounts: candidates.length,
    };
  }

  const scored = eligible
    .map((c) => computePatientMatchScore(liveEmbedding, c, humanOverride, liveBucket, probeRatios))
    .sort(
      (a, b) =>
        Number(b.passesMultiExemplarGate) - Number(a.passesMultiExemplarGate) ||
        b.bestSim - a.bestSim ||
        b.centeredSimilarity - a.centeredSimilarity ||
        b.consensusTop3Avg - a.consensusTop3Avg,
    );

  const top = scored[0];
  const topName = normalizeNameForIdentity(top.candidate.fullName);
  const distinctRunners = scored
    .slice(1)
    .filter((s) => !(topName && normalizeNameForIdentity(s.candidate.fullName) === topName));
  const runnerUp = distinctRunners[0] ?? null;

  const runnerUpSim = runnerUp ? runnerUp.bestSim : 0;
  const runnerUpFinalScore = runnerUp ? runnerUp.finalScore : 0;
  const margin = runnerUp ? top.bestSim - runnerUp.bestSim : 1;

  const maxCohortScore = computeMaxImposterCohortScore(liveEmbedding);
  const cohortMargin = top.bestSim - maxCohortScore;
  const passesCohort = top.bestSim > maxCohortScore + STRICT_COHORT_MARGIN;

  // Same strict security thresholds regardless of whether 1 patient or 1,000 patients are enrolled
  const passesThreshold =
    top.bestSim >= BIOMETRIC_MATCH_THRESHOLD &&
    top.centeredSimilarity >= BIOMETRIC_CENTERED_COSINE_THRESHOLD &&
    top.minEuclideanDist <= BIOMETRIC_MAX_EUCLIDEAN_DISTANCE &&
    top.passesMultiExemplarGate;
  const passesMargin = distinctRunners.length === 0 || margin >= BIOMETRIC_MARGIN_THRESHOLD;
  const matched = passesThreshold && passesCohort && passesMargin;

  const reason: Live1toNMatchEvaluation['reason'] = matched
    ? 'OK'
    : passesThreshold && !passesCohort
      ? 'IMPOSTER_COHORT_REJECTED'
      : passesThreshold && !passesMargin
        ? 'AMBIGUOUS_CLOSE_CANDIDATES'
        : top.tripleGateFailureReasons.some((r) => r.includes('LOW_VARIANCE'))
          ? 'LOW_VARIANCE_REENROLL_REQUIRED'
          : 'BELOW_THRESHOLD';

  const confidencePercent = Number((Math.max(0, Math.min(1, top.bestSim)) * 100).toFixed(1));

  return {
    matched,
    topCandidate: matched ? top.candidate : null,
    runnerUpCandidate: matched ? (runnerUp?.candidate ?? null) : null,
    topSimilarity: top.bestSim,
    topCenteredSimilarity: top.centeredSimilarity,
    topCentroidSimilarity: top.centroidSim,
    topExemplarSimilarity: top.exemplarSim,
    topHumanSimilarity: top.humanSim,
    topFinalScore: top.finalScore,
    topEuclideanDistance: top.minEuclideanDist,
    topConsensusTop3Avg: top.consensusTop3Avg,
    topPatchZncc: top.patchZncc,
    topCraniofacialDeviation: top.craniofacialDeviation,
    cohortMargin,
    runnerUpSimilarity: runnerUpSim,
    runnerUpFinalScore,
    margin,
    confidencePercent,
    bestSource: top.bestSource,
    reason,
    validAccounts: dimensionCompatible.length,
    totalAccounts: candidates.length,
  };
}

/**
 * Gate 4: Enforces 6/6 consecutive matching frames for the exact same `patientId`
 * with low score variance (`< 0.025`). Any frame below `0.74`, failed gate, spoof,
 * or identity switch immediately resets streak to `0/6`.
 */
export class LiveIdentityStreakTracker {
  readonly requiredStreak: number;
  private activePatientId: string | null = null;
  private streakScores: number[] = [];

  constructor(requiredStreak: number = BIOMETRIC_REQUIRED_STREAK) {
    this.requiredStreak = requiredStreak;
  }

  get currentStreak(): number {
    return this.streakScores.length;
  }

  reset(): void {
    this.activePatientId = null;
    this.streakScores = [];
  }

  step(
    evalResult: Live1toNMatchEvaluation,
    options: { spoofDetected?: boolean; biologicalVerified?: boolean } = {},
  ): LiveStreakEvaluation {
    if (options.spoofDetected) {
      this.reset();
      return {
        confirmed: false,
        streak: 0,
        requiredStreak: this.requiredStreak,
        candidate: null,
        runnerUp: evalResult.runnerUpCandidate,
        similarity: evalResult.topSimilarity,
        finalScore: evalResult.topFinalScore,
        runnerUpSimilarity: evalResult.runnerUpSimilarity,
        margin: evalResult.margin,
        confidencePercent: evalResult.confidencePercent,
        scoreVariance: 0,
        bestSource: evalResult.bestSource,
        reason: 'SPOOF_BLOCKED',
      };
    }

    if (
      !evalResult.matched ||
      !evalResult.topCandidate ||
      evalResult.topSimilarity < BIOMETRIC_STREAK_MIN_FRAME_COSINE
    ) {
      this.reset();
      return {
        confirmed: false,
        streak: 0,
        requiredStreak: this.requiredStreak,
        candidate: null,
        runnerUp: evalResult.runnerUpCandidate,
        similarity: evalResult.topSimilarity,
        finalScore: evalResult.topFinalScore,
        runnerUpSimilarity: evalResult.runnerUpSimilarity,
        margin: evalResult.margin,
        confidencePercent: evalResult.confidencePercent,
        scoreVariance: 0,
        bestSource: evalResult.bestSource,
        reason: evalResult.reason === 'OK' ? 'BELOW_THRESHOLD' : evalResult.reason,
      };
    }

    const pid = evalResult.topCandidate.patientId;
    if (this.activePatientId !== pid) {
      this.activePatientId = pid;
      this.streakScores = [evalResult.topSimilarity];
    } else {
      this.streakScores.push(evalResult.topSimilarity);
      if (this.streakScores.length > this.requiredStreak) {
        this.streakScores.shift();
      }
    }

    const scoreSpread =
      this.streakScores.length >= 2
        ? Math.max(...this.streakScores) - Math.min(...this.streakScores)
        : 0;

    if (this.streakScores.length >= 2 && scoreSpread >= BIOMETRIC_MAX_SCORE_VARIANCE) {
      this.reset();
      return {
        confirmed: false,
        streak: 0,
        requiredStreak: this.requiredStreak,
        candidate: null,
        runnerUp: evalResult.runnerUpCandidate,
        similarity: evalResult.topSimilarity,
        finalScore: evalResult.topFinalScore,
        runnerUpSimilarity: evalResult.runnerUpSimilarity,
        margin: evalResult.margin,
        confidencePercent: evalResult.confidencePercent,
        scoreVariance: scoreSpread,
        bestSource: evalResult.bestSource,
        reason: 'HIGH_SCORE_VARIANCE',
      };
    }

    const streak = this.streakScores.length;
    const livenessOk = options.biologicalVerified !== false;
    const confirmed = streak >= this.requiredStreak && livenessOk;

    return {
      confirmed,
      streak,
      requiredStreak: this.requiredStreak,
      candidate: confirmed ? evalResult.topCandidate : null,
      runnerUp: evalResult.runnerUpCandidate,
      similarity: evalResult.topSimilarity,
      finalScore: evalResult.topFinalScore,
      runnerUpSimilarity: evalResult.runnerUpSimilarity,
      margin: evalResult.margin,
      confidencePercent: evalResult.confidencePercent,
      scoreVariance: scoreSpread,
      bestSource: evalResult.bestSource,
      reason: livenessOk ? 'OK' : 'LIVENESS_PENDING',
    };
  }
}
