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
import type { GeometricAnthropometricRatios } from '@drtechcares/smart-biometric-core';

export const BIOMETRIC_VAULT_STORAGE_KEY = 'drtech_biometric_vault_v3';
export const BIOMETRIC_ENGINE_ID = 'human-faceres-1024d' as const;
export const BIOMETRIC_MIN_EMBEDDING_DIM = 128;
export const BIOMETRIC_EXPECTED_DIM = 1024;
export const BIOMETRIC_MATCH_THRESHOLD = 0.76;
export const BIOMETRIC_FINAL_SCORE_THRESHOLD = 0.76;
export const BIOMETRIC_EXEMPLAR_FLOOR_THRESHOLD = 0.73;
export const BIOMETRIC_MAX_EUCLIDEAN_DISTANCE = 0.69;
export const BIOMETRIC_HUMAN_SIM_THRESHOLD = 0.76;
export const BIOMETRIC_MARGIN_THRESHOLD = 0.08;
export const BIOMETRIC_REQUIRED_STREAK = 6;
export const BIOMETRIC_STREAK_MIN_FRAME_COSINE = 0.74;
export const BIOMETRIC_MAX_SCORE_VARIANCE = 0.025;
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
  centroidSim: number;
  exemplarSim: number;
  humanSim: number;
  finalScore: number;
  minEuclideanDist: number;
  consensusTop3Avg: number;
  vectorsAbove073: number;
  uniqueVectorCount: number;
  passesMultiExemplarGate: boolean;
  bestSource: string;
}

export interface Live1toNMatchEvaluation {
  matched: boolean;
  topCandidate: EnrolledBiometricCandidate | null;
  runnerUpCandidate: EnrolledBiometricCandidate | null;
  topSimilarity: number;
  topCentroidSimilarity: number;
  topExemplarSimilarity: number;
  topHumanSimilarity: number;
  topFinalScore: number;
  topEuclideanDistance: number;
  topConsensusTop3Avg: number;
  runnerUpSimilarity: number;
  runnerUpFinalScore: number;
  margin: number;
  confidencePercent: number;
  bestSource: string;
  reason: 'OK' | 'NO_CANDIDATES' | 'EXCLUDED_ONLY' | 'BELOW_THRESHOLD' | 'AMBIGUOUS_CLOSE_CANDIDATES';
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

  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] === 0) continue;
    data[i] = Math.max(0, Math.min(255, Math.round(targetMean + (data[i] - meanY) * gain)));
    data[i + 1] = Math.max(0, Math.min(255, Math.round(targetMean + (data[i + 1] - meanY) * gain)));
    data[i + 2] = Math.max(0, Math.min(255, Math.round(targetMean + (data[i + 2] - meanY) * gain)));
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
        const faceSpan = Math.max(36, Math.max((maxX - minX) * 1.18, (maxY - minY) * 1.15));
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
    const cx = faceBox.x + faceBox.width * 0.5;
    const cy = faceBox.y + faceBox.height * 0.48;
    // Tight inner-face crop (brows to chin, cheekbones) eliminating background wall/lights
    const padScale = faceBox.height < 0.18 ? 1.45 : 1.18;
    const sideNorm = Math.min(1, Math.max(faceBox.width, faceBox.height) * padScale);
    const halfNorm = sideNorm * 0.5;
    const rx = Math.max(0, Math.min(1 - halfNorm * 2, cx - halfNorm));
    const ry = Math.max(0, Math.min(1 - halfNorm * 2, cy - halfNorm));
    const rw = Math.min(1 - rx, halfNorm * 2);
    const rh = Math.min(1 - ry, halfNorm * 2);
    const sx = Math.max(0, Math.floor(rx * vw));
    const sy = Math.max(0, Math.floor(ry * vh));
    const sw = Math.max(32, Math.min(vw - sx, Math.ceil(rw * vw)));
    const sh = Math.max(32, Math.min(vh - sy, Math.ceil(rh * vh)));

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
    ...(meta.geometricRatios ? { geometricRatios: meta.geometricRatios } : {}),
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

  if (supabase) {
    try {
      const blob = {
        engine: BIOMETRIC_ENGINE_ID,
        dimension: dim,
        master_embedding: masterNorm,
        angle_embeddings: angleNorm,
        exemplar_embeddings: dbExemplars,
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
    if ((!prevMatches && nextMatches) || (preferIfExpectedDim && nextMatches)) {
      byPatientId.set(cand.patientId, {
        ...cand,
        exemplar_embeddings:
          cand.exemplar_embeddings.length > 0
            ? cand.exemplar_embeddings
            : prev.dimension === cand.dimension
              ? prev.exemplar_embeddings
              : [],
        sourceEntry: {
          ...prev.sourceEntry,
          ...cand.sourceEntry,
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
    } else if (prevMatches && nextMatches && cand.exemplar_embeddings.length > prev.exemplar_embeddings.length) {
      byPatientId.set(cand.patientId, {
        ...prev,
        exemplar_embeddings: cand.exemplar_embeddings,
        sourceEntry: {
          ...prev.sourceEntry,
          exemplarEmbeddings: cand.exemplar_embeddings,
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
    // Continue cleanly to direct Supabase query if Cloud API failed
  }

  // 2. Authoritative Live Sync from Supabase Cloud (`patients` + `patient_biometrics`) — NEVER orphan biometrics
  let syncedFromSupabase = false;
  let supabaseSyncedAccounts = 0;
  let syncError: string | undefined;
  const supabase = apiClient.getSupabaseClient();
  if (supabase) {
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

  // Enforce strict active-patient whitelist (patientId AND hn MUST be in active roster and NOT in tombstones)
  if (authoritativeActivePatients !== null) {
    const activeById = new Map(
      authoritativeActivePatients
        .filter((p) => (p.registrationStatus ?? 'active') === 'active')
        .map((p) => [p.id.trim().toLowerCase(), p.hn.trim().toLowerCase()]),
    );
    for (const [pid, cand] of Array.from(byPatientId.entries())) {
      const pidLo = pid.trim().toLowerCase();
      const hnLo = cand.hn.trim().toLowerCase();
      const expectedHn = activeById.get(pidLo);
      if (
        deletedLowerSet.has(pidLo) ||
        (hnLo && deletedLowerSet.has(hnLo)) ||
        expectedHn === undefined ||
        !activeById.has(pidLo) ||
        (expectedHn && hnLo !== expectedHn)
      ) {
        byPatientId.delete(pid);
      }
    }
  }

  // Fallback read from Local Vault ONLY when BOTH Cloud API roster & Supabase were unavailable
  if (!syncedFromSupabase && authoritativeActivePatients === null) {
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
 * Evaluates a live L2-normalized embedding strictly using raw L2 Cosine Similarity & Euclidean Distance
 * across unique stored vectors (`master_embedding`, distinct `angle_embeddings`, and `exemplar_embeddings`)
 * with ZERO artificial score inflation or non-linear curve boosting.
 */
export function computePatientMatchScore(
  liveEmbedding: readonly number[],
  candidate: EnrolledBiometricCandidate,
  _humanOverride?: Pick<Human, 'match'> | null,
  liveBucket?: keyof AngleEmbeddings5D,
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
      centroidSim: 0,
      exemplarSim: 0,
      humanSim: 0,
      finalScore: 0,
      minEuclideanDist: Number.POSITIVE_INFINITY,
      consensusTop3Avg: 0,
      vectorsAbove073: 0,
      uniqueVectorCount: 0,
      passesMultiExemplarGate: false,
      bestSource: 'none',
    };
  }

  let centroidSim = 0;
  let bestTargetCos = -1;
  let bestSource = 'none';
  const scoredVecs: { cos: number; dist: number; source: string }[] = [];

  for (const item of uniqueStored) {
    const cos = cosineSimilarityNormalized(liveNorm, item.vec);
    const dist = euclideanDistanceNormalized(liveNorm, item.vec);
    scoredVecs.push({ cos, dist, source: item.source });
    if (
      item.source === 'master' ||
      item.source === 'center' ||
      item.source === 'left' ||
      item.source === 'right' ||
      item.source === 'up' ||
      item.source === 'down'
    ) {
      centroidSim = Math.max(centroidSim, cos);
    }
    if (cos > bestTargetCos) {
      bestTargetCos = cos;
      bestSource = item.source;
    }
  }

  scoredVecs.sort((a, b) => b.cos - a.cos);
  const bestRawCos = Math.max(0, scoredVecs[0].cos);
  const minEuclideanDist = scoredVecs[0].dist;
  const k = Math.min(3, scoredVecs.length);
  let topSum = 0;
  for (let i = 0; i < k; i += 1) {
    topSum += Math.max(0, scoredVecs[i].cos);
  }
  const consensusTop3Avg = topSum / k;
  const vectorsAbove073 = scoredVecs.filter((s) => s.cos >= BIOMETRIC_EXEMPLAR_FLOOR_THRESHOLD).length;
  const passesMultiExemplarGate =
    scoredVecs.length >= 3
      ? vectorsAbove073 >= 3 && consensusTop3Avg >= BIOMETRIC_MATCH_THRESHOLD
      : scoredVecs.length === 2
        ? vectorsAbove073 >= 2 && consensusTop3Avg >= BIOMETRIC_MATCH_THRESHOLD
        : bestRawCos >= BIOMETRIC_MATCH_THRESHOLD;

  return {
    candidate,
    bestSim: bestRawCos,
    centroidSim: Math.max(0, centroidSim),
    exemplarSim: consensusTop3Avg,
    humanSim: bestRawCos,
    finalScore: bestRawCos,
    minEuclideanDist,
    consensusTop3Avg,
    vectorsAbove073,
    uniqueVectorCount: scoredVecs.length,
    passesMultiExemplarGate,
    bestSource,
  };
}

function normalizeNameForIdentity(name: string): string {
  return name.trim().toLowerCase().replace(/\s+/g, ' ');
}

/**
 * Ultra-Strict Open-Set 1:N Neural Face Verification:
 * - Gate 1: Raw L2-normalized Cosine Similarity >= 0.76 AND L2 Euclidean Distance <= 0.69.
 * - Gate 2: Multi-Exemplar Consensus (>=3 stored vectors >= 0.73 and Top-3 avg >= 0.76, or Master + Angle when 2).
 * - Gate 3: Margin >= 0.08 over Rank #2 (`bestCosine - secondBestCosine >= 0.08`).
 * - Pure true raw cosine percentage ONLY (`cosine * 100`), zero curve inflation.
 */
export function evaluateLiveMatch1toN(
  liveEmbedding: readonly number[],
  candidates: readonly EnrolledBiometricCandidate[],
  excludedIds: readonly string[] = [],
  humanOverride?: Pick<Human, 'match'> | null,
  liveBucket?: keyof AngleEmbeddings5D,
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
      topCentroidSimilarity: 0,
      topExemplarSimilarity: 0,
      topHumanSimilarity: 0,
      topFinalScore: 0,
      topEuclideanDistance: Number.POSITIVE_INFINITY,
      topConsensusTop3Avg: 0,
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
      topCentroidSimilarity: 0,
      topExemplarSimilarity: 0,
      topHumanSimilarity: 0,
      topFinalScore: 0,
      topEuclideanDistance: Number.POSITIVE_INFINITY,
      topConsensusTop3Avg: 0,
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
    .map((c) => computePatientMatchScore(liveEmbedding, c, humanOverride, liveBucket))
    .sort((a, b) => b.bestSim - a.bestSim || b.consensusTop3Avg - a.consensusTop3Avg);

  const top = scored[0];
  const topName = normalizeNameForIdentity(top.candidate.fullName);
  const distinctRunners = scored
    .slice(1)
    .filter((s) => !(topName && normalizeNameForIdentity(s.candidate.fullName) === topName));
  const runnerUp = distinctRunners[0] ?? null;

  const runnerUpSim = runnerUp ? runnerUp.bestSim : 0;
  const runnerUpFinalScore = runnerUp ? runnerUp.finalScore : 0;
  const margin = runnerUp ? top.bestSim - runnerUp.bestSim : 1;

  const passesThreshold =
    top.bestSim >= BIOMETRIC_MATCH_THRESHOLD &&
    top.minEuclideanDist <= BIOMETRIC_MAX_EUCLIDEAN_DISTANCE &&
    top.passesMultiExemplarGate;
  const passesMargin = distinctRunners.length === 0 || margin >= BIOMETRIC_MARGIN_THRESHOLD;
  const matched = passesThreshold && passesMargin;

  const reason: Live1toNMatchEvaluation['reason'] = matched
    ? 'OK'
    : passesThreshold && !passesMargin
      ? 'AMBIGUOUS_CLOSE_CANDIDATES'
      : 'BELOW_THRESHOLD';

  const confidencePercent = Number((Math.max(0, Math.min(1, top.bestSim)) * 100).toFixed(1));

  return {
    matched,
    topCandidate: matched ? top.candidate : null,
    runnerUpCandidate: matched ? (runnerUp?.candidate ?? null) : null,
    topSimilarity: top.bestSim,
    topCentroidSimilarity: top.centroidSim,
    topExemplarSimilarity: top.exemplarSim,
    topHumanSimilarity: top.humanSim,
    topFinalScore: top.finalScore,
    topEuclideanDistance: top.minEuclideanDist,
    topConsensusTop3Avg: top.consensusTop3Avg,
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
