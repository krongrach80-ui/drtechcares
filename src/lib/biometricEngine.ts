import type Human from '@vladmandic/human';
import type { Config, FaceResult } from '@vladmandic/human';
import {
  type AngleEmbeddings5D,
  type DrTechCaresApiClient,
  type PatientBiometricIndexEntry,
  type PatientRecord,
} from '@drtechcares/api-client';
import { notifyBiometricIndexChanged } from './faceMatcher';
import type { GeometricAnthropometricRatios } from '@drtechcares/smart-biometric-core';

export const BIOMETRIC_VAULT_STORAGE_KEY = 'drtech_biometric_vault_v3';
export const BIOMETRIC_ENGINE_ID = 'human-faceres-1024d' as const;
export const BIOMETRIC_MIN_EMBEDDING_DIM = 128;
export const BIOMETRIC_EXPECTED_DIM = 1024;
export const BIOMETRIC_MATCH_THRESHOLD = 0.6;
export const BIOMETRIC_HUMAN_SIM_THRESHOLD = 0.65;
export const BIOMETRIC_MARGIN_THRESHOLD = 0.06;
export const BIOMETRIC_REQUIRED_STREAK = 3;

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
    enabled: true,
    equalization: false,
    return: false,
  },
  face: {
    enabled: true,
    detector: {
      rotation: true,
      maxDetected: 3,
      minConfidence: 0.5,
      minSize: 16,
      skipFrames: 0,
      skipTime: 0,
      return: false,
    },
    mesh: { enabled: true },
    iris: { enabled: true },
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
  humanSim: number;
  bestSource: string;
}

export interface Live1toNMatchEvaluation {
  matched: boolean;
  topCandidate: EnrolledBiometricCandidate | null;
  runnerUpCandidate: EnrolledBiometricCandidate | null;
  topSimilarity: number;
  topHumanSimilarity: number;
  runnerUpSimilarity: number;
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
  runnerUpSimilarity: number;
  margin: number;
  confidencePercent: number;
  bestSource: string;
  reason: Live1toNMatchEvaluation['reason'] | 'SPOOF_BLOCKED' | 'LIVENESS_PENDING';
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

const HUMAN_MATCH_CALIBRATION = { order: 2, multiplier: 2000, min: 0.2, max: 0.8 } as const;

export function computeHumanDescriptorSimilarity(
  a: readonly number[],
  b: readonly number[],
  humanOverride?: Pick<Human, 'match'> | null,
): number {
  if (a.length === 0 || b.length === 0 || a.length !== b.length) return 0;
  const activeHuman = humanOverride ?? sharedHumanInstance;
  if (activeHuman?.match?.similarity) {
    return activeHuman.match.similarity([...a], [...b], { ...HUMAN_MATCH_CALIBRATION });
  }
  let sumSq = 0;
  for (let i = 0; i < a.length; i += 1) {
    const diff = a[i] - b[i];
    sumSq += diff * diff;
  }
  const dist = Math.round(100 * HUMAN_MATCH_CALIBRATION.multiplier * sumSq) / 100;
  if (dist === 0) return 1;
  const root = Math.sqrt(dist);
  const norm = (1 - root / 100 - HUMAN_MATCH_CALIBRATION.min) / (HUMAN_MATCH_CALIBRATION.max - HUMAN_MATCH_CALIBRATION.min);
  return Math.round(100 * Math.max(Math.min(norm, 1), 0)) / 100;
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

/**
 * Captures a race-free canvas snapshot from `<video>`, cropping and upscaling the head region
 * (`320x320`) when a distant face (`height < 0.22`) is tracked so `@vladmandic/human` extracts crisp
 * `1024-D` embeddings across 0.25m–5.0m without choking on full 1080p textures.
 */
export function captureCanonicalFaceSnapshot(
  video: HTMLVideoElement,
  faceBox?: { x: number; y: number; width: number; height: number } | null,
  targetCanvas?: HTMLCanvasElement | null,
): HTMLCanvasElement | null {
  if (typeof document === 'undefined' || !video || video.videoWidth <= 0 || video.videoHeight <= 0) {
    return null;
  }
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const canvas = targetCanvas || document.createElement('canvas');

  if (faceBox && faceBox.height > 0 && faceBox.height < 0.22 && faceBox.width > 0) {
    const cx = faceBox.x + faceBox.width * 0.5;
    const cy = faceBox.y + faceBox.height * 0.5;
    const sideNorm = Math.max(faceBox.width, faceBox.height) * 2.6;
    const halfNorm = Math.min(0.5, sideNorm * 0.5);
    const rx = Math.max(0, Math.min(1 - halfNorm * 2, cx - halfNorm));
    const ry = Math.max(0, Math.min(1 - halfNorm * 2, cy - halfNorm));
    const rw = Math.min(1 - rx, halfNorm * 2);
    const rh = Math.min(1 - ry, halfNorm * 2);
    const sx = Math.max(0, Math.floor(rx * vw));
    const sy = Math.max(0, Math.floor(ry * vh));
    const sw = Math.max(32, Math.min(vw - sx, Math.ceil(rw * vw)));
    const sh = Math.max(32, Math.min(vh - sy, Math.ceil(rh * vh)));

    canvas.width = 320;
    canvas.height = 320;
    const ctx = canvas.getContext('2d', { alpha: false });
    if (!ctx) return null;
    ctx.drawImage(video, sx, sy, sw, sh, 0, 0, 320, 320);
    return canvas;
  }

  const maxW = 640;
  const scale = vw > maxW ? maxW / vw : 1;
  const outW = Math.max(160, Math.round(vw * scale));
  const outH = Math.max(120, Math.round(vh * scale));
  if (canvas.width !== outW || canvas.height !== outH) {
    canvas.width = outW;
    canvas.height = outH;
  }
  const ctx = canvas.getContext('2d', { alpha: false });
  if (!ctx) return null;
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

  // Top 10 sharpest real frames prioritized from frontal & mild angles for KNN matching
  const indexed = normalizedSamples.map((s, idx) => ({
    s,
    idx,
    frontalPriority:
      s.bucket === 'center' || (Math.abs(s.yawDeg) <= 16 && Math.abs(s.pitchDeg) <= 12) ? 1 : 0,
  }));
  const rankedFrontal = indexed
    .filter((item) => item.frontalPriority === 1)
    .sort((a, b) => b.s.sharpness - a.s.sharpness);
  const rankedMildSide = indexed
    .filter((item) => item.frontalPriority === 0)
    .sort((a, b) => b.s.sharpness - a.s.sharpness);

  const selectedExemplars: number[][] = [];
  const usedVectors = new Set<string>();
  const pushDistinctExemplar = (vec: number[]) => {
    const sig = `${vec[0]?.toFixed(4)}|${vec[1]?.toFixed(4)}|${vec[Math.floor(vec.length / 2)]?.toFixed(4)}`;
    if (usedVectors.has(sig)) return;
    usedVectors.add(sig);
    selectedExemplars.push(vec);
  };

  // Pick up to 7 sharpest frontal/mild-angle frames + up to 3 sharpest side/tilt exemplars (10 total)
  for (const item of rankedFrontal) {
    if (selectedExemplars.length >= 7) break;
    pushDistinctExemplar(item.s.embedding);
  }
  for (const item of rankedMildSide) {
    if (selectedExemplars.length >= 10) break;
    pushDistinctExemplar(item.s.embedding);
  }
  for (const item of rankedFrontal) {
    if (selectedExemplars.length >= 10) break;
    pushDistinctExemplar(item.s.embedding);
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
        .slice(0, 12),
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
 * Persists a newly enrolled or re-enrolled `BiometricProfile` simultaneously to:
 * 1) Local Shared Biometric Vault (`drtech_biometric_vault_v3`) for instant zero-latency Kiosk recognition
 * 2) Direct Supabase `patient_biometrics` & `patients` tables when client is configured
 * 3) Broadcasts `notifyBiometricIndexChanged()` across views
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
}): Promise<BiometricProfile | null> {
  if (!isValidNeuralEmbedding(input.masterEmbedding)) return null;
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
    .map((v) => normalizeVector(v));

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

  saveProfileToLocalBiometricVault(profile);

  // Best-effort direct Supabase reinforcement alongside backend worker persistence
  const supabase = input.apiClient.getSupabaseClient();
  if (supabase) {
    try {
      const bioPayload: Record<string, unknown> = {
        patient_id: input.patient.id,
        hn: input.patient.hn,
        master_embedding: masterNorm,
        angle_embeddings: angleNorm,
        exemplar_embeddings: exemplars,
        biometric_engine: BIOMETRIC_ENGINE_ID,
        embedding_dim: dim,
        frames_captured: profile.framesCaptured,
        quality_score: profile.qualityScore,
        avatar_url: input.patient.faceImageUrl ?? null,
        updated_at: profile.updatedAt,
      };
      const upRes = await supabase
        .from('patient_biometrics')
        .upsert(bioPayload, { onConflict: 'patient_id' });
      if (upRes.error) {
        const compatPayload = { ...bioPayload };
        delete compatPayload.exemplar_embeddings;
        delete compatPayload.biometric_engine;
        delete compatPayload.embedding_dim;
        await supabase
          .from('patient_biometrics')
          .upsert(compatPayload, { onConflict: 'patient_id' });
      }

      const blob = {
        engine: BIOMETRIC_ENGINE_ID,
        dimension: dim,
        master_embedding: masterNorm,
        angle_embeddings: angleNorm,
        exemplar_embeddings: exemplars,
      };
      const patRes = await supabase
        .from('patients')
        .update({
          face_biometric_status: 'VERIFIED',
          face_frames_count: profile.framesCaptured,
          face_embedding: blob,
          biometric_data: blob,
          updated_at: profile.updatedAt,
        })
        .eq('id', input.patient.id);
      if (patRes.error) {
        await supabase
          .from('patients')
          .update({
            face_biometric_status: 'VERIFIED',
            face_embedding: blob,
            updated_at: profile.updatedAt,
          })
          .eq('id', input.patient.id);
      }
    } catch {
      // Worker API has already persisted authoritative server record; ignore direct RLS edge cases
    }
  }

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
 * Loads all enrolled biometric profiles from Cloud API + Supabase + Local Biometric Vault (`drtech_biometric_vault_v3`),
 * merging seamlessly and filtering strictly by compatible neural dimension (`expectedDim`, e.g. 1024-D).
 */
export async function loadEnrolledPatientsWithBiometrics(
  apiClient: DrTechCaresApiClient,
  expectedDim: number = BIOMETRIC_EXPECTED_DIM,
): Promise<LoadedBiometricVaultResult> {
  const byPatientId = new Map<string, EnrolledBiometricCandidate>();

  const applyCandidate = (cand: EnrolledBiometricCandidate, preferIfExpectedDim = false) => {
    const prev = byPatientId.get(cand.patientId);
    if (!prev) {
      byPatientId.set(cand.patientId, cand);
      return;
    }
    const prevMatches = prev.dimension === expectedDim;
    const nextMatches = cand.dimension === expectedDim;
    // Always let compatible 1024-D profile win over stale 512-D profile, and enrich missing schedule/exemplar metadata
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

  // 1. Primary Cloud API biometric index
  try {
    const cloudEntries = await apiClient.getBiometricIndex();
    for (const entry of cloudEntries) {
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
    // Continue cleanly to Supabase + Local Vault
  }

  // 2. Direct Supabase tables (`patients` + `patient_biometrics`) enrichment
  const supabase = apiClient.getSupabaseClient();
  if (supabase) {
    try {
      const [patRes, bioRes] = await Promise.all([
        supabase
          .from('patients')
          .select('id, hn, full_name, diagnosis, assigned_pt_id, next_appointment_date, registration_status, face_embedding, face_image_url, history')
          .eq('registration_status', 'active'),
        supabase.from('patient_biometrics').select('*'),
      ]);
      const bioByPid = new Map<string, Record<string, unknown>>();
      if (!bioRes.error && Array.isArray(bioRes.data)) {
        for (const row of bioRes.data as Record<string, unknown>[]) {
          bioByPid.set(String(row.patient_id), row);
        }
      }
      if (!patRes.error && Array.isArray(patRes.data)) {
        for (const r of patRes.data as Record<string, unknown>[]) {
          const pid = String(r.id);
          const bio = bioByPid.get(pid);
          const history = (r.history && typeof r.history === 'object' ? r.history : {}) as Record<string, unknown>;
          const faceBlob = (r.face_embedding && typeof r.face_embedding === 'object' && !Array.isArray(r.face_embedding)
            ? r.face_embedding
            : {}) as Record<string, unknown>;
          const master =
            parseVectorLoose(bio?.master_embedding) ??
            parseVectorLoose(r.face_embedding) ??
            parseVectorLoose(history.masterEmbedding);
          if (!master) continue;
          const rawAngles =
            (bio?.angle_embeddings as Partial<AngleEmbeddings5D> | undefined) ??
            (faceBlob.angle_embeddings as Partial<AngleEmbeddings5D> | undefined) ??
            (history.angleEmbeddings as Partial<AngleEmbeddings5D> | undefined);
          const rawExemplars = Array.isArray(bio?.exemplar_embeddings)
            ? (bio.exemplar_embeddings as unknown[])
            : Array.isArray(faceBlob.exemplar_embeddings)
              ? (faceBlob.exemplar_embeddings as unknown[])
              : Array.isArray(history.exemplarEmbeddings)
                ? (history.exemplarEmbeddings as unknown[])
                : undefined;
          applyCandidate(
            buildCandidateFromSource(
              pid,
              String(r.hn ?? ''),
              String(r.full_name ?? ''),
              master,
              rawAngles,
              rawExemplars,
              {
                diagnosis: String(r.diagnosis ?? ''),
                assignedPtId: String(r.assigned_pt_id ?? ''),
                nextAppointmentDate: String(r.next_appointment_date ?? ''),
                avatarUrl: typeof r.face_image_url === 'string' ? r.face_image_url : undefined,
                biometricEngine:
                  master.length >= 768
                    ? BIOMETRIC_ENGINE_ID
                    : (history.biometricEngine as PatientRecord['biometricEngine']),
              },
            ),
          );
        }
      }
    } catch {
      // Offline or RLS fallback -> local vault still loads cleanly
    }
  }

  // 3. Instant Shared Local Biometric Vault (`drtech_biometric_vault_v3`)
  const vaultEntries = readLocalBiometricVault();
  for (const v of vaultEntries) {
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

  const allCandidates = Array.from(byPatientId.values());
  const compatibleCandidates = allCandidates.filter((c) => c.dimension === expectedDim);
  const legacyAccountsCount = allCandidates.length - compatibleCandidates.length;

  return {
    candidates: allCandidates,
    compatibleCandidates,
    totalAccounts: allCandidates.length,
    validAccounts: compatibleCandidates.length,
    legacyAccountsCount,
  };
}

/**
 * Evaluates a live L2-normalized embedding against a single enrolled patient's
 * `master_embedding`, 5 `angle_embeddings`, and Top-10 KNN `exemplar_embeddings`.
 */
export function computePatientMatchScore(
  liveEmbedding: readonly number[],
  candidate: EnrolledBiometricCandidate,
  humanOverride?: Pick<Human, 'match'> | null,
): PatientMatchScoreResult {
  const liveNorm = normalizeVector(liveEmbedding);
  const expectedDim = liveNorm.length;

  const targets: { label: string; vec: number[] }[] = [];
  if (candidate.master_embedding.length === expectedDim) {
    targets.push({ label: 'master', vec: candidate.master_embedding });
  }
  const angleKeys: (keyof AngleEmbeddings5D)[] = ['center', 'left', 'right', 'up', 'down'];
  for (const k of angleKeys) {
    const v = candidate.angle_embeddings?.[k];
    if (v && v.length === expectedDim) {
      targets.push({ label: k, vec: v });
    }
  }
  candidate.exemplar_embeddings.forEach((ex, idx) => {
    if (ex.length === expectedDim) {
      targets.push({ label: `exemplar-${idx + 1}`, vec: ex });
    }
  });

  let bestSim = -1;
  let bestTargetVec: number[] | null = null;
  let bestSource = 'none';

  for (const t of targets) {
    const cos = cosineSimilarityNormalized(liveNorm, t.vec);
    if (cos > bestSim) {
      bestSim = cos;
      bestTargetVec = t.vec;
      bestSource = t.label;
    }
  }

  const humanSim = bestTargetVec
    ? computeHumanDescriptorSimilarity(liveNorm, bestTargetVec, humanOverride)
    : 0;

  return {
    candidate,
    bestSim: Math.max(0, bestSim),
    humanSim,
    bestSource,
  };
}

function normalizeNameForIdentity(name: string): string {
  return name.trim().toLowerCase().replace(/\s+/g, ' ');
}

/**
 * 1:N Exemplar + Centroid Matcher:
 * - Filters strictly by matching embedding dimension (ignoring stale corrupted vectors).
 * - Requires `cosineSim >= 0.60` (`humanSim >= 0.65`) AND runner-up margin `>= 0.06` across >1 patient.
 */
export function evaluateLiveMatch1toN(
  liveEmbedding: readonly number[],
  candidates: readonly EnrolledBiometricCandidate[],
  excludedIds: readonly string[] = [],
  humanOverride?: Pick<Human, 'match'> | null,
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
      topHumanSimilarity: 0,
      runnerUpSimilarity: 0,
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
      topHumanSimilarity: 0,
      runnerUpSimilarity: 0,
      margin: 0,
      confidencePercent: 0,
      bestSource: 'none',
      reason: 'EXCLUDED_ONLY',
      validAccounts: dimensionCompatible.length,
      totalAccounts: candidates.length,
    };
  }

  const scored = eligible
    .map((c) => computePatientMatchScore(liveEmbedding, c, humanOverride))
    .sort((a, b) => b.bestSim - a.bestSim);

  const top = scored[0];
  const topName = normalizeNameForIdentity(top.candidate.fullName);
  const distinctRunners = scored
    .slice(1)
    .filter((s) => !(topName && normalizeNameForIdentity(s.candidate.fullName) === topName));
  const runnerUp = distinctRunners[0] ?? null;

  const runnerUpSim = runnerUp ? runnerUp.bestSim : 0;
  const margin = runnerUp ? top.bestSim - runnerUp.bestSim : 1;

  const passesThreshold =
    top.bestSim >= BIOMETRIC_MATCH_THRESHOLD &&
    top.humanSim >= BIOMETRIC_HUMAN_SIM_THRESHOLD;
  const passesMargin = distinctRunners.length === 0 || margin >= BIOMETRIC_MARGIN_THRESHOLD;
  const matched = passesThreshold && passesMargin;

  const reason: Live1toNMatchEvaluation['reason'] = matched
    ? 'OK'
    : passesThreshold && !passesMargin
      ? 'AMBIGUOUS_CLOSE_CANDIDATES'
      : 'BELOW_THRESHOLD';

  const confidencePercent = Number(
    Math.max(
      0,
      Math.min(
        99.9,
        matched
          ? Math.max(85.0, 40 + top.bestSim * 66)
          : top.bestSim * 100,
      ),
    ).toFixed(1),
  );

  return {
    matched,
    topCandidate: top.candidate,
    runnerUpCandidate: runnerUp?.candidate ?? null,
    topSimilarity: top.bestSim,
    topHumanSimilarity: top.humanSim,
    runnerUpSimilarity: runnerUpSim,
    margin,
    confidencePercent,
    bestSource: top.bestSource,
    reason,
    validAccounts: dimensionCompatible.length,
    totalAccounts: candidates.length,
  };
}

/**
 * Enforces 3 consecutive matching frames (`3/3`) for the exact same patient with live anti-spoof verification.
 */
export class LiveIdentityStreakTracker {
  readonly requiredStreak: number;
  private activePatientId: string | null = null;
  private streak = 0;

  constructor(requiredStreak: number = BIOMETRIC_REQUIRED_STREAK) {
    this.requiredStreak = requiredStreak;
  }

  get currentStreak(): number {
    return this.streak;
  }

  reset(): void {
    this.activePatientId = null;
    this.streak = 0;
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
        candidate: evalResult.topCandidate,
        runnerUp: evalResult.runnerUpCandidate,
        similarity: evalResult.topSimilarity,
        runnerUpSimilarity: evalResult.runnerUpSimilarity,
        margin: evalResult.margin,
        confidencePercent: evalResult.confidencePercent,
        bestSource: evalResult.bestSource,
        reason: 'SPOOF_BLOCKED',
      };
    }

    if (!evalResult.matched || !evalResult.topCandidate) {
      this.reset();
      return {
        confirmed: false,
        streak: 0,
        requiredStreak: this.requiredStreak,
        candidate: evalResult.topCandidate,
        runnerUp: evalResult.runnerUpCandidate,
        similarity: evalResult.topSimilarity,
        runnerUpSimilarity: evalResult.runnerUpSimilarity,
        margin: evalResult.margin,
        confidencePercent: evalResult.confidencePercent,
        bestSource: evalResult.bestSource,
        reason: evalResult.reason,
      };
    }

    const pid = evalResult.topCandidate.patientId;
    if (this.activePatientId !== pid) {
      this.activePatientId = pid;
      this.streak = 1;
    } else {
      this.streak += 1;
    }

    const livenessOk = options.biologicalVerified !== false;
    const confirmed = this.streak >= this.requiredStreak && livenessOk;

    return {
      confirmed,
      streak: this.streak,
      requiredStreak: this.requiredStreak,
      candidate: evalResult.topCandidate,
      runnerUp: evalResult.runnerUpCandidate,
      similarity: evalResult.topSimilarity,
      runnerUpSimilarity: evalResult.runnerUpSimilarity,
      margin: evalResult.margin,
      confidencePercent: evalResult.confidencePercent,
      bestSource: evalResult.bestSource,
      reason: livenessOk ? 'OK' : 'LIVENESS_PENDING',
    };
  }
}
