import type {
  DrTechCaresApiClient,
  PatientBiometricIndexEntry,
} from '@drtechcares/api-client';
import {
  CANONICAL_HUMAN_FACE_BASELINE_512,
  COMMON_MODE_ALPHA,
  FACE_EMBEDDING_DIM,
  computeFaceCoreMetrics,
  cosineSimilarity as coreCosineSimilarity,
  l2Normalize,
  whitenEmbedding,
} from '@drtechcares/smart-biometric-core';

export interface MultiAngleMatchOptions {
  similarityThreshold?: number;
  whitenedThreshold?: number;
  marginThreshold?: number;
  faceHeightRatio?: number;
}

export interface MultiAngleMatchResult {
  matched: boolean;
  bestCandidate: PatientBiometricIndexEntry | null;
  bestAngle: 'master' | 'center' | 'left' | 'right' | 'up' | 'down' | 'none';
  similarity: number;
  confidencePercent: number;
  margin: number;
  runnerUpSimilarity: number;
}

export interface FaceBoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
  centerX: number;
  centerY: number;
  confidence: number;
}

export interface LiveFrameObservation {
  probeVector: readonly number[];
  depthScore: number;
  yawDeg: number;
  pitchDeg: number;
  microMotion: number;
  blinkDetected?: boolean;
  nonRigidMotion?: number;
  biologicalVerified?: boolean;
  spoofDetected?: boolean;
  rppgScore?: number;
  faceHeightRatio?: number;
}

export interface TemporalMatchResult {
  confirmed: boolean;
  streakCount: number;
  requiredStreak: number;
  candidate: PatientBiometricIndexEntry | null;
  confidencePercent: number;
  bestAngle: string;
  livenessPassed: boolean;
  livenessReason?: string;
}

/**
 * Computes normalized Cosine Similarity in [-1, 1] between two embedding vectors.
 */
export function cosineSimilarity(a: readonly number[], b: readonly number[]): number {
  if (a.length === 0 || b.length === 0 || a.length !== b.length) return 0;
  if (a.length === FACE_EMBEDDING_DIM) {
    return coreCosineSimilarity(a, b);
  }
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom > 0 ? Math.max(-1, Math.min(1, dot / denom)) : 0;
}

function computeWhitenedSimilarity(a: readonly number[], b: readonly number[]): number {
  if (a.length !== FACE_EMBEDDING_DIM || b.length !== FACE_EMBEDDING_DIM) {
    return cosineSimilarity(a, b);
  }
  const wa = whitenEmbedding(a, CANONICAL_HUMAN_FACE_BASELINE_512, COMMON_MODE_ALPHA);
  const wb = whitenEmbedding(b, CANONICAL_HUMAN_FACE_BASELINE_512, COMMON_MODE_ALPHA);
  return coreCosineSimilarity(wa, wb);
}

function normalizePatientFullName(name: string | undefined): string {
  return (name ?? '').trim().toLowerCase().replace(/\s+/g, ' ');
}

function isSameIdentityRecord(
  a: Pick<PatientBiometricIndexEntry, 'patientId' | 'fullName'>,
  b: Pick<PatientBiometricIndexEntry, 'patientId' | 'fullName'>,
): boolean {
  if (a.patientId === b.patientId) return true;
  const na = normalizePatientFullName(a.fullName);
  const nb = normalizePatientFullName(b.fullName);
  return Boolean(na && nb && na === nb);
}

/**
 * Matches a single live face embedding against all enrolled patients' master_embedding
 * and 5-angle centroids (center, left, right, up, down) using Cosine Similarity + margin gate.
 */
export function matchMultiAngleEmbedding(
  probeVector: readonly number[],
  candidates: readonly PatientBiometricIndexEntry[],
  options: MultiAngleMatchOptions = {},
): MultiAngleMatchResult {
  const isLongRange =
    typeof options.faceHeightRatio === 'number' && options.faceHeightRatio < 0.2;
  const isNearDistanceExplicit =
    typeof options.faceHeightRatio === 'number' && options.faceHeightRatio >= 0.2;
  const similarityThreshold = isLongRange
    ? 0.62
    : isNearDistanceExplicit
      ? Math.min(options.similarityThreshold ?? 0.72, 0.72)
      : (options.similarityThreshold ?? 0.84);
  const whitenedThreshold = isLongRange
    ? (options.whitenedThreshold ?? 0.44)
    : isNearDistanceExplicit
      ? Math.min(options.whitenedThreshold ?? 0.52, 0.56)
      : (options.whitenedThreshold ?? 0.68);
  const marginThreshold = isLongRange
    ? (options.marginThreshold ?? 0.05)
    : (options.marginThreshold ?? 0.06);

  if (probeVector.length === 0 || candidates.length === 0) {
    return {
      matched: false,
      bestCandidate: null,
      bestAngle: 'none',
      similarity: 0,
      confidencePercent: 0,
      margin: 0,
      runnerUpSimilarity: 0,
    };
  }

  const normalizedProbe = l2Normalize(probeVector);

  const scored: {
    candidate: PatientBiometricIndexEntry;
    orderIdx: number;
    angle: 'master' | 'center' | 'left' | 'right' | 'up' | 'down';
    referenceVec: readonly number[];
    rawSim: number;
    combinedSim: number;
    whitenedSim: number;
    hasCore: boolean;
    bestCoreSim: number;
    bestCoreWhitenedSim: number;
    bestCos3D: number;
    bestCraniofacialCos: number;
    bestInnerDctCos: number;
    bestInnerDctWhite: number;
    bestInnerPatchCos: number;
    bestInnerPatchWhite: number;
  }[] = [];

  for (let orderIdx = 0; orderIdx < candidates.length; orderIdx += 1) {
    const cand = candidates[orderIdx];
    const vectorsToCompare: {
      angle: 'master' | 'center' | 'left' | 'right' | 'up' | 'down';
      vec: readonly number[];
    }[] = [
      { angle: 'master', vec: cand.masterEmbedding },
      { angle: 'center', vec: cand.angleEmbeddings.center },
      { angle: 'left', vec: cand.angleEmbeddings.left },
      { angle: 'right', vec: cand.angleEmbeddings.right },
      { angle: 'up', vec: cand.angleEmbeddings.up },
      { angle: 'down', vec: cand.angleEmbeddings.down },
    ];

    let bestAngle: 'master' | 'center' | 'left' | 'right' | 'up' | 'down' = 'master';
    let bestRaw = -1;
    let bestWhite = -1;
    let masterRaw = -1;
    let hasCore = false;
    let bestCoreSim = 0;
    let bestCoreWhitenedSim = 0;
    let bestCos3D = 0;
    let bestCraniofacialCos = 0;
    let bestInnerDctCos = 0;
    let bestInnerDctWhite = 0;
    let bestInnerPatchCos = 0;
    let bestInnerPatchWhite = 0;
    let bestCoreCombined = -1;

    for (const item of vectorsToCompare) {
      if (!Array.isArray(item.vec) || item.vec.length !== normalizedProbe.length) continue;
      const normalizedItem = l2Normalize(item.vec);
      const globalRawSim = cosineSimilarity(normalizedProbe, normalizedItem);
      const globalWhitenedSim = computeWhitenedSimilarity(normalizedProbe, normalizedItem);
      const core =
        normalizedProbe.length === FACE_EMBEDDING_DIM
          ? computeFaceCoreMetrics(normalizedProbe, normalizedItem)
          : null;
      const rawSim = core ? Math.max(globalRawSim, core.faceCoreSim) : globalRawSim;
      const whitenedSim = core
        ? Math.max(globalWhitenedSim, core.faceCoreWhitenedSim)
        : globalWhitenedSim;

      if (core) {
        hasCore = true;
        const coreCombined = core.faceCoreSim + core.faceCoreWhitenedSim;
        if (coreCombined > bestCoreCombined) {
          bestCoreCombined = coreCombined;
          bestCoreSim = core.faceCoreSim;
          bestCoreWhitenedSim = core.faceCoreWhitenedSim;
          bestCos3D = core.cos3DNormalized;
          bestCraniofacialCos = core.craniofacialCos;
          bestInnerDctCos = core.innerDctCos;
          bestInnerDctWhite = core.innerDctWhitenedCos;
          bestInnerPatchCos = core.innerPatchCos;
          bestInnerPatchWhite = core.innerPatchWhitenedCos;
        }
      }

      if (item.angle === 'master') {
        masterRaw = rawSim;
      }
      if (rawSim > bestRaw) {
        bestRaw = rawSim;
        bestWhite = whitenedSim;
        bestAngle = item.angle;
      }
    }

    if (bestRaw >= 0) {
      const effectiveMaster = masterRaw >= 0 ? masterRaw : bestRaw;
      const combinedSim = Math.max(bestRaw * 0.75 + effectiveMaster * 0.25, effectiveMaster);
      const refRaw =
        Array.isArray(cand.masterEmbedding) && cand.masterEmbedding.length === normalizedProbe.length
          ? cand.masterEmbedding
          : cand.angleEmbeddings.center;
      const referenceVec =
        Array.isArray(refRaw) && refRaw.length === normalizedProbe.length ? l2Normalize(refRaw) : [];
      scored.push({
        candidate: cand,
        orderIdx,
        angle: bestAngle,
        referenceVec,
        rawSim: bestRaw,
        combinedSim,
        whitenedSim: bestWhite,
        hasCore,
        bestCoreSim,
        bestCoreWhitenedSim,
        bestCos3D,
        bestCraniofacialCos,
        bestInnerDctCos,
        bestInnerDctWhite,
        bestInnerPatchCos,
        bestInnerPatchWhite,
      });
    }
  }

  if (scored.length === 0) {
    return {
      matched: false,
      bestCandidate: null,
      bestAngle: 'none',
      similarity: 0,
      confidencePercent: 0,
      margin: 0,
      runnerUpSimilarity: 0,
    };
  }

  scored.sort((a, b) => {
    if (
      isSameIdentityRecord(a.candidate, b.candidate) &&
      Math.abs(b.combinedSim - a.combinedSim) < 0.035
    ) {
      return b.orderIdx - a.orderIdx;
    }
    return b.combinedSim - a.combinedSim;
  });
  const top = scored[0];
  const distinctRunners = scored.slice(1).filter((s) => {
    if (isSameIdentityRecord(top.candidate, s.candidate)) {
      return false;
    }
    if (
      s.rawSim >= 0.80 &&
      top.rawSim >= 0.82 &&
      top.referenceVec.length > 0 &&
      top.referenceVec.length === s.referenceVec.length &&
      cosineSimilarity(top.referenceVec, s.referenceVec) >= 0.78
    ) {
      return false;
    }
    return true;
  });
  const runnerUp = distinctRunners[0];
  const runnerUpSim = runnerUp ? runnerUp.rawSim : 0;
  const margin = runnerUp
    ? Math.max(top.rawSim - runnerUp.rawSim, top.combinedSim - runnerUp.combinedSim)
    : 1;

  const passesMargin = distinctRunners.length === 0 || margin >= marginThreshold;
  const subDctCosMin = isLongRange ? 0.52 : 0.68;
  const subDctWhiteMin = isLongRange ? 0.36 : 0.52;
  const subPatchCosMin = isLongRange ? 0.44 : 0.6;
  const subPatchWhiteMin = isLongRange ? 0.3 : 0.46;
  const sub3DMin = isLongRange ? 0.64 : 0.74;
  const passesSubbandFloor =
    !top.hasCore ||
    (top.bestInnerDctCos >= subDctCosMin &&
      top.bestInnerDctWhite >= subDctWhiteMin &&
      top.bestInnerPatchCos >= subPatchCosMin &&
      top.bestInnerPatchWhite >= subPatchWhiteMin &&
      top.bestCos3D >= sub3DMin);
  const passesStandardGate =
    top.rawSim >= similarityThreshold &&
    top.whitenedSim >= whitenedThreshold &&
    passesSubbandFloor &&
    passesMargin;
  const coreSimFloor = isLongRange ? 0.62 : Math.max(0.8, similarityThreshold);
  const coreWhitenedFloor = isLongRange ? 0.44 : Math.max(0.6, whitenedThreshold - 0.04);
  const passesFaceCoreGate =
    top.hasCore &&
    top.bestCoreSim >= coreSimFloor &&
    top.bestCoreWhitenedSim >= coreWhitenedFloor &&
    top.bestInnerDctCos >= (isLongRange ? 0.54 : 0.7) &&
    top.bestInnerDctWhite >= (isLongRange ? 0.38 : 0.54) &&
    top.bestInnerPatchCos >= (isLongRange ? 0.46 : 0.62) &&
    top.bestInnerPatchWhite >= (isLongRange ? 0.32 : 0.48) &&
    top.bestCos3D >= (isLongRange ? 0.66 : 0.76) &&
    top.bestCraniofacialCos >= (isLongRange ? 0.44 : 0.54) &&
    passesMargin;
  const matched = passesStandardGate || passesFaceCoreGate;

  const rawMatchedScore = top.hasCore
    ? Math.max(top.rawSim, top.bestCoreSim)
    : top.rawSim;
  const calibratedScore = matched
    ? Math.max(isLongRange ? 0.84 : 0.86, rawMatchedScore + (isLongRange ? 0.18 : 0))
    : Math.min(0.49, top.rawSim * 0.5);
  const displayPercent = Number(
    Math.max(0, Math.min(99.9, calibratedScore * 100)).toFixed(1),
  );

  return {
    matched,
    bestCandidate: top.candidate,
    bestAngle: top.angle,
    similarity: top.rawSim,
    confidencePercent: displayPercent,
    margin,
    runnerUpSimilarity: runnerUpSim,
  };
}

/**
 * Temporal consensus & anti-spoof liveness verifier.
 * Requires consecutive matching frames (default 4) for the same patientId plus natural 3D depth
 * and facial micro-movement / pose variance across frames (rejects printed photos).
 */
export class TemporalBiometricMatcher {
  readonly requiredConsecutiveFrames: number;
  readonly defaultSimilarityThreshold?: number;
  private streakPatientId: string | null = null;
  private streakPatientName: string | null = null;
  private streakFrames: {
    match: MultiAngleMatchResult;
    depthScore: number;
    yawDeg: number;
    pitchDeg: number;
    microMotion: number;
    blinkDetected: boolean;
  }[] = [];

  constructor(requiredConsecutiveFrames = 4, defaultSimilarityThreshold?: number) {
    this.requiredConsecutiveFrames = requiredConsecutiveFrames;
    this.defaultSimilarityThreshold = defaultSimilarityThreshold;
  }

  get currentStreak(): number {
    return this.streakFrames.length;
  }

  reset(): void {
    this.streakPatientId = null;
    this.streakPatientName = null;
    this.streakFrames = [];
  }

  evaluateFrame(
    observation: LiveFrameObservation,
    candidates: readonly PatientBiometricIndexEntry[],
    options?: MultiAngleMatchOptions,
  ): TemporalMatchResult {
    if (observation.spoofDetected) {
      this.reset();
      return {
        confirmed: false,
        streakCount: 0,
        requiredStreak: this.requiredConsecutiveFrames,
        candidate: null,
        confidencePercent: 0,
        bestAngle: 'none',
        livenessPassed: false,
        livenessReason: 'SPOOF_ATTEMPT_BLOCKED',
      };
    }

    const mergedOptions: MultiAngleMatchOptions =
      this.defaultSimilarityThreshold !== undefined
        ? {
            similarityThreshold: this.defaultSimilarityThreshold,
            ...(observation.faceHeightRatio !== undefined
              ? { faceHeightRatio: observation.faceHeightRatio }
              : {}),
            ...options,
          }
        : {
            ...(observation.faceHeightRatio !== undefined
              ? { faceHeightRatio: observation.faceHeightRatio }
              : {}),
            ...options,
          };
    const match = matchMultiAngleEmbedding(observation.probeVector, candidates, mergedOptions);

    if (!match.matched || !match.bestCandidate) {
      this.reset();
      return {
        confirmed: false,
        streakCount: 0,
        requiredStreak: this.requiredConsecutiveFrames,
        candidate: match.bestCandidate,
        confidencePercent: match.confidencePercent,
        bestAngle: match.bestAngle,
        livenessPassed:
          observation.depthScore >= 0.25 && observation.biologicalVerified !== false,
        livenessReason: match.bestCandidate ? 'LOW_CONFIDENCE_OR_MARGIN' : 'NO_MATCH',
      };
    }

    const pid = match.bestCandidate.patientId;
    const pName = normalizePatientFullName(match.bestCandidate.fullName);
    const sameIdentityStreak =
      this.streakPatientId === pid ||
      (Boolean(this.streakPatientName) && Boolean(pName) && this.streakPatientName === pName);
    if (!sameIdentityStreak) {
      this.streakPatientId = pid;
      this.streakPatientName = pName || null;
      this.streakFrames = [];
    } else {
      this.streakPatientId = pid;
      if (pName) this.streakPatientName = pName;
    }

    this.streakFrames.push({
      match,
      depthScore: observation.depthScore,
      yawDeg: observation.yawDeg,
      pitchDeg: observation.pitchDeg,
      microMotion: observation.microMotion,
      blinkDetected: Boolean(observation.blinkDetected),
    });

    if (this.streakFrames.length > this.requiredConsecutiveFrames) {
      this.streakFrames.shift();
    }

    const streakCount = this.streakFrames.length;
    const avgDepth =
      this.streakFrames.reduce((acc, f) => acc + f.depthScore, 0) / Math.max(1, streakCount);
    const maxMicroMotion = Math.max(...this.streakFrames.map((f) => f.microMotion));
    const yawRange =
      Math.max(...this.streakFrames.map((f) => f.yawDeg)) -
      Math.min(...this.streakFrames.map((f) => f.yawDeg));
    const pitchRange =
      Math.max(...this.streakFrames.map((f) => f.pitchDeg)) -
      Math.min(...this.streakFrames.map((f) => f.pitchDeg));
    const anyBlink = this.streakFrames.some((f) => f.blinkDetected);

    const hasMotionVariance =
      maxMicroMotion >= 0.0001 || yawRange >= 0.1 || pitchRange >= 0.1 || anyBlink;
    const basePhysicalLiveness =
      avgDepth >= 0.25 && (streakCount < 2 || hasMotionVariance);
    const bioVerifiedOk =
      observation.biologicalVerified === undefined ? true : observation.biologicalVerified === true;
    const livenessPassed = basePhysicalLiveness && bioVerifiedOk;

    const avgConf = Number(
      (
        this.streakFrames.reduce((acc, f) => acc + f.match.confidencePercent, 0) /
        Math.max(1, streakCount)
      ).toFixed(1),
    );

    const confirmed = streakCount >= this.requiredConsecutiveFrames && livenessPassed;

    return {
      confirmed,
      streakCount,
      requiredStreak: this.requiredConsecutiveFrames,
      candidate: match.bestCandidate,
      confidencePercent: avgConf,
      bestAngle: match.bestAngle,
      livenessPassed,
      livenessReason: !livenessPassed
        ? !bioVerifiedOk
          ? 'BIOLOGICAL_BLINK_REQUIRED'
          : avgDepth < 0.25
            ? 'FLAT_SPOOF_SUSPECTED'
            : 'STATIC_PHOTO_SUSPECTED'
        : undefined,
    };
  }
}

/**
 * Computes a normalized bounding box [0..1] in camera image coordinates from MediaPipe landmarks.
 */
export function computeBoundingBoxFromLandmarks(
  landmarks: readonly { x: number; y: number; z?: number }[],
): FaceBoundingBox | null {
  if (!landmarks || landmarks.length < 10) return null;
  let minX = 1;
  let minY = 1;
  let maxX = 0;
  let maxY = 0;
  for (const lm of landmarks) {
    if (lm.x < minX) minX = lm.x;
    if (lm.y < minY) minY = lm.y;
    if (lm.x > maxX) maxX = lm.x;
    if (lm.y > maxY) maxY = lm.y;
  }
  const padX = (maxX - minX) * 0.14;
  const padY = (maxY - minY) * 0.18;
  const x = Math.max(0.01, minX - padX);
  const y = Math.max(0.01, minY - padY);
  const right = Math.min(0.99, maxX + padX);
  const bottom = Math.min(0.99, maxY + padY);
  const width = Math.max(0.08, right - x);
  const height = Math.max(0.1, bottom - y);
  return {
    x,
    y,
    width,
    height,
    centerX: x + width / 2,
    centerY: y + height / 2,
    confidence: landmarks.length >= 400 ? 0.96 : 0.88,
  };
}

export const BIOMETRIC_INDEX_CHANGED_EVENT = 'drtechcares:biometric-index-changed';

/**
 * Broadcasts that patient biometric templates changed (enrolled/updated/deleted)
 * so active scanner views immediately refresh their local index.
 */
export function notifyBiometricIndexChanged(): void {
  if (typeof window !== 'undefined' && typeof CustomEvent !== 'undefined') {
    window.dispatchEvent(new CustomEvent(BIOMETRIC_INDEX_CHANGED_EVENT));
  }
}

/**
 * Loads the full multi-angle biometric index from the cloud API and keeps it synchronized
 * via Supabase Realtime subscriptions, focus/delete events, and periodic polling.
 */
export function subscribeBiometricIndex(
  apiClient: DrTechCaresApiClient,
  onUpdate: (entries: PatientBiometricIndexEntry[]) => void,
): () => void {
  let active = true;

  const refresh = () => {
    apiClient
      .getBiometricIndex()
      .then((entries) => {
        if (active) onUpdate(entries);
      })
      .catch(() => {});
  };

  refresh();

  const onVisibilityChange = () => {
    if (typeof document !== 'undefined' && document.visibilityState === 'visible') {
      refresh();
    }
  };

  if (typeof window !== 'undefined') {
    window.addEventListener(BIOMETRIC_INDEX_CHANGED_EVENT, refresh);
    window.addEventListener('focus', refresh);
  }
  if (typeof document !== 'undefined') {
    document.addEventListener('visibilitychange', onVisibilityChange);
  }
  const intervalId = globalThis.setInterval(refresh, 5000);

  const supabase = apiClient.getSupabaseClient();
  const channel = supabase
    ? supabase
        .channel(`biometrics-index-rt-${Date.now()}`)
        .on('postgres_changes', { event: '*', schema: 'public', table: 'patient_biometrics' }, refresh)
        .on('postgres_changes', { event: '*', schema: 'public', table: 'face_templates' }, refresh)
        .on('postgres_changes', { event: '*', schema: 'public', table: 'patients' }, refresh)
        .subscribe()
    : null;

  return () => {
    active = false;
    globalThis.clearInterval(intervalId);
    if (typeof window !== 'undefined') {
      window.removeEventListener(BIOMETRIC_INDEX_CHANGED_EVENT, refresh);
      window.removeEventListener('focus', refresh);
    }
    if (typeof document !== 'undefined') {
      document.removeEventListener('visibilitychange', onVisibilityChange);
    }
    if (channel && supabase) {
      void supabase.removeChannel(channel);
    }
  };
}

