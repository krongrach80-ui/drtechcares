import { useEffect, useMemo, useRef, useState, type RefObject } from 'react';
import {
  cosineSimilarityNormalized,
  extractAllCanonicalFaceDescriptors,
  isValidNeuralEmbedding,
  normalizeVector,
  type DetectedCanonicalFaceEntry,
} from '../lib/biometricEngine';
import {
  computeObjectCoverLayout,
  projectMirroredNormToCoverViewport,
  type OwnerBiometricTemplate,
} from '../lib/ownerBiometricLock';

export type AccountOwnerIdentityStatus =
  | 'VERIFIED_OWNER'
  | 'UNAUTHORIZED_PERSON'
  | 'NO_OWNER_FACE';

export interface NormalizedFaceBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface MirroredOverlayFaceBox {
  leftPct: number;
  topPct: number;
  widthPct: number;
  heightPct: number;
}

export interface AccountOwnerGuardMachineState {
  identityStatus: AccountOwnerIdentityStatus;
  canExercise: boolean;
  similarity: number;
  mismatchStreak: number;
  lastFaceSeenAtMs: number;
  trackedRawBox: NormalizedFaceBox | null;
  faceCount: number;
  unauthorizedSinceMs: number | null;
}

export interface AccountOwnerGuardFrameEvaluation {
  matchedOwner: boolean;
  bestSimilarity: number;
  trackedRawBox: NormalizedFaceBox | null;
  faceCount: number;
  hasEvaluatedEmbedding: boolean;
}

export interface AccountOwnerFaceGuardResult {
  active: boolean;
  identityStatus: AccountOwnerIdentityStatus;
  canExercise: boolean;
  similarity: number;
  trackedRawBox: NormalizedFaceBox | null;
  faceCount: number;
  effectiveThreshold: number;
  ownerEmbeddingCount: number;
}

export const OWNER_GUARD_CHECK_INTERVAL_MS = 450;
export const OWNER_GUARD_MAX_CANVAS_WIDTH = 320;
export const OWNER_GUARD_MISMATCH_STREAK_REQUIRED = 2;
export const OWNER_GUARD_NO_FACE_GRACE_MS = 1500;
export const OWNER_GUARD_DEFAULT_BASE_THRESHOLD = 0.72;
export const OWNER_GUARD_MIN_MATCH_THRESHOLD = 0.66;
export const OWNER_GUARD_THRESHOLD_MARGIN = 0.04;
export const OWNER_GUARD_VOICE_COOLDOWN_MS = 6000;
export const OWNER_GUARD_UNAUTHORIZED_TTS_TH =
  'ตรวจพบบุคคลอื่นหน้ากล้องครับ ระบบระงับการนับรอบชั่วคราว กรุณาให้เจ้าของบัญชีเป็นผู้ทำกายภาพบำบัดนะครับ';

/**
 * Computes the continuous exercise-session owner match threshold:
 * `Math.max(0.66, baseMatchThreshold - 0.04)` so natural exercise head tilts
 * remain smooth while clearly rejecting unauthorized individuals.
 */
export function computeOwnerGuardMatchThreshold(
  baseMatchThreshold: number = OWNER_GUARD_DEFAULT_BASE_THRESHOLD,
): number {
  const safeBase = Number.isFinite(baseMatchThreshold)
    ? baseMatchThreshold
    : OWNER_GUARD_DEFAULT_BASE_THRESHOLD;
  return Number(
    Math.max(OWNER_GUARD_MIN_MATCH_THRESHOLD, safeBase - OWNER_GUARD_THRESHOLD_MARGIN).toFixed(4),
  );
}

/**
 * Collects ALL valid L2-normalized reference embeddings belonging exclusively to the
 * logged-in account owner (`masterEmbedding`, all 5 `angleEmbeddings`, and `exemplarEmbeddings`).
 */
export function collectOwnerReferenceEmbeddings(
  ownerBiometrics?: OwnerBiometricTemplate | null,
): number[][] {
  if (!ownerBiometrics) return [];
  const rawList: Array<readonly number[] | undefined | null> = [
    ownerBiometrics.masterEmbedding,
    ownerBiometrics.angleEmbeddings?.center,
    ownerBiometrics.angleEmbeddings?.left,
    ownerBiometrics.angleEmbeddings?.right,
    ownerBiometrics.angleEmbeddings?.up,
    ownerBiometrics.angleEmbeddings?.down,
    ...(ownerBiometrics.exemplarEmbeddings ?? []),
  ];

  const collected: number[][] = [];
  let expectedDim = 0;
  for (const candidate of rawList) {
    if (!candidate || !isValidNeuralEmbedding(candidate, 128)) continue;
    if (expectedDim === 0) {
      expectedDim = candidate.length;
    }
    if (candidate.length !== expectedDim) continue;
    const norm = normalizeVector(candidate);
    if (norm.length === expectedDim) {
      collected.push(norm);
    }
  }
  return collected;
}

/**
 * Evaluates all detected faces in the current camera frame against the logged-in
 * account owner's exclusive embedding bank (`master` + `5 angles` + `exemplars`).
 */
export function evaluateDetectedFacesAgainstOwner(params: {
  faces: readonly DetectedCanonicalFaceEntry[];
  ownerEmbeddings: readonly (readonly number[])[];
  baseMatchThreshold?: number;
}): AccountOwnerGuardFrameEvaluation {
  const { faces, ownerEmbeddings } = params;
  const threshold = computeOwnerGuardMatchThreshold(params.baseMatchThreshold);
  if (!faces || faces.length === 0) {
    return {
      matchedOwner: false,
      bestSimilarity: 0,
      trackedRawBox: null,
      faceCount: 0,
      hasEvaluatedEmbedding: false,
    };
  }
  if (!ownerEmbeddings || ownerEmbeddings.length === 0) {
    return {
      matchedOwner: true,
      bestSimilarity: 1,
      trackedRawBox: faces[0]?.box ?? null,
      faceCount: faces.length,
      hasEvaluatedEmbedding: false,
    };
  }

  const targetDim = ownerEmbeddings[0].length;
  let bestOwnerMatch: { similarity: number; box: NormalizedFaceBox | null } | null = null;
  let highestAnySimilarity = -1;
  let largestFaceBox: NormalizedFaceBox | null = null;
  let largestFaceArea = -1;
  let hasEvaluatedEmbedding = false;

  for (const face of faces) {
    const area = face.box ? face.box.width * face.box.height : 0;
    if (face.box && area > largestFaceArea) {
      largestFaceArea = area;
      largestFaceBox = face.box;
    }

    if (!face.embedding || face.embedding.length !== targetDim) {
      continue;
    }
    hasEvaluatedEmbedding = true;
    const probeNorm = normalizeVector(face.embedding);
    let faceBestSim = -1;
    for (const ref of ownerEmbeddings) {
      if (ref.length !== targetDim) continue;
      const sim = cosineSimilarityNormalized(probeNorm, ref);
      if (sim > faceBestSim) {
        faceBestSim = sim;
      }
    }
    if (faceBestSim > highestAnySimilarity) {
      highestAnySimilarity = faceBestSim;
    }
    if (faceBestSim >= threshold) {
      if (!bestOwnerMatch || faceBestSim > bestOwnerMatch.similarity) {
        bestOwnerMatch = {
          similarity: faceBestSim,
          box: face.box ?? largestFaceBox,
        };
      }
    }
  }

  if (bestOwnerMatch) {
    return {
      matchedOwner: true,
      bestSimilarity: Math.max(0, Math.min(1, bestOwnerMatch.similarity)),
      trackedRawBox: bestOwnerMatch.box,
      faceCount: faces.length,
      hasEvaluatedEmbedding,
    };
  }

  return {
    matchedOwner: false,
    bestSimilarity: Math.max(0, Math.min(1, highestAnySimilarity)),
    trackedRawBox: largestFaceBox,
    faceCount: faces.length,
    hasEvaluatedEmbedding,
  };
}

export function createInitialAccountOwnerGuardState(nowMs: number): AccountOwnerGuardMachineState {
  return {
    identityStatus: 'VERIFIED_OWNER',
    canExercise: true,
    similarity: 1,
    mismatchStreak: 0,
    lastFaceSeenAtMs: nowMs,
    trackedRawBox: null,
    faceCount: 0,
    unauthorizedSinceMs: null,
  };
}

/**
 * Pure 3-state anti-flicker identity state machine:
 * 1. `'VERIFIED_OWNER'` -> immediately unlocks exercise (`canExercise = true`).
 * 2. `'UNAUTHORIZED_PERSON'` -> requires 2 consecutive mismatched checks (~900ms)
 *    before locking (`canExercise = false`) to avoid transient motion-blur false alarms.
 * 3. `'NO_OWNER_FACE'` -> allows a 1500ms grace buffer when no face is visible before locking.
 */
export function stepAccountOwnerGuardState(
  prev: AccountOwnerGuardMachineState,
  observation: AccountOwnerGuardFrameEvaluation & { nowMs: number },
): AccountOwnerGuardMachineState {
  const { nowMs, matchedOwner, bestSimilarity, trackedRawBox, faceCount, hasEvaluatedEmbedding } =
    observation;

  if (matchedOwner) {
    return {
      identityStatus: 'VERIFIED_OWNER',
      canExercise: true,
      similarity: bestSimilarity,
      mismatchStreak: 0,
      lastFaceSeenAtMs: nowMs,
      trackedRawBox,
      faceCount,
      unauthorizedSinceMs: null,
    };
  }

  if (faceCount > 0 && hasEvaluatedEmbedding) {
    const nextStreak = prev.mismatchStreak + 1;
    if (nextStreak >= OWNER_GUARD_MISMATCH_STREAK_REQUIRED) {
      return {
        identityStatus: 'UNAUTHORIZED_PERSON',
        canExercise: false,
        similarity: bestSimilarity,
        mismatchStreak: nextStreak,
        lastFaceSeenAtMs: nowMs,
        trackedRawBox,
        faceCount,
        unauthorizedSinceMs: prev.unauthorizedSinceMs ?? nowMs,
      };
    }
    return {
      ...prev,
      similarity: prev.identityStatus === 'VERIFIED_OWNER' ? prev.similarity : bestSimilarity,
      mismatchStreak: nextStreak,
      lastFaceSeenAtMs: nowMs,
      trackedRawBox: trackedRawBox ?? prev.trackedRawBox,
      faceCount,
    };
  }

  // Face present without extracted neural descriptor (transient blur): keep current grace state
  if (faceCount > 0 && !hasEvaluatedEmbedding) {
    return {
      ...prev,
      lastFaceSeenAtMs: nowMs,
      trackedRawBox: trackedRawBox ?? prev.trackedRawBox,
      faceCount,
    };
  }

  // No face detected in frame: enforce 1500ms grace buffer before locking to NO_OWNER_FACE
  const elapsedNoFaceMs = nowMs - prev.lastFaceSeenAtMs;
  if (elapsedNoFaceMs > OWNER_GUARD_NO_FACE_GRACE_MS) {
    return {
      identityStatus: 'NO_OWNER_FACE',
      canExercise: false,
      similarity: 0,
      mismatchStreak: 0,
      lastFaceSeenAtMs: prev.lastFaceSeenAtMs,
      trackedRawBox: null,
      faceCount: 0,
      unauthorizedSinceMs: null,
    };
  }

  return {
    ...prev,
    mismatchStreak: 0,
    trackedRawBox: null,
    faceCount: 0,
  };
}

/**
 * Projects a raw normalized face bounding box (`[0..1]` in video space) into mirrored
 * `object-cover` viewport percentage coordinates so the HTML/SVG face tracking box aligns
 * cleanly over `<video className="object-cover scale-x-[-1]" />`.
 */
export function projectRawFaceBoxToMirroredCover(
  rawBox: NormalizedFaceBox | null,
  videoW: number,
  videoH: number,
  containerW: number,
  containerH: number,
): MirroredOverlayFaceBox | null {
  if (!rawBox || videoW <= 0 || videoH <= 0 || containerW <= 0 || containerH <= 0) {
    return null;
  }
  const layout = computeObjectCoverLayout(videoW, videoH, containerW, containerH);
  const padX = rawBox.width * 0.08;
  const padY = rawBox.height * 0.1;
  const rxMin = Math.max(0, rawBox.x - padX);
  const rxMax = Math.min(1, rawBox.x + rawBox.width + padX);
  const ryMin = Math.max(0, rawBox.y - padY);
  const ryMax = Math.min(1, rawBox.y + rawBox.height + padY);

  const pRightMirrored = projectMirroredNormToCoverViewport(rxMax, ryMin, layout);
  const pLeftMirrored = projectMirroredNormToCoverViewport(rxMin, ryMax, layout);

  const left = Math.max(0, Math.min(0.95, Math.min(pRightMirrored.x, pLeftMirrored.x)));
  const right = Math.max(left + 0.04, Math.min(1, Math.max(pRightMirrored.x, pLeftMirrored.x)));
  const top = Math.max(0, Math.min(0.95, Math.min(pRightMirrored.y, pLeftMirrored.y)));
  const bottom = Math.max(top + 0.04, Math.min(1, Math.max(pRightMirrored.y, pLeftMirrored.y)));

  return {
    leftPct: Number((left * 100).toFixed(2)),
    topPct: Number((top * 100).toFixed(2)),
    widthPct: Number(((right - left) * 100).toFixed(2)),
    heightPct: Number(((bottom - top) * 100).toFixed(2)),
  };
}

export interface UseAccountOwnerFaceGuardOptions {
  enabled: boolean;
  videoRef: RefObject<HTMLVideoElement | null>;
  ownerBiometrics?: OwnerBiometricTemplate | null;
  baseMatchThreshold?: number;
  checkIntervalMs?: number;
}

/**
 * Non-blocking, mutex-locked continuous account-owner biometric verification hook.
 * Runs async 1024D face embedding checks every ~450ms on a 320px offscreen canvas
 * so 60FPS MediaPipe pose tracking and ROM calculation never stall.
 */
export function useAccountOwnerFaceGuard(
  options: UseAccountOwnerFaceGuardOptions,
): AccountOwnerFaceGuardResult {
  const {
    enabled,
    videoRef,
    ownerBiometrics,
    baseMatchThreshold = OWNER_GUARD_DEFAULT_BASE_THRESHOLD,
    checkIntervalMs = OWNER_GUARD_CHECK_INTERVAL_MS,
  } = options;

  const ownerEmbeddings = useMemo(
    () => collectOwnerReferenceEmbeddings(ownerBiometrics),
    [ownerBiometrics],
  );
  const effectiveThreshold = useMemo(
    () => computeOwnerGuardMatchThreshold(baseMatchThreshold),
    [baseMatchThreshold],
  );
  const isActive = enabled && ownerEmbeddings.length > 0;

  const [guardState, setGuardState] = useState<AccountOwnerGuardMachineState>(() =>
    createInitialAccountOwnerGuardState(Date.now()),
  );
  const stateRef = useRef<AccountOwnerGuardMachineState>(guardState);
  const isCheckingRef = useRef(false);
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const ownerEmbeddingsRef = useRef(ownerEmbeddings);
  ownerEmbeddingsRef.current = ownerEmbeddings;
  const baseThresholdRef = useRef(baseMatchThreshold);
  baseThresholdRef.current = baseMatchThreshold;

  const ownerIdentityKey = `${ownerBiometrics?.patientId ?? ''}:${ownerEmbeddings.length}`;

  useEffect(() => {
    const initial = createInitialAccountOwnerGuardState(Date.now());
    stateRef.current = initial;
    setGuardState(initial);
    isCheckingRef.current = false;

    if (!isActive || typeof window === 'undefined' || typeof document === 'undefined') {
      return undefined;
    }

    let cancelled = false;

    const runSingleCheck = async () => {
      if (cancelled || isCheckingRef.current) return;
      const video = videoRef.current;
      if (!video || video.readyState < 2 || video.videoWidth <= 0 || video.videoHeight <= 0) {
        return;
      }

      isCheckingRef.current = true;
      try {
        let canvas = offscreenCanvasRef.current;
        if (!canvas) {
          canvas = document.createElement('canvas');
          offscreenCanvasRef.current = canvas;
        }
        const targetW = Math.min(OWNER_GUARD_MAX_CANVAS_WIDTH, video.videoWidth);
        const targetH = Math.max(
          120,
          Math.round(targetW * (video.videoHeight / Math.max(1, video.videoWidth))),
        );
        if (canvas.width !== targetW || canvas.height !== targetH) {
          canvas.width = targetW;
          canvas.height = targetH;
        }
        const ctx = canvas.getContext('2d', { alpha: false, willReadFrequently: true });
        if (!ctx) return;
        ctx.drawImage(video, 0, 0, targetW, targetH);

        const faces = await extractAllCanonicalFaceDescriptors(canvas);
        if (cancelled) return;

        const evaluation = evaluateDetectedFacesAgainstOwner({
          faces,
          ownerEmbeddings: ownerEmbeddingsRef.current,
          baseMatchThreshold: baseThresholdRef.current,
        });
        const nextState = stepAccountOwnerGuardState(stateRef.current, {
          ...evaluation,
          nowMs: Date.now(),
        });
        const prev = stateRef.current;
        stateRef.current = nextState;

        const boxChanged =
          Boolean(prev.trackedRawBox) !== Boolean(nextState.trackedRawBox) ||
          (prev.trackedRawBox &&
            nextState.trackedRawBox &&
            (Math.abs(prev.trackedRawBox.x - nextState.trackedRawBox.x) > 0.008 ||
              Math.abs(prev.trackedRawBox.y - nextState.trackedRawBox.y) > 0.008 ||
              Math.abs(prev.trackedRawBox.width - nextState.trackedRawBox.width) > 0.008 ||
              Math.abs(prev.trackedRawBox.height - nextState.trackedRawBox.height) > 0.008));

        if (
          prev.identityStatus !== nextState.identityStatus ||
          prev.canExercise !== nextState.canExercise ||
          prev.faceCount !== nextState.faceCount ||
          Math.round(prev.similarity * 100) !== Math.round(nextState.similarity * 100) ||
          boxChanged
        ) {
          setGuardState(nextState);
        }
      } catch {
        // Never disrupt 60FPS pose tracking on transient Human/WebGL detection errors
      } finally {
        isCheckingRef.current = false;
      }
    };

    const intervalId = window.setInterval(() => {
      void runSingleCheck();
    }, Math.max(250, checkIntervalMs));

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
      isCheckingRef.current = false;
    };
  }, [isActive, ownerIdentityKey, checkIntervalMs, videoRef]);

  return {
    active: isActive,
    identityStatus: isActive ? guardState.identityStatus : 'VERIFIED_OWNER',
    canExercise: isActive ? guardState.canExercise : true,
    similarity: isActive ? guardState.similarity : 1,
    trackedRawBox: isActive ? guardState.trackedRawBox : null,
    faceCount: isActive ? guardState.faceCount : 0,
    effectiveThreshold,
    ownerEmbeddingCount: ownerEmbeddings.length,
  };
}

