import { HAND21_CONNECTIONS, POSE_INDEX, type PoseLandmark } from '@drtechcares/pose-engine';

export const SKELETON_JOINT_MIN_SCORE = 0.52;
export const SKELETON_EDGE_MARGIN_NORM = 0.03;
export const SKELETON_CANVAS_MARGIN_PX = 4;
export const HAND21_CANVAS_MARGIN_PX = 6;
export const MAX_JOINT_TELEPORT_RATIO = 0.38;
export const MAX_JOINT_CLAMP_STEP_RATIO = 0.12;
export const UPPER_ARM_MIN_RATIO = 0.20;
export const UPPER_ARM_MAX_RATIO = 1.30;
export const FOREARM_MIN_RATIO = 0.18;
export const FOREARM_MAX_RATIO = 1.30;
export const HAND21_WRIST_TETHER_MAX_RATIO = 0.40;
export const HAND21_BONE_MAX_RATIO = 0.30;
export const HAND_OUT_OF_FRAME_BADGE_TH = 'มืออยู่นอกกรอบกล้อง';

export type ScoredLandmark = PoseLandmark & { score?: number };

export interface CanvasPoint2D {
  x: number;
  y: number;
}

/**
 * Maps raw video normalized coordinates `[0..1]` onto the `object-cover` canvas
 * with optional horizontal mirroring (`scale-x-[-1]`).
 */
export function mapVideoPointToCoverCanvas(
  vx: number,
  vy: number,
  videoW: number,
  videoH: number,
  canvasW: number,
  canvasH: number,
  isMirrored = true,
): CanvasPoint2D {
  const cw = canvasW > 0 ? canvasW : 640;
  const ch = canvasH > 0 ? canvasH : 480;
  const vw = videoW > 0 ? videoW : cw;
  const vh = videoH > 0 ? videoH : ch;
  // Normalized MediaPipe landmarks may extend slightly beyond [0..1] (e.g., 1.06..1.50 when limbs leave frame);
  // only interpret inputs as raw pixels when magnitude clearly exceeds normalized range (> 2.5).
  const isRawPixel = Math.abs(vx) > 2.5 || Math.abs(vy) > 2.5;
  const px = isRawPixel ? vx : vx * vw;
  const py = isRawPixel ? vy : vy * vh;
  const scale = Math.max(cw / vw, ch / vh);
  const scaledW = vw * scale;
  const scaledH = vh * scale;
  const offsetX = (cw - scaledW) / 2;
  const offsetY = (ch - scaledH) / 2;
  const rawCanvasX = px * scale + offsetX;
  const canvasY = py * scale + offsetY;
  const canvasX = isMirrored ? cw - rawCanvasX : rawCanvasX;
  return { x: canvasX, y: canvasY };
}

/**
 * Viewport Edge Margin & Confidence Gate:
 * - Confidence: requires `score >= minScore` (if present) and `visibility >= minScore` (if present)
 * - Camera Border Clipping Gate: normalized coord outside `[0.03, 0.97]` OR canvas coord
 *   outside `[marginPx, canvasW - marginPx]` x `[marginPx, canvasH - marginPx]` -> OUT_OF_FRAME.
 */
export function isJointValidInFrame(params: {
  lm: ScoredLandmark | undefined;
  canvasPt: CanvasPoint2D;
  canvasW: number;
  canvasH: number;
  minScore?: number;
  marginPx?: number;
  edgeMarginNorm?: number;
  requireConfidence?: boolean;
}): boolean {
  const { lm, canvasPt, canvasW, canvasH } = params;
  if (!lm || !Number.isFinite(lm.x) || !Number.isFinite(lm.y)) return false;
  if (!Number.isFinite(canvasPt.x) || !Number.isFinite(canvasPt.y)) return false;

  const minScore = params.minScore ?? SKELETON_JOINT_MIN_SCORE;
  const marginPx = params.marginPx ?? SKELETON_CANVAS_MARGIN_PX;
  const edge = params.edgeMarginNorm ?? SKELETON_EDGE_MARGIN_NORM;
  const requireConf = params.requireConfidence ?? true;

  if (typeof lm.score === 'number' && lm.score < minScore) return false;
  if (typeof lm.visibility === 'number' && lm.visibility < minScore) return false;
  if (requireConf && typeof lm.score !== 'number' && typeof lm.visibility !== 'number') {
    return false;
  }

  if (lm.x < edge || lm.x > 1 - edge || lm.y < edge || lm.y > 1 - edge) {
    return false;
  }
  if (
    canvasPt.x < marginPx ||
    canvasPt.x > canvasW - marginPx ||
    canvasPt.y < marginPx ||
    canvasPt.y > canvasH - marginPx
  ) {
    return false;
  }
  return true;
}

/**
 * Computes anatomical shoulder width in isotropic canvas pixels, falling back to `canvasW * 0.22`.
 */
export function computeCanvasShoulderWidth(
  landmarks: readonly ScoredLandmark[],
  videoW: number,
  videoH: number,
  canvasW: number,
  canvasH: number,
  isMirrored = true,
): number {
  const cw = canvasW > 0 ? canvasW : 640;
  const fallback = Math.max(24, cw * 0.22);
  const lSh = landmarks[POSE_INDEX.LEFT_SHOULDER];
  const rSh = landmarks[POSE_INDEX.RIGHT_SHOULDER];
  if (!lSh || !rSh) return fallback;
  const lConf = Math.min(lSh.visibility ?? lSh.score ?? 0, lSh.score ?? lSh.visibility ?? 0);
  const rConf = Math.min(rSh.visibility ?? rSh.score ?? 0, rSh.score ?? rSh.visibility ?? 0);
  if (lConf < 0.2 || rConf < 0.2) return fallback;
  const lPt = mapVideoPointToCoverCanvas(lSh.x, lSh.y, videoW, videoH, cw, canvasH, isMirrored);
  const rPt = mapVideoPointToCoverCanvas(rSh.x, rSh.y, videoW, videoH, cw, canvasH, isMirrored);
  const span = Math.hypot(rPt.x - lPt.x, rPt.y - lPt.y);
  return span >= Math.max(18, cw * 0.06) ? span : fallback;
}

const UPPER_LIMB_VEL_CLAMP_INDICES: readonly number[] = [
  POSE_INDEX.LEFT_ELBOW,
  POSE_INDEX.RIGHT_ELBOW,
  POSE_INDEX.LEFT_WRIST,
  POSE_INDEX.RIGHT_WRIST,
  POSE_INDEX.LEFT_PINKY,
  POSE_INDEX.RIGHT_PINKY,
  POSE_INDEX.LEFT_INDEX,
  POSE_INDEX.RIGHT_INDEX,
  POSE_INDEX.LEFT_THUMB,
  POSE_INDEX.RIGHT_THUMB,
];

function invalidateLimbJoint<T extends ScoredLandmark>(lm: T | undefined): T {
  const base = (lm ?? { x: 0, y: 0, z: 0 }) as T;
  return {
    ...base,
    z: base.z ?? 0,
    visibility: 0,
    ...(typeof base.score === 'number' ? { score: 0 } : {}),
  };
}

export interface ArmChainSafetyEvaluation {
  shoulderWidthPx: number;
  points: CanvasPoint2D[];
  validMask: boolean[];
  leftUpperArmValid: boolean;
  leftForearmValid: boolean;
  rightUpperArmValid: boolean;
  rightForearmValid: boolean;
  leftWristOutOfFrame: boolean;
  rightWristOutOfFrame: boolean;
  shouldDrawBone: (aIdx: number, bIdx: number) => boolean;
}

/**
 * Evaluates all body joints against:
 * 1) `isJointValidInFrame` (3% viewport margin, 4px canvas margin, confidence >= 0.52 for limb/hand joints)
 * 2) Anatomical arm bone-length ratio `[0.20..1.30] * shoulderWidth` (upper arm) and `[0.18..1.30] * shoulderWidth` (forearm)
 */
export function evaluatePoseCanvasSkeletonSafety(params: {
  landmarks: readonly ScoredLandmark[];
  videoW: number;
  videoH: number;
  canvasW: number;
  canvasH: number;
  isMirrored?: boolean;
  minScoreForIdx?: (idx: number) => number;
}): ArmChainSafetyEvaluation {
  const { landmarks, videoW, videoH, canvasW, canvasH } = params;
  const isMirrored = params.isMirrored ?? true;
  const cw = canvasW > 0 ? canvasW : 640;
  const ch = canvasH > 0 ? canvasH : 480;
  const shoulderWidthPx = computeCanvasShoulderWidth(landmarks, videoW, videoH, cw, ch, isMirrored);

  const points: CanvasPoint2D[] = landmarks.map((lm) =>
    mapVideoPointToCoverCanvas(lm.x, lm.y, videoW, videoH, cw, ch, isMirrored),
  );

  const defaultMinScore = (idx: number): number => {
    if (
      idx === POSE_INDEX.LEFT_ELBOW ||
      idx === POSE_INDEX.RIGHT_ELBOW ||
      idx === POSE_INDEX.LEFT_WRIST ||
      idx === POSE_INDEX.RIGHT_WRIST ||
      idx === POSE_INDEX.LEFT_PINKY ||
      idx === POSE_INDEX.RIGHT_PINKY ||
      idx === POSE_INDEX.LEFT_INDEX ||
      idx === POSE_INDEX.RIGHT_INDEX ||
      idx === POSE_INDEX.LEFT_THUMB ||
      idx === POSE_INDEX.RIGHT_THUMB
    ) {
      return SKELETON_JOINT_MIN_SCORE;
    }
    if (idx === POSE_INDEX.LEFT_SHOULDER || idx === POSE_INDEX.RIGHT_SHOULDER) {
      return 0.28;
    }
    return 0.52;
  };
  const minScoreFor = params.minScoreForIdx ?? defaultMinScore;

  const validMask: boolean[] = landmarks.map((lm, idx) =>
    isJointValidInFrame({
      lm,
      canvasPt: points[idx],
      canvasW: cw,
      canvasH: ch,
      minScore: minScoreFor(idx),
    }),
  );

  const checkSide = (shIdx: number, elIdx: number, wrIdx: number, palmIndices: readonly number[]) => {
    const shOk = validMask[shIdx] ?? false;
    const elOk = validMask[elIdx] ?? false;
    const wrOk = validMask[wrIdx] ?? false;

    const shLm = landmarks[shIdx];
    const elLm = landmarks[elIdx];
    const wrLm = landmarks[wrIdx];
    const shPt = points[shIdx];
    const elPt = points[elIdx];
    const wrPt = points[wrIdx];

    const isForeshortenedTowardLens = (a: ScoredLandmark | undefined, b: ScoredLandmark | undefined) => {
      if (!a || !b) return false;
      const confB = Math.min(b.visibility ?? 1, b.score ?? 1);
      const insideFrame = b.x >= 0.06 && b.x <= 0.94 && b.y >= 0.06 && b.y <= 0.94;
      const depthDz = Math.abs((b.z ?? 0) - (a.z ?? 0));
      return confB >= 0.55 && insideFrame && depthDz >= 0.1;
    };

    const upperLen = shPt && elPt ? Math.hypot(elPt.x - shPt.x, elPt.y - shPt.y) : 0;
    const upperForeshortened = isForeshortenedTowardLens(shLm, elLm);
    const upperRatioOk =
      shOk &&
      elOk &&
      (upperForeshortened || upperLen >= UPPER_ARM_MIN_RATIO * shoulderWidthPx) &&
      upperLen <= UPPER_ARM_MAX_RATIO * shoulderWidthPx;

    if (!upperRatioOk) {
      validMask[elIdx] = false;
      validMask[wrIdx] = false;
      for (const pIdx of palmIndices) validMask[pIdx] = false;
      return { upperArmValid: false, forearmValid: false, wristOutOfFrame: false };
    }

    const foreLen = elPt && wrPt ? Math.hypot(wrPt.x - elPt.x, wrPt.y - elPt.y) : 0;
    const foreForeshortened = isForeshortenedTowardLens(elLm, wrLm);
    const foreRatioOk =
      wrOk &&
      (foreForeshortened || foreLen >= FOREARM_MIN_RATIO * shoulderWidthPx) &&
      foreLen <= FOREARM_MAX_RATIO * shoulderWidthPx;

    if (!foreRatioOk) {
      validMask[wrIdx] = false;
      for (const pIdx of palmIndices) validMask[pIdx] = false;
      return { upperArmValid: true, forearmValid: false, wristOutOfFrame: true };
    }

    for (const pIdx of palmIndices) {
      if (!validMask[pIdx]) continue;
      const pPt = points[pIdx];
      if (!pPt || Math.hypot(pPt.x - wrPt.x, pPt.y - wrPt.y) > 0.45 * shoulderWidthPx) {
        validMask[pIdx] = false;
      }
    }

    return { upperArmValid: true, forearmValid: true, wristOutOfFrame: false };
  };

  const leftSide = checkSide(
    POSE_INDEX.LEFT_SHOULDER,
    POSE_INDEX.LEFT_ELBOW,
    POSE_INDEX.LEFT_WRIST,
    [POSE_INDEX.LEFT_PINKY, POSE_INDEX.LEFT_INDEX, POSE_INDEX.LEFT_THUMB],
  );
  const rightSide = checkSide(
    POSE_INDEX.RIGHT_SHOULDER,
    POSE_INDEX.RIGHT_ELBOW,
    POSE_INDEX.RIGHT_WRIST,
    [POSE_INDEX.RIGHT_PINKY, POSE_INDEX.RIGHT_INDEX, POSE_INDEX.RIGHT_THUMB],
  );

  const shouldDrawBone = (aIdx: number, bIdx: number): boolean => {
    if (!validMask[aIdx] || !validMask[bIdx]) return false;
    if (
      (aIdx === POSE_INDEX.LEFT_SHOULDER && bIdx === POSE_INDEX.LEFT_ELBOW) ||
      (aIdx === POSE_INDEX.LEFT_ELBOW && bIdx === POSE_INDEX.LEFT_SHOULDER)
    ) {
      return leftSide.upperArmValid;
    }
    if (
      (aIdx === POSE_INDEX.LEFT_ELBOW && bIdx === POSE_INDEX.LEFT_WRIST) ||
      (aIdx === POSE_INDEX.LEFT_WRIST && bIdx === POSE_INDEX.LEFT_ELBOW)
    ) {
      return leftSide.forearmValid;
    }
    if (
      (aIdx === POSE_INDEX.RIGHT_SHOULDER && bIdx === POSE_INDEX.RIGHT_ELBOW) ||
      (aIdx === POSE_INDEX.RIGHT_ELBOW && bIdx === POSE_INDEX.RIGHT_SHOULDER)
    ) {
      return rightSide.upperArmValid;
    }
    if (
      (aIdx === POSE_INDEX.RIGHT_ELBOW && bIdx === POSE_INDEX.RIGHT_WRIST) ||
      (aIdx === POSE_INDEX.RIGHT_WRIST && bIdx === POSE_INDEX.RIGHT_ELBOW)
    ) {
      return rightSide.forearmValid;
    }
    return true;
  };

  return {
    shoulderWidthPx,
    points,
    validMask,
    leftUpperArmValid: leftSide.upperArmValid,
    leftForearmValid: leftSide.forearmValid,
    rightUpperArmValid: rightSide.upperArmValid,
    rightForearmValid: rightSide.forearmValid,
    leftWristOutOfFrame: leftSide.wristOutOfFrame,
    rightWristOutOfFrame: rightSide.wristOutOfFrame,
    shouldDrawBone,
  };
}

/**
 * Kinematic Skeleton Lock applied BEFORE frame smoothing/lerp:
 * - Enforces viewport border (`3%` / `4px`) and `>= 0.52` confidence on elbow/wrist/palm.
 * - Rejects single-frame teleportation `> 0.38 * shoulderWidth`: if near border or low conf, marks
 *   OUT_OF_FRAME (`visibility: 0`); otherwise clamps movement step to at most `0.12 * shoulderWidth`.
 * - Enforces anatomical upper-arm (`[0.20, 1.30] * shoulderWidth`) & forearm (`[0.18, 1.30] * shoulderWidth`).
 */
export function applyKinematicSkeletonLock<T extends ScoredLandmark>(params: {
  prevSmoothed: readonly T[] | null;
  rawLandmarks: readonly T[];
  videoW: number;
  videoH: number;
  canvasW: number;
  canvasH: number;
  isMirrored?: boolean;
}): { landmarks: T[]; teleportedCount: number } {
  const { prevSmoothed, rawLandmarks, videoW, videoH, canvasW, canvasH } = params;
  const isMirrored = params.isMirrored ?? true;
  const cw = canvasW > 0 ? canvasW : 640;
  const ch = canvasH > 0 ? canvasH : 480;
  const shoulderWidthPx = computeCanvasShoulderWidth(rawLandmarks, videoW, videoH, cw, ch, isMirrored);
  const out: T[] = rawLandmarks.map((lm) => ({ ...lm, z: lm.z ?? 0 }));
  let teleportedCount = 0;

  for (const idx of UPPER_LIMB_VEL_CLAMP_INDICES) {
    const rawLm = out[idx];
    if (!rawLm) continue;
    const rawPt = mapVideoPointToCoverCanvas(rawLm.x, rawLm.y, videoW, videoH, cw, ch, isMirrored);
    const inFrameNow = isJointValidInFrame({
      lm: rawLm,
      canvasPt: rawPt,
      canvasW: cw,
      canvasH: ch,
      minScore: SKELETON_JOINT_MIN_SCORE,
    });
    if (!inFrameNow) {
      out[idx] = invalidateLimbJoint(rawLm);
      continue;
    }

    const prevLm = prevSmoothed?.[idx];
    if (!prevLm) continue;
    const prevPt = mapVideoPointToCoverCanvas(prevLm.x, prevLm.y, videoW, videoH, cw, ch, isMirrored);
    const prevValid = isJointValidInFrame({
      lm: prevLm,
      canvasPt: prevPt,
      canvasW: cw,
      canvasH: ch,
      minScore: SKELETON_JOINT_MIN_SCORE,
    });
    if (!prevValid) continue;

    const jumpPx = Math.hypot(rawPt.x - prevPt.x, rawPt.y - prevPt.y);
    if (jumpPx > MAX_JOINT_TELEPORT_RATIO * shoulderWidthPx) {
      teleportedCount += 1;
      const nearBorder =
        rawLm.x < 0.05 || rawLm.x > 0.95 || rawLm.y < 0.05 || rawLm.y > 0.95;
      const conf = Math.min(rawLm.visibility ?? 1, rawLm.score ?? 1);
      if (nearBorder || conf < 0.65) {
        out[idx] = invalidateLimbJoint(rawLm);
      } else {
        const stepRatio = (MAX_JOINT_CLAMP_STEP_RATIO * shoulderWidthPx) / jumpPx;
        out[idx] = {
          ...rawLm,
          x: prevLm.x + (rawLm.x - prevLm.x) * stepRatio,
          y: prevLm.y + (rawLm.y - prevLm.y) * stepRatio,
          z: (prevLm.z ?? 0) + ((rawLm.z ?? 0) - (prevLm.z ?? 0)) * stepRatio,
        };
      }
    }
  }

  // Enforce anatomical arm chain ratio after velocity clamping
  const safety = evaluatePoseCanvasSkeletonSafety({
    landmarks: out,
    videoW,
    videoH,
    canvasW: cw,
    canvasH: ch,
    isMirrored,
  });
  for (const idx of UPPER_LIMB_VEL_CLAMP_INDICES) {
    if (!safety.validMask[idx]) {
      out[idx] = invalidateLimbJoint(out[idx]);
    }
  }

  return { landmarks: out, teleportedCount };
}

export type NormalizedWithScore = ScoredLandmark;

/**
 * Validates all 4 tethering rules for 21-point HandLandmarker overlay:
 * 1) Corresponding body wrist is valid and inside frame (`!OUT_OF_FRAME` + valid arm chain)
 * 2) Hand root landmark (`hand[0]`) is within `0.40 * shoulderWidth` of that body wrist
 * 3) Every finger point (0..20) satisfies confidence `>= 0.52` (if present), `0.03..0.97`,
 *    and canvas bounds `[6, canvasW - 6]` x `[6, canvasH - 6]`
 * 4) No individual finger bone in `HAND21_CONNECTIONS` exceeds `0.30 * shoulderWidth`
 */
export function validateTetheredHandLandmarks21(params: {
  handLandmarks21: readonly ScoredLandmark[] | undefined;
  poseLandmarks: readonly ScoredLandmark[];
  wristSide: 'left' | 'right';
  videoW: number;
  videoH: number;
  canvasW: number;
  canvasH: number;
  isMirrored?: boolean;
  armSafety?: ArmChainSafetyEvaluation;
}): { valid: boolean; canvasPoints: CanvasPoint2D[]; outOfFrame: boolean } {
  const { handLandmarks21, poseLandmarks, wristSide, videoW, videoH, canvasW, canvasH } = params;
  if (!handLandmarks21 || handLandmarks21.length < 21) {
    return { valid: false, canvasPoints: [], outOfFrame: false };
  }
  const cw = canvasW > 0 ? canvasW : 640;
  const ch = canvasH > 0 ? canvasH : 480;
  const isMirrored = params.isMirrored ?? true;
  const safety =
    params.armSafety ??
    evaluatePoseCanvasSkeletonSafety({
      landmarks: poseLandmarks,
      videoW,
      videoH,
      canvasW: cw,
      canvasH: ch,
      isMirrored,
    });

  const wristIdx = wristSide === 'left' ? POSE_INDEX.LEFT_WRIST : POSE_INDEX.RIGHT_WRIST;
  if (!safety.validMask[wristIdx]) {
    return { valid: false, canvasPoints: [], outOfFrame: true };
  }

  const bodyWristPt = safety.points[wristIdx];
  const shoulderWidthPx = safety.shoulderWidthPx;
  const canvasPoints: CanvasPoint2D[] = [];

  for (let i = 0; i < 21; i += 1) {
    const lm = handLandmarks21[i];
    if (!lm) return { valid: false, canvasPoints: [], outOfFrame: true };
    const pt = mapVideoPointToCoverCanvas(lm.x, lm.y, videoW, videoH, cw, ch, isMirrored);
    const ptValid = isJointValidInFrame({
      lm,
      canvasPt: pt,
      canvasW: cw,
      canvasH: ch,
      minScore: SKELETON_JOINT_MIN_SCORE,
      marginPx: HAND21_CANVAS_MARGIN_PX,
      edgeMarginNorm: SKELETON_EDGE_MARGIN_NORM,
      requireConfidence: false,
    });
    if (!ptValid) {
      return { valid: false, canvasPoints: [], outOfFrame: true };
    }
    canvasPoints.push(pt);
  }

  const rootDist = Math.hypot(canvasPoints[0].x - bodyWristPt.x, canvasPoints[0].y - bodyWristPt.y);
  if (rootDist > HAND21_WRIST_TETHER_MAX_RATIO * shoulderWidthPx) {
    return { valid: false, canvasPoints: [], outOfFrame: true };
  }

  for (const [a, b] of HAND21_CONNECTIONS) {
    const pa = canvasPoints[a];
    const pb = canvasPoints[b];
    if (!pa || !pb || Math.hypot(pb.x - pa.x, pb.y - pa.y) > HAND21_BONE_MAX_RATIO * shoulderWidthPx) {
      return { valid: false, canvasPoints: [], outOfFrame: true };
    }
  }

  return { valid: true, canvasPoints, outOfFrame: false };
}
