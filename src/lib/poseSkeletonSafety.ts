import { HAND21_CONNECTIONS, POSE_INDEX, computeJointAngle3D, type PoseLandmark } from '@drtechcares/pose-engine';

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
export const HAND21_WRIST_TETHER_MAX_RATIO = 0.42;
export const HAND21_WRIST_TETHER_FORESHORTENED_MAX_RATIO = 0.54;
export const HAND21_BONE_MAX_RATIO = 0.42;
export const HAND21_PALM_BONE_MAX_RATIO = 0.62;
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
  const shIdx = wristSide === 'left' ? POSE_INDEX.LEFT_SHOULDER : POSE_INDEX.RIGHT_SHOULDER;
  const poseSh = poseLandmarks[shIdx];
  const poseWr = poseLandmarks[wristIdx];
  const isForwardForeshortened = Boolean(
    poseSh && poseWr && (poseWr.z ?? 0) < (poseSh.z ?? 0) - 0.14,
  );
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
      marginPx: 2,
      edgeMarginNorm: 0.005,
      requireConfidence: false,
    });
    if (!ptValid) {
      return { valid: false, canvasPoints: [], outOfFrame: true };
    }
    canvasPoints.push(pt);
  }

  const maxTetherRatio = isForwardForeshortened
    ? HAND21_WRIST_TETHER_FORESHORTENED_MAX_RATIO
    : HAND21_WRIST_TETHER_MAX_RATIO;
  const rootDist = Math.hypot(canvasPoints[0].x - bodyWristPt.x, canvasPoints[0].y - bodyWristPt.y);
  if (rootDist > maxTetherRatio * shoulderWidthPx) {
    return { valid: false, canvasPoints: [], outOfFrame: true };
  }

  for (const [a, b] of HAND21_CONNECTIONS) {
    const pa = canvasPoints[a];
    const pb = canvasPoints[b];
    const maxBonePx =
      (a === 0 || b === 0 ? HAND21_PALM_BONE_MAX_RATIO : HAND21_BONE_MAX_RATIO) * shoulderWidthPx;
    if (!pa || !pb || Math.hypot(pb.x - pa.x, pb.y - pa.y) > maxBonePx) {
      return { valid: false, canvasPoints: [], outOfFrame: true };
    }
  }

  return { valid: true, canvasPoints, outOfFrame: false };
}

/**
 * Corrects monocular 2D elbow V-kink hallucinations when a patient extends an arm straight
 * (especially straight forward toward the camera lens where the forearm occludes the elbow).
 *
 * Uses two strictly bounded geometric invariants:
 * 1) Projective 3D Collinearity (all exercises): a straight 3D arm (`worldElbowAngle >= 158 deg`)
 *    MUST project onto a straight 2D line between shoulder and wrist.
 * 2) Forward-Extension Occlusion Repair (straight-arm modes only): when the wrist points toward
 *    the lens (`wrist.z < shoulder.z - 0.12`) at roughly shoulder height and MediaPipe drops the
 *    occluded 2D elbow below the shoulder->wrist segment (or pushes it past 68% of the segment),
 *    smoothly aligns the occluded 2D elbow onto the proportional shoulder->wrist ray without
 *    affecting genuine bent-elbow exercises or true elbow flexion.
 */
export function stabilizeStraightArmElbows<T extends ScoredLandmark>(params: {
  landmarks: readonly T[];
  worldLandmarks?: readonly PoseLandmark[] | null;
  allowForwardOcclusionRepair?: boolean;
  videoAspect?: number;
}): T[] {
  const { landmarks, worldLandmarks, allowForwardOcclusionRepair = true } = params;
  const aspect =
    params.videoAspect && Number.isFinite(params.videoAspect) && params.videoAspect > 0.4
      ? params.videoAspect
      : 1.333;
  const out = landmarks.map((lm) => ({ ...lm }));
  const lSh = out[POSE_INDEX.LEFT_SHOULDER];
  const rSh = out[POSE_INDEX.RIGHT_SHOULDER];
  if (!lSh || !rSh) return out;
  const shSpan = Math.max(0.08, Math.hypot((rSh.x - lSh.x) * aspect, rSh.y - lSh.y));

  const fixSide = (shIdx: number, elIdx: number, wrIdx: number) => {
    const sh = out[shIdx];
    const el = out[elIdx];
    const wr = out[wrIdx];
    if (!sh || !el || !wr) return;
    if ((sh.visibility ?? 1) < 0.25 || (el.visibility ?? 1) < 0.2 || (wr.visibility ?? 1) < 0.3) {
      return;
    }

    const dxSW = (wr.x - sh.x) * aspect;
    const dySW = wr.y - sh.y;
    const lenSW = Math.hypot(dxSW, dySW);
    if (!Number.isFinite(lenSW) || lenSW < 0.012) return;

    const dxSE = (el.x - sh.x) * aspect;
    const dySE = el.y - sh.y;
    const projT = (dxSE * dxSW + dySE * dySW) / (lenSW * lenSW);
    const perpDist = Math.abs(dxSE * dySW - dySE * dxSW) / lenSW;

    let worldElbowDeg: number | null = null;
    let t3d = 0.52;
    const wSh = worldLandmarks?.[shIdx];
    const wEl = worldLandmarks?.[elIdx];
    const wWr = worldLandmarks?.[wrIdx];
    if (wSh && wEl && wWr) {
      const uLen = Math.hypot(wEl.x - wSh.x, wEl.y - wSh.y, (wEl.z ?? 0) - (wSh.z ?? 0));
      const fLen = Math.hypot(wWr.x - wEl.x, wWr.y - wEl.y, (wWr.z ?? 0) - (wEl.z ?? 0));
      if (uLen >= 0.08 && fLen >= 0.08) {
        worldElbowDeg = computeJointAngle3D(wSh, wEl, wWr);
        t3d = Math.max(0.42, Math.min(0.58, uLen / (uLen + fLen)));
      }
    }

    // Rule 1: Exact projective 3D-to-2D straightness invariant (applies to every exercise)
    let blend = 0;
    if (worldElbowDeg !== null && worldElbowDeg >= 156) {
      blend = Math.max(0, Math.min(0.92, (worldElbowDeg - 154) / 18));
    }

    // Rule 2: Monocular forward-occlusion sag repair (arm extended straight toward camera)
    const wristForwardDz = (sh.z ?? 0) - (wr.z ?? 0);
    const isForwardTowardCamera =
      allowForwardOcclusionRepair &&
      wristForwardDz >= 0.11 &&
      lenSW <= shSpan * 1.18 &&
      Math.abs(wr.y - sh.y) <= shSpan * 0.58 &&
      el.y <= sh.y + shSpan * 0.55;

    if (isForwardTowardCamera && (worldElbowDeg === null || worldElbowDeg >= 120)) {
      const elbowSagBelowRay = el.y > Math.max(sh.y, wr.y) + 0.012;
      const elbowPushedPastMid = projT > 0.66 || projT < 0.28;
      const kinkRatio = perpDist / Math.max(0.03, lenSW);
      if (elbowSagBelowRay || elbowPushedPastMid || kinkRatio >= 0.14) {
        const occlBlend =
          worldElbowDeg !== null && worldElbowDeg < 138
            ? 0.55
            : 0.86;
        blend = Math.max(blend, occlBlend);
      }
    }

    if (blend <= 0.01) return;

    const targetT = worldElbowDeg !== null && worldElbowDeg >= 156 ? t3d : 0.52;
    const idealX = sh.x + (wr.x - sh.x) * targetT;
    const idealY = sh.y + (wr.y - sh.y) * targetT;
    const idealZ = (sh.z ?? 0) + ((wr.z ?? 0) - (sh.z ?? 0)) * targetT;

    out[elIdx] = {
      ...el,
      x: el.x + (idealX - el.x) * blend,
      y: el.y + (idealY - el.y) * blend,
      z: (el.z ?? 0) + (idealZ - (el.z ?? 0)) * (blend * 0.5),
    };
  };

  fixSide(POSE_INDEX.LEFT_SHOULDER, POSE_INDEX.LEFT_ELBOW, POSE_INDEX.LEFT_WRIST);
  fixSide(POSE_INDEX.RIGHT_SHOULDER, POSE_INDEX.RIGHT_ELBOW, POSE_INDEX.RIGHT_WRIST);
  return out;
}

export interface DetectedHandCandidate<T extends ScoredLandmark> {
  landmarks: readonly T[];
  worldLandmarks?: readonly T[];
  handednessCategory?: string;
  handednessScore?: number;
}

/**
 * Strictly filters MediaPipe HandLandmarker detections against anatomical scale, confidence,
 * anti-chest-phantom guards, and 1-to-1 body wrist tethering so textured shirts or background
 * clutter never spawn floating phantom hands on the chest when extending arms forward.
 */
export function filterAndAssignDetectedHands<T extends ScoredLandmark>(params: {
  candidates: readonly DetectedHandCandidate<T>[];
  poseLandmarks: readonly ScoredLandmark[];
  videoAspect?: number;
  minConfidence?: number;
}): {
  handsBySide: { left?: T[]; right?: T[] };
  worldHandsBySide: { left?: T[]; right?: T[] };
} {
  const { candidates, poseLandmarks, minConfidence = 0.34 } = params;
  const aspect =
    params.videoAspect && Number.isFinite(params.videoAspect) && params.videoAspect > 0.4
      ? params.videoAspect
      : 1.333;

  const lSh = poseLandmarks[POSE_INDEX.LEFT_SHOULDER];
  const rSh = poseLandmarks[POSE_INDEX.RIGHT_SHOULDER];
  const lWr = poseLandmarks[POSE_INDEX.LEFT_WRIST];
  const rWr = poseLandmarks[POSE_INDEX.RIGHT_WRIST];

  const lWrVis = Boolean(lWr && (lWr.visibility ?? 0) >= 0.25 && Number.isFinite(lWr.x) && Number.isFinite(lWr.y));
  const rWrVis = Boolean(rWr && (rWr.visibility ?? 0) >= 0.25 && Number.isFinite(rWr.x) && Number.isFinite(rWr.y));
  const shSpan =
    lSh && rSh
      ? Math.max(0.09, Math.hypot((rSh.x - lSh.x) * aspect, rSh.y - lSh.y))
      : 0.22;

  const minShX = lSh && rSh ? Math.min(lSh.x, rSh.x) : 0.35;
  const maxShX = lSh && rSh ? Math.max(lSh.x, rSh.x) : 0.65;
  const shSpanRawX = Math.max(0.08, maxShX - minShX);
  const midShY = lSh && rSh ? (lSh.y + rSh.y) / 2 : 0.38;

  const distIso = (a: { x: number; y: number }, b: { x: number; y: number }) =>
    Math.hypot((a.x - b.x) * aspect, a.y - b.y);

  const lForward = Boolean(lSh && lWr && (lWr.z ?? 0) < (lSh.z ?? 0) - 0.12);
  const rForward = Boolean(rSh && rWr && (rWr.z ?? 0) < (rSh.z ?? 0) - 0.12);
  const maxTetherL = shSpan * (lForward ? 0.56 : 0.44);
  const maxTetherR = shSpan * (rForward ? 0.56 : 0.44);

  const normalizeHandLandmarks = (raw: readonly T[]): T[] =>
    raw.slice(0, 21).map((lm) => ({
      ...lm,
      z: lm.z ?? 0,
      // MediaPipe tasks-vision HandLandmarker sets visibility=0/undefined because it validates at hand level;
      // normalize 0/undefined -> 1 so strict downstream joint validators do not falsely drop valid hands.
      visibility: lm.visibility === undefined || lm.visibility === 0 ? 1 : lm.visibility,
    }));

  interface ScoredMatch {
    side: 'left' | 'right';
    candIdx: number;
    dist: number;
    score: number;
    hand: T[];
    worldHand?: T[];
  }

  const validMatches: ScoredMatch[] = [];

  candidates.forEach((cand, candIdx) => {
    if (!cand.landmarks || cand.landmarks.length < 21) return;
    if (typeof cand.handednessScore === 'number' && cand.handednessScore < minConfidence) {
      return;
    }

    const hWrist = cand.landmarks[0];
    const hMidMcp = cand.landmarks[9];
    const hIdxTip = cand.landmarks[8];
    if (
      !hWrist ||
      !hMidMcp ||
      !hIdxTip ||
      !Number.isFinite(hWrist.x) ||
      !Number.isFinite(hWrist.y)
    ) {
      return;
    }

    // Anatomical palm scale sanity check (rejects tiny noise blobs & oversized shirt patterns)
    const palmLen = distIso(hWrist, hMidMcp);
    if (palmLen < 0.018 || palmLen > shSpan * 1.15) return;

    // Anti-chest-phantom check: detections sitting strictly inside the torso/shirt area require
    // an actual body wrist right next to them (<= 0.26 * shoulderWidth)
    const palmCenter = { x: (hWrist.x + hMidMcp.x) / 2, y: (hWrist.y + hMidMcp.y) / 2 };
    const insideChestBox =
      Boolean(lSh && rSh) &&
      palmCenter.x >= minShX + shSpanRawX * 0.14 &&
      palmCenter.x <= maxShX - shSpanRawX * 0.14 &&
      palmCenter.y >= midShY + 0.035 &&
      palmCenter.y <= midShY + shSpan * 1.18;

    const dLeft = lWrVis && lWr ? distIso(hWrist, lWr) : Number.POSITIVE_INFINITY;
    const dRight = rWrVis && rWr ? distIso(hWrist, rWr) : Number.POSITIVE_INFINITY;
    const effMaxL = insideChestBox ? Math.min(maxTetherL, shSpan * 0.26) : maxTetherL;
    const effMaxR = insideChestBox ? Math.min(maxTetherR, shSpan * 0.26) : maxTetherR;

    const fitsL = dLeft <= effMaxL;
    const fitsR = dRight <= effMaxR;
    if (!fitsL && !fitsR) {
      // Strictly reject untethered hands floating on chest/background far from BOTH wrists
      return;
    }

    const normalized = normalizeHandLandmarks(cand.landmarks);
    const normalizedWorld =
      cand.worldLandmarks && cand.worldLandmarks.length >= 21
        ? normalizeHandLandmarks(cand.worldLandmarks)
        : undefined;
    const score = typeof cand.handednessScore === 'number' ? cand.handednessScore : 0.85;

    if (fitsL) {
      validMatches.push({
        side: 'left',
        candIdx,
        dist: dLeft,
        score,
        hand: normalized,
        worldHand: normalizedWorld,
      });
    }
    if (fitsR) {
      validMatches.push({
        side: 'right',
        candIdx,
        dist: dRight,
        score,
        hand: normalized,
        worldHand: normalizedWorld,
      });
    }
  });

  // Greedy globally closest 1-to-1 assignment (each detected hand & each body wrist used at most once)
  validMatches.sort((a, b) => a.dist - b.dist || b.score - a.score);
  const usedCandidates = new Set<number>();
  const handsBySide: { left?: T[]; right?: T[] } = {};
  const worldHandsBySide: { left?: T[]; right?: T[] } = {};

  for (const m of validMatches) {
    if (usedCandidates.has(m.candIdx)) continue;
    if (handsBySide[m.side]) continue;
    usedCandidates.add(m.candIdx);
    handsBySide[m.side] = m.hand;
    if (m.worldHand) {
      worldHandsBySide[m.side] = m.worldHand;
    }
  }

  return { handsBySide, worldHandsBySide };
}

export interface HandRoiCropBox {
  x: number;
  y: number;
  size: number;
  videoW: number;
  videoH: number;
  wristZ: number;
}

/**
 * Computes an isotropic square ROI crop box in raw video pixels centered around a patient's
 * left or right hand so a dedicated 256x256 HandLandmarker pass sees ~6x-8x finer finger detail
 * (crucial for distant hands, closed fists, and fingers gripping objects like a mouse).
 */
export function computeHandRoiCropBox(params: {
  poseLandmarks: readonly ScoredLandmark[];
  side: 'left' | 'right';
  videoW: number;
  videoH: number;
}): HandRoiCropBox | null {
  const { poseLandmarks, side, videoW, videoH } = params;
  if (videoW < 64 || videoH < 64) return null;

  const shIdx = side === 'left' ? POSE_INDEX.LEFT_SHOULDER : POSE_INDEX.RIGHT_SHOULDER;
  const elIdx = side === 'left' ? POSE_INDEX.LEFT_ELBOW : POSE_INDEX.RIGHT_ELBOW;
  const wrIdx = side === 'left' ? POSE_INDEX.LEFT_WRIST : POSE_INDEX.RIGHT_WRIST;
  const pinkyIdx = side === 'left' ? POSE_INDEX.LEFT_PINKY : POSE_INDEX.RIGHT_PINKY;
  const indexIdx = side === 'left' ? POSE_INDEX.LEFT_INDEX : POSE_INDEX.RIGHT_INDEX;
  const thumbIdx = side === 'left' ? POSE_INDEX.LEFT_THUMB : POSE_INDEX.RIGHT_THUMB;

  const lSh = poseLandmarks[POSE_INDEX.LEFT_SHOULDER];
  const rSh = poseLandmarks[POSE_INDEX.RIGHT_SHOULDER];
  const sh = poseLandmarks[shIdx];
  const el = poseLandmarks[elIdx];
  const wr = poseLandmarks[wrIdx];
  if (!wr || (wr.visibility ?? 0) < 0.28 || !Number.isFinite(wr.x) || !Number.isFinite(wr.y)) {
    return null;
  }
  if (wr.x < 0.01 || wr.x > 0.99 || wr.y < 0.01 || wr.y > 0.99) return null;

  const shSpanPx =
    lSh && rSh && Number.isFinite(lSh.x) && Number.isFinite(rSh.x)
      ? Math.max(
          videoW * 0.10,
          Math.hypot((rSh.x - lSh.x) * videoW, (rSh.y - lSh.y) * videoH),
        )
      : videoW * 0.22;

  const wrPx = { x: wr.x * videoW, y: wr.y * videoH };
  let cx = wrPx.x;
  let cy = wrPx.y;

  // Bias crop center toward Pose palm hints (index/pinky/thumb) so all 5 fingers sit centered
  const palmPts: Array<{ x: number; y: number }> = [];
  for (const pIdx of [indexIdx, pinkyIdx, thumbIdx]) {
    const p = poseLandmarks[pIdx];
    if (p && (p.visibility ?? 0) >= 0.22 && Number.isFinite(p.x) && Number.isFinite(p.y)) {
      const px = p.x * videoW;
      const py = p.y * videoH;
      if (Math.hypot(px - wrPx.x, py - wrPx.y) <= shSpanPx * 0.55) {
        palmPts.push({ x: px, y: py });
      }
    }
  }

  if (palmPts.length > 0) {
    const avgX = palmPts.reduce((acc, p) => acc + p.x, 0) / palmPts.length;
    const avgY = palmPts.reduce((acc, p) => acc + p.y, 0) / palmPts.length;
    cx = wrPx.x * 0.35 + avgX * 0.65;
    cy = wrPx.y * 0.35 + avgY * 0.65;
  } else if (el && (el.visibility ?? 0) >= 0.25 && Number.isFinite(el.x) && Number.isFinite(el.y)) {
    const dx = wrPx.x - el.x * videoW;
    const dy = wrPx.y - el.y * videoH;
    const d = Math.hypot(dx, dy);
    if (d >= 8) {
      const shift = Math.min(shSpanPx * 0.26, d * 0.28);
      cx = wrPx.x + (dx / d) * shift;
      cy = wrPx.y + (dy / d) * shift;
    }
  }

  const isForwardTowardCamera = Boolean(sh && (wr.z ?? 0) < (sh.z ?? 0) - 0.12);
  const minSide = Math.min(videoW, videoH);
  const rawSize = shSpanPx * (isForwardTowardCamera ? 1.48 : 1.24);
  const size = Math.round(Math.max(Math.min(180, minSide), Math.min(minSide * 0.60, rawSize)));
  const x = Math.round(Math.max(0, Math.min(videoW - size, cx - size / 2)));
  const y = Math.round(Math.max(0, Math.min(videoH - size, cy - size / 2)));

  return {
    x,
    y,
    size,
    videoW,
    videoH,
    wristZ: wr.z ?? 0,
  };
}

/** Projects 21 normalized landmarks from a square ROI hand crop back into exact full-video [0..1] space. */
export function projectRoiHandToFullVideo<T extends ScoredLandmark>(
  roiLandmarks: readonly T[],
  crop: HandRoiCropBox,
): T[] {
  const zScale = crop.size / Math.max(1, crop.videoW);
  return roiLandmarks.slice(0, 21).map((lm) => ({
    ...lm,
    x: (crop.x + lm.x * crop.size) / crop.videoW,
    y: (crop.y + lm.y * crop.size) / crop.videoH,
    z: crop.wristZ + (lm.z ?? 0) * zScale,
    visibility: 1,
  }));
}
