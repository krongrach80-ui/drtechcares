import { POSE_INDEX } from '@drtechcares/pose-engine';
import type { AngleEmbeddings5D } from '@drtechcares/api-client';
import {
  classifyLivePoseZone,
  computeDualFactorBiometricScore,
  isLegacyV1HybridVector,
  l2Normalize,
  mirrorPoseZone,
  type FacePoseZone5,
  type GeometricAnthropometricRatios,
} from '@drtechcares/smart-biometric-core';

export interface OwnerBiometricTemplate {
  patientId: string;
  hn?: string;
  fullName: string;
  masterEmbedding: number[];
  angleEmbeddings?: Partial<AngleEmbeddings5D>;
  geometricRatios?: GeometricAnthropometricRatios;
}

export interface Point2DPx {
  x: number;
  y: number;
}

export interface OwnerFaceBoxMetrics {
  /** Normalized [0..1] coordinates inside the visible mirrored camera viewport (matching on-screen left/right) */
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
  centerX: number;
  centerY: number;
  width: number;
  height: number;
  /** Unmirrored raw pixel coordinates inside a `scale-x-[-1]` full-bleed canvas */
  rawPx: {
    xMin: number;
    xMax: number;
    yMin: number;
    yMax: number;
    centerX: number;
    centerY: number;
    width: number;
    height: number;
  };
}

export interface OwnerFace1to1Result {
  hasTemplate: boolean;
  matched: boolean;
  similarity: number;
  livePoseZone: FacePoseZone5;
  bestAngle: 'master' | FacePoseZone5 | 'none';
}

export type OwnerLockStatus =
  | 'OK_BOTH_HANDS'
  | 'OK_WAITING_BOTH'
  | 'NO_OWNER_FACE'
  | 'IMPOSTOR_FACE'
  | 'MULTI_PERSON'
  | 'INTRUDER_HAND';

export interface OwnerBoundPalmPoint {
  rawX: number;
  rawY: number;
  screenNormX: number;
  screenNormY: number;
}

export interface OwnerBoundHandEvaluation {
  status: OwnerLockStatus;
  leftUp: boolean;
  rightUp: boolean;
  bothUp: boolean;
  singleOnly: boolean;
  intruderHandDetected: boolean;
  poseLinkedToOwner: boolean;
  alertBannerTh?: string;
  alertBannerEn?: string;
  alertBannerCn?: string;
  boundLeftPalm?: OwnerBoundPalmPoint;
  boundRightPalm?: OwnerBoundPalmPoint;
  intruderPalms: OwnerBoundPalmPoint[];
}

export interface ObjectCoverLayout {
  scale: number;
  renderW: number;
  renderH: number;
  offsetX: number;
  offsetY: number;
  canvasW: number;
  canvasH: number;
}

export function computeObjectCoverLayout(
  videoW: number,
  videoH: number,
  canvasW: number,
  canvasH: number,
): ObjectCoverLayout {
  const cw = canvasW > 0 ? canvasW : 640;
  const ch = canvasH > 0 ? canvasH : 480;
  const vw = videoW > 0 ? videoW : cw;
  const vh = videoH > 0 ? videoH : ch;
  const scale = Math.max(cw / vw, ch / vh);
  const renderW = vw * scale;
  const renderH = vh * scale;
  const offsetX = (cw - renderW) / 2;
  const offsetY = (ch - renderH) / 2;
  return { scale, renderW, renderH, offsetX, offsetY, canvasW: cw, canvasH: ch };
}

/**
 * Projects raw camera normalized [0..1] point onto a `<canvas>` that shares
 * `object-cover scale-x-[-1]` CSS transforms with `<video>`.
 */
export function projectRawNormToCoverCanvas(
  normX: number,
  normY: number,
  layout: ObjectCoverLayout,
): Point2DPx {
  return {
    x: layout.offsetX + normX * layout.renderW,
    y: layout.offsetY + normY * layout.renderH,
  };
}

/**
 * Projects raw camera normalized [0..1] point into normalized [0..1] visible mirrored
 * screen coordinates (after `object-cover` cropping + `scale-x-[-1]` mirroring).
 */
export function projectMirroredNormToCoverViewport(
  normX: number,
  normY: number,
  layout: ObjectCoverLayout,
): Point2DPx {
  const pxX = layout.offsetX + (1 - normX) * layout.renderW;
  const pxY = layout.offsetY + normY * layout.renderH;
  return {
    x: pxX / layout.canvasW,
    y: pxY / layout.canvasH,
  };
}

/**
 * Computes tightly-fitted Owner Face bounding box in both visible mirrored viewport [0..1]
 * and `scale-x-[-1]` canvas pixel space.
 */
export function computeOwnerFaceBoxMetrics(
  landmarks: ReadonlyArray<{ x: number; y: number }>,
  videoW: number,
  videoH: number,
  canvasW: number,
  canvasH: number,
): OwnerFaceBoxMetrics | null {
  if (!landmarks || landmarks.length < 10) return null;
  const layout = computeObjectCoverLayout(videoW, videoH, canvasW, canvasH);

  let sMinX = Infinity;
  let sMaxX = -Infinity;
  let sMinY = Infinity;
  let sMaxY = -Infinity;

  let rMinX = Infinity;
  let rMaxX = -Infinity;
  let rMinY = Infinity;
  let rMaxY = -Infinity;

  for (const pt of landmarks) {
    if (!Number.isFinite(pt.x) || !Number.isFinite(pt.y)) continue;
    const sv = projectMirroredNormToCoverViewport(pt.x, pt.y, layout);
    if (sv.x < sMinX) sMinX = sv.x;
    if (sv.x > sMaxX) sMaxX = sv.x;
    if (sv.y < sMinY) sMinY = sv.y;
    if (sv.y > sMaxY) sMaxY = sv.y;

    const rp = projectRawNormToCoverCanvas(pt.x, pt.y, layout);
    if (rp.x < rMinX) rMinX = rp.x;
    if (rp.x > rMaxX) rMaxX = rp.x;
    if (rp.y < rMinY) rMinY = rp.y;
    if (rp.y > rMaxY) rMaxY = rp.y;
  }

  if (!Number.isFinite(sMinX) || sMaxX <= sMinX || sMaxY <= sMinY) return null;
  const padSX = (sMaxX - sMinX) * 0.12;
  const padSY = (sMaxY - sMinY) * 0.14;
  const xMin = sMinX - padSX;
  const xMax = sMaxX + padSX;
  const yMin = sMinY - padSY;
  const yMax = sMaxY + padSY;

  const padRX = (rMaxX - rMinX) * 0.12;
  const padRY = (rMaxY - rMinY) * 0.14;
  const rawXMin = rMinX - padRX;
  const rawXMax = rMaxX + padRX;
  const rawYMin = rMinY - padRY;
  const rawYMax = rMaxY + padRY;

  return {
    xMin,
    xMax,
    yMin,
    yMax,
    centerX: (xMin + xMax) * 0.5,
    centerY: (yMin + yMax) * 0.5,
    width: Math.max(0.03, xMax - xMin),
    height: Math.max(0.04, yMax - yMin),
    rawPx: {
      xMin: rawXMin,
      xMax: rawXMax,
      yMin: rawYMin,
      yMax: rawYMax,
      centerX: (rawXMin + rawXMax) * 0.5,
      centerY: (rawYMin + rawYMax) * 0.5,
      width: Math.max(12, rawXMax - rawXMin),
      height: Math.max(16, rawYMax - rawYMin),
    },
  };
}

/**
 * Continuous 1:1 Account Owner Face Lock verification:
 * compares live face probe strictly 1:1 against the currently logged-in account owner's
 * pose-synchronized angle template + master embedding + craniofacial ratios.
 */
export function verifyOwnerFace1to1(input: {
  probeVector: readonly number[];
  mirroredProbeVector?: readonly number[];
  legacyV1ProbeVector?: readonly number[];
  mirroredLegacyV1ProbeVector?: readonly number[];
  yawDeg?: number;
  pitchDeg?: number;
  rollDeg?: number;
  probeRatios?: GeometricAnthropometricRatios;
  ownerTemplate?: OwnerBiometricTemplate | null;
}): OwnerFace1to1Result {
  const livePoseZone = classifyLivePoseZone(input.yawDeg ?? 0, input.pitchDeg ?? 0);
  const template = input.ownerTemplate;
  if (
    !template ||
    !Array.isArray(template.masterEmbedding) ||
    template.masterEmbedding.length === 0 ||
    input.probeVector.length !== template.masterEmbedding.length
  ) {
    return {
      hasTemplate: false,
      matched: false,
      similarity: 0,
      livePoseZone,
      bestAngle: 'none',
    };
  }

  const normMaster = l2Normalize(template.masterEmbedding);
  const ownerIsLegacyV1 = isLegacyV1HybridVector(normMaster);
  const mirroredZone = mirrorPoseZone(livePoseZone);
  const activePrimaryProbe =
    ownerIsLegacyV1 &&
    input.legacyV1ProbeVector &&
    input.legacyV1ProbeVector.length === template.masterEmbedding.length
      ? input.legacyV1ProbeVector
      : input.probeVector;
  const activeMirroredProbe =
    ownerIsLegacyV1 &&
    input.mirroredLegacyV1ProbeVector &&
    input.mirroredLegacyV1ProbeVector.length === template.masterEmbedding.length
      ? input.mirroredLegacyV1ProbeVector
      : !ownerIsLegacyV1
        ? input.mirroredProbeVector
        : input.legacyV1ProbeVector
          ? undefined
          : input.mirroredProbeVector;

  const variants: { zone: FacePoseZone5; vec: number[] }[] = [
    { zone: livePoseZone, vec: l2Normalize(activePrimaryProbe) },
  ];
  if (
    activeMirroredProbe &&
    activeMirroredProbe.length === template.masterEmbedding.length
  ) {
    variants.push({ zone: mirroredZone, vec: l2Normalize(activeMirroredProbe) });
  }

  let bestScore = -1;
  let bestAngle: 'master' | FacePoseZone5 = 'master';
  let bestMeanCentered = 0;
  let bestWhitened = 0;
  let bestGeomPenalty = 0;

  for (const variant of variants) {
    const rawAngle = template.angleEmbeddings?.[variant.zone];
    const hasAngle = Array.isArray(rawAngle) && rawAngle.length === normMaster.length;
    const sameZoneVec = hasAngle ? l2Normalize(rawAngle) : normMaster;

    const sameZoneDual = computeDualFactorBiometricScore(variant.vec, sameZoneVec, {
      probeRatios: input.probeRatios ?? null,
      candidateRatios: template.geometricRatios ?? null,
    });
    const masterDual = computeDualFactorBiometricScore(variant.vec, normMaster, {
      probeRatios: input.probeRatios ?? null,
      candidateRatios: template.geometricRatios ?? null,
    });

    const dualScore = hasAngle
      ? 0.55 * sameZoneDual.dualFactorScore + 0.45 * masterDual.dualFactorScore
      : masterDual.dualFactorScore;

    if (dualScore > bestScore) {
      bestScore = dualScore;
      const dom =
        sameZoneDual.dualFactorScore >= masterDual.dualFactorScore ? sameZoneDual : masterDual;
      bestAngle =
        hasAngle && sameZoneDual.dualFactorScore >= masterDual.dualFactorScore - 0.04
          ? variant.zone
          : 'master';
      bestMeanCentered = dom.meanCenteredCos;
      bestWhitened = Math.max(sameZoneDual.whitenedCos, masterDual.whitenedCos);
      bestGeomPenalty = Math.max(sameZoneDual.geometricPenalty, masterDual.geometricPenalty);
    }
  }

  const isAngled =
    livePoseZone !== 'center' ||
    Math.abs(input.yawDeg ?? 0) > 8 ||
    Math.abs(input.pitchDeg ?? 0) > 6 ||
    Math.abs(input.rollDeg ?? 0) > 10;

  const ownerThreshold = ownerIsLegacyV1
    ? isAngled
      ? 0.74
      : 0.78
    : isAngled
      ? 0.70
      : 0.75;
  const matched =
    bestScore >= ownerThreshold &&
    bestMeanCentered >= (ownerIsLegacyV1 ? (isAngled ? 0.54 : 0.62) : isAngled ? 0.50 : 0.58) &&
    bestWhitened >= (ownerIsLegacyV1 ? (isAngled ? 0.34 : 0.40) : isAngled ? 0.30 : 0.35) &&
    bestGeomPenalty <= 0.16;

  return {
    hasTemplate: true,
    matched,
    similarity: Math.max(0, bestScore),
    livePoseZone,
    bestAngle,
  };
}

/**
 * Checks if a MediaPipe 33-pt Pose skeleton's head keypoints fall inside the owner's face box
 * (Condition 1: Skeleton & Shoulder linkage to Owner Face Box).
 */
export function isPoseSkeletonLinkedToOwnerFace(
  pose: ReadonlyArray<{ x: number; y: number; visibility?: number }> | null | undefined,
  ownerFaceBox: OwnerFaceBoxMetrics | null,
  layout: ObjectCoverLayout,
): boolean {
  if (!pose || pose.length < 13 || !ownerFaceBox) return false;
  const padX = Math.max(0.04, ownerFaceBox.width * 0.42);
  const padY = Math.max(0.04, ownerFaceBox.height * 0.48);
  const minX = ownerFaceBox.xMin - padX;
  const maxX = ownerFaceBox.xMax + padX;
  const minY = ownerFaceBox.yMin - padY;
  const maxY = ownerFaceBox.yMax + padY;

  let visibleHeadCount = 0;
  for (const idx of [0, 2, 5]) {
    const pt = pose[idx];
    if (!pt || (pt.visibility ?? 1) < 0.25) continue;
    visibleHeadCount += 1;
    const sv = projectMirroredNormToCoverViewport(pt.x, pt.y, layout);
    if (sv.x >= minX && sv.x <= maxX && sv.y >= minY && sv.y <= maxY) {
      return true;
    }
  }
  return visibleHeadCount === 0;
}

/**
 * Owner-Bound Kinematic Hand Filter:
 * - Blocks immediately when owner face is missing, multi-person (`faceCount > 1`),
 *   impostor face (`!ownerFaceVerified`), or `hands.length > 2`.
 * - Accepts ONLY hands inside the Owner Anatomical Envelope anchored to `ownerFaceBox`:
 *   - Left hand (viewer left): `centerX - 2.8 * faceW <= handX <= centerX - 0.15 * faceW`
 *   - Right hand (viewer right): `centerX + 0.15 * faceW <= handX <= centerX + 2.8 * faceW`
 *   - Elevation height: `topY - 0.8 * faceH <= handY <= bottomY + 1.2 * faceH`
 * - Rejects external hands poking from far edges, behind, or from unlinked skeletons.
 */
export function evaluateOwnerBoundKinematicHands(input: {
  ownerFaceVerified: boolean;
  ownerFaceBox: OwnerFaceBoxMetrics | null;
  faceCount: number;
  ownerName: string;
  hands: ReadonlyArray<ReadonlyArray<{ x: number; y: number }>>;
  pose?: ReadonlyArray<{ x: number; y: number; visibility?: number }> | null;
  poseCount?: number;
  videoW: number;
  videoH: number;
  canvasW: number;
  canvasH: number;
}): OwnerBoundHandEvaluation {
  const layout = computeObjectCoverLayout(
    input.videoW,
    input.videoH,
    input.canvasW,
    input.canvasH,
  );

  if (input.faceCount > 1 || (typeof input.poseCount === 'number' && input.poseCount > 1)) {
    return {
      status: 'MULTI_PERSON',
      leftUp: false,
      rightUp: false,
      bothUp: false,
      singleOnly: false,
      intruderHandDetected: false,
      poseLinkedToOwner: false,
      alertBannerTh: '⚠️ พบผู้อื่นอยู่ในเฟรม กรุณาให้เจ้าของบัญชีอยู่หน้ากล้องเพียงคนเดียว',
      alertBannerEn: '⚠️ Multiple people detected — account owner must be alone in front of the camera',
      alertBannerCn: '⚠️ 画面中检测到多名人员，请仅由账户本人面向摄像头',
      intruderPalms: [],
    };
  }

  if (input.faceCount === 0 || !input.ownerFaceBox) {
    return {
      status: 'NO_OWNER_FACE',
      leftUp: false,
      rightUp: false,
      bothUp: false,
      singleOnly: false,
      intruderHandDetected: false,
      poseLinkedToOwner: false,
      alertBannerTh: '⚠️ ไม่พบใบหน้าเจ้าของบัญชี กรุณามองกล้องขณะยกมือ',
      alertBannerEn: '⚠️ Owner face not found — please face the camera while raising hands',
      alertBannerCn: '⚠️ 未检测到本人面部，请在举手时直视摄像头',
      intruderPalms: [],
    };
  }

  if (!input.ownerFaceVerified) {
    return {
      status: 'IMPOSTOR_FACE',
      leftUp: false,
      rightUp: false,
      bothUp: false,
      singleOnly: false,
      intruderHandDetected: false,
      poseLinkedToOwner: false,
      alertBannerTh: `⛔ คุณไม่ใช่เจ้าของบัญชี (${input.ownerName}) ไม่สามารถทำรายการแทนกันได้`,
      alertBannerEn: `⛔ Identity mismatch — you are not the account owner (${input.ownerName})`,
      alertBannerCn: `⛔ 身份不匹配 — 您不是账户本人 (${input.ownerName})，无法代为操作`,
      intruderPalms: [],
    };
  }

  const box = input.ownerFaceBox;
  const poseLinkedToOwner = input.pose
    ? isPoseSkeletonLinkedToOwnerFace(input.pose, box, layout)
    : true;

  if (input.hands.length > 2) {
    const intruderPalms: OwnerBoundPalmPoint[] = input.hands.map((h) => {
      const anchor = h[9] ?? h[0] ?? { x: 0.5, y: 0.5 };
      const rp = projectRawNormToCoverCanvas(anchor.x, anchor.y, layout);
      const sv = projectMirroredNormToCoverViewport(anchor.x, anchor.y, layout);
      return { rawX: rp.x, rawY: rp.y, screenNormX: sv.x, screenNormY: sv.y };
    });
    return {
      status: 'INTRUDER_HAND',
      leftUp: false,
      rightUp: false,
      bothUp: false,
      singleOnly: false,
      intruderHandDetected: true,
      poseLinkedToOwner,
      alertBannerTh: '⚠️ ตรวจพบมือของบุคคลอื่น กรุณาให้เจ้าของบัญชียกมือด้วยตนเองเท่านั้น',
      alertBannerEn: '⚠️ Another person’s hand detected — only the account owner may raise hands',
      alertBannerCn: '⚠️ 检测到他人手部，请仅由账户本人亲自举起双手',
      intruderPalms,
    };
  }

  const leftMinX = box.centerX - 2.8 * box.width;
  const leftMaxX = box.centerX - 0.15 * box.width;
  const rightMinX = box.centerX + 0.15 * box.width;
  const rightMaxX = box.centerX + 2.8 * box.width;
  const minHandY = box.yMin - 0.8 * box.height;
  const maxHandY = box.yMax + 1.2 * box.height;

  let boundLeftPalm: OwnerBoundPalmPoint | undefined;
  let boundRightPalm: OwnerBoundPalmPoint | undefined;
  let leftCountOnSide = 0;
  let rightCountOnSide = 0;
  let intruderHandDetected = false;
  const intruderPalms: OwnerBoundPalmPoint[] = [];

  for (const h of input.hands) {
    if (h.length < 21) continue;
    const anchor = h[9] ?? h[0];
    if (!anchor) continue;
    const sv = projectMirroredNormToCoverViewport(anchor.x, anchor.y, layout);
    const rp = projectRawNormToCoverCanvas(anchor.x, anchor.y, layout);
    const pt: OwnerBoundPalmPoint = {
      rawX: rp.x,
      rawY: rp.y,
      screenNormX: sv.x,
      screenNormY: sv.y,
    };

    const horizontalInOwnerSpan =
      sv.x >= leftMinX && sv.x <= rightMaxX;
    const verticalInRaiseWindow =
      sv.y >= minHandY && sv.y <= maxHandY;

    if (!horizontalInOwnerSpan) {
      // Hand pokes in horizontally outside owner's 2.8 * faceW wingspan
      intruderHandDetected = true;
      intruderPalms.push(pt);
      continue;
    }

    if (!verticalInRaiseWindow) {
      // If raised way above head (>0.8 * faceH above brow), classify as external overhead hand
      if (sv.y < minHandY) {
        intruderHandDetected = true;
        intruderPalms.push(pt);
      }
      continue;
    }

    const inLeftZone = sv.x >= leftMinX && sv.x <= leftMaxX;
    const inRightZone = sv.x >= rightMinX && sv.x <= rightMaxX;

    if (inLeftZone) {
      leftCountOnSide += 1;
      if (leftCountOnSide > 1) {
        intruderHandDetected = true;
        intruderPalms.push(pt);
      } else {
        boundLeftPalm = pt;
      }
    } else if (inRightZone) {
      rightCountOnSide += 1;
      if (rightCountOnSide > 1) {
        intruderHandDetected = true;
        intruderPalms.push(pt);
      } else {
        boundRightPalm = pt;
      }
    } else {
      // Hand overlaps directly across face centerline [centerX - 0.15W .. centerX + 0.15W]
      // Wait gently unless 2 hands exist on one side or >2 hands; skip ambiguous overlap
    }
  }

  // Fallback Condition 1: owner-linked pose skeleton wrist check when one hand blurs briefly
  if (
    !intruderHandDetected &&
    poseLinkedToOwner &&
    input.pose &&
    input.pose.length >= 21 &&
    (!boundLeftPalm || !boundRightPalm)
  ) {
    const lSh = input.pose[POSE_INDEX.LEFT_SHOULDER];
    const rSh = input.pose[POSE_INDEX.RIGHT_SHOULDER];
    const lWr = input.pose[POSE_INDEX.LEFT_WRIST];
    const rWr = input.pose[POSE_INDEX.RIGHT_WRIST];
    const lShY = lSh && (lSh.visibility ?? 1) >= 0.3 ? lSh.y : box.yMax;
    const rShY = rSh && (rSh.visibility ?? 1) >= 0.3 ? rSh.y : box.yMax;

    for (const [wr, shY] of [
      [lWr, lShY],
      [rWr, rShY],
    ] as const) {
      if (!wr || (wr.visibility ?? 0) < 0.45 || wr.y > Math.min(0.84, shY + 0.08)) continue;
      const sv = projectMirroredNormToCoverViewport(wr.x, wr.y, layout);
      const rp = projectRawNormToCoverCanvas(wr.x, wr.y, layout);
      if (sv.y < minHandY || sv.y > maxHandY) continue;
      const pt: OwnerBoundPalmPoint = {
        rawX: rp.x,
        rawY: rp.y,
        screenNormX: sv.x,
        screenNormY: sv.y,
      };
      if (sv.x >= leftMinX && sv.x <= leftMaxX && !boundLeftPalm) {
        boundLeftPalm = pt;
      } else if (sv.x >= rightMinX && sv.x <= rightMaxX && !boundRightPalm) {
        boundRightPalm = pt;
      }
    }
  }

  if (intruderHandDetected) {
    return {
      status: 'INTRUDER_HAND',
      leftUp: false,
      rightUp: false,
      bothUp: false,
      singleOnly: false,
      intruderHandDetected: true,
      poseLinkedToOwner,
      alertBannerTh: '⚠️ ตรวจพบมือของบุคคลอื่น กรุณาให้เจ้าของบัญชียกมือด้วยตนเองเท่านั้น',
      alertBannerEn: '⚠️ Another person’s hand detected — only the account owner may raise hands',
      alertBannerCn: '⚠️ 检测到他人手部，请仅由账户本人亲自举起双手',
      intruderPalms,
    };
  }

  const leftUp = Boolean(boundLeftPalm);
  const rightUp = Boolean(boundRightPalm);
  const bothUp = leftUp && rightUp;
  const singleOnly = (leftUp || rightUp) && !bothUp;

  return {
    status: bothUp ? 'OK_BOTH_HANDS' : 'OK_WAITING_BOTH',
    leftUp,
    rightUp,
    bothUp,
    singleOnly,
    intruderHandDetected: false,
    poseLinkedToOwner,
    boundLeftPalm,
    boundRightPalm,
    intruderPalms: [],
  };
}

/**
 * Draws upright, non-mirrored text + background pill onto a CSS `scale-x-[-1]` mirrored `<canvas>`.
 */
export function drawUprightLabelOnMirroredCanvas(
  ctx: CanvasRenderingContext2D,
  rawX: number,
  rawY: number,
  text: string,
  bgColor: string,
  textColor = '#FFFFFF',
  align: 'left' | 'center' = 'left',
  font = '800 12px "Sarabun", sans-serif',
): void {
  ctx.save();
  ctx.font = font;
  const padX = 10;
  const tw = ctx.measureText(text).width;
  const pillW = tw + padX * 2;
  const pillH = 22;

  ctx.translate(rawX, rawY);
  // Cancel outer CSS `scale-x-[-1]` so glyphs appear left-to-right readable!
  ctx.scale(-1, 1);

  const rx = align === 'center' ? -pillW / 2 : 0;
  const ry = -pillH / 2;

  ctx.fillStyle = bgColor;
  ctx.beginPath();
  ctx.roundRect(rx, ry, pillW, pillH, 10);
  ctx.fill();

  ctx.fillStyle = textColor;
  ctx.textAlign = align === 'center' ? 'center' : 'left';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, align === 'center' ? 0 : rx + padX, 0);
  ctx.restore();
}

/**
 * Renders the Owner Face Bounding Box + Label + Owner Biometric Tether HUD + Intruder warnings
 * directly onto a full-bleed `scale-x-[-1]` canvas matching `<video class="object-cover scale-x-[-1]" />`.
 */
export function drawOwnerBiometricOverlayOnCanvas(
  ctx: CanvasRenderingContext2D,
  cw: number,
  ch: number,
  params: {
    ownerName: string;
    faceBox: OwnerFaceBoxMetrics | null;
    extraFaceBoxes?: OwnerFaceBoxMetrics[];
    ownerFaceVerified: boolean;
    faceCount: number;
    handEval?: OwnerBoundHandEvaluation | null;
  },
): void {
  const { ownerName, faceBox, extraFaceBoxes = [], ownerFaceVerified, faceCount, handEval } = params;

  const drawFaceBox = (
    box: OwnerFaceBoxMetrics,
    color: string,
    labelText: string,
    dashed = false,
  ) => {
    const { xMin, yMin, width, height, xMax } = box.rawPx;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.shadowColor = color;
    ctx.shadowBlur = 10;
    if (dashed) ctx.setLineDash([7, 5]);
    ctx.beginPath();
    ctx.roundRect(xMin, yMin, width, height, 14);
    ctx.stroke();
    ctx.restore();

    // Put upright label above right edge (since scale-x-[-1] flips xMax -> visual left!)
    const labelAnchorX = xMax;
    const labelAnchorY = Math.max(16, yMin - 14);
    drawUprightLabelOnMirroredCanvas(ctx, labelAnchorX, labelAnchorY, labelText, color, '#FFFFFF', 'left');
  };

  if (faceBox) {
    if (faceCount > 1) {
      drawFaceBox(faceBox, '#F59E0B', '⚠️ พบมากกว่า 1 คนในเฟรม', true);
    } else if (ownerFaceVerified) {
      drawFaceBox(faceBox, '#10B981', `✓ เจ้าของบัญชี: ${ownerName}`, false);
    } else {
      drawFaceBox(faceBox, '#EF4444', `⛔ ไม่ใช่เจ้าของบัญชี (${ownerName})`, false);
    }
  }

  for (const extra of extraFaceBoxes) {
    drawFaceBox(extra, '#EF4444', '⚠️ บุคคลอื่น', true);
  }

  if (handEval) {
    // Glowing Emerald Dashed Biometric Tether connecting Owner Face Box to bound Left & Right hands
    if (ownerFaceVerified && faceBox && !handEval.intruderHandDetected) {
      const faceLAnchorX = faceBox.rawPx.xMax - faceBox.rawPx.width * 0.15;
      const faceRAnchorX = faceBox.rawPx.xMin + faceBox.rawPx.width * 0.15;
      const faceAnchorY = faceBox.rawPx.centerY;

      const drawTetherAndPalm = (
        palm: OwnerBoundPalmPoint | undefined,
        anchorX: number,
        anchorY: number,
      ) => {
        if (!palm) return;
        ctx.save();
        ctx.strokeStyle = '#10B981';
        ctx.lineWidth = 2.5;
        ctx.shadowColor = '#10B981';
        ctx.shadowBlur = 10;
        ctx.setLineDash([7, 5]);
        ctx.beginPath();
        ctx.moveTo(anchorX, anchorY);
        ctx.lineTo(palm.rawX, palm.rawY);
        ctx.stroke();

        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.arc(palm.rawX, palm.rawY, 16, 0, Math.PI * 2);
        ctx.fillStyle = handEval.bothUp ? 'rgba(16,185,129,0.34)' : 'rgba(16,185,129,0.20)';
        ctx.fill();
        ctx.lineWidth = 3;
        ctx.strokeStyle = '#10B981';
        ctx.stroke();
        ctx.restore();
      };

      // Note: screen visual Left corresponds to higher rawX when camera is CSS scale-x-[-1]
      drawTetherAndPalm(handEval.boundLeftPalm, faceLAnchorX, faceAnchorY);
      drawTetherAndPalm(handEval.boundRightPalm, faceRAnchorX, faceAnchorY);
    }

    // Intruder hand markers (#EF4444)
    for (const bad of handEval.intruderPalms) {
      ctx.save();
      ctx.strokeStyle = '#EF4444';
      ctx.lineWidth = 3.5;
      ctx.shadowColor = '#EF4444';
      ctx.shadowBlur = 10;
      ctx.beginPath();
      ctx.arc(bad.rawX, bad.rawY, 18, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(239,68,68,0.28)';
      ctx.fill();
      ctx.stroke();
      ctx.restore();
      drawUprightLabelOnMirroredCanvas(
        ctx,
        bad.rawX,
        Math.max(16, bad.rawY - 24),
        '⛔ มือบุคคลอื่น',
        '#EF4444',
        '#FFFFFF',
        'center',
      );
    }
  }

  void cw;
  void ch;
}

