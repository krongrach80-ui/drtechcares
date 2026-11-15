import React, { useEffect, useMemo, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { FilesetResolver, PoseLandmarker, type NormalizedLandmark } from '@mediapipe/tasks-vision';
import { selectClinicalCameraDeviceId } from '@drtechcares/face-client';
import { POSE_INDEX, computeJointAngle3D } from '@drtechcares/pose-engine';
import {
  Activity,
  AlertOctagon,
  AlertTriangle,
  Camera,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Gamepad2,
  HeartPulse,
  Pause,
  Play,
  PlusCircle,
  RotateCcw,
  SlidersHorizontal,
  Sparkles,
  Timer,
  UserCheck,
  Wifi,
  WifiOff,
} from 'lucide-react';
import { FaceScanOverlay } from './FaceScanOverlay';
import { RomGauge } from './RomGauge';
import { SkeletonCanvas } from './SkeletonCanvas';
import { VoiceCoachBanner } from './VoiceCoachBanner';
import {
  getPrescribedLimbPromptTh,
  useKioskSocket,
  type DemoPresetId,
  type ExercisePrescriptionOption,
  type PatientSummary,
} from '../hooks/useKioskSocket';

interface SmartRehabKioskUIProps {
  patient?: PatientSummary | null;
  prescribedExercises?: ExercisePrescriptionOption[];
  onExerciseCompleted?: (exercise: ExercisePrescriptionOption) => void;
  onAllExercisesCompleted?: () => void;
  onEmergencyStop?: (info: { reason: string; patientId?: string; exerciseNameTh?: string }) => void;
  onOpenStaffPortal?: () => void;
  onOpenLegacyKiosk?: () => void;
}

const DEMO_PRESETS: { id: DemoPresetId; label: string; shortLabel: string }[] = [
  { id: 'FACE_SCAN', label: 'Face Scan Login', shortLabel: '1. Face Scan' },
  { id: 'ACTIVE_REP', label: 'Active Exercise', shortLabel: '2. Active' },
  { id: 'TRUNK_LEAN', label: 'Cheating / Trunk Lean', shortLabel: '3. Trunk Lean' },
  { id: 'WRONG_LIMB', label: 'Wrong Arm (ยกผิดข้าง)', shortLabel: '3b. Wrong Arm' },
  { id: 'HOLD_PHASE', label: 'Hold Phase', shortLabel: '4. Hold' },
  { id: 'REP_COMPLETED', label: 'Rep Completed', shortLabel: '5. Rep +' },
  { id: 'SET_FINISHED', label: 'Set Finished', shortLabel: '6. Finished' },
  { id: 'MULTI_PERSON', label: 'Multi-Person Alert (2+ คน)', shortLabel: '7. Multi-Person' },
  { id: 'IMPOSTOR', label: 'Impostor Blocked (ผิดคน)', shortLabel: '8. Impostor' },
];

export interface LiveJointMeasurement {
  romDeg: number | null;
  trunkTiltDeg: number;
  wrongSideRaised: boolean;
  oppositeRomDeg: number | null;
  wrongSideBannerTh: string | null;
}

export function measureLiveJointAngleAndTilt(
  landmarks: readonly NormalizedLandmark[],
  targetJoint: ExercisePrescriptionOption['target_joint'],
): LiveJointMeasurement {
  const UPPER_MIN_VIS = 0.02;
  const LOWER_MIN_VIS = 0.2;
  const lSh = landmarks[POSE_INDEX.LEFT_SHOULDER];
  const rSh = landmarks[POSE_INDEX.RIGHT_SHOULDER];
  const lHip = landmarks[POSE_INDEX.LEFT_HIP];
  const rHip = landmarks[POSE_INDEX.RIGHT_HIP];

  let trunkTiltDeg = 2.0;
  if (
    lSh &&
    rSh &&
    lHip &&
    rHip &&
    (lHip.visibility ?? 0) >= LOWER_MIN_VIS &&
    (rHip.visibility ?? 0) >= LOWER_MIN_VIS
  ) {
    const midShX = (lSh.x + rSh.x) / 2;
    const midShY = (lSh.y + rSh.y) / 2;
    const midHipX = (lHip.x + rHip.x) / 2;
    const midHipY = (lHip.y + rHip.y) / 2;
    trunkTiltDeg = Math.abs(
      (Math.atan2(midShX - midHipX, Math.max(0.05, midHipY - midShY)) * 180) / Math.PI,
    );
  } else if (
    lSh &&
    rSh &&
    (lSh.visibility ?? 0) >= UPPER_MIN_VIS &&
    (rSh.visibility ?? 0) >= UPPER_MIN_VIS
  ) {
    trunkTiltDeg =
      Math.abs((Math.atan2(rSh.y - lSh.y, Math.max(0.08, Math.abs(lSh.x - rSh.x))) * 180) / Math.PI) *
      0.55;
  }

  const calibrateShoulderElevationDeg = (rawDeg: number): number =>
    rawDeg > 85 ? Math.min(180, 85 + (rawDeg - 85) * 1.25) : rawDeg;

  const computeArmElevationDeg = (side: 'left' | 'right'): number | null => {
    const sh = landmarks[side === 'left' ? POSE_INDEX.LEFT_SHOULDER : POSE_INDEX.RIGHT_SHOULDER];
    const elbow = landmarks[side === 'left' ? POSE_INDEX.LEFT_ELBOW : POSE_INDEX.RIGHT_ELBOW];
    const wrist = landmarks[side === 'left' ? POSE_INDEX.LEFT_WRIST : POSE_INDEX.RIGHT_WRIST];
    const hip = landmarks[side === 'left' ? POSE_INDEX.LEFT_HIP : POSE_INDEX.RIGHT_HIP];
    if (!sh || (sh.visibility ?? 0) < UPPER_MIN_VIS) return null;

    const shScaled = { x: sh.x, y: sh.y, z: (sh.z ?? 0) * 0.35 };
    const refHip =
      hip && (hip.visibility ?? 0) >= LOWER_MIN_VIS && hip.y <= 0.98
        ? { x: hip.x, y: hip.y, z: (hip.z ?? 0) * 0.35 }
        : { x: sh.x, y: sh.y + 0.35, z: (sh.z ?? 0) * 0.35 };

    const candidates: number[] = [];
    for (const pt of [wrist, elbow]) {
      if (!pt || (pt.visibility ?? 0) < UPPER_MIN_VIS) continue;
      // Scale horizontal span for 4:3 / 16:9 webcam aspect ratio so abducted arms read true angle
      const dx = (pt.x - sh.x) * 1.22;
      const dy = pt.y - sh.y;
      const len = Math.hypot(dx, dy);
      if (len >= 0.03) {
        const cosDown = Math.max(-1, Math.min(1, dy / len));
        const elev2D = (Math.acos(cosDown) * 180) / Math.PI;
        const ptScaled = { x: pt.x, y: pt.y, z: (pt.z ?? 0) * 0.35 };
        const angle3D = computeJointAngle3D(refHip, shScaled, ptScaled);
        candidates.push(calibrateShoulderElevationDeg(Math.max(elev2D, angle3D)));
      }
    }

    // Measure forearm elevation vector (elbow -> wrist) when raised for close-up webcam perspective
    if (
      elbow &&
      wrist &&
      (elbow.visibility ?? 0) >= UPPER_MIN_VIS &&
      (wrist.visibility ?? 0) >= UPPER_MIN_VIS &&
      wrist.y < elbow.y
    ) {
      const fDx = (wrist.x - elbow.x) * 1.22;
      const fDy = wrist.y - elbow.y;
      const fLen = Math.hypot(fDx, fDy);
      if (fLen >= 0.03) {
        const fCosDown = Math.max(-1, Math.min(1, fDy / fLen));
        const forearmElev2D = (Math.acos(fCosDown) * 180) / Math.PI;
        candidates.push(calibrateShoulderElevationDeg(forearmElev2D));
      }
    }

    return candidates.length > 0 ? Math.max(...candidates) : null;
  };

  let romDeg: number | null = null;
  let oppositeRomDeg: number | null = null;
  let wrongSideRaised = false;
  let wrongSideBannerTh: string | null = null;
  if (targetJoint === 'left_shoulder' || targetJoint === 'right_shoulder') {
    const primarySide: 'left' | 'right' = targetJoint === 'left_shoulder' ? 'left' : 'right';
    const oppositeSide: 'left' | 'right' = primarySide === 'left' ? 'right' : 'left';
    const primaryDeg = computeArmElevationDeg(primarySide);
    const oppositeDeg = computeArmElevationDeg(oppositeSide);
    wrongSideRaised =
      oppositeDeg !== null &&
      oppositeDeg >= 45 &&
      (primaryDeg === null || (primaryDeg < 65 && oppositeDeg > primaryDeg + 18));
    // Strict side enforcement: measure ONLY the prescribed arm; hold last reading on transient drop
    romDeg =
      primaryDeg !== null ? Number(primaryDeg.toFixed(1)) : wrongSideRaised ? 0 : null;
    oppositeRomDeg = oppositeDeg;
    if (wrongSideRaised) {
      wrongSideBannerTh = getPrescribedLimbPromptTh(targetJoint).bannerAlertTh;
    }
  } else if (targetJoint === 'left_knee' || targetJoint === 'right_knee') {
    const isLeft = targetJoint === 'left_knee';
    const hip = landmarks[isLeft ? POSE_INDEX.LEFT_HIP : POSE_INDEX.RIGHT_HIP];
    const knee = landmarks[isLeft ? POSE_INDEX.LEFT_KNEE : POSE_INDEX.RIGHT_KNEE];
    const ankle = landmarks[isLeft ? POSE_INDEX.LEFT_ANKLE : POSE_INDEX.RIGHT_ANKLE];
    if (
      hip &&
      knee &&
      ankle &&
      (hip.visibility ?? 0) >= LOWER_MIN_VIS &&
      (knee.visibility ?? 0) >= LOWER_MIN_VIS &&
      (ankle.visibility ?? 0) >= LOWER_MIN_VIS
    ) {
      romDeg = computeJointAngle3D(hip, knee, ankle);
    } else if (
      lSh &&
      rSh &&
      lHip &&
      rHip &&
      (lSh.visibility ?? 0) >= UPPER_MIN_VIS &&
      (lHip.visibility ?? 0) >= LOWER_MIN_VIS
    ) {
      // Fallback for close webcam framing: approximate squat/knee flexion from hip-shoulder vertical span
      const midShY = (lSh.y + rSh.y) / 2;
      const midHipY = (lHip.y + rHip.y) / 2;
      const dropNorm = Math.max(0, Math.min(1, (midHipY - midShY - 0.18) / 0.25));
      romDeg = Number((105 + dropNorm * 65).toFixed(1));
    }
  } else if (targetJoint === 'spine') {
    romDeg = Number(Math.max(0, 15 - trunkTiltDeg).toFixed(1));
  }

  return {
    romDeg,
    trunkTiltDeg: Number(trunkTiltDeg.toFixed(1)),
    wrongSideRaised,
    oppositeRomDeg,
    wrongSideBannerTh,
  };
}

/**
 * Smooths noisy joint ROM readings with a compact anti-jitter deadband and responsive slew rate
 * so gauge arcs and numeric degree readouts track real body movement without artificial lag.
 */
export function stepSmoothRomAngle(
  currentRom: number | null,
  targetRom: number,
  dtMs: number,
): number {
  const safeTarget = Math.max(0, Math.min(180, targetRom));
  if (currentRom === null || !Number.isFinite(currentRom)) {
    return safeTarget;
  }
  const safeDt = Math.min(100, Math.max(1, dtMs));
  const rawDelta = safeTarget - currentRom;
  // Deadband of 0.8 deg suppresses stationary jitter while staying tight to real motion
  if (Math.abs(rawDelta) < 0.8) {
    return currentRom;
  }
  // Responsive exponential filter (tau = 50ms) + natural biomechanical slew cap (260/280 deg/s)
  const alpha = 1 - Math.exp(-safeDt / 50);
  const emaStep = rawDelta * alpha;
  const maxUpStep = (260 * safeDt) / 1000;
  const maxDownStep = (280 * safeDt) / 1000;
  const step =
    emaStep > 0 ? Math.min(emaStep, maxUpStep) : Math.max(emaStep, -maxDownStep);
  return Math.max(0, Math.min(180, currentRom + step));
}

/**
 * Interpolates 33 pose landmarks at 120Hz between camera inference ticks so canvas lines follow
 * live movement smoothly without visible frame-stepping.
 */
export function interpolatePoseLandmarks(
  prev: NormalizedLandmark[] | null,
  next: readonly NormalizedLandmark[],
  dtMs: number,
): NormalizedLandmark[] {
  if (!prev || prev.length !== next.length) {
    return next.map((lm) => ({ ...lm }));
  }
  const safeDt = Math.min(100, Math.max(1, dtMs));
  const alpha = Math.min(1, 1 - Math.exp(-safeDt / 20));
  return next.map((target, idx) => {
    const cur = prev[idx];
    if (!cur) return { ...target };
    const vis =
      target.visibility !== undefined
        ? (cur.visibility ?? target.visibility) +
          (target.visibility - (cur.visibility ?? target.visibility)) * alpha
        : cur.visibility;
    return {
      x: cur.x + (target.x - cur.x) * alpha,
      y: cur.y + (target.y - cur.y) * alpha,
      z: cur.z + (target.z - cur.z) * alpha,
      visibility: vis,
    };
  });
}

const LIVE_BODY_BONELINE_PAIRS: readonly [number, number][] = [
  [POSE_INDEX.LEFT_SHOULDER, POSE_INDEX.RIGHT_SHOULDER],
  [POSE_INDEX.LEFT_SHOULDER, POSE_INDEX.LEFT_ELBOW],
  [POSE_INDEX.LEFT_ELBOW, POSE_INDEX.LEFT_WRIST],
  [POSE_INDEX.RIGHT_SHOULDER, POSE_INDEX.RIGHT_ELBOW],
  [POSE_INDEX.RIGHT_ELBOW, POSE_INDEX.RIGHT_WRIST],
  [POSE_INDEX.LEFT_SHOULDER, POSE_INDEX.LEFT_HIP],
  [POSE_INDEX.RIGHT_SHOULDER, POSE_INDEX.RIGHT_HIP],
  [POSE_INDEX.LEFT_HIP, POSE_INDEX.RIGHT_HIP],
  [POSE_INDEX.LEFT_HIP, POSE_INDEX.LEFT_KNEE],
  [POSE_INDEX.LEFT_KNEE, POSE_INDEX.LEFT_ANKLE],
  [POSE_INDEX.RIGHT_HIP, POSE_INDEX.RIGHT_KNEE],
  [POSE_INDEX.RIGHT_KNEE, POSE_INDEX.RIGHT_ANKLE],
];

const LIVE_KEYPOINT_INDICES: readonly number[] = [
  POSE_INDEX.NOSE,
  POSE_INDEX.LEFT_SHOULDER,
  POSE_INDEX.RIGHT_SHOULDER,
  POSE_INDEX.LEFT_ELBOW,
  POSE_INDEX.RIGHT_ELBOW,
  POSE_INDEX.LEFT_WRIST,
  POSE_INDEX.RIGHT_WRIST,
  POSE_INDEX.LEFT_HIP,
  POSE_INDEX.RIGHT_HIP,
  POSE_INDEX.LEFT_KNEE,
  POSE_INDEX.RIGHT_KNEE,
  POSE_INDEX.LEFT_ANKLE,
  POSE_INDEX.RIGHT_ANKLE,
];

const UPPER_BODY_KEYPOINTS = new Set<number>([
  POSE_INDEX.NOSE,
  POSE_INDEX.LEFT_SHOULDER,
  POSE_INDEX.RIGHT_SHOULDER,
  POSE_INDEX.LEFT_ELBOW,
  POSE_INDEX.RIGHT_ELBOW,
  POSE_INDEX.LEFT_WRIST,
  POSE_INDEX.RIGHT_WRIST,
]);

function drawLivePoseOverlayOnCanvas(
  canvas: HTMLCanvasElement | null,
  landmarksList: readonly (readonly NormalizedLandmark[])[],
  opts: {
    targetJoint: ExercisePrescriptionOption['target_joint'];
    romDeg: number | null;
    targetRomDeg: number;
    holdRatio: number;
    trunkTiltDeg: number;
    isMultiPerson: boolean;
    isImpostorOrEmergency: boolean;
    wrongSideRaised?: boolean;
    wrongSideBannerTh?: string | null;
    videoWidth?: number;
    videoHeight?: number;
  },
): void {
  if (!canvas) return;
  const w = canvas.clientWidth || 640;
  const h = canvas.clientHeight || 480;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (landmarksList.length === 0) return;

  // Match CSS object-cover scale/crop so mirrored landmarks align 1:1 over the video feed
  const vw = opts.videoWidth && opts.videoWidth > 0 ? opts.videoWidth : w;
  const vh = opts.videoHeight && opts.videoHeight > 0 ? opts.videoHeight : h;
  const coverScale = Math.max(w / vw, h / vh);
  const renderW = vw * coverScale;
  const renderH = vh * coverScale;
  const offsetX = (w - renderW) / 2;
  const offsetY = (h - renderH) / 2;
  const rawCanvasX = (normX: number) => offsetX + (1 - normX) * renderW;
  const rawCanvasY = (normY: number) => offsetY + normY * renderH;
  const toCanvasPt = (idx: number, lm: NormalizedLandmark) => {
    const cx = rawCanvasX(lm.x);
    const cy = rawCanvasY(lm.y);
    if (UPPER_BODY_KEYPOINTS.has(idx)) {
      return {
        x: Math.max(18, Math.min(w - 18, cx)),
        y: Math.max(18, Math.min(h - 18, cy)),
      };
    }
    return { x: cx, y: cy };
  };
  const minVisFor = (idx: number) => (UPPER_BODY_KEYPOINTS.has(idx) ? 0.02 : 0.2);

  const activeColor = opts.isImpostorOrEmergency
    ? '#EF4444'
    : opts.isMultiPerson || opts.trunkTiltDeg > 10
      ? '#F59E0B'
      : '#10B981';

  const isHighlightedEdge = (a: number, b: number) => {
    if (opts.targetJoint === 'left_shoulder') {
      return (
        (a === POSE_INDEX.LEFT_SHOULDER && b === POSE_INDEX.LEFT_ELBOW) ||
        (a === POSE_INDEX.LEFT_ELBOW && b === POSE_INDEX.LEFT_WRIST)
      );
    }
    if (opts.targetJoint === 'right_shoulder') {
      return (
        (a === POSE_INDEX.RIGHT_SHOULDER && b === POSE_INDEX.RIGHT_ELBOW) ||
        (a === POSE_INDEX.RIGHT_ELBOW && b === POSE_INDEX.RIGHT_WRIST)
      );
    }
    if (opts.targetJoint === 'left_knee') {
      return (
        (a === POSE_INDEX.LEFT_HIP && b === POSE_INDEX.LEFT_KNEE) ||
        (a === POSE_INDEX.LEFT_KNEE && b === POSE_INDEX.LEFT_ANKLE)
      );
    }
    if (opts.targetJoint === 'right_knee') {
      return (
        (a === POSE_INDEX.RIGHT_HIP && b === POSE_INDEX.RIGHT_KNEE) ||
        (a === POSE_INDEX.RIGHT_KNEE && b === POSE_INDEX.RIGHT_ANKLE)
      );
    }
    return (
      (a === POSE_INDEX.LEFT_SHOULDER && b === POSE_INDEX.RIGHT_SHOULDER) ||
      (a === POSE_INDEX.LEFT_SHOULDER && b === POSE_INDEX.LEFT_HIP) ||
      (a === POSE_INDEX.RIGHT_SHOULDER && b === POSE_INDEX.RIGHT_HIP)
    );
  };

  const isOppositeArmEdge = (a: number, b: number) => {
    if (opts.targetJoint === 'left_shoulder') {
      return (
        (a === POSE_INDEX.RIGHT_SHOULDER && b === POSE_INDEX.RIGHT_ELBOW) ||
        (a === POSE_INDEX.RIGHT_ELBOW && b === POSE_INDEX.RIGHT_WRIST)
      );
    }
    if (opts.targetJoint === 'right_shoulder') {
      return (
        (a === POSE_INDEX.LEFT_SHOULDER && b === POSE_INDEX.LEFT_ELBOW) ||
        (a === POSE_INDEX.LEFT_ELBOW && b === POSE_INDEX.LEFT_WRIST)
      );
    }
    return false;
  };

  // Primary patient live skeleton overlay
  const primary = landmarksList[0];
  ctx.save();
  ctx.lineCap = 'round';
  for (const [aIdx, bIdx] of LIVE_BODY_BONELINE_PAIRS) {
    const a = primary[aIdx];
    const b = primary[bIdx];
    if (!a || !b || (a.visibility ?? 0) < minVisFor(aIdx) || (b.visibility ?? 0) < minVisFor(bIdx)) {
      continue;
    }
    const aPt = toCanvasPt(aIdx, a);
    const bPt = toCanvasPt(bIdx, b);
    const emphasized = isHighlightedEdge(aIdx, bIdx);
    const isWrongRaisedEdge = Boolean(opts.wrongSideRaised && isOppositeArmEdge(aIdx, bIdx));

    ctx.beginPath();
    ctx.moveTo(aPt.x, aPt.y);
    ctx.lineTo(bPt.x, bPt.y);
    ctx.lineWidth = isWrongRaisedEdge ? 6 : emphasized ? 6 : 3.5;
    ctx.strokeStyle = isWrongRaisedEdge
      ? '#EF4444'
      : emphasized
        ? activeColor
        : 'rgba(56, 189, 248, 0.9)';
    if (isWrongRaisedEdge || emphasized) {
      ctx.shadowColor = isWrongRaisedEdge ? '#EF4444' : activeColor;
      ctx.shadowBlur = 12;
    } else {
      ctx.shadowBlur = 0;
    }
    ctx.stroke();
  }

  // Head/neck guide lines (NOSE -> shoulder midpoint NECK -> both shoulders)
  const noseLm = primary[POSE_INDEX.NOSE];
  const lShLm = primary[POSE_INDEX.LEFT_SHOULDER];
  const rShLm = primary[POSE_INDEX.RIGHT_SHOULDER];
  if (
    lShLm &&
    rShLm &&
    (lShLm.visibility ?? 0) >= 0.02 &&
    (rShLm.visibility ?? 0) >= 0.02
  ) {
    const lShPt = toCanvasPt(POSE_INDEX.LEFT_SHOULDER, lShLm);
    const rShPt = toCanvasPt(POSE_INDEX.RIGHT_SHOULDER, rShLm);
    const neckPt = { x: (lShPt.x + rShPt.x) / 2, y: (lShPt.y + rShPt.y) / 2 };

    ctx.shadowBlur = 0;
    ctx.strokeStyle = 'rgba(13, 148, 136, 0.85)';
    ctx.lineWidth = 3.2;
    ctx.beginPath();
    ctx.moveTo(lShPt.x, lShPt.y);
    ctx.lineTo(neckPt.x, neckPt.y);
    ctx.lineTo(rShPt.x, rShPt.y);
    if (noseLm && (noseLm.visibility ?? 0) >= 0.02) {
      const nosePt = toCanvasPt(POSE_INDEX.NOSE, noseLm);
      ctx.moveTo(neckPt.x, neckPt.y);
      ctx.lineTo(nosePt.x, nosePt.y);
    }
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(neckPt.x, neckPt.y, 4.5, 0, Math.PI * 2);
    ctx.fillStyle = '#FFFFFF';
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = activeColor;
    ctx.stroke();
  }
  ctx.shadowBlur = 0;

  // Joint keypoints
  for (const idx of LIVE_KEYPOINT_INDICES) {
    const lm = primary[idx];
    if (!lm || (lm.visibility ?? 0) < minVisFor(idx)) continue;
    const pt = toCanvasPt(idx, lm);
    const r = idx === POSE_INDEX.NOSE ? 5 : 6;
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, r, 0, Math.PI * 2);
    ctx.fillStyle = '#FFFFFF';
    ctx.fill();
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = activeColor;
    ctx.stroke();
  }

  // On-body floating angle badge & radial hold indicator
  const anchorIdx =
    opts.targetJoint === 'left_shoulder'
      ? POSE_INDEX.LEFT_SHOULDER
      : opts.targetJoint === 'right_shoulder'
        ? POSE_INDEX.RIGHT_SHOULDER
        : opts.targetJoint === 'left_knee'
          ? POSE_INDEX.LEFT_KNEE
          : opts.targetJoint === 'right_knee'
            ? POSE_INDEX.RIGHT_KNEE
            : POSE_INDEX.NOSE;
  const anchorLm = primary[anchorIdx] ?? primary[POSE_INDEX.NOSE];
  if (anchorLm && opts.romDeg !== null) {
    const anchorPt = toCanvasPt(
      primary[anchorIdx] ? anchorIdx : POSE_INDEX.NOSE,
      anchorLm,
    );
    const jx = Math.max(64, Math.min(w - 64, anchorPt.x));
    const jy = Math.max(54, Math.min(h - 40, anchorPt.y - 42));

    if (opts.holdRatio > 0) {
      ctx.beginPath();
      ctx.arc(jx, jy, 34, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * Math.min(1, opts.holdRatio));
      ctx.strokeStyle = '#10B981';
      ctx.lineWidth = 5;
      ctx.stroke();
    }

    const badgeText = `${Math.round(opts.romDeg)}° / ${opts.targetRomDeg}°`;
    ctx.font = '800 14px "Sarabun", sans-serif';
    const tw = ctx.measureText(badgeText).width;
    const bw = Math.max(82, tw + 20);
    const bh = 30;
    const bx = jx - bw / 2;
    const by = jy - bh / 2;

    ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
    ctx.strokeStyle = activeColor;
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.roundRect(bx, by, bw, bh, 15);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#FFFFFF';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(badgeText, jx, jy);
  }

  // On-body alert badge when patient raises the wrong arm
  if (opts.wrongSideRaised) {
    const wrongElbowIdx =
      opts.targetJoint === 'right_shoulder' ? POSE_INDEX.LEFT_ELBOW : POSE_INDEX.RIGHT_ELBOW;
    const wrongShIdx =
      opts.targetJoint === 'right_shoulder' ? POSE_INDEX.LEFT_SHOULDER : POSE_INDEX.RIGHT_SHOULDER;
    const wrongLm = primary[wrongElbowIdx] ?? primary[wrongShIdx] ?? primary[POSE_INDEX.NOSE];
    const limbInfo = getPrescribedLimbPromptTh(opts.targetJoint);
    const warnText = `✗ ยก${limbInfo.wrongLimbTh} • กรุณายก${limbInfo.requiredLimbTh}`;
    ctx.font = '800 13px "Sarabun", sans-serif';
    const wtw = ctx.measureText(warnText).width;
    const wbw = Math.min(w - 24, wtw + 24);
    const wbh = 32;
    const wx = wrongLm
      ? Math.max(wbw / 2 + 12, Math.min(w - wbw / 2 - 12, rawCanvasX(wrongLm.x)))
      : w / 2;
    const wy = wrongLm ? Math.max(88, Math.min(h - 36, rawCanvasY(wrongLm.y) - 34)) : 96;

    ctx.fillStyle = 'rgba(220, 38, 38, 0.92)';
    ctx.strokeStyle = '#FCA5A5';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(wx - wbw / 2, wy - wbh / 2, wbw, wbh, 16);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#FFFFFF';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(warnText, wx, wy);
  }

  // Multi-person camera bounding boxes when 2+ people enter view
  if (opts.isMultiPerson || landmarksList.length > 1) {
    landmarksList.forEach((person, pIdx) => {
      let minX = w;
      let maxX = 0;
      let minY = h;
      let maxY = 0;
      let any = false;
      for (const idx of LIVE_KEYPOINT_INDICES) {
        const lm = person[idx];
        if (!lm || (lm.visibility ?? 0) < minVisFor(idx)) continue;
        const pt = toCanvasPt(idx, lm);
        const cx = pt.x;
        const cy = pt.y;
        minX = Math.min(minX, cx);
        maxX = Math.max(maxX, cx);
        minY = Math.min(minY, cy);
        maxY = Math.max(maxY, cy);
        any = true;
      }
      if (!any) return;
      const x1 = Math.max(8, minX - 24);
      const y1 = Math.max(8, minY - 28);
      const bw = Math.min(w - x1 - 8, Math.max(72, maxX - minX + 48));
      const bh = Math.min(h - y1 - 8, Math.max(96, maxY - minY + 44));
      const boxHex = pIdx === 0 ? '#10B981' : '#F59E0B';
      const label = pIdx === 0 ? '✓ คนไข้หลัก' : '⚠ บุคคลเกินในเฟรม';

      ctx.strokeStyle = boxHex;
      ctx.lineWidth = 3;
      ctx.setLineDash(pIdx === 0 ? [] : [8, 5]);
      ctx.strokeRect(x1, y1, bw, bh);
      ctx.setLineDash([]);

      ctx.font = '800 12px "Sarabun", sans-serif';
      const lw = ctx.measureText(label).width + 16;
      const ly = Math.max(6, y1 - 24);
      ctx.fillStyle = boxHex;
      ctx.fillRect(x1, ly, lw, 22);
      ctx.fillStyle = '#FFFFFF';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, x1 + 8, ly + 11);
    });
  }
  ctx.restore();
}

export function SmartRehabKioskUI({
  patient: patientProp,
  prescribedExercises,
  onExerciseCompleted,
  onAllExercisesCompleted,
  onEmergencyStop,
  onOpenStaffPortal,
  onOpenLegacyKiosk,
}: SmartRehabKioskUIProps) {
  const socketOptions = useMemo(
    () =>
      prescribedExercises || patientProp !== undefined
        ? { prescribedExercises, patientOverride: patientProp }
        : undefined,
    [prescribedExercises, patientProp],
  );

  const {
    payload,
    connectionStatus,
    isDemoMode,
    toggleDemoMode,
    exerciseOptions,
    selectedExercise,
    setSelectedExercise,
    demoPreset,
    setDemoPreset,
    setDemoPresetState,
    demoRomDeg,
    setDemoRomDeg,
    demoTrunkTiltDeg,
    setDemoTrunkTiltDeg,
    demoDistanceM,
    setDemoDistanceM,
    demoValidReps,
    setDemoHoldRemainingSec,
    setDemoValidReps,
    demoStreakFrames,
    demoFaceMatched,
    autoCycleDemo,
    setAutoCycleDemo,
    triggerEmergencyStop,
    resetDemoSession,
    completeCurrentExercise,
  } = useKioskSocket('ws://localhost:8000/ws/kiosk', socketOptions);

  const [voiceMuted, setVoiceMuted] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(() => !patientProp);
  const [completedIds, setCompletedIds] = useState<string[]>([]);
  const [livePoseActive, setLivePoseActive] = useState<boolean>(true);
  const [cameraReady, setCameraReady] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const poseCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const onExerciseCompletedRef = useRef(onExerciseCompleted);
  const onAllExercisesCompletedRef = useRef(onAllExercisesCompleted);
  const selectedExerciseRef = useRef(selectedExercise);
  const demoValidRepsRef = useRef(demoValidReps);
  const demoPresetRef = useRef(demoPreset);
  const countingLockedRef = useRef(false);

  onExerciseCompletedRef.current = onExerciseCompleted;
  onAllExercisesCompletedRef.current = onAllExercisesCompleted;
  selectedExerciseRef.current = selectedExercise;
  demoValidRepsRef.current = demoValidReps;
  demoPresetRef.current = demoPreset;
  countingLockedRef.current =
    payload.kiosk_state === 'PAUSED_MULTI_PERSON' ||
    payload.kiosk_state === 'PAUSED_IMPOSTOR' ||
    payload.identity_guard?.is_counting_allowed === false;

  const { kiosk_state, patient, telemetry_analysis, rep_counter, ui_directives, tts_voice_th } = payload;
  const compensations = telemetry_analysis.compensation_detected;
  const trunkLeanAlert = compensations.includes('TRUNK_LEAN');
  const shoulderHikeAlert = compensations.includes('SHOULDER_HIKE');
  const kneeValgusAlert = compensations.includes('KNEE_VALGUS');
  const tooFastAlert = compensations.includes('TOO_FAST');
  const wrongLimbAlert = compensations.includes('WRONG_LIMB');
  const isHoldPhase = kiosk_state === 'HOLD_PHASE';
  const isRepCelebration = kiosk_state === 'REP_COMPLETED' || kiosk_state === 'SET_FINISHED';
  const isEmergency = kiosk_state === 'EMERGENCY_HALT';

  const handleEmergencyStopClick = () => {
    triggerEmergencyStop();
    onEmergencyStop?.({
      reason: `กดปุ่มหยุดฉุกเฉินระหว่างฝึกท่า ${selectedExercise.name_th}`,
      patientId: patient?.id ?? patientProp?.id ?? undefined,
      exerciseNameTh: selectedExercise.name_th,
    });
  };

  const markCurrentExerciseCompleted = () => {
    completeCurrentExercise();
    if (!completedIds.includes(selectedExercise.exercise_id)) {
      setCompletedIds((prev) =>
        prev.includes(selectedExercise.exercise_id)
          ? prev
          : [...prev, selectedExercise.exercise_id],
      );
      onExerciseCompletedRef.current?.(selectedExercise);
    }
  };

  const exerciseListSignature = exerciseOptions.map((item) => item.exercise_id).join('|');
  useEffect(() => {
    setCompletedIds([]);
  }, [exerciseListSignature]);

  const currentExerciseIndex = Math.max(
    0,
    exerciseOptions.findIndex((opt) => opt.exercise_id === selectedExercise.exercise_id),
  );
  const allExercisesCompleted =
    exerciseOptions.length > 0 && completedIds.length >= exerciseOptions.length;
  const nextUncompletedExercise = useMemo(() => {
    const afterCurrent = exerciseOptions.find(
      (opt, idx) =>
        idx > currentExerciseIndex &&
        opt.exercise_id !== selectedExercise.exercise_id &&
        !completedIds.includes(opt.exercise_id),
    );
    if (afterCurrent) return afterCurrent;
    return exerciseOptions.find(
      (opt) => opt.exercise_id !== selectedExercise.exercise_id && !completedIds.includes(opt.exercise_id),
    );
  }, [exerciseOptions, currentExerciseIndex, selectedExercise.exercise_id, completedIds]);

  // Mark exercise completed once target reps are reached
  useEffect(() => {
    if (rep_counter.valid_reps < selectedExercise.target_reps) return;
    if (completedIds.includes(selectedExercise.exercise_id)) return;
    setCompletedIds((prev) =>
      prev.includes(selectedExercise.exercise_id) ? prev : [...prev, selectedExercise.exercise_id],
    );
    onExerciseCompletedRef.current?.(selectedExercise);
  }, [rep_counter.valid_reps, selectedExercise, completedIds]);

  // Auto-advance to the next uncompleted exercise after completing one set
  useEffect(() => {
    if (!completedIds.includes(selectedExercise.exercise_id)) return;
    if (allExercisesCompleted) {
      if (!onAllExercisesCompletedRef.current) return;
      const doneTimer = window.setTimeout(() => {
        onAllExercisesCompletedRef.current?.();
      }, 2200);
      return () => window.clearTimeout(doneTimer);
    }
    if (!nextUncompletedExercise) return;
    const nextTimer = window.setTimeout(() => {
      setSelectedExercise(nextUncompletedExercise);
    }, 1800);
    return () => window.clearTimeout(nextTimer);
  }, [
    completedIds,
    selectedExercise.exercise_id,
    allExercisesCompleted,
    nextUncompletedExercise,
    setSelectedExercise,
  ]);

  // Live MediaPipe PoseLandmarker camera tracking (when enabled & not auto-cycling demo)
  useEffect(() => {
    if (!livePoseActive || autoCycleDemo || isEmergency) {
      setCameraReady(false);
      return;
    }

    let active = true;
    let stream: MediaStream | null = null;
    let rafId = 0;
    let poseLandmarker: PoseLandmarker | null = null;
    let offscreenCanvas: HTMLCanvasElement | null = null;
    let useOffscreenFallback = false;
    let lastVideoTime = -1;
    let lastInferenceAt = 0;
    let lastDetectTs = 0;
    let lastFrameAt = 0;
    let smoothedRom: number | null = 0;
    let smoothedTilt = 0;
    let holdAccumMs = 0;
    let awaitingReturn = false;
    let repCelebrateUntil = 0;
    let lastRepCompletedAt = 0;
    let singlePersonRecoveryFrames = 0;
    let trackedExId = selectedExerciseRef.current.exercise_id;
    let targetRomDeg: number | null = null;
    let latestLandmarks: NormalizedLandmark[] | null = null;
    let renderPrimaryLandmarks: NormalizedLandmark[] | null = null;
    let latestAllLandmarks: NormalizedLandmark[][] = [];
    let latestPoseCount = 0;
    let latestRawTiltDeg = 0;
    let latestTrunkTiltDeg = 0;
    let latestWrongSideRaised = false;
    let latestWrongSideBannerTh: string | null = null;
    let lastEmittedRom = -999;
    let lastEmittedTilt = -999;
    let lastEmittedHold = -999;
    let lastRomEmitAt = 0;
    let lastTiltEmitAt = 0;
    let lastHoldEmitAt = 0;

    const cleanup = () => {
      if (rafId) {
        cancelAnimationFrame(rafId);
        rafId = 0;
      }
      if (poseCanvasRef.current) {
        const ctx = poseCanvasRef.current.getContext('2d');
        ctx?.clearRect(0, 0, poseCanvasRef.current.width, poseCanvasRef.current.height);
      }
      if (stream) {
        stream.getTracks().forEach((t) => t.stop());
        stream = null;
      }
      if (poseLandmarker) {
        try {
          poseLandmarker.close();
        } catch {
          // ignore close errors
        }
        poseLandmarker = null;
      }
    };

    async function initPoseCamera() {
      try {
        if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
          setCameraError('เบราว์เซอร์ไม่รองรับกล้อง (สามารถฝึกผ่านโหมดควบคุมได้)');
          return;
        }
        setCameraError(null);
        // Brief pause allows the previous step's camera stream to release cleanly on Windows
        await new Promise((r) => setTimeout(r, 180));
        if (!active) return;

        const deviceId = await selectClinicalCameraDeviceId();
        if (!active) return;

        stream = await navigator.mediaDevices.getUserMedia({
          video: deviceId
            ? {
                deviceId: { exact: deviceId },
                width: { ideal: 640 },
                height: { ideal: 480 },
                frameRate: { ideal: 60, max: 120 },
                facingMode: 'user',
              }
            : {
                width: { ideal: 640 },
                height: { ideal: 480 },
                frameRate: { ideal: 60, max: 120 },
                facingMode: 'user',
              },
          audio: false,
        });
        if (!active || !videoRef.current) {
          cleanup();
          return;
        }
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        if (!active) {
          cleanup();
          return;
        }
        setCameraReady(true);

        let vision;
        for (const wasmBase of [
          '/mediapipe/wasm',
          '/models/wasm',
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm',
        ]) {
          try {
            vision = await FilesetResolver.forVisionTasks(wasmBase);
            break;
          } catch {
            // Try next WASM base URL
          }
        }
        if (!vision || !active) return;

        for (const modelPath of ['/models/pose_landmarker.task', '/models/pose_landmarker_full.task']) {
          for (const delegate of ['GPU', 'CPU', undefined] as const) {
            try {
              const created = await PoseLandmarker.createFromOptions(vision, {
                baseOptions: {
                  modelAssetPath: modelPath,
                  ...(delegate ? { delegate } : {}),
                },
                runningMode: 'VIDEO',
                numPoses: 2,
                minPoseDetectionConfidence: 0.3,
                minPosePresenceConfidence: 0.3,
                minTrackingConfidence: 0.3,
              });
              if (!active) {
                created.close();
                return;
              }
              poseLandmarker = created;
              break;
            } catch {
              // Try fallback
            }
          }
          if (poseLandmarker) break;
        }
        if (!poseLandmarker || !active) return;

        const loop = () => {
          if (!active || !videoRef.current || !poseLandmarker) return;
          const curEx = selectedExerciseRef.current;
          if (trackedExId !== curEx.exercise_id) {
            trackedExId = curEx.exercise_id;
            smoothedRom = 0;
            smoothedTilt = 0;
            targetRomDeg = null;
            latestLandmarks = null;
            renderPrimaryLandmarks = null;
            latestAllLandmarks = [];
            latestPoseCount = 0;
            latestRawTiltDeg = 0;
            latestTrunkTiltDeg = 0;
            latestWrongSideRaised = false;
            latestWrongSideBannerTh = null;
            lastEmittedRom = -999;
            lastEmittedTilt = -999;
            lastEmittedHold = -999;
            lastRomEmitAt = 0;
            lastTiltEmitAt = 0;
            lastHoldEmitAt = 0;
            holdAccumMs = 0;
            awaitingReturn = false;
            repCelebrateUntil = 0;
            lastRepCompletedAt = 0;
          }

          const v = videoRef.current;
          const now = performance.now();
          const wallDtMs = lastFrameAt > 0 ? Math.min(500, Math.max(1, now - lastFrameAt)) : 16.7;
          const kinematicDtMs = Math.min(100, wallDtMs);
          lastFrameAt = now;

          // Step A: Run AI pose inference when a new video frame arrives (zero-copy HTMLVideoElement first)
          if (
            v.readyState >= 2 &&
            v.videoWidth > 0 &&
            (v.currentTime !== lastVideoTime || now - lastInferenceAt >= 36)
          ) {
            lastVideoTime = v.currentTime;
            lastInferenceAt = now;

            const runDetect = (source: HTMLVideoElement | HTMLCanvasElement, ts: number) =>
              poseLandmarker!.detectForVideo(source, ts);

            try {
              const ts = Math.max(lastDetectTs + 1, Math.round(now));
              lastDetectTs = ts;
              let poseRes;
              if (!useOffscreenFallback) {
                try {
                  poseRes = runDetect(v, ts);
                } catch {
                  useOffscreenFallback = true;
                }
              }
              if (useOffscreenFallback) {
                if (!offscreenCanvas) {
                  offscreenCanvas = document.createElement('canvas');
                }
                const offW = 256;
                const offH = Math.max(144, Math.round(offW * ((v.videoHeight || 480) / (v.videoWidth || 640))));
                if (offscreenCanvas.width !== offW || offscreenCanvas.height !== offH) {
                  offscreenCanvas.width = offW;
                  offscreenCanvas.height = offH;
                }
                const offCtx = offscreenCanvas.getContext('2d', { alpha: false });
                if (offCtx) {
                  offCtx.drawImage(v, 0, 0, offW, offH);
                }
                poseRes = runDetect(offCtx ? offscreenCanvas : v, ts);
              }

              const allLandmarks = poseRes?.landmarks ?? [];
              latestAllLandmarks = allLandmarks;
              latestPoseCount = allLandmarks.length;

              if (latestPoseCount > 1) {
                singlePersonRecoveryFrames = 0;
                latestLandmarks = null;
                renderPrimaryLandmarks = null;
              } else if (demoPresetRef.current !== 'IMPOSTOR') {
                const landmarks = allLandmarks[0] ?? null;
                latestLandmarks = landmarks;
                if (landmarks) {
                  if (demoPresetRef.current === 'MULTI_PERSON') {
                    singlePersonRecoveryFrames += 1;
                    if (singlePersonRecoveryFrames < 3) {
                      holdAccumMs = 0;
                      if (lastEmittedHold !== curEx.hold_sec) {
                        lastEmittedHold = curEx.hold_sec;
                        lastHoldEmitAt = now;
                        setDemoHoldRemainingSec(curEx.hold_sec);
                      }
                    } else {
                      singlePersonRecoveryFrames = 0;
                      setDemoPresetState('ACTIVE_REP');
                    }
                  }
                  const measured = measureLiveJointAngleAndTilt(landmarks, curEx.target_joint);
                  latestRawTiltDeg = measured.trunkTiltDeg;
                  latestWrongSideRaised = measured.wrongSideRaised;
                  latestWrongSideBannerTh = measured.wrongSideBannerTh;
                  if (measured.romDeg !== null) {
                    targetRomDeg = measured.romDeg;
                  }
                } else {
                  renderPrimaryLandmarks = null;
                }
              }
            } catch {
              // Ignore transient inference errors
            }
          }

          // Step B: 120Hz landmark & ROM interpolation + 1:1 wall-clock hold timer + canvas redraw
          const holdTargetMs = Math.max(1000, curEx.hold_sec * 1000);
          if (latestPoseCount > 1) {
            holdAccumMs = 0;
            if (lastEmittedHold !== curEx.hold_sec) {
              lastEmittedHold = curEx.hold_sec;
              lastHoldEmitAt = now;
              setDemoHoldRemainingSec(curEx.hold_sec);
            }
            setDemoPresetState('MULTI_PERSON');
            drawLivePoseOverlayOnCanvas(poseCanvasRef.current, latestAllLandmarks, {
              targetJoint: curEx.target_joint,
              romDeg: targetRomDeg !== null ? smoothedRom : null,
              targetRomDeg: curEx.target_rom_deg,
              holdRatio: 0,
              trunkTiltDeg: 0,
              isMultiPerson: true,
              isImpostorOrEmergency: false,
              videoWidth: v.videoWidth,
              videoHeight: v.videoHeight,
            });
          } else if (demoPresetRef.current === 'IMPOSTOR') {
            holdAccumMs = 0;
            drawLivePoseOverlayOnCanvas(poseCanvasRef.current, latestAllLandmarks, {
              targetJoint: curEx.target_joint,
              romDeg: targetRomDeg !== null ? smoothedRom : null,
              targetRomDeg: curEx.target_rom_deg,
              holdRatio: 0,
              trunkTiltDeg: 0,
              isMultiPerson: false,
              isImpostorOrEmergency: true,
              videoWidth: v.videoWidth,
              videoHeight: v.videoHeight,
            });
          } else if (latestLandmarks) {
            renderPrimaryLandmarks = interpolatePoseLandmarks(
              renderPrimaryLandmarks,
              latestLandmarks,
              kinematicDtMs,
            );
            const renderLandmarksList: NormalizedLandmark[][] = [
              renderPrimaryLandmarks,
              ...latestAllLandmarks.slice(1),
            ];

            const tiltDelta = latestRawTiltDeg - smoothedTilt;
            if (Math.abs(tiltDelta) >= 0.4) {
              const tiltAlpha = 1 - Math.exp(-kinematicDtMs / 120);
              smoothedTilt = Math.max(0, smoothedTilt + tiltDelta * tiltAlpha);
            }
            latestTrunkTiltDeg = Number(smoothedTilt.toFixed(1));
            if (now - lastTiltEmitAt >= 90 && Math.abs(latestTrunkTiltDeg - lastEmittedTilt) >= 0.3) {
              lastEmittedTilt = latestTrunkTiltDeg;
              lastTiltEmitAt = now;
              setDemoTrunkTiltDeg(latestTrunkTiltDeg);
            }

            if (targetRomDeg !== null) {
              smoothedRom = stepSmoothRomAngle(smoothedRom, targetRomDeg, kinematicDtMs);
              const nextEmitRom = Number(smoothedRom.toFixed(1));
              const hitBoundary =
                (nextEmitRom <= 0.1 || Math.abs(nextEmitRom - curEx.target_rom_deg) <= 0.2) &&
                Math.abs(nextEmitRom - lastEmittedRom) >= 0.1;
              if (
                (now - lastRomEmitAt >= 24 && Math.abs(nextEmitRom - lastEmittedRom) >= 0.25) ||
                hitBoundary
              ) {
                lastEmittedRom = nextEmitRom;
                lastRomEmitAt = now;
                setDemoRomDeg(nextEmitRom);
              }

              const isKneeExercise = curEx.target_joint.includes('knee');
              const targetThreshold = isKneeExercise
                ? curEx.target_rom_deg + 15
                : curEx.target_rom_deg * 0.82;
              const holdMaintainThreshold = isKneeExercise
                ? curEx.target_rom_deg + 22
                : curEx.target_rom_deg * 0.75;
              const atTarget = isKneeExercise
                ? smoothedRom <= targetThreshold
                : smoothedRom >= targetThreshold;
              const atMaintainTarget = isKneeExercise
                ? smoothedRom <= holdMaintainThreshold
                : smoothedRom >= holdMaintainThreshold;
              const atRest = isKneeExercise
                ? smoothedRom >= 145
                : smoothedRom <= Math.min(55, curEx.target_rom_deg * 0.45);

              if (atRest && now - lastRepCompletedAt >= 1200) {
                awaitingReturn = false;
              }

              const blockCounting =
                demoPresetRef.current === 'MULTI_PERSON' ||
                countingLockedRef.current;

              if (demoValidRepsRef.current < curEx.target_reps) {
                if (latestWrongSideRaised) {
                  holdAccumMs = 0;
                  if (lastEmittedHold !== curEx.hold_sec) {
                    lastEmittedHold = curEx.hold_sec;
                    lastHoldEmitAt = now;
                    setDemoHoldRemainingSec(curEx.hold_sec);
                  }
                  setDemoPresetState('WRONG_LIMB');
                } else if (!blockCounting) {
                  const holdingZone = atTarget || (holdAccumMs > 0 && atMaintainTarget);
                  if (holdingZone && !awaitingReturn && latestTrunkTiltDeg <= 14) {
                    holdAccumMs += wallDtMs;
                    const remSec = Math.max(0, Number(((holdTargetMs - holdAccumMs) / 1000).toFixed(1)));
                    if (now - lastHoldEmitAt >= 40 && remSec !== lastEmittedHold) {
                      lastEmittedHold = remSec;
                      lastHoldEmitAt = now;
                      setDemoHoldRemainingSec(remSec);
                    }
                    if (holdAccumMs >= holdTargetMs) {
                      holdAccumMs = 0;
                      awaitingReturn = true;
                      lastRepCompletedAt = now;
                      repCelebrateUntil = now + 4000;
                      lastEmittedHold = 0;
                      lastHoldEmitAt = now;
                      setDemoHoldRemainingSec(0);
                      setDemoValidReps((r) => {
                        const next = Math.min(curEx.target_reps, r + 1);
                        setDemoPresetState(next >= curEx.target_reps ? 'SET_FINISHED' : 'REP_COMPLETED');
                        return next;
                      });
                    } else {
                      setDemoPresetState('HOLD_PHASE');
                    }
                  } else {
                    holdAccumMs = atRest ? 0 : Math.max(0, holdAccumMs - wallDtMs * 0.45);
                    if (awaitingReturn || now <= repCelebrateUntil) {
                      if (lastEmittedHold !== 0) {
                        lastEmittedHold = 0;
                        lastHoldEmitAt = now;
                        setDemoHoldRemainingSec(0);
                      }
                      setDemoPresetState(
                        demoValidRepsRef.current >= curEx.target_reps ? 'SET_FINISHED' : 'REP_COMPLETED',
                      );
                    } else {
                      const fallbackHoldSec =
                        holdAccumMs > 0
                          ? Math.max(0, Number(((holdTargetMs - holdAccumMs) / 1000).toFixed(1)))
                          : curEx.hold_sec;
                      if (lastEmittedHold !== fallbackHoldSec && now - lastHoldEmitAt >= 40) {
                        lastEmittedHold = fallbackHoldSec;
                        lastHoldEmitAt = now;
                        setDemoHoldRemainingSec(fallbackHoldSec);
                      }
                      setDemoPresetState(latestTrunkTiltDeg > 10 ? 'TRUNK_LEAN' : 'ACTIVE_REP');
                    }
                  }
                }
              }
            }

            drawLivePoseOverlayOnCanvas(poseCanvasRef.current, renderLandmarksList, {
              targetJoint: curEx.target_joint,
              romDeg: smoothedRom,
              targetRomDeg: curEx.target_rom_deg,
              holdRatio: holdAccumMs > 0 ? Math.min(1, holdAccumMs / holdTargetMs) : 0,
              trunkTiltDeg: latestTrunkTiltDeg,
              isMultiPerson: demoPresetRef.current === 'MULTI_PERSON',
              isImpostorOrEmergency: countingLockedRef.current,
              wrongSideRaised: latestWrongSideRaised,
              wrongSideBannerTh: latestWrongSideBannerTh,
              videoWidth: v.videoWidth,
              videoHeight: v.videoHeight,
            });
          } else {
            renderPrimaryLandmarks = null;
            drawLivePoseOverlayOnCanvas(poseCanvasRef.current, [], {
              targetJoint: curEx.target_joint,
              romDeg: null,
              targetRomDeg: curEx.target_rom_deg,
              holdRatio: 0,
              trunkTiltDeg: 0,
              isMultiPerson: false,
              isImpostorOrEmergency: false,
            });
          }

          if (active) {
            rafId = requestAnimationFrame(loop);
          }
        };

        rafId = requestAnimationFrame(loop);
      } catch {
        if (active) {
          setCameraError('ไม่พบกล้องหรือไม่สามารถเปิดกล้องได้');
        }
      }
    }

    void initPoseCamera();
    return () => {
      active = false;
      cleanup();
    };
  }, [
    livePoseActive,
    autoCycleDemo,
    isEmergency,
    setDemoHoldRemainingSec,
    setDemoPresetState,
    setDemoRomDeg,
    setDemoTrunkTiltDeg,
    setDemoValidReps,
  ]);

  const patientRevealed = Boolean(patient?.name && (kiosk_state !== 'AUTH_FRAMING' || demoFaceMatched));

  const checklistItems = [
    ...(selectedExercise.target_joint === 'left_shoulder' ||
    selectedExercise.target_joint === 'right_shoulder'
      ? [
          {
            label:
              selectedExercise.target_joint === 'left_shoulder'
                ? 'ยกเฉพาะแขนซ้ายตามท่าที่กำหนด'
                : 'ยกเฉพาะแขนขวาตามท่าที่กำหนด',
            ok: !wrongLimbAlert && !isEmergency,
            warnText:
              selectedExercise.target_joint === 'left_shoulder'
                ? 'กรุณายกแขนซ้าย'
                : 'กรุณายกแขนขวา',
          },
        ]
      : []),
    {
      label: 'ลำตัวตั้งตรง ไม่เอียงชดเชย',
      ok: !trunkLeanAlert && !isEmergency,
      warnText: 'เอียงลำตัว',
    },
    {
      label:
        selectedExercise.target_joint === 'right_knee'
          ? 'เข่าตรงแนว ไม่บิดเข้าด้านใน'
          : 'ผ่อนหัวไหล่ ไม่ยักไหล่ขึ้น',
      ok:
        selectedExercise.target_joint === 'right_knee'
          ? !kneeValgusAlert && !isEmergency
          : !shoulderHikeAlert && !isEmergency,
      warnText: selectedExercise.target_joint === 'right_knee' ? 'เข่าบิดใน' : 'ยักไหล่',
    },
    {
      label: 'ความเร็วเคลื่อนไหวสม่ำเสมอ',
      ok: !tooFastAlert && !isEmergency,
      warnText: 'เร็วเกินไป',
    },
    {
      label: 'ยืนยันตัวตนคนไข้ตัวจริง (1 คนในเฟรม)',
      ok:
        kiosk_state !== 'PAUSED_MULTI_PERSON' &&
        kiosk_state !== 'PAUSED_IMPOSTOR' &&
        payload.identity_guard?.is_counting_allowed !== false &&
        !isEmergency,
      warnText:
        kiosk_state === 'PAUSED_IMPOSTOR' || payload.identity_guard?.status === 'IMPOSTOR'
          ? 'ไม่ใช่คนไข้'
          : 'พบหลายคน',
    },
  ];

  return (
    <div className="clinical-mist-bg relative min-h-screen w-full overflow-x-hidden text-[#0F4C81]">
      {/* 1. Ambient Breathing Aurora Background */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <motion.div
          animate={{ x: [0, 36, -18, 0], y: [0, -24, 20, 0], scale: [1, 1.08, 0.96, 1] }}
          transition={{ repeat: Infinity, duration: 18, ease: 'easeInOut' }}
          className="absolute -left-24 -top-24 h-96 w-96 rounded-full bg-[#A7F3D0]/55 blur-3xl"
        />
        <motion.div
          animate={{ x: [0, -40, 24, 0], y: [0, 28, -16, 0], scale: [1, 0.94, 1.07, 1] }}
          transition={{ repeat: Infinity, duration: 22, ease: 'easeInOut' }}
          className="absolute -bottom-24 -right-24 h-[440px] w-[440px] rounded-full bg-[#BAE6FD]/60 blur-3xl"
        />
        <motion.div
          animate={{ opacity: [0.25, 0.45, 0.25] }}
          transition={{ repeat: Infinity, duration: 10, ease: 'easeInOut' }}
          className="absolute left-1/2 top-1/3 h-80 w-80 -translate-x-1/2 rounded-full bg-[#E0F2FE]/70 blur-3xl"
        />
      </div>

      {/* Main Container */}
      <div className="relative z-10 mx-auto flex min-h-screen max-w-[1600px] flex-col justify-between gap-4 p-3 sm:p-5">
        {/* 2. Top Status Bar (Glass Header) */}
        <header className="glass-card flex flex-wrap items-center justify-between gap-3 rounded-3xl px-4 py-3 sm:px-6">
          {/* Left: Logo + Edge Vision Connection Status */}
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-2.5">
              <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-gradient-to-br from-[#0284C7] to-[#10B981] text-white shadow-md">
                <HeartPulse className="h-6 w-6" />
              </div>
              <div>
                <h1 className="text-lg font-extrabold tracking-tight text-[#0F4C81] sm:text-xl">
                  DRTECH<span className="text-[#0D9488]">CARES</span>
                </h1>
                <p className="text-[11px] font-semibold text-[#0284C7]">Smart Rehabilitation Kiosk</p>
              </div>
            </div>

            <div
              className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-bold ${
                !isDemoMode && connectionStatus === 'connected'
                  ? 'border-emerald-200 bg-emerald-50 text-[#0D9488]'
                  : 'border-sky-200 bg-sky-50 text-[#0284C7]'
              }`}
            >
              <span className="relative flex h-2.5 w-2.5">
                <span
                  className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-75 ${
                    !isDemoMode && connectionStatus === 'connected' ? 'bg-[#10B981]' : 'bg-[#38BDF8]'
                  }`}
                />
                <span
                  className={`relative inline-flex h-2.5 w-2.5 rounded-full ${
                    !isDemoMode && connectionStatus === 'connected' ? 'bg-[#10B981]' : 'bg-[#0284C7]'
                  }`}
                />
              </span>
              {!isDemoMode && connectionStatus === 'connected' ? (
                <>
                  <Wifi className="h-3.5 w-3.5" />
                  <span>Connected to Edge Vision</span>
                </>
              ) : (
                <>
                  <WifiOff className="h-3.5 w-3.5" />
                  <span>{connectionStatus === 'connected' ? 'Demo Simulator (Live Paused)' : 'Demo Simulator Mode'}</span>
                </>
              )}
            </div>
          </div>

          {/* Center: Patient Profile Badge */}
          <AnimatePresence>
            {patientRevealed && patient && (
              <motion.div
                initial={{ opacity: 0, y: -10, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -10 }}
                className="flex items-center gap-3 rounded-2xl border border-emerald-200/90 bg-white/90 px-4 py-1.5 shadow-sm"
              >
                <div className="flex h-8 w-8 items-center justify-center rounded-full bg-emerald-100 text-[#0D9488]">
                  <UserCheck className="h-4 w-4" />
                </div>
                <div className="leading-tight">
                  <p className="text-sm font-bold text-[#0F4C81]">{patient.name}</p>
                  <p className="tabular-mono text-[11px] font-semibold text-[#0284C7]">
                    {patient.id} • ความคืบหน้า {rep_counter.valid_reps}/{selectedExercise.target_reps} ครั้ง
                  </p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Right: Live Camera Toggle + Demo Mode Switch + Portal Switch + Emergency Stop */}
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => setLivePoseActive((v) => !v)}
              className={`inline-flex items-center gap-1.5 rounded-2xl border px-3 py-2 text-xs font-bold transition-all sm:text-sm ${
                livePoseActive
                  ? 'border-emerald-300 bg-emerald-50 text-[#0D9488]'
                  : 'border-slate-200 bg-white/80 text-slate-600 hover:bg-slate-50'
              }`}
            >
              <Camera className="h-4 w-4" />
              <span>{livePoseActive ? 'กล้อง AI: ON' : 'กล้อง AI: OFF'}</span>
            </button>

            <button
              type="button"
              onClick={() => toggleDemoMode()}
              className={`inline-flex items-center gap-2 rounded-2xl border px-3.5 py-2 text-xs font-bold transition-all sm:text-sm ${
                isDemoMode
                  ? 'border-[#38BDF8] bg-[#E0F2FE] text-[#0F4C81]'
                  : 'border-emerald-300 bg-emerald-50 text-[#0D9488]'
              }`}
            >
              <SlidersHorizontal className="h-4 w-4" />
              <span>{isDemoMode ? 'Demo Simulator: ON' : 'Live Edge: ON'}</span>
            </button>

            {onOpenLegacyKiosk && (
              <button
                type="button"
                onClick={onOpenLegacyKiosk}
                className="rounded-2xl border border-sky-200 bg-white/80 px-3 py-2 text-xs font-bold text-[#0F4C81] hover:bg-sky-50"
              >
                โหมดสแกนเต็มระบบ
              </button>
            )}

            {onOpenStaffPortal && (
              <button
                type="button"
                onClick={onOpenStaffPortal}
                className="rounded-2xl border border-sky-200 bg-white/80 px-3 py-2 text-xs font-bold text-[#0284C7] hover:bg-sky-50"
              >
                Staff Portal
              </button>
            )}

            <button
              type="button"
              onClick={handleEmergencyStopClick}
              className="inline-flex items-center gap-2 rounded-2xl bg-[#EF4444] px-4 py-2 text-xs font-extrabold text-white shadow-md transition-transform hover:bg-red-600 active:scale-95 sm:text-sm"
            >
              <AlertOctagon className="h-4 w-4" />
              <span>หยุดฉุกเฉิน</span>
            </button>
          </div>
        </header>

        {/* Completion Banner when all prescribed exercises are finished */}
        {allExercisesCompleted && onAllExercisesCompleted && (
          <div className="glass-card flex flex-wrap items-center justify-between gap-3 rounded-3xl border-2 border-emerald-400 bg-emerald-50/95 px-5 py-3.5 shadow-lg">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-emerald-500 text-white shadow">
                <CheckCircle2 className="h-5 w-5" />
              </div>
              <div>
                <p className="text-sm font-extrabold text-emerald-950">
                  ทำท่ากายภาพบำบัดครบทั้งหมด {exerciseOptions.length} ท่าแล้ว!
                </p>
                <p className="text-xs font-semibold text-emerald-800">
                  ระบบกำลังพาเข้าสู่ขั้นตอนสุดท้าย: มินิเกมฝึกสมองและการ์ดกายภาพ...
                </p>
              </div>
            </div>
            <button
              type="button"
              onClick={onAllExercisesCompleted}
              className="inline-flex items-center gap-2 rounded-2xl bg-gradient-to-r from-emerald-600 to-teal-600 px-5 py-2.5 text-xs font-extrabold text-white shadow-md hover:from-emerald-500 hover:to-teal-500 cursor-pointer sm:text-sm"
            >
              <Gamepad2 className="h-4 w-4" />
              <span>เข้าสู่มินิเกม (ขั้นตอนสุดท้าย) ➔</span>
            </button>
          </div>
        )}

        {/* 3. Main Clinical Workspace: 16:9 Landscape (~65% Left / ~35% Right), 9:16 Portrait Responsive */}
        <main className="grid flex-1 grid-cols-1 gap-4 lg:grid-cols-12">
          {/* Left Stage (~65% width) */}
          <section className="lg:col-span-8">
            <SkeletonCanvas
              payload={payload}
              exercise={selectedExercise}
              isDemoMode={isDemoMode}
              distanceM={demoDistanceM}
              trunkTiltDeg={demoTrunkTiltDeg}
              hasLiveCamera={livePoseActive && !cameraError}
              liveVideoOverlay={
                livePoseActive && !cameraError ? (
                  <div className="relative h-full w-full bg-gradient-to-br from-sky-100 via-emerald-50 to-white">
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      muted
                      className="h-full w-full object-cover"
                      style={{ transform: 'scaleX(-1)' }}
                    />
                    <canvas
                      ref={poseCanvasRef}
                      aria-label="Live Pose Skeleton & Angle Overlay"
                      className="pointer-events-none absolute inset-0 h-full w-full"
                    />
                    <div className="pointer-events-none absolute left-4 top-16 z-10 inline-flex items-center gap-1.5 rounded-full border border-emerald-300 bg-white/90 px-3 py-1 text-[11px] font-bold text-[#0F4C81] shadow-sm backdrop-blur-md">
                      <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-400" />
                      <span>
                        {cameraReady ? '📷 กล้องตรวจจับท่าทางสด (Live AI Pose)' : '⏳ กำลังเปิดกล้อง...'}
                      </span>
                    </div>
                  </div>
                ) : undefined
              }
            >
              {kiosk_state === 'AUTH_FRAMING' && (
                <FaceScanOverlay
                  verified={demoFaceMatched || Boolean(patient?.id)}
                  streakFrames={demoStreakFrames}
                  requiredFrames={5}
                  distanceM={demoDistanceM}
                  patientName={patient?.name}
                />
              )}

              {isEmergency && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="absolute inset-0 z-30 flex flex-col items-center justify-center bg-rose-950/75 p-6 text-center text-white backdrop-blur-sm"
                >
                  <AlertOctagon className="h-20 w-20 text-rose-400" />
                  <h2 className="mt-4 text-3xl font-extrabold">หยุดพักฉุกเฉิน (EMERGENCY HALT)</h2>
                  <p className="mt-2 max-w-md text-base text-rose-100">
                    ระบบตรวจพบสัญญาณหยุดพักฉุกเฉินและได้แจ้งเตือนเจ้าหน้าที่พยาบาลเรียบร้อยแล้ว
                  </p>
                  <button
                    type="button"
                    onClick={resetDemoSession}
                    className="mt-6 inline-flex items-center gap-2 rounded-2xl bg-white px-6 py-3 text-sm font-bold text-rose-700 shadow-lg hover:bg-rose-50"
                  >
                    <RotateCcw className="h-4 w-4" />
                    รีเซ็ตสถานะเพื่อฝึกต่อ
                  </button>
                </motion.div>
              )}
            </SkeletonCanvas>
          </section>

          {/* Right Telemetry Panel (~35% width) */}
          <aside className="flex flex-col gap-4 lg:col-span-4">
            {/* Card 1: Exercise Prescription Header */}
            <div className="glass-card rounded-3xl p-5">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <span className="inline-flex items-center gap-1.5 rounded-full bg-sky-100/90 px-3 py-1 text-[11px] font-bold text-[#0284C7]">
                    <Sparkles className="h-3.5 w-3.5" />
                    ท่าที่ {currentExerciseIndex + 1} จาก {exerciseOptions.length} • เซต{' '}
                    {selectedExercise.target_sets ?? 1} × {selectedExercise.target_reps} ครั้ง
                  </span>
                  <h2 className="mt-2 text-xl font-extrabold text-[#0F4C81]">{selectedExercise.name_th}</h2>
                  <p className="text-xs font-semibold text-[#0284C7]">{selectedExercise.name_en}</p>
                  {(selectedExercise.target_joint === 'left_shoulder' ||
                    selectedExercise.target_joint === 'right_shoulder') && (
                    <div
                      className={`mt-2 inline-flex items-center gap-1.5 rounded-xl border px-3 py-1 text-xs font-extrabold ${
                        selectedExercise.target_joint === 'left_shoulder'
                          ? 'border-orange-300 bg-orange-50 text-orange-800'
                          : 'border-emerald-300 bg-emerald-50 text-emerald-800'
                      }`}
                    >
                      <span>✋ ท่านี้ต้องยกเฉพาะ:</span>
                      <span>
                        {selectedExercise.target_joint === 'left_shoulder'
                          ? 'แขนซ้าย (Left Arm Only)'
                          : 'แขนขวา (Right Arm Only)'}
                      </span>
                    </div>
                  )}
                </div>

                {/* Animated Miniature Guide Illustration */}
                <div className="flex h-16 w-16 shrink-0 items-center justify-center rounded-2xl border border-sky-100 bg-gradient-to-br from-sky-50 to-emerald-50">
                  <svg viewBox="0 0 64 64" className="h-12 w-12">
                    <circle cx="32" cy="14" r="5" fill="#0F4C81" />
                    <line x1="32" y1="19" x2="32" y2="40" stroke="#0F4C81" strokeWidth="3.5" strokeLinecap="round" />
                    <line x1="32" y1="40" x2="24" y2="56" stroke="#0284C7" strokeWidth="3" strokeLinecap="round" />
                    <line x1="32" y1="40" x2="40" y2="56" stroke="#0284C7" strokeWidth="3" strokeLinecap="round" />
                    <motion.line
                      x1="32"
                      y1="24"
                      x2="14"
                      y2="34"
                      stroke="#10B981"
                      strokeWidth="3.5"
                      strokeLinecap="round"
                      animate={{ x2: [18, 12, 18], y2: [38, 10, 38] }}
                      transition={{ repeat: Infinity, duration: 3, ease: 'easeInOut' }}
                    />
                  </svg>
                </div>
              </div>

              <div className="mt-3 flex flex-wrap gap-1.5">
                {exerciseOptions.map((opt, idx) => {
                  const active = opt.exercise_id === selectedExercise.exercise_id;
                  const done = completedIds.includes(opt.exercise_id);
                  return (
                    <button
                      key={opt.exercise_id}
                      type="button"
                      onClick={() => setSelectedExercise(opt)}
                      className={`inline-flex items-center gap-1 rounded-xl px-2.5 py-1.5 text-xs font-bold transition-colors ${
                        active
                          ? 'bg-[#0F4C81] text-white shadow-sm'
                          : done
                            ? 'bg-emerald-100 text-emerald-900 border border-emerald-300'
                            : 'bg-sky-50 text-[#0F4C81] hover:bg-sky-100'
                      }`}
                    >
                      <span>{done ? '✓' : `${idx + 1}.`}</span>
                      <span>{opt.name_th}</span>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Card 2: Hero ROM Angle Gauge */}
            <RomGauge
              currentRomDeg={telemetry_analysis.current_rom_deg}
              targetRomDeg={telemetry_analysis.target_rom_deg}
              completionPct={telemetry_analysis.rom_completion_pct}
              kioskState={kiosk_state}
              highlightColor={ui_directives.highlight_color}
              holdRemainingSec={rep_counter.hold_remaining_sec}
              holdTargetSec={selectedExercise.hold_sec}
            />

            {/* Card 3: Repetition & Hold Counter */}
            <div className="glass-card relative overflow-hidden rounded-3xl p-5">
              {/* Emerald Shockwave Ripple on Rep Completion */}
              <AnimatePresence>
                {isRepCelebration && (
                  <motion.div
                    key={`ripple-${rep_counter.valid_reps}`}
                    initial={{ scale: 0.75, opacity: 0.65 }}
                    animate={{ scale: 1.45, opacity: 0 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.85, ease: 'easeOut' }}
                    className="pointer-events-none absolute inset-0 rounded-3xl border-4 border-[#10B981]"
                  />
                )}
              </AnimatePresence>

              <div className="grid grid-cols-2 gap-3">
                {/* Valid Reps */}
                <div className="rounded-2xl border border-emerald-100 bg-gradient-to-br from-white to-emerald-50/70 p-3.5">
                  <p className="text-xs font-bold text-[#0D9488]">จำนวนครั้งที่ถูกต้อง</p>
                  <div className="mt-1 flex items-baseline gap-1.5">
                    <motion.span
                      key={rep_counter.valid_reps}
                      initial={{ y: -10, scale: 1.25, opacity: 0.4 }}
                      animate={{ y: 0, scale: 1, opacity: 1 }}
                      transition={{ type: 'spring', stiffness: 320, damping: 18 }}
                      className="tabular-mono text-3xl font-extrabold text-[#0F4C81]"
                    >
                      {rep_counter.valid_reps}
                    </motion.span>
                    <span className="tabular-mono text-sm font-bold text-[#0F4C81]/60">
                      / {selectedExercise.target_reps} ครั้ง
                    </span>
                  </div>
                </div>

                {/* Live Hold Timer */}
                <div
                  className={`rounded-2xl border p-3.5 transition-colors ${
                    isHoldPhase
                      ? 'border-emerald-300 bg-emerald-500 text-white'
                      : 'border-sky-100 bg-gradient-to-br from-white to-sky-50/70 text-[#0F4C81]'
                  }`}
                >
                  <p className={`flex items-center gap-1 text-xs font-bold ${isHoldPhase ? 'text-emerald-50' : 'text-[#0284C7]'}`}>
                    <Timer className="h-3.5 w-3.5" />
                    ค้างท่าเป้าหมาย
                  </p>
                  <div className="tabular-mono mt-1 text-3xl font-extrabold">
                    {rep_counter.hold_remaining_sec.toFixed(1)}s
                  </div>
                  {(() => {
                    const holdTargetSec = Math.max(0.5, selectedExercise.hold_sec);
                    const holdRatio =
                      kiosk_state === 'REP_COMPLETED' || kiosk_state === 'SET_FINISHED'
                        ? 1
                        : isHoldPhase || rep_counter.hold_remaining_sec < holdTargetSec
                          ? Math.max(
                              0,
                              Math.min(1, (holdTargetSec - rep_counter.hold_remaining_sec) / holdTargetSec),
                            )
                          : 0;
                    return (
                      <div
                        className={`mt-2 h-1.5 w-full overflow-hidden rounded-full ${
                          isHoldPhase ? 'bg-white/30' : 'bg-sky-100'
                        }`}
                      >
                        <div
                          className={`h-full w-full origin-left rounded-full ${
                            isHoldPhase ? 'bg-white' : 'bg-gradient-to-r from-[#0284C7] to-[#10B981]'
                          }`}
                          style={{
                            transform: `scaleX(${holdRatio})`,
                            willChange: 'transform',
                          }}
                        />
                      </div>
                    );
                  })()}
                </div>
              </div>

              {/* Rep Visual Dots */}
              <div className="mt-3.5 flex items-center gap-1.5">
                {Array.from({ length: selectedExercise.target_reps }).map((_, idx) => {
                  const done = idx < rep_counter.valid_reps;
                  return (
                    <motion.div
                      key={idx}
                      animate={{ scale: done ? [1, 1.15, 1] : 1 }}
                      transition={{ duration: 0.25 }}
                      className={`h-2.5 flex-1 rounded-full transition-colors ${
                        done ? 'bg-[#10B981]' : 'bg-sky-100'
                      }`}
                    />
                  );
                })}
              </div>

              {/* Action Controls: Count Rep / Complete Current Exercise -> Next or Final Minigame */}
              <div className="mt-3.5 flex flex-col gap-2">
                <button
                  type="button"
                  onClick={() => setDemoPreset('REP_COMPLETED')}
                  disabled={rep_counter.valid_reps >= selectedExercise.target_reps}
                  className="inline-flex w-full items-center justify-center gap-1.5 rounded-xl border border-sky-200 bg-sky-50 px-3 py-2 text-xs font-bold text-[#0F4C81] hover:bg-sky-100 disabled:opacity-50 cursor-pointer"
                >
                  <PlusCircle className="h-3.5 w-3.5" />
                  <span>+1 ครั้ง (นับครั้งสำเร็จ)</span>
                </button>

                {nextUncompletedExercise ? (
                  <button
                    type="button"
                    onClick={() => {
                      markCurrentExerciseCompleted();
                      setSelectedExercise(nextUncompletedExercise);
                    }}
                    className="inline-flex w-full items-center justify-center gap-1.5 rounded-xl bg-gradient-to-r from-[#0284C7] to-[#10B981] px-3 py-2.5 text-xs font-extrabold text-white shadow hover:opacity-95 cursor-pointer"
                  >
                    <CheckCircle2 className="h-4 w-4" />
                    <span>ทำท่านี้ครบแล้ว • ไปท่าถัดไป ({nextUncompletedExercise.name_th}) ➔</span>
                  </button>
                ) : (
                  onAllExercisesCompleted && (
                    <button
                      type="button"
                      onClick={() => {
                        markCurrentExerciseCompleted();
                        onAllExercisesCompleted();
                      }}
                      className="inline-flex w-full items-center justify-center gap-1.5 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 px-3 py-2.5 text-xs font-extrabold text-white shadow hover:from-emerald-500 hover:to-teal-500 cursor-pointer"
                    >
                      <Gamepad2 className="h-4 w-4" />
                      <span>ทำกายภาพครบทุกท่าแล้ว • เข้าสู่มินิเกม (ขั้นตอนสุดท้าย) ➔</span>
                    </button>
                  )
                )}
              </div>
            </div>

            {/* Card 4: Biomechanical Form Checklist */}
            <div className="glass-card rounded-3xl p-5">
              <div className="mb-3 flex items-center justify-between">
                <h3 className="flex items-center gap-2 text-sm font-extrabold text-[#0F4C81]">
                  <Activity className="h-4 w-4 text-[#0D9488]" />
                  การประเมินท่าทางแบบเรียลไทม์
                </h3>
                <span className="tabular-mono text-xs font-bold text-[#0284C7]">
                  เอียงลำตัว {demoTrunkTiltDeg.toFixed(1)}°
                </span>
              </div>

              <div className="space-y-2">
                {checklistItems.map((item) => (
                  <div
                    key={item.label}
                    className={`flex items-center justify-between rounded-2xl border px-3.5 py-2.5 text-sm font-semibold transition-colors ${
                      item.ok
                        ? 'border-emerald-100 bg-emerald-50/60 text-[#0F4C81]'
                        : 'border-amber-300 bg-amber-50 text-amber-900'
                    }`}
                  >
                    <span className="flex items-center gap-2">
                      {item.ok ? (
                        <CheckCircle2 className="h-4 w-4 shrink-0 text-[#10B981]" />
                      ) : (
                        <AlertTriangle className="h-4 w-4 shrink-0 text-[#F59E0B]" />
                      )}
                      {item.label}
                    </span>
                    <span
                      className={`rounded-full px-2 py-0.5 text-[11px] font-bold ${
                        item.ok ? 'bg-emerald-100 text-[#0D9488]' : 'bg-amber-200 text-amber-900'
                      }`}
                    >
                      {item.ok ? 'ผ่าน' : item.warnText}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </aside>
        </main>

        {/* 4. Bottom Floating Dock: Thai AI Coach + Demo Simulator Drawer */}
        <footer className="space-y-3">
          <VoiceCoachBanner
            textTh={tts_voice_th}
            kioskState={kiosk_state}
            highlightColor={ui_directives.highlight_color}
            muted={voiceMuted}
            onToggleMute={() => setVoiceMuted((m) => !m)}
          />

          {/* Interactive Demo Simulator Floating Control Drawer */}
          {isDemoMode && (
            <div className="glass-card rounded-3xl px-4 py-3 sm:px-6">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <button
                  type="button"
                  onClick={() => setDrawerOpen((o) => !o)}
                  className="flex items-center gap-2 text-xs font-extrabold uppercase tracking-wider text-[#0F4C81] sm:text-sm"
                >
                  <SlidersHorizontal className="h-4 w-4 text-[#0284C7]" />
                  <span>Demo Simulator Controls (พรีวิวสถานะ &amp; มุมข้อต่อสด)</span>
                  {drawerOpen ? <ChevronDown className="h-4 w-4" /> : <ChevronUp className="h-4 w-4" />}
                </button>

                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => setAutoCycleDemo((a) => !a)}
                    className={`inline-flex items-center gap-1.5 rounded-full px-3.5 py-1.5 text-xs font-bold transition-colors ${
                      autoCycleDemo
                        ? 'bg-[#10B981] text-white'
                        : 'bg-sky-100 text-[#0F4C81] hover:bg-sky-200'
                    }`}
                  >
                    {autoCycleDemo ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
                    <span>{autoCycleDemo ? 'หยุดจำลองอัตโนมัติ' : 'เล่นจำลองต่อเนื่อง'}</span>
                  </button>
                  <button
                    type="button"
                    onClick={resetDemoSession}
                    className="inline-flex items-center gap-1 rounded-full border border-sky-200 bg-white px-3 py-1.5 text-xs font-bold text-[#0F4C81] hover:bg-sky-50"
                  >
                    <RotateCcw className="h-3.5 w-3.5" />
                    รีเซ็ต
                  </button>
                </div>
              </div>

              {drawerOpen && (
                <div className="mt-3 space-y-3 border-t border-sky-100 pt-3">
                  {/* State Switch Pills */}
                  <div className="flex flex-wrap gap-1.5">
                    {DEMO_PRESETS.map((p) => {
                      const active = demoPreset === p.id;
                      return (
                        <button
                          key={p.id}
                          type="button"
                          onClick={() => setDemoPreset(p.id)}
                          className={`rounded-xl px-3 py-1.5 text-xs font-bold transition-all ${
                            active
                              ? p.id === 'IMPOSTOR'
                                ? 'bg-[#EF4444] text-white shadow-sm'
                                : p.id === 'TRUNK_LEAN' || p.id === 'MULTI_PERSON' || p.id === 'WRONG_LIMB'
                                  ? 'bg-[#F59E0B] text-white shadow-sm'
                                  : 'bg-[#0284C7] text-white shadow-sm'
                              : 'bg-white/80 text-[#0F4C81] hover:bg-sky-50'
                          }`}
                        >
                          {p.label}
                        </button>
                      );
                    })}
                  </div>

                  {/* Sliders: current_rom_deg, trunk_tilt_deg, distance_m */}
                  <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
                    <label className="flex flex-col gap-1">
                      <div className="flex justify-between text-xs font-bold">
                        <span>มุมข้อต่อ (current_rom_deg)</span>
                        <span className="tabular-mono text-[#0284C7]">{demoRomDeg.toFixed(0)}°</span>
                      </div>
                      <input
                        type="range"
                        min={0}
                        max={180}
                        step={1}
                        value={demoRomDeg}
                        onChange={(e) => {
                          setAutoCycleDemo(false);
                          setDemoRomDeg(Number(e.target.value));
                        }}
                        className="accent-[#0284C7]"
                      />
                    </label>

                    <label className="flex flex-col gap-1">
                      <div className="flex justify-between text-xs font-bold">
                        <span>เอียงลำตัว (trunk_tilt_deg)</span>
                        <span
                          className={`tabular-mono ${
                            demoTrunkTiltDeg > 10 ? 'text-[#F59E0B]' : 'text-[#0D9488]'
                          }`}
                        >
                          {demoTrunkTiltDeg.toFixed(1)}°
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0}
                        max={25}
                        step={0.5}
                        value={demoTrunkTiltDeg}
                        onChange={(e) => {
                          setAutoCycleDemo(false);
                          setDemoTrunkTiltDeg(Number(e.target.value));
                        }}
                        className="accent-[#F59E0B]"
                      />
                    </label>

                    <label className="flex flex-col gap-1">
                      <div className="flex justify-between text-xs font-bold">
                        <span>ระยะยืนหน้าตู้ (distance_m)</span>
                        <span className="tabular-mono text-[#0D9488]">{demoDistanceM.toFixed(2)} ม.</span>
                      </div>
                      <input
                        type="range"
                        min={1.4}
                        max={2.6}
                        step={0.05}
                        value={demoDistanceM}
                        onChange={(e) => setDemoDistanceM(Number(e.target.value))}
                        className="accent-[#10B981]"
                      />
                    </label>
                  </div>
                </div>
              )}
            </div>
          )}
        </footer>
      </div>
    </div>
  );
}

