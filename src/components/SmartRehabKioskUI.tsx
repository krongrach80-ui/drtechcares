import React, { useEffect, useMemo, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { FilesetResolver, PoseLandmarker, type NormalizedLandmark } from '@mediapipe/tasks-vision';
import { selectClinicalCameraDeviceId } from '@drtechcares/face-client';
import { POSE_INDEX, ClinicalOneEuroLandmarkFilter, computeJointAngle3D } from '@drtechcares/pose-engine';
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
  getPrescribedPlanePromptTh,
  resolveExpectedPlaneForExercise,
  useKioskSocket,
  type DemoPresetId,
  type ExercisePrescriptionOption,
  type ExpectedMovementPlane,
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
  { id: 'WRONG_PLANE', label: 'Wrong Plane (ยกผิดทิศทาง)', shortLabel: '3c. Wrong Plane' },
  { id: 'HOLD_PHASE', label: 'Hold Phase', shortLabel: '4. Hold' },
  { id: 'REP_COMPLETED', label: 'Rep Completed', shortLabel: '5. Rep +' },
  { id: 'SET_FINISHED', label: 'Set Finished', shortLabel: '6. Finished' },
  { id: 'MULTI_PERSON', label: 'Multi-Person Alert (2+ คน)', shortLabel: '7. Multi-Person' },
  { id: 'IMPOSTOR', label: 'Impostor Blocked (ผิดคน)', shortLabel: '8. Impostor' },
];

export type FormAssessmentRuleKey =
  | 'limb'
  | 'plane'
  | 'trunk'
  | 'posture'
  | 'velocity'
  | 'identity';

export interface FormAssessmentRulesConfig {
  enabled: boolean;
  limb: boolean;
  plane: boolean;
  trunk: boolean;
  posture: boolean;
  velocity: boolean;
  identity: boolean;
}

export const DEFAULT_FORM_ASSESSMENT_RULES: FormAssessmentRulesConfig = {
  enabled: true,
  limb: true,
  plane: true,
  trunk: true,
  posture: true,
  velocity: true,
  identity: true,
};

const FORM_ASSESSMENT_STORAGE_KEY = 'drtechcares_form_assessment_rules_v1';

export function isFormRuleActive(
  config: FormAssessmentRulesConfig,
  rule: FormAssessmentRuleKey,
): boolean {
  return Boolean(config.enabled && config[rule]);
}

export interface LiveJointMeasurement {
  romDeg: number | null;
  rawPrimaryRomDeg?: number | null;
  trunkTiltDeg: number;
  wrongSideRaised: boolean;
  oppositeRomDeg: number | null;
  wrongSideBannerTh: string | null;
  wrongPlaneRaised: boolean;
  wrongPlaneBannerTh: string | null;
  xyzVector?: { x: number; y: number; z: number } | null;
  lowVisibilityAlertTh?: string | null;
  postureCompensated?: boolean;
  postureBannerTh?: string | null;
  elbowAngleDeg?: number | null;
  shoulderTiltDeg?: number;
}

export interface TorsoAnchorState {
  leftShoulder: NormalizedLandmark;
  rightShoulder: NormalizedLandmark;
  nose?: NormalizedLandmark;
  spanX: number;
}

const LOWER_BODY_POSE_INDICES = [
  POSE_INDEX.LEFT_HIP,
  POSE_INDEX.RIGHT_HIP,
  POSE_INDEX.LEFT_KNEE,
  POSE_INDEX.RIGHT_KNEE,
  POSE_INDEX.LEFT_ANKLE,
  POSE_INDEX.RIGHT_ANKLE,
] as const;

/**
 * Validates anatomical torso landmarks and prevents MediaPipe self-occlusion collapse
 * (e.g., shoulders/hips jumping onto an outstretched hand approaching the camera lens).
 */
export function sanitizeAndStabilizePoseLandmarks(
  rawLandmarks: readonly NormalizedLandmark[],
  prevAnchor: TorsoAnchorState | null,
  targetJoint: ExercisePrescriptionOption['target_joint'],
): { landmarks: NormalizedLandmark[]; anchor: TorsoAnchorState | null; occludedFallback: boolean } {
  const stabilized = rawLandmarks.map((lm) => ({ ...lm }));
  const lSh = stabilized[POSE_INDEX.LEFT_SHOULDER];
  const rSh = stabilized[POSE_INDEX.RIGHT_SHOULDER];
  const nose = stabilized[POSE_INDEX.NOSE];
  const lWrist = stabilized[POSE_INDEX.LEFT_WRIST];
  const rWrist = stabilized[POSE_INDEX.RIGHT_WRIST];

  let occludedFallback = false;
  let nextAnchor = prevAnchor;

  const hasShoulderVis = Boolean(
    lSh && rSh && (lSh.visibility ?? 0) >= 0.15 && (rSh.visibility ?? 0) >= 0.15,
  );
  const rawSpanX = lSh && rSh ? Math.abs(lSh.x - rSh.x) : 0;

  const isWristCollapsedCluster = (wrist: NormalizedLandmark | undefined) => {
    if (!wrist || (wrist.visibility ?? 0) < 0.35 || !lSh || !rSh) return false;
    const dL = Math.hypot(lSh.x - wrist.x, lSh.y - wrist.y);
    const dR = Math.hypot(rSh.x - wrist.x, rSh.y - wrist.y);
    return dL < 0.075 && dR < 0.075;
  };

  const shouldersAnatomical =
    hasShoulderVis &&
    rawSpanX >= 0.085 &&
    !isWristCollapsedCluster(rWrist) &&
    !isWristCollapsedCluster(lWrist);

  if (shouldersAnatomical && lSh && rSh) {
    nextAnchor = {
      leftShoulder: { ...lSh },
      rightShoulder: { ...rSh },
      nose:
        nose && (nose.visibility ?? 0) >= 0.25
          ? { ...nose }
          : prevAnchor?.nose
            ? { ...prevAnchor.nose }
            : undefined,
      spanX: rawSpanX,
    };
  } else if (prevAnchor) {
    occludedFallback = true;
    stabilized[POSE_INDEX.LEFT_SHOULDER] = {
      ...prevAnchor.leftShoulder,
      visibility: Math.max(0.65, prevAnchor.leftShoulder.visibility ?? 0.65),
    };
    stabilized[POSE_INDEX.RIGHT_SHOULDER] = {
      ...prevAnchor.rightShoulder,
      visibility: Math.max(0.65, prevAnchor.rightShoulder.visibility ?? 0.65),
    };
    if (prevAnchor.nose) {
      const midX = (prevAnchor.leftShoulder.x + prevAnchor.rightShoulder.x) / 2;
      const midY = (prevAnchor.leftShoulder.y + prevAnchor.rightShoulder.y) / 2;
      const noseOk =
        nose &&
        (nose.visibility ?? 0) >= 0.25 &&
        Math.hypot(nose.x - midX, nose.y - midY) <= prevAnchor.spanX * 1.85;
      if (!noseOk) {
        stabilized[POSE_INDEX.NOSE] = {
          ...prevAnchor.nose,
          visibility: Math.max(0.65, prevAnchor.nose.visibility ?? 0.65),
        };
      }
    }
  }

  const effLSh = stabilized[POSE_INDEX.LEFT_SHOULDER];
  const effRSh = stabilized[POSE_INDEX.RIGHT_SHOULDER];

  // Suppress anatomically impossible chest/hand-height hallucinated hips/knees/ankles across all exercises
  const maxShY = effLSh && effRSh ? Math.max(effLSh.y, effRSh.y) : 0.5;
  for (const lowerIdx of LOWER_BODY_POSE_INDICES) {
    const lm = stabilized[lowerIdx];
    const isHipIdx = lowerIdx === POSE_INDEX.LEFT_HIP || lowerIdx === POSE_INDEX.RIGHT_HIP;
    const maxHallucinatedOffset = isHipIdx ? 0.05 : 0.12;
    if (lm && lm.y <= maxShY + maxHallucinatedOffset) {
      stabilized[lowerIdx] = { ...lm, visibility: 0 };
    }
  }

  // Repair overhead out-of-frame wrist foldback when hand exits the top of the camera frame
  if (targetJoint === 'left_shoulder' || targetJoint === 'right_shoulder') {
    const shIdx =
      targetJoint === 'left_shoulder' ? POSE_INDEX.LEFT_SHOULDER : POSE_INDEX.RIGHT_SHOULDER;
    const elbowIdx =
      targetJoint === 'left_shoulder' ? POSE_INDEX.LEFT_ELBOW : POSE_INDEX.RIGHT_ELBOW;
    const wristIdx =
      targetJoint === 'left_shoulder' ? POSE_INDEX.LEFT_WRIST : POSE_INDEX.RIGHT_WRIST;
    const activeSh = stabilized[shIdx];
    const activeElbow = stabilized[elbowIdx];
    const activeWrist = stabilized[wristIdx];
    if (
      activeSh &&
      activeElbow &&
      (activeElbow.visibility ?? 0) >= 0.3 &&
      activeElbow.y < activeSh.y - 0.08 &&
      activeWrist &&
      activeWrist.y > activeElbow.y + 0.08 &&
      (activeWrist.visibility ?? 0) < 0.65
    ) {
      stabilized[wristIdx] = {
        x: activeElbow.x + (activeElbow.x - activeSh.x) * 0.85,
        y: activeElbow.y + (activeElbow.y - activeSh.y) * 0.85,
        z:
          activeElbow.z !== undefined && activeSh.z !== undefined
            ? activeElbow.z + (activeElbow.z - activeSh.z) * 0.85
            : (activeElbow.z ?? 0),
        visibility: Math.max(0.55, activeElbow.visibility ?? 0.55),
      };
    }
  }

  return { landmarks: stabilized, anchor: nextAnchor, occludedFallback };
}

export type ExerciseJointMode = 'elbow' | 'neck' | 'hip_march' | 'default';

export interface ExerciseMetaHint {
  exercise_id?: string;
  catalog_exercise_id?: string;
  name_th?: string;
  name_en?: string;
}

export function classifyExerciseJointMode(
  exerciseMeta?: ExerciseMetaHint,
): ExerciseJointMode {
  if (!exerciseMeta) return 'default';
  const id = (exerciseMeta.catalog_exercise_id || exerciseMeta.exercise_id || '').toLowerCase();
  const name = `${exerciseMeta.name_th ?? ''} ${exerciseMeta.name_en ?? ''}`.toLowerCase();
  if (
    id === 'ex-05' ||
    id === 'ex-06' ||
    id.includes('elb') ||
    name.includes('ข้อศอก') ||
    name.includes('elbow')
  ) {
    return 'elbow';
  }
  if (
    id === 'ex-19' ||
    id === 'ex-20' ||
    id.includes('neck') ||
    name.includes('เอียงคอ') ||
    name.includes('neck')
  ) {
    return 'neck';
  }
  if (
    id === 'ex-15' ||
    id === 'ex-16' ||
    id.includes('hip_flex') ||
    name.includes('ยกเข่าสูง') ||
    name.includes('march')
  ) {
    return 'hip_march';
  }
  return 'default';
}

export function measureLiveJointAngleAndTilt(
  landmarks: readonly NormalizedLandmark[],
  targetJoint: ExercisePrescriptionOption['target_joint'],
  expectedPlane: ExpectedMovementPlane = 'any',
  worldLandmarks?: readonly NormalizedLandmark[],
  exerciseMeta?: ExerciseMetaHint,
): LiveJointMeasurement {
  const UPPER_MIN_VIS = 0.02;
  const LOWER_MIN_VIS = 0.2;
  const jointMode = classifyExerciseJointMode(exerciseMeta);
  const lSh = landmarks[POSE_INDEX.LEFT_SHOULDER];
  const rSh = landmarks[POSE_INDEX.RIGHT_SHOULDER];
  const lHip = landmarks[POSE_INDEX.LEFT_HIP];
  const rHip = landmarks[POSE_INDEX.RIGHT_HIP];

  const shoulderSpanX = lSh && rSh ? Math.max(0.12, Math.abs(lSh.x - rSh.x)) : 0.18;
  const shouldersSeparated = Boolean(lSh && rSh && Math.abs(lSh.x - rSh.x) >= 0.11);
  const hipsVisible = Boolean(
    lSh &&
      rSh &&
      lHip &&
      rHip &&
      (lHip.visibility ?? 0) >= LOWER_MIN_VIS &&
      (rHip.visibility ?? 0) >= LOWER_MIN_VIS &&
      (lHip.y + rHip.y) / 2 <= 0.94 &&
      (lHip.y + rHip.y) / 2 - (lSh.y + rSh.y) / 2 >= 0.15,
  );
  const torsoSpanY =
    hipsVisible && lSh && rSh && lHip && rHip
      ? Math.abs((lHip.y + rHip.y) / 2 - (lSh.y + rSh.y) / 2)
      : 0;

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
    shouldersSeparated &&
    lSh &&
    rSh &&
    (lSh.visibility ?? 0) >= 0.35 &&
    (rSh.visibility ?? 0) >= 0.35
  ) {
    trunkTiltDeg =
      Math.abs((Math.atan2(rSh.y - lSh.y, Math.max(0.11, Math.abs(lSh.x - rSh.x))) * 180) / Math.PI) *
      0.55;
  }

  const computeArmElevationDeg = (
    side: 'left' | 'right',
  ): { angleDeg: number | null; xyzVector: { x: number; y: number; z: number } | null } => {
    const shIdx = side === 'left' ? POSE_INDEX.LEFT_SHOULDER : POSE_INDEX.RIGHT_SHOULDER;
    const elbowIdx = side === 'left' ? POSE_INDEX.LEFT_ELBOW : POSE_INDEX.RIGHT_ELBOW;
    const wristIdx = side === 'left' ? POSE_INDEX.LEFT_WRIST : POSE_INDEX.RIGHT_WRIST;
    const sh = landmarks[shIdx];
    const elbow = landmarks[elbowIdx];
    const wrist = landmarks[wristIdx];
    if (!sh || (sh.visibility ?? 0) < UPPER_MIN_VIS) {
      return { angleDeg: null, xyzVector: null };
    }

    const wristUpRef = hipsVisible
      ? Math.max(torsoSpanY * 1.02, shoulderSpanX * 1.88)
      : shoulderSpanX * 2.72;

    const measureSegment3D = (
      ptIdx: number,
      pt: NormalizedLandmark | undefined,
      isWrist: boolean,
    ): { angleDeg: number; xyzVector: { x: number; y: number; z: number } } | null => {
      if (!pt || (pt.visibility ?? 0) < UPPER_MIN_VIS) return null;

      const dx = (pt.x - sh.x) * 1.22;
      const dy = pt.y - sh.y;
      const dzSensor = ((pt.z ?? 0) - (sh.z ?? 0)) * 1.22;

      const refLen = isWrist ? wristUpRef : wristUpRef * 0.54;
      const len2D = Math.hypot(dx, dy);
      const dzSphere = Math.sqrt(Math.max(0, refLen * refLen - dx * dx - dy * dy));
      const isPlaneRestricted = expectedPlane === 'flexion' || expectedPlane === 'abduction';
      const sagittalWeight =
        isPlaneRestricted || Math.abs(dx) <= shoulderSpanX * 0.45
          ? 1
          : Math.max(0, 1 - (Math.abs(dx) - shoulderSpanX * 0.45) / (shoulderSpanX * 0.45));

      // When arm reaches straight toward the camera (near shoulder height, strong 2D foreshortening),
      // reconstruct forward Z depth smoothly while leaving neutral hanging/seated rest near 0..10 deg.
      const foreshortenedForward =
        expectedPlane !== 'abduction' && dy <= shoulderSpanX * 0.38
          ? Math.max(0, Math.min(1, (shoulderSpanX * 0.38 - dy) / (shoulderSpanX * 0.25))) *
            Math.max(0, Math.min(1, (refLen * 0.68 - len2D) / (refLen * 0.26)))
          : 0;
      const downForwardWeight =
        dy > 0
          ? expectedPlane === 'abduction'
            ? 0
            : Math.max(Math.min(1, Math.abs(dzSensor) / 0.14), foreshortenedForward)
          : 1;
      const absDz = Math.max(
        Math.abs(dzSensor),
        dzSphere * sagittalWeight * downForwardWeight,
      );
      const dz = (dzSensor > 1e-4 ? 1 : -1) * absDz;
      const horizMagXZ = Math.hypot(dx, absDz);
      const mag3D = Math.hypot(dx, dy, dz);
      const angle3D = mag3D >= 0.02 ? (Math.atan2(horizMagXZ, dy) * 180) / Math.PI : null;
      const xyz2DSphere =
        mag3D >= 0.02
          ? {
              x: Number((dx / mag3D).toFixed(3)),
              y: Number((dy / mag3D).toFixed(3)),
              z: Number((dz / mag3D).toFixed(3)),
            }
          : null;

      // Prefer true isotropic metric 3D coordinates when MediaPipe worldLandmarks are valid,
      // but rescue self-occluded forward-reach dropouts where 2D perspective clearly shows ~90 deg reach.
      if (worldLandmarks) {
        const wSh = worldLandmarks[shIdx];
        const wPt = worldLandmarks[ptIdx];
        if (wSh && wPt) {
          const wDx = wPt.x - wSh.x;
          const wDy = wPt.y - wSh.y;
          const wDz = (wPt.z ?? 0) - (wSh.z ?? 0);
          const wLen3D = Math.hypot(wDx, wDy, wDz);
          if (wLen3D >= 0.08) {
            const worldHorizXZ = Math.hypot(wDx, wDz);
            const worldAngleDeg = (Math.atan2(worldHorizXZ, wDy) * 180) / Math.PI;
            const stronglyForeshortenedFront =
              expectedPlane !== 'abduction' &&
              angle3D !== null &&
              angle3D >= 65 &&
              dy <= shoulderSpanX * 0.45 &&
              Math.abs(dx) <= shoulderSpanX * 0.85 &&
              len2D <= refLen * 0.68;
            if (stronglyForeshortenedFront && worldAngleDeg < angle3D - 18 && xyz2DSphere) {
              return {
                angleDeg: Number((0.15 * worldAngleDeg + 0.85 * angle3D).toFixed(1)),
                xyzVector: xyz2DSphere,
              };
            }
            return {
              angleDeg: worldAngleDeg,
              xyzVector: {
                x: Number((wDx / wLen3D).toFixed(3)),
                y: Number((wDy / wLen3D).toFixed(3)),
                z: Number((wDz / wLen3D).toFixed(3)),
              },
            };
          }
        }
      }

      if (angle3D === null || !xyz2DSphere) return null;

      return {
        angleDeg: angle3D,
        xyzVector: xyz2DSphere,
      };
    };

    const wristMeasure = measureSegment3D(wristIdx, wrist, true);
    const elbowMeasure = measureSegment3D(elbowIdx, elbow, false);

    let angleDeg: number | null = null;
    if (wristMeasure && elbowMeasure) {
      angleDeg = 0.75 * wristMeasure.angleDeg + 0.25 * elbowMeasure.angleDeg;
    } else if (wristMeasure) {
      angleDeg = wristMeasure.angleDeg;
    } else if (elbowMeasure) {
      angleDeg = elbowMeasure.angleDeg;
    }

    return {
      angleDeg,
      xyzVector: wristMeasure?.xyzVector ?? elbowMeasure?.xyzVector ?? null,
    };
  };

  const computeElbowFlexionDeg = (
    side: 'left' | 'right',
  ): { flexionDeg: number | null; xyzVector: { x: number; y: number; z: number } | null } => {
    const shIdx = side === 'left' ? POSE_INDEX.LEFT_SHOULDER : POSE_INDEX.RIGHT_SHOULDER;
    const elbowIdx = side === 'left' ? POSE_INDEX.LEFT_ELBOW : POSE_INDEX.RIGHT_ELBOW;
    const wristIdx = side === 'left' ? POSE_INDEX.LEFT_WRIST : POSE_INDEX.RIGHT_WRIST;
    const sh = landmarks[shIdx];
    const elbow = landmarks[elbowIdx];
    const wrist = landmarks[wristIdx];
    if (
      !sh ||
      !elbow ||
      !wrist ||
      (sh.visibility ?? 0) < UPPER_MIN_VIS ||
      (elbow.visibility ?? 0) < UPPER_MIN_VIS ||
      (wrist.visibility ?? 0) < UPPER_MIN_VIS
    ) {
      return { flexionDeg: null, xyzVector: null };
    }

    const wSh = worldLandmarks?.[shIdx];
    const wElbow = worldLandmarks?.[elbowIdx];
    const wWrist = worldLandmarks?.[wristIdx];
    const hasValidWorldElbow = Boolean(
      wSh &&
        wElbow &&
        wWrist &&
        Math.hypot(wSh.x - wElbow.x, wSh.y - wElbow.y, (wSh.z ?? 0) - (wElbow.z ?? 0)) >= 0.08 &&
        Math.hypot(wWrist.x - wElbow.x, wWrist.y - wElbow.y, (wWrist.z ?? 0) - (wElbow.z ?? 0)) >=
          0.08,
    );
    const interior3D =
      hasValidWorldElbow && wSh && wElbow && wWrist
        ? computeJointAngle3D(wSh, wElbow, wWrist)
        : computeJointAngle3D(sh, elbow, wrist);
    const baseFlexion = Math.max(0, Math.min(165, 175 - interior3D));

    let foreshortenedProxDeg = 0;
    if (wrist.y <= elbow.y + 0.02) {
      const upperLen2D = Math.max(0.06, Math.hypot(elbow.x - sh.x, elbow.y - sh.y));
      const shToWrist2D = Math.hypot(wrist.x - sh.x, wrist.y - sh.y);
      const ratio = shToWrist2D / upperLen2D;
      if (ratio < 1.55) {
        foreshortenedProxDeg = Math.max(0, Math.min(150, ((1.55 - ratio) / 1.15) * 140));
      }
    }

    const flexionDeg = Math.max(baseFlexion, foreshortenedProxDeg);
    const fDx = wrist.x - elbow.x;
    const fDy = wrist.y - elbow.y;
    const fDz = (wrist.z ?? 0) - (elbow.z ?? 0);
    const fMag = Math.hypot(fDx, fDy, fDz);
    const forearmVec =
      fMag >= 0.02
        ? {
            x: Number((fDx / fMag).toFixed(3)),
            y: Number((fDy / fMag).toFixed(3)),
            z: Number((fDz / fMag).toFixed(3)),
          }
        : null;

    return { flexionDeg, xyzVector: forearmVec };
  };

  let romDeg: number | null = null;
  let rawPrimaryRomDeg: number | null = null;
  let oppositeRomDeg: number | null = null;
  let wrongSideRaised = false;
  let wrongSideBannerTh: string | null = null;
  let wrongPlaneRaised = false;
  let wrongPlaneBannerTh: string | null = null;
  let xyzVector: { x: number; y: number; z: number } | null = null;
  let lowVisibilityAlertTh: string | null = null;
  let postureCompensated = false;
  let postureBannerTh: string | null = null;
  let elbowAngleDeg: number | null = null;
  const shoulderTiltDeg =
    lSh && rSh && (lSh.visibility ?? 0) >= 0.35 && (rSh.visibility ?? 0) >= 0.35
      ? Number(
          Math.abs(
            (Math.atan2(rSh.y - lSh.y, Math.max(0.11, Math.abs(lSh.x - rSh.x))) * 180) / Math.PI,
          ).toFixed(1),
        )
      : 0;

  if (jointMode === 'elbow' && (targetJoint === 'left_shoulder' || targetJoint === 'right_shoulder')) {
    const primarySide: 'left' | 'right' = targetJoint === 'left_shoulder' ? 'left' : 'right';
    const oppositeSide: 'left' | 'right' = primarySide === 'left' ? 'right' : 'left';
    const primaryElbow = computeElbowFlexionDeg(primarySide);
    const oppositeElbow = computeElbowFlexionDeg(oppositeSide);
    const primaryDeg = primaryElbow.flexionDeg;
    const oppositeDeg = oppositeElbow.flexionDeg;
    rawPrimaryRomDeg = primaryDeg !== null ? Number(primaryDeg.toFixed(1)) : null;
    oppositeRomDeg = oppositeDeg !== null ? Number(oppositeDeg.toFixed(1)) : null;
    xyzVector = primaryElbow.xyzVector;
    wrongSideRaised =
      oppositeDeg !== null &&
      oppositeDeg >= 35 &&
      (primaryDeg === null || (primaryDeg < 30 && oppositeDeg > primaryDeg + 18));
    wrongPlaneRaised = false;
    romDeg = wrongSideRaised ? 0 : rawPrimaryRomDeg;
    if (wrongSideRaised) {
      wrongSideBannerTh = getPrescribedLimbPromptTh(targetJoint).bannerAlertTh;
    }
  } else if (targetJoint === 'left_shoulder' || targetJoint === 'right_shoulder') {
    const primarySide: 'left' | 'right' = targetJoint === 'left_shoulder' ? 'left' : 'right';
    const oppositeSide: 'left' | 'right' = primarySide === 'left' ? 'right' : 'left';
    const primaryResult = computeArmElevationDeg(primarySide);
    const oppositeResult = computeArmElevationDeg(oppositeSide);
    const primaryDeg = primaryResult.angleDeg;
    const oppositeDeg = oppositeResult.angleDeg;
    rawPrimaryRomDeg = primaryDeg !== null ? Number(primaryDeg.toFixed(1)) : null;
    xyzVector = primaryResult.xyzVector;

    const pShIdx = primarySide === 'left' ? POSE_INDEX.LEFT_SHOULDER : POSE_INDEX.RIGHT_SHOULDER;
    const pElIdx = primarySide === 'left' ? POSE_INDEX.LEFT_ELBOW : POSE_INDEX.RIGHT_ELBOW;
    const pWrIdx = primarySide === 'left' ? POSE_INDEX.LEFT_WRIST : POSE_INDEX.RIGHT_WRIST;
    const pSh = landmarks[pShIdx];
    const pEl = landmarks[pElIdx];
    const pWr = landmarks[pWrIdx];
    if (
      pSh &&
      pEl &&
      pWr &&
      (pSh.visibility ?? 0) >= 0.35 &&
      (pEl.visibility ?? 0) >= 0.35 &&
      (pWr.visibility ?? 0) >= 0.35
    ) {
      const wSh = worldLandmarks?.[pShIdx];
      const wEl = worldLandmarks?.[pElIdx];
      const wWr = worldLandmarks?.[pWrIdx];
      const validWorldArm = Boolean(
        wSh &&
          wEl &&
          wWr &&
          Math.hypot(wSh.x - wEl.x, wSh.y - wEl.y, (wSh.z ?? 0) - (wEl.z ?? 0)) >= 0.08 &&
          Math.hypot(wWr.x - wEl.x, wWr.y - wEl.y, (wWr.z ?? 0) - (wEl.z ?? 0)) >= 0.08,
      );
      elbowAngleDeg =
        validWorldArm && wSh && wEl && wWr
          ? computeJointAngle3D(wSh, wEl, wWr)
          : computeJointAngle3D(pSh, pEl, pWr);
    }

    wrongSideRaised =
      oppositeDeg !== null &&
      oppositeDeg >= 45 &&
      (primaryDeg === null || (primaryDeg < 65 && oppositeDeg > primaryDeg + 18));

    if (!wrongSideRaised && primaryDeg !== null && (expectedPlane === 'flexion' || expectedPlane === 'abduction')) {
      const sh = pSh;
      const elbow = pEl;
      const wrist = pWr;
      const lIsPositiveX = !lSh || !rSh || lSh.x >= rSh.x;
      const outwardSign = lIsPositiveX
        ? primarySide === 'right'
          ? -1
          : 1
        : primarySide === 'right'
          ? 1
          : -1;
      const elbowLatRatio =
        sh && elbow && (elbow.visibility ?? 0) >= UPPER_MIN_VIS
          ? Math.abs(elbow.x - sh.x) / shoulderSpanX
          : 0;
      const wristLatRatio =
        sh && wrist && (wrist.visibility ?? 0) >= UPPER_MIN_VIS
          ? Math.abs(wrist.x - sh.x) / shoulderSpanX
          : 0;
      const elbowOutwardLatRatio =
        sh && elbow && (elbow.visibility ?? 0) >= UPPER_MIN_VIS
          ? ((elbow.x - sh.x) * outwardSign) / shoulderSpanX
          : 0;
      const wristOutwardLatRatio =
        sh && wrist && (wrist.visibility ?? 0) >= UPPER_MIN_VIS
          ? ((wrist.x - sh.x) * outwardSign) / shoulderSpanX
          : 0;
      const distalPt = wrist && (wrist.visibility ?? 0) >= UPPER_MIN_VIS ? wrist : elbow;
      const dxAbs = sh && distalPt ? Math.abs(distalPt.x - sh.x) : 0;
      const dyAbs = sh && distalPt ? Math.abs(distalPt.y - sh.y) : 0;
      const sensorDzAbs = sh && distalPt ? Math.abs((distalPt.z ?? 0) - (sh.z ?? 0)) : 0;
      const distalIdx = distalPt === wrist ? pWrIdx : pElIdx;
      const wShFlex = worldLandmarks?.[pShIdx];
      const wDistalFlex = worldLandmarks?.[distalIdx];
      const wDxFlex = wShFlex && wDistalFlex ? Math.abs(wDistalFlex.x - wShFlex.x) : 0;
      const wDzFlex =
        wShFlex && wDistalFlex ? Math.abs((wDistalFlex.z ?? 0) - (wShFlex.z ?? 0)) : 0;
      const worldForwardDominant =
        Boolean(
          wShFlex &&
            wDistalFlex &&
            Math.hypot(wDxFlex, wDistalFlex.y - wShFlex.y, wDzFlex) >= 0.08,
        ) && wDzFlex > wDxFlex * 1.15;
      const forwardDepthProxy = Math.max(dyAbs, sensorDzAbs * 0.65);
      const horizRatioFlex = dxAbs / Math.max(0.04, Math.hypot(dxAbs, forwardDepthProxy));
      const horizRatioAbd = dxAbs / Math.max(shoulderSpanX, Math.hypot(dxAbs, dyAbs));

      if (
        expectedPlane === 'flexion' &&
        primaryDeg >= 42 &&
        !worldForwardDominant &&
        ((wristOutwardLatRatio >= 0.85 && elbowOutwardLatRatio >= 0.45) ||
          elbowOutwardLatRatio >= 0.7) &&
        dxAbs > forwardDepthProxy * 1.15 &&
        horizRatioFlex >= 0.62
      ) {
        wrongPlaneRaised = true;
        wrongPlaneBannerTh = getPrescribedPlanePromptTh('flexion').bannerAlertTh;
      } else if (
        expectedPlane === 'abduction' &&
        primaryDeg >= 45 &&
        primaryDeg <= 140 &&
        wristLatRatio < 0.34 &&
        elbowLatRatio < 0.28 &&
        horizRatioAbd < 0.32
      ) {
        wrongPlaneRaised = true;
        wrongPlaneBannerTh = getPrescribedPlanePromptTh('abduction').bannerAlertTh;
      }
    }

    if (rawPrimaryRomDeg === null) {
      lowVisibilityAlertTh = 'มองเห็นข้อต่อแขนไม่ชัดเจน (< 0.70) กรุณาขยับให้อยู่ในกรอบกล้อง';
    } else if (!wrongSideRaised && !wrongPlaneRaised && rawPrimaryRomDeg >= 40) {
      if (shoulderTiltDeg > 13.5) {
        postureCompensated = true;
        postureBannerTh = `ห้ามยกไหล่หรือเอียงไหล่ช่วย (${shoulderTiltDeg.toFixed(1)}°)`;
      } else if (
        (expectedPlane === 'flexion' || expectedPlane === 'abduction') &&
        elbowAngleDeg !== null &&
        elbowAngleDeg < 138
      ) {
        postureCompensated = true;
        postureBannerTh = `กรุณาเหยียดข้อศอกให้ตรง (${elbowAngleDeg.toFixed(0)}°)`;
      }
    }

    // Strict side & plane enforcement: block wrong-side or wrong-plane movements at 0 deg
    romDeg =
      wrongSideRaised || wrongPlaneRaised
        ? 0
        : rawPrimaryRomDeg;
    oppositeRomDeg = oppositeDeg;
    if (wrongSideRaised) {
      wrongSideBannerTh = getPrescribedLimbPromptTh(targetJoint).bannerAlertTh;
    }
  } else if (targetJoint === 'left_knee' || targetJoint === 'right_knee') {
    const isLeft = targetJoint === 'left_knee';
    const hipIdx = isLeft ? POSE_INDEX.LEFT_HIP : POSE_INDEX.RIGHT_HIP;
    const kneeIdx = isLeft ? POSE_INDEX.LEFT_KNEE : POSE_INDEX.RIGHT_KNEE;
    const ankleIdx = isLeft ? POSE_INDEX.LEFT_ANKLE : POSE_INDEX.RIGHT_ANKLE;
    const hip = landmarks[hipIdx];
    const knee = landmarks[kneeIdx];
    const ankle = landmarks[ankleIdx];
    const KNEE_JOINT_MIN_VIS = 0.55;
    const hasShouldersForOrder = Boolean(
      lSh && rSh && (lSh.visibility ?? 0) >= 0.35 && (rSh.visibility ?? 0) >= 0.35,
    );
    const maxShY = hasShouldersForOrder && lSh && rSh ? Math.max(lSh.y, rSh.y) : 0;
    const anatomicalOrderOk = Boolean(
      hip &&
        knee &&
        ankle &&
        (!hasShouldersForOrder || hip.y > maxShY + 0.10) &&
        knee.y >= hip.y - 0.08 &&
        ankle.y > knee.y + 0.06,
    );
    const inFrameKneeJoint = Boolean(
      hip &&
        knee &&
        ankle &&
        (hip.visibility ?? 0) >= KNEE_JOINT_MIN_VIS &&
        (knee.visibility ?? 0) >= KNEE_JOINT_MIN_VIS &&
        (ankle.visibility ?? 0) >= KNEE_JOINT_MIN_VIS &&
        hip.y <= 0.96 &&
        knee.y <= 0.96 &&
        ankle.y <= 0.96 &&
        anatomicalOrderOk,
    );
    if (inFrameKneeJoint && hip && knee && ankle) {
      const wHip = worldLandmarks?.[hipIdx];
      const wKnee = worldLandmarks?.[kneeIdx];
      const wAnkle = worldLandmarks?.[ankleIdx];
      const hasValidWorldKnee = Boolean(
        wHip &&
          wKnee &&
          wAnkle &&
          Math.hypot(wHip.x - wKnee.x, wHip.y - wKnee.y, (wHip.z ?? 0) - (wKnee.z ?? 0)) >= 0.08 &&
          Math.hypot(wAnkle.x - wKnee.x, wAnkle.y - wKnee.y, (wAnkle.z ?? 0) - (wKnee.z ?? 0)) >= 0.08,
      );
      const kneeInteriorDeg =
        hasValidWorldKnee && wHip && wKnee && wAnkle
          ? computeJointAngle3D(wHip, wKnee, wAnkle)
          : computeJointAngle3D(hip, knee, ankle);
      romDeg =
        jointMode === 'hip_march'
          ? Number(Math.max(0, 180 - kneeInteriorDeg).toFixed(1))
          : kneeInteriorDeg;
      rawPrimaryRomDeg = romDeg;
    } else {
      lowVisibilityAlertTh = 'มองไม่เห็นสะโพก-เข่า-ข้อเท้าชัดเจน กรุณาถอยห่างจากกล้องให้เห็นขาเต็มช่วง';
    }
  } else if (jointMode === 'neck') {
    const nose = landmarks[POSE_INDEX.NOSE];
    if (
      nose &&
      lSh &&
      rSh &&
      (nose.visibility ?? 0) >= UPPER_MIN_VIS &&
      (lSh.visibility ?? 0) >= UPPER_MIN_VIS &&
      (rSh.visibility ?? 0) >= UPPER_MIN_VIS
    ) {
      const neckMidX = (lSh.x + rSh.x) / 2;
      const neckMidY = (lSh.y + rSh.y) / 2;
      const lateralTiltRad = Math.atan2(nose.x - neckMidX, Math.max(0.05, neckMidY - nose.y));
      romDeg = Number(
        Math.min(60, ((Math.abs(lateralTiltRad) * 180) / Math.PI) * 1.35).toFixed(1),
      );
      rawPrimaryRomDeg = romDeg;
    }
  } else if (targetJoint === 'spine') {
    romDeg = Number(Math.max(0, 15 - trunkTiltDeg).toFixed(1));
  }

  return {
    romDeg,
    rawPrimaryRomDeg,
    trunkTiltDeg: Number(trunkTiltDeg.toFixed(1)),
    wrongSideRaised,
    oppositeRomDeg,
    wrongSideBannerTh,
    wrongPlaneRaised,
    wrongPlaneBannerTh,
    xyzVector,
    lowVisibilityAlertTh,
    postureCompensated,
    postureBannerTh,
    elbowAngleDeg,
    shoulderTiltDeg,
  };
}

/**
 * Smooths noisy joint ROM readings with a compact anti-jitter deadband and instant
 * fast-pursuit slew rate so gauge arcs and degree readouts lock onto rapid arm motion.
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
  // Tight deadband of 0.35 deg suppresses stationary jitter while locking onto motion without lag
  if (Math.abs(rawDelta) < 0.35) {
    return currentRom;
  }
  // Instant fast-pursuit filter (tau: 3..12ms) + high slew cap (1800 deg/s)
  const absDelta = Math.abs(rawDelta);
  const adaptiveTau = absDelta >= 12 ? 3 : absDelta >= 4 ? 6 : 12;
  const alpha = 1 - Math.exp(-safeDt / adaptiveTau);
  const emaStep = rawDelta * alpha;
  const maxStep = (1800 * safeDt) / 1000;
  const step = emaStep > 0 ? Math.min(emaStep, maxStep) : Math.max(emaStep, -maxStep);
  return Math.max(0, Math.min(180, currentRom + step));
}

export interface LandmarkSmootherOptions {
  alpha?: number;
  adaptive?: boolean;
  minAlpha?: number;
  maxAlpha?: number;
  velocityScale?: number;
}

/**
 * Dual-mode EMA & Velocity-based Adaptive EMA smoother over (N, 3) pose [x, y, z] coordinates.
 */
export class LandmarkSmoother {
  alpha: number;
  adaptive: boolean;
  minAlpha: number;
  maxAlpha: number;
  velocityScale: number;
  private smoothedCoords: Array<[number, number, number]> | null = null;

  constructor(options: LandmarkSmootherOptions = {}) {
    this.alpha = options.alpha ?? 0.3;
    this.adaptive = options.adaptive ?? true;
    this.minAlpha = options.minAlpha ?? 0.1;
    this.maxAlpha = options.maxAlpha ?? 0.8;
    this.velocityScale = options.velocityScale ?? 15.0;
  }

  apply(currentLandmarks: readonly NormalizedLandmark[]): NormalizedLandmark[] {
    if (currentLandmarks.length === 0) return [];
    if (!this.smoothedCoords || this.smoothedCoords.length !== currentLandmarks.length) {
      this.smoothedCoords = currentLandmarks.map((lm) => [lm.x, lm.y, lm.z ?? 0]);
      return currentLandmarks.map((lm) => ({ ...lm }));
    }

    const nextCoords: Array<[number, number, number]> = [];
    const output: NormalizedLandmark[] = new Array(currentLandmarks.length);

    for (let i = 0; i < currentLandmarks.length; i += 1) {
      const lm = currentLandmarks[i]!;
      const prev = this.smoothedCoords[i]!;
      const curX = lm.x;
      const curY = lm.y;
      const curZ = lm.z ?? 0;

      let effectiveAlpha = this.alpha;
      if (this.adaptive) {
        const velocity = Math.hypot(curX - prev[0], curY - prev[1], curZ - prev[2]);
        effectiveAlpha = Math.max(
          this.minAlpha,
          Math.min(this.maxAlpha, this.minAlpha + velocity * this.velocityScale),
        );
      }

      const sx = effectiveAlpha * curX + (1 - effectiveAlpha) * prev[0];
      const sy = effectiveAlpha * curY + (1 - effectiveAlpha) * prev[1];
      const sz = effectiveAlpha * curZ + (1 - effectiveAlpha) * prev[2];
      nextCoords.push([sx, sy, sz]);
      output[i] = {
        ...lm,
        x: sx,
        y: sy,
        z: sz,
      };
    }

    this.smoothedCoords = nextCoords;
    return output;
  }

  reset(): void {
    this.smoothedCoords = null;
  }
}

/**
 * Zero-delay 1:1 landmark follower: MediaPipe VIDEO mode already applies its internal C++ One-Euro
 * filter, so only sub-pixel stationary tremor (< 0.0015 norm) is damped while active movement snaps 1:1.
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
  const jitterAlpha = Math.min(1, 1 - Math.exp(-safeDt / 22));
  return next.map((target, idx) => {
    const cur = prev[idx];
    if (!cur) return { ...target };
    const dx = target.x - cur.x;
    const dy = target.y - cur.y;
    const dist2D = Math.hypot(dx, dy);
    const alpha = dist2D < 0.0015 ? jitterAlpha : dist2D < 0.0025 ? 0.9 : 1.0;
    if (alpha >= 1) {
      return { ...target };
    }
    const vis =
      target.visibility !== undefined
        ? (cur.visibility ?? target.visibility) +
          (target.visibility - (cur.visibility ?? target.visibility)) * alpha
        : cur.visibility;
    return {
      x: cur.x + dx * alpha,
      y: cur.y + dy * alpha,
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
    expectedPlane?: ExpectedMovementPlane;
    exerciseMeta?: ExerciseMetaHint;
    romDeg: number | null;
    targetRomDeg: number;
    holdRatio: number;
    trunkTiltDeg: number;
    isMultiPerson: boolean;
    isImpostorOrEmergency: boolean;
    wrongSideRaised?: boolean;
    wrongSideBannerTh?: string | null;
    wrongPlaneRaised?: boolean;
    wrongPlaneBannerTh?: string | null;
    lowVisibilityAlertTh?: string | null;
    postureBannerTh?: string | null;
    videoWidth?: number;
    videoHeight?: number;
    smoothingStatusText?: string;
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

  // Match CSS object-contain wide-angle framing so mirrored landmarks align 1:1 over the uncropped video feed
  const vw = opts.videoWidth && opts.videoWidth > 0 ? opts.videoWidth : w;
  const vh = opts.videoHeight && opts.videoHeight > 0 ? opts.videoHeight : h;
  const fitScale = Math.min(w / vw, h / vh);
  const renderW = vw * fitScale;
  const renderH = vh * fitScale;
  const offsetX = (w - renderW) / 2;
  const offsetY = (h - renderH) / 2;
  const rawCanvasX = (normX: number) => offsetX + (1 - normX) * renderW;
  const rawCanvasY = (normY: number) => offsetY + normY * renderH;
  const toCanvasPt = (_idx: number, lm: NormalizedLandmark) => ({
    x: rawCanvasX(lm.x),
    y: rawCanvasY(lm.y),
  });
  const minVisFor = (idx: number) => (UPPER_BODY_KEYPOINTS.has(idx) ? 0.35 : 0.55);

  const activeColor =
    opts.isImpostorOrEmergency || opts.postureBannerTh
      ? '#EF4444'
      : opts.wrongPlaneRaised || opts.isMultiPerson || opts.trunkTiltDeg > 10 || opts.lowVisibilityAlertTh
        ? '#F59E0B'
        : '#10B981';

  const primary = landmarksList[0];
  const isUpperLimbOrSpine =
    opts.targetJoint === 'left_shoulder' ||
    opts.targetJoint === 'right_shoulder' ||
    opts.targetJoint === 'spine';
  const lShLm = primary[POSE_INDEX.LEFT_SHOULDER];
  const rShLm = primary[POSE_INDEX.RIGHT_SHOULDER];
  const maxShY =
    lShLm && rShLm ? Math.max(lShLm.y, rShLm.y) : (lShLm?.y ?? rShLm?.y ?? 0.5);
  const lHipLm = primary[POSE_INDEX.LEFT_HIP];
  const rHipLm = primary[POSE_INDEX.RIGHT_HIP];
  const hipsCleanlyBelowShoulders = Boolean(
    lHipLm &&
      rHipLm &&
      (lHipLm.visibility ?? 0) >= 0.55 &&
      (rHipLm.visibility ?? 0) >= 0.55 &&
      lHipLm.y > maxShY + 0.16 &&
      rHipLm.y > maxShY + 0.16 &&
      Math.abs(lHipLm.x - rHipLm.x) >= 0.09,
  );
  const activeWristLm =
    opts.targetJoint === 'left_shoulder'
      ? primary[POSE_INDEX.LEFT_WRIST]
      : opts.targetJoint === 'right_shoulder'
        ? primary[POSE_INDEX.RIGHT_WRIST]
        : undefined;

  const isOppositeArmIdx = (idx: number) => {
    if (opts.targetJoint === 'left_shoulder') {
      return idx === POSE_INDEX.RIGHT_ELBOW || idx === POSE_INDEX.RIGHT_WRIST;
    }
    if (opts.targetJoint === 'right_shoulder') {
      return idx === POSE_INDEX.LEFT_ELBOW || idx === POSE_INDEX.LEFT_WRIST;
    }
    return false;
  };

  const isPointDrawable = (idx: number): boolean => {
    const lm = primary[idx];
    if (!lm) return false;
    const isLowerLeg =
      idx === POSE_INDEX.LEFT_KNEE ||
      idx === POSE_INDEX.RIGHT_KNEE ||
      idx === POSE_INDEX.LEFT_ANKLE ||
      idx === POSE_INDEX.RIGHT_ANKLE;
    const isHip = idx === POSE_INDEX.LEFT_HIP || idx === POSE_INDEX.RIGHT_HIP;
    if (isUpperLimbOrSpine) {
      // Never draw hallucinated knee/ankle lines during upper-body camera exercises
      if (isLowerLeg) return false;
      if (isHip && !hipsCleanlyBelowShoulders) return false;
    }
    const minV = isOppositeArmIdx(idx) ? 0.28 : minVisFor(idx);
    if ((lm.visibility ?? 0) < minV) return false;
    // Prevent phantom opposite elbow/wrist keypoints collapsed onto the reaching hand
    if (
      isOppositeArmIdx(idx) &&
      !opts.wrongSideRaised &&
      activeWristLm &&
      (activeWristLm.visibility ?? 0) >= 0.28 &&
      Math.hypot(lm.x - activeWristLm.x, lm.y - activeWristLm.y) < 0.08
    ) {
      return false;
    }
    return true;
  };

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
  ctx.save();
  ctx.lineCap = 'round';
  for (const [aIdx, bIdx] of LIVE_BODY_BONELINE_PAIRS) {
    if (!isPointDrawable(aIdx) || !isPointDrawable(bIdx)) {
      continue;
    }
    const a = primary[aIdx]!;
    const b = primary[bIdx]!;
    const aPt = toCanvasPt(aIdx, a);
    const bPt = toCanvasPt(bIdx, b);
    const emphasized = isHighlightedEdge(aIdx, bIdx);
    const isWrongRaisedEdge = Boolean(
      (opts.wrongSideRaised && isOppositeArmEdge(aIdx, bIdx)) ||
        (opts.wrongPlaneRaised && emphasized),
    );

    const strokeHex = isWrongRaisedEdge
      ? '#EF4444'
      : emphasized
        ? activeColor
        : 'rgba(56, 189, 248, 0.9)';
    ctx.shadowBlur = 0;
    if (isWrongRaisedEdge || emphasized) {
      ctx.beginPath();
      ctx.moveTo(aPt.x, aPt.y);
      ctx.lineTo(bPt.x, bPt.y);
      ctx.lineWidth = 13;
      ctx.globalAlpha = 0.24;
      ctx.strokeStyle = strokeHex;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    ctx.beginPath();
    ctx.moveTo(aPt.x, aPt.y);
    ctx.lineTo(bPt.x, bPt.y);
    ctx.lineWidth = isWrongRaisedEdge ? 6 : emphasized ? 6 : 3.5;
    ctx.strokeStyle = strokeHex;
    ctx.stroke();
  }

  // Head/neck guide lines (NOSE -> shoulder midpoint NECK -> both shoulders)
  const noseLm = primary[POSE_INDEX.NOSE];
  if (
    lShLm &&
    rShLm &&
    (lShLm.visibility ?? 0) >= 0.18 &&
    (rShLm.visibility ?? 0) >= 0.18
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
    if (noseLm && (noseLm.visibility ?? 0) >= 0.18) {
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
    if (!isPointDrawable(idx)) continue;
    const lm = primary[idx]!;
    if (lm.x < -0.02 || lm.x > 1.02 || lm.y < -0.02 || lm.y > 1.02) {
      continue;
    }
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
  const overlayJointMode = classifyExerciseJointMode(opts.exerciseMeta);
  const anchorIdx =
    opts.targetJoint === 'left_shoulder'
      ? overlayJointMode === 'elbow'
        ? POSE_INDEX.LEFT_ELBOW
        : POSE_INDEX.LEFT_SHOULDER
      : opts.targetJoint === 'right_shoulder'
        ? overlayJointMode === 'elbow'
          ? POSE_INDEX.RIGHT_ELBOW
          : POSE_INDEX.RIGHT_SHOULDER
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

    // Draw real-time geometric angle arc directly at the active joint
    if (
      opts.targetJoint === 'left_shoulder' ||
      opts.targetJoint === 'right_shoulder' ||
      opts.targetJoint === 'left_knee' ||
      opts.targetJoint === 'right_knee'
    ) {
      const isLeftSide =
        opts.targetJoint === 'left_shoulder' || opts.targetJoint === 'left_knee';
      let refLm: NormalizedLandmark | undefined;
      let movLm: NormalizedLandmark | undefined;
      if (opts.targetJoint === 'left_shoulder' || opts.targetJoint === 'right_shoulder') {
        if (overlayJointMode === 'elbow') {
          refLm = primary[isLeftSide ? POSE_INDEX.LEFT_SHOULDER : POSE_INDEX.RIGHT_SHOULDER];
          movLm = primary[isLeftSide ? POSE_INDEX.LEFT_WRIST : POSE_INDEX.RIGHT_WRIST];
        } else {
          refLm = primary[isLeftSide ? POSE_INDEX.LEFT_HIP : POSE_INDEX.RIGHT_HIP];
          movLm = primary[isLeftSide ? POSE_INDEX.LEFT_ELBOW : POSE_INDEX.RIGHT_ELBOW];
        }
      } else {
        refLm = primary[isLeftSide ? POSE_INDEX.LEFT_HIP : POSE_INDEX.RIGHT_HIP];
        movLm = primary[isLeftSide ? POSE_INDEX.LEFT_ANKLE : POSE_INDEX.RIGHT_ANKLE];
      }

      if (movLm && (movLm.visibility ?? 0) >= 0.18) {
        const movPt = toCanvasPt(anchorIdx, movLm);
        const refPt =
          refLm && (refLm.visibility ?? 0) >= 0.18
            ? toCanvasPt(anchorIdx, refLm)
            : { x: anchorPt.x, y: anchorPt.y + 80 };
        const aRef = Math.atan2(refPt.y - anchorPt.y, refPt.x - anchorPt.x);
        const aMov = Math.atan2(movPt.y - anchorPt.y, movPt.x - anchorPt.x);
        let deltaAng = aMov - aRef;
        while (deltaAng <= -Math.PI) deltaAng += Math.PI * 2;
        while (deltaAng > Math.PI) deltaAng -= Math.PI * 2;
        const anticlockwise = deltaAng < 0;

        ctx.save();
        ctx.beginPath();
        ctx.moveTo(anchorPt.x, anchorPt.y);
        ctx.arc(anchorPt.x, anchorPt.y, 32, aRef, aRef + deltaAng, anticlockwise);
        ctx.closePath();
        ctx.globalAlpha = 0.24;
        ctx.fillStyle = activeColor;
        ctx.fill();
        ctx.globalAlpha = 0.92;
        ctx.beginPath();
        ctx.arc(anchorPt.x, anchorPt.y, 32, aRef, aRef + deltaAng, anticlockwise);
        ctx.lineWidth = 3;
        ctx.strokeStyle = activeColor;
        ctx.stroke();
        ctx.restore();
      }
    }

    const jx = Math.max(64, Math.min(w - 64, anchorPt.x));
    const jy = Math.max(54, Math.min(h - 40, anchorPt.y - 42));

    if (opts.holdRatio > 0) {
      ctx.beginPath();
      ctx.arc(jx, jy, 34, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * Math.min(1, opts.holdRatio));
      ctx.strokeStyle = '#10B981';
      ctx.lineWidth = 5;
      ctx.stroke();
    }

    const badgeText = `${opts.romDeg.toFixed(1)}° / ${opts.targetRomDeg}°`;
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

  // On-body alert badge when patient raises in wrong movement plane (e.g. side abduction instead of forward flexion)
  if (opts.wrongPlaneRaised && !opts.wrongSideRaised) {
    const activeElbowIdx =
      opts.targetJoint === 'left_shoulder' ? POSE_INDEX.LEFT_ELBOW : POSE_INDEX.RIGHT_ELBOW;
    const activeShIdx =
      opts.targetJoint === 'left_shoulder' ? POSE_INDEX.LEFT_SHOULDER : POSE_INDEX.RIGHT_SHOULDER;
    const activeLm = primary[activeElbowIdx] ?? primary[activeShIdx] ?? primary[POSE_INDEX.NOSE];
    const planeInfo = getPrescribedPlanePromptTh(opts.expectedPlane ?? 'any');
    const warnText = `✗ ${planeInfo.wrongPlaneTh} • กรุณา${planeInfo.requiredPlaneTh}`;
    ctx.font = '800 13px "Sarabun", sans-serif';
    const wtw = ctx.measureText(warnText).width;
    const wbw = Math.min(w - 24, wtw + 24);
    const wbh = 32;
    const wx = activeLm
      ? Math.max(wbw / 2 + 12, Math.min(w - wbw / 2 - 12, rawCanvasX(activeLm.x)))
      : w / 2;
    const wy = activeLm ? Math.max(88, Math.min(h - 36, rawCanvasY(activeLm.y) - 34)) : 96;

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

  const clinicalAlertBanner = !opts.wrongSideRaised && !opts.wrongPlaneRaised
    ? opts.postureBannerTh
      ? { text: `⚠ ${opts.postureBannerTh}`, bg: 'rgba(220, 38, 38, 0.92)', border: '#FCA5A5' }
      : opts.lowVisibilityAlertTh
        ? { text: `⚠ ${opts.lowVisibilityAlertTh}`, bg: 'rgba(217, 119, 6, 0.92)', border: '#FDE68A' }
        : null
    : null;
  if (clinicalAlertBanner) {
    ctx.font = '800 13px "Sarabun", sans-serif';
    const tw = ctx.measureText(clinicalAlertBanner.text).width;
    const bw = Math.min(w - 24, tw + 26);
    const bh = 32;
    const bx = w / 2;
    const by = h - 68;
    ctx.fillStyle = clinicalAlertBanner.bg;
    ctx.strokeStyle = clinicalAlertBanner.border;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(bx - bw / 2, by - bh / 2, bw, bh, 16);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = '#FFFFFF';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(clinicalAlertBanner.text, bx, by, bw - 18);
  }

  if (opts.smoothingStatusText) {
    ctx.font = '700 11px "Sarabun", sans-serif';
    const textW = ctx.measureText(opts.smoothingStatusText).width;
    const pillW = Math.min(w - 28, textW + 18);
    const pillH = 22;
    const pillX = 14;
    const pillY = h - 32;
    ctx.fillStyle = 'rgba(15, 23, 42, 0.72)';
    ctx.beginPath();
    ctx.roundRect(pillX, pillY, pillW, pillH, 11);
    ctx.fill();
    ctx.fillStyle = 'rgba(226, 232, 240, 0.96)';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(opts.smoothingStatusText, pillX + 9, pillY + pillH / 2, pillW - 18);
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
    isEdgeDirectCamera,
    sendClientLandmarks,
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

  const sendClientLandmarksRef = useRef(sendClientLandmarks);
  sendClientLandmarksRef.current = sendClientLandmarks;

  const [voiceMuted, setVoiceMuted] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(() => !patientProp);
  const [formRules, setFormRules] = useState<FormAssessmentRulesConfig>(() => {
    try {
      const saved =
        typeof window !== 'undefined' ? window.localStorage.getItem(FORM_ASSESSMENT_STORAGE_KEY) : null;
      if (saved) {
        const parsed = JSON.parse(saved) as Partial<FormAssessmentRulesConfig>;
        return { ...DEFAULT_FORM_ASSESSMENT_RULES, ...parsed };
      }
    } catch {
      // Ignore storage read errors
    }
    return DEFAULT_FORM_ASSESSMENT_RULES;
  });
  const [formChecklistExpanded, setFormChecklistExpanded] = useState<boolean>(true);
  const [completedIds, setCompletedIds] = useState<string[]>([]);
  const [livePoseActive, setLivePoseActive] = useState<boolean>(true);

  const handleToggleDemoMode = () => {
    if (isDemoMode) {
      setAutoCycleDemo(false);
      setLivePoseActive(true);
    }
    toggleDemoMode();
  };
  const [cameraReady, setCameraReady] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const poseCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const onExerciseCompletedRef = useRef(onExerciseCompleted);
  const onAllExercisesCompletedRef = useRef(onAllExercisesCompleted);
  const selectedExerciseRef = useRef(selectedExercise);
  const demoValidRepsRef = useRef(demoValidReps);
  const demoPresetRef = useRef(demoPreset);
  const formRulesRef = useRef<FormAssessmentRulesConfig>(formRules);
  const countingLockedRef = useRef(false);

  onExerciseCompletedRef.current = onExerciseCompleted;
  onAllExercisesCompletedRef.current = onAllExercisesCompleted;
  selectedExerciseRef.current = selectedExercise;
  demoValidRepsRef.current = demoValidReps;
  demoPresetRef.current = demoPreset;
  formRulesRef.current = formRules;
  countingLockedRef.current =
    isFormRuleActive(formRules, 'identity') &&
    (payload.kiosk_state === 'PAUSED_MULTI_PERSON' ||
      payload.kiosk_state === 'PAUSED_IMPOSTOR' ||
      payload.identity_guard?.is_counting_allowed === false);

  useEffect(() => {
    try {
      if (typeof window !== 'undefined') {
        window.localStorage.setItem(FORM_ASSESSMENT_STORAGE_KEY, JSON.stringify(formRules));
      }
    } catch {
      // Ignore storage write errors
    }
  }, [formRules]);

  useEffect(() => {
    if (
      (demoPreset === 'WRONG_LIMB' && !isFormRuleActive(formRules, 'limb')) ||
      (demoPreset === 'WRONG_PLANE' && !isFormRuleActive(formRules, 'plane')) ||
      (demoPreset === 'TRUNK_LEAN' && !isFormRuleActive(formRules, 'trunk')) ||
      (demoPreset === 'MULTI_PERSON' && !isFormRuleActive(formRules, 'identity'))
    ) {
      setDemoPresetState('ACTIVE_REP');
    }
  }, [formRules, demoPreset, setDemoPresetState]);

  const toggleMasterFormAssessment = () => {
    setFormRules((prev) => {
      const nextEnabled = !prev.enabled;
      if (nextEnabled) {
        setFormChecklistExpanded(true);
      }
      return { ...prev, enabled: nextEnabled };
    });
  };

  const toggleSingleFormRule = (rule: FormAssessmentRuleKey) => {
    setFormRules((prev) => {
      if (!prev.enabled) {
        return { ...prev, enabled: true, [rule]: true };
      }
      return { ...prev, [rule]: !prev[rule] };
    });
  };

  const resetAllFormRules = () => {
    setFormRules(DEFAULT_FORM_ASSESSMENT_RULES);
    setFormChecklistExpanded(true);
  };

  const { kiosk_state, patient, telemetry_analysis, rep_counter, ui_directives, tts_voice_th } = payload;
  const compensations = telemetry_analysis.compensation_detected;
  const trunkLeanAlert = isFormRuleActive(formRules, 'trunk') && compensations.includes('TRUNK_LEAN');
  const shoulderHikeAlert = isFormRuleActive(formRules, 'posture') && compensations.includes('SHOULDER_HIKE');
  const kneeValgusAlert = isFormRuleActive(formRules, 'posture') && compensations.includes('KNEE_VALGUS');
  const tooFastAlert = isFormRuleActive(formRules, 'velocity') && compensations.includes('TOO_FAST');
  const wrongLimbAlert = isFormRuleActive(formRules, 'limb') && compensations.includes('WRONG_LIMB');
  const wrongPlaneAlert = isFormRuleActive(formRules, 'plane') && compensations.includes('WRONG_PLANE');
  const activeExpectedPlane = resolveExpectedPlaneForExercise(selectedExercise);
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

  // Live MediaPipe PoseLandmarker camera tracking (skip when Python Edge Vision owns the physical webcam)
  useEffect(() => {
    if (!livePoseActive || autoCycleDemo || isEmergency || isEdgeDirectCamera) {
      setCameraReady(false);
      return;
    }

    let active = true;
    let stream: MediaStream | null = null;
    let rafId = 0;
    let poseLandmarker: PoseLandmarker | null = null;
    let offscreenCanvas: HTMLCanvasElement | null = null;
    let useOffscreenFallback = false;
    let lastValidPoseAt = 0;
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
    let torsoAnchor: TorsoAnchorState | null = null;
    const landmarkSmoother = new LandmarkSmoother({
      alpha: 0.25,
      adaptive: true,
      minAlpha: 0.1,
      maxAlpha: 0.85,
      velocityScale: 15.0,
    });
    const pose2dOneEuroFilter = new ClinicalOneEuroLandmarkFilter(1.1, 0.06, 1.0);
    const worldOneEuroFilter = new ClinicalOneEuroLandmarkFilter(0.7, 0.04, 1.0);
    let smoothingMode: 'one_euro' | 'adaptive_ema' | 'fixed_ema' = 'one_euro';
    let enableCustomSmooth = true;
    const formatSmoothingHud = () => {
      if (!enableCustomSmooth) {
        return `Smoothing: OFF • [1] On/Off [2] Mode [+/-] Alpha`;
      }
      if (smoothingMode === 'one_euro') {
        return `Smoothing: 1€ Filter (Heavy 3D+2D) • [1] On/Off [2] Mode [+/-] Alpha`;
      }
      if (smoothingMode === 'adaptive_ema') {
        return `Smoothing: Adaptive EMA (min=${landmarkSmoother.minAlpha.toFixed(2)}, max=${landmarkSmoother.maxAlpha.toFixed(2)}) • [1] On/Off [2] Mode [+/-] Alpha`;
      }
      return `Smoothing: Fixed EMA (alpha=${landmarkSmoother.alpha.toFixed(2)}) • [1] On/Off [2] Mode [+/-] Alpha`;
    };
    const handleSmoothingKeyDown = (event: KeyboardEvent) => {
      const targetEl = event.target as HTMLElement | null;
      if (
        targetEl &&
        (targetEl.tagName === 'INPUT' ||
          targetEl.tagName === 'TEXTAREA' ||
          targetEl.tagName === 'SELECT' ||
          targetEl.isContentEditable)
      ) {
        return;
      }
      if (event.key === '1') {
        enableCustomSmooth = !enableCustomSmooth;
        if (!enableCustomSmooth) {
          landmarkSmoother.reset();
          pose2dOneEuroFilter.reset();
          worldOneEuroFilter.reset();
        }
      } else if (event.key === '2') {
        smoothingMode =
          smoothingMode === 'one_euro'
            ? 'adaptive_ema'
            : smoothingMode === 'adaptive_ema'
              ? 'fixed_ema'
              : 'one_euro';
        landmarkSmoother.adaptive = smoothingMode === 'adaptive_ema';
      } else if (event.key === '+' || event.key === '=') {
        landmarkSmoother.alpha = Math.min(0.95, Number((landmarkSmoother.alpha + 0.05).toFixed(2)));
      } else if (event.key === '-' || event.key === '_') {
        landmarkSmoother.alpha = Math.max(0.05, Number((landmarkSmoother.alpha - 0.05).toFixed(2)));
      }
    };
    window.addEventListener('keydown', handleSmoothingKeyDown);
    let occludedFrames = 0;
    let lastVideoTime = -1;
    let lastDetectWallMs = 0;
    let wrongSideStreak = 0;
    let wrongPlaneStreak = 0;
    let latestRawPrimaryRomDeg: number | null = null;
    let targetRomDeg: number | null = null;
    let latestLandmarks: NormalizedLandmark[] | null = null;
    let renderPrimaryLandmarks: NormalizedLandmark[] | null = null;
    let latestAllLandmarks: NormalizedLandmark[][] = [];
    let latestPoseCount = 0;
    let latestRawTiltDeg = 0;
    let latestTrunkTiltDeg = 0;
    let latestWrongSideRaised = false;
    let latestWrongSideBannerTh: string | null = null;
    let latestWrongPlaneRaised = false;
    let latestWrongPlaneBannerTh: string | null = null;
    let latestLowVisibilityAlertTh: string | null = null;
    let latestPostureCompensated = false;
    let latestPostureBannerTh: string | null = null;
    let lastEmittedRom = -999;
    let lastEmittedTilt = -999;
    let lastEmittedHold = -999;
    let lastRomEmitAt = 0;
    let lastTiltEmitAt = 0;
    let lastHoldEmitAt = 0;
    let lastEdgeBridgeSentAt = 0;

    const cleanup = () => {
      window.removeEventListener('keydown', handleSmoothingKeyDown);
      landmarkSmoother.reset();
      pose2dOneEuroFilter.reset();
      worldOneEuroFilter.reset();
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

        const baseVideoConstraints: MediaTrackConstraints = {
          width: { ideal: 640 },
          height: { ideal: 360 },
          aspectRatio: { ideal: 16 / 9 },
          frameRate: { ideal: 60, max: 120 },
          facingMode: 'user',
        };
        stream = await navigator.mediaDevices.getUserMedia({
          video: deviceId
            ? {
                deviceId: { exact: deviceId },
                ...baseVideoConstraints,
              }
            : baseVideoConstraints,
          audio: false,
        });
        if (!active || !videoRef.current) {
          cleanup();
          return;
        }
        const videoTrack = stream.getVideoTracks()[0];
        const caps = videoTrack?.getCapabilities?.() as { zoom?: { min?: number } } | undefined;
        if (videoTrack && caps?.zoom && typeof caps.zoom.min === 'number') {
          await videoTrack
            .applyConstraints({
              advanced: [{ zoom: caps.zoom.min } as MediaTrackConstraintSet],
            })
            .catch(() => undefined);
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

        const POSE_MODEL_PATHS = [
          'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task',
          '/models/pose_landmarker_full.task',
          '/models/pose_landmarker.task',
        ];
        for (const minConf of [0.7, 0.5] as const) {
          for (const modelPath of POSE_MODEL_PATHS) {
            for (const delegate of ['GPU', 'CPU', undefined] as const) {
              try {
                const created = await PoseLandmarker.createFromOptions(vision, {
                  baseOptions: {
                    modelAssetPath: modelPath,
                    ...(delegate ? { delegate } : {}),
                  },
                  runningMode: 'VIDEO',
                  numPoses: 1,
                  minPoseDetectionConfidence: minConf,
                  minPosePresenceConfidence: minConf,
                  minTrackingConfidence: minConf,
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
          if (poseLandmarker) break;
        }
        if (!poseLandmarker || !active) return;

        const emitPresetState = (nextPreset: DemoPresetId) => {
          if (demoPresetRef.current !== nextPreset) {
            demoPresetRef.current = nextPreset;
            setDemoPresetState(nextPreset);
          }
        };

        const loop = () => {
          if (!active || !videoRef.current || !poseLandmarker) return;
          const curEx = selectedExerciseRef.current;
          const curExpectedPlane = resolveExpectedPlaneForExercise(curEx);
          if (trackedExId !== curEx.exercise_id) {
            trackedExId = curEx.exercise_id;
            torsoAnchor = null;
            landmarkSmoother.reset();
            pose2dOneEuroFilter.reset();
            worldOneEuroFilter.reset();
            occludedFrames = 0;
            wrongSideStreak = 0;
            wrongPlaneStreak = 0;
            latestRawPrimaryRomDeg = null;
            smoothedRom = 0;
            smoothedTilt = 0;
            targetRomDeg = null;
            latestLandmarks = null;
            lastValidPoseAt = 0;
            renderPrimaryLandmarks = null;
            latestAllLandmarks = [];
            latestPoseCount = 0;
            latestRawTiltDeg = 0;
            latestTrunkTiltDeg = 0;
            latestWrongSideRaised = false;
            latestWrongSideBannerTh = null;
            latestWrongPlaneRaised = false;
            latestWrongPlaneBannerTh = null;
            latestLowVisibilityAlertTh = null;
            latestPostureCompensated = false;
            latestPostureBannerTh = null;
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

          // Step A: Run zero-copy MediaPipe pose inference on new video frames (skip duplicate currentTime frames)
          if (
            v.readyState >= 2 &&
            v.videoWidth > 0 &&
            (v.currentTime !== lastVideoTime ||
              (useOffscreenFallback && now - lastDetectWallMs >= 14))
          ) {
            lastVideoTime = v.currentTime;
            lastDetectWallMs = now;
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
                const offH = Math.max(
                  144,
                  Math.round(offW * ((v.videoHeight || 360) / (v.videoWidth || 640))),
                );
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

              const stepARules = formRulesRef.current;
              const stepACheckIdentity = isFormRuleActive(stepARules, 'identity');

              if (latestPoseCount > 1 && stepACheckIdentity) {
                singlePersonRecoveryFrames = 0;
                latestLandmarks = null;
                renderPrimaryLandmarks = null;
                landmarkSmoother.reset();
                pose2dOneEuroFilter.reset();
                worldOneEuroFilter.reset();
              } else if (demoPresetRef.current !== 'IMPOSTOR' || !stepACheckIdentity) {
                const rawLandmarks = allLandmarks[0] ?? null;
                if (rawLandmarks) {
                  const stabilizedResult = sanitizeAndStabilizePoseLandmarks(
                    rawLandmarks,
                    torsoAnchor,
                    curEx.target_joint,
                  );
                  torsoAnchor = stabilizedResult.anchor;
                  if (stabilizedResult.occludedFallback) {
                    occludedFrames += 1;
                    if (occludedFrames > 12) {
                      torsoAnchor = null;
                      occludedFrames = 0;
                    }
                  } else {
                    occludedFrames = 0;
                  }
                  const applyOneEuro = (
                    flt: ClinicalOneEuroLandmarkFilter,
                    src: readonly NormalizedLandmark[],
                  ): NormalizedLandmark[] =>
                    flt.filterLandmarks(src, now).map((lm, idx) => ({
                      x: lm.x,
                      y: lm.y,
                      z: lm.z ?? src[idx]?.z ?? 0,
                      visibility: lm.visibility ?? src[idx]?.visibility ?? 0,
                    }));
                  const landmarks = !enableCustomSmooth
                    ? stabilizedResult.landmarks
                    : smoothingMode === 'one_euro'
                      ? applyOneEuro(pose2dOneEuroFilter, stabilizedResult.landmarks)
                      : landmarkSmoother.apply(stabilizedResult.landmarks);
                  latestLandmarks = landmarks;
                  latestAllLandmarks = [landmarks, ...allLandmarks.slice(1)];
                  lastValidPoseAt = now;
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
                      emitPresetState('ACTIVE_REP');
                    }
                  }
                  const rawWorldLandmarks = poseRes?.worldLandmarks?.[0];
                  const worldLandmarks =
                    rawWorldLandmarks && enableCustomSmooth
                      ? applyOneEuro(worldOneEuroFilter, rawWorldLandmarks)
                      : rawWorldLandmarks;
                  if (now - lastEdgeBridgeSentAt >= 65) {
                    lastEdgeBridgeSentAt = now;
                    sendClientLandmarksRef.current(landmarks, worldLandmarks, latestPoseCount);
                  }
                  const measured = measureLiveJointAngleAndTilt(
                    landmarks,
                    curEx.target_joint,
                    curExpectedPlane,
                    worldLandmarks,
                    curEx,
                  );
                  latestRawTiltDeg = measured.trunkTiltDeg;
                  wrongSideStreak = measured.wrongSideRaised ? wrongSideStreak + 1 : 0;
                  wrongPlaneStreak = measured.wrongPlaneRaised ? wrongPlaneStreak + 1 : 0;
                  latestWrongSideRaised = wrongSideStreak >= 4;
                  latestWrongSideBannerTh = measured.wrongSideBannerTh;
                  latestWrongPlaneRaised = wrongPlaneStreak >= 4;
                  latestWrongPlaneBannerTh = measured.wrongPlaneBannerTh;
                  latestLowVisibilityAlertTh = measured.lowVisibilityAlertTh ?? null;
                  latestPostureCompensated = Boolean(measured.postureCompensated);
                  latestPostureBannerTh = measured.postureBannerTh ?? null;
                  latestRawPrimaryRomDeg = measured.rawPrimaryRomDeg ?? measured.romDeg;
                } else if (now - lastValidPoseAt > 220 || !latestLandmarks) {
                  wrongSideStreak = 0;
                  wrongPlaneStreak = 0;
                  latestWrongSideRaised = false;
                  latestWrongPlaneRaised = false;
                  latestLowVisibilityAlertTh = null;
                  latestPostureCompensated = false;
                  latestPostureBannerTh = null;
                  latestRawPrimaryRomDeg = null;
                  latestLandmarks = null;
                  renderPrimaryLandmarks = null;
                  landmarkSmoother.reset();
                  pose2dOneEuroFilter.reset();
                  worldOneEuroFilter.reset();
                }
              }
            } catch {
              // Ignore transient inference errors
            }
          }

          // Step B: 1:1 landmark tracking & instantaneous ROM update + hold timer + canvas redraw
          const activeRules = formRulesRef.current;
          const checkLimb = isFormRuleActive(activeRules, 'limb');
          const checkPlane = isFormRuleActive(activeRules, 'plane');
          const checkTrunk = isFormRuleActive(activeRules, 'trunk');
          const checkPosture = isFormRuleActive(activeRules, 'posture');
          const checkIdentity = isFormRuleActive(activeRules, 'identity');
          const holdTargetMs = Math.max(1000, curEx.hold_sec * 1000);
          if (latestPoseCount > 1 && checkIdentity) {
            holdAccumMs = 0;
            if (lastEmittedHold !== curEx.hold_sec) {
              lastEmittedHold = curEx.hold_sec;
              lastHoldEmitAt = now;
              setDemoHoldRemainingSec(curEx.hold_sec);
            }
            emitPresetState('MULTI_PERSON');
            drawLivePoseOverlayOnCanvas(poseCanvasRef.current, latestAllLandmarks, {
              targetJoint: curEx.target_joint,
              exerciseMeta: curEx,
              romDeg: targetRomDeg !== null ? smoothedRom : null,
              targetRomDeg: curEx.target_rom_deg,
              holdRatio: 0,
              trunkTiltDeg: 0,
              isMultiPerson: true,
              isImpostorOrEmergency: false,
              videoWidth: v.videoWidth,
              videoHeight: v.videoHeight,
            });
          } else if (demoPresetRef.current === 'IMPOSTOR' && checkIdentity) {
            holdAccumMs = 0;
            drawLivePoseOverlayOnCanvas(poseCanvasRef.current, latestAllLandmarks, {
              targetJoint: curEx.target_joint,
              exerciseMeta: curEx,
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

            const effRawTiltDeg = checkTrunk ? latestRawTiltDeg : 0;
            const effWrongSideRaised = checkLimb ? latestWrongSideRaised : false;
            const effWrongSideBannerTh = checkLimb ? latestWrongSideBannerTh : null;
            const effWrongPlaneRaised = checkPlane ? latestWrongPlaneRaised : false;
            const effWrongPlaneBannerTh = checkPlane ? latestWrongPlaneBannerTh : null;
            const effPostureCompensated = checkPosture ? latestPostureCompensated : false;
            const effPostureBannerTh = checkPosture ? latestPostureBannerTh : null;
            targetRomDeg =
              effWrongSideRaised || effWrongPlaneRaised ? 0 : latestRawPrimaryRomDeg;

            const tiltDelta = effRawTiltDeg - smoothedTilt;
            if (!checkTrunk) {
              smoothedTilt = 0;
            } else if (Math.abs(tiltDelta) >= 0.4) {
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
              if (Math.abs(nextEmitRom - lastEmittedRom) >= 0.2 || hitBoundary) {
                lastEmittedRom = nextEmitRom;
                lastRomEmitAt = now;
                setDemoRomDeg(nextEmitRom);
              }

              const isKneeExercise = curEx.target_joint.includes('knee');
              const isKneeExtension = isKneeExercise && curEx.target_rom_deg >= 140;
              const isKneeFlexion = isKneeExercise && curEx.target_rom_deg < 140;
              const targetThreshold = isKneeFlexion
                ? curEx.target_rom_deg + 15
                : isKneeExtension
                  ? curEx.target_rom_deg * 0.92
                  : curEx.target_rom_deg * 0.82;
              const holdMaintainThreshold = isKneeFlexion
                ? curEx.target_rom_deg + 22
                : isKneeExtension
                  ? curEx.target_rom_deg * 0.85
                  : curEx.target_rom_deg * 0.75;
              const atTarget = isKneeFlexion
                ? smoothedRom <= targetThreshold
                : smoothedRom >= targetThreshold;
              const atMaintainTarget = isKneeFlexion
                ? smoothedRom <= holdMaintainThreshold
                : smoothedRom >= holdMaintainThreshold;
              const atRest = isKneeFlexion
                ? smoothedRom >= 145
                : isKneeExtension
                  ? smoothedRom <= 125
                  : smoothedRom <= Math.min(55, curEx.target_rom_deg * 0.45);

              if (atRest && now - lastRepCompletedAt >= 1200) {
                awaitingReturn = false;
              }

              const blockCounting =
                (demoPresetRef.current === 'MULTI_PERSON' && checkIdentity) ||
                countingLockedRef.current;

              const isCompensatedNow =
                (checkTrunk && latestTrunkTiltDeg > 12) || effPostureCompensated;

              if (demoValidRepsRef.current < curEx.target_reps) {
                if (effWrongSideRaised) {
                  holdAccumMs = 0;
                  if (lastEmittedHold !== curEx.hold_sec) {
                    lastEmittedHold = curEx.hold_sec;
                    lastHoldEmitAt = now;
                    setDemoHoldRemainingSec(curEx.hold_sec);
                  }
                  emitPresetState('WRONG_LIMB');
                } else if (effWrongPlaneRaised) {
                  holdAccumMs = 0;
                  if (lastEmittedHold !== curEx.hold_sec) {
                    lastEmittedHold = curEx.hold_sec;
                    lastHoldEmitAt = now;
                    setDemoHoldRemainingSec(curEx.hold_sec);
                  }
                  emitPresetState('WRONG_PLANE');
                } else if (!blockCounting) {
                  const holdingZone = atTarget || (holdAccumMs > 0 && atMaintainTarget);
                  if (holdingZone && !awaitingReturn) {
                    if (isCompensatedNow) {
                      // Freeze Hold Timer immediately when posture compensation is detected
                      emitPresetState('TRUNK_LEAN');
                    } else {
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
                          emitPresetState(next >= curEx.target_reps ? 'SET_FINISHED' : 'REP_COMPLETED');
                          return next;
                        });
                      } else {
                        emitPresetState('HOLD_PHASE');
                      }
                    }
                  } else if (awaitingReturn || now <= repCelebrateUntil) {
                    holdAccumMs = 0;
                    if (lastEmittedHold !== 0) {
                      lastEmittedHold = 0;
                      lastHoldEmitAt = now;
                      setDemoHoldRemainingSec(0);
                    }
                    emitPresetState(
                      demoValidRepsRef.current >= curEx.target_reps ? 'SET_FINISHED' : 'REP_COMPLETED',
                    );
                  } else {
                    // Immediate Hold Timer reset on leaving the target zone before completion
                    holdAccumMs = 0;
                    if (lastEmittedHold !== curEx.hold_sec) {
                      lastEmittedHold = curEx.hold_sec;
                      lastHoldEmitAt = now;
                      setDemoHoldRemainingSec(curEx.hold_sec);
                    }
                    emitPresetState(isCompensatedNow ? 'TRUNK_LEAN' : 'ACTIVE_REP');
                  }
                }
              }
            } else {
              holdAccumMs = 0;
              smoothedRom = 0;
              if (lastEmittedRom !== 0) {
                lastEmittedRom = 0;
                lastRomEmitAt = now;
                setDemoRomDeg(0);
              }
              if (lastEmittedHold !== curEx.hold_sec) {
                lastEmittedHold = curEx.hold_sec;
                lastHoldEmitAt = now;
                setDemoHoldRemainingSec(curEx.hold_sec);
              }
              emitPresetState('ACTIVE_REP');
            }

            drawLivePoseOverlayOnCanvas(poseCanvasRef.current, renderLandmarksList, {
              targetJoint: curEx.target_joint,
              expectedPlane: curExpectedPlane,
              exerciseMeta: curEx,
              romDeg: targetRomDeg !== null ? smoothedRom : null,
              targetRomDeg: curEx.target_rom_deg,
              holdRatio: holdAccumMs > 0 ? Math.min(1, holdAccumMs / holdTargetMs) : 0,
              trunkTiltDeg: latestTrunkTiltDeg,
              isMultiPerson: demoPresetRef.current === 'MULTI_PERSON' && checkIdentity,
              isImpostorOrEmergency: countingLockedRef.current,
              wrongSideRaised: effWrongSideRaised,
              wrongSideBannerTh: effWrongSideBannerTh,
              wrongPlaneRaised: effWrongPlaneRaised,
              wrongPlaneBannerTh: effWrongPlaneBannerTh,
              lowVisibilityAlertTh: latestLowVisibilityAlertTh,
              postureBannerTh: effPostureBannerTh,
              videoWidth: v.videoWidth,
              videoHeight: v.videoHeight,
              smoothingStatusText: formatSmoothingHud(),
            });
          } else {
            renderPrimaryLandmarks = null;
            drawLivePoseOverlayOnCanvas(poseCanvasRef.current, [], {
              targetJoint: curEx.target_joint,
              exerciseMeta: curEx,
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
    isEdgeDirectCamera,
    setDemoHoldRemainingSec,
    setDemoPresetState,
    setDemoRomDeg,
    setDemoTrunkTiltDeg,
    setDemoValidReps,
  ]);

  const patientRevealed = Boolean(patient?.name && (kiosk_state !== 'AUTH_FRAMING' || demoFaceMatched));

  const checklistItems: {
    ruleKey: FormAssessmentRuleKey;
    label: string;
    ok: boolean;
    ruleActive: boolean;
    warnText: string;
  }[] = [
    ...(selectedExercise.target_joint === 'left_shoulder' ||
    selectedExercise.target_joint === 'right_shoulder'
      ? [
          {
            ruleKey: 'limb' as const,
            label:
              selectedExercise.target_joint === 'left_shoulder'
                ? 'ยกเฉพาะแขนซ้ายตามท่าที่กำหนด'
                : 'ยกเฉพาะแขนขวาตามท่าที่กำหนด',
            ok: !wrongLimbAlert && !isEmergency,
            ruleActive: isFormRuleActive(formRules, 'limb'),
            warnText:
              selectedExercise.target_joint === 'left_shoulder'
                ? 'กรุณายกแขนซ้าย'
                : 'กรุณายกแขนขวา',
          },
          ...(activeExpectedPlane === 'flexion' || activeExpectedPlane === 'abduction'
            ? [
                {
                  ruleKey: 'plane' as const,
                  label:
                    activeExpectedPlane === 'flexion'
                      ? 'ทิศทางถูกต้อง (ยกแขนไปด้านหน้าเท่านั้น)'
                      : 'ทิศทางถูกต้อง (กางแขนออกด้านข้างเท่านั้น)',
                  ok: !wrongPlaneAlert && !isEmergency,
                  ruleActive: isFormRuleActive(formRules, 'plane'),
                  warnText:
                    activeExpectedPlane === 'flexion' ? 'ห้ามกางด้านข้าง' : 'ห้ามยกด้านหน้า',
                },
              ]
            : []),
        ]
      : []),
    {
      ruleKey: 'trunk',
      label: 'ลำตัวตั้งตรง ไม่เอียงชดเชย',
      ok: !trunkLeanAlert && !isEmergency,
      ruleActive: isFormRuleActive(formRules, 'trunk'),
      warnText: 'เอียงลำตัว',
    },
    {
      ruleKey: 'posture',
      label:
        selectedExercise.target_joint === 'right_knee'
          ? 'เข่าตรงแนว ไม่บิดเข้าด้านใน'
          : 'ผ่อนหัวไหล่ ไม่ยักไหล่ขึ้น',
      ok:
        selectedExercise.target_joint === 'right_knee'
          ? !kneeValgusAlert && !isEmergency
          : !shoulderHikeAlert && !isEmergency,
      ruleActive: isFormRuleActive(formRules, 'posture'),
      warnText: selectedExercise.target_joint === 'right_knee' ? 'เข่าบิดใน' : 'ยักไหล่',
    },
    {
      ruleKey: 'velocity',
      label: 'ความเร็วเคลื่อนไหวสม่ำเสมอ',
      ok: !tooFastAlert && !isEmergency,
      ruleActive: isFormRuleActive(formRules, 'velocity'),
      warnText: 'เร็วเกินไป',
    },
    {
      ruleKey: 'identity',
      label: 'ยืนยันตัวตนคนไข้ตัวจริง (1 คนในเฟรม)',
      ok:
        !isFormRuleActive(formRules, 'identity') ||
        (kiosk_state !== 'PAUSED_MULTI_PERSON' &&
          kiosk_state !== 'PAUSED_IMPOSTOR' &&
          payload.identity_guard?.is_counting_allowed !== false &&
          !isEmergency),
      ruleActive: isFormRuleActive(formRules, 'identity'),
      warnText:
        kiosk_state === 'PAUSED_IMPOSTOR' || payload.identity_guard?.status === 'IMPOSTOR'
          ? 'ไม่ใช่คนไข้'
          : 'พบหลายคน',
    },
  ];
  const hasDisabledFormRule =
    !formRules.enabled || checklistItems.some((item) => !item.ruleActive);

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
                !isDemoMode && (connectionStatus === 'connected' || livePoseActive)
                  ? 'border-emerald-200 bg-emerald-50 text-[#0D9488]'
                  : 'border-sky-200 bg-sky-50 text-[#0284C7]'
              }`}
            >
              <span className="relative flex h-2.5 w-2.5">
                <span
                  className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-75 ${
                    !isDemoMode && (connectionStatus === 'connected' || livePoseActive)
                      ? 'bg-[#10B981]'
                      : 'bg-[#38BDF8]'
                  }`}
                />
                <span
                  className={`relative inline-flex h-2.5 w-2.5 rounded-full ${
                    !isDemoMode && (connectionStatus === 'connected' || livePoseActive)
                      ? 'bg-[#10B981]'
                      : 'bg-[#0284C7]'
                  }`}
                />
              </span>
              {!isDemoMode && (connectionStatus === 'connected' || livePoseActive) ? (
                <>
                  <Wifi className="h-3.5 w-3.5" />
                  <span>
                    {connectionStatus === 'connected'
                      ? 'Connected to Edge Vision (Heavy 3D)'
                      : 'Live AI Heavy 3D Active'}
                  </span>
                </>
              ) : !isDemoMode ? (
                <>
                  <WifiOff className="h-3.5 w-3.5" />
                  <span>Edge Vision & Camera Paused</span>
                </>
              ) : (
                <>
                  <WifiOff className="h-3.5 w-3.5" />
                  <span>
                    {connectionStatus === 'connected'
                      ? 'Demo Simulator (Live Paused)'
                      : 'Demo Simulator Mode'}
                  </span>
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
              onClick={handleToggleDemoMode}
              className={`inline-flex items-center gap-2 rounded-2xl border px-3.5 py-2 text-xs font-bold transition-all sm:text-sm ${
                isDemoMode
                  ? 'border-[#38BDF8] bg-[#E0F2FE] text-[#0F4C81]'
                  : 'border-emerald-300 bg-emerald-50 text-[#0D9488]'
              }`}
            >
              <SlidersHorizontal className="h-4 w-4" />
              <span>
                {isDemoMode
                  ? 'Demo Simulator: ON'
                  : isEdgeDirectCamera
                    ? 'Live Edge: ON'
                    : 'Demo Simulator: OFF'}
              </span>
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

        {/* Camera Hardware Alert Banner (only shown if local camera access fails) */}
        {!isDemoMode && !isEdgeDirectCamera && cameraError && (
          <div
            role="alert"
            className="glass-card flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-rose-300/90 bg-rose-50/95 px-4 py-2.5 text-rose-950 shadow-sm"
          >
            <div className="flex min-w-0 flex-1 items-center gap-2.5">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-rose-500/15 text-rose-700">
                <AlertTriangle className="h-4 w-4" />
              </div>
              <p className="text-xs font-extrabold text-rose-950">
                ไม่สามารถเปิดกล้องเว็บแคมได้: {cameraError}
              </p>
            </div>
            <button
              type="button"
              onClick={() => {
                setCameraError(null);
                setLivePoseActive(false);
                window.setTimeout(() => setLivePoseActive(true), 120);
              }}
              className="inline-flex items-center gap-1.5 rounded-xl bg-rose-600 px-3 py-1.5 text-xs font-extrabold text-white shadow-sm hover:bg-rose-500"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              <span>ลองเปิดกล้องใหม่</span>
            </button>
          </div>
        )}

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
          <section className="lg:col-span-8 lg:sticky lg:top-4 lg:self-start">
            <SkeletonCanvas
              payload={payload}
              exercise={selectedExercise}
              isDemoMode={isDemoMode}
              distanceM={demoDistanceM}
              trunkTiltDeg={demoTrunkTiltDeg}
              hasLiveCamera={!isEdgeDirectCamera && livePoseActive && !cameraError}
              liveVideoOverlay={
                !isEdgeDirectCamera && livePoseActive && !cameraError ? (
                  <div className="relative h-full w-full bg-[#0B192C]">
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      muted
                      className="h-full w-full object-contain"
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
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      <div
                        className={`inline-flex items-center gap-1.5 rounded-xl border px-3 py-1 text-xs font-extrabold ${
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
                      {(activeExpectedPlane === 'flexion' || activeExpectedPlane === 'abduction') && (
                        <div className="inline-flex items-center gap-1.5 rounded-xl border border-sky-300 bg-sky-50 px-3 py-1 text-xs font-extrabold text-[#0F4C81]">
                          <span>
                            {activeExpectedPlane === 'flexion'
                              ? '⬆️ ทิศทาง: ยกตรงไปด้านหน้า • ตรวจจับมุม 3 มิติ (แกน X-Y-Z)'
                              : '↔️ ทิศทาง: กางออกด้านข้าง • ตรวจจับมุม 3 มิติ (แกน X-Y-Z)'}
                          </span>
                        </div>
                      )}
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

              <div className="mt-3 flex max-h-36 flex-wrap gap-1.5 overflow-y-auto pr-1">
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
              <div className="flex flex-wrap items-center justify-between gap-2">
                <button
                  type="button"
                  onClick={() => setFormChecklistExpanded((prev) => !prev)}
                  className="flex items-center gap-2 text-left text-sm font-extrabold text-[#0F4C81] hover:text-[#0284C7] cursor-pointer"
                  aria-expanded={formChecklistExpanded}
                >
                  <Activity
                    className={`h-4 w-4 shrink-0 ${formRules.enabled ? 'text-[#0D9488]' : 'text-slate-400'}`}
                  />
                  <span>การประเมินท่าทางแบบเรียลไทม์</span>
                </button>

                <div className="flex items-center gap-1.5">
                  <button
                    type="button"
                    onClick={toggleMasterFormAssessment}
                    aria-pressed={formRules.enabled}
                    title={formRules.enabled ? 'กดเพื่อปิดระบบประเมินท่าทางแบบเรียลไทม์' : 'กดเพื่อเปิดระบบประเมินท่าทางแบบเรียลไทม์'}
                    className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-extrabold transition-colors cursor-pointer ${
                      formRules.enabled
                        ? 'border-emerald-300 bg-emerald-50 text-[#0D9488] hover:bg-emerald-100'
                        : 'border-slate-300 bg-slate-100 text-slate-600 hover:bg-slate-200'
                    }`}
                  >
                    <span
                      className={`inline-flex h-3.5 w-6 items-center rounded-full p-0.5 transition-colors ${
                        formRules.enabled ? 'bg-[#10B981] justify-end' : 'bg-slate-400 justify-start'
                      }`}
                    >
                      <span className="h-2.5 w-2.5 rounded-full bg-white shadow-sm" />
                    </span>
                    <span>{formRules.enabled ? 'เปิดประเมิน' : 'ปิดประเมิน'}</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => setFormChecklistExpanded((prev) => !prev)}
                    aria-label={formChecklistExpanded ? 'ซ่อนรายการประเมินท่าทาง' : 'เปิดดูรายการประเมินท่าทาง'}
                    className="inline-flex items-center gap-1 rounded-full border border-sky-200 bg-white/80 px-2.5 py-1 text-[11px] font-bold text-[#0F4C81] hover:bg-sky-50 cursor-pointer"
                  >
                    <span>{formChecklistExpanded ? 'ซ่อน' : 'เปิดดู'}</span>
                    {formChecklistExpanded ? (
                      <ChevronUp className="h-3.5 w-3.5" />
                    ) : (
                      <ChevronDown className="h-3.5 w-3.5" />
                    )}
                  </button>
                </div>
              </div>

              <AnimatePresence initial={false}>
                {formChecklistExpanded ? (
                  <motion.div
                    key="form-checklist-expanded"
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.2, ease: 'easeOut' }}
                    className="overflow-hidden"
                  >
                    <div className="mt-2.5 mb-2.5 flex flex-wrap items-center justify-between gap-2 text-xs">
                      <span className="tabular-mono font-bold text-[#0284C7]">
                        {isFormRuleActive(formRules, 'trunk')
                          ? `เอียงลำตัว ${demoTrunkTiltDeg.toFixed(1)}°`
                          : 'เอียงลำตัว: ปิดเช็ก'}
                      </span>
                      <div className="flex items-center gap-2">
                        <span className="text-[11px] font-semibold text-slate-500">
                          กดแต่ละหัวข้อเพื่อเปิด/ปิด
                        </span>
                        {hasDisabledFormRule && (
                          <button
                            type="button"
                            onClick={resetAllFormRules}
                            className="inline-flex items-center gap-1 rounded-full border border-sky-200 bg-sky-50 px-2 py-0.5 text-[11px] font-bold text-[#0284C7] hover:bg-sky-100 cursor-pointer"
                          >
                            <RotateCcw className="h-3 w-3" />
                            <span>เปิดทั้งหมด</span>
                          </button>
                        )}
                      </div>
                    </div>

                    {!formRules.enabled && (
                      <div className="mb-2.5 flex items-center justify-between gap-2 rounded-2xl border border-slate-200 bg-slate-50 px-3.5 py-2 text-xs font-semibold text-slate-600">
                        <span>ปิดการประเมินท่าทางอยู่ • นับรอบตามมุมองศาอย่างเดียว</span>
                        <button
                          type="button"
                          onClick={toggleMasterFormAssessment}
                          className="shrink-0 rounded-full bg-emerald-600 px-2.5 py-1 text-[11px] font-bold text-white hover:bg-emerald-500 cursor-pointer"
                        >
                          กดเปิดประเมิน
                        </button>
                      </div>
                    )}

                    <div className="space-y-2">
                      {checklistItems.map((item) => (
                        <button
                          key={item.label}
                          type="button"
                          onClick={() => toggleSingleFormRule(item.ruleKey)}
                          aria-pressed={item.ruleActive}
                          className={`flex w-full items-center justify-between gap-2 rounded-2xl border px-3.5 py-2.5 text-left text-sm font-semibold transition-colors cursor-pointer ${
                            !item.ruleActive
                              ? 'border-dashed border-slate-200 bg-slate-50/70 text-slate-400 hover:border-slate-300 hover:bg-slate-100/70'
                              : item.ok
                                ? 'border-emerald-100 bg-emerald-50/60 text-[#0F4C81] hover:bg-emerald-50'
                                : 'border-amber-300 bg-amber-50 text-amber-900 hover:bg-amber-100/70'
                          }`}
                        >
                          <span className="flex items-center gap-2">
                            {!item.ruleActive ? (
                              <Pause className="h-4 w-4 shrink-0 text-slate-400" />
                            ) : item.ok ? (
                              <CheckCircle2 className="h-4 w-4 shrink-0 text-[#10B981]" />
                            ) : (
                              <AlertTriangle className="h-4 w-4 shrink-0 text-[#F59E0B]" />
                            )}
                            <span className={!item.ruleActive ? 'line-through decoration-slate-300' : ''}>
                              {item.label}
                            </span>
                          </span>
                          <span
                            className={`shrink-0 rounded-full px-2 py-0.5 text-[11px] font-bold ${
                              !item.ruleActive
                                ? 'bg-slate-200 text-slate-600'
                                : item.ok
                                  ? 'bg-emerald-100 text-[#0D9488]'
                                  : 'bg-amber-200 text-amber-900'
                            }`}
                          >
                            {!item.ruleActive ? 'ปิดอยู่' : item.ok ? 'ผ่าน' : item.warnText}
                          </span>
                        </button>
                      ))}
                    </div>
                  </motion.div>
                ) : (
                  <div className="mt-2 flex items-center justify-between text-xs font-semibold text-slate-500">
                    <span>
                      {formRules.enabled
                        ? `กำลังเปิดประเมิน (${checklistItems.filter((item) => item.ruleActive).length}/${checklistItems.length} หัวข้อ)`
                        : 'ปิดการประเมินท่าทางอยู่ • นับรอบตามมุมองศาอย่างเดียว'}
                    </span>
                    <button
                      type="button"
                      onClick={() => setFormChecklistExpanded(true)}
                      className="font-bold text-[#0284C7] hover:underline cursor-pointer"
                    >
                      คลิกเพื่อเปิดดูรายการ
                    </button>
                  </div>
                )}
              </AnimatePresence>
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
                                : p.id === 'TRUNK_LEAN' ||
                                    p.id === 'MULTI_PERSON' ||
                                    p.id === 'WRONG_LIMB' ||
                                    p.id === 'WRONG_PLANE'
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

