import type { BodySide, PoseLandmark, TrilingualText } from './types';

export const MIN_LANDMARK_VISIBILITY = 0.5;
export const FRAME_BOUNDARY_MIN = 0.02;
export const FRAME_BOUNDARY_MAX = 0.98;
export const MIN_STRAIGHT_ELBOW_DEG = 135;

export const POSE_INDEX = {
  NOSE: 0,
  LEFT_EYE: 2,
  RIGHT_EYE: 5,
  LEFT_EAR: 7,
  RIGHT_EAR: 8,
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,
  RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,
  RIGHT_WRIST: 16,
  LEFT_PINKY: 17,
  RIGHT_PINKY: 18,
  LEFT_INDEX: 19,
  RIGHT_INDEX: 20,
  LEFT_THUMB: 21,
  RIGHT_THUMB: 22,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
  LEFT_KNEE: 25,
  RIGHT_KNEE: 26,
  LEFT_ANKLE: 27,
  RIGHT_ANKLE: 28,
  LEFT_HEEL: 29,
  RIGHT_HEEL: 30,
  LEFT_FOOT_INDEX: 31,
  RIGHT_FOOT_INDEX: 32,
} as const;

export interface LandmarkGateResult {
  valid: boolean;
  reason: 'OK' | 'OUT_OF_FRAME' | 'LOW_VISIBILITY';
  missingJoints: { index: number; side: BodySide | 'center'; joint: 'shoulder' | 'elbow' | 'wrist' | 'hip' | 'knee' | 'ankle' }[];
  warningMessage: TrilingualText | null;
}

export interface ArmKinematicsResult {
  angleDeg: number | null;
  elbowAngleDeg: number | null;
  gate: LandmarkGateResult;
  elbowGuardFailed: boolean;
  elbowWarning: TrilingualText | null;
}

function isInFrame(lm: PoseLandmark): boolean {
  return (
    Number.isFinite(lm.x) &&
    Number.isFinite(lm.y) &&
    lm.x >= FRAME_BOUNDARY_MIN &&
    lm.x <= FRAME_BOUNDARY_MAX &&
    lm.y >= FRAME_BOUNDARY_MIN &&
    lm.y <= FRAME_BOUNDARY_MAX
  );
}

function isVisible(lm: PoseLandmark | undefined): lm is PoseLandmark {
  if (!lm) return false;
  const vis = lm.visibility ?? 1.0;
  return vis >= MIN_LANDMARK_VISIBILITY;
}

function jointLabel(
  joint: 'shoulder' | 'elbow' | 'wrist' | 'hip' | 'knee' | 'ankle',
  side: BodySide | 'center',
): TrilingualText {
  const map = {
    shoulder: { th: 'หัวไหล่', en: 'shoulder', cn: '肩膀' },
    elbow: { th: 'ข้อศอก', en: 'elbow', cn: '手肘' },
    wrist: { th: 'ข้อมือ', en: 'wrist', cn: '手腕' },
    hip: { th: 'สะโพก', en: 'hip', cn: '髋部' },
    knee: { th: 'หัวเข่า', en: 'knee', cn: '膝盖' },
    ankle: { th: 'ข้อเท้า', en: 'ankle', cn: '脚踝' },
  }[joint];
  const sideLabel =
    side === 'left'
      ? { th: 'ซ้าย', en: 'left ', cn: '左' }
      : side === 'right'
        ? { th: 'ขวา', en: 'right ', cn: '右' }
        : { th: '', en: '', cn: '' };

  return {
    th: `${map.th}${sideLabel.th}`,
    en: `${sideLabel.en}${map.en}`,
    cn: `${sideLabel.cn}${map.cn}`,
  };
}

/**
 * Computes 3D (or 2D fallback) joint angle at vertex B formed by A-B-C in degrees [0..180].
 */
export function computeJointAngle3D(a: PoseLandmark, b: PoseLandmark, c: PoseLandmark): number {
  if (
    !Number.isFinite(a?.x) ||
    !Number.isFinite(a?.y) ||
    !Number.isFinite(b?.x) ||
    !Number.isFinite(b?.y) ||
    !Number.isFinite(c?.x) ||
    !Number.isFinite(c?.y)
  ) {
    return 0;
  }
  const az = Number.isFinite(a.z) ? (a.z as number) : 0;
  const bz = Number.isFinite(b.z) ? (b.z as number) : 0;
  const cz = Number.isFinite(c.z) ? (c.z as number) : 0;

  const bax = a.x - b.x;
  const bay = a.y - b.y;
  const baz = az - bz;

  const bcx = c.x - b.x;
  const bcy = c.y - b.y;
  const bcz = cz - bz;

  const magBA = Math.hypot(bax, bay, baz);
  const magBC = Math.hypot(bcx, bcy, bcz);
  if (!Number.isFinite(magBA) || !Number.isFinite(magBC) || magBA < 1e-6 || magBC < 1e-6) return 0;

  const cos = Math.max(-1, Math.min(1, (bax * bcx + bay * bcy + baz * bcz) / (magBA * magBC)));
  const deg = (Math.acos(cos) * 180) / Math.PI;
  return Number.isFinite(deg) ? Number(deg.toFixed(2)) : 0;
}

/**
 * Checks Visibility (>= 0.5) & Boundary ([0.02, 0.98]) on required joints before computing any angle.
 */
export function checkVisibilityAndBoundaryGate(
  landmarks: readonly PoseLandmark[],
  required: readonly { index: number; side: BodySide | 'center'; joint: 'shoulder' | 'elbow' | 'wrist' | 'hip' | 'knee' | 'ankle' }[],
): LandmarkGateResult {
  const outOfFrameJoints: typeof required[number][] = [];
  const lowVisJoints: typeof required[number][] = [];

  for (const req of required) {
    const lm = landmarks[req.index];
    if (!isVisible(lm)) {
      lowVisJoints.push(req);
      continue;
    }
    if (!isInFrame(lm)) {
      outOfFrameJoints.push(req);
    }
  }

  if (outOfFrameJoints.length > 0) {
    const hasArmOut = outOfFrameJoints.some((j) => j.joint === 'wrist' || j.joint === 'elbow');
    return {
      valid: false,
      reason: 'OUT_OF_FRAME',
      missingJoints: [...outOfFrameJoints],
      warningMessage: hasArmOut
        ? {
            th: 'ข้อมือ/ข้อศอกอยู่นอกกรอบกล้อง กรุณาถอยห่างจากกล้องเล็กน้อย',
            en: 'Wrist/elbow is outside the camera frame. Please step back slightly.',
            cn: '手腕或手肘超出画面范围，请稍微向后退一步。',
          }
        : {
            th: 'ร่างกายบางส่วนอยู่นอกกรอบกล้อง กรุณาถอยห่างจากกล้องเล็กน้อย',
            en: 'Part of your body is outside the camera frame. Please step back slightly.',
            cn: '部分身体超出镜头范围，请稍微向后退。',
          },
    };
  }

  if (lowVisJoints.length > 0) {
    const first = lowVisJoints[0];
    const lbl = jointLabel(first.joint, first.side);
    const armText =
      first.side === 'left'
        ? { th: 'แขนซ้าย', en: 'your left arm', cn: '左臂' }
        : first.side === 'right'
          ? { th: 'แขนขวา', en: 'your right arm', cn: '右臂' }
          : { th: 'ร่างกาย', en: 'your body', cn: '身体' };

    return {
      valid: false,
      reason: 'LOW_VISIBILITY',
      missingJoints: [...lowVisJoints],
      warningMessage: {
        th: `มองไม่เห็น${lbl.th} กรุณาขยับ${armText.th}ให้อยู่ในกรอบกล้อง`,
        en: `Cannot see ${lbl.en}. Please move ${armText.en} into the camera frame.`,
        cn: `无法识别${lbl.cn}，请将${armText.cn}移入画面内。`,
      },
    };
  }

  return {
    valid: true,
    reason: 'OK',
    missingJoints: [],
    warningMessage: null,
  };
}

/**
 * Evaluates Arm/Shoulder elevation angle and optional Straight-Arm Elbow Guard (>= 135°).
 * Never returns 0° or 180° when landmarks fail visibility/boundary gating; returns angleDeg: null instead.
 */
export function evaluateArmKinematics(
  landmarks: readonly PoseLandmark[],
  side: BodySide,
  options: {
    jointGroup?: 'shoulder' | 'elbow';
    requireStraightElbow?: boolean;
    minElbowDeg?: number;
  } = {},
): ArmKinematicsResult {
  const shoulderIdx = side === 'left' ? POSE_INDEX.LEFT_SHOULDER : POSE_INDEX.RIGHT_SHOULDER;
  const elbowIdx = side === 'left' ? POSE_INDEX.LEFT_ELBOW : POSE_INDEX.RIGHT_ELBOW;
  const wristIdx = side === 'left' ? POSE_INDEX.LEFT_WRIST : POSE_INDEX.RIGHT_WRIST;
  const hipIdx = side === 'left' ? POSE_INDEX.LEFT_HIP : POSE_INDEX.RIGHT_HIP;

  const gate = checkVisibilityAndBoundaryGate(landmarks, [
    { index: shoulderIdx, side, joint: 'shoulder' },
    { index: elbowIdx, side, joint: 'elbow' },
    { index: wristIdx, side, joint: 'wrist' },
  ]);

  if (!gate.valid) {
    return {
      angleDeg: null,
      elbowAngleDeg: null,
      gate,
      elbowGuardFailed: false,
      elbowWarning: null,
    };
  }

  const shoulder = landmarks[shoulderIdx];
  const elbow = landmarks[elbowIdx];
  const wrist = landmarks[wristIdx];
  const rawHip = landmarks[hipIdx];
  // In seated / wheelchair upper-body framing, hip may be cropped below frame; fall back to vertical torso axis
  const hip: PoseLandmark =
    rawHip && isVisible(rawHip) && isInFrame(rawHip)
      ? rawHip
      : { x: shoulder.x, y: Math.min(0.99, shoulder.y + 0.35), z: shoulder.z ?? 0, visibility: 1 };

  const elbowAngleDeg = computeJointAngle3D(shoulder, elbow, wrist);
  const jointGroup = options.jointGroup ?? 'shoulder';
  const angleDeg =
    jointGroup === 'elbow'
      ? elbowAngleDeg
      : computeJointAngle3D(hip, shoulder, wrist);

  const minElbow = options.minElbowDeg ?? MIN_STRAIGHT_ELBOW_DEG;
  const elbowGuardFailed = Boolean(options.requireStraightElbow && elbowAngleDeg < minElbow);

  return {
    angleDeg,
    elbowAngleDeg,
    gate,
    elbowGuardFailed,
    elbowWarning: elbowGuardFailed
      ? {
          th: 'กรุณาเหยียดข้อศอกให้ตรง',
          en: 'Please keep your elbow straight.',
          cn: '请保持手肘伸直。',
        }
      : null,
  };
}

