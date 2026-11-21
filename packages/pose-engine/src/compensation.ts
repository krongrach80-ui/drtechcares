import { POSE_INDEX } from './armJointProcessor';
import type { BodySide, CompensationAlert, PoseLandmark } from './types';

/**
 * Anti-cheat clinical compensation checks:
 * - trunk_lean: lateral torso tilt > 14°
 * - scapular_hiking: excessive shoulder elevation asymmetry
 * - elbow_flexion_cheat: bent elbow (< 135°) during straight-arm raises
 * - elevation_plane: verifies flexion vs abduction plane alignment
 */
export function evaluateCompensations(
  landmarks: readonly PoseLandmark[],
  side: BodySide,
  options: {
    elbowAngleDeg: number | null;
    requireStraightElbow?: boolean;
    minElbowDeg?: number;
    expectedPlane?: 'flexion' | 'abduction' | 'any';
  },
): CompensationAlert[] {
  const alerts: CompensationAlert[] = [];
  const lShoulder = landmarks[POSE_INDEX.LEFT_SHOULDER];
  const rShoulder = landmarks[POSE_INDEX.RIGHT_SHOULDER];
  const lHip = landmarks[POSE_INDEX.LEFT_HIP];
  const rHip = landmarks[POSE_INDEX.RIGHT_HIP];

  const lHipVis = (lHip?.visibility ?? 1) >= 0.45;
  const rHipVis = (rHip?.visibility ?? 1) >= 0.45;
  const lShVis = (lShoulder?.visibility ?? 1) >= 0.45;
  const rShVis = (rShoulder?.visibility ?? 1) >= 0.45;

  if (lShoulder && rShoulder && lShVis && rShVis) {
    if (lHip && rHip && lHipVis && rHipVis) {
      const midShoulderX = (lShoulder.x + rShoulder.x) / 2;
      const midShoulderY = (lShoulder.y + rShoulder.y) / 2;
      const midHipX = (lHip.x + rHip.x) / 2;
      const midHipY = (lHip.y + rHip.y) / 2;

      const trunkAngleDeg = Math.abs(
        (Math.atan2(midShoulderX - midHipX, Math.max(0.05, midHipY - midShoulderY)) * 180) / Math.PI,
      );
      if (trunkAngleDeg > 14) {
        alerts.push({
          type: 'trunk_lean',
          severity: 'warning',
          message: {
            th: 'กรุณาตั้งลำตัวให้ตรง อย่าเอียงตัวช่วยยก',
            en: 'Keep your torso upright; avoid leaning to lift.',
            cn: '请保持躯干直立，不要借力侧倾。',
          },
        });
      }
    }

    const shoulderTilt = Math.abs(lShoulder.y - rShoulder.y);
    if (shoulderTilt > 0.08) {
      alerts.push({
        type: 'scapular_hiking',
        severity: 'warning',
        message: {
          th: 'ผ่อนไหล่ลง อย่ายักไหล่ขึ้นสูง',
          en: 'Relax your shoulder; avoid shrugging upward.',
          cn: '请放松肩膀，不要耸肩。',
        },
      });
    }
  }

  const minElbow = options.minElbowDeg ?? 135;
  if (options.requireStraightElbow && options.elbowAngleDeg !== null && options.elbowAngleDeg < minElbow) {
    alerts.push({
      type: 'elbow_flexion_cheat',
      severity: 'blocking',
      message: {
        th: 'กรุณาเหยียดข้อศอกให้ตรง',
        en: 'Please straighten your elbow.',
        cn: '请伸直手肘。',
      },
    });
  }

  if (options.expectedPlane && options.expectedPlane !== 'any') {
    const shoulder = landmarks[side === 'left' ? POSE_INDEX.LEFT_SHOULDER : POSE_INDEX.RIGHT_SHOULDER];
    const wrist = landmarks[side === 'left' ? POSE_INDEX.LEFT_WRIST : POSE_INDEX.RIGHT_WRIST];
    if (shoulder && wrist) {
      const lateralDx = Math.abs(wrist.x - shoulder.x);
      const depthDz = Math.abs((wrist.z ?? 0) - (shoulder.z ?? 0));
      if (options.expectedPlane === 'abduction' && lateralDx < 0.08 && depthDz > 0.14) {
        alerts.push({
          type: 'elevation_plane',
          severity: 'warning',
          message: {
            th: 'กรุณากางแขนออกด้านข้างให้ถูกระนาบ',
            en: 'Please raise your arm out to the side in the abduction plane.',
            cn: '请向身体侧面外展抬臂。',
          },
        });
      }
    }
  }

  return alerts;
}

