import type { Landmark3D } from './embed';

export const MIN_DEPTH_SCORE = 0.28;
export const BLINK_EAR_CLOSED = 0.20;
export const BLINK_EAR_OPEN = 0.26;
export const MIN_MICRO_MOTION = 0.0008;

export interface LivenessAndPoseReport {
  isLive: boolean;
  faceCount: number;
  depthScore: number;
  microMotion: number;
  earLeft: number;
  earRight: number;
  blinkDetected: boolean;
  yawDeg: number;
  pitchDeg: number;
  rollDeg: number;
  reasons: string[];
}

function dist2D(a: Landmark3D, b: Landmark3D): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function computeEyeAspectRatio(
  p1: Landmark3D,
  p2: Landmark3D,
  p3: Landmark3D,
  p4: Landmark3D,
  p5: Landmark3D,
  p6: Landmark3D,
): number {
  const verticalA = dist2D(p2, p6);
  const verticalB = dist2D(p3, p5);
  const horizontal = Math.max(1e-5, dist2D(p1, p4));
  return (verticalA + verticalB) / (2 * horizontal);
}

/**
 * Evaluates 3D depth protrusion, micro-motion, Eye Aspect Ratio (EAR) blink state,
 * and head pose angles (yaw, pitch, roll) from MediaPipe FaceLandmarker 3D landmarks.
 */
export function evaluateLivenessAndPose(
  landmarks3D: readonly Landmark3D[],
  prevLandmarks3D: readonly Landmark3D[] | null,
  faceCount: number,
  wasEyeClosed: boolean = false,
): LivenessAndPoseReport {
  if (faceCount !== 1 || landmarks3D.length < 300) {
    return {
      isLive: false,
      faceCount,
      depthScore: 0,
      microMotion: 0,
      earLeft: 0.3,
      earRight: 0.3,
      blinkDetected: false,
      yawDeg: 0,
      pitchDeg: 0,
      rollDeg: 0,
      reasons: [faceCount === 0 ? 'NO_FACE' : faceCount > 1 ? 'MULTIPLE_FACES' : 'INSUFFICIENT_LANDMARKS'],
    };
  }

  const noseTip = landmarks3D[1];
  const leftCheek = landmarks3D[234];
  const rightCheek = landmarks3D[454];
  const forehead = landmarks3D[10];
  const chin = landmarks3D[152];

  const faceWidth = Math.max(1e-4, dist2D(leftCheek, rightCheek));
  const faceHeight = Math.max(1e-4, dist2D(forehead, chin));

  // 3D depth protrusion: nose z relative to cheek plane normalized by faceWidth
  const cheekMeanZ = ((leftCheek.z ?? 0) + (rightCheek.z ?? 0)) / 2;
  const noseProtrusion = Math.abs(cheekMeanZ - (noseTip.z ?? 0));
  const depthScore = Math.min(1, noseProtrusion / (faceWidth * 0.25));

  // Head pose approximation (degrees)
  const midX = (leftCheek.x + rightCheek.x) / 2;
  const yawDeg = ((noseTip.x - midX) / faceWidth) * 90;
  const midY = (forehead.y + chin.y) / 2;
  const pitchDeg = ((noseTip.y - midY) / faceHeight) * 90;
  const rollDeg = (Math.atan2(rightCheek.y - leftCheek.y, rightCheek.x - leftCheek.x) * 180) / Math.PI;

  // Eye Aspect Ratio (MediaPipe standard eye contour indices)
  const earLeft = computeEyeAspectRatio(
    landmarks3D[33],
    landmarks3D[160],
    landmarks3D[158],
    landmarks3D[133],
    landmarks3D[153],
    landmarks3D[144],
  );
  const earRight = computeEyeAspectRatio(
    landmarks3D[362],
    landmarks3D[385],
    landmarks3D[387],
    landmarks3D[263],
    landmarks3D[373],
    landmarks3D[380],
  );
  const meanEar = (earLeft + earRight) / 2;
  const blinkDetected = wasEyeClosed && meanEar >= BLINK_EAR_OPEN;

  // Micro-motion across frames
  let microMotion = 0;
  if (prevLandmarks3D && prevLandmarks3D.length === landmarks3D.length) {
    let diffSum = 0;
    const sampleIndices = [1, 10, 33, 133, 152, 263, 362];
    for (const idx of sampleIndices) {
      diffSum += dist2D(landmarks3D[idx], prevLandmarks3D[idx]);
    }
    microMotion = diffSum / sampleIndices.length;
  }

  const reasons: string[] = [];
  if (depthScore < MIN_DEPTH_SCORE) reasons.push('FLAT_SPOOF_SUSPECTED');
  if (prevLandmarks3D && microMotion < MIN_MICRO_MOTION * 0.1) reasons.push('STATIC_IMAGE_SUSPECTED');

  return {
    isLive: reasons.length === 0,
    faceCount,
    depthScore: Number(depthScore.toFixed(3)),
    microMotion: Number(microMotion.toFixed(5)),
    earLeft: Number(earLeft.toFixed(3)),
    earRight: Number(earRight.toFixed(3)),
    blinkDetected,
    yawDeg: Number(yawDeg.toFixed(1)),
    pitchDeg: Number(pitchDeg.toFixed(1)),
    rollDeg: Number(rollDeg.toFixed(1)),
    reasons,
  };
}

