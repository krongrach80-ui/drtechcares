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

export interface PrimaryFaceSelection {
  primaryFace: readonly Landmark3D[] | null;
  effectiveFaceCount: number;
  primaryCenter?: { x: number; y: number };
  primaryBox?: { width: number; height: number; area: number };
}

/**
 * Normalizes a head-pose angle to degrees.
 * Values within (-1.5, 1.5) excluding near-zero noise are treated as radians and converted.
 */
export function normalizeAngleToDegrees(angleOrRad: number): number {
  if (!Number.isFinite(angleOrRad)) return 0;
  const abs = Math.abs(angleOrRad);
  if (abs > 0.02 && abs <= 1.5) {
    return angleOrRad * (180 / Math.PI);
  }
  return angleOrRad;
}

/**
 * Selects the dominant primary face near the viewfinder center while ignoring
 * small background shadows/noise and duplicate overlapping detections.
 */
export function selectPrimaryFace(
  faces: readonly (readonly Landmark3D[])[],
): PrimaryFaceSelection {
  interface Candidate {
    face: readonly Landmark3D[];
    cx: number;
    cy: number;
    width: number;
    height: number;
    area: number;
    score: number;
  }

  const rawCandidates: Candidate[] = [];
  for (const landmarks of faces) {
    if (!landmarks || landmarks.length < 300) continue;
    let minX = 1;
    let maxX = 0;
    let minY = 1;
    let maxY = 0;
    for (const pt of landmarks) {
      if (pt.x < minX) minX = pt.x;
      if (pt.x > maxX) maxX = pt.x;
      if (pt.y < minY) minY = pt.y;
      if (pt.y > maxY) maxY = pt.y;
    }
    const width = maxX - minX;
    const height = maxY - minY;
    const area = width * height;
    // Allow ultra-long-range faces (>= ~2.2% frame height, up to ~5m)
    if (width < 0.018 || height < 0.022 || area < 0.0004) continue;

    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const score = area * 2.5 - Math.hypot(cx - 0.5, cy - 0.5) * 0.4;
    rawCandidates.push({ face: landmarks, cx, cy, width, height, area, score });
  }

  if (rawCandidates.length === 0) {
    return { primaryFace: null, effectiveFaceCount: 0 };
  }

  // Deduplicate overlapping detections on the same face, keeping the larger box.
  rawCandidates.sort((a, b) => b.area - a.area);
  const distinct: Candidate[] = [];
  for (const cand of rawCandidates) {
    const overlapsExisting = distinct.some(
      (kept) => Math.hypot(cand.cx - kept.cx, cand.cy - kept.cy) < 0.18,
    );
    if (!overlapsExisting) {
      distinct.push(cand);
    }
  }

  // Rank remaining distinct faces by combined area & centrality.
  distinct.sort((a, b) => b.score - a.score);
  const primary = distinct[0];

  const secondCloseLargeFaces = distinct.slice(1).filter(
    (other) =>
      other.area >= primary.area * 0.65 &&
      Math.hypot(other.cx - primary.cx, other.cy - primary.cy) >= 0.22 &&
      Math.hypot(other.cx - 0.5, other.cy - 0.5) <= 0.35,
  );

  return {
    primaryFace: primary.face,
    effectiveFaceCount: secondCloseLargeFaces.length > 0 ? 1 + secondCloseLargeFaces.length : 1,
    primaryCenter: { x: primary.cx, y: primary.cy },
    primaryBox: { width: primary.width, height: primary.height, area: primary.area },
  };
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

  // 3D Head pose in degrees combining 2D nose displacement + 3D depth differential
  const midX = (leftCheek.x + rightCheek.x) / 2;
  const yawOffsetDeg = ((noseTip.x - midX) / faceWidth) * 90;
  const yawDepthDeg =
    (Math.atan2((rightCheek.z ?? 0) - (leftCheek.z ?? 0), faceWidth) * 180) / Math.PI;
  const yawDeg = yawOffsetDeg * 0.6 + yawDepthDeg * 0.4;

  const neutralNoseY = forehead.y + faceHeight * 0.55;
  const pitchOffsetDeg = ((noseTip.y - neutralNoseY) / faceHeight) * 90;
  const pitchDepthDeg =
    (Math.atan2((chin.z ?? 0) - (forehead.z ?? 0), faceHeight) * 180) / Math.PI;
  const pitchDeg = pitchOffsetDeg * 0.55 + pitchDepthDeg * 0.45;

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

/**
 * Estimates patient distance in meters (0.4m - 4.8m) from normalized vertical face height ratio.
 */
export function estimateCameraDistanceMeters(faceHeightRatio: number): number {
  if (!Number.isFinite(faceHeightRatio) || faceHeightRatio <= 0) return 2.5;
  const raw = 0.46 / Math.max(0.055, faceHeightRatio);
  return Number(Math.max(0.4, Math.min(4.8, raw)).toFixed(1));
}

