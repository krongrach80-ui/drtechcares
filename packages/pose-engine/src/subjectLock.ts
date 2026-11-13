import type { PoseLandmark } from './types';

export interface FaceAnchorBox {
  centerX: number;
  centerY: number;
  width: number;
}

/**
 * Locks onto the primary patient closest to the verified face anchor or frame center,
 * preventing passers-by in the background from hijacking pose evaluation.
 */
export function selectSubjectPoseWithIdentity(
  detectedPoses: readonly (readonly PoseLandmark[])[],
  faceAnchor?: FaceAnchorBox | null,
): readonly PoseLandmark[] | null {
  if (detectedPoses.length === 0) return null;
  if (detectedPoses.length === 1) return detectedPoses[0];

  const targetX = faceAnchor?.centerX ?? 0.5;
  const targetY = faceAnchor?.centerY ?? 0.35;

  let bestPose = detectedPoses[0];
  let bestScore = -Infinity;

  for (const pose of detectedPoses) {
    const nose = pose[0];
    const lShoulder = pose[11];
    const rShoulder = pose[12];
    if (!nose || !lShoulder || !rShoulder) continue;

    const shoulderWidth = Math.hypot(rShoulder.x - lShoulder.x, rShoulder.y - lShoulder.y);
    const distToAnchor = Math.hypot(nose.x - targetX, nose.y - targetY);
    const score = shoulderWidth * 2.5 - distToAnchor;
    if (score > bestScore) {
      bestScore = score;
      bestPose = pose;
    }
  }

  return bestPose;
}

