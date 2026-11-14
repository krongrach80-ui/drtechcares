import { describe, expect, it } from 'vitest';
import { POSE_INDEX, type PoseLandmark } from '@drtechcares/pose-engine';
import {
  detectRaiseBothHands,
  detectTwoRaisedHandLandmarks,
  extractRaisedHandSides,
  FacePresenceTracker,
  HandGestureVerifier,
  HeadNodDetector,
  VoiceFeedbackController,
  type DetectedHandFrame,
  type HandLandmarkPoint,
} from '../index';

function buildOpenRaisedHand(handedness: 'Left' | 'Right', centerX = 0.5): DetectedHandFrame {
  const landmarks: HandLandmarkPoint[] = Array.from({ length: 21 }, () => ({ x: centerX, y: 0.6 }));
  landmarks[0] = { x: centerX, y: 0.65 }; // wrist
  landmarks[5] = { x: centerX - 0.03, y: 0.55 }; // index mcp
  landmarks[8] = { x: centerX - 0.04, y: 0.36 }; // index tip
  landmarks[9] = { x: centerX, y: 0.54 }; // middle mcp
  landmarks[12] = { x: centerX, y: 0.33 }; // middle tip
  landmarks[13] = { x: centerX + 0.03, y: 0.55 }; // ring mcp
  landmarks[16] = { x: centerX + 0.04, y: 0.36 }; // ring tip
  landmarks[17] = { x: centerX + 0.05, y: 0.56 }; // pinky mcp
  landmarks[20] = { x: centerX + 0.07, y: 0.40 }; // pinky tip
  return { landmarks, handedness, score: 0.95 };
}

describe('Touchless Kiosk Input & Cognitive Hand Verification', () => {
  it('maps anatomical right hand to YES and left hand to NO after 600ms hold', () => {
    const verifier = new HandGestureVerifier();
    const rightHand = buildOpenRaisedHand('Right', 0.65);

    const t0 = verifier.update([rightHand], 1000);
    expect(t0.choice).toBe('yes');
    expect(t0.confirmedAnswer).toBeNull();

    const t600 = verifier.update([rightHand], 1650);
    expect(t600.confirmedAnswer).toBe('yes');

    verifier.reset();
    const leftHand = buildOpenRaisedHand('Left', 0.35);
    verifier.update([leftHand], 2000);
    const leftConfirm = verifier.update([leftHand], 2650);
    expect(leftConfirm.confirmedAnswer).toBe('no');
  });

  it('blocks answer and triggers warning when both hands are raised simultaneously', () => {
    const verifier = new HandGestureVerifier();
    const both = verifier.update(
      [buildOpenRaisedHand('Left', 0.3), buildOpenRaisedHand('Right', 0.7)],
      1000,
    );
    expect(both.bothHandsRaisedWarning).toBe(true);
    expect(both.confirmedAnswer).toBeNull();
  });

  it('detects portrait crop boundary when hand is too close to edge', () => {
    const verifier = new HandGestureVerifier();
    const edgeHand = buildOpenRaisedHand('Right', 0.05);
    const res = verifier.update([edgeHand], 1000, { isPortrait: true, cropRatio: 0.36 });
    expect(res.outsidePortraitCropWarning).toBe(true);
  });

  it('handles presence wake-up, head nod confirmation, and TTS cooldown (>=3500ms)', () => {
    const presence = new FacePresenceTracker();
    expect(presence.update({ facePresent: true, faceSizeRatio: 0.2, timestampMs: 0 }).awake).toBe(false);
    expect(presence.update({ facePresent: true, faceSizeRatio: 0.2, timestampMs: 1100 }).awake).toBe(true);

    const nod = new HeadNodDetector();
    expect(nod.update(15, 100).confirmed).toBe(false);
    expect(nod.update(15, 450).confirmed).toBe(false);
    expect(nod.update(2, 600).confirmed).toBe(true);

    const tts = new VoiceFeedbackController(3500);
    expect(tts.speak('สวัสดีค่ะ', 'th', false, 1000)).toBe(true);
    expect(tts.speak('ยกแขนขึ้น', 'th', false, 2000)).toBe(false);
    expect(tts.speak('ยกแขนขึ้น', 'th', false, 5000)).toBe(true);
  });

  it('detects two raised hands for seated kiosk users without visible hips and via HandLandmarker', () => {
    const leftHand = buildOpenRaisedHand('Left', 0.3).landmarks;
    const rightHand = buildOpenRaisedHand('Right', 0.7).landmarks;
    expect(detectTwoRaisedHandLandmarks([leftHand, rightHand])).toBe(true);
    expect(detectTwoRaisedHandLandmarks([leftHand])).toBe(false);
    expect(detectTwoRaisedHandLandmarks([leftHand, leftHand])).toBe(false);

    const poseLandmarks: PoseLandmark[] = Array.from({ length: 33 }, () => ({ x: 0, y: 0, z: 0, visibility: 0 }));
    poseLandmarks[POSE_INDEX.LEFT_SHOULDER] = { x: 0.35, y: 0.55, z: 0, visibility: 0.9 };
    poseLandmarks[POSE_INDEX.RIGHT_SHOULDER] = { x: 0.65, y: 0.55, z: 0, visibility: 0.9 };
    poseLandmarks[POSE_INDEX.LEFT_WRIST] = { x: 0.3, y: 0.58, z: 0, visibility: 0.85 };
    poseLandmarks[POSE_INDEX.RIGHT_WRIST] = { x: 0.7, y: 0.57, z: 0, visibility: 0.85 };
    // Hips remain out-of-frame (visibility 0)
    const seatedRes = detectRaiseBothHands(poseLandmarks);
    expect(seatedRes.bothRaised).toBe(true);
  });

  it('supports elbow-level high raises and hybrid HandLandmarker + PoseLandmarker bilateral detection', () => {
    const elbowOnlyPose: PoseLandmark[] = Array.from({ length: 33 }, () => ({ x: 0, y: 0, z: 0, visibility: 0 }));
    elbowOnlyPose[POSE_INDEX.LEFT_SHOULDER] = { x: 0.65, y: 0.65, z: 0, visibility: 0.9 };
    elbowOnlyPose[POSE_INDEX.RIGHT_SHOULDER] = { x: 0.35, y: 0.65, z: 0, visibility: 0.9 };
    elbowOnlyPose[POSE_INDEX.LEFT_ELBOW] = { x: 0.72, y: 0.62, z: 0, visibility: 0.85 };
    elbowOnlyPose[POSE_INDEX.RIGHT_ELBOW] = { x: 0.28, y: 0.60, z: 0, visibility: 0.85 };
    expect(extractRaisedHandSides([], elbowOnlyPose).bothRaised).toBe(true);

    // Hybrid: Pose sees anatomical right raised, HandLandmarker sees one hand on anatomical left shoulder side
    const leftSideHand = buildOpenRaisedHand('Left', 0.68).landmarks;
    const rightOnlyPose: PoseLandmark[] = Array.from({ length: 33 }, () => ({ x: 0, y: 0, z: 0, visibility: 0 }));
    rightOnlyPose[POSE_INDEX.LEFT_SHOULDER] = { x: 0.65, y: 0.65, z: 0, visibility: 0.9 };
    rightOnlyPose[POSE_INDEX.RIGHT_SHOULDER] = { x: 0.35, y: 0.65, z: 0, visibility: 0.9 };
    rightOnlyPose[POSE_INDEX.RIGHT_WRIST] = { x: 0.3, y: 0.45, z: 0, visibility: 0.9 };
    const hybrid = extractRaisedHandSides([leftSideHand], rightOnlyPose);
    expect(hybrid.leftRaised).toBe(true);
    expect(hybrid.rightRaised).toBe(true);
    expect(hybrid.bothRaised).toBe(true);

    // Resting pose below shoulders must NOT false-trigger
    const restingPose: PoseLandmark[] = Array.from({ length: 33 }, () => ({ x: 0, y: 0, z: 0, visibility: 0 }));
    restingPose[POSE_INDEX.LEFT_SHOULDER] = { x: 0.65, y: 0.35, z: 0, visibility: 0.95 };
    restingPose[POSE_INDEX.RIGHT_SHOULDER] = { x: 0.35, y: 0.35, z: 0, visibility: 0.95 };
    restingPose[POSE_INDEX.LEFT_ELBOW] = { x: 0.67, y: 0.56, z: 0, visibility: 0.95 };
    restingPose[POSE_INDEX.RIGHT_ELBOW] = { x: 0.33, y: 0.56, z: 0, visibility: 0.95 };
    restingPose[POSE_INDEX.LEFT_WRIST] = { x: 0.69, y: 0.72, z: 0, visibility: 0.95 };
    restingPose[POSE_INDEX.RIGHT_WRIST] = { x: 0.31, y: 0.72, z: 0, visibility: 0.95 };
    expect(extractRaisedHandSides([], restingPose).bothRaised).toBe(false);
  });
});

