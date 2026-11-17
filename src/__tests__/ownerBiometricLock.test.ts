import { describe, expect, it } from 'vitest';
import {
  CANONICAL_HUMAN_FACE_BASELINE_512,
  FACE_EMBEDDING_DIM,
  l2Normalize,
} from '@drtechcares/smart-biometric-core';
import {
  computeObjectCoverLayout,
  computeOwnerFaceBoxMetrics,
  evaluateOwnerBoundKinematicHands,
  projectMirroredNormToCoverViewport,
  projectRawNormToCoverCanvas,
  verifyOwnerFace1to1,
  type OwnerBiometricTemplate,
} from '../lib/ownerBiometricLock';

function mkPersonVec(seed: number): number[] {
  const out = new Array<number>(FACE_EMBEDDING_DIM);
  let s = seed >>> 0;
  for (let i = 0; i < FACE_EMBEDDING_DIM; i += 1) {
    s = (s * 1664525 + 1013904223 + i * 31) >>> 0;
    const wave = ((s % 2000) / 1000 - 1) * 0.45;
    out[i] = CANONICAL_HUMAN_FACE_BASELINE_512[i] * 0.48 + wave;
  }
  return l2Normalize(out);
}

function mkHand(anchorX: number, anchorY: number) {
  return Array.from({ length: 21 }, (_, idx) => ({
    x: anchorX + (idx % 3) * 0.005,
    y: anchorY + Math.floor(idx / 3) * 0.004,
  }));
}

describe('Continuous 1:1 Owner Lock, Full-Bleed Cover Math & Owner-Bound Hand Filter', () => {
  const ownerVec = mkPersonVec(123456);
  const impostorVec = mkPersonVec(987654);

  const ownerTemplate: OwnerBiometricTemplate = {
    patientId: 'pat-owner-1',
    hn: 'HN-44524',
    fullName: 'Phattadon Phongarrom',
    masterEmbedding: ownerVec,
    angleEmbeddings: {
      center: ownerVec,
      left: ownerVec,
      right: ownerVec,
      up: ownerVec,
      down: ownerVec,
    },
  };

  it('computes exact full-bleed object-cover scaling and mirrored projections without black bars', () => {
    // 4:3 webcam (640x480) rendered inside wider 16:9 container (640x360)
    const layout = computeObjectCoverLayout(640, 480, 640, 360);
    expect(layout.scale).toBeCloseTo(1, 5);
    expect(layout.renderW).toBe(640);
    expect(layout.renderH).toBe(480);
    expect(layout.offsetX).toBe(0);
    expect(layout.offsetY).toBe(-60);

    const centerPt = projectRawNormToCoverCanvas(0.5, 0.5, layout);
    expect(centerPt.x).toBeCloseTo(320, 4);
    expect(centerPt.y).toBeCloseTo(180, 4);

    // Mirrored X maps raw left (0.25) -> screen right (0.75)
    const mirrorPt = projectMirroredNormToCoverViewport(0.25, 0.5, layout);
    expect(mirrorPt.x).toBeCloseTo(0.75, 4);
    expect(mirrorPt.y).toBeCloseTo(0.5, 4);
  });

  it('verifies 1:1 account owner cleanly and blocks impersonator with similarity < ownerThreshold', () => {
    const ownerRes = verifyOwnerFace1to1({
      probeVector: ownerVec,
      yawDeg: 0,
      pitchDeg: 0,
      ownerTemplate,
    });
    expect(ownerRes.hasTemplate).toBe(true);
    expect(ownerRes.matched).toBe(true);
    expect(ownerRes.similarity).toBeGreaterThanOrEqual(0.95);

    const impostorRes = verifyOwnerFace1to1({
      probeVector: impostorVec,
      yawDeg: 0,
      pitchDeg: 0,
      ownerTemplate,
    });
    expect(impostorRes.hasTemplate).toBe(true);
    expect(impostorRes.matched).toBe(false);
    expect(impostorRes.similarity).toBeLessThan(0.76);
  });

  it('allows owner-bound raised hands inside anatomical envelope and blocks impostors, multi-person, missing face, and intruder hands', () => {
    // Face landmarks centered at (0.5, 0.45), width ~0.12 -> with pad ~0.149 width, ~0.18 height
    const ownerFaceLandmarks = [
      { x: 0.44, y: 0.38 },
      { x: 0.56, y: 0.38 },
      { x: 0.50, y: 0.45 },
      { x: 0.45, y: 0.52 },
      { x: 0.55, y: 0.52 },
      { x: 0.44, y: 0.44 },
      { x: 0.56, y: 0.44 },
      { x: 0.47, y: 0.41 },
      { x: 0.53, y: 0.41 },
      { x: 0.50, y: 0.37 },
    ];
    const ownerBox = computeOwnerFaceBoxMetrics(ownerFaceLandmarks, 640, 360, 640, 360)!;
    expect(ownerBox).not.toBeNull();
    expect(ownerBox.centerX).toBeCloseTo(0.5, 2);

    // Raw X=0.70 -> Mirrored viewport X=0.30 (viewer LEFT, within centerX - 2.8*W..centerX - 0.15*W)
    // Raw X=0.30 -> Mirrored viewport X=0.70 (viewer RIGHT, within centerX + 0.15*W..centerX + 2.8*W)
    const ownLeftRaw = mkHand(0.70, 0.48);
    const ownRightRaw = mkHand(0.30, 0.48);

    // 1. Owner alone lifting both own hands inside anatomical envelope -> PASSES!
    const valid = evaluateOwnerBoundKinematicHands({
      ownerFaceVerified: true,
      ownerFaceBox: ownerBox,
      faceCount: 1,
      ownerName: ownerTemplate.fullName,
      hands: [ownLeftRaw, ownRightRaw],
      videoW: 640,
      videoH: 360,
      canvasW: 640,
      canvasH: 360,
    });
    expect(valid.status).toBe('OK_BOTH_HANDS');
    expect(valid.bothUp).toBe(true);
    expect(valid.leftUp).toBe(true);
    expect(valid.rightUp).toBe(true);
    expect(valid.boundLeftPalm).toBeDefined();
    expect(valid.boundRightPalm).toBeDefined();

    // 2. Non-owner sitting & raising BOTH hands -> IMPOSTOR_FACE block!
    const impostorSitting = evaluateOwnerBoundKinematicHands({
      ownerFaceVerified: false,
      ownerFaceBox: ownerBox,
      faceCount: 1,
      ownerName: ownerTemplate.fullName,
      hands: [ownLeftRaw, ownRightRaw],
      videoW: 640,
      videoH: 360,
      canvasW: 640,
      canvasH: 360,
    });
    expect(impostorSitting.status).toBe('IMPOSTOR_FACE');
    expect(impostorSitting.bothUp).toBe(false);
    expect(impostorSitting.alertBannerTh).toContain('คุณไม่ใช่เจ้าของบัญชี');

    // 3. No face visible -> NO_OWNER_FACE block!
    const missingFace = evaluateOwnerBoundKinematicHands({
      ownerFaceVerified: false,
      ownerFaceBox: null,
      faceCount: 0,
      ownerName: ownerTemplate.fullName,
      hands: [ownLeftRaw, ownRightRaw],
      videoW: 640,
      videoH: 360,
      canvasW: 640,
      canvasH: 360,
    });
    expect(missingFace.status).toBe('NO_OWNER_FACE');
    expect(missingFace.bothUp).toBe(false);
    expect(missingFace.alertBannerTh).toContain('ไม่พบใบหน้าเจ้าของบัญชี');

    // 4. Two faces in frame -> MULTI_PERSON block!
    const twoFaces = evaluateOwnerBoundKinematicHands({
      ownerFaceVerified: true,
      ownerFaceBox: ownerBox,
      faceCount: 2,
      ownerName: ownerTemplate.fullName,
      hands: [ownLeftRaw, ownRightRaw],
      videoW: 640,
      videoH: 360,
      canvasW: 640,
      canvasH: 360,
    });
    expect(twoFaces.status).toBe('MULTI_PERSON');
    expect(twoFaces.bothUp).toBe(false);
    expect(twoFaces.alertBannerTh).toContain('พบผู้อื่นอยู่ในเฟรม');

    // 5a. External hand reaching from far frame edge (> 2.8 * faceW away) -> INTRUDER_HAND!
    const farEdgeHand = mkHand(0.96, 0.48); // mirroredX = 0.04, diff = 0.46 >> 2.8 * 0.148 = 0.414
    const intruderSide = evaluateOwnerBoundKinematicHands({
      ownerFaceVerified: true,
      ownerFaceBox: ownerBox,
      faceCount: 1,
      ownerName: ownerTemplate.fullName,
      hands: [farEdgeHand, ownRightRaw],
      videoW: 640,
      videoH: 360,
      canvasW: 640,
      canvasH: 360,
    });
    expect(intruderSide.status).toBe('INTRUDER_HAND');
    expect(intruderSide.intruderHandDetected).toBe(true);
    expect(intruderSide.bothUp).toBe(false);
    expect(intruderSide.alertBannerTh).toContain('ตรวจพบมือของบุคคลอื่น');

    // 5b. Owner raises both hands AND third helper hand enters frame (totalHands > 2) -> INTRUDER_HAND!
    const thirdHand = mkHand(0.65, 0.45);
    const threeHands = evaluateOwnerBoundKinematicHands({
      ownerFaceVerified: true,
      ownerFaceBox: ownerBox,
      faceCount: 1,
      ownerName: ownerTemplate.fullName,
      hands: [ownLeftRaw, ownRightRaw, thirdHand],
      videoW: 640,
      videoH: 360,
      canvasW: 640,
      canvasH: 360,
    });
    expect(threeHands.status).toBe('INTRUDER_HAND');
    expect(threeHands.intruderHandDetected).toBe(true);
    expect(threeHands.bothUp).toBe(false);
  });

  it('verifies 1:1 Legacy V1 account owners via legacyV1ProbeVector while rejecting V1 impostors', () => {
    const mkLegacy = (seed: number): number[] => {
      const out = new Array<number>(FACE_EMBEDDING_DIM);
      let s = seed >>> 0;
      for (let i = 0; i < FACE_EMBEDDING_DIM; i += 1) {
        s = (s * 1664525 + 1013904223 + i * 29) >>> 0;
        const wave = ((s % 2000) / 1000 - 1) * (i >= 256 ? 0.01 : 0.35);
        out[i] = CANONICAL_HUMAN_FACE_BASELINE_512[i] * (i >= 256 ? 0.65 : 0.5) + wave;
      }
      return l2Normalize(out);
    };
    const v1Owner = mkLegacy(111222);
    const v1Impostor = mkLegacy(999888);
    const dummyV2Probe = mkPersonVec(444555);
    const legacyTemplate: OwnerBiometricTemplate = {
      patientId: 'pat-legacy-owner',
      hn: 'HN-44524',
      fullName: 'Phattadon Phongarrom',
      masterEmbedding: v1Owner,
      angleEmbeddings: {
        center: v1Owner,
        left: v1Owner,
        right: v1Owner,
        up: v1Owner,
        down: v1Owner,
      },
    };

    const okOwner = verifyOwnerFace1to1({
      probeVector: dummyV2Probe,
      legacyV1ProbeVector: v1Owner,
      yawDeg: 0,
      pitchDeg: 0,
      ownerTemplate: legacyTemplate,
    });
    expect(okOwner.matched).toBe(true);

    const blockedImpostor = verifyOwnerFace1to1({
      probeVector: dummyV2Probe,
      legacyV1ProbeVector: v1Impostor,
      yawDeg: 0,
      pitchDeg: 0,
      ownerTemplate: legacyTemplate,
    });
    expect(blockedImpostor.matched).toBe(false);
  });

  it('verifies 1:1 ArcFace ResNet-50 owners strictly with engine isolation', () => {
    const mkArc = (seed: number): number[] => {
      const out = new Array<number>(FACE_EMBEDDING_DIM);
      let s = seed >>> 0;
      for (let i = 0; i < FACE_EMBEDDING_DIM; i += 1) {
        s = (s * 1664525 + 1013904223 + i * 41) >>> 0;
        out[i] = ((s % 2000) / 1000 - 1) * (i >= 256 ? 1.4 : 1.0);
      }
      return l2Normalize(out);
    };
    const arcOwner = mkArc(7001);
    const arcImpostor = mkArc(7002);
    const arcTemplate: OwnerBiometricTemplate = {
      patientId: 'pat-arc-owner',
      hn: 'HN-ARC-LOCK',
      fullName: 'คุณวิชัย อาร์คเฟซ',
      biometricEngine: 'arcface-onnx',
      masterEmbedding: arcOwner,
      angleEmbeddings: {
        center: arcOwner,
        left: arcOwner,
        right: arcOwner,
        up: arcOwner,
        down: arcOwner,
      },
    };

    expect(
      verifyOwnerFace1to1({
        probeVector: l2Normalize(arcOwner.map((v, i) => v * 0.72 + mkArc(5)[i] * 0.35)),
        probeEngine: 'arcface-onnx',
        ownerTemplate: arcTemplate,
      }).matched,
    ).toBe(true);

    expect(
      verifyOwnerFace1to1({
        probeVector: arcImpostor,
        probeEngine: 'arcface-onnx',
        ownerTemplate: arcTemplate,
      }).matched,
    ).toBe(false);

    // Hybrid probe cannot spoof an ArcFace owner lock
    expect(
      verifyOwnerFace1to1({
        probeVector: arcOwner,
        probeEngine: 'hybrid-512d',
        ownerTemplate: arcTemplate,
      }).matched,
    ).toBe(false);
  });
});

