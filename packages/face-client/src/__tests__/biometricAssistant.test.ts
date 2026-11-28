import { describe, expect, it } from 'vitest';
import { BiologicalLivenessTracker } from '../biologicalLiveness';
import { estimateCameraDistanceMeters, selectPrimaryFace } from '../liveness';
import { analyzeScreenAndBezelArtifacts } from '../screenAntiSpoof';
import {
  FACE_ID_SCREEN_PROMPTS,
  classifyDetected3DPose,
  inspectBiometricFrame3D,
  resolveFaceIdScreenPrompt,
} from '../biometricAssistant';
import { extractGenuineNeuralFaceDescriptor } from '../embed';

describe('3D Biometric Verification Assistant & Face ID Prompts', () => {
  it('evaluates checkpoints in strict order: framing -> lighting -> obstruction -> liveness', () => {
    // 1. TOO_FAR (< 30% frame height) short-circuits before all later checks
    const tooFar = inspectBiometricFrame3D({
      faceCount: 1,
      metrics: {
        faceHeightRatio: 0.22,
        brightness: 10,
        sharpness: 1,
        eyesVisible: false,
        moireOrBezelDetected: true,
      },
    });
    expect(tooFar.scan_status).toBe('RETRY');
    expect(tooFar.error_code).toBe('TOO_FAR');
    expect(tooFar.checks).toEqual({
      framing_ok: false,
      lighting_ok: false,
      no_obstruction: false,
      liveness_ok: false,
    });

    // 2. TOO_CLOSE (cropped edge)
    const tooClose = inspectBiometricFrame3D({
      faceCount: 1,
      metrics: { faceHeightRatio: 0.88, croppedEdge: true },
    });
    expect(tooClose.scan_status).toBe('RETRY');
    expect(tooClose.error_code).toBe('TOO_CLOSE');
    expect(tooClose.checks.framing_ok).toBe(false);

    // 3. LOW_LIGHT
    const lowLight = inspectBiometricFrame3D({
      faceCount: 1,
      metrics: { faceHeightRatio: 0.5, brightness: 20, sharpness: 1, eyesVisible: false },
    });
    expect(lowLight.scan_status).toBe('RETRY');
    expect(lowLight.error_code).toBe('LOW_LIGHT');
    expect(lowLight.checks).toEqual({
      framing_ok: true,
      lighting_ok: false,
      no_obstruction: false,
      liveness_ok: false,
    });

    // 4. BLURRY
    const blurry = inspectBiometricFrame3D({
      faceCount: 1,
      metrics: { faceHeightRatio: 0.5, brightness: 120, sharpness: 2.1 },
    });
    expect(blurry.scan_status).toBe('RETRY');
    expect(blurry.error_code).toBe('BLURRY');
    expect(blurry.checks.framing_ok).toBe(true);
    expect(blurry.checks.lighting_ok).toBe(false);

    // 5. OBSTRUCTED_EYES
    const eyesBlocked = inspectBiometricFrame3D({
      faceCount: 1,
      metrics: {
        faceHeightRatio: 0.5,
        brightness: 120,
        sharpness: 15,
        eyesVisible: false,
        mouthVisible: false,
      },
    });
    expect(eyesBlocked.scan_status).toBe('RETRY');
    expect(eyesBlocked.error_code).toBe('OBSTRUCTED_EYES');
    expect(eyesBlocked.checks).toEqual({
      framing_ok: true,
      lighting_ok: true,
      no_obstruction: false,
      liveness_ok: false,
    });

    // 6. OBSTRUCTED_MOUTH
    const mouthBlocked = inspectBiometricFrame3D({
      faceCount: 1,
      metrics: {
        faceHeightRatio: 0.5,
        brightness: 120,
        sharpness: 15,
        eyesVisible: true,
        mouthVisible: false,
      },
    });
    expect(mouthBlocked.scan_status).toBe('RETRY');
    expect(mouthBlocked.error_code).toBe('OBSTRUCTED_MOUTH');

    // 7. SPOOF_ATTEMPT (returns SPOOF_DETECTED immediately)
    const spoof = inspectBiometricFrame3D({
      faceCount: 1,
      metrics: {
        faceHeightRatio: 0.5,
        brightness: 120,
        sharpness: 15,
        eyesVisible: true,
        mouthVisible: true,
        moireOrBezelDetected: true,
        depthScore: 0.12,
      },
    });
    expect(spoof.scan_status).toBe('SPOOF_DETECTED');
    expect(spoof.error_code).toBe('SPOOF_ATTEMPT');
    expect(spoof.checks).toEqual({
      framing_ok: true,
      lighting_ok: true,
      no_obstruction: true,
      liveness_ok: false,
    });

    // 8. PASS when all 4 checkpoints succeed
    const pass = inspectBiometricFrame3D({
      faceCount: 1,
      requestedPose: 'CENTER',
      metrics: {
        faceHeightRatio: 0.52,
        brightness: 125,
        sharpness: 19,
        eyesVisible: true,
        mouthVisible: true,
        isLive: true,
        depthScore: 0.55,
        yawDeg: 1,
        pitchDeg: 0,
      },
    });
    expect(pass.scan_status).toBe('PASS');
    expect(pass.error_code).toBe('NONE');
    expect(pass.detected_pose).toBe('CENTER');
    expect(pass.confidence_score).toBeGreaterThanOrEqual(0.75);
    expect(pass.checks).toEqual({
      framing_ok: true,
      lighting_ok: true,
      no_obstruction: true,
      liveness_ok: true,
    });
  });

  it('classifies all 5 3D poses accurately', () => {
    expect(classifyDetected3DPose(0, 0)).toBe('CENTER');
    expect(classifyDetected3DPose(-25, 0)).toBe('TURN_LEFT');
    expect(classifyDetected3DPose(24, 2)).toBe('TURN_RIGHT');
    expect(classifyDetected3DPose(0, -18)).toBe('TILT_UP');
    expect(classifyDetected3DPose(0, 17)).toBe('TILT_DOWN');
    expect(classifyDetected3DPose(65, 45)).toBe('UNKNOWN');
  });

  it('resolves all 7 iPhone Face ID UI/UX screen prompts correctly', () => {
    expect(FACE_ID_SCREEN_PROMPTS).toHaveLength(7);

    expect(resolveFaceIdScreenPrompt(null, 'initial').primaryPrompt).toBe('วางใบหน้าของคุณให้อยู่ในกรอบ');
    expect(resolveFaceIdScreenPrompt(null, 'round1').primaryPrompt).toBe('ค่อยๆ ขยับศีรษะเป็นวงกลม');
    expect(resolveFaceIdScreenPrompt(null, 'round2').primaryPrompt).toBe('ขยับศีรษะเป็นวงกลมอีกครั้ง');
    expect(resolveFaceIdScreenPrompt(null, 'completed').primaryPrompt).toBe(
      'ตั้งค่าการสแกนใบหน้าเรียบร้อยแล้ว',
    );

    const obstructJson = inspectBiometricFrame3D({
      faceCount: 1,
      metrics: { faceHeightRatio: 0.5, brightness: 120, sharpness: 14, eyesVisible: false },
    });
    expect(resolveFaceIdScreenPrompt(obstructJson, 'round1').primaryPrompt).toBe('มองไม่เห็นใบหน้าชัดเจน');

    const farJson = inspectBiometricFrame3D({
      faceCount: 1,
      metrics: { faceHeightRatio: 0.2 },
    });
    expect(resolveFaceIdScreenPrompt(farJson, 'initial').primaryPrompt).toBe(
      'ขยับเข้ามาใกล้ขึ้น / ถอยห่างออกไปอีกนิด',
    );

    const darkJson = inspectBiometricFrame3D({
      faceCount: 1,
      metrics: { faceHeightRatio: 0.5, brightness: 15 },
    });
    expect(resolveFaceIdScreenPrompt(darkJson, 'initial').primaryPrompt).toBe('สภาพแสงไม่เพียงพอ');
  });

  it('detects smartphone bezel / moire screen spoofs and accepts natural human skin without false alarm', () => {
    // 1. Clean natural warm human skin 112x112 + soft room 160x120 context
    const cleanCrop = new Uint8ClampedArray(112 * 112 * 4);
    for (let y = 0; y < 112; y += 1) {
      for (let x = 0; x < 112; x += 1) {
        const p = (y * 112 + x) * 4;
        const grad = Math.round(x * 0.2 + y * 0.1);
        cleanCrop[p] = 175 + grad;
        cleanCrop[p + 1] = 140 + grad;
        cleanCrop[p + 2] = 120 + grad;
        cleanCrop[p + 3] = 255;
      }
    }
    const cleanContext = new Uint8ClampedArray(160 * 120 * 4);
    for (let i = 0; i < 160 * 120; i += 1) {
      const p = i * 4;
      cleanContext[p] = 135;
      cleanContext[p + 1] = 128;
      cleanContext[p + 2] = 122;
      cleanContext[p + 3] = 255;
    }
    const naturalRes = analyzeScreenAndBezelArtifacts({
      faceCrop: { width: 112, height: 112, data: cleanCrop },
      contextFrame: { width: 160, height: 120, data: cleanContext },
      faceBox: { x: 0.3, y: 0.25, width: 0.4, height: 0.5 },
    });
    expect(naturalRes.spoofDetected).toBe(false);

    // 2. High-frequency moire lattice + cool blue LCD screen emission
    const moireCrop = new Uint8ClampedArray(112 * 112 * 4);
    for (let y = 0; y < 112; y += 1) {
      for (let x = 0; x < 112; x += 1) {
        const p = (y * 112 + x) * 4;
        const ripple = ((x + y) % 2 === 0 ? 18 : -18);
        moireCrop[p] = Math.max(0, Math.min(255, 130 + ripple));
        moireCrop[p + 1] = Math.max(0, Math.min(255, 155 - ripple));
        moireCrop[p + 2] = Math.max(0, Math.min(255, 175 + ripple));
        moireCrop[p + 3] = 255;
      }
    }
    const moireRes = analyzeScreenAndBezelArtifacts({
      faceCrop: { width: 112, height: 112, data: moireCrop },
    });
    expect(moireRes.spoofDetected).toBe(true);
    expect(moireRes.moireScore).toBeGreaterThan(0.5);

    // 3. Rectangular smartphone bezel + glowing bright screen around face
    const bezelCtx = new Uint8ClampedArray(160 * 120 * 4);
    for (let y = 0; y < 120; y += 1) {
      for (let x = 0; x < 160; x += 1) {
        const p = (y * 160 + x) * 4;
        const inPhone = x >= 44 && x <= 116 && y >= 22 && y <= 98;
        const onBezel =
          (x >= 44 && x <= 46 && y >= 22 && y <= 98) ||
          (x >= 114 && x <= 116 && y >= 22 && y <= 98) ||
          (y >= 22 && y <= 24 && x >= 44 && x <= 116) ||
          (y >= 96 && y <= 98 && x >= 44 && x <= 116);
        const val = onBezel ? 15 : inPhone ? 215 : 85;
        bezelCtx[p] = val;
        bezelCtx[p + 1] = val;
        bezelCtx[p + 2] = inPhone && !onBezel ? 235 : val;
        bezelCtx[p + 3] = 255;
      }
    }
    const bezelRes = analyzeScreenAndBezelArtifacts({
      faceCrop: { width: 112, height: 112, data: cleanCrop },
      contextFrame: { width: 160, height: 120, data: bezelCtx },
      faceBox: { x: 0.3, y: 0.25, width: 0.4, height: 0.5 },
    });
    expect(bezelRes.spoofDetected).toBe(true);
    expect(bezelRes.bezelScore).toBeGreaterThan(0.6);

    const frameInspection = inspectBiometricFrame3D({
      faceCount: 1,
      image: { width: 112, height: 112, data: cleanCrop },
      contextImage: { width: 160, height: 120, data: bezelCtx },
      faceBox: { x: 0.3, y: 0.25, width: 0.4, height: 0.5 },
      metrics: {
        faceHeightRatio: 0.5,
        brightness: 145,
        sharpness: 18,
        eyesVisible: true,
        mouthVisible: true,
        depthScore: 0.52,
      },
    });
    expect(frameInspection.scan_status).toBe('SPOOF_DETECTED');
    expect(frameInspection.error_code).toBe('SPOOF_ATTEMPT');
  });

  it('requires real eye blink and rejects rigid 2D planar photo motion in BiologicalLivenessTracker', () => {
    const makeFace = (shiftX = 0, shiftY = 0, earGap = 0.03) => {
      const lms = Array.from({ length: 478 }, () => ({ x: 0.5 + shiftX, y: 0.5 + shiftY, z: 0 }));
      lms[33] = { x: 0.42 + shiftX, y: 0.45 + shiftY, z: 0 };
      lms[263] = { x: 0.58 + shiftX, y: 0.45 + shiftY, z: 0 };
      lms[159] = { x: 0.44 + shiftX, y: 0.45 - earGap / 2 + shiftY, z: 0 };
      lms[145] = { x: 0.44 + shiftX, y: 0.45 + earGap / 2 + shiftY, z: 0 };
      lms[386] = { x: 0.56 + shiftX, y: 0.45 - earGap / 2 + shiftY, z: 0 };
      lms[374] = { x: 0.56 + shiftX, y: 0.45 + earGap / 2 + shiftY, z: 0 };
      lms[105] = { x: 0.44 + shiftX, y: 0.40 + shiftY, z: 0 };
      lms[334] = { x: 0.56 + shiftX, y: 0.40 + shiftY, z: 0 };
      lms[13] = { x: 0.50 + shiftX, y: 0.62 + shiftY, z: 0 };
      lms[14] = { x: 0.50 + shiftX, y: 0.64 + shiftY, z: 0 };
      lms[61] = { x: 0.46 + shiftX, y: 0.63 + shiftY, z: 0 };
      lms[291] = { x: 0.54 + shiftX, y: 0.63 + shiftY, z: 0 };
      lms[1] = { x: 0.50 + shiftX, y: 0.54 + shiftY, z: -0.05 };
      return lms;
    };

    // First 4.5s window avoids false-alarming stationary human before blink/head-turn
    const spoofTracker = new BiologicalLivenessTracker();
    let spoofOut = spoofTracker.update({
      landmarks3D: makeFace(0, 0),
      earLeft: 0.31,
      earRight: 0.31,
      yawDeg: 0,
      pitchDeg: 0,
      timestampMs: 1000,
    });
    for (let i = 1; i <= 15; i += 1) {
      spoofOut = spoofTracker.update({
        landmarks3D: makeFace(i * 0.002, 0),
        earLeft: 0.31,
        earRight: 0.31,
        yawDeg: 0.05,
        pitchDeg: 0.05,
        timestampMs: 1000 + i * 80,
      });
    }
    expect(spoofOut.biologicalVerified).toBe(false);
    expect(spoofOut.spoofDetected).toBe(false);
    expect(spoofOut.spoofScore).toBe(0);

    // Rigid 2D planar face (yaw/pitch variance < 0.15° and zero blink) sustained > 4.5s triggers PLANAR_RIGID_MOTION
    for (let i = 16; i <= 60; i += 1) {
      spoofOut = spoofTracker.update({
        landmarks3D: makeFace(i * 0.002, 0),
        earLeft: 0.31,
        earRight: 0.31,
        yawDeg: (i % 2) * 0.05,
        pitchDeg: (i % 2) * 0.05,
        timestampMs: 1000 + i * 100,
      });
    }
    expect(spoofOut.biologicalVerified).toBe(false);
    expect(spoofOut.planarSpoofDetected).toBe(true);
    expect(spoofOut.spoofDetected).toBe(true);
    expect(spoofOut.spoofScore).toBeGreaterThan(0);
    expect(spoofOut.spoofReason).toBe('PLANAR_RIGID_MOTION');

    // Immediate State Clear: turning head > 2.5° immediately clears spoofScore=0 & sets livenessPassed=true
    const clearedByHeadTurn = spoofTracker.update({
      landmarks3D: makeFace(0.12, 0),
      earLeft: 0.31,
      earRight: 0.31,
      yawDeg: 3.4,
      pitchDeg: 0.4,
      timestampMs: 7100,
    });
    expect(clearedByHeadTurn.headPoseVerified).toBe(true);
    expect(clearedByHeadTurn.biologicalVerified).toBe(true);
    expect(clearedByHeadTurn.livenessPassed).toBe(true);
    expect(clearedByHeadTurn.spoofDetected).toBe(false);
    expect(clearedByHeadTurn.spoofScore).toBe(0);

    // Distance-adaptive blink detection at far range (faceScale=0.22): relative drop <= openEyeBaseline * 0.83 triggers blink
    const distantTracker = new BiologicalLivenessTracker();
    for (let i = 0; i < 4; i += 1) {
      distantTracker.update({
        landmarks3D: makeFace(0, 0, 0.025),
        earLeft: 0.27,
        earRight: 0.27,
        faceBox: { x: 0.4, y: 0.35, width: 0.16, height: 0.22 },
        timestampMs: 3000 + i * 80,
      });
    }
    // Subtle distant EAR dip to 0.21 (< 0.27 * 0.83 = 0.224) then reopen to 0.27
    distantTracker.update({
      landmarks3D: makeFace(0, 0, 0.012),
      earLeft: 0.21,
      earRight: 0.21,
      faceBox: { x: 0.4, y: 0.35, width: 0.16, height: 0.22 },
      timestampMs: 3400,
    });
    const distantReopen = distantTracker.update({
      landmarks3D: makeFace(0, 0, 0.025),
      earLeft: 0.27,
      earRight: 0.27,
      faceBox: { x: 0.4, y: 0.35, width: 0.16, height: 0.22 },
      timestampMs: 3520,
    });
    expect(distantReopen.faceScale).toBeLessThan(0.35);
    expect(distantReopen.blinkCount).toBe(1);
    expect(distantReopen.blinkVerified).toBe(true);
    expect(distantReopen.biologicalVerified).toBe(true);
    expect(distantReopen.spoofDetected).toBe(false);

    // Genuine live person performing one natural blendshape blink passes (including elderly unilateral blink)
    const liveTracker = new BiologicalLivenessTracker();
    liveTracker.update({
      landmarks3D: makeFace(0, 0, 0.03),
      blendshapes: [
        { categoryName: 'eyeBlinkLeft', score: 0.05 },
        { categoryName: 'eyeBlinkRight', score: 0.05 },
      ],
      earLeft: 0.31,
      earRight: 0.31,
      timestampMs: 2000,
    });
    liveTracker.update({
      landmarks3D: makeFace(0, 0, 0.004),
      blendshapes: [
        { categoryName: 'eyeBlinkLeft', score: 0.72 },
        { categoryName: 'eyeBlinkRight', score: 0.12 },
      ],
      earLeft: 0.14,
      earRight: 0.28,
      timestampMs: 2120,
    });
    const liveAfterBlink = liveTracker.update({
      landmarks3D: makeFace(0, 0, 0.03),
      blendshapes: [
        { categoryName: 'eyeBlinkLeft', score: 0.08 },
        { categoryName: 'eyeBlinkRight', score: 0.06 },
      ],
      earLeft: 0.31,
      earRight: 0.31,
      timestampMs: 2240,
    });
    expect(liveAfterBlink.blinkCount).toBe(1);
    expect(liveAfterBlink.blinkVerified).toBe(true);
    expect(liveAfterBlink.biologicalVerified).toBe(true);
    expect(liveAfterBlink.spoofDetected).toBe(false);
  });

  it('supports Ultra-Long-Range detection (28px+ / 4% height at 1.5-4.5m) and quick swaying liveness without requiring blink', () => {
    // 1. selectPrimaryFace accepts distant small face (~0.032 width x ~0.048 height at ~4m)
    const farLandmarks = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.4, z: 0 }));
    farLandmarks[10] = { x: 0.5, y: 0.376, z: 0 };
    farLandmarks[152] = { x: 0.5, y: 0.424, z: 0 };
    farLandmarks[234] = { x: 0.484, y: 0.4, z: 0 };
    farLandmarks[454] = { x: 0.516, y: 0.4, z: 0 };
    farLandmarks[33] = { x: 0.49, y: 0.395, z: 0 };
    farLandmarks[263] = { x: 0.51, y: 0.395, z: 0 };
    farLandmarks[1] = { x: 0.5, y: 0.405, z: -0.015 };
    farLandmarks[61] = { x: 0.493, y: 0.415, z: 0 };
    farLandmarks[291] = { x: 0.507, y: 0.415, z: 0 };
    farLandmarks[13] = { x: 0.5, y: 0.414, z: 0 };
    farLandmarks[14] = { x: 0.5, y: 0.416, z: 0 };
    farLandmarks[159] = { x: 0.49, y: 0.394, z: 0 };
    farLandmarks[145] = { x: 0.49, y: 0.396, z: 0 };
    farLandmarks[386] = { x: 0.51, y: 0.394, z: 0 };
    farLandmarks[374] = { x: 0.51, y: 0.396, z: 0 };
    farLandmarks[105] = { x: 0.49, y: 0.388, z: 0 };
    farLandmarks[334] = { x: 0.51, y: 0.388, z: 0 };

    const picked = selectPrimaryFace([farLandmarks]);
    expect(picked.effectiveFaceCount).toBe(1);
    expect(picked.primaryFace).not.toBeNull();
    expect(estimateCameraDistanceMeters(0.11)).toBeGreaterThanOrEqual(4.0);

    // 2. inspectBiometricFrame3D accepts long-range framing when allowLongRange: true
    const farInspect = inspectBiometricFrame3D({
      faceCount: 1,
      allowLongRange: true,
      metrics: {
        faceHeightRatio: 0.06,
        brightness: 120,
        sharpness: 6,
        eyesVisible: true,
        mouthVisible: true,
        depthScore: 0.5,
      },
    });
    expect(farInspect.scan_status).toBe('PASS');

    // 3. Long-Range Liveness passes within 0.6s on subtle >1.5px body sway with ZERO eye blink required
    const lrTracker = new BiologicalLivenessTracker();
    const shiftLm = (dx: number) => farLandmarks.map((pt) => ({ ...pt, x: pt.x + dx }));
    lrTracker.update({
      landmarks3D: shiftLm(0),
      earLeft: 0.25,
      earRight: 0.25,
      frameWidth: 1920,
      frameHeight: 1080,
      faceBox: { x: 0.48, y: 0.37, width: 0.04, height: 0.08 },
      timestampMs: 5000,
    });
    const lrPass = lrTracker.update({
      landmarks3D: shiftLm(0.002), // ~3.8px shift on 1920px HD frame
      earLeft: 0.25,
      earRight: 0.25,
      frameWidth: 1920,
      frameHeight: 1080,
      faceBox: { x: 0.482, y: 0.37, width: 0.04, height: 0.08 },
      timestampMs: 5250,
    });
    expect(lrPass.livenessMode).toBe('LONG_RANGE');
    expect(lrPass.centerShiftPx).toBeGreaterThan(1.5);
    expect(lrPass.longRangeMotionVerified).toBe(true);
    expect(lrPass.biologicalVerified).toBe(true);
    expect(lrPass.livenessPassed).toBe(true);
    expect(lrPass.spoofDetected).toBe(false);
  });

  it('supports Full-Range 0.25m-5.0m scanning (close-up border clip, 2.5% far face, and left/right off-center)', () => {
    // 1. Extreme Close-Up (~0.28m): forehead/chin clipped near top/bottom, core 5pt visible inside frame
    const closeLandmarks = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
    closeLandmarks[10] = { x: 0.5, y: 0.002, z: 0 };
    closeLandmarks[152] = { x: 0.5, y: 0.998, z: 0 };
    closeLandmarks[33] = { x: 0.32, y: 0.38, z: 0 };
    closeLandmarks[263] = { x: 0.68, y: 0.38, z: 0 };
    closeLandmarks[1] = { x: 0.5, y: 0.55, z: -0.05 };
    closeLandmarks[61] = { x: 0.38, y: 0.72, z: 0 };
    closeLandmarks[291] = { x: 0.62, y: 0.72, z: 0 };

    const closePass = inspectBiometricFrame3D({
      landmarks3D: closeLandmarks,
      faceCount: 1,
      allowLongRange: true,
      metrics: {
        faceHeightRatio: 0.91,
        faceCenterX: 0.5,
        faceCenterY: 0.5,
        brightness: 120,
        sharpness: 65,
        eyesVisible: true,
        mouthVisible: true,
        depthScore: 0.65,
      },
    });
    expect(closePass.scan_status).toBe('PASS');

    // 2. Off-Center Full-Width Kiosk Login (e.g. seated left at x=0.14)
    const offCenterPass = inspectBiometricFrame3D({
      faceCount: 1,
      allowLongRange: true,
      metrics: {
        faceHeightRatio: 0.24,
        faceCenterX: 0.14,
        faceCenterY: 0.45,
        brightness: 125,
        sharpness: 35,
        eyesVisible: true,
        mouthVisible: true,
        depthScore: 0.55,
      },
    });
    expect(offCenterPass.scan_status).toBe('PASS');

    // 3. Far distance (~4.5-5.0m, 2.5% frame height)
    const ultraFarPass = inspectBiometricFrame3D({
      faceCount: 1,
      allowLongRange: true,
      metrics: {
        faceHeightRatio: 0.025,
        faceCenterX: 0.5,
        faceCenterY: 0.42,
        brightness: 120,
        sharpness: 6,
        eyesVisible: true,
        mouthVisible: true,
        depthScore: 0.5,
      },
    });
    expect(ultraFarPass.scan_status).toBe('PASS');

    // 4. Default strict 120-frame registration bounds still reject 0.22 far faces
    const regFarReject = inspectBiometricFrame3D({
      faceCount: 1,
      metrics: { faceHeightRatio: 0.22, brightness: 120, sharpness: 50, eyesVisible: true, mouthVisible: true },
    });
    expect(regFarReject.error_code).toBe('TOO_FAR');
  });

  it('extractGenuineNeuralFaceDescriptor rejects sub-128D or zero vectors and L2-normalizes valid deep descriptors', () => {
    expect(extractGenuineNeuralFaceDescriptor(undefined)).toBeNull();
    expect(extractGenuineNeuralFaceDescriptor([])).toBeNull();
    expect(extractGenuineNeuralFaceDescriptor(new Array(64).fill(0.5))).toBeNull();
    expect(extractGenuineNeuralFaceDescriptor(new Array(512).fill(0))).toBeNull();
    expect(extractGenuineNeuralFaceDescriptor([Number.NaN, ...new Array(127).fill(0.2)])).toBeNull();

    const raw1024 = Array.from({ length: 1024 }, (_, idx) => Math.sin(idx * 0.13) + 0.5);
    const norm = extractGenuineNeuralFaceDescriptor(raw1024);
    expect(norm).not.toBeNull();
    expect(norm!).toHaveLength(1024);
    const mag = Math.sqrt(norm!.reduce((acc, v) => acc + v * v, 0));
    expect(mag).toBeCloseTo(1, 5);
  });
});

