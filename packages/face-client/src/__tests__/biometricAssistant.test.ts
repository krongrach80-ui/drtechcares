import { describe, expect, it } from 'vitest';
import {
  FACE_ID_SCREEN_PROMPTS,
  classifyDetected3DPose,
  inspectBiometricFrame3D,
  resolveFaceIdScreenPrompt,
} from '../biometricAssistant';

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
});

