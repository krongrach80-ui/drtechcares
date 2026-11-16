import {
  MAX_BRIGHTNESS,
  MAX_BURNOUT_RATIO,
  MIN_BRIGHTNESS,
  MIN_SHARPNESS_LAPLACIAN,
  evaluateFrameQuality,
  type SimpleImageBuffer,
} from '@drtechcares/smart-biometric-core';
import type { Landmark3D } from './embed';
import { MIN_DEPTH_SCORE, evaluateLivenessAndPose } from './liveness';
import { analyzeScreenAndBezelArtifacts, type NormalizedFaceBox } from './screenAntiSpoof';

export type BiometricScanStatus = 'PASS' | 'RETRY' | 'SPOOF_DETECTED';

export type Detected3DPose =
  | 'CENTER'
  | 'TURN_LEFT'
  | 'TURN_RIGHT'
  | 'TILT_UP'
  | 'TILT_DOWN'
  | 'UNKNOWN';

export type BiometricErrorCode =
  | 'NONE'
  | 'TOO_CLOSE'
  | 'TOO_FAR'
  | 'LOW_LIGHT'
  | 'BLURRY'
  | 'OBSTRUCTED_EYES'
  | 'OBSTRUCTED_MOUTH'
  | 'SPOOF_ATTEMPT';

export interface BiometricInspectionChecks {
  framing_ok: boolean;
  lighting_ok: boolean;
  no_obstruction: boolean;
  liveness_ok: boolean;
}

export interface BiometricVerificationJson {
  scan_status: BiometricScanStatus;
  confidence_score: number;
  detected_pose: Detected3DPose;
  checks: BiometricInspectionChecks;
  error_code: BiometricErrorCode;
  user_instruction_th: string;
}

export type FaceIdScanStage = 'initial' | 'round1' | 'round2' | 'completed';

export interface FaceIdScreenPromptSpec {
  step: number;
  key:
    | 'INITIAL'
    | 'ROUND_1_3D'
    | 'ROUND_2_DEPTH'
    | 'OBSTRUCTION'
    | 'DISTANCE'
    | 'LIGHTING'
    | 'COMPLETED';
  titleTh: string;
  primaryPrompt: string;
  subPrompt: string;
}

export const FACE_ID_SCREEN_PROMPTS: readonly FaceIdScreenPromptSpec[] = [
  {
    step: 1,
    key: 'INITIAL',
    titleTh: '1. เริ่มต้นสแกน',
    primaryPrompt: 'วางใบหน้าของคุณให้อยู่ในกรอบ',
    subPrompt: 'มองตรงไปที่กล้องและถืออุปกรณ์ให้อยู่ในระดับสายตา',
  },
  {
    step: 2,
    key: 'ROUND_1_3D',
    titleTh: '2. สแกนรอบที่ 1 (เก็บมิติ 3D)',
    primaryPrompt: 'ค่อยๆ ขยับศีรษะเป็นวงกลม',
    subPrompt: 'หมุนใบหน้าช้าๆ เพื่อเก็บรายละเอียดให้ครบทุกมุม',
  },
  {
    step: 3,
    key: 'ROUND_2_DEPTH',
    titleTh: '3. สแกนรอบที่ 2 (ยืนยันความลึก)',
    primaryPrompt: 'ขยับศีรษะเป็นวงกลมอีกครั้ง',
    subPrompt: 'ทำซ้ำช้าๆ เพื่อยืนยันโครงสร้างใบหน้า',
  },
  {
    step: 4,
    key: 'OBSTRUCTION',
    titleTh: '4. แจ้งเตือนเมื่อมีสิ่งกีดขวาง',
    primaryPrompt: 'มองไม่เห็นใบหน้าชัดเจน',
    subPrompt: 'กรุณาถอดแว่นกันแดด หน้ากาก หรือสิ่งบดบังใบหน้า',
  },
  {
    step: 5,
    key: 'DISTANCE',
    titleTh: '5. แจ้งเตือนระยะห่าง',
    primaryPrompt: 'ขยับเข้ามาใกล้ขึ้น / ถอยห่างออกไปอีกนิด',
    subPrompt: 'ปรับระยะห่างให้ใบหน้าพอดีกับเส้นกรอบวงกลม',
  },
  {
    step: 6,
    key: 'LIGHTING',
    titleTh: '6. แจ้งเตือนแสงสว่าง',
    primaryPrompt: 'สภาพแสงไม่เพียงพอ',
    subPrompt: 'ย้ายไปยังบริเวณที่มีแสงสว่างสม่ำเสมอ',
  },
  {
    step: 7,
    key: 'COMPLETED',
    titleTh: '7. เสร็จสิ้น',
    primaryPrompt: 'ตั้งค่าการสแกนใบหน้าเรียบร้อยแล้ว',
    subPrompt: 'พร้อมใช้งานสำหรับการปลดล็อกและยืนยันตัวตน',
  },
];

export interface BiometricTelemetryOverrides {
  faceCount?: number;
  faceHeightRatio?: number;
  faceCenterX?: number;
  faceCenterY?: number;
  croppedEdge?: boolean;
  brightness?: number;
  sharpness?: number;
  burnoutRatio?: number;
  lightingAsymmetry?: number;
  eyesVisible?: boolean;
  mouthVisible?: boolean;
  eyeGlareRatio?: number;
  sunglassesDetected?: boolean;
  moireOrBezelDetected?: boolean;
  planarRigidSpoof?: boolean;
  biologicalVerified?: boolean;
  allowLongRange?: boolean;
  isLive?: boolean;
  depthScore?: number;
  microMotion?: number;
  yawDeg?: number;
  pitchDeg?: number;
}

export interface BiometricAssistantInput {
  landmarks3D?: readonly Landmark3D[];
  prevLandmarks3D?: readonly Landmark3D[] | null;
  faceCount?: number;
  wasEyeClosed?: boolean;
  image?: SimpleImageBuffer | null;
  contextImage?: SimpleImageBuffer | null;
  faceBox?: NormalizedFaceBox | null;
  biologicalVerified?: boolean;
  allowLongRange?: boolean;
  requestedPose?: Detected3DPose;
  metrics?: BiometricTelemetryOverrides;
}

const POSE_INSTRUCTION_TH: Record<Detected3DPose, string> = {
  CENTER: 'มองตรงไปที่กล้อง',
  TURN_LEFT: 'ค่อยๆ หันหน้าไปทางซ้าย',
  TURN_RIGHT: 'ค่อยๆ หันหน้าไปทางขวา',
  TILT_UP: 'เงยหน้าขึ้นเล็กน้อย',
  TILT_DOWN: 'ก้มหน้าลงเล็กน้อย',
  UNKNOWN: 'ขยับใบหน้าให้อยู่ในกรอบ',
};

export function classifyDetected3DPose(yawDeg: number, pitchDeg: number): Detected3DPose {
  if (Math.abs(yawDeg) <= 14 && Math.abs(pitchDeg) <= 12) {
    return 'CENTER';
  }
  if (yawDeg <= -15 && yawDeg >= -42 && Math.abs(pitchDeg) <= 20) {
    return 'TURN_LEFT';
  }
  if (yawDeg >= 15 && yawDeg <= 42 && Math.abs(pitchDeg) <= 20) {
    return 'TURN_RIGHT';
  }
  if (pitchDeg <= -12 && pitchDeg >= -35 && Math.abs(yawDeg) <= 20) {
    return 'TILT_UP';
  }
  if (pitchDeg >= 12 && pitchDeg <= 35 && Math.abs(yawDeg) <= 20) {
    return 'TILT_DOWN';
  }
  return 'UNKNOWN';
}

function analyzeCropAppearance(image: SimpleImageBuffer | null | undefined): {
  lightingAsymmetry: number;
  eyeGlareRatio: number;
  sunglassesDetected: boolean;
} {
  if (!image || image.width < 16 || image.height < 16 || image.data.length < image.width * image.height * 4) {
    return { lightingAsymmetry: 0, eyeGlareRatio: 0, sunglassesDetected: false };
  }

  const { width, height, data } = image;
  let leftSum = 0;
  let leftCount = 0;
  let rightSum = 0;
  let rightCount = 0;
  let eyeLumaSum = 0;
  let eyeCount = 0;
  let eyeGlareCount = 0;
  let cheekLumaSum = 0;
  let cheekCount = 0;

  const eyeYMin = Math.floor(height * 0.22);
  const eyeYMax = Math.floor(height * 0.44);
  const cheekYMin = Math.floor(height * 0.52);
  const cheekYMax = Math.floor(height * 0.68);

  for (let y = 0; y < height; y += 2) {
    for (let x = 0; x < width; x += 2) {
      const p = (y * width + x) * 4;
      const luma = 0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2];

      if (x < width / 2) {
        leftSum += luma;
        leftCount += 1;
      } else {
        rightSum += luma;
        rightCount += 1;
      }

      if (y >= eyeYMin && y <= eyeYMax && x >= width * 0.18 && x <= width * 0.82) {
        eyeLumaSum += luma;
        eyeCount += 1;
        if (luma >= 244) eyeGlareCount += 1;
      }

      if (y >= cheekYMin && y <= cheekYMax && x >= width * 0.25 && x <= width * 0.75) {
        cheekLumaSum += luma;
        cheekCount += 1;
      }
    }
  }

  const leftMean = leftCount > 0 ? leftSum / leftCount : 120;
  const rightMean = rightCount > 0 ? rightSum / rightCount : 120;
  const lightingAsymmetry = Math.abs(leftMean - rightMean) / Math.max(40, (leftMean + rightMean) / 2);
  const eyeGlareRatio = eyeCount > 0 ? eyeGlareCount / eyeCount : 0;
  const eyeMean = eyeCount > 0 ? eyeLumaSum / eyeCount : 100;
  const cheekMean = cheekCount > 0 ? cheekLumaSum / cheekCount : 100;
  const sunglassesDetected = eyeMean < 20 && cheekMean > 68;

  return {
    lightingAsymmetry: Number(lightingAsymmetry.toFixed(3)),
    eyeGlareRatio: Number(eyeGlareRatio.toFixed(3)),
    sunglassesDetected,
  };
}

function isLandmarkWithinFrame(pt: Landmark3D | undefined): boolean {
  return Boolean(pt && Number.isFinite(pt.x) && Number.isFinite(pt.y) && pt.x >= 0.02 && pt.x <= 0.98 && pt.y >= 0.02 && pt.y <= 0.98);
}

/**
 * Evaluates a camera frame strictly across the 4 biometric checkpoints in order:
 * 1. Framing & Alignment (TOO_CLOSE / TOO_FAR)
 * 2. Lighting & Clarity (LOW_LIGHT / BLURRY)
 * 3. Obstructions & Accessories (OBSTRUCTED_EYES / OBSTRUCTED_MOUTH)
 * 4. Liveness & Spoof Detection (SPOOF_ATTEMPT -> scan_status: SPOOF_DETECTED)
 */
export function inspectBiometricFrame3D(input: BiometricAssistantInput): BiometricVerificationJson {
  const m = input.metrics ?? {};
  const landmarks = input.landmarks3D ?? [];
  const faceCount = m.faceCount ?? input.faceCount ?? (landmarks.length >= 300 ? 1 : 0);

  const liveness =
    landmarks.length >= 300
      ? evaluateLivenessAndPose(landmarks, input.prevLandmarks3D ?? null, faceCount, input.wasEyeClosed ?? false)
      : null;

  const yawDeg = m.yawDeg ?? liveness?.yawDeg ?? 0;
  const pitchDeg = m.pitchDeg ?? liveness?.pitchDeg ?? 0;
  const detectedPose: Detected3DPose =
    faceCount === 1 ? classifyDetected3DPose(yawDeg, pitchDeg) : 'UNKNOWN';

  // Compute framing geometry from forehead(10), chin(152), leftCheek(234), rightCheek(454)
  let faceHeightRatio = m.faceHeightRatio;
  let faceCenterX = m.faceCenterX;
  let faceCenterY = m.faceCenterY;
  let croppedEdge = m.croppedEdge;

  if (landmarks.length >= 455) {
    const forehead = landmarks[10];
    const chin = landmarks[152];
    const leftCheek = landmarks[234];
    const rightCheek = landmarks[454];
    faceHeightRatio ??= Math.abs(chin.y - forehead.y);
    faceCenterX ??= (leftCheek.x + rightCheek.x) / 2;
    faceCenterY ??= (forehead.y + chin.y) / 2;
    croppedEdge ??=
      !isLandmarkWithinFrame(forehead) ||
      !isLandmarkWithinFrame(chin) ||
      !isLandmarkWithinFrame(leftCheek) ||
      !isLandmarkWithinFrame(rightCheek);
  }

  faceHeightRatio ??= faceCount === 1 ? 0.5 : 0;
  faceCenterX ??= 0.5;
  faceCenterY ??= 0.5;
  croppedEdge ??= false;

  // Checkpoint 1: Framing & Alignment
  const minFaceHeightThreshold = (input.allowLongRange ?? m.allowLongRange) ? 0.035 : 0.3;
  if (faceCount === 0 || faceHeightRatio < minFaceHeightThreshold) {
    return {
      scan_status: 'RETRY',
      confidence_score: 0.15,
      detected_pose: 'UNKNOWN',
      checks: {
        framing_ok: false,
        lighting_ok: false,
        no_obstruction: false,
        liveness_ok: false,
      },
      error_code: 'TOO_FAR',
      user_instruction_th: faceCount === 0 ? 'ขยับใบหน้าให้อยู่ในกรอบ' : 'ขยับเข้ามาใกล้กล้องอีกนิด',
    };
  }

  if (faceCount > 1 || croppedEdge || faceHeightRatio > 0.82) {
    return {
      scan_status: 'RETRY',
      confidence_score: 0.25,
      detected_pose: detectedPose,
      checks: {
        framing_ok: false,
        lighting_ok: false,
        no_obstruction: false,
        liveness_ok: false,
      },
      error_code: 'TOO_CLOSE',
      user_instruction_th: faceCount > 1 ? 'กรุณาสแกนครั้งละ 1 ท่าน' : 'ถอยห่างจากกล้องเล็กน้อย',
    };
  }

  if (Math.abs(faceCenterX - 0.5) > 0.24 || Math.abs(faceCenterY - 0.5) > 0.24) {
    return {
      scan_status: 'RETRY',
      confidence_score: 0.35,
      detected_pose: detectedPose,
      checks: {
        framing_ok: false,
        lighting_ok: false,
        no_obstruction: false,
        liveness_ok: false,
      },
      error_code: 'TOO_FAR',
      user_instruction_th: 'ขยับใบหน้าให้อยู่ตรงกลางกรอบ',
    };
  }

  // Checkpoint 2: Lighting & Clarity
  const imgReport = input.image
    ? evaluateFrameQuality(input.image, { yawDeg, pitchDeg, rollDeg: liveness?.rollDeg ?? 0 }, 45, 45)
    : null;
  const cropApp = analyzeCropAppearance(input.image);

  const brightness = m.brightness ?? imgReport?.brightness ?? 118;
  const sharpness = m.sharpness ?? imgReport?.sharpness ?? 18;
  const burnoutRatio = m.burnoutRatio ?? imgReport?.burnoutRatio ?? 0;
  const lightingAsymmetry = m.lightingAsymmetry ?? cropApp.lightingAsymmetry;

  if (
    brightness < MIN_BRIGHTNESS ||
    brightness > MAX_BRIGHTNESS ||
    burnoutRatio > MAX_BURNOUT_RATIO ||
    lightingAsymmetry > 0.52
  ) {
    return {
      scan_status: 'RETRY',
      confidence_score: 0.45,
      detected_pose: detectedPose,
      checks: {
        framing_ok: true,
        lighting_ok: false,
        no_obstruction: false,
        liveness_ok: false,
      },
      error_code: 'LOW_LIGHT',
      user_instruction_th:
        brightness > MAX_BRIGHTNESS || burnoutRatio > MAX_BURNOUT_RATIO
          ? 'ลดแสงย้อนด้านหลังแล้วลองใหม่'
          : 'ขยับไปบริเวณที่มีแสงสว่างสม่ำเสมอ',
    };
  }

  const faceScale = input.faceBox?.height ?? faceHeightRatio;
  const effectiveMinSharpness = faceScale < 0.35 ? 1.0 : MIN_SHARPNESS_LAPLACIAN;

  if (sharpness < effectiveMinSharpness) {
    return {
      scan_status: 'RETRY',
      confidence_score: 0.48,
      detected_pose: detectedPose,
      checks: {
        framing_ok: true,
        lighting_ok: false,
        no_obstruction: false,
        liveness_ok: false,
      },
      error_code: 'BLURRY',
      user_instruction_th: 'กรุณาหยุดนิ่งสักครู่ ภาพไม่คมชัด',
    };
  }

  // Checkpoint 3: Obstructions & Accessories
  const eyeGlareRatio = m.eyeGlareRatio ?? cropApp.eyeGlareRatio;
  const sunglassesDetected = m.sunglassesDetected ?? cropApp.sunglassesDetected;
  const landmarkEyesOk =
    landmarks.length >= 400
      ? isLandmarkWithinFrame(landmarks[33]) &&
        isLandmarkWithinFrame(landmarks[133]) &&
        isLandmarkWithinFrame(landmarks[362]) &&
        isLandmarkWithinFrame(landmarks[263])
      : true;
  const eyesVisible = m.eyesVisible ?? (landmarkEyesOk && !sunglassesDetected && eyeGlareRatio <= 0.42);

  if (!eyesVisible) {
    return {
      scan_status: 'RETRY',
      confidence_score: 0.52,
      detected_pose: detectedPose,
      checks: {
        framing_ok: true,
        lighting_ok: true,
        no_obstruction: false,
        liveness_ok: false,
      },
      error_code: 'OBSTRUCTED_EYES',
      user_instruction_th: 'กรุณาถอดแว่นหรือเปิดหน้าผากให้เห็นตา',
    };
  }

  const landmarkMouthOk =
    landmarks.length >= 300
      ? isLandmarkWithinFrame(landmarks[13]) &&
        isLandmarkWithinFrame(landmarks[14]) &&
        isLandmarkWithinFrame(landmarks[61]) &&
        isLandmarkWithinFrame(landmarks[291])
      : true;
  const mouthVisible = m.mouthVisible ?? landmarkMouthOk;

  if (!mouthVisible) {
    return {
      scan_status: 'RETRY',
      confidence_score: 0.55,
      detected_pose: detectedPose,
      checks: {
        framing_ok: true,
        lighting_ok: true,
        no_obstruction: false,
        liveness_ok: false,
      },
      error_code: 'OBSTRUCTED_MOUTH',
      user_instruction_th: 'กรุณาถอดหน้ากากอนามัยก่อนสแกน',
    };
  }

  // Checkpoint 4: Liveness & Spoof Detection
  const inferredFaceBox: NormalizedFaceBox | null =
    input.faceBox ??
    (landmarks.length >= 455
      ? {
          x: Math.max(0, Math.min(landmarks[234].x, landmarks[454].x) - 0.04),
          y: Math.max(0, landmarks[10].y - 0.04),
          width: Math.max(0.08, Math.abs(landmarks[454].x - landmarks[234].x) + 0.08),
          height: Math.max(0.1, Math.abs(landmarks[152].y - landmarks[10].y) + 0.08),
        }
      : null);

  const biologicalPassed = Boolean(input.biologicalVerified ?? m.biologicalVerified);
  const screenArtifacts =
    !biologicalPassed && (input.image || input.contextImage)
      ? analyzeScreenAndBezelArtifacts({
          faceCrop: input.image,
          contextFrame: input.contextImage,
          faceBox: inferredFaceBox,
          faceScale,
        })
      : null;

  const depthScore = m.depthScore ?? liveness?.depthScore ?? 0.52;
  const isLive = biologicalPassed ? true : (m.isLive ?? depthScore >= MIN_DEPTH_SCORE);
  const moireOrBezelDetected = biologicalPassed
    ? false
    : (m.moireOrBezelDetected ?? screenArtifacts?.spoofDetected ?? false);
  const planarRigidSpoof = biologicalPassed ? false : (m.planarRigidSpoof ?? false);

  if (!biologicalPassed && (moireOrBezelDetected || planarRigidSpoof || !isLive || depthScore < MIN_DEPTH_SCORE)) {
    return {
      scan_status: 'SPOOF_DETECTED',
      confidence_score: Number(Math.max(0.05, Math.min(0.39, depthScore)).toFixed(2)),
      detected_pose: detectedPose,
      checks: {
        framing_ok: true,
        lighting_ok: true,
        no_obstruction: true,
        liveness_ok: false,
      },
      error_code: 'SPOOF_ATTEMPT',
      user_instruction_th: 'ตรวจพบภาพถ่ายหรือหน้าจอมือถือ กรุณาใช้ใบหน้าจริง',
    };
  }

  // All 4 checkpoints passed! Check requested 3D pose if specified
  const qualityFactor = Math.min(1, sharpness / 24) * 0.2 + Math.min(1, depthScore / 0.65) * 0.25 + 0.54;
  const confidenceScore = Number(Math.max(0.75, Math.min(0.99, qualityFactor)).toFixed(2));

  if (input.requestedPose && detectedPose !== input.requestedPose) {
    return {
      scan_status: 'RETRY',
      confidence_score: Number((confidenceScore * 0.85).toFixed(2)),
      detected_pose: detectedPose,
      checks: {
        framing_ok: true,
        lighting_ok: true,
        no_obstruction: true,
        liveness_ok: true,
      },
      error_code: 'NONE',
      user_instruction_th: POSE_INSTRUCTION_TH[input.requestedPose],
    };
  }

  return {
    scan_status: 'PASS',
    confidence_score: confidenceScore,
    detected_pose: detectedPose,
    checks: {
      framing_ok: true,
      lighting_ok: true,
      no_obstruction: true,
      liveness_ok: true,
    },
    error_code: 'NONE',
    user_instruction_th: 'ผ่านการตรวจสอบ วางตำแหน่งดีมาก',
  };
}

/**
 * Maps live biometric inspection JSON + scan round to the exact 7-row
 * Face ID UI/UX Screen Prompts specification.
 */
export function resolveFaceIdScreenPrompt(
  inspection: BiometricVerificationJson | null,
  stage: FaceIdScanStage,
): FaceIdScreenPromptSpec {
  if (stage === 'completed') {
    return FACE_ID_SCREEN_PROMPTS[6];
  }
  if (inspection && inspection.scan_status !== 'PASS') {
    switch (inspection.error_code) {
      case 'OBSTRUCTED_EYES':
      case 'OBSTRUCTED_MOUTH':
      case 'SPOOF_ATTEMPT':
        return FACE_ID_SCREEN_PROMPTS[3];
      case 'TOO_CLOSE':
      case 'TOO_FAR':
        return FACE_ID_SCREEN_PROMPTS[4];
      case 'LOW_LIGHT':
      case 'BLURRY':
        return FACE_ID_SCREEN_PROMPTS[5];
      default:
        break;
    }
  }
  if (stage === 'round1') return FACE_ID_SCREEN_PROMPTS[1];
  if (stage === 'round2') return FACE_ID_SCREEN_PROMPTS[2];
  return FACE_ID_SCREEN_PROMPTS[0];
}

