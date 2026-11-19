import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Camera,
  Check,
  CheckCircle2,
  Loader2,
  Lock,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Volume2,
  VolumeX,
} from 'lucide-react';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import {
  alignFaceTo112,
  computeHeadZoneRoiFromFaceBox,
  evaluateLivenessAndPose,
  extractGeometricAnthropometricRatios,
  remapRoiLandmarksToFrame,
  renderUpscaledRoi256,
  selectClinicalCameraDeviceId,
  selectPrimaryFace,
} from '@drtechcares/face-client';
import {
  computeCraniofacialRatios12,
  evaluateFrameQuality,
  type GeometricAnthropometricRatios,
} from '@drtechcares/smart-biometric-core';
import type { AngleEmbeddings5D } from '@drtechcares/api-client';
import {
  useClinicalI18n,
  type ClinicalLang,
} from '@drtechcares/clinical-i18n';
import {
  buildTtsProxyUrl,
  pickSpeechVoice,
  readStoredKioskVoiceMuted,
  sanitizeSpeechText,
  subscribeKioskVoiceMuted,
  writeStoredKioskVoiceMuted,
  cancelAllKioskVoiceSpeech,
  getSeniorVoiceToggleLabel,
  type VoiceCoachLanguage,
} from './VoiceCoachBanner';
import {
  BIOMETRIC_ENGINE_ID,
  buildBiometricProfileFromSamples,
  captureCanonicalFaceSnapshot,
  extractCanonicalFaceDescriptor,
  extractFacePatchSignature384,
  normalizeVector,
  warmupBiometricEngine,
} from '../lib/biometricEngine';

export type Bucket5Key = keyof AngleEmbeddings5D; // 'center' | 'left' | 'right' | 'up' | 'down'
export type AngleSnapshots5D = Record<Bucket5Key, string>;

export interface FaceID120Result {
  masterEmbedding: number[];
  angleEmbeddings: AngleEmbeddings5D;
  exemplarEmbeddings?: number[][];
  biometricEngine?: 'human-faceres-1024d' | 'arcface-onnx' | 'hybrid-512d';
  geometricRatios?: GeometricAnthropometricRatios;
  framesCaptured: number;
  qualityScore: number;
  avatarDataUrl: string;
  compositeSheetDataUrl?: string;
  angleSnapshots?: Partial<AngleSnapshots5D>;
  faceImageUrls?: string[];
}

export interface FaceID120ScannerProps {
  active: boolean;
  patientName?: string;
  hn?: string;
  onComplete: (result: FaceID120Result) => void;
  onReadyStateChange?: (
    ready: boolean,
    triggerFinalize: () => Promise<FaceID120Result | null>,
  ) => void;
  onCancel?: () => void;
}

const TOTAL_TICKS = 120;

export const ENROLLMENT_SEQUENCE = [
  { step: 1, key: 'center', labelTh: 'หน้าตรง', target: 40, instructionTh: 'ขั้นตอนที่ 1/5: มองตรงหน้ากล้องนิ่งๆ' },
  { step: 2, key: 'left',   labelTh: 'หันซ้าย', target: 20, instructionTh: 'ขั้นตอนที่ 2/5: ค่อยๆ หันหน้าไปทางซ้าย' },
  { step: 3, key: 'right',  labelTh: 'หันขวา', target: 20, instructionTh: 'ขั้นตอนที่ 3/5: ค่อยๆ หันหน้าไปทางขวา' },
  { step: 4, key: 'up',     labelTh: 'เงยหน้า', target: 20, instructionTh: 'ขั้นตอนที่ 4/5: ค่อยๆ เงยหน้าขึ้นเล็กน้อย' },
  { step: 5, key: 'down',   labelTh: 'ก้มหน้า', target: 20, instructionTh: 'ขั้นตอนที่ 5/5: ค่อยๆ ก้มหน้าลงเล็กน้อย' },
] as const;

const BUCKET_TARGETS: Record<Bucket5Key, number> = {
  center: ENROLLMENT_SEQUENCE[0].target,
  left: ENROLLMENT_SEQUENCE[1].target,
  right: ENROLLMENT_SEQUENCE[2].target,
  up: ENROLLMENT_SEQUENCE[3].target,
  down: ENROLLMENT_SEQUENCE[4].target,
};

/**
 * Partition all 120 radial ticks (0..119 clockwise from 12 o'clock) to match the
 * exact 5-angle quotas: 40 interleaved center + 20 up + 20 right + 20 down + 20 left.
 */
const TICK_BUCKET_ASSIGNMENT: Bucket5Key[] = (() => {
  const result = new Array<Bucket5Key>(TOTAL_TICKS);
  const quadBuckets: Exclude<Bucket5Key, 'center'>[] = ['up', 'right', 'down', 'left'];
  for (let q = 0; q < 4; q++) {
    const startIdx = (105 + q * 30) % TOTAL_TICKS; // top=105..14, right=15..44, down=45..74, left=75..104
    const bucket = quadBuckets[q];
    for (let offset = 0; offset < 30; offset++) {
      const idx = (startIdx + offset) % TOTAL_TICKS;
      result[idx] = offset % 3 === 0 ? 'center' : bucket;
    }
  }
  return result;
})();

const BUCKET_TICK_INDICES: Record<Bucket5Key, number[]> = (() => {
  const map: Record<Bucket5Key, number[]> = {
    center: [],
    up: [],
    right: [],
    down: [],
    left: [],
  };
  TICK_BUCKET_ASSIGNMENT.forEach((b, idx) => {
    map[b].push(idx);
  });
  return map;
})();

export interface FaceId120GuidanceItem {
  label: string;
  stepTitle: string;
  turnCue: string;
  holdCue: string;
  voice: string;
}

const FACE_ID_120_GUIDANCE: Record<ClinicalLang, Record<Bucket5Key, FaceId120GuidanceItem>> = {
  th: {
    center: {
      label: 'หน้าตรง',
      stepTitle: 'ขั้นตอนที่ 1/5: มองตรงหน้ากล้องนิ่งๆ',
      turnCue: 'มองตรงเข้าหากล้องให้อยู่ในกรอบวงกลม',
      holdCue: 'กำลังเก็บมิติหน้าตรง กรุณามองตรงนิ่งๆ',
      voice: 'ขั้นตอนที่ 1 กรุณามองตรงเข้าหากล้องนิ่งๆ ครับ',
    },
    left: {
      label: 'หันซ้าย',
      stepTitle: 'ขั้นตอนที่ 2/5: ค่อยๆ หันหน้าไปทางซ้าย',
      turnCue: 'สแกนหน้าตรงครบ 40/40 แล้ว! ขั้นต่อไป: กรุณาหันหน้าไปทางซ้าย',
      holdCue: 'กำลังเก็บมิติใบหน้าด้านซ้าย',
      voice: 'สแกนหน้าตรงครบแล้ว ขั้นตอนที่ 2 กรุณาหันหน้าไปทางซ้ายครับ',
    },
    right: {
      label: 'หันขวา',
      stepTitle: 'ขั้นตอนที่ 3/5: ค่อยๆ หันหน้าไปทางขวา',
      turnCue: 'หันซ้ายครบ 20/20 แล้ว! ขั้นต่อไป: กรุณาหันหน้าไปทางขวา',
      holdCue: 'กำลังเก็บมิติใบหน้าด้านขวา',
      voice: 'หันซ้ายครบแล้ว ขั้นตอนที่ 3 กรุณาหันหน้าไปทางขวาครับ',
    },
    up: {
      label: 'เงยหน้า',
      stepTitle: 'ขั้นตอนที่ 4/5: ค่อยๆ เงยหน้าขึ้นเล็กน้อย',
      turnCue: 'หันขวาครบ 20/20 แล้ว! ขั้นต่อไป: กรุณาเงยหน้าขึ้นเล็กน้อย',
      holdCue: 'กำลังเก็บมิติมุมเงยหน้า',
      voice: 'หันขวาครบแล้ว ขั้นตอนที่ 4 กรุณาเงยหน้าขึ้นเล็กน้อยครับ',
    },
    down: {
      label: 'ก้มหน้า',
      stepTitle: 'ขั้นตอนที่ 5/5: ค่อยๆ ก้มหน้าลงเล็กน้อย',
      turnCue: 'เงยหน้าครบ 20/20 แล้ว! ขั้นสุดท้าย: กรุณาก้มหน้าลงเล็กน้อย',
      holdCue: 'กำลังเก็บมิติมุมก้มหน้า',
      voice: 'เงยหน้าครบแล้ว ขั้นตอนที่ 5 ขั้นสุดท้าย กรุณาก้มหน้าลงเล็กน้อยครับ',
    },
  },
  en: {
    center: {
      label: 'Center',
      stepTitle: 'Step 1/5: Look Straight at the Camera',
      turnCue: 'Please look straight into the circular frame',
      holdCue: 'Hold your face straight for a moment',
      voice: 'Step 1. Please look straight at the camera.',
    },
    left: {
      label: 'Left (<-8°)',
      stepTitle: 'Step 2/5: Turn Your Face Left (<-8°)',
      turnCue: 'Center complete — please turn your head slightly to the left',
      holdCue: 'Capturing left facial angle',
      voice: 'Step 2. Please turn your face slightly to the left.',
    },
    right: {
      label: 'Right (>8°)',
      stepTitle: 'Step 3/5: Turn Your Face Right (>8°)',
      turnCue: 'Left side complete — please turn your head slightly to the right',
      holdCue: 'Capturing right facial angle',
      voice: 'Step 3. Please turn your face slightly to the right.',
    },
    up: {
      label: 'Tilt Up (<-6°)',
      stepTitle: 'Step 4/5: Tilt Your Chin Up (<-6°)',
      turnCue: 'Right side complete — please tilt your head slightly up',
      holdCue: 'Capturing upward facial angle',
      voice: 'Step 4. Please tilt your head slightly up.',
    },
    down: {
      label: 'Tilt Down (>6°)',
      stepTitle: 'Step 5/5: Tilt Your Chin Down (>6°)',
      turnCue: 'Upward angle complete — please tilt your head slightly down',
      holdCue: 'Capturing downward facial angle',
      voice: 'Step 5, final step. Please tilt your head slightly down.',
    },
  },
  cn: {
    center: {
      label: '正视',
      stepTitle: '步骤 1/5：请正视镜头',
      turnCue: '请正视摄像头并保持面部在圆框内',
      holdCue: '请保持正视片刻',
      voice: '第一步，请正视摄像头。',
    },
    left: {
      label: '向左 (<-8°)',
      stepTitle: '步骤 2/5：请向左转头 (<-8°)',
      turnCue: '正视已完成 — 请将头部微微向左转',
      holdCue: '正在采集左侧面部角度',
      voice: '第二步，请将脸微微向左转。',
    },
    right: {
      label: '向右 (>8°)',
      stepTitle: '步骤 3/5：请向右转头 (>8°)',
      turnCue: '左侧已完成 — 请将头部微微向右转',
      holdCue: '正在采集右侧面部角度',
      voice: '第三步，请将脸微微向右转。',
    },
    up: {
      label: '抬头 (<-6°)',
      stepTitle: '步骤 4/5：请微微抬头 (<-6°)',
      turnCue: '右侧已完成 — 请微微抬头',
      holdCue: '正在采集仰视面部角度',
      voice: '第四步，请微微抬头。',
    },
    down: {
      label: '低头 (>6°)',
      stepTitle: '步骤 5/5：请微微低头 (>6°)',
      turnCue: '仰视已完成 — 请微微低头',
      holdCue: '正在采集俯视面部角度',
      voice: '第五步，最后一步，请微微低头。',
    },
  },
};

export function getFaceId120TargetGuidance(
  lang: ClinicalLang,
  bucket: Bucket5Key,
): FaceId120GuidanceItem {
  return (FACE_ID_120_GUIDANCE[lang] ?? FACE_ID_120_GUIDANCE.th)[bucket];
}

const BUCKET_ORDER: Bucket5Key[] = ['center', 'left', 'right', 'up', 'down'];

const FACE_ID_120_UI_COPY: Record<
  ClinicalLang,
  {
    preparingModel: string;
    checkingEdge: string;
    loadingWasm: string;
    readyEdge: string;
    readyWeb: string;
    placeFaceCircle: string;
    detecting3d: string;
    readyStartCue: string;
    noFaceDetectedSub: string;
    computingTitle: string;
    computingSub: string;
    completedVoice: string;
    completedBadge: string;
    qualityLabel: (pct: number) => string;
    completedHeading: (frames: number) => string;
    completedSub: (pct: number) => string;
    framesUnit: string;
    angleFrameUnit: string;
    voiceOnBtn: string;
    voiceOffBtn: string;
    replayBtn: string;
    resetBtn: string;
    pauseCamBtn: string;
    camErrTitle: string;
    camRetryBtn: string;
    waitMoment: string;
  }
> = {
  th: {
    preparingModel: 'กำลังเตรียมโมเดล Deep FaceRes 1024-D & 3D Landmarker...',
    checkingEdge: 'กำลังปลุก Shared Biometric Engine (@vladmandic/human)...',
    loadingWasm: 'กำลังโหลด MediaPipe Face Landmarker & WASM...',
    readyEdge: 'พร้อมสแกนด้วย Shared Deep FaceRes 1024-D',
    readyWeb: 'พร้อมสแกนใบหน้า 3 มิติ (Deep FaceRes 1024-D Real-Time)',
    placeFaceCircle: 'วางใบหน้าให้อยู่ในกรอบวงกลม',
    detecting3d: 'ระบบกำลังตรวจจับตำแหน่งใบหน้า 3 มิติ...',
    readyStartCue: 'เตรียมพร้อม... มองตรงเข้าหากล้องเพื่อเริ่มต้น',
    noFaceDetectedSub: 'ยังไม่พบใบหน้า — ขยับเข้าใกล้กล้องในระยะ 40–60 ซม.',
    computingTitle: 'กำลังประมวลผล 120 เฟรม...',
    computingSub: 'คำนวณ Master + 5 Angle Embeddings',
    completedVoice: 'บันทึกใบหน้าสามมิติครบเรียบร้อยแล้วครับ สามารถกดบันทึกและเปิดใช้งานตู้คีออสได้เลยครับ',
    completedBadge: '120 / 120 เฟรม สมบูรณ์แบบ',
    qualityLabel: (pct) => `คุณภาพแม่แบบ 3D: ${pct}%`,
    completedHeading: (frames) => `ยืนยันการเก็บใบหน้า ${frames} เฟรม (3D) สำเร็จ!`,
    completedSub: (pct) => `บันทึก 5 มุมมองครบถ้วน • คุณภาพ ${pct}%`,
    framesUnit: 'เฟรม',
    angleFrameUnit: 'เฟรม',
    voiceOnBtn: 'เสียงนำทางเปิด',
    voiceOffBtn: 'ปิดเสียง',
    replayBtn: 'พูดซ้ำ',
    resetBtn: 'รีเซ็ต 0/120',
    pauseCamBtn: 'ปิดกล้องชั่วคราว',
    camErrTitle: 'ไม่พบสิทธิ์เข้าถึงกล้อง',
    camRetryBtn: 'ลองเปิดกล้องอีกครั้ง',
    waitMoment: 'กรุณารอสักครู่...',
  },
  en: {
    preparingModel: 'Preparing Deep FaceRes 1024-D & 3D Landmarker...',
    checkingEdge: 'Warming Shared Biometric Engine (@vladmandic/human)...',
    loadingWasm: 'Loading MediaPipe Face Landmarker & WASM...',
    readyEdge: 'Ready with Shared Deep FaceRes 1024-D',
    readyWeb: 'Ready for 3D Multi-Angle Scan (Deep FaceRes 1024-D)',
    placeFaceCircle: 'Position your face inside the circle',
    detecting3d: 'Detecting 3D facial landmarks...',
    readyStartCue: 'Get ready... look straight into the camera to begin',
    noFaceDetectedSub: 'No face detected — move within 40–60 cm of the camera',
    computingTitle: 'Processing 120 frames...',
    computingSub: 'Computing Master + 5 Angle Embeddings',
    completedVoice: '3D multi-angle Face ID capture is complete. You can now save and activate the kiosk.',
    completedBadge: '120 / 120 Frames Complete',
    qualityLabel: (pct) => `3D Template Quality: ${pct}%`,
    completedHeading: (frames) => `Captured ${frames} 3D Face ID frames successfully!`,
    completedSub: (pct) => `All 5 angles saved • Quality ${pct}%`,
    framesUnit: 'frames',
    angleFrameUnit: 'frames',
    voiceOnBtn: 'Voice On',
    voiceOffBtn: 'Muted',
    replayBtn: 'Replay',
    resetBtn: 'Reset 0/120',
    pauseCamBtn: 'Pause Camera',
    camErrTitle: 'Camera access unavailable',
    camRetryBtn: 'Retry Camera',
    waitMoment: 'Please wait a moment...',
  },
  cn: {
    preparingModel: '正在准备 Deep FaceRes 1024-D 3D 人脸模型...',
    checkingEdge: '正在预热共享人脸识别引擎 (@vladmandic/human)...',
    loadingWasm: '正在加载 MediaPipe 人脸模型与 WASM...',
    readyEdge: '已就绪：Shared Deep FaceRes 1024 维',
    readyWeb: '已就绪：3D 多角度人脸扫描 (Deep FaceRes 1024-D)',
    placeFaceCircle: '请将面部置于圆形框内',
    detecting3d: '正在实时检测 3D 面部位置...',
    readyStartCue: '准备开始... 请先正视摄像头',
    noFaceDetectedSub: '未检测到人脸 — 请保持距离镜头 40–60 厘米',
    computingTitle: '正在处理 120 帧数据...',
    computingSub: '正在生成主特征与 5 角度向量模板',
    completedVoice: '3D 多角度人脸采集已完成，现在可以保存并启用自助机了。',
    completedBadge: '120 / 120 帧 采集完成',
    qualityLabel: (pct) => `3D 模板质量：${pct}%`,
    completedHeading: (frames) => `已成功采集 ${frames} 帧 3D 人脸特征！`,
    completedSub: (pct) => `5 个角度全部完成 • 质量 ${pct}%`,
    framesUnit: '帧',
    angleFrameUnit: '帧',
    voiceOnBtn: '语音开启',
    voiceOffBtn: '静音',
    replayBtn: '重播',
    resetBtn: '重置 0/120',
    pauseCamBtn: '暂停摄像头',
    camErrTitle: '无法访问摄像头',
    camRetryBtn: '重新打开摄像头',
    waitMoment: '请稍候...',
  },
};

function toTtsLangAndBcp47(lang: ClinicalLang): { ttsLang: VoiceCoachLanguage; bcp47: string } {
  if (lang === 'en') return { ttsLang: 'en', bcp47: 'en-US' };
  if (lang === 'cn') return { ttsLang: 'zh', bcp47: 'zh-CN' };
  return { ttsLang: 'th', bcp47: 'th-TH' };
}

interface CapturedSample {
  yawDeg: number;
  pitchDeg: number;
  rawYawDeg?: number;
  rawPitchDeg?: number;
  noseBoxRatio?: number;
  sharpness: number;
  bucket: Bucket5Key;
  embedding: number[];
  geometricRatios?: GeometricAnthropometricRatios;
}

export interface CaptureGateContext {
  inCircleGuide?: boolean;
}

export interface PoseBoundingBoxHint {
  x: number;
  y: number;
  width: number;
  height: number;
}

export function computeNormalizedMirroredHeadPoseDeg(
  landmarks3D: readonly { x: number; y: number; z?: number }[],
  aspectRatio = 1,
  faceBox?: PoseBoundingBoxHint | null,
): { yawDeg: number; pitchDeg: number; rollDeg: number; normYawRatio: number; eyeToNoseRatio: number } {
  if (!landmarks3D || landmarks3D.length < 300) {
    return { yawDeg: 0, pitchDeg: 0, rollDeg: 0, normYawRatio: 0, eyeToNoseRatio: 0.65 };
  }
  const rOuter = landmarks3D[33] ?? landmarks3D[234];
  const rInner = landmarks3D[133] ?? rOuter;
  const lInner = landmarks3D[362] ?? landmarks3D[454];
  const lOuter = landmarks3D[263] ?? lInner;
  const noseTip = landmarks3D[1];
  if (!rOuter || !rInner || !lInner || !lOuter || !noseTip) {
    return { yawDeg: 0, pitchDeg: 0, rollDeg: 0, normYawRatio: 0, eyeToNoseRatio: 0.65 };
  }

  const safeAspect =
    Number.isFinite(aspectRatio) && aspectRatio > 0.25 && aspectRatio < 4 ? aspectRatio : 1;

  const rightEyeX = (rOuter.x + rInner.x) * 0.5;
  const rightEyeY = (rOuter.y + rInner.y) * 0.5;
  const leftEyeX = (lOuter.x + lInner.x) * 0.5;
  const leftEyeY = (lOuter.y + lInner.y) * 0.5;

  const eyeMidX = (leftEyeX + rightEyeX) * 0.5;
  const eyeMidY = (leftEyeY + rightEyeY) * 0.5;
  const eyeDx = (leftEyeX - rightEyeX) * safeAspect;
  const eyeDy = leftEyeY - rightEyeY;
  const rawEyeDist = Math.hypot(eyeDx, eyeDy);
  const isPixelSpace = eyeMidX > 2 || rawEyeDist > 2;
  const eyeDistance = Math.max(isPixelSpace ? 12 : 0.02, rawEyeDist);

  // Preview video uses CSS -scale-x-100 mirror:
  // Turning head toward LEFT side of screen shrinks (1 - noseTip.x) -> raw noseTip.x increases relative to eyeMidX & boxMidX -> negative normYawRatio (< 0)
  const mirroredEyeOffset = (eyeMidX - noseTip.x) * safeAspect;
  const boxMidX = faceBox ? faceBox.x + faceBox.width * 0.5 : eyeMidX;
  const boxHalfW = faceBox
    ? Math.max(isPixelSpace ? 20 : 0.03, faceBox.width * safeAspect * 0.5)
    : eyeDistance;
  const mirroredBoxOffset = (boxMidX - noseTip.x) * safeAspect;
  const normYawRatio = faceBox
    ? 0.65 * (mirroredEyeOffset / eyeDistance) + 0.35 * (mirroredBoxOffset / boxHalfW)
    : mirroredEyeOffset / eyeDistance;
  const yawDeg = Math.max(-55, Math.min(55, Number((normYawRatio * 65).toFixed(1))));

  // Anatomical vertical tilt using eye-to-nose ratio, mouth-to-eye span, and bounding-box vertical ratio
  const mouthRight = landmarks3D[61];
  const mouthLeft = landmarks3D[291];
  const upperLip = landmarks3D[13];
  const lowerLip = landmarks3D[14];
  const mouthCandidates = [mouthRight, mouthLeft, upperLip, lowerLip].filter(
    (pt): pt is { x: number; y: number; z?: number } => Boolean(pt && Number.isFinite(pt.y)),
  );
  const mouthSpanX =
    mouthLeft && mouthRight ? Math.abs(mouthLeft.x - mouthRight.x) : 0;
  const lipOpenSpanY =
    upperLip && lowerLip ? Math.abs(lowerLip.y - upperLip.y) : 0;
  const mouthMidY =
    mouthCandidates.length > 0
      ? mouthCandidates.reduce((acc, pt) => acc + pt.y, 0) / mouthCandidates.length
      : 0;
  const hasGenuineMouth =
    mouthCandidates.length >= 2 &&
    (mouthSpanX > (isPixelSpace ? 4 : 0.005) || lipOpenSpanY > (isPixelSpace ? 2 : 0.002)) &&
    mouthMidY > noseTip.y + (isPixelSpace ? 2 : 0.004) &&
    mouthMidY > eyeMidY + (isPixelSpace ? 10 : 0.02);

  const eyeToNoseRatio = (noseTip.y - eyeMidY) / eyeDistance;
  let rawPitch: number;
  if (faceBox) {
    const boxHalfH = Math.max(isPixelSpace ? 20 : 0.03, faceBox.height * 0.5);
    const boxVerticalRatio = (noseTip.y - (faceBox.y + faceBox.height * 0.52)) / boxHalfH;
    if (hasGenuineMouth) {
      const faceVerticalSpan = Math.max(isPixelSpace ? 15 : 0.015, mouthMidY - eyeMidY);
      const verticalRatio = (noseTip.y - eyeMidY) / faceVerticalSpan;
      rawPitch = (verticalRatio - 0.56) * 55 + (eyeToNoseRatio - 0.62) * 30 + boxVerticalRatio * 45;
    } else {
      rawPitch = (eyeToNoseRatio - 0.62) * 55 + boxVerticalRatio * 45;
    }
  } else if (hasGenuineMouth) {
    const faceVerticalSpan = Math.max(isPixelSpace ? 15 : 0.015, mouthMidY - eyeMidY);
    const verticalRatio = (noseTip.y - eyeMidY) / faceVerticalSpan;
    rawPitch = (verticalRatio - 0.56) * 110;
  } else {
    rawPitch = (eyeToNoseRatio - 0.68) * 85;
  }
  const pitchDeg = Math.max(-50, Math.min(50, Number(rawPitch.toFixed(1))));

  const rollRad = Math.atan2(eyeDy, eyeDx);
  const rollDeg = Math.max(-45, Math.min(45, Number(((rollRad * 180) / Math.PI).toFixed(1))));

  return { yawDeg, pitchDeg, rollDeg, normYawRatio, eyeToNoseRatio };
}

export function getActiveEnrollmentStage(
  counts: Record<Bucket5Key, number>,
): (typeof ENROLLMENT_SEQUENCE)[number] | null {
  return ENROLLMENT_SEQUENCE.find((s) => (counts[s.key] ?? 0) < s.target) ?? null;
}

export function classifyDisplayBucket(yawDeg: number, pitchDeg: number): Bucket5Key {
  if (Math.abs(yawDeg) >= Math.abs(pitchDeg) * 1.15) {
    if (yawDeg <= -11) return 'left';
    if (yawDeg >= 11) return 'right';
  }
  if (pitchDeg <= -8 && Math.abs(yawDeg) <= 16) return 'up';
  if (pitchDeg >= 8 && Math.abs(yawDeg) <= 16) return 'down';
  if (yawDeg <= -11) return 'left';
  if (yawDeg >= 11) return 'right';
  return 'center';
}

/**
 * Computes the peak pose score for capturing a distinct thumbnail per step.
 * Higher score always represents stronger alignment with the step's target head pose.
 */
export function computePeakAngleScoreForStep(
  stepKey: Bucket5Key,
  deltaYaw: number,
  deltaPitch: number,
): number {
  switch (stepKey) {
    case 'center':
      return 100 - Math.hypot(deltaYaw, deltaPitch);
    case 'left':
      return -deltaYaw - Math.abs(deltaPitch) * 0.15;
    case 'right':
      return deltaYaw - Math.abs(deltaPitch) * 0.15;
    case 'up':
      return -deltaPitch - Math.abs(deltaYaw) * 0.15;
    case 'down':
      return deltaPitch - Math.abs(deltaYaw) * 0.15;
  }
}

/**
 * Strict 5-Stage Sequential Gate with ZERO fake or auto-pass fallback:
 * Frames ONLY count when the user physically achieves the current step's required angle relative to Step 1.
 */
export function chooseCaptureBucket(
  deltaYaw: number,
  deltaPitch: number,
  counts: Record<Bucket5Key, number>,
  context?: CaptureGateContext,
): Bucket5Key | null {
  const currentStage = getActiveEnrollmentStage(counts);
  if (!currentStage) return null;

  const inCircleGuide = context?.inCircleGuide ?? true;
  if (!inCircleGuide) return null;

  switch (currentStage.key) {
    case 'center':
      return Math.abs(deltaYaw) <= 8 && Math.abs(deltaPitch) <= 9 ? 'center' : null;
    case 'left':
      return deltaYaw <= -11 ? 'left' : null;
    case 'right':
      return deltaYaw >= 11 ? 'right' : null;
    case 'up':
      return deltaPitch <= -8 && Math.abs(deltaYaw) <= 16 ? 'up' : null;
    case 'down':
      return deltaPitch >= 8 && Math.abs(deltaYaw) <= 16 ? 'down' : null;
  }
}

export function buildWrongDirectionCue(
  targetBucket: Bucket5Key,
  deltaYaw: number,
  deltaPitch: number,
  lang: ClinicalLang,
): string | null {
  if (targetBucket === 'center') {
    if (Math.abs(deltaYaw) <= 8 && Math.abs(deltaPitch) <= 9) return null;
    if (lang === 'en') return 'Look straight into the camera to capture frontal baseline frames.';
    if (lang === 'cn') return '请保持面部正对摄像头静止，采集正面图像。';
    return 'กรุณามองตรงหน้ากล้องนิ่งๆ เพื่อเก็บภาพหน้าตรง';
  }
  if (targetBucket === 'left') {
    if (deltaYaw <= -11) return null;
    const progressDeg = Math.max(0, Math.round(-deltaYaw));
    if (lang === 'en') {
      return `Turn head LEFT clearly (${progressDeg}° / > 11° required)`;
    }
    if (lang === 'cn') {
      return `请明显向屏幕左侧转头 (${progressDeg}° / 需大于 11°)`;
    }
    return deltaYaw > 5
      ? `กำลังหันขวา — กรุณาหันหน้าไปทางซ้ายให้ชัดเจน (ตอนนี้ 0° / ต้องมากกว่า 11°)`
      : `กรุณาหันหน้าไปทางซ้ายให้ชัดเจน (ตอนนี้ ${progressDeg}° / ต้องมากกว่า 11°)`;
  }
  if (targetBucket === 'right') {
    if (deltaYaw >= 11) return null;
    const progressDeg = Math.max(0, Math.round(deltaYaw));
    if (lang === 'en') {
      return `Turn head RIGHT clearly (${progressDeg}° / > 11° required)`;
    }
    if (lang === 'cn') {
      return `请明显向屏幕右侧转头 (${progressDeg}° / 需大于 11°)`;
    }
    return deltaYaw < -5
      ? `กำลังหันซ้าย — กรุณาหันหน้าไปทางขวาให้ชัดเจน (ตอนนี้ 0° / ต้องมากกว่า 11°)`
      : `กรุณาหันหน้าไปทางขวาให้ชัดเจน (ตอนนี้ ${progressDeg}° / ต้องมากกว่า 11°)`;
  }
  if (targetBucket === 'up') {
    if (deltaPitch <= -8 && Math.abs(deltaYaw) <= 16) return null;
    const progressDeg = Math.max(0, Math.round(-deltaPitch));
    if (lang === 'en') {
      return `Tilt chin UP clearly (${progressDeg}° / > 8° required)`;
    }
    if (lang === 'cn') {
      return `请明显向上抬头 (${progressDeg}° / 需大于 8°)`;
    }
    return `กรุณาเงยหน้าขึ้นด้านบนให้ชัดเจน (ตอนนี้ ${progressDeg}° / ต้องมากกว่า 8°)`;
  }
  if (targetBucket === 'down') {
    if (deltaPitch >= 8 && Math.abs(deltaYaw) <= 16) return null;
    const progressDeg = Math.max(0, Math.round(deltaPitch));
    if (lang === 'en') {
      return `Tilt chin DOWN clearly (${progressDeg}° / > 8° required)`;
    }
    if (lang === 'cn') {
      return `请明显向下低头 (${progressDeg}° / 需大于 8°)`;
    }
    return `กรุณาก้มหน้าลงด้านล่างให้ชัดเจน (ตอนนี้ ${progressDeg}° / ต้องมากกว่า 8°)`;
  }
  return null;
}

function findNextMissingBucket(counts: Record<Bucket5Key, number>): Bucket5Key {
  return getActiveEnrollmentStage(counts)?.key ?? 'down';
}

function averageNormalizedEmbeddings(vectors: number[][]): number[] {
  if (vectors.length === 0) return [];
  const dim = vectors[0].length;
  const sum = new Array<number>(dim).fill(0);
  for (const vec of vectors) {
    for (let i = 0; i < dim; i++) {
      sum[i] += vec[i] ?? 0;
    }
  }
  return normalizeVector(sum);
}

function playSuccessChime() {
  try {
    const AudioCtx =
      window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AudioCtx) return;
    const ctx = new AudioCtx();
    const now = ctx.currentTime;
    const notes = [880, 1174.66]; // A5 -> D6 clean medical chime
    notes.forEach((freq, i) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(freq, now + i * 0.11);
      gain.gain.setValueAtTime(0.001, now + i * 0.11);
      gain.gain.exponentialRampToValueAtTime(0.14, now + i * 0.11 + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.001, now + i * 0.11 + 0.28);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(now + i * 0.11);
      osc.stop(now + i * 0.11 + 0.3);
    });
    window.setTimeout(() => {
      void ctx.close().catch(() => {});
    }, 900);
  } catch {
    // Ignore audio context blocks in strict kiosk browsers
  }
}

const ANGLE_COMPOSITE_LABELS: Record<Bucket5Key, string> = {
  center: '1. หน้าตรง (Center)',
  left: '2. หันซ้าย (Left)',
  right: '3. หันขวา (Right)',
  up: '4. เงยหน้า (Up)',
  down: '5. ก้มหน้า (Down)',
};

function loadHtmlImageFromDataUrl(src: string): Promise<HTMLImageElement | null> {
  return new Promise((resolve) => {
    if (typeof Image === 'undefined' || !src) {
      resolve(null);
      return;
    }
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => resolve(null);
    img.src = src;
  });
}

async function buildFiveAngleCompositeDataUrl(
  angleSnapshots: Partial<AngleSnapshots5D>,
  fallbackDataUrl: string,
  meta: { patientName?: string; hn?: string; framesCaptured: number; qualityPct: number },
): Promise<string> {
  if (typeof document === 'undefined') return fallbackDataUrl;
  const srcs = BUCKET_ORDER.map((k) => angleSnapshots[k] || fallbackDataUrl);
  if (!srcs.some(Boolean)) return fallbackDataUrl;

  try {
    const imgs = await Promise.all(srcs.map((s) => loadHtmlImageFromDataUrl(s)));
    const width = 1200;
    const height = 360;
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return fallbackDataUrl;

    const bgGrad = ctx.createLinearGradient(0, 0, width, height);
    bgGrad.addColorStop(0, '#F0FDF9');
    bgGrad.addColorStop(1, '#EFF6FF');
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, width, height);

    ctx.fillStyle = '#0F4C81';
    ctx.fillRect(0, 0, width, 54);
    ctx.fillStyle = '#FFFFFF';
    ctx.font = 'bold 20px sans-serif';
    ctx.fillText(
      `DRTECHCARES 3D Face ID • ${meta.framesCaptured} Frames • 5 Angles (${meta.qualityPct}%)`,
      24,
      34,
    );
    const identityLabel = [meta.patientName, meta.hn ? `HN: ${meta.hn}` : '']
      .filter(Boolean)
      .join(' • ');
    if (identityLabel) {
      ctx.font = 'bold 16px sans-serif';
      const tw = ctx.measureText(identityLabel).width;
      ctx.fillStyle = '#A7F3D0';
      ctx.fillText(identityLabel, Math.max(660, width - tw - 24), 33);
    }

    const tileSize = 216;
    const gap = 18;
    const startX = 24;
    const tileY = 74;

    BUCKET_ORDER.forEach((key, idx) => {
      const x = startX + idx * (tileSize + gap);
      ctx.fillStyle = '#FFFFFF';
      ctx.fillRect(x - 2, tileY - 2, tileSize + 4, tileSize + 40);
      ctx.strokeStyle = '#10B981';
      ctx.lineWidth = 2;
      ctx.strokeRect(x - 2, tileY - 2, tileSize + 4, tileSize + 40);

      const img = imgs[idx];
      if (img) {
        ctx.drawImage(img, x, tileY, tileSize, tileSize);
      } else {
        ctx.fillStyle = '#E2E8F0';
        ctx.fillRect(x, tileY, tileSize, tileSize);
      }

      ctx.fillStyle = '#0F4C81';
      ctx.font = 'bold 15px sans-serif';
      ctx.fillText(ANGLE_COMPOSITE_LABELS[key], x + 10, tileY + tileSize + 24);
    });

    return canvas.toDataURL('image/jpeg', 0.88);
  } catch {
    return fallbackDataUrl;
  }
}

export default function FaceID120Scanner({
  active,
  patientName,
  hn,
  onComplete,
  onReadyStateChange,
  onCancel,
}: FaceID120ScannerProps) {
  const { lang, setLang } = useClinicalI18n();
  const ui = FACE_ID_120_UI_COPY[lang] ?? FACE_ID_120_UI_COPY.th;
  const langRef = useRef<ClinicalLang>(lang);
  langRef.current = lang;

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const workCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const snapshotCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const roiLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const rafRef = useRef<number | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const webFallbackTimerRef = useRef<number | null>(null);
  const speakSeqRef = useRef(0);
  const isSpeakingRef = useRef(false);
  const queuedSpeechRef = useRef<{ phrase: string; lang: ClinicalLang } | null>(null);

  const [modelStatus, setModelStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [modelProgressMsg, setModelProgressMsg] = useState(ui.preparingModel);
  const [cameraReady, setCameraReady] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [voiceEnabled, setVoiceEnabled] = useState<boolean>(() => !readStoredKioskVoiceMuted());
  const [phase, setPhase] = useState<'scanning' | 'computing' | 'completed'>('scanning');

  const [capturedCount, setCapturedCount] = useState(0);
  const [bucketCounts, setBucketCounts] = useState<Record<Bucket5Key, number>>({
    center: 0,
    left: 0,
    right: 0,
    up: 0,
    down: 0,
  });
  const [activeBucket, setActiveBucket] = useState<Bucket5Key>('center');
  const [livePose, setLivePose] = useState({ yaw: 0, pitch: 0, roll: 0, sharpness: 0, faceDetected: false });
  const [guidanceText, setGuidanceText] = useState(ui.placeFaceCircle);
  const [subCueText, setSubCueText] = useState(ui.detecting3d);
  const [wrongDirCue, setWrongDirCue] = useState<string | null>(null);
  const [transitionBanner, setTransitionBanner] = useState<string | null>(null);
  const [targetBucketHint, setTargetBucketHint] = useState<Bucket5Key>('center');
  const [finalQualityPct, setFinalQualityPct] = useState(0);
  const [retryNonce, setRetryNonce] = useState(0);
  const [completedAngleSnapshots, setCompletedAngleSnapshots] = useState<Partial<AngleSnapshots5D>>({});
  const [stepThumbnails, setStepThumbnails] = useState<Record<Bucket5Key, string | null>>({
    center: null,
    left: null,
    right: null,
    up: null,
    down: null,
  });

  const samplesRef = useRef<CapturedSample[]>([]);
  const bucketCountsRef = useRef<Record<Bucket5Key, number>>({ ...bucketCounts });
  const prevLandmarksRef = useRef<Array<{ x: number; y: number; z?: number }> | null>(null);
  const lastCaptureAtRef = useRef(0);
  const lastVideoTimeRef = useRef(-1);
  const lastSpokenAtRef = useRef(0);
  const lastSpokenCueRef = useRef('');
  const lastGuidedTargetRef = useRef<Bucket5Key | null>(null);
  const stageTransitionLockUntilRef = useRef<number>(0);
  const baselineCenterYawRef = useRef<number>(0);
  const baselineCenterPitchRef = useRef<number>(0);
  const baselineCenterNoseRatioRef = useRef<number>(0);
  const baselineNoseToEyeRatioRef = useRef<number>(0.65);
  const centerPoseSamplesRef = useRef<{
    yawSum: number;
    pitchSum: number;
    noseRatioSum: number;
    eyeToNoseSum: number;
    count: number;
  }>({
    yawSum: 0,
    pitchSum: 0,
    noseRatioSum: 0,
    eyeToNoseSum: 0,
    count: 0,
  });
  const hasSnapshotFrameRef = useRef(false);
  const bestFrontalDataUrlRef = useRef<string>('');
  const bestFrontalScoreRef = useRef<number>(-1);
  const angleCanvasesRef = useRef<Partial<Record<Bucket5Key, HTMLCanvasElement>>>({});
  const stepThumbnailsRef = useRef<Record<Bucket5Key, string | null>>({
    center: null,
    left: null,
    right: null,
    up: null,
    down: null,
  });
  const bestAngleScoresRef = useRef<Record<Bucket5Key, number>>({
    center: -Infinity,
    left: -Infinity,
    right: -Infinity,
    up: -Infinity,
    down: -Infinity,
  });

  const captureAngleFrameToCanvas = useCallback(
    (
      vid: HTMLVideoElement,
      bucket: Bucket5Key,
      peakScore: number,
      faceBox?: PoseBoundingBoxHint | null,
    ) => {
      if (typeof document === 'undefined' || vid.videoWidth <= 0 || vid.videoHeight <= 0) return;
      const existing = angleCanvasesRef.current[bucket];
      if (existing && peakScore <= bestAngleScoresRef.current[bucket]) return;
      const canvas = existing || document.createElement('canvas');
      canvas.width = 480;
      canvas.height = 480;
      angleCanvasesRef.current[bucket] = canvas;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const vw = vid.videoWidth;
      const vh = vid.videoHeight;
      let sx = Math.max(0, (vw - Math.min(vw, vh)) / 2);
      let sy = Math.max(0, (vh - Math.min(vw, vh)) / 2);
      let sw = Math.min(vw, vh);
      let sh = sw;

      if (faceBox && faceBox.width > 0.04 && faceBox.height > 0.04) {
        const cxPx = (faceBox.x + faceBox.width * 0.5) * vw;
        const cyPx = (faceBox.y + faceBox.height * 0.48) * vh;
        const boxSidePx = Math.max(faceBox.width * vw, faceBox.height * vh) * 1.65;
        const clampedSide = Math.max(120, Math.min(vw, vh, boxSidePx));
        sx = Math.max(0, Math.min(vw - clampedSide, cxPx - clampedSide * 0.5));
        sy = Math.max(0, Math.min(vh - clampedSide, cyPx - clampedSide * 0.5));
        sw = clampedSide;
        sh = clampedSide;
      }

      ctx.save();
      ctx.translate(480, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(vid, sx, sy, sw, sh, 0, 0, 480, 480);
      ctx.restore();
      bestAngleScoresRef.current[bucket] = peakScore;

      try {
        const dataUrl = canvas.toDataURL('image/jpeg', 0.86);
        stepThumbnailsRef.current[bucket] = dataUrl;
        setStepThumbnails((prev) =>
          prev[bucket] === dataUrl ? prev : { ...prev, [bucket]: dataUrl },
        );
      } catch {
        // Ignore dataURL failures in headless environments
      }
    },
    [],
  );

  const buildFinalAngleBundle = useCallback(
    async (framesCaptured: number, qualityScore: number, fallbackOverride?: string) => {
      if (!bestFrontalDataUrlRef.current && hasSnapshotFrameRef.current && snapshotCanvasRef.current) {
        try {
          bestFrontalDataUrlRef.current = snapshotCanvasRef.current.toDataURL('image/jpeg', 0.88);
        } catch {
          bestFrontalDataUrlRef.current = '';
        }
      }
      const centerCanvas = angleCanvasesRef.current.center;
      if (!bestFrontalDataUrlRef.current && centerCanvas) {
        try {
          bestFrontalDataUrlRef.current = centerCanvas.toDataURL('image/jpeg', 0.88);
        } catch {
          // ignore
        }
      }
      const primaryFallback = fallbackOverride || bestFrontalDataUrlRef.current || '';
    const angleSnapshots: Partial<AngleSnapshots5D> = {};
    for (const key of BUCKET_ORDER) {
      const capturedUrl = stepThumbnailsRef.current[key];
      if (capturedUrl) {
        angleSnapshots[key] = capturedUrl;
        continue;
      }
      const canvas = angleCanvasesRef.current[key];
      if (canvas) {
        try {
          angleSnapshots[key] = canvas.toDataURL('image/jpeg', 0.86);
          continue;
        } catch {
          // fallback below
        }
      }
      if (primaryFallback) {
        angleSnapshots[key] = primaryFallback;
      }
    }
      const compositeSheetDataUrl = primaryFallback
        ? await buildFiveAngleCompositeDataUrl(angleSnapshots, primaryFallback, {
            patientName,
            hn,
            framesCaptured,
            qualityPct: Math.round(qualityScore * 100),
          })
        : '';
      const avatarDataUrl = angleSnapshots.center || primaryFallback;
      const faceImageUrls = Array.from(
        new Set(
          [
            compositeSheetDataUrl,
            ...BUCKET_ORDER.map((k) => angleSnapshots[k] ?? ''),
          ].filter(Boolean),
        ),
      );
      return {
        avatarDataUrl,
        compositeSheetDataUrl: compositeSheetDataUrl || avatarDataUrl,
        angleSnapshots,
        faceImageUrls,
      };
    },
    [hn, patientName],
  );
  const completedRef = useRef(false);
  const finalizePromiseRef = useRef<Promise<FaceID120Result | null> | null>(null);
  const finalizedResultRef = useRef<FaceID120Result | null>(null);
  const readyNotifiedRef = useRef(false);
  const unmountedRef = useRef(false);
  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;
  const onReadyStateChangeRef = useRef(onReadyStateChange);
  onReadyStateChangeRef.current = onReadyStateChange;
  const voiceEnabledRef = useRef(voiceEnabled);
  voiceEnabledRef.current = voiceEnabled;

  useEffect(() => {
    return subscribeKioskVoiceMuted((muted) => {
      const enabled = !muted;
      voiceEnabledRef.current = enabled;
      setVoiceEnabled(enabled);
      if (muted) {
        speakSeqRef.current += 1;
        isSpeakingRef.current = false;
        queuedSpeechRef.current = null;
        cancelAllKioskVoiceSpeech(audioRef.current);
      }
    });
  }, []);

  const stopVoicePlayback = useCallback((clearQueue = true) => {
    speakSeqRef.current += 1;
    isSpeakingRef.current = false;
    if (clearQueue) {
      queuedSpeechRef.current = null;
    }
    if (typeof window !== 'undefined') {
      if (webFallbackTimerRef.current !== null) {
        window.clearTimeout(webFallbackTimerRef.current);
        webFallbackTimerRef.current = null;
      }
      if ('speechSynthesis' in window) {
        try {
          window.speechSynthesis.cancel();
        } catch {
          // ignore
        }
      }
    }
    if (audioRef.current) {
      try {
        audioRef.current.pause();
        audioRef.current.src = '';
      } catch {
        // ignore
      }
      audioRef.current = null;
    }
  }, []);

  const playUtterance = useCallback((phrase: string, speakLang: ClinicalLang) => {
    if (!voiceEnabledRef.current || typeof window === 'undefined' || unmountedRef.current) {
      isSpeakingRef.current = false;
      return;
    }
    const trimmed = phrase.trim();
    if (!trimmed) {
      isSpeakingRef.current = false;
      return;
    }

    const seq = ++speakSeqRef.current;
    const { ttsLang, bcp47 } = toTtsLangAndBcp47(speakLang);
    isSpeakingRef.current = true;
    lastSpokenAtRef.current = Date.now();
    lastSpokenCueRef.current = trimmed;

    if (webFallbackTimerRef.current !== null) {
      window.clearTimeout(webFallbackTimerRef.current);
      webFallbackTimerRef.current = null;
    }
    if (audioRef.current) {
      try {
        audioRef.current.pause();
        audioRef.current.src = '';
      } catch {
        // ignore
      }
      audioRef.current = null;
    }
    if ('speechSynthesis' in window) {
      try {
        window.speechSynthesis.cancel();
      } catch {
        // ignore
      }
    }

    const handleFinished = () => {
      if (seq !== speakSeqRef.current) return;
      if (webFallbackTimerRef.current !== null) {
        window.clearTimeout(webFallbackTimerRef.current);
        webFallbackTimerRef.current = null;
      }
      isSpeakingRef.current = false;
      const queued = queuedSpeechRef.current;
      queuedSpeechRef.current = null;
      if (queued && voiceEnabledRef.current && !unmountedRef.current) {
        playUtterance(queued.phrase, queued.lang);
      }
    };

    let started = false;
    let fallbackUsed = false;

    const triggerWebSpeech = () => {
      if (seq !== speakSeqRef.current || started || fallbackUsed) return;
      fallbackUsed = true;
      if (audioRef.current) {
        try {
          audioRef.current.pause();
          audioRef.current.src = '';
        } catch {
          // ignore
        }
        audioRef.current = null;
      }
      if (!('speechSynthesis' in window)) {
        handleFinished();
        return;
      }
      try {
        const wasSpeaking = window.speechSynthesis.speaking || window.speechSynthesis.pending;
        window.speechSynthesis.cancel();
        const cleanSpeechText = sanitizeSpeechText(trimmed);
        if (!cleanSpeechText) {
          handleFinished();
          return;
        }
        const utter = new SpeechSynthesisUtterance(cleanSpeechText);
        utter.lang = bcp47;
        utter.rate = speakLang === 'en' ? 1.0 : 1.04;
        utter.pitch = 1.02;
        const matched = pickSpeechVoice(window.speechSynthesis.getVoices(), bcp47);
        if (matched) {
          utter.voice = matched;
        }
        utter.onend = handleFinished;
        utter.onerror = handleFinished;
        if (wasSpeaking) {
          webFallbackTimerRef.current = window.setTimeout(() => {
            if (seq === speakSeqRef.current) {
              window.speechSynthesis.speak(utter);
            }
          }, 55);
        } else {
          window.speechSynthesis.speak(utter);
        }
      } catch {
        handleFinished();
      }
    };

    const audio = new Audio(buildTtsProxyUrl(trimmed, ttsLang));
    audio.preload = 'auto';
    audio.playbackRate = speakLang === 'en' ? 1.0 : 1.04;
    audioRef.current = audio;
    audio.onplaying = () => {
      if (seq !== speakSeqRef.current) return;
      started = true;
      if (webFallbackTimerRef.current !== null) {
        window.clearTimeout(webFallbackTimerRef.current);
        webFallbackTimerRef.current = null;
      }
    };
    audio.onended = handleFinished;
    audio.onerror = triggerWebSpeech;

    webFallbackTimerRef.current = window.setTimeout(() => {
      if (!started && seq === speakSeqRef.current) {
        triggerWebSpeech();
      }
    }, 1100);

    try {
      const playPromise = audio.play();
      if (playPromise && typeof playPromise.catch === 'function') {
        playPromise.catch(() => {
          triggerWebSpeech();
        });
      }
    } catch {
      triggerWebSpeech();
    }
  }, []);

  const speakGuidance = useCallback(
    (phrase: string, options: { force?: boolean; interrupt?: boolean; speakLang?: ClinicalLang } | boolean = false) => {
      if (!voiceEnabledRef.current || typeof window === 'undefined' || unmountedRef.current) {
        return;
      }
      const opts = typeof options === 'boolean' ? { force: options, interrupt: false } : options;
      const force = Boolean(opts.force);
      const interrupt = Boolean(opts.interrupt);
      const speakLang = opts.speakLang ?? langRef.current;
      const cleanPhrase = phrase.trim();
      if (!cleanPhrase) return;

      const now = Date.now();
      if (!force && (cleanPhrase === lastSpokenCueRef.current && now - lastSpokenAtRef.current < 6200)) {
        return;
      }
      if (!force && now - lastSpokenAtRef.current < 3200) {
        return;
      }

      if (interrupt) {
        queuedSpeechRef.current = null;
        playUtterance(cleanPhrase, speakLang);
        return;
      }

      if (isSpeakingRef.current) {
        if (cleanPhrase !== lastSpokenCueRef.current) {
          queuedSpeechRef.current = { phrase: cleanPhrase, lang: speakLang };
        }
        return;
      }

      playUtterance(cleanPhrase, speakLang);
    },
    [playUtterance],
  );

  const stopCameraHardware = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    if (streamRef.current) {
      for (const track of streamRef.current.getTracks()) {
        try {
          track.stop();
        } catch {
          // ignore
        }
      }
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    if (landmarkerRef.current) {
      try {
        landmarkerRef.current.close();
      } catch {
        // ignore
      }
      landmarkerRef.current = null;
    }
    if (roiLandmarkerRef.current) {
      try {
        roiLandmarkerRef.current.close();
      } catch {
        // ignore
      }
      roiLandmarkerRef.current = null;
    }
  }, []);

  const stopHardware = useCallback(() => {
    stopCameraHardware();
    stopVoicePlayback(true);
  }, [stopCameraHardware, stopVoicePlayback]);

  // Speak immediately in the new language if staff or patient switches TH / EN / 中文 mid-scan
  const prevLangRef = useRef<ClinicalLang>(lang);
  useEffect(() => {
    if (prevLangRef.current === lang) return;
    prevLangRef.current = lang;
    const copy = FACE_ID_120_UI_COPY[lang] ?? FACE_ID_120_UI_COPY.th;
    if (phase === 'completed') {
      const doneFrames = finalizedResultRef.current?.framesCaptured ?? capturedCount;
      const donePct = finalQualityPct;
      setGuidanceText(copy.completedHeading(doneFrames));
      setSubCueText(copy.completedSub(donePct));
      speakGuidance(copy.completedVoice, { force: true, interrupt: true, speakLang: lang });
      return;
    }
    if (phase === 'scanning' && modelStatus === 'ready') {
      const cue = getFaceId120TargetGuidance(lang, targetBucketHint);
      setGuidanceText(`${cue.stepTitle} (${capturedCount}/120)`);
      setSubCueText(
        `${cue.turnCue} (${bucketCountsRef.current[targetBucketHint] ?? 0}/${BUCKET_TARGETS[targetBucketHint]} ${copy.angleFrameUnit})`,
      );
      speakGuidance(cue.voice, { force: true, interrupt: true, speakLang: lang });
    }
  }, [capturedCount, finalQualityPct, lang, modelStatus, phase, speakGuidance, targetBucketHint]);

  const finalizeScan = useCallback(async (): Promise<FaceID120Result | null> => {
    if (finalizedResultRef.current) return finalizedResultRef.current;
    if (finalizePromiseRef.current) return finalizePromiseRef.current;
    if (samplesRef.current.length === 0) return null;

    const currentLang = langRef.current;
    const copy = FACE_ID_120_UI_COPY[currentLang] ?? FACE_ID_120_UI_COPY.th;
    completedRef.current = true;
    setPhase('computing');
    setGuidanceText(copy.computingTitle);
    setSubCueText(copy.computingSub);

    const run = async (): Promise<FaceID120Result | null> => {
      const samples = [...samplesRef.current];
      if (unmountedRef.current) return null;

      const profile = buildBiometricProfileFromSamples(samples, {
        hn,
        fullName: patientName,
      });
      if (!profile) {
        return null;
      }

      const masterEmbedding = profile.master_embedding;
      const angleEmbeddings = profile.angle_embeddings;
      const exemplarEmbeddings = profile.exemplar_embeddings;

      const centerRatios = samples
        .filter((s) => s.bucket === 'center' && s.geometricRatios)
        .map((s) => s.geometricRatios!);
      const fallbackRatios = samples
        .filter((s) => s.geometricRatios)
        .map((s) => s.geometricRatios!);
      const ratioPool = centerRatios.length >= 3 ? centerRatios : fallbackRatios;
      const r12Pool = ratioPool
        .map((r) => r.craniofacialRatios12)
        .filter((arr): arr is number[] => Array.isArray(arr) && arr.length === 12);
      const avgCranio12 =
        r12Pool.length > 0
          ? Array.from({ length: 12 }, (_, idx) =>
              Number((r12Pool.reduce((acc, a) => acc + a[idx], 0) / r12Pool.length).toFixed(4)),
            )
          : undefined;
      const centerPatch384 = profile.geometricRatios?.appearancePatch384;
      const geometricRatios: GeometricAnthropometricRatios | undefined =
        ratioPool.length > 0 || centerPatch384
          ? {
              eyeToNoseRatio:
                ratioPool.length > 0
                  ? Number((ratioPool.reduce((acc, r) => acc + r.eyeToNoseRatio, 0) / ratioPool.length).toFixed(4))
                  : 1,
              faceWidthToUpperHeight:
                ratioPool.length > 0
                  ? Number((ratioPool.reduce((acc, r) => acc + r.faceWidthToUpperHeight, 0) / ratioPool.length).toFixed(4))
                  : 1,
              mouthToEyeRatio:
                ratioPool.length > 0
                  ? Number((ratioPool.reduce((acc, r) => acc + r.mouthToEyeRatio, 0) / ratioPool.length).toFixed(4))
                  : 1,
              noseToChinRatio:
                ratioPool.length > 0
                  ? Number((ratioPool.reduce((acc, r) => acc + r.noseToChinRatio, 0) / ratioPool.length).toFixed(4))
                  : 1,
              ...(centerPatch384 ? { appearancePatch384: centerPatch384 } : {}),
              ...(avgCranio12 ? { craniofacialRatios12: avgCranio12 } : {}),
            }
          : undefined;

      const meanSharpness =
        samples.length > 0
          ? samples.reduce((acc, s) => acc + s.sharpness, 0) / samples.length
          : 60;
      const diversityCount = (Object.keys(bucketCountsRef.current) as Bucket5Key[]).filter(
        (k) => bucketCountsRef.current[k] > 0,
      ).length;
      const qualityScore = Number(
        Math.min(0.99, 0.72 + Math.min(0.18, meanSharpness / 350) + diversityCount * 0.02).toFixed(3),
      );
      const pct = Math.round(qualityScore * 100);
      setFinalQualityPct(pct);

      const visualBundle = await buildFinalAngleBundle(samples.length, qualityScore);
      if (unmountedRef.current) return null;
      setCompletedAngleSnapshots(visualBundle.angleSnapshots);

      const resultPayload: FaceID120Result = {
        masterEmbedding,
        angleEmbeddings,
        exemplarEmbeddings,
        biometricEngine: BIOMETRIC_ENGINE_ID,
        ...(geometricRatios ? { geometricRatios } : {}),
        framesCaptured: samples.length,
        qualityScore,
        avatarDataUrl: visualBundle.avatarDataUrl,
        compositeSheetDataUrl: visualBundle.compositeSheetDataUrl,
        angleSnapshots: visualBundle.angleSnapshots,
        faceImageUrls: visualBundle.faceImageUrls,
      };
      finalizedResultRef.current = resultPayload;

      const doneLang = langRef.current;
      const doneCopy = FACE_ID_120_UI_COPY[doneLang] ?? FACE_ID_120_UI_COPY.th;
      setPhase('completed');
      setGuidanceText(doneCopy.completedHeading(samples.length));
      setSubCueText(doneCopy.completedSub(pct));
      playSuccessChime();
      speakGuidance(doneCopy.completedVoice, { force: true, speakLang: doneLang });

      stopCameraHardware();

      if (!unmountedRef.current) {
        onCompleteRef.current(resultPayload);
        onReadyStateChangeRef.current?.(true, async () => resultPayload);
      }
      return resultPayload;
    };

    finalizePromiseRef.current = run();
    return finalizePromiseRef.current;
  }, [buildFinalAngleBundle, hn, patientName, speakGuidance, stopCameraHardware]);

  const resetScannerState = useCallback(() => {
    const copy = FACE_ID_120_UI_COPY[langRef.current] ?? FACE_ID_120_UI_COPY.th;
    samplesRef.current = [];
    const emptyBucket: Record<Bucket5Key, number> = {
      center: 0,
      left: 0,
      right: 0,
      up: 0,
      down: 0,
    };
    bucketCountsRef.current = emptyBucket;
    setBucketCounts(emptyBucket);
    setCapturedCount(0);
    setPhase('scanning');
    setGuidanceText(copy.placeFaceCircle);
    setSubCueText(copy.readyStartCue);
    setTargetBucketHint('center');
    setWrongDirCue(null);
    setTransitionBanner(null);
    hasSnapshotFrameRef.current = false;
    lastVideoTimeRef.current = -1;
    bestFrontalDataUrlRef.current = '';
    bestFrontalScoreRef.current = -1;
    angleCanvasesRef.current = {};
    bestAngleScoresRef.current = {
      center: -Infinity,
      left: -Infinity,
      right: -Infinity,
      up: -Infinity,
      down: -Infinity,
    };
    const emptyThumbs: Record<Bucket5Key, string | null> = {
      center: null,
      left: null,
      right: null,
      up: null,
      down: null,
    };
    stepThumbnailsRef.current = emptyThumbs;
    setStepThumbnails(emptyThumbs);
    setCompletedAngleSnapshots({});
    completedRef.current = false;
    finalizePromiseRef.current = null;
    finalizedResultRef.current = null;
    readyNotifiedRef.current = false;
    lastGuidedTargetRef.current = null;
    stageTransitionLockUntilRef.current = 0;
    baselineCenterYawRef.current = 0;
    baselineCenterPitchRef.current = 0;
    baselineCenterNoseRatioRef.current = 0;
    baselineNoseToEyeRatioRef.current = 0.65;
    centerPoseSamplesRef.current = { yawSum: 0, pitchSum: 0, noseRatioSum: 0, eyeToNoseSum: 0, count: 0 };
    prevLandmarksRef.current = null;
    onReadyStateChangeRef.current?.(false, finalizeScan);
  }, [finalizeScan]);

    const captureAngleFrameToCanvasRef = useRef(captureAngleFrameToCanvas);
    captureAngleFrameToCanvasRef.current = captureAngleFrameToCanvas;
    const finalizeScanRef = useRef(finalizeScan);
    finalizeScanRef.current = finalizeScan;
    const resetScannerStateRef = useRef(resetScannerState);
    resetScannerStateRef.current = resetScannerState;
    const speakGuidanceRef = useRef(speakGuidance);
    speakGuidanceRef.current = speakGuidance;
    const stopHardwareRef = useRef(stopHardware);
    stopHardwareRef.current = stopHardware;

    useEffect(() => {
    unmountedRef.current = false;
    if (!active) {
      stopHardwareRef.current();
      return;
    }

    let cancelled = false;
    resetScannerStateRef.current();

    const startEngineAndCamera = async () => {
      try {
        const initCopy = FACE_ID_120_UI_COPY[langRef.current] ?? FACE_ID_120_UI_COPY.th;
        setModelStatus('loading');
        setCameraError(null);
        setModelProgressMsg(initCopy.checkingEdge);
        // Warm up the Shared Biometric Engine (@vladmandic/human faceres 1024-D) in parallel with camera access
        const humanWarmupPromise = warmupBiometricEngine();

        const preferredDeviceId = await selectClinicalCameraDeviceId();
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            deviceId: preferredDeviceId ? { ideal: preferredDeviceId } : undefined,
            facingMode: 'user',
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }

        streamRef.current = stream;
        const video = videoRef.current;
        if (!video) {
          stream.getTracks().forEach((t) => t.stop());
          streamRef.current = null;
          return;
        }
        video.srcObject = stream;
        await video.play();
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          streamRef.current = null;
          return;
        }
        setCameraReady(true);

        const snapCanvas = snapshotCanvasRef.current || document.createElement('canvas');
        snapCanvas.width = 480;
        snapCanvas.height = 480;
        snapshotCanvasRef.current = snapCanvas;

        // Single Shared Deep Neural Face Engine + MediaPipe 478-pt Pose Ring
        const loadCopy = FACE_ID_120_UI_COPY[langRef.current] ?? FACE_ID_120_UI_COPY.th;
        setModelProgressMsg(loadCopy.loadingWasm);

        const resolveVision = async () => {
          for (const wasmBase of [
            '/mediapipe/wasm',
            '/models/wasm',
            'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm',
          ]) {
            try {
              return await FilesetResolver.forVisionTasks(wasmBase);
            } catch {
              // Try next WASM base path
            }
          }
          throw new Error('Failed to load MediaPipe WASM files');
        };

        const [humanReady, vision] = await Promise.all([
          humanWarmupPromise,
          resolveVision(),
        ]);
        if (cancelled) return;
        if (!humanReady) {
          throw new Error('Failed to load shared deep face biometric engine (@vladmandic/human)');
        }

        let landmarker: FaceLandmarker | null = null;
        for (const delegate of ['GPU', 'CPU'] as const) {
          try {
            landmarker = await FaceLandmarker.createFromOptions(vision, {
              baseOptions: {
                modelAssetPath: '/models/face_landmarker.task',
                delegate,
              },
              runningMode: 'VIDEO',
              numFaces: 2,
              minFaceDetectionConfidence: 0.36,
              minFacePresenceConfidence: 0.36,
              minTrackingConfidence: 0.36,
              outputFaceBlendshapes: false,
              outputFacialTransformationMatrixes: false,
            });
            break;
          } catch {
            // Try CPU fallback if GPU unavailable
          }
        }
        if (!landmarker) {
          throw new Error('Failed to initialize FaceLandmarker');
        }
        if (cancelled) {
          landmarker.close();
          return;
        }
        landmarkerRef.current = landmarker;

        let roiLandmarker: FaceLandmarker | null = null;
        try {
          roiLandmarker = await FaceLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: '/models/face_landmarker.task', delegate: 'GPU' },
            runningMode: 'IMAGE',
            numFaces: 1,
            minFaceDetectionConfidence: 0.3,
            minFacePresenceConfidence: 0.3,
          });
        } catch {
          try {
            roiLandmarker = await FaceLandmarker.createFromOptions(vision, {
              baseOptions: { modelAssetPath: '/models/face_landmarker.task', delegate: 'CPU' },
              runningMode: 'IMAGE',
              numFaces: 1,
              minFaceDetectionConfidence: 0.3,
              minFacePresenceConfidence: 0.3,
            });
          } catch {
            roiLandmarker = null;
          }
        }
        if (cancelled) {
          roiLandmarker?.close();
          return;
        }
        roiLandmarkerRef.current = roiLandmarker;

        const readyLang = langRef.current;
        const readyCopy = FACE_ID_120_UI_COPY[readyLang] ?? FACE_ID_120_UI_COPY.th;
        const firstCue = getFaceId120TargetGuidance(readyLang, 'center');
        setModelProgressMsg(readyCopy.readyWeb);
        setModelStatus('ready');
        setGuidanceText(`${firstCue.stepTitle} (0/120)`);
        setSubCueText(firstCue.turnCue);
        lastGuidedTargetRef.current = 'center';
        speakGuidance(firstCue.voice, { force: true, interrupt: true, speakLang: readyLang });

        const workCanvas = workCanvasRef.current || document.createElement('canvas');
        workCanvas.width = 112;
        workCanvas.height = 112;
        workCanvasRef.current = workCanvas;
        const roiZoomCanvas = document.createElement('canvas');
        const neuralExtractCanvas = document.createElement('canvas');
        let neuralExtractInFlight = false;
        let lastExtractDispatchedAt = 0;
        let lastSharpnessAt = 0;
        let lastSharpnessVal = 68;
        let lastAngleSnapAt = 0;
        let latestRealAnchor: {
          bucket: Bucket5Key;
          yawDeg: number;
          pitchDeg: number;
          sharpness: number;
          embedding: number[];
          geom?: GeometricAnthropometricRatios;
          arrivedAt: number;
          companionsUsed: number;
        } | null = null;
        let lastRegRoiBox: { x: number; y: number; width: number; height: number } | null = null;
        let regSweepIdx = 0;

        const recordAcceptedSample = (sample: CapturedSample) => {
          if (cancelled || completedRef.current || unmountedRef.current) return false;
          if (samplesRef.current.length >= TOTAL_TICKS) return false;
          const nowRec = performance.now();
          if (nowRec < stageTransitionLockUntilRef.current) return false;
          if (nowRec - lastCaptureAtRef.current < 70) return false;
          const activeStageNow = getActiveEnrollmentStage(bucketCountsRef.current);
          if (!activeStageNow || sample.bucket !== activeStageNow.key) {
            return false;
          }
          if (bucketCountsRef.current[sample.bucket] >= BUCKET_TARGETS[sample.bucket]) {
            return false;
          }
          lastCaptureAtRef.current = nowRec;
          if (sample.bucket === 'center') {
            centerPoseSamplesRef.current.yawSum += sample.rawYawDeg ?? sample.yawDeg;
            centerPoseSamplesRef.current.pitchSum += sample.rawPitchDeg ?? sample.pitchDeg;
            if (typeof sample.noseBoxRatio === 'number') {
              centerPoseSamplesRef.current.noseRatioSum += sample.noseBoxRatio;
              centerPoseSamplesRef.current.eyeToNoseSum += sample.noseBoxRatio;
            }
            centerPoseSamplesRef.current.count += 1;
            const c = centerPoseSamplesRef.current.count;
            baselineCenterYawRef.current = centerPoseSamplesRef.current.yawSum / c;
            baselineCenterPitchRef.current = centerPoseSamplesRef.current.pitchSum / c;
            baselineCenterNoseRatioRef.current = centerPoseSamplesRef.current.noseRatioSum / c;
            baselineNoseToEyeRatioRef.current = centerPoseSamplesRef.current.eyeToNoseSum / c;
          }
          samplesRef.current.push(sample);
          bucketCountsRef.current = {
            ...bucketCountsRef.current,
            [sample.bucket]: bucketCountsRef.current[sample.bucket] + 1,
          };
          const nextTotalCount = samplesRef.current.length;
          setCapturedCount(nextTotalCount);
          setBucketCounts({ ...bucketCountsRef.current });

          const allFilled = (Object.keys(BUCKET_TARGETS) as Bucket5Key[]).every(
            (k) => bucketCountsRef.current[k] >= BUCKET_TARGETS[k],
          );
          if (allFilled || nextTotalCount >= TOTAL_TICKS) {
            readyNotifiedRef.current = true;
            void finalizeScan();
            return true;
          }

          const nextStageAfter = getActiveEnrollmentStage(bucketCountsRef.current);
          if (
            bucketCountsRef.current[activeStageNow.key] >= activeStageNow.target &&
            nextStageAfter &&
            nextStageAfter.key !== activeStageNow.key
          ) {
            stageTransitionLockUntilRef.current = nowRec + 350;
            latestRealAnchor = null;
            if (activeStageNow.key === 'center' && centerPoseSamplesRef.current.count > 0) {
              baselineCenterYawRef.current =
                centerPoseSamplesRef.current.yawSum / centerPoseSamplesRef.current.count;
              baselineCenterPitchRef.current =
                centerPoseSamplesRef.current.pitchSum / centerPoseSamplesRef.current.count;
              baselineCenterNoseRatioRef.current =
                centerPoseSamplesRef.current.noseRatioSum / centerPoseSamplesRef.current.count;
              baselineNoseToEyeRatioRef.current =
                centerPoseSamplesRef.current.eyeToNoseSum / centerPoseSamplesRef.current.count;
            }
            const transLang = langRef.current;
            const nextCue = getFaceId120TargetGuidance(transLang, nextStageAfter.key);
            lastGuidedTargetRef.current = nextStageAfter.key;
            setTargetBucketHint(nextStageAfter.key);
            setWrongDirCue(null);
            setTransitionBanner(nextCue.turnCue);
            setGuidanceText(`${nextCue.stepTitle} (${nextTotalCount}/120)`);
            setSubCueText(nextCue.turnCue);
            speakGuidance(nextCue.voice, { force: true, interrupt: true, speakLang: transLang });
          }
          return true;
        };

        let lastDetectTs = 0;
        const tickLoop = () => {
          if (cancelled || completedRef.current) return;
          const vid = videoRef.current;
          const lm = landmarkerRef.current;
          if (!vid || !lm || vid.readyState < 2 || vid.videoWidth === 0 || vid.videoHeight === 0) {
            rafRef.current = requestAnimationFrame(tickLoop);
            return;
          }

          if (vid.currentTime === lastVideoTimeRef.current) {
            rafRef.current = requestAnimationFrame(tickLoop);
            return;
          }
          lastVideoTimeRef.current = vid.currentTime;

          const now = performance.now();
          const detectTs = Math.max(lastDetectTs + 1, Math.round(now));
          lastDetectTs = detectTs;
          let result;
          try {
            result = lm.detectForVideo(vid, detectTs);
          } catch {
            rafRef.current = requestAnimationFrame(tickLoop);
            return;
          }

          let rawFaces = result?.faceLandmarks ?? [];
          let selection = selectPrimaryFace(rawFaces);
          const coarseH = selection.primaryBox?.height ?? 0;
          if (roiLandmarker && (!selection.primaryFace || coarseH < 0.11)) {
            const sweepRects = [
              { x: 0.25, y: 0.05, width: 0.5, height: 0.6 },
              { x: 0.02, y: 0.05, width: 0.48, height: 0.6 },
              { x: 0.5, y: 0.05, width: 0.48, height: 0.6 },
            ];
            const roiRect = selection.primaryFace && selection.primaryBox
              ? computeHeadZoneRoiFromFaceBox(
                  {
                    x: Math.max(0, (selection.primaryCenter?.x ?? 0.5) - selection.primaryBox.width / 2),
                    y: Math.max(0, (selection.primaryCenter?.y ?? 0.5) - selection.primaryBox.height / 2),
                    width: selection.primaryBox.width,
                    height: selection.primaryBox.height,
                  },
                  2.25,
                )
              : lastRegRoiBox
                ? computeHeadZoneRoiFromFaceBox(lastRegRoiBox, 2.5)
                : sweepRects[regSweepIdx++ % sweepRects.length];
            const upscaled = renderUpscaledRoi256(vid, roiRect, roiZoomCanvas, 320);
            if (upscaled) {
              try {
                const roiRes = roiLandmarker.detect(upscaled.canvas);
                const roiSel = selectPrimaryFace(roiRes.faceLandmarks ?? []);
                if (roiSel.primaryFace && roiSel.primaryFace.length >= 300) {
                  const remapped = remapRoiLandmarksToFrame(roiSel.primaryFace, upscaled.cropPx);
                  rawFaces = [
                    remapped.map((p) => ({
                      x: p.x,
                      y: p.y,
                      z: p.z ?? 0,
                      visibility: 1,
                    })),
                  ];
                  selection = selectPrimaryFace(rawFaces);
                }
              } catch {
                // Keep primary selection
              }
            }
          }
          const activeLang = langRef.current;
          const activeCopy = FACE_ID_120_UI_COPY[activeLang] ?? FACE_ID_120_UI_COPY.th;

          if (!selection.primaryFace && rawFaces.length === 0) {
            lastRegRoiBox = null;
            setLivePose((p) => ({ ...p, faceDetected: false }));
            setGuidanceText(activeCopy.placeFaceCircle);
            setSubCueText(activeCopy.noFaceDetectedSub);
            prevLandmarksRef.current = null;
            rafRef.current = requestAnimationFrame(tickLoop);
            return;
          }

          if (selection.primaryBox && selection.primaryCenter) {
            lastRegRoiBox = {
              x: Math.max(0, selection.primaryCenter.x - selection.primaryBox.width / 2),
              y: Math.max(0, selection.primaryCenter.y - selection.primaryBox.height / 2),
              width: selection.primaryBox.width,
              height: selection.primaryBox.height,
            };
          }
          const landmarks3D = selection.primaryFace ?? rawFaces[0];
          const liveness = evaluateLivenessAndPose(
            landmarks3D,
            prevLandmarksRef.current,
            1,
            false,
          );
          prevLandmarksRef.current = [...landmarks3D];

          // Baseline-calibrated 3D pose estimation (-55°..+55° yaw, -50°..+50° pitch) combining eyes, bounding box & mouth geometry
          const aspectRatio =
            vid.videoWidth > 0 && vid.videoHeight > 0 ? vid.videoWidth / vid.videoHeight : 1;
          const normBox: PoseBoundingBoxHint | null = lastRegRoiBox;
          const normPose = computeNormalizedMirroredHeadPoseDeg(landmarks3D, aspectRatio, normBox);
          const stageAtFrame = getActiveEnrollmentStage(bucketCountsRef.current);
          const stageKey = stageAtFrame?.key ?? 'center';
          const isCenterStage = stageKey === 'center';
          const faceCx = selection.primaryCenter?.x ?? landmarks3D[1].x;
          const faceCy = selection.primaryCenter?.y ?? landmarks3D[1].y;
          const inCircleGuide = Math.hypot(faceCx - 0.5, faceCy - 0.48) <= 0.38;

          if (
            isCenterStage &&
            centerPoseSamplesRef.current.count < 6 &&
            inCircleGuide &&
            Math.abs(normPose.normYawRatio) <= 0.16 &&
            Math.abs(normPose.pitchDeg) <= 18
          ) {
            const c = centerPoseSamplesRef.current;
            const n = c.count + 1;
            baselineCenterYawRef.current = (c.yawSum + normPose.yawDeg) / n;
            baselineCenterPitchRef.current = (c.pitchSum + normPose.pitchDeg) / n;
            baselineNoseToEyeRatioRef.current = (c.eyeToNoseSum + normPose.eyeToNoseRatio) / n;
            baselineCenterNoseRatioRef.current = (c.eyeToNoseSum + normPose.eyeToNoseRatio) / n;
          }

          const yawDeg = Math.max(
            -55,
            Math.min(55, Number((normPose.yawDeg - baselineCenterYawRef.current).toFixed(1))),
          );
          const pitchDeg = Math.max(
            -50,
            Math.min(50, Number((normPose.pitchDeg - baselineCenterPitchRef.current).toFixed(1))),
          );
          const rollDeg = normPose.rollDeg;

          // Throttle CPU getImageData sharpness inspection (~90ms) so 60 FPS video capture never stalls
          if (now - lastSharpnessAt >= 90 || !hasSnapshotFrameRef.current) {
            const crop = alignFaceTo112(vid, landmarks3D, workCanvas);
            if (crop) {
              const quality = evaluateFrameQuality(
                crop,
                { yawDeg, pitchDeg, rollDeg },
                52,
                52,
              );
              lastSharpnessVal = Math.max(1, Math.round(quality.sharpness));
              lastSharpnessAt = now;
            }
          }
          const currentSharpness = lastSharpnessVal;

          setLivePose({
            yaw: Math.round(yawDeg),
            pitch: Math.round(pitchDeg),
            roll: Math.round(rollDeg),
            sharpness: currentSharpness,
            faceDetected: true,
          });

          const displayBucket = classifyDisplayBucket(yawDeg, pitchDeg);
          setActiveBucket(displayBucket);

          // Draw crispest frontal snapshot onto offscreen snapCanvas ONLY while looking centered
          if (
            !hasSnapshotFrameRef.current ||
            (displayBucket === 'center' &&
              Math.abs(yawDeg) <= 8 &&
              Math.abs(pitchDeg) <= 9 &&
              currentSharpness > bestFrontalScoreRef.current)
          ) {
            bestFrontalScoreRef.current = displayBucket === 'center' ? currentSharpness : 0;
            const sctx = snapCanvas.getContext('2d');
            if (sctx && vid.videoWidth > 0 && vid.videoHeight > 0) {
              const side = Math.min(vid.videoWidth, vid.videoHeight);
              const sx = Math.max(0, (vid.videoWidth - side) / 2);
              const sy = Math.max(0, (vid.videoHeight - side) / 2);
              sctx.save();
              sctx.translate(480, 0);
              sctx.scale(-1, 1);
              sctx.drawImage(vid, sx, sy, side, side, 0, 0, 480, 480);
              sctx.restore();
              hasSnapshotFrameRef.current = true;
            }
          }

          const inTransitionLock = now < stageTransitionLockUntilRef.current;
          const gateCtx: CaptureGateContext = {
            inCircleGuide,
          };
          const captureBucket = inTransitionLock
            ? null
            : chooseCaptureBucket(yawDeg, pitchDeg, bucketCountsRef.current, gateCtx);
          const totalSoFar = samplesRef.current.length;
          const meanEar = (liveness.earLeft + liveness.earRight) * 0.5;
          const minOpenEar =
            captureBucket === 'down'
              ? 0.08
              : lastRegRoiBox && lastRegRoiBox.height < 0.22
                ? 0.11
                : 0.13;

          // Capture peak-pose thumbnail ONLY when strict angle gate for current active step passes!
          if (!inTransitionLock && captureBucket && meanEar >= minOpenEar) {
            const peakScore = computePeakAngleScoreForStep(captureBucket, yawDeg, pitchDeg);
            const prevBest = bestAngleScoresRef.current[captureBucket];
            const needsInitialOrPeak =
              !angleCanvasesRef.current[captureBucket] ||
              (peakScore > prevBest && (now - lastAngleSnapAt >= 160 || peakScore > prevBest + 2));
            if (needsInitialOrPeak) {
              captureAngleFrameToCanvas(vid, captureBucket, peakScore, normBox);
              lastAngleSnapAt = now;
            }
          }

          // Strict sequential capture ONLY when captureBucket passes angle gate (ZERO auto-pass fallback!)
          if (!inTransitionLock && captureBucket && meanEar >= minOpenEar && totalSoFar < TOTAL_TICKS) {
            const shouldDispatchNeural =
              !neuralExtractInFlight &&
              (!latestRealAnchor ||
                latestRealAnchor.bucket !== captureBucket ||
                now - lastExtractDispatchedAt >= 95);
            if (shouldDispatchNeural) {
              const snapInput = captureCanonicalFaceSnapshot(vid, lastRegRoiBox, neuralExtractCanvas, landmarks3D);
              if (snapInput) {
                neuralExtractInFlight = true;
                lastExtractDispatchedAt = now;
                const dispatchedStageKey = captureBucket;
                const snapYaw = yawDeg;
                const snapPitch = pitchDeg;
                const snapRawYaw = normPose.yawDeg;
                const snapRawPitch = normPose.pitchDeg;
                const snapEyeToNoseRatio = normPose.eyeToNoseRatio;
                const snapSharpness = currentSharpness;
                const snapGateCtx = { ...gateCtx };
                const rawSnapGeom = extractGeometricAnthropometricRatios(landmarks3D);
                const snapPatch384 = extractFacePatchSignature384(snapInput);
                const snapRatios12 = computeCraniofacialRatios12(
                  landmarks3D,
                  (vid.videoWidth || 1280) / (vid.videoHeight || 720),
                );
                const snapGeom: GeometricAnthropometricRatios | undefined =
                  rawSnapGeom || snapPatch384 || snapRatios12
                    ? {
                        eyeToNoseRatio: rawSnapGeom?.eyeToNoseRatio ?? 1,
                        faceWidthToUpperHeight: rawSnapGeom?.faceWidthToUpperHeight ?? 1,
                        mouthToEyeRatio: rawSnapGeom?.mouthToEyeRatio ?? 1,
                        noseToChinRatio: rawSnapGeom?.noseToChinRatio ?? 1,
                        ...(snapPatch384 ? { appearancePatch384: snapPatch384 } : {}),
                        ...(snapRatios12 ? { craniofacialRatios12: snapRatios12 } : {}),
                      }
                    : undefined;
                void extractCanonicalFaceDescriptor(snapInput)
                  .then((desc) => {
                    if (cancelled || completedRef.current || unmountedRef.current) return;
                    if (!desc.embedding || desc.embedding.length < 128) return;
                    const resolvedAt = performance.now();
                    if (resolvedAt < stageTransitionLockUntilRef.current) return;
                    const activeAtResolve = getActiveEnrollmentStage(bucketCountsRef.current);
                    if (!activeAtResolve || activeAtResolve.key !== dispatchedStageKey) return;
                    const targetBkt = chooseCaptureBucket(
                      snapYaw,
                      snapPitch,
                      bucketCountsRef.current,
                      snapGateCtx,
                    );
                    if (!targetBkt || targetBkt !== activeAtResolve.key) return;
                    latestRealAnchor = {
                      bucket: targetBkt,
                      yawDeg: snapYaw,
                      pitchDeg: snapPitch,
                      sharpness: snapSharpness,
                      embedding: desc.embedding,
                      geom: snapGeom,
                      arrivedAt: resolvedAt,
                      companionsUsed: 0,
                    };
                    recordAcceptedSample({
                      yawDeg: snapYaw,
                      pitchDeg: snapPitch,
                      rawYawDeg: snapRawYaw,
                      rawPitchDeg: snapRawPitch,
                      noseBoxRatio: snapEyeToNoseRatio,
                      sharpness: snapSharpness,
                      bucket: targetBkt,
                      embedding: desc.embedding,
                      geometricRatios: snapGeom,
                    });
                  })
                  .catch(() => {})
                  .finally(() => {
                    neuralExtractInFlight = false;
                  });
              }
            }
            if (
              latestRealAnchor &&
              latestRealAnchor.bucket === captureBucket &&
              now - lastCaptureAtRef.current >= 72 &&
              now - latestRealAnchor.arrivedAt <= 900 &&
              latestRealAnchor.companionsUsed < 6 &&
              Math.abs(yawDeg - latestRealAnchor.yawDeg) <= 6 &&
              Math.abs(pitchDeg - latestRealAnchor.pitchDeg) <= 6
            ) {
              latestRealAnchor.companionsUsed += 1;
              recordAcceptedSample({
                yawDeg,
                pitchDeg,
                rawYawDeg: normPose.yawDeg,
                rawPitchDeg: normPose.pitchDeg,
                noseBoxRatio: normPose.eyeToNoseRatio,
                sharpness: currentSharpness,
                bucket: captureBucket,
                embedding: latestRealAnchor.embedding,
                geometricRatios: latestRealAnchor.geom,
              });
            }
          }

          // Dynamic strict 5-stage sequential guidance & wrong-direction coaching
          const bc = bucketCountsRef.current;
          const nextTotal = samplesRef.current.length;
          const nextTarget = findNextMissingBucket(bc);
          setTargetBucketHint(nextTarget);
          const cue = getFaceId120TargetGuidance(activeLang, nextTarget);

          if (lastGuidedTargetRef.current !== nextTarget) {
            lastGuidedTargetRef.current = nextTarget;
            speakGuidance(cue.voice, { force: true, speakLang: activeLang });
          } else if (!captureBucket && !inTransitionLock) {
            speakGuidance(cue.voice, { speakLang: activeLang });
          }

          if (inTransitionLock) {
            setWrongDirCue(null);
            setGuidanceText(`${cue.stepTitle} (${nextTotal}/120)`);
            setSubCueText(cue.turnCue);
          } else if (captureBucket) {
            setWrongDirCue(null);
            setTransitionBanner(null);
            const capCue = getFaceId120TargetGuidance(activeLang, captureBucket);
            setGuidanceText(`${capCue.stepTitle} (${nextTotal}/120)`);
            setSubCueText(
              `${capCue.holdCue} • ${bc[captureBucket]}/${BUCKET_TARGETS[captureBucket]} ${activeCopy.angleFrameUnit}`,
            );
          } else {
            const wrongCue = buildWrongDirectionCue(nextTarget, yawDeg, pitchDeg, activeLang);
            setWrongDirCue(wrongCue);
            setGuidanceText(`${cue.stepTitle} (${nextTotal}/120)`);
            setSubCueText(
              wrongCue ??
                `${cue.turnCue} (${bc[nextTarget]}/${BUCKET_TARGETS[nextTarget]} ${activeCopy.angleFrameUnit})`,
            );
          }

          rafRef.current = requestAnimationFrame(tickLoop);
        };

        rafRef.current = requestAnimationFrame(tickLoop);
      } catch (err) {
        if (!cancelled) {
          setModelStatus('error');
          setCameraError(
            err instanceof Error
              ? err.message
              : 'ไม่สามารถเปิดกล้องหรือโหลดโมเดล Face ID ได้ กรุณาอนุญาตการใช้กล้อง',
          );
        }
      }
    };

    void startEngineAndCamera();

    return () => {
      cancelled = true;
      unmountedRef.current = true;
      stopHardwareRef.current();
    };
  }, [active, retryNonce]);

  // Compute which of the 120 ticks are lit based on 5-angle capture progress (40+20+20+20+20)
  const tickStates = useMemo(() => {
    const lit = new Array<boolean>(TOTAL_TICKS).fill(false);
    const activeTargetTicks = new Set(BUCKET_TICK_INDICES[targetBucketHint] ?? []);
    (Object.keys(BUCKET_TICK_INDICES) as Bucket5Key[]).forEach((bucketKey) => {
      const indices = BUCKET_TICK_INDICES[bucketKey];
      const filled = Math.min(indices.length, bucketCounts[bucketKey] ?? 0);
      for (let i = 0; i < filled; i++) {
        lit[indices[i]] = true;
      }
    });
    return { lit, activeTargetTicks };
  }, [bucketCounts, targetBucketHint]);

  const progressPct = Math.round((capturedCount / TOTAL_TICKS) * 100);
  const circumferenceSize = 248;
  const cx = circumferenceSize / 2;
  const cy = circumferenceSize / 2;

  return (
    <div className="rounded-3xl border-2 border-emerald-200 bg-gradient-to-b from-[#F0FDF9] via-[#EFF6FF] to-white p-3.5 sm:p-4 text-[#0F4C81] shadow-xl">
      {/* Top Status & Voice Toggle */}
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-300 bg-gradient-to-r from-emerald-50 to-sky-50 px-3 py-1 text-xs font-extrabold text-emerald-700 shadow-xs">
            <Sparkles className="h-3.5 w-3.5 text-emerald-600" />
            Deep FaceRes 1024-D • 120 เฟรม
          </span>
          {patientName && (
            <span className="text-xs font-bold text-[#0F4C81]">
              {patientName} {hn ? `(${hn})` : ''}
            </span>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-1.5">
          <div className="inline-flex items-center rounded-lg border border-sky-200 bg-white p-0.5 text-[11px] font-bold shadow-xs">
            {(
              [
                { code: 'th', label: '1. ไทย' },
                { code: 'en', label: '2. EN' },
                { code: 'cn', label: '3. 中文' },
              ] as const
            ).map((opt) => (
              <button
                key={opt.code}
                type="button"
                onClick={() => {
                  setLang(opt.code);
                  const nextCopy = FACE_ID_120_UI_COPY[opt.code];
                  const nextText =
                    phase === 'completed'
                      ? nextCopy.completedVoice
                      : getFaceId120TargetGuidance(opt.code, targetBucketHint).voice;
                  if (opt.code === lang) {
                    speakGuidance(nextText, { force: true, interrupt: true, speakLang: opt.code });
                  }
                }}
                className={`rounded-md px-2 py-0.5 transition cursor-pointer ${
                  lang === opt.code
                    ? 'bg-emerald-600 text-white shadow-xs'
                    : 'text-[#0F4C81] hover:bg-sky-50'
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
          <button
            type="button"
            onClick={() => {
              const phrase =
                phase === 'completed'
                  ? ui.completedVoice
                  : getFaceId120TargetGuidance(lang, targetBucketHint).voice;
              if (!voiceEnabled) {
                setVoiceEnabled(true);
                voiceEnabledRef.current = true;
                writeStoredKioskVoiceMuted(false);
              }
              speakGuidance(phrase, { force: true, interrupt: true, speakLang: lang });
            }}
            className="inline-flex items-center gap-1 rounded-lg border border-sky-200 bg-white px-2.5 py-1 text-[11px] font-bold text-[#0F4C81] shadow-xs hover:bg-sky-50 cursor-pointer"
          >
            <Volume2 className="h-3.5 w-3.5 text-[#0284C7]" />
            {ui.replayBtn}
          </button>
          <button
            type="button"
            onClick={() => {
              const next = !voiceEnabledRef.current;
              voiceEnabledRef.current = next;
              setVoiceEnabled(next);
              writeStoredKioskVoiceMuted(!next);
              if (!next) {
                stopVoicePlayback(true);
                cancelAllKioskVoiceSpeech(audioRef.current);
              } else {
                const phrase =
                  phase === 'completed'
                    ? ui.completedVoice
                    : getFaceId120TargetGuidance(lang, targetBucketHint).voice;
                speakGuidance(phrase, { force: true, interrupt: true, speakLang: lang });
              }
            }}
            aria-pressed={!voiceEnabled}
            className={`inline-flex items-center gap-1.5 rounded-xl border-2 px-3 py-1.5 text-xs font-extrabold shadow-xs transition-all cursor-pointer ${
              voiceEnabled
                ? 'border-emerald-400 bg-emerald-100 text-emerald-950 hover:bg-emerald-200'
                : 'border-amber-400 bg-amber-100 text-amber-950 hover:bg-amber-200'
            }`}
          >
            {voiceEnabled ? <Volume2 className="h-3.5 w-3.5 text-emerald-700" /> : <VolumeX className="h-3.5 w-3.5 text-amber-700" />}
            <span>{getSeniorVoiceToggleLabel(lang, !voiceEnabled)}</span>
          </button>
          <button
            type="button"
            onClick={() => setRetryNonce((n) => n + 1)}
            className="inline-flex items-center gap-1 rounded-lg border border-sky-200 bg-white px-2.5 py-1 text-[11px] font-bold text-[#0F4C81] shadow-xs hover:bg-sky-50 cursor-pointer"
          >
            <RefreshCw className="h-3.5 w-3.5 text-[#0284C7]" />
            {ui.resetBtn}
          </button>
        </div>
      </div>

      {/* Compact Circular 208x208 Viewfinder + 120-Tick Radial Ring (fits cleanly without clipping head) */}
      <div className="relative mx-auto flex h-[248px] w-[248px] items-center justify-center">
        {/* 120-Tick Radial SVG */}
        <svg
          width={circumferenceSize}
          height={circumferenceSize}
          viewBox={`0 0 ${circumferenceSize} ${circumferenceSize}`}
          className="pointer-events-none absolute inset-0"
          aria-label="120 radial biometric scan progress ring"
        >
          <defs>
            <filter id="emeraldGlow" x="-40%" y="-40%" width="180%" height="180%">
              <feGaussianBlur stdDeviation="2.5" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>
          {Array.from({ length: TOTAL_TICKS }).map((_, idx) => {
            const deg = idx * 3 - 90; // 0 tick at 12 o'clock
            const rad = (deg * Math.PI) / 180;
            const isLit = phase === 'completed' || tickStates.lit[idx];
            const isHint = !isLit && phase === 'scanning' && tickStates.activeTargetTicks.has(idx);
            const innerR = 109;
            const outerR = isLit ? 121 : isHint ? 118 : 115;
            const x1 = cx + innerR * Math.cos(rad);
            const y1 = cy + innerR * Math.sin(rad);
            const x2 = cx + outerR * Math.cos(rad);
            const y2 = cy + outerR * Math.sin(rad);
            const stroke = isLit ? '#10B981' : isHint ? '#0284C7' : '#93C5FD';
            const opacity = isLit ? 1 : isHint ? 0.85 : 0.55;

            return (
              <line
                key={idx}
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                stroke={stroke}
                strokeWidth={isLit ? 2.8 : 1.9}
                strokeLinecap="round"
                opacity={opacity}
                filter={isLit ? 'url(#emeraldGlow)' : undefined}
                style={{ transition: 'all 100ms ease-out' }}
              />
            );
          })}
        </svg>

        {/* Exact 208px x 208px Circular Live Webcam Feed */}
        <div className="relative h-[208px] w-[208px] overflow-hidden rounded-full border-4 border-sky-200 bg-gradient-to-br from-sky-100 to-emerald-50 shadow-[0_0_32px_rgba(2,132,199,0.2)]">
          <video
            ref={videoRef}
            muted
            playsInline
            className="h-full w-full -scale-x-100 object-cover object-center"
          />

          {/* Subtle Inner Reticle */}
          <div className="pointer-events-none absolute inset-3 rounded-full border border-dashed border-white/60" />

          {/* Directional Visual Guide Arrows on Circle Border (Mirror-Aware) */}
          {phase === 'scanning' && cameraReady && !cameraError && (
            <>
              {targetBucketHint === 'left' && (
                <div className="pointer-events-none absolute left-1.5 top-1/2 -translate-y-1/2 z-10 rounded-full bg-amber-500/95 px-2.5 py-1 text-[11px] font-black text-white shadow-lg animate-pulse">
                  ← หันซ้าย
                </div>
              )}
              {targetBucketHint === 'right' && (
                <div className="pointer-events-none absolute right-1.5 top-1/2 -translate-y-1/2 z-10 rounded-full bg-amber-500/95 px-2.5 py-1 text-[11px] font-black text-white shadow-lg animate-pulse">
                  หันขวา →
                </div>
              )}
              {targetBucketHint === 'up' && (
                <div className="pointer-events-none absolute top-2 left-1/2 -translate-x-1/2 z-10 rounded-full bg-amber-500/95 px-2.5 py-1 text-[11px] font-black text-white shadow-lg animate-pulse whitespace-nowrap">
                  ↑ เงยหน้าขึ้น
                </div>
              )}
              {targetBucketHint === 'down' && (
                <div className="pointer-events-none absolute bottom-2 left-1/2 -translate-x-1/2 z-10 rounded-full bg-amber-500/95 px-2.5 py-1 text-[11px] font-black text-white shadow-lg animate-pulse whitespace-nowrap">
                  ↓ ก้มหน้าลง
                </div>
              )}
            </>
          )}

          {/* Model Loading / Camera Error Overlay */}
          {(modelStatus === 'loading' || (!cameraReady && !cameraError)) && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-white/90 p-5 text-center backdrop-blur-xs">
              <Loader2 className="mb-2 h-8 w-8 animate-spin text-emerald-600" />
              <div className="text-xs font-extrabold text-[#0F4C81]">{modelProgressMsg}</div>
              <div className="mt-1 text-[11px] text-[#0284C7]">{ui.waitMoment}</div>
            </div>
          )}

          {cameraError && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-white/95 p-5 text-center">
              <Camera className="mb-2 h-8 w-8 text-rose-500" />
              <div className="text-xs font-extrabold text-rose-700">{ui.camErrTitle}</div>
              <div className="mt-1 text-[11px] text-slate-600">{cameraError}</div>
              <button
                type="button"
                onClick={() => setRetryNonce((n) => n + 1)}
                className="mt-3 rounded-lg bg-gradient-to-r from-[#0284C7] to-[#10B981] px-3 py-1.5 text-xs font-bold text-white"
              >
                {ui.camRetryBtn}
              </button>
            </div>
          )}

          {/* Computing Phase Overlay */}
          {phase === 'computing' && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-white/88 p-4 text-center backdrop-blur-xs">
              <Loader2 className="mb-2 h-9 w-9 animate-spin text-[#0284C7]" />
              <div className="text-sm font-black text-[#0F4C81]">{ui.computingTitle}</div>
              <div className="mt-1 text-[11px] text-[#0284C7]">{ui.computingSub}</div>
            </div>
          )}

          {/* Apple Face ID Checkmark Burst on Completion */}
          {phase === 'completed' && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-emerald-50/92 p-4 text-center backdrop-blur-xs animate-popup-spring">
              <div className="flex h-20 w-20 items-center justify-center rounded-full bg-gradient-to-br from-[#10B981] to-[#0284C7] shadow-[0_0_32px_rgba(16,185,129,0.5)] animate-rep-pop">
                <CheckCircle2 className="h-12 w-12 text-white" />
              </div>
              <div className="mt-3 text-sm font-black text-emerald-800">
                {ui.completedBadge}
              </div>
              <div className="mt-0.5 text-xs font-bold text-[#0F4C81]">
                {ui.qualityLabel(finalQualityPct)}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Live Counter & Dynamic Multi-Language Guidance */}
      <div className="mt-3 text-center">
        <div className="inline-flex items-baseline gap-2 rounded-full border border-emerald-300 bg-white px-4 py-1 shadow-xs">
          <span className="font-mono text-2xl font-black text-emerald-600">{capturedCount}</span>
          <span className="text-xs font-bold text-[#0F4C81]">/ 120 {ui.framesUnit} ({progressPct}%)</span>
        </div>

        <h4 className="mt-2 text-base font-black text-[#0F4C81]">{guidanceText}</h4>
        <p className="mt-0.5 text-xs font-semibold text-[#0284C7]">{subCueText}</p>
        {phase === 'scanning' && wrongDirCue && (
          <div
            role="status"
            className="mx-auto mt-1.5 inline-flex items-center gap-1.5 rounded-full border border-rose-300 bg-rose-50 px-3 py-1 text-xs font-extrabold text-rose-700 animate-toast-in animate-gentle-shake"
          >
            {wrongDirCue}
          </div>
        )}
        {phase === 'scanning' && !wrongDirCue && transitionBanner && (
          <div
            role="status"
            className="mx-auto mt-1.5 inline-flex items-center gap-1.5 rounded-full border border-emerald-300 bg-emerald-50 px-3 py-1 text-xs font-extrabold text-emerald-800 animate-toast-in"
          >
            {transitionBanner}
          </div>
        )}
      </div>

      {/* Strict 5-Stage Sequential Step Cards (Done -> Active -> Locked Queue) */}
      <div className="mt-4 grid grid-cols-5 gap-1.5 text-center text-[11px]">
        {(() => {
          const activeStage = getActiveEnrollmentStage(bucketCounts);
          const activeStageStep = activeStage?.step ?? 6;
          return ENROLLMENT_SEQUENCE.map((stage) => {
            const itemLabel =
              lang === 'th' ? stage.labelTh : getFaceId120TargetGuidance(lang, stage.key).label;
            const count = bucketCounts[stage.key] ?? 0;
            const target = stage.target;
            const ok = count >= target;
            const isActiveStep = phase === 'scanning' && !ok && activeStage?.key === stage.key;
            const isLockedQueue = phase === 'scanning' && !ok && stage.step > activeStageStep;
            const isActivelyCapturing = isActiveStep && activeBucket === stage.key;

            return (
              <div
                key={stage.key}
                className={`rounded-xl border px-1.5 py-2 font-bold transition-all ${
                  ok
                    ? 'border-emerald-400 bg-emerald-50 text-emerald-800 shadow-xs'
                    : isActivelyCapturing
                      ? 'scale-[1.03] border-[#0284C7] bg-sky-50 text-[#0F4C81] ring-2 ring-[#0284C7]/60 shadow-sm'
                      : isActiveStep
                        ? 'scale-[1.03] border-amber-400 bg-amber-50 text-amber-950 ring-2 ring-amber-400/70 shadow-sm'
                        : isLockedQueue
                          ? 'border-slate-200 bg-slate-100/90 text-slate-400 opacity-80'
                          : 'border-sky-200 bg-white text-slate-600'
                }`}
              >
                <div className="flex items-center justify-center gap-0.5 truncate">
                  {isLockedQueue ? (
                    <Lock className="h-3 w-3 shrink-0 text-slate-400" />
                  ) : isActiveStep ? (
                    <span className="inline-block w-2 h-2 rounded-full bg-amber-500 animate-pulse shrink-0" />
                  ) : null}
                  <span className="truncate">
                    {stage.step}. {itemLabel}
                  </span>
                </div>
                <div className="mt-0.5 inline-flex items-center justify-center gap-1 font-mono text-xs font-black">
                  <span>{count}/{target}</span>
                  {ok ? (
                    <Check className="h-3 w-3 text-emerald-600 shrink-0" />
                  ) : isLockedQueue ? (
                    <Lock className="h-3 w-3 text-slate-400 shrink-0" />
                  ) : null}
                </div>
                <div className="mt-0.5 text-[10px] font-bold">
                  {ok
                    ? 'เสร็จสิ้น'
                    : isActiveStep
                      ? '(กำลังสแกน...)'
                      : isLockedQueue
                        ? '(รอคิว)'
                        : ''}
                </div>
              </div>
            );
          });
        })()}
      </div>

      {/* Captured 5-Angle Distinct Peak-Pose Thumbnails (only shown per step once real frames are captured) */}
      {(BUCKET_ORDER.some((k) => Boolean(stepThumbnails[k])) || phase === 'completed') && (
        <div className="mt-3 grid grid-cols-5 gap-2">
          {BUCKET_ORDER.map((bucketKey) => {
            const hasRealFrames = (bucketCounts[bucketKey] ?? 0) > 0;
            const thumb = hasRealFrames
              ? (stepThumbnails[bucketKey] ?? completedAngleSnapshots[bucketKey] ?? null)
              : null;
            const itemLabel = getFaceId120TargetGuidance(lang, bucketKey).label;
            return (
              <div
                key={`thumb-${bucketKey}`}
                className={`overflow-hidden rounded-xl border bg-white shadow-xs card-hover-lift animate-card-enter ${
                  thumb ? 'border-emerald-400' : 'border-slate-200'
                }`}
              >
                {thumb ? (
                  <img
                    src={thumb}
                    alt={itemLabel}
                    className="aspect-square w-full object-cover"
                  />
                ) : (
                  <div className="aspect-square w-full flex flex-col items-center justify-center bg-slate-100/90 text-slate-400">
                    <Camera className="h-5 w-5 opacity-60" />
                  </div>
                )}
                <div className="truncate px-1 py-1 text-center text-[10px] font-bold text-[#0F4C81]">
                  {itemLabel}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Live Head Pose Diagnostics */}
      <div className="mt-3 flex flex-wrap items-center justify-between gap-2 rounded-xl border border-sky-200 bg-gradient-to-r from-emerald-50/80 via-sky-50/80 to-white px-3 py-2 text-[11px] text-[#0F4C81]">
        <div className="flex items-center gap-2">
          <ShieldCheck className={`h-3.5 w-3.5 ${livePose.faceDetected ? 'text-emerald-600' : 'text-amber-500'}`} />
          <span>
            Yaw: <strong className="font-mono text-[#0F4C81]">{livePose.yaw}°</strong> • Pitch:{' '}
            <strong className="font-mono text-[#0F4C81]">{livePose.pitch}°</strong> • Angle:{' '}
            <strong className="font-mono text-emerald-700">{activeBucket}</strong>
          </span>
        </div>
        <span>
          Sharpness: <strong className="font-mono text-[#0F4C81]">{livePose.sharpness}</strong>
        </span>
      </div>

      {onCancel && phase === 'scanning' && (
        <div className="mt-3 flex justify-end">
          <button
            type="button"
            onClick={() => {
              stopHardware();
              onCancel();
            }}
            className="rounded-lg border border-sky-200 bg-white px-3 py-1.5 text-xs font-bold text-[#0F4C81] hover:bg-sky-50"
          >
            {ui.pauseCamBtn}
          </button>
        </div>
      )}
    </div>
  );
}
