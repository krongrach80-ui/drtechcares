import {
  STRICT_LOGIN_MAX_PITCH_DEG,
  STRICT_LOGIN_MAX_YAW_DEG,
  type SimpleImageBuffer,
} from '@drtechcares/smart-biometric-core';
import type { ClinicalLang } from '@drtechcares/clinical-i18n';
import type { BiometricVerificationJson } from '@drtechcares/face-client';

export interface SmoothedBoxState {
  left: number;
  top: number;
  width: number;
  height: number;
}

export interface Point3DLike {
  x: number;
  y: number;
  z?: number;
}

/**
 * Frame-rate-independent adaptive exponential smoothing for face bounding boxes.
 * Suppresses sub-pixel MediaPipe jitter when the user holds still while staying
 * near-instantaneous when the head moves quickly.
 */
export function smoothBiometricBox(
  prev: SmoothedBoxState | null,
  target: SmoothedBoxState,
  dtMs: number,
): SmoothedBoxState {
  if (!prev) {
    return { ...target };
  }
  const dtSec = Math.max(0.004, Math.min(0.1, (Number.isFinite(dtMs) ? dtMs : 16) / 1000));
  const cxPrev = prev.left + prev.width * 0.5;
  const cyPrev = prev.top + prev.height * 0.5;
  const cxNext = target.left + target.width * 0.5;
  const cyNext = target.top + target.height * 0.5;
  const movePx = Math.hypot(cxNext - cxPrev, cyNext - cyPrev);
  const sizeDeltaPx = Math.abs(target.width - prev.width) + Math.abs(target.height - prev.height);
  const motionPx = movePx + sizeDeltaPx * 0.5;

  // Large jump (new face entered / sudden shift) snaps cleanly without rubber-band drag
  if (motionPx > Math.max(85, target.width * 0.45)) {
    return { ...target };
  }

  // Adaptive cutoff frequency: calm deadband (~9Hz) when still, brisk (~28Hz) during movement
  const freqHz = motionPx < 1.2 ? 9.5 : motionPx < 8 ? 16 : 28;
  const alpha = 1 - Math.exp(-dtSec * freqHz);

  return {
    left: prev.left + (target.left - prev.left) * alpha,
    top: prev.top + (target.top - prev.top) * alpha,
    width: prev.width + (target.width - prev.width) * alpha,
    height: prev.height + (target.height - prev.height) * alpha,
  };
}

/**
 * Adaptive One-Euro-style temporal smoother for 478 facial + dual-iris landmarks.
 * Keeps contours and iris reticles liquid-smooth at 60-120 FPS without motion lag.
 */
export function smoothFacialLandmarks(
  prev: readonly Point3DLike[] | null,
  target: readonly Point3DLike[],
  dtMs: number,
): Point3DLike[] {
  if (!prev || prev.length !== target.length || target.length === 0) {
    return target.map((pt) => ({ x: pt.x, y: pt.y, z: pt.z }));
  }

  const dtSec = Math.max(0.004, Math.min(0.1, (Number.isFinite(dtMs) ? dtMs : 16) / 1000));
  const anchorIndices = [1, 33, 263, 61, 291];
  let anchorMoveSum = 0;
  let anchorCount = 0;
  for (const idx of anchorIndices) {
    const a = prev[idx];
    const b = target[idx];
    if (a && b) {
      anchorMoveSum += Math.hypot(b.x - a.x, b.y - a.y);
      anchorCount += 1;
    }
  }
  const meanAnchorMove = anchorCount > 0 ? anchorMoveSum / anchorCount : 0;

  // Immediate snap on large scene / person change
  if (meanAnchorMove > 0.085) {
    return target.map((pt) => ({ x: pt.x, y: pt.y, z: pt.z }));
  }

  // Smooth calm landmarks gently (~12Hz) when steady, accelerate to ~34Hz when turning head
  const baseFreqHz = meanAnchorMove < 0.0022 ? 12 : meanAnchorMove < 0.009 ? 20 : 34;
  const baseAlpha = 1 - Math.exp(-dtSec * baseFreqHz);
  // Iris landmarks (468..477) track micro-saccades slightly faster for responsive reticle lock
  const irisAlpha = 1 - Math.exp(-dtSec * Math.min(42, baseFreqHz * 1.25));

  const out = new Array<Point3DLike>(target.length);
  for (let i = 0; i < target.length; i += 1) {
    const p = prev[i];
    const t = target[i];
    const a = i >= 468 ? irisAlpha : baseAlpha;
    out[i] = {
      x: p.x + (t.x - p.x) * a,
      y: p.y + (t.y - p.y) * a,
      z:
        typeof p.z === 'number' && typeof t.z === 'number'
          ? p.z + (t.z - p.z) * a
          : t.z,
    };
  }
  return out;
}

/**
 * Computes contrast-compensated Laplacian variance sharpness on an aligned face crop
 * so a sharp backlit/shadowed face is NOT falsely flagged as BLURRY while genuine
 * out-of-focus or motion-blurred frames remain strictly rejected.
 */
export function computeContrastNormalizedSharpness(
  image: SimpleImageBuffer | null | undefined,
): number {
  if (!image || image.width < 16 || image.height < 16 || image.data.length < image.width * image.height * 4) {
    return 0;
  }
  const { width, height, data } = image;
  const totalPixels = width * height;
  const luma = new Float32Array(totalPixels);

  let innerSum = 0;
  let innerSqSum = 0;
  let innerCount = 0;
  const xMin = Math.floor(width * 0.16);
  const xMax = Math.ceil(width * 0.84);
  const yMin = Math.floor(height * 0.16);
  const yMax = Math.ceil(height * 0.84);

  for (let y = 0; y < height; y += 1) {
    const inY = y >= yMin && y <= yMax;
    for (let x = 0; x < width; x += 1) {
      const idx = y * width + x;
      const p = idx * 4;
      const l = 0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2];
      luma[idx] = l;
      if (inY && x >= xMin && x <= xMax) {
        innerSum += l;
        innerSqSum += l * l;
        innerCount += 1;
      }
    }
  }

  let lapSum = 0;
  let lapSqSum = 0;
  let lapCount = 0;
  for (let y = Math.max(1, yMin); y < Math.min(height - 1, yMax); y += 1) {
    for (let x = Math.max(1, xMin); x < Math.min(width - 1, xMax); x += 1) {
      const idx = y * width + x;
      const lap =
        luma[idx - width] +
        luma[idx + width] +
        luma[idx - 1] +
        luma[idx + 1] -
        4 * luma[idx];
      lapSum += lap;
      lapSqSum += lap * lap;
      lapCount += 1;
    }
  }

  if (lapCount === 0 || innerCount === 0) return 0;
  const meanLap = lapSum / lapCount;
  const rawSharpness = Math.max(0, lapSqSum / lapCount - meanLap * meanLap) / 100;

  // Truly flat / heavily motion-blurred frames stay strictly unboosted so blur protection holds
  if (rawSharpness < 1.15) {
    return Number(rawSharpness.toFixed(2));
  }

  const innerMean = innerSum / innerCount;
  const innerStd = Math.sqrt(Math.max(0, innerSqSum / innerCount - innerMean * innerMean));
  if (innerStd >= 36 || innerStd < 7) {
    return Number(rawSharpness.toFixed(2));
  }

  // Compensate strictly for shadow-compressed dynamic range (capped at 2.4x variance boost)
  const scaleFactor = Math.min(2.4, Math.pow(38 / Math.max(12, innerStd), 1.65));
  return Number((rawSharpness * scaleFactor).toFixed(2));
}

export interface KioskReadinessGuidanceInput {
  isSpoof: boolean;
  isBioVerified: boolean;
  yawDeg: number;
  pitchDeg: number;
  inspection: Pick<BiometricVerificationJson, 'scan_status' | 'error_code'> | null;
  blurryFrameStreak: number;
  lang: ClinicalLang;
}

/**
 * Resolves clear, actionable clinical posture/focus guidance whenever a face is visible
 * but blocked from completing biometric matching, preventing silent "(0/3)" stalls.
 */
export function resolveKioskReadinessGuidance(
  input: KioskReadinessGuidanceInput,
): string | null {
  if (input.isSpoof || !input.isBioVerified) return null;

  const { lang } = input;
  if (input.inspection?.error_code === 'OBSTRUCTED_EYES') {
    if (lang === 'en') return 'Keep eyes uncovered and look directly at the lens';
    if (lang === 'cn') return '请勿遮挡双眼并直视摄像头';
    return 'กรุณาเปิดใบหน้าและมองตรงที่กล้องให้เห็นดวงตาชัดเจน';
  }
  if (input.inspection?.error_code === 'OBSTRUCTED_MOUTH') {
    if (lang === 'en') return 'Please remove face mask or obstruction before scanning';
    if (lang === 'cn') return '请摘下口罩或移开面部遮挡物';
    return 'กรุณาถอดหน้ากากหรือเปิดช่วงปากให้ชัดเจน';
  }
  if (Math.abs(input.yawDeg) > STRICT_LOGIN_MAX_YAW_DEG) {
    if (lang === 'en') return 'Turn your face straight toward the camera';
    if (lang === 'cn') return '请正对摄像头保持面部居中';
    return 'กรุณาหันหน้าตรงเข้าหากล้อง';
  }
  if (Math.abs(input.pitchDeg) > STRICT_LOGIN_MAX_PITCH_DEG) {
    const tiltingUp = input.pitchDeg < -STRICT_LOGIN_MAX_PITCH_DEG;
    if (lang === 'en') {
      return tiltingUp
        ? 'Lower your chin slightly to look straight into the camera'
        : 'Raise your chin slightly to look straight into the camera';
    }
    if (lang === 'cn') {
      return tiltingUp ? '请略微低头正视摄像头' : '请略微抬头正视摄像头';
    }
    return tiltingUp
      ? 'กรุณาก้มหน้าลงเล็กน้อยเพื่อมองตรงเข้ากล้อง'
      : 'กรุณาเงยหน้าขึ้นเล็กน้อยเพื่อมองตรงเข้ากล้อง';
  }
  if (input.inspection?.error_code === 'BLURRY' || input.blurryFrameStreak >= 2) {
    if (lang === 'en') return 'Hold still briefly so the camera can focus clearly';
    if (lang === 'cn') return '请保持静止片刻以便镜头清晰对焦';
    return 'กรุณาหยุดนิ่งสักครู่ เพื่อให้กล้องโฟกัสใบหน้าและม่านตาชัดขึ้น';
  }
  return null;
}

