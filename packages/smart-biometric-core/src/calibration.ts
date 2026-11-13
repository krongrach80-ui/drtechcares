/**
 * Camera frame quality & exposure deadband calibration for clinical face capture.
 */

export const MIN_SHARPNESS_LAPLACIAN = 4.0;
export const MIN_BRIGHTNESS = 35;
export const MAX_BRIGHTNESS = 225;
export const TARGET_LUMINANCE_MIN = 95;
export const TARGET_LUMINANCE_MAX = 145;
export const MAX_BURNOUT_RATIO = 0.25;
export const MAX_LOGIN_YAW_DEG = 20;
export const MAX_LOGIN_PITCH_DEG = 20;

export interface FrameHeadPose {
  yawDeg: number;
  pitchDeg: number;
  rollDeg?: number;
}

export interface FrameQualityReport {
  acceptable: boolean;
  sharpness: number;
  brightness: number;
  burnoutRatio: number;
  exposureAdjustment: 'increase' | 'decrease' | 'hold';
  reasons: string[];
}

export interface SimpleImageBuffer {
  width: number;
  height: number;
  data: Uint8ClampedArray | Uint8Array;
}

/**
 * Computes Laplacian variance sharpness, mean luminance, and overexposure burnout ratio.
 */
export function evaluateFrameQuality(
  image: SimpleImageBuffer,
  headPose: FrameHeadPose,
  maxYawDeg: number = MAX_LOGIN_YAW_DEG,
  maxPitchDeg: number = MAX_LOGIN_PITCH_DEG,
): FrameQualityReport {
  const { width, height, data } = image;
  const totalPixels = width * height;
  if (totalPixels === 0 || data.length < totalPixels * 4) {
    return {
      acceptable: false,
      sharpness: 0,
      brightness: 0,
      burnoutRatio: 1,
      exposureAdjustment: 'hold',
      reasons: ['EMPTY_FRAME'],
    };
  }

  const luma = new Float32Array(totalPixels);
  let sumLuma = 0;
  let burnedPixels = 0;

  for (let i = 0; i < totalPixels; i += 1) {
    const p = i * 4;
    const y = 0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2];
    luma[i] = y;
    sumLuma += y;
    if (y >= 245) burnedPixels += 1;
  }

  const brightness = sumLuma / totalPixels;
  const burnoutRatio = burnedPixels / totalPixels;

  // 3x3 Laplacian variance for edge sharpness
  let lapSum = 0;
  let lapSqSum = 0;
  let lapCount = 0;

  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
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

  const meanLap = lapCount > 0 ? lapSum / lapCount : 0;
  const sharpness = lapCount > 0 ? Math.max(0, lapSqSum / lapCount - meanLap * meanLap) / 100 : 0;

  const reasons: string[] = [];
  if (sharpness < MIN_SHARPNESS_LAPLACIAN) reasons.push('BLURRY_IMAGE');
  if (brightness < MIN_BRIGHTNESS) reasons.push('TOO_DARK');
  if (brightness > MAX_BRIGHTNESS) reasons.push('TOO_BRIGHT');
  if (burnoutRatio > MAX_BURNOUT_RATIO) reasons.push('OVEREXPOSED_BURNOUT');
  if (Math.abs(headPose.yawDeg) > maxYawDeg) reasons.push('YAW_OUT_OF_RANGE');
  if (Math.abs(headPose.pitchDeg) > maxPitchDeg) reasons.push('PITCH_OUT_OF_RANGE');

  let exposureAdjustment: 'increase' | 'decrease' | 'hold' = 'hold';
  if (brightness < TARGET_LUMINANCE_MIN) exposureAdjustment = 'increase';
  else if (brightness > TARGET_LUMINANCE_MAX || burnoutRatio > MAX_BURNOUT_RATIO) exposureAdjustment = 'decrease';

  return {
    acceptable: reasons.length === 0,
    sharpness: Number(sharpness.toFixed(2)),
    brightness: Number(brightness.toFixed(1)),
    burnoutRatio: Number(burnoutRatio.toFixed(3)),
    exposureAdjustment,
    reasons,
  };
}

