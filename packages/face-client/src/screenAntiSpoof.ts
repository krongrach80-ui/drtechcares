import type { SimpleImageBuffer } from '@drtechcares/smart-biometric-core';

export type ScreenSpoofReason =
  | 'NONE'
  | 'MOIRE_GRID'
  | 'PHONE_BEZEL'
  | 'SCREEN_BACKLIGHT_CAST'
  | 'GLASS_SPECULAR';

export interface NormalizedFaceBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface ScreenAntiSpoofReport {
  spoofDetected: boolean;
  moireScore: number;
  bezelScore: number;
  screenCastScore: number;
  specularScore: number;
  combinedSpoofRisk: number;
  reason: ScreenSpoofReason;
}

export interface AnalyzeScreenAntiSpoofInput {
  faceCrop?: SimpleImageBuffer | null;
  contextFrame?: SimpleImageBuffer | null;
  faceBox?: NormalizedFaceBox | null;
  faceScale?: number;
}

function clamp01(v: number): number {
  if (!Number.isFinite(v)) return 0;
  return Math.max(0, Math.min(1, v));
}

function lumaAt(data: Uint8ClampedArray | Uint8Array, width: number, x: number, y: number): number {
  const p = (y * width + x) * 4;
  return 0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2];
}

function rgDiffAt(data: Uint8ClampedArray | Uint8Array, width: number, x: number, y: number): number {
  const p = (y * width + x) * 4;
  return data[p] - data[p + 1];
}

/**
 * Inspects aligned face-crop skin patches (forehead + cheeks) for:
 * - LCD/OLED subpixel moire & 1-px periodic checkerboard lattice ripples
 * - Screen glass specular glare spots
 * - Narrow-band cool LED screen backlight color shift (B > R on skin)
 * Distance-adaptive: dampens high-frequency upscaling/sharpening noise when standing 1.2-2.5m away.
 */
export function analyzeFaceCropScreenSignals(
  image: SimpleImageBuffer | null | undefined,
  faceScale: number = 0.4,
): {
  moireScore: number;
  specularScore: number;
  screenCastScore: number;
} {
  if (!image || image.width < 24 || image.height < 24 || image.data.length < image.width * image.height * 4) {
    return { moireScore: 0, specularScore: 0, screenCastScore: 0 };
  }

  const { width, height, data } = image;
  const patches = [
    {
      x0: Math.max(2, Math.floor(width * 0.28)),
      x1: Math.min(width - 3, Math.floor(width * 0.72)),
      y0: Math.max(2, Math.floor(height * 0.11)),
      y1: Math.min(height - 3, Math.floor(height * 0.26)),
    },
    {
      x0: Math.max(2, Math.floor(width * 0.16)),
      x1: Math.min(width - 3, Math.floor(width * 0.34)),
      y0: Math.max(2, Math.floor(height * 0.52)),
      y1: Math.min(height - 3, Math.floor(height * 0.7)),
    },
    {
      x0: Math.max(2, Math.floor(width * 0.66)),
      x1: Math.min(width - 3, Math.floor(width * 0.84)),
      y0: Math.max(2, Math.floor(height * 0.52)),
      y1: Math.min(height - 3, Math.floor(height * 0.7)),
    },
  ];

  let alternatingHighFreqCount = 0;
  let totalTripleSamples = 0;
  let highFreqLapSum = 0;
  let smoothGradSum = 0;
  let subpixelJitterCount = 0;

  let pixelCount = 0;
  let specularCount = 0;
  let sumR = 0;
  let sumG = 0;
  let sumB = 0;

  for (const patch of patches) {
    for (let y = patch.y0; y <= patch.y1; y += 1) {
      for (let x = patch.x0; x <= patch.x1; x += 1) {
        const p = (y * width + x) * 4;
        const r = data[p];
        const g = data[p + 1];
        const b = data[p + 2];
        const luma = 0.299 * r + 0.587 * g + 0.114 * b;

        pixelCount += 1;
        sumR += r;
        sumG += g;
        sumB += b;

        const maxC = Math.max(r, g, b);
        const minC = Math.min(r, g, b);
        if (luma >= 242 && maxC - minC <= 18) {
          specularCount += 1;
        }

        const lLeft = lumaAt(data, width, x - 1, y);
        const lRight = lumaAt(data, width, x + 1, y);
        const lUp = lumaAt(data, width, x, y - 1);
        const lDown = lumaAt(data, width, x, y + 1);

        const d1x = luma - lLeft;
        const d2x = lRight - luma;
        const d1y = luma - lUp;
        const d2y = lDown - luma;

        if (d1x * d2x < -16 || d1y * d2y < -16) {
          alternatingHighFreqCount += 1;
        }

        const rgL = rgDiffAt(data, width, x - 1, y);
        const rgC = r - g;
        const rgR = rgDiffAt(data, width, x + 1, y);
        if ((rgC - rgL) * (rgR - rgC) < -14) {
          subpixelJitterCount += 1;
        }

        const lap1 = Math.abs(lLeft - 2 * luma + lRight) + Math.abs(lUp - 2 * luma + lDown);
        const smooth2 = Math.abs(lRight - lLeft) + Math.abs(lDown - lUp);
        highFreqLapSum += lap1;
        smoothGradSum += smooth2 + 2.0;
        totalTripleSamples += 1;
      }
    }
  }

  if (totalTripleSamples === 0 || pixelCount === 0) {
    return { moireScore: 0, specularScore: 0, screenCastScore: 0 };
  }

  const altRatio = alternatingHighFreqCount / totalTripleSamples;
  const subpixelRatio = subpixelJitterCount / totalTripleSamples;
  const freqRatio = highFreqLapSum / Math.max(1, smoothGradSum);

  // Dampen moire/sharpening score at far distances (< 0.35 faceScale) where webcam digital zoom sharpens noise
  const distanceDampen = faceScale < 0.35 ? Math.max(0.35, faceScale / 0.35) : 1;
  const moireFromAlt = clamp01((altRatio - 0.24) / 0.24);
  const moireFromSubpixel = clamp01((subpixelRatio - 0.2) / 0.24);
  const moireFromFreq = clamp01((freqRatio - 1.45) / 0.95);
  const moireScore = clamp01(
    (moireFromAlt * 0.45 + moireFromSubpixel * 0.35 + moireFromFreq * 0.28) * distanceDampen,
  );

  const specRatio = specularCount / pixelCount;
  const specularScore = clamp01((specRatio - 0.05) / 0.14);

  const meanR = sumR / pixelCount;
  const meanG = sumG / pixelCount;
  const meanB = sumB / pixelCount;
  const blueToRedRatio = meanB / Math.max(14, meanR);
  const coolBlueCast = clamp01((blueToRedRatio - 1.07) / 0.22);
  const flatGreenCast = meanB > meanG * 1.08 ? clamp01((meanB / Math.max(14, meanG) - 1.06) / 0.18) : 0;
  const screenCastScore = clamp01(coolBlueCast * 0.72 + flatGreenCast * 0.28);

  return {
    moireScore: Number(moireScore.toFixed(3)),
    specularScore: Number(specularScore.toFixed(3)),
    screenCastScore: Number(screenCastScore.toFixed(3)),
  };
}

/**
 * Inspects uncropped camera frame around the detected face bounding box for
 * enclosed 4-sided smartphone/tablet bezels + screen luminance jumps while ignoring
 * ordinary room walls, air conditioners, or window frames behind a distant live person.
 */
export function analyzeContextBezelAndStep(
  contextFrame: SimpleImageBuffer | null | undefined,
  faceBox?: NormalizedFaceBox | null,
  faceScale?: number,
): number {
  if (
    !contextFrame ||
    contextFrame.width < 48 ||
    contextFrame.height < 48 ||
    contextFrame.data.length < contextFrame.width * contextFrame.height * 4
  ) {
    return 0;
  }

  const { width, height, data } = contextFrame;
  const bx = faceBox ? clamp01(faceBox.x) : 0.28;
  const by = faceBox ? clamp01(faceBox.y) : 0.2;
  const bw = faceBox ? clamp01(faceBox.width) : 0.44;
  const bh = faceBox ? clamp01(faceBox.height) : 0.6;
  const scale = clamp01(faceScale ?? bh);

  const yStart = Math.max(3, Math.floor((by - bh * 0.14) * height));
  const yEnd = Math.min(height - 4, Math.floor((by + bh * 1.12) * height));
  const xStart = Math.max(3, Math.floor((bx - bw * 0.22) * width));
  const xEnd = Math.min(width - 4, Math.floor((bx + bw * 1.22) * width));

  const leftMinX = Math.max(2, Math.floor((bx - bw * 0.3) * width));
  const leftMaxX = Math.max(leftMinX + 2, Math.floor((bx + bw * 0.03) * width));
  const rightMinX = Math.min(width - 4, Math.floor((bx + bw * 0.97) * width));
  const rightMaxX = Math.min(width - 3, Math.floor((bx + bw * 1.3) * width));

  let bestLeftEdge = 0;
  let bestLeftX = -1;
  const rowSpan = Math.max(1, yEnd - yStart);
  for (let x = leftMinX; x <= leftMaxX; x += 1) {
    let strongRows = 0;
    for (let y = yStart; y <= yEnd; y += 2) {
      const gx = Math.abs(lumaAt(data, width, x + 1, y) - lumaAt(data, width, x - 1, y));
      if (gx >= 24) strongRows += 1;
    }
    const frac = strongRows / Math.max(1, Math.ceil(rowSpan / 2));
    if (frac > bestLeftEdge) {
      bestLeftEdge = frac;
      bestLeftX = x;
    }
  }

  let bestRightEdge = 0;
  let bestRightX = -1;
  for (let x = rightMinX; x <= rightMaxX; x += 1) {
    let strongRows = 0;
    for (let y = yStart; y <= yEnd; y += 2) {
      const gx = Math.abs(lumaAt(data, width, x + 1, y) - lumaAt(data, width, x - 1, y));
      if (gx >= 24) strongRows += 1;
    }
    const frac = strongRows / Math.max(1, Math.ceil(rowSpan / 2));
    if (frac > bestRightEdge) {
      bestRightEdge = frac;
      bestRightX = x;
    }
  }

  const topMinY = Math.max(2, Math.floor((by - bh * 0.28) * height));
  const topMaxY = Math.max(topMinY + 2, Math.floor((by + bh * 0.05) * height));
  const botMinY = Math.min(height - 4, Math.floor((by + bh * 0.95) * height));
  const botMaxY = Math.min(height - 3, Math.floor((by + bh * 1.28) * height));

  let bestTopEdge = 0;
  const colSpan = Math.max(1, xEnd - xStart);
  for (let y = topMinY; y <= topMaxY; y += 1) {
    let strongCols = 0;
    for (let x = xStart; x <= xEnd; x += 2) {
      const gy = Math.abs(lumaAt(data, width, x, y + 1) - lumaAt(data, width, x, y - 1));
      if (gy >= 24) strongCols += 1;
    }
    const frac = strongCols / Math.max(1, Math.ceil(colSpan / 2));
    if (frac > bestTopEdge) bestTopEdge = frac;
  }

  let bestBottomEdge = 0;
  for (let y = botMinY; y <= botMaxY; y += 1) {
    let strongCols = 0;
    for (let x = xStart; x <= xEnd; x += 2) {
      const gy = Math.abs(lumaAt(data, width, x, y + 1) - lumaAt(data, width, x, y - 1));
      if (gy >= 24) strongCols += 1;
    }
    const frac = strongCols / Math.max(1, Math.ceil(colSpan / 2));
    if (frac > bestBottomEdge) bestBottomEdge = frac;
  }

  // Real phone/paper screens form an enclosed 4-sided rectangle (ceiling/wall corners only have 1-2 partial lines)
  const hasEnclosedFourSidedRect =
    bestLeftEdge >= 0.56 &&
    bestRightEdge >= 0.56 &&
    bestTopEdge >= 0.5 &&
    bestBottomEdge >= 0.48;

  let luminanceStepBoost = 0;
  if (bestLeftX > 3 && bestRightX > bestLeftX + 16 && bestRightX < width - 4) {
    const innerInset = Math.max(5, Math.min(12, Math.floor((bestRightX - bestLeftX) * 0.12)));
    const outerInset = 5;
    let insideSum = 0;
    let insideN = 0;
    let outsideSum = 0;
    let outsideN = 0;
    const innerYStart = Math.min(yEnd, yStart + 6);
    const innerYEnd = Math.max(innerYStart, yEnd - 6);
    for (let y = innerYStart; y <= innerYEnd; y += 3) {
      insideSum += lumaAt(data, width, Math.min(width - 2, bestLeftX + innerInset), y);
      insideSum += lumaAt(data, width, Math.max(1, bestRightX - innerInset), y);
      insideN += 2;
      outsideSum += lumaAt(data, width, Math.max(0, bestLeftX - outerInset), y);
      outsideSum += lumaAt(data, width, Math.min(width - 1, bestRightX + outerInset), y);
      outsideN += 2;
    }
    if (insideN > 0 && outsideN > 0) {
      const diff = Math.abs(insideSum / insideN - outsideSum / outsideN);
      luminanceStepBoost = clamp01((diff - 26) / 55);
    }
  }

  if (!hasEnclosedFourSidedRect) {
    return 0;
  }

  const verticalPair = (bestLeftEdge + bestRightEdge) / 2;
  const horizontalPair = (bestTopEdge + bestBottomEdge) / 2;
  const rawRectScore =
    verticalPair * 0.42 +
    horizontalPair * 0.34 +
    luminanceStepBoost * 0.3;
  // Require a distinct inside-vs-outside luminance jump around the enclosed rectangle
  const stepGate = luminanceStepBoost >= 0.2 ? 1.12 : 0.6;
  // Distant room boxes (< 0.32 scale) require extra-strong contrast step
  const distanceWeight = scale < 0.32 ? 0.8 : 1.0;

  return Number(clamp01(rawRectScore * stepGate * distanceWeight).toFixed(3));
}

/**
 * Combines aligned face-crop subpixel/glare/color-cast metrics and enclosed 4-sided
 * phone bezel inspection into an anti-spoof presentation attack score.
 */
export function analyzeScreenAndBezelArtifacts(
  input: AnalyzeScreenAntiSpoofInput,
): ScreenAntiSpoofReport {
  const effectiveScale = clamp01(input.faceScale ?? input.faceBox?.height ?? 0.4);
  const cropSignals = analyzeFaceCropScreenSignals(input.faceCrop, effectiveScale);
  const bezelScore = analyzeContextBezelAndStep(input.contextFrame, input.faceBox, effectiveScale);

  const combinedSpoofRisk = Number(
    clamp01(
      bezelScore * 0.56 +
        cropSignals.moireScore * 0.42 +
        cropSignals.screenCastScore * 0.22 +
        cropSignals.specularScore * 0.2,
    ).toFixed(3),
  );

  let reason: ScreenSpoofReason = 'NONE';
  if (bezelScore >= 0.7) {
    reason = 'PHONE_BEZEL';
  } else if (cropSignals.moireScore >= 0.72) {
    reason = 'MOIRE_GRID';
  } else if (cropSignals.screenCastScore >= 0.65 && cropSignals.moireScore >= 0.4) {
    reason = 'SCREEN_BACKLIGHT_CAST';
  } else if (cropSignals.specularScore >= 0.7 && (cropSignals.moireScore >= 0.4 || bezelScore >= 0.45)) {
    reason = 'GLASS_SPECULAR';
  }

  const spoofDetected =
    bezelScore >= 0.7 ||
    cropSignals.moireScore >= 0.76 ||
    (cropSignals.moireScore >= 0.54 &&
      (cropSignals.screenCastScore >= 0.45 || cropSignals.specularScore >= 0.5)) ||
    combinedSpoofRisk >= 0.68;

  if (spoofDetected && reason === 'NONE') {
    if (bezelScore >= cropSignals.moireScore) {
      reason = 'PHONE_BEZEL';
    } else if (cropSignals.moireScore >= cropSignals.screenCastScore) {
      reason = 'MOIRE_GRID';
    } else {
      reason = 'SCREEN_BACKLIGHT_CAST';
    }
  }

  return {
    spoofDetected,
    moireScore: cropSignals.moireScore,
    bezelScore,
    screenCastScore: cropSignals.screenCastScore,
    specularScore: cropSignals.specularScore,
    combinedSpoofRisk,
    reason,
  };
}
