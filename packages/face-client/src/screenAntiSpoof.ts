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
 */
export function analyzeFaceCropScreenSignals(image: SimpleImageBuffer | null | undefined): {
  moireScore: number;
  specularScore: number;
  screenCastScore: number;
} {
  if (!image || image.width < 24 || image.height < 24 || image.data.length < image.width * image.height * 4) {
    return { moireScore: 0, specularScore: 0, screenCastScore: 0 };
  }

  const { width, height, data } = image;
  // Forehead + malar cheek skin patches where real human skin has smooth local gradients
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
      y1: Math.min(height - 3, Math.floor(height * 0.70)),
    },
    {
      x0: Math.max(2, Math.floor(width * 0.66)),
      x1: Math.min(width - 3, Math.floor(width * 0.84)),
      y0: Math.max(2, Math.floor(height * 0.52)),
      y1: Math.min(height - 3, Math.floor(height * 0.70)),
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
        if (luma >= 238 && maxC - minC <= 22) {
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

        // Opposite consecutive 1-px finite differences indicate high-frequency pixel grid aliasing
        if ((d1x * d2x < -12) || (d1y * d2y < -12)) {
          alternatingHighFreqCount += 1;
        }

        const rgL = rgDiffAt(data, width, x - 1, y);
        const rgC = r - g;
        const rgR = rgDiffAt(data, width, x + 1, y);
        if ((rgC - rgL) * (rgR - rgC) < -10) {
          subpixelJitterCount += 1;
        }

        const lap1 = Math.abs(lLeft - 2 * luma + lRight) + Math.abs(lUp - 2 * luma + lDown);
        const smooth2 = Math.abs(lRight - lLeft) + Math.abs(lDown - lUp);
        highFreqLapSum += lap1;
        smoothGradSum += smooth2 + 1.5;
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

  // Real camera noise sits ~0.04-0.14 altRatio; moire / subpixel grid spikes >= 0.28+
  const moireFromAlt = clamp01((altRatio - 0.17) / 0.22);
  const moireFromSubpixel = clamp01((subpixelRatio - 0.14) / 0.22);
  const moireFromFreq = clamp01((freqRatio - 1.28) / 0.9);
  const moireScore = clamp01(moireFromAlt * 0.48 + moireFromSubpixel * 0.34 + moireFromFreq * 0.28);

  const specRatio = specularCount / pixelCount;
  const specularScore = clamp01((specRatio - 0.035) / 0.12);

  const meanR = sumR / pixelCount;
  const meanG = sumG / pixelCount;
  const meanB = sumB / pixelCount;
  const blueToRedRatio = meanB / Math.max(12, meanR);
  // Living human skin exhibits hemoglobin R >= B; digital LCD/OLED screens shift cool blue-white
  const coolBlueCast = clamp01((blueToRedRatio - 1.03) / 0.22);
  const flatGreenCast = meanB > meanG * 1.06 ? clamp01((meanB / Math.max(12, meanG) - 1.04) / 0.18) : 0;
  const screenCastScore = clamp01(coolBlueCast * 0.72 + flatGreenCast * 0.28);

  return {
    moireScore: Number(moireScore.toFixed(3)),
    specularScore: Number(specularScore.toFixed(3)),
    screenCastScore: Number(screenCastScore.toFixed(3)),
  };
}

/**
 * Inspects uncropped camera frame around the detected face bounding box for
 * straight rectangular smartphone/tablet bezels and abrupt screen-vs-room luminance steps.
 */
export function analyzeContextBezelAndStep(
  contextFrame: SimpleImageBuffer | null | undefined,
  faceBox?: NormalizedFaceBox | null,
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

  const yStart = Math.max(3, Math.floor((by - bh * 0.18) * height));
  const yEnd = Math.min(height - 4, Math.floor((by + bh * 1.15) * height));
  const xStart = Math.max(3, Math.floor((bx - bw * 0.25) * width));
  const xEnd = Math.min(width - 4, Math.floor((bx + bw * 1.25) * width));

  const leftMinX = Math.max(2, Math.floor((bx - bw * 0.34) * width));
  const leftMaxX = Math.max(leftMinX + 2, Math.floor((bx + bw * 0.04) * width));
  const rightMinX = Math.min(width - 4, Math.floor((bx + bw * 0.96) * width));
  const rightMaxX = Math.min(width - 3, Math.floor((bx + bw * 1.34) * width));

  let bestLeftEdge = 0;
  let bestLeftX = -1;
  const rowSpan = Math.max(1, yEnd - yStart);
  for (let x = leftMinX; x <= leftMaxX; x += 1) {
    let strongRows = 0;
    for (let y = yStart; y <= yEnd; y += 2) {
      const gx = Math.abs(lumaAt(data, width, x + 1, y) - lumaAt(data, width, x - 1, y));
      if (gx >= 18) strongRows += 1;
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
      if (gx >= 18) strongRows += 1;
    }
    const frac = strongRows / Math.max(1, Math.ceil(rowSpan / 2));
    if (frac > bestRightEdge) {
      bestRightEdge = frac;
      bestRightX = x;
    }
  }

  const topMinY = Math.max(2, Math.floor((by - bh * 0.36) * height));
  const topMaxY = Math.max(topMinY + 2, Math.floor((by + bh * 0.06) * height));
  const botMinY = Math.min(height - 4, Math.floor((by + bh * 0.94) * height));
  const botMaxY = Math.min(height - 3, Math.floor((by + bh * 1.34) * height));

  let bestTopEdge = 0;
  const colSpan = Math.max(1, xEnd - xStart);
  for (let y = topMinY; y <= topMaxY; y += 1) {
    let strongCols = 0;
    for (let x = xStart; x <= xEnd; x += 2) {
      const gy = Math.abs(lumaAt(data, width, x, y + 1) - lumaAt(data, width, x, y - 1));
      if (gy >= 18) strongCols += 1;
    }
    const frac = strongCols / Math.max(1, Math.ceil(colSpan / 2));
    if (frac > bestTopEdge) bestTopEdge = frac;
  }

  let bestBottomEdge = 0;
  for (let y = botMinY; y <= botMaxY; y += 1) {
    let strongCols = 0;
    for (let x = xStart; x <= xEnd; x += 2) {
      const gy = Math.abs(lumaAt(data, width, x, y + 1) - lumaAt(data, width, x, y - 1));
      if (gy >= 18) strongCols += 1;
    }
    const frac = strongCols / Math.max(1, Math.ceil(colSpan / 2));
    if (frac > bestBottomEdge) bestBottomEdge = frac;
  }

  // Require multiple straight orthogonal edges simultaneously (rectangles have both vertical & horizontal straight edges)
  const verticalPair = (bestLeftEdge + bestRightEdge) / 2;
  const horizontalBest = Math.max(bestTopEdge, bestBottomEdge);
  const hasRectangularFrame =
    bestLeftEdge >= 0.45 && bestRightEdge >= 0.45 && horizontalBest >= 0.42;

  let luminanceStepBoost = 0;
  if (bestLeftX > 3 && bestRightX > bestLeftX + 10 && bestRightX < width - 4) {
    let insideSum = 0;
    let insideN = 0;
    let outsideSum = 0;
    let outsideN = 0;
    for (let y = yStart; y <= yEnd; y += 3) {
      insideSum += lumaAt(data, width, Math.min(width - 2, bestLeftX + 3), y);
      insideSum += lumaAt(data, width, Math.max(1, bestRightX - 3), y);
      insideN += 2;
      outsideSum += lumaAt(data, width, Math.max(0, bestLeftX - 4), y);
      outsideSum += lumaAt(data, width, Math.min(width - 1, bestRightX + 4), y);
      outsideN += 2;
    }
    if (insideN > 0 && outsideN > 0) {
      const diff = Math.abs(insideSum / insideN - outsideSum / outsideN);
      luminanceStepBoost = clamp01((diff - 22) / 55);
    }
  }

  const rawRectScore =
    verticalPair * 0.46 +
    horizontalBest * 0.34 +
    luminanceStepBoost * 0.28;
  const bezelScore = hasRectangularFrame
    ? clamp01(rawRectScore * 1.15)
    : clamp01(rawRectScore * 0.55);

  return Number(bezelScore.toFixed(3));
}

/**
 * Combines aligned face-crop subpixel/glare/color-cast metrics and full-context rectangular
 * phone bezel inspection into an anti-spoof presentation attack score.
 */
export function analyzeScreenAndBezelArtifacts(
  input: AnalyzeScreenAntiSpoofInput,
): ScreenAntiSpoofReport {
  const cropSignals = analyzeFaceCropScreenSignals(input.faceCrop);
  const bezelScore = analyzeContextBezelAndStep(input.contextFrame, input.faceBox);

  const combinedSpoofRisk = Number(
    clamp01(
      bezelScore * 0.52 +
        cropSignals.moireScore * 0.44 +
        cropSignals.screenCastScore * 0.24 +
        cropSignals.specularScore * 0.22,
    ).toFixed(3),
  );

  let reason: ScreenSpoofReason = 'NONE';
  if (bezelScore >= 0.68) {
    reason = 'PHONE_BEZEL';
  } else if (cropSignals.moireScore >= 0.66) {
    reason = 'MOIRE_GRID';
  } else if (cropSignals.screenCastScore >= 0.62 && cropSignals.moireScore >= 0.35) {
    reason = 'SCREEN_BACKLIGHT_CAST';
  } else if (cropSignals.specularScore >= 0.68 && (cropSignals.moireScore >= 0.35 || bezelScore >= 0.4)) {
    reason = 'GLASS_SPECULAR';
  }

  const spoofDetected =
    bezelScore >= 0.68 ||
    cropSignals.moireScore >= 0.72 ||
    (cropSignals.moireScore >= 0.48 &&
      (cropSignals.screenCastScore >= 0.42 || cropSignals.specularScore >= 0.45)) ||
    combinedSpoofRisk >= 0.62;

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

