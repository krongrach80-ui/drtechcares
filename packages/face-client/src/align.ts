export interface Point2D {
  x: number;
  y: number;
}

export const ARCFACE_INPUT_SIZE = 112;

/**
 * Standard InsightFace/ArcFace 5-point reference landmarks on a 112x112 canvas:
 * [leftEye, rightEye, noseTip, leftMouthCorner, rightMouthCorner]
 */
export const ARCFACE_REFERENCE_5PT: readonly Point2D[] = [
  { x: 38.2946, y: 51.6963 },
  { x: 73.5318, y: 51.5014 },
  { x: 56.0252, y: 71.7366 },
  { x: 41.5493, y: 92.3655 },
  { x: 70.7299, y: 92.2041 },
];

export interface SimilarityMatrix2D {
  a: number;
  b: number;
  tx: number;
  ty: number;
}

/**
 * Computes closed-form 2D Umeyama Similarity Transform (scale, rotation, translation)
 * mapping source 5-point landmarks (in pixel coordinates) to target 112x112 reference points.
 */
export function estimateUmeyamaSimilarity2D(
  srcPoints: readonly Point2D[],
  dstPoints: readonly Point2D[] = ARCFACE_REFERENCE_5PT,
): SimilarityMatrix2D {
  const n = Math.min(srcPoints.length, dstPoints.length);
  if (n < 2) {
    return { a: 1, b: 0, tx: 0, ty: 0 };
  }

  let srcMeanX = 0;
  let srcMeanY = 0;
  let dstMeanX = 0;
  let dstMeanY = 0;

  for (let i = 0; i < n; i += 1) {
    srcMeanX += srcPoints[i].x;
    srcMeanY += srcPoints[i].y;
    dstMeanX += dstPoints[i].x;
    dstMeanY += dstPoints[i].y;
  }
  srcMeanX /= n;
  srcMeanY /= n;
  dstMeanX /= n;
  dstMeanY /= n;

  let srcVar = 0;
  let numA = 0;
  let numB = 0;

  for (let i = 0; i < n; i += 1) {
    const sx = srcPoints[i].x - srcMeanX;
    const sy = srcPoints[i].y - srcMeanY;
    const dx = dstPoints[i].x - dstMeanX;
    const dy = dstPoints[i].y - dstMeanY;

    srcVar += sx * sx + sy * sy;
    numA += sx * dx + sy * dy;
    numB += sx * dy - sy * dx;
  }

  if (srcVar < 1e-8) {
    return { a: 1, b: 0, tx: dstMeanX - srcMeanX, ty: dstMeanY - srcMeanY };
  }

  const a = numA / srcVar;
  const b = numB / srcVar;
  const tx = dstMeanX - (a * srcMeanX - b * srcMeanY);
  const ty = dstMeanY - (b * srcMeanX + a * srcMeanY);

  return { a, b, tx, ty };
}

/**
 * Aligns a video frame to canonical 112x112 ImageData using Umeyama 5-point transform.
 */
export function alignFaceTo112(
  source: HTMLVideoElement | HTMLCanvasElement,
  landmarksNormalized: readonly { x: number; y: number }[],
  targetCanvas?: HTMLCanvasElement,
): ImageData | null {
  const width = source instanceof HTMLVideoElement ? source.videoWidth : source.width;
  const height = source instanceof HTMLVideoElement ? source.videoHeight : source.height;
  if (!width || !height) return null;

  // MediaPipe FaceLandmarker 468/478 indices:
  // Left eye center ~ 468 or avg(33,133), Right eye ~ 473 or avg(362,263), Nose tip 1, Mouth corners 61 & 291
  let src5: Point2D[];
  if (landmarksNormalized.length >= 468) {
    const lEye = landmarksNormalized[468] ?? {
      x: (landmarksNormalized[33].x + landmarksNormalized[133].x) / 2,
      y: (landmarksNormalized[33].y + landmarksNormalized[133].y) / 2,
    };
    const rEye = landmarksNormalized[473] ?? {
      x: (landmarksNormalized[362].x + landmarksNormalized[263].x) / 2,
      y: (landmarksNormalized[362].y + landmarksNormalized[263].y) / 2,
    };
    const nose = landmarksNormalized[1];
    const lMouth = landmarksNormalized[61];
    const rMouth = landmarksNormalized[291];
    src5 = [
      { x: lEye.x * width, y: lEye.y * height },
      { x: rEye.x * width, y: rEye.y * height },
      { x: nose.x * width, y: nose.y * height },
      { x: lMouth.x * width, y: lMouth.y * height },
      { x: rMouth.x * width, y: rMouth.y * height },
    ];
  } else if (landmarksNormalized.length >= 5) {
    src5 = landmarksNormalized.slice(0, 5).map((p) => ({ x: p.x * width, y: p.y * height }));
  } else {
    return null;
  }

  const mat = estimateUmeyamaSimilarity2D(src5, ARCFACE_REFERENCE_5PT);
  const canvas = targetCanvas ?? (typeof document !== 'undefined' ? document.createElement('canvas') : null);
  if (!canvas) return null;
  canvas.width = ARCFACE_INPUT_SIZE;
  canvas.height = ARCFACE_INPUT_SIZE;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return null;

  ctx.setTransform(mat.a, mat.b, -mat.b, mat.a, mat.tx, mat.ty);
  ctx.drawImage(source, 0, 0);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  return ctx.getImageData(0, 0, ARCFACE_INPUT_SIZE, ARCFACE_INPUT_SIZE);
}

/**
 * Captures a lightweight downscaled full camera frame (default width 160px)
 * so rectangular smartphone/tablet bezels and screen luminance boundaries around
 * the detected face box can be analyzed without blocking the UI thread.
 */
export function captureDownscaledContextFrame(
  source: HTMLVideoElement | HTMLCanvasElement,
  targetCanvas?: HTMLCanvasElement | null,
  maxWidth = 160,
): ImageData | null {
  const srcWidth = source instanceof HTMLVideoElement ? source.videoWidth : source.width;
  const srcHeight = source instanceof HTMLVideoElement ? source.videoHeight : source.height;
  if (!srcWidth || !srcHeight) return null;

  const targetW = Math.max(48, Math.min(srcWidth, maxWidth));
  const targetH = Math.max(48, Math.round((srcHeight / srcWidth) * targetW));
  const canvas = targetCanvas ?? (typeof document !== 'undefined' ? document.createElement('canvas') : null);
  if (!canvas) return null;

  canvas.width = targetW;
  canvas.height = targetH;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return null;

  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.drawImage(source, 0, 0, targetW, targetH);
  return ctx.getImageData(0, 0, targetW, targetH);
}


