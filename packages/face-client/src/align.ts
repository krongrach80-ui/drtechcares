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

/**
 * Tighter inner-face 5-point reference (eyebrows to chin, cheekbone to cheekbone inside 112x112)
 * used to minimize background/hair/collar intrusion on tight crop inspection.
 */
export const INNER_FACE_REFERENCE_5PT: readonly Point2D[] = [
  { x: 34.0, y: 40.0 },
  { x: 78.0, y: 40.0 },
  { x: 56.0, y: 66.0 },
  { x: 39.0, y: 89.0 },
  { x: 73.0, y: 89.0 },
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
 * Extracts left & right eye center coordinates from MediaPipe 468/478 or 5-point landmarks.
 */
export function resolveEyeCentersFromLandmarks(
  landmarksNormalized: readonly Point2D[],
): { leftEye: Point2D; rightEye: Point2D } | null {
  if (landmarksNormalized.length >= 468) {
    const leftEye = landmarksNormalized[468] ?? {
      x: (landmarksNormalized[33].x + landmarksNormalized[133].x) / 2,
      y: (landmarksNormalized[33].y + landmarksNormalized[133].y) / 2,
    };
    const rightEye = landmarksNormalized[473] ?? {
      x: (landmarksNormalized[362].x + landmarksNormalized[263].x) / 2,
      y: (landmarksNormalized[362].y + landmarksNormalized[263].y) / 2,
    };
    return { leftEye, rightEye };
  }
  if (landmarksNormalized.length >= 2) {
    return { leftEye: landmarksNormalized[0], rightEye: landmarksNormalized[1] };
  }
  return null;
}

/**
 * Computes head roll angle (degrees) in pixel aspect coordinates from eye centers.
 */
export function computeEyeRollAngleDeg(
  landmarksNormalized: readonly Point2D[],
  widthPx = 1,
  heightPx = 1,
): number {
  const eyes = resolveEyeCentersFromLandmarks(landmarksNormalized);
  if (!eyes) return 0;
  const dx = (eyes.rightEye.x - eyes.leftEye.x) * Math.max(1, widthPx);
  const dy = (eyes.rightEye.y - eyes.leftEye.y) * Math.max(1, heightPx);
  if (Math.hypot(dx, dy) < 1e-6) return 0;
  return (Math.atan2(dy, dx) * 180) / Math.PI;
}

/**
 * Rotates normalized [0..1] landmarks in pixel-aspect space around a pivot by angleRad.
 */
export function rotateNormalizedLandmarksForRoll<T extends Point2D>(
  landmarks: readonly T[],
  pivotNormX: number,
  pivotNormY: number,
  angleRad: number,
  widthPx = 1,
  heightPx = 1,
): T[] {
  const w = Math.max(1, widthPx);
  const h = Math.max(1, heightPx);
  const pxCenter = pivotNormX * w;
  const pyCenter = pivotNormY * h;
  const cos = Math.cos(angleRad);
  const sin = Math.sin(angleRad);
  return landmarks.map((pt) => {
    const rx = pt.x * w - pxCenter;
    const ry = pt.y * h - pyCenter;
    const nx = (rx * cos - ry * sin + pxCenter) / w;
    const ny = (rx * sin + ry * cos + pyCenter) / h;
    return { ...pt, x: nx, y: ny };
  });
}

/**
 * Aligns a video frame to canonical 112x112 ImageData using Roll Tilt Compensation (|roll| > 5°)
 * followed by Umeyama 5-point similarity warping.
 * For distant faces (height < 0.22), performs Stage-2 256x256 contrast-boosted Smart ROI
 * upscaling from full-resolution source before warping.
 */
export function alignFaceTo112(
  source: HTMLVideoElement | HTMLCanvasElement,
  landmarksNormalized: readonly { x: number; y: number }[],
  targetCanvas?: HTMLCanvasElement,
  roiWorkCanvas?: HTMLCanvasElement | null,
): ImageData | null {
  const width = source instanceof HTMLVideoElement ? source.videoWidth : source.width;
  const height = source instanceof HTMLVideoElement ? source.videoHeight : source.height;
  if (!width || !height) return null;

  let minY = 1;
  let maxY = 0;
  let minX = 1;
  let maxX = 0;
  for (const pt of landmarksNormalized) {
    if (pt.x < minX) minX = pt.x;
    if (pt.x > maxX) maxX = pt.x;
    if (pt.y < minY) minY = pt.y;
    if (pt.y > maxY) maxY = pt.y;
  }
  const faceHeightRatio = maxY - minY;

  // Stage-2 Two-Stage Smart ROI Super-Crop for distant faces (< 22% frame height)
  let warpSource: HTMLVideoElement | HTMLCanvasElement = source;
  let srcW = width;
  let srcH = height;
  let mappedLandmarks = landmarksNormalized;

  if (faceHeightRatio > 0.01 && faceHeightRatio < 0.22 && typeof document !== 'undefined') {
    const roiBox = computeHeadZoneRoiFromFaceBox(
      { x: minX, y: minY, width: Math.max(0.02, maxX - minX), height: faceHeightRatio },
      2.15,
    );
    const workCanvas = roiWorkCanvas ?? document.createElement('canvas');
    const upscaled = renderUpscaledRoi256(source, roiBox, workCanvas);
    if (upscaled) {
      warpSource = upscaled.canvas;
      srcW = upscaled.canvas.width;
      srcH = upscaled.canvas.height;
      mappedLandmarks = landmarksNormalized.map((pt) => ({
        x: Math.max(0, Math.min(1, (pt.x * width - upscaled.cropPx.sx) / upscaled.cropPx.sw)),
        y: Math.max(0, Math.min(1, (pt.y * height - upscaled.cropPx.sy) / upscaled.cropPx.sh)),
      }));
    }
  }

  // Explicit Roll Tilt Compensation when |rollAngle| > 2.5°: level eyes horizontally before 5-pt warp
  const rawRollDeg = computeEyeRollAngleDeg(mappedLandmarks, srcW, srcH);
  if (Math.abs(rawRollDeg) > 2.5 && typeof document !== 'undefined') {
    const eyes = resolveEyeCentersFromLandmarks(mappedLandmarks);
    if (eyes) {
      const eyeMidNormX = (eyes.leftEye.x + eyes.rightEye.x) / 2;
      const eyeMidNormY = (eyes.leftEye.y + eyes.rightEye.y) / 2;
      const rollCompRad = (-rawRollDeg * Math.PI) / 180;
      const levelCanvas = document.createElement('canvas');
      levelCanvas.width = srcW;
      levelCanvas.height = srcH;
      const lCtx = levelCanvas.getContext('2d', { willReadFrequently: true });
      if (lCtx) {
        const cx = eyeMidNormX * srcW;
        const cy = eyeMidNormY * srcH;
        lCtx.imageSmoothingEnabled = true;
        lCtx.imageSmoothingQuality = 'high';
        lCtx.translate(cx, cy);
        lCtx.rotate(rollCompRad);
        lCtx.translate(-cx, -cy);
        lCtx.drawImage(warpSource, 0, 0, srcW, srcH);
        lCtx.setTransform(1, 0, 0, 1, 0, 0);
        warpSource = levelCanvas;
        mappedLandmarks = rotateNormalizedLandmarksForRoll(
          mappedLandmarks,
          eyeMidNormX,
          eyeMidNormY,
          rollCompRad,
          srcW,
          srcH,
        );
      }
    }
  }

  // MediaPipe FaceLandmarker 468/478 indices:
  // Left eye center ~ 468 or avg(33,133), Right eye ~ 473 or avg(362,263), Nose tip 1, Mouth corners 61 & 291
  let src5: Point2D[];
  if (mappedLandmarks.length >= 468) {
    const lEye = mappedLandmarks[468] ?? {
      x: (mappedLandmarks[33].x + mappedLandmarks[133].x) / 2,
      y: (mappedLandmarks[33].y + mappedLandmarks[133].y) / 2,
    };
    const rEye = mappedLandmarks[473] ?? {
      x: (mappedLandmarks[362].x + mappedLandmarks[263].x) / 2,
      y: (mappedLandmarks[362].y + mappedLandmarks[263].y) / 2,
    };
    const nose = mappedLandmarks[1];
    const lMouth = mappedLandmarks[61];
    const rMouth = mappedLandmarks[291];
    src5 = [
      { x: lEye.x * srcW, y: lEye.y * srcH },
      { x: rEye.x * srcW, y: rEye.y * srcH },
      { x: nose.x * srcW, y: nose.y * srcH },
      { x: lMouth.x * srcW, y: lMouth.y * srcH },
      { x: rMouth.x * srcW, y: rMouth.y * srcH },
    ];
  } else if (mappedLandmarks.length >= 5) {
    src5 = mappedLandmarks.slice(0, 5).map((p) => ({ x: p.x * srcW, y: p.y * srcH }));
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

  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.setTransform(mat.a, mat.b, -mat.b, mat.a, mat.tx, mat.ty);
  ctx.drawImage(warpSource, 0, 0);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  return ctx.getImageData(0, 0, ARCFACE_INPUT_SIZE, ARCFACE_INPUT_SIZE);
}

/**
 * Tight Inner-Face 112x112 alignment helper focusing strictly on eyebrows-to-chin & cheekbones,
 * with neutral mid-gray elliptical border masking to suppress wall/ceiling/shirt interference.
 */
export function alignInnerFaceTo112(
  source: HTMLVideoElement | HTMLCanvasElement,
  landmarksNormalized: readonly { x: number; y: number }[],
  targetCanvas?: HTMLCanvasElement,
  roiWorkCanvas?: HTMLCanvasElement | null,
): ImageData | null {
  const aligned = alignFaceTo112(source, landmarksNormalized, targetCanvas, roiWorkCanvas);
  if (!aligned) return null;
  const { width, height, data } = aligned;
  const cx = width * 0.5;
  const cy = height * 0.56;
  const rx = width * 0.34;
  const ry = height * 0.40;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const ellip = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2;
      if (ellip > 1.0) {
        const idx = (y * width + x) * 4;
        data[idx] = 128;
        data[idx + 1] = 128;
        data[idx + 2] = 128;
      }
    }
  }
  return aligned;
}

export interface NormalizedRoiRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface RoiCropMetrics {
  sx: number;
  sy: number;
  sw: number;
  sh: number;
  srcW: number;
  srcH: number;
}

export function computeHeadZoneRoiFromFaceBox(
  box: NormalizedRoiRect,
  expandFactor: number = 2.2,
): NormalizedRoiRect {
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;
  const size = Math.max(0.14, Math.min(0.85, Math.max(box.width, box.height) * expandFactor));
  const x = Math.max(0, Math.min(1 - size, cx - size / 2));
  const y = Math.max(0, Math.min(1 - size, cy - size / 2));
  return { x, y, width: Math.min(size, 1 - x), height: Math.min(size, 1 - y) };
}

/**
 * Computes a high-probability square Head-Zone focal crop from MediaPipe Pose landmarks
 * so tiny faces at 2.0-4.5m can be upscaled to 256x256 before running detailed FaceLandmarker.
 */
export function computeHeadZoneRoiFromPose(
  poseLandmarks: readonly { x: number; y: number; visibility?: number }[] | null | undefined,
): NormalizedRoiRect {
  if (poseLandmarks && poseLandmarks.length >= 13) {
    const nose = poseLandmarks[0];
    const lEar = poseLandmarks[7];
    const rEar = poseLandmarks[8];
    const lShoulder = poseLandmarks[11];
    const rShoulder = poseLandmarks[12];
    const anchors = [nose, lEar, rEar].filter(
      (pt) => pt && (pt.visibility ?? 1) >= 0.2 && Number.isFinite(pt.x) && Number.isFinite(pt.y),
    );
    if (anchors.length > 0) {
      const cx = anchors.reduce((a, p) => a + p.x, 0) / anchors.length;
      const cy = anchors.reduce((a, p) => a + p.y, 0) / anchors.length;
      const earSpan =
        lEar && rEar && (lEar.visibility ?? 1) >= 0.2 && (rEar.visibility ?? 1) >= 0.2
          ? Math.hypot(rEar.x - lEar.x, rEar.y - lEar.y)
          : 0;
      const shoulderSpan =
        lShoulder && rShoulder && (lShoulder.visibility ?? 1) >= 0.2 && (rShoulder.visibility ?? 1) >= 0.2
          ? Math.hypot(rShoulder.x - lShoulder.x, rShoulder.y - lShoulder.y)
          : 0;
      const size = Math.max(0.18, Math.min(0.65, Math.max(earSpan * 3.4, shoulderSpan * 0.9, 0.22)));
      const x = Math.max(0, Math.min(1 - size, cx - size / 2));
      const y = Math.max(0, Math.min(1 - size, cy - size / 2));
      return { x, y, width: Math.min(size, 1 - x), height: Math.min(size, 1 - y) };
    }
  }
  return { x: 0.26, y: 0.05, width: 0.48, height: 0.56 };
}

/**
 * Upscales the selected head/face ROI from full-res video onto a 256x256 canvas with mild
 * brightness + contrast boost so distant eye/brow/jaw features lock reliably.
 */
export function renderUpscaledRoi256(
  source: HTMLVideoElement | HTMLCanvasElement,
  roiNorm: NormalizedRoiRect,
  targetCanvas?: HTMLCanvasElement | null,
  outSize: number = 256,
): { canvas: HTMLCanvasElement; cropPx: RoiCropMetrics } | null {
  const srcW = source instanceof HTMLVideoElement ? source.videoWidth : source.width;
  const srcH = source instanceof HTMLVideoElement ? source.videoHeight : source.height;
  if (!srcW || !srcH) return null;

  const sx = Math.max(0, Math.floor(roiNorm.x * srcW));
  const sy = Math.max(0, Math.floor(roiNorm.y * srcH));
  const sw = Math.max(24, Math.min(srcW - sx, Math.round(roiNorm.width * srcW)));
  const sh = Math.max(24, Math.min(srcH - sy, Math.round(roiNorm.height * srcH)));

  const canvas = targetCanvas ?? (typeof document !== 'undefined' ? document.createElement('canvas') : null);
  if (!canvas) return null;
  canvas.width = outSize;
  canvas.height = outSize;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return null;

  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  try {
    ctx.filter = 'contrast(1.16) brightness(1.06)';
  } catch {
    // Ignore filter in constrained contexts
  }
  ctx.drawImage(source, sx, sy, sw, sh, 0, 0, outSize, outSize);
  try {
    ctx.filter = 'none';
  } catch {
    // Ignore filter reset errors
  }

  return { canvas, cropPx: { sx, sy, sw, sh, srcW, srcH } };
}

/**
 * Projects landmarks detected on a zoomed ROI crop cleanly back to full-frame normalized coordinates.
 */
export function remapRoiLandmarksToFrame<T extends { x: number; y: number; z?: number }>(
  landmarks: readonly T[],
  cropPx: RoiCropMetrics,
): T[] {
  const scaleX = cropPx.sw / Math.max(1, cropPx.srcW);
  const scaleY = cropPx.sh / Math.max(1, cropPx.srcH);
  const offsetX = cropPx.sx / Math.max(1, cropPx.srcW);
  const offsetY = cropPx.sy / Math.max(1, cropPx.srcH);
  return landmarks.map((lm) => ({
    ...lm,
    x: offsetX + lm.x * scaleX,
    y: offsetY + lm.y * scaleY,
    z: lm.z !== undefined ? lm.z * scaleX : lm.z,
  }));
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


