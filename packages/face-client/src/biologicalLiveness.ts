import type { SimpleImageBuffer } from '@drtechcares/smart-biometric-core';
import type { Landmark3D } from './embed';
import {
  analyzeScreenAndBezelArtifacts,
  type NormalizedFaceBox,
  type ScreenAntiSpoofReport,
} from './screenAntiSpoof';

export interface BlendshapeCategory {
  categoryName: string;
  score: number;
}

export interface BiologicalUpdateInput {
  landmarks3D: readonly Landmark3D[];
  blendshapes?: readonly BlendshapeCategory[] | null;
  earLeft: number;
  earRight: number;
  faceCrop?: SimpleImageBuffer | null;
  contextFrame?: SimpleImageBuffer | null;
  faceBox?: NormalizedFaceBox | null;
  timestampMs: number;
}

export type BiologicalSpoofReason =
  | 'NONE'
  | 'SCREEN_DETECTED'
  | 'PLANAR_RIGID_MOTION'
  | 'STATIC_FRAME_INJECTION';

export interface BiologicalLivenessReport {
  biologicalVerified: boolean;
  blinkVerified: boolean;
  blinkCount: number;
  eyesClosedNow: boolean;
  baselineEar: number;
  blendshapeBlinkMax: number;
  nonRigidScore: number;
  nonRigidMotion: number;
  globalMotion: number;
  planarSpoofDetected: boolean;
  screenSpoofDetected: boolean;
  spoofDetected: boolean;
  spoofReason: BiologicalSpoofReason;
  rppgScore: number;
  rppgPulseAlive: boolean;
  screenReport: ScreenAntiSpoofReport;
}

interface AnatomicalState {
  iod: number;
  centerX: number;
  centerY: number;
  rollRad: number;
  leftEyeOpen: number;
  rightEyeOpen: number;
  mouthOpen: number;
  browLiftL: number;
  browLiftR: number;
}

interface RppgSample {
  tMs: number;
  chrom: number;
  luma: number;
}

const DEFAULT_SCREEN_REPORT: ScreenAntiSpoofReport = {
  spoofDetected: false,
  moireScore: 0,
  bezelScore: 0,
  screenCastScore: 0,
  specularScore: 0,
  combinedSpoofRisk: 0,
  reason: 'NONE',
};

function dist2D(a: Landmark3D, b: Landmark3D): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function clamp01(v: number): number {
  if (!Number.isFinite(v)) return 0;
  return Math.max(0, Math.min(1, v));
}

function readBlendshapeScore(
  categories: readonly BlendshapeCategory[] | null | undefined,
  name: string,
): number {
  if (!categories || categories.length === 0) return 0;
  for (const c of categories) {
    if (c.categoryName === name) return c.score;
  }
  return 0;
}

function extractAnatomicalState(landmarks: readonly Landmark3D[]): AnatomicalState | null {
  if (landmarks.length < 390) return null;
  const p33 = landmarks[33];
  const p263 = landmarks[263];
  const p159 = landmarks[159];
  const p145 = landmarks[145];
  const p386 = landmarks[386];
  const p374 = landmarks[374];
  const p13 = landmarks[13];
  const p14 = landmarks[14];
  const p61 = landmarks[61];
  const p291 = landmarks[291];
  const p105 = landmarks[105];
  const p334 = landmarks[334];
  const nose = landmarks[1];
  if (
    !p33 ||
    !p263 ||
    !p159 ||
    !p145 ||
    !p386 ||
    !p374 ||
    !p13 ||
    !p14 ||
    !p61 ||
    !p291 ||
    !p105 ||
    !p334 ||
    !nose
  ) {
    return null;
  }
  const iod = Math.max(1e-4, dist2D(p33, p263));
  const mouthW = Math.max(1e-4, dist2D(p61, p291));
  return {
    iod,
    centerX: (p33.x + p263.x + nose.x) / 3,
    centerY: (p33.y + p263.y + nose.y) / 3,
    rollRad: Math.atan2(p263.y - p33.y, p263.x - p33.x),
    leftEyeOpen: dist2D(p159, p145) / iod,
    rightEyeOpen: dist2D(p386, p374) / iod,
    mouthOpen: dist2D(p13, p14) / mouthW,
    browLiftL: (p159.y - p105.y) / iod,
    browLiftR: (p386.y - p334.y) / iod,
  };
}

/**
 * Extracts normalized green-red chrominance (CHROM) & skin luminance from aligned 112x112
 * forehead and cheek skin patches for contactless rPPG biological pulse tracking.
 */
function sampleSkinChrominance(
  faceCrop: SimpleImageBuffer | null | undefined,
): { chrom: number; luma: number; warmSkinRatio: number } | null {
  if (!faceCrop || faceCrop.width < 24 || faceCrop.height < 24 || faceCrop.data.length < faceCrop.width * faceCrop.height * 4) {
    return null;
  }
  const { width, height, data } = faceCrop;
  const rois = [
    {
      x0: Math.floor(width * 0.3),
      x1: Math.floor(width * 0.7),
      y0: Math.floor(height * 0.12),
      y1: Math.floor(height * 0.25),
    },
    {
      x0: Math.floor(width * 0.18),
      x1: Math.floor(width * 0.34),
      y0: Math.floor(height * 0.54),
      y1: Math.floor(height * 0.68),
    },
    {
      x0: Math.floor(width * 0.66),
      x1: Math.floor(width * 0.82),
      y0: Math.floor(height * 0.54),
      y1: Math.floor(height * 0.68),
    },
  ];
  let sumR = 0;
  let sumG = 0;
  let sumB = 0;
  let n = 0;
  for (const r of rois) {
    for (let y = r.y0; y <= r.y1; y += 2) {
      for (let x = r.x0; x <= r.x1; x += 2) {
        const p = (y * width + x) * 4;
        sumR += data[p];
        sumG += data[p + 1];
        sumB += data[p + 2];
        n += 1;
      }
    }
  }
  if (n === 0) return null;
  const meanR = sumR / n;
  const meanG = sumG / n;
  const meanB = sumB / n;
  const luma = 0.299 * meanR + 0.587 * meanG + 0.114 * meanB;
  if (luma < 12) return null;

  // Normalized De Haan & Jeanne CHROM signal (Xs = 3R - 2G normalized by DC)
  const rn = meanR / Math.max(20, luma);
  const gn = meanG / Math.max(20, luma);
  const chrom = 3 * rn - 2 * gn;
  const warmSkinRatio = meanR / Math.max(12, meanB);
  return { chrom, luma, warmSkinRatio };
}

/**
 * Stateful multi-factor biological liveness & presentation-attack detector.
 * Combines:
 * 1. Physiological eye-blink cycle (neural blendshapes + adaptive baseline EAR, supports unilateral blink)
 * 2. 2D planar rigid motion vs non-rigid facial biological articulation
 * 3. Pixel-level screen moire / rectangular bezel / glass glare inspection
 * 4. Contactless skin hemoglobin CHROM rPPG variance & frozen-stream detection
 */
export class BiologicalLivenessTracker {
  private readonly blinkExpiryMs: number;
  private baselineEar = 0.29;
  private hasBaseline = false;
  private eyeClosedSinceMs: number | null = null;
  private blinkCount = 0;
  private lastBlinkAtMs: number | null = null;

  private prevAnatomy: AnatomicalState | null = null;
  private cumulativeNonRigid = 0;
  private rigidMovingStreak = 0;
  private planarSpoofDetected = false;

  private screenSpoofStreak = 0;
  private screenSpoofLatchedUntilMs = 0;
  private latestScreenReport: ScreenAntiSpoofReport = DEFAULT_SCREEN_REPORT;

  private rppgSamples: RppgSample[] = [];
  private staticInjectionDetected = false;
  private latestRppgScore = 0;
  private rppgPulseAlive = false;

  constructor(options?: { blinkExpiryMs?: number }) {
    this.blinkExpiryMs = options?.blinkExpiryMs ?? 15_000;
  }

  reset(): void {
    this.baselineEar = 0.29;
    this.hasBaseline = false;
    this.eyeClosedSinceMs = null;
    this.blinkCount = 0;
    this.lastBlinkAtMs = null;
    this.prevAnatomy = null;
    this.cumulativeNonRigid = 0;
    this.rigidMovingStreak = 0;
    this.planarSpoofDetected = false;
    this.screenSpoofStreak = 0;
    this.screenSpoofLatchedUntilMs = 0;
    this.latestScreenReport = DEFAULT_SCREEN_REPORT;
    this.rppgSamples = [];
    this.staticInjectionDetected = false;
    this.latestRppgScore = 0;
    this.rppgPulseAlive = false;
  }

  update(input: BiologicalUpdateInput): BiologicalLivenessReport {
    const now = input.timestampMs;

    // 1. Update Eye Blink Detection (supports unilateral blink for stroke / Bell's palsy safety)
    const blinkL = readBlendshapeScore(input.blendshapes, 'eyeBlinkLeft');
    const blinkR = readBlendshapeScore(input.blendshapes, 'eyeBlinkRight');
    const blendshapeBlinkMax = Math.max(blinkL, blinkR);

    const validEarL = Number.isFinite(input.earLeft) ? input.earLeft : 0.3;
    const validEarR = Number.isFinite(input.earRight) ? input.earRight : 0.3;
    const minEar = Math.min(validEarL, validEarR);
    const meanEar = (validEarL + validEarR) / 2;

    if (!this.hasBaseline && meanEar >= 0.21) {
      this.baselineEar = meanEar;
      this.hasBaseline = true;
    }

    const isEyeClosedByBlendshape = blendshapeBlinkMax >= 0.38;
    const isEyeClosedByEar =
      minEar < 0.205 || (this.hasBaseline && this.baselineEar >= 0.23 && this.baselineEar - minEar >= 0.052);
    const eyesClosedNow = isEyeClosedByBlendshape || isEyeClosedByEar;

    const isEyeReopened =
      blendshapeBlinkMax <= 0.24 &&
      meanEar >= Math.min(0.225, Math.max(0.205, this.baselineEar * 0.87));

    if (!eyesClosedNow && this.eyeClosedSinceMs === null && meanEar >= 0.215) {
      // Track slow resting baseline while clearly open
      this.baselineEar = this.baselineEar * 0.9 + meanEar * 0.1;
    }

    if (eyesClosedNow) {
      if (this.eyeClosedSinceMs === null) {
        this.eyeClosedSinceMs = now;
      }
    } else if (this.eyeClosedSinceMs !== null && isEyeReopened) {
      const closedDurationMs = now - this.eyeClosedSinceMs;
      if (closedDurationMs >= 30 && closedDurationMs <= 1800) {
        this.blinkCount += 1;
        this.lastBlinkAtMs = now;
        this.rigidMovingStreak = 0;
        this.planarSpoofDetected = false;
      }
      this.eyeClosedSinceMs = null;
    } else if (this.eyeClosedSinceMs !== null && now - this.eyeClosedSinceMs > 2200) {
      this.eyeClosedSinceMs = null;
    }

    // 2. Non-rigid biological deformation vs 2D rigid planar motion
    let globalMotion = 0;
    let nonRigidMotion = 0;
    const anatomy = extractAnatomicalState(input.landmarks3D);
    if (anatomy && this.prevAnatomy) {
      globalMotion =
        Math.hypot(anatomy.centerX - this.prevAnatomy.centerX, anatomy.centerY - this.prevAnatomy.centerY) +
        Math.abs(anatomy.iod - this.prevAnatomy.iod) * 0.8 +
        Math.abs(anatomy.rollRad - this.prevAnatomy.rollRad) * 0.08;

      nonRigidMotion =
        Math.abs(anatomy.leftEyeOpen - this.prevAnatomy.leftEyeOpen) +
        Math.abs(anatomy.rightEyeOpen - this.prevAnatomy.rightEyeOpen) +
        Math.abs(anatomy.mouthOpen - this.prevAnatomy.mouthOpen) * 0.7 +
        Math.abs(anatomy.browLiftL - this.prevAnatomy.browLiftL) * 0.6 +
        Math.abs(anatomy.browLiftR - this.prevAnatomy.browLiftR) * 0.6;

      this.cumulativeNonRigid = Math.min(1, this.cumulativeNonRigid * 0.92 + nonRigidMotion * 2.5);

      if (globalMotion >= 0.0045 && nonRigidMotion < 0.004 && this.blinkCount === 0) {
        this.rigidMovingStreak += 1;
      } else if (nonRigidMotion >= 0.009) {
        this.rigidMovingStreak = Math.max(0, this.rigidMovingStreak - 2);
      }

      if (this.rigidMovingStreak >= 12 && this.blinkCount === 0) {
        this.planarSpoofDetected = true;
      }
    }
    this.prevAnatomy = anatomy;

    // 3. Pixel-level Screen Bezel / Moire / Specular Anti-Spoof
    if (input.faceCrop || input.contextFrame) {
      const screenReport = analyzeScreenAndBezelArtifacts({
        faceCrop: input.faceCrop,
        contextFrame: input.contextFrame,
        faceBox: input.faceBox,
      });
      this.latestScreenReport = screenReport;
      if (screenReport.spoofDetected) {
        this.screenSpoofStreak += 1;
        // Require 2 consecutive screen-spoof detections OR high confidence bezel/moire to reject cleanly
        if (
          this.screenSpoofStreak >= 2 ||
          screenReport.bezelScore >= 0.72 ||
          screenReport.combinedSpoofRisk >= 0.72
        ) {
          this.screenSpoofLatchedUntilMs = Math.max(this.screenSpoofLatchedUntilMs, now + 1800);
        }
      } else {
        this.screenSpoofStreak = Math.max(0, this.screenSpoofStreak - 1);
      }
    }

    // 4. Micro-vascular CHROM rPPG & Frozen-Frame Injection Guard
    const rppgSample = sampleSkinChrominance(input.faceCrop);
    if (rppgSample) {
      this.rppgSamples.push({ tMs: now, chrom: rppgSample.chrom, luma: rppgSample.luma });
      if (this.rppgSamples.length > 36) {
        this.rppgSamples.shift();
      }
      if (this.rppgSamples.length >= 12) {
        const meanChrom =
          this.rppgSamples.reduce((acc, s) => acc + s.chrom, 0) / this.rppgSamples.length;
        const meanLuma =
          this.rppgSamples.reduce((acc, s) => acc + s.luma, 0) / this.rppgSamples.length;
        let chromVar = 0;
        let lumaVar = 0;
        let signChanges = 0;
        let prevDelta = 0;
        for (let i = 0; i < this.rppgSamples.length; i += 1) {
          const dc = this.rppgSamples[i].chrom - meanChrom;
          const dl = this.rppgSamples[i].luma - meanLuma;
          chromVar += dc * dc;
          lumaVar += dl * dl;
          if (i > 0) {
            const diff = this.rppgSamples[i].chrom - this.rppgSamples[i - 1].chrom;
            if (diff * prevDelta < 0) signChanges += 1;
            prevDelta = diff;
          }
        }
        const chromStd = Math.sqrt(chromVar / this.rppgSamples.length);
        const lumaStd = Math.sqrt(lumaVar / this.rppgSamples.length);

        if (chromStd < 1e-5 && lumaStd < 1e-4) {
          this.staticInjectionDetected = true;
        } else {
          this.staticInjectionDetected = false;
          const acScore = clamp01(chromStd / 0.028) * clamp01(1 - Math.max(0, chromStd - 0.12) / 0.15);
          const oscillationScore = clamp01(signChanges / (this.rppgSamples.length * 0.45));
          const warmthBoost = clamp01((rppgSample.warmSkinRatio - 0.92) / 0.35);
          this.latestRppgScore = Number(
            clamp01(acScore * 0.55 + oscillationScore * 0.25 + warmthBoost * 0.25).toFixed(3),
          );
          this.rppgPulseAlive =
            this.latestRppgScore >= 0.35 && rppgSample.warmSkinRatio >= 0.96;
        }
      }
    }

    const screenSpoofDetected = now <= this.screenSpoofLatchedUntilMs;
    const spoofDetected =
      screenSpoofDetected || this.planarSpoofDetected || this.staticInjectionDetected;

    let spoofReason: BiologicalSpoofReason = 'NONE';
    if (screenSpoofDetected) {
      spoofReason = 'SCREEN_DETECTED';
    } else if (this.planarSpoofDetected) {
      spoofReason = 'PLANAR_RIGID_MOTION';
    } else if (this.staticInjectionDetected) {
      spoofReason = 'STATIC_FRAME_INJECTION';
    }

    const blinkVerified =
      this.blinkCount >= 1 &&
      this.lastBlinkAtMs !== null &&
      now - this.lastBlinkAtMs <= this.blinkExpiryMs;

    const biologicalVerified = blinkVerified && !spoofDetected;

    return {
      biologicalVerified,
      blinkVerified,
      blinkCount: this.blinkCount,
      eyesClosedNow,
      baselineEar: Number(this.baselineEar.toFixed(3)),
      blendshapeBlinkMax: Number(blendshapeBlinkMax.toFixed(3)),
      nonRigidScore: Number(this.cumulativeNonRigid.toFixed(3)),
      nonRigidMotion: Number(nonRigidMotion.toFixed(5)),
      globalMotion: Number(globalMotion.toFixed(5)),
      planarSpoofDetected: this.planarSpoofDetected,
      screenSpoofDetected,
      spoofDetected,
      spoofReason,
      rppgScore: this.latestRppgScore,
      rppgPulseAlive: this.rppgPulseAlive,
      screenReport: this.latestScreenReport,
    };
  }
}

