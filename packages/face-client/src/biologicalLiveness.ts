import type { SimpleImageBuffer } from '@drtechcares/smart-biometric-core';
import type { Landmark3D } from './embed';
import { estimateCameraDistanceMeters } from './liveness';
import {
  analyzeScreenAndBezelArtifacts,
  type NormalizedFaceBox,
  type ScreenAntiSpoofReport,
} from './screenAntiSpoof';

export interface BlendshapeCategory {
  categoryName: string;
  score: number;
}

export type BiologicalLivenessMode = 'NEAR_RANGE' | 'LONG_RANGE';

export interface BiologicalUpdateInput {
  landmarks3D: readonly Landmark3D[];
  blendshapes?: readonly BlendshapeCategory[] | null;
  earLeft: number;
  earRight: number;
  yawDeg?: number;
  pitchDeg?: number;
  frameWidth?: number;
  frameHeight?: number;
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
  livenessPassed: boolean;
  livenessMode: BiologicalLivenessMode;
  estimatedDistanceM: number;
  centerShiftPx: number;
  longRangeMotionVerified: boolean;
  blinkVerified: boolean;
  blinkDetectedNow: boolean;
  headPoseVerified: boolean;
  lipMotionVerified: boolean;
  blinkCount: number;
  eyesClosedNow: boolean;
  openEyeBaseline: number;
  baselineEar: number;
  faceScale: number;
  deltaYawDeg: number;
  deltaPitchDeg: number;
  yawVariance: number;
  pitchVariance: number;
  observationDurationMs: number;
  blendshapeBlinkMax: number;
  nonRigidScore: number;
  nonRigidMotion: number;
  globalMotion: number;
  planarSpoofDetected: boolean;
  screenSpoofDetected: boolean;
  spoofDetected: boolean;
  spoofScore: number;
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
  yawDeg: number;
  pitchDeg: number;
}

interface PoseHistorySample {
  tMs: number;
  yawDeg: number;
  pitchDeg: number;
  mouthOpen: number;
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

function stdDev(values: readonly number[]): number {
  if (values.length < 2) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  let variance = 0;
  for (const v of values) {
    const d = v - mean;
    variance += d * d;
  }
  return Math.sqrt(variance / values.length);
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

function extractAnatomicalState(
  landmarks: readonly Landmark3D[],
  overrideYaw?: number,
  overridePitch?: number,
): AnatomicalState | null {
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

  // Derive stable head pose angles from facial geometry if not provided explicitly
  let yawDeg = overrideYaw ?? 0;
  let pitchDeg = overridePitch ?? 0;
  if (overrideYaw === undefined || overridePitch === undefined) {
    const lCheek = landmarks[234] ?? p33;
    const rCheek = landmarks[454] ?? p263;
    const forehead = landmarks[10] ?? p105;
    const chin = landmarks[152] ?? p14;
    const fw = Math.max(1e-4, dist2D(lCheek, rCheek));
    const fh = Math.max(1e-4, dist2D(forehead, chin));
    if (overrideYaw === undefined) {
      const midX = (lCheek.x + rCheek.x) / 2;
      const yaw2D = ((nose.x - midX) / fw) * 90;
      const yawZ = (Math.atan2((rCheek.z ?? 0) - (lCheek.z ?? 0), fw) * 180) / Math.PI;
      yawDeg = yaw2D * 0.6 + yawZ * 0.4;
    }
    if (overridePitch === undefined) {
      const neutralY = forehead.y + fh * 0.55;
      const pitch2D = ((nose.y - neutralY) / fh) * 90;
      const pitchZ = (Math.atan2((chin.z ?? 0) - (forehead.z ?? 0), fh) * 180) / Math.PI;
      pitchDeg = pitch2D * 0.55 + pitchZ * 0.45;
    }
  }

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
    yawDeg,
    pitchDeg,
  };
}

/**
 * Samples upper-eye strip mean brightness in the aligned 112x112 face crop to detect
 * rapid 100-400ms eyelid closure/opening intensity shifts at far camera distances (1.2-2.5m).
 */
function sampleEyePatchLuma(faceCrop: SimpleImageBuffer | null | undefined): number | null {
  if (!faceCrop || faceCrop.width < 48 || faceCrop.height < 48 || faceCrop.data.length < faceCrop.width * faceCrop.height * 4) {
    return null;
  }
  const { width, height, data } = faceCrop;
  const boxes = [
    {
      x0: Math.floor(width * 0.25),
      x1: Math.floor(width * 0.43),
      y0: Math.floor(height * 0.41),
      y1: Math.floor(height * 0.52),
    },
    {
      x0: Math.floor(width * 0.57),
      x1: Math.floor(width * 0.75),
      y0: Math.floor(height * 0.41),
      y1: Math.floor(height * 0.52),
    },
  ];
  let sum = 0;
  let n = 0;
  for (const b of boxes) {
    for (let y = b.y0; y <= b.y1; y += 2) {
      for (let x = b.x0; x <= b.x1; x += 2) {
        const p = (y * width + x) * 4;
        sum += 0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2];
        n += 1;
      }
    }
  }
  return n > 0 ? sum / n : null;
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

  const rn = meanR / Math.max(20, luma);
  const gn = meanG / Math.max(20, luma);
  const chrom = 3 * rn - 2 * gn;
  const warmSkinRatio = meanR / Math.max(12, meanB);
  return { chrom, luma, warmSkinRatio };
}

/**
 * Distance-Adaptive Multi-Cue Biological Liveness & Anti-Spoof Tracker.
 * Features:
 * 1. Distance-adaptive openEyeBaseline moving average (EAR < openEyeBaseline * ratio, plus eyelid pixel delta)
 * 2. Multi-cue biological proof: eye blink OR 3D head rotation (>2.5 deg Yaw/Pitch) OR lip articulation
 * 3. Immediate State Clear: any genuine biological cue clears spoofScore=0 & spoofDetected=false immediately
 * 4. 4.5-second (120+ frame) patience window before static-planar warning unless enclosed phone bezel confirmed
 */
export class BiologicalLivenessTracker {
  private readonly blinkExpiryMs: number;
  private readonly staticSpoofWindowMs: number;

  private firstSeenAtMs: number | null = null;
  private frameCount = 0;

  private openEyeBaseline = 0.28;
  private hasBaseline = false;
  private eyeClosedSinceMs: number | null = null;
  private blinkCount = 0;
  private lastBlinkAtMs: number | null = null;

  private eyePatchBaseline: number | null = null;
  private eyePatchPulseAtMs: number | null = null;

  private liveVerifiedUntilMs = 0;
  private headPoseVerified = false;
  private lipMotionVerified = false;
  private longRangeMotionVerified = false;

  private prevAnatomy: AnatomicalState | null = null;
  private poseHistory: (PoseHistorySample & { cxPx: number; cyPx: number })[] = [];
  private cumulativeNonRigid = 0;
  private rigidMovingStreak = 0;
  private planarSpoofDetected = false;

  private screenSpoofStreak = 0;
  private latestScreenReport: ScreenAntiSpoofReport = DEFAULT_SCREEN_REPORT;

  private rppgSamples: RppgSample[] = [];
  private staticInjectionDetected = false;
  private latestRppgScore = 0;
  private rppgPulseAlive = false;

  constructor(options?: { blinkExpiryMs?: number; staticSpoofWindowMs?: number }) {
    this.blinkExpiryMs = options?.blinkExpiryMs ?? 15_000;
    this.staticSpoofWindowMs = options?.staticSpoofWindowMs ?? 4_500;
  }

  reset(): void {
    this.firstSeenAtMs = null;
    this.frameCount = 0;
    this.openEyeBaseline = 0.28;
    this.hasBaseline = false;
    this.eyeClosedSinceMs = null;
    this.blinkCount = 0;
    this.lastBlinkAtMs = null;
    this.eyePatchBaseline = null;
    this.eyePatchPulseAtMs = null;
    this.liveVerifiedUntilMs = 0;
    this.headPoseVerified = false;
    this.lipMotionVerified = false;
    this.longRangeMotionVerified = false;
    this.prevAnatomy = null;
    this.poseHistory = [];
    this.cumulativeNonRigid = 0;
    this.rigidMovingStreak = 0;
    this.planarSpoofDetected = false;
    this.screenSpoofStreak = 0;
    this.latestScreenReport = DEFAULT_SCREEN_REPORT;
    this.rppgSamples = [];
    this.staticInjectionDetected = false;
    this.latestRppgScore = 0;
    this.rppgPulseAlive = false;
  }

  private clearPlanarSpoofOnly(now: number): void {
    this.planarSpoofDetected = false;
    this.staticInjectionDetected = false;
    this.rigidMovingStreak = 0;
    this.liveVerifiedUntilMs = Math.max(this.liveVerifiedUntilMs, now + this.blinkExpiryMs);
  }

  private clearSpoofStateImmediately(now: number): void {
    this.planarSpoofDetected = false;
    this.staticInjectionDetected = false;
    this.screenSpoofStreak = 0;
    this.rigidMovingStreak = 0;
    this.latestScreenReport = DEFAULT_SCREEN_REPORT;
    this.liveVerifiedUntilMs = Math.max(this.liveVerifiedUntilMs, now + this.blinkExpiryMs);
  }

  update(input: BiologicalUpdateInput): BiologicalLivenessReport {
    const now = input.timestampMs;
    if (this.firstSeenAtMs === null) {
      this.firstSeenAtMs = now;
    }
    this.frameCount += 1;
    const observationDurationMs = Math.max(0, now - this.firstSeenAtMs);

    const rawLandmarkSpan =
      input.landmarks3D[152] && input.landmarks3D[10]
        ? Math.abs(input.landmarks3D[152].y - input.landmarks3D[10].y)
        : 0;
    const landmarksFaceHeight = rawLandmarkSpan > 0.01 ? rawLandmarkSpan : 0.4;
    const faceScale = Number(
      clamp01(input.faceBox?.height ?? landmarksFaceHeight).toFixed(3),
    );
    const livenessMode: BiologicalLivenessMode = faceScale < 0.22 ? 'LONG_RANGE' : 'NEAR_RANGE';
    const estimatedDistanceM = estimateCameraDistanceMeters(faceScale);
    const isDistantFace = faceScale < 0.35;
    const nominalW = input.frameWidth && input.frameWidth >= 320 ? input.frameWidth : 1920;
    const nominalH = input.frameHeight && input.frameHeight >= 240 ? input.frameHeight : 1080;

    // 1. Distance-Adaptive Dynamic Eye Baseline & Blink Detection
    const blinkL = readBlendshapeScore(input.blendshapes, 'eyeBlinkLeft');
    const blinkR = readBlendshapeScore(input.blendshapes, 'eyeBlinkRight');
    const blendshapeBlinkMax = Math.max(blinkL, blinkR);

    const validEarL = Number.isFinite(input.earLeft) ? input.earLeft : 0.29;
    const validEarR = Number.isFinite(input.earRight) ? input.earRight : 0.29;
    const minEar = Math.min(validEarL, validEarR);
    const meanEar = (validEarL + validEarR) / 2;

    if (!this.hasBaseline && meanEar >= 0.19) {
      this.openEyeBaseline = meanEar;
      this.hasBaseline = true;
    }

    // Distant faces (< 0.35 height) allow gentler 0.80-0.83 relative drop; standard 0.78
    const earCloseRatio = faceScale < 0.24 ? 0.83 : isDistantFace ? 0.8 : 0.78;
    const earThreshold = this.openEyeBaseline * earCloseRatio;
    const blendshapeThreshold = isDistantFace ? 0.26 : 0.36;

    // Also track sudden 100-400ms eye-patch brightness jump (brighter closed upper skin vs dark iris)
    const eyeLuma = sampleEyePatchLuma(input.faceCrop);
    let eyePatchClosing = false;
    let eyePatchReopening = false;
    if (eyeLuma !== null) {
      if (this.eyePatchBaseline === null) {
        this.eyePatchBaseline = eyeLuma;
      } else {
        const diff = eyeLuma - this.eyePatchBaseline;
        if (Math.abs(diff) >= 9.5) {
          eyePatchClosing = true;
          this.eyePatchPulseAtMs = now;
        } else if (this.eyePatchPulseAtMs !== null && Math.abs(diff) <= 4.0) {
          const dt = now - this.eyePatchPulseAtMs;
          if (dt >= 70 && dt <= 550) {
            eyePatchReopening = true;
          }
          this.eyePatchPulseAtMs = null;
        } else if (Math.abs(diff) < 5.0) {
          this.eyePatchBaseline = this.eyePatchBaseline * 0.92 + eyeLuma * 0.08;
        }
      }
    }

    const isEyeClosedByBlendshape = blendshapeBlinkMax >= blendshapeThreshold;
    const isEyeClosedByEar =
      minEar < 0.195 ||
      (this.hasBaseline && (meanEar < earThreshold || minEar < earThreshold));
    const eyesClosedNow = isEyeClosedByBlendshape || isEyeClosedByEar || eyePatchClosing;

    const isEyeReopened =
      eyePatchReopening ||
      (blendshapeBlinkMax <= 0.24 &&
        meanEar >= Math.max(0.2, this.openEyeBaseline * 0.89));

    if (!eyesClosedNow && this.eyeClosedSinceMs === null && meanEar >= 0.195) {
      // Smooth moving average ONLY when eyes are open
      this.openEyeBaseline = this.openEyeBaseline * 0.9 + meanEar * 0.1;
    }

    let blinkDetectedNow = false;
    if (eyesClosedNow) {
      if (this.eyeClosedSinceMs === null) {
        this.eyeClosedSinceMs = now;
      }
    } else if (this.eyeClosedSinceMs !== null && isEyeReopened) {
      const closedDurationMs = now - this.eyeClosedSinceMs;
      if (closedDurationMs >= 30 && closedDurationMs <= 1800) {
        this.blinkCount += 1;
        this.lastBlinkAtMs = now;
        blinkDetectedNow = true;
        this.clearSpoofStateImmediately(now);
      }
      this.eyeClosedSinceMs = null;
    } else if (this.eyeClosedSinceMs !== null && now - this.eyeClosedSinceMs > 2200) {
      this.eyeClosedSinceMs = null;
    }

    // 2. Non-Rigid Biological Deformation & Multi-Cue 3D Head Pose / Lip Motion / Long-Range Motion
    let globalMotion = 0;
    let nonRigidMotion = 0;
    const anatomy = extractAnatomicalState(input.landmarks3D, input.yawDeg, input.pitchDeg);
    if (anatomy) {
      this.poseHistory.push({
        tMs: now,
        yawDeg: anatomy.yawDeg,
        pitchDeg: anatomy.pitchDeg,
        mouthOpen: anatomy.mouthOpen,
        cxPx: anatomy.centerX * nominalW,
        cyPx: anatomy.centerY * nominalH,
      });
      while (this.poseHistory.length > 0 && now - this.poseHistory[0].tMs > this.staticSpoofWindowMs) {
        this.poseHistory.shift();
      }
    }

    if (anatomy && this.prevAnatomy) {
      globalMotion =
        Math.hypot(anatomy.centerX - this.prevAnatomy.centerX, anatomy.centerY - this.prevAnatomy.centerY) +
        Math.abs(anatomy.iod - this.prevAnatomy.iod) * 0.8 +
        Math.abs(anatomy.rollRad - this.prevAnatomy.rollRad) * 0.08;

      nonRigidMotion =
        Math.abs(anatomy.leftEyeOpen - this.prevAnatomy.leftEyeOpen) +
        Math.abs(anatomy.rightEyeOpen - this.prevAnatomy.rightEyeOpen) +
        Math.abs(anatomy.mouthOpen - this.prevAnatomy.mouthOpen) * 0.75 +
        Math.abs(anatomy.browLiftL - this.prevAnatomy.browLiftL) * 0.6 +
        Math.abs(anatomy.browLiftR - this.prevAnatomy.browLiftR) * 0.6;

      this.cumulativeNonRigid = Math.min(1, this.cumulativeNonRigid * 0.92 + nonRigidMotion * 2.5);
    }
    this.prevAnatomy = anatomy;

    // Recent 2.5s window + 0.6s rapid long-range sway window
    let deltaYawDeg = 0;
    let deltaPitchDeg = 0;
    let deltaLip = 0;
    let centerShiftPx = 0;
    let fastDeltaYaw = 0;
    let fastDeltaPitch = 0;
    const recentSamples = this.poseHistory.filter((s) => now - s.tMs <= 2500);
    if (recentSamples.length >= 2) {
      const yaws = recentSamples.map((s) => s.yawDeg);
      const pitches = recentSamples.map((s) => s.pitchDeg);
      const lips = recentSamples.map((s) => s.mouthOpen);
      deltaYawDeg = Math.max(...yaws) - Math.min(...yaws);
      deltaPitchDeg = Math.max(...pitches) - Math.min(...pitches);
      deltaLip = Math.max(...lips) - Math.min(...lips);
    }
    const rapidSamples = this.poseHistory.filter((s) => now - s.tMs <= 700);
    if (rapidSamples.length >= 2) {
      const rCx = rapidSamples.map((s) => s.cxPx);
      const rCy = rapidSamples.map((s) => s.cyPx);
      const rYaw = rapidSamples.map((s) => s.yawDeg);
      const rPitch = rapidSamples.map((s) => s.pitchDeg);
      centerShiftPx = Math.hypot(Math.max(...rCx) - Math.min(...rCx), Math.max(...rCy) - Math.min(...rCy));
      fastDeltaYaw = Math.max(...rYaw) - Math.min(...rYaw);
      fastDeltaPitch = Math.max(...rPitch) - Math.min(...rPitch);
    }

    if (deltaYawDeg >= 2.5 || deltaPitchDeg >= 2.5) {
      this.headPoseVerified = true;
      // Rigid head sway clears static-planar uncertainty ONLY — keep phone/bezel streak intact
      this.clearPlanarSpoofOnly(now);
    }
    if (deltaLip >= 0.045 || nonRigidMotion >= 0.022) {
      this.lipMotionVerified = true;
      this.clearSpoofStateImmediately(now);
    }
    // Distance-Aware Multi-Mode Liveness: at 1.5-4.5m (< 0.22 faceHeight), do NOT require blinking;
    // pass within 0.6s on natural body/head sway (> 1.5px shift or > 1.2 deg tilt) or non-frozen posture
    if (
      livenessMode === 'LONG_RANGE' &&
      (centerShiftPx > 1.5 ||
        fastDeltaYaw > 1.2 ||
        fastDeltaPitch > 1.2 ||
        (this.frameCount >= 12 && (centerShiftPx > 0.4 || nonRigidMotion >= 0.002)))
    ) {
      this.longRangeMotionVerified = true;
      this.clearPlanarSpoofOnly(now);
    }

    // 4.5s window variance for static 2D planar check
    const yawVariance = Number(stdDev(this.poseHistory.map((s) => s.yawDeg)).toFixed(3));
    const pitchVariance = Number(stdDev(this.poseHistory.map((s) => s.pitchDeg)).toFixed(3));

    const blinkVerified =
      this.blinkCount >= 1 &&
      this.lastBlinkAtMs !== null &&
      now - this.lastBlinkAtMs <= this.blinkExpiryMs;

    const liveVerifiedNow = blinkVerified || now <= this.liveVerifiedUntilMs;
    const genuineNonRigidVerified = blinkVerified || this.lipMotionVerified;

    // 3. Enclosed Phone Bezel Check + Strict 4.5s Static Planar Rule
    // Evaluate bezel/screen artifacts even during rigid sway so tilting a phone cannot bypass bezel inspection.
    let clearBezelSpoof = false;
    if (input.faceCrop || input.contextFrame) {
      const screenReport = analyzeScreenAndBezelArtifacts({
        faceCrop: input.faceCrop,
        contextFrame: input.contextFrame,
        faceBox: input.faceBox,
        faceScale,
      });
      this.latestScreenReport = screenReport;
      const unmistakableScreen =
        screenReport.bezelScore >= 0.72 && screenReport.combinedSpoofRisk >= 0.68;
      if (unmistakableScreen) {
        this.screenSpoofStreak += 1;
        if (this.screenSpoofStreak >= 4) {
          clearBezelSpoof = true;
        }
      } else if (genuineNonRigidVerified) {
        this.screenSpoofStreak = 0;
        this.latestScreenReport = DEFAULT_SCREEN_REPORT;
      } else {
        this.screenSpoofStreak = Math.max(0, this.screenSpoofStreak - 1);
      }
    } else if (genuineNonRigidVerified) {
      this.latestScreenReport = DEFAULT_SCREEN_REPORT;
      this.screenSpoofStreak = 0;
    }

    // Only flag 2D flat rigid spoof AFTER >= 4.5s (120+ frames or 4500ms) with practically zero 3D yaw/pitch variance & zero blink
    const observedLongEnoughForStaticCheck =
      observationDurationMs >= this.staticSpoofWindowMs && this.frameCount >= 35;

    if (
      !liveVerifiedNow &&
      observedLongEnoughForStaticCheck &&
      yawVariance < 0.15 &&
      pitchVariance < 0.15 &&
      !this.headPoseVerified &&
      !this.lipMotionVerified &&
      !this.longRangeMotionVerified
    ) {
      this.planarSpoofDetected = true;
    } else if (liveVerifiedNow) {
      this.planarSpoofDetected = false;
    }

    // 4. Micro-vascular CHROM rPPG & Bit-Identical Frozen Injection Guard (post-4.5s only if unverified)
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

        if (!liveVerifiedNow && observedLongEnoughForStaticCheck && chromStd < 1e-5 && lumaStd < 1e-4) {
          this.staticInjectionDetected = true;
        } else {
          this.staticInjectionDetected = false;
          const acScore =
            clamp01(chromStd / 0.028) * clamp01(1 - Math.max(0, chromStd - 0.12) / 0.15);
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

    const screenSpoofDetected = clearBezelSpoof;
    const spoofDetected =
      screenSpoofDetected ||
      (!liveVerifiedNow && (this.planarSpoofDetected || this.staticInjectionDetected));

    const spoofScore = spoofDetected
      ? Number(
          Math.max(
            0.75,
            this.latestScreenReport.combinedSpoofRisk,
            this.planarSpoofDetected ? 0.85 : 0,
          ).toFixed(3),
        )
      : 0;

    let spoofReason: BiologicalSpoofReason = 'NONE';
    if (screenSpoofDetected) {
      spoofReason = 'SCREEN_DETECTED';
    } else if (spoofDetected && this.planarSpoofDetected) {
      spoofReason = 'PLANAR_RIGID_MOTION';
    } else if (spoofDetected && this.staticInjectionDetected) {
      spoofReason = 'STATIC_FRAME_INJECTION';
    }

    const biologicalVerified = liveVerifiedNow && !spoofDetected;

    return {
      biologicalVerified,
      livenessPassed: biologicalVerified,
      livenessMode,
      estimatedDistanceM,
      centerShiftPx: Number(centerShiftPx.toFixed(2)),
      longRangeMotionVerified: this.longRangeMotionVerified,
      blinkVerified,
      blinkDetectedNow,
      headPoseVerified: this.headPoseVerified,
      lipMotionVerified: this.lipMotionVerified,
      blinkCount: this.blinkCount,
      eyesClosedNow,
      openEyeBaseline: Number(this.openEyeBaseline.toFixed(3)),
      baselineEar: Number(this.openEyeBaseline.toFixed(3)),
      faceScale,
      deltaYawDeg: Number(deltaYawDeg.toFixed(2)),
      deltaPitchDeg: Number(deltaPitchDeg.toFixed(2)),
      yawVariance,
      pitchVariance,
      observationDurationMs,
      blendshapeBlinkMax: Number(blendshapeBlinkMax.toFixed(3)),
      nonRigidScore: Number(this.cumulativeNonRigid.toFixed(3)),
      nonRigidMotion: Number(nonRigidMotion.toFixed(5)),
      globalMotion: Number(globalMotion.toFixed(5)),
      planarSpoofDetected: spoofDetected && this.planarSpoofDetected,
      screenSpoofDetected,
      spoofDetected,
      spoofScore,
      spoofReason,
      rppgScore: this.latestRppgScore,
      rppgPulseAlive: this.rppgPulseAlive,
      screenReport: this.latestScreenReport,
    };
  }
}
