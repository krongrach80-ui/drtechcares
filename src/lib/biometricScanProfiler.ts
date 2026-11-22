/**
 * Staff-Only Kiosk Biometric Scan Performance Profiler (Step 0 Instrumentation).
 *
 * Strictly records numeric timings, hardware delegate labels, and frame counters ONLY.
 * NEVER stores patient names, HNs, embeddings, or raw camera images, and NEVER logs to console.
 */

export type BiometricLivenessCue =
  | 'pending'
  | 'iris_saccade'
  | 'blink'
  | 'head_pose'
  | 'lip'
  | 'long_range';

export interface TimingPercentileSummary {
  count: number;
  avgMs: number;
  p50Ms: number;
  p95Ms: number;
  lastMs: number;
}

export interface KioskScanRoundMetrics {
  timestampMs: number;
  outcome: 'matched' | 'not_found' | 'aborted';
  cameraOpenMs: number;
  cameraProbeCount: number;
  usedRememberedCamera: boolean;
  firstFrameMs: number;
  exposureSettleMs: number;
  modelWarmupMs: number;
  wasPrewarmed: boolean;
  faceDelegate: 'GPU' | 'CPU' | 'unknown';
  humanBackend: 'webgl' | 'cpu' | 'unknown';
  faceDetectAvgMs: number;
  roiZoomAvgMs: number;
  livenessStepAvgMs: number;
  irisExtractAvgMs: number;
  humanEmbedAvgMs: number;
  localMatchAvgMs: number;
  overlayDrawAvgMs: number;
  scanFps: number;
  overlayFps: number;
  skippedStaleFrames: number;
  detectInputLabel: string;
  faceToReadyMs: number;
  livenessLockMs: number;
  livenessCue: BiometricLivenessCue;
  serverVerifyMs: number;
  pipelinedOverlapSavedMs: number;
  postLockToModalMs: number;
  totalFaceToVerifiedMs: number;
}

export interface KioskProfilerLiveSnapshot {
  roundsCount: number;
  targetRounds: number;
  live: {
    cameraOpenMs: number;
    cameraProbeCount: number;
    usedRememberedCamera: boolean;
    firstFrameMs: number;
    exposureSettleMs: number;
    modelWarmupMs: number;
    wasPrewarmed: boolean;
    faceDelegate: 'GPU' | 'CPU' | 'unknown';
    humanBackend: 'webgl' | 'cpu' | 'unknown';
    executionThread: 'main';
    detectInputLabel: string;
    faceDetectMs: number;
    roiZoomMs: number;
    livenessStepMs: number;
    irisExtractMs: number;
    humanEmbedMs: number;
    localMatchMs: number;
    overlayDrawMs: number;
    scanFps: number;
    overlayFps: number;
    skippedStaleFrames: number;
    faceToReadyMs: number;
    livenessLockMs: number;
    livenessCue: BiometricLivenessCue;
    serverVerifyMs: number;
    pipelinedOverlapSavedMs: number;
    postLockToModalMs: number;
    totalFaceToVerifiedMs: number;
  };
  stats: {
    cameraOpenMs: TimingPercentileSummary;
    modelWarmupMs: TimingPercentileSummary;
    faceDetectAvgMs: TimingPercentileSummary;
    humanEmbedAvgMs: TimingPercentileSummary;
    overlayDrawAvgMs: TimingPercentileSummary;
    faceToReadyMs: TimingPercentileSummary;
    livenessLockMs: TimingPercentileSummary;
    serverVerifyMs: TimingPercentileSummary;
    postLockToModalMs: TimingPercentileSummary;
    totalFaceToVerifiedMs: TimingPercentileSummary;
  };
  topBottlenecks: { stage: string; avgMs: number }[];
}

const SESSION_STORAGE_KEY = 'drtech_kiosk_bio_perf_rounds_v1';
const MAX_STORED_ROUNDS = 50;

function round1(v: number): number {
  if (!Number.isFinite(v) || v < 0) return 0;
  return Number(v.toFixed(1));
}

export function computeTimingPercentiles(samples: readonly number[]): TimingPercentileSummary {
  const valid = samples.filter((v) => Number.isFinite(v) && v >= 0);
  const count = valid.length;
  if (count === 0) {
    return { count: 0, avgMs: 0, p50Ms: 0, p95Ms: 0, lastMs: 0 };
  }
  const sorted = [...valid].sort((a, b) => a - b);
  let sum = 0;
  for (let i = 0; i < count; i += 1) sum += valid[i];
  const pickPercentile = (p: number) => {
    const idx = Math.min(count - 1, Math.max(0, Math.ceil(p * count) - 1));
    return sorted[idx];
  };
  return {
    count,
    avgMs: round1(sum / count),
    p50Ms: round1(pickPercentile(0.5)),
    p95Ms: round1(pickPercentile(0.95)),
    lastMs: round1(valid[count - 1]),
  };
}

function readStoredRounds(): KioskScanRoundMetrics[] {
  if (typeof window === 'undefined' || !window.sessionStorage) return [];
  try {
    const raw = window.sessionStorage.getItem(SESSION_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    return Array.isArray(parsed) ? (parsed as KioskScanRoundMetrics[]).slice(-MAX_STORED_ROUNDS) : [];
  } catch {
    return [];
  }
}

function writeStoredRounds(rounds: readonly KioskScanRoundMetrics[]): void {
  if (typeof window === 'undefined' || !window.sessionStorage) return;
  try {
    window.sessionStorage.setItem(
      SESSION_STORAGE_KEY,
      JSON.stringify(rounds.slice(-MAX_STORED_ROUNDS)),
    );
  } catch {
    // Ignore sessionStorage quota issues
  }
}

export class KioskBiometricScanProfiler {
  private rounds: KioskScanRoundMetrics[] = readStoredRounds();

  private viewOpenedAt = 0;
  private cameraOpenMs = 0;
  private cameraProbeCount = 0;
  private usedRememberedCamera = false;
  private firstFrameMs = 0;
  private exposureSettleMs = 0;
  private exposureDone = false;
  private modelWarmupMs = 0;
  private wasPrewarmed = false;
  private faceDelegate: 'GPU' | 'CPU' | 'unknown' = 'unknown';
  private humanBackend: 'webgl' | 'cpu' | 'unknown' = 'unknown';
  private detectInputLabel = '480x270';

  private faceFirstSeenAt = 0;
  private faceToReadyMs = 0;
  private livenessLockMs = 0;
  private livenessCue: BiometricLivenessCue = 'pending';
  private serverVerifyMs = 0;
  private pipelinedOverlapSavedMs = 0;
  private postLockToModalMs = 0;
  private totalFaceToVerifiedMs = 0;

  private faceDetectSamples: number[] = [];
  private roiZoomSamples: number[] = [];
  private livenessStepSamples: number[] = [];
  private irisExtractSamples: number[] = [];
  private humanEmbedSamples: number[] = [];
  private localMatchSamples: number[] = [];
  private overlayDrawSamples: number[] = [];
  private scanFrameTimestamps: number[] = [];
  private overlayFrameTimestamps: number[] = [];
  private skippedStaleFrames = 0;

  private lastFaceDetectMs = 0;
  private lastRoiZoomMs = 0;
  private lastLivenessStepMs = 0;
  private lastIrisExtractMs = 0;
  private lastHumanEmbedMs = 0;
  private lastLocalMatchMs = 0;
  private lastOverlayDrawMs = 0;

  private pushCapped(arr: number[], val: number, maxLen = 40): void {
    if (!Number.isFinite(val) || val < 0) return;
    arr.push(val);
    if (arr.length > maxLen) arr.shift();
  }

  private avgOf(arr: readonly number[]): number {
    if (arr.length === 0) return 0;
    let sum = 0;
    for (let i = 0; i < arr.length; i += 1) sum += arr[i];
    return round1(sum / arr.length);
  }

  private computeFps(timestamps: number[], now: number): number {
    while (timestamps.length > 0 && now - timestamps[0] > 1500) {
      timestamps.shift();
    }
    if (timestamps.length < 2) return 0;
    const spanMs = timestamps[timestamps.length - 1] - timestamps[0];
    if (spanMs <= 0) return 0;
    return round1(((timestamps.length - 1) * 1000) / spanMs);
  }

  startSession(now: number): void {
    this.viewOpenedAt = now;
    this.cameraOpenMs = 0;
    this.cameraProbeCount = 0;
    this.usedRememberedCamera = false;
    this.firstFrameMs = 0;
    this.exposureSettleMs = 0;
    this.exposureDone = false;
    this.resetFaceAttempt();
  }

  resetFaceAttempt(): void {
    this.faceFirstSeenAt = 0;
    this.faceToReadyMs = 0;
    this.livenessLockMs = 0;
    this.livenessCue = 'pending';
    this.serverVerifyMs = 0;
    this.pipelinedOverlapSavedMs = 0;
    this.postLockToModalMs = 0;
    this.totalFaceToVerifiedMs = 0;
    this.faceDetectSamples = [];
    this.roiZoomSamples = [];
    this.livenessStepSamples = [];
    this.irisExtractSamples = [];
    this.humanEmbedSamples = [];
    this.localMatchSamples = [];
    this.overlayDrawSamples = [];
    this.skippedStaleFrames = 0;
  }

  clearStoredRounds(): void {
    this.rounds = [];
    writeStoredRounds([]);
  }

  recordCameraOpen(info: {
    openDurationMs: number;
    probeCount: number;
    usedRememberedDevice: boolean;
  }): void {
    this.cameraOpenMs = round1(info.openDurationMs);
    this.cameraProbeCount = info.probeCount;
    this.usedRememberedCamera = info.usedRememberedDevice;
  }

  recordModelsReady(info: {
    modelWarmupMs: number;
    wasPrewarmed: boolean;
    faceDelegate: 'GPU' | 'CPU' | 'unknown';
    humanBackend: 'webgl' | 'cpu' | 'unknown';
  }): void {
    this.modelWarmupMs = round1(info.modelWarmupMs);
    this.wasPrewarmed = info.wasPrewarmed;
    this.faceDelegate = info.faceDelegate;
    this.humanBackend = info.humanBackend;
  }

  recordFirstVideoFrame(now: number): void {
    if (this.firstFrameMs === 0 && this.viewOpenedAt > 0) {
      this.firstFrameMs = round1(now - this.viewOpenedAt);
    }
  }

  recordExposureSettled(durationMs: number): void {
    if (!this.exposureDone) {
      this.exposureDone = true;
      this.exposureSettleMs = round1(durationMs);
    }
  }

  recordSkippedStaleFrame(): void {
    this.skippedStaleFrames += 1;
  }

  recordDetectFrame(info: {
    now: number;
    detectMs: number;
    roiMs: number;
    livenessMs: number;
    inputWidth: number;
    inputHeight: number;
  }): void {
    this.scanFrameTimestamps.push(info.now);
    this.lastFaceDetectMs = round1(info.detectMs);
    this.lastRoiZoomMs = round1(info.roiMs);
    this.lastLivenessStepMs = round1(info.livenessMs);
    this.detectInputLabel = `${info.inputWidth}x${info.inputHeight}`;
    this.pushCapped(this.faceDetectSamples, info.detectMs);
    if (info.roiMs > 0) this.pushCapped(this.roiZoomSamples, info.roiMs);
    this.pushCapped(this.livenessStepSamples, info.livenessMs);
  }

  recordOverlayDraw(now: number, drawMs: number): void {
    this.overlayFrameTimestamps.push(now);
    this.lastOverlayDrawMs = round1(drawMs);
    this.pushCapped(this.overlayDrawSamples, drawMs);
  }

  markFaceSeen(now: number): void {
    if (this.faceFirstSeenAt === 0) {
      this.faceFirstSeenAt = now;
    }
  }

  markFaceReady(now: number): void {
    if (this.faceFirstSeenAt > 0 && this.faceToReadyMs === 0) {
      this.faceToReadyMs = round1(now - this.faceFirstSeenAt);
    }
  }

  markLivenessVerified(now: number, cue: BiometricLivenessCue): void {
    if (this.faceFirstSeenAt > 0 && this.livenessLockMs === 0) {
      this.livenessLockMs = round1(now - this.faceFirstSeenAt);
      this.livenessCue = cue;
    }
  }

  recordEmbedStep(info: {
    irisMs: number;
    humanEmbedMs: number;
    localMatchMs: number;
  }): void {
    this.lastIrisExtractMs = round1(info.irisMs);
    this.lastHumanEmbedMs = round1(info.humanEmbedMs);
    this.lastLocalMatchMs = round1(info.localMatchMs);
    if (info.irisMs > 0) this.pushCapped(this.irisExtractSamples, info.irisMs);
    if (info.humanEmbedMs > 0) this.pushCapped(this.humanEmbedSamples, info.humanEmbedMs);
    if (info.localMatchMs >= 0) this.pushCapped(this.localMatchSamples, info.localMatchMs);
  }

  recordServerVerify(rttMs: number, overlapSavedMs = 0): void {
    this.serverVerifyMs = round1(rttMs);
    this.pipelinedOverlapSavedMs = round1(Math.max(0, overlapSavedMs));
  }

  commitRound(
    outcome: 'matched' | 'not_found' | 'aborted',
    now: number,
    postLockMs = 0,
  ): KioskScanRoundMetrics {
    this.postLockToModalMs = round1(postLockMs);
    this.totalFaceToVerifiedMs =
      this.faceFirstSeenAt > 0 ? round1(now - this.faceFirstSeenAt) : 0;

    const record: KioskScanRoundMetrics = {
      timestampMs: Date.now(),
      outcome,
      cameraOpenMs: this.cameraOpenMs,
      cameraProbeCount: this.cameraProbeCount,
      usedRememberedCamera: this.usedRememberedCamera,
      firstFrameMs: this.firstFrameMs,
      exposureSettleMs: this.exposureSettleMs,
      modelWarmupMs: this.modelWarmupMs,
      wasPrewarmed: this.wasPrewarmed,
      faceDelegate: this.faceDelegate,
      humanBackend: this.humanBackend,
      faceDetectAvgMs: this.avgOf(this.faceDetectSamples),
      roiZoomAvgMs: this.avgOf(this.roiZoomSamples),
      livenessStepAvgMs: this.avgOf(this.livenessStepSamples),
      irisExtractAvgMs: this.avgOf(this.irisExtractSamples),
      humanEmbedAvgMs: this.avgOf(this.humanEmbedSamples),
      localMatchAvgMs: this.avgOf(this.localMatchSamples),
      overlayDrawAvgMs: this.avgOf(this.overlayDrawSamples),
      scanFps: this.computeFps(this.scanFrameTimestamps, now),
      overlayFps: this.computeFps(this.overlayFrameTimestamps, now),
      skippedStaleFrames: this.skippedStaleFrames,
      detectInputLabel: this.detectInputLabel,
      faceToReadyMs: this.faceToReadyMs,
      livenessLockMs: this.livenessLockMs,
      livenessCue: this.livenessCue,
      serverVerifyMs: this.serverVerifyMs,
      pipelinedOverlapSavedMs: this.pipelinedOverlapSavedMs,
      postLockToModalMs: this.postLockToModalMs,
      totalFaceToVerifiedMs: this.totalFaceToVerifiedMs,
    };

    if (outcome !== 'aborted') {
      this.rounds.push(record);
      if (this.rounds.length > MAX_STORED_ROUNDS) {
        this.rounds.shift();
      }
      writeStoredRounds(this.rounds);
    }
    return record;
  }

  getSnapshot(now = typeof performance !== 'undefined' ? performance.now() : 0): KioskProfilerLiveSnapshot {
    const matchedOrAll = this.rounds;
    const stats = {
      cameraOpenMs: computeTimingPercentiles(matchedOrAll.map((r) => r.cameraOpenMs)),
      modelWarmupMs: computeTimingPercentiles(matchedOrAll.map((r) => r.modelWarmupMs)),
      faceDetectAvgMs: computeTimingPercentiles(matchedOrAll.map((r) => r.faceDetectAvgMs)),
      humanEmbedAvgMs: computeTimingPercentiles(matchedOrAll.map((r) => r.humanEmbedAvgMs)),
      overlayDrawAvgMs: computeTimingPercentiles(matchedOrAll.map((r) => r.overlayDrawAvgMs)),
      faceToReadyMs: computeTimingPercentiles(matchedOrAll.map((r) => r.faceToReadyMs)),
      livenessLockMs: computeTimingPercentiles(matchedOrAll.map((r) => r.livenessLockMs)),
      serverVerifyMs: computeTimingPercentiles(
        matchedOrAll.filter((r) => r.serverVerifyMs > 0).map((r) => r.serverVerifyMs),
      ),
      postLockToModalMs: computeTimingPercentiles(
        matchedOrAll.filter((r) => r.outcome === 'matched').map((r) => r.postLockToModalMs),
      ),
      totalFaceToVerifiedMs: computeTimingPercentiles(
        matchedOrAll.filter((r) => r.outcome === 'matched').map((r) => r.totalFaceToVerifiedMs),
      ),
    };

    const bottleneckCandidates = [
      { stage: 'humanEmbed1024', avgMs: stats.humanEmbedAvgMs.avgMs || this.lastHumanEmbedMs },
      { stage: 'serverVerifyRTT', avgMs: stats.serverVerifyMs.avgMs || this.serverVerifyMs },
      { stage: 'livenessLock', avgMs: stats.livenessLockMs.avgMs || this.livenessLockMs },
      { stage: 'cameraOpen', avgMs: stats.cameraOpenMs.avgMs || this.cameraOpenMs },
      { stage: 'faceDetect478', avgMs: stats.faceDetectAvgMs.avgMs || this.lastFaceDetectMs },
      { stage: 'overlayDraw', avgMs: stats.overlayDrawAvgMs.avgMs || this.lastOverlayDrawMs },
    ]
      .filter((item) => item.avgMs > 0)
      .sort((a, b) => b.avgMs - a.avgMs)
      .slice(0, 3);

    return {
      roundsCount: this.rounds.length,
      targetRounds: 20,
      live: {
        cameraOpenMs: this.cameraOpenMs,
        cameraProbeCount: this.cameraProbeCount,
        usedRememberedCamera: this.usedRememberedCamera,
        firstFrameMs: this.firstFrameMs,
        exposureSettleMs: this.exposureSettleMs,
        modelWarmupMs: this.modelWarmupMs,
        wasPrewarmed: this.wasPrewarmed,
        faceDelegate: this.faceDelegate,
        humanBackend: this.humanBackend,
        executionThread: 'main',
        detectInputLabel: this.detectInputLabel,
        faceDetectMs: this.lastFaceDetectMs,
        roiZoomMs: this.lastRoiZoomMs,
        livenessStepMs: this.lastLivenessStepMs,
        irisExtractMs: this.lastIrisExtractMs,
        humanEmbedMs: this.lastHumanEmbedMs,
        localMatchMs: this.lastLocalMatchMs,
        overlayDrawMs: this.lastOverlayDrawMs,
        scanFps: this.computeFps(this.scanFrameTimestamps, now),
        overlayFps: this.computeFps(this.overlayFrameTimestamps, now),
        skippedStaleFrames: this.skippedStaleFrames,
        faceToReadyMs: this.faceToReadyMs,
        livenessLockMs: this.livenessLockMs,
        livenessCue: this.livenessCue,
        serverVerifyMs: this.serverVerifyMs,
        pipelinedOverlapSavedMs: this.pipelinedOverlapSavedMs,
        postLockToModalMs: this.postLockToModalMs,
        totalFaceToVerifiedMs: this.totalFaceToVerifiedMs,
      },
      stats,
      topBottlenecks: bottleneckCandidates,
    };
  }
}

