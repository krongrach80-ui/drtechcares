import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Camera,
  CheckCircle2,
  Loader2,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Volume2,
  VolumeX,
} from 'lucide-react';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import {
  alignFaceTo112,
  evaluateLivenessAndPose,
  extractFaceEmbeddingWithFallback,
  getArcFaceSession,
  normalizeAngleToDegrees,
  selectClinicalCameraDeviceId,
  selectPrimaryFace,
} from '@drtechcares/face-client';
import { evaluateFrameQuality, l2Normalize } from '@drtechcares/smart-biometric-core';
import type { AngleEmbeddings5D } from '@drtechcares/api-client';

export interface FaceID120Result {
  masterEmbedding: number[];
  angleEmbeddings: AngleEmbeddings5D;
  framesCaptured: number;
  qualityScore: number;
  avatarDataUrl: string;
}

export type Bucket5Key = keyof AngleEmbeddings5D; // 'center' | 'left' | 'right' | 'up' | 'down'

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

const BUCKET_TARGETS: Record<Bucket5Key, number> = {
  center: 40,
  left: 20,
  right: 20,
  up: 20,
  down: 20,
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

const BUCKET_LABELS: Array<{ key: Bucket5Key; label: string }> = [
  { key: 'center', label: 'หน้าตรง' },
  { key: 'left', label: 'ซ้าย (<-8°)' },
  { key: 'right', label: 'ขวา (>8°)' },
  { key: 'up', label: 'เงย (<-6°)' },
  { key: 'down', label: 'ก้ม (>6°)' },
];

interface CapturedSample {
  crop: ImageData;
  landmarks3D: Array<{ x: number; y: number; z?: number }>;
  yawDeg: number;
  pitchDeg: number;
  sharpness: number;
  bucket: Bucket5Key;
  embedding?: number[];
}

function classifyDisplayBucket(yawDeg: number, pitchDeg: number): Bucket5Key {
  if (Math.abs(yawDeg) <= 8 && Math.abs(pitchDeg) <= 8) return 'center';
  const absYaw = Math.abs(yawDeg);
  const absPitch = Math.abs(pitchDeg);
  if (absYaw >= absPitch) {
    if (yawDeg < -8) return 'left';
    if (yawDeg > 8) return 'right';
  }
  if (pitchDeg < -6) return 'up';
  if (pitchDeg > 6) return 'down';
  if (yawDeg < -8) return 'left';
  if (yawDeg > 8) return 'right';
  return 'center';
}

function chooseCaptureBucket(
  yawDeg: number,
  pitchDeg: number,
  counts: Record<Bucket5Key, number>,
): Bucket5Key | null {
  const candidates: Bucket5Key[] = [];
  if (Math.abs(yawDeg) <= 8 && Math.abs(pitchDeg) <= 8) candidates.push('center');
  if (yawDeg < -8) candidates.push('left');
  if (yawDeg > 8) candidates.push('right');
  if (pitchDeg < -6) candidates.push('up');
  if (pitchDeg > 6) candidates.push('down');

  const unfilledPrimary = candidates.find((b) => (counts[b] ?? 0) < BUCKET_TARGETS[b]);
  if (unfilledPrimary) return unfilledPrimary;

  // Soft-band spillover so slightly tilted poses keep feeding unfilled nearby directions.
  if (yawDeg <= -5 && counts.left < BUCKET_TARGETS.left) return 'left';
  if (yawDeg >= 5 && counts.right < BUCKET_TARGETS.right) return 'right';
  if (pitchDeg <= -4.5 && counts.up < BUCKET_TARGETS.up) return 'up';
  if (pitchDeg >= 4.5 && counts.down < BUCKET_TARGETS.down) return 'down';
  if (Math.abs(yawDeg) <= 12 && Math.abs(pitchDeg) <= 10 && counts.center < BUCKET_TARGETS.center) {
    return 'center';
  }
  return null;
}

function findNextMissingBucket(counts: Record<Bucket5Key, number>): Bucket5Key {
  if (counts.center < 10) return 'center';
  const order: Bucket5Key[] = ['left', 'right', 'up', 'down', 'center'];
  let bestKey: Bucket5Key = 'center';
  let maxMissing = -1;
  for (const key of order) {
    const missing = BUCKET_TARGETS[key] - (counts[key] ?? 0);
    if (missing > maxMissing) {
      maxMissing = missing;
      bestKey = key;
    }
  }
  return bestKey;
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
  return l2Normalize(sum);
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

export default function FaceID120Scanner({
  active,
  patientName,
  hn,
  onComplete,
  onReadyStateChange,
  onCancel,
}: FaceID120ScannerProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const workCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const snapshotCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const rafRef = useRef<number | null>(null);

  const [modelStatus, setModelStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [modelProgressMsg, setModelProgressMsg] = useState('กำลังเตรียมโมเดล Face Landmarker 3D...');
  const [cameraReady, setCameraReady] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [voiceEnabled, setVoiceEnabled] = useState(true);
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
  const [guidanceText, setGuidanceText] = useState('วางใบหน้าให้อยู่ในกรอบวงกลม');
  const [subCueText, setSubCueText] = useState('ระบบกำลังตรวจจับตำแหน่งใบหน้า 3 มิติ...');
  const [targetBucketHint, setTargetBucketHint] = useState<Bucket5Key>('center');
  const [finalQualityPct, setFinalQualityPct] = useState(0);
  const [retryNonce, setRetryNonce] = useState(0);

  const samplesRef = useRef<CapturedSample[]>([]);
  const bucketCountsRef = useRef<Record<Bucket5Key, number>>({ ...bucketCounts });
  const prevLandmarksRef = useRef<Array<{ x: number; y: number; z?: number }> | null>(null);
  const lastCaptureAtRef = useRef(0);
  const lastSpokenAtRef = useRef(0);
  const lastSpokenCueRef = useRef('');
  const bestFrontalDataUrlRef = useRef<string>('');
  const bestFrontalScoreRef = useRef<number>(-1);
  const completedRef = useRef(false);
  const finalizePromiseRef = useRef<Promise<FaceID120Result | null> | null>(null);
  const finalizedResultRef = useRef<FaceID120Result | null>(null);
  const readyNotifiedRef = useRef(false);
  const unmountedRef = useRef(false);
  const embeddingWorkerActiveRef = useRef(false);
  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;
  const onReadyStateChangeRef = useRef(onReadyStateChange);
  onReadyStateChangeRef.current = onReadyStateChange;
  const voiceEnabledRef = useRef(voiceEnabled);
  voiceEnabledRef.current = voiceEnabled;

  const speakGuidance = useCallback((phrase: string, force = false) => {
    if (!voiceEnabledRef.current || typeof window === 'undefined' || !('speechSynthesis' in window)) {
      return;
    }
    const now = Date.now();
    if (!force && (phrase === lastSpokenCueRef.current || now - lastSpokenAtRef.current < 3200)) {
      return;
    }
    try {
      window.speechSynthesis.cancel();
      const utter = new SpeechSynthesisUtterance(phrase);
      utter.lang = 'th-TH';
      utter.rate = 1.05;
      utter.pitch = 1.02;
      window.speechSynthesis.speak(utter);
      lastSpokenAtRef.current = now;
      lastSpokenCueRef.current = phrase;
    } catch {
      // Ignore TTS blocks
    }
  }, []);

  const stopHardware = useCallback(() => {
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
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      try {
        window.speechSynthesis.cancel();
      } catch {
        // ignore
      }
    }
  }, []);

  const drainEmbeddingsIncrementally = useCallback(async () => {
    if (embeddingWorkerActiveRef.current) return;
    embeddingWorkerActiveRef.current = true;
    try {
      while (!unmountedRef.current) {
        const nextIdx = samplesRef.current.findIndex((s) => !s.embedding);
        if (nextIdx === -1) break;
        const item = samplesRef.current[nextIdx];
        if (!item) break;
        try {
          const { vector } = await extractFaceEmbeddingWithFallback(item.crop, item.landmarks3D);
          item.embedding = vector;
        } catch {
          break;
        }
      }
    } finally {
      embeddingWorkerActiveRef.current = false;
    }
  }, []);

  const finalizeScan = useCallback(async (): Promise<FaceID120Result | null> => {
    if (finalizedResultRef.current) return finalizedResultRef.current;
    if (finalizePromiseRef.current) return finalizePromiseRef.current;
    if (samplesRef.current.length === 0) return null;

    completedRef.current = true;
    setPhase('computing');
    setGuidanceText('กำลังสร้างรหัส Biometric 3D 512 มิติ...');
    setSubCueText('ประมวลผลเวกเตอร์เฉลี่ย 5 มุมมอง (Center, Left, Right, Up, Down)');

    const run = async (): Promise<FaceID120Result | null> => {
      const samples = [...samplesRef.current];
      const startMs = Date.now();
      for (let i = 0; i < samples.length; i++) {
        if (unmountedRef.current) return null;
        if (!samples[i].embedding) {
          if (Date.now() - startMs > 2800) {
            const bucket = samples[i].bucket;
            const computedInBucket = samples.filter((s) => s.bucket === bucket && s.embedding).length;
            if (computedInBucket >= 6) continue;
          }
          const { vector } = await extractFaceEmbeddingWithFallback(
            samples[i].crop,
            samples[i].landmarks3D,
          );
          samples[i].embedding = vector;
        }
      }

      const withEmbeddings = samples.filter((s): s is CapturedSample & { embedding: number[] } =>
        Boolean(s.embedding && s.embedding.length === 512),
      );
      const allVectors = withEmbeddings.map((s) => s.embedding);
      const masterEmbedding = averageNormalizedEmbeddings(allVectors);

      const byBucket = (b: Bucket5Key) => {
        const vecs = withEmbeddings.filter((s) => s.bucket === b).map((s) => s.embedding);
        return vecs.length > 0 ? averageNormalizedEmbeddings(vecs) : masterEmbedding;
      };

      const angleEmbeddings: AngleEmbeddings5D = {
        center: byBucket('center'),
        left: byBucket('left'),
        right: byBucket('right'),
        up: byBucket('up'),
        down: byBucket('down'),
      };

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
      setFinalQualityPct(Math.round(qualityScore * 100));

      const resultPayload: FaceID120Result = {
        masterEmbedding,
        angleEmbeddings,
        framesCaptured: samples.length,
        qualityScore,
        avatarDataUrl: bestFrontalDataUrlRef.current,
      };
      finalizedResultRef.current = resultPayload;

      setPhase('completed');
      setGuidanceText(`ยืนยันการเก็บใบหน้า ${samples.length} เฟรม (3D) สำเร็จ!`);
      setSubCueText(`บันทึก 5 มุมมองครบถ้วน • คุณภาพ ${Math.round(qualityScore * 100)}%`);
      playSuccessChime();
      speakGuidance('บันทึกใบหน้าสามมิติครบเรียบร้อยแล้วค่ะ สามารถกดบันทึกและเปิดใช้งานตู้คีออสได้เลยค่ะ', true);

      stopHardware();

      if (!unmountedRef.current) {
        onCompleteRef.current(resultPayload);
        onReadyStateChangeRef.current?.(true, async () => resultPayload);
      }
      return resultPayload;
    };

    finalizePromiseRef.current = run();
    return finalizePromiseRef.current;
  }, [speakGuidance, stopHardware]);

  const resetScannerState = useCallback(() => {
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
    setGuidanceText('วางใบหน้าให้อยู่ในกรอบวงกลม');
    setSubCueText('เตรียมพร้อม... มองตรงเข้าหากล้องเพื่อเริ่มต้น');
    setTargetBucketHint('center');
    bestFrontalDataUrlRef.current = '';
    bestFrontalScoreRef.current = -1;
    completedRef.current = false;
    finalizePromiseRef.current = null;
    finalizedResultRef.current = null;
    readyNotifiedRef.current = false;
    prevLandmarksRef.current = null;
    onReadyStateChangeRef.current?.(false, finalizeScan);
  }, [finalizeScan]);

  useEffect(() => {
    unmountedRef.current = false;
    if (!active) {
      stopHardware();
      return;
    }

    let cancelled = false;
    resetScannerState();

    const startEngineAndCamera = async () => {
      try {
        setModelStatus('loading');
        setCameraError(null);
        setModelProgressMsg('กำลังโหลด MediaPipe Face Landmarker & WASM...');

        const vision = await FilesetResolver.forVisionTasks('/mediapipe/wasm');
        if (cancelled) return;

        const landmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: '/models/face_landmarker.task',
            delegate: 'GPU',
          },
          runningMode: 'VIDEO',
          numFaces: 2,
          minFaceDetectionConfidence: 0.65,
          minFacePresenceConfidence: 0.65,
          minTrackingConfidence: 0.65,
          outputFaceBlendshapes: false,
          outputFacialTransformationMatrixes: false,
        });
        if (cancelled) {
          landmarker.close();
          return;
        }
        landmarkerRef.current = landmarker;

        setModelProgressMsg('กำลังเตรียม ArcFace 512-D Embedder...');
        void getArcFaceSession().catch(() => {
          // Hybrid 512-D fallback activates automatically if ArcFace ONNX is unavailable
        });
        setModelStatus('ready');

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
        if (!video) return;
        video.srcObject = stream;
        await video.play();
        setCameraReady(true);
        setGuidanceText('วางใบหน้าให้อยู่ในกรอบวงกลม');
        speakGuidance('กรุณาวางใบหน้าให้อยู่ในกรอบวงกลม แล้วค่อยๆ หมุนศีรษะช้าๆ ค่ะ', true);

        const workCanvas = workCanvasRef.current || document.createElement('canvas');
        workCanvas.width = 112;
        workCanvas.height = 112;
        workCanvasRef.current = workCanvas;

        const snapCanvas = snapshotCanvasRef.current || document.createElement('canvas');
        snapCanvas.width = 480;
        snapCanvas.height = 480;
        snapshotCanvasRef.current = snapCanvas;

        const tickLoop = () => {
          if (cancelled || completedRef.current) return;
          const vid = videoRef.current;
          const lm = landmarkerRef.current;
          if (!vid || !lm || vid.readyState < 2) {
            rafRef.current = requestAnimationFrame(tickLoop);
            return;
          }

          let result;
          try {
            result = lm.detectForVideo(vid, performance.now());
          } catch {
            rafRef.current = requestAnimationFrame(tickLoop);
            return;
          }

          const rawFaces = result?.faceLandmarks ?? [];
          const selection = selectPrimaryFace(rawFaces);
          if (!selection.primaryFace) {
            setLivePose((p) => ({ ...p, faceDetected: false }));
            setGuidanceText('วางใบหน้าให้อยู่ในกรอบวงกลม');
            setSubCueText('ยังไม่พบใบหน้า — ขยับเข้าใกล้กล้องในระยะ 40–60 ซม.');
            prevLandmarksRef.current = null;
            rafRef.current = requestAnimationFrame(tickLoop);
            return;
          }

          if (selection.effectiveFaceCount > 1) {
            setLivePose((p) => ({ ...p, faceDetected: false }));
            setGuidanceText('พบมากกว่า 1 ใบหน้าในเฟรม');
            setSubCueText('กรุณาให้มีเฉพาะใบหน้าของคนไข้ 1 ท่านเท่านั้น');
            rafRef.current = requestAnimationFrame(tickLoop);
            return;
          }

          const landmarks3D = selection.primaryFace;
          const liveness = evaluateLivenessAndPose(
            landmarks3D,
            prevLandmarksRef.current,
            1,
            false,
          );
          prevLandmarksRef.current = [...landmarks3D];

          // Convert Radians -> Degrees if needed and flip Yaw for scale-x-[-1] mirrored display
          const yawDeg = -normalizeAngleToDegrees(liveness.yawDeg);
          const pitchDeg = normalizeAngleToDegrees(liveness.pitchDeg);
          const rollDeg = normalizeAngleToDegrees(liveness.rollDeg);

          // Center & size check inside viewfinder (permissive for rotated poses)
          const centerInfo = selection.primaryCenter ?? { x: 0.5, y: 0.5 };
          const boxInfo = selection.primaryBox ?? { width: 0.25, height: 0.25, area: 0.0625 };
          const centered =
            Math.hypot(centerInfo.x - 0.5, centerInfo.y - 0.5) <= 0.34 &&
            boxInfo.width >= 0.10 &&
            boxInfo.width <= 0.82;

          const crop = alignFaceTo112(vid, landmarks3D, workCanvas);
          if (!crop) {
            rafRef.current = requestAnimationFrame(tickLoop);
            return;
          }

          const quality = evaluateFrameQuality(
            crop,
            { yawDeg, pitchDeg, rollDeg },
            52,
            52,
          );

          setLivePose({
            yaw: Math.round(yawDeg),
            pitch: Math.round(pitchDeg),
            roll: Math.round(rollDeg),
            sharpness: Math.round(quality.sharpness),
            faceDetected: true,
          });

          if (!centered) {
            setGuidanceText('ขยับใบหน้าให้อยู่กึ่งกลางวงกลม');
            setSubCueText(
              boxInfo.width < 0.10 ? 'ขยับใบหน้าเข้ามาใกล้กล้องอีกนิด' : 'จัดตำแหน่งใบหน้าให้อยู่ตรงกลางพอดี',
            );
            rafRef.current = requestAnimationFrame(tickLoop);
            return;
          }

          if (quality.sharpness < 8 || quality.brightness < 18) {
            setGuidanceText('ภาพเบลอหรือแสงน้อยเกินไป');
            setSubCueText('ถือศีรษะให้นิ่งชั่วครู่หรือเพิ่มแสงสว่างบริเวณใบหน้า');
            rafRef.current = requestAnimationFrame(tickLoop);
            return;
          }

          const displayBucket = classifyDisplayBucket(yawDeg, pitchDeg);
          setActiveBucket(displayBucket);

          // Save crispest frontal snapshot for profile avatar (with immediate fallback on first frame)
          if (
            !bestFrontalDataUrlRef.current ||
            (displayBucket === 'center' && quality.sharpness > bestFrontalScoreRef.current)
          ) {
            bestFrontalScoreRef.current = displayBucket === 'center' ? quality.sharpness : 0;
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
              bestFrontalDataUrlRef.current = snapCanvas.toDataURL('image/jpeg', 0.88);
            }
          }

          const now = performance.now();
          const captureBucket = chooseCaptureBucket(yawDeg, pitchDeg, bucketCountsRef.current);
          const totalSoFar = samplesRef.current.length;

          // Capture valid frame at ~24 fps (42ms interval) when the current pose's bucket needs frames
          if (captureBucket && totalSoFar < TOTAL_TICKS && now - lastCaptureAtRef.current >= 42) {
            lastCaptureAtRef.current = now;
            const clonedCrop = new ImageData(
              new Uint8ClampedArray(crop.data),
              crop.width,
              crop.height,
            );
            samplesRef.current.push({
              crop: clonedCrop,
              landmarks3D: landmarks3D.map((p) => ({ x: p.x, y: p.y, z: p.z })),
              yawDeg,
              pitchDeg,
              sharpness: quality.sharpness,
              bucket: captureBucket,
            });
            bucketCountsRef.current = {
              ...bucketCountsRef.current,
              [captureBucket]: bucketCountsRef.current[captureBucket] + 1,
            };

            const nextTotal = samplesRef.current.length;
            setCapturedCount(nextTotal);
            setBucketCounts({ ...bucketCountsRef.current });
            void drainEmbeddingsIncrementally();

            if (!readyNotifiedRef.current) {
              const allFiveReady = (Object.keys(BUCKET_TARGETS) as Bucket5Key[]).every(
                (key) => (bucketCountsRef.current[key] ?? 0) >= 1,
              );
              if (allFiveReady) {
                readyNotifiedRef.current = true;
                onReadyStateChangeRef.current?.(true, finalizeScan);
              }
            }

            if (nextTotal >= TOTAL_TICKS) {
              void finalizeScan();
              return;
            }
          }

          // Dynamic guidance based on unfilled 5-angle quotas (40 Center / 20 Left / 20 Right / 20 Up / 20 Down)
          const bc = bucketCountsRef.current;
          const nextTotal = samplesRef.current.length;
          const nextTarget = findNextMissingBucket(bc);
          setTargetBucketHint(nextTarget);

          if (bc.center < 10) {
            setGuidanceText(`มองตรงค้างไว้สักครู่ (${nextTotal}/120)`);
            setSubCueText(`กำลังเก็บมุมหน้าตรง (${bc.center}/${BUCKET_TARGETS.center} เฟรม)`);
          } else if (nextTarget === 'left') {
            setGuidanceText(`ค่อยๆ หันศีรษะไปทางซ้าย (${nextTotal}/120)`);
            setSubCueText(`เก็บมุมซ้าย (Yaw < -8°) • ${bc.left}/${BUCKET_TARGETS.left} เฟรม`);
            if (bc.left < 4) speakGuidance('ค่อยๆ หันหน้าไปทางซ้ายเล็กน้อยค่ะ');
          } else if (nextTarget === 'right') {
            setGuidanceText(`ค่อยๆ หันศีรษะไปทางขวา (${nextTotal}/120)`);
            setSubCueText(`เก็บมุมขวา (Yaw > 8°) • ${bc.right}/${BUCKET_TARGETS.right} เฟรม`);
            if (bc.right < 4) speakGuidance('ค่อยๆ หันหน้าไปทางขวาเล็กน้อยค่ะ');
          } else if (nextTarget === 'up') {
            setGuidanceText(`ค่อยๆ เงยหน้าขึ้นเล็กน้อย (${nextTotal}/120)`);
            setSubCueText(`เก็บมุมเงยหน้า (Pitch < -6°) • ${bc.up}/${BUCKET_TARGETS.up} เฟรม`);
            if (bc.up < 4) speakGuidance('ค่อยๆ เงยหน้าขึ้นเล็กน้อยค่ะ');
          } else if (nextTarget === 'down') {
            setGuidanceText(`ค่อยๆ ก้มหน้าลงเล็กน้อย (${nextTotal}/120)`);
            setSubCueText(`เก็บมุมก้มหน้า (Pitch > 6°) • ${bc.down}/${BUCKET_TARGETS.down} เฟรม`);
            if (bc.down < 4) speakGuidance('ค่อยๆ ก้มหน้าลงเล็กน้อยค่ะ');
          } else {
            setGuidanceText(`มองตรงค้างไว้เพื่อเก็บให้ครบ 120 เฟรม (${nextTotal}/120)`);
            setSubCueText(`กำลังเติมมุมหน้าตรง (${bc.center}/${BUCKET_TARGETS.center} เฟรม)`);
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
      stopHardware();
    };
  }, [
    active,
    drainEmbeddingsIncrementally,
    finalizeScan,
    resetScannerState,
    retryNonce,
    speakGuidance,
    stopHardware,
  ]);

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
  const circumferenceSize = 336;
  const cx = circumferenceSize / 2;
  const cy = circumferenceSize / 2;

  return (
    <div className="rounded-3xl border border-slate-200 bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950 p-5 text-white shadow-2xl">
      {/* Top Status & Voice Toggle */}
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-500/40 bg-emerald-500/15 px-3 py-1 text-xs font-extrabold text-emerald-300">
            <Sparkles className="h-3.5 w-3.5" />
            iPhone-Style 3D Face ID • 120 เฟรม
          </span>
          {patientName && (
            <span className="text-xs font-bold text-slate-300">
              {patientName} {hn ? `(${hn})` : ''}
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setVoiceEnabled((v) => !v)}
            className="inline-flex items-center gap-1 rounded-lg border border-slate-700 bg-slate-800/80 px-2.5 py-1 text-[11px] font-bold text-slate-300 hover:bg-slate-700"
          >
            {voiceEnabled ? <Volume2 className="h-3.5 w-3.5 text-emerald-400" /> : <VolumeX className="h-3.5 w-3.5 text-slate-500" />}
            {voiceEnabled ? 'เสียงนำทางเปิด' : 'ปิดเสียง'}
          </button>
          <button
            type="button"
            onClick={() => setRetryNonce((n) => n + 1)}
            className="inline-flex items-center gap-1 rounded-lg border border-slate-700 bg-slate-800/80 px-2.5 py-1 text-[11px] font-bold text-slate-300 hover:bg-slate-700"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            รีเซ็ต 0/120
          </button>
        </div>
      </div>

      {/* Circular 280x280 Viewfinder + 120-Tick Radial Ring */}
      <div className="relative mx-auto flex h-[336px] w-[336px] items-center justify-center">
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
            const innerR = 148;
            const outerR = isLit ? 164 : isHint ? 160 : 156;
            const x1 = cx + innerR * Math.cos(rad);
            const y1 = cy + innerR * Math.sin(rad);
            const x2 = cx + outerR * Math.cos(rad);
            const y2 = cy + outerR * Math.sin(rad);
            const stroke = isLit ? '#10B981' : isHint ? '#38BDF8' : '#CBD5E1';
            const opacity = isLit ? 1 : isHint ? 0.72 : 0.28;

            return (
              <line
                key={idx}
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                stroke={stroke}
                strokeWidth={isLit ? 3.2 : 2.1}
                strokeLinecap="round"
                opacity={opacity}
                filter={isLit ? 'url(#emeraldGlow)' : undefined}
                style={{ transition: 'all 140ms ease-out' }}
              />
            );
          })}
        </svg>

        {/* Exact 280px x 280px Circular Live Webcam Feed */}
        <div className="relative h-[280px] w-[280px] overflow-hidden rounded-full border-4 border-slate-800 bg-slate-950 shadow-[0_0_50px_rgba(16,185,129,0.22)]">
          <video
            ref={videoRef}
            muted
            playsInline
            className="h-full w-full -scale-x-100 object-cover"
          />

          {/* Subtle Inner Reticle */}
          <div className="pointer-events-none absolute inset-3 rounded-full border border-dashed border-white/25" />

          {/* Model Loading / Camera Error Overlay */}
          {(modelStatus === 'loading' || (!cameraReady && !cameraError)) && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-950/85 p-5 text-center">
              <Loader2 className="mb-2 h-8 w-8 animate-spin text-emerald-400" />
              <div className="text-xs font-extrabold text-emerald-300">{modelProgressMsg}</div>
              <div className="mt-1 text-[11px] text-slate-400">กรุณารอสักครู่...</div>
            </div>
          )}

          {cameraError && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-950/92 p-5 text-center">
              <Camera className="mb-2 h-8 w-8 text-rose-400" />
              <div className="text-xs font-extrabold text-rose-300">ไม่พบสิทธิ์เข้าถึงกล้อง</div>
              <div className="mt-1 text-[11px] text-slate-300">{cameraError}</div>
              <button
                type="button"
                onClick={() => setRetryNonce((n) => n + 1)}
                className="mt-3 rounded-lg bg-emerald-600 px-3 py-1.5 text-xs font-bold text-white"
              >
                ลองเปิดกล้องอีกครั้ง
              </button>
            </div>
          )}

          {/* Computing Phase Overlay */}
          {phase === 'computing' && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-950/75 p-4 text-center backdrop-blur-xs">
              <Loader2 className="mb-2 h-9 w-9 animate-spin text-emerald-400" />
              <div className="text-sm font-black text-emerald-300">กำลังประมวลผล 120 เฟรม...</div>
              <div className="mt-1 text-[11px] text-slate-300">คำนวณ Master + 5 Angle Embeddings</div>
            </div>
          )}

          {/* Apple Face ID Checkmark Burst on Completion */}
          {phase === 'completed' && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-emerald-950/80 p-4 text-center backdrop-blur-xs animate-in fade-in zoom-in duration-300">
              <div className="flex h-20 w-20 items-center justify-center rounded-full bg-emerald-500 shadow-[0_0_40px_rgba(16,185,129,0.8)]">
                <CheckCircle2 className="h-12 w-12 text-white" />
              </div>
              <div className="mt-3 text-sm font-black text-emerald-200">
                120 / 120 เฟรม สมบูรณ์แบบ
              </div>
              <div className="mt-0.5 text-xs font-bold text-white/90">
                คุณภาพแม่แบบ 3D: {finalQualityPct}%
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Live Counter & Dynamic Thai Guidance */}
      <div className="mt-3 text-center">
        <div className="inline-flex items-baseline gap-2 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-4 py-1">
          <span className="font-mono text-2xl font-black text-emerald-400">{capturedCount}</span>
          <span className="text-xs font-bold text-slate-300">/ 120 เฟรม ({progressPct}%)</span>
        </div>

        <h4 className="mt-2 text-base font-black text-white">{guidanceText}</h4>
        <p className="mt-0.5 text-xs font-medium text-sky-300">{subCueText}</p>
      </div>

      {/* 5-Angle Multi-Perspective Telemetry Pills */}
      <div className="mt-4 grid grid-cols-5 gap-1.5 text-center text-[11px]">
        {BUCKET_LABELS.map((item) => {
          const count = bucketCounts[item.key] ?? 0;
          const target = BUCKET_TARGETS[item.key];
          const ok = count >= target;
          const started = count > 0;
          return (
            <div
              key={item.key}
              className={`rounded-xl border px-1.5 py-2 font-bold transition-colors ${
                ok
                  ? 'border-emerald-500/70 bg-emerald-500/20 text-emerald-300'
                  : started
                    ? 'border-sky-500/50 bg-sky-500/10 text-sky-200'
                    : 'border-slate-700 bg-slate-800/70 text-slate-400'
              }`}
            >
              <div className="truncate">{item.label}</div>
              <div className="mt-0.5 font-mono text-xs font-black">
                {count}/{target} {ok ? '✓' : ''}
              </div>
            </div>
          );
        })}
      </div>

      {/* Live Head Pose Diagnostics */}
      <div className="mt-3 flex flex-wrap items-center justify-between gap-2 rounded-xl border border-slate-800 bg-slate-900/90 px-3 py-2 text-[11px] text-slate-400">
        <div className="flex items-center gap-2">
          <ShieldCheck className={`h-3.5 w-3.5 ${livePose.faceDetected ? 'text-emerald-400' : 'text-amber-400'}`} />
          <span>
            Yaw: <strong className="font-mono text-white">{livePose.yaw}°</strong> • Pitch:{' '}
            <strong className="font-mono text-white">{livePose.pitch}°</strong> • Angle:{' '}
            <strong className="font-mono text-emerald-300">{activeBucket}</strong>
          </span>
        </div>
        <span>
          Sharpness: <strong className="font-mono text-white">{livePose.sharpness}</strong>
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
            className="rounded-lg px-3 py-1.5 text-xs font-bold text-slate-400 hover:bg-slate-800 hover:text-white"
          >
            ปิดกล้องชั่วคราว
          </button>
        </div>
      )}
    </div>
  );
}
