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
  extractHybrid512Descriptor,
  selectClinicalCameraDeviceId,
  selectPrimaryFace,
} from '@drtechcares/face-client';
import { evaluateFrameQuality, l2Normalize } from '@drtechcares/smart-biometric-core';
import type { AngleEmbeddings5D } from '@drtechcares/api-client';
import {
  edgeFaceEnrollFinalize,
  edgeFaceEnrollFrame,
  edgeFaceEnrollStart,
  probeEdgeVisionHealth,
} from '../hooks/useEdgeVisionSocket';

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
  yawDeg: number;
  pitchDeg: number;
  sharpness: number;
  bucket: Bucket5Key;
  embedding: number[];
}

export function classifyDisplayBucket(yawDeg: number, pitchDeg: number): Bucket5Key {
  if (Math.abs(yawDeg) >= Math.abs(pitchDeg) * 1.15) {
    if (yawDeg <= -8) return 'left';
    if (yawDeg >= 8) return 'right';
  }
  if (pitchDeg <= -6) return 'up';
  if (pitchDeg >= 6) return 'down';
  if (yawDeg <= -8) return 'left';
  if (yawDeg >= 8) return 'right';
  return 'center';
}

export function chooseCaptureBucket(
  yawDeg: number,
  pitchDeg: number,
  counts: Record<Bucket5Key, number>,
): Bucket5Key | null {
  const natural = classifyDisplayBucket(yawDeg, pitchDeg);
  if ((counts[natural] ?? 0) < BUCKET_TARGETS[natural]) {
    return natural;
  }

  // Only allow diagonal overflow into angles whose threshold is genuinely satisfied by current pose.
  // Never spill frontal frames into left/right/up/down or vice versa.
  const candidates: Bucket5Key[] = [];
  if (yawDeg <= -8) candidates.push('left');
  if (yawDeg >= 8) candidates.push('right');
  if (pitchDeg <= -6) candidates.push('up');
  if (pitchDeg >= 6) candidates.push('down');
  if (Math.abs(yawDeg) < 8 && Math.abs(pitchDeg) < 6) candidates.push('center');

  return candidates.find((b) => (counts[b] ?? 0) < BUCKET_TARGETS[b]) ?? null;
}

const TARGET_GUIDANCE: Record<
  Bucket5Key,
  { stepTitle: string; turnCue: string; holdCue: string; voice: string }
> = {
  center: {
    stepTitle: 'ขั้นตอนที่ 1/5: มองตรงเข้าหากล้อง',
    turnCue: 'กรุณามองตรงเข้าหากล้องให้อยู่ในกรอบ',
    holdCue: 'ค้างหน้าตรงไว้สักครู่',
    voice: 'ขั้นตอนที่หนึ่ง กรุณามองตรงเข้าหากล้องค่ะ',
  },
  left: {
    stepTitle: 'ขั้นตอนที่ 2/5: หันหน้าไปทางซ้าย (<-8°)',
    turnCue: 'หน้าตรงครบแล้ว — กรุณาหันหน้าไปทางซ้ายเล็กน้อย',
    holdCue: 'กำลังเก็บมิติใบหน้าด้านซ้าย',
    voice: 'ขั้นตอนที่สอง กรุณาหันหน้าไปทางซ้ายเล็กน้อยค่ะ',
  },
  right: {
    stepTitle: 'ขั้นตอนที่ 3/5: หันหน้าไปทางขวา (>8°)',
    turnCue: 'ด้านซ้ายครบแล้ว — กรุณาหันหน้าไปทางขวาเล็กน้อย',
    holdCue: 'กำลังเก็บมิติใบหน้าด้านขวา',
    voice: 'ขั้นตอนที่สาม กรุณาหันหน้าไปทางขวาเล็กน้อยค่ะ',
  },
  up: {
    stepTitle: 'ขั้นตอนที่ 4/5: เงยหน้าขึ้นเล็กน้อย (<-6°)',
    turnCue: 'ด้านขวาครบแล้ว — กรุณาเงยหน้าขึ้นเล็กน้อย',
    holdCue: 'กำลังเก็บมิติมุมเงยและแนวคาง',
    voice: 'ขั้นตอนที่สี่ กรุณาเงยหน้าขึ้นเล็กน้อยค่ะ',
  },
  down: {
    stepTitle: 'ขั้นตอนที่ 5/5: ก้มหน้าลงเล็กน้อย (>6°)',
    turnCue: 'มุมเงยครบแล้ว — กรุณาก้มหน้าลงเล็กน้อย',
    holdCue: 'กำลังเก็บมิติมุมก้มและหน้าผาก',
    voice: 'ขั้นตอนสุดท้าย กรุณาก้มหน้าลงเล็กน้อยค่ะ',
  },
};

function findNextMissingBucket(counts: Record<Bucket5Key, number>): Bucket5Key {
  const order: Bucket5Key[] = ['center', 'left', 'right', 'up', 'down'];
  return order.find((key) => (counts[key] ?? 0) < BUCKET_TARGETS[key]) ?? 'down';
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
  const [edgeBackendActive, setEdgeBackendActive] = useState(false);

  const edgeSessionIdRef = useRef<string | null>(null);
  const samplesRef = useRef<CapturedSample[]>([]);
  const bucketCountsRef = useRef<Record<Bucket5Key, number>>({ ...bucketCounts });
  const prevLandmarksRef = useRef<Array<{ x: number; y: number; z?: number }> | null>(null);
  const lastCaptureAtRef = useRef(0);
  const lastVideoTimeRef = useRef(-1);
  const lastSpokenAtRef = useRef(0);
  const lastSpokenCueRef = useRef('');
  const lastGuidedTargetRef = useRef<Bucket5Key | null>(null);
  const hasSnapshotFrameRef = useRef(false);
  const bestFrontalDataUrlRef = useRef<string>('');
  const bestFrontalScoreRef = useRef<number>(-1);
  const completedRef = useRef(false);
  const finalizePromiseRef = useRef<Promise<FaceID120Result | null> | null>(null);
  const finalizedResultRef = useRef<FaceID120Result | null>(null);
  const readyNotifiedRef = useRef(false);
  const unmountedRef = useRef(false);
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

  const finalizeScan = useCallback(async (): Promise<FaceID120Result | null> => {
    if (finalizedResultRef.current) return finalizedResultRef.current;
    if (finalizePromiseRef.current) return finalizePromiseRef.current;
    const edgeSid = edgeSessionIdRef.current;
    if (!edgeSid && samplesRef.current.length === 0) return null;

    completedRef.current = true;
    setPhase('computing');
    setGuidanceText('กำลังสร้างรหัส Biometric 3D 512 มิติ...');
    setSubCueText('ประมวลผลเวกเตอร์เฉลี่ย 5 มุมมอง (Center, Left, Right, Up, Down)');

    const run = async (): Promise<FaceID120Result | null> => {
      if (edgeSid && samplesRef.current.length === 0) {
        const edgeRes = await edgeFaceEnrollFinalize({
          session_id: edgeSid,
          hn,
          persist_to_supabase: false,
        });
        if (unmountedRef.current) return null;

        if (!edgeRes.avatar_data_url && hasSnapshotFrameRef.current && snapshotCanvasRef.current) {
          try {
            bestFrontalDataUrlRef.current = snapshotCanvasRef.current.toDataURL('image/jpeg', 0.88);
          } catch {
            bestFrontalDataUrlRef.current = '';
          }
        }
        const avatarDataUrl = edgeRes.avatar_data_url || bestFrontalDataUrlRef.current;
        const resultPayload: FaceID120Result = {
          masterEmbedding: edgeRes.master_embedding,
          angleEmbeddings: {
            center: edgeRes.angle_embeddings.center ?? edgeRes.master_embedding,
            left: edgeRes.angle_embeddings.left ?? edgeRes.master_embedding,
            right: edgeRes.angle_embeddings.right ?? edgeRes.master_embedding,
            up: edgeRes.angle_embeddings.up ?? edgeRes.master_embedding,
            down: edgeRes.angle_embeddings.down ?? edgeRes.master_embedding,
          },
          framesCaptured: edgeRes.frames_captured,
          qualityScore: edgeRes.quality_score,
          avatarDataUrl,
        };
        finalizedResultRef.current = resultPayload;
        setFinalQualityPct(Math.round(edgeRes.quality_score * 100));
        setPhase('completed');
        setGuidanceText(`ยืนยันการเก็บใบหน้า ${edgeRes.frames_captured} เฟรม (Edge ArcFace 512-D) สำเร็จ!`);
        setSubCueText(`บันทึก 5 มุมมองครบถ้วน • คุณภาพ ${Math.round(edgeRes.quality_score * 100)}%`);
        playSuccessChime();
        speakGuidance('บันทึกใบหน้าสามมิติครบเรียบร้อยแล้วค่ะ สามารถกดบันทึกและเปิดใช้งานตู้คีออสได้เลยค่ะ', true);
        stopHardware();
        if (!unmountedRef.current) {
          onCompleteRef.current(resultPayload);
          onReadyStateChangeRef.current?.(true, async () => resultPayload);
        }
        return resultPayload;
      }

      const samples = [...samplesRef.current];
      if (unmountedRef.current) return null;

      const withEmbeddings = samples.filter((s) => s.embedding.length === 512);
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

      if (!bestFrontalDataUrlRef.current && hasSnapshotFrameRef.current && snapshotCanvasRef.current) {
        try {
          bestFrontalDataUrlRef.current = snapshotCanvasRef.current.toDataURL('image/jpeg', 0.88);
        } catch {
          bestFrontalDataUrlRef.current = '';
        }
      }

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
  }, [hn, speakGuidance, stopHardware]);

  const resetScannerState = useCallback(() => {
    edgeSessionIdRef.current = null;
    setEdgeBackendActive(false);
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
    hasSnapshotFrameRef.current = false;
    lastVideoTimeRef.current = -1;
    bestFrontalDataUrlRef.current = '';
    bestFrontalScoreRef.current = -1;
    completedRef.current = false;
    finalizePromiseRef.current = null;
    finalizedResultRef.current = null;
    readyNotifiedRef.current = false;
    lastGuidedTargetRef.current = null;
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
        setModelProgressMsg('กำลังตรวจสอบ Edge InsightFace Server...');

        const edgeHealth = await probeEdgeVisionHealth(450);
        if (cancelled) return;

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

        const snapCanvas = snapshotCanvasRef.current || document.createElement('canvas');
        snapCanvas.width = 480;
        snapCanvas.height = 480;
        snapshotCanvasRef.current = snapCanvas;

        // Path A: Edge Vision InsightFace (buffalo_l ArcFace 512-D)
        if (edgeHealth) {
          try {
            const edgeSess = await edgeFaceEnrollStart({
              hn,
              full_name: patientName,
            });
            if (cancelled) return;

            if (edgeSess?.session_id) {
              edgeSessionIdRef.current = edgeSess.session_id;
              setEdgeBackendActive(true);
              setModelProgressMsg('พร้อมสแกนด้วย Edge InsightFace ArcFace 512-D');
              setModelStatus('ready');
              setGuidanceText('วางใบหน้าให้อยู่ในกรอบวงกลม');
              speakGuidance('เชื่อมต่อระบบ Edge InsightFace สำเร็จ กรุณามองกล้องและหมุนศีรษะช้าๆ ค่ะ', true);

              const edgeCanvas = workCanvasRef.current || document.createElement('canvas');
              edgeCanvas.width = 240;
              edgeCanvas.height = 240;
              workCanvasRef.current = edgeCanvas;

              while (!cancelled && !completedRef.current) {
                const vid = videoRef.current;
                if (!vid || vid.readyState < 2 || vid.videoWidth <= 0) {
                  await new Promise((r) => window.setTimeout(r, 60));
                  continue;
                }

                const ectx = edgeCanvas.getContext('2d');
                if (!ectx) break;
                const side = Math.min(vid.videoWidth, vid.videoHeight);
                const sx = Math.max(0, (vid.videoWidth - side) / 2);
                const sy = Math.max(0, (vid.videoHeight - side) / 2);
                ectx.save();
                ectx.translate(240, 0);
                ectx.scale(-1, 1);
                ectx.drawImage(vid, sx, sy, side, side, 0, 0, 240, 240);
                ectx.restore();

                if (!hasSnapshotFrameRef.current) {
                  const sctx = snapCanvas.getContext('2d');
                  if (sctx) {
                    sctx.save();
                    sctx.translate(480, 0);
                    sctx.scale(-1, 1);
                    sctx.drawImage(vid, sx, sy, side, side, 0, 0, 480, 480);
                    sctx.restore();
                    hasSnapshotFrameRef.current = true;
                  }
                }

                const image_base64 = edgeCanvas.toDataURL('image/jpeg', 0.78);
                const state = await edgeFaceEnrollFrame({
                  session_id: edgeSess.session_id,
                  image_base64,
                });
                if (cancelled || completedRef.current) return;

                const nextCounts: Record<Bucket5Key, number> = {
                  center: state.bucket_counts.center ?? 0,
                  left: state.bucket_counts.left ?? 0,
                  right: state.bucket_counts.right ?? 0,
                  up: state.bucket_counts.up ?? 0,
                  down: state.bucket_counts.down ?? 0,
                };
                bucketCountsRef.current = nextCounts;
                setBucketCounts(nextCounts);
                setCapturedCount(state.frames_captured);

                const poseKey = (['center', 'left', 'right', 'up', 'down'].includes(state.current_pose)
                  ? state.current_pose
                  : 'center') as Bucket5Key;
                setActiveBucket(poseKey);
                const nextMissing = findNextMissingBucket(nextCounts);
                setTargetBucketHint(nextMissing);
                setLivePose({
                  yaw: Math.round(state.yaw_deg),
                  pitch: Math.round(state.pitch_deg),
                  roll: Math.round(state.roll_deg),
                  sharpness: Math.round(state.sharpness),
                  faceDetected: state.face_detected,
                });
                setGuidanceText(state.guidance_th || `${TARGET_GUIDANCE[nextMissing].stepTitle} (${state.frames_captured}/120)`);
                setSubCueText(`Edge InsightFace 512-D • มุม ${nextMissing} (${nextCounts[nextMissing]}/${BUCKET_TARGETS[nextMissing]})`);

                if (state.frames_captured >= TOTAL_TICKS || state.phase === 'completed') {
                  readyNotifiedRef.current = true;
                  await finalizeScan();
                  return;
                }

                await new Promise((r) => window.setTimeout(r, 45));
              }
              return;
            }
          } catch {
            // Fall through gracefully to browser MediaPipe Web AI
            edgeSessionIdRef.current = null;
            setEdgeBackendActive(false);
          }
        }

        // Path B: Auto-Fallback to Browser MediaPipe Web AI
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

        setModelProgressMsg('พร้อมสแกนใบหน้า 3 มิติ (Hybrid 512-D Real-Time)');
        setModelStatus('ready');
        setGuidanceText('วางใบหน้าให้อยู่ในกรอบวงกลม');
        speakGuidance('กรุณาวางใบหน้าให้อยู่ในกรอบวงกลม แล้วค่อยๆ หมุนศีรษะช้าๆ ค่ะ', true);

        const workCanvas = workCanvasRef.current || document.createElement('canvas');
        workCanvas.width = 112;
        workCanvas.height = 112;
        workCanvasRef.current = workCanvas;

        const tickLoop = () => {
          if (cancelled || completedRef.current) return;
          const vid = videoRef.current;
          const lm = landmarkerRef.current;
          if (!vid || !lm || vid.readyState < 2) {
            rafRef.current = requestAnimationFrame(tickLoop);
            return;
          }

          if (vid.currentTime === lastVideoTimeRef.current) {
            rafRef.current = requestAnimationFrame(tickLoop);
            return;
          }
          lastVideoTimeRef.current = vid.currentTime;

          const now = performance.now();
          let result;
          try {
            result = lm.detectForVideo(vid, now);
          } catch {
            rafRef.current = requestAnimationFrame(tickLoop);
            return;
          }

          const rawFaces = result?.faceLandmarks ?? [];
          if (rawFaces.length === 0) {
            setLivePose((p) => ({ ...p, faceDetected: false }));
            setGuidanceText('วางใบหน้าให้อยู่ในกรอบวงกลม');
            setSubCueText('ยังไม่พบใบหน้า — ขยับเข้าใกล้กล้องในระยะ 40–60 ซม.');
            prevLandmarksRef.current = null;
            rafRef.current = requestAnimationFrame(tickLoop);
            return;
          }

          const selection = selectPrimaryFace(rawFaces);
          const landmarks3D = selection.primaryFace ?? rawFaces[0];
          const liveness = evaluateLivenessAndPose(
            landmarks3D,
            prevLandmarksRef.current,
            1,
            false,
          );
          prevLandmarksRef.current = [...landmarks3D];

          // evaluateLivenessAndPose already returns degrees; mirror Yaw for scale-x-[-1] display
          const yawDeg = -liveness.yawDeg;
          const pitchDeg = liveness.pitchDeg;
          const rollDeg = liveness.rollDeg;

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

          const displayBucket = classifyDisplayBucket(yawDeg, pitchDeg);
          setActiveBucket(displayBucket);

          // Draw crispest frontal snapshot onto offscreen snapCanvas (encode JPEG only once inside finalizeScan)
          if (
            !hasSnapshotFrameRef.current ||
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
              hasSnapshotFrameRef.current = true;
            }
          }

          const captureBucket = chooseCaptureBucket(yawDeg, pitchDeg, bucketCountsRef.current);
          const totalSoFar = samplesRef.current.length;

          // Capture frames only when the live head pose matches an unfilled target angle
          if (captureBucket && totalSoFar < TOTAL_TICKS && now - lastCaptureAtRef.current >= 28) {
            lastCaptureAtRef.current = now;
            const embedding = extractHybrid512Descriptor(crop, landmarks3D);
            samplesRef.current.push({
              yawDeg,
              pitchDeg,
              sharpness: quality.sharpness,
              bucket: captureBucket,
              embedding,
            });
            bucketCountsRef.current = {
              ...bucketCountsRef.current,
              [captureBucket]: bucketCountsRef.current[captureBucket] + 1,
            };

            const nextTotal = samplesRef.current.length;
            setCapturedCount(nextTotal);
            setBucketCounts({ ...bucketCountsRef.current });

            const allBucketsFilled = (Object.keys(BUCKET_TARGETS) as Bucket5Key[]).every(
              (k) => bucketCountsRef.current[k] >= BUCKET_TARGETS[k],
            );
            if (allBucketsFilled || nextTotal >= TOTAL_TICKS) {
              readyNotifiedRef.current = true;
              void finalizeScan();
              return;
            }
          }

          // Dynamic strict 5-angle visual & voice guidance
          const bc = bucketCountsRef.current;
          const nextTotal = samplesRef.current.length;
          const nextTarget = findNextMissingBucket(bc);
          setTargetBucketHint(nextTarget);
          const cue = TARGET_GUIDANCE[nextTarget];

          if (lastGuidedTargetRef.current !== nextTarget) {
            lastGuidedTargetRef.current = nextTarget;
            speakGuidance(cue.voice, true);
          } else if (!captureBucket) {
            speakGuidance(cue.voice, false);
          }

          if (captureBucket) {
            const capCue = TARGET_GUIDANCE[captureBucket];
            setGuidanceText(`${capCue.stepTitle} (${nextTotal}/120)`);
            setSubCueText(
              `${capCue.holdCue} • มุมนี้ ${bc[captureBucket]}/${BUCKET_TARGETS[captureBucket]} เฟรม`,
            );
          } else {
            setGuidanceText(`${cue.stepTitle} (${nextTotal}/120)`);
            setSubCueText(`${cue.turnCue} (${bc[nextTarget]}/${BUCKET_TARGETS[nextTarget]} เฟรม)`);
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
    <div className="rounded-3xl border-2 border-emerald-200 bg-gradient-to-b from-[#F0FDF9] via-[#EFF6FF] to-white p-5 text-[#0F4C81] shadow-xl">
      {/* Top Status & Voice Toggle */}
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-300 bg-gradient-to-r from-emerald-50 to-sky-50 px-3 py-1 text-xs font-extrabold text-emerald-700 shadow-xs">
            <Sparkles className="h-3.5 w-3.5 text-emerald-600" />
            {edgeBackendActive
              ? 'Edge InsightFace 512-D • 120 เฟรม'
              : 'iPhone-Style 3D Face ID • 120 เฟรม'}
          </span>
          {patientName && (
            <span className="text-xs font-bold text-[#0F4C81]">
              {patientName} {hn ? `(${hn})` : ''}
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setVoiceEnabled((v) => !v)}
            className="inline-flex items-center gap-1 rounded-lg border border-sky-200 bg-white px-2.5 py-1 text-[11px] font-bold text-[#0F4C81] shadow-xs hover:bg-sky-50"
          >
            {voiceEnabled ? <Volume2 className="h-3.5 w-3.5 text-emerald-600" /> : <VolumeX className="h-3.5 w-3.5 text-slate-400" />}
            {voiceEnabled ? 'เสียงนำทางเปิด' : 'ปิดเสียง'}
          </button>
          <button
            type="button"
            onClick={() => setRetryNonce((n) => n + 1)}
            className="inline-flex items-center gap-1 rounded-lg border border-sky-200 bg-white px-2.5 py-1 text-[11px] font-bold text-[#0F4C81] shadow-xs hover:bg-sky-50"
          >
            <RefreshCw className="h-3.5 w-3.5 text-[#0284C7]" />
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
            const stroke = isLit ? '#10B981' : isHint ? '#0284C7' : '#93C5FD';
            const opacity = isLit ? 1 : isHint ? 0.85 : 0.55;

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
        <div className="relative h-[280px] w-[280px] overflow-hidden rounded-full border-4 border-sky-200 bg-gradient-to-br from-sky-100 to-emerald-50 shadow-[0_0_40px_rgba(2,132,199,0.2)]">
          <video
            ref={videoRef}
            muted
            playsInline
            className="h-full w-full -scale-x-100 object-cover"
          />

          {/* Subtle Inner Reticle */}
          <div className="pointer-events-none absolute inset-3 rounded-full border border-dashed border-white/60" />

          {/* Model Loading / Camera Error Overlay */}
          {(modelStatus === 'loading' || (!cameraReady && !cameraError)) && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-white/90 p-5 text-center backdrop-blur-xs">
              <Loader2 className="mb-2 h-8 w-8 animate-spin text-emerald-600" />
              <div className="text-xs font-extrabold text-[#0F4C81]">{modelProgressMsg}</div>
              <div className="mt-1 text-[11px] text-[#0284C7]">กรุณารอสักครู่...</div>
            </div>
          )}

          {cameraError && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-white/95 p-5 text-center">
              <Camera className="mb-2 h-8 w-8 text-rose-500" />
              <div className="text-xs font-extrabold text-rose-700">ไม่พบสิทธิ์เข้าถึงกล้อง</div>
              <div className="mt-1 text-[11px] text-slate-600">{cameraError}</div>
              <button
                type="button"
                onClick={() => setRetryNonce((n) => n + 1)}
                className="mt-3 rounded-lg bg-gradient-to-r from-[#0284C7] to-[#10B981] px-3 py-1.5 text-xs font-bold text-white"
              >
                ลองเปิดกล้องอีกครั้ง
              </button>
            </div>
          )}

          {/* Computing Phase Overlay */}
          {phase === 'computing' && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-white/88 p-4 text-center backdrop-blur-xs">
              <Loader2 className="mb-2 h-9 w-9 animate-spin text-[#0284C7]" />
              <div className="text-sm font-black text-[#0F4C81]">กำลังประมวลผล 120 เฟรม...</div>
              <div className="mt-1 text-[11px] text-[#0284C7]">คำนวณ Master + 5 Angle Embeddings</div>
            </div>
          )}

          {/* Apple Face ID Checkmark Burst on Completion */}
          {phase === 'completed' && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-emerald-50/92 p-4 text-center backdrop-blur-xs animate-in fade-in zoom-in duration-300">
              <div className="flex h-20 w-20 items-center justify-center rounded-full bg-gradient-to-br from-[#10B981] to-[#0284C7] shadow-[0_0_32px_rgba(16,185,129,0.5)]">
                <CheckCircle2 className="h-12 w-12 text-white" />
              </div>
              <div className="mt-3 text-sm font-black text-emerald-800">
                120 / 120 เฟรม สมบูรณ์แบบ
              </div>
              <div className="mt-0.5 text-xs font-bold text-[#0F4C81]">
                คุณภาพแม่แบบ 3D: {finalQualityPct}%
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Live Counter & Dynamic Thai Guidance */}
      <div className="mt-3 text-center">
        <div className="inline-flex items-baseline gap-2 rounded-full border border-emerald-300 bg-white px-4 py-1 shadow-xs">
          <span className="font-mono text-2xl font-black text-emerald-600">{capturedCount}</span>
          <span className="text-xs font-bold text-[#0F4C81]">/ 120 เฟรม ({progressPct}%)</span>
        </div>

        <h4 className="mt-2 text-base font-black text-[#0F4C81]">{guidanceText}</h4>
        <p className="mt-0.5 text-xs font-semibold text-[#0284C7]">{subCueText}</p>
      </div>

      {/* 5-Angle Multi-Perspective Telemetry Pills */}
      <div className="mt-4 grid grid-cols-5 gap-1.5 text-center text-[11px]">
        {BUCKET_LABELS.map((item) => {
          const count = bucketCounts[item.key] ?? 0;
          const target = BUCKET_TARGETS[item.key];
          const ok = count >= target;
          const isTarget = phase === 'scanning' && !ok && targetBucketHint === item.key;
          const isActivelyCapturing = isTarget && activeBucket === item.key;
          const started = count > 0;
          return (
            <div
              key={item.key}
              className={`rounded-xl border px-1.5 py-2 font-bold transition-colors ${
                ok
                  ? 'border-emerald-400 bg-emerald-50 text-emerald-800 shadow-xs'
                  : isActivelyCapturing
                    ? 'border-[#0284C7] bg-sky-50 text-[#0F4C81] ring-2 ring-[#0284C7]/40'
                    : isTarget
                      ? 'border-amber-400 bg-amber-50 text-amber-800 ring-2 ring-amber-400/50 animate-pulse'
                      : started
                        ? 'border-sky-300 bg-sky-50/70 text-[#0284C7]'
                        : 'border-sky-200 bg-white text-slate-600'
              }`}
            >
              <div className="truncate">{item.label}</div>
              <div className="mt-0.5 font-mono text-xs font-black">
                {count}/{target} {ok ? '✓' : isTarget ? '•' : ''}
              </div>
            </div>
          );
        })}
      </div>

      {/* Live Head Pose Diagnostics */}
      <div className="mt-3 flex flex-wrap items-center justify-between gap-2 rounded-xl border border-sky-200 bg-gradient-to-r from-emerald-50/80 via-sky-50/80 to-white px-3 py-2 text-[11px] text-[#0F4C81]">
        <div className="flex items-center gap-2">
          <ShieldCheck className={`h-3.5 w-3.5 ${livePose.faceDetected ? 'text-emerald-600' : 'text-amber-500'}`} />
          <span>
            Yaw: <strong className="font-mono text-[#0F4C81]">{livePose.yaw}°</strong> • Pitch:{' '}
            <strong className="font-mono text-[#0F4C81]">{livePose.pitch}°</strong> • Angle:{' '}
            <strong className="font-mono text-emerald-700">{activeBucket}</strong>
          </span>
        </div>
        <span>
          Sharpness: <strong className="font-mono text-[#0F4C81]">{livePose.sharpness}</strong>
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
            className="rounded-lg border border-sky-200 bg-white px-3 py-1.5 text-xs font-bold text-[#0F4C81] hover:bg-sky-50"
          >
            ปิดกล้องชั่วคราว
          </button>
        </div>
      )}
    </div>
  );
}
