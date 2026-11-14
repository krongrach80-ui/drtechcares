import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { FilesetResolver, HandLandmarker, PoseLandmarker, type NormalizedLandmark } from '@mediapipe/tasks-vision';
import { useClinicalI18n } from '@drtechcares/clinical-i18n';
import {
  AudioChimeSynthesizer,
  HandGestureVerifier,
  VoiceFeedbackController,
} from '@drtechcares/kiosk-input';
import { selectClinicalCameraDeviceId } from '@drtechcares/face-client';
import { AlertTriangle, Award, CheckCircle2, Clock, Hand, RotateCcw, Sparkles, XCircle } from 'lucide-react';
import { sampleTenCognitiveCards, type CognitiveCardQuestion } from './PhysicalCardGameData';

const DWELL_CONFIRM_MS = 800;
const DWELL_GRACE_MS = 180;

const HAND_CONNECTIONS: ReadonlyArray<[number, number]> = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  [0, 17],
];

const POSE_HAND_CONNECTIONS_A: ReadonlyArray<[number, number]> = [
  [11, 13],
  [13, 15],
  [15, 17],
  [15, 19],
  [15, 21],
  [17, 19],
];

const POSE_HAND_CONNECTIONS_B: ReadonlyArray<[number, number]> = [
  [12, 14],
  [14, 16],
  [16, 18],
  [16, 20],
  [16, 22],
  [18, 20],
];

export interface PhysicalCardGameSummary {
  correct: number;
  total: number;
  category: string;
  durationSec: number;
  accuracyPct: number;
}

interface PhysicalCardGameProps {
  patientName: string;
  onComplete: (summary: PhysicalCardGameSummary) => void;
  onSessionFinished?: (summary: PhysicalCardGameSummary) => Promise<void> | void;
}

function classifyMirroredSide(screenX: number): 'yes' | 'no' | null {
  if (screenX < 0.48) return 'no';
  if (screenX > 0.52) return 'yes';
  return null;
}

function drawHandSkeletonOnCanvas(
  ctx: CanvasRenderingContext2D,
  landmarks: NormalizedLandmark[],
  width: number,
  height: number,
  color: string,
) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
    const start = landmarks[startIdx];
    const end = landmarks[endIdx];
    if (!start || !end) continue;
    ctx.beginPath();
    ctx.moveTo((1 - start.x) * width, start.y * height);
    ctx.lineTo((1 - end.x) * width, end.y * height);
    ctx.stroke();
  }

  for (let i = 0; i < landmarks.length; i += 1) {
    const lm = landmarks[i];
    if (!lm) continue;
    const cx = (1 - lm.x) * width;
    const cy = lm.y * height;
    ctx.fillStyle = i === 0 || i === 9 ? '#ffffff' : color;
    ctx.beginPath();
    ctx.arc(cx, cy, i === 0 || i === 9 ? 5.5 : 3.8, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#0f172a';
    ctx.lineWidth = 1.25;
    ctx.stroke();
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
  }
  ctx.restore();
}

function drawPoseArmSkeletonOnCanvas(
  ctx: CanvasRenderingContext2D,
  landmarks: NormalizedLandmark[],
  connections: ReadonlyArray<[number, number]>,
  width: number,
  height: number,
  color: string,
) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 3.5;
  ctx.lineCap = 'round';
  const drawnPoints = new Set<number>();

  for (const [a, b] of connections) {
    const start = landmarks[a];
    const end = landmarks[b];
    if (!start || !end) continue;
    if ((start.visibility ?? 1) < 0.25 || (end.visibility ?? 1) < 0.25) continue;
    ctx.beginPath();
    ctx.moveTo((1 - start.x) * width, start.y * height);
    ctx.lineTo((1 - end.x) * width, end.y * height);
    ctx.stroke();
    drawnPoints.add(a);
    drawnPoints.add(b);
  }

  for (const idx of drawnPoints) {
    const lm = landmarks[idx];
    if (!lm) continue;
    ctx.fillStyle = idx === 15 || idx === 16 ? '#ffffff' : color;
    ctx.beginPath();
    ctx.arc((1 - lm.x) * width, lm.y * height, 5, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function drawDwellHalo(
  ctx: CanvasRenderingContext2D,
  screenX: number,
  screenY: number,
  width: number,
  height: number,
  progress: number,
  color: string,
) {
  if (progress <= 0) return;
  ctx.save();
  const cx = screenX * width;
  const cy = screenY * height;
  ctx.beginPath();
  ctx.strokeStyle = 'rgba(255,255,255,0.28)';
  ctx.lineWidth = 4;
  ctx.arc(cx, cy, 34, 0, Math.PI * 2);
  ctx.stroke();

  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 5;
  ctx.lineCap = 'round';
  ctx.arc(cx, cy, 34, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * Math.min(1, progress));
  ctx.stroke();
  ctx.restore();
}

async function resolveVisionTasks() {
  try {
    return await FilesetResolver.forVisionTasks('/mediapipe/wasm');
  } catch {
    return await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm',
    );
  }
}

export function PhysicalCardGame({ patientName, onComplete, onSessionFinished }: PhysicalCardGameProps) {
  const { lang, t, pick } = useClinicalI18n();
  const [cards, setCards] = useState<CognitiveCardQuestion[]>(() => sampleTenCognitiveCards());
  const [index, setIndex] = useState(0);
  const [score, setScore] = useState(0);
  const [feedback, setFeedback] = useState<{ correct: boolean; message: string } | null>(null);
  const [holdProgress, setHoldProgress] = useState(0);
  const [pendingChoice, setPendingChoice] = useState<'yes' | 'no' | null>(null);
  const [bothHandsWarn, setBothHandsWarn] = useState(false);
  const [cropWarn, setCropWarn] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [detectorReady, setDetectorReady] = useState(false);
  const [completedSummary, setCompletedSummary] = useState<PhysicalCardGameSummary | null>(null);
  const [savingResult, setSavingResult] = useState(false);
  const [sessionKey, setSessionKey] = useState(0);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const verifierRef = useRef(new HandGestureVerifier());
  const voiceRef = useRef(new VoiceFeedbackController());
  const chimeRef = useRef(new AudioChimeSynthesizer());
  const dwellRef = useRef<{ choice: 'yes' | 'no' | null; startedAt: number; lastSeenAt: number }>({
    choice: null,
    startedAt: 0,
    lastSeenAt: 0,
  });
  const startedAtRef = useRef<number>(Date.now());
  const submittingRef = useRef(false);
  const timerRef = useRef<number | null>(null);

  const feedbackRef = useRef(feedback);
  feedbackRef.current = feedback;
  const indexRef = useRef(index);
  indexRef.current = index;
  const scoreRef = useRef(score);
  scoreRef.current = score;
  const cardsRef = useRef(cards);
  cardsRef.current = cards;

  const stopCameraTracks = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((tr) => tr.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  const currentCard = cards[index];

  const questionFontStyle = useMemo(() => {
    const text = currentCard ? pick(currentCard.question) : '';
    const size = text.length > 44 ? 'clamp(1.1rem, 2.3vw, 1.55rem)' : 'clamp(1.3rem, 2.8vw, 1.9rem)';
    return { fontSize: size };
  }, [currentCard, pick]);

  const submitAnswer = useCallback(
    (choice: 'yes' | 'no') => {
      const card = cardsRef.current[indexRef.current];
      if (!card || feedbackRef.current || submittingRef.current) return;
      submittingRef.current = true;
      dwellRef.current = { choice: null, startedAt: 0, lastSeenAt: 0 };

      const isCorrect = choice === card.correctAnswer;
      chimeRef.current.playTone(isCorrect ? 'correct' : 'wrong');
      const nextScore = isCorrect ? scoreRef.current + 1 : scoreRef.current;
      if (isCorrect) setScore(nextScore);
      const msg = pick(card.explanation);
      setFeedback({ correct: isCorrect, message: msg });
      voiceRef.current.speak(msg, lang, true);

      if (timerRef.current) window.clearTimeout(timerRef.current);
      timerRef.current = window.setTimeout(async () => {
        verifierRef.current.reset();
        setFeedback(null);
        setHoldProgress(0);
        setPendingChoice(null);
        submittingRef.current = false;

        if (indexRef.current + 1 >= cardsRef.current.length) {
          stopCameraTracks();
          const durationSec = Math.max(1, Math.round((Date.now() - startedAtRef.current) / 1000));
          const total = cardsRef.current.length;
          const accuracyPct = total > 0 ? Math.round((nextScore / total) * 100) : 0;
          const summary: PhysicalCardGameSummary = {
            correct: nextScore,
            total,
            category: 'mixed-99',
            durationSec,
            accuracyPct,
          };
          setCompletedSummary(summary);
          if (onSessionFinished) {
            setSavingResult(true);
            try {
              await onSessionFinished(summary);
            } catch {
              // ignore logging failures so summary remains responsive
            } finally {
              setSavingResult(false);
            }
          }
        } else {
          setIndex((i) => i + 1);
        }
      }, 1350);
    },
    [lang, onSessionFinished, pick, stopCameraTracks],
  );

  const submitAnswerRef = useRef(submitAnswer);
  submitAnswerRef.current = submitAnswer;

  useEffect(() => {
    if (completedSummary) return;

    let active = true;
    let rafId = 0;
    let handLandmarker: HandLandmarker | null = null;
    let poseLandmarker: PoseLandmarker | null = null;
    setCameraError(null);
    setDetectorReady(false);

    async function createHandOrPoseDetector() {
      const vision = await resolveVisionTasks();
      const handUrls = [
        '/models/hand_landmarker.task',
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
      ];
      for (const modelAssetPath of handUrls) {
        try {
          handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath },
            runningMode: 'VIDEO',
            numHands: 2,
            minHandDetectionConfidence: 0.45,
            minHandPresenceConfidence: 0.45,
            minTrackingConfidence: 0.45,
          });
          return 'hand' as const;
        } catch {
          // try next asset path or fall back to pose landmarker
        }
      }

      const poseUrls = ['/models/pose_landmarker.task', '/models/pose_landmarker_full.task'];
      for (const modelAssetPath of poseUrls) {
        try {
          poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath },
            runningMode: 'VIDEO',
            numPoses: 1,
            minPoseDetectionConfidence: 0.45,
            minPosePresenceConfidence: 0.45,
            minTrackingConfidence: 0.45,
          });
          return 'pose' as const;
        } catch {
          // try next pose model path
        }
      }

      return 'none' as const;
    }

    async function init() {
      let stream: MediaStream | null = null;
      try {
        const deviceId = await selectClinicalCameraDeviceId();
        stream = await navigator.mediaDevices.getUserMedia({
          video: deviceId
            ? { deviceId: { exact: deviceId }, width: { ideal: 640 }, height: { ideal: 480 } }
            : { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
          audio: false,
        });
      } catch {
        if (active) {
          setCameraError('ไม่สามารถเปิดกล้องได้ กรุณาอนุญาตการใช้กล้อง หรือคลิกปุ่มคำตอบด้านซ้ายได้เลย');
        }
        return;
      }

      if (!active || !videoRef.current) {
        stream.getTracks().forEach((tr) => tr.stop());
        return;
      }

      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      try {
        await videoRef.current.play();
      } catch {
        // video element play might settle slightly later
      }

      const mode = await createHandOrPoseDetector();
      if (!active) return;

      if (mode === 'none') {
        setCameraError('ระบบตรวจจับมือไม่พร้อมใช้งาน กรุณาคลิกเลือกคำตอบด้านซ้ายแทน');
        return;
      }

      setDetectorReady(true);

      const loop = () => {
        if (!active || !videoRef.current) return;
        const video = videoRef.current;
        const canvas = canvasRef.current;

        if (canvas) {
          const vw = video.videoWidth || 640;
          const vh = video.videoHeight || 480;
          if (canvas.width !== vw || canvas.height !== vh) {
            canvas.width = vw;
            canvas.height = vh;
          }
        }

        const ctx = canvas?.getContext('2d') ?? null;
        if (ctx && canvas) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
        }

        if (video.readyState >= 2 && !feedbackRef.current && !submittingRef.current) {
          const now = performance.now();
          let hasNo = false;
          let hasYes = false;
          let outOfBounds = false;
          let activeAnchor: { screenX: number; screenY: number; side: 'yes' | 'no' } | null = null;
          const isPortrait = typeof window !== 'undefined' && window.innerHeight > window.innerWidth;

          if (mode === 'hand' && handLandmarker) {
            const res = handLandmarker.detectForVideo(video, now);
            const allHands = res.landmarks ?? [];
            for (const handLms of allHands) {
              const wrist = handLms[0];
              const middleMcp = handLms[9];
              if (!wrist || !middleMcp) continue;
              const anchorX = (wrist.x + middleMcp.x) / 2;
              const anchorY = (wrist.y + middleMcp.y) / 2;
              const screenX = 1 - anchorX;
              const side = classifyMirroredSide(screenX);
              const isRaised = anchorY < 0.85;
              const color = side === 'yes' ? '#10b981' : side === 'no' ? '#f43f5e' : '#38bdf8';

              if (ctx && canvas) {
                drawHandSkeletonOnCanvas(ctx, handLms, canvas.width, canvas.height, color);
              }

              if (isPortrait && (screenX < 0.08 || screenX > 0.92)) {
                outOfBounds = true;
              }

              if (isRaised && side === 'no') {
                hasNo = true;
                activeAnchor = { screenX, screenY: anchorY, side: 'no' };
              } else if (isRaised && side === 'yes') {
                hasYes = true;
                activeAnchor = { screenX, screenY: anchorY, side: 'yes' };
              }
            }
          } else if (mode === 'pose' && poseLandmarker) {
            const res = poseLandmarker.detectForVideo(video, now);
            const pose = res.landmarks?.[0];
            if (pose) {
              const groups = [
                {
                  shoulder: pose[11],
                  wrist: pose[15],
                  index: pose[19],
                  connections: POSE_HAND_CONNECTIONS_A,
                },
                {
                  shoulder: pose[12],
                  wrist: pose[16],
                  index: pose[20],
                  connections: POSE_HAND_CONNECTIONS_B,
                },
              ];
              for (const g of groups) {
                if (!g.wrist || (g.wrist.visibility ?? 1) < 0.35) continue;
                const palmX = g.index && (g.index.visibility ?? 1) > 0.3 ? (g.wrist.x + g.index.x) / 2 : g.wrist.x;
                const palmY = g.index && (g.index.visibility ?? 1) > 0.3 ? (g.wrist.y + g.index.y) / 2 : g.wrist.y;
                const screenX = 1 - palmX;
                const side = classifyMirroredSide(screenX);
                const shoulderCeiling = g.shoulder ? Math.min(0.84, g.shoulder.y + 0.08) : 0.78;
                const isRaised = palmY < shoulderCeiling;
                const color = side === 'yes' ? '#10b981' : side === 'no' ? '#f43f5e' : '#38bdf8';

                if (ctx && canvas && isRaised) {
                  drawPoseArmSkeletonOnCanvas(ctx, pose, g.connections, canvas.width, canvas.height, color);
                }

                if (isPortrait && (screenX < 0.08 || screenX > 0.92)) {
                  outOfBounds = true;
                }

                if (isRaised && side === 'no') {
                  hasNo = true;
                  activeAnchor = { screenX, screenY: palmY, side: 'no' };
                } else if (isRaised && side === 'yes') {
                  hasYes = true;
                  activeAnchor = { screenX, screenY: palmY, side: 'yes' };
                }
              }
            }
          }

          setCropWarn(outOfBounds);

          if (hasNo && hasYes) {
            setBothHandsWarn(true);
            dwellRef.current = { choice: null, startedAt: 0, lastSeenAt: 0 };
            setPendingChoice(null);
            setHoldProgress(0);
          } else {
            setBothHandsWarn(false);
            const detectedChoice: 'yes' | 'no' | null = hasNo ? 'no' : hasYes ? 'yes' : null;
            if (detectedChoice) {
              if (dwellRef.current.choice !== detectedChoice) {
                dwellRef.current = { choice: detectedChoice, startedAt: now, lastSeenAt: now };
              } else {
                dwellRef.current.lastSeenAt = now;
              }
              const elapsed = now - dwellRef.current.startedAt;
              const progress = Math.min(1, elapsed / DWELL_CONFIRM_MS);
              setPendingChoice(detectedChoice);
              setHoldProgress(progress);

              if (ctx && canvas && activeAnchor) {
                drawDwellHalo(
                  ctx,
                  activeAnchor.screenX,
                  activeAnchor.screenY,
                  canvas.width,
                  canvas.height,
                  progress,
                  detectedChoice === 'yes' ? '#10b981' : '#f43f5e',
                );
              }

              if (elapsed >= DWELL_CONFIRM_MS) {
                submitAnswerRef.current(detectedChoice);
              }
            } else if (dwellRef.current.choice && now - dwellRef.current.lastSeenAt <= DWELL_GRACE_MS) {
              // preserve progress briefly during minor frame jitter
            } else {
              dwellRef.current = { choice: null, startedAt: 0, lastSeenAt: 0 };
              setPendingChoice(null);
              setHoldProgress(0);
            }
          }
        }

        rafId = requestAnimationFrame(loop);
      };
      rafId = requestAnimationFrame(loop);
    }

    init();
    return () => {
      active = false;
      cancelAnimationFrame(rafId);
      handLandmarker?.close();
      poseLandmarker?.close();
      stopCameraTracks();
      voiceRef.current.flush();
    };
  }, [completedSummary, sessionKey, stopCameraTracks]);

  useEffect(() => {
    return () => {
      if (timerRef.current) window.clearTimeout(timerRef.current);
      stopCameraTracks();
    };
  }, [stopCameraTracks]);

  const handleReplay = () => {
    if (timerRef.current) window.clearTimeout(timerRef.current);
    verifierRef.current.reset();
    dwellRef.current = { choice: null, startedAt: 0, lastSeenAt: 0 };
    submittingRef.current = false;
    startedAtRef.current = Date.now();
    setCards(sampleTenCognitiveCards());
    setIndex(0);
    setScore(0);
    setFeedback(null);
    setHoldProgress(0);
    setPendingChoice(null);
    setBothHandsWarn(false);
    setCropWarn(false);
    setCompletedSummary(null);
    setSessionKey((k) => k + 1);
  };

  if (completedSummary) {
    return (
      <div className="mx-auto max-w-2xl rounded-3xl bg-white p-8 shadow-2xl border border-slate-200 text-center space-y-6">
        <div className="mx-auto flex h-20 w-20 items-center justify-center rounded-full bg-emerald-50 text-emerald-600 shadow-inner">
          <Award className="h-11 w-11" />
        </div>
        <div>
          <span className="inline-flex items-center gap-1.5 rounded-full bg-indigo-50 px-3 py-1 text-xs font-semibold text-indigo-700">
            <Sparkles className="h-3.5 w-3.5" /> {patientName}
          </span>
          <h3 className="mt-3 text-2xl font-black text-slate-900">สรุปผลการเล่นเกมการ์ดกายภาพบำบัด</h3>
          <p className="mt-1 text-sm text-slate-500">
            {savingResult ? 'กำลังบันทึกคะแนนลงฐานข้อมูล Supabase...' : 'บันทึกผลการฝึกครบทั้ง 10 ข้อเรียบร้อยแล้ว'}
          </p>
        </div>

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          <div className="rounded-2xl border border-emerald-200 bg-emerald-50 p-4">
            <p className="text-xs font-semibold text-emerald-700">คะแนนที่ได้</p>
            <p className="mt-1 text-3xl font-black text-emerald-900">
              {completedSummary.correct} / {completedSummary.total}
            </p>
          </div>
          <div className="rounded-2xl border border-indigo-200 bg-indigo-50 p-4">
            <p className="text-xs font-semibold text-indigo-700">ความแม่นยำ</p>
            <p className="mt-1 text-3xl font-black text-indigo-900">{completedSummary.accuracyPct}%</p>
          </div>
          <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
            <p className="text-xs font-semibold text-slate-600 inline-flex items-center gap-1">
              <Clock className="h-3.5 w-3.5" /> เวลาที่ใช้
            </p>
            <p className="mt-1 text-3xl font-black text-slate-900">{completedSummary.durationSec} วินาที</p>
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-center gap-3 pt-2">
          <button
            type="button"
            onClick={handleReplay}
            className="inline-flex items-center gap-2 rounded-2xl border border-indigo-200 bg-indigo-50 px-6 py-3 text-sm font-bold text-indigo-700 hover:bg-indigo-100 active:scale-95 transition cursor-pointer"
          >
            <RotateCcw className="h-4 w-4" /> เล่นใหม่อีกครั้ง
          </button>
          <button
            type="button"
            onClick={() => {
              stopCameraTracks();
              onComplete(completedSummary);
            }}
            className="inline-flex items-center gap-2 rounded-2xl bg-emerald-600 px-6 py-3 text-sm font-bold text-white shadow-lg shadow-emerald-600/20 hover:bg-emerald-700 active:scale-95 transition cursor-pointer"
          >
            <CheckCircle2 className="h-4 w-4" /> ปิดหน้าต่าง
          </button>
        </div>
      </div>
    );
  }

  if (!currentCard) return null;

  return (
    <div className="mx-auto max-w-5xl rounded-3xl bg-white p-6 shadow-2xl border border-slate-200">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 pb-4">
        <div>
          <span className="inline-flex items-center gap-1.5 rounded-full bg-indigo-50 px-3 py-1 text-xs font-semibold text-indigo-700">
            <Sparkles className="h-3.5 w-3.5" /> {t('startCognitiveGameBtn')} • {patientName}
          </span>
          <p className="mt-1 text-xs text-slate-500">
            ยกมือค้าง 0.8 วินาทีเพื่อตอบ (ซ้าย = ไม่ใช่ / ขวา = ใช่) หรือคลิกปุ่มคำตอบได้ทันที
          </p>
        </div>
        <span className="rounded-xl bg-slate-100 px-3 py-1.5 text-sm font-bold text-slate-700">
          ข้อที่ {index + 1} / {cards.length} • คะแนน {score}
        </span>
      </div>

      <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="flex flex-col justify-between rounded-3xl bg-gradient-to-br from-indigo-50 via-white to-sky-50 p-6 border border-indigo-100">
          <div className="text-center">
            <span className="inline-block rounded-full bg-white px-3 py-1 text-xs font-semibold uppercase tracking-wider text-indigo-600 shadow-sm">
              {currentCard.category.replace('_', ' ')}
            </span>
            <div className="my-6 text-7xl select-none">{currentCard.icon}</div>
            <h2 className="font-bold text-slate-900 leading-snug" style={questionFontStyle}>
              {pick(currentCard.question)}
            </h2>
          </div>

          {feedback ? (
            <div
              className={`mt-6 flex items-center gap-3 rounded-2xl p-4 text-sm font-semibold ${
                feedback.correct ? 'bg-emerald-100 text-emerald-900' : 'bg-rose-100 text-rose-900'
              }`}
            >
              {feedback.correct ? <CheckCircle2 className="h-6 w-6 shrink-0" /> : <XCircle className="h-6 w-6 shrink-0" />}
              <span>{feedback.message}</span>
            </div>
          ) : (
            <div className="mt-6 grid grid-cols-2 gap-3 text-center">
              <button
                type="button"
                onClick={() => submitAnswer('no')}
                className={`rounded-2xl border p-3.5 active:scale-95 transition cursor-pointer ${
                  pendingChoice === 'no'
                    ? 'border-rose-500 bg-rose-100 ring-2 ring-rose-400 shadow-md'
                    : 'border-rose-200 bg-rose-50 hover:bg-rose-100'
                }`}
              >
                <p className="text-sm font-bold text-rose-700">{t('minigameLeftNo')}</p>
              </button>
              <button
                type="button"
                onClick={() => submitAnswer('yes')}
                className={`rounded-2xl border p-3.5 active:scale-95 transition cursor-pointer ${
                  pendingChoice === 'yes'
                    ? 'border-emerald-500 bg-emerald-100 ring-2 ring-emerald-400 shadow-md'
                    : 'border-emerald-200 bg-emerald-50 hover:bg-emerald-100'
                }`}
              >
                <p className="text-sm font-bold text-emerald-700">{t('minigameRightYes')}</p>
              </button>
            </div>
          )}
        </div>

        <div className="relative overflow-hidden rounded-3xl bg-slate-900 aspect-[4/3]">
          <video ref={videoRef} playsInline muted className="h-full w-full object-cover -scale-x-100" />
          <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full object-cover" />

          {!cameraError && (
            <div className="pointer-events-none absolute top-3 right-3 z-10">
              {pendingChoice === 'yes' ? (
                <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-600/95 px-3 py-1.5 text-xs font-bold text-white shadow-lg backdrop-blur">
                  <span className="h-2 w-2 rounded-full bg-white animate-pulse" />
                  ตรวจพบมือขวา (ใช่)... {Math.round(holdProgress * 100)}%
                </span>
              ) : pendingChoice === 'no' ? (
                <span className="inline-flex items-center gap-1.5 rounded-full bg-rose-600/95 px-3 py-1.5 text-xs font-bold text-white shadow-lg backdrop-blur">
                  <span className="h-2 w-2 rounded-full bg-white animate-pulse" />
                  ตรวจพบมือซ้าย (ไม่ใช่)... {Math.round(holdProgress * 100)}%
                </span>
              ) : (
                <span className="inline-flex items-center gap-1.5 rounded-full bg-slate-950/80 px-3 py-1.5 text-xs font-semibold text-slate-100 shadow backdrop-blur border border-white/10">
                  <span className={`h-2 w-2 rounded-full ${detectorReady ? 'bg-emerald-400' : 'bg-amber-400 animate-pulse'}`} />
                  {detectorReady ? 'พร้อมตรวจจับมือ' : 'กำลังเตรียมระบบตรวจจับมือ...'}
                </span>
              )}
            </div>
          )}

          {cameraError && (
            <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-slate-950/90 p-6 text-center text-amber-200 text-sm">
              <Hand className="h-10 w-10 mb-3 text-amber-400" />
              <p className="max-w-xs font-semibold leading-relaxed">{cameraError}</p>
            </div>
          )}

          {bothHandsWarn && (
            <div className="absolute top-12 inset-x-3 z-10 flex items-center gap-2 rounded-xl bg-amber-500/95 px-4 py-2 text-xs font-bold text-slate-950 shadow-lg">
              <AlertTriangle className="h-4 w-4 shrink-0" />
              <span>{t('bothHandsWarning')}</span>
            </div>
          )}

          {cropWarn && !bothHandsWarn && (
            <div className="absolute top-12 inset-x-3 z-10 flex items-center gap-2 rounded-xl bg-sky-500/95 px-4 py-2 text-xs font-bold text-white shadow-lg">
              <AlertTriangle className="h-4 w-4 shrink-0" />
              <span>{t('portraitCropWarning')}</span>
            </div>
          )}

          {!cameraError && !pendingChoice && (
            <div className="pointer-events-none absolute bottom-3 inset-x-3 flex items-center justify-between text-[11px] font-bold">
              <span className="rounded-lg bg-rose-950/75 px-2.5 py-1 text-rose-200 border border-rose-400/30">
                ← ซ้าย: ไม่ใช่ (NO)
              </span>
              <span className="rounded-lg bg-emerald-950/75 px-2.5 py-1 text-emerald-200 border border-emerald-400/30">
                ขวา: ใช่ (YES) →
              </span>
            </div>
          )}

          {pendingChoice && (
            <div className="absolute bottom-4 inset-x-4 z-10 rounded-2xl bg-slate-950/85 p-3 text-white backdrop-blur border border-white/10">
              <div className="flex items-center justify-between text-xs font-semibold mb-1.5">
                <span>{pendingChoice === 'yes' ? t('minigameRightYes') : t('minigameLeftNo')}</span>
                <span>{Math.round(holdProgress * 100)}%</span>
              </div>
              <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-700">
                <div
                  className={`h-full transition-all duration-75 ${
                    pendingChoice === 'yes' ? 'bg-emerald-400' : 'bg-rose-500'
                  }`}
                  style={{ width: `${Math.round(holdProgress * 100)}%` }}
                />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
