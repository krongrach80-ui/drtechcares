import React, { useCallback, useEffect, useRef, useState } from 'react';
import { FilesetResolver, HandLandmarker, PoseLandmarker, type NormalizedLandmark } from '@mediapipe/tasks-vision';
import { useClinicalI18n } from '@drtechcares/clinical-i18n';
import {
  AudioChimeSynthesizer,
  VoiceFeedbackController,
} from '@drtechcares/kiosk-input';
import { selectClinicalCameraDeviceId } from '@drtechcares/face-client';
import { AlertTriangle, Award, CheckCircle2, Clock, Hand, RotateCcw, Sparkles, XCircle } from 'lucide-react';
import { sampleTenCognitiveCards, type CognitiveCardQuestion } from './PhysicalCardGameData';

const HOLD_DURATION_SEC = 0.8;
const HAND_RELEASE_DECAY_RATE = 2.5;
const HAND_SWITCH_DECAY_RATE = 4.0;
const LANDMARK_SMOOTH_SPEED = 38;
const OFF_CANVAS_WIDTH = 256;

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

type ActiveHandSide = 'LEFT' | 'RIGHT' | null;

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

interface CoverTransform {
  cw: number;
  ch: number;
  renderedW: number;
  renderedH: number;
  offsetX: number;
  offsetY: number;
}

function computeMirroredCoverTransform(cw: number, ch: number, vw: number, vh: number): CoverTransform {
  const safeVw = vw > 0 ? vw : 640;
  const safeVh = vh > 0 ? vh : 480;
  const safeCw = cw > 0 ? cw : 640;
  const safeCh = ch > 0 ? ch : 480;
  const scale = Math.max(safeCw / safeVw, safeCh / safeVh);
  const renderedW = safeVw * scale;
  const renderedH = safeVh * scale;
  return {
    cw: safeCw,
    ch: safeCh,
    renderedW,
    renderedH,
    offsetX: (safeCw - renderedW) / 2,
    offsetY: (safeCh - renderedH) / 2,
  };
}

function mapMirroredLandmarkToScreen(
  lm: { x: number; y: number },
  tf: CoverTransform,
): { px: number; py: number; screenX: number; screenY: number } {
  const px = tf.offsetX + (1 - lm.x) * tf.renderedW;
  const py = tf.offsetY + lm.y * tf.renderedH;
  return {
    px,
    py,
    screenX: px / tf.cw,
    screenY: py / tf.ch,
  };
}

function classifyMirroredHandZone(screenX: number, lockedHand: ActiveHandSide): 'LEFT' | 'RIGHT' {
  const splitThreshold = lockedHand === 'LEFT' ? 0.52 : lockedHand === 'RIGHT' ? 0.48 : 0.5;
  return screenX < splitThreshold ? 'LEFT' : 'RIGHT';
}

function lerpLandmarks(
  prev: NormalizedLandmark[] | null,
  target: NormalizedLandmark[],
  factor: number,
): NormalizedLandmark[] {
  if (!prev || prev.length !== target.length) {
    return target.map((lm) => ({ ...lm }));
  }
  const wristPrev = prev[0];
  const wristNext = target[0];
  if (wristPrev && wristNext) {
    const jump = Math.hypot(wristNext.x - wristPrev.x, wristNext.y - wristPrev.y);
    if (jump > 0.35) {
      return target.map((lm) => ({ ...lm }));
    }
  }
  return target.map((tLm, idx) => {
    const pLm = prev[idx];
    if (!pLm) return { ...tLm };
    return {
      x: pLm.x + (tLm.x - pLm.x) * factor,
      y: pLm.y + (tLm.y - pLm.y) * factor,
      z: pLm.z + (tLm.z - pLm.z) * factor,
      visibility: tLm.visibility,
    };
  });
}

function drawCenterZoneDivider(ctx: CanvasRenderingContext2D, cw: number, ch: number) {
  ctx.save();
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.28)';
  ctx.lineWidth = 3;
  ctx.setLineDash([10, 10]);
  ctx.beginPath();
  ctx.moveTo(cw / 2, 16);
  ctx.lineTo(cw / 2, ch - 16);
  ctx.stroke();
  ctx.restore();
}

function drawHandSkeletonOnCanvas(
  ctx: CanvasRenderingContext2D,
  landmarks: NormalizedLandmark[],
  tf: CoverTransform,
  color: string,
) {
  const pts = landmarks.map((lm) => mapMirroredLandmarkToScreen(lm, tf));
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 6;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
    const start = pts[startIdx];
    const end = pts[endIdx];
    if (!start || !end) continue;
    ctx.beginPath();
    ctx.moveTo(start.px, start.py);
    ctx.lineTo(end.px, end.py);
    ctx.stroke();
  }

  for (let i = 0; i < pts.length; i += 1) {
    const pt = pts[i];
    if (!pt) continue;
    ctx.fillStyle = i === 0 || i === 9 ? '#ffffff' : color;
    ctx.beginPath();
    ctx.arc(pt.px, pt.py, i === 0 || i === 9 ? 8.5 : 5.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#0f172a';
    ctx.lineWidth = 2;
    ctx.stroke();
  }
  ctx.restore();
}

function drawPoseArmSkeletonOnCanvas(
  ctx: CanvasRenderingContext2D,
  landmarks: NormalizedLandmark[],
  connections: ReadonlyArray<[number, number]>,
  tf: CoverTransform,
  color: string,
) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 6;
  ctx.lineCap = 'round';
  const drawnPoints = new Set<number>();

  for (const [a, b] of connections) {
    const start = landmarks[a];
    const end = landmarks[b];
    if (!start || !end) continue;
    if ((start.visibility ?? 1) < 0.25 || (end.visibility ?? 1) < 0.25) continue;
    const pA = mapMirroredLandmarkToScreen(start, tf);
    const pB = mapMirroredLandmarkToScreen(end, tf);
    ctx.beginPath();
    ctx.moveTo(pA.px, pA.py);
    ctx.lineTo(pB.px, pB.py);
    ctx.stroke();
    drawnPoints.add(a);
    drawnPoints.add(b);
  }

  for (const idx of drawnPoints) {
    const lm = landmarks[idx];
    if (!lm) continue;
    const pt = mapMirroredLandmarkToScreen(lm, tf);
    ctx.fillStyle = idx === 15 || idx === 16 ? '#ffffff' : color;
    ctx.beginPath();
    ctx.arc(pt.px, pt.py, 8, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function drawDwellHalo(
  ctx: CanvasRenderingContext2D,
  px: number,
  py: number,
  progress: number,
  color: string,
) {
  if (progress <= 0.01) return;
  ctx.save();
  ctx.beginPath();
  ctx.strokeStyle = 'rgba(255,255,255,0.32)';
  ctx.lineWidth = 7;
  ctx.arc(px, py, 54, 0, Math.PI * 2);
  ctx.stroke();

  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 9;
  ctx.lineCap = 'round';
  ctx.arc(px, py, 54, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * Math.min(1, progress));
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
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const progressBarFillRef = useRef<HTMLDivElement | null>(null);
  const progressPctRef = useRef<HTMLSpanElement | null>(null);
  const badgePctRef = useRef<HTMLSpanElement | null>(null);

  const voiceRef = useRef(new VoiceFeedbackController());
  const chimeRef = useRef(new AudioChimeSynthesizer());

  const activeHandRef = useRef<ActiveHandSide>(null);
  const lockedHandRef = useRef<ActiveHandSide>(null);
  const holdProgressRef = useRef(0);
  const lastFrameTimeRef = useRef(0);
  const lastVideoTimeRef = useRef(-1);
  const lastInferenceTsRef = useRef(0);
  const targetLandmarksRef = useRef<NormalizedLandmark[] | null>(null);
  const smoothedLandmarksRef = useRef<NormalizedLandmark[] | null>(null);
  const poseLandmarksRef = useRef<NormalizedLandmark[] | null>(null);
  const palmPxRef = useRef<{ px: number; py: number } | null>(null);
  const pendingChoiceRef = useRef<'yes' | 'no' | null>(null);
  const cropWarnRef = useRef(false);
  const bothHandsWarnRef = useRef(false);

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

  const resetTrackingRefs = useCallback(() => {
    activeHandRef.current = null;
    lockedHandRef.current = null;
    holdProgressRef.current = 0;
    targetLandmarksRef.current = null;
    smoothedLandmarksRef.current = null;
    poseLandmarksRef.current = null;
    palmPxRef.current = null;
    pendingChoiceRef.current = null;
    bothHandsWarnRef.current = false;
    cropWarnRef.current = false;
    if (progressBarFillRef.current) progressBarFillRef.current.style.width = '0%';
    if (progressPctRef.current) progressPctRef.current.textContent = '0%';
    if (badgePctRef.current) badgePctRef.current.textContent = '0%';
  }, []);

  const currentCard = cards[index];

  const submitAnswer = useCallback(
    (choice: 'yes' | 'no') => {
      const card = cardsRef.current[indexRef.current];
      if (!card || feedbackRef.current || submittingRef.current) return;
      submittingRef.current = true;
      resetTrackingRefs();
      setPendingChoice(null);
      setBothHandsWarn(false);
      setCropWarn(false);

      const isCorrect = choice === card.correctAnswer;
      chimeRef.current.playTone(isCorrect ? 'correct' : 'wrong');
      const nextScore = isCorrect ? scoreRef.current + 1 : scoreRef.current;
      if (isCorrect) setScore(nextScore);
      const msg = pick(card.explanation);
      setFeedback({ correct: isCorrect, message: msg });
      voiceRef.current.speak(msg, lang, true);

      if (timerRef.current) window.clearTimeout(timerRef.current);
      timerRef.current = window.setTimeout(async () => {
        resetTrackingRefs();
        setFeedback(null);
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
              // ignore background sync failures so summary stays responsive
            } finally {
              setSavingResult(false);
            }
          }
        } else {
          setIndex((i) => i + 1);
        }
      }, 1350);
    },
    [lang, onSessionFinished, pick, resetTrackingRefs, stopCameraTracks],
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
    resetTrackingRefs();

    async function createHandOrPoseDetector() {
      const vision = await resolveVisionTasks();
      const handUrls = [
        '/models/hand_landmarker.task',
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
      ];
      for (const modelAssetPath of handUrls) {
        for (const delegate of ['GPU', 'CPU'] as const) {
          try {
            handLandmarker = await HandLandmarker.createFromOptions(vision, {
              baseOptions: { modelAssetPath, delegate },
              runningMode: 'VIDEO',
              numHands: 1,
              minHandDetectionConfidence: 0.4,
              minHandPresenceConfidence: 0.4,
              minTrackingConfidence: 0.4,
            });
            return 'hand' as const;
          } catch {
            // try CPU delegate or next model path
          }
        }
      }

      const poseUrls = ['/models/pose_landmarker.task', '/models/pose_landmarker_full.task'];
      for (const modelAssetPath of poseUrls) {
        for (const delegate of ['GPU', 'CPU'] as const) {
          try {
            poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
              baseOptions: { modelAssetPath, delegate },
              runningMode: 'VIDEO',
              numPoses: 1,
              minPoseDetectionConfidence: 0.45,
              minPosePresenceConfidence: 0.45,
              minTrackingConfidence: 0.45,
            });
            return 'pose' as const;
          } catch {
            // try CPU delegate or next pose model path
          }
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
            ? { deviceId: { exact: deviceId }, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 120 } }
            : { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 120 } },
          audio: false,
        });
      } catch {
        if (active) {
          setCameraError('ไม่สามารถเปิดกล้องได้ กรุณาอนุญาตการใช้กล้อง หรือคลิกปุ่มคำตอบด้านล่างได้เลย');
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
        // video play settles shortly after metadata loads
      }

      const mode = await createHandOrPoseDetector();
      if (!active) return;

      if (mode === 'none') {
        setCameraError('ระบบตรวจจับมือไม่พร้อมใช้งาน กรุณาคลิกเลือกคำตอบด้านล่างแทน');
        return;
      }

      setDetectorReady(true);
      lastFrameTimeRef.current = performance.now();
      lastVideoTimeRef.current = -1;
      lastInferenceTsRef.current = 0;

      const loop = (now: number) => {
        if (!active || !videoRef.current) return;
        const video = videoRef.current;
        const canvas = canvasRef.current;

        const dt = Math.min(0.1, Math.max(0.001, (now - (lastFrameTimeRef.current || now)) / 1000));
        lastFrameTimeRef.current = now;

        const vw = video.videoWidth || 640;
        const vh = video.videoHeight || 480;
        const cw = canvas?.clientWidth || vw;
        const ch = canvas?.clientHeight || vh;
        const dpr = typeof window !== 'undefined' ? Math.min(window.devicePixelRatio || 1, 2) : 1;

        if (canvas) {
          const targetW = Math.max(1, Math.round(cw * dpr));
          const targetH = Math.max(1, Math.round(ch * dpr));
          if (canvas.width !== targetW || canvas.height !== targetH) {
            canvas.width = targetW;
            canvas.height = targetH;
          }
        }

        const tf = computeMirroredCoverTransform(cw, ch, vw, vh);
        const ctx = canvas?.getContext('2d', { desynchronized: true }) ?? null;
        if (ctx) {
          ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
          ctx.clearRect(0, 0, cw, ch);
          drawCenterZoneDivider(ctx, cw, ch);
        }

        const isTrackingPaused = Boolean(feedbackRef.current) || submittingRef.current;
        if (!isTrackingPaused && video.readyState >= 2) {
          const isNewVideoFrame = video.currentTime !== lastVideoTimeRef.current;
          if (isNewVideoFrame) {
            lastVideoTimeRef.current = video.currentTime;

            let offscreen = offscreenCanvasRef.current;
            if (!offscreen) {
              offscreen = document.createElement('canvas');
              offscreenCanvasRef.current = offscreen;
            }
            const offH = Math.max(144, Math.round(OFF_CANVAS_WIDTH * (vh / vw)));
            if (offscreen.width !== OFF_CANVAS_WIDTH || offscreen.height !== offH) {
              offscreen.width = OFF_CANVAS_WIDTH;
              offscreen.height = offH;
            }
            const offCtx = offscreen.getContext('2d', { alpha: false });
            if (offCtx) {
              offCtx.drawImage(video, 0, 0, offscreen.width, offscreen.height);
            }
            const inferenceSource = offCtx ? offscreen : video;
            const inferenceTs = Math.max(lastInferenceTsRef.current + 1, Math.round(now));
            lastInferenceTsRef.current = inferenceTs;
            const isPortrait = typeof window !== 'undefined' && window.innerHeight > window.innerWidth;

            if (mode === 'hand' && handLandmarker) {
              const res = handLandmarker.detectForVideo(inferenceSource, inferenceTs);
              const handLms = res.landmarks?.[0];
              if (handLms && handLms.length >= 21) {
                targetLandmarksRef.current = handLms;
                const wrist = handLms[0];
                const indexMcp = handLms[5];
                const middleMcp = handLms[9];
                const pinkyMcp = handLms[17];
                const middleTip = handLms[12];
                const palmRaw = {
                  x: (wrist.x + indexMcp.x + middleMcp.x + pinkyMcp.x + middleTip.x) / 5,
                  y: (wrist.y + indexMcp.y + middleMcp.y + pinkyMcp.y + middleTip.y) / 5,
                };
                const mappedPalm = mapMirroredLandmarkToScreen(palmRaw, tf);
                let minTipY = 1;
                for (const tipIdx of [4, 8, 12, 16, 20]) {
                  const tip = handLms[tipIdx];
                  if (tip) {
                    const mappedTip = mapMirroredLandmarkToScreen(tip, tf);
                    if (mappedTip.screenY < minTipY) minTipY = mappedTip.screenY;
                  }
                }
                const isRaised = mappedPalm.screenY < 0.78 || minTipY < 0.65;
                palmPxRef.current = { px: mappedPalm.px, py: mappedPalm.py };
                activeHandRef.current = isRaised
                  ? classifyMirroredHandZone(mappedPalm.screenX, lockedHandRef.current)
                  : null;
                bothHandsWarnRef.current = false;

                const nextCropWarn = isPortrait && (mappedPalm.screenX < 0.08 || mappedPalm.screenX > 0.92);
                if (cropWarnRef.current !== nextCropWarn) {
                  cropWarnRef.current = nextCropWarn;
                  setCropWarn(nextCropWarn);
                }
              } else {
                targetLandmarksRef.current = null;
                smoothedLandmarksRef.current = null;
                palmPxRef.current = null;
                activeHandRef.current = null;
                if (cropWarnRef.current) {
                  cropWarnRef.current = false;
                  setCropWarn(false);
                }
              }
            } else if (mode === 'pose' && poseLandmarker) {
              const res = poseLandmarker.detectForVideo(inferenceSource, inferenceTs);
              const pose = res.landmarks?.[0] ?? null;
              poseLandmarksRef.current = pose;
              let detectedLeft = false;
              let detectedRight = false;
              let detectedPalmPx: { px: number; py: number } | null = null;
              let nextCropWarn = false;

              if (pose) {
                const groups = [
                  { shoulder: pose[11], wrist: pose[15], index: pose[19] },
                  { shoulder: pose[12], wrist: pose[16], index: pose[20] },
                ];
                for (const g of groups) {
                  if (!g.wrist || (g.wrist.visibility ?? 1) < 0.35) continue;
                  const palmRaw = {
                    x: g.index && (g.index.visibility ?? 1) > 0.3 ? (g.wrist.x + g.index.x) / 2 : g.wrist.x,
                    y: g.index && (g.index.visibility ?? 1) > 0.3 ? (g.wrist.y + g.index.y) / 2 : g.wrist.y,
                  };
                  const mappedPalm = mapMirroredLandmarkToScreen(palmRaw, tf);
                  const shoulderY = g.shoulder ? mapMirroredLandmarkToScreen(g.shoulder, tf).screenY : 0.72;
                  const isRaised = mappedPalm.screenY < Math.min(0.78, shoulderY + 0.08) || mappedPalm.screenY < 0.65;
                  if (isPortrait && (mappedPalm.screenX < 0.08 || mappedPalm.screenX > 0.92)) {
                    nextCropWarn = true;
                  }
                  if (isRaised) {
                    const side = classifyMirroredHandZone(mappedPalm.screenX, lockedHandRef.current);
                    if (side === 'LEFT') detectedLeft = true;
                    if (side === 'RIGHT') detectedRight = true;
                    detectedPalmPx = { px: mappedPalm.px, py: mappedPalm.py };
                  }
                }
              }

              const nextBothWarn = detectedLeft && detectedRight;
              if (bothHandsWarnRef.current !== nextBothWarn) {
                bothHandsWarnRef.current = nextBothWarn;
                setBothHandsWarn(nextBothWarn);
              }
              if (cropWarnRef.current !== nextCropWarn) {
                cropWarnRef.current = nextCropWarn;
                setCropWarn(nextCropWarn);
              }

              if (nextBothWarn) {
                activeHandRef.current = null;
                palmPxRef.current = null;
              } else {
                activeHandRef.current = detectedLeft ? 'LEFT' : detectedRight ? 'RIGHT' : null;
                palmPxRef.current = detectedPalmPx;
              }
            }
          }

          // Smooth 60fps delta-time hold progress (0.8s confirmation, smooth decay)
          const activeHand = activeHandRef.current;
          if (activeHand) {
            if (!lockedHandRef.current || holdProgressRef.current <= 0.05) {
              lockedHandRef.current = activeHand;
              holdProgressRef.current = Math.min(1, holdProgressRef.current + dt / HOLD_DURATION_SEC);
            } else if (lockedHandRef.current === activeHand) {
              holdProgressRef.current = Math.min(1, holdProgressRef.current + dt / HOLD_DURATION_SEC);
            } else {
              holdProgressRef.current = Math.max(0, holdProgressRef.current - dt * HAND_SWITCH_DECAY_RATE);
              if (holdProgressRef.current === 0) {
                lockedHandRef.current = activeHand;
              }
            }
          } else {
            holdProgressRef.current = Math.max(0, holdProgressRef.current - dt * HAND_RELEASE_DECAY_RATE);
            if (holdProgressRef.current === 0) {
              lockedHandRef.current = null;
            }
          }

          const nextPendingChoice: 'yes' | 'no' | null =
            lockedHandRef.current === 'RIGHT' ? 'yes' : lockedHandRef.current === 'LEFT' ? 'no' : null;
          if (pendingChoiceRef.current !== nextPendingChoice) {
            pendingChoiceRef.current = nextPendingChoice;
            setPendingChoice(nextPendingChoice);
          }

          // Update progress width and percentage directly on DOM at 60fps without React re-renders
          const pctFloat = Math.min(100, Math.max(0, holdProgressRef.current * 100));
          const pctRounded = `${Math.round(pctFloat)}%`;
          if (progressBarFillRef.current) {
            progressBarFillRef.current.style.width = `${pctFloat.toFixed(1)}%`;
          }
          if (progressPctRef.current) {
            progressPctRef.current.textContent = pctRounded;
          }
          if (badgePctRef.current) {
            badgePctRef.current.textContent = pctRounded;
          }

          // Render interpolated 120fps hand or pose skeleton + progress halo directly on canvas
          if (ctx) {
            const smoothAlpha = 1 - Math.exp(-LANDMARK_SMOOTH_SPEED * dt);
            if (mode === 'hand' && targetLandmarksRef.current) {
              smoothedLandmarksRef.current = lerpLandmarks(
                smoothedLandmarksRef.current,
                targetLandmarksRef.current,
                smoothAlpha,
              );
              const skelColor =
                (activeHand ?? lockedHandRef.current) === 'RIGHT'
                  ? '#10b981'
                  : (activeHand ?? lockedHandRef.current) === 'LEFT'
                    ? '#f43f5e'
                    : '#38bdf8';
              drawHandSkeletonOnCanvas(ctx, smoothedLandmarksRef.current, tf, skelColor);
            } else if (mode === 'pose' && poseLandmarksRef.current && activeHand) {
              const skelColor = activeHand === 'RIGHT' ? '#10b981' : '#f43f5e';
              drawPoseArmSkeletonOnCanvas(ctx, poseLandmarksRef.current, POSE_HAND_CONNECTIONS_A, tf, skelColor);
              drawPoseArmSkeletonOnCanvas(ctx, poseLandmarksRef.current, POSE_HAND_CONNECTIONS_B, tf, skelColor);
            }

            if (palmPxRef.current && holdProgressRef.current > 0.01 && lockedHandRef.current) {
              drawDwellHalo(
                ctx,
                palmPxRef.current.px,
                palmPxRef.current.py,
                holdProgressRef.current,
                lockedHandRef.current === 'RIGHT' ? '#10b981' : '#f43f5e',
              );
            }
          }

          if (holdProgressRef.current >= 1 && lockedHandRef.current) {
            const confirmed: 'yes' | 'no' = lockedHandRef.current === 'RIGHT' ? 'yes' : 'no';
            submitAnswerRef.current(confirmed);
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
  }, [completedSummary, resetTrackingRefs, sessionKey, stopCameraTracks]);

  useEffect(() => {
    return () => {
      if (timerRef.current) window.clearTimeout(timerRef.current);
      stopCameraTracks();
    };
  }, [stopCameraTracks]);

  const handleReplay = () => {
    if (timerRef.current) window.clearTimeout(timerRef.current);
    resetTrackingRefs();
    submittingRef.current = false;
    startedAtRef.current = Date.now();
    setCards(sampleTenCognitiveCards());
    setIndex(0);
    setScore(0);
    setFeedback(null);
    setPendingChoice(null);
    setBothHandsWarn(false);
    setCropWarn(false);
    setCompletedSummary(null);
    setSessionKey((k) => k + 1);
  };

  if (completedSummary) {
    return (
      <div className="w-full flex-1 flex flex-col items-center justify-center rounded-3xl bg-white p-6 sm:p-12 shadow-xl border-2 border-slate-200 text-center space-y-8">
        <div className="mx-auto flex h-32 w-32 items-center justify-center rounded-full bg-emerald-50 text-emerald-600 shadow-inner">
          <Award className="h-20 w-20" />
        </div>
        <div>
          <span className="mg-badge-top inline-flex items-center gap-2.5 rounded-full bg-indigo-50 px-6 py-2.5 font-black text-indigo-700">
            <Sparkles className="h-6 w-6 shrink-0" /> {patientName}
          </span>
          <h3 className="mg-summary-title mt-4 font-black text-slate-900">สรุปผลการเล่นเกมการ์ดกายภาพบำบัด</h3>
          <p className="mg-sub-instruction mt-2 font-bold text-slate-600">
            {savingResult ? 'กำลังบันทึกคะแนนลงฐานข้อมูล Supabase...' : 'บันทึกผลการฝึกครบทั้ง 10 ข้อเรียบร้อยแล้ว'}
          </p>
        </div>

        <div className="w-full max-w-5xl grid grid-cols-1 gap-6 sm:grid-cols-3">
          <div className="rounded-3xl border-2 border-emerald-200 bg-emerald-50 p-6 sm:p-8">
            <p className="mg-summary-stat-label font-extrabold text-emerald-700">คะแนนที่ได้</p>
            <p className="mg-summary-stat-val mt-2 font-black text-emerald-900">
              {completedSummary.correct} / {completedSummary.total}
            </p>
          </div>
          <div className="rounded-3xl border-2 border-indigo-200 bg-indigo-50 p-6 sm:p-8">
            <p className="mg-summary-stat-label font-extrabold text-indigo-700">ความแม่นยำ</p>
            <p className="mg-summary-stat-val mt-2 font-black text-indigo-900">{completedSummary.accuracyPct}%</p>
          </div>
          <div className="rounded-3xl border-2 border-slate-200 bg-slate-50 p-6 sm:p-8">
            <p className="mg-summary-stat-label font-extrabold text-slate-600 inline-flex items-center gap-2">
              <Clock className="h-7 w-7 shrink-0" /> เวลาที่ใช้
            </p>
            <p className="mg-summary-stat-val mt-2 font-black text-slate-900">{completedSummary.durationSec} วินาที</p>
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-center gap-5 pt-2">
          <button
            type="button"
            onClick={handleReplay}
            className="mg-summary-btn inline-flex items-center gap-3 rounded-3xl border-2 border-indigo-200 bg-indigo-50 min-h-[84px] px-10 py-5 font-black text-indigo-700 hover:bg-indigo-100 active:scale-95 transition cursor-pointer"
          >
            <RotateCcw className="h-8 w-8 shrink-0" /> เล่นใหม่อีกครั้ง
          </button>
          <button
            type="button"
            onClick={() => {
              stopCameraTracks();
              onComplete(completedSummary);
            }}
            className="mg-summary-btn inline-flex items-center gap-3 rounded-3xl bg-emerald-600 min-h-[84px] px-10 py-5 font-black text-white shadow-lg shadow-emerald-600/20 hover:bg-emerald-700 active:scale-95 transition cursor-pointer"
          >
            <CheckCircle2 className="h-8 w-8 shrink-0" /> ปิดหน้าต่าง
          </button>
        </div>
      </div>
    );
  }

  if (!currentCard) return null;

  return (
    <div className="w-full flex-1 flex flex-col justify-between rounded-3xl bg-white p-4 sm:p-7 shadow-xl border-2 border-slate-200 gap-4 sm:gap-5">
      {/* Header Section */}
      <div className="shrink-0 flex flex-col gap-3 border-b-2 border-slate-100 pb-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <span className="mg-badge-top inline-flex items-center gap-2.5 rounded-full bg-indigo-50 px-5 py-2.5 font-black text-indigo-700">
            <Sparkles className="h-6 w-6 shrink-0" /> {t('startCognitiveGameBtn')} • {patientName}
          </span>
          <span className="mg-score-pill rounded-2xl bg-slate-900 px-7 py-3.5 font-black text-white shadow-md whitespace-nowrap">
            ข้อที่ {index + 1} / {cards.length} • คะแนน {score}
          </span>
        </div>
        <p className="mg-sub-instruction font-extrabold text-slate-700">
          ยกมือค้าง 0.8 วินาทีเพื่อตอบ (ซ้าย = ไม่ใช่ / ขวา = ใช่) หรือคลิกปุ่มคำตอบได้ทันที
        </p>
      </div>

      {/* Main Full-Size Vertical Stack: Top = Question & Illustration, Middle = Camera, Bottom = Answer Buttons */}
      <div className="flex-1 flex flex-col justify-between gap-4 sm:gap-5 min-h-0">
        {/* Top Card: Question & Illustration */}
        <div className="flex-[1] min-h-[240px] sm:min-h-[290px] rounded-3xl bg-gradient-to-br from-indigo-50 via-white to-sky-50 p-6 sm:p-8 border-2 border-indigo-200 text-center flex flex-col items-center justify-center shadow-sm">
          <span className="mg-category-pill inline-block rounded-full bg-white px-7 py-2.5 font-black uppercase tracking-wider text-indigo-700 shadow-sm border-2 border-indigo-100">
            {currentCard.category.replace('_', ' ')}
          </span>
          <div className="mg-icon-hero my-3 sm:my-4 select-none">
            {currentCard.icon}
          </div>
          <h2 className="mg-question-title font-black text-slate-950 max-w-5xl">
            {pick(currentCard.question)}
          </h2>
        </div>

        {/* Middle Card: Full-Size Live Camera Viewport (120Hz Hand Tracking) */}
        <div className="flex-[1.4] min-h-[360px] sm:min-h-[480px] relative w-full overflow-hidden rounded-3xl bg-slate-900 border-4 border-slate-800 shadow-inner">
          <video ref={videoRef} playsInline muted className="h-full w-full object-cover -scale-x-100" />
          <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />

          {!cameraError && (
            <div className="pointer-events-none absolute top-4 right-4 z-10">
              {pendingChoice === 'yes' ? (
                <span className="mg-cam-status inline-flex items-center gap-2.5 rounded-full bg-emerald-600/95 px-6 py-3 font-black text-white shadow-xl backdrop-blur">
                  <span className="h-4 w-4 rounded-full bg-white animate-pulse shrink-0" />
                  <span>ตรวจพบมือขวา (ใช่)... </span>
                  <span ref={badgePctRef}>0%</span>
                </span>
              ) : pendingChoice === 'no' ? (
                <span className="mg-cam-status inline-flex items-center gap-2.5 rounded-full bg-rose-600/95 px-6 py-3 font-black text-white shadow-xl backdrop-blur">
                  <span className="h-4 w-4 rounded-full bg-white animate-pulse shrink-0" />
                  <span>ตรวจพบมือซ้าย (ไม่ใช่)... </span>
                  <span ref={badgePctRef}>0%</span>
                </span>
              ) : (
                <span className="mg-cam-status inline-flex items-center gap-2.5 rounded-full bg-slate-950/85 px-6 py-3 font-black text-slate-100 shadow-xl backdrop-blur border-2 border-white/15">
                  <span className={`h-4 w-4 rounded-full shrink-0 ${detectorReady ? 'bg-emerald-400' : 'bg-amber-400 animate-pulse'}`} />
                  {detectorReady ? 'พร้อมตรวจจับมือ 120Hz' : 'กำลังเตรียมระบบตรวจจับมือ...'}
                </span>
              )}
            </div>
          )}

          {cameraError && (
            <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-slate-950/90 p-8 text-center">
              <Hand className="h-24 w-24 mb-5 text-amber-400" />
              <p className="mg-cam-status max-w-2xl font-black text-amber-200">{cameraError}</p>
            </div>
          )}

          {bothHandsWarn && (
            <div className="mg-cam-status absolute top-20 inset-x-4 z-10 flex items-center gap-3 rounded-2xl bg-amber-500/95 px-6 py-3.5 font-black text-slate-950 shadow-xl">
              <AlertTriangle className="h-8 w-8 shrink-0" />
              <span>{t('bothHandsWarning')}</span>
            </div>
          )}

          {cropWarn && !bothHandsWarn && (
            <div className="mg-cam-status absolute top-20 inset-x-4 z-10 flex items-center gap-3 rounded-2xl bg-sky-500/95 px-6 py-3.5 font-black text-white shadow-xl">
              <AlertTriangle className="h-8 w-8 shrink-0" />
              <span>{t('portraitCropWarning')}</span>
            </div>
          )}

          {!cameraError && !pendingChoice && (
            <div className="pointer-events-none absolute bottom-4 inset-x-4 flex items-center justify-between gap-3">
              <span className="mg-cam-zone rounded-2xl bg-rose-950/85 px-6 py-3 font-black text-rose-100 border-2 border-rose-400/50 shadow-lg">
                ← ซ้าย: ไม่ใช่ (NO)
              </span>
              <span className="mg-cam-zone rounded-2xl bg-emerald-950/85 px-6 py-3 font-black text-emerald-100 border-2 border-emerald-400/50 shadow-lg">
                ขวา: ใช่ (YES) →
              </span>
            </div>
          )}

          {pendingChoice && (
            <div className="absolute bottom-4 inset-x-4 z-10 rounded-2xl bg-slate-950/90 p-5 sm:p-6 text-white backdrop-blur border-2 border-white/20 shadow-2xl">
              <div className="mg-hold-label flex items-center justify-between font-black mb-2.5">
                <span>{pendingChoice === 'yes' ? t('minigameRightYes') : t('minigameLeftNo')}</span>
                <span ref={progressPctRef}>0%</span>
              </div>
              <div className="h-7 sm:h-8 w-full overflow-hidden rounded-full bg-slate-700">
                <div
                  ref={progressBarFillRef}
                  className={`h-full will-change-[width] ${
                    pendingChoice === 'yes' ? 'bg-emerald-400' : 'bg-rose-500'
                  }`}
                  style={{ width: '0%' }}
                />
              </div>
            </div>
          )}
        </div>

        {/* Bottom Controls: Feedback Banner or Giant YES/NO Answer Buttons */}
        <div className="shrink-0">
          {feedback ? (
            <div
              className={`mg-feedback-banner min-h-[108px] sm:min-h-[132px] flex items-center justify-center gap-4 rounded-3xl border-4 p-6 font-black text-center ${
                feedback.correct
                  ? 'border-emerald-300 bg-emerald-100 text-emerald-950'
                  : 'border-rose-300 bg-rose-100 text-rose-950'
              }`}
            >
              {feedback.correct ? (
                <CheckCircle2 className="h-12 w-12 shrink-0 text-emerald-700" />
              ) : (
                <XCircle className="h-12 w-12 shrink-0 text-rose-700" />
              )}
              <span>{feedback.message}</span>
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-4 sm:gap-6 text-center">
              <button
                type="button"
                onClick={() => submitAnswer('no')}
                className={`min-h-[108px] sm:min-h-[132px] rounded-3xl border-4 p-5 sm:p-7 flex items-center justify-center active:scale-95 transition cursor-pointer ${
                  pendingChoice === 'no'
                    ? 'border-rose-500 bg-rose-100 ring-4 ring-rose-300 shadow-lg'
                    : 'border-rose-300 bg-rose-50 hover:bg-rose-100'
                }`}
              >
                <p className="mg-answer-btn font-black text-rose-700">
                  {t('minigameLeftNo')}
                </p>
              </button>
              <button
                type="button"
                onClick={() => submitAnswer('yes')}
                className={`min-h-[108px] sm:min-h-[132px] rounded-3xl border-4 p-5 sm:p-7 flex items-center justify-center active:scale-95 transition cursor-pointer ${
                  pendingChoice === 'yes'
                    ? 'border-emerald-500 bg-emerald-100 ring-4 ring-emerald-300 shadow-lg'
                    : 'border-emerald-300 bg-emerald-50 hover:bg-emerald-100'
                }`}
              >
                <p className="mg-answer-btn font-black text-emerald-700">
                  {t('minigameRightYes')}
                </p>
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
