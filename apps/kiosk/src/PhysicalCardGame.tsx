import React, { useEffect, useMemo, useRef, useState } from 'react';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import { useClinicalI18n } from '@drtechcares/clinical-i18n';
import {
  AudioChimeSynthesizer,
  HandGestureVerifier,
  VoiceFeedbackController,
  type DetectedHandFrame,
} from '@drtechcares/kiosk-input';
import { selectClinicalCameraDeviceId } from '@drtechcares/face-client';
import { AlertTriangle, CheckCircle2, Hand, Sparkles, XCircle } from 'lucide-react';
import { sampleTenCognitiveCards, type CognitiveCardQuestion } from './PhysicalCardGameData';

interface PhysicalCardGameProps {
  patientName: string;
  onComplete: (summary: { correct: number; total: number; category: string }) => void;
}

export function PhysicalCardGame({ patientName, onComplete }: PhysicalCardGameProps) {
  const { lang, t, pick } = useClinicalI18n();
  const [cards] = useState<CognitiveCardQuestion[]>(() => sampleTenCognitiveCards());
  const [index, setIndex] = useState(0);
  const [score, setScore] = useState(0);
  const [feedback, setFeedback] = useState<{ correct: boolean; message: string } | null>(null);
  const [holdProgress, setHoldProgress] = useState(0);
  const [pendingChoice, setPendingChoice] = useState<'yes' | 'no' | null>(null);
  const [bothHandsWarn, setBothHandsWarn] = useState(false);
  const [cropWarn, setCropWarn] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const verifierRef = useRef(new HandGestureVerifier());
  const voiceRef = useRef(new VoiceFeedbackController());
  const chimeRef = useRef(new AudioChimeSynthesizer());
  const feedbackRef = useRef(feedback);
  feedbackRef.current = feedback;
  const indexRef = useRef(index);
  indexRef.current = index;
  const scoreRef = useRef(score);
  scoreRef.current = score;

  const currentCard = cards[index];

  const questionFontStyle = useMemo(() => {
    const text = currentCard ? pick(currentCard.question) : '';
    const size = text.length > 44 ? 'clamp(1.1rem, 2.3vw, 1.55rem)' : 'clamp(1.3rem, 2.8vw, 1.9rem)';
    return { fontSize: size };
  }, [currentCard, pick]);

  const submitAnswer = (choice: 'yes' | 'no') => {
    const card = cards[indexRef.current];
    if (!card || feedbackRef.current) return;
    const isCorrect = choice === card.correctAnswer;
    chimeRef.current.playTone(isCorrect ? 'correct' : 'wrong');
    const nextScore = isCorrect ? scoreRef.current + 1 : scoreRef.current;
    if (isCorrect) setScore(nextScore);
    const msg = pick(card.explanation);
    setFeedback({ correct: isCorrect, message: msg });
    voiceRef.current.speak(msg, lang, true);

    setTimeout(() => {
      verifierRef.current.reset();
      setFeedback(null);
      setHoldProgress(0);
      setPendingChoice(null);
      if (indexRef.current + 1 >= cards.length) {
        onComplete({ correct: nextScore, total: cards.length, category: 'mixed-99' });
      } else {
        setIndex((i) => i + 1);
      }
    }, 1600);
  };

  useEffect(() => {
    let active = true;
    let stream: MediaStream | null = null;
    let rafId = 0;
    let handLandmarker: HandLandmarker | null = null;

    async function init() {
      try {
        const deviceId = await selectClinicalCameraDeviceId();
        stream = await navigator.mediaDevices.getUserMedia({
          video: deviceId ? { deviceId: { exact: deviceId }, width: 640, height: 480 } : { width: 640, height: 480 },
          audio: false,
        });
        if (!active || !videoRef.current) return;
        videoRef.current.srcObject = stream;
        await videoRef.current.play();

        const vision = await FilesetResolver.forVisionTasks('/mediapipe/wasm');
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: '/models/hand_landmarker.task' },
          runningMode: 'VIDEO',
          numHands: 2,
        });

        const loop = () => {
          if (!active || !videoRef.current || !handLandmarker) return;
          if (videoRef.current.readyState >= 2 && !feedbackRef.current) {
            const now = performance.now();
            const res = handLandmarker.detectForVideo(videoRef.current, now);
            const hands: DetectedHandFrame[] = (res.landmarks ?? []).map((lms, idx) => ({
              landmarks: lms,
              handedness: (res.handedness?.[idx]?.[0]?.categoryName === 'Left' ? 'Left' : 'Right') as 'Left' | 'Right',
              score: res.handedness?.[idx]?.[0]?.score ?? 0.9,
            }));

            const isPortrait = typeof window !== 'undefined' && window.innerHeight > window.innerWidth;
            const out = verifierRef.current.update(hands, now, { isPortrait, cropRatio: isPortrait ? 0.35 : 0 });
            setHoldProgress(out.holdProgress);
            setPendingChoice(out.choice);
            setBothHandsWarn(out.bothHandsRaisedWarning);
            setCropWarn(out.outsidePortraitCropWarning);

            if (out.confirmedAnswer) {
              submitAnswer(out.confirmedAnswer);
            }
          }
          rafId = requestAnimationFrame(loop);
        };
        rafId = requestAnimationFrame(loop);
      } catch (err) {
        setCameraError(err instanceof Error ? err.message : 'Hand camera/model initialization error');
      }
    }

    init();
    return () => {
      active = false;
      cancelAnimationFrame(rafId);
      handLandmarker?.close();
      stream?.getTracks().forEach((tr) => tr.stop());
      voiceRef.current.flush();
    };
  }, []);

  if (!currentCard) return null;

  return (
    <div className="mx-auto max-w-5xl rounded-3xl bg-white p-6 shadow-2xl border border-slate-200">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 pb-4">
        <div>
          <span className="inline-flex items-center gap-1.5 rounded-full bg-indigo-50 px-3 py-1 text-xs font-semibold text-indigo-700">
            <Sparkles className="h-3.5 w-3.5" /> {t('startCognitiveGameBtn')} • {patientName}
          </span>
          <p className="mt-1 text-xs text-slate-500">{t('cleanCameraBadge')}</p>
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
                className="rounded-2xl border border-rose-200 bg-rose-50 p-3 hover:bg-rose-100 active:scale-95 transition cursor-pointer"
              >
                <p className="text-sm font-bold text-rose-700">{t('minigameLeftNo')}</p>
              </button>
              <button
                type="button"
                onClick={() => submitAnswer('yes')}
                className="rounded-2xl border border-emerald-200 bg-emerald-50 p-3 hover:bg-emerald-100 active:scale-95 transition cursor-pointer"
              >
                <p className="text-sm font-bold text-emerald-700">{t('minigameRightYes')}</p>
              </button>
            </div>
          )}
        </div>

        <div className="relative overflow-hidden rounded-3xl bg-slate-900 aspect-[4/3]">
          <video ref={videoRef} playsInline muted className="h-full w-full object-cover -scale-x-100" />
          {cameraError && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-950/90 p-6 text-center text-amber-300 text-sm">
              <Hand className="h-10 w-10 mb-2" />
              <span>{cameraError}</span>
            </div>
          )}

          {bothHandsWarn && (
            <div className="absolute top-3 inset-x-3 flex items-center gap-2 rounded-xl bg-amber-500/95 px-4 py-2 text-xs font-bold text-slate-950 shadow-lg">
              <AlertTriangle className="h-4 w-4 shrink-0" />
              <span>{t('bothHandsWarning')}</span>
            </div>
          )}

          {cropWarn && !bothHandsWarn && (
            <div className="absolute top-3 inset-x-3 flex items-center gap-2 rounded-xl bg-sky-500/95 px-4 py-2 text-xs font-bold text-white shadow-lg">
              <AlertTriangle className="h-4 w-4 shrink-0" />
              <span>{t('portraitCropWarning')}</span>
            </div>
          )}

          {pendingChoice && (
            <div className="absolute bottom-4 inset-x-4 rounded-2xl bg-slate-950/80 p-3 text-white backdrop-blur">
              <div className="flex items-center justify-between text-xs font-semibold mb-1.5">
                <span>{pendingChoice === 'yes' ? t('minigameRightYes') : t('minigameLeftNo')}</span>
                <span>{Math.round(holdProgress * 100)}%</span>
              </div>
              <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-700">
                <div
                  className={`h-full transition-all duration-75 ${
                    pendingChoice === 'yes' ? 'bg-emerald-400' : 'bg-rose-400'
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
