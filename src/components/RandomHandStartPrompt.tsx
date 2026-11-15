import React, { useEffect, useRef, useState } from 'react';
import {
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
  type NormalizedLandmark,
} from '@mediapipe/tasks-vision';
import { selectClinicalCameraDeviceId } from '@drtechcares/face-client';
import { useClinicalI18n, type ClinicalLang } from '@drtechcares/clinical-i18n';
import { extractRaisedHandSides } from '@drtechcares/kiosk-input';
import type { PoseLandmark } from '@drtechcares/pose-engine';

const RANDOM_HAND_COPY: Record<
  ClinicalLang,
  {
    openingCamera: string;
    errUnsupported: string;
    errOpenFailed: string;
    matchedBadge: (pct: number) => string;
    wrongSideBadge: string;
    preparingBadge: string;
    holdPromptBadge: string;
    confirmingBtn: (pct: number) => string;
    idleBtn: string;
  }
> = {
  th: {
    openingCamera: 'กำลังเปิดกล้องตรวจจับการยกมือ...',
    errUnsupported: 'เบราว์เซอร์ไม่รองรับกล้อง สามารถกดปุ่มด้านล่างเพื่อเริ่มกายภาพได้ทันที',
    errOpenFailed: 'ไม่สามารถเปิดกล้องได้ — สามารถกดปุ่มด้านล่างเพื่อเริ่มการกายภาพได้ทันที',
    matchedBadge: (pct) => `✅ ตรวจพบการยกมือถูกต้องแล้ว! ค้างไว้ ${pct}%`,
    wrongSideBadge: '⚠️ กรุณายกมืออีกข้างตามที่ระบุไว้ด้านบน',
    preparingBadge: '⏳ กำลังเตรียมระบบตรวจจับการยกมือ...',
    holdPromptBadge: '📷 ชูมือค้างไว้หน้ากล้องตามคำสั่งด้านบนเพื่อเริ่มอัตโนมัติ',
    confirmingBtn: (pct) => `🖐️ กำลังยืนยันท่าทาง... (${pct}%)`,
    idleBtn: '🖐️ ยกมือตามคำสั่งด้านบน หรือ กดที่นี่เพื่อเริ่มการกายภาพบำบัด ➔',
  },
  en: {
    openingCamera: 'Opening camera for hand-raise detection...',
    errUnsupported: 'Camera unavailable in browser. Tap below to start physical therapy.',
    errOpenFailed: 'Unable to open camera — you can tap the button below to start immediately.',
    matchedBadge: (pct) => `✅ Correct hand raised! Hold steady ${pct}%`,
    wrongSideBadge: '⚠️ Please raise the other hand as shown above',
    preparingBadge: '⏳ Preparing hand gesture detector...',
    holdPromptBadge: '📷 Hold your hand up as instructed above to start automatically',
    confirmingBtn: (pct) => `🖐️ Confirming gesture... (${pct}%)`,
    idleBtn: '🖐️ Raise the instructed hand or tap here to start rehabilitation ➔',
  },
  cn: {
    openingCamera: '正在开启举手识别摄像头...',
    errUnsupported: '当前浏览器不支持摄像头，可直接点击下方按钮开始康复训练。',
    errOpenFailed: '无法开启摄像头 — 请直接点击下方按钮开始物理治疗训练。',
    matchedBadge: (pct) => `✅ 已识别到正确手部！请保持 ${pct}%`,
    wrongSideBadge: '⚠️ 请举起上方指示的另一只手',
    preparingBadge: '⏳ 正在准备手势识别系统...',
    holdPromptBadge: '📷 请按上方提示举手并保持，系统将自动开始训练',
    confirmingBtn: (pct) => `🖐️ 正在确认手势... (${pct}%)`,
    idleBtn: '🖐️ 请按上方提示举手 或 点击此处开始物理治疗训练 ➔',
  },
};

export type HandChallengeSide = 'left' | 'right';

export interface RandomHandEvaluationResult {
  leftRaised: boolean;
  rightRaised: boolean;
  targetMatched: boolean;
  wrongSideRaised: boolean;
}

const HAND_HOLD_CONFIRM_MS = 420;
const OFF_CANVAS_WIDTH = 256;

export function pickRandomHandSide(): HandChallengeSide {
  return Math.random() < 0.5 ? 'left' : 'right';
}

/**
 * Evaluates raised hand landmarks / pose landmarks against a target side ('left' or 'right').
 * Uses raw front-camera coordinates before CSS `scaleX(-1)` mirroring:
 * `x >= 0.5` corresponds to the user's left hand on screen; `x < 0.5` corresponds to the right hand.
 */
export function evaluateRandomHandChallenge(
  hands: ReadonlyArray<ReadonlyArray<{ x: number; y: number }>>,
  poseLandmarks: readonly PoseLandmark[] | null | undefined,
  targetSide: HandChallengeSide,
): RandomHandEvaluationResult {
  const validHands = hands.filter((hand) => {
    if (hand.length < 21) return false;
    const wrist = hand[0];
    return Boolean(wrist && wrist.y <= 0.95);
  });

  let leftRaised = false;
  let rightRaised = false;

  if (validHands.length === 1) {
    const singleHand = validHands[0];
    const anchorX = (singleHand[9] ?? singleHand[0]).x;
    if (anchorX >= 0.5) {
      leftRaised = true;
    } else {
      rightRaised = true;
    }
  } else {
    const sides = extractRaisedHandSides(hands, poseLandmarks);
    leftRaised = sides.leftRaised;
    rightRaised = sides.rightRaised;
  }

  const targetRaised = targetSide === 'left' ? leftRaised : rightRaised;
  const oppositeRaised = targetSide === 'left' ? rightRaised : leftRaised;

  return {
    leftRaised,
    rightRaised,
    targetMatched: targetRaised,
    wrongSideRaised: !targetRaised && oppositeRaised,
  };
}

interface RandomHandStartPromptProps {
  onConfirmStart: () => void;
  initialSide?: HandChallengeSide;
  targetSide?: HandChallengeSide;
}

export function RandomHandStartPrompt({
  onConfirmStart,
  initialSide,
  targetSide: controlledTargetSide,
}: RandomHandStartPromptProps) {
  const { lang } = useClinicalI18n();
  const rhCopy = RANDOM_HAND_COPY[lang];
  const [internalSide] = useState<HandChallengeSide>(
    () => initialSide ?? pickRandomHandSide(),
  );
  const targetSide = controlledTargetSide ?? internalSide;
  const [cameraReady, setCameraReady] = useState(false);
  const [detectorReady, setDetectorReady] = useState(false);
  const [cameraErrorKey, setCameraErrorKey] = useState<'unsupported' | 'open_failed' | null>(null);
  const [holdProgress, setHoldProgress] = useState(0);
  const [evalState, setEvalState] = useState<RandomHandEvaluationResult>({
    leftRaised: false,
    rightRaised: false,
    targetMatched: false,
    wrongSideRaised: false,
  });

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const targetSideRef = useRef<HandChallengeSide>(targetSide);
  const onConfirmStartRef = useRef(onConfirmStart);
  const confirmedRef = useRef(false);

  targetSideRef.current = targetSide;
  onConfirmStartRef.current = onConfirmStart;

  useEffect(() => {
    setHoldProgress(0);
    setEvalState({
      leftRaised: false,
      rightRaised: false,
      targetMatched: false,
      wrongSideRaised: false,
    });
  }, [targetSide]);

  useEffect(() => {
    let active = true;
    let stream: MediaStream | null = null;
    let rafId = 0;
    let handLandmarker: HandLandmarker | null = null;
    let poseLandmarker: PoseLandmarker | null = null;
    let offscreenCanvas: HTMLCanvasElement | null = null;
    let lastVideoTime = -1;
    let lastInferenceAt = 0;
    let lastDetectTs = 0;
    let lastFrameAt = 0;
    let holdMs = 0;
    let lastReportedProgress = -1;
    let lastMatched = false;
    let lastWrong = false;
    let prevTargetSide = targetSideRef.current;
    let latestEval: RandomHandEvaluationResult = {
      leftRaised: false,
      rightRaised: false,
      targetMatched: false,
      wrongSideRaised: false,
    };

    const cleanupMedia = () => {
      if (rafId) {
        cancelAnimationFrame(rafId);
        rafId = 0;
      }
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
        stream = null;
      }
      if (handLandmarker) {
        try {
          handLandmarker.close();
        } catch {
          // ignore close errors
        }
        handLandmarker = null;
      }
      if (poseLandmarker) {
        try {
          poseLandmarker.close();
        } catch {
          // ignore close errors
        }
        poseLandmarker = null;
      }
    };

    async function initHandChallengeCamera() {
      try {
        if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
          setCameraErrorKey('unsupported');
          return;
        }

        const deviceId = await selectClinicalCameraDeviceId();
        if (!active) return;

        stream = await navigator.mediaDevices.getUserMedia({
          video: deviceId
            ? {
                deviceId: { exact: deviceId },
                width: { ideal: 1280 },
                height: { ideal: 720 },
                aspectRatio: { ideal: 16 / 9 },
                facingMode: 'user',
              }
            : {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                aspectRatio: { ideal: 16 / 9 },
                facingMode: 'user',
              },
          audio: false,
        });

        if (!active || !videoRef.current) {
          cleanupMedia();
          return;
        }

        const videoTrack = stream.getVideoTracks()[0];
        const caps = videoTrack?.getCapabilities?.() as { zoom?: { min?: number } } | undefined;
        if (videoTrack && caps?.zoom && typeof caps.zoom.min === 'number') {
          await videoTrack
            .applyConstraints({
              advanced: [{ zoom: caps.zoom.min } as MediaTrackConstraintSet],
            })
            .catch(() => undefined);
        }

        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        if (!active) {
          cleanupMedia();
          return;
        }
        setCameraReady(true);

        let vision;
        for (const wasmBase of [
          '/mediapipe/wasm',
          '/models/wasm',
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm',
        ]) {
          try {
            vision = await FilesetResolver.forVisionTasks(wasmBase);
            break;
          } catch {
            // Try next WASM base URL
          }
        }
        if (!vision || !active) return;

        const initHand = async () => {
          for (const delegate of ['CPU', undefined] as const) {
            try {
              const created = await HandLandmarker.createFromOptions(vision, {
                baseOptions: delegate
                  ? { modelAssetPath: '/models/hand_landmarker.task', delegate }
                  : { modelAssetPath: '/models/hand_landmarker.task' },
                runningMode: 'VIDEO',
                numHands: 2,
                minHandDetectionConfidence: 0.35,
                minHandPresenceConfidence: 0.35,
                minTrackingConfidence: 0.35,
              });
              if (!active) {
                created.close();
                return;
              }
              handLandmarker = created;
              setDetectorReady(true);
              return;
            } catch {
              // Try fallback delegate
            }
          }
        };

        const initPose = async () => {
          for (const modelAssetPath of ['/models/pose_landmarker.task', '/models/pose_landmarker_full.task']) {
            for (const delegate of ['CPU', undefined] as const) {
              try {
                const created = await PoseLandmarker.createFromOptions(vision, {
                  baseOptions: delegate ? { modelAssetPath, delegate } : { modelAssetPath },
                  runningMode: 'VIDEO',
                  numPoses: 1,
                  minPoseDetectionConfidence: 0.35,
                  minPosePresenceConfidence: 0.35,
                  minTrackingConfidence: 0.35,
                });
                if (!active) {
                  created.close();
                  return;
                }
                poseLandmarker = created;
                setDetectorReady(true);
                return;
              } catch {
                // Try next pose model / delegate
              }
            }
          }
        };

        await Promise.all([initHand(), initPose()]);
        if (!active) {
          cleanupMedia();
          return;
        }

        const loop = () => {
          if (!active || confirmedRef.current || !videoRef.current) return;
          if (prevTargetSide !== targetSideRef.current) {
            prevTargetSide = targetSideRef.current;
            holdMs = 0;
            lastReportedProgress = -1;
            lastMatched = false;
            lastWrong = false;
          }
          const v = videoRef.current;
          const now = performance.now();
          const dt = lastFrameAt > 0 ? Math.min(100, Math.max(1, now - lastFrameAt)) : 16;
          lastFrameAt = now;

          if (v.readyState >= 2 && v.videoWidth > 0 && (handLandmarker || poseLandmarker)) {
            if (v.currentTime !== lastVideoTime || now - lastInferenceAt >= 33) {
              lastVideoTime = v.currentTime;
              lastInferenceAt = now;

              if (!offscreenCanvas) {
                offscreenCanvas = document.createElement('canvas');
              }
              const offW = OFF_CANVAS_WIDTH;
              const offH = Math.max(144, Math.round(offW * ((v.videoHeight || 480) / (v.videoWidth || 640))));
              if (offscreenCanvas.width !== offW || offscreenCanvas.height !== offH) {
                offscreenCanvas.width = offW;
                offscreenCanvas.height = offH;
              }
              const offCtx = offscreenCanvas.getContext('2d', { alpha: false });
              if (offCtx) {
                offCtx.drawImage(v, 0, 0, offW, offH);
              }
              const inferenceSource = offCtx ? offscreenCanvas : v;

              let detectedHands: NormalizedLandmark[][] = [];
              let detectedPose: NormalizedLandmark[] | null = null;

              if (handLandmarker) {
                try {
                  const ts = Math.max(lastDetectTs + 1, Math.round(now));
                  lastDetectTs = ts;
                  detectedHands = handLandmarker.detectForVideo(inferenceSource, ts).landmarks ?? [];
                } catch {
                  // Ignore transient hand inference error
                }
              }

              if (poseLandmarker) {
                try {
                  const ts = Math.max(lastDetectTs + 1, Math.round(now));
                  lastDetectTs = ts;
                  detectedPose = poseLandmarker.detectForVideo(inferenceSource, ts).landmarks?.[0] ?? null;
                } catch {
                  // Ignore transient pose inference error
                }
              }

              latestEval = evaluateRandomHandChallenge(
                detectedHands,
                detectedPose,
                targetSideRef.current,
              );
            }

            if (latestEval.targetMatched) {
              holdMs = Math.min(HAND_HOLD_CONFIRM_MS, holdMs + dt);
            } else {
              holdMs = Math.max(0, holdMs - dt * 1.1);
            }

            const progress = Math.min(1, holdMs / HAND_HOLD_CONFIRM_MS);
            if (
              latestEval.targetMatched !== lastMatched ||
              latestEval.wrongSideRaised !== lastWrong
            ) {
              lastMatched = latestEval.targetMatched;
              lastWrong = latestEval.wrongSideRaised;
              setEvalState(latestEval);
            }
            if (
              Math.abs(progress - lastReportedProgress) >= 0.03 ||
              (progress === 0 && lastReportedProgress !== 0) ||
              progress >= 1
            ) {
              lastReportedProgress = progress;
              setHoldProgress(progress);
            }

            if (progress >= 1 && !confirmedRef.current) {
              confirmedRef.current = true;
              active = false;
              cleanupMedia();
              onConfirmStartRef.current();
              return;
            }
          }

          if (active) {
            rafId = requestAnimationFrame(loop);
          }
        };

        rafId = requestAnimationFrame(loop);
      } catch {
        if (active) {
          setCameraErrorKey('open_failed');
        }
      }
    }

    void initHandChallengeCamera();

    return () => {
      active = false;
      cleanupMedia();
    };
  }, []);

  const handleManualConfirm = () => {
    if (confirmedRef.current) return;
    confirmedRef.current = true;
    onConfirmStart();
  };

  const isLeft = targetSide === 'left';
  const cameraErrorText =
    cameraErrorKey === 'unsupported'
      ? rhCopy.errUnsupported
      : cameraErrorKey === 'open_failed'
        ? rhCopy.errOpenFailed
        : null;

  return (
    <div
      className={`mt-5 rounded-2xl border-2 p-4 text-center space-y-3 transition-colors ${
        isLeft
          ? 'border-orange-300 bg-orange-50/60'
          : 'border-emerald-300 bg-emerald-50/70'
      }`}
    >
      {/* Live Mirrored Camera Preview (Clean unobstructed frame; target side shown under main card heading) */}
      <div
        className={`relative mx-auto w-full h-72 sm:h-80 md:h-96 rounded-2xl overflow-hidden bg-gradient-to-br from-sky-100 via-emerald-50 to-white border-2 shadow-inner transition-colors ${
          isLeft ? 'border-sky-300' : 'border-emerald-300'
        }`}
      >
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-full object-cover"
          style={{ transform: 'scaleX(-1)' }}
        />

        {!cameraReady && !cameraErrorText && (
          <div className="absolute inset-0 flex items-center justify-center bg-white/90 text-xs sm:text-sm font-bold text-[#0F4C81] px-4">
            {rhCopy.openingCamera}
          </div>
        )}
        {cameraErrorText && (
          <div className="absolute inset-0 flex items-center justify-center bg-white/95 text-xs sm:text-sm font-bold text-[#0284C7] px-4">
            {cameraErrorText}
          </div>
        )}

        <div className="absolute inset-x-3 bottom-3 flex items-center justify-center">
          <span
            className={`px-3.5 py-1.5 rounded-full text-xs font-extrabold backdrop-blur border shadow ${
              evalState.targetMatched
                ? isLeft
                  ? 'bg-[#0284C7] text-white border-sky-200 animate-pulse'
                  : 'bg-emerald-600 text-white border-emerald-200 animate-pulse'
                : evalState.wrongSideRaised
                  ? 'bg-amber-500 text-white border-amber-200'
                  : 'bg-white/95 text-[#0F4C81] border-emerald-300'
            }`}
          >
            {evalState.targetMatched
              ? rhCopy.matchedBadge(Math.round(holdProgress * 100))
              : evalState.wrongSideRaised
                ? rhCopy.wrongSideBadge
                : !detectorReady
                  ? rhCopy.preparingBadge
                  : rhCopy.holdPromptBadge}
          </span>
        </div>
      </div>

      <button
        type="button"
        onClick={handleManualConfirm}
        className={`relative overflow-hidden w-full py-3.5 px-5 rounded-2xl text-white font-extrabold text-sm shadow-lg cursor-pointer transition-colors ${
          isLeft
            ? 'bg-gradient-to-r from-orange-500 to-amber-500 hover:from-orange-400 hover:to-amber-400'
            : 'bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500'
        }`}
      >
        {holdProgress > 0 && (
          <span
            className={`pointer-events-none absolute inset-y-0 left-0 transition-[width] duration-75 ${
              isLeft ? 'bg-orange-200/40' : 'bg-emerald-300/35'
            }`}
            style={{ width: `${Math.round(holdProgress * 100)}%` }}
          />
        )}
        <span className="relative z-10 flex items-center justify-center gap-2">
          <span>
            {holdProgress > 0
              ? rhCopy.confirmingBtn(Math.round(holdProgress * 100))
              : rhCopy.idleBtn}
          </span>
        </span>
      </button>
    </div>
  );
}
