import React, { useEffect, useRef, useState } from 'react';
import {
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
  type NormalizedLandmark,
} from '@mediapipe/tasks-vision';
import { selectClinicalCameraDeviceId } from '@drtechcares/face-client';
import { extractRaisedHandSides } from '@drtechcares/kiosk-input';
import type { PoseLandmark } from '@drtechcares/pose-engine';
import { Shuffle } from 'lucide-react';

export type HandChallengeSide = 'left' | 'right';

export interface RandomHandEvaluationResult {
  leftRaised: boolean;
  rightRaised: boolean;
  targetMatched: boolean;
  wrongSideRaised: boolean;
}

const HAND_HOLD_CONFIRM_MS = 420;
const OFF_CANVAS_WIDTH = 256;

function pickRandomHandSide(): HandChallengeSide {
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
}

export function RandomHandStartPrompt({
  onConfirmStart,
  initialSide,
}: RandomHandStartPromptProps) {
  const [targetSide, setTargetSide] = useState<HandChallengeSide>(
    () => initialSide ?? pickRandomHandSide(),
  );
  const [cameraReady, setCameraReady] = useState(false);
  const [detectorReady, setDetectorReady] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
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

  const sideLabel = targetSide === 'left' ? 'มือซ้าย' : 'มือขวา';
  const sideLabelEn = targetSide === 'left' ? 'Left Hand' : 'Right Hand';
  const oppositeLabel = targetSide === 'left' ? 'มือขวา' : 'มือซ้าย';

  const rerollSide = () => {
    setTargetSide((prev) => (prev === 'left' ? 'right' : 'left'));
    setHoldProgress(0);
  };

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
          setCameraError('เบราว์เซอร์ไม่รองรับกล้อง สามารถกดปุ่มด้านล่างเพื่อเริ่มกายภาพได้ทันที');
          return;
        }

        const deviceId = await selectClinicalCameraDeviceId();
        if (!active) return;

        stream = await navigator.mediaDevices.getUserMedia({
          video: deviceId
            ? { deviceId: { exact: deviceId }, width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
            : { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
          audio: false,
        });

        if (!active || !videoRef.current) {
          cleanupMedia();
          return;
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
          setCameraError('ไม่สามารถเปิดกล้องได้ — สามารถกดปุ่มด้านล่างเพื่อเริ่มการกายภาพได้ทันที');
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

  return (
    <div className="mt-5 rounded-2xl border-2 border-emerald-300 bg-emerald-50/70 p-4 text-center space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="text-lg">{targetSide === 'left' ? '🫲' : '🫱'}</span>
          <div className="text-left">
            <p className="text-xs font-extrabold text-emerald-950">
              {targetSide === 'left'
                ? 'ระบบสุ่มให้ยก: มือซ้าย (Left Hand)'
                : 'ระบบสุ่มให้ยก: มือขวา (Right Hand)'}
            </p>
            <p className="text-[11px] font-semibold text-emerald-700">
              {targetSide === 'left'
                ? '🫲 โจทย์สุ่ม: กรุณายกมือซ้ายค้างไว้เพื่อเริ่มการกายภาพ'
                : '🫱 โจทย์สุ่ม: กรุณายกมือขวาค้างไว้เพื่อเริ่มการกายภาพ'}
            </p>
          </div>
        </div>
        <button
          type="button"
          onClick={rerollSide}
          className="inline-flex items-center gap-1 rounded-xl border border-emerald-300 bg-white px-2.5 py-1 text-[11px] font-bold text-emerald-800 hover:bg-emerald-100 cursor-pointer"
        >
          <Shuffle className="h-3.5 w-3.5" />
          สุ่มข้างใหม่
        </button>
      </div>

      {/* Live Mirrored Camera Preview with Highlighted Left / Right Target Side */}
      <div className="relative mx-auto w-full h-40 rounded-2xl overflow-hidden bg-slate-900 border-2 border-emerald-400/60 shadow-inner">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-full object-cover"
          style={{ transform: 'scaleX(-1)' }}
        />

        {/* Target Left/Right Side Highlight Overlay */}
        <div className="pointer-events-none absolute inset-0 flex">
          <div
            className={`w-1/2 h-full border-r border-dashed border-white/40 flex items-start justify-start p-2 transition-colors ${
              targetSide === 'left' ? 'bg-emerald-500/20 ring-2 ring-inset ring-emerald-400/80' : 'bg-slate-950/35'
            }`}
          >
            <span
              className={`px-2 py-0.5 rounded-full text-[10px] font-extrabold ${
                targetSide === 'left'
                  ? 'bg-emerald-500 text-white shadow'
                  : 'bg-slate-900/70 text-slate-300'
              }`}
            >
              ฝั่งซ้าย • มือซ้าย {targetSide === 'left' ? '(เป้าหมาย)' : ''}
            </span>
          </div>
          <div
            className={`w-1/2 h-full flex items-start justify-end p-2 transition-colors ${
              targetSide === 'right' ? 'bg-emerald-500/20 ring-2 ring-inset ring-emerald-400/80' : 'bg-slate-950/35'
            }`}
          >
            <span
              className={`px-2 py-0.5 rounded-full text-[10px] font-extrabold ${
                targetSide === 'right'
                  ? 'bg-emerald-500 text-white shadow'
                  : 'bg-slate-900/70 text-slate-300'
              }`}
            >
              {targetSide === 'right' ? '(เป้าหมาย) ' : ''}ฝั่งขวา • มือขวา
            </span>
          </div>
        </div>

        {!cameraReady && !cameraError && (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-900/85 text-xs font-bold text-slate-200 px-4">
            กำลังเปิดกล้องตรวจจับการยก{sideLabel} ({sideLabelEn})...
          </div>
        )}
        {cameraError && (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-900/90 text-xs font-bold text-amber-200 px-4">
            {cameraError}
          </div>
        )}

        <div className="absolute inset-x-2 bottom-2 flex items-center justify-center">
          <span
            className={`px-3 py-1 rounded-full text-[11px] font-extrabold backdrop-blur border shadow ${
              evalState.targetMatched
                ? 'bg-emerald-500/95 text-white border-emerald-200 animate-pulse'
                : evalState.wrongSideRaised
                  ? 'bg-amber-500/95 text-slate-950 border-amber-200'
                  : 'bg-slate-950/80 text-emerald-200 border-emerald-500/40'
            }`}
          >
            {evalState.targetMatched
              ? `✅ ตรวจพบการยก${sideLabel}แล้ว! ค้างไว้ ${Math.round(holdProgress * 100)}%`
              : evalState.wrongSideRaised
                ? `⚠️ คุณกำลังยก${oppositeLabel} กรุณายก "${sideLabel}" ตามโจทย์`
                : !detectorReady
                  ? `⏳ กำลังเตรียม AI ตรวจจับการยก${sideLabel}...`
                  : `📷 ชู "${sideLabel}" ขึ้นในฝั่งที่ไฮไลต์ค้างไว้เพื่อเริ่มกายภาพอัตโนมัติ`}
          </span>
        </div>
      </div>

      <button
        type="button"
        onClick={handleManualConfirm}
        className="relative overflow-hidden w-full py-3.5 px-5 rounded-2xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-extrabold text-sm shadow-lg cursor-pointer"
      >
        {holdProgress > 0 && (
          <span
            className="pointer-events-none absolute inset-y-0 left-0 bg-emerald-300/35 transition-[width] duration-75"
            style={{ width: `${Math.round(holdProgress * 100)}%` }}
          />
        )}
        <span className="relative z-10 flex items-center justify-center gap-2">
          <span>
            {holdProgress > 0
              ? `🖐️ ยก${sideLabel}ค้างไว้... (${Math.round(holdProgress * 100)}%)`
              : `🖐️ ยก${sideLabel}ขึ้น หรือ กดที่นี่เพื่อเริ่มการกายภาพบำบัด ➔`}
          </span>
        </span>
      </button>
    </div>
  );
}
