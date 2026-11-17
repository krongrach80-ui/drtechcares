import React, { useEffect, useRef, useState } from 'react';
import {
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
  type NormalizedLandmark,
} from '@mediapipe/tasks-vision';
import type {
  DailyScheduleDay,
  DrTechCaresApiClient,
  SystemSettingsRecord,
} from '@drtechcares/api-client';
import {
  alignFaceTo112,
  evaluateLivenessAndPose,
  inspectBiometricFrame3D,
  resolveFaceIdScreenPrompt,
  selectClinicalCameraDeviceId,
  SingleScanLoginController,
  type BiometricVerificationJson,
} from '@drtechcares/face-client';
import {
  captureCanonicalFaceSnapshot,
  extractCanonicalFaceDescriptor,
} from '../../../src/lib/biometricEngine';
import {
  extractRaisedHandSides,
  FacePresenceTracker,
  RaiseBothHandsTracker,
} from '@drtechcares/kiosk-input';
import {
  BURST_FRAME_COUNT,
  evaluateFrameQuality,
  purgeLegacyBiometricCache,
} from '@drtechcares/smart-biometric-core';
import {
  AlertCircle,
  Calendar,
  Camera,
  CheckCircle2,
  Hand,
  HeartPulse,
  PhoneCall,
  Wrench,
} from 'lucide-react';
import { DynamicExerciseSessionView } from './DynamicExerciseSessionView';
import { PhysicalCardGame } from './PhysicalCardGame';
import { FaceIdCircularOverlay } from '../../../src/components/FaceIdCircularOverlay';

type KioskFlowState =
  | 'STANDBY'
  | 'FACIAL_AUTH'
  | 'NOT_FOUND'
  | 'GESTURE_CONTROL'
  | 'CONDITIONAL_EXECUTION'
  | 'SUMMARY_AND_RESET';

interface VerifiedKioskPatient {
  id: string;
  hn: string;
  fullName: string;
  nextAppointmentDate: string;
}

interface KioskPortalAppProps {
  apiClient: DrTechCaresApiClient;
}

export function KioskPortalApp({ apiClient }: KioskPortalAppProps) {
  const [flowState, setFlowState] = useState<KioskFlowState>('STANDBY');
  const [settings, setSettings] = useState<SystemSettingsRecord>({
    aiFaceScanEnabled: true,
    aiFaceThreshold: 0.925,
    aiPoseDetectionEnabled: false,
    kioskAutoResetSec: 6,
  });

  const [verifiedPatient, setVerifiedPatient] = useState<VerifiedKioskPatient | null>(null);
  const [todayRoutine, setTodayRoutine] = useState<DailyScheduleDay | null>(null);
  const [countdownSec, setCountdownSec] = useState(3);
  const [burstCollected, setBurstCollected] = useState(0);
  const [liveInspection, setLiveInspection] = useState<BiometricVerificationJson | null>(null);
  const [scanHint, setScanHint] = useState('กรุณามองตรงเข้ากล้องเพื่อสแกนใบหน้า');
  const [raiseProgress, setRaiseProgress] = useState(0);
  const [staffCalled, setStaffCalled] = useState(false);
  const [summaryText, setSummaryText] = useState('วันนี้คุณทำได้ดีมาก!');
  const [autoResetLeft, setAutoResetLeft] = useState(6);
  const [physioWaitLeft, setPhysioWaitLeft] = useState(4);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const alignCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const presenceRef = useRef(new FacePresenceTracker());
  const loginCtrlRef = useRef(new SingleScanLoginController());
  const raiseTrackerRef = useRef(new RaiseBothHandsTracker());

  useEffect(() => {
    purgeLegacyBiometricCache();
    apiClient
      .getSystemSettings()
      .then((res) => setSettings(res.settings))
      .catch(() => undefined);
  }, [apiClient]);

  const resetToStandby = () => {
    presenceRef.current.reset();
    loginCtrlRef.current.reset();
    raiseTrackerRef.current.reset();
    setVerifiedPatient(null);
    setTodayRoutine(null);
    setBurstCollected(0);
    setLiveInspection(null);
    setRaiseProgress(0);
    setStaffCalled(false);
    setFlowState('STANDBY');
  };

  // STATE 1 (STANDBY) & STATE 2 (FACIAL_AUTH): Real Camera Motion/Presence -> Slide Logo Up -> Facial Auth
  useEffect(() => {
    if (flowState !== 'STANDBY' && flowState !== 'FACIAL_AUTH') return;

    let active = true;
    let stream: MediaStream | null = null;
    let rafId = 0;
    let landmarker: FaceLandmarker | null = null;
    let prevLandmarks: { x: number; y: number; z?: number }[] | null = null;
    let authStartedAt: number | null = null;

    async function runFacePipeline() {
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
        landmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: '/models/face_landmarker.task' },
          runningMode: 'VIDEO',
          numFaces: 2,
        });

        const loop = async () => {
          if (!active || !videoRef.current || !landmarker) return;
          if (videoRef.current.readyState >= 2) {
            const now = performance.now();
            const res = landmarker.detectForVideo(videoRef.current, now);
            const faceCount = res.faceLandmarks?.length ?? 0;
            const landmarks = res.faceLandmarks?.[0] ?? [];
            const faceRatio = landmarks.length > 400 ? Math.abs(landmarks[454].x - landmarks[234].x) : 0;

            // Standby -> Motion/Presence wake-up
            const presence = presenceRef.current.update({
              facePresent: faceCount >= 1,
              faceSizeRatio: faceRatio,
              timestampMs: now,
            });

            if (flowState === 'STANDBY' && presence.awake) {
              setFlowState('FACIAL_AUTH');
              return;
            }

            if (flowState === 'FACIAL_AUTH') {
              if (authStartedAt === null) authStartedAt = now;
              const rem = Math.max(0, 3 - Math.floor((now - authStartedAt) / 1000));
              setCountdownSec(rem);

              const liveness = evaluateLivenessAndPose(landmarks, prevLandmarks, faceCount);
              prevLandmarks = landmarks;

              if (!alignCanvasRef.current) alignCanvasRef.current = document.createElement('canvas');
              const crop =
                landmarks.length >= 300 ? alignFaceTo112(videoRef.current, landmarks, alignCanvasRef.current) : null;

              const inspection = inspectBiometricFrame3D({
                landmarks3D: landmarks,
                prevLandmarks3D: prevLandmarks,
                faceCount,
                image: crop,
                requestedPose: 'CENTER',
              });
              setLiveInspection(inspection);
              setScanHint(inspection.user_instruction_th);

              if (rem === 0 && inspection.scan_status === 'PASS' && liveness.isLive && crop) {
                const q = evaluateFrameQuality(crop, {
                  yawDeg: liveness.yawDeg,
                  pitchDeg: liveness.pitchDeg,
                  rollDeg: liveness.rollDeg,
                });
                if (q.acceptable) {
                  loginCtrlRef.current.startCountdownComplete();
                  const snap = captureCanonicalFaceSnapshot(
                    videoRef.current,
                    null,
                    undefined,
                    landmarks,
                  );
                  const extracted = snap ? await extractCanonicalFaceDescriptor(snap) : null;
                  if (!extracted?.embedding) {
                    rafId = requestAnimationFrame(loop);
                    return;
                  }
                  const vector = extracted.embedding;
                  const push = loginCtrlRef.current.pushFrame({
                    vector,
                    qualityScore: q.sharpness,
                    capturedAfterCountdown: true,
                    pose: 'front',
                    sharpness: q.sharpness,
                    yaw: liveness.yawDeg,
                    pitch: liveness.pitchDeg,
                  });
                  setBurstCollected(push.collected);
                  setScanHint('กำลังยืนยันตัวตนกับฐานข้อมูลกลาง...');

                  if (push.readyToVerify) {
                    const probe = loginCtrlRef.current.buildProbeCentroid();
                    const verifyRes = await apiClient.verifyFaceLogin(probe);
                    if (verifyRes.matched && verifyRes.patient) {
                      setVerifiedPatient(verifyRes.patient);
                      setTodayRoutine(verifyRes.todayRoutine ?? null);
                      setFlowState('GESTURE_CONTROL');
                      return;
                    }
                    loginCtrlRef.current.recordUnmatchedRound();
                    setFlowState('NOT_FOUND');
                    return;
                  }
                }
              }
            }
          }
          rafId = requestAnimationFrame(loop);
        };
        rafId = requestAnimationFrame(loop);
      } catch (err) {
        setScanHint(err instanceof Error ? err.message : 'ไม่พบกล้องเว็บแคม');
      }
    }

    runFacePipeline();
    return () => {
      active = false;
      cancelAnimationFrame(rafId);
      landmarker?.close();
      stream?.getTracks().forEach((tr) => tr.stop());
    };
  }, [flowState, apiClient]);

  // STATE 3 (GESTURE_CONTROL): "ยก 2 มือขึ้นเพื่อเริ่ม" via PoseLandmarker + HandLandmarker (2 hands)
  useEffect(() => {
    if (flowState !== 'GESTURE_CONTROL') return;

    let active = true;
    let stream: MediaStream | null = null;
    let rafId = 0;
    let poseLandmarker: PoseLandmarker | null = null;
    let handLandmarker: HandLandmarker | null = null;
    let offscreenCanvas: HTMLCanvasElement | null = null;
    let lastTs = 0;
    let lastFrameAt = 0;
    let lastVideoTime = -1;
    let lastInferenceAt = 0;
    let handHoldMs = 0;
    let lastLeftRaisedAt = -Infinity;
    let lastRightRaisedAt = -Infinity;
    let latestBothHandsUp = false;
    const HAND_HOLD_MS = 350;

    async function runRaiseBothHands() {
      try {
        const deviceId = await selectClinicalCameraDeviceId();
        const constraints: MediaStreamConstraints = {
          video: deviceId ? { deviceId: { exact: deviceId }, width: 640, height: 480 } : { width: 640, height: 480 },
          audio: false,
        };
        try {
          stream = await navigator.mediaDevices.getUserMedia(constraints);
        } catch {
          await new Promise((r) => window.setTimeout(r, 200));
          if (!active) return;
          stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
        }
        if (!active || !videoRef.current) return;
        videoRef.current.srcObject = stream;
        await videoRef.current.play();

        const vision = await FilesetResolver.forVisionTasks('/mediapipe/wasm');
        if (!active) return;

        for (const delegate of ['CPU', undefined] as const) {
          try {
            handLandmarker = await HandLandmarker.createFromOptions(vision, {
              baseOptions: delegate
                ? { modelAssetPath: '/models/hand_landmarker.task', delegate }
                : { modelAssetPath: '/models/hand_landmarker.task' },
              runningMode: 'VIDEO',
              numHands: 2,
              minHandDetectionConfidence: 0.35,
              minHandPresenceConfidence: 0.35,
              minTrackingConfidence: 0.35,
            });
            break;
          } catch {
            // Try fallback delegate
          }
        }

        for (const modelAssetPath of ['/models/pose_landmarker.task', '/models/pose_landmarker_full.task']) {
          for (const delegate of ['CPU', undefined] as const) {
            try {
              poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
                baseOptions: delegate ? { modelAssetPath, delegate } : { modelAssetPath },
                runningMode: 'VIDEO',
                numPoses: 1,
                minPoseDetectionConfidence: 0.35,
                minPosePresenceConfidence: 0.35,
                minTrackingConfidence: 0.35,
              });
              break;
            } catch {
              // Try fallback pose model
            }
          }
          if (poseLandmarker) break;
        }

        const loop = () => {
          if (!active || !videoRef.current || (!poseLandmarker && !handLandmarker)) return;
          let shouldContinue = true;
          try {
            const v = videoRef.current;
            const now = performance.now();
            const dt = lastFrameAt > 0 ? Math.min(100, Math.max(1, now - lastFrameAt)) : 16;
            lastFrameAt = now;

            if (v.readyState >= 2 && v.videoWidth > 0) {
              if (v.currentTime !== lastVideoTime || now - lastInferenceAt >= 33) {
                lastVideoTime = v.currentTime;
                lastInferenceAt = now;
                if (!offscreenCanvas) offscreenCanvas = document.createElement('canvas');
                const offW = 256;
                const offH = Math.max(144, Math.round(offW * ((v.videoHeight || 480) / (v.videoWidth || 640))));
                if (offscreenCanvas.width !== offW || offscreenCanvas.height !== offH) {
                  offscreenCanvas.width = offW;
                  offscreenCanvas.height = offH;
                }
                const offCtx = offscreenCanvas.getContext('2d', { alpha: false });
                if (offCtx) offCtx.drawImage(v, 0, 0, offW, offH);
                const inferenceSource = offCtx ? offscreenCanvas : v;

                let detectedHands: NormalizedLandmark[][] = [];
                let detectedPose: NormalizedLandmark[] | null = null;
                if (handLandmarker) {
                  try {
                    const detectTs = Math.max(lastTs + 1, Math.round(now));
                    lastTs = detectTs;
                    const handRes = handLandmarker.detectForVideo(inferenceSource, detectTs);
                    detectedHands = handRes.landmarks ?? [];
                  } catch {
                    // Continue across transient MediaPipe frame errors
                  }
                }
                if (poseLandmarker) {
                  try {
                    const poseTs = Math.max(lastTs + 1, Math.round(now));
                    lastTs = poseTs;
                    const poseRes = poseLandmarker.detectForVideo(inferenceSource, poseTs);
                    detectedPose = poseRes.landmarks?.[0] ?? null;
                  } catch {
                    // Continue across transient MediaPipe frame errors
                  }
                }

                const sides = extractRaisedHandSides(detectedHands, detectedPose);
                if (sides.leftRaised) lastLeftRaisedAt = now;
                if (sides.rightRaised) lastRightRaisedAt = now;
                latestBothHandsUp =
                  sides.bothRaised ||
                  (now - lastLeftRaisedAt <= 450 && now - lastRightRaisedAt <= 450);

                if (detectedPose) {
                  const status = raiseTrackerRef.current.update(detectedPose, now);
                  if (status.confirmed) latestBothHandsUp = true;
                }
              }

              handHoldMs = latestBothHandsUp
                ? Math.min(HAND_HOLD_MS, handHoldMs + dt)
                : Math.max(0, handHoldMs - dt * 0.9);
              const bestProgress = Math.min(1, handHoldMs / HAND_HOLD_MS);
              setRaiseProgress(bestProgress);
              if (bestProgress >= 1) {
                shouldContinue = false;
                setFlowState('CONDITIONAL_EXECUTION');
                return;
              }
            }
          } catch {
            // Keep animation loop alive across transient frame errors
          } finally {
            if (active && shouldContinue) {
              rafId = requestAnimationFrame(loop);
            }
          }
        };
        rafId = requestAnimationFrame(loop);
      } catch {
        // Camera error handled gracefully
      }
    }

    runRaiseBothHands();
    return () => {
      active = false;
      cancelAnimationFrame(rafId);
      poseLandmarker?.close();
      handLandmarker?.close();
      stream?.getTracks().forEach((tr) => tr.stop());
    };
  }, [flowState]);

  // STATE 4B: Physio Exercise Notice ("ยังไม่ได้เพิ่มระบบดีเทคกายภาพ โปรดรอการพัฒนา" 3-5s -> Summary)
  useEffect(() => {
    if (flowState !== 'CONDITIONAL_EXECUTION') return;
    const isPhysio = todayRoutine?.activityType === 'physio_exercise';
    if (!isPhysio || settings.aiPoseDetectionEnabled) return;

    setPhysioWaitLeft(4);
    const interval = setInterval(() => {
      setPhysioWaitLeft((prev) => {
        if (prev <= 1) {
          clearInterval(interval);
          setSummaryText('วันนี้คุณทำได้ดีมาก! (บันทึกการเข้ารับบริการเรียบร้อย)');
          setFlowState('SUMMARY_AND_RESET');
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(interval);
  }, [flowState, todayRoutine, settings.aiPoseDetectionEnabled]);

  // STATE 5: Summary & Auto-Reset to Standby
  useEffect(() => {
    if (flowState !== 'SUMMARY_AND_RESET') return;
    const initialSec = Math.max(3, settings.kioskAutoResetSec || 6);
    setAutoResetLeft(initialSec);
    const interval = setInterval(() => {
      setAutoResetLeft((prev) => {
        if (prev <= 1) {
          clearInterval(interval);
          resetToStandby();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(interval);
  }, [flowState, settings.kioskAutoResetSec]);

  const handleCallStaff = async () => {
    await apiClient.createStaffCall(
      'KIOSK-OPD-01',
      'ไม่พบบัญชีผู้ป่วยระหว่างสแกนใบหน้า ต้องการเจ้าหน้าที่ช่วยเหลือ',
      verifiedPatient?.id ?? null,
    );
    setStaffCalled(true);
  };

  return (
    <div className="min-h-[calc(100vh-2rem)] rounded-3xl bg-slate-950 text-white p-6 shadow-2xl flex flex-col justify-between relative overflow-hidden">
      {/* Hidden presence video stream during STANDBY */}
      {flowState === 'STANDBY' && (
        <video ref={videoRef} playsInline muted className="pointer-events-none absolute opacity-0 h-1 w-1" />
      )}

      {/* Animated DrTech.Care Logo Header: centered on STANDBY, slides up on FACIAL_AUTH+ */}
      <div
        className={`transition-all duration-700 ease-out flex items-center justify-center gap-3 ${
          flowState === 'STANDBY' ? 'my-auto flex-col py-16' : 'pt-1 pb-4 border-b border-slate-800'
        }`}
      >
        <div
          className={`flex items-center justify-center rounded-3xl bg-gradient-to-br from-teal-400 to-emerald-600 shadow-xl transition-all duration-700 ${
            flowState === 'STANDBY' ? 'h-28 w-28 animate-pulse' : 'h-11 w-11'
          }`}
        >
          <HeartPulse className={flowState === 'STANDBY' ? 'h-16 w-16 text-white' : 'h-6 w-6 text-white'} />
        </div>
        <div className={flowState === 'STANDBY' ? 'text-center mt-4' : 'text-left'}>
          <h1 className={flowState === 'STANDBY' ? 'text-5xl font-black tracking-tight' : 'text-xl font-black'}>
            DrTech.<span className="text-teal-400">Care</span>
          </h1>
          {flowState === 'STANDBY' && (
            <div className="mt-4 space-y-2">
              <span className="inline-flex items-center gap-2 rounded-full bg-emerald-500/20 border border-emerald-400/40 px-4 py-1.5 text-sm font-bold text-emerald-300">
                <span className="h-2.5 w-2.5 rounded-full bg-emerald-400 animate-ping" />
                สถานะ: Kiosk Standby — กรุณาเดินมานั่งหน้าตู้เพื่อเริ่มสแกนอัตโนมัติ
              </span>
              <p className="text-xs text-slate-400">Motion & Presence Sensor Active • Touchless 100%</p>
            </div>
          )}
        </div>
      </div>

      {/* STATE 2: FACIAL_AUTH */}
      {flowState === 'FACIAL_AUTH' && (
        <div className="my-auto mx-auto w-full max-w-3xl space-y-4">
          <div className="relative overflow-hidden rounded-3xl bg-slate-900 aspect-video border border-slate-800">
            <video ref={videoRef} playsInline muted className="h-full w-full object-cover -scale-x-100" />
            <FaceIdCircularOverlay
              progressRatio={burstCollected / BURST_FRAME_COUNT}
              inspection={liveInspection}
              screenPrompt={resolveFaceIdScreenPrompt(liveInspection, 'initial')}
            />
            <div className="absolute top-3 right-4 rounded-full bg-slate-950/85 px-3 py-1 text-xs font-bold text-white border border-slate-700">
              {countdownSec > 0 ? `นับถอยหลัง ${countdownSec}s` : `Burst ${burstCollected}/${BURST_FRAME_COUNT}`}
            </div>
          </div>
        </div>
      )}

      {/* STATE 2B: NOT_FOUND */}
      {flowState === 'NOT_FOUND' && (
        <div className="my-auto mx-auto max-w-md rounded-3xl bg-white p-8 text-center text-slate-900 shadow-2xl">
          <AlertCircle className="mx-auto h-14 w-14 text-rose-500" />
          <h2 className="mt-4 text-2xl font-extrabold text-slate-900">ไม่พบบัญชี</h2>
          <p className="mt-2 text-sm text-slate-600">
            สแกนไม่ผ่านหรือไม่พบข้อมูลผู้ป่วยในระบบ กรุณากดปุ่มมุมขวาล่างเพื่อเรียกเจ้าหน้าที่
          </p>
          {staffCalled && (
            <div className="mt-4 rounded-xl bg-amber-50 border border-amber-200 px-3 py-2 text-xs font-bold text-amber-800">
              ส่งสัญญาณเรียกเจ้าหน้าที่แล้ว กรุณารอสักครู่
            </div>
          )}
          <div className="mt-6 flex justify-end">
            <button
              onClick={resetToStandby}
              className="rounded-xl border border-slate-300 px-4 py-2 text-xs font-semibold text-slate-600 hover:bg-slate-50"
            >
              กลับหน้า Standby
            </button>
          </div>
        </div>
      )}

      {/* STATE 3: GESTURE_CONTROL ("ยก 2 มือขึ้นเพื่อเริ่ม") */}
      {flowState === 'GESTURE_CONTROL' && verifiedPatient && (
        <div className="my-auto mx-auto w-full max-w-4xl grid grid-cols-1 gap-6 lg:grid-cols-2 items-center">
          <div className="rounded-3xl bg-slate-900 border border-slate-800 p-6 space-y-4">
            <span className="inline-block rounded-full bg-teal-500/20 px-3 py-1 text-xs font-bold text-teal-300">
              สแกนผ่าน • HN: {verifiedPatient.hn}
            </span>
            <h2 className="text-3xl font-extrabold">{verifiedPatient.fullName}</h2>
            <div className="rounded-2xl bg-slate-800/80 p-4 border border-slate-700">
              <p className="text-xs font-semibold text-slate-400">ตารางฝึกของวันนี้</p>
              <p className="mt-1 text-lg font-bold text-white">
                {todayRoutine?.activityType === 'physio_exercise'
                  ? '🚧 ท่ากายภาพบำบัด'
                  : '🎮 มินิเกมฝึกสมองและการตัดสินใจ'}
              </p>
            </div>
            <div className="rounded-2xl bg-teal-600/20 border border-teal-500/40 p-4">
              <p className="text-lg font-bold text-teal-300 flex items-center gap-2">
                <Hand className="h-6 w-6" /> ยก 2 มือขึ้นเพื่อเริ่ม
              </p>
              <div className="mt-3 h-3 w-full overflow-hidden rounded-full bg-slate-800">
                <div
                  className="h-full bg-teal-400 transition-all duration-100"
                  style={{ width: `${Math.round(raiseProgress * 100)}%` }}
                />
              </div>
            </div>
          </div>
          <div className="relative overflow-hidden rounded-3xl bg-slate-900 aspect-[4/3] border border-slate-800">
            <video ref={videoRef} playsInline muted className="h-full w-full object-cover -scale-x-100" />
          </div>
        </div>
      )}

      {/* STATE 4: CONDITIONAL_EXECUTION */}
      {flowState === 'CONDITIONAL_EXECUTION' && verifiedPatient && (
        <div className="my-auto text-slate-900">
          {todayRoutine?.activityType === 'physio_exercise' ? (
            settings.aiPoseDetectionEnabled ? (
              <DynamicExerciseSessionView
                patient={{
                  id: verifiedPatient.id,
                  hn: verifiedPatient.hn,
                  fullName: verifiedPatient.fullName,
                  age: 65,
                  phone: '',
                  diagnosis: '',
                  assignedPtId: '',
                  delegatedToPtId: null,
                  delegationNote: null,
                  nextAppointmentDate: verifiedPatient.nextAppointmentDate,
                  registrationStatus: 'active',
                  enrolledAt: '',
                }}
                apiClient={apiClient}
                onFinished={() => {
                  setSummaryText('วันนี้คุณทำได้ดีมาก! ฝึกท่ากายภาพบำบัดครบถ้วน');
                  setFlowState('SUMMARY_AND_RESET');
                }}
              />
            ) : (
              <div className="mx-auto max-w-xl rounded-3xl bg-amber-50 p-10 text-center shadow-2xl border-2 border-amber-300">
                <Wrench className="mx-auto h-16 w-16 text-amber-600" />
                <h2 className="mt-4 text-2xl font-extrabold text-amber-950">
                  ยังไม่ได้เพิ่มระบบดีเทคกายภาพ โปรดรอการพัฒนา
                </h2>
                <p className="mt-3 text-sm font-semibold text-amber-800">
                  กำลังนำท่านไปสู่หน้าสรุปผลอัตโนมัติใน {physioWaitLeft} วินาที...
                </p>
              </div>
            )
          ) : (
            <PhysicalCardGame
              patientName={verifiedPatient.fullName}
              onComplete={async (summary) => {
                const accuracy = Math.round((summary.correct / summary.total) * 100);
                await apiClient.recordRehabSession({
                  patientId: verifiedPatient.id,
                  activityType: 'minigame',
                  exerciseId: 'cognitive-card-game',
                  exerciseName: 'มินิเกมฝึกสมอง (การ์ดกายภาพ)',
                  completedSets: 1,
                  completedReps: summary.total,
                  accuracyPercent: accuracy,
                  maxRomDeg: 0,
                  painStopTriggered: false,
                  aiFeedback: `Cognitive Accuracy ${accuracy}% (${summary.correct}/${summary.total})`,
                  kinematicsReplay: [],
                  minigameScore: summary,
                });
                setSummaryText(`วันนี้คุณทำได้ดีมาก! ได้คะแนน ${summary.correct}/${summary.total} (${accuracy}%)`);
                setFlowState('SUMMARY_AND_RESET');
              }}
            />
          )}
        </div>
      )}

      {/* STATE 5: SUMMARY_AND_RESET */}
      {flowState === 'SUMMARY_AND_RESET' && verifiedPatient && (
        <div className="my-auto mx-auto max-w-lg rounded-3xl bg-white p-10 text-center text-slate-900 shadow-2xl">
          <CheckCircle2 className="mx-auto h-16 w-16 text-emerald-500" />
          <h2 className="mt-4 text-3xl font-extrabold text-slate-900">{summaryText}</h2>
          <div className="mt-6 rounded-2xl bg-teal-50 border border-teal-200 p-4">
            <p className="text-xs font-semibold text-teal-700 flex items-center justify-center gap-1.5">
              <Calendar className="h-4 w-4" /> วันที่นัดหมายครั้งต่อไป
            </p>
            <p className="mt-1 text-xl font-black text-teal-900">
              {verifiedPatient.nextAppointmentDate}
            </p>
          </div>
          <p className="mt-6 text-xs font-semibold text-slate-500">
            รีเซ็ตตู้ Kiosk อัตโนมัติกลับสู่หน้า Standby ใน {autoResetLeft} วินาที...
          </p>
        </div>
      )}

      {/* Bottom-right "เรียกเจ้าหน้าที่" button visible on NOT_FOUND per Diagram 3 */}
      {flowState === 'NOT_FOUND' && (
        <div className="flex justify-end">
          <button
            onClick={handleCallStaff}
            className="inline-flex items-center gap-2 rounded-2xl bg-amber-500 px-5 py-3 text-sm font-extrabold text-slate-950 shadow-xl hover:bg-amber-400"
          >
            <PhoneCall className="h-5 w-5" />
            เรียกเจ้าหน้าที่
          </button>
        </div>
      )}
    </div>
  );
}
