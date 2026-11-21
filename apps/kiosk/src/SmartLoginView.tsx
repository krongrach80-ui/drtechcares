import React, { useEffect, useRef, useState } from 'react';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { useClinicalI18n } from '@drtechcares/clinical-i18n';
import type { DrTechCaresApiClient, PatientRecord } from '@drtechcares/api-client';
import {
  alignFaceTo112,
  evaluateLivenessAndPose,
  selectClinicalCameraDeviceId,
  SingleScanLoginController,
} from '@drtechcares/face-client';
import {
  captureCanonicalFaceSnapshot,
  extractCanonicalFaceDescriptor,
} from '../../../src/lib/biometricEngine';
import {
  BURST_FRAME_COUNT,
  evaluateFrameQuality,
  purgeLegacyBiometricCache,
} from '@drtechcares/smart-biometric-core';
import { FacePresenceTracker } from '@drtechcares/kiosk-input';
import { AlertCircle, Camera, PhoneCall, ShieldCheck, UserPlus } from 'lucide-react';

interface SmartLoginViewProps {
  apiClient: DrTechCaresApiClient;
  onAuthenticated: (patient: PatientRecord) => void;
  onNavigateEnroll: () => void;
}

export function SmartLoginView({ apiClient, onAuthenticated, onNavigateEnroll }: SmartLoginViewProps) {
  const { t } = useClinicalI18n();
  const [status, setStatus] = useState<'idle' | 'countdown' | 'burst' | 'verifying' | 'not_found'>('idle');
  const [countdown, setCountdown] = useState(3);
  const [burstCount, setBurstCount] = useState(0);
  const [qualityMsg, setQualityMsg] = useState<string>('');
  const [staffCalled, setStaffCalled] = useState(false);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const alignCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const loginCtrlRef = useRef(new SingleScanLoginController());
  const presenceRef = useRef(new FacePresenceTracker());
  const statusRef = useRef(status);
  statusRef.current = status;
  const isNotFound = status === 'not_found';

  useEffect(() => {
    // Zero Ghost Cache: wipe any stale biometric entries on Kiosk mount
    purgeLegacyBiometricCache();
  }, []);

  useEffect(() => {
    if (isNotFound) return;

    let active = true;
    let stream: MediaStream | null = null;
    let rafId = 0;
    let landmarker: FaceLandmarker | null = null;
    let prevLandmarks: { x: number; y: number; z?: number }[] | null = null;
    let countdownStartedAt: number | null = null;

    async function startCamera() {
      try {
        const deviceId = await selectClinicalCameraDeviceId();
        stream = await navigator.mediaDevices.getUserMedia({
          video: deviceId ? { deviceId: { exact: deviceId }, width: 640, height: 480 } : { width: 640, height: 480 },
          audio: false,
        });
        if (!active || !videoRef.current) {
          stream.getTracks().forEach((tr) => tr.stop());
          stream = null;
          return;
        }
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        if (!active) {
          stream.getTracks().forEach((tr) => tr.stop());
          stream = null;
          return;
        }

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
            // Try next WASM base path
          }
        }
        if (!vision || !active) {
          stream.getTracks().forEach((tr) => tr.stop());
          stream = null;
          return;
        }
        let createdLandmarker: FaceLandmarker | null = null;
        for (const delegate of ['GPU', 'CPU'] as const) {
          try {
            createdLandmarker = await FaceLandmarker.createFromOptions(vision, {
              baseOptions: { modelAssetPath: '/models/face_landmarker.task', delegate },
              runningMode: 'VIDEO',
              numFaces: 2,
            });
            break;
          } catch {
            // Try CPU fallback
          }
        }
        if (!createdLandmarker || !active) {
          createdLandmarker?.close();
          stream.getTracks().forEach((tr) => tr.stop());
          stream = null;
          return;
        }
        landmarker = createdLandmarker;

        let lastDetectTs = 0;
        let lastVideoTime = -1;
        const tick = async () => {
          if (!active || !videoRef.current || !landmarker) return;
          try {
            if (
              videoRef.current.readyState >= 2 &&
              videoRef.current.videoWidth > 0 &&
              videoRef.current.videoHeight > 0 &&
              videoRef.current.currentTime !== lastVideoTime
            ) {
              lastVideoTime = videoRef.current.currentTime;
              const now = performance.now();
              const detectTs = Math.max(lastDetectTs + 1, Math.round(now));
              lastDetectTs = detectTs;
              const res = landmarker.detectForVideo(videoRef.current, detectTs);
              const faceCount = res.faceLandmarks?.length ?? 0;
              const landmarks = res.faceLandmarks?.[0] ?? [];
              const liveness = evaluateLivenessAndPose(landmarks, prevLandmarks, faceCount);
              prevLandmarks = landmarks;

              const faceBoxRatio =
                landmarks.length > 400
                  ? Math.abs(landmarks[454].x - landmarks[234].x)
                  : 0;
              const presence = presenceRef.current.update({
                facePresent: faceCount === 1,
                faceSizeRatio: faceBoxRatio,
                timestampMs: now,
              });

              if (presence.awake && countdownStartedAt === null) {
                countdownStartedAt = now;
                setStatus('countdown');
              }

              if (countdownStartedAt !== null && statusRef.current !== 'not_found') {
                const elapsedSec = Math.floor((now - countdownStartedAt) / 1000);
                const rem = Math.max(0, 3 - elapsedSec);
                setCountdown(rem);

                if (rem === 0 && liveness.isLive) {
                  if (!alignCanvasRef.current) alignCanvasRef.current = document.createElement('canvas');
                  const crop = alignFaceTo112(videoRef.current, landmarks, alignCanvasRef.current);
                  if (crop) {
                    const q = evaluateFrameQuality(crop, {
                      yawDeg: liveness.yawDeg,
                      pitchDeg: liveness.pitchDeg,
                      rollDeg: liveness.rollDeg,
                    });
                    if (q.acceptable) {
                      setStatus('burst');
                      loginCtrlRef.current.startCountdownComplete();
                      const snap = captureCanonicalFaceSnapshot(
                        videoRef.current,
                        null,
                        undefined,
                        landmarks,
                      );
                      const extracted = snap ? await extractCanonicalFaceDescriptor(snap) : null;
                      if (!active) return;
                      if (!extracted?.embedding) {
                        rafId = requestAnimationFrame(tick);
                        return;
                      }
                      const push = loginCtrlRef.current.pushFrame({
                        vector: extracted.embedding,
                        qualityScore: q.sharpness,
                        capturedAfterCountdown: true,
                        pose: 'front',
                        sharpness: q.sharpness,
                        yaw: liveness.yawDeg,
                        pitch: liveness.pitchDeg,
                      });
                      setBurstCount(push.collected);

                      if (push.readyToVerify) {
                        setStatus('verifying');
                        const probe = loginCtrlRef.current.buildProbeCentroid();
                        const verifyRes = await apiClient.verifyFaceLogin(probe);
                        if (!active) return;
                        if (verifyRes.matched && verifyRes.patientId) {
                          const patients = await apiClient.listPatients();
                          if (!active) return;
                          const found = patients.find((p) => p.id === verifyRes.patientId);
                          if (found) {
                            onAuthenticated(found);
                            return;
                          }
                        }
                        // NOT_FOUND_CONFIRM_ROUNDS = 1 -> pause camera & show modal immediately
                        const miss = loginCtrlRef.current.recordUnmatchedRound();
                        if (miss.shouldShowNotFoundModal) {
                          setStatus('not_found');
                          stream?.getTracks().forEach((tr) => tr.stop());
                          return;
                        }
                      }
                    } else {
                      setQualityMsg(q.reasons.join(', '));
                    }
                  }
                }
              }
            }
          } catch {
            // Keep RAF alive across transient frame errors
          }
          if (active) {
            rafId = requestAnimationFrame(tick);
          }
        };
        rafId = requestAnimationFrame(tick);
      } catch {
        setQualityMsg('__CAMERA_STANDBY__');
      }
    }

    startCamera();
    return () => {
      active = false;
      cancelAnimationFrame(rafId);
      landmarker?.close();
      stream?.getTracks().forEach((tr) => tr.stop());
    };
  }, [isNotFound, apiClient, onAuthenticated]);

  const handleCallStaff = async () => {
    await apiClient.createStaffCall('KIOSK-OPD-01', 'Patient needs assistance on Face Login');
    setStaffCalled(true);
  };

  return (
    <div className="mx-auto max-w-4xl rounded-3xl bg-white p-6 shadow-xl border border-slate-200 animate-view-enter">
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-100 pb-4">
        <div className="min-w-0">
          <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
            <ShieldCheck className="h-7 w-7 text-teal-600 shrink-0" />
            {t('touchlessLoginTitle')}
          </h1>
          <p className="mt-1 text-sm text-slate-500">{t('touchlessLoginSubtitle')}</p>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={onNavigateEnroll}
            className="whitespace-nowrap h-10 px-4 inline-flex items-center justify-center gap-1.5 rounded-xl bg-teal-600 text-sm font-semibold text-white hover:bg-teal-700"
          >
            <UserPlus className="h-4 w-4 shrink-0" />
            {t('staffRegisterBtn')}
          </button>
          <button
            onClick={handleCallStaff}
            className="whitespace-nowrap h-10 px-4 inline-flex items-center justify-center gap-1.5 rounded-xl border border-amber-300 bg-amber-50 text-sm font-semibold text-amber-800 hover:bg-amber-100"
          >
            <PhoneCall className="h-4 w-4 shrink-0" />
            {t('callStaffBtn')}
          </button>
        </div>
      </div>

      {staffCalled && (
        <div className="mt-4 rounded-2xl bg-amber-50 border border-amber-200 px-4 py-3 text-sm font-semibold text-amber-900">
          {t('staffCalledConfirm')}
        </div>
      )}

      <div className="mt-6 relative overflow-hidden rounded-3xl bg-slate-950 aspect-video flex items-center justify-center">
        <video ref={videoRef} playsInline muted className="h-full w-full object-cover -scale-x-100" />

        {/* Clean Oval Face Guide */}
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <div className="h-64 w-48 rounded-[999px] border-4 border-dashed border-teal-400/80 shadow-[0_0_0_9999px_rgba(2,6,23,0.38)]" />
        </div>

        <div className="absolute bottom-4 inset-x-4 flex flex-wrap items-center justify-between gap-2 rounded-2xl bg-slate-900/85 px-4 py-3 text-white backdrop-blur">
          <div className="flex items-center gap-2 text-sm font-medium">
            <Camera className="h-4 w-4 text-teal-400 shrink-0" />
            <span>
              {status === 'countdown' && `${t('holdSteadyCountdown')} ${countdown}s`}
              {status === 'burst' && `${t('scanningBurst')} (${burstCount}/${BURST_FRAME_COUNT})`}
              {status === 'verifying' && t('verifyingGate')}
              {status === 'idle' &&
                (qualityMsg === '__CAMERA_STANDBY__' ? t('cameraStandbyMsg') : qualityMsg || t('alignFaceOval'))}
            </span>
          </div>
        </div>

        {/* Single-Scan Decisive Account Not Found Modal (NOT_FOUND_CONFIRM_ROUNDS = 1) */}
        {status === 'not_found' && (
          <div className="absolute inset-0 z-20 flex items-center justify-center bg-slate-950/85 p-6 backdrop-blur-sm">
            <div className="max-w-md rounded-3xl bg-white p-6 text-center shadow-2xl">
              <AlertCircle className="mx-auto h-12 w-12 text-rose-500" />
              <h2 className="mt-3 text-xl font-bold text-slate-900">{t('accountNotFoundTitle')}</h2>
              <p className="mt-2 text-sm text-slate-600">{t('accountNotFoundDesc')}</p>
              <div className="mt-6 flex flex-col gap-2.5">
                <button
                  onClick={onNavigateEnroll}
                  className="w-full whitespace-nowrap h-11 rounded-xl bg-teal-600 px-4 text-sm font-bold text-white hover:bg-teal-700"
                >
                  <UserPlus className="inline h-4 w-4 mr-1.5" />
                  {t('staffRegisterBtn')}
                </button>
                <button
                  onClick={handleCallStaff}
                  className="w-full whitespace-nowrap h-11 rounded-xl border border-amber-300 bg-amber-50 px-4 text-sm font-bold text-amber-800 hover:bg-amber-100"
                >
                  <PhoneCall className="inline h-4 w-4 mr-1.5" />
                  {t('callStaffBtn')}
                </button>
                <button
                  onClick={() => {
                    loginCtrlRef.current.reset();
                    presenceRef.current.reset();
                    setBurstCount(0);
                    setStatus('idle');
                  }}
                  className="w-full whitespace-nowrap h-10 rounded-xl border border-slate-200 px-4 text-sm font-semibold text-slate-600 hover:bg-slate-50"
                >
                  {t('retrySingleScanBtn')}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

