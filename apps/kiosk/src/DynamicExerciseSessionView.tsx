import React, { useEffect, useRef, useState } from 'react';
import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision';
import { useClinicalI18n } from '@drtechcares/clinical-i18n';
import type { DrTechCaresApiClient, PatientRecord } from '@drtechcares/api-client';
import { selectClinicalCameraDeviceId } from '@drtechcares/face-client';
import {
  AudioChimeSynthesizer,
  VoiceFeedbackController,
} from '@drtechcares/kiosk-input';
import {
  AutoBoneBaselineCalibrator,
  CLINICAL_EXERCISES,
  ClinicalBiomechanicsAgent,
  ClinicalOneEuroLandmarkFilter,
  RehabExercisePipeline,
  buildTelemetryPacketFromLandmarks,
  selectSubjectPoseWithIdentity,
  solveStrokeClinicalJointBiomechanics,
  stabilizeArmIdentity,
  type ClinicalBiomechanicsDecision,
  type ExerciseConfig,
  type PoseLandmark,
  type SessionProgressSnapshot,
  type StrokeJointMode,
} from '@drtechcares/pose-engine';
import { AlertOctagon, CheckCircle2, PhoneCall, RotateCcw, ShieldAlert } from 'lucide-react';

interface DynamicExerciseSessionViewProps {
  patient: PatientRecord;
  exercise?: ExerciseConfig;
  apiClient: DrTechCaresApiClient;
  onFinished: () => void;
}

export function DynamicExerciseSessionView({
  patient,
  exercise = CLINICAL_EXERCISES[0],
  apiClient,
  onFinished,
}: DynamicExerciseSessionViewProps) {
  const { lang, t, pick } = useClinicalI18n();
  const [selectedExercise, setSelectedExercise] = useState<ExerciseConfig>(exercise);
  const [snapshot, setSnapshot] = useState<SessionProgressSnapshot | null>(null);
  const [aiDecision, setAiDecision] = useState<ClinicalBiomechanicsDecision | null>(null);
  const [painStopped, setPainStopped] = useState(false);
  const [continuousGuardOk, setContinuousGuardOk] = useState(true);
  const [multiPersonPaused, setMultiPersonPaused] = useState(false);
  const [pinOverrideOpen, setPinOverrideOpen] = useState(false);
  const [staffPin, setStaffPin] = useState('');
  const [strokeWarningTh, setStrokeWarningTh] = useState<string | null>(null);
  const [strokeClinicalAngleDeg, setStrokeClinicalAngleDeg] = useState<number | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const pipelineRef = useRef(new RehabExercisePipeline(selectedExercise));
  const agentRef = useRef(new ClinicalBiomechanicsAgent());
  const filterRef = useRef(new ClinicalOneEuroLandmarkFilter(1.15, 0.06, 1.0));
  const worldFilterRef = useRef(new ClinicalOneEuroLandmarkFilter(1.15, 0.06, 1.0));
  const baselineRef = useRef(new AutoBoneBaselineCalibrator());
  const voiceRef = useRef(new VoiceFeedbackController());
  const chimeRef = useRef(new AudioChimeSynthesizer());
  const replayLogRef = useRef<{ tMs: number; angleDeg: number }[]>([]);
  const lastMultiPersonTtsAtRef = useRef<number>(0);
  const recordedSessionRef = useRef<boolean>(false);

  useEffect(() => {
    setSelectedExercise(exercise);
  }, [exercise]);

  useEffect(() => {
    recordedSessionRef.current = false;
    pipelineRef.current.reset(selectedExercise);
    agentRef.current.reset();
    filterRef.current.reset();
    worldFilterRef.current.reset();
    baselineRef.current.reset();
    replayLogRef.current = [];
    setSnapshot(null);
    setAiDecision(null);
    setStrokeWarningTh(null);
    setStrokeClinicalAngleDeg(null);
  }, [selectedExercise]);

  useEffect(() => {
    if (painStopped) return;

    let active = true;
    let stream: MediaStream | null = null;
    let rafId = 0;
    let poseLandmarker: PoseLandmarker | null = null;
    let prevPose: PoseLandmark[] | null = null;
    let prevAngle: number | null = null;
    let lastTickAt = performance.now();
    let lastDetectTsMs = 0;
    let singlePersonStreak = 0;
    const startedAt = performance.now();

    async function initPoseCamera() {
      const releaseInitResources = () => {
        try {
          poseLandmarker?.close();
        } catch {
          // ignore
        }
        poseLandmarker = null;
        stream?.getTracks().forEach((tr) => tr.stop());
        stream = null;
        if (videoRef.current) {
          videoRef.current.srcObject = null;
        }
      };

      try {
        setCameraError(null);
        const deviceId = await selectClinicalCameraDeviceId();
        if (!active) return;
        stream = await navigator.mediaDevices.getUserMedia({
          video: deviceId ? { deviceId: { exact: deviceId }, width: 640, height: 480 } : { width: 640, height: 480 },
          audio: false,
        });
        if (!active || !videoRef.current) {
          releaseInitResources();
          return;
        }
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        if (!active) {
          releaseInitResources();
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
            // Try next WASM base
          }
        }
        if (!vision || !active) {
          if (active) setCameraError('ไม่สามารถโหลด MediaPipe WASM สำหรับตรวจจับท่าทางได้');
          releaseInitResources();
          return;
        }

        const POSE_MODEL_PATHS = [
          '/models/pose_landmarker_full.task',
          '/models/pose_landmarker.task',
          'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task',
        ];
        for (const minConf of [0.7, 0.5] as const) {
          for (const modelAssetPath of POSE_MODEL_PATHS) {
            for (const delegate of ['GPU', 'CPU', undefined] as const) {
              try {
                const created = await PoseLandmarker.createFromOptions(vision, {
                  baseOptions: delegate ? { modelAssetPath, delegate } : { modelAssetPath },
                  runningMode: 'VIDEO',
                  numPoses: 2,
                  minPoseDetectionConfidence: minConf,
                  minPosePresenceConfidence: minConf,
                  minTrackingConfidence: minConf,
                });
                if (!active) {
                  created.close();
                  releaseInitResources();
                  return;
                }
                poseLandmarker = created;
                break;
              } catch {
                // Try next model / delegate / confidence tier
              }
            }
            if (poseLandmarker) break;
          }
          if (poseLandmarker) break;
        }
        if (!poseLandmarker || !active) {
          if (active) setCameraError('ไม่สามารถเริ่มโมเดลตรวจจับท่าทางกายภาพได้');
          releaseInitResources();
          return;
        }

        let lastVideoTime = -1;
        const loop = () => {
          if (!active || !videoRef.current || !poseLandmarker) return;
          try {
            const vid = videoRef.current;
            if (vid.readyState >= 2 && vid.videoWidth > 0 && vid.videoHeight > 0 && vid.currentTime !== lastVideoTime) {
              lastVideoTime = vid.currentTime;
              const now = performance.now();
              const detectTsMs = Math.max(lastDetectTsMs + 1, Math.round(now));
              lastDetectTsMs = detectTsMs;
              const res = poseLandmarker.detectForVideo(vid, detectTsMs);
              const detectedPeople = res.landmarks?.length ?? 0;

              if (detectedPeople > 1) {
                singlePersonStreak = 0;
                setMultiPersonPaused(true);
                setContinuousGuardOk(false);
                if (now - lastMultiPersonTtsAtRef.current >= 4000) {
                  lastMultiPersonTtsAtRef.current = now;
                  voiceRef.current.speak(
                    'ตรวจพบมีบุคคลอยู่หลายคนในเฟรม กรุณาให้คนไข้ยืนหน้ากล้องเพียงลำพัง ระบบจึงจะนับรอบการฝึกต่อครับ',
                    'th',
                  );
                }
                rafId = requestAnimationFrame(loop);
                return;
              }

              if (detectedPeople === 1) {
                singlePersonStreak += 1;
                if (singlePersonStreak >= 3) {
                  setMultiPersonPaused(false);
                  setContinuousGuardOk(true);
                }
              }

              const candidates = res.landmarks ?? [];
              const subject = selectSubjectPoseWithIdentity(candidates);
              if (subject) {
                const subjectIdx = Math.max(
                  0,
                  candidates.findIndex(
                    (c) =>
                      c === subject ||
                      (c.length > 12 &&
                        subject.length > 12 &&
                        Math.abs((c[11]?.x ?? 0) - (subject[11]?.x ?? 0)) < 1e-5 &&
                        Math.abs((c[12]?.x ?? 0) - (subject[12]?.x ?? 0)) < 1e-5),
                  ),
                );
                const stabilized = stabilizeArmIdentity(subject, prevPose);
                const smoothedLandmarks = filterRef.current.filterLandmarks(stabilized, now);
                const rawWorld = res.worldLandmarks?.[subjectIdx] ?? res.worldLandmarks?.[0];
                const smoothedWorld = rawWorld
                  ? worldFilterRef.current.filterLandmarks(rawWorld, now)
                  : undefined;
                baselineRef.current.update(smoothedLandmarks);

                const det = selectedExercise.detection;
                const strokeTargetJoint =
                  det.jointGroup === 'neck' || det.expectedPlane === 'shrug_roll'
                    ? 'spine'
                    : det.side === 'left'
                      ? 'left_shoulder'
                      : 'right_shoulder';
                const strokeJointMode: StrokeJointMode =
                  det.expectedPlane === 'bilateral_push'
                    ? 'bobath_push'
                    : det.jointGroup === 'elbow'
                      ? 'elbow'
                      : det.expectedPlane === 'cross_body'
                        ? 'cross_body'
                        : det.expectedPlane === 'scap_protract'
                          ? 'scap_protract'
                          : det.expectedPlane === 'hand_to_head'
                            ? 'hand_to_head'
                            : det.expectedPlane === 'forearm_turn'
                              ? 'forearm_turn'
                              : det.expectedPlane === 'wrist_finger'
                                ? 'wrist_finger'
                                : det.expectedPlane === 'thumb_pinch'
                                  ? 'thumb_pinch'
                                  : det.expectedPlane === 'rotation'
                                    ? 'shoulder_rotation'
                                    : det.expectedPlane === 'shrug_roll'
                                      ? 'shoulder_shrug'
                                      : det.expectedPlane === 'bilateral'
                                        ? 'bilateral_raise'
                                        : det.expectedPlane === 'neck_turn'
                                          ? 'neck_turn'
                                          : det.jointGroup === 'neck'
                                            ? 'neck'
                                            : 'default';
                const strokeSolved = solveStrokeClinicalJointBiomechanics({
                  landmarks: smoothedLandmarks,
                  worldLandmarks: smoothedWorld,
                  targetJoint: strokeTargetJoint,
                  exerciseSide:
                    det.side === 'left'
                      ? 'LEFT'
                      : det.side === 'right'
                        ? 'RIGHT'
                        : det.side === 'both'
                          ? 'BOTH'
                          : undefined,
                  expectedPlane: det.expectedPlane ?? 'any',
                  targetRomDeg: det.targetRomDeg ?? det.targetMinDeg,
                  jointMode: strokeJointMode,
                  baseline: baselineRef.current.getBaseline() ?? undefined,
                  videoAspect: vid.videoWidth / vid.videoHeight,
                });
                setStrokeWarningTh(
                  strokeSolved.sideViolationBannerTh ??
                    strokeSolved.wrongPlaneBannerTh ??
                    strokeSolved.postureBannerTh,
                );
                setStrokeClinicalAngleDeg(strokeSolved.romDeg);

                const snap = pipelineRef.current.step(smoothedLandmarks, now);
                setSnapshot(snap);

                const dtSec = Math.max(0.03, (now - lastTickAt) / 1000);
                const rawAngle = strokeSolved.romDeg ?? snap.currentAngleDeg ?? snap.lastReliableAngleDeg ?? 0;
                const packet = buildTelemetryPacketFromLandmarks({
                  landmarks: smoothedLandmarks,
                  prevLandmarks: prevPose,
                  exercise: selectedExercise,
                  primaryAngleDeg: rawAngle,
                  prevAngleDeg: prevAngle,
                  dtSec,
                });
                const decision = agentRef.current.evaluate(packet);
                setAiDecision(decision);
                prevPose = smoothedLandmarks;
                prevAngle = rawAngle;
                lastTickAt = now;

                if (decision.kiosk_state === 'EMERGENCY_HALT') {
                  voiceRef.current.speak(decision.tts_voice_th, 'th');
                  chimeRef.current.playTone('alert');
                  setPainStopped(true);
                  void apiClient
                    .createStaffCall(
                      'KIOSK-OPD-01',
                      `EMERGENCY_HALT during ${selectedExercise.code}: ${decision.tts_voice_th}`,
                      patient.id,
                    )
                    .catch(() => {});
                  return;
                }

                if (snap.currentAngleDeg !== null && replayLogRef.current.length < 600) {
                  replayLogRef.current.push({
                    tMs: Math.round(now - startedAt),
                    angleDeg: snap.currentAngleDeg,
                  });
                }

                if (snap.repJustCompleted) {
                  chimeRef.current.playTone('rep');
                }
                if (decision.telemetry_analysis.compensation_detected[0] !== 'NONE' && decision.tts_voice_th) {
                  voiceRef.current.speak(decision.tts_voice_th, 'th');
                } else if (snap.warningMessage) {
                  voiceRef.current.speak(pick(snap.warningMessage), lang);
                }
                if (snap.sessionCompleted && !recordedSessionRef.current) {
                  recordedSessionRef.current = true;
                  chimeRef.current.playTone('confirm');
                  void apiClient.recordRehabSession({
                    patientId: patient.id,
                    activityType: 'physio_exercise',
                    exerciseId: selectedExercise.id,
                    exerciseName: pick(selectedExercise.name),
                    completedSets: snap.targetSets,
                    completedReps: snap.targetReps,
                    accuracyPercent: snap.accuracyPercent,
                    maxRomDeg: snap.maxRomDeg,
                    painStopTriggered: false,
                    aiFeedback: `Max ROM ${snap.maxRomDeg}°, Accuracy ${snap.accuracyPercent}%`,
                    kinematicsReplay: replayLogRef.current.slice(-180),
                  }).catch(() => {});
                  return;
                }
              }
            }
          } catch {
            // Keep pose RAF loop resilient across transient frame failures
          }
          if (active && !recordedSessionRef.current) {
            rafId = requestAnimationFrame(loop);
          }
        };
        rafId = requestAnimationFrame(loop);
      } catch (err) {
        if (active) {
          setCameraError(err instanceof Error ? err.message : 'ไม่สามารถเปิดกล้องสำหรับกายภาพบำบัดได้');
        }
        releaseInitResources();
      }
    }

    initPoseCamera();
    return () => {
      active = false;
      cancelAnimationFrame(rafId);
      poseLandmarker?.close();
      stream?.getTracks().forEach((tr) => tr.stop());
      voiceRef.current.flush();
    };
  }, [selectedExercise, painStopped, apiClient, patient.id, lang, pick]);

  const handlePainStop = async () => {
    setPainStopped(true);
    voiceRef.current.flush();
    chimeRef.current.playTone('alert');
    await apiClient.recordRehabSession({
      patientId: patient.id,
      activityType: 'physio_exercise',
      exerciseId: selectedExercise.id,
      exerciseName: pick(selectedExercise.name),
      completedSets: snapshot?.currentSet ?? 1,
      completedReps: snapshot?.currentRep ?? 0,
      accuracyPercent: snapshot?.accuracyPercent ?? 100,
      maxRomDeg: snapshot?.maxRomDeg ?? 0,
      painStopTriggered: true,
      aiFeedback: 'Emergency Pain Stop triggered by patient',
      kinematicsReplay: replayLogRef.current.slice(-180),
    });
    await apiClient.createStaffCall('KIOSK-OPD-01', `EMERGENCY PAIN STOP: ${patient.fullName}`, patient.id);
  };

  return (
    <div className="mx-auto max-w-6xl space-y-4">
      {/* Top Clinical Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-3xl bg-white p-4 shadow border border-slate-200">
        <div>
          <span className="inline-block rounded-full bg-teal-50 px-3 py-1 text-xs font-bold text-teal-700">
            {t('cleanCameraBadge')}
          </span>
          <h2 className="mt-1 text-xl font-bold text-slate-900">{pick(selectedExercise.name)}</h2>
          <p className="text-xs text-slate-500">{pick(selectedExercise.instruction)}</p>
        </div>

        <div className="flex items-center gap-2">
          <select
            value={selectedExercise.id}
            onChange={(e) => {
              const found = CLINICAL_EXERCISES.find((ex) => ex.id === e.target.value);
              if (found) setSelectedExercise(found);
            }}
            className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs font-semibold text-slate-700"
          >
            {CLINICAL_EXERCISES.map((ex) => (
              <option key={ex.id} value={ex.id}>
                {ex.code} — {pick(ex.name)}
              </option>
            ))}
          </select>
          <button
            onClick={onFinished}
            className="rounded-xl border border-slate-200 px-3 py-2 text-xs font-semibold text-slate-600 hover:bg-slate-50"
          >
            <RotateCcw className="inline h-3.5 w-3.5 mr-1" /> Finish
          </button>
        </div>
      </div>

      {/* Main Clean Camera & HUD Grid */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="relative lg:col-span-2 overflow-hidden rounded-3xl bg-slate-950 aspect-[4/3] shadow-xl">
          {/* Clean Mirror Video — Zero MediaPipe Skeleton Lines Overlay */}
          <video ref={videoRef} playsInline muted className="h-full w-full object-cover -scale-x-100" />

          {/* Clean Framing Guide Box */}
          <div className="pointer-events-none absolute inset-6 rounded-3xl border-2 border-dashed border-white/25" />

          {cameraError && (
            <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-slate-950/85 p-6 text-center text-white">
              <p className="text-sm font-bold text-rose-300">{cameraError}</p>
            </div>
          )}

          {multiPersonPaused && (
            <div className="absolute top-4 inset-x-4 z-20 rounded-2xl border-2 border-amber-300 bg-amber-500/95 px-4 py-3 text-slate-950 shadow-xl">
              <div className="text-sm font-extrabold">
                <ShieldAlert className="inline h-4 w-4 mr-1.5" />
                ตรวจพบมีบุคคลอยู่หลายคนในเฟรม
              </div>
              <div className="mt-0.5 text-xs font-bold">
                กรุณาให้คนไข้ยืนหน้ากล้องเพียงลำพัง ระบบจึงจะนับรอบการฝึกต่อครับ
              </div>
            </div>
          )}

          {!continuousGuardOk && !multiPersonPaused && (
            <div className="absolute top-4 inset-x-4 z-20 rounded-2xl border-2 border-rose-300 bg-rose-600/95 px-4 py-3 text-white shadow-xl">
              <div className="text-sm font-extrabold">
                <ShieldAlert className="inline h-4 w-4 mr-1.5" />
                คุณไม่ใช่คนไข้ที่ลงทะเบียนไว้
              </div>
              <div className="mt-0.5 text-xs font-semibold text-rose-100">
                ระบบจะไม่นับรอบการฝึก กรุณาให้คนไข้ตัวจริงเข้าฝึกครับ
              </div>
            </div>
          )}

          {(strokeWarningTh || snapshot?.warningMessage) && (
            <div className="absolute top-4 inset-x-4 rounded-2xl bg-amber-500/95 px-4 py-3 text-sm font-bold text-slate-950 shadow-lg">
              {strokeWarningTh ?? (snapshot?.warningMessage ? pick(snapshot.warningMessage) : '')}
            </div>
          )}

          {snapshot?.sessionCompleted && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-emerald-950/85 text-white p-6 text-center">
              <CheckCircle2 className="h-16 w-16 text-emerald-400" />
              <h3 className="mt-3 text-2xl font-bold">Session Completed!</h3>
              <p className="mt-1 text-sm text-emerald-100">
                Accuracy {snapshot.accuracyPercent}% • Max ROM {snapshot.maxRomDeg}°
              </p>
              <button
                onClick={onFinished}
                className="mt-6 rounded-2xl bg-white px-6 py-3 text-sm font-bold text-emerald-900"
              >
                Return to Dashboard
              </button>
            </div>
          )}

          {painStopped && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-rose-950/90 text-white p-6 text-center">
              <AlertOctagon className="h-16 w-16 text-rose-400" />
              <h3 className="mt-3 text-2xl font-bold">{t('painEmergencyStop')}</h3>
              <p className="mt-2 text-sm text-rose-100">{t('staffCalledConfirm')}</p>
              <button
                onClick={() => setPinOverrideOpen(true)}
                className="mt-6 rounded-xl border border-white/30 px-4 py-2 text-xs font-semibold text-white"
              >
                Staff Override PIN
              </button>
            </div>
          )}
        </div>

        {/* Live Telemetry HUD */}
        <div className="flex flex-col justify-between rounded-3xl bg-white p-6 shadow-lg border border-slate-200">
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-2xl bg-slate-50 p-4 border border-slate-100">
                <p className="text-xs font-semibold text-slate-500">Joint Angle</p>
                <p className="mt-1 text-3xl font-extrabold text-teal-600">
                  {strokeClinicalAngleDeg !== null && Number.isFinite(strokeClinicalAngleDeg)
                    ? `${strokeClinicalAngleDeg.toFixed(0)}°`
                    : snapshot?.currentAngleDeg !== null && snapshot?.currentAngleDeg !== undefined
                      ? `${snapshot.currentAngleDeg.toFixed(0)}°`
                      : snapshot?.lastReliableAngleDeg !== null && snapshot?.lastReliableAngleDeg !== undefined
                        ? `${snapshot.lastReliableAngleDeg.toFixed(0)}° (Hold)`
                        : '--'}
                </p>
              </div>
              <div className="rounded-2xl bg-slate-50 p-4 border border-slate-100">
                <p className="text-xs font-semibold text-slate-500">Elbow Angle</p>
                <p className="mt-1 text-3xl font-extrabold text-slate-800">
                  {snapshot?.elbowAngleDeg !== null && snapshot?.elbowAngleDeg !== undefined
                    ? `${snapshot.elbowAngleDeg.toFixed(0)}°`
                    : '--'}
                </p>
              </div>
            </div>

            <div className="rounded-2xl bg-teal-50 p-4 border border-teal-100">
              <div className="flex items-center justify-between text-sm font-bold text-teal-900">
                <span>Set {snapshot?.currentSet ?? 1} / {snapshot?.targetSets ?? selectedExercise.detection.defaultSets}</span>
                <span>Rep {snapshot?.currentRep ?? 0} / {snapshot?.targetReps ?? selectedExercise.detection.defaultReps}</span>
              </div>
              <div className="mt-3">
                <div className="flex justify-between text-xs font-semibold text-teal-700 mb-1">
                  <span>Hold Timer {snapshot?.holdPaused ? '(Paused)' : ''}</span>
                  <span>
                    {snapshot?.holdElapsedSec ?? 0}s / {snapshot?.holdTargetSec ?? selectedExercise.detection.holdSec}s
                  </span>
                </div>
                <div className="h-3 w-full overflow-hidden rounded-full bg-teal-200">
                  <div
                    className="h-full bg-teal-600 transition-all"
                    style={{
                      width: `${Math.min(
                        100,
                        ((snapshot?.holdElapsedSec ?? 0) /
                          Math.max(1, snapshot?.holdTargetSec ?? selectedExercise.detection.holdSec)) *
                          100,
                      )}%`,
                    }}
                  />
                </div>
              </div>
            </div>

            <div className="rounded-2xl bg-slate-50 p-3 text-xs text-slate-600">
              State: <span className="font-bold text-slate-900">{aiDecision?.kiosk_state ?? snapshot?.state ?? 'IDLE'}</span> • Max ROM:{' '}
              <span className="font-bold text-slate-900">{snapshot?.maxRomDeg ?? 0}°</span>
            </div>

            {aiDecision && (
              <div className="rounded-2xl bg-slate-950 p-3 text-[11px] text-emerald-300 font-mono border border-slate-800 space-y-1.5">
                <div className="flex items-center justify-between font-sans text-[10px] font-bold uppercase tracking-wider text-slate-400">
                  <span>Clinical Biomechanics AI Agent</span>
                  <span
                    className={`rounded-full px-2 py-0.5 text-[10px] ${
                      aiDecision.ui_directives.highlight_color === 'RED'
                        ? 'bg-rose-600 text-white'
                        : aiDecision.ui_directives.highlight_color === 'AMBER'
                          ? 'bg-amber-500 text-slate-950'
                          : 'bg-emerald-500 text-slate-950'
                    }`}
                  >
                    {aiDecision.ui_directives.highlight_color} • {aiDecision.ui_directives.skeleton_highlight_joint}
                  </span>
                </div>
                <p className="font-sans text-xs font-bold text-white">“{aiDecision.tts_voice_th}”</p>
                <div className="text-[10px] text-slate-400 font-sans">
                  ROM {aiDecision.telemetry_analysis.rom_completion_pct}% • Valid {aiDecision.rep_counter.valid_reps} • Invalid{' '}
                  {aiDecision.rep_counter.invalid_reps} • Flags: {aiDecision.telemetry_analysis.compensation_detected.join(', ')}
                </div>
              </div>
            )}
          </div>

          <div className="mt-6 space-y-2.5">
            <button
              onClick={handlePainStop}
              className="w-full flex items-center justify-center gap-2 rounded-2xl bg-rose-600 py-4 text-base font-bold text-white shadow-lg hover:bg-rose-700"
            >
              <AlertOctagon className="h-5 w-5" />
              {t('painEmergencyStop')}
            </button>
            <button
              onClick={() => apiClient.createStaffCall('KIOSK-OPD-01', `Assistance needed during ${selectedExercise.code}`, patient.id)}
              className="w-full flex items-center justify-center gap-2 rounded-2xl border border-amber-300 bg-amber-50 py-3 text-sm font-bold text-amber-800 hover:bg-amber-100"
            >
              <PhoneCall className="h-4 w-4" />
              {t('callStaffBtn')}
            </button>
          </div>
        </div>
      </div>

      {pinOverrideOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/70 p-4 animate-backdrop-fade">
          <div className="w-full max-w-sm rounded-3xl bg-white p-6 shadow-2xl animate-popup-spring">
            <h3 className="text-lg font-bold text-slate-900">Staff Override PIN</h3>
            <input
              type="password"
              inputMode="numeric"
              pattern="[0-9]*"
              maxLength={6}
              value={staffPin}
              onChange={(e) => setStaffPin(e.target.value.replace(/\D/g, '').slice(0, 6))}
              placeholder="Enter Staff Override PIN (e.g. 2026)"
              className="mt-3 w-full rounded-xl border border-slate-300 px-3 py-2 text-sm"
            />
            <div className="mt-4 flex gap-2">
              <button
                onClick={() => {
                  if (staffPin === '2026') {
                    setPainStopped(false);
                    setContinuousGuardOk(true);
                    setPinOverrideOpen(false);
                    setStaffPin('');
                  }
                }}
                className="flex-1 rounded-xl bg-teal-600 py-2 text-sm font-bold text-white"
              >
                Unlock
              </button>
              <button
                onClick={() => setPinOverrideOpen(false)}
                className="flex-1 rounded-xl border border-slate-200 py-2 text-sm font-semibold text-slate-600"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

