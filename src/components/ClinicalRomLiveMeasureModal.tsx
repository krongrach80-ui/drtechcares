import React, { useEffect, useMemo, useRef, useState } from 'react';
import type { HandLandmarker, NormalizedLandmark, PoseLandmarker } from '@mediapipe/tasks-vision';
import { selectClinicalCameraDeviceId } from '@drtechcares/face-client';
import { CLINICAL_EXERCISES, POSE_INDEX } from '@drtechcares/pose-engine';
import { Camera, CheckCircle2, Lock, RefreshCw, Unlock, X } from 'lucide-react';
import {
  getCachedHandLandmarker,
  getCachedPoseLandmarker,
  getNextMonotonicHandDetectTs,
  getNextMonotonicPoseDetectTs,
} from '../lib/biometricVisionCache';
import { deriveScore0To6FromRomRatio } from '../lib/clinicalAssessmentStore';
import {
  computeTargetAchievementPct,
  getMahidolRubricItem,
  type AssessmentTargetSide,
  type MahidolScore0To6,
} from '../types/clinicalAssessment';
import {
  classifyExerciseJointMode,
  measureLiveJointAngleAndTilt,
} from './SmartRehabKioskUI';

export interface LiveMeasuredRomResult {
  measuredRomDeg: number;
  peakRomDeg: number;
  suggestedScore0To6: MahidolScore0To6;
  applySuggestedScore: boolean;
  sampleFrames: number;
}

export interface ClinicalRomLiveMeasureModalProps {
  open: boolean;
  exerciseCode: string;
  exerciseNameTh: string;
  targetSide: AssessmentTargetSide;
  targetRomDeg: number;
  onClose: () => void;
  onConfirmMeasuredRom: (result: LiveMeasuredRomResult) => void;
}

const UPPER_SKELETON_EDGES: readonly [number, number][] = [
  [POSE_INDEX.LEFT_SHOULDER, POSE_INDEX.RIGHT_SHOULDER],
  [POSE_INDEX.LEFT_SHOULDER, POSE_INDEX.LEFT_ELBOW],
  [POSE_INDEX.LEFT_ELBOW, POSE_INDEX.LEFT_WRIST],
  [POSE_INDEX.RIGHT_SHOULDER, POSE_INDEX.RIGHT_ELBOW],
  [POSE_INDEX.RIGHT_ELBOW, POSE_INDEX.RIGHT_WRIST],
  [POSE_INDEX.LEFT_SHOULDER, POSE_INDEX.LEFT_HIP],
  [POSE_INDEX.RIGHT_SHOULDER, POSE_INDEX.RIGHT_HIP],
  [POSE_INDEX.LEFT_HIP, POSE_INDEX.RIGHT_HIP],
];

function inferExpectedPlane(exerciseCode: string): 'flexion' | 'abduction' | 'any' {
  const upper = exerciseCode.toUpperCase();
  if (upper.includes('FLEXION') || upper.includes('OVERHEAD') || upper.includes('FORWARD')) {
    return 'flexion';
  }
  if (upper.includes('ABDUCTION')) {
    return 'abduction';
  }
  return 'any';
}

function inferTargetJoint(
  exerciseCode: string,
  targetSide: AssessmentTargetSide,
): 'left_shoulder' | 'right_shoulder' | 'spine' {
  const upper = exerciseCode.toUpperCase();
  if (upper.includes('NECK') || upper.includes('TRUNK') || upper.includes('SHRUG')) {
    return 'spine';
  }
  if (targetSide === 'LEFT') return 'left_shoulder';
  if (targetSide === 'BOTH') return 'spine';
  return 'right_shoulder';
}

export function ClinicalRomLiveMeasureModal({
  open,
  exerciseCode,
  exerciseNameTh,
  targetSide,
  targetRomDeg,
  onClose,
  onConfirmMeasuredRom,
}: ClinicalRomLiveMeasureModalProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [cameraReady, setCameraReady] = useState(false);
  const [detectorReady, setDetectorReady] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [liveAngleDeg, setLiveAngleDeg] = useState<number | null>(null);
  const [sustainedPeakDeg, setSustainedPeakDeg] = useState<number>(0);
  const [rawPeakDeg, setRawPeakDeg] = useState<number>(0);
  const [sampleFrames, setSampleFrames] = useState<number>(0);
  const [statusCueTh, setStatusCueTh] = useState<string>('กำลังเตรียมกล้องและระบบวัดองศา AI...');
  const [lockedAngleDeg, setLockedAngleDeg] = useState<number | null>(null);
  const [applySuggestedScore, setApplySuggestedScore] = useState<boolean>(true);
  const [retryNonce, setRetryNonce] = useState(0);

  const smoothedAngleRef = useRef<number | null>(null);
  const sustainedPeakRef = useRef<number>(0);
  const rawPeakRef = useRef<number>(0);
  const sampleFramesRef = useRef<number>(0);
  const stabilityWindowRef = useRef<Array<{ ts: number; deg: number }>>([]);
  const lockedRef = useRef<boolean>(false);

  const exerciseMeta = useMemo(() => {
    const matched = CLINICAL_EXERCISES.find((e) => e.code === exerciseCode || e.id === exerciseCode);
    return {
      exercise_id: matched?.code || exerciseCode,
      catalog_exercise_id: matched?.code || exerciseCode,
      name_th: matched?.name.th || exerciseNameTh,
      name_en: matched?.name.en || '',
      prescribed_side: targetSide,
    };
  }, [exerciseCode, exerciseNameTh, targetSide]);

  const jointMode = useMemo(() => classifyExerciseJointMode(exerciseMeta), [exerciseMeta]);
  const needsHandDetector =
    jointMode === 'wrist_finger' || jointMode === 'thumb_pinch' || jointMode === 'forearm_turn';
  const expectedPlane = useMemo(() => inferExpectedPlane(exerciseCode), [exerciseCode]);
  const targetJoint = useMemo(
    () => inferTargetJoint(exerciseCode, targetSide),
    [exerciseCode, targetSide],
  );

  const resetMeasurements = () => {
    smoothedAngleRef.current = null;
    sustainedPeakRef.current = 0;
    rawPeakRef.current = 0;
    sampleFramesRef.current = 0;
    stabilityWindowRef.current = [];
    lockedRef.current = false;
    setLockedAngleDeg(null);
    setLiveAngleDeg(null);
    setSustainedPeakDeg(0);
    setRawPeakDeg(0);
    setSampleFrames(0);
  };

  useEffect(() => {
    if (!open) return;
    resetMeasurements();
    let active = true;
    let stream: MediaStream | null = null;
    let poseLandmarker: PoseLandmarker | null = null;
    let handLandmarker: HandLandmarker | null = null;
    let rafId = 0;
    let lastVideoTime = -1;
    let lastUiSyncAt = 0;

    setCameraReady(false);
    setDetectorReady(false);
    setCameraError(null);
    setStatusCueTh('กำลังเชื่อมต่อกล้องและตัวประมวลผลท่าทาง MediaPipe...');

    async function start() {
      try {
        const deviceId = await selectClinicalCameraDeviceId().catch(() => undefined);
        const chains: MediaTrackConstraints[] = deviceId
          ? [
              { deviceId: { ideal: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } },
              { deviceId: { ideal: deviceId } },
              { width: { ideal: 1280 }, height: { ideal: 720 } },
            ]
          : [
              { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
              { width: { ideal: 640 }, height: { ideal: 480 } },
            ];

        let opened: MediaStream | null = null;
        let lastErr: unknown = null;
        for (const vc of chains) {
          try {
            opened = await navigator.mediaDevices.getUserMedia({ video: vc, audio: false });
            break;
          } catch (err) {
            lastErr = err;
          }
        }
        if (!opened) {
          try {
            opened = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
          } catch (err) {
            throw lastErr ?? err;
          }
        }
        stream = opened;
        if (!active || !videoRef.current) {
          stream.getTracks().forEach((t) => t.stop());
          stream = null;
          return;
        }
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        if (!active) return;
        setCameraReady(true);

        const [poseHandle, handHandle] = await Promise.all([
          getCachedPoseLandmarker({ numPoses: 1 }),
          needsHandDetector
            ? getCachedHandLandmarker({ numHands: 2, minConfidence: 0.35 }).catch(() => null)
            : Promise.resolve(null),
        ]);
        if (!active) return;
        poseLandmarker = poseHandle.instance;
        handLandmarker = handHandle?.instance ?? null;
        setDetectorReady(true);
        setStatusCueTh('พร้อมวัดองศาจริง — ให้ผู้ป่วยทำท่าค้างไว้ที่จุดสูงสุดอย่างน้อย 0.3 วินาที');

        const drawOverlay = (
          lms: readonly NormalizedLandmark[],
          angleVal: number | null,
        ) => {
          const canvas = canvasRef.current;
          const video = videoRef.current;
          if (!canvas || !video) return;
          const cw = video.clientWidth || 640;
          const ch = video.clientHeight || 360;
          if (canvas.width !== cw || canvas.height !== ch) {
            canvas.width = cw;
            canvas.height = ch;
          }
          const ctx = canvas.getContext('2d');
          if (!ctx) return;
          ctx.clearRect(0, 0, cw, ch);
          ctx.shadowBlur = 0;

          const vw = video.videoWidth || cw;
          const vh = video.videoHeight || ch;
          const scale = Math.max(cw / vw, ch / vh);
          const renderedW = vw * scale;
          const renderedH = vh * scale;
          const offsetX = (renderedW - cw) * 0.5;
          const offsetY = (renderedH - ch) * 0.5;

          const mapPt = (lm: NormalizedLandmark) => ({
            x: (1 - lm.x) * renderedW - offsetX,
            y: lm.y * renderedH - offsetY,
          });

          const isTargetReached = angleVal !== null && angleVal >= targetRomDeg * 0.85;
          const edgeColor = isTargetReached ? '#10b981' : '#0284c7';

          ctx.lineCap = 'round';
          for (const [aIdx, bIdx] of UPPER_SKELETON_EDGES) {
            const a = lms[aIdx];
            const b = lms[bIdx];
            if (!a || !b || (a.visibility ?? 0) < 0.25 || (b.visibility ?? 0) < 0.25) continue;
            const pa = mapPt(a);
            const pb = mapPt(b);
            ctx.beginPath();
            ctx.moveTo(pa.x, pa.y);
            ctx.lineTo(pb.x, pb.y);
            ctx.strokeStyle = 'rgba(15, 23, 42, 0.55)';
            ctx.lineWidth = 6;
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(pa.x, pa.y);
            ctx.lineTo(pb.x, pb.y);
            ctx.strokeStyle = edgeColor;
            ctx.lineWidth = 3.2;
            ctx.stroke();
          }

          const jointIndices = [
            POSE_INDEX.LEFT_SHOULDER,
            POSE_INDEX.RIGHT_SHOULDER,
            POSE_INDEX.LEFT_ELBOW,
            POSE_INDEX.RIGHT_ELBOW,
            POSE_INDEX.LEFT_WRIST,
            POSE_INDEX.RIGHT_WRIST,
          ];
          for (const idx of jointIndices) {
            const lm = lms[idx];
            if (!lm || (lm.visibility ?? 0) < 0.25) continue;
            const p = mapPt(lm);
            ctx.beginPath();
            ctx.arc(p.x, p.y, 5.5, 0, Math.PI * 2);
            ctx.fillStyle = '#ffffff';
            ctx.fill();
            ctx.lineWidth = 2.5;
            ctx.strokeStyle = edgeColor;
            ctx.stroke();
          }
        };

        const loop = () => {
          if (!active || !videoRef.current || !poseLandmarker) return;
          try {
            const video = videoRef.current;
            const now = performance.now();
            if (video.readyState >= 2 && video.videoWidth > 0 && video.currentTime !== lastVideoTime) {
              lastVideoTime = video.currentTime;
              const poseTs = getNextMonotonicPoseDetectTs(now);
              const poseRes = poseLandmarker.detectForVideo(video, poseTs);
              const landmarks = poseRes.landmarks?.[0];
              const worldLandmarks = poseRes.worldLandmarks?.[0];

              if (!landmarks || landmarks.length < 25) {
                const canvas = canvasRef.current;
                if (canvas) {
                  const ctx = canvas.getContext('2d');
                  ctx?.clearRect(0, 0, canvas.width, canvas.height);
                }
                if (now - lastUiSyncAt >= 180) {
                  lastUiSyncAt = now;
                  setLiveAngleDeg(null);
                  setStatusCueTh('ไม่พบลำตัวและหัวไหล่ในกรอบภาพ กรุณาถอยห่างจากกล้องเล็กน้อย');
                }
              } else {
                let handLandmarksBySide:
                  | { left?: readonly NormalizedLandmark[]; right?: readonly NormalizedLandmark[] }
                  | undefined;
                if (handLandmarker) {
                  try {
                    const handTs = getNextMonotonicHandDetectTs(now);
                    const handRes = handLandmarker.detectForVideo(video, handTs);
                    if (handRes.landmarks && handRes.landmarks.length > 0) {
                      handLandmarksBySide = {};
                      handRes.landmarks.forEach((hlms, idx) => {
                        const handed = handRes.handedness?.[idx]?.[0]?.categoryName?.toLowerCase() ?? '';
                        if (handed.includes('left')) {
                          handLandmarksBySide!.left = hlms;
                        } else if (handed.includes('right')) {
                          handLandmarksBySide!.right = hlms;
                        }
                      });
                    }
                  } catch {
                    // Hand detector optional fallback
                  }
                }

                const measured = measureLiveJointAngleAndTilt(
                  landmarks,
                  targetJoint,
                  expectedPlane,
                  worldLandmarks,
                  exerciseMeta,
                  {
                    handLandmarksBySide,
                    targetRomDeg,
                    videoAspect: (video.videoWidth || 1280) / Math.max(1, video.videoHeight || 720),
                  },
                );

                const rawRom =
                  typeof measured.romDeg === 'number' && Number.isFinite(measured.romDeg)
                    ? Math.max(0, Math.min(220, measured.romDeg))
                    : null;

                if (rawRom !== null && !lockedRef.current) {
                  const prevSmooth = smoothedAngleRef.current;
                  const smooth = prevSmooth === null ? rawRom : prevSmooth * 0.65 + rawRom * 0.35;
                  smoothedAngleRef.current = smooth;
                  sampleFramesRef.current += 1;

                  if (smooth > rawPeakRef.current) {
                    rawPeakRef.current = Number(smooth.toFixed(1));
                  }

                  // Sustained peak gate: require >= 300ms continuous holding within +/- 9 deg to reject spikes
                  const win = stabilityWindowRef.current;
                  win.push({ ts: now, deg: smooth });
                  while (win.length > 0 && now - win[0].ts > 420) {
                    win.shift();
                  }
                  if (win.length >= 4 && now - win[0].ts >= 280) {
                    let minDeg = Infinity;
                    let maxDeg = -Infinity;
                    let sumDeg = 0;
                    for (const pt of win) {
                      if (pt.deg < minDeg) minDeg = pt.deg;
                      if (pt.deg > maxDeg) maxDeg = pt.deg;
                      sumDeg += pt.deg;
                    }
                    if (maxDeg - minDeg <= 10) {
                      const avgStable = Number((sumDeg / win.length).toFixed(1));
                      if (avgStable > sustainedPeakRef.current) {
                        sustainedPeakRef.current = avgStable;
                      }
                    }
                  }
                }

                drawOverlay(landmarks, smoothedAngleRef.current);

                if (now - lastUiSyncAt >= 90) {
                  lastUiSyncAt = now;
                  const curVal =
                    smoothedAngleRef.current !== null
                      ? Number(smoothedAngleRef.current.toFixed(1))
                      : null;
                  setLiveAngleDeg(curVal);
                  setSustainedPeakDeg(sustainedPeakRef.current);
                  setRawPeakDeg(rawPeakRef.current);
                  setSampleFrames(sampleFramesRef.current);
                  if (measured.wrongSideBannerTh) {
                    setStatusCueTh(measured.wrongSideBannerTh);
                  } else if (measured.postureBannerTh) {
                    setStatusCueTh(measured.postureBannerTh);
                  } else if (measured.lowVisibilityAlertTh) {
                    setStatusCueTh(measured.lowVisibilityAlertTh);
                  } else if (curVal !== null) {
                    setStatusCueTh(
                      `กำลังจับองศาจริง: ${curVal.toFixed(1)}° • ค้างที่จุดสูงสุด 0.3 วินาทีเพื่อยืนยันค่าเสถียร`,
                    );
                  }
                }
              }
            }
          } catch {
            // Keep measurement loop alive across transient frame errors
          }
          if (active) {
            rafId = requestAnimationFrame(loop);
          }
        };

        rafId = requestAnimationFrame(loop);
      } catch (err) {
        if (active) {
          setCameraError(
            err instanceof Error
              ? err.message
              : 'ไม่สามารถเปิดกล้องเพื่อวัดองศาได้ กรุณาอนุญาตสิทธิ์กล้อง',
          );
        }
      }
    }

    void start();
    return () => {
      active = false;
      cancelAnimationFrame(rafId);
      poseLandmarker = null;
      handLandmarker = null;
      stream?.getTracks().forEach((t) => t.stop());
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
    };
  }, [open, retryNonce, exerciseMeta, expectedPlane, targetJoint, targetRomDeg, needsHandDetector]);

  if (!open) return null;

  const effectiveMeasuredDeg =
    lockedAngleDeg !== null
      ? lockedAngleDeg
      : sustainedPeakDeg > 0
        ? sustainedPeakDeg
        : rawPeakDeg > 0
          ? rawPeakDeg
          : liveAngleDeg ?? 0;

  const targetPct = computeTargetAchievementPct(effectiveMeasuredDeg, targetRomDeg);
  const suggestedScore = deriveScore0To6FromRomRatio(effectiveMeasuredDeg, targetRomDeg);
  const suggestedRubric = getMahidolRubricItem(suggestedScore);

  const handleToggleLock = () => {
    if (lockedAngleDeg !== null) {
      lockedRef.current = false;
      setLockedAngleDeg(null);
      return;
    }
    const snap = liveAngleDeg ?? sustainedPeakDeg ?? rawPeakDeg;
    if (snap > 0) {
      lockedRef.current = true;
      setLockedAngleDeg(Number(snap.toFixed(1)));
    }
  };

  return (
    <div className="fixed inset-0 z-[70] flex items-center justify-center bg-[#0F4C81]/45 p-3 sm:p-5 animate-backdrop-fade">
      <div className="w-full max-w-3xl max-h-[92vh] overflow-y-auto rounded-3xl border border-sky-200 bg-white p-5 shadow-2xl space-y-4 animate-popup-spring">
        <div className="flex items-start justify-between gap-3 border-b border-slate-100 pb-3">
          <div className="flex items-start gap-2.5 min-w-0">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-[#0F4C81] to-[#059669] text-white shadow-sm">
              <Camera className="h-5 w-5" />
            </div>
            <div className="min-w-0">
              <h3 className="text-base font-black text-slate-900 truncate">
                เครื่องมือวัดองศาข้อต่อจริงด้วยกล้อง AI (Live Clinical Goniometer)
              </h3>
              <p className="text-xs text-slate-500">
                ท่าที่วัด: <span className="font-bold text-[#0F4C81]">{exerciseNameTh}</span> • ข้าง:{' '}
                <span className="font-bold text-emerald-700">
                  {targetSide === 'LEFT' ? 'ซ้าย' : targetSide === 'RIGHT' ? 'ขวา' : 'ทั้งสองข้าง'}
                </span>{' '}
                • เป้าหมายคลินิก <span className="font-bold text-slate-800">{targetRomDeg.toFixed(1)}°</span>
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-xl p-1.5 text-slate-400 hover:bg-slate-100 hover:text-slate-700 cursor-pointer"
            aria-label="ปิดหน้าต่างวัดองศา"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Live Camera Viewport (opaque high-contrast overlays for 60FPS smoothness) */}
        <div className="relative aspect-video w-full overflow-hidden rounded-2xl border-2 border-sky-200 bg-slate-950">
          <video
            ref={videoRef}
            muted
            playsInline
            className="h-full w-full -scale-x-100 object-cover"
          />
          <canvas
            ref={canvasRef}
            className="pointer-events-none absolute inset-0 h-full w-full"
          />

          <div className="pointer-events-none absolute left-3 top-3 flex flex-wrap items-center gap-2">
            <span className="rounded-full bg-slate-900/90 border border-white/20 px-3 py-1 text-[11px] font-extrabold text-white">
              {detectorReady ? 'AI Pose พร้อมวัดผลจริง' : 'กำลังโหลดโมเดลตรวจจับ...'}
            </span>
            {lockedAngleDeg !== null && (
              <span className="rounded-full bg-amber-500 px-3 py-1 text-[11px] font-black text-white">
                ล็อกค่าไว้ที่ {lockedAngleDeg.toFixed(1)}°
              </span>
            )}
          </div>

          <div className="pointer-events-none absolute right-3 top-3 rounded-2xl bg-white/95 border border-emerald-300 px-3.5 py-2 text-right shadow-lg">
            <div className="text-[10px] font-bold text-slate-500">องศาขณะนี้ (Live)</div>
            <div className="text-2xl font-black text-[#0F4C81] tabular-nums">
              {liveAngleDeg !== null ? `${liveAngleDeg.toFixed(1)}°` : '--.-°'}
            </div>
          </div>

          <div className="pointer-events-none absolute inset-x-3 bottom-3 rounded-xl bg-slate-900/90 border border-white/15 px-3.5 py-2 text-xs font-bold text-white">
            {statusCueTh}
          </div>

          {(!cameraReady || cameraError) && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-white/95 p-5 text-center">
              <Camera className="h-9 w-9 text-[#0F4C81] mb-2" />
              {cameraError ? (
                <>
                  <div className="text-sm font-black text-rose-700">ไม่สามารถเปิดกล้องได้</div>
                  <div className="mt-1 text-xs text-slate-600 max-w-md">{cameraError}</div>
                  <button
                    type="button"
                    onClick={() => setRetryNonce((n) => n + 1)}
                    className="mt-3 rounded-xl bg-[#0F4C81] px-4 py-2 text-xs font-bold text-white cursor-pointer"
                  >
                    ลองเปิดกล้องอีกครั้ง
                  </button>
                </>
              ) : (
                <div className="text-xs font-bold text-[#0F4C81]">กำลังเปิดกล้องและเตรียมตัววัดองศา...</div>
              )}
            </div>
          )}
        </div>

        {/* Live Telemetry & Sustained Peak Readouts */}
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          <div className="rounded-2xl border border-emerald-200 bg-emerald-50/60 p-3.5">
            <div className="text-[11px] font-bold text-emerald-800">
              ค่าองศาสูงสุดแบบเสถียร (แนะนำบันทึก)
            </div>
            <div className="mt-1 text-2xl font-black text-emerald-700 tabular-nums">
              {effectiveMeasuredDeg.toFixed(1)}°
            </div>
            <div className="mt-0.5 text-[10px] font-semibold text-slate-600">
              Peak ดิบสูงสุด: {rawPeakDeg.toFixed(1)}° • เก็บข้อมูล {sampleFrames} เฟรม
            </div>
          </div>

          <div className="rounded-2xl border border-sky-200 bg-sky-50/60 p-3.5">
            <div className="text-[11px] font-bold text-[#0F4C81]">ความสำเร็จเทียบเป้าหมายคลินิก</div>
            <div className="mt-1 text-2xl font-black text-[#0F4C81] tabular-nums">
              {targetPct.toFixed(1)}%
            </div>
            <div className="mt-1.5 h-2 w-full overflow-hidden rounded-full bg-sky-100">
              <div
                className="h-full rounded-full bg-gradient-to-r from-[#0284C7] to-[#10B981]"
                style={{ width: `${Math.min(100, targetPct)}%` }}
              />
            </div>
          </div>

          <div className="rounded-2xl border border-indigo-200 bg-indigo-50/60 p-3.5">
            <div className="text-[11px] font-bold text-indigo-800">ระดับมหิดลแนะนำตามมุมจริง</div>
            <div className="mt-1 text-2xl font-black text-indigo-700 tabular-nums">
              ระดับ {suggestedScore} / 6
            </div>
            <div className="mt-0.5 text-[10px] font-semibold text-indigo-900 line-clamp-2">
              {suggestedRubric.shortLabelTh}
            </div>
          </div>
        </div>

        <label className="flex items-center gap-2 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs font-bold text-slate-700 cursor-pointer">
          <input
            type="checkbox"
            checked={applySuggestedScore}
            onChange={(e) => setApplySuggestedScore(e.target.checked)}
            className="h-4 w-4 accent-emerald-600"
          />
          <span>อัปเดตคะแนนประเมินมหิดล (ระดับ {suggestedScore}/6) ตามมุมที่วัดได้ให้อัตโนมัติ (แก้ไขภายหลังได้)</span>
        </label>

        {/* Action Controls */}
        <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={resetMeasurements}
              className="btn-tactile inline-flex items-center gap-1.5 rounded-xl border border-slate-200 bg-white px-3.5 py-2 text-xs font-bold text-slate-700 hover:bg-slate-50 cursor-pointer"
            >
              <RefreshCw className="h-3.5 w-3.5" />
              <span>เริ่มจับค่าใหม่</span>
            </button>
            <button
              type="button"
              disabled={!liveAngleDeg && !sustainedPeakDeg}
              onClick={handleToggleLock}
              className={`btn-tactile inline-flex items-center gap-1.5 rounded-xl border px-3.5 py-2 text-xs font-bold cursor-pointer disabled:opacity-50 ${
                lockedAngleDeg !== null
                  ? 'border-amber-300 bg-amber-50 text-amber-800'
                  : 'border-sky-200 bg-sky-50 text-[#0F4C81]'
              }`}
            >
              {lockedAngleDeg !== null ? <Unlock className="h-3.5 w-3.5" /> : <Lock className="h-3.5 w-3.5" />}
              <span>{lockedAngleDeg !== null ? 'ปลดล็อกการวัด' : 'ล็อกค่าขณะนี้'}</span>
            </button>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={onClose}
              className="rounded-xl border border-slate-200 bg-white px-4 py-2 text-xs font-bold text-slate-600 hover:bg-slate-50 cursor-pointer"
            >
              ยกเลิก
            </button>
            <button
              type="button"
              disabled={effectiveMeasuredDeg <= 0}
              onClick={() => {
                onConfirmMeasuredRom({
                  measuredRomDeg: Number(effectiveMeasuredDeg.toFixed(1)),
                  peakRomDeg: Number(Math.max(effectiveMeasuredDeg, rawPeakDeg).toFixed(1)),
                  suggestedScore0To6: suggestedScore,
                  applySuggestedScore,
                  sampleFrames,
                });
                onClose();
              }}
              className="btn-tactile inline-flex items-center gap-1.5 rounded-xl bg-gradient-to-r from-[#0F4C81] to-[#059669] px-4 py-2 text-xs font-black text-white shadow-md hover:brightness-105 disabled:opacity-45 cursor-pointer"
            >
              <CheckCircle2 className="h-4 w-4" />
              <span>นำค่า {effectiveMeasuredDeg.toFixed(1)}° ไปบันทึก</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

