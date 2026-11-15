import React, { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { Activity, AlertTriangle, CheckCircle2, Ruler, ShieldAlert, UserCheck, Users, UserX } from 'lucide-react';
import {
  getPrescribedLimbPromptTh,
  getPrescribedPlanePromptTh,
  resolveExpectedPlaneForExercise,
  type ExercisePrescriptionOption,
  type KioskDecisionPayload,
} from '../hooks/useKioskSocket';
import { drawWholeBody133OnCanvas, type Keypoint3DPoint } from '../hooks/useEdgeVisionSocket';

interface SkeletonCanvasProps {
  payload: KioskDecisionPayload;
  exercise: ExercisePrescriptionOption;
  isDemoMode: boolean;
  distanceM: number;
  trunkTiltDeg: number;
  hasLiveCamera?: boolean;
  liveVideoOverlay?: React.ReactNode;
  children?: React.ReactNode;
}

function polarPoint(cx: number, cy: number, radius: number, angleDegFromRight: number) {
  const rad = (angleDegFromRight * Math.PI) / 180;
  return {
    x: cx + radius * Math.cos(rad),
    y: cy + radius * Math.sin(rad),
  };
}

function describeArcSector(
  cx: number,
  cy: number,
  radius: number,
  startDeg: number,
  endDeg: number,
  sweepFlag: 0 | 1,
): string {
  const start = polarPoint(cx, cy, radius, startDeg);
  const end = polarPoint(cx, cy, radius, endDeg);
  const largeArc = Math.abs(endDeg - startDeg) > 180 ? 1 : 0;
  return `M ${cx} ${cy} L ${start.x.toFixed(1)} ${start.y.toFixed(1)} A ${radius} ${radius} 0 ${largeArc} ${sweepFlag} ${end.x.toFixed(1)} ${end.y.toFixed(1)} Z`;
}

function describeArcStroke(
  cx: number,
  cy: number,
  radius: number,
  startDeg: number,
  endDeg: number,
  sweepFlag: 0 | 1,
): string {
  const start = polarPoint(cx, cy, radius, startDeg);
  const end = polarPoint(cx, cy, radius, endDeg);
  const largeArc = Math.abs(endDeg - startDeg) > 180 ? 1 : 0;
  return `M ${start.x.toFixed(1)} ${start.y.toFixed(1)} A ${radius} ${radius} 0 ${largeArc} ${sweepFlag} ${end.x.toFixed(1)} ${end.y.toFixed(1)}`;
}

export function SkeletonCanvas({
  payload,
  exercise,
  isDemoMode,
  distanceM,
  trunkTiltDeg,
  hasLiveCamera = false,
  liveVideoOverlay,
  children,
}: SkeletonCanvasProps) {
  const { kiosk_state, telemetry_analysis, rep_counter, ui_directives } = payload;
  const edgeCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const latestPayloadRef = useRef(payload);
  latestPayloadRef.current = payload;

  const MEDIAPIPE_33_TO_COCO_17 = [
    0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28,
  ] as const;

  const resolveEdgeKeypoints = (cur: KioskDecisionPayload): Keypoint3DPoint[] | undefined => {
    if (cur.keypoints_3d && cur.keypoints_3d.length >= 17) return cur.keypoints_3d;
    if (cur.pose_landmarks && cur.pose_landmarks.length >= 29) {
      return MEDIAPIPE_33_TO_COCO_17.map((mpIdx) => cur.pose_landmarks![mpIdx]);
    }
    return undefined;
  };

  const hasEdge3D = Boolean(!isDemoMode && resolveEdgeKeypoints(payload));

  useEffect(() => {
    if (!hasEdge3D) return;
    let rafId = 0;
    const renderFrame = () => {
      const canvas = edgeCanvasRef.current;
      const cur = latestPayloadRef.current;
      const keypointsToDraw = resolveEdgeKeypoints(cur);
      if (canvas && keypointsToDraw && keypointsToDraw.length >= 17) {
        const w = canvas.clientWidth || 800;
        const h = canvas.clientHeight || 540;
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
          canvas.width = Math.round(w * dpr);
          canvas.height = Math.round(h * dpr);
        }
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
          drawWholeBody133OnCanvas(ctx, w, h, keypointsToDraw, {
            highlightColor: cur.ui_directives?.highlight_color ?? 'GREEN',
            mirrorX: Boolean(cur.frame_base64),
            detectedBoxes: cur.identity_guard?.detected_boxes,
            guardStatus: cur.identity_guard?.status,
          });
        }
      }
      rafId = requestAnimationFrame(renderFrame);
    };
    rafId = requestAnimationFrame(renderFrame);
    return () => cancelAnimationFrame(rafId);
  }, [hasEdge3D]);

  const currentRom = telemetry_analysis.current_rom_deg;
  const targetRom = telemetry_analysis.target_rom_deg || exercise.target_rom_deg;
  const hasTrunkLean = telemetry_analysis.compensation_detected.includes('TRUNK_LEAN');
  const hasShoulderHike = telemetry_analysis.compensation_detected.includes('SHOULDER_HIKE');
  const hasKneeValgus = telemetry_analysis.compensation_detected.includes('KNEE_VALGUS');
  const wrongLimbAlert = telemetry_analysis.compensation_detected.includes('WRONG_LIMB');
  const wrongPlaneAlert = telemetry_analysis.compensation_detected.includes('WRONG_PLANE');
  const limbPrompt = getPrescribedLimbPromptTh(exercise.target_joint);
  const expectedPlane = resolveExpectedPlaneForExercise(exercise);
  const planePrompt = getPrescribedPlanePromptTh(expectedPlane);
  const guard = payload.identity_guard;
  const isMultiPerson = kiosk_state === 'PAUSED_MULTI_PERSON' || guard?.status === 'MULTI_PERSON';
  const isImpostor = kiosk_state === 'PAUSED_IMPOSTOR' || guard?.status === 'IMPOSTOR';
  const isEmergency = kiosk_state === 'EMERGENCY_HALT';
  const isHold = kiosk_state === 'HOLD_PHASE';
  const isRepCompletedOrFinished = kiosk_state === 'REP_COMPLETED' || kiosk_state === 'SET_FINISHED';
  const isLowerBody = exercise.target_joint.includes('knee') || exercise.target_joint.includes('hip');

  const distanceIdeal = distanceM >= 1.8 && distanceM <= 2.2;
  const accentColor =
    ui_directives.highlight_color === 'RED'
      ? '#EF4444'
      : ui_directives.highlight_color === 'AMBER'
        ? '#F59E0B'
        : '#10B981';

  // Compute deterministic clinical skeleton joint positions inside 800x540 viewport
  const pelvisCenter = { x: 400, y: 318 };
  const effectiveTilt = hasTrunkLean ? Math.max(12, trunkTiltDeg) : trunkTiltDeg;
  const tiltDir = exercise.target_joint === 'left_shoulder' ? 1 : -1;
  const tiltRad = ((effectiveTilt * tiltDir) * Math.PI) / 180;
  const spineLen = 148;
  const neckCenter = {
    x: pelvisCenter.x + Math.sin(tiltRad) * spineLen,
    y: pelvisCenter.y - Math.cos(tiltRad) * spineLen,
  };
  const headCenter = {
    x: neckCenter.x + Math.sin(tiltRad) * 44,
    y: neckCenter.y - Math.cos(tiltRad) * 44,
  };

  const shoulderOffset = 58;
  const leftShoulder = {
    x: neckCenter.x + Math.cos(tiltRad) * shoulderOffset,
    y: neckCenter.y + Math.sin(tiltRad) * shoulderOffset,
  };
  const rightShoulder = {
    x: neckCenter.x - Math.cos(tiltRad) * shoulderOffset,
    y: neckCenter.y - Math.sin(tiltRad) * shoulderOffset - (hasShoulderHike ? 14 : 0),
  };

  // Arm angles (from straight-down 90deg in SVG coordinates):
  // Right arm abducts clockwise toward left-of-screen (90 -> 270 deg)
  // Left arm abducts counter-clockwise toward right-of-screen (90 -> -90 deg)
  const rightArmDeg =
    exercise.target_joint === 'right_shoulder' ? 90 + Math.min(180, currentRom) : 98;
  const leftArmDeg =
    exercise.target_joint === 'left_shoulder' ? 90 - Math.min(180, currentRom) : 82;

  const rightElbow = polarPoint(rightShoulder.x, rightShoulder.y, 76, rightArmDeg);
  const rightWrist = polarPoint(rightElbow.x, rightElbow.y, 68, rightArmDeg);
  const leftElbow = polarPoint(leftShoulder.x, leftShoulder.y, 76, leftArmDeg);
  const leftWrist = polarPoint(leftElbow.x, leftElbow.y, 68, leftArmDeg);

  // Lower-body squat kinematics when chair_squat is active
  const squatDropRatio =
    exercise.target_joint === 'right_knee' ? Math.min(1, Math.max(0, (175 - currentRom) / 90)) : 0;
  const hipDropY = squatDropRatio * 62;
  const adjustedPelvis = { x: pelvisCenter.x, y: pelvisCenter.y + hipDropY };
  const adjustedNeck = { x: neckCenter.x, y: neckCenter.y + hipDropY };
  const adjustedHead = { x: headCenter.x, y: headCenter.y + hipDropY };

  const leftHip = { x: adjustedPelvis.x + 36, y: adjustedPelvis.y };
  const rightHip = { x: adjustedPelvis.x - 36, y: adjustedPelvis.y };
  const kneeSpread = hasKneeValgus ? -12 : 10;
  const leftKnee = { x: 438 - kneeSpread + squatDropRatio * 18, y: 406 + squatDropRatio * 14 };
  const rightKnee = { x: 362 + kneeSpread - squatDropRatio * 18, y: 406 + squatDropRatio * 14 };
  const leftAnkle = { x: 442, y: 492 };
  const rightAnkle = { x: 358, y: 492 };

  // Target ROM Fan coordinates
  const activeJointPos =
    exercise.target_joint === 'left_shoulder'
      ? { x: leftShoulder.x, y: leftShoulder.y + hipDropY }
      : exercise.target_joint === 'right_knee'
        ? rightKnee
        : { x: rightShoulder.x, y: rightShoulder.y + hipDropY };

  const fanRadius = 138;
  let targetFanPath = '';
  let progressFanPath = '';
  let targetArcStrokePath = '';
  if (exercise.target_joint === 'right_shoulder') {
    const startDeg = 90;
    const endTarget = 90 + Math.min(178, targetRom);
    const endCurrent = 90 + Math.min(178, currentRom);
    targetFanPath = describeArcSector(activeJointPos.x, activeJointPos.y, fanRadius, startDeg, endTarget, 1);
    progressFanPath = describeArcSector(activeJointPos.x, activeJointPos.y, fanRadius - 4, startDeg, endCurrent, 1);
    targetArcStrokePath = describeArcStroke(activeJointPos.x, activeJointPos.y, fanRadius, startDeg, endTarget, 1);
  } else if (exercise.target_joint === 'left_shoulder') {
    const startDeg = 90;
    const endTarget = 90 - Math.min(178, targetRom);
    const endCurrent = 90 - Math.min(178, currentRom);
    targetFanPath = describeArcSector(activeJointPos.x, activeJointPos.y, fanRadius, startDeg, endTarget, 0);
    progressFanPath = describeArcSector(activeJointPos.x, activeJointPos.y, fanRadius - 4, startDeg, endCurrent, 0);
    targetArcStrokePath = describeArcStroke(activeJointPos.x, activeJointPos.y, fanRadius, startDeg, endTarget, 0);
  } else {
    const startDeg = -90;
    const endTarget = -90 - (175 - targetRom);
    const endCurrent = -90 - Math.max(5, 175 - currentRom);
    targetFanPath = describeArcSector(activeJointPos.x, activeJointPos.y, 118, startDeg, endTarget, 0);
    progressFanPath = describeArcSector(activeJointPos.x, activeJointPos.y, 114, startDeg, endCurrent, 0);
    targetArcStrokePath = describeArcStroke(activeJointPos.x, activeJointPos.y, 118, startDeg, endTarget, 0);
  }

  const holdProgressRatio =
    exercise.hold_sec > 0
      ? Math.max(0, Math.min(1, (exercise.hold_sec - rep_counter.hold_remaining_sec) / exercise.hold_sec))
      : 1;

  return (
    <div
      className={`glass-card relative flex min-h-[420px] h-[460px] sm:h-[540px] lg:h-[620px] w-full flex-col justify-between overflow-hidden rounded-3xl transition-colors duration-300 ${
        isEmergency ? 'ring-4 ring-rose-500/70' : ''
      }`}
    >
      {/* Live Camera Frame (Edge Vision / Local Webcam) or Clinical Zen Stage Backdrop */}
      {!isDemoMode && payload.frame_base64 ? (
        <img
          src={`data:image/jpeg;base64,${payload.frame_base64}`}
          alt="Live Rehabilitation Vision Feed"
          className="absolute inset-0 h-full w-full object-cover"
          style={{ transform: 'scaleX(-1)' }}
        />
      ) : hasLiveCamera && liveVideoOverlay ? (
        <div className="absolute inset-0 z-0 h-full w-full overflow-hidden bg-gradient-to-br from-sky-100 via-emerald-50 to-white">
          {liveVideoOverlay}
        </div>
      ) : (
        <div className="absolute inset-0 bg-gradient-to-br from-[#F4F9F9] via-[#E6F4F6] to-[#DCF0F7]">
          {/* Subtle Clinical Calibration Grid */}
          <svg className="h-full w-full opacity-45" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <pattern id="clinicalGrid" width="40" height="40" patternUnits="userSpaceOnUse">
                <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#BAE6FD" strokeWidth="1" />
              </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#clinicalGrid)" />
          </svg>
          {/* Floor horizon ellipse */}
          <div className="pointer-events-none absolute bottom-6 left-1/2 h-10 w-72 -translate-x-1/2 rounded-full bg-[#0284C7]/10 blur-md" />
        </div>
      )}

      {/* Top HUD Overlay: Distance & Posture Status */}
      <div className="relative z-10 flex flex-wrap items-center justify-between gap-2 p-4 sm:p-5">
        <div
          className={`inline-flex items-center gap-2 rounded-full border px-3.5 py-1.5 text-xs font-bold shadow-sm backdrop-blur-md sm:text-sm ${
            distanceIdeal
              ? 'border-emerald-200 bg-white/85 text-[#0F4C81]'
              : 'border-amber-300 bg-amber-50/90 text-amber-800'
          }`}
        >
          <Ruler className={`h-4 w-4 ${distanceIdeal ? 'text-[#10B981]' : 'text-[#F59E0B]'}`} />
          <span>
            ระยะยืน: <span className="tabular-mono">{distanceM.toFixed(1)}</span> ม. -{' '}
            {distanceIdeal ? 'เหมาะสม' : distanceM < 1.8 ? 'กรุณาถอยหลังเล็กน้อย' : 'กรุณาขยับเข้าอีกนิด'}
          </span>
        </div>

        <motion.div
          key={`${kiosk_state}-${ui_directives.highlight_color}-${wrongLimbAlert ? 'wrong' : wrongPlaneAlert ? 'wrong_plane' : 'ok'}`}
          initial={{ scale: 0.96, opacity: 0.85 }}
          animate={{ scale: 1, opacity: 1 }}
          className={`inline-flex items-center gap-2 rounded-full border px-4 py-1.5 text-xs font-bold shadow-sm backdrop-blur-md sm:text-sm ${
            ui_directives.highlight_color === 'RED' || wrongLimbAlert || wrongPlaneAlert
              ? 'border-rose-300 bg-rose-500 text-white'
              : ui_directives.highlight_color === 'AMBER'
                ? 'border-amber-300 bg-amber-100/95 text-amber-900'
                : 'border-emerald-200 bg-white/88 text-[#0D9488]'
          }`}
        >
          {ui_directives.highlight_color === 'RED' || wrongLimbAlert || wrongPlaneAlert ? (
            <ShieldAlert className="h-4 w-4" />
          ) : ui_directives.highlight_color === 'AMBER' ? (
            <AlertTriangle className="h-4 w-4 text-[#F59E0B]" />
          ) : (
            <CheckCircle2 className="h-4 w-4 text-[#10B981]" />
          )}
          <span>
            {isEmergency
              ? 'หยุดพักฉุกเฉิน (EMERGENCY HALT)'
              : isImpostor
                ? 'ผู้ใช้งานไม่ใช่คนไข้ที่ลงทะเบียน'
                : isMultiPerson
                  ? `พบบุคคล ${guard?.person_count ?? 2} คน • หยุดนับชั่วคราว`
                  : wrongLimbAlert
                    ? `⚠️ ยกแขนผิดข้าง • ${limbPrompt.bannerAlertTh}`
                    : wrongPlaneAlert
                      ? `⚠️ ผิดทิศทาง • ${planePrompt.bannerAlertTh}`
                      : hasTrunkLean
                      ? 'ตรวจพบการเอียงลำตัวชดเชย'
                      : isRepCompletedOrFinished
                        ? `ครบ ${exercise.hold_sec} วินาทีแล้ว • ${isLowerBody ? 'ค่อยๆ เอาขาลงช้าๆ' : 'ค่อยๆ เอาแขนลงช้าๆ'}`
                        : isHold
                          ? 'คงท่าตามมุมเป้าหมาย'
                          : telemetry_analysis.is_form_valid
                            ? 'ท่าทางถูกต้อง • เคลื่อนไหวต่อเนื่อง'
                            : 'กรุณาจัดตำแหน่งร่างกาย'}
          </span>
        </motion.div>
      </div>

      {/* Wrong-Arm Enforcement Alert Banner */}
      {!isEmergency && !isImpostor && !isMultiPerson && wrongLimbAlert && (
        <div
          role="alert"
          className="relative z-20 mx-4 mb-2 flex items-center gap-3 rounded-2xl border-2 border-rose-400 bg-rose-600/95 px-4 py-3 text-white shadow-lg backdrop-blur-md sm:mx-5"
        >
          <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-white/20 text-white">
            <AlertTriangle className="h-6 w-6" />
          </div>
          <div className="min-w-0 flex-1">
            <div className="text-sm font-black tracking-tight sm:text-base">
              ⚠️ ยกแขนผิดข้าง! ระบบไม่นับรอบ (ยก{limbPrompt.wrongLimbTh}อยู่)
            </div>
            <div className="mt-0.5 text-xs font-bold text-rose-100 sm:text-sm">
              กรุณายก{limbPrompt.requiredLimbTh} (ท่ากายภาพนี้ต้องทำด้วย{limbPrompt.requiredLimbTh}เท่านั้น)
            </div>
          </div>
          <span className="shrink-0 rounded-full bg-rose-900/60 px-3 py-1 text-[11px] font-black uppercase tracking-wider text-rose-100">
            0 REPS ADDED
          </span>
        </div>
      )}

      {/* Wrong-Plane Enforcement Alert Banner */}
      {!isEmergency && !isImpostor && !isMultiPerson && !wrongLimbAlert && wrongPlaneAlert && (
        <div
          role="alert"
          className="relative z-20 mx-4 mb-2 flex items-center gap-3 rounded-2xl border-2 border-rose-400 bg-rose-600/95 px-4 py-3 text-white shadow-lg backdrop-blur-md sm:mx-5"
        >
          <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-white/20 text-white">
            <AlertTriangle className="h-6 w-6" />
          </div>
          <div className="min-w-0 flex-1">
            <div className="text-sm font-black tracking-tight sm:text-base">
              ⚠️ เคลื่อนไหวผิดทิศทาง! ระบบไม่นับรอบ (กำลัง{planePrompt.wrongPlaneTh}อยู่)
            </div>
            <div className="mt-0.5 text-xs font-bold text-rose-100 sm:text-sm">
              {planePrompt.bannerAlertTh}
            </div>
          </div>
          <span className="shrink-0 rounded-full bg-rose-900/60 px-3 py-1 text-[11px] font-black uppercase tracking-wider text-rose-100">
            0 REPS ADDED
          </span>
        </div>
      )}

      {/* Continuous Anti-Cheat & Identity Guard Prominent Banner (Rule 1 + Rule 2) */}
      {(isMultiPerson || isImpostor) && (
        <div
          role="alert"
          className={`relative z-20 mx-4 mb-2 flex items-center gap-3 rounded-2xl border-2 px-4 py-3 shadow-lg backdrop-blur-md sm:mx-5 ${
            isImpostor
              ? 'border-rose-400 bg-rose-600/95 text-white'
              : 'border-amber-400 bg-amber-500/95 text-slate-950'
          }`}
        >
          <div
            className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-xl ${
              isImpostor ? 'bg-white/20 text-white' : 'bg-slate-950/15 text-slate-950'
            }`}
          >
            {isImpostor ? <UserX className="h-6 w-6" /> : <Users className="h-6 w-6" />}
          </div>
          <div className="min-w-0 flex-1">
            <div className="text-sm font-black tracking-tight sm:text-base">
              {guard?.alert_title_th ||
                (isImpostor
                  ? 'คุณไม่ใช่คนไข้ที่ลงทะเบียนไว้'
                  : 'ตรวจพบมีบุคคลอยู่หลายคนในเฟรม')}
            </div>
            <div
              className={`mt-0.5 text-xs font-bold sm:text-sm ${
                isImpostor ? 'text-rose-100' : 'text-slate-900'
              }`}
            >
              {guard?.alert_subtitle_th ||
                (isImpostor
                  ? 'ระบบจะไม่นับรอบการฝึก กรุณาให้คนไข้ตัวจริงเข้าฝึกครับ'
                  : 'กรุณาให้คนไข้ยืนหน้ากล้องเพียงลำพัง ระบบจึงจะนับรอบการฝึกต่อครับ')}
            </div>
          </div>
          <span
            className={`shrink-0 rounded-full px-3 py-1 text-[11px] font-black uppercase tracking-wider ${
              isImpostor ? 'bg-rose-900/60 text-rose-100' : 'bg-amber-900/20 text-slate-950'
            }`}
          >
            {isImpostor ? 'LOCKED • 0 REPS' : 'PAUSED'}
          </span>
        </div>
      )}

      {!isMultiPerson && !isImpostor && guard?.status === 'VERIFIED' && guard.is_counting_allowed && (
        <div className="relative z-20 mx-4 mb-1 inline-flex w-fit items-center gap-1.5 rounded-full border border-emerald-300/80 bg-emerald-500/15 px-3 py-1 text-[11px] font-extrabold text-emerald-800 backdrop-blur-sm sm:mx-5">
          <UserCheck className="h-3.5 w-3.5 text-emerald-600" />
          <span>ยืนยันคนไข้ตัวจริง ({Math.round((guard.match_confidence || 1) * 100)}%)</span>
        </div>
      )}

      {/* Glowing 3D RTMW3D-l Canvas (when Edge Server active) or SVG Anatomical Skeleton + Target ROM Fan */}
      <div className="relative z-10 flex flex-1 items-center justify-center px-2 pb-2">
        {hasEdge3D && (
          <canvas
            ref={edgeCanvasRef}
            aria-label="RTMW3D-l 133-Keypoints 3D WholeBody Skeleton"
            className="pointer-events-none absolute inset-0 z-10 h-full w-full"
          />
        )}

        {!isDemoMode && payload.calibration && !payload.calibration.is_calibrated && (
          <div className="pointer-events-none absolute bottom-4 left-1/2 z-20 -translate-x-1/2 rounded-2xl border border-sky-300 bg-white/95 px-5 py-2.5 text-center text-[#0F4C81] shadow-xl backdrop-blur-md">
            <div className="text-xs font-extrabold text-[#0284C7]">
              {payload.calibration.message_th}
            </div>
            <div className="mt-1.5 h-2 w-56 overflow-hidden rounded-full bg-sky-100">
              <div
                className="h-full rounded-full bg-gradient-to-r from-[#0284C7] to-[#10B981] transition-all duration-150"
                style={{ width: `${Math.round((payload.calibration.progress || 0) * 100)}%` }}
              />
            </div>
          </div>
        )}

        {!hasLiveCamera && (
          <svg
            viewBox="0 0 800 540"
            className={`h-full max-h-[500px] w-full select-none overflow-visible ${hasEdge3D ? 'opacity-20' : ''}`}
          >
          <defs>
            <linearGradient id="romFillGrad" x1="0%" y1="100%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#0284C7" stopOpacity="0.22" />
              <stop offset="100%" stopColor="#10B981" stopOpacity="0.45" />
            </linearGradient>
            <linearGradient id="romAmberGrad" x1="0%" y1="100%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#F59E0B" stopOpacity="0.25" />
              <stop offset="100%" stopColor="#FBBF24" stopOpacity="0.45" />
            </linearGradient>
            <filter id="glowNode" x="-40%" y="-40%" width="180%" height="180%">
              <feGaussianBlur stdDeviation="4" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Translucent Anatomical Silhouette Backdrop (Demo / Zen mode) */}
          {(isDemoMode || !payload.frame_base64) && (
            <g opacity="0.22" fill="#0F4C81">
              <circle cx={adjustedHead.x} cy={adjustedHead.y} r="28" />
              <path
                d={`M ${leftShoulder.x} ${leftShoulder.y + hipDropY} L ${rightShoulder.x} ${rightShoulder.y + hipDropY} L ${rightHip.x} ${rightHip.y} L ${leftHip.x} ${leftHip.y} Z`}
              />
            </g>
          )}

          {/* Target ROM Fan & Current Fill Arc */}
          {ui_directives.show_target_arc && (
            <g>
              <path
                d={targetFanPath}
                fill="#E0F2FE"
                fillOpacity="0.35"
                stroke="#38BDF8"
                strokeOpacity="0.45"
                strokeWidth="1.5"
                strokeDasharray="6 6"
              />
              <path
                d={progressFanPath}
                fill={ui_directives.highlight_color === 'AMBER' ? 'url(#romAmberGrad)' : 'url(#romFillGrad)'}
              />
              <path
                d={targetArcStrokePath}
                fill="none"
                stroke={accentColor}
                strokeWidth="4"
                strokeLinecap="round"
                strokeDasharray="8 6"
              />
            </g>
          )}

          {/* Ideal Vertical Reference Spine Guide when TRUNK_LEAN detected */}
          {hasTrunkLean && (
            <g>
              <line
                x1={adjustedPelvis.x}
                y1={adjustedPelvis.y}
                x2={adjustedPelvis.x}
                y2={adjustedPelvis.y - 165}
                stroke="#10B981"
                strokeWidth="2.5"
                strokeDasharray="6 6"
                strokeOpacity="0.7"
              />
              {/* Animated directional straightening arrow */}
              <motion.g
                animate={{ x: [0, tiltDir * -14, 0] }}
                transition={{ repeat: Infinity, duration: 1.2, ease: 'easeInOut' }}
              >
                <path
                  d={`M ${adjustedNeck.x} ${adjustedNeck.y + 18} L ${adjustedPelvis.x} ${adjustedNeck.y + 18}`}
                  stroke="#F59E0B"
                  strokeWidth="4"
                  strokeLinecap="round"
                  markerEnd="url(#arrowHead)"
                />
                <circle cx={adjustedPelvis.x} cy={adjustedNeck.y + 18} r="5" fill="#F59E0B" />
              </motion.g>
              <rect
                x={adjustedPelvis.x - 74}
                y={adjustedNeck.y - 38}
                width="148"
                height="30"
                rx="15"
                fill="#FEF3C7"
                stroke="#F59E0B"
                strokeWidth="1.5"
              />
              <text
                x={adjustedPelvis.x}
                y={adjustedNeck.y - 18}
                textAnchor="middle"
                fill="#92400E"
                fontSize="13"
                fontWeight="700"
              >
                ยืดลำตัวตรง ({trunkTiltDeg.toFixed(1)}°)
              </text>
            </g>
          )}

          {/* Skeleton Bones */}
          <g strokeLinecap="round" strokeLinejoin="round">
            {/* Pelvis & Lower Limbs */}
            <line x1={leftHip.x} y1={leftHip.y} x2={rightHip.x} y2={rightHip.y} stroke="#0284C7" strokeWidth="6" />
            <line x1={leftHip.x} y1={leftHip.y} x2={leftKnee.x} y2={leftKnee.y} stroke="#0284C7" strokeWidth="6" />
            <line x1={leftKnee.x} y1={leftKnee.y} x2={leftAnkle.x} y2={leftAnkle.y} stroke="#0284C7" strokeWidth="6" />
            <line
              x1={rightHip.x}
              y1={rightHip.y}
              x2={rightKnee.x}
              y2={rightKnee.y}
              stroke={exercise.target_joint === 'right_knee' ? accentColor : '#0284C7'}
              strokeWidth="6"
            />
            <line
              x1={rightKnee.x}
              y1={rightKnee.y}
              x2={rightAnkle.x}
              y2={rightAnkle.y}
              stroke={exercise.target_joint === 'right_knee' ? accentColor : '#0284C7'}
              strokeWidth="6"
            />

            {/* Spine Line */}
            <line
              x1={adjustedPelvis.x}
              y1={adjustedPelvis.y}
              x2={adjustedNeck.x}
              y2={adjustedNeck.y}
              stroke={hasTrunkLean ? '#F59E0B' : '#0D9488'}
              strokeWidth={hasTrunkLean ? 8 : 6}
              filter={hasTrunkLean ? 'url(#glowNode)' : undefined}
            />

            {/* Shoulders & Neck */}
            <line
              x1={leftShoulder.x}
              y1={leftShoulder.y + hipDropY}
              x2={rightShoulder.x}
              y2={rightShoulder.y + hipDropY}
              stroke="#0F4C81"
              strokeWidth="6"
            />
            <line
              x1={adjustedNeck.x}
              y1={adjustedNeck.y}
              x2={adjustedHead.x}
              y2={adjustedHead.y}
              stroke="#0F4C81"
              strokeWidth="5"
            />

            {/* Left Arm */}
            <line
              x1={leftShoulder.x}
              y1={leftShoulder.y + hipDropY}
              x2={leftElbow.x}
              y2={leftElbow.y + hipDropY}
              stroke={exercise.target_joint === 'left_shoulder' ? accentColor : '#0284C7'}
              strokeWidth={exercise.target_joint === 'left_shoulder' ? 7 : 5.5}
              filter={exercise.target_joint === 'left_shoulder' ? 'url(#glowNode)' : undefined}
            />
            <line
              x1={leftElbow.x}
              y1={leftElbow.y + hipDropY}
              x2={leftWrist.x}
              y2={leftWrist.y + hipDropY}
              stroke={exercise.target_joint === 'left_shoulder' ? accentColor : '#0284C7'}
              strokeWidth={exercise.target_joint === 'left_shoulder' ? 7 : 5.5}
            />

            {/* Right Arm */}
            <line
              x1={rightShoulder.x}
              y1={rightShoulder.y + hipDropY}
              x2={rightElbow.x}
              y2={rightElbow.y + hipDropY}
              stroke={exercise.target_joint === 'right_shoulder' ? accentColor : '#0284C7'}
              strokeWidth={exercise.target_joint === 'right_shoulder' ? 7 : 5.5}
              filter={exercise.target_joint === 'right_shoulder' ? 'url(#glowNode)' : undefined}
            />
            <line
              x1={rightElbow.x}
              y1={rightElbow.y + hipDropY}
              x2={rightWrist.x}
              y2={rightWrist.y + hipDropY}
              stroke={exercise.target_joint === 'right_shoulder' ? accentColor : '#0284C7'}
              strokeWidth={exercise.target_joint === 'right_shoulder' ? 7 : 5.5}
            />
          </g>

          {/* Pulsing Joint Nodes */}
          {[
            { x: adjustedHead.x, y: adjustedHead.y, r: 16, highlight: false },
            { x: leftShoulder.x, y: leftShoulder.y + hipDropY, r: 8, highlight: exercise.target_joint === 'left_shoulder' },
            { x: rightShoulder.x, y: rightShoulder.y + hipDropY, r: 8, highlight: exercise.target_joint === 'right_shoulder' },
            { x: leftElbow.x, y: leftElbow.y + hipDropY, r: 6.5, highlight: false },
            { x: rightElbow.x, y: rightElbow.y + hipDropY, r: 6.5, highlight: false },
            { x: leftWrist.x, y: leftWrist.y + hipDropY, r: 6.5, highlight: false },
            { x: rightWrist.x, y: rightWrist.y + hipDropY, r: 6.5, highlight: false },
            { x: adjustedPelvis.x, y: adjustedPelvis.y, r: 8, highlight: hasTrunkLean },
            { x: leftKnee.x, y: leftKnee.y, r: 7, highlight: false },
            { x: rightKnee.x, y: rightKnee.y, r: 8, highlight: exercise.target_joint === 'right_knee' },
          ].map((node, i) => (
            <g key={i}>
              {node.highlight && (
                <motion.circle
                  cx={node.x}
                  cy={node.y}
                  r={node.r + 10}
                  fill={accentColor}
                  fillOpacity="0.25"
                  animate={{ scale: [1, 1.35, 1], opacity: [0.45, 0.15, 0.45] }}
                  transition={{ repeat: Infinity, duration: 1.8 }}
                  style={{ originX: `${node.x}px`, originY: `${node.y}px` }}
                />
              )}
              <circle
                cx={node.x}
                cy={node.y}
                r={node.r}
                fill={node.highlight ? accentColor : '#FFFFFF'}
                stroke={node.highlight ? '#FFFFFF' : '#0F4C81'}
                strokeWidth="3"
              />
            </g>
          ))}

          {/* Live Joint Angle Floating Badge & Hold-Phase Radial Ring */}
          <g transform={`translate(${activeJointPos.x}, ${Math.max(85, activeJointPos.y - 54)})`}>
            {isHold && (
              <g>
                <circle r="48" fill="rgba(255,255,255,0.9)" stroke="#D1FAE5" strokeWidth="7" />
                <motion.circle
                  r="48"
                  fill="none"
                  stroke="#10B981"
                  strokeWidth="7"
                  strokeLinecap="round"
                  strokeDasharray={2 * Math.PI * 48}
                  strokeDashoffset={2 * Math.PI * 48 * (1 - holdProgressRatio)}
                  transform="rotate(-90)"
                />
              </g>
            )}
            <rect
              x="-44"
              y="-22"
              width="88"
              height="44"
              rx="22"
              fill={isHold ? '#0D9488' : 'rgba(255,255,255,0.92)'}
              stroke={accentColor}
              strokeWidth="2.5"
            />
            <text
              x="0"
              y="6"
              textAnchor="middle"
              fill={isHold ? '#FFFFFF' : '#0F4C81'}
              fontSize="20"
              fontWeight="800"
              className="tabular-mono"
            >
              {currentRom.toFixed(0)}°
            </text>
          </g>

          {/* Identity Guard Bounding Boxes overlay for Demo / SVG mode */}
          {!hasEdge3D &&
            (isMultiPerson || isImpostor) &&
            (guard?.detected_boxes ?? []).map((box, bIdx) => {
              const bx = (box.bbox[0] / 1280) * 800;
              const by = (box.bbox[1] / 720) * 540;
              const bw = Math.max(40, ((box.bbox[2] - box.bbox[0]) / 1280) * 800);
              const bh = Math.max(40, ((box.bbox[3] - box.bbox[1]) / 720) * 540);
              const boxHex = box.is_authorized
                ? '#10B981'
                : isImpostor
                  ? '#EF4444'
                  : '#F59E0B';
              return (
                <g key={`guard-box-${bIdx}`}>
                  <rect
                    x={bx}
                    y={by}
                    width={bw}
                    height={bh}
                    rx="12"
                    fill="none"
                    stroke={boxHex}
                    strokeWidth="4"
                    strokeDasharray={box.is_authorized ? undefined : '10 6'}
                  />
                  <rect
                    x={bx}
                    y={Math.max(4, by - 28)}
                    width={Math.min(bw + 40, Math.max(150, box.label.length * 8.5))}
                    height="24"
                    rx="6"
                    fill={boxHex}
                  />
                  <text
                    x={bx + 8}
                    y={Math.max(20, by - 12)}
                    fill="#FFFFFF"
                    fontSize="12"
                    fontWeight="800"
                  >
                    {box.label}
                  </text>
                </g>
              );
            })}
          </svg>
        )}
      </div>

      {/* Bottom Camera Status Strip */}
      <div className="relative z-10 flex items-center justify-between border-t border-white/70 bg-white/85 px-5 py-2.5 text-xs font-semibold text-[#0F4C81]/90 backdrop-blur-md">
        <span className="inline-flex items-center gap-1.5">
          <Activity className="h-4 w-4 text-[#0D9488]" />
          {hasLiveCamera
            ? 'กล้องตรวจจับท่าทางกายภาพบำบัดสด (Live AI Camera Pose Tracking)'
            : isDemoMode
              ? 'โหมดจำลองกายวิภาคเชิงโต้ตอบ (Interactive Demo Canvas)'
              : `${payload.engine_backend || 'RTMW3D-l 133-Keypoints + One-Euro Filter Active'}${
                  payload.fps ? ` • ${payload.fps.toFixed(0)} FPS` : ''
                }`}
        </span>
        <span className="tabular-mono text-[#0284C7]">
          TARGET ROM {targetRom}° • HOLD {exercise.hold_sec}s
        </span>
      </div>

      {children}
    </div>
  );
}

