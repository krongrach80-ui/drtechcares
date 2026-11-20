import React, { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { AlertTriangle, Ruler, Target } from 'lucide-react';
import { resolveActiveExerciseSide } from '@drtechcares/pose-engine';
import {
  resolveExpectedPlaneForExercise,
  type ExercisePrescriptionOption,
  type KioskDecisionPayload,
} from '../hooks/useKioskSocket';
import {
  drawMediaPipe33OnCanvas,
  drawWholeBody133OnCanvas,
  isUpperBodyVisibleInFrame,
  type Keypoint3DPoint,
} from '../hooks/useEdgeVisionSocket';

interface SkeletonCanvasProps {
  payload: KioskDecisionPayload;
  exercise: ExercisePrescriptionOption;
  isDemoMode: boolean;
  distanceM: number;
  trunkTiltDeg: number;
  hasLiveCamera?: boolean;
  isPrivacyMode?: boolean;
  showSkeletonOverlay?: boolean;
  framingHintTh?: string | null;
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
  isPrivacyMode = false,
  showSkeletonOverlay = true,
  framingHintTh: _framingHintTh = null,
  liveVideoOverlay,
  children,
}: SkeletonCanvasProps) {
  const { kiosk_state, telemetry_analysis, rep_counter, ui_directives } = payload;
  const edgeCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const edgeImgRef = useRef<HTMLImageElement | null>(null);
  const latestPayloadRef = useRef(payload);
  latestPayloadRef.current = payload;

  const MEDIAPIPE_33_TO_COCO_23 = [
    0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28,
    31, 31, 29, 32, 32, 30,
  ] as const;

  const resolveEdgeKeypoints = (cur: KioskDecisionPayload): Keypoint3DPoint[] | undefined => {
    if (cur.error_code) return undefined;
    if (cur.keypoints_3d && cur.keypoints_3d.length >= 17) return cur.keypoints_3d;
    if (cur.pose_landmarks && cur.pose_landmarks.length >= 33) {
      return MEDIAPIPE_33_TO_COCO_23.map((mpIdx) => cur.pose_landmarks![mpIdx]);
    }
    if (cur.pose_landmarks && cur.pose_landmarks.length >= 29) {
      return MEDIAPIPE_33_TO_COCO_23.slice(0, 17).map((mpIdx) => cur.pose_landmarks![mpIdx]);
    }
    return undefined;
  };

  const hasValidMp33 = Boolean(
    payload.pose_landmarks &&
      payload.pose_landmarks.length >= 33 &&
      isUpperBodyVisibleInFrame(payload.pose_landmarks),
  );
  const hasEdgePose = Boolean(
    !isDemoMode &&
      showSkeletonOverlay &&
      !payload.error_code &&
      (hasValidMp33 || resolveEdgeKeypoints(payload)),
  );
  const hasEdge3D = hasEdgePose;

  useEffect(() => {
    const canvas = edgeCanvasRef.current;
    if (isDemoMode || !showSkeletonOverlay || (hasLiveCamera && Boolean(liveVideoOverlay))) {
      if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx?.clearRect(0, 0, canvas.width, canvas.height);
      }
      return;
    }

    let rafId = 0;
    const renderFrame = () => {
      const c = edgeCanvasRef.current;
      const cur = latestPayloadRef.current;
      if (c) {
        const w = c.clientWidth || 800;
        const h = c.clientHeight || 540;
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
          c.width = Math.round(w * dpr);
          c.height = Math.round(h * dpr);
        }
        const ctx = c.getContext('2d');
        if (ctx) {
          ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
          if (cur.error_code) {
            ctx.clearRect(0, 0, w, h);
          } else {
            const img = edgeImgRef.current;
            const videoWidth = !isPrivacyMode && img && img.naturalWidth > 0 ? img.naturalWidth : undefined;
            const videoHeight = !isPrivacyMode && img && img.naturalHeight > 0 ? img.naturalHeight : undefined;
            const drawOpts = {
              highlightColor: cur.ui_directives?.highlight_color ?? 'GREEN',
              mirrorX: Boolean(cur.frame_base64),
              videoWidth,
              videoHeight,
              isPrivacyMode,
              showSkeleton: showSkeletonOverlay,
              detectedBoxes: cur.identity_guard?.detected_boxes,
              guardStatus: cur.identity_guard?.status,
            };
            if (cur.pose_landmarks && cur.pose_landmarks.length >= 33) {
              if (isUpperBodyVisibleInFrame(cur.pose_landmarks)) {
                drawMediaPipe33OnCanvas(ctx, w, h, cur.pose_landmarks, drawOpts);
              } else {
                ctx.clearRect(0, 0, w, h);
              }
            } else {
              const keypointsToDraw = resolveEdgeKeypoints(cur);
              if (keypointsToDraw && keypointsToDraw.length >= 17) {
                drawWholeBody133OnCanvas(ctx, w, h, keypointsToDraw, drawOpts);
              } else {
                ctx.clearRect(0, 0, w, h);
              }
            }
          }
        }
      }
      rafId = requestAnimationFrame(renderFrame);
    };
    rafId = requestAnimationFrame(renderFrame);
    return () => cancelAnimationFrame(rafId);
  }, [isDemoMode, isPrivacyMode, showSkeletonOverlay, hasEdge3D, hasLiveCamera, Boolean(liveVideoOverlay)]);

  const currentRom = telemetry_analysis.current_rom_deg;
  const targetRom = telemetry_analysis.target_rom_deg || exercise.target_rom_deg;
  const hasTrunkLean = telemetry_analysis.compensation_detected.includes('TRUNK_LEAN');
  const hasShoulderHike = telemetry_analysis.compensation_detected.includes('SHOULDER_HIKE');
  const hasKneeValgus = telemetry_analysis.compensation_detected.includes('KNEE_VALGUS');
  const expectedPlane = resolveExpectedPlaneForExercise(exercise);
  const activeSide =
    resolveActiveExerciseSide(exercise, exercise, exercise.target_joint, expectedPlane) ??
    (/both|ทั้งสองข้าง|ทั้ง\s*2\s*ข้าง|ประสานมือ|bobath/i.test(`${exercise.name_th} ${exercise.name_en}`)
      ? 'BOTH'
      : undefined);
  const guard = payload.identity_guard;
  const isMultiPerson = kiosk_state === 'PAUSED_MULTI_PERSON' || guard?.status === 'MULTI_PERSON';
  const isImpostor = kiosk_state === 'PAUSED_IMPOSTOR' || guard?.status === 'IMPOSTOR';
  const isEmergency = kiosk_state === 'EMERGENCY_HALT';
  const isHold = kiosk_state === 'HOLD_PHASE';

  const distanceIdeal = distanceM > 0 && distanceM >= 1.8 && distanceM <= 2.2;
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
  const isActiveRightArm =
    activeSide === 'BOTH' ||
    activeSide === 'RIGHT' ||
    (!activeSide && exercise.target_joint === 'right_shoulder');
  const isActiveLeftArm =
    activeSide === 'BOTH' ||
    activeSide === 'LEFT' ||
    (!activeSide && exercise.target_joint === 'left_shoulder');
  const rightArmDeg = isActiveRightArm ? 90 + Math.min(180, currentRom) : 98;
  const leftArmDeg = isActiveLeftArm ? 90 - Math.min(180, currentRom) : 82;

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
      className={`glass-card relative aspect-[16/10] sm:aspect-[16/9] w-full flex flex-col justify-between overflow-hidden rounded-3xl shadow-lg transition-colors duration-300 ${
        isEmergency ? 'ring-4 ring-rose-500/70' : ''
      }`}
    >
      {/* Live Camera Frame (Edge Vision / Local Webcam) or Privacy Stickman Black Stage + 1:1 33-Point Skeleton Overlay */}
      <div className="absolute inset-0 z-0 overflow-hidden">
        {!isDemoMode && payload.frame_base64 ? (
          isPrivacyMode ? (
            <div
              aria-label="Privacy Stickman Black Stage"
              className="h-full w-full bg-[#050811]"
            />
          ) : (
            <img
              ref={edgeImgRef}
              src={`data:image/jpeg;base64,${payload.frame_base64}`}
              alt="Live Rehabilitation Vision Feed"
              className="h-full w-full bg-slate-950 object-contain"
              style={{ transform: 'scaleX(-1)' }}
            />
          )
        ) : hasLiveCamera && liveVideoOverlay ? (
          <div
            className={`h-full w-full overflow-hidden ${
              isPrivacyMode ? 'bg-[#050811]' : 'bg-gradient-to-br from-sky-100 via-emerald-50 to-white'
            }`}
          >
            {liveVideoOverlay}
          </div>
        ) : !isDemoMode && isPrivacyMode ? (
          <div
            aria-label="Privacy Stickman Black Stage"
            className="h-full w-full bg-[#050811]"
          />
        ) : !isDemoMode ? (
          <div
            aria-label="Live Vision Stage"
            className="h-full w-full bg-slate-950"
          />
        ) : (
          <div className="h-full w-full bg-gradient-to-br from-[#F4F9F9] via-[#E6F4F6] to-[#DCF0F7]">
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

        {!isDemoMode && showSkeletonOverlay && !(hasLiveCamera && liveVideoOverlay) && (
          <canvas
            ref={edgeCanvasRef}
            aria-label="MediaPipe 33-Keypoints 1:1 Skeleton Overlay"
            className="pointer-events-none absolute inset-0 h-full w-full"
          />
        )}
      </div>

      {/* Top HUD Corner Badges ONLY (100% Unobstructed Video) */}
      <div className="pointer-events-none relative z-10 flex items-start justify-between gap-2 p-3">
        <div
          className={`inline-flex max-w-[45%] items-center gap-1.5 truncate rounded-full border px-3 py-1 text-xs font-bold shadow-sm backdrop-blur-md sm:text-sm ${
            distanceM <= 0
              ? 'border-sky-200 bg-white/90 text-[#0F4C81]'
              : distanceIdeal
                ? 'border-emerald-200 bg-white/88 text-[#0F4C81]'
                : 'border-amber-300 bg-amber-50/92 text-amber-900'
          }`}
        >
          <Ruler
            className={`h-3.5 w-3.5 shrink-0 ${
              distanceM <= 0 ? 'text-[#0284C7]' : distanceIdeal ? 'text-[#10B981]' : 'text-[#F59E0B]'
            }`}
          />
          <span className="truncate">
            {distanceM > 0
              ? `ระยะยืน: ${distanceM.toFixed(1)} ม. (${
                  distanceIdeal ? 'เหมาะสม' : distanceM < 1.8 ? 'ถอยหลังเล็กน้อย' : 'ขยับเข้าอีกนิด'
                })`
              : 'ระยะยืน: กำลังวัดระยะ...'}
          </span>
        </div>

        <div
          className={`inline-flex max-w-[48%] items-center gap-1.5 truncate rounded-full border px-3 py-1 text-xs font-extrabold shadow-sm backdrop-blur-md sm:text-sm ${
            activeSide === 'BOTH'
              ? 'border-teal-300 bg-white/92 text-teal-800'
              : activeSide === 'LEFT'
                ? 'border-orange-300 bg-white/92 text-orange-800'
                : 'border-emerald-300 bg-white/92 text-emerald-800'
          }`}
        >
          <Target className="h-3.5 w-3.5 shrink-0" />
          <span className="truncate">
            เป้าหมาย:{' '}
            {activeSide === 'BOTH'
              ? 'ยกทั้ง 2 ข้างพร้อมกัน'
              : activeSide === 'LEFT'
                ? 'ข้างซ้าย (Left)'
                : activeSide === 'RIGHT'
                  ? 'ข้างขวา (Right)'
                  : `ทำตามท่า ${targetRom}°`}
          </span>
        </div>
      </div>

      {/* Clinical Live Error or Demo Procedural SVG Stickman (Zero Blocking Overlay Banners Inside Video) */}
      <div className="relative z-10 flex flex-1 items-center justify-center px-2 pb-2">
        {!isDemoMode && (payload.error_code === 'CAMERA_DISCONNECTED' || payload.error_code === 'AI_MODEL_ERROR') && (
          <div
            role="alert"
            className="relative z-20 mx-auto max-w-md rounded-2xl border-2 border-rose-400 bg-rose-950/90 px-5 py-4 text-center text-white shadow-2xl backdrop-blur-md"
          >
            <div className="mx-auto mb-2 flex h-10 w-10 items-center justify-center rounded-full bg-rose-500/20 text-rose-300">
              <AlertTriangle className="h-6 w-6" />
            </div>
            <div className="text-sm font-extrabold text-rose-200">
              {payload.error_code === 'CAMERA_DISCONNECTED'
                ? 'ไม่พบกล้องเว็บแคม หรือเปิดกล้องไม่สำเร็จ'
                : 'ไม่สามารถโหลดโมเดล MediaPipe Pose AI'}
            </div>
            <div className="mt-1 text-xs font-medium text-rose-100/90">
              {payload.error_message || 'กรุณาตรวจสอบการเชื่อมต่อกล้องเว็บแคมและสิทธิ์การใช้งาน'}
            </div>
          </div>
        )}

        {isDemoMode && isPrivacyMode && !hasLiveCamera && !payload.frame_base64 && showSkeletonOverlay && (
          <svg
            viewBox="0 0 800 540"
            className="h-full max-h-[500px] w-full select-none overflow-visible"
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

      {children}
    </div>
  );
}

