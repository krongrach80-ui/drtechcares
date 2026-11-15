import { useCallback, useEffect, useRef, useState } from 'react';

export interface Keypoint3DPoint {
  index: number;
  name?: string;
  x: number;
  y: number;
  z: number;
  score: number;
}

export interface BoneLengths3D {
  upper_arm_len: number;
  forearm_len: number;
  thigh_len: number;
  shin_len: number;
  shoulder_width: number;
  torso_height: number;
  cm_per_unit: number;
  spine_vertical_unit: [number, number, number];
  baseline_shoulder_y: number;
  baseline_heel_y: number;
}

export interface CalibrationState {
  is_calibrated: boolean;
  progress: number;
  remaining_sec: number;
  message_th: string;
  estimated_distance_m?: number;
  bone_lengths: BoneLengths3D | null;
}

export interface HandRaiseTelemetry {
  left_raised: boolean;
  right_raised: boolean;
  active_card_zone: 'LEFT' | 'RIGHT' | 'BOTH' | 'NONE';
  left_palm_2d: [number, number] | null;
  right_palm_2d: [number, number] | null;
  left_hand_landmarks: Keypoint3DPoint[];
  right_hand_landmarks: Keypoint3DPoint[];
}

export interface FaceEnrollmentState {
  active: boolean;
  session_id: string | null;
  phase: 'idle' | 'scanning' | 'computing' | 'completed';
  frames_captured: number;
  target_frames: number;
  bucket_counts: Record<string, number>;
  bucket_targets: Record<string, number>;
  current_pose: string;
  yaw_deg: number;
  pitch_deg: number;
  roll_deg: number;
  sharpness: number;
  face_detected: boolean;
  guidance_th: string;
}

export interface EdgePatientMatch {
  id: string | null;
  hn?: string | null;
  name: string | null;
  similarity: number;
  latency_ms: number;
}

export interface EdgeTelemetryAnalysis {
  current_rom_deg: number;
  target_rom_deg: number;
  rom_completion_pct: number;
  trunk_tilt_deg?: number;
  shoulder_hike_cm?: number;
  knee_valgus_deg?: number;
  heel_lift_cm?: number;
  angular_velocity_deg_s?: number;
  is_form_valid: boolean;
  compensation_detected: string[];
  pain_alert: boolean;
}

export interface IdentityDetectedBox {
  bbox: [number, number, number, number];
  label: string;
  is_authorized: boolean;
}

export interface IdentityGuardState {
  status: 'VERIFIED' | 'MULTI_PERSON' | 'IMPOSTOR' | 'FACE_NOT_VISIBLE';
  person_count: number;
  face_count: number;
  match_confidence: number;
  is_counting_allowed: boolean;
  alert_title_th: string | null;
  alert_subtitle_th: string | null;
  detected_boxes: IdentityDetectedBox[];
}

export interface EdgeKioskTelemetryPayload {
  kiosk_state:
    | 'AUTH_FRAMING'
    | 'CALIBRATING'
    | 'ACTIVE_REP'
    | 'HOLD_PHASE'
    | 'REP_COMPLETED'
    | 'SET_FINISHED'
    | 'EMERGENCY_HALT'
    | 'PAUSED_MULTI_PERSON'
    | 'PAUSED_IMPOSTOR';
  fsm_state?:
    | 'FACE_AUTH'
    | 'CALIBRATE_2S'
    | 'ACTIVE'
    | 'HOLD'
    | 'REP_DONE'
    | 'SET_FINISHED'
    | 'EMERGENCY'
    | 'PAUSED_MULTI_PERSON'
    | 'PAUSED_IMPOSTOR';
  timestamp_ms?: number;
  fps?: number;
  engine_backend?: string;
  camera_mode?: string;
  patient: EdgePatientMatch | null;
  telemetry_analysis: EdgeTelemetryAnalysis;
  rep_counter: {
    valid_reps: number;
    invalid_reps: number;
    hold_remaining_sec: number;
    target_reps?: number;
  };
  ui_directives: {
    skeleton_highlight_joint: string;
    highlight_color: 'GREEN' | 'AMBER' | 'RED';
    show_target_arc: boolean;
  };
  tts_voice_th: string;
  error_code?: string | null;
  error_message?: string | null;
  calibration?: CalibrationState;
  hand_status?: HandRaiseTelemetry;
  face_enrollment?: FaceEnrollmentState | null;
  identity_guard?: IdentityGuardState;
  keypoints_3d?: Keypoint3DPoint[];
  pose_landmarks?: Keypoint3DPoint[];
  pose_world_landmarks?: Keypoint3DPoint[];
  frame_base64: string | null;
}

export interface EdgeHealthInfo {
  status: string;
  service: string;
  engine: string;
  face_engine: string;
  camera_mode: string;
  fps: number;
  enrolled_candidates: number;
}

export interface EdgeFaceID120Response {
  session_id: string;
  master_embedding: number[];
  angle_embeddings: Record<string, number[]>;
  frames_captured: number;
  quality_score: number;
  avatar_data_url: string;
  bucket_counts: Record<string, number>;
  supabase_synced: boolean;
}

export interface EdgeVerifyResponse {
  matched: boolean;
  patient: EdgePatientMatch | null;
  similarity: number;
  latency_ms: number;
  candidate_count: number;
}

function readStoredEndpoint(storageKey: string): string | null {
  if (typeof window === 'undefined' || !window.localStorage) return null;
  try {
    const val = window.localStorage.getItem(storageKey)?.trim();
    return val ? val : null;
  } catch {
    return null;
  }
}

export function getEdgeVisionHttpBase(): string {
  const envHttp = (import.meta.env.VITE_EDGE_VISION_HTTP as string | undefined)?.trim();
  if (envHttp) return envHttp.replace(/\/$/, '');
  const storedHttp = readStoredEndpoint('drtechcares_edge_http_url');
  if (storedHttp) return storedHttp.replace(/\/$/, '');
  return 'http://localhost:8000';
}

export function getEdgeVisionWsUrl(): string {
  const envWs = (import.meta.env.VITE_EDGE_VISION_WS as string | undefined)?.trim();
  if (envWs) return envWs;
  const storedWs = readStoredEndpoint('drtechcares_edge_ws_url');
  if (storedWs) return storedWs;
  return 'ws://localhost:8000/ws/kiosk';
}

export async function probeEdgeVisionHealth(timeoutMs = 650): Promise<EdgeHealthInfo | null> {
  if (typeof window === 'undefined' || typeof fetch === 'undefined') return null;
  // Skip localhost probe if user explicitly disables it or running non-local without tunnel
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${getEdgeVisionHttpBase()}/health`, {
      method: 'GET',
      signal: controller.signal,
    });
    if (!res.ok) return null;
    const data = (await res.json()) as EdgeHealthInfo;
    return data?.status === 'ok' ? data : null;
  } catch {
    return null;
  } finally {
    window.clearTimeout(timer);
  }
}

export async function edgeFaceEnrollStart(params: {
  patient_id?: string;
  hn?: string;
  full_name?: string;
}): Promise<FaceEnrollmentState> {
  const res = await fetch(`${getEdgeVisionHttpBase()}/api/enroll/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error(`Edge enroll/start HTTP ${res.status}`);
  return (await res.json()) as FaceEnrollmentState;
}

export async function edgeFaceEnrollFrame(params: {
  session_id: string;
  image_base64?: string;
}): Promise<FaceEnrollmentState> {
  const res = await fetch(`${getEdgeVisionHttpBase()}/api/enroll/frame`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error(`Edge enroll/frame HTTP ${res.status}`);
  return (await res.json()) as FaceEnrollmentState;
}

export async function edgeFaceEnrollFinalize(params: {
  session_id: string;
  patient_id?: string;
  hn?: string;
  persist_to_supabase?: boolean;
}): Promise<EdgeFaceID120Response> {
  const res = await fetch(`${getEdgeVisionHttpBase()}/api/enroll/finalize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error(`Edge enroll/finalize HTTP ${res.status}`);
  return (await res.json()) as EdgeFaceID120Response;
}

export async function edgeBiometricsVerify(image_base64?: string): Promise<EdgeVerifyResponse> {
  const res = await fetch(`${getEdgeVisionHttpBase()}/api/biometrics/verify`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_base64, use_edge_camera: !image_base64 }),
  });
  if (!res.ok) throw new Error(`Edge verify HTTP ${res.status}`);
  return (await res.json()) as EdgeVerifyResponse;
}

const COCO_BODY_CONNECTIONS: ReadonlyArray<[number, number]> = [
  [5, 6],
  [5, 7],
  [7, 9],
  [6, 8],
  [8, 10],
  [5, 11],
  [6, 12],
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
  [15, 17],
  [15, 19],
  [17, 19],
  [16, 20],
  [16, 22],
  [20, 22],
];

const HAND_BONE_PAIRS: ReadonlyArray<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
];

/**
 * Draw 133-point RTMW3D WholeBody 3D landmarks onto an HTML5 Canvas at 60 FPS
 * as a clean Stickman silhouette (head circle, neck guide, Z-depth joints, feet, and 21-point hands),
 * intentionally omitting the 5 raw facial dots (indices 0..4) for a clean clinical look.
 */
export interface PoseCanvasDrawOptions {
  highlightColor?: 'GREEN' | 'AMBER' | 'RED';
  mirrorX?: boolean;
  videoWidth?: number;
  videoHeight?: number;
  minVisibility?: number;
  detectedBoxes?: IdentityDetectedBox[];
  guardStatus?: IdentityGuardState['status'];
}

export function computeContainRect(
  containerWidth: number,
  containerHeight: number,
  mediaWidth?: number,
  mediaHeight?: number,
): { offsetX: number; offsetY: number; renderW: number; renderH: number } {
  if (
    !mediaWidth ||
    !mediaHeight ||
    mediaWidth <= 0 ||
    mediaHeight <= 0 ||
    containerWidth <= 0 ||
    containerHeight <= 0
  ) {
    return { offsetX: 0, offsetY: 0, renderW: containerWidth, renderH: containerHeight };
  }
  const containerAspect = containerWidth / containerHeight;
  const mediaAspect = mediaWidth / mediaHeight;
  if (mediaAspect > containerAspect) {
    const renderW = containerWidth;
    const renderH = containerWidth / mediaAspect;
    return {
      offsetX: 0,
      offsetY: (containerHeight - renderH) / 2,
      renderW,
      renderH,
    };
  }
  const renderH = containerHeight;
  const renderW = containerHeight * mediaAspect;
  return {
    offsetX: (containerWidth - renderW) / 2,
    offsetY: 0,
    renderW,
    renderH,
  };
}

function resolveHighlightPalette(highlightColor?: 'GREEN' | 'AMBER' | 'RED'): {
  bone: string;
  joint: string;
  glow: string;
} {
  if (highlightColor === 'RED') {
    return { bone: '#EF4444', joint: '#FCA5A5', glow: 'rgba(239, 68, 68, 0.45)' };
  }
  if (highlightColor === 'AMBER') {
    return { bone: '#F59E0B', joint: '#FDE68A', glow: 'rgba(245, 158, 11, 0.45)' };
  }
  return { bone: '#10B981', joint: '#6EE7B7', glow: 'rgba(16, 185, 129, 0.45)' };
}

const MEDIAPIPE33_BODY_EDGES: ReadonlyArray<readonly [number, number, number]> = [
  // Torso & shoulders (width weight 4.2)
  [11, 12, 4.2],
  [11, 23, 4.2],
  [12, 24, 4.2],
  [23, 24, 4.2],
  // Upper & lower arms (bent elbows preserved at 13, 14)
  [11, 13, 4.5],
  [13, 15, 4.2],
  [12, 14, 4.5],
  [14, 16, 4.2],
  // Hands & wrist rotation wireframe (15..22)
  [15, 17, 2.2],
  [15, 19, 2.2],
  [15, 21, 2.2],
  [17, 19, 2.0],
  [16, 18, 2.2],
  [16, 20, 2.2],
  [16, 22, 2.2],
  [18, 20, 2.0],
  // Legs & feet (23..32)
  [23, 25, 4.2],
  [25, 27, 4.0],
  [24, 26, 4.2],
  [26, 28, 4.0],
  [27, 29, 2.6],
  [29, 31, 2.6],
  [27, 31, 2.4],
  [28, 30, 2.6],
  [30, 32, 2.6],
  [28, 32, 2.4],
];

function drawIdentityBoxesOnCanvas(
  ctx: CanvasRenderingContext2D,
  rect: { offsetX: number; offsetY: number; renderW: number; renderH: number },
  mirror: boolean,
  opts?: PoseCanvasDrawOptions,
): void {
  const boxes = opts?.detectedBoxes ?? [];
  if (
    boxes.length === 0 ||
    !(
      opts?.guardStatus === 'MULTI_PERSON' ||
      opts?.guardStatus === 'IMPOSTOR' ||
      boxes.length > 1 ||
      !boxes[0]?.is_authorized
    )
  ) {
    return;
  }

  const refW = opts?.videoWidth && opts.videoWidth > 0 ? opts.videoWidth : 1280;
  const refH = opts?.videoHeight && opts.videoHeight > 0 ? opts.videoHeight : 720;
  const sx = rect.renderW / refW;
  const sy = rect.renderH / refH;

  ctx.save();
  for (const box of boxes) {
    const [rawX1, rawY1, rawX2, rawY2] = box.bbox;
    const x1 = rect.offsetX + (mirror ? rect.renderW - rawX2 * sx : rawX1 * sx);
    const x2 = rect.offsetX + (mirror ? rect.renderW - rawX1 * sx : rawX2 * sx);
    const y1 = rect.offsetY + rawY1 * sy;
    const y2 = rect.offsetY + rawY2 * sy;
    const bw = Math.max(24, x2 - x1);
    const bh = Math.max(24, y2 - y1);
    const boxHex = box.is_authorized
      ? '#10B981'
      : opts?.guardStatus === 'MULTI_PERSON'
        ? '#F59E0B'
        : '#EF4444';

    ctx.strokeStyle = boxHex;
    ctx.lineWidth = Math.max(2.5, rect.renderW * 0.004);
    ctx.setLineDash(box.is_authorized ? [] : [8, 4]);
    ctx.strokeRect(x1, y1, bw, bh);
    ctx.setLineDash([]);

    if (box.label) {
      ctx.font = `bold ${Math.max(12, Math.round(rect.renderW * 0.018))}px "Sarabun", sans-serif`;
      const textWidth = ctx.measureText(box.label).width;
      const padX = 8;
      const labelH = Math.max(22, Math.round(rect.renderW * 0.028));
      const ly = Math.max(4, y1 - labelH - 4);
      ctx.fillStyle = boxHex;
      ctx.fillRect(x1, ly, textWidth + padX * 2, labelH);
      ctx.fillStyle = '#FFFFFF';
      ctx.fillText(box.label, x1 + padX, ly + labelH - 6);
    }
  }
  ctx.restore();
}

/**
 * Draw 33-point MediaPipe Pose Landmarks with 1:1 object-contain alignment
 * directly on top of the live webcam stream (clean Stickman head circle, hiding 5 face dots).
 */
export function drawMediaPipe33OnCanvas(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  landmarks: Keypoint3DPoint[],
  opts?: PoseCanvasDrawOptions,
): void {
  ctx.clearRect(0, 0, width, height);
  if (!landmarks || landmarks.length < 33) return;

  const mirror = opts?.mirrorX ?? false;
  const minVis = opts?.minVisibility ?? 0.15;
  const rect = computeContainRect(width, height, opts?.videoWidth, opts?.videoHeight);
  const palette = resolveHighlightPalette(opts?.highlightColor);

  const toXY = (kp: Keypoint3DPoint) => ({
    px: rect.offsetX + (mirror ? 1 - kp.x : kp.x) * rect.renderW,
    py: rect.offsetY + kp.y * rect.renderH,
    z: kp.z ?? 0,
    score: kp.score ?? 1,
  });

  const pts = landmarks.map(toXY);

  // Optional identity guard bounding boxes
  drawIdentityBoxesOnCanvas(ctx, rect, mirror, opts);

  // Draw crisp bone lines (dark halo under glowing clinical accent for visibility on bright video)
  ctx.save();
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  for (const [i, j, lineW] of MEDIAPIPE33_BODY_EDGES) {
    const a = pts[i];
    const b = pts[j];
    if (!a || !b || a.score < minVis || b.score < minVis) continue;

    ctx.strokeStyle = 'rgba(15, 23, 42, 0.55)';
    ctx.lineWidth = lineW + 3;
    ctx.shadowBlur = 0;
    ctx.beginPath();
    ctx.moveTo(a.px, a.py);
    ctx.lineTo(b.px, b.py);
    ctx.stroke();

    ctx.strokeStyle = palette.bone;
    ctx.lineWidth = lineW;
    ctx.shadowColor = palette.glow;
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.moveTo(a.px, a.py);
    ctx.lineTo(b.px, b.py);
    ctx.stroke();
  }

  // Clean Stickman neck & head circle (omitting 5 face dots 0..10 for clinical clarity)
  const shL = pts[11];
  const shR = pts[12];
  const nose = pts[0];
  const earL = pts[7];
  const earR = pts[8];
  if (shL && shR && shL.score >= minVis && shR.score >= minVis) {
    const neckX = (shL.px + shR.px) / 2;
    const neckY = (shL.py + shR.py) / 2;
    const shoulderSpan = Math.hypot(shL.px - shR.px, shL.py - shR.py);
    let headX = neckX;
    let headY = neckY - Math.max(shoulderSpan * 0.48, 28);
    if (nose && nose.score >= minVis) {
      headX = nose.px;
      headY = nose.py;
    } else if (earL && earR && earL.score >= minVis && earR.score >= minVis) {
      headX = (earL.px + earR.px) / 2;
      headY = (earL.py + earR.py) / 2;
    }
    const earSpan =
      earL && earR && earL.score >= minVis && earR.score >= minVis
        ? Math.hypot(earL.px - earR.px, earL.py - earR.py)
        : shoulderSpan * 0.42;
    const headRadius = Math.max(14, Math.min(44, earSpan * 0.68));

    const dx = headX - neckX;
    const dy = headY - neckY;
    const dist = Math.hypot(dx, dy);
    if (dist > headRadius) {
      const stopX = headX - (dx / dist) * headRadius;
      const stopY = headY - (dy / dist) * headRadius;
      ctx.strokeStyle = 'rgba(15, 23, 42, 0.55)';
      ctx.lineWidth = 6.5;
      ctx.shadowBlur = 0;
      ctx.beginPath();
      ctx.moveTo(neckX, neckY);
      ctx.lineTo(stopX, stopY);
      ctx.stroke();

      ctx.strokeStyle = palette.bone;
      ctx.lineWidth = 3.5;
      ctx.shadowColor = palette.glow;
      ctx.shadowBlur = 8;
      ctx.beginPath();
      ctx.moveTo(neckX, neckY);
      ctx.lineTo(stopX, stopY);
      ctx.stroke();
    }

    ctx.strokeStyle = 'rgba(15, 23, 42, 0.55)';
    ctx.lineWidth = 6.2;
    ctx.shadowBlur = 0;
    ctx.beginPath();
    ctx.arc(headX, headY, headRadius, 0, Math.PI * 2);
    ctx.stroke();

    ctx.strokeStyle = palette.bone;
    ctx.lineWidth = 3.2;
    ctx.shadowColor = palette.glow;
    ctx.shadowBlur = 10;
    ctx.beginPath();
    ctx.arc(headX, headY, headRadius, 0, Math.PI * 2);
    ctx.stroke();
  }

  // Draw anatomical body, hand & foot joints (11..32) — face landmarks (0..10) intentionally excluded
  for (let idx = 11; idx < 33; idx += 1) {
    const p = pts[idx];
    if (!p || p.score < minVis) continue;
    const isMajor = idx >= 11 && idx <= 16 ? true : idx >= 23 && idx <= 28;
    const radius = isMajor ? 5.2 : 3.2;
    ctx.shadowBlur = 0;
    ctx.fillStyle = 'rgba(15, 23, 42, 0.75)';
    ctx.beginPath();
    ctx.arc(p.px, p.py, radius + 1.6, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = palette.joint;
    ctx.beginPath();
    ctx.arc(p.px, p.py, radius, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

export function drawWholeBody133OnCanvas(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  keypoints: Keypoint3DPoint[],
  opts?: PoseCanvasDrawOptions,
): void {
  ctx.clearRect(0, 0, width, height);
  if (!keypoints || keypoints.length < 17) return;

  const mirror = opts?.mirrorX ?? false;
  const rect = computeContainRect(width, height, opts?.videoWidth, opts?.videoHeight);
  const color =
    opts?.highlightColor === 'RED'
      ? '#EF4444'
      : opts?.highlightColor === 'AMBER'
        ? '#F59E0B'
        : '#10B981';

  const toXY = (kp: Keypoint3DPoint) => ({
    px: rect.offsetX + (mirror ? 1 - kp.x : kp.x) * rect.renderW,
    py: rect.offsetY + kp.y * rect.renderH,
    z: kp.z ?? 0,
    score: kp.score ?? 1,
  });

  // Body & foot bones (indices 5..22 — no raw facial bone clutter)
  ctx.save();
  ctx.lineCap = 'round';
  ctx.lineWidth = Math.max(2.5, width * 0.0055);
  ctx.strokeStyle = color;
  for (const [aIdx, bIdx] of COCO_BODY_CONNECTIONS) {
    const a = keypoints[aIdx];
    const b = keypoints[bIdx];
    if (!a || !b || a.score < 0.2 || b.score < 0.2) continue;
    const pa = toXY(a);
    const pb = toXY(b);
    ctx.beginPath();
    ctx.moveTo(pa.px, pa.py);
    ctx.lineTo(pb.px, pb.py);
    ctx.stroke();
  }

  // Stickman head circle + neck stem (hides the 5 facial keypoints 0..4 while keeping head posture clear)
  const nose = keypoints[0];
  const lEar = keypoints[3];
  const rEar = keypoints[4];
  const lSh = keypoints[5];
  const rSh = keypoints[6];
  const lShValid = Boolean(lSh && lSh.score >= 0.2);
  const rShValid = Boolean(rSh && rSh.score >= 0.2);

  let headCenterPx: { px: number; py: number } | null = null;
  if (nose && nose.score >= 0.2) {
    headCenterPx = toXY(nose);
  } else if (lEar && rEar && lEar.score >= 0.2 && rEar.score >= 0.2) {
    const peL = toXY(lEar);
    const peR = toXY(rEar);
    headCenterPx = { px: (peL.px + peR.px) / 2, py: (peL.py + peR.py) / 2 };
  }

  let shoulderSpanPx = width * 0.16;
  let neckPt: { px: number; py: number } | null = null;
  if (lShValid && rShValid && lSh && rSh) {
    const pL = toXY(lSh);
    const pR = toXY(rSh);
    shoulderSpanPx = Math.max(40, Math.hypot(pL.px - pR.px, pL.py - pR.py));
    neckPt = { px: (pL.px + pR.px) / 2, py: (pL.py + pR.py) / 2 };
  }

  if (headCenterPx) {
    const maxR = Math.max(26, width * 0.065);
    const headRadius = Math.max(16, Math.min(maxR, shoulderSpanPx * 0.28));

    if (neckPt) {
      const dx = headCenterPx.px - neckPt.px;
      const dy = headCenterPx.py - neckPt.py;
      const dist = Math.hypot(dx, dy);
      if (dist > headRadius + 2) {
        const edgeX = headCenterPx.px - (dx / dist) * headRadius;
        const edgeY = headCenterPx.py - (dy / dist) * headRadius;
        ctx.beginPath();
        ctx.moveTo(neckPt.px, neckPt.py);
        ctx.lineTo(edgeX, edgeY);
        ctx.lineWidth = Math.max(2.5, width * 0.005);
        ctx.strokeStyle = color;
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.arc(neckPt.px, neckPt.py, 4.5, 0, Math.PI * 2);
      ctx.fillStyle = '#FFFFFF';
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = color;
      ctx.stroke();
    }

    ctx.beginPath();
    ctx.arc(headCenterPx.px, headCenterPx.py, headRadius, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(15, 23, 42, 0.6)';
    ctx.fill();
    ctx.lineWidth = Math.max(2.5, width * 0.0055);
    ctx.strokeStyle = color;
    ctx.stroke();
  }

  // Body & foot keypoint nodes (start at index 5 — omit 5 raw facial dots 0..4)
  for (let i = 5; i < Math.min(23, keypoints.length); i++) {
    const kp = keypoints[i];
    if (!kp || kp.score < 0.2) continue;
    const { px, py, z } = toXY(kp);
    const depthScale = Math.max(0.75, Math.min(1.45, 1 - z * 0.8));
    const r = 5.5 * depthScale;
    ctx.beginPath();
    ctx.arc(px, py, r, 0, Math.PI * 2);
    ctx.fillStyle = i >= 17 ? '#38BDF8' : '#FFFFFF';
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = i >= 17 ? '#0284C7' : color;
    ctx.stroke();
  }

  // Draw Left Hand (91..111) & Right Hand (112..132)
  const drawHand = (offset: number, strokeHex: string) => {
    if (keypoints.length < offset + 21) return;
    ctx.strokeStyle = strokeHex;
    ctx.lineWidth = 2;
    for (const [u, v] of HAND_BONE_PAIRS) {
      const a = keypoints[offset + u];
      const b = keypoints[offset + v];
      if (!a || !b || a.score < 0.2 || b.score < 0.2) continue;
      const pa = toXY(a);
      const pb = toXY(b);
      ctx.beginPath();
      ctx.moveTo(pa.px, pa.py);
      ctx.lineTo(pb.px, pb.py);
      ctx.stroke();
    }
    for (let i = 0; i < 21; i++) {
      const kp = keypoints[offset + i];
      if (!kp || kp.score < 0.2) continue;
      const { px, py } = toXY(kp);
      ctx.beginPath();
      ctx.arc(px, py, i === 0 ? 4 : 2.5, 0, Math.PI * 2);
      ctx.fillStyle = strokeHex;
      ctx.fill();
    }
  };

  drawHand(91, '#818CF8');
  drawHand(112, '#34D399');

  // Identity Guard person/face bounding boxes
  drawIdentityBoxesOnCanvas(ctx, rect, mirror, opts);
  ctx.restore();
}

export function useEdgeVisionSocket(options?: {
  enabled?: boolean;
  wsUrl?: string;
  streamVideo?: boolean;
  onPatientMatch?: (patient: EdgePatientMatch) => void;
}) {
  const enabled = options?.enabled ?? true;
  const wsUrl = options?.wsUrl ?? getEdgeVisionWsUrl();
  const streamVideo = options?.streamVideo ?? false;

  const [connected, setConnected] = useState(false);
  const [telemetry, setTelemetry] = useState<EdgeKioskTelemetryPayload | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const onMatchRef = useRef(options?.onPatientMatch);
  onMatchRef.current = options?.onPatientMatch;
  const lastMatchedIdRef = useRef<string | null>(null);

  const sendCommand = useCallback((cmd: Record<string, unknown>) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(cmd));
      return true;
    }
    return false;
  }, []);

  useEffect(() => {
    if (!enabled || typeof window === 'undefined') return;
    let disposed = false;
    let reconnectTimer: number | null = null;
    let attempt = 0;

    const connect = () => {
      if (disposed) return;
      try {
        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;

        ws.onopen = () => {
          if (disposed) return;
          attempt = 0;
          setConnected(true);
          if (streamVideo) {
            ws.send(JSON.stringify({ type: 'config', stream_video: true }));
          }
        };

        ws.onmessage = (evt) => {
          if (disposed) return;
          try {
            const payload = JSON.parse(evt.data) as EdgeKioskTelemetryPayload;
            if (payload && payload.kiosk_state && payload.telemetry_analysis) {
              setTelemetry(payload);
              if (
                payload.patient?.id &&
                payload.patient.id !== lastMatchedIdRef.current
              ) {
                lastMatchedIdRef.current = payload.patient.id;
                onMatchRef.current?.(payload.patient);
              }
            }
          } catch {
            // ignore non-telemetry ack frames
          }
        };

        ws.onerror = () => {
          ws.close();
        };

        ws.onclose = () => {
          if (disposed) return;
          wsRef.current = null;
          setConnected(false);
          const delayMs = Math.min(12000, 1500 * Math.pow(1.6, attempt));
          attempt += 1;
          reconnectTimer = window.setTimeout(connect, delayMs);
        };
      } catch {
        setConnected(false);
      }
    };

    connect();
    return () => {
      disposed = true;
      if (reconnectTimer !== null) window.clearTimeout(reconnectTimer);
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [enabled, streamVideo, wsUrl]);

  return {
    connected,
    telemetry,
    sendCommand,
  };
}

