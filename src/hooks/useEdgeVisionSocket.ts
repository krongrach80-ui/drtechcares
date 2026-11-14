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
  calibration?: CalibrationState;
  hand_status?: HandRaiseTelemetry;
  face_enrollment?: FaceEnrollmentState | null;
  identity_guard?: IdentityGuardState;
  keypoints_3d?: Keypoint3DPoint[];
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

export function getEdgeVisionHttpBase(): string {
  const envHttp = (import.meta.env.VITE_EDGE_VISION_HTTP as string | undefined)?.trim();
  if (envHttp) return envHttp.replace(/\/$/, '');
  return 'http://localhost:8000';
}

export function getEdgeVisionWsUrl(): string {
  const envWs = (import.meta.env.VITE_EDGE_VISION_WS as string | undefined)?.trim();
  if (envWs) return envWs;
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
  [16, 20],
  [16, 22],
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
 * with Z-depth node scaling, target-joint clinical glow, and 21-point hand overlays.
 */
export function drawWholeBody133OnCanvas(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  keypoints: Keypoint3DPoint[],
  opts?: {
    highlightColor?: 'GREEN' | 'AMBER' | 'RED';
    mirrorX?: boolean;
    detectedBoxes?: IdentityDetectedBox[];
    guardStatus?: IdentityGuardState['status'];
  },
): void {
  if (!keypoints || keypoints.length < 17) return;
  ctx.clearRect(0, 0, width, height);

  const mirror = opts?.mirrorX ?? false;
  const color =
    opts?.highlightColor === 'RED'
      ? '#EF4444'
      : opts?.highlightColor === 'AMBER'
        ? '#F59E0B'
        : '#10B981';

  const toXY = (kp: Keypoint3DPoint) => ({
    px: (mirror ? 1 - kp.x : kp.x) * width,
    py: kp.y * height,
    z: kp.z ?? 0,
    score: kp.score ?? 1,
  });

  // Body & foot bones (0..22)
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

  // Body & foot keypoint nodes
  for (let i = 0; i < Math.min(23, keypoints.length); i++) {
    const kp = keypoints[i];
    if (!kp || kp.score < 0.2) continue;
    const { px, py, z } = toXY(kp);
    const depthScale = Math.max(0.75, Math.min(1.45, 1 - z * 0.8));
    const r = (i <= 4 ? 3.5 : 5.5) * depthScale;
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

  // Identity Guard person/face bounding boxes (scaled from 1280x720)
  const boxes = opts?.detectedBoxes ?? [];
  if (boxes.length > 0 && (opts?.guardStatus === 'MULTI_PERSON' || opts?.guardStatus === 'IMPOSTOR' || boxes.length > 1 || !boxes[0]?.is_authorized)) {
    const sx = width / 1280;
    const sy = height / 720;
    for (const box of boxes) {
      const [rawX1, rawY1, rawX2, rawY2] = box.bbox;
      const x1 = mirror ? width - rawX2 * sx : rawX1 * sx;
      const x2 = mirror ? width - rawX1 * sx : rawX2 * sx;
      const y1 = rawY1 * sy;
      const y2 = rawY2 * sy;
      const bw = Math.max(24, x2 - x1);
      const bh = Math.max(24, y2 - y1);
      const boxHex = box.is_authorized
        ? '#10B981'
        : opts?.guardStatus === 'MULTI_PERSON'
          ? '#F59E0B'
          : '#EF4444';

      ctx.strokeStyle = boxHex;
      ctx.lineWidth = Math.max(2.5, width * 0.004);
      ctx.setLineDash(box.is_authorized ? [] : [8, 4]);
      ctx.strokeRect(x1, y1, bw, bh);
      ctx.setLineDash([]);

      if (box.label) {
        ctx.font = `bold ${Math.max(12, Math.round(width * 0.018))}px "Sarabun", sans-serif`;
        const textWidth = ctx.measureText(box.label).width;
        const padX = 8;
        const labelH = Math.max(22, Math.round(width * 0.028));
        const ly = Math.max(4, y1 - labelH - 4);
        ctx.fillStyle = boxHex;
        ctx.fillRect(x1, ly, textWidth + padX * 2, labelH);
        ctx.fillStyle = '#FFFFFF';
        ctx.fillText(box.label, x1 + padX, ly + labelH - 6);
      }
    }
  }
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

