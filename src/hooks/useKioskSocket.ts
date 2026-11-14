import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { CLINICAL_EXERCISES } from '@drtechcares/pose-engine';
import type { TodayPrescribedExerciseItem } from '@drtechcares/api-client';

export type KioskState =
  | 'AUTH_FRAMING'
  | 'CALIBRATING'
  | 'ACTIVE_REP'
  | 'HOLD_PHASE'
  | 'REP_COMPLETED'
  | 'SET_FINISHED'
  | 'EMERGENCY_HALT';

export type CompensationCode =
  | 'NONE'
  | 'TRUNK_LEAN'
  | 'SHOULDER_HIKE'
  | 'KNEE_VALGUS'
  | 'TOO_FAST'
  | 'INCOMPLETE_ROM';

export type HighlightJoint =
  | 'none'
  | 'left_shoulder'
  | 'right_shoulder'
  | 'spine'
  | 'left_knee'
  | 'right_knee';

export type HighlightColor = 'GREEN' | 'AMBER' | 'RED';

export interface PatientSummary {
  id: string | null;
  name: string | null;
}

export interface TelemetryAnalysis {
  current_rom_deg: number;
  target_rom_deg: number;
  rom_completion_pct: number;
  is_form_valid: boolean;
  compensation_detected: CompensationCode[];
  pain_alert: boolean;
}

export interface RepCounter {
  valid_reps: number;
  invalid_reps: number;
  hold_remaining_sec: number;
}

export interface UIDirectives {
  skeleton_highlight_joint: HighlightJoint;
  highlight_color: HighlightColor;
  show_target_arc: boolean;
}

export interface KioskDecisionPayload {
  kiosk_state: KioskState;
  patient: PatientSummary | null;
  telemetry_analysis: TelemetryAnalysis;
  rep_counter: RepCounter;
  ui_directives: UIDirectives;
  tts_voice_th: string;
  frame_base64: string | null;
}

export interface ExercisePrescriptionOption {
  exercise_id: string;
  catalog_exercise_id?: string;
  schedule_id?: string;
  name_th: string;
  name_en: string;
  target_joint: Exclude<HighlightJoint, 'none'>;
  target_rom_deg: number;
  hold_sec: number;
  target_reps: number;
  target_sets?: number;
}

export type DemoPresetId =
  | 'FACE_SCAN'
  | 'ACTIVE_REP'
  | 'TRUNK_LEAN'
  | 'HOLD_PHASE'
  | 'REP_COMPLETED'
  | 'SET_FINISHED'
  | 'EMERGENCY_HALT';

export type ConnectionStatus = 'connecting' | 'connected' | 'offline';

export const EXERCISE_OPTIONS: ExercisePrescriptionOption[] = [
  {
    exercise_id: 'right_shoulder_abduction',
    name_th: 'กางแขนขวาด้านข้าง',
    name_en: 'Right Shoulder Abduction',
    target_joint: 'right_shoulder',
    target_rom_deg: 150,
    hold_sec: 3,
    target_reps: 10,
  },
  {
    exercise_id: 'left_shoulder_abduction',
    name_th: 'กางแขนซ้ายด้านข้าง',
    name_en: 'Left Shoulder Abduction',
    target_joint: 'left_shoulder',
    target_rom_deg: 150,
    hold_sec: 3,
    target_reps: 10,
  },
  {
    exercise_id: 'chair_squat',
    name_th: 'ลุกนั่งฝึกกล้ามเนื้อขา',
    name_en: 'Chair Squat Rehabilitation',
    target_joint: 'right_knee',
    target_rom_deg: 100,
    hold_sec: 2,
    target_reps: 10,
  },
];

export const DEMO_PATIENT: PatientSummary = {
  id: 'PT-8841',
  name: 'คุณสมชาย ใจดี',
};

function resolveTargetJointForTodayExercise(
  item: TodayPrescribedExerciseItem,
  catalogMatch?: (typeof CLINICAL_EXERCISES)[number],
): Exclude<HighlightJoint, 'none'> {
  if (catalogMatch) {
    const group = catalogMatch.detection.jointGroup;
    if (group === 'knee' || group === 'hip' || group === 'ankle') {
      return catalogMatch.detection.side === 'left' ? 'left_knee' : 'right_knee';
    }
    if (group === 'trunk' || group === 'neck') {
      return 'spine';
    }
    return catalogMatch.detection.side === 'left' ? 'left_shoulder' : 'right_shoulder';
  }
  const combined = `${item.nameTh} ${item.nameEn} ${item.targetJoint}`.toLowerCase();
  if (
    combined.includes('เข่า') ||
    combined.includes('ขา') ||
    combined.includes('knee') ||
    combined.includes('squat')
  ) {
    return combined.includes('ซ้าย') || combined.includes('left') ? 'left_knee' : 'right_knee';
  }
  if (combined.includes('ซ้าย') || combined.includes('left')) {
    return 'left_shoulder';
  }
  return 'right_shoulder';
}

export function mapTodayExercisesToPrescriptions(
  items: readonly TodayPrescribedExerciseItem[],
): ExercisePrescriptionOption[] {
  const physioItems = items.filter(
    (item) =>
      item.activityType !== 'minigame' &&
      !item.exerciseId.includes('card') &&
      !item.exerciseId.includes('minigame'),
  );
  const sourceItems = physioItems.length > 0 ? physioItems : items;
  if (sourceItems.length === 0) {
    return EXERCISE_OPTIONS;
  }

  return sourceItems.map((item) => {
    const found = CLINICAL_EXERCISES.find(
      (c) => c.id === item.exerciseId || c.code === item.exerciseId,
    );
    const target_joint = resolveTargetJointForTodayExercise(item, found);
    const target_rom_deg = found
      ? Math.min(
          160,
          Math.max(90, Math.round((found.detection.targetMinDeg + found.detection.targetMaxDeg) / 2)),
        )
      : target_joint.includes('knee')
        ? 100
        : 150;

    return {
      exercise_id: item.id || item.exerciseId,
      catalog_exercise_id: item.exerciseId,
      schedule_id: item.id,
      name_th: item.nameTh,
      name_en: item.nameEn,
      target_joint,
      target_rom_deg,
      hold_sec: item.holdSec || (found?.detection.holdSec ?? 3),
      target_reps: item.reps || (found?.detection.defaultReps ?? 10),
      target_sets: item.sets || 1,
    };
  });
}

export interface DemoSynthesisParams {
  preset: DemoPresetId;
  romDeg: number;
  trunkTiltDeg: number;
  distanceM: number;
  exercise: ExercisePrescriptionOption;
  validReps: number;
  invalidReps: number;
  holdRemainingSec: number;
  faceMatched?: boolean;
  patientOverride?: PatientSummary | null;
}

export function synthesizeDemoPayload({
  preset,
  romDeg,
  trunkTiltDeg,
  distanceM,
  exercise,
  validReps,
  invalidReps,
  holdRemainingSec,
  faceMatched = false,
  patientOverride,
}: DemoSynthesisParams): KioskDecisionPayload {
  const activePatient = patientOverride ?? DEMO_PATIENT;
  const targetRom = exercise.target_rom_deg;
  const clampedRom = Math.max(0, Math.min(180, Number(romDeg.toFixed(1))));
  const romPct = Math.max(0, Math.min(120, Math.round((clampedRom / targetRom) * 100)));

  if (preset === 'EMERGENCY_HALT') {
    return {
      kiosk_state: 'EMERGENCY_HALT',
      patient: activePatient,
      telemetry_analysis: {
        current_rom_deg: clampedRom,
        target_rom_deg: targetRom,
        rom_completion_pct: romPct,
        is_form_valid: false,
        compensation_detected: ['NONE'],
        pain_alert: true,
      },
      rep_counter: {
        valid_reps: validReps,
        invalid_reps: invalidReps,
        hold_remaining_sec: exercise.hold_sec,
      },
      ui_directives: {
        skeleton_highlight_joint: exercise.target_joint,
        highlight_color: 'RED',
        show_target_arc: false,
      },
      tts_voice_th: 'หยุดพักทันทีนะครับ ระบบกำลังเรียกเจ้าหน้าที่ดูแล',
      frame_base64: null,
    };
  }

  if (preset === 'FACE_SCAN') {
    const distanceOk = distanceM >= 1.8 && distanceM <= 2.2;
    return {
      kiosk_state: 'AUTH_FRAMING',
      patient: faceMatched ? activePatient : null,
      telemetry_analysis: {
        current_rom_deg: 12.0,
        target_rom_deg: targetRom,
        rom_completion_pct: 8,
        is_form_valid: false,
        compensation_detected: ['NONE'],
        pain_alert: false,
      },
      rep_counter: {
        valid_reps: 0,
        invalid_reps: 0,
        hold_remaining_sec: exercise.hold_sec,
      },
      ui_directives: {
        skeleton_highlight_joint: 'none',
        highlight_color: faceMatched && distanceOk ? 'GREEN' : 'AMBER',
        show_target_arc: false,
      },
      tts_voice_th: !faceMatched
        ? 'กรุณามองตรงเข้ากรอบวงกลม เพื่อยืนยันตัวตนอัตโนมัติครับ'
        : distanceM < 1.8
          ? 'ยืนยันตัวตนสำเร็จ กรุณาถอยหลังเล็กน้อยให้อยู่ระยะ 2 เมตรครับ'
          : distanceM > 2.2
            ? 'ยืนยันตัวตนสำเร็จ กรุณาเดินเข้ามาอีกนิดให้อยู่ระยะ 2 เมตรครับ'
            : 'ยืนยันตัวตนสำเร็จ ระยะยืนเหมาะสม พร้อมเริ่มฝึกได้เลยครับ',
      frame_base64: null,
    };
  }

  if (preset === 'SET_FINISHED') {
    const patientName = activePatient.name || 'ผู้ป่วย';
    return {
      kiosk_state: 'SET_FINISHED',
      patient: activePatient,
      telemetry_analysis: {
        current_rom_deg: targetRom,
        target_rom_deg: targetRom,
        rom_completion_pct: 100,
        is_form_valid: true,
        compensation_detected: ['NONE'],
        pain_alert: false,
      },
      rep_counter: {
        valid_reps: exercise.target_reps,
        invalid_reps: invalidReps,
        hold_remaining_sec: 0,
      },
      ui_directives: {
        skeleton_highlight_joint: exercise.target_joint,
        highlight_color: 'GREEN',
        show_target_arc: false,
      },
      tts_voice_th: `เยี่ยมมากครับ ${patientName} ฝึกครบ ${exercise.target_reps} ครั้ง จบเซ็ตแล้วครับ`,
      frame_base64: null,
    };
  }

  if (preset === 'REP_COMPLETED') {
    return {
      kiosk_state: 'REP_COMPLETED',
      patient: activePatient,
      telemetry_analysis: {
        current_rom_deg: Math.max(clampedRom, targetRom * 0.96),
        target_rom_deg: targetRom,
        rom_completion_pct: Math.max(romPct, 96),
        is_form_valid: true,
        compensation_detected: ['NONE'],
        pain_alert: false,
      },
      rep_counter: {
        valid_reps: validReps,
        invalid_reps: invalidReps,
        hold_remaining_sec: 0,
      },
      ui_directives: {
        skeleton_highlight_joint: exercise.target_joint,
        highlight_color: 'GREEN',
        show_target_arc: true,
      },
      tts_voice_th: `ดีมากครับ ครบ ${validReps} ครั้ง ค่อยๆ ผ่อนแขนลงช้าๆ ครับ`,
      frame_base64: null,
    };
  }

  const trunkLeaning = preset === 'TRUNK_LEAN' || Math.abs(trunkTiltDeg) > 10.0;
  if (trunkLeaning) {
    const compensations: CompensationCode[] =
      clampedRom < targetRom * 0.95 ? ['TRUNK_LEAN', 'INCOMPLETE_ROM'] : ['TRUNK_LEAN'];
    return {
      kiosk_state: 'ACTIVE_REP',
      patient: activePatient,
      telemetry_analysis: {
        current_rom_deg: clampedRom,
        target_rom_deg: targetRom,
        rom_completion_pct: romPct,
        is_form_valid: false,
        compensation_detected: compensations,
        pain_alert: false,
      },
      rep_counter: {
        valid_reps: validReps,
        invalid_reps: Math.max(1, invalidReps),
        hold_remaining_sec: exercise.hold_sec,
      },
      ui_directives: {
        skeleton_highlight_joint: 'spine',
        highlight_color: 'AMBER',
        show_target_arc: true,
      },
      tts_voice_th: 'ยืดตัวตรงครับ อย่าเอียงลำตัวช่วยยกนะครับ',
      frame_base64: null,
    };
  }

  const inHoldPhase = preset === 'HOLD_PHASE' || clampedRom >= targetRom * 0.95;
  if (inHoldPhase) {
    const rem = Math.max(0.1, Number(holdRemainingSec.toFixed(1)));
    return {
      kiosk_state: 'HOLD_PHASE',
      patient: activePatient,
      telemetry_analysis: {
        current_rom_deg: clampedRom,
        target_rom_deg: targetRom,
        rom_completion_pct: romPct,
        is_form_valid: true,
        compensation_detected: ['NONE'],
        pain_alert: false,
      },
      rep_counter: {
        valid_reps: validReps,
        invalid_reps: invalidReps,
        hold_remaining_sec: rem,
      },
      ui_directives: {
        skeleton_highlight_joint: exercise.target_joint,
        highlight_color: 'GREEN',
        show_target_arc: true,
      },
      tts_voice_th: `ดีมากครับ ค้างท่าไว้ ${Math.ceil(rem)} วินาที`,
      frame_base64: null,
    };
  }

  return {
    kiosk_state: 'ACTIVE_REP',
    patient: activePatient,
    telemetry_analysis: {
      current_rom_deg: clampedRom,
      target_rom_deg: targetRom,
      rom_completion_pct: romPct,
      is_form_valid: true,
      compensation_detected: ['NONE'],
      pain_alert: false,
    },
    rep_counter: {
      valid_reps: validReps,
      invalid_reps: invalidReps,
      hold_remaining_sec: exercise.hold_sec,
    },
    ui_directives: {
      skeleton_highlight_joint: exercise.target_joint,
      highlight_color: 'GREEN',
      show_target_arc: true,
    },
    tts_voice_th: `ค่อยๆ เคลื่อนไหวไปยังเป้าหมาย ${targetRom} องศาครับ`,
    frame_base64: null,
  };
}

export function useKioskSocket(
  wsUrl = 'ws://localhost:8000/ws/kiosk',
  options?: {
    prescribedExercises?: ExercisePrescriptionOption[];
    patientOverride?: PatientSummary | null;
  },
) {
  const exerciseOptions = useMemo(
    () =>
      options?.prescribedExercises && options.prescribedExercises.length > 0
        ? options.prescribedExercises
        : EXERCISE_OPTIONS,
    [options?.prescribedExercises],
  );
  const hasExplicitContext = Boolean(
    options?.prescribedExercises?.length || options?.patientOverride,
  );

  const [selectedExercise, setSelectedExerciseState] = useState<ExercisePrescriptionOption>(
    () => exerciseOptions[0],
  );
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('connecting');
  const [isDemoMode, setIsDemoMode] = useState<boolean>(false);
  const [livePayload, setLivePayload] = useState<KioskDecisionPayload | null>(null);

  // Interactive Demo Simulator state
  const [demoPreset, setDemoPresetState] = useState<DemoPresetId>('ACTIVE_REP');
  const [demoRomDeg, setDemoRomDeg] = useState<number>(() => (hasExplicitContext ? 15 : 112));
  const [demoTrunkTiltDeg, setDemoTrunkTiltDeg] = useState<number>(2.4);
  const [demoDistanceM, setDemoDistanceM] = useState<number>(2.0);
  const [demoValidReps, setDemoValidReps] = useState<number>(() => (hasExplicitContext ? 0 : 4));
  const [demoInvalidReps, setDemoInvalidReps] = useState<number>(0);
  const [demoHoldRemainingSec, setDemoHoldRemainingSec] = useState<number>(
    () => exerciseOptions[0].hold_sec,
  );
  const [demoFaceMatched, setDemoFaceMatched] = useState<boolean>(false);
  const [demoStreakFrames, setDemoStreakFrames] = useState<number>(2);
  const [autoCycleDemo, setAutoCycleDemo] = useState<boolean>(false);

  const wsRef = useRef<WebSocket | null>(null);
  const userToggledModeRef = useRef<boolean>(false);

  const sendExercisePrescription = useCallback((rx: ExercisePrescriptionOption) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          include_frame: true,
          prescription: {
            exercise_id: rx.catalog_exercise_id ?? rx.exercise_id,
            exercise_name: `${rx.name_en} (${rx.name_th})`,
            target_joint: rx.target_joint,
            target_rom_deg: rx.target_rom_deg,
            hold_sec: rx.hold_sec,
            target_reps: rx.target_reps,
          },
        }),
      );
    }
  }, []);

  useEffect(() => {
    if (!exerciseOptions.some((opt) => opt.exercise_id === selectedExercise.exercise_id)) {
      const first = exerciseOptions[0];
      setSelectedExerciseState(first);
      setDemoValidReps(0);
      setDemoInvalidReps(0);
      setDemoHoldRemainingSec(first.hold_sec);
      setDemoPresetState('ACTIVE_REP');
      sendExercisePrescription(first);
    }
  }, [exerciseOptions, selectedExercise.exercise_id, sendExercisePrescription]);

  const setSelectedExercise = useCallback(
    (rx: ExercisePrescriptionOption) => {
      setSelectedExerciseState(rx);
      setDemoValidReps(0);
      setDemoInvalidReps(0);
      setDemoPresetState('ACTIVE_REP');
      setDemoHoldRemainingSec(rx.hold_sec);
      sendExercisePrescription(rx);
    },
    [sendExercisePrescription],
  );

  // WebSocket client with automatic exponential backoff reconnect
  useEffect(() => {
    let disposed = false;
    let attempt = 0;
    let reconnectTimer: number | null = null;

    const connect = () => {
      if (disposed) return;
      setConnectionStatus('connecting');

      try {
        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;

        ws.onopen = () => {
          if (disposed) return;
          attempt = 0;
          setConnectionStatus('connected');
          if (!userToggledModeRef.current) {
            setIsDemoMode(false);
          }
          sendExercisePrescription(selectedExercise);
        };

        ws.onmessage = (event) => {
          if (disposed) return;
          try {
            const data = JSON.parse(event.data) as KioskDecisionPayload;
            if (data && data.kiosk_state && data.telemetry_analysis) {
              setLivePayload(data);
            }
          } catch {
            // Ignore malformed non-JSON frame
          }
        };

        ws.onerror = () => {
          if (disposed) return;
          ws.close();
        };

        ws.onclose = () => {
          if (disposed) return;
          wsRef.current = null;
          setConnectionStatus('offline');
          if (!userToggledModeRef.current) {
            setIsDemoMode(true);
          }
          const delayMs = Math.min(10000, 1500 * Math.pow(1.6, attempt));
          attempt += 1;
          reconnectTimer = window.setTimeout(connect, delayMs);
        };
      } catch {
        setConnectionStatus('offline');
        if (!userToggledModeRef.current) {
          setIsDemoMode(true);
        }
      }
    };

    connect();
    return () => {
      disposed = true;
      if (reconnectTimer !== null) {
        window.clearTimeout(reconnectTimer);
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [wsUrl, sendExercisePrescription]);

  // Face scan animated streak in FACE_SCAN preset
  useEffect(() => {
    if (!isDemoMode || demoPreset !== 'FACE_SCAN') return;
    setDemoFaceMatched(false);
    setDemoStreakFrames(1);
    const timer = window.setInterval(() => {
      setDemoStreakFrames((prev) => {
        if (prev >= 5) {
          setDemoFaceMatched(true);
          return 5;
        }
        const next = prev + 1;
        if (next >= 5) setDemoFaceMatched(true);
        return next;
      });
    }, 550);
    return () => window.clearInterval(timer);
  }, [isDemoMode, demoPreset]);

  // Hold phase smooth countdown when manually inspecting HOLD_PHASE
  useEffect(() => {
    if (!isDemoMode || demoPreset !== 'HOLD_PHASE' || autoCycleDemo) return;
    setDemoHoldRemainingSec(selectedExercise.hold_sec);
    const interval = window.setInterval(() => {
      setDemoHoldRemainingSec((prev) => {
        if (prev <= 0.2) return selectedExercise.hold_sec;
        return Number((prev - 0.1).toFixed(1));
      });
    }, 100);
    return () => window.clearInterval(interval);
  }, [isDemoMode, demoPreset, autoCycleDemo, selectedExercise.hold_sec]);

  // Smooth continuous auto-simulation loop when Auto-Cycle Demo is toggled ON
  useEffect(() => {
    if (!isDemoMode || !autoCycleDemo) return;
    let phaseT = 0;
    let holdAccum = 0;
    const interval = window.setInterval(() => {
      phaseT += 0.06;
      const target = selectedExercise.target_rom_deg;
      const sine = (Math.sin(phaseT - Math.PI / 2) + 1) / 2;
      const rom = Math.round(sine * (target + 4));
      setDemoRomDeg(rom);

      if (rom >= target * 0.95) {
        holdAccum += 0.05;
        const rem = Math.max(0.1, Number((selectedExercise.hold_sec - holdAccum).toFixed(1)));
        setDemoHoldRemainingSec(rem);
        setDemoPresetState('HOLD_PHASE');
        if (holdAccum >= selectedExercise.hold_sec) {
          holdAccum = 0;
          setDemoValidReps((r) => {
            const next = Math.min(selectedExercise.target_reps, r + 1);
            if (next >= selectedExercise.target_reps) {
              setDemoPresetState('SET_FINISHED');
              setAutoCycleDemo(false);
            } else {
              setDemoPresetState('REP_COMPLETED');
            }
            return next;
          });
        }
      } else {
        holdAccum = 0;
        setDemoHoldRemainingSec(selectedExercise.hold_sec);
        setDemoPresetState(demoTrunkTiltDeg > 10 ? 'TRUNK_LEAN' : 'ACTIVE_REP');
      }
    }, 50);
    return () => window.clearInterval(interval);
  }, [isDemoMode, autoCycleDemo, selectedExercise.target_rom_deg, selectedExercise.hold_sec, selectedExercise.target_reps, demoTrunkTiltDeg]);

  const setDemoPreset = useCallback(
    (preset: DemoPresetId) => {
      setAutoCycleDemo(false);
      setDemoPresetState(preset);
      switch (preset) {
        case 'FACE_SCAN':
          setDemoTrunkTiltDeg(1.5);
          setDemoRomDeg(12);
          break;
        case 'ACTIVE_REP':
          setDemoTrunkTiltDeg(2.2);
          setDemoRomDeg(Math.round(selectedExercise.target_rom_deg * 0.72));
          break;
        case 'TRUNK_LEAN':
          setDemoTrunkTiltDeg(16.4);
          setDemoRomDeg(Math.round(selectedExercise.target_rom_deg * 0.88));
          setDemoInvalidReps((r) => Math.max(1, r));
          break;
        case 'HOLD_PHASE':
          setDemoTrunkTiltDeg(1.8);
          setDemoRomDeg(selectedExercise.target_rom_deg);
          setDemoHoldRemainingSec(selectedExercise.hold_sec);
          break;
        case 'REP_COMPLETED':
          setDemoTrunkTiltDeg(1.5);
          setDemoRomDeg(selectedExercise.target_rom_deg);
          setDemoValidReps((r) => Math.min(selectedExercise.target_reps, r + 1));
          setDemoHoldRemainingSec(0);
          break;
        case 'SET_FINISHED':
          setDemoTrunkTiltDeg(1.2);
          setDemoRomDeg(selectedExercise.target_rom_deg);
          setDemoValidReps(selectedExercise.target_reps);
          setDemoHoldRemainingSec(0);
          break;
        case 'EMERGENCY_HALT':
          setDemoTrunkTiltDeg(4.0);
          break;
      }
    },
    [selectedExercise],
  );

  const toggleDemoMode = useCallback((next?: boolean) => {
    userToggledModeRef.current = true;
    setIsDemoMode((prev) => (typeof next === 'boolean' ? next : !prev));
  }, []);

  const triggerEmergencyStop = useCallback(() => {
    setAutoCycleDemo(false);
    userToggledModeRef.current = true;
    setIsDemoMode(true);
    setDemoPresetState('EMERGENCY_HALT');
  }, []);

  const resetDemoSession = useCallback(() => {
    setAutoCycleDemo(false);
    setDemoValidReps(0);
    setDemoInvalidReps(0);
    setDemoTrunkTiltDeg(1.8);
    setDemoRomDeg(Math.round(selectedExercise.target_rom_deg * 0.65));
    setDemoPresetState('ACTIVE_REP');
  }, [selectedExercise.target_rom_deg]);

  const completeCurrentExercise = useCallback(() => {
    setAutoCycleDemo(false);
    setDemoTrunkTiltDeg(1.2);
    setDemoRomDeg(selectedExercise.target_rom_deg);
    setDemoValidReps(selectedExercise.target_reps);
    setDemoHoldRemainingSec(0);
    setDemoPresetState('SET_FINISHED');
  }, [selectedExercise.target_rom_deg, selectedExercise.target_reps]);

  const demoPayload = synthesizeDemoPayload({
    preset: demoPreset,
    romDeg: demoRomDeg,
    trunkTiltDeg: demoTrunkTiltDeg,
    distanceM: demoDistanceM,
    exercise: selectedExercise,
    validReps: demoValidReps,
    invalidReps: demoInvalidReps,
    holdRemainingSec: demoHoldRemainingSec,
    faceMatched: demoFaceMatched,
    patientOverride: options?.patientOverride,
  });

  const payload: KioskDecisionPayload = !isDemoMode && livePayload ? livePayload : demoPayload;

  return {
    payload,
    connectionStatus,
    isDemoMode,
    toggleDemoMode,
    exerciseOptions,
    selectedExercise,
    setSelectedExercise,
    demoPreset,
    setDemoPreset,
    setDemoPresetState,
    demoRomDeg,
    setDemoRomDeg,
    demoTrunkTiltDeg,
    setDemoTrunkTiltDeg,
    demoDistanceM,
    setDemoDistanceM,
    demoValidReps,
    setDemoValidReps,
    demoInvalidReps,
    demoHoldRemainingSec,
    setDemoHoldRemainingSec,
    demoFaceMatched,
    demoStreakFrames,
    autoCycleDemo,
    setAutoCycleDemo,
    triggerEmergencyStop,
    resetDemoSession,
    completeCurrentExercise,
  };
}

