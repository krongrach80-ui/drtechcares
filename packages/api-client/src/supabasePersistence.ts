import type { SupabaseClient } from '@supabase/supabase-js';
import type { EnrolledTemplateCandidate, BiometricPose } from '@drtechcares/smart-biometric-core';
import type {
  ActiveSessionRecord,
  AuditLogRecord,
  BannedTargetRecord,
  DailyScheduleDay,
  KioskStatusRecord,
  ManagedExerciseRecord,
  MedicalNoteRecord,
  ModpackRecord,
  PatientCourseRecord,
  PatientRecord,
  RehabSessionRecord,
  StaffCallRecord,
  SystemSettingsRecord,
  TreatmentReportRecord,
  UserAccount,
  UserRole,
} from './index';

export interface StoredUserAccount extends UserAccount {
  passwordHash: string;
}

export interface CloudRouterState {
  users: StoredUserAccount[];
  patients: PatientRecord[];
  face_templates: EnrolledTemplateCandidate[];
  medical_notes: MedicalNoteRecord[];
  patient_courses: PatientCourseRecord[];
  rehab_sessions: RehabSessionRecord[];
  treatment_reports: TreatmentReportRecord[];
  exercises: ManagedExerciseRecord[];
  modpacks: ModpackRecord[];
  staff_calls: StaffCallRecord[];
  audit_logs: AuditLogRecord[];
  banned_targets: BannedTargetRecord[];
  active_sessions: ActiveSessionRecord[];
  system_settings: SystemSettingsRecord;
  kiosks: KioskStatusRecord[];
}

export function newRecordId(useUuid: boolean, prefix: string): string {
  if (useUuid && typeof globalThis.crypto?.randomUUID === 'function') {
    return globalThis.crypto.randomUUID();
  }
  const bytes = new Uint8Array(4);
  globalThis.crypto.getRandomValues(bytes);
  const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
  return `${prefix}-${Date.now()}-${hex}`;
}

export async function hydrateStateFromSupabase(
  supabase: SupabaseClient,
  state: CloudRouterState,
): Promise<void> {
  const [
    usersRes,
    patientsRes,
    templatesRes,
    notesRes,
    coursesRes,
    sessionsRes,
    reportsRes,
    callsRes,
    bansRes,
    activeSessionsRes,
    settingsRes,
    kiosksRes,
  ] = await Promise.allSettled([
    supabase.from('staff_users').select('*'),
    supabase.from('patients').select('*'),
    supabase.from('face_templates').select('patient_id, pose, embedding'),
    supabase.from('medical_notes').select('*').order('created_at', { ascending: true }),
    supabase.from('patient_courses').select('*'),
    supabase.from('rehab_sessions').select('*').order('created_at', { ascending: false }),
    supabase.from('treatment_reports').select('*').order('created_at', { ascending: false }),
    supabase.from('staff_calls').select('*').order('created_at', { ascending: false }),
    supabase.from('banned_targets').select('*').order('created_at', { ascending: false }),
    supabase.from('active_sessions').select('*').eq('revoked', false),
    supabase.from('system_settings').select('*').eq('id', 1).maybeSingle(),
    supabase.from('kiosks').select('*'),
  ]);

  if (usersRes.status === 'fulfilled' && Array.isArray(usersRes.value.data) && usersRes.value.data.length > 0) {
    state.users = usersRes.value.data.map((r: Record<string, unknown>) => ({
      id: String(r.id),
      username: String(r.username),
      passwordHash: String(r.password_hash ?? ''),
      role: (r.role as UserRole) ?? 'pt',
      fullName: String(r.full_name ?? ''),
      phone: String(r.phone ?? ''),
      specialty: r.specialty ? String(r.specialty) : undefined,
      licenseNo: r.license_no ? String(r.license_no) : undefined,
      patientId: r.patient_id ? String(r.patient_id) : undefined,
      active: Boolean(r.active ?? true),
      forceLoggedOutAt: r.force_logged_out_at ? String(r.force_logged_out_at) : null,
      createdAt: String(r.created_at ?? new Date().toISOString()),
    }));
  }

  if (patientsRes.status === 'fulfilled' && Array.isArray(patientsRes.value.data)) {
    state.patients = patientsRes.value.data.map((r: Record<string, unknown>) => ({
      id: String(r.id),
      hn: String(r.hn),
      fullName: String(r.full_name),
      age: Number(r.age ?? 0),
      phone: String(r.phone ?? ''),
      diagnosis: String(r.diagnosis ?? ''),
      assignedPtId: String(r.assigned_pt_id ?? ''),
      delegatedToPtId: r.delegated_to_pt_id ? String(r.delegated_to_pt_id) : null,
      delegationNote: r.delegation_note ? String(r.delegation_note) : null,
      nextAppointmentDate: String(r.next_appointment_date ?? ''),
      registrationStatus: (r.registration_status as 'active' | 'archived') ?? 'active',
      enrolledAt: String(r.created_at ?? new Date().toISOString()),
    }));
  }

  if (templatesRes.status === 'fulfilled' && Array.isArray(templatesRes.value.data)) {
    const patientNames = new Map(state.patients.map((p) => [p.id, p.fullName]));
    state.face_templates = templatesRes.value.data
      .filter((r: Record<string, unknown>) => Array.isArray(r.embedding))
      .map((r: Record<string, unknown>) => ({
        patientId: String(r.patient_id),
        patientName: patientNames.get(String(r.patient_id)),
        pose: (r.pose as BiometricPose) ?? 'frontal_centroid',
        vector: r.embedding as number[],
      }));
  }

  if (notesRes.status === 'fulfilled' && Array.isArray(notesRes.value.data)) {
    state.medical_notes = notesRes.value.data.map((r: Record<string, unknown>) => ({
      id: String(r.id),
      patientId: String(r.patient_id),
      authorId: String(r.author_id),
      authorName: String(r.author_name),
      content: String(r.content),
      createdAt: String(r.created_at),
    }));
  }

  if (coursesRes.status === 'fulfilled' && Array.isArray(coursesRes.value.data)) {
    state.patient_courses = coursesRes.value.data.map((r: Record<string, unknown>) => ({
      id: String(r.id),
      patientId: String(r.patient_id),
      title: String(r.title),
      weeklySchedule: (r.weekly_schedule as DailyScheduleDay[]) ?? [],
      cognitiveModpackId: String(r.cognitive_modpack_id ?? 'modpack-core-99'),
      updatedAt: String(r.updated_at ?? new Date().toISOString()),
    }));
  }

  if (sessionsRes.status === 'fulfilled' && Array.isArray(sessionsRes.value.data)) {
    state.rehab_sessions = sessionsRes.value.data.map((r: Record<string, unknown>) => ({
      id: String(r.id),
      patientId: String(r.patient_id),
      activityType: 'physio_exercise',
      exerciseId: String(r.exercise_id),
      exerciseName: String(r.exercise_id),
      completedSets: Number(r.completed_sets ?? 0),
      completedReps: Number(r.completed_reps ?? 0),
      accuracyPercent: Number(r.accuracy_percent ?? 100),
      maxRomDeg: Number(r.max_rom_deg ?? 0),
      painStopTriggered: Boolean(r.pain_stop_triggered),
      aiFeedback: 'Synced from Supabase',
      kinematicsReplay: (r.kinematics_replay as { tMs: number; angleDeg: number }[]) ?? [],
      minigameScore: r.minigame_score as { correct: number; total: number; category: string } | undefined,
      createdAt: String(r.created_at),
    }));
  }

  if (reportsRes.status === 'fulfilled' && Array.isArray(reportsRes.value.data)) {
    state.treatment_reports = reportsRes.value.data.map((r: Record<string, unknown>) => ({
      id: String(r.id),
      patientId: String(r.patient_id),
      ptId: String(r.pt_id),
      ptName: String(r.pt_name),
      summary: String(r.summary),
      aiRecommendation: String(r.ai_recommendation ?? ''),
      scoreSummary: String(r.score_summary ?? ''),
      createdAt: String(r.created_at),
      updatedAt: String(r.updated_at),
    }));
  }

  if (callsRes.status === 'fulfilled' && Array.isArray(callsRes.value.data)) {
    state.staff_calls = callsRes.value.data.map((r: Record<string, unknown>) => ({
      id: String(r.id),
      kioskId: String(r.kiosk_id),
      patientId: r.patient_id ? String(r.patient_id) : null,
      reason: String(r.reason),
      status: (r.status as 'open' | 'acknowledged' | 'resolved') ?? 'open',
      createdAt: String(r.created_at),
    }));
  }

  if (bansRes.status === 'fulfilled' && Array.isArray(bansRes.value.data)) {
    state.banned_targets = bansRes.value.data.map((r: Record<string, unknown>) => ({
      id: String(r.id),
      targetType: (r.target_type as 'ip' | 'device') ?? 'device',
      value: String(r.value),
      reason: String(r.reason ?? ''),
      bannedBy: String(r.banned_by ?? ''),
      createdAt: String(r.created_at),
    }));
  }

  if (activeSessionsRes.status === 'fulfilled' && Array.isArray(activeSessionsRes.value.data)) {
    state.active_sessions = activeSessionsRes.value.data.map((r: Record<string, unknown>) => ({
      token: String(r.token),
      userId: String(r.user_id),
      role: (r.role as UserRole) ?? 'pt',
      ipAddress: String(r.ip_address ?? 'unknown'),
      deviceId: String(r.device_id ?? 'unknown'),
      createdAt: String(r.created_at),
      revoked: Boolean(r.revoked),
    }));
  }

  if (settingsRes.status === 'fulfilled' && settingsRes.value.data) {
    const s = settingsRes.value.data as Record<string, unknown>;
    state.system_settings = {
      aiFaceScanEnabled: Boolean(s.ai_face_scan_enabled ?? true),
      aiFaceThreshold: Number(s.ai_face_threshold ?? 0.925),
      aiPoseDetectionEnabled: Boolean(s.ai_pose_detection_enabled ?? false),
      kioskAutoResetSec: Number(s.kiosk_auto_reset_sec ?? 6),
    };
  }

  if (kiosksRes.status === 'fulfilled' && Array.isArray(kiosksRes.value.data) && kiosksRes.value.data.length > 0) {
    state.kiosks = kiosksRes.value.data.map((r: Record<string, unknown>) => ({
      kioskId: String(r.kiosk_id),
      location: String(r.location ?? ''),
      online: Boolean(r.online ?? true),
      currentState: String(r.current_state ?? 'STANDBY'),
      lastHeartbeatAt: String(r.last_heartbeat_at ?? new Date().toISOString()),
    }));
  }
}

export function persistToSupabase(
  supabase: SupabaseClient | null,
  task: (client: SupabaseClient) => PromiseLike<unknown>,
): void {
  if (!supabase) return;
  void Promise.resolve(task(supabase)).catch(() => {
    // Non-blocking best-effort persistence for edge sync
  });
}

