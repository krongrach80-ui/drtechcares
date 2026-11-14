import type { SupabaseClient } from '@supabase/supabase-js';
import type { EnrolledTemplateCandidate, BiometricPose } from '@drtechcares/smart-biometric-core';
import type {
  ActiveSessionRecord,
  AngleEmbeddings5D,
  AuditLogRecord,
  BannedTargetRecord,
  DailyScheduleDay,
  KioskStatusRecord,
  ManagedExerciseRecord,
  MedicalNoteRecord,
  ModpackRecord,
  PatientCourseRecord,
  PatientRecord,
  RehabScheduleItem,
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

export function normalizeStaffRole(rawRole: unknown): UserRole {
  const r = String(rawRole ?? 'pt').trim().toLowerCase();
  if (r === 'super_admin' || r === 'admin' || r === 'director') return 'super_admin';
  if (r === 'patient') return 'patient';
  return 'pt';
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
    exercisesRes,
    modpacksRes,
    callsRes,
    auditRes,
    bansRes,
    activeSessionsRes,
    settingsRes,
    kiosksRes,
    biometrics120Res,
  ] = await Promise.allSettled([
    supabase.from('staff_users').select('*'),
    supabase.from('patients').select('*'),
    supabase.from('face_templates').select('patient_id, pose, embedding, quality_score'),
    supabase.from('medical_notes').select('*').order('created_at', { ascending: true }),
    supabase.from('patient_courses').select('*'),
    supabase.from('rehab_sessions').select('*').order('created_at', { ascending: false }),
    supabase.from('treatment_reports').select('*').order('created_at', { ascending: false }),
    supabase.from('exercises').select('*').eq('active', true),
    supabase.from('modpacks').select('*').eq('active', true),
    supabase.from('staff_calls').select('*').order('created_at', { ascending: false }),
    supabase.from('audit_logs').select('*').order('created_at', { ascending: false }).limit(250),
    supabase.from('banned_targets').select('*').order('created_at', { ascending: false }),
    supabase.from('active_sessions').select('*').eq('revoked', false),
    supabase.from('system_settings').select('*').eq('id', 1).maybeSingle(),
    supabase.from('kiosks').select('*'),
    supabase.from('patient_biometrics').select('*'),
  ]);

  if (usersRes.status === 'fulfilled' && Array.isArray(usersRes.value.data) && usersRes.value.data.length > 0) {
    const dbUsers: StoredUserAccount[] = usersRes.value.data.map((r: Record<string, unknown>) => ({
      id: String(r.id),
      username: String(r.username),
      passwordHash: String(r.password_hash ?? ''),
      role: normalizeStaffRole(r.role),
      fullName: String(r.full_name ?? ''),
      phone: String(r.phone ?? ''),
      specialty: r.specialty ? String(r.specialty) : undefined,
      licenseNo: r.license_no
        ? String(r.license_no)
        : r.license_number
          ? String(r.license_number)
          : undefined,
      patientId: r.patient_id ? String(r.patient_id) : undefined,
      active:
        r.active !== undefined && r.active !== null
          ? Boolean(r.active)
          : String(r.status ?? 'ACTIVE').toUpperCase() !== 'SUSPENDED',
      forceLoggedOutAt: r.force_logged_out_at ? String(r.force_logged_out_at) : null,
      createdAt: String(r.created_at ?? new Date().toISOString()),
    }));
    const dbIds = new Set(dbUsers.map((u) => u.id));
    const dbUsernames = new Set(dbUsers.map((u) => u.username.toLowerCase()));
    const localOnly = state.users.filter(
      (u) => !dbIds.has(u.id) && !dbUsernames.has(u.username.toLowerCase()),
    );
    state.users = [...dbUsers, ...localOnly];
  }

  const parsedTemplatesByPatient = new Map<
    string,
    { frontal?: number[]; overall?: number[]; raw: { pose: 'front' | 'left' | 'right' | 'nod'; vector: number[]; qualityScore: number }[] }
  >();

  if (templatesRes.status === 'fulfilled' && Array.isArray(templatesRes.value.data)) {
    const rawPatientNameMap = new Map<string, string>();
    if (patientsRes.status === 'fulfilled' && Array.isArray(patientsRes.value.data)) {
      patientsRes.value.data.forEach((p: Record<string, unknown>) => {
        rawPatientNameMap.set(String(p.id), String(p.full_name ?? ''));
      });
    }
    state.face_templates = [];
    for (const row of templatesRes.value.data as Record<string, unknown>[]) {
      const vec = Array.isArray(row.embedding)
        ? (row.embedding as number[])
        : typeof row.embedding === 'string' && row.embedding.startsWith('[')
          ? (JSON.parse(row.embedding) as number[])
          : null;
      if (!vec) continue;
      const pid = String(row.patient_id);
      const pose = (row.pose as BiometricPose) ?? 'frontal_centroid';
      state.face_templates.push({
        patientId: pid,
        patientName: rawPatientNameMap.get(pid),
        pose,
        vector: vec,
      });
      const entry = parsedTemplatesByPatient.get(pid) ?? { raw: [] };
      if (pose === 'frontal_centroid') entry.frontal = vec;
      else if (pose === 'overall_centroid') entry.overall = vec;
      else if (pose === 'front' || pose === 'left' || pose === 'right' || pose === 'nod') {
        entry.raw.push({ pose, vector: vec, qualityScore: Number(row.quality_score ?? 1.0) });
      }
      parsedTemplatesByPatient.set(pid, entry);
    }
  }

  const biometrics120ByPatient = new Map<
    string,
    {
      masterEmbedding?: number[];
      angleEmbeddings?: AngleEmbeddings5D;
      framesCaptured?: number;
      avatarUrl?: string;
    }
  >();
  if (biometrics120Res.status === 'fulfilled' && Array.isArray(biometrics120Res.value.data)) {
    for (const row of biometrics120Res.value.data as Record<string, unknown>[]) {
      const pid = String(row.patient_id);
      biometrics120ByPatient.set(pid, {
        masterEmbedding: Array.isArray(row.master_embedding) ? (row.master_embedding as number[]) : undefined,
        angleEmbeddings:
          row.angle_embeddings && typeof row.angle_embeddings === 'object'
            ? (row.angle_embeddings as AngleEmbeddings5D)
            : undefined,
        framesCaptured: Number(row.frames_captured ?? 120),
        avatarUrl: typeof row.avatar_url === 'string' ? row.avatar_url : undefined,
      });
    }
  }

  if (patientsRes.status === 'fulfilled' && Array.isArray(patientsRes.value.data)) {
    state.patients = patientsRes.value.data.map((r: Record<string, unknown>) => {
      const pid = String(r.id);
      const history = (r.history as Record<string, unknown>) ?? {};
      const tpl = parsedTemplatesByPatient.get(pid);
      const b120 = biometrics120ByPatient.get(pid);
      const hasTemplates = Boolean((tpl && tpl.raw.length > 0) || b120?.masterEmbedding);
      const historyStatus = history.faceBiometricStatus;
      const colStatus = r.face_biometric_status;
      const resolvedFaceStatus: 'PENDING' | 'VERIFIED' =
        colStatus === 'VERIFIED' || historyStatus === 'VERIFIED' || hasTemplates ? 'VERIFIED' : 'PENDING';

      const fallbackMaster =
        b120?.masterEmbedding ??
        (Array.isArray(history.masterEmbedding) ? (history.masterEmbedding as number[]) : undefined) ??
        tpl?.overall ??
        tpl?.frontal;

      let fallbackAngles: AngleEmbeddings5D | undefined =
        b120?.angleEmbeddings ??
        (history.angleEmbeddings && typeof history.angleEmbeddings === 'object'
          ? (history.angleEmbeddings as AngleEmbeddings5D)
          : undefined);
      if (!fallbackAngles && fallbackMaster) {
        const frontVec = tpl?.raw.find((f) => f.pose === 'front')?.vector ?? tpl?.frontal ?? fallbackMaster;
        const leftVec = tpl?.raw.find((f) => f.pose === 'left')?.vector ?? fallbackMaster;
        const rightVec = tpl?.raw.find((f) => f.pose === 'right')?.vector ?? fallbackMaster;
        const nodVec = tpl?.raw.find((f) => f.pose === 'nod')?.vector ?? fallbackMaster;
        fallbackAngles = {
          center: frontVec,
          left: leftVec,
          right: rightVec,
          up: nodVec,
          down: nodVec,
        };
      }

      const resolvedFramesCount =
        b120?.framesCaptured ??
        (typeof r.face_frames_count === 'number' ? r.face_frames_count : undefined) ??
        (typeof history.faceFramesCount === 'number' ? history.faceFramesCount : undefined) ??
        (resolvedFaceStatus === 'VERIFIED' ? 120 : 0);

      return {
        id: pid,
        hn: String(r.hn),
        fullName: String(r.full_name),
        age: Number(r.age ?? 0),
        gender: typeof r.gender === 'string' ? r.gender : typeof history.gender === 'string' ? history.gender : 'ชาย',
        phone: String(r.phone ?? ''),
        diagnosis: String(r.diagnosis ?? ''),
        assignedPtId: String(r.assigned_pt_id ?? ''),
        delegatedToPtId: r.delegated_to_pt_id ? String(r.delegated_to_pt_id) : null,
        delegationNote: r.delegation_note ? String(r.delegation_note) : null,
        nextAppointmentDate: String(r.next_appointment_date ?? ''),
        registrationStatus: (r.registration_status as 'active' | 'archived') ?? 'active',
        enrolledAt: String(r.created_at ?? new Date().toISOString()),
        faceBiometricStatus: resolvedFaceStatus,
        faceFramesCount: resolvedFramesCount,
        faceImageUrl:
          b120?.avatarUrl ??
          (typeof r.face_image_url === 'string'
            ? r.face_image_url
            : typeof history.faceImageUrl === 'string'
              ? history.faceImageUrl
              : undefined),
        masterEmbedding: fallbackMaster,
        angleEmbeddings: fallbackAngles,
        rehabSchedules: Array.isArray(history.rehabSchedules)
          ? (history.rehabSchedules as RehabScheduleItem[])
          : [],
        biometricHistory:
          tpl && tpl.frontal && tpl.overall
            ? {
                frontal_centroid: tpl.frontal,
                overall_centroid: tpl.overall,
                raw_frames: tpl.raw,
              }
            : fallbackMaster
              ? {
                  frontal_centroid: fallbackAngles?.center ?? fallbackMaster,
                  overall_centroid: fallbackMaster,
                  raw_frames: [],
                }
              : undefined,
      };
    });
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

  if (exercisesRes.status === 'fulfilled' && Array.isArray(exercisesRes.value.data) && exercisesRes.value.data.length > 0) {
    const dbExIds = new Set(exercisesRes.value.data.map((r: Record<string, unknown>) => String(r.id)));
    const dbExercises: ManagedExerciseRecord[] = exercisesRes.value.data.map((r: Record<string, unknown>) => ({
      id: String(r.id),
      code: String(r.code),
      category: (r.category as ManagedExerciseRecord['category']) ?? 'upper_limb',
      name: (r.name_i18n as ManagedExerciseRecord['name']) ?? {
        th: String(r.name_th ?? r.code),
        en: String(r.name_th ?? r.code),
        cn: String(r.name_th ?? r.code),
      },
      instruction: (r.instruction_i18n as ManagedExerciseRecord['instruction']) ?? {
        th: String(r.instructions_th ?? ''),
        en: String(r.instructions_th ?? ''),
        cn: String(r.instructions_th ?? ''),
      },
      detection: (r.detection_config as ManagedExerciseRecord['detection']) ?? {
        mode: 'angle_rep_hold',
        jointGroup: 'shoulder',
        side: 'right',
        targetMinDeg: 30,
        targetMaxDeg: 135,
        startRestMaxDeg: 25,
        holdSec: 2,
        defaultSets: 2,
        defaultReps: 8,
      },
      createdBy: 'system',
    }));
    const defaultsOnly = state.exercises.filter((e) => !dbExIds.has(e.id));
    state.exercises = [...defaultsOnly, ...dbExercises];
  }

  if (modpacksRes.status === 'fulfilled' && Array.isArray(modpacksRes.value.data) && modpacksRes.value.data.length > 0) {
    state.modpacks = modpacksRes.value.data.map((r: Record<string, unknown>) => ({
      id: String(r.id),
      name: String(r.name),
      categories: Array.isArray(r.categories) ? (r.categories as string[]) : [String(r.category ?? 'cognitive')],
      questionCount: Number(r.card_count ?? (Array.isArray(r.questions) && r.questions.length > 0 ? r.questions.length : 10)),
      createdBy: 'system',
      active: Boolean(r.active ?? true),
    }));
  }

  if (callsRes.status === 'fulfilled' && Array.isArray(callsRes.value.data)) {
    state.staff_calls = callsRes.value.data.map((r: Record<string, unknown>) => {
      const rawReason = String(r.reason ?? '');
      const rawStatus = String(r.status ?? 'open');
      const isAcked = rawStatus === 'open' && rawReason.startsWith('[ACKNOWLEDGED]');
      return {
        id: String(r.id),
        kioskId: String(r.kiosk_id),
        patientId: r.patient_id ? String(r.patient_id) : null,
        reason: isAcked ? rawReason.replace(/^\[ACKNOWLEDGED\]\s*/, '') : rawReason,
        status: isAcked ? 'acknowledged' : ((rawStatus as 'open' | 'acknowledged' | 'resolved') ?? 'open'),
        createdAt: String(r.created_at),
      };
    });
  }

  if (auditRes.status === 'fulfilled' && Array.isArray(auditRes.value.data)) {
    state.audit_logs = auditRes.value.data.map((r: Record<string, unknown>) => {
      const detailObj = (r.detail as Record<string, unknown> | null) ?? (r.details as Record<string, unknown> | null);
      const detailMsg =
        typeof detailObj?.message === 'string'
          ? detailObj.message
          : typeof r.action === 'string'
            ? `${r.action} ${r.target ?? ''}`.trim()
            : JSON.stringify(detailObj ?? {});
      return {
        id: String(r.id),
        eventType: String(r.event_type ?? r.action ?? 'SYSTEM_EVENT'),
        actorRole: String(r.actor_role ?? 'system'),
        actorId: String(r.actor_id ?? ''),
        actorName: String(r.actor_name ?? ''),
        ipAddress: String(r.ip_address ?? 'unknown'),
        deviceId: String(r.device_id ?? 'unknown'),
        targetId: r.target_id ? String(r.target_id) : r.target ? String(r.target) : undefined,
        detail: detailMsg,
        createdAt: String(r.created_at ?? new Date().toISOString()),
      };
    });
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
    const merged = new Map<string, ActiveSessionRecord>();
    for (const s of state.active_sessions) {
      merged.set(s.token, s);
    }
    for (const r of activeSessionsRes.value.data as Record<string, unknown>[]) {
      const token = String(r.token);
      const prev = merged.get(token);
      merged.set(token, {
        token,
        userId: String(r.user_id),
        role: normalizeStaffRole(r.role),
        ipAddress: String(r.ip_address ?? 'unknown'),
        deviceId: String(r.device_id ?? 'unknown'),
        createdAt: String(r.created_at),
        revoked: Boolean(prev?.revoked) || Boolean(r.revoked),
      });
    }
    state.active_sessions = Array.from(merged.values());
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
    state.kiosks = kiosksRes.value.data.map((r: Record<string, unknown>) => {
      const rawState = String(r.current_state ?? 'STANDBY');
      const [parsedState, parsedToken] = rawState.includes(':') ? rawState.split(':', 2) : [rawState, undefined];
      return {
        kioskId: String(r.kiosk_id),
        location: String(r.location ?? ''),
        online: Boolean(r.online ?? true),
        currentState: parsedState || 'STANDBY',
        pairingToken: typeof r.pairing_token === 'string' ? r.pairing_token : parsedToken,
        lastHeartbeatAt: String(r.last_heartbeat_at ?? new Date().toISOString()),
      };
    });
  }
}

export async function persistToSupabase(
  supabase: SupabaseClient | null,
  task: (client: SupabaseClient) => PromiseLike<unknown>,
): Promise<void> {
  if (!supabase) return;
  try {
    await task(supabase);
  } catch {
    // Non-fatal fallback for offline / missing column edge cases
  }
}
