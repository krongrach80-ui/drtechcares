import { createClient, type SupabaseClient } from '@supabase/supabase-js';
import {
  purgeLegacyBiometricCache,
  verifyHybridBiometricIdentity,
  type BiometricPose,
  type BiometricVerificationResult,
  type EnrolledTemplateCandidate,
} from '@drtechcares/smart-biometric-core';
import { CLINICAL_EXERCISES, type BodySide, type ExerciseConfig } from '@drtechcares/pose-engine';

export type UserRole = 'super_admin' | 'pt' | 'patient';
export type ActivityType = 'minigame' | 'physio_exercise';

export interface UserAccount {
  id: string;
  username: string;
  role: UserRole;
  fullName: string;
  phone: string;
  specialty?: string;
  licenseNo?: string;
  patientId?: string;
  active: boolean;
  forceLoggedOutAt?: string | null;
  createdAt: string;
}

export interface RehabScheduleItem {
  id: string;
  patientId: string;
  ptId?: string;
  exerciseId?: string;
  modpackId?: string;
  scheduledDate: string; // YYYY-MM-DD
  sets: number;
  reps: number;
  status: 'ACTIVE' | 'REST' | 'COMPLETED' | 'MISSED';
  notes?: string;
}

export interface PatientRecord {
  id: string;
  hn: string;
  fullName: string;
  age: number;
  gender?: string;
  phone: string;
  diagnosis: string;
  assignedPtId: string;
  delegatedToPtId: string | null;
  delegationNote: string | null;
  nextAppointmentDate: string;
  registrationStatus: 'active' | 'archived';
  enrolledAt: string;
  faceBiometricStatus?: 'PENDING' | 'VERIFIED';
  faceImageUrl?: string;
  rehabSchedules?: RehabScheduleItem[];
  biometricHistory?: {
    frontal_centroid: number[];
    overall_centroid: number[];
    raw_frames: { pose: 'front' | 'left' | 'right' | 'nod'; vector: number[]; qualityScore: number }[];
  };
}

export interface MedicalNoteRecord {
  id: string;
  patientId: string;
  authorId: string;
  authorName: string;
  content: string;
  createdAt: string;
}

export interface PrescribedExerciseItem {
  exerciseId: string;
  side: BodySide;
  sets: number;
  reps: number;
  holdSec: number;
  targetMinDeg: number;
  targetMaxDeg: number;
  allowedStarterSide: BodySide | 'both';
}

export interface DailyScheduleDay {
  dayIndex: number;
  dayType: 'training' | 'rest';
  activityType: ActivityType;
  difficulty: 'easy' | 'medium' | 'hard';
  label: string;
  exercises: PrescribedExerciseItem[];
  cognitiveModpackId?: string;
}

export interface PatientCourseRecord {
  id: string;
  patientId: string;
  title: string;
  weeklySchedule: DailyScheduleDay[];
  cognitiveModpackId: string;
  updatedAt: string;
}

export interface RehabSessionRecord {
  id: string;
  patientId: string;
  activityType: ActivityType;
  exerciseId: string;
  exerciseName: string;
  completedSets: number;
  completedReps: number;
  accuracyPercent: number;
  maxRomDeg: number;
  painStopTriggered: boolean;
  aiFeedback: string;
  kinematicsReplay: { tMs: number; angleDeg: number }[];
  minigameScore?: { correct: number; total: number; category: string };
  createdAt: string;
}

export interface TreatmentReportRecord {
  id: string;
  patientId: string;
  ptId: string;
  ptName: string;
  summary: string;
  aiRecommendation: string;
  scoreSummary: string;
  createdAt: string;
  updatedAt: string;
}

export interface ManagedExerciseRecord extends ExerciseConfig {
  createdBy: string;
}

export interface ModpackRecord {
  id: string;
  name: string;
  categories: string[];
  questionCount: number;
  createdBy: string;
  active: boolean;
}

export interface StaffCallRecord {
  id: string;
  kioskId: string;
  patientId?: string | null;
  reason: string;
  status: 'open' | 'acknowledged' | 'resolved';
  createdAt: string;
}

export interface AuditLogRecord {
  id: string;
  eventType: string;
  actorRole: string;
  actorId: string;
  actorName: string;
  ipAddress: string;
  deviceId: string;
  targetId?: string;
  detail: string;
  createdAt: string;
}

export interface BannedTargetRecord {
  id: string;
  targetType: 'ip' | 'device';
  value: string;
  reason: string;
  bannedBy: string;
  createdAt: string;
}

export interface ActiveSessionRecord {
  token: string;
  userId: string;
  role: UserRole;
  ipAddress: string;
  deviceId: string;
  createdAt: string;
  revoked: boolean;
}

export interface SystemSettingsRecord {
  aiFaceScanEnabled: boolean;
  aiFaceThreshold: number;
  aiPoseDetectionEnabled: boolean;
  kioskAutoResetSec: number;
}

export interface KioskStatusRecord {
  kioskId: string;
  location: string;
  online: boolean;
  currentState: string;
  lastHeartbeatAt: string;
  pairingToken?: string;
}

export interface KioskFaceVerifyResponse extends BiometricVerificationResult {
  patient?: {
    id: string;
    hn: string;
    fullName: string;
    nextAppointmentDate: string;
  } | null;
  todayRoutine?: DailyScheduleDay | null;
}

export interface CloudClientConfig {
  supabaseUrl?: string;
  supabaseAnonKey?: string;
  supabaseServiceKey?: string;
  apiBaseUrl?: string;
}

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly code: string,
    message: string,
  ) {
    super(message);
  }
}

export class DrTechCaresApiClient {
  private supabase: SupabaseClient | null = null;
  private authToken: string | null = null;
  private readonly baseUrl: string;
  private readonly deviceId: string;

  constructor(config: CloudClientConfig = {}) {
    purgeLegacyBiometricCache();
    this.baseUrl = config.apiBaseUrl ?? '';
    this.deviceId = this.resolveDeviceId();

    if (typeof globalThis.sessionStorage !== 'undefined') {
      this.authToken = globalThis.sessionStorage.getItem('drtechcares_auth_token');
    }

    const url = config.supabaseUrl || '';
    const key = config.supabaseServiceKey || config.supabaseAnonKey || '';
    if (url && key) {
      this.supabase = createClient(url, key, {
        auth: { persistSession: false, autoRefreshToken: false },
      });
    }
  }

  private resolveDeviceId(): string {
    if (typeof globalThis.sessionStorage !== 'undefined') {
      let id = globalThis.sessionStorage.getItem('drtechcares_device_id');
      if (!id) {
        id = `DEV-${Math.random().toString(36).slice(2, 8).toUpperCase()}`;
        globalThis.sessionStorage.setItem('drtechcares_device_id', id);
      }
      return id;
    }
    return 'server-worker';
  }

  setAuthToken(token: string | null): void {
    this.authToken = token;
    if (typeof globalThis.sessionStorage !== 'undefined') {
      if (token) globalThis.sessionStorage.setItem('drtechcares_auth_token', token);
      else globalThis.sessionStorage.removeItem('drtechcares_auth_token');
    }
  }

  getAuthToken(): string | null {
    return this.authToken;
  }

  getSupabaseClient(): SupabaseClient | null {
    return this.supabase;
  }

  async checkHealth(): Promise<{
    ok: boolean;
    persistentDb: boolean;
    supabaseConnected: boolean;
    latencyMs: number;
  }> {
    const start = Date.now();
    try {
      const res = await this.request<{
        ok: boolean;
        persistentDb?: boolean;
        supabaseConnected?: boolean;
      }>('/api/health');
      return {
        ok: Boolean(res.ok),
        persistentDb: Boolean(res.persistentDb),
        supabaseConnected: res.supabaseConnected ?? Boolean(res.persistentDb),
        latencyMs: Date.now() - start,
      };
    } catch {
      return {
        ok: false,
        persistentDb: false,
        supabaseConnected: false,
        latencyMs: Date.now() - start,
      };
    }
  }

  private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'x-device-id': this.deviceId,
      ...(init.headers as Record<string, string> | undefined),
    };
    if (this.authToken) {
      headers.Authorization = `Bearer ${this.authToken}`;
    }
    const res = await fetch(`${this.baseUrl}${path}`, { ...init, headers });
    const text = await res.text();
    const data = text ? (JSON.parse(text) as Record<string, unknown>) : {};
    if (!res.ok) {
      if (res.status === 401 && path !== '/api/auth/login' && path !== '/api/biometric/verify') {
        this.setAuthToken(null);
      }
      throw new ApiError(
        res.status,
        String(data.error ?? 'API_ERROR'),
        String(data.message ?? `Request failed (${res.status})`),
      );
    }
    return data as T;
  }

  // Auth
  async login(username: string, password: string): Promise<{ token: string; user: UserAccount }> {
    const res = await this.request<{ token: string; user: UserAccount }>('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    });
    this.setAuthToken(res.token);
    return res;
  }

  async logout(): Promise<void> {
    try {
      await this.request('/api/auth/logout', { method: 'POST' });
    } finally {
      this.setAuthToken(null);
    }
  }

  async getMe(): Promise<UserAccount | null> {
    if (!this.authToken) return null;
    try {
      const res = await this.request<{ user: UserAccount }>('/api/auth/me');
      return res.user;
    } catch {
      return null;
    }
  }

  async changePassword(currentPassword: string, newPassword: string): Promise<void> {
    await this.request('/api/auth/change-password', {
      method: 'POST',
      body: JSON.stringify({ currentPassword, newPassword }),
    });
  }

  // Users & Staff RBAC
  async listUsers(): Promise<UserAccount[]> {
    const res = await this.request<{ users: UserAccount[] }>('/api/users');
    return res.users;
  }

  async createUser(input: {
    username: string;
    password: string;
    role: UserRole;
    fullName: string;
    phone?: string;
    specialty?: string;
    licenseNo?: string;
  }): Promise<UserAccount> {
    const res = await this.request<{ user: UserAccount }>('/api/users', {
      method: 'POST',
      body: JSON.stringify(input),
    });
    return res.user;
  }

  async updateUser(id: string, patch: Partial<UserAccount>): Promise<UserAccount> {
    const res = await this.request<{ user: UserAccount }>(`/api/users/${id}`, {
      method: 'PUT',
      body: JSON.stringify(patch),
    });
    return res.user;
  }

  async deleteUser(id: string): Promise<void> {
    await this.request(`/api/users/${id}`, { method: 'DELETE' });
  }

  async resetUserPassword(id: string, newPassword: string): Promise<void> {
    await this.request(`/api/users/${id}/reset-password`, {
      method: 'POST',
      body: JSON.stringify({ newPassword }),
    });
  }

  // Biometric Kiosk Login & 10-Shot Enrollment
  async verifyFaceLogin(probeVector: readonly number[]): Promise<KioskFaceVerifyResponse> {
    if (this.supabase) {
      const { data, error } = await this.supabase.rpc('match_patient_face', {
        query_embedding: Array.from(probeVector),
        match_count: 5,
      });
      if (!error && Array.isArray(data) && data.length > 0) {
        const candidates: EnrolledTemplateCandidate[] = data
          .filter((row: Record<string, unknown>) => Array.isArray(row.embedding))
          .map((row: Record<string, unknown>) => ({
            patientId: String(row.patient_id),
            patientName: typeof row.full_name === 'string' ? row.full_name : undefined,
            pose: (row.pose as BiometricPose) || 'frontal_centroid',
            vector: row.embedding as number[],
          }));
        const verified = verifyHybridBiometricIdentity(probeVector, candidates, {
          totalEnrolledPatients: new Set(candidates.map((c) => c.patientId)).size || 1,
        });
        if (!verified.matched) return verified;
      }
    }

    const res = await fetch(`${this.baseUrl}/api/biometric/verify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'x-device-id': this.deviceId },
      body: JSON.stringify({ probeVector }),
    });
    const body = (await res.json()) as KioskFaceVerifyResponse;
    return body;
  }

  async enrollPatient10Shot(input: {
    hn: string;
    fullName: string;
    age: number;
    gender?: string;
    phone?: string;
    diagnosis: string;
    assignedPtId?: string;
    nextAppointmentDate?: string;
    username?: string;
    password?: string;
    skipFaceScan?: boolean;
    faceBiometricStatus?: 'PENDING' | 'VERIFIED';
    faceImageUrl?: string;
    frontalCentroid?: number[];
    overallCentroid?: number[];
    rawFrames?: { pose: 'front' | 'left' | 'right' | 'nod'; vector: number[]; qualityScore: number }[];
  }): Promise<PatientRecord> {
    const res = await this.request<{ patient: PatientRecord }>('/api/biometric/enroll', {
      method: 'POST',
      body: JSON.stringify(input),
    });
    return res.patient;
  }

  async enrollExistingPatientFace(
    patientId: string,
    input: {
      frontalCentroid: number[];
      overallCentroid: number[];
      rawFrames: { pose: 'front' | 'left' | 'right' | 'nod'; vector: number[]; qualityScore: number }[];
      faceImageUrl?: string;
    },
  ): Promise<PatientRecord> {
    const res = await this.request<{ patient: PatientRecord }>(`/api/patients/${patientId}/enroll-face`, {
      method: 'POST',
      body: JSON.stringify(input),
    });
    return res.patient;
  }

  async quickCheckInPatient(
    patientId: string,
    note?: string,
  ): Promise<{ patient: PatientRecord; session: RehabSessionRecord }> {
    return this.request<{ patient: PatientRecord; session: RehabSessionRecord }>(
      `/api/patients/${patientId}/checkin`,
      {
        method: 'POST',
        body: JSON.stringify({ note }),
      },
    );
  }

  // Patients, Delegation, Notes
  async listPatients(): Promise<PatientRecord[]> {
    const res = await this.request<{ patients: PatientRecord[] }>('/api/patients');
    return res.patients;
  }

  async updatePatient(id: string, patch: Partial<PatientRecord>): Promise<PatientRecord> {
    const res = await this.request<{ patient: PatientRecord }>(`/api/patients/${id}`, {
      method: 'PUT',
      body: JSON.stringify(patch),
    });
    return res.patient;
  }

  async deletePatientCascading(patientId: string): Promise<void> {
    await this.request(`/api/patients/${patientId}`, { method: 'DELETE' });
    purgeLegacyBiometricCache();
  }

  async delegatePatientCase(patientId: string, targetPtId: string | null, note?: string): Promise<PatientRecord> {
    const res = await this.request<{ patient: PatientRecord }>(`/api/patients/${patientId}/delegate`, {
      method: 'POST',
      body: JSON.stringify({ targetPtId, note }),
    });
    return res.patient;
  }

  async listMedicalNotes(patientId: string): Promise<MedicalNoteRecord[]> {
    const res = await this.request<{ notes: MedicalNoteRecord[] }>(`/api/patients/${patientId}/notes`);
    return res.notes;
  }

  async appendMedicalNote(patientId: string, content: string): Promise<MedicalNoteRecord> {
    const res = await this.request<{ note: MedicalNoteRecord }>(`/api/patients/${patientId}/notes`, {
      method: 'POST',
      body: JSON.stringify({ content }),
    });
    return res.note;
  }

  // Courses, Sessions, Reports
  async getPatientCourse(patientId: string): Promise<PatientCourseRecord | null> {
    try {
      const res = await this.request<{ course: PatientCourseRecord }>(`/api/courses/${patientId}`);
      return res.course;
    } catch {
      return null;
    }
  }

  async upsertPatientCourse(course: Omit<PatientCourseRecord, 'id' | 'updatedAt'>): Promise<PatientCourseRecord> {
    const res = await this.request<{ course: PatientCourseRecord }>(`/api/courses/${course.patientId}`, {
      method: 'PUT',
      body: JSON.stringify(course),
    });
    return res.course;
  }

  async listRehabSessions(patientId?: string): Promise<RehabSessionRecord[]> {
    const q = patientId ? `?patientId=${encodeURIComponent(patientId)}` : '';
    const res = await this.request<{ sessions: RehabSessionRecord[] }>(`/api/sessions${q}`);
    return res.sessions;
  }

  async recordRehabSession(session: Omit<RehabSessionRecord, 'id' | 'createdAt'>): Promise<RehabSessionRecord> {
    const res = await this.request<{ session: RehabSessionRecord }>('/api/sessions', {
      method: 'POST',
      body: JSON.stringify(session),
    });
    return res.session;
  }

  async listTreatmentReports(): Promise<TreatmentReportRecord[]> {
    const res = await this.request<{ reports: TreatmentReportRecord[] }>('/api/reports');
    return res.reports;
  }

  async createTreatmentReport(input: {
    patientId: string;
    summary: string;
    aiRecommendation: string;
    scoreSummary: string;
  }): Promise<TreatmentReportRecord> {
    const res = await this.request<{ report: TreatmentReportRecord }>('/api/reports', {
      method: 'POST',
      body: JSON.stringify(input),
    });
    return res.report;
  }

  async updateTreatmentReport(id: string, patch: Partial<TreatmentReportRecord>): Promise<TreatmentReportRecord> {
    const res = await this.request<{ report: TreatmentReportRecord }>(`/api/reports/${id}`, {
      method: 'PUT',
      body: JSON.stringify(patch),
    });
    return res.report;
  }

  // Exercises & Modpacks
  async listExercises(): Promise<ManagedExerciseRecord[]> {
    const res = await this.request<{ exercises: ManagedExerciseRecord[] }>('/api/exercises');
    return res.exercises;
  }

  async createExercise(ex: ExerciseConfig): Promise<ManagedExerciseRecord> {
    const res = await this.request<{ exercise: ManagedExerciseRecord }>('/api/exercises', {
      method: 'POST',
      body: JSON.stringify(ex),
    });
    return res.exercise;
  }

  async deleteExercise(id: string): Promise<void> {
    await this.request(`/api/exercises/${id}`, { method: 'DELETE' });
  }

  async listModpacks(): Promise<ModpackRecord[]> {
    const res = await this.request<{ modpacks: ModpackRecord[] }>('/api/modpacks');
    return res.modpacks;
  }

  async createModpack(input: { name: string; categories: string[]; questionCount: number }): Promise<ModpackRecord> {
    const res = await this.request<{ modpack: ModpackRecord }>('/api/modpacks', {
      method: 'POST',
      body: JSON.stringify(input),
    });
    return res.modpack;
  }

  async deleteModpack(id: string): Promise<void> {
    await this.request(`/api/modpacks/${id}`, { method: 'DELETE' });
  }

  // Staff Calls & Security & Settings
  async createStaffCall(kioskId: string, reason: string, patientId?: string | null): Promise<StaffCallRecord> {
    const res = await this.request<{ call: StaffCallRecord }>('/api/staff-calls', {
      method: 'POST',
      body: JSON.stringify({ kioskId, reason, patientId }),
    });
    return res.call;
  }

  async listStaffCalls(): Promise<StaffCallRecord[]> {
    const res = await this.request<{ calls: StaffCallRecord[] }>('/api/staff-calls');
    return res.calls;
  }

  async acknowledgeStaffCall(callId: string): Promise<void> {
    await this.request(`/api/staff-calls/${callId}/acknowledge`, { method: 'PATCH' });
  }

  async resolveStaffCall(callId: string): Promise<void> {
    await this.request(`/api/staff-calls/${callId}/resolve`, { method: 'PATCH' });
  }

  async registerKiosk(input: { kioskId: string; location: string }): Promise<KioskStatusRecord> {
    const res = await this.request<{ kiosk: KioskStatusRecord }>('/api/kiosks', {
      method: 'POST',
      body: JSON.stringify(input),
    });
    return res.kiosk;
  }

  async deleteKiosk(kioskId: string): Promise<void> {
    await this.request(`/api/kiosks/${encodeURIComponent(kioskId)}`, { method: 'DELETE' });
  }

  async listSecurityAudit(): Promise<{
    logs: AuditLogRecord[];
    bannedTargets: BannedTargetRecord[];
    activeSessions: ActiveSessionRecord[];
  }> {
    return this.request('/api/audit-logs');
  }

  async banTarget(targetType: 'ip' | 'device', value: string, reason: string): Promise<BannedTargetRecord> {
    const res = await this.request<{ ban: BannedTargetRecord }>('/api/security/ban', {
      method: 'POST',
      body: JSON.stringify({ targetType, value, reason }),
    });
    return res.ban;
  }

  async unbanTarget(banId: string): Promise<void> {
    await this.request(`/api/security/ban/${banId}`, { method: 'DELETE' });
  }

  async forceLogoutUser(userId: string): Promise<void> {
    await this.request(`/api/security/force-logout/${userId}`, { method: 'POST' });
  }

  async getSystemSettings(): Promise<{ settings: SystemSettingsRecord; kiosks: KioskStatusRecord[] }> {
    return this.request('/api/settings');
  }

  async updateSystemSettings(patch: Partial<SystemSettingsRecord>): Promise<SystemSettingsRecord> {
    const res = await this.request<{ settings: SystemSettingsRecord }>('/api/settings', {
      method: 'PUT',
      body: JSON.stringify(patch),
    });
    return res.settings;
  }
}

import { createCloudRouter } from './cloudRouter';

export { createCloudRouter };

export function createCloudDelegatedRouter(config: CloudClientConfig = {}) {
  return createCloudRouter(config);
}
