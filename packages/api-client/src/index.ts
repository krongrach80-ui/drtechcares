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
  avatarUrl?: string;
  bio?: string;
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

export interface TodayPrescribedExerciseItem {
  id: string;
  exerciseId: string;
  nameTh: string;
  nameEn: string;
  targetJoint: string;
  activityType: ActivityType;
  sets: number;
  reps: number;
  holdSec: number;
  status: 'ACTIVE' | 'COMPLETED';
  notes?: string;
}

export interface AngleEmbeddings5D {
  center: number[];
  left: number[];
  right: number[];
  up: number[];
  down: number[];
}

export interface PatientBiometricIndexEntry {
  patientId: string;
  hn: string;
  fullName: string;
  diagnosis: string;
  assignedPtId: string;
  nextAppointmentDate: string;
  avatarUrl?: string;
  framesCaptured: number;
  qualityScore: number;
  masterEmbedding: number[];
  angleEmbeddings: AngleEmbeddings5D;
  todayRoutine?: DailyScheduleDay | null;
  rehabSchedules?: RehabScheduleItem[];
  todayExercises?: TodayPrescribedExerciseItem[];
  hasAppointmentToday?: boolean;
  completedToday?: boolean;
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
  delegationDate?: string | null;
  nextAppointmentDate: string;
  registrationStatus: 'active' | 'archived';
  enrolledAt: string;
  faceBiometricStatus?: 'PENDING' | 'VERIFIED';
  faceFramesCount?: number;
  faceImageUrl?: string;
  masterEmbedding?: number[];
  angleEmbeddings?: AngleEmbeddings5D;
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
  rehabSchedules?: RehabScheduleItem[];
  todayExercises?: TodayPrescribedExerciseItem[];
  hasAppointmentToday?: boolean;
  completedToday?: boolean;
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

const DRTECH_SESSION_STORAGE_KEY = 'drtech_session';
const DRTECH_TOKEN_STORAGE_KEY = 'drtechcares_auth_token';
const DRTECH_STAFF_CALLS_STORAGE_KEY = 'drtechcares_staff_calls_v1';
const DRTECH_DELETED_CALL_IDS_STORAGE_KEY = 'drtechcares_deleted_staff_calls_v1';
const DRTECH_CASE_DELEGATIONS_STORAGE_KEY = 'drtechcares_case_delegations_v1';

interface StoredCaseDelegationEntry {
  delegatedToPtId: string | null;
  delegationDate: string | null;
  delegationNote: string | null;
}

const STAFF_CALL_STATUS_RANK: Record<StaffCallRecord['status'], number> = {
  open: 0,
  acknowledged: 1,
  resolved: 2,
};

function mergeStaffCallLists(
  primary: readonly StaffCallRecord[],
  secondary: readonly StaffCallRecord[],
  deletedIds: ReadonlySet<string>,
): StaffCallRecord[] {
  const merged = new Map<string, StaffCallRecord>();
  const apply = (item: StaffCallRecord) => {
    if (!item?.id || deletedIds.has(item.id)) return;
    const prev = merged.get(item.id);
    if (!prev) {
      merged.set(item.id, item);
      return;
    }
    const status =
      STAFF_CALL_STATUS_RANK[item.status] >= STAFF_CALL_STATUS_RANK[prev.status]
        ? item.status
        : prev.status;
    merged.set(item.id, {
      ...prev,
      ...item,
      status,
      reason: (item.reason || prev.reason).replace(/^\[ACKNOWLEDGED\]\s*/, ''),
    });
  };
  secondary.forEach(apply);
  primary.forEach(apply);
  return Array.from(merged.values())
    .sort((a, b) => (Date.parse(b.createdAt) || 0) - (Date.parse(a.createdAt) || 0))
    .slice(0, 200);
}

export interface StaffRegisterInput {
  username: string;
  password: string;
  role: 'super_admin' | 'pt' | 'ADMIN' | 'PT';
  fullName: string;
  phone?: string;
  specialty?: string;
  licenseNo?: string;
}

export class DrTechCaresApiClient {
  private supabase: SupabaseClient | null = null;
  private authToken: string | null = null;
  private storedUser: UserAccount | null = null;
  private readonly baseUrl: string;
  private readonly deviceId: string;

  constructor(config: CloudClientConfig = {}) {
    purgeLegacyBiometricCache();
    this.baseUrl = config.apiBaseUrl ?? '';
    this.deviceId = this.resolveDeviceId();

    const persisted = this.readPersistedSession();
    this.authToken = persisted.token;
    this.storedUser = persisted.user;

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

  private readPersistedSession(): { token: string | null; user: UserAccount | null } {
    let token: string | null = null;
    let user: UserAccount | null = null;
    try {
      if (typeof globalThis.localStorage !== 'undefined') {
        const rawSession = globalThis.localStorage.getItem(DRTECH_SESSION_STORAGE_KEY);
        if (rawSession) {
          const parsed = JSON.parse(rawSession) as { token?: string; user?: UserAccount } | null;
          if (parsed?.token) {
            token = parsed.token;
            user = parsed.user ?? null;
          }
        }
        if (!token) {
          token = globalThis.localStorage.getItem(DRTECH_TOKEN_STORAGE_KEY);
        }
      }
      if (!token && typeof globalThis.sessionStorage !== 'undefined') {
        token = globalThis.sessionStorage.getItem(DRTECH_TOKEN_STORAGE_KEY);
      }
    } catch {
      // Ignore storage access or malformed JSON errors
    }
    return { token, user };
  }

  /** Persist both Bearer token and authenticated user across localStorage and sessionStorage. */
  setSession(token: string | null, user: UserAccount | null = null): void {
    this.authToken = token;
    this.storedUser = token ? (user ?? this.storedUser) : null;
    try {
      if (typeof globalThis.sessionStorage !== 'undefined') {
        if (token) globalThis.sessionStorage.setItem(DRTECH_TOKEN_STORAGE_KEY, token);
        else globalThis.sessionStorage.removeItem(DRTECH_TOKEN_STORAGE_KEY);
      }
      if (typeof globalThis.localStorage !== 'undefined') {
        if (token) {
          globalThis.localStorage.setItem(DRTECH_TOKEN_STORAGE_KEY, token);
          globalThis.localStorage.setItem(
            DRTECH_SESSION_STORAGE_KEY,
            JSON.stringify({ token, user: this.storedUser }),
          );
        } else {
          globalThis.localStorage.removeItem(DRTECH_TOKEN_STORAGE_KEY);
          globalThis.localStorage.removeItem(DRTECH_SESSION_STORAGE_KEY);
        }
      }
    } catch {
      // Ignore storage quota or security restrictions
    }
  }

  setAuthToken(token: string | null): void {
    this.setSession(token, token ? this.storedUser : null);
  }

  getAuthToken(): string | null {
    if (!this.authToken) {
      const persisted = this.readPersistedSession();
      this.authToken = persisted.token;
      if (!this.storedUser && persisted.user) {
        this.storedUser = persisted.user;
      }
    }
    return this.authToken;
  }

  /** Return the cached authenticated user if an active token is present. */
  getStoredUser(): UserAccount | null {
    const token = this.getAuthToken();
    if (!token) return null;
    if (!this.storedUser) {
      this.storedUser = this.readPersistedSession().user;
    }
    return this.storedUser;
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
    const activeToken = this.getAuthToken();
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'x-device-id': this.deviceId,
      ...(init.headers as Record<string, string> | undefined),
    };
    if (activeToken) {
      headers.Authorization = `Bearer ${activeToken}`;
      headers['X-Staff-Token'] = activeToken;
    }
    const res = await fetch(`${this.baseUrl}${path}`, { ...init, headers });
    const text = await res.text();
    const data = text ? (JSON.parse(text) as Record<string, unknown>) : {};
    if (!res.ok) {
      if (res.status === 401 && path === '/api/auth/me') {
        this.setSession(null, null);
        if (typeof globalThis.dispatchEvent === 'function' && typeof globalThis.CustomEvent === 'function') {
          globalThis.dispatchEvent(new globalThis.CustomEvent('drtech:auth-expired'));
        }
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
    this.setSession(res.token, res.user);
    return res;
  }

  /** Register a new Admin or PT staff member and persist their active session immediately. */
  async registerStaff(input: StaffRegisterInput): Promise<{ token: string; user: UserAccount }> {
    const res = await this.request<{ token: string; user: UserAccount }>('/api/auth/register', {
      method: 'POST',
      body: JSON.stringify(input),
    });
    this.setSession(res.token, res.user);
    return res;
  }

  async logout(): Promise<void> {
    try {
      await this.request('/api/auth/logout', { method: 'POST' });
    } finally {
      this.setSession(null, null);
    }
  }

  async getMe(): Promise<UserAccount | null> {
    const activeToken = this.getAuthToken();
    if (!activeToken) return null;
    try {
      const res = await this.request<{ user: UserAccount }>('/api/auth/me');
      this.setSession(activeToken, res.user);
      return res.user;
    } catch (err) {
      if (err instanceof ApiError && (err.status === 401 || err.status === 403)) {
        this.setSession(null, null);
      }
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
    try {
      const res = await fetch(`${this.baseUrl}/api/biometric/verify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'x-device-id': this.deviceId },
        body: JSON.stringify({ probeVector }),
      });
      const body = (await res.json()) as KioskFaceVerifyResponse;
      return body;
    } catch (err) {
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
          return verifyHybridBiometricIdentity(probeVector, candidates, {
            totalEnrolledPatients: new Set(candidates.map((c) => c.patientId)).size || 1,
          });
        }
      }
      throw err;
    }
  }

  async getBiometricIndex(): Promise<PatientBiometricIndexEntry[]> {
    const res = await this.request<{ index: PatientBiometricIndexEntry[] }>('/api/biometrics/index');
    return res.index ?? [];
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
    masterEmbedding?: number[];
    angleEmbeddings?: AngleEmbeddings5D;
    framesCaptured?: number;
    qualityScore?: number;
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
      masterEmbedding?: number[];
      angleEmbeddings?: AngleEmbeddings5D;
      framesCaptured?: number;
      qualityScore?: number;
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

  private readCaseDelegationOverrides(): Record<string, StoredCaseDelegationEntry> {
    try {
      if (typeof globalThis.localStorage === 'undefined') return {};
      const raw = globalThis.localStorage.getItem(DRTECH_CASE_DELEGATIONS_STORAGE_KEY);
      if (!raw) return {};
      const parsed = JSON.parse(raw) as unknown;
      return parsed && typeof parsed === 'object' ? (parsed as Record<string, StoredCaseDelegationEntry>) : {};
    } catch {
      return {};
    }
  }

  private writeCaseDelegationOverride(
    patientId: string,
    entry: StoredCaseDelegationEntry | null,
  ): void {
    try {
      if (typeof globalThis.localStorage === 'undefined') return;
      const current = this.readCaseDelegationOverrides();
      if (entry && entry.delegatedToPtId) {
        current[patientId] = entry;
      } else {
        delete current[patientId];
      }
      globalThis.localStorage.setItem(DRTECH_CASE_DELEGATIONS_STORAGE_KEY, JSON.stringify(current));
    } catch {
      // ignore storage quota errors
    }
  }

  // Patients, Delegation, Notes
  async listPatients(): Promise<PatientRecord[]> {
    const res = await this.request<{ patients: PatientRecord[] }>('/api/patients');
    const overrides = this.readCaseDelegationOverrides();
    return res.patients.map((p) => {
      const ov = overrides[p.id];
      if (ov && ov.delegatedToPtId) {
        return {
          ...p,
          delegatedToPtId: p.delegatedToPtId || ov.delegatedToPtId,
          delegationDate: p.delegationDate || ov.delegationDate,
          delegationNote: p.delegationNote ?? ov.delegationNote,
        };
      }
      return p;
    });
  }

  async updatePatient(id: string, patch: Partial<PatientRecord>): Promise<PatientRecord> {
    const res = await this.request<{ patient: PatientRecord }>(`/api/patients/${id}`, {
      method: 'PUT',
      body: JSON.stringify(patch),
    });
    return res.patient;
  }

  async deletePatientCascading(patientId: string): Promise<void> {
    this.writeCaseDelegationOverride(patientId, null);
    await this.request(`/api/patients/${patientId}`, { method: 'DELETE' });
    purgeLegacyBiometricCache();
  }

  async delegatePatientCase(
    patientId: string,
    targetPtId: string | null,
    note?: string,
    delegationDate?: string,
  ): Promise<PatientRecord> {
    const res = await this.request<{ patient: PatientRecord }>(`/api/patients/${patientId}/delegate`, {
      method: 'POST',
      body: JSON.stringify({ targetPtId, note, delegationDate }),
    });
    if (targetPtId) {
      this.writeCaseDelegationOverride(patientId, {
        delegatedToPtId: res.patient.delegatedToPtId ?? targetPtId,
        delegationDate: res.patient.delegationDate ?? delegationDate ?? new Date().toISOString().slice(0, 10),
        delegationNote: res.patient.delegationNote ?? (note || null),
      });
    } else {
      this.writeCaseDelegationOverride(patientId, null);
    }
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

  async recordRehabSession(
    session: Omit<RehabSessionRecord, 'id' | 'createdAt'> & {
      scheduledDate?: string;
      markDailyCompleted?: boolean;
    },
  ): Promise<RehabSessionRecord> {
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

  private readLocalStaffCalls(): StaffCallRecord[] {
    try {
      if (typeof globalThis.localStorage === 'undefined') return [];
      const raw = globalThis.localStorage.getItem(DRTECH_STAFF_CALLS_STORAGE_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw) as unknown;
      return Array.isArray(parsed) ? (parsed as StaffCallRecord[]) : [];
    } catch {
      return [];
    }
  }

  private writeLocalStaffCalls(calls: StaffCallRecord[]): void {
    try {
      if (typeof globalThis.localStorage === 'undefined') return;
      globalThis.localStorage.setItem(DRTECH_STAFF_CALLS_STORAGE_KEY, JSON.stringify(calls.slice(0, 200)));
    } catch {
      // ignore storage quota errors
    }
  }

  private readDeletedStaffCallIds(): Set<string> {
    try {
      if (typeof globalThis.localStorage === 'undefined') return new Set();
      const raw = globalThis.localStorage.getItem(DRTECH_DELETED_CALL_IDS_STORAGE_KEY);
      if (!raw) return new Set();
      const parsed = JSON.parse(raw) as unknown;
      return Array.isArray(parsed) ? new Set(parsed.map(String)) : new Set();
    } catch {
      return new Set();
    }
  }

  private writeDeletedStaffCallId(callId: string, deleted: boolean): void {
    try {
      if (typeof globalThis.localStorage === 'undefined') return;
      const current = this.readDeletedStaffCallIds();
      if (deleted) current.add(callId);
      else current.delete(callId);
      globalThis.localStorage.setItem(
        DRTECH_DELETED_CALL_IDS_STORAGE_KEY,
        JSON.stringify(Array.from(current).slice(-300)),
      );
    } catch {
      // ignore storage quota errors
    }
  }

  // Staff Calls & Security & Settings
  async createStaffCall(kioskId: string, reason: string, patientId?: string | null): Promise<StaffCallRecord> {
    let created: StaffCallRecord;
    try {
      const res = await this.request<{ call: StaffCallRecord }>('/api/staff-calls', {
        method: 'POST',
        body: JSON.stringify({ kioskId, reason, patientId }),
      });
      created = res.call;
    } catch (err) {
      if (err instanceof ApiError && err.status < 500) {
        throw err;
      }
      created = {
        id: `call-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`,
        kioskId: kioskId || 'KIOSK-OPD-01',
        patientId: patientId ?? null,
        reason: reason.trim(),
        status: 'open',
        createdAt: new Date().toISOString(),
      };
    }
    this.writeDeletedStaffCallId(created.id, false);
    const merged = mergeStaffCallLists([created], this.readLocalStaffCalls(), this.readDeletedStaffCallIds());
    this.writeLocalStaffCalls(merged);
    return created;
  }

  async listStaffCalls(): Promise<StaffCallRecord[]> {
    const deleted = this.readDeletedStaffCallIds();
    const localCalls = this.readLocalStaffCalls();
    try {
      const res = await this.request<{ calls: StaffCallRecord[] }>('/api/staff-calls');
      const merged = mergeStaffCallLists(res.calls, localCalls, deleted);
      this.writeLocalStaffCalls(merged);
      return merged;
    } catch (err) {
      if (err instanceof ApiError && (err.status === 401 || err.status === 403)) {
        throw err;
      }
      return mergeStaffCallLists([], localCalls, deleted);
    }
  }

  async acknowledgeStaffCall(callId: string): Promise<void> {
    const updatedLocal = this.readLocalStaffCalls().map((c) =>
      c.id === callId && c.status === 'open' ? { ...c, status: 'acknowledged' as const } : c,
    );
    this.writeLocalStaffCalls(updatedLocal);
    await this.request(`/api/staff-calls/${encodeURIComponent(callId)}/acknowledge`, { method: 'PATCH' });
  }

  async resolveStaffCall(callId: string): Promise<void> {
    const updatedLocal = this.readLocalStaffCalls().map((c) =>
      c.id === callId ? { ...c, status: 'resolved' as const } : c,
    );
    this.writeLocalStaffCalls(updatedLocal);
    await this.request(`/api/staff-calls/${encodeURIComponent(callId)}/resolve`, { method: 'PATCH' });
  }

  async deleteStaffCall(callId: string): Promise<void> {
    this.writeDeletedStaffCallId(callId, true);
    this.writeLocalStaffCalls(this.readLocalStaffCalls().filter((c) => c.id !== callId));
    try {
      await this.request(`/api/staff-calls/${encodeURIComponent(callId)}`, { method: 'DELETE' });
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) return;
      throw err;
    }
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

const JOINT_GROUP_LABELS: Record<ExerciseConfig['detection']['jointGroup'], string> = {
  shoulder: 'ข้อไหล่ (Shoulder)',
  elbow: 'ข้อศอก (Elbow)',
  knee: 'ข้อเข่า (Knee)',
  hip: 'ข้อสะโพก (Hip)',
  ankle: 'ข้อเท้า (Ankle)',
  neck: 'ลำคอ (Neck)',
  trunk: 'ลำตัว (Trunk)',
  balance: 'การทรงตัว (Balance)',
};

/** Formats a Date object as YYYY-MM-DD in the local calendar timezone. */
export function formatLocalIsoDate(date: Date = new Date()): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

function buildExerciseSummary(
  exerciseId: string,
  catalog: readonly ExerciseConfig[],
  fallbackById?: Map<string, TodayPrescribedExerciseItem>,
  modpackId?: string,
): { nameTh: string; nameEn: string; targetJoint: string; holdSec: number; activityType: ActivityType } {
  const found = catalog.find((c) => c.id === exerciseId);
  if (found) {
    return {
      nameTh: found.name.th,
      nameEn: found.name.en,
      targetJoint: JOINT_GROUP_LABELS[found.detection.jointGroup] ?? found.detection.jointGroup,
      holdSec: found.detection.holdSec ?? 3,
      activityType:
        exerciseId.includes('card') || exerciseId.includes('minigame') ? 'minigame' : 'physio_exercise',
    };
  }
  const existingMeta = fallbackById?.get(exerciseId);
  if (existingMeta) {
    return {
      nameTh: existingMeta.nameTh,
      nameEn: existingMeta.nameEn,
      targetJoint: existingMeta.targetJoint,
      holdSec: existingMeta.holdSec,
      activityType: existingMeta.activityType,
    };
  }
  const isMinigame =
    !exerciseId ||
    exerciseId.includes('card') ||
    exerciseId.includes('minigame') ||
    Boolean(modpackId && !exerciseId);
  return {
    nameTh: isMinigame ? 'มินิเกมฝึกสมอง (การ์ดกายภาพ)' : exerciseId || 'ท่ากายภาพบำบัดตามนัด',
    nameEn: isMinigame ? 'Physical Therapy Card Mini-Game' : exerciseId || 'Prescribed Rehab Exercise',
    targetJoint: isMinigame ? 'สมองและระบบประสาทสั่งการ (Cognitive & Hand)' : 'ร่างกายส่วนที่ฟื้นฟู (Targeted Rehab)',
    holdSec: 3,
    activityType: isMinigame ? 'minigame' : 'physio_exercise',
  };
}

export const DAILY_REHAB_DONE_STORAGE_KEY = 'drtechcares_daily_rehab_done_v1';

function readDailyRehabDoneMap(): Record<string, string> {
  if (typeof globalThis.window === 'undefined') return {};
  try {
    const raw = globalThis.window.localStorage.getItem(DAILY_REHAB_DONE_STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as unknown;
    return parsed && typeof parsed === 'object' ? (parsed as Record<string, string>) : {};
  } catch {
    return {};
  }
}

export function isPatientCompletedTodayInStorage(
  patientKey: string,
  todayIso: string = formatLocalIsoDate(),
): boolean {
  if (!patientKey) return false;
  const map = readDailyRehabDoneMap();
  return map[patientKey] === todayIso;
}

export function markPatientCompletedTodayInStorage(
  patientKeys: readonly string[],
  todayIso: string = formatLocalIsoDate(),
): void {
  if (typeof globalThis.window === 'undefined') return;
  try {
    const map = readDailyRehabDoneMap();
    for (const key of patientKeys) {
      const trimmed = key.trim();
      if (trimmed) {
        map[trimmed] = todayIso;
      }
    }
    globalThis.window.localStorage.setItem(DAILY_REHAB_DONE_STORAGE_KEY, JSON.stringify(map));
  } catch {
    // Ignore storage quota or private mode restrictions
  }
}

function isSameCalendarDate(isoOrTimestamp: string | undefined, targetIso: string): boolean {
  if (!isoOrTimestamp) return false;
  if (isoOrTimestamp.slice(0, 10) === targetIso) return true;
  const parsed = new Date(isoOrTimestamp);
  if (Number.isNaN(parsed.getTime())) return false;
  return formatLocalIsoDate(parsed) === targetIso;
}

/**
 * Resolves whether a patient has an appointment today (`todayIso`) and the list of
 * prescribed exercises/poses to perform today. Explicit `rehabSchedules` entries take
 * precedence over generic weekly course templates so rest days are never misreported.
 */
export function resolveTodayPatientSchedule(input: {
  rehabSchedules?: readonly RehabScheduleItem[];
  rehabSessions?: readonly RehabSessionRecord[];
  todayRoutine?: DailyScheduleDay | null;
  nextAppointmentDate?: string;
  exercisesCatalog?: readonly ExerciseConfig[];
  serverTodayExercises?: readonly TodayPrescribedExerciseItem[];
  completedTodayOverride?: boolean;
  now?: Date;
}): {
  todayIso: string;
  hasAppointmentToday: boolean;
  completedToday: boolean;
  todayExercises: TodayPrescribedExerciseItem[];
} {
  const todayIso = formatLocalIsoDate(input.now ?? new Date());
  const catalog =
    input.exercisesCatalog && input.exercisesCatalog.length > 0 ? input.exercisesCatalog : CLINICAL_EXERCISES;
  const fallbackById = new Map<string, TodayPrescribedExerciseItem>();
  for (const item of input.serverTodayExercises ?? []) {
    fallbackById.set(item.exerciseId, item);
  }

  const hasCompletedSessionToday = (input.rehabSessions ?? []).some(
    (s) =>
      s.exerciseId !== 'CLINIC_CHECKIN' &&
      isSameCalendarDate(s.createdAt, todayIso) &&
      (s.activityType === 'minigame' || (s.completedReps ?? 0) > 0 || (s.completedSets ?? 0) > 0),
  );

  const schedules = input.rehabSchedules ?? [];
  if (schedules.length > 0) {
    const scheduledToday = schedules.filter(
      (item) => item.scheduledDate === todayIso && (item.status === 'ACTIVE' || item.status === 'COMPLETED'),
    );
    if (scheduledToday.length === 0) {
      return {
        todayIso,
        hasAppointmentToday: false,
        completedToday: Boolean(input.completedTodayOverride || hasCompletedSessionToday),
        todayExercises: [],
      };
    }
    const allScheduledCompleted = scheduledToday.every((item) => item.status === 'COMPLETED');
    const completedToday = Boolean(
      input.completedTodayOverride || hasCompletedSessionToday || allScheduledCompleted,
    );
    const todayExercises: TodayPrescribedExerciseItem[] = scheduledToday.map((item) => {
      const exId = item.exerciseId || item.modpackId || 'minigame';
      const meta = buildExerciseSummary(exId, catalog, fallbackById, item.modpackId);
      return {
        id: item.id,
        exerciseId: exId,
        nameTh: meta.nameTh,
        nameEn: meta.nameEn,
        targetJoint: meta.targetJoint,
        activityType: meta.activityType,
        sets: item.sets,
        reps: item.reps,
        holdSec: meta.holdSec,
        status: completedToday || item.status === 'COMPLETED' ? 'COMPLETED' : 'ACTIVE',
        notes: item.notes,
      };
    });
    return { todayIso, hasAppointmentToday: true, completedToday, todayExercises };
  }

  const completedToday = Boolean(input.completedTodayOverride || hasCompletedSessionToday);
  const routine = input.todayRoutine;
  if (routine?.dayType === 'training' && (routine.exercises ?? []).length > 0) {
    const todayExercises: TodayPrescribedExerciseItem[] = routine.exercises.map((item, idx) => {
      const meta = buildExerciseSummary(item.exerciseId, catalog, fallbackById);
      return {
        id: `routine-${routine.dayIndex}-${item.exerciseId}-${idx}`,
        exerciseId: item.exerciseId,
        nameTh: meta.nameTh,
        nameEn: meta.nameEn,
        targetJoint: meta.targetJoint,
        activityType: meta.activityType,
        sets: item.sets,
        reps: item.reps,
        holdSec: item.holdSec ?? meta.holdSec,
        status: completedToday ? 'COMPLETED' : 'ACTIVE',
      };
    });
    return { todayIso, hasAppointmentToday: true, completedToday, todayExercises };
  }

  if (routine?.dayType === 'training' && input.nextAppointmentDate === todayIso) {
    return {
      todayIso,
      hasAppointmentToday: true,
      completedToday,
      todayExercises: [
        {
          id: 'routine-today-default',
          exerciseId: 'ex-card-01',
          nameTh: routine.label || 'มินิเกมฝึกสมอง (การ์ดกายภาพ)',
          nameEn: 'Physical Therapy Card Mini-Game',
          targetJoint: 'สมองและระบบประสาทสั่งการ (Cognitive & Hand)',
          activityType: routine.activityType ?? 'minigame',
          sets: 1,
          reps: 10,
          holdSec: 3,
          status: completedToday ? 'COMPLETED' : 'ACTIVE',
        },
      ],
    };
  }

  return { todayIso, hasAppointmentToday: false, completedToday, todayExercises: [] };
}

import { createCloudRouter } from './cloudRouter';
export {
  buildUpstreamTtsUrls,
  clampTtsText,
  normalizeTtsLang,
  synthesizeTtsAudio,
  type SupportedTtsLang,
  type SynthesizedTtsAudio,
} from './ttsProxy';

export { createCloudRouter };

export function createCloudDelegatedRouter(config: CloudClientConfig = {}) {
  return createCloudRouter(config);
}
