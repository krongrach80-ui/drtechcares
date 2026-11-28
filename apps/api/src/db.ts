import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import {
  CLINICAL_EXERCISES,
  ELDERLY_CANONICAL_EXERCISE_SET,
  isDeprecatedExerciseRecord,
  type BodySide,
  type ExerciseConfig,
} from '@drtechcares/pose-engine';
import type { EnrolledTemplateCandidate } from '@drtechcares/smart-biometric-core';

export type UserRole = 'super_admin' | 'pt' | 'patient';

export interface UserAccount {
  id: string;
  username: string;
  passwordHash: string;
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

export interface PatientBiometricHistory {
  frontal_centroid: number[];
  overall_centroid: number[];
  raw_frames: { pose: 'front' | 'left' | 'right' | 'nod'; vector: number[]; qualityScore: number }[];
}

export interface PatientRecord {
  id: string;
  hn: string;
  fullName: string;
  age: number;
  phone: string;
  diagnosis: string;
  assignedPtId: string;
  delegatedToPtId: string | null;
  delegationNote: string | null;
  delegationDate?: string | null;
  nextAppointmentDate: string;
  registrationStatus: 'active' | 'archived';
  enrolledAt: string;
  biometricHistory?: PatientBiometricHistory;
}

export interface MedicalNoteRecord {
  id: string;
  patientId: string;
  authorId: string;
  authorName: string;
  content: string;
  createdAt: string;
}

export interface ClinicalAssessmentRecord {
  id: string;
  patient_id: string;
  patient_hn: string;
  universal_link_code: string;
  visit_number: number;
  assessed_at: string;
  facility_name: string;
  assessor_staff_id: string;
  assessor_staff_name: string;
  exercise_code: string;
  exercise_name_th: string;
  target_side: 'LEFT' | 'RIGHT' | 'BOTH';
  initial_score_0_to_6: 0 | 1 | 2 | 3 | 4 | 5 | 6;
  measured_rom_deg: number;
  target_rom_deg: number;
  pain_vas_0_to_10?: number | null;
  chief_complaint: string;
  clinical_handover_note: string;
  next_session_goal: string;
  source_channel: 'STAFF_MANUAL_ASSESSMENT' | 'KIOSK_SESSION_AUTO_SYNC';
  measurement_method?: 'MANUAL_INPUT' | 'CAMERA_AI_LIVE' | 'KIOSK_SESSION_AUTO_SYNC';
  completed_reps?: number | null;
  accuracy_percent?: number | null;
  session_id?: string | null;
  created_at?: string;
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

export type ActivityType = 'minigame' | 'physio_exercise';

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
  createdBy: string; // 'system' | userId
}

export interface ModpackRecord {
  id: string;
  name: string;
  categories: string[];
  questionCount: number;
  createdBy: string; // 'system' | userId
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
}

export interface DatabaseState {
  users: UserAccount[];
  patients: PatientRecord[];
  face_templates: EnrolledTemplateCandidate[];
  medical_notes: MedicalNoteRecord[];
  clinical_assessments?: ClinicalAssessmentRecord[];
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

export function hashPassword(password: string): string {
  const salt = crypto.randomBytes(16).toString('hex');
  const derived = crypto.scryptSync(password, salt, 64).toString('hex');
  return `${salt}:${derived}`;
}

export function verifyPassword(password: string, storedHash: string): boolean {
  const parts = storedHash.split(':');
  if (parts.length !== 2) return false;
  const [salt, originalHex] = parts;
  const derived = crypto.scryptSync(password, salt, 64);
  const original = Buffer.from(originalHex, 'hex');
  if (derived.length !== original.length) return false;
  return crypto.timingSafeEqual(derived, original);
}

function createInitialState(): DatabaseState {
  const now = new Date().toISOString();
  return {
    users: [
      {
        id: 'user-super-1',
        username: 'admin',
        passwordHash: hashPassword('Admin@2026'),
        role: 'super_admin',
        fullName: 'นพ.สมชาย ผู้อำนวยการโรงพยาบาล (Super Admin)',
        phone: '02-000-0001',
        specialty: 'Hospital Director',
        licenseNo: 'MD-10001',
        active: true,
        forceLoggedOutAt: null,
        createdAt: now,
      },
      {
        id: 'user-pt-1',
        username: 'pt_somchai',
        passwordHash: hashPassword('Pt@2026'),
        role: 'pt',
        fullName: 'กภ.สมชาย ใจดี (PT)',
        phone: '081-111-2233',
        specialty: 'Orthopedic & Shoulder Rehab',
        licenseNo: 'PT-44012',
        active: true,
        forceLoggedOutAt: null,
        createdAt: now,
      },
      {
        id: 'user-pt-2',
        username: 'pt_waranya',
        passwordHash: hashPassword('Pt@2026'),
        role: 'pt',
        fullName: 'กภ.วรัญญา สุขสันต์ (PT)',
        phone: '089-222-3344',
        specialty: 'Neuro & Cognitive Rehab',
        licenseNo: 'PT-44089',
        active: true,
        forceLoggedOutAt: null,
        createdAt: now,
      },
    ],
    patients: [],
    face_templates: [],
    medical_notes: [],
    patient_courses: [],
    rehab_sessions: [],
    treatment_reports: [],
    exercises: CLINICAL_EXERCISES.map((ex) => ({ ...ex, createdBy: 'system' })),
    modpacks: [
      {
        id: 'modpack-core-99',
        name: 'Clinical Cognitive Core (99 Cards)',
        categories: ['food_fruit', 'objects', 'animals', 'counting', 'safety_routine', 'color_shape_time', 'position'],
        questionCount: 99,
        createdBy: 'system',
        active: true,
      },
      {
        id: 'modpack-memory-safety',
        name: 'Fall Prevention & Daily Safety Pack',
        categories: ['safety_routine', 'counting', 'position'],
        questionCount: 42,
        createdBy: 'system',
        active: true,
      },
    ],
    staff_calls: [],
    audit_logs: [],
    banned_targets: [],
    active_sessions: [],
    clinical_assessments: [],
    system_settings: {
      aiFaceScanEnabled: true,
      aiFaceThreshold: 0.925,
      aiPoseDetectionEnabled: false, // Default locked per diagram 2 & 3 until Super Admin enables
      kioskAutoResetSec: 6,
    },
    kiosks: [
      {
        kioskId: 'KIOSK-OPD-01',
        location: 'OPD Rehab Zone A',
        online: true,
        currentState: 'STANDBY',
        lastHeartbeatAt: now,
      },
    ],
  };
}

export class ClinicalDatabase {
  public state: DatabaseState;
  private readonly filePath: string | null;

  constructor(options: { filePath?: string | null; persistToDisk?: boolean } = {}) {
    const shouldPersist = options.persistToDisk ?? true;
    this.filePath = shouldPersist
      ? (options.filePath ?? path.resolve(process.cwd(), 'data', 'drtechcares-db.json'))
      : null;

    if (this.filePath && fs.existsSync(this.filePath)) {
      try {
        const raw = fs.readFileSync(this.filePath, 'utf-8');
        this.state = JSON.parse(raw) as DatabaseState;
        if (!Array.isArray(this.state.clinical_assessments)) {
          this.state.clinical_assessments = [];
        }
        this.syncElderlyExerciseLibrary();
      } catch {
        this.state = createInitialState();
        this.save();
      }
    } else {
      this.state = createInitialState();
      this.save();
    }
  }

  /**
   * Purges deprecated leg/orthogonal/legacy ex-01..ex-40 exercises from local DB state
   * and upserts all 26 canonical Elderly Care & MOPH upper-body exercises.
   */
  syncElderlyExerciseLibrary(): { exercises: ManagedExerciseRecord[]; purgedCount: number } {
    const existing = Array.isArray(this.state.exercises) ? this.state.exercises : [];
    const nonDeprecatedCustom = existing.filter(
      (ex) =>
        !isDeprecatedExerciseRecord(ex) &&
        !ELDERLY_CANONICAL_EXERCISE_SET.has(ex.code) &&
        !ELDERLY_CANONICAL_EXERCISE_SET.has(ex.id),
    );
    const purgedCount = existing.filter((ex) => isDeprecatedExerciseRecord(ex)).length;
    const canonical: ManagedExerciseRecord[] = CLINICAL_EXERCISES.map((ex) => ({
      ...ex,
      createdBy: 'system',
    }));
    this.state.exercises = [...canonical, ...nonDeprecatedCustom];
    this.save();
    return { exercises: this.state.exercises, purgedCount };
  }

  save(): void {
    if (!this.filePath) return;
    const dir = path.dirname(this.filePath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    const tmpPath = `${this.filePath}.tmp`;
    fs.writeFileSync(tmpPath, JSON.stringify(this.state, null, 2), 'utf-8');
    fs.renameSync(tmpPath, this.filePath);
  }

  recordAudit(entry: Omit<AuditLogRecord, 'id' | 'createdAt'>): AuditLogRecord {
    const log: AuditLogRecord = {
      id: `audit-${Date.now()}-${crypto.randomBytes(3).toString('hex')}`,
      ...entry,
      createdAt: new Date().toISOString(),
    };
    this.state.audit_logs.unshift(log);
    if (this.state.audit_logs.length > 1000) {
      this.state.audit_logs.length = 1000;
    }
    this.save();
    return log;
  }

  publicUser(u: UserAccount): Omit<UserAccount, 'passwordHash'> {
    const { passwordHash: _omit, ...rest } = u;
    return rest;
  }
}

