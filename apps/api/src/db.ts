import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import {
  CLINICAL_EXERCISES,
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
      } catch {
        this.state = createInitialState();
        this.save();
      }
    } else {
      this.state = createInitialState();
      this.save();
    }
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

