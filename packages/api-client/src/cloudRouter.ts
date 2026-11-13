import { createClient, type SupabaseClient } from '@supabase/supabase-js';
import {
  CLINICAL_EXERCISES,
  evaluateClinicalBiomechanicsPacket,
} from '@drtechcares/pose-engine';
import {
  FACE_EMBEDDING_DIM,
  verifyHybridBiometricIdentity,
  type BiometricPose,
  type EnrolledTemplateCandidate,
} from '@drtechcares/smart-biometric-core';
import {
  inspectBiometricFrame3D,
  type Detected3DPose,
} from '../../face-client/src/biometricAssistant';

type BiomechanicsEvaluationPacket = Parameters<typeof evaluateClinicalBiomechanicsPacket>[0];
type RepCounterState = NonNullable<Parameters<typeof evaluateClinicalBiomechanicsPacket>[1]>;
type FrameInspectionInput = Parameters<typeof inspectBiometricFrame3D>[0];
import type {
  AuditLogRecord,
  BannedTargetRecord,
  CloudClientConfig,
  DailyScheduleDay,
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
import {
  hydrateStateFromSupabase,
  newRecordId,
  persistToSupabase,
  type CloudRouterState,
  type StoredUserAccount,
} from './supabasePersistence';

function randomHex(byteLength: number): string {
  const bytes = new Uint8Array(byteLength);
  globalThis.crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
}

function timingSafeEqualStr(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i += 1) {
    diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return diff === 0;
}

async function hashPasswordEdge(password: string): Promise<string> {
  const salt = randomHex(16);
  const enc = new TextEncoder();
  const keyMaterial = await globalThis.crypto.subtle.importKey(
    'raw',
    enc.encode(password),
    'PBKDF2',
    false,
    ['deriveBits'],
  );
  const bits = await globalThis.crypto.subtle.deriveBits(
    {
      name: 'PBKDF2',
      salt: enc.encode(salt),
      iterations: 10_000,
      hash: 'SHA-256',
    },
    keyMaterial,
    256,
  );
  const hex = Array.from(new Uint8Array(bits), (b) => b.toString(16).padStart(2, '0')).join('');
  return `pbkdf2:${salt}:${hex}`;
}

async function verifyPasswordEdge(password: string, storedHash: string): Promise<boolean> {
  if (storedHash.startsWith('pbkdf2:')) {
    const parts = storedHash.split(':');
    if (parts.length !== 3) return false;
    const [, salt, expectedHex] = parts;
    const enc = new TextEncoder();
    const keyMaterial = await globalThis.crypto.subtle.importKey(
      'raw',
      enc.encode(password),
      'PBKDF2',
      false,
      ['deriveBits'],
    );
    const bits = await globalThis.crypto.subtle.deriveBits(
      {
        name: 'PBKDF2',
        salt: enc.encode(salt),
        iterations: 10_000,
        hash: 'SHA-256',
      },
      keyMaterial,
      256,
    );
    const actualHex = Array.from(new Uint8Array(bits), (b) => b.toString(16).padStart(2, '0')).join('');
    return timingSafeEqualStr(actualHex, expectedHex);
  }
  if (storedHash.startsWith('plain:')) {
    return timingSafeEqualStr(password, storedHash.slice(6));
  }
  return false;
}

function defaultWeeklySchedule(): DailyScheduleDay[] {
  return [0, 1, 2, 3, 4, 5, 6].map((dayIndex) => {
    const isTraining = [1, 3, 5].includes(dayIndex);
    return {
      dayIndex,
      dayType: isTraining ? 'training' : 'rest',
      activityType: isTraining ? 'minigame' : 'physio_exercise',
      difficulty: 'medium',
      label: isTraining ? 'Cognitive Rehab Card Game' : 'Physical Rehab (Pending AI Unlock)',
      exercises: [],
      cognitiveModpackId: 'modpack-core-99',
    };
  });
}

function createInitialState(useUuidIds = false): CloudRouterState {
  const now = new Date().toISOString();
  return {
    users: [
      {
        id: useUuidIds ? '00000000-0000-4000-8000-000000000001' : 'user-super-1',
        username: 'admin',
        passwordHash: 'plain:Admin@2026',
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
        id: useUuidIds ? '00000000-0000-4000-8000-000000000002' : 'user-pt-1',
        username: 'pt_somchai',
        passwordHash: 'plain:Pt@2026',
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
        id: useUuidIds ? '00000000-0000-4000-8000-000000000003' : 'user-pt-2',
        username: 'pt_waranya',
        passwordHash: 'plain:Pt@2026',
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
        categories: [
          'food_fruit',
          'objects',
          'animals',
          'counting',
          'safety_routine',
          'color_shape_time',
          'position',
        ],
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
      aiPoseDetectionEnabled: false,
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

function isVector512(value: unknown): value is number[] {
  return (
    Array.isArray(value) &&
    value.length === FACE_EMBEDDING_DIM &&
    value.every((n) => typeof n === 'number' && Number.isFinite(n))
  );
}

/** Uploads a base64 face snapshot into Supabase Storage and falls back cleanly on offline/errors. */
async function uploadFaceImageIfNeeded(
  supabase: SupabaseClient | null,
  patientId: string,
  rawUrl: string | undefined,
): Promise<string | undefined> {
  if (!supabase || !rawUrl || !rawUrl.startsWith('data:image/')) {
    return rawUrl;
  }
  const match = rawUrl.match(/^data:image\/([a-zA-Z0-9.+-]+);base64,(.+)$/);
  if (!match) return rawUrl;
  const rawSubtype = match[1].toLowerCase();
  const ext = rawSubtype === 'jpeg' ? 'jpg' : rawSubtype.replace(/[^a-z0-9]/g, '') || 'jpg';
  const contentType = `image/${rawSubtype}`;
  try {
    const binary = globalThis.atob(match[2]);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) {
      bytes[i] = binary.charCodeAt(i);
    }
    await supabase.storage.createBucket('patient-faces', { public: true }).catch(() => {});
    const objectPath = `patients/${patientId}-${Date.now()}.${ext}`;
    const { error } = await supabase.storage
      .from('patient-faces')
      .upload(objectPath, bytes, { contentType, upsert: true });
    if (error) return rawUrl;
    const { data } = supabase.storage.from('patient-faces').getPublicUrl(objectPath);
    return data?.publicUrl || rawUrl;
  } catch {
    return rawUrl;
  }
}

function publicUser(u: StoredUserAccount): UserAccount {
  const { passwordHash: _omit, ...rest } = u;
  return rest;
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      'content-type': 'application/json; charset=utf-8',
      'access-control-allow-origin': '*',
      'access-control-allow-headers': 'Content-Type, Authorization, x-device-id',
      'access-control-allow-methods': 'GET, POST, PUT, PATCH, DELETE, OPTIONS',
    },
  });
}

async function readJsonBody<T = Record<string, unknown>>(request: Request): Promise<T | null> {
  try {
    return (await request.json()) as T;
  } catch {
    return null;
  }
}

/**
 * Full-parity edge/cloud API router for Cloudflare Workers & Supabase delegation.
 * Mirrors every route, RBAC rule, biometric gate, and clinical guard in apps/api/src/app.ts.
 */
export function createCloudRouter(config: CloudClientConfig = {}) {
  const url = config.supabaseUrl || '';
  const key = config.supabaseServiceKey || config.supabaseAnonKey || '';
  const supabase: SupabaseClient | null =
    url && key
      ? createClient(url, key, { auth: { persistSession: false, autoRefreshToken: false } })
      : null;
  const state = createInitialState(Boolean(supabase));

  let hydrationPromise: Promise<void> | null = null;
  let lastHydratedAt = 0;
  const ensureHydrated = async (force = false) => {
    if (!supabase) return;
    if (!hydrationPromise || (force && Date.now() - lastHydratedAt > 2500)) {
      hydrationPromise = hydrateStateFromSupabase(supabase, state).then(() => {
        lastHydratedAt = Date.now();
      });
    }
    await hydrationPromise;
  };

  const recordAudit = (entry: Omit<AuditLogRecord, 'id' | 'createdAt'>): AuditLogRecord => {
    const log: AuditLogRecord = {
      id: `audit-${Date.now()}-${randomHex(3)}`,
      ...entry,
      createdAt: new Date().toISOString(),
    };
    state.audit_logs.unshift(log);
    if (state.audit_logs.length > 1000) {
      state.audit_logs.length = 1000;
    }
    if (supabase) {
      void supabase.from('audit_logs').insert({
        event_type: log.eventType,
        actor_role: log.actorRole,
        actor_id: log.actorId,
        actor_name: log.actorName,
        ip_address: log.ipAddress,
        device_id: log.deviceId,
        target_id: log.targetId ?? null,
        detail: { message: log.detail },
      });
    }
    return log;
  };

  const authenticate = (request: Request): { user?: StoredUserAccount; token?: string } => {
    const header = request.headers.get('authorization');
    if (!header || !header.startsWith('Bearer ')) return {};
    const token = header.slice(7).trim();
    const session = state.active_sessions.find((s) => s.token === token && !s.revoked);
    if (!session) return {};
    const user = state.users.find((u) => u.id === session.userId && u.active);
    if (!user) return {};
    const forcedBefore =
      user.forceLoggedOutAt && new Date(session.createdAt) <= new Date(user.forceLoggedOutAt);
    if (forcedBefore) return {};
    return { user, token };
  };

  const requireAuth = (
    authUser: StoredUserAccount | undefined,
    roles?: UserRole[],
  ): Response | null => {
    if (!authUser) {
      return jsonResponse({ error: 'UNAUTHORIZED', message: 'Authentication required' }, 401);
    }
    if (roles && !roles.includes(authUser.role)) {
      return jsonResponse({ error: 'FORBIDDEN', message: 'Insufficient role permissions' }, 403);
    }
    return null;
  };

  return async function handleApiRequest(request: Request): Promise<Response> {
    if (request.method === 'OPTIONS') {
      return jsonResponse({ ok: true }, 204);
    }

    const reqUrl = new URL(request.url);
    const pathname = reqUrl.pathname;
    const method = request.method.toUpperCase();

    await ensureHydrated(method === 'GET' && pathname !== '/api/health');

    const clientIp =
      request.headers.get('x-forwarded-for')?.split(',')[0]?.trim() ||
      request.headers.get('cf-connecting-ip') ||
      '127.0.0.1';
    const clientDeviceId = request.headers.get('x-device-id') || 'kiosk-browser';

    // Enforce IP / Device Bans
    const banned = state.banned_targets.find(
      (b) =>
        (b.targetType === 'ip' && b.value === clientIp) ||
        (b.targetType === 'device' && b.value === clientDeviceId),
    );
    if (banned) {
      return jsonResponse(
        {
          error: 'BANNED_IP_OR_DEVICE',
          message: `Access denied: ${banned.targetType.toUpperCase()} is banned (${banned.reason})`,
        },
        403,
      );
    }

    const { user: authUser, token: authToken } = authenticate(request);

    // Health
    if (pathname === '/api/health' && method === 'GET') {
      let supabaseConnected = false;
      const t0 = Date.now();
      if (supabase) {
        try {
          const { error } = await supabase.from('system_settings').select('id').limit(1);
          supabaseConnected = !error;
        } catch {
          supabaseConnected = false;
        }
      }
      return jsonResponse({
        ok: true,
        service: 'drtechcares-cloud-router',
        persistentDb: Boolean(supabase),
        supabaseConnected,
        latencyMs: Date.now() - t0,
        zeroGhostCache: true,
      });
    }

    // ========================
    // 1. AUTHENTICATION
    // ========================
    if (pathname === '/api/auth/login' && method === 'POST') {
      const body = await readJsonBody<{ username?: string; password?: string }>(request);
      if (!body?.username || !body?.password) {
        return jsonResponse({ error: 'VALIDATION_ERROR' }, 400);
      }
      const rawUsername = body.username.trim().toLowerCase();
      const normalizedUsername = rawUsername === 'director' ? 'admin' : rawUsername;
      const user = state.users.find(
        (u) => u.username.toLowerCase() === normalizedUsername && u.active,
      );
      if (!user || !(await verifyPasswordEdge(body.password, user.passwordHash))) {
        recordAudit({
          eventType: 'LOGIN_FAILED',
          actorRole: 'anonymous',
          actorId: body.username,
          actorName: body.username,
          ipAddress: clientIp,
          deviceId: clientDeviceId,
          detail: `Failed login attempt for username "${body.username}"`,
        });
        return jsonResponse(
          { error: 'INVALID_CREDENTIALS', message: 'ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง' },
          401,
        );
      }

      const token = randomHex(32);
      const sessionRow = {
        token,
        userId: user.id,
        role: user.role,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        createdAt: new Date().toISOString(),
        revoked: false,
      };
      state.active_sessions.push(sessionRow);
      persistToSupabase(supabase, (db) =>
        db.from('active_sessions').insert({
          token: sessionRow.token,
          user_id: sessionRow.userId,
          role: sessionRow.role,
          ip_address: sessionRow.ipAddress,
          device_id: sessionRow.deviceId,
          revoked: false,
        }),
      );

      recordAudit({
        eventType: 'USER_LOGIN',
        actorRole: user.role,
        actorId: user.id,
        actorName: user.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        detail: `Successful login as ${user.role}`,
      });

      return jsonResponse({ token, user: publicUser(user) });
    }

    if (pathname === '/api/auth/logout' && method === 'POST') {
      const err = requireAuth(authUser);
      if (err) return err;
      const session = state.active_sessions.find((s) => s.token === authToken);
      if (session) {
        session.revoked = true;
        persistToSupabase(supabase, (db) =>
          db.from('active_sessions').update({ revoked: true }).eq('token', session.token),
        );
      }
      return jsonResponse({ ok: true });
    }

    if (pathname === '/api/auth/me' && method === 'GET') {
      const err = requireAuth(authUser);
      if (err) return err;
      return jsonResponse({ user: publicUser(authUser!) });
    }

    if (pathname === '/api/auth/change-password' && method === 'POST') {
      const err = requireAuth(authUser);
      if (err) return err;
      const body = await readJsonBody<{ currentPassword?: string; newPassword?: string }>(request);
      if (!body?.currentPassword || !body?.newPassword || body.newPassword.length < 6) {
        return jsonResponse(
          { error: 'INVALID_PASSWORD', message: 'New password must be at least 6 characters' },
          400,
        );
      }
      if (!(await verifyPasswordEdge(body.currentPassword, authUser!.passwordHash))) {
        return jsonResponse(
          { error: 'WRONG_CURRENT_PASSWORD', message: 'รหัสผ่านเดิมไม่ถูกต้อง' },
          400,
        );
      }
      authUser!.passwordHash = await hashPasswordEdge(body.newPassword);
      persistToSupabase(supabase, (db) =>
        db.from('staff_users').update({ password_hash: authUser!.passwordHash }).eq('id', authUser!.id),
      );
      recordAudit({
        eventType: 'PASSWORD_CHANGED',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        detail: 'User changed own password',
      });
      return jsonResponse({ ok: true });
    }

    // ========================
    // 2. USER & RBAC MANAGEMENT
    // ========================
    if (pathname === '/api/users' && method === 'GET') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      const users =
        authUser!.role === 'super_admin'
          ? state.users
          : state.users.filter((u) => u.role === 'pt' || u.role === 'patient');
      return jsonResponse({ users: users.map(publicUser) });
    }

    if (pathname === '/api/users' && method === 'POST') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      const body = await readJsonBody<{
        username?: string;
        password?: string;
        role?: UserRole;
        fullName?: string;
        phone?: string;
        specialty?: string;
        licenseNo?: string;
        patientId?: string;
      }>(request);
      if (!body?.username || !body?.password || !body?.role || !body?.fullName) {
        return jsonResponse({ error: 'VALIDATION_ERROR' }, 400);
      }
      if (authUser!.role === 'pt' && body.role !== 'patient') {
        return jsonResponse(
          { error: 'FORBIDDEN', message: 'PT cannot create Admin or PT accounts' },
          403,
        );
      }
      if (state.users.some((u) => u.username.toLowerCase() === body.username!.toLowerCase())) {
        return jsonResponse({ error: 'USERNAME_EXISTS' }, 409);
      }
      const created: StoredUserAccount = {
        id: newRecordId(Boolean(supabase), 'user'),
        username: body.username,
        passwordHash: await hashPasswordEdge(body.password),
        role: body.role,
        fullName: body.fullName,
        phone: body.phone ?? '',
        specialty: body.specialty,
        licenseNo: body.licenseNo,
        patientId: body.patientId,
        active: true,
        forceLoggedOutAt: null,
        createdAt: new Date().toISOString(),
      };
      state.users.push(created);
      persistToSupabase(supabase, (db) =>
        db.from('staff_users').insert({
          id: created.id,
          username: created.username,
          password_hash: created.passwordHash,
          full_name: created.fullName,
          phone: created.phone,
          specialty: created.specialty ?? null,
          license_no: created.licenseNo ?? null,
          patient_id: created.patientId ?? null,
          role: created.role,
          active: created.active,
        }),
      );
      recordAudit({
        eventType: 'USER_CREATED',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: created.id,
        detail: `Created ${body.role} account @${body.username}`,
      });
      return jsonResponse({ user: publicUser(created) }, 201);
    }

    const userResetMatch = pathname.match(/^\/api\/users\/([^/]+)\/reset-password$/);
    if (userResetMatch && method === 'POST') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      const target = state.users.find((u) => u.id === userResetMatch[1]);
      if (!target) return jsonResponse({ error: 'USER_NOT_FOUND' }, 404);
      if (authUser!.role === 'pt' && target.role !== 'patient') {
        return jsonResponse(
          { error: 'FORBIDDEN', message: 'PT can only reset Patient passwords' },
          403,
        );
      }
      const body = await readJsonBody<{ newPassword?: string }>(request);
      target.passwordHash = await hashPasswordEdge(body?.newPassword || 'Patient@2026');
      persistToSupabase(supabase, (db) =>
        db.from('staff_users').update({ password_hash: target.passwordHash }).eq('id', target.id),
      );
      recordAudit({
        eventType: 'PASSWORD_RESET',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: target.id,
        detail: `Reset password for @${target.username}`,
      });
      return jsonResponse({ ok: true });
    }

    const userMatch = pathname.match(/^\/api\/users\/([^/]+)$/);
    if (userMatch && method === 'PUT') {
      const err = requireAuth(authUser);
      if (err) return err;
      const target = state.users.find((u) => u.id === userMatch[1]);
      if (!target) return jsonResponse({ error: 'USER_NOT_FOUND' }, 404);
      const actor = authUser!;
      if (actor.role === 'pt' && target.id !== actor.id && target.role !== 'patient') {
        return jsonResponse(
          { error: 'FORBIDDEN', message: 'PT cannot edit Admin or other PT accounts' },
          403,
        );
      }
      if (actor.role === 'patient' && target.id !== actor.id) {
        return jsonResponse({ error: 'FORBIDDEN' }, 403);
      }
      const body = (await readJsonBody<Partial<UserAccount>>(request)) ?? {};
      if (body.fullName !== undefined) target.fullName = body.fullName;
      if (body.phone !== undefined) target.phone = body.phone;
      if (body.specialty !== undefined && actor.role !== 'patient') target.specialty = body.specialty;
      if (body.licenseNo !== undefined && actor.role !== 'patient') target.licenseNo = body.licenseNo;
      if (body.role !== undefined && actor.role === 'super_admin') target.role = body.role;
      if (body.active !== undefined && actor.role === 'super_admin') target.active = body.active;

      persistToSupabase(supabase, (db) =>
        db
          .from('staff_users')
          .update({
            full_name: target.fullName,
            phone: target.phone,
            specialty: target.specialty ?? null,
            license_no: target.licenseNo ?? null,
            role: target.role,
            active: target.active,
          })
          .eq('id', target.id),
      );

      recordAudit({
        eventType: 'USER_UPDATED',
        actorRole: actor.role,
        actorId: actor.id,
        actorName: actor.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: target.id,
        detail: `Updated user @${target.username}`,
      });
      return jsonResponse({ user: publicUser(target) });
    }

    if (userMatch && method === 'DELETE') {
      const err = requireAuth(authUser, ['super_admin']);
      if (err) return err;
      const idx = state.users.findIndex((u) => u.id === userMatch[1]);
      if (idx === -1) return jsonResponse({ error: 'USER_NOT_FOUND' }, 404);
      const removed = state.users.splice(idx, 1)[0];
      persistToSupabase(supabase, (db) => db.from('staff_users').delete().eq('id', removed.id));
      recordAudit({
        eventType: 'USER_DELETED',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: removed.id,
        detail: `Deleted user @${removed.username}`,
      });
      return jsonResponse({ deleted: true });
    }

    // ========================
    // 3. BIOMETRICS & KIOSK FACE AUTH
    // ========================
    if (pathname === '/api/biometric/enroll' && method === 'POST') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      const body = await readJsonBody<{
        hn?: string;
        fullName?: string;
        age?: number;
        gender?: string;
        phone?: string;
        diagnosis?: string;
        assignedPtId?: string;
        nextAppointmentDate?: string;
        username?: string;
        password?: string;
        skipFaceScan?: boolean;
        faceBiometricStatus?: 'PENDING' | 'VERIFIED';
        faceImageUrl?: string;
        frontalCentroid?: unknown;
        overallCentroid?: unknown;
        rawFrames?: { pose?: string; vector?: unknown; qualityScore?: number }[];
      }>(request);

      const isPendingSkip = Boolean(body?.skipFaceScan);

      if (
        !body ||
        typeof body.hn !== 'string' ||
        body.hn.length < 2 ||
        typeof body.fullName !== 'string' ||
        body.fullName.length < 2 ||
        typeof body.age !== 'number' ||
        !Number.isInteger(body.age) ||
        body.age < 1 ||
        body.age > 120 ||
        (!isPendingSkip &&
          (!isVector512(body.frontalCentroid) ||
            !isVector512(body.overallCentroid) ||
            !Array.isArray(body.rawFrames) ||
            body.rawFrames.length !== 10 ||
            !body.rawFrames.every(
              (f) =>
                (f.pose === 'front' || f.pose === 'left' || f.pose === 'right' || f.pose === 'nod') &&
                isVector512(f.vector) &&
                typeof f.qualityScore === 'number' &&
                f.qualityScore > 0,
            )))
      ) {
        return jsonResponse({ error: 'VALIDATION_ERROR' }, 400);
      }

      const patientId = newRecordId(Boolean(supabase), 'pat');
      const assignedPtId =
        authUser!.role === 'pt'
          ? authUser!.id
          : body.assignedPtId || state.users.find((u) => u.role === 'pt')?.id || authUser!.id;
      const nextAppt =
        body.nextAppointmentDate ||
        new Date(Date.now() + 2 * 24 * 60 * 60 * 1000).toISOString().slice(0, 10);

      const rawFrames = (!isPendingSkip && Array.isArray(body.rawFrames) ? body.rawFrames : []) as {
        pose: 'front' | 'left' | 'right' | 'nod';
        vector: number[];
        qualityScore: number;
      }[];

      const faceStatus: 'PENDING' | 'VERIFIED' = isPendingSkip
        ? 'PENDING'
        : body.faceBiometricStatus ?? 'VERIFIED';
      const gender = body.gender || 'ชาย';
      const storedFaceImageUrl = await uploadFaceImageIfNeeded(supabase, patientId, body.faceImageUrl);

      const patient: PatientRecord = {
        id: patientId,
        hn: body.hn,
        fullName: body.fullName,
        age: body.age,
        gender,
        phone: body.phone ?? '',
        diagnosis: body.diagnosis ?? 'General Rehabilitation',
        assignedPtId,
        delegatedToPtId: null,
        delegationNote: null,
        nextAppointmentDate: nextAppt,
        registrationStatus: 'active',
        enrolledAt: new Date().toISOString(),
        faceBiometricStatus: faceStatus,
        faceImageUrl: storedFaceImageUrl,
        rehabSchedules: [],
        biometricHistory:
          !isPendingSkip && isVector512(body.frontalCentroid) && isVector512(body.overallCentroid)
            ? {
                frontal_centroid: body.frontalCentroid,
                overall_centroid: body.overallCentroid,
                raw_frames: rawFrames,
              }
            : undefined,
      };

      state.patients.push(patient);
      await persistToSupabase(supabase, (db) =>
        db.from('patients').insert({
          id: patient.id,
          hn: patient.hn,
          full_name: patient.fullName,
          age: patient.age,
          phone: patient.phone,
          diagnosis: patient.diagnosis,
          assigned_pt_id: patient.assignedPtId,
          next_appointment_date: patient.nextAppointmentDate,
          registration_status: 'active',
          history: {
            gender,
            faceBiometricStatus: faceStatus,
            faceImageUrl: storedFaceImageUrl ?? null,
            rehabSchedules: [],
          },
        }),
      );
      if (!isPendingSkip && isVector512(body.frontalCentroid) && isVector512(body.overallCentroid)) {
        const fc = body.frontalCentroid;
        const oc = body.overallCentroid;
        state.face_templates.push(
          {
            patientId,
            patientName: body.fullName,
            pose: 'frontal_centroid',
            vector: fc,
          },
          {
            patientId,
            patientName: body.fullName,
            pose: 'overall_centroid',
            vector: oc,
          },
          ...rawFrames.map((f) => ({
            patientId,
            patientName: body.fullName,
            pose: f.pose as BiometricPose,
            vector: f.vector,
          })),
        );
        await persistToSupabase(supabase, async (db) => {
          await db.from('face_enrollments').insert({
            patient_id: patientId,
            shot_count: 10,
            liveness_verified: true,
            enrolled_by: authUser!.id,
          });
          await db.from('face_templates').insert([
            { patient_id: patientId, pose: 'frontal_centroid', embedding: fc, quality_score: 1.0 },
            { patient_id: patientId, pose: 'overall_centroid', embedding: oc, quality_score: 1.0 },
            ...rawFrames.map((f) => ({
              patient_id: patientId,
              pose: f.pose,
              embedding: f.vector,
              quality_score: f.qualityScore,
            })),
          ]);
        });
      }

      const defaultCourse: PatientCourseRecord = {
        id: newRecordId(Boolean(supabase), 'course'),
        patientId,
        title: `${body.fullName} — แผนฟื้นฟูสมรรถภาพ`,
        weeklySchedule: defaultWeeklySchedule(),
        cognitiveModpackId: 'modpack-core-99',
        updatedAt: new Date().toISOString(),
      };
      state.patient_courses.push(defaultCourse);
      persistToSupabase(supabase, (db) =>
        db.from('patient_courses').insert({
          id: defaultCourse.id,
          patient_id: defaultCourse.patientId,
          title: defaultCourse.title,
          weekly_schedule: defaultCourse.weeklySchedule,
          cognitive_modpack_id: defaultCourse.cognitiveModpackId,
        }),
      );

      const patUsername = (body.username || body.hn.replace(/[^a-zA-Z0-9]/g, '')).toLowerCase();
      if (!state.users.some((u) => u.username.toLowerCase() === patUsername)) {
        const patUser: StoredUserAccount = {
          id: newRecordId(Boolean(supabase), 'user-pat'),
          username: patUsername,
          passwordHash: await hashPasswordEdge(body.password || 'Patient@2026'),
          role: 'patient',
          fullName: body.fullName,
          phone: body.phone ?? '',
          patientId,
          active: true,
          forceLoggedOutAt: null,
          createdAt: new Date().toISOString(),
        };
        state.users.push(patUser);
        persistToSupabase(supabase, (db) =>
          db.from('staff_users').insert({
            id: patUser.id,
            username: patUser.username,
            password_hash: patUser.passwordHash,
            full_name: patUser.fullName,
            phone: patUser.phone,
            patient_id: patientId,
            role: 'patient',
            active: true,
          }),
        );
      }

      recordAudit({
        eventType: isPendingSkip ? 'PATIENT_REGISTERED_PENDING_FACE' : 'PATIENT_ENROLLED_10SHOT',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: patientId,
        detail: isPendingSkip
          ? `Registered patient ${body.fullName} (${body.hn}) — Face ID pending`
          : `Enrolled patient ${body.fullName} (${body.hn}) with 10-frame biometric scan`,
      });

      return jsonResponse({ patient }, 201);
    }

    if (pathname === '/api/biometric/inspect-frame' && method === 'POST') {
      const body = await readJsonBody<FrameInspectionInput & { requestedPose?: Detected3DPose }>(
        request,
      );
      if (!body) return jsonResponse({ error: 'VALIDATION_ERROR' }, 400);

      const inspection = inspectBiometricFrame3D({
        landmarks3D: body.landmarks3D,
        faceCount: body.faceCount,
        requestedPose: body.requestedPose,
        metrics: body.metrics,
      });

      if (inspection.scan_status === 'SPOOF_DETECTED') {
        recordAudit({
          eventType: 'BIOMETRIC_SPOOF_DETECTED',
          actorRole: authUser?.role ?? 'patient_kiosk',
          actorId: authUser?.id ?? clientDeviceId,
          actorName: authUser?.fullName ?? 'Kiosk Face Sensor',
          ipAddress: clientIp,
          deviceId: clientDeviceId,
          detail: `Blocked spoof attempt (${inspection.error_code})`,
        });
      }

      return jsonResponse(inspection);
    }

    if (pathname === '/api/biometric/verify' && method === 'POST') {
      if (!state.system_settings.aiFaceScanEnabled) {
        return jsonResponse(
          {
            error: 'FACE_SCAN_DISABLED',
            message: 'ระบบสแกนใบหน้าถูกปิดชั่วคราวโดยผู้ดูแลระบบ',
          },
          503,
        );
      }
      const body = await readJsonBody<{ probeVector?: unknown }>(request);
      if (!body || !isVector512(body.probeVector)) {
        return jsonResponse({ error: 'VALIDATION_ERROR' }, 400);
      }

      let candidates: EnrolledTemplateCandidate[] = [...state.face_templates];
      let totalPatients = state.patients.filter((p) => p.registrationStatus === 'active').length;

      if (supabase && candidates.length === 0) {
        const { data, error } = await supabase.rpc('match_patient_face', {
          query_embedding: body.probeVector,
          match_count: 5,
        });
        if (!error && Array.isArray(data) && data.length > 0) {
          candidates = data
            .filter((row: Record<string, unknown>) => Array.isArray(row.embedding))
            .map((row: Record<string, unknown>) => ({
              patientId: String(row.patient_id),
              patientName: typeof row.full_name === 'string' ? row.full_name : undefined,
              pose: (row.pose as BiometricPose) || 'frontal_centroid',
              vector: row.embedding as number[],
            }));
          totalPatients = Math.max(1, new Set(candidates.map((c) => c.patientId)).size);
        }
      }

      const result = verifyHybridBiometricIdentity(body.probeVector, candidates, {
        totalEnrolledPatients: totalPatients,
        thresholds: {
          minRawCosMulti: state.system_settings.aiFaceThreshold,
        },
      });

      if (!result.matched || !result.patientId) {
        return jsonResponse(result, 401);
      }

      const patient = state.patients.find((p) => p.id === result.patientId) ?? null;
      const course = state.patient_courses.find((c) => c.patientId === result.patientId) ?? null;
      const todayIdx = new Date().getDay();
      const todayRoutine = course?.weeklySchedule.find((d) => d.dayIndex === todayIdx) ?? null;

      recordAudit({
        eventType: 'KIOSK_FACE_LOGIN',
        actorRole: 'patient_kiosk',
        actorId: result.patientId,
        actorName: result.patientName ?? 'Patient',
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: result.patientId,
        detail: `Kiosk Face ID matched (${result.confidencePercent}%)`,
      });

      return jsonResponse({
        ...result,
        patient: patient
          ? {
              id: patient.id,
              hn: patient.hn,
              fullName: patient.fullName,
              nextAppointmentDate: patient.nextAppointmentDate,
            }
          : null,
        todayRoutine,
      });
    }

    // ========================
    // 4. PATIENTS, DELEGATION & APPEND-ONLY NOTES
    // ========================
    if (pathname === '/api/patients' && method === 'GET') {
      const err = requireAuth(authUser);
      if (err) return err;
      const actor = authUser!;
      let list = state.patients;
      if (actor.role === 'pt') {
        list = list.filter((p) => p.assignedPtId === actor.id || p.delegatedToPtId === actor.id);
      } else if (actor.role === 'patient') {
        list = list.filter((p) => p.id === actor.patientId);
      }
      return jsonResponse({ patients: list });
    }

    const delegateMatch = pathname.match(/^\/api\/patients\/([^/]+)\/delegate$/);
    if (delegateMatch && method === 'POST') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      const patient = state.patients.find((p) => p.id === delegateMatch[1]);
      if (!patient) return jsonResponse({ error: 'PATIENT_NOT_FOUND' }, 404);
      const actor = authUser!;
      if (actor.role === 'pt' && patient.assignedPtId !== actor.id) {
        return jsonResponse({ error: 'FORBIDDEN', message: 'Only primary PT can delegate case' }, 403);
      }
      const body = (await readJsonBody<{ targetPtId?: string | null; note?: string }>(request)) ?? {};
      patient.delegatedToPtId = body.targetPtId || null;
      patient.delegationNote = body.note || null;
      persistToSupabase(supabase, (db) =>
        db
          .from('patients')
          .update({
            delegated_to_pt_id: patient.delegatedToPtId,
            delegation_note: patient.delegationNote,
            updated_at: new Date().toISOString(),
          })
          .eq('id', patient.id),
      );
      recordAudit({
        eventType: 'CASE_DELEGATED',
        actorRole: actor.role,
        actorId: actor.id,
        actorName: actor.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: patient.id,
        detail: body.targetPtId
          ? `Delegated case ${patient.fullName} to PT ${body.targetPtId}`
          : `Reclaimed case ${patient.fullName}`,
      });
      return jsonResponse({ patient });
    }

    const noteItemMatch = pathname.match(/^\/api\/patients\/([^/]+)\/notes\/([^/]+)$/);
    if (noteItemMatch && method === 'PUT') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      if (authUser!.role === 'pt') {
        return jsonResponse(
          {
            error: 'APPEND_ONLY_NOTES',
            message: 'นักกายภาพบำบัดไม่สามารถแก้ไขโน้ตประวัติเก่าที่เคยเขียนไปแล้วได้',
          },
          403,
        );
      }
      const note = state.medical_notes.find((n) => n.id === noteItemMatch[2]);
      if (!note) return jsonResponse({ error: 'NOTE_NOT_FOUND' }, 404);
      const body = await readJsonBody<{ content?: string }>(request);
      note.content = String(body?.content ?? note.content);
      persistToSupabase(supabase, (db) =>
        db.from('medical_notes').update({ content: note.content }).eq('id', note.id),
      );
      return jsonResponse({ note });
    }

    if (noteItemMatch && method === 'DELETE') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      if (authUser!.role === 'pt') {
        return jsonResponse(
          {
            error: 'APPEND_ONLY_NOTES',
            message: 'นักกายภาพบำบัดไม่สามารถลบโน้ตประวัติเก่าที่เคยเขียนไปแล้วได้',
          },
          403,
        );
      }
      state.medical_notes = state.medical_notes.filter((n) => n.id !== noteItemMatch[2]);
      persistToSupabase(supabase, (db) => db.from('medical_notes').delete().eq('id', noteItemMatch[2]));
      return jsonResponse({ deleted: true });
    }

    const notesMatch = pathname.match(/^\/api\/patients\/([^/]+)\/notes$/);
    if (notesMatch && method === 'GET') {
      const err = requireAuth(authUser);
      if (err) return err;
      const notes = state.medical_notes.filter((n) => n.patientId === notesMatch[1]);
      return jsonResponse({ notes });
    }

    if (notesMatch && method === 'POST') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      const body = await readJsonBody<{ content?: string }>(request);
      const content = body?.content?.trim();
      if (!content) return jsonResponse({ error: 'EMPTY_NOTE' }, 400);
      const note: MedicalNoteRecord = {
        id: `note-${Date.now()}-${randomHex(2)}`,
        patientId: notesMatch[1],
        authorId: authUser!.id,
        authorName: authUser!.fullName,
        content,
        createdAt: new Date().toISOString(),
      };
      state.medical_notes.push(note);
      persistToSupabase(supabase, (db) =>
        db.from('medical_notes').insert({
          patient_id: note.patientId,
          author_id: note.authorId,
          author_name: note.authorName,
          content: note.content,
        }),
      );
      recordAudit({
        eventType: 'MEDICAL_NOTE_APPENDED',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: note.patientId,
        detail: `Appended medical note to patient ${note.patientId}`,
      });
      return jsonResponse({ note }, 201);
    }

    const enrollExistingMatch = pathname.match(/^\/api\/patients\/([^/]+)\/enroll-face$/);
    if (enrollExistingMatch && method === 'POST') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      const patient = state.patients.find((p) => p.id === enrollExistingMatch[1]);
      if (!patient) return jsonResponse({ error: 'PATIENT_NOT_FOUND' }, 404);
      const body = await readJsonBody<{
        frontalCentroid?: unknown;
        overallCentroid?: unknown;
        rawFrames?: { pose?: string; vector?: unknown; qualityScore?: number }[];
        faceImageUrl?: string;
      }>(request);
      if (
        !body ||
        !isVector512(body.frontalCentroid) ||
        !isVector512(body.overallCentroid) ||
        !Array.isArray(body.rawFrames) ||
        body.rawFrames.length !== 10
      ) {
        return jsonResponse({ error: 'VALIDATION_ERROR' }, 400);
      }

      const rawFrames = body.rawFrames as {
        pose: 'front' | 'left' | 'right' | 'nod';
        vector: number[];
        qualityScore: number;
      }[];

      patient.faceBiometricStatus = 'VERIFIED';
      if (body.faceImageUrl) {
        patient.faceImageUrl = await uploadFaceImageIfNeeded(supabase, patient.id, body.faceImageUrl);
      }
      patient.biometricHistory = {
        frontal_centroid: body.frontalCentroid,
        overall_centroid: body.overallCentroid,
        raw_frames: rawFrames,
      };

      state.face_templates = state.face_templates.filter((t) => t.patientId !== patient.id);
      state.face_templates.push(
        {
          patientId: patient.id,
          patientName: patient.fullName,
          pose: 'frontal_centroid',
          vector: body.frontalCentroid,
        },
        {
          patientId: patient.id,
          patientName: patient.fullName,
          pose: 'overall_centroid',
          vector: body.overallCentroid,
        },
        ...rawFrames.map((f) => ({
          patientId: patient.id,
          patientName: patient.fullName,
          pose: f.pose as BiometricPose,
          vector: f.vector,
        })),
      );

      await persistToSupabase(supabase, async (db) => {
        await db.from('face_templates').delete().eq('patient_id', patient.id);
        await db.from('face_enrollments').insert({
          patient_id: patient.id,
          shot_count: 10,
          liveness_verified: true,
          enrolled_by: authUser!.id,
        });
        await db.from('face_templates').insert([
          { patient_id: patient.id, pose: 'frontal_centroid', embedding: body.frontalCentroid, quality_score: 1.0 },
          { patient_id: patient.id, pose: 'overall_centroid', embedding: body.overallCentroid, quality_score: 1.0 },
          ...rawFrames.map((f) => ({
            patient_id: patient.id,
            pose: f.pose,
            embedding: f.vector,
            quality_score: f.qualityScore,
          })),
        ]);
        await db
          .from('patients')
          .update({
            history: {
              gender: patient.gender ?? 'ชาย',
              faceBiometricStatus: 'VERIFIED',
              faceImageUrl: patient.faceImageUrl ?? null,
              rehabSchedules: patient.rehabSchedules ?? [],
            },
            updated_at: new Date().toISOString(),
          })
          .eq('id', patient.id);
      });

      recordAudit({
        eventType: 'PATIENT_FACE_ENROLLED',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: patient.id,
        detail: `Captured 10-frame Face ID for ${patient.fullName} (${patient.hn})`,
      });
      return jsonResponse({ patient });
    }

    const checkinMatch = pathname.match(/^\/api\/patients\/([^/]+)\/checkin$/);
    if (checkinMatch && method === 'POST') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      const patient = state.patients.find((p) => p.id === checkinMatch[1]);
      if (!patient) return jsonResponse({ error: 'PATIENT_NOT_FOUND' }, 404);
      const body = await readJsonBody<{ note?: string }>(request);
      const todayIso = new Date().toISOString().slice(0, 10);
      patient.nextAppointmentDate = todayIso;

      const existingSchedules = [...(patient.rehabSchedules ?? [])];
      const todayScheduleIdx = existingSchedules.findIndex((s) => s.scheduledDate === todayIso);
      if (todayScheduleIdx >= 0) {
        existingSchedules[todayScheduleIdx] = { ...existingSchedules[todayScheduleIdx], status: 'ACTIVE' };
      } else {
        existingSchedules.push({
          id: `sched-${Date.now()}`,
          patientId: patient.id,
          ptId: patient.delegatedToPtId || patient.assignedPtId,
          scheduledDate: todayIso,
          sets: 2,
          reps: 8,
          status: 'ACTIVE',
          notes: body?.note || 'Quick Check-in at clinic reception',
        });
      }
      patient.rehabSchedules = existingSchedules;

      const session: RehabSessionRecord = {
        id: `sess-checkin-${Date.now()}`,
        patientId: patient.id,
        activityType: 'physio_exercise',
        exerciseId: 'CLINIC_CHECKIN',
        exerciseName: body?.note?.trim() || 'เช็คอินเข้าฝึกกายภาพบำบัด',
        completedSets: 0,
        completedReps: 0,
        accuracyPercent: 100,
        maxRomDeg: 0,
        painStopTriggered: false,
        aiFeedback: `Checked in by ${authUser!.fullName}`,
        kinematicsReplay: [],
        createdAt: new Date().toISOString(),
      };
      state.rehab_sessions.unshift(session);

      await persistToSupabase(supabase, async (db) => {
        await db
          .from('patients')
          .update({
            next_appointment_date: todayIso,
            history: {
              gender: patient.gender ?? 'ชาย',
              faceBiometricStatus: patient.faceBiometricStatus ?? 'PENDING',
              faceImageUrl: patient.faceImageUrl ?? null,
              rehabSchedules: existingSchedules,
            },
            updated_at: new Date().toISOString(),
          })
          .eq('id', patient.id);
        await db.from('rehab_sessions').insert({
          patient_id: session.patientId,
          exercise_id: session.exerciseId,
          completed_sets: 0,
          completed_reps: 0,
          accuracy_percent: 100,
          max_rom_deg: 0,
          pain_stop_triggered: false,
          kinematics_replay: [],
        });
      });

      recordAudit({
        eventType: 'PATIENT_QUICK_CHECKIN',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: patient.id,
        detail: `Quick check-in for ${patient.fullName} (${patient.hn})`,
      });
      return jsonResponse({ patient, session });
    }

    const patientMatch = pathname.match(/^\/api\/patients\/([^/]+)$/);
    if (patientMatch && method === 'PUT') {
      const err = requireAuth(authUser);
      if (err) return err;
      const patient = state.patients.find((p) => p.id === patientMatch[1]);
      if (!patient) return jsonResponse({ error: 'PATIENT_NOT_FOUND' }, 404);
      const actor = authUser!;
      if (actor.role === 'pt' && patient.assignedPtId !== actor.id && patient.delegatedToPtId !== actor.id) {
        return jsonResponse(
          { error: 'FORBIDDEN', message: 'PT can only edit own assigned patients' },
          403,
        );
      }
      if (actor.role === 'patient' && patient.id !== actor.patientId) {
        return jsonResponse({ error: 'FORBIDDEN' }, 403);
      }
      const body = (await readJsonBody<Partial<PatientRecord>>(request)) ?? {};
      if (actor.role === 'patient') {
        if (body.phone !== undefined) patient.phone = body.phone;
      } else {
        if (body.fullName !== undefined) patient.fullName = body.fullName;
        if (body.age !== undefined) patient.age = body.age;
        if (body.gender !== undefined) patient.gender = body.gender;
        if (body.phone !== undefined) patient.phone = body.phone;
        if (body.diagnosis !== undefined) patient.diagnosis = body.diagnosis;
        if (body.nextAppointmentDate !== undefined) {
          patient.nextAppointmentDate = body.nextAppointmentDate;
        }
        if (body.assignedPtId !== undefined && actor.role === 'super_admin') {
          patient.assignedPtId = body.assignedPtId;
        }
        if (body.faceBiometricStatus !== undefined) {
          patient.faceBiometricStatus = body.faceBiometricStatus;
        }
        if (body.faceImageUrl !== undefined) {
          patient.faceImageUrl = await uploadFaceImageIfNeeded(supabase, patient.id, body.faceImageUrl);
        }
        if (Array.isArray(body.rehabSchedules)) {
          patient.rehabSchedules = body.rehabSchedules as RehabScheduleItem[];
        }
      }
      await persistToSupabase(supabase, (db) =>
        db
          .from('patients')
          .update({
            full_name: patient.fullName,
            age: patient.age,
            phone: patient.phone,
            diagnosis: patient.diagnosis,
            assigned_pt_id: patient.assignedPtId,
            next_appointment_date: patient.nextAppointmentDate,
            history: {
              gender: patient.gender ?? 'ชาย',
              faceBiometricStatus: patient.faceBiometricStatus ?? 'PENDING',
              faceImageUrl: patient.faceImageUrl ?? null,
              rehabSchedules: patient.rehabSchedules ?? [],
            },
            updated_at: new Date().toISOString(),
          })
          .eq('id', patient.id),
      );
      recordAudit({
        eventType: 'PATIENT_UPDATED',
        actorRole: actor.role,
        actorId: actor.id,
        actorName: actor.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: patient.id,
        detail: `Updated patient ${patient.fullName}`,
      });
      return jsonResponse({ patient });
    }

    if (patientMatch && method === 'DELETE') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      const patient = state.patients.find((p) => p.id === patientMatch[1]);
      if (!patient) return jsonResponse({ error: 'PATIENT_NOT_FOUND' }, 404);
      const actor = authUser!;
      if (actor.role === 'pt' && patient.assignedPtId !== actor.id) {
        return jsonResponse(
          { error: 'FORBIDDEN', message: 'PT can only delete own assigned patients' },
          403,
        );
      }
      const pid = patient.id;
      state.patients = state.patients.filter((p) => p.id !== pid);
      state.face_templates = state.face_templates.filter((t) => t.patientId !== pid);
      state.patient_courses = state.patient_courses.filter((c) => c.patientId !== pid);
      state.rehab_sessions = state.rehab_sessions.filter((s) => s.patientId !== pid);
      state.medical_notes = state.medical_notes.filter((n) => n.patientId !== pid);
      state.treatment_reports = state.treatment_reports.filter((r) => r.patientId !== pid);
      state.users = state.users.filter((u) => u.patientId !== pid);
      persistToSupabase(supabase, (db) => db.from('patients').delete().eq('id', pid));

      recordAudit({
        eventType: 'PATIENT_CASCADING_PURGE',
        actorRole: actor.role,
        actorId: actor.id,
        actorName: actor.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: pid,
        detail: `Cascading purge for patient ${patient.fullName} (${patient.hn})`,
      });
      return jsonResponse({ deleted: true, patientId: pid, cascadingPurge: true });
    }

    // ========================
    // 5. COURSES, SESSIONS, BIOMECHANICS & REPORTS
    // ========================
    const courseMatch = pathname.match(/^\/api\/courses\/([^/]+)$/);
    if (courseMatch && method === 'GET') {
      const err = requireAuth(authUser);
      if (err) return err;
      const course = state.patient_courses.find((c) => c.patientId === courseMatch[1]);
      if (!course) return jsonResponse({ error: 'COURSE_NOT_FOUND' }, 404);
      return jsonResponse({ course });
    }

    if (courseMatch && method === 'PUT') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      const body = await readJsonBody<{
        title?: string;
        cognitiveModpackId?: string;
        weeklySchedule?: DailyScheduleDay[];
      }>(request);
      if (!body?.title || !body?.cognitiveModpackId || !Array.isArray(body.weeklySchedule)) {
        return jsonResponse({ error: 'VALIDATION_ERROR' }, 400);
      }
      if (
        authUser!.role === 'pt' &&
        !state.system_settings.aiPoseDetectionEnabled &&
        body.weeklySchedule.some((d) => d.dayType === 'training' && d.activityType === 'physio_exercise')
      ) {
        return jsonResponse(
          {
            error: 'PHYSIO_LOCKED_UNDER_DEVELOPMENT',
            message:
              'ท่ากายภาพบำบัดถูกล็อกสถานะ "อยู่ระหว่างพัฒนา" PT สามารถสั่งจ่ายตารางได้เฉพาะ "มินิเกม"',
          },
          403,
        );
      }

      const pid = courseMatch[1];
      const existing = state.patient_courses.find((c) => c.patientId === pid);
      if (existing) {
        existing.title = body.title;
        existing.cognitiveModpackId = body.cognitiveModpackId;
        existing.weeklySchedule = body.weeklySchedule;
        existing.updatedAt = new Date().toISOString();
        await persistToSupabase(supabase, (db) =>
          db
            .from('patient_courses')
            .update({
              title: existing.title,
              cognitive_modpack_id: existing.cognitiveModpackId,
              weekly_schedule: existing.weeklySchedule,
              updated_at: existing.updatedAt,
            })
            .eq('id', existing.id),
        );
        recordAudit({
          eventType: 'COURSE_SCHEDULE_UPDATED',
          actorRole: authUser!.role,
          actorId: authUser!.id,
          actorName: authUser!.fullName,
          ipAddress: clientIp,
          deviceId: clientDeviceId,
          targetId: pid,
          detail: `Updated weekly course schedule "${existing.title}" for patient ${pid}`,
        });
        return jsonResponse({ course: existing });
      }
      const created: PatientCourseRecord = {
        id: newRecordId(Boolean(supabase), 'course'),
        patientId: pid,
        title: body.title,
        cognitiveModpackId: body.cognitiveModpackId,
        weeklySchedule: body.weeklySchedule,
        updatedAt: new Date().toISOString(),
      };
      state.patient_courses.push(created);
      await persistToSupabase(supabase, (db) =>
        db.from('patient_courses').insert({
          id: created.id,
          patient_id: created.patientId,
          title: created.title,
          weekly_schedule: created.weeklySchedule,
          cognitive_modpack_id: created.cognitiveModpackId,
        }),
      );
      recordAudit({
        eventType: 'COURSE_SCHEDULE_UPDATED',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: pid,
        detail: `Created weekly course schedule "${created.title}" for patient ${pid}`,
      });
      return jsonResponse({ course: created });
    }

    if (pathname === '/api/sessions' && method === 'GET') {
      const err = requireAuth(authUser);
      if (err) return err;
      const pid = reqUrl.searchParams.get('patientId') ?? undefined;
      let list = state.rehab_sessions;
      if (authUser!.role === 'patient') {
        list = list.filter((s) => s.patientId === authUser!.patientId);
      } else if (pid) {
        list = list.filter((s) => s.patientId === pid);
      }
      return jsonResponse({ sessions: list });
    }

    if (pathname === '/api/sessions' && method === 'POST') {
      const body = await readJsonBody<Partial<RehabSessionRecord>>(request);
      if (!body?.patientId || !body?.exerciseId || !body?.exerciseName) {
        return jsonResponse({ error: 'VALIDATION_ERROR' }, 400);
      }
      const session: RehabSessionRecord = {
        id: `sess-${Date.now()}-${randomHex(2)}`,
        patientId: body.patientId,
        activityType: body.activityType ?? 'minigame',
        exerciseId: body.exerciseId,
        exerciseName: body.exerciseName,
        completedSets: body.completedSets ?? 0,
        completedReps: body.completedReps ?? 0,
        accuracyPercent: body.accuracyPercent ?? 100,
        maxRomDeg: body.maxRomDeg ?? 0,
        painStopTriggered: body.painStopTriggered ?? false,
        aiFeedback: body.aiFeedback ?? 'Good posture stability observed.',
        kinematicsReplay: body.kinematicsReplay ?? [],
        minigameScore: body.minigameScore,
        createdAt: new Date().toISOString(),
      };
      state.rehab_sessions.unshift(session);
      persistToSupabase(supabase, (db) =>
        db.from('rehab_sessions').insert({
          patient_id: session.patientId,
          exercise_id: session.exerciseId,
          completed_sets: session.completedSets,
          completed_reps: session.completedReps,
          accuracy_percent: session.accuracyPercent,
          max_rom_deg: session.maxRomDeg,
          pain_stop_triggered: session.painStopTriggered,
          kinematics_replay: session.kinematicsReplay,
          minigame_score: session.minigameScore ?? null,
        }),
      );
      return jsonResponse({ session }, 201);
    }

    if (pathname === '/api/biomechanics/evaluate' && method === 'POST') {
      const body = await readJsonBody<
        BiomechanicsEvaluationPacket & { counterState?: RepCounterState }
      >(request);
      if (!body?.auth_status || !body?.spatial_framing || !body?.exercise_prescription || !body?.live_telemetry) {
        return jsonResponse({ error: 'VALIDATION_ERROR' }, 400);
      }
      const { counterState, ...packet } = body;
      const { decision } = evaluateClinicalBiomechanicsPacket(
        packet,
        counterState ?? { valid_reps: 0, invalid_reps: 0, hold_elapsed_sec: 0 },
      );

      if (decision.kiosk_state === 'EMERGENCY_HALT') {
        recordAudit({
          eventType: 'BIOMECHANICS_EMERGENCY_HALT',
          actorRole: authUser?.role ?? 'patient_kiosk',
          actorId: packet.auth_status.patient_id ?? authUser?.id ?? clientDeviceId,
          actorName: authUser?.fullName ?? 'Kiosk Biomechanics AI Agent',
          ipAddress: clientIp,
          deviceId: clientDeviceId,
          detail: `Emergency halt during ${packet.exercise_prescription.exercise_name ?? packet.exercise_prescription.exercise_id ?? 'rehab'}: ${decision.tts_voice_th}`,
        });
      }

      return jsonResponse(decision);
    }

    if (pathname === '/api/reports' && method === 'GET') {
      const err = requireAuth(authUser);
      if (err) return err;
      const actor = authUser!;
      let list = state.treatment_reports;
      if (actor.role === 'patient') {
        list = list.filter((r) => r.patientId === actor.patientId);
      } else if (actor.role === 'pt') {
        const allowed = new Set(
          state.patients
            .filter((p) => p.assignedPtId === actor.id || p.delegatedToPtId === actor.id)
            .map((p) => p.id),
        );
        list = list.filter((r) => allowed.has(r.patientId));
      }
      return jsonResponse({ reports: list });
    }

    if (pathname === '/api/reports' && method === 'POST') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      const body = await readJsonBody<{
        patientId?: string;
        summary?: string;
        aiRecommendation?: string;
        scoreSummary?: string;
      }>(request);
      if (!body?.patientId || !body?.summary) {
        return jsonResponse({ error: 'VALIDATION_ERROR' }, 400);
      }
      const now = new Date().toISOString();
      const report: TreatmentReportRecord = {
        id: newRecordId(Boolean(supabase), 'rep'),
        patientId: body.patientId,
        ptId: authUser!.id,
        ptName: authUser!.fullName,
        summary: body.summary,
        aiRecommendation: body.aiRecommendation ?? 'Continue daily cognitive & guided rehab routine.',
        scoreSummary: body.scoreSummary ?? 'Stable progress',
        createdAt: now,
        updatedAt: now,
      };
      state.treatment_reports.unshift(report);
      persistToSupabase(supabase, (db) =>
        db.from('treatment_reports').insert({
          id: report.id,
          patient_id: report.patientId,
          pt_id: report.ptId,
          pt_name: report.ptName,
          summary: report.summary,
          ai_recommendation: report.aiRecommendation,
          score_summary: report.scoreSummary,
        }),
      );
      return jsonResponse({ report }, 201);
    }

    const reportMatch = pathname.match(/^\/api\/reports\/([^/]+)$/);
    if (reportMatch && method === 'PUT') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      const rep = state.treatment_reports.find((r) => r.id === reportMatch[1]);
      if (!rep) return jsonResponse({ error: 'REPORT_NOT_FOUND' }, 404);
      const body = (await readJsonBody<Partial<TreatmentReportRecord>>(request)) ?? {};
      if (body.summary) rep.summary = String(body.summary);
      if (body.aiRecommendation) rep.aiRecommendation = String(body.aiRecommendation);
      if (body.scoreSummary) rep.scoreSummary = String(body.scoreSummary);
      rep.updatedAt = new Date().toISOString();
      persistToSupabase(supabase, (db) =>
        db
          .from('treatment_reports')
          .update({
            summary: rep.summary,
            ai_recommendation: rep.aiRecommendation,
            score_summary: rep.scoreSummary,
            updated_at: rep.updatedAt,
          })
          .eq('id', rep.id),
      );
      return jsonResponse({ report: rep });
    }

    // ========================
    // 6. EXERCISES & MODPACKS
    // ========================
    if (pathname === '/api/exercises' && method === 'GET') {
      return jsonResponse({ exercises: state.exercises });
    }

    if (pathname === '/api/exercises' && method === 'POST') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      const body = await readJsonBody<ManagedExerciseRecord>(request);
      if (!body?.code || !body?.name) {
        return jsonResponse({ error: 'VALIDATION_ERROR' }, 400);
      }
      const created: ManagedExerciseRecord = {
        ...body,
        id: body.id || `ex-custom-${Date.now()}`,
        createdBy: authUser!.role === 'super_admin' ? 'system' : authUser!.id,
      };
      state.exercises.push(created);
      await persistToSupabase(supabase, (db) =>
        db.from('exercises').upsert({
          id: created.id,
          code: created.code,
          category: created.category,
          name_i18n: created.name,
          instruction_i18n: created.instruction,
          detection_config: created.detection,
          active: true,
        }),
      );
      recordAudit({
        eventType: 'EXERCISE_CREATED',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: created.id,
        detail: `Added exercise ${created.code} (${created.name.th})`,
      });
      return jsonResponse({ exercise: created }, 201);
    }

    const exerciseMatch = pathname.match(/^\/api\/exercises\/([^/]+)$/);
    if (exerciseMatch && method === 'DELETE') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      const target = state.exercises.find((e) => e.id === exerciseMatch[1]);
      if (!target) return jsonResponse({ error: 'NOT_FOUND' }, 404);
      if (authUser!.role === 'pt' && target.createdBy !== authUser!.id) {
        return jsonResponse(
          {
            error: 'FORBIDDEN_GLOBAL_LIBRARY',
            message: 'ห้ามลบหรือแก้ไขคลังท่าฝึกส่วนกลางของโรงพยาบาล',
          },
          403,
        );
      }
      state.exercises = state.exercises.filter((e) => e.id !== exerciseMatch[1]);
      await persistToSupabase(supabase, (db) => db.from('exercises').delete().eq('id', exerciseMatch[1]));
      recordAudit({
        eventType: 'EXERCISE_DELETED',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: exerciseMatch[1],
        detail: `Deleted exercise ${target.code}`,
      });
      return jsonResponse({ deleted: true });
    }

    if (pathname === '/api/modpacks' && method === 'GET') {
      return jsonResponse({ modpacks: state.modpacks });
    }

    if (pathname === '/api/modpacks' && method === 'POST') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      const body = await readJsonBody<{ name?: string; categories?: string[]; questionCount?: number }>(
        request,
      );
      if (!body?.name) return jsonResponse({ error: 'VALIDATION_ERROR' }, 400);
      const created: ModpackRecord = {
        id: `modpack-${Date.now()}`,
        name: body.name,
        categories: body.categories ?? ['food_fruit', 'safety_routine'],
        questionCount: body.questionCount ?? 20,
        createdBy: authUser!.role === 'super_admin' ? 'system' : authUser!.id,
        active: true,
      };
      state.modpacks.push(created);
      await persistToSupabase(supabase, (db) =>
        db.from('modpacks').upsert({
          id: created.id,
          name: created.name,
          categories: created.categories,
          questions: Array.from({ length: created.questionCount }, (_, i) => ({ index: i + 1 })),
          active: true,
        }),
      );
      recordAudit({
        eventType: 'MODPACK_CREATED',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: created.id,
        detail: `Created modpack "${created.name}" (${created.questionCount} cards)`,
      });
      return jsonResponse({ modpack: created }, 201);
    }

    const modpackMatch = pathname.match(/^\/api\/modpacks\/([^/]+)$/);
    if (modpackMatch && method === 'DELETE') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      const target = state.modpacks.find((m) => m.id === modpackMatch[1]);
      if (!target) return jsonResponse({ error: 'NOT_FOUND' }, 404);
      if (authUser!.role === 'pt' && target.createdBy !== authUser!.id) {
        return jsonResponse(
          {
            error: 'FORBIDDEN_GLOBAL_LIBRARY',
            message: 'ห้ามลบหรือแก้ไขชุดมินิเกมส่วนกลางของโรงพยาบาล',
          },
          403,
        );
      }
      state.modpacks = state.modpacks.filter((m) => m.id !== modpackMatch[1]);
      await persistToSupabase(supabase, (db) => db.from('modpacks').delete().eq('id', modpackMatch[1]));
      recordAudit({
        eventType: 'MODPACK_DELETED',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: modpackMatch[1],
        detail: `Deleted modpack "${target.name}"`,
      });
      return jsonResponse({ deleted: true });
    }

    // ========================
    // 7. STAFF CALLS, SECURITY, AUDIT LOGS & SETTINGS
    // ========================
    if (pathname === '/api/staff-calls' && method === 'GET') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      return jsonResponse({ calls: state.staff_calls });
    }

    if (pathname === '/api/staff-calls' && method === 'POST') {
      const body = await readJsonBody<{ kioskId?: string; patientId?: string | null; reason?: string }>(
        request,
      );
      if (!body?.reason || body.reason.length > 240) {
        return jsonResponse({ error: 'VALIDATION_ERROR' }, 400);
      }
      const kioskId = body.kioskId || 'KIOSK-OPD-01';
      const call: StaffCallRecord = {
        id: `call-${Date.now()}`,
        kioskId,
        patientId: body.patientId ?? null,
        reason: body.reason,
        status: 'open',
        createdAt: new Date().toISOString(),
      };
      state.staff_calls.unshift(call);
      persistToSupabase(supabase, (db) =>
        db.from('staff_calls').insert({
          kiosk_id: call.kioskId,
          patient_id: call.patientId ?? null,
          reason: call.reason,
          status: call.status,
        }),
      );
      recordAudit({
        eventType: 'KIOSK_STAFF_CALL',
        actorRole: 'patient_kiosk',
        actorId: kioskId,
        actorName: kioskId,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        detail: body.reason,
      });
      return jsonResponse({ call }, 201);
    }

    const staffCallAckMatch = pathname.match(/^\/api\/staff-calls\/([^/]+)\/acknowledge$/);
    if (staffCallAckMatch && method === 'PATCH') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      const target = state.staff_calls.find((c) => c.id === staffCallAckMatch[1]);
      if (target) {
        target.status = 'acknowledged';
        const ackReason = target.reason.startsWith('[ACKNOWLEDGED]')
          ? target.reason
          : `[ACKNOWLEDGED] ${target.reason}`;
        await persistToSupabase(supabase, (db) =>
          db.from('staff_calls').update({ status: 'open', reason: ackReason }).eq('id', target.id),
        );
        recordAudit({
          eventType: 'STAFF_CALL_ACKNOWLEDGED',
          actorRole: authUser!.role,
          actorId: authUser!.id,
          actorName: authUser!.fullName,
          ipAddress: clientIp,
          deviceId: clientDeviceId,
          targetId: target.id,
          detail: `Acknowledged Kiosk call from ${target.kioskId}`,
        });
      }
      return jsonResponse({ ok: true });
    }

    const staffCallResolveMatch = pathname.match(/^\/api\/staff-calls\/([^/]+)\/resolve$/);
    if (staffCallResolveMatch && method === 'PATCH') {
      const err = requireAuth(authUser, ['super_admin', 'pt']);
      if (err) return err;
      const target = state.staff_calls.find((c) => c.id === staffCallResolveMatch[1]);
      if (target) {
        target.status = 'resolved';
        target.reason = target.reason.replace(/^\[ACKNOWLEDGED\]\s*/, '');
        await persistToSupabase(supabase, (db) =>
          db.from('staff_calls').update({ status: 'resolved', reason: target.reason }).eq('id', target.id),
        );
        recordAudit({
          eventType: 'STAFF_CALL_RESOLVED',
          actorRole: authUser!.role,
          actorId: authUser!.id,
          actorName: authUser!.fullName,
          ipAddress: clientIp,
          deviceId: clientDeviceId,
          targetId: target.id,
          detail: `Resolved Kiosk call from ${target.kioskId}`,
        });
      }
      return jsonResponse({ ok: true });
    }

    if (pathname === '/api/audit-logs' && method === 'GET') {
      const err = requireAuth(authUser, ['super_admin']);
      if (err) return err;
      return jsonResponse({
        logs: state.audit_logs,
        bannedTargets: state.banned_targets,
        activeSessions: state.active_sessions.filter((s) => !s.revoked),
      });
    }

    if (pathname === '/api/security/ban' && method === 'POST') {
      const err = requireAuth(authUser, ['super_admin']);
      if (err) return err;
      const body = await readJsonBody<{ targetType?: 'ip' | 'device'; value?: string; reason?: string }>(
        request,
      );
      if ((body?.targetType !== 'ip' && body?.targetType !== 'device') || !body?.value) {
        return jsonResponse({ error: 'VALIDATION_ERROR' }, 400);
      }
      const ban: BannedTargetRecord = {
        id: newRecordId(Boolean(supabase), 'ban'),
        targetType: body.targetType,
        value: body.value.trim(),
        reason: body.reason?.trim() || 'Banned by Hospital Super Admin',
        bannedBy: authUser!.fullName,
        createdAt: new Date().toISOString(),
      };
      state.banned_targets.unshift(ban);
      persistToSupabase(supabase, (db) =>
        db.from('banned_targets').insert({
          id: ban.id,
          target_type: ban.targetType,
          value: ban.value,
          reason: ban.reason,
          banned_by: ban.bannedBy,
        }),
      );
      recordAudit({
        eventType: 'SECURITY_TARGET_BANNED',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        detail: `Banned ${body.targetType}: ${body.value}`,
      });
      return jsonResponse({ ban }, 201);
    }

    const banDeleteMatch = pathname.match(/^\/api\/security\/ban\/([^/]+)$/);
    if (banDeleteMatch && method === 'DELETE') {
      const err = requireAuth(authUser, ['super_admin']);
      if (err) return err;
      state.banned_targets = state.banned_targets.filter((b) => b.id !== banDeleteMatch[1]);
      await persistToSupabase(supabase, (db) =>
        db.from('banned_targets').delete().eq('id', banDeleteMatch[1]),
      );
      recordAudit({
        eventType: 'SECURITY_TARGET_UNBANNED',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: banDeleteMatch[1],
        detail: `Removed ban ${banDeleteMatch[1]}`,
      });
      return jsonResponse({ ok: true });
    }

    if (pathname === '/api/kiosks' && method === 'POST') {
      const err = requireAuth(authUser, ['super_admin']);
      if (err) return err;
      const body = await readJsonBody<{ kioskId?: string; location?: string }>(request);
      if (!body?.kioskId?.trim()) return jsonResponse({ error: 'VALIDATION_ERROR' }, 400);
      const cleanId = body.kioskId.trim().toUpperCase();
      const pairingToken = `KPAIR-${randomHex(3).toUpperCase()}`;
      const kiosk = {
        kioskId: cleanId,
        location: body.location?.trim() || 'OPD Rehab Zone',
        online: true,
        currentState: 'STANDBY',
        pairingToken,
        lastHeartbeatAt: new Date().toISOString(),
      };
      state.kiosks = state.kiosks.filter((k) => k.kioskId !== cleanId);
      state.kiosks.push(kiosk);
      await persistToSupabase(supabase, (db) =>
        db.from('kiosks').upsert({
          kiosk_id: kiosk.kioskId,
          location: kiosk.location,
          online: true,
          current_state: `STANDBY:${pairingToken}`,
          last_heartbeat_at: kiosk.lastHeartbeatAt,
        }),
      );
      recordAudit({
        eventType: 'KIOSK_REGISTERED',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: kiosk.kioskId,
        detail: `Registered Kiosk ${kiosk.kioskId} (${kiosk.location})`,
      });
      return jsonResponse({ kiosk }, 201);
    }

    const kioskDeleteMatch = pathname.match(/^\/api\/kiosks\/([^/]+)$/);
    if (kioskDeleteMatch && method === 'DELETE') {
      const err = requireAuth(authUser, ['super_admin']);
      if (err) return err;
      const kid = decodeURIComponent(kioskDeleteMatch[1]);
      state.kiosks = state.kiosks.filter((k) => k.kioskId !== kid);
      await persistToSupabase(supabase, (db) => db.from('kiosks').delete().eq('kiosk_id', kid));
      recordAudit({
        eventType: 'KIOSK_REMOVED',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: kid,
        detail: `Removed Kiosk ${kid}`,
      });
      return jsonResponse({ deleted: true });
    }

    const forceLogoutMatch = pathname.match(/^\/api\/security\/force-logout\/([^/]+)$/);
    if (forceLogoutMatch && method === 'POST') {
      const err = requireAuth(authUser, ['super_admin']);
      if (err) return err;
      const user = state.users.find((u) => u.id === forceLogoutMatch[1]);
      if (!user) return jsonResponse({ error: 'USER_NOT_FOUND' }, 404);
      user.forceLoggedOutAt = new Date().toISOString();
      for (const s of state.active_sessions) {
        if (s.userId === user.id) s.revoked = true;
      }
      persistToSupabase(supabase, async (db) => {
        await db
          .from('staff_users')
          .update({ force_logged_out_at: user.forceLoggedOutAt })
          .eq('id', user.id);
        await db.from('active_sessions').update({ revoked: true }).eq('user_id', user.id);
      });
      recordAudit({
        eventType: 'FORCE_LOGOUT',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        targetId: user.id,
        detail: `Forced logout for @${user.username}`,
      });
      return jsonResponse({ ok: true });
    }

    if (pathname === '/api/settings' && method === 'GET') {
      return jsonResponse({ settings: state.system_settings, kiosks: state.kiosks });
    }

    if (pathname === '/api/settings' && method === 'PUT') {
      const err = requireAuth(authUser, ['super_admin']);
      if (err) return err;
      const patch = (await readJsonBody<Partial<SystemSettingsRecord>>(request)) ?? {};
      state.system_settings = {
        ...state.system_settings,
        ...patch,
      };
      persistToSupabase(supabase, (db) =>
        db
          .from('system_settings')
          .upsert({
            id: 1,
            ai_face_scan_enabled: state.system_settings.aiFaceScanEnabled,
            ai_face_threshold: state.system_settings.aiFaceThreshold,
            ai_pose_detection_enabled: state.system_settings.aiPoseDetectionEnabled,
            kiosk_auto_reset_sec: state.system_settings.kioskAutoResetSec,
            updated_at: new Date().toISOString(),
          }),
      );
      recordAudit({
        eventType: 'SYSTEM_SETTINGS_UPDATED',
        actorRole: authUser!.role,
        actorId: authUser!.id,
        actorName: authUser!.fullName,
        ipAddress: clientIp,
        deviceId: clientDeviceId,
        detail: JSON.stringify(state.system_settings),
      });
      return jsonResponse({ settings: state.system_settings });
    }

    return jsonResponse({ error: 'NOT_FOUND', path: pathname }, 404);
  };
}
