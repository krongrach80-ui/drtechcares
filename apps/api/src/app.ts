import crypto from 'node:crypto';
import cors from 'cors';
import express, { type Express, type NextFunction, type Request, type Response } from 'express';
import rateLimit from 'express-rate-limit';
import helmet from 'helmet';
import { z } from 'zod';
import {
  FACE_EMBEDDING_DIM,
  verifyHybridBiometricIdentity,
} from '@drtechcares/smart-biometric-core';
import {
  evaluateClinicalBiomechanicsPacket,
} from '@drtechcares/pose-engine';
import {
  inspectBiometricFrame3D,
  type Detected3DPose,
} from '../../../packages/face-client/src/biometricAssistant';
import {
  clampTtsText,
  synthesizeTtsAudio,
} from '../../../packages/api-client/src/ttsProxy';
import {
  ClinicalDatabase,
  hashPassword,
  verifyPassword,
  type DailyScheduleDay,
  type PatientRecord,
  type UserAccount,
  type UserRole,
} from './db';

const vectorSchema = z.array(z.number().finite()).length(FACE_EMBEDDING_DIM);

const loginSchema = z.object({
  username: z.string().min(1),
  password: z.string().min(1),
});

const enrollSchema = z.object({
  hn: z.string().min(2).max(40),
  fullName: z.string().min(2).max(120),
  age: z.number().int().min(1).max(120),
  phone: z.string().max(40).default(''),
  diagnosis: z.string().max(300).default('General Rehabilitation'),
  assignedPtId: z.string().optional(),
  nextAppointmentDate: z.string().optional(),
  username: z.string().optional(),
  password: z.string().optional(),
  frontalCentroid: vectorSchema,
  overallCentroid: vectorSchema,
  rawFrames: z
    .array(
      z.object({
        pose: z.enum(['front', 'left', 'right', 'nod']),
        vector: vectorSchema,
        qualityScore: z.number().positive(),
      }),
    )
    .length(10),
});

const verifySchema = z.object({
  probeVector: vectorSchema,
});

const inspectFrameSchema = z.object({
  faceCount: z.number().int().min(0).optional(),
  requestedPose: z
    .enum(['CENTER', 'TURN_LEFT', 'TURN_RIGHT', 'TILT_UP', 'TILT_DOWN', 'UNKNOWN'])
    .optional(),
  landmarks3D: z
    .array(
      z.object({
        x: z.number(),
        y: z.number(),
        z: z.number().optional(),
      }),
    )
    .optional(),
  metrics: z
    .object({
      faceCount: z.number().int().min(0).optional(),
      faceHeightRatio: z.number().min(0).max(2).optional(),
      faceCenterX: z.number().optional(),
      faceCenterY: z.number().optional(),
      croppedEdge: z.boolean().optional(),
      brightness: z.number().optional(),
      sharpness: z.number().optional(),
      burnoutRatio: z.number().optional(),
      lightingAsymmetry: z.number().optional(),
      eyesVisible: z.boolean().optional(),
      mouthVisible: z.boolean().optional(),
      eyeGlareRatio: z.number().optional(),
      sunglassesDetected: z.boolean().optional(),
      moireOrBezelDetected: z.boolean().optional(),
      isLive: z.boolean().optional(),
      depthScore: z.number().optional(),
      microMotion: z.number().optional(),
      yawDeg: z.number().optional(),
      pitchDeg: z.number().optional(),
    })
    .optional(),
});

const sessionSchema = z.object({
  patientId: z.string().min(1),
  activityType: z.enum(['minigame', 'physio_exercise']).default('minigame'),
  exerciseId: z.string().min(1),
  exerciseName: z.string().min(1),
  completedSets: z.number().int().min(0),
  completedReps: z.number().int().min(0),
  accuracyPercent: z.number().min(0).max(100),
  maxRomDeg: z.number().min(0).max(360),
  painStopTriggered: z.boolean().default(false),
  aiFeedback: z.string().default('Good posture stability observed.'),
  kinematicsReplay: z
    .array(z.object({ tMs: z.number(), angleDeg: z.number() }))
    .default([]),
  minigameScore: z
    .object({
      correct: z.number().int().min(0),
      total: z.number().int().min(1),
      category: z.string(),
    })
    .optional(),
});

const staffCallSchema = z.object({
  kioskId: z.string().min(1).default('KIOSK-OPD-01'),
  patientId: z.string().nullable().optional(),
  reason: z.string().min(1).max(240),
});

const biomechanicsEvaluateSchema = z.object({
  auth_status: z.object({
    verified: z.boolean(),
    patient_id: z.string().optional(),
    confidence: z.number().min(0).max(1),
    liveness: z.boolean(),
  }),
  spatial_framing: z.object({
    distance_m: z.number().min(0).max(10),
    full_body_visible: z.boolean(),
    feet_visible: z.boolean().optional(),
    lighting_score: z.number().min(0).max(1).optional(),
  }),
  exercise_prescription: z.object({
    exercise_id: z.string().min(1).optional(),
    exercise_name: z.string().min(1).optional(),
    target_joint: z.enum(['left_shoulder', 'right_shoulder', 'left_knee', 'right_knee', 'spine', 'hip']).optional(),
    target_rom_deg: z.number().min(10).max(180),
    hold_sec: z.number().min(0).max(60),
    target_reps: z.number().int().min(1).max(50),
    max_safe_rom_deg: z.number().optional(),
    max_safe_velocity_deg_per_sec: z.number().optional(),
  }),
  live_telemetry: z.object({
    primary_angle_deg: z.number(),
    trunk_tilt_deg: z.number(),
    contralateral_shoulder_elevation_cm: z.number().optional(),
    shoulder_elevation_cm: z.number().optional(),
    knee_valgus_deg: z.number().optional(),
    heel_lifted: z.boolean().optional(),
    pelvic_tilt_deg: z.number().optional(),
    angular_velocity_deg_per_sec: z.number().optional(),
    angular_velocity_deg_s: z.number().optional(),
    hold_elapsed_sec: z.number().min(0).optional(),
    pain_score: z.number().min(0).max(1).optional(),
    facial_pain_detected: z.boolean().optional(),
    rapid_vertical_drop_detected: z.boolean().optional(),
    dt_sec: z.number().optional(),
  }),
  counterState: z
    .object({
      valid_reps: z.number().int().min(0),
      invalid_reps: z.number().int().min(0),
      hold_elapsed_sec: z.number().min(0),
      awaiting_return_to_rest: z.boolean().optional(),
      rep_invalidated_in_cycle: z.boolean().optional(),
      reached_attempt_threshold: z.boolean().optional(),
    })
    .optional(),
});

const courseUpdateSchema = z.object({
  title: z.string().min(1),
  cognitiveModpackId: z.string().min(1),
  weeklySchedule: z.array(
    z.object({
      dayIndex: z.number().int().min(0).max(6),
      dayType: z.enum(['training', 'rest']),
      activityType: z.enum(['minigame', 'physio_exercise']).default('minigame'),
      difficulty: z.enum(['easy', 'medium', 'hard']).default('medium'),
      label: z.string(),
      exercises: z.array(
        z.object({
          exerciseId: z.string(),
          side: z.enum(['left', 'right']),
          sets: z.number().int().min(1).max(10),
          reps: z.number().int().min(1).max(50),
          holdSec: z.number().int().min(0).max(60),
          targetMinDeg: z.number().min(0).max(180),
          targetMaxDeg: z.number().min(0).max(180),
          allowedStarterSide: z.enum(['left', 'right', 'both']),
        }),
      ),
      cognitiveModpackId: z.string().optional(),
    }),
  ),
});

interface AuthenticatedRequest extends Request {
  authUser?: UserAccount;
  authToken?: string;
  clientIp?: string;
  clientDeviceId?: string;
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

export function createDrTechCaresApiApp(
  db = new ClinicalDatabase({ persistToDisk: true }),
): Express {
  const app = express();
  app.use(helmet());
  app.use(cors());
  app.use(express.json({ limit: '4mb' }));

  const limiter = rateLimit({
    windowMs: 60_000,
    max: 600,
    standardHeaders: true,
    legacyHeaders: false,
  });
  app.use('/api', limiter);

  // Extract IP/Device and enforce IP/Device Bans
  app.use('/api', (req: AuthenticatedRequest, res: Response, next: NextFunction) => {
    const ip =
      (req.headers['x-forwarded-for'] as string | undefined)?.split(',')[0]?.trim() ||
      req.ip ||
      '127.0.0.1';
    const deviceId = (req.headers['x-device-id'] as string | undefined) || 'kiosk-browser';
    req.clientIp = ip;
    req.clientDeviceId = deviceId;

    const banned = db.state.banned_targets.find(
      (b) => (b.targetType === 'ip' && b.value === ip) || (b.targetType === 'device' && b.value === deviceId),
    );
    if (banned) {
      res.status(403).json({
        error: 'BANNED_IP_OR_DEVICE',
        message: `Access denied: ${banned.targetType.toUpperCase()} is banned (${banned.reason})`,
      });
      return;
    }
    next();
  });

  // Attach optional auth session
  app.use('/api', (req: AuthenticatedRequest, _res: Response, next: NextFunction) => {
    const authHeader = req.headers.authorization;
    const xStaffHeader = Array.isArray(req.headers['x-staff-token'])
      ? req.headers['x-staff-token'][0]
      : req.headers['x-staff-token'];
    const token =
      authHeader && authHeader.startsWith('Bearer ')
        ? authHeader.slice(7).trim()
        : typeof xStaffHeader === 'string' && xStaffHeader.trim()
          ? xStaffHeader.trim()
          : '';
    if (token) {
      const session = db.state.active_sessions.find((s) => s.token === token && !s.revoked);
      if (session) {
        const user = db.state.users.find((u) => u.id === session.userId && u.active);
        if (user) {
          const forcedBefore =
            user.forceLoggedOutAt && new Date(session.createdAt) <= new Date(user.forceLoggedOutAt);
          if (!forcedBefore) {
            req.authUser = user;
            req.authToken = token;
          }
        }
      }
    }
    next();
  });

  const requireAuth = (roles?: UserRole[]) => (req: AuthenticatedRequest, res: Response, next: NextFunction) => {
    if (!req.authUser) {
      res.status(401).json({ error: 'UNAUTHORIZED', message: 'Authentication required' });
      return;
    }
    if (roles && !roles.includes(req.authUser.role)) {
      res.status(403).json({ error: 'FORBIDDEN', message: 'Insufficient role permissions' });
      return;
    }
    next();
  };

  /** Create and persist an active session token for a newly logged-in or registered user. */
  const issueSessionForUser = (user: UserAccount, req: AuthenticatedRequest): string => {
    const token = crypto.randomBytes(32).toString('hex');
    db.state.active_sessions.push({
      token,
      userId: user.id,
      role: user.role,
      ipAddress: req.clientIp ?? 'unknown',
      deviceId: req.clientDeviceId ?? 'unknown',
      createdAt: new Date().toISOString(),
      revoked: false,
    });
    db.save();
    return token;
  };

  /** Normalize external role aliases into canonical Express UserRole values. */
  const normalizeStaffRole = (rawRole?: string): UserRole => {
    const upper = (rawRole ?? 'pt').trim().toUpperCase();
    if (upper === 'SUPER_ADMIN' || upper === 'ADMIN' || upper === 'DIRECTOR') return 'super_admin';
    if (upper === 'PATIENT') return 'patient';
    return 'pt';
  };

  app.get('/api/health', (_req: Request, res: Response) => {
    res.json({ ok: true, service: 'drtechcares-production-api', persistentDb: true });
  });

  app.get('/api/tts', async (req: Request, res: Response) => {
    const rawText = typeof req.query.text === 'string' ? req.query.text : typeof req.query.q === 'string' ? req.query.q : '';
    const cleanText = clampTtsText(rawText);
    if (!cleanText) {
      res.status(400).json({ error: 'VALIDATION_ERROR', message: 'Missing text parameter' });
      return;
    }
    const langParam = typeof req.query.lang === 'string' ? req.query.lang : 'th';
    const synthesized = await synthesizeTtsAudio(langParam, cleanText);
    if (!synthesized) {
      res.status(502).json({ error: 'TTS_UPSTREAM_UNAVAILABLE', message: 'TTS synthesis unavailable' });
      return;
    }
    res
      .status(200)
      .setHeader('Content-Type', synthesized.contentType)
      .setHeader('Cache-Control', 'public, max-age=86400')
      .setHeader('X-TTS-Provider', synthesized.provider)
      .send(Buffer.from(synthesized.audio));
  });

  // ========================
  // 1. AUTHENTICATION
  // ========================
  app.post('/api/auth/login', (req: AuthenticatedRequest, res: Response) => {
    const parsed = loginSchema.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: 'VALIDATION_ERROR', issues: parsed.error.issues });
      return;
    }
    const user = db.state.users.find(
      (u) => u.username.toLowerCase() === parsed.data.username.trim().toLowerCase() && u.active,
    );
    if (!user || !verifyPassword(parsed.data.password, user.passwordHash)) {
      db.recordAudit({
        eventType: 'LOGIN_FAILED',
        actorRole: 'anonymous',
        actorId: parsed.data.username,
        actorName: parsed.data.username,
        ipAddress: req.clientIp ?? 'unknown',
        deviceId: req.clientDeviceId ?? 'unknown',
        detail: `Failed login attempt for username "${parsed.data.username}"`,
      });
      res.status(401).json({ error: 'INVALID_CREDENTIALS', message: 'ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง' });
      return;
    }

    const token = issueSessionForUser(user, req);

    db.recordAudit({
      eventType: 'USER_LOGIN',
      actorRole: user.role,
      actorId: user.id,
      actorName: user.fullName,
      ipAddress: req.clientIp ?? 'unknown',
      deviceId: req.clientDeviceId ?? 'unknown',
      detail: `Successful login as ${user.role}`,
    });

    res.json({ token, user: db.publicUser(user) });
  });

  app.post('/api/auth/register', (req: AuthenticatedRequest, res: Response) => {
    const body = (req.body ?? {}) as {
      username?: string;
      password?: string;
      role?: string;
      fullName?: string;
      phone?: string;
      specialty?: string;
      licenseNo?: string;
    };
    const cleanUsername = body.username?.trim() ?? '';
    const cleanFullName = body.fullName?.trim() ?? '';
    const rawPassword = body.password ?? '';
    if (!cleanUsername || !cleanFullName || rawPassword.length < 4) {
      res.status(400).json({
        error: 'VALIDATION_ERROR',
        message: 'กรุณากรอกชื่อ-นามสกุล, Username และรหัสผ่านอย่างน้อย 4 ตัวอักษร',
      });
      return;
    }
    const targetRole = normalizeStaffRole(body.role);
    if (targetRole === 'patient') {
      res.status(400).json({ error: 'FORBIDDEN_ROLE', message: 'Invalid staff registration role' });
      return;
    }
    if (db.state.users.some((u) => u.username.toLowerCase() === cleanUsername.toLowerCase())) {
      res.status(409).json({
        error: 'USERNAME_EXISTS',
        message: `ชื่อผู้ใช้ "@${cleanUsername}" ถูกใช้งานแล้ว`,
      });
      return;
    }

    const created: UserAccount = {
      id: `user-${Date.now()}-${crypto.randomBytes(2).toString('hex')}`,
      username: cleanUsername,
      passwordHash: hashPassword(rawPassword),
      role: targetRole,
      fullName: cleanFullName,
      phone: body.phone?.trim() ?? '',
      specialty:
        body.specialty?.trim() ||
        (targetRole === 'super_admin' ? 'บริหารโรงพยาบาล' : 'เวชศาสตร์ฟื้นฟูกายภาพบำบัด'),
      licenseNo: body.licenseNo?.trim() || undefined,
      active: true,
      forceLoggedOutAt: null,
      createdAt: new Date().toISOString(),
    };
    db.state.users.push(created);
    const token = issueSessionForUser(created, req);

    db.recordAudit({
      eventType: 'USER_CREATED',
      actorRole: created.role,
      actorId: created.id,
      actorName: created.fullName,
      ipAddress: req.clientIp ?? 'unknown',
      deviceId: req.clientDeviceId ?? 'unknown',
      targetId: created.id,
      detail: `Self-registered ${created.role} account @${created.username}`,
    });

    res.status(201).json({ token, user: db.publicUser(created) });
  });

  app.post('/api/auth/logout', requireAuth(), (req: AuthenticatedRequest, res: Response) => {
    const session = db.state.active_sessions.find((s) => s.token === req.authToken);
    if (session) {
      session.revoked = true;
      db.save();
    }
    res.json({ ok: true });
  });

  app.get('/api/auth/me', requireAuth(), (req: AuthenticatedRequest, res: Response) => {
    res.json({ user: db.publicUser(req.authUser!) });
  });

  app.post('/api/auth/change-password', requireAuth(), (req: AuthenticatedRequest, res: Response) => {
    const { currentPassword, newPassword } = req.body as { currentPassword?: string; newPassword?: string };
    if (!currentPassword || !newPassword || newPassword.length < 6) {
      res.status(400).json({ error: 'INVALID_PASSWORD', message: 'New password must be at least 6 characters' });
      return;
    }
    if (!verifyPassword(currentPassword, req.authUser!.passwordHash)) {
      res.status(400).json({ error: 'WRONG_CURRENT_PASSWORD', message: 'รหัสผ่านเดิมไม่ถูกต้อง' });
      return;
    }
    req.authUser!.passwordHash = hashPassword(newPassword);
    db.save();
    db.recordAudit({
      eventType: 'PASSWORD_CHANGED',
      actorRole: req.authUser!.role,
      actorId: req.authUser!.id,
      actorName: req.authUser!.fullName,
      ipAddress: req.clientIp ?? 'unknown',
      deviceId: req.clientDeviceId ?? 'unknown',
      detail: 'User changed own password',
    });
    res.json({ ok: true });
  });

  // ========================
  // 2. USER & RBAC MANAGEMENT
  // ========================
  app.get(['/api/users', '/api/staff'], requireAuth(['super_admin', 'pt']), (req: AuthenticatedRequest, res: Response) => {
    const users =
      req.authUser!.role === 'super_admin'
        ? db.state.users
        : db.state.users.filter((u) => u.role === 'pt' || u.role === 'patient');
    res.json({ users: users.map((u) => db.publicUser(u)) });
  });

  app.post(['/api/users', '/api/staff'], requireAuth(['super_admin', 'pt']), (req: AuthenticatedRequest, res: Response) => {
    const { username, password, role, fullName, phone, specialty, licenseNo, patientId } = req.body as {
      username?: string;
      password?: string;
      role?: UserRole;
      fullName?: string;
      phone?: string;
      specialty?: string;
      licenseNo?: string;
      patientId?: string;
    };
    const normalizedRole = role ? normalizeStaffRole(role) : undefined;
    if (!username || !password || !normalizedRole || !fullName) {
      res.status(400).json({ error: 'VALIDATION_ERROR' });
      return;
    }
    if (req.authUser!.role === 'pt' && normalizedRole !== 'patient') {
      res.status(403).json({
        error: 'FORBIDDEN',
        message: 'PT cannot create Admin or PT accounts',
      });
      return;
    }
    if (db.state.users.some((u) => u.username.toLowerCase() === username.toLowerCase())) {
      res.status(409).json({ error: 'USERNAME_EXISTS' });
      return;
    }
    const created: UserAccount = {
      id: `user-${Date.now()}-${crypto.randomBytes(2).toString('hex')}`,
      username,
      passwordHash: hashPassword(password),
      role: normalizedRole,
      fullName,
      phone: phone ?? '',
      specialty,
      licenseNo,
      patientId,
      active: true,
      forceLoggedOutAt: null,
      createdAt: new Date().toISOString(),
    };
    db.state.users.push(created);
    db.save();
    db.recordAudit({
      eventType: 'USER_CREATED',
      actorRole: req.authUser!.role,
      actorId: req.authUser!.id,
      actorName: req.authUser!.fullName,
      ipAddress: req.clientIp ?? 'unknown',
      deviceId: req.clientDeviceId ?? 'unknown',
      targetId: created.id,
      detail: `Created ${role} account @${username}`,
    });
    res.status(201).json({ user: db.publicUser(created) });
  });

  app.put('/api/users/:id', requireAuth(), (req: AuthenticatedRequest, res: Response) => {
    const target = db.state.users.find((u) => u.id === req.params.id);
    if (!target) {
      res.status(404).json({ error: 'USER_NOT_FOUND' });
      return;
    }
    const actor = req.authUser!;
    if (actor.role === 'pt' && target.id !== actor.id && target.role !== 'patient') {
      res.status(403).json({ error: 'FORBIDDEN', message: 'PT cannot edit Admin or other PT accounts' });
      return;
    }
    if (actor.role === 'patient' && target.id !== actor.id) {
      res.status(403).json({ error: 'FORBIDDEN' });
      return;
    }

    const { fullName, phone, specialty, licenseNo, avatarUrl, bio, role, active } =
      req.body as Partial<UserAccount>;
    if (fullName !== undefined) target.fullName = fullName;
    if (phone !== undefined) target.phone = phone;
    if (specialty !== undefined && actor.role !== 'patient') target.specialty = specialty;
    if (licenseNo !== undefined && actor.role !== 'patient') target.licenseNo = licenseNo;
    if (avatarUrl !== undefined) target.avatarUrl = avatarUrl;
    if (bio !== undefined) target.bio = bio;
    if (role !== undefined && actor.role === 'super_admin') target.role = role;
    if (active !== undefined && actor.role === 'super_admin') target.active = active;
    db.save();

    db.recordAudit({
      eventType: 'USER_UPDATED',
      actorRole: actor.role,
      actorId: actor.id,
      actorName: actor.fullName,
      ipAddress: req.clientIp ?? 'unknown',
      deviceId: req.clientDeviceId ?? 'unknown',
      targetId: target.id,
      detail: `Updated user @${target.username}`,
    });
    res.json({ user: db.publicUser(target) });
  });

  app.delete('/api/users/:id', requireAuth(['super_admin']), (req: AuthenticatedRequest, res: Response) => {
    const idx = db.state.users.findIndex((u) => u.id === req.params.id);
    if (idx === -1) {
      res.status(404).json({ error: 'USER_NOT_FOUND' });
      return;
    }
    const removed = db.state.users.splice(idx, 1)[0];
    db.save();
    db.recordAudit({
      eventType: 'USER_DELETED',
      actorRole: req.authUser!.role,
      actorId: req.authUser!.id,
      actorName: req.authUser!.fullName,
      ipAddress: req.clientIp ?? 'unknown',
      deviceId: req.clientDeviceId ?? 'unknown',
      targetId: removed.id,
      detail: `Deleted user @${removed.username}`,
    });
    res.json({ deleted: true });
  });

  app.post('/api/users/:id/reset-password', requireAuth(['super_admin', 'pt']), (req: AuthenticatedRequest, res: Response) => {
    const target = db.state.users.find((u) => u.id === req.params.id);
    if (!target) {
      res.status(404).json({ error: 'USER_NOT_FOUND' });
      return;
    }
    if (req.authUser!.role === 'pt' && target.role !== 'patient') {
      res.status(403).json({ error: 'FORBIDDEN', message: 'PT can only reset Patient passwords' });
      return;
    }
    const newPassword = (req.body?.newPassword as string | undefined) || 'Patient@2026';
    target.passwordHash = hashPassword(newPassword);
    db.save();
    db.recordAudit({
      eventType: 'PASSWORD_RESET',
      actorRole: req.authUser!.role,
      actorId: req.authUser!.id,
      actorName: req.authUser!.fullName,
      ipAddress: req.clientIp ?? 'unknown',
      deviceId: req.clientDeviceId ?? 'unknown',
      targetId: target.id,
      detail: `Reset password for @${target.username}`,
    });
    res.json({ ok: true });
  });

  // ========================
  // 3. BIOMETRICS & KIOSK FACE AUTH
  // ========================
  app.post('/api/biometric/enroll', requireAuth(['super_admin', 'pt']), (req: AuthenticatedRequest, res: Response) => {
    const parsed = enrollSchema.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: 'VALIDATION_ERROR', issues: parsed.error.issues });
      return;
    }

    const d = parsed.data;
    const patientId = `pat-${Date.now()}-${crypto.randomBytes(2).toString('hex')}`;
    const assignedPtId =
      req.authUser!.role === 'pt'
        ? req.authUser!.id
        : (d.assignedPtId || db.state.users.find((u) => u.role === 'pt')?.id || req.authUser!.id);

    const nextAppt =
      d.nextAppointmentDate ||
      new Date(Date.now() + 2 * 24 * 60 * 60 * 1000).toISOString().slice(0, 10);

    const patient: PatientRecord = {
      id: patientId,
      hn: d.hn,
      fullName: d.fullName,
      age: d.age,
      phone: d.phone,
      diagnosis: d.diagnosis,
      assignedPtId,
      delegatedToPtId: null,
      delegationNote: null,
      nextAppointmentDate: nextAppt,
      registrationStatus: 'active',
      enrolledAt: new Date().toISOString(),
      biometricHistory: {
        frontal_centroid: d.frontalCentroid,
        overall_centroid: d.overallCentroid,
        raw_frames: d.rawFrames,
      },
    };

    db.state.patients.push(patient);
    db.state.face_templates.push({
      patientId,
      patientName: d.fullName,
      pose: 'frontal_centroid',
      vector: d.frontalCentroid,
    });
    db.state.face_templates.push({
      patientId,
      patientName: d.fullName,
      pose: 'overall_centroid',
      vector: d.overallCentroid,
    });
    for (const f of d.rawFrames) {
      db.state.face_templates.push({
        patientId,
        patientName: d.fullName,
        pose: f.pose,
        vector: f.vector,
      });
    }

    // Create default weekly course
    db.state.patient_courses.push({
      id: `course-${patientId}`,
      patientId,
      title: `${d.fullName} — แผนฟื้นฟูสมรรถภาพ`,
      weeklySchedule: defaultWeeklySchedule(),
      cognitiveModpackId: 'modpack-core-99',
      updatedAt: new Date().toISOString(),
    });

    // Provision patient portal login account
    const patUsername = (d.username || d.hn.replace(/[^a-zA-Z0-9]/g, '')).toLowerCase();
    if (!db.state.users.some((u) => u.username.toLowerCase() === patUsername)) {
      db.state.users.push({
        id: `user-pat-${Date.now()}`,
        username: patUsername,
        passwordHash: hashPassword(d.password || 'Patient@2026'),
        role: 'patient',
        fullName: d.fullName,
        phone: d.phone,
        patientId,
        active: true,
        forceLoggedOutAt: null,
        createdAt: new Date().toISOString(),
      });
    }

    db.save();
    db.recordAudit({
      eventType: 'PATIENT_ENROLLED_10SHOT',
      actorRole: req.authUser!.role,
      actorId: req.authUser!.id,
      actorName: req.authUser!.fullName,
      ipAddress: req.clientIp ?? 'unknown',
      deviceId: req.clientDeviceId ?? 'unknown',
      targetId: patientId,
      detail: `Enrolled patient ${d.fullName} (${d.hn}) with 10-frame biometric scan`,
    });

    res.status(201).json({ patient });
  });

  // Real-time 4-Checkpoint 3D Face Frame Quality & Liveness Assistant
  app.post('/api/biometric/inspect-frame', (req: AuthenticatedRequest, res: Response) => {
    const parsed = inspectFrameSchema.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: 'VALIDATION_ERROR', issues: parsed.error.issues });
      return;
    }

    const inspection = inspectBiometricFrame3D({
      landmarks3D: parsed.data.landmarks3D,
      faceCount: parsed.data.faceCount,
      requestedPose: parsed.data.requestedPose as Detected3DPose | undefined,
      metrics: parsed.data.metrics,
    });

    if (inspection.scan_status === 'SPOOF_DETECTED') {
      db.recordAudit({
        eventType: 'BIOMETRIC_SPOOF_DETECTED',
        actorRole: req.authUser?.role ?? 'patient_kiosk',
        actorId: req.authUser?.id ?? (req.clientDeviceId ?? 'kiosk'),
        actorName: req.authUser?.fullName ?? 'Kiosk Face Sensor',
        ipAddress: req.clientIp ?? 'unknown',
        deviceId: req.clientDeviceId ?? 'unknown',
        detail: `Blocked spoof attempt (${inspection.error_code})`,
      });
    }

    res.json(inspection);
  });

  // Public Kiosk Face Verify (returns ONLY matched patient + today's routine; never exposes other patients)
  app.post('/api/biometric/verify', (req: AuthenticatedRequest, res: Response) => {
    if (!db.state.system_settings.aiFaceScanEnabled) {
      res.status(503).json({ error: 'FACE_SCAN_DISABLED', message: 'ระบบสแกนใบหน้าถูกปิดชั่วคราวโดยผู้ดูแลระบบ' });
      return;
    }
    const parsed = verifySchema.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: 'VALIDATION_ERROR', issues: parsed.error.issues });
      return;
    }

    const totalPatients = db.state.patients.filter((p) => p.registrationStatus === 'active').length;
    const result = verifyHybridBiometricIdentity(parsed.data.probeVector, db.state.face_templates, {
      totalEnrolledPatients: totalPatients,
      thresholds: {
        minRawCosMulti: db.state.system_settings.aiFaceThreshold,
      },
    });

    if (!result.matched || !result.patientId) {
      res.status(401).json(result);
      return;
    }

    const patient = db.state.patients.find((p) => p.id === result.patientId) ?? null;
    const course = db.state.patient_courses.find((c) => c.patientId === result.patientId) ?? null;
    const todayIdx = new Date().getDay();
    const todayRoutine = course?.weeklySchedule.find((d) => d.dayIndex === todayIdx) ?? null;

    db.recordAudit({
      eventType: 'KIOSK_FACE_LOGIN',
      actorRole: 'patient_kiosk',
      actorId: result.patientId,
      actorName: result.patientName ?? 'Patient',
      ipAddress: req.clientIp ?? 'unknown',
      deviceId: req.clientDeviceId ?? 'unknown',
      targetId: result.patientId,
      detail: `Kiosk Face ID matched (${result.confidencePercent}%)`,
    });

    res.json({
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
  });

  // ========================
  // 4. PATIENTS, DELEGATION & APPEND-ONLY NOTES
  // ========================
  app.get('/api/patients', requireAuth(), (req: AuthenticatedRequest, res: Response) => {
    const actor = req.authUser!;
    let list = db.state.patients;
    if (actor.role === 'pt') {
      list = list.filter((p) => p.assignedPtId === actor.id || p.delegatedToPtId === actor.id);
    } else if (actor.role === 'patient') {
      list = list.filter((p) => p.id === actor.patientId);
    }
    res.json({ patients: list });
  });

  app.put('/api/patients/:id', requireAuth(), (req: AuthenticatedRequest, res: Response) => {
    const patient = db.state.patients.find((p) => p.id === req.params.id);
    if (!patient) {
      res.status(404).json({ error: 'PATIENT_NOT_FOUND' });
      return;
    }
    const actor = req.authUser!;
    if (actor.role === 'pt' && patient.assignedPtId !== actor.id) {
      res.status(403).json({ error: 'FORBIDDEN', message: 'PT can only edit own assigned patients' });
      return;
    }
    if (actor.role === 'patient' && patient.id !== actor.patientId) {
      res.status(403).json({ error: 'FORBIDDEN' });
      return;
    }

    const { fullName, age, phone, diagnosis, assignedPtId, nextAppointmentDate } = req.body as Partial<PatientRecord>;
    if (actor.role === 'patient') {
      if (phone !== undefined) patient.phone = phone;
    } else {
      if (fullName !== undefined) patient.fullName = fullName;
      if (age !== undefined) patient.age = age;
      if (phone !== undefined) patient.phone = phone;
      if (diagnosis !== undefined) patient.diagnosis = diagnosis;
      if (nextAppointmentDate !== undefined) patient.nextAppointmentDate = nextAppointmentDate;
      if (assignedPtId !== undefined && actor.role === 'super_admin') patient.assignedPtId = assignedPtId;
    }
    db.save();
    db.recordAudit({
      eventType: 'PATIENT_UPDATED',
      actorRole: actor.role,
      actorId: actor.id,
      actorName: actor.fullName,
      ipAddress: req.clientIp ?? 'unknown',
      deviceId: req.clientDeviceId ?? 'unknown',
      targetId: patient.id,
      detail: `Updated patient ${patient.fullName}`,
    });
    res.json({ patient });
  });

  app.delete('/api/patients/:id', requireAuth(['super_admin', 'pt']), (req: AuthenticatedRequest, res: Response) => {
    const patient = db.state.patients.find((p) => p.id === req.params.id);
    if (!patient) {
      res.status(404).json({ error: 'PATIENT_NOT_FOUND' });
      return;
    }
    const actor = req.authUser!;
    if (actor.role === 'pt' && patient.assignedPtId !== actor.id) {
      res.status(403).json({ error: 'FORBIDDEN', message: 'PT can only delete own assigned patients' });
      return;
    }

    const pid = patient.id;
    db.state.patients = db.state.patients.filter((p) => p.id !== pid);
    db.state.face_templates = db.state.face_templates.filter((t) => t.patientId !== pid);
    db.state.patient_courses = db.state.patient_courses.filter((c) => c.patientId !== pid);
    db.state.rehab_sessions = db.state.rehab_sessions.filter((s) => s.patientId !== pid);
    db.state.medical_notes = db.state.medical_notes.filter((n) => n.patientId !== pid);
    db.state.treatment_reports = db.state.treatment_reports.filter((r) => r.patientId !== pid);
    db.state.users = db.state.users.filter((u) => u.patientId !== pid);
    db.save();

    db.recordAudit({
      eventType: 'PATIENT_CASCADING_PURGE',
      actorRole: actor.role,
      actorId: actor.id,
      actorName: actor.fullName,
      ipAddress: req.clientIp ?? 'unknown',
      deviceId: req.clientDeviceId ?? 'unknown',
      targetId: pid,
      detail: `Cascading purge for patient ${patient.fullName} (${patient.hn})`,
    });

    res.json({ deleted: true, patientId: pid, cascadingPurge: true });
  });

  // PT Special Permission: [ฝากเคสชั่วคราว]
  app.post('/api/patients/:id/delegate', requireAuth(['super_admin', 'pt']), (req: AuthenticatedRequest, res: Response) => {
    const patient = db.state.patients.find((p) => p.id === req.params.id);
    if (!patient) {
      res.status(404).json({ error: 'PATIENT_NOT_FOUND' });
      return;
    }
    const actor = req.authUser!;
    if (actor.role === 'pt' && patient.assignedPtId !== actor.id && patient.delegatedToPtId !== actor.id) {
      res.status(403).json({ error: 'FORBIDDEN', message: 'Only primary PT can delegate case' });
      return;
    }
    const { targetPtId, note, delegationDate } = req.body as {
      targetPtId?: string | null;
      note?: string;
      delegationDate?: string;
    };
    patient.delegatedToPtId = targetPtId || null;
    patient.delegationNote = targetPtId ? (note || null) : null;
    patient.delegationDate = targetPtId
      ? (delegationDate || new Date().toISOString().slice(0, 10))
      : null;
    db.save();

    db.recordAudit({
      eventType: 'CASE_DELEGATED',
      actorRole: actor.role,
      actorId: actor.id,
      actorName: actor.fullName,
      ipAddress: req.clientIp ?? 'unknown',
      deviceId: req.clientDeviceId ?? 'unknown',
      targetId: patient.id,
      detail: targetPtId
        ? `Delegated case ${patient.fullName} to PT ${targetPtId} on ${patient.delegationDate}`
        : `Reclaimed case ${patient.fullName}`,
    });
    res.json({ patient });
  });

  // Append-Only Medical Notes
  app.get('/api/patients/:id/notes', requireAuth(), (req: AuthenticatedRequest, res: Response) => {
    const notes = db.state.medical_notes.filter((n) => n.patientId === req.params.id);
    res.json({ notes });
  });

  app.post('/api/patients/:id/notes', requireAuth(['super_admin', 'pt']), (req: AuthenticatedRequest, res: Response) => {
    const content = (req.body?.content as string | undefined)?.trim();
    if (!content) {
      res.status(400).json({ error: 'EMPTY_NOTE' });
      return;
    }
    const note = {
      id: `note-${Date.now()}-${crypto.randomBytes(2).toString('hex')}`,
      patientId: String(req.params.id),
      authorId: req.authUser!.id,
      authorName: req.authUser!.fullName,
      content,
      createdAt: new Date().toISOString(),
    };
    db.state.medical_notes.push(note);
    db.save();
    db.recordAudit({
      eventType: 'MEDICAL_NOTE_APPENDED',
      actorRole: req.authUser!.role,
      actorId: req.authUser!.id,
      actorName: req.authUser!.fullName,
      ipAddress: req.clientIp ?? 'unknown',
      deviceId: req.clientDeviceId ?? 'unknown',
      targetId: note.patientId,
      detail: `Appended medical note to patient ${note.patientId}`,
    });
    res.status(201).json({ note });
  });

  app.put('/api/patients/:id/notes/:noteId', requireAuth(['super_admin', 'pt']), (req: AuthenticatedRequest, res: Response) => {
    if (req.authUser!.role === 'pt') {
      res.status(403).json({
        error: 'APPEND_ONLY_NOTES',
        message: 'นักกายภาพบำบัดไม่สามารถแก้ไขโน้ตประวัติเก่าที่เคยเขียนไปแล้วได้',
      });
      return;
    }
    const note = db.state.medical_notes.find((n) => n.id === req.params.noteId);
    if (!note) {
      res.status(404).json({ error: 'NOTE_NOT_FOUND' });
      return;
    }
    note.content = String(req.body?.content ?? note.content);
    db.save();
    res.json({ note });
  });

  app.delete('/api/patients/:id/notes/:noteId', requireAuth(['super_admin', 'pt']), (req: AuthenticatedRequest, res: Response) => {
    if (req.authUser!.role === 'pt') {
      res.status(403).json({
        error: 'APPEND_ONLY_NOTES',
        message: 'นักกายภาพบำบัดไม่สามารถลบโน้ตประวัติเก่าที่เคยเขียนไปแล้วได้',
      });
      return;
    }
    db.state.medical_notes = db.state.medical_notes.filter((n) => n.id !== req.params.noteId);
    db.save();
    res.json({ deleted: true });
  });

  // ========================
  // 5. COURSES, SESSIONS & REPORTS
  // ========================
  app.get('/api/courses/:patientId', requireAuth(), (req: AuthenticatedRequest, res: Response) => {
    const course = db.state.patient_courses.find((c) => c.patientId === req.params.patientId);
    if (!course) {
      res.status(404).json({ error: 'COURSE_NOT_FOUND' });
      return;
    }
    res.json({ course });
  });

  app.put('/api/courses/:patientId', requireAuth(['super_admin', 'pt']), (req: AuthenticatedRequest, res: Response) => {
    const parsed = courseUpdateSchema.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: 'VALIDATION_ERROR', issues: parsed.error.issues });
      return;
    }
    // Enforce PT restriction when AI Pose Detection is still locked ("อยู่ระหว่างพัฒนา")
    if (
      req.authUser!.role === 'pt' &&
      !db.state.system_settings.aiPoseDetectionEnabled &&
      parsed.data.weeklySchedule.some((d) => d.dayType === 'training' && d.activityType === 'physio_exercise')
    ) {
      res.status(403).json({
        error: 'PHYSIO_LOCKED_UNDER_DEVELOPMENT',
        message: 'ท่ากายภาพบำบัดถูกล็อกสถานะ "อยู่ระหว่างพัฒนา" PT สามารถสั่งจ่ายตารางได้เฉพาะ "มินิเกม"',
      });
      return;
    }

    const pid = String(req.params.patientId);
    const existing = db.state.patient_courses.find((c) => c.patientId === pid);
    if (existing) {
      existing.title = parsed.data.title;
      existing.cognitiveModpackId = parsed.data.cognitiveModpackId;
      existing.weeklySchedule = parsed.data.weeklySchedule;
      existing.updatedAt = new Date().toISOString();
      db.save();
      res.json({ course: existing });
      return;
    }

    const created = {
      id: `course-${pid}`,
      patientId: pid,
      title: parsed.data.title,
      cognitiveModpackId: parsed.data.cognitiveModpackId,
      weeklySchedule: parsed.data.weeklySchedule,
      updatedAt: new Date().toISOString(),
    };
    db.state.patient_courses.push(created);
    db.save();
    res.json({ course: created });
  });

  app.get('/api/sessions', requireAuth(), (req: AuthenticatedRequest, res: Response) => {
    const actor = req.authUser!;
    const pid = typeof req.query.patientId === 'string' ? req.query.patientId : undefined;
    let list = db.state.rehab_sessions;
    if (actor.role === 'patient') {
      list = list.filter((s) => s.patientId === actor.patientId);
    } else if (pid) {
      list = list.filter((s) => s.patientId === pid);
    }
    res.json({ sessions: list });
  });

  // Kiosk & Patient Portal record rehab/minigame session
  app.post('/api/sessions', (req: AuthenticatedRequest, res: Response) => {
    const parsed = sessionSchema.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: 'VALIDATION_ERROR', issues: parsed.error.issues });
      return;
    }
    const session = {
      id: `sess-${Date.now()}-${crypto.randomBytes(2).toString('hex')}`,
      ...parsed.data,
      createdAt: new Date().toISOString(),
    };
    db.state.rehab_sessions.unshift(session);
    db.save();
    res.status(201).json({ session });
  });

  // Clinical Biomechanics AI Agent Telemetry Evaluator (Strict JSON Output)
  app.post('/api/biomechanics/evaluate', (req: AuthenticatedRequest, res: Response) => {
    const parsed = biomechanicsEvaluateSchema.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: 'VALIDATION_ERROR', issues: parsed.error.issues });
      return;
    }

    const { counterState, ...packet } = parsed.data;
    const { decision } = evaluateClinicalBiomechanicsPacket(
      packet,
      counterState ?? { valid_reps: 0, invalid_reps: 0, hold_elapsed_sec: 0 },
    );

    if (decision.kiosk_state === 'EMERGENCY_HALT') {
      db.recordAudit({
        eventType: 'BIOMECHANICS_EMERGENCY_HALT',
        actorRole: req.authUser?.role ?? 'patient_kiosk',
        actorId: packet.auth_status.patient_id ?? req.authUser?.id ?? (req.clientDeviceId ?? 'kiosk'),
        actorName: req.authUser?.fullName ?? 'Kiosk Biomechanics AI Agent',
        ipAddress: req.clientIp ?? 'unknown',
        deviceId: req.clientDeviceId ?? 'unknown',
        detail: `Emergency halt during ${packet.exercise_prescription.exercise_name ?? packet.exercise_prescription.exercise_id ?? 'rehab'}: ${decision.tts_voice_th}`,
      });
    }

    res.json(decision);
  });

  app.get('/api/reports', requireAuth(), (req: AuthenticatedRequest, res: Response) => {
    const actor = req.authUser!;
    let list = db.state.treatment_reports;
    if (actor.role === 'patient') {
      list = list.filter((r) => r.patientId === actor.patientId);
    } else if (actor.role === 'pt') {
      const allowedPatientIds = new Set(
        db.state.patients
          .filter((p) => p.assignedPtId === actor.id || p.delegatedToPtId === actor.id)
          .map((p) => p.id),
      );
      list = list.filter((r) => allowedPatientIds.has(r.patientId));
    }
    res.json({ reports: list });
  });

  app.post('/api/reports', requireAuth(['super_admin', 'pt']), (req: AuthenticatedRequest, res: Response) => {
    const { patientId, summary, aiRecommendation, scoreSummary } = req.body as {
      patientId?: string;
      summary?: string;
      aiRecommendation?: string;
      scoreSummary?: string;
    };
    if (!patientId || !summary) {
      res.status(400).json({ error: 'VALIDATION_ERROR' });
      return;
    }
    const now = new Date().toISOString();
    const report = {
      id: `rep-${Date.now()}`,
      patientId,
      ptId: req.authUser!.id,
      ptName: req.authUser!.fullName,
      summary,
      aiRecommendation: aiRecommendation ?? 'Continue daily cognitive & guided rehab routine.',
      scoreSummary: scoreSummary ?? 'Stable progress',
      createdAt: now,
      updatedAt: now,
    };
    db.state.treatment_reports.unshift(report);
    db.save();
    res.status(201).json({ report });
  });

  app.put('/api/reports/:id', requireAuth(['super_admin', 'pt']), (req: AuthenticatedRequest, res: Response) => {
    const rep = db.state.treatment_reports.find((r) => r.id === req.params.id);
    if (!rep) {
      res.status(404).json({ error: 'REPORT_NOT_FOUND' });
      return;
    }
    if (req.body.summary) rep.summary = String(req.body.summary);
    if (req.body.aiRecommendation) rep.aiRecommendation = String(req.body.aiRecommendation);
    if (req.body.scoreSummary) rep.scoreSummary = String(req.body.scoreSummary);
    rep.updatedAt = new Date().toISOString();
    db.save();
    res.json({ report: rep });
  });

  // ========================
  // 6. EXERCISES & MODPACKS (Scoped Ownership for PT vs Full CRUD for Super Admin)
  // ========================
  app.get('/api/exercises', (_req: Request, res: Response) => {
    res.json({ exercises: db.state.exercises });
  });

  app.post('/api/exercises', requireAuth(['super_admin', 'pt']), (req: AuthenticatedRequest, res: Response) => {
    const ex = req.body;
    if (!ex?.code || !ex?.name) {
      res.status(400).json({ error: 'VALIDATION_ERROR' });
      return;
    }
    const created = {
      ...ex,
      id: ex.id || `ex-custom-${Date.now()}`,
      createdBy: req.authUser!.role === 'super_admin' ? 'system' : req.authUser!.id,
    };
    db.state.exercises.push(created);
    db.save();
    res.status(201).json({ exercise: created });
  });

  app.delete('/api/exercises/:id', requireAuth(['super_admin', 'pt']), (req: AuthenticatedRequest, res: Response) => {
    const target = db.state.exercises.find((e) => e.id === req.params.id);
    if (!target) {
      res.status(404).json({ error: 'NOT_FOUND' });
      return;
    }
    if (req.authUser!.role === 'pt' && target.createdBy !== req.authUser!.id) {
      res.status(403).json({
        error: 'FORBIDDEN_GLOBAL_LIBRARY',
        message: 'ห้ามลบหรือแก้ไขคลังท่าฝึกส่วนกลางของโรงพยาบาล',
      });
      return;
    }
    db.state.exercises = db.state.exercises.filter((e) => e.id !== req.params.id);
    db.save();
    res.json({ deleted: true });
  });

  app.get('/api/modpacks', (_req: Request, res: Response) => {
    res.json({ modpacks: db.state.modpacks });
  });

  app.post('/api/modpacks', requireAuth(['super_admin', 'pt']), (req: AuthenticatedRequest, res: Response) => {
    const { name, categories, questionCount } = req.body as {
      name?: string;
      categories?: string[];
      questionCount?: number;
    };
    if (!name) {
      res.status(400).json({ error: 'VALIDATION_ERROR' });
      return;
    }
    const created = {
      id: `modpack-${Date.now()}`,
      name,
      categories: categories ?? ['food_fruit', 'safety_routine'],
      questionCount: questionCount ?? 20,
      createdBy: req.authUser!.role === 'super_admin' ? 'system' : req.authUser!.id,
      active: true,
    };
    db.state.modpacks.push(created);
    db.save();
    res.status(201).json({ modpack: created });
  });

  app.delete('/api/modpacks/:id', requireAuth(['super_admin', 'pt']), (req: AuthenticatedRequest, res: Response) => {
    const target = db.state.modpacks.find((m) => m.id === req.params.id);
    if (!target) {
      res.status(404).json({ error: 'NOT_FOUND' });
      return;
    }
    if (req.authUser!.role === 'pt' && target.createdBy !== req.authUser!.id) {
      res.status(403).json({
        error: 'FORBIDDEN_GLOBAL_LIBRARY',
        message: 'ห้ามลบหรือแก้ไขชุดมินิเกมส่วนกลางของโรงพยาบาล',
      });
      return;
    }
    db.state.modpacks = db.state.modpacks.filter((m) => m.id !== req.params.id);
    db.save();
    res.json({ deleted: true });
  });

  // ========================
  // 7. STAFF CALLS, SECURITY, AUDIT LOGS & SYSTEM SETTINGS
  // ========================
  app.get('/api/staff-calls', requireAuth(['super_admin', 'pt']), (_req: Request, res: Response) => {
    res.json({ calls: db.state.staff_calls });
  });

  app.post('/api/staff-calls', (req: AuthenticatedRequest, res: Response) => {
    const parsed = staffCallSchema.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: 'VALIDATION_ERROR', issues: parsed.error.issues });
      return;
    }
    const call = {
      id: `call-${Date.now()}`,
      kioskId: parsed.data.kioskId,
      patientId: parsed.data.patientId ?? null,
      reason: parsed.data.reason,
      status: 'open' as const,
      createdAt: new Date().toISOString(),
    };
    db.state.staff_calls.unshift(call);
    db.save();
    db.recordAudit({
      eventType: 'KIOSK_STAFF_CALL',
      actorRole: 'patient_kiosk',
      actorId: parsed.data.kioskId,
      actorName: parsed.data.kioskId,
      ipAddress: req.clientIp ?? 'unknown',
      deviceId: req.clientDeviceId ?? 'unknown',
      detail: parsed.data.reason,
    });
    res.status(201).json({ call });
  });

  app.patch('/api/staff-calls/:id/resolve', requireAuth(['super_admin', 'pt']), (req: Request, res: Response) => {
    const target = db.state.staff_calls.find((c) => c.id === req.params.id);
    if (target) {
      target.status = 'resolved';
      db.save();
    }
    res.json({ ok: true });
  });

  // Strictly Super Admin Only (PT gets 403 Forbidden)
  app.get('/api/audit-logs', requireAuth(['super_admin']), (_req: Request, res: Response) => {
    res.json({
      logs: db.state.audit_logs,
      bannedTargets: db.state.banned_targets,
      activeSessions: db.state.active_sessions.filter((s) => !s.revoked),
    });
  });

  app.post('/api/security/ban', requireAuth(['super_admin']), (req: AuthenticatedRequest, res: Response) => {
    const { targetType, value, reason } = req.body as {
      targetType?: 'ip' | 'device';
      value?: string;
      reason?: string;
    };
    if (!targetType || !value) {
      res.status(400).json({ error: 'VALIDATION_ERROR' });
      return;
    }
    const ban = {
      id: `ban-${Date.now()}`,
      targetType,
      value: value.trim(),
      reason: reason?.trim() || 'Banned by Hospital Super Admin',
      bannedBy: req.authUser!.fullName,
      createdAt: new Date().toISOString(),
    };
    db.state.banned_targets.unshift(ban);
    db.save();
    db.recordAudit({
      eventType: 'SECURITY_TARGET_BANNED',
      actorRole: req.authUser!.role,
      actorId: req.authUser!.id,
      actorName: req.authUser!.fullName,
      ipAddress: req.clientIp ?? 'unknown',
      deviceId: req.clientDeviceId ?? 'unknown',
      detail: `Banned ${targetType}: ${value}`,
    });
    res.status(201).json({ ban });
  });

  app.delete('/api/security/ban/:id', requireAuth(['super_admin']), (req: Request, res: Response) => {
    db.state.banned_targets = db.state.banned_targets.filter((b) => b.id !== req.params.id);
    db.save();
    res.json({ ok: true });
  });

  app.post('/api/security/force-logout/:userId', requireAuth(['super_admin']), (req: AuthenticatedRequest, res: Response) => {
    const user = db.state.users.find((u) => u.id === req.params.userId);
    if (!user) {
      res.status(404).json({ error: 'USER_NOT_FOUND' });
      return;
    }
    user.forceLoggedOutAt = new Date().toISOString();
    for (const s of db.state.active_sessions) {
      if (s.userId === user.id) s.revoked = true;
    }
    db.save();
    db.recordAudit({
      eventType: 'FORCE_LOGOUT',
      actorRole: req.authUser!.role,
      actorId: req.authUser!.id,
      actorName: req.authUser!.fullName,
      ipAddress: req.clientIp ?? 'unknown',
      deviceId: req.clientDeviceId ?? 'unknown',
      targetId: user.id,
      detail: `Forced logout for @${user.username}`,
    });
    res.json({ ok: true });
  });

  app.get('/api/settings', (_req: Request, res: Response) => {
    res.json({ settings: db.state.system_settings, kiosks: db.state.kiosks });
  });

  app.put('/api/settings', requireAuth(['super_admin']), (req: AuthenticatedRequest, res: Response) => {
    const patch = req.body as Partial<typeof db.state.system_settings>;
    db.state.system_settings = {
      ...db.state.system_settings,
      ...patch,
    };
    db.save();
    db.recordAudit({
      eventType: 'SYSTEM_SETTINGS_UPDATED',
      actorRole: req.authUser!.role,
      actorId: req.authUser!.id,
      actorName: req.authUser!.fullName,
      ipAddress: req.clientIp ?? 'unknown',
      deviceId: req.clientDeviceId ?? 'unknown',
      detail: JSON.stringify(db.state.system_settings),
    });
    res.json({ settings: db.state.system_settings });
  });

  return app;
}
