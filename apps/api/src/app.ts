import crypto from 'node:crypto';
import cors from 'cors';
import express, { type Express, type NextFunction, type Request, type Response } from 'express';
import rateLimit from 'express-rate-limit';
import helmet from 'helmet';
import { z } from 'zod';
import {
  ACTIVE_BIOMETRIC_DIM,
  ACTIVE_BIOMETRIC_PIPELINE_VERSION,
  FACE_EMBEDDING_DIM,
  STRICT_RAW_COSINE_THRESHOLD,
  STRICT_RELU6_CENTERED_COSINE_THRESHOLD,
  STRICT_RELU6_RAW_COSINE_THRESHOLD,
  computeZeroMeanCenteredVectorStrict,
  cosineSimilarity,
  isNonNegativeRelu6Vector,
  l2Normalize,
  verifyHybridBiometricIdentity,
  type BiometricVerifyReasonCode,
} from '@drtechcares/smart-biometric-core';
import {
  CLINICAL_EXERCISES,
  evaluateClinicalBiomechanicsPacket,
  isDeprecatedExerciseRecord,
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
  isAdminRole,
  resolveTodayPatientSchedule,
  type BiometricLibraryStatsResponse,
  type BiometricVerifyDiagnostics,
  type DuplicateFaceCheckResponse,
} from '../../../packages/api-client/src/index';
import {
  ClinicalDatabase,
  hashPassword,
  verifyPassword,
  type DailyScheduleDay,
  type PatientRecord,
  type UserAccount,
  type UserRole,
} from './db';

const vectorSchema = z
  .array(z.number().finite())
  .refine((v) => v.length >= 128 && v.length <= 2048, {
    message: 'Embedding vector must contain between 128 and 2048 elements',
  });

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
  angleEmbeddings: z
    .object({
      center: vectorSchema,
      left: vectorSchema,
      right: vectorSchema,
      up: vectorSchema,
      down: vectorSchema,
    })
    .optional(),
  exemplarEmbeddings: z.array(vectorSchema).max(24).optional(),
  biometricEngine: z.enum(['human-faceres-1024d', 'arcface-onnx', 'hybrid-512d']).optional(),
  rawFrames: z
    .array(
      z.object({
        pose: z.enum(['front', 'left', 'right', 'nod', 'center', 'up', 'down']),
        vector: vectorSchema,
        qualityScore: z.number().positive(),
      }),
    )
    .min(1)
    .max(120),
});

const verifySchema = z.object({
  probeVector: vectorSchema,
  frameVectors: z.array(vectorSchema).max(6).optional(),
  frameTimestamps: z.array(z.number().finite()).max(6).optional(),
});

const checkDuplicateFaceSchema = z.object({
  masterEmbedding: vectorSchema.optional(),
  centerEmbedding: vectorSchema.optional(),
  excludePatientId: z.string().optional(),
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
    const actorIsAdmin = isAdminRole(actor.role);
    if (actor.role === 'pt' && target.id !== actor.id && target.role !== 'patient') {
      res.status(403).json({ error: 'FORBIDDEN', message: 'PT cannot edit Admin or other PT accounts' });
      return;
    }
    if (actor.role === 'patient' && target.id !== actor.id) {
      res.status(403).json({ error: 'FORBIDDEN' });
      return;
    }

    const { fullName, phone, specialty, licenseNo, license_number, license_no, avatarUrl, bio, role, active } =
      req.body as Partial<UserAccount> & { license_number?: string; license_no?: string };
    if (fullName !== undefined) target.fullName = fullName;
    if (phone !== undefined) target.phone = phone;
    if (specialty !== undefined && actor.role !== 'patient') target.specialty = specialty;
    // Strict RBAC Field Lock: Only Admin may modify licenseNo / license_number; PT keeps existing DB licenseNo.
    const requestedLicense = licenseNo ?? license_number ?? license_no;
    if (requestedLicense !== undefined && actorIsAdmin) {
      target.licenseNo = String(requestedLicense).trim() || undefined;
    }
    if (avatarUrl !== undefined) target.avatarUrl = avatarUrl;
    if (bio !== undefined) target.bio = bio;
    if (role !== undefined && actorIsAdmin) target.role = role;
    if (active !== undefined && actorIsAdmin) target.active = active;
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
    const paramId = Array.isArray(req.params.id) ? req.params.id[0] : req.params.id;
    let rawKey = String(paramId || '');
    try {
      rawKey = decodeURIComponent(rawKey);
    } catch {
      // Keep raw segment if malformed percent encoding
    }
    rawKey = rawKey.trim();
    const keyLower = rawKey.replace(/^@/, '').toLowerCase();
    const idx = db.state.users.findIndex(
      (u) => u.id === rawKey || (keyLower.length > 0 && u.username.toLowerCase() === keyLower),
    );
    const removed = idx !== -1 ? db.state.users.splice(idx, 1)[0] : undefined;
    const effectiveId = removed?.id ?? rawKey;
    for (const p of db.state.patients) {
      if (p.assignedPtId === effectiveId || (rawKey && p.assignedPtId === rawKey)) {
        p.assignedPtId = '';
      }
      if (p.delegatedToPtId === effectiveId || (rawKey && p.delegatedToPtId === rawKey)) {
        p.delegatedToPtId = null;
        p.delegationNote = null;
        p.delegationDate = null;
      }
    }
    db.save();
    db.recordAudit({
      eventType: 'USER_DELETED',
      actorRole: req.authUser!.role,
      actorId: req.authUser!.id,
      actorName: req.authUser!.fullName,
      ipAddress: req.clientIp ?? 'unknown',
      deviceId: req.clientDeviceId ?? 'unknown',
      targetId: effectiveId || 'unknown',
      detail: `Deleted user @${removed?.username ?? rawKey}`,
    });
    res.json({ ok: true, deleted: true, id: effectiveId });
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
  const findLocalDuplicateFace = (
    masterVec: readonly number[],
    centerVec?: readonly number[],
    excludePatientId?: string,
  ): { id: string; hn: string; fullName: string; similarityPercent: number } | null => {
    const probeMaster = l2Normalize(masterVec);
    const probeCenter =
      centerVec && centerVec.length === probeMaster.length ? l2Normalize(centerVec) : probeMaster;
    const expectedDim = probeMaster.length;
    const isRelu6 = isNonNegativeRelu6Vector(probeMaster) || expectedDim >= ACTIVE_BIOMETRIC_DIM;
    const probeMasterC = computeZeroMeanCenteredVectorStrict(probeMaster);
    const probeCenterC = computeZeroMeanCenteredVectorStrict(probeCenter);
    const excludeLo = (excludePatientId ?? '').trim().toLowerCase();
    let best: { id: string; hn: string; fullName: string; raw: number } | null = null;

    for (const p of db.state.patients) {
      const regLo = String(p.registrationStatus ?? '').trim().toLowerCase();
      if (regLo !== '' && regLo !== 'active') continue;
      if (
        excludeLo &&
        (p.id.trim().toLowerCase() === excludeLo || p.hn.trim().toLowerCase() === excludeLo)
      ) {
        continue;
      }
      const existingRaw =
        p.biometricHistory?.overall_centroid ??
        p.biometricHistory?.frontal_centroid ??
        db.state.face_templates.find((t) => t.patientId === p.id)?.vector;
      if (!Array.isArray(existingRaw) || existingRaw.length !== expectedDim) continue;
      const existingMaster = l2Normalize(existingRaw);
      const existingCenterRaw = p.biometricHistory?.frontal_centroid ?? existingRaw;
      const existingCenter =
        existingCenterRaw.length === expectedDim ? l2Normalize(existingCenterRaw) : existingMaster;
      const bestRaw = Math.max(
        cosineSimilarity(probeMaster, existingMaster),
        cosineSimilarity(probeCenter, existingCenter),
        cosineSimilarity(probeCenter, existingMaster),
      );
      const exMasterC = computeZeroMeanCenteredVectorStrict(existingMaster);
      const exCenterC = computeZeroMeanCenteredVectorStrict(existingCenter);
      const bestCentered = Math.max(
        probeMasterC && exMasterC ? cosineSimilarity(probeMasterC, exMasterC) : 0,
        probeCenterC && exCenterC ? cosineSimilarity(probeCenterC, exCenterC) : 0,
      );
      const dup = isRelu6
        ? bestRaw >= STRICT_RELU6_RAW_COSINE_THRESHOLD &&
          bestCentered >= STRICT_RELU6_CENTERED_COSINE_THRESHOLD
        : bestRaw >= 0.88 && bestCentered >= 0.82;
      if (dup && (!best || bestRaw > best.raw)) {
        best = { id: p.id, hn: p.hn, fullName: p.fullName, raw: bestRaw };
      }
    }
    return best
      ? {
          id: best.id,
          hn: best.hn,
          fullName: best.fullName,
          similarityPercent: Number((Math.max(0, Math.min(1, best.raw)) * 100).toFixed(1)),
        }
      : null;
  };

  app.post('/api/biometric/check-duplicate', (req: AuthenticatedRequest, res: Response) => {
    const parsed = checkDuplicateFaceSchema.safeParse(req.body);
    const primary = parsed.success ? (parsed.data.masterEmbedding ?? parsed.data.centerEmbedding) : null;
    if (!parsed.success || !primary) {
      res.status(400).json({ error: 'VALIDATION_ERROR' });
      return;
    }
    const dup = findLocalDuplicateFace(
      primary,
      parsed.data.centerEmbedding,
      parsed.data.excludePatientId,
    );
    if (!dup) {
      const okPayload: DuplicateFaceCheckResponse = { duplicate: false };
      res.json(okPayload);
      return;
    }
    const isStaff = req.authUser?.role === 'super_admin' || req.authUser?.role === 'pt';
    if (isStaff) {
      const staffPayload: DuplicateFaceCheckResponse = {
        duplicate: true,
        similarityPercent: dup.similarityPercent,
        existingPatient: { id: dup.id, hn: dup.hn, fullName: dup.fullName },
        message: `ตรวจพบใบหน้าซ้ำกับบัญชีที่มีอยู่แล้วในระบบ: ${dup.fullName} (${dup.hn}) (ความคล้าย ${dup.similarityPercent.toFixed(1)}%) กรุณาอัปเดตใบหน้าในบัญชีเดิมแทนการสร้างบัญชีใหม่`,
      };
      res.json(staffPayload);
      return;
    }
    const publicPayload: DuplicateFaceCheckResponse = {
      duplicate: true,
      similarityPercent: dup.similarityPercent,
      message: 'ใบหน้านี้มีบัญชีอยู่ในระบบแล้ว กรุณากลับไปสแกนหน้าเข้าสู่ระบบ หรือติดต่อเจ้าหน้าที่',
    };
    res.json(publicPayload);
  });

  app.post('/api/biometric/enroll', requireAuth(['super_admin', 'pt']), (req: AuthenticatedRequest, res: Response) => {
    const parsed = enrollSchema.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: 'VALIDATION_ERROR', issues: parsed.error.issues });
      return;
    }

    const d = parsed.data;
    const duplicate = findLocalDuplicateFace(d.overallCentroid, d.frontalCentroid);
    if (duplicate) {
      res.status(409).json({
        error: 'DUPLICATE_FACE_ENROLLMENT',
        similarityPercent: duplicate.similarityPercent,
        existingPatient: { id: duplicate.id, hn: duplicate.hn, fullName: duplicate.fullName },
        message: `ตรวจพบใบหน้าซ้ำกับบัญชีที่มีอยู่แล้วในระบบ: ${duplicate.fullName} (${duplicate.hn}) (ความคล้าย ${duplicate.similarityPercent.toFixed(1)}%) ห้ามสร้างบัญชีซ้ำ กรุณาอัปเดตใบหน้าในบัญชีเดิมแทน`,
      });
      return;
    }
    const patientId = `pat-${Date.now()}-${crypto.randomBytes(2).toString('hex')}`;
    const assignedPtId =
      req.authUser!.role === 'pt'
        ? req.authUser!.id
        : (d.assignedPtId || db.state.users.find((u) => u.role === 'pt')?.id || req.authUser!.id);

    const nextAppt =
      d.nextAppointmentDate ||
      new Date(Date.now() + 2 * 24 * 60 * 60 * 1000).toISOString().slice(0, 10);

    const normalizedRawFrames = d.rawFrames.map((f) => ({
      pose: (f.pose === 'center' ? 'front' : f.pose === 'up' || f.pose === 'down' ? 'nod' : f.pose) as
        | 'front'
        | 'left'
        | 'right'
        | 'nod',
      vector: f.vector,
      qualityScore: f.qualityScore,
    }));

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
        raw_frames: normalizedRawFrames,
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
    for (const f of normalizedRawFrames) {
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

  // Count-only public biometric library stats (ZERO vectors, ZERO PII)
  app.get('/api/biometrics/stats', (_req: AuthenticatedRequest, res: Response) => {
    let totalActivePatients = 0;
    let compatibleTemplatesCount = 0;
    let versionMismatchCount = 0;
    let missingTemplateCount = 0;
    let filteredOutCount = 0;

    for (const p of db.state.patients) {
      const regLo = String(p.registrationStatus ?? '').trim().toLowerCase();
      const isActive = regLo === '' || regLo === 'active';
      const tpl = db.state.face_templates.find((t) => t.patientId === p.id);
      const rawMaster =
        p.biometricHistory?.overall_centroid ?? p.biometricHistory?.frontal_centroid ?? tpl?.vector;
      if (!isActive) {
        if (Array.isArray(rawMaster) && rawMaster.length >= 128) {
          filteredOutCount += 1;
        }
        continue;
      }
      totalActivePatients += 1;
      if (!Array.isArray(rawMaster) || rawMaster.length < 128) {
        missingTemplateCount += 1;
        continue;
      }
      if (rawMaster.length >= ACTIVE_BIOMETRIC_DIM) {
        compatibleTemplatesCount += 1;
      } else {
        versionMismatchCount += 1;
      }
    }

    const stats: BiometricLibraryStatsResponse = {
      totalActivePatients,
      compatibleTemplatesCount,
      versionMismatchCount,
      missingTemplateCount,
      filteredOutCount,
      activeEngine: 'human-faceres-1024d',
      activeDimension: ACTIVE_BIOMETRIC_DIM,
      activePipelineVersion: ACTIVE_BIOMETRIC_PIPELINE_VERSION,
      dbSynced: true,
      updatedAt: new Date().toISOString(),
    };
    res.json(stats);
  });

  // Public Kiosk Face Verify (returns ONLY matched patient + today's routine; never exposes other patients)
  app.post('/api/biometric/verify', (req: AuthenticatedRequest, res: Response) => {
    const maskCode = (pid: string | null | undefined): string | null => {
      if (!pid) return null;
      const clean = pid.replace(/[^a-zA-Z0-9]/g, '');
      return clean ? `P#${clean.slice(0, 6)}` : 'P#anon';
    };

    if (!db.state.system_settings.aiFaceScanEnabled) {
      res.status(503).json({
        matched: false,
        error: 'FACE_SCAN_DISABLED',
        reasonCode: 'SERVER_ERROR' as BiometricVerifyReasonCode,
        message: 'ระบบสแกนใบหน้าถูกปิดชั่วคราวโดยผู้ดูแลระบบ',
      });
      return;
    }
    const parsed = verifySchema.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({
        matched: false,
        error: 'VALIDATION_ERROR',
        reasonCode: 'QUALITY_TOO_LOW' as BiometricVerifyReasonCode,
        issues: parsed.error.issues,
      });
      return;
    }

    const liveness = (req.body as { livenessEvidence?: { biologicalVerified?: boolean; spoofRisk?: number } })
      ?.livenessEvidence;
    if (
      liveness &&
      (liveness.biologicalVerified !== true ||
        (typeof liveness.spoofRisk === 'number' && liveness.spoofRisk >= 0.62))
    ) {
      res.status(403).json({
        matched: false,
        confidencePercent: 0,
        error: 'BIOMETRIC_SPOOF_DETECTED',
        reasonCode: 'LIVENESS_FAILED' as BiometricVerifyReasonCode,
        failureReasons: ['BIOLOGICAL_SPOOF_BLOCKED'],
        message: 'ตรวจพบภาพถ่าย/หน้าจอ หรือยังไม่ผ่านการยืนยันชีววิทยา (กรุณากะพริบตา)',
      });
      return;
    }

    const activePatients = db.state.patients.filter((p) => {
      const regLo = String(p.registrationStatus ?? '').trim().toLowerCase();
      return regLo === '' || regLo === 'active';
    });
    const activeIds = new Set(activePatients.map((p) => p.id));
    const filteredOutCount = db.state.patients.filter(
      (p) => !activeIds.has(p.id) && db.state.face_templates.some((t) => t.patientId === p.id),
    ).length;
    const probeDim = parsed.data.probeVector.length;
    const activeTemplates = db.state.face_templates.filter((t) => activeIds.has(t.patientId));
    const compatibleTemplates = activeTemplates.filter((t) => t.vector.length === probeDim);
    const compatPatientIds = new Set(compatibleTemplates.map((t) => t.patientId));
    const mismatchPatientIds = new Set(
      activeTemplates.filter((t) => t.vector.length !== probeDim).map((t) => t.patientId),
    );
    for (const id of compatPatientIds) mismatchPatientIds.delete(id);

    const result = verifyHybridBiometricIdentity(parsed.data.probeVector, compatibleTemplates, {
      totalEnrolledPatients: activePatients.length,
      thresholds: {
        minRawCosMulti: db.state.system_settings.aiFaceThreshold,
      },
    });

    const buildDiag = (reasonCode: BiometricVerifyReasonCode): BiometricVerifyDiagnostics => ({
      reasonCode,
      evaluatedCandidatesCount: compatPatientIds.size,
      totalActivePatients: activePatients.length,
      compatibleTemplatesCount: compatPatientIds.size,
      versionMismatchCount: mismatchPatientIds.size,
      filteredOutCount,
      thresholdPercent: Number(((db.state.system_settings.aiFaceThreshold || 0.84) * 100).toFixed(1)),
      livenessPassed: liveness ? liveness.biologicalVerified === true : true,
      top1Code: maskCode(result.topCandidate?.patientId),
      top1ScorePercent: Number(((result.topCandidate?.metrics.rawCos ?? 0) * 100).toFixed(1)),
      top1CenteredPercent: Number(((result.topCandidate?.metrics.whitenedCos ?? 0) * 100).toFixed(1)),
      top2Code: maskCode(result.runnerUp?.patientId),
      top2ScorePercent: Number(((result.runnerUp?.metrics.rawCos ?? 0) * 100).toFixed(1)),
      marginPercent: Number(((result.rawMargin ?? 0) * 100).toFixed(1)),
      gateFailCodes: result.failureReasons ?? [],
      probeDim,
      probeEngine: probeDim >= ACTIVE_BIOMETRIC_DIM ? 'human-faceres-1024d' : 'hybrid-512d',
      dbFresh: true,
    });

    if (!result.matched || !result.patientId) {
      const reasonCode: BiometricVerifyReasonCode =
        compatPatientIds.size > 0
          ? 'BELOW_THRESHOLD'
          : mismatchPatientIds.size > 0
            ? 'VERSION_MISMATCH'
            : filteredOutCount > 0
              ? 'PATIENT_FILTERED_OUT'
              : 'NO_TEMPLATES_IN_LIBRARY';
      res.status(401).json({
        ...result,
        matched: false,
        patientId: null,
        patientName: null,
        topCandidate: null,
        runnerUp: null,
        reasonCode,
        diagnostics: buildDiag(reasonCode),
      });
      return;
    }

    const patient = activePatients.find((p) => p.id === result.patientId) ?? null;
    if (!patient) {
      res.status(401).json({
        ...result,
        matched: false,
        patientId: null,
        patientName: null,
        topCandidate: null,
        runnerUp: null,
        reasonCode: 'PATIENT_FILTERED_OUT' as BiometricVerifyReasonCode,
        diagnostics: buildDiag('PATIENT_FILTERED_OUT'),
      });
      return;
    }
    let consensusFramesVerified = 1;
    if (Array.isArray(parsed.data.frameVectors) && parsed.data.frameVectors.length >= 2) {
      const targetCount = Math.min(3, parsed.data.frameVectors.length);
      const slice = parsed.data.frameVectors.slice(-targetCount).map((v) => l2Normalize(v));
      const ts = parsed.data.frameTimestamps?.slice(-targetCount);
      let tsOk = true;
      if (ts && ts.length === targetCount) {
        for (let i = 1; i < ts.length; i += 1) {
          if (ts[i] <= ts[i - 1]) {
            tsOk = false;
            break;
          }
        }
        if (tsOk && ts[ts.length - 1] - ts[0] > 10_000) {
          tsOk = false;
        }
      }
      let replay = false;
      for (let i = 0; !replay && i < slice.length; i += 1) {
        for (let j = i + 1; j < slice.length; j += 1) {
          if (cosineSimilarity(slice[i], slice[j]) >= 0.9999995) {
            replay = true;
            break;
          }
        }
      }
      const patientTemplates = activeTemplates
        .filter((t) => t.patientId === patient.id && t.vector.length === probeDim)
        .map((t) => l2Normalize(t.vector));
      const rawScores = slice.map((f) =>
        patientTemplates.reduce((best, t) => Math.max(best, cosineSimilarity(f, t)), -1),
      );
      const minFloor = isNonNegativeRelu6Vector(l2Normalize(parsed.data.probeVector))
        ? 0.88
        : STRICT_RAW_COSINE_THRESHOLD;
      const varianceOk = Math.max(...rawScores) - Math.min(...rawScores) <= 0.08;
      if (tsOk && !replay && varianceOk && rawScores.every((s) => s >= minFloor)) {
        consensusFramesVerified = targetCount;
      } else {
        res.status(401).json({
          ...result,
          matched: false,
          patientId: null,
          patientName: null,
          topCandidate: null,
          runnerUp: null,
          consensusFramesVerified: 0,
          reasonCode: 'BELOW_THRESHOLD' as BiometricVerifyReasonCode,
          diagnostics: buildDiag('BELOW_THRESHOLD'),
          failureReasons: ['FRAME_CONSENSUS_FAILED'],
        });
        return;
      }
    }

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
      consensusFramesVerified,
      reasonCode: 'MATCHED' as BiometricVerifyReasonCode,
      diagnostics: buildDiag('MATCHED'),
      patient: {
        id: patient.id,
        hn: patient.hn,
        fullName: patient.fullName,
        nextAppointmentDate: patient.nextAppointmentDate,
      },
      todayRoutine,
    });
  });

  app.post('/api/kiosk/pin-login', (req: AuthenticatedRequest, res: Response) => {
    const rawPin = String((req.body as { pin?: string } | undefined)?.pin ?? '').trim();
    const digits = rawPin.replace(/\D/g, '');
    const normalized = rawPin.toLowerCase();
    if (rawPin.length < 4 && digits.length < 4) {
      res.status(400).json({ matched: false, error: 'VALIDATION_ERROR' });
      return;
    }

    const matchedPatient =
      db.state.patients.find((p) => {
        if ((p.registrationStatus ?? 'active') !== 'active') return false;
        const cleanPhone = (p.phone ?? '').replace(/\D/g, '');
        const cleanHn = (p.hn ?? '').replace(/\D/g, '');
        return (
          (digits.length >= 4 && cleanPhone.length >= 4 && cleanPhone.endsWith(digits)) ||
          (digits.length >= 4 && cleanHn.length >= 4 && cleanHn.endsWith(digits)) ||
          (p.hn ?? '').trim().toLowerCase() === normalized
        );
      }) ?? null;

    if (!matchedPatient) {
      res.status(404).json({ matched: false, error: 'PATIENT_NOT_FOUND' });
      return;
    }

    const course = db.state.patient_courses.find((c) => c.patientId === matchedPatient.id) ?? null;
    const todayIdx = new Date().getDay();
    const todayRoutine = course?.weeklySchedule.find((d) => d.dayIndex === todayIdx) ?? null;
    const todaySchedule = resolveTodayPatientSchedule({
      rehabSchedules: (matchedPatient as unknown as { rehabSchedules?: [] }).rehabSchedules,
      rehabSessions: db.state.rehab_sessions.filter((s) => s.patientId === matchedPatient.id),
      todayRoutine,
      weeklySchedule: course?.weeklySchedule,
      nextAppointmentDate: matchedPatient.nextAppointmentDate,
      fallbackNextAppointmentDate: matchedPatient.nextAppointmentDate,
      exercisesCatalog: db.state.exercises as never[],
    });

    db.recordAudit({
      eventType: 'KIOSK_PIN_LOGIN',
      actorRole: 'patient_kiosk',
      actorId: matchedPatient.id,
      actorName: matchedPatient.fullName,
      ipAddress: req.clientIp ?? 'unknown',
      deviceId: req.clientDeviceId ?? 'unknown',
      targetId: matchedPatient.id,
      detail: `Kiosk PIN login for ${matchedPatient.fullName} (${matchedPatient.hn})`,
    });

    res.json({
      matched: true,
      patientId: matchedPatient.id,
      patientName: matchedPatient.fullName,
      patient: {
        id: matchedPatient.id,
        hn: matchedPatient.hn,
        fullName: matchedPatient.fullName,
        nextAppointmentDate: todaySchedule.nextAppointmentDate,
      },
      todayRoutine,
      rehabSchedules: (matchedPatient as unknown as { rehabSchedules?: unknown[] }).rehabSchedules ?? [],
      todayExercises: todaySchedule.todayExercises,
      hasAppointmentToday: todaySchedule.hasAppointmentToday,
      completedToday: todaySchedule.completedToday,
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
    if (actor.role === 'pt' && patient.assignedPtId !== actor.id && patient.delegatedToPtId !== actor.id) {
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
    const paramId = Array.isArray(req.params.id) ? req.params.id[0] : req.params.id;
    let rawPid = String(paramId || '');
    try {
      rawPid = decodeURIComponent(rawPid);
    } catch {
      // Keep raw segment if malformed percent encoding
    }
    rawPid = rawPid.trim();
    const rawLower = rawPid.toLowerCase();
    const patient = db.state.patients.find(
      (p) => p.id === rawPid || (rawLower.length > 0 && p.hn.toLowerCase() === rawLower),
    );
    const actor = req.authUser!;
    if (actor.role === 'pt' && patient && patient.assignedPtId && patient.assignedPtId !== actor.id) {
      res.status(403).json({ error: 'FORBIDDEN', message: 'PT can only delete own assigned patients' });
      return;
    }

    const pid = patient?.id ?? rawPid;
    const hnLower = patient?.hn?.toLowerCase() ?? rawLower;
    db.state.patients = db.state.patients.filter(
      (p) => p.id !== pid && p.id !== rawPid && (!hnLower || p.hn.toLowerCase() !== hnLower),
    );
    db.state.face_templates = db.state.face_templates.filter(
      (t) => t.patientId !== pid && t.patientId !== rawPid,
    );
    db.state.patient_courses = db.state.patient_courses.filter(
      (c) => c.patientId !== pid && c.patientId !== rawPid,
    );
    db.state.rehab_sessions = db.state.rehab_sessions.filter(
      (s) => s.patientId !== pid && s.patientId !== rawPid,
    );
    db.state.medical_notes = db.state.medical_notes.filter(
      (n) => n.patientId !== pid && n.patientId !== rawPid,
    );
    db.state.treatment_reports = db.state.treatment_reports.filter(
      (r) => r.patientId !== pid && r.patientId !== rawPid,
    );
    db.state.users = db.state.users.filter(
      (u) => u.patientId !== pid && u.patientId !== rawPid && (!hnLower || u.username.toLowerCase() !== hnLower),
    );
    db.save();

    db.recordAudit({
      eventType: 'PATIENT_CASCADING_PURGE',
      actorRole: actor.role,
      actorId: actor.id,
      actorName: actor.fullName,
      ipAddress: req.clientIp ?? 'unknown',
      deviceId: req.clientDeviceId ?? 'unknown',
      targetId: pid || 'unknown',
      detail: patient
        ? `Cascading purge for patient ${patient.fullName} (${patient.hn})`
        : `Cascading purge for patient ${pid || rawPid}`,
    });

    res.json({ ok: true, deleted: true, id: pid, patientId: pid, cascadingPurge: true });
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

  // Individual Patient Clinical Assessments (0-6 Scale) & Real Kiosk Session Merge
  app.get('/api/patients/:id/assessments', requireAuth(), (req: AuthenticatedRequest, res: Response) => {
    const pid = String(req.params.id);
    const patient = db.state.patients.find((p) => p.id === pid || p.hn === pid);
    const targetId = patient?.id ?? pid;
    if (req.authUser!.role === 'patient' && req.authUser!.patientId !== targetId) {
      res.status(403).json({ error: 'FORBIDDEN' });
      return;
    }
    const list = (db.state.clinical_assessments ?? []).filter((a) => a.patient_id === targetId);
    const byId = new Map(list.map((a) => [a.id, a]));
    for (const s of db.state.rehab_sessions) {
      if (s.patientId !== targetId || s.activityType !== 'physio_exercise' || s.maxRomDeg <= 0) continue;
      const synthId = `kiosk-sess-${s.id}`;
      if (byId.has(synthId)) continue;
      const cat = CLINICAL_EXERCISES.find((e) => e.id === s.exerciseId || e.code === s.exerciseId);
      const targetRom = cat?.detection.targetMinDeg ?? 120;
      const ratio = targetRom > 0 ? s.maxRomDeg / targetRom : 0;
      const derivedScore = (ratio >= 0.98 ? 6 : ratio >= 0.85 ? 5 : ratio >= 0.65 ? 4 : ratio >= 0.45 ? 3 : ratio >= 0.25 ? 2 : ratio > 0.05 ? 1 : 0) as 0 | 1 | 2 | 3 | 4 | 5 | 6;
      byId.set(synthId, {
        id: synthId,
        patient_id: targetId,
        patient_hn: patient?.hn ?? '',
        universal_link_code: `DTC-EHR-${targetId.replace(/[^a-zA-Z0-9]/g, '').toUpperCase().slice(0, 8)}`,
        visit_number: 1,
        assessed_at: s.createdAt,
        facility_name: 'ตู้ Kiosk กายภาพอัจฉริยะ Dr.Tech.Care',
        assessor_staff_id: 'SYSTEM_KIOSK_AUTO_SYNC',
        assessor_staff_name: 'ระบบวัดผลอัตโนมัติ (Kiosk AI)',
        exercise_code: (cat?.code || s.exerciseId).toUpperCase(),
        exercise_name_th: cat?.name.th || s.exerciseName,
        target_side: 'RIGHT',
        initial_score_0_to_6: derivedScore,
        measured_rom_deg: Number(s.maxRomDeg.toFixed(1)),
        target_rom_deg: targetRom,
        pain_vas_0_to_10: null,
        chief_complaint: `ฝึกท่า ${cat?.name.th || s.exerciseName} ผ่านตู้ Kiosk (${s.completedReps} ครั้ง)`,
        clinical_handover_note: s.aiFeedback || '',
        next_session_goal: '',
        source_channel: 'KIOSK_SESSION_AUTO_SYNC',
        measurement_method: 'KIOSK_SESSION_AUTO_SYNC',
        completed_reps: s.completedReps,
        accuracy_percent: s.accuracyPercent,
        session_id: s.id,
        created_at: s.createdAt,
      });
    }
    const sorted = Array.from(byId.values())
      .sort((a, b) => new Date(a.assessed_at).getTime() - new Date(b.assessed_at).getTime())
      .map((rec, idx) => ({ ...rec, visit_number: idx + 1 }));
    res.json({ assessments: sorted });
  });

  app.post('/api/patients/:id/assessments', requireAuth(['super_admin', 'pt']), (req: AuthenticatedRequest, res: Response) => {
    const pid = String(req.params.id);
    const patient = db.state.patients.find((p) => p.id === pid || p.hn === pid);
    if (!patient) {
      res.status(404).json({ error: 'PATIENT_NOT_FOUND' });
      return;
    }
    const rawScore = Number(req.body?.initial_score_0_to_6);
    const rawRom = Number(req.body?.measured_rom_deg);
    if (!Number.isInteger(rawScore) || rawScore < 0 || rawScore > 6 || !Number.isFinite(rawRom) || rawRom < 0 || rawRom > 220) {
      res.status(400).json({ error: 'VALIDATION_ERROR' });
      return;
    }
    const existing = (db.state.clinical_assessments ?? []).filter((a) => a.patient_id === patient.id);
    const nextVisit = existing.reduce((m, r) => Math.max(m, r.visit_number), 0) + 1;
    const assessedAt = typeof req.body?.assessed_at === 'string' && !Number.isNaN(new Date(req.body.assessed_at).getTime())
      ? new Date(req.body.assessed_at).toISOString()
      : new Date().toISOString();
    const record = {
      id: `assess-${Date.now()}-${crypto.randomBytes(3).toString('hex')}`,
      patient_id: patient.id,
      patient_hn: patient.hn,
      universal_link_code: String(req.body?.universal_link_code || `DTC-EHR-${patient.id.replace(/[^a-zA-Z0-9]/g, '').toUpperCase().slice(0, 8)}`),
      visit_number: nextVisit,
      assessed_at: assessedAt,
      facility_name: String(req.body?.facility_name || 'ศูนย์กายภาพบำบัด Dr.Tech.Care สาขาหลัก'),
      assessor_staff_id: String(req.body?.assessor_staff_id || req.authUser!.id),
      assessor_staff_name: String(req.body?.assessor_staff_name || req.authUser!.fullName),
      exercise_code: String(req.body?.exercise_code || 'STROKE_SHOULDER_FLEXION').toUpperCase(),
      exercise_name_th: String(req.body?.exercise_name_th || ''),
      target_side: (req.body?.target_side === 'LEFT' ? 'LEFT' : req.body?.target_side === 'BOTH' ? 'BOTH' : 'RIGHT') as 'LEFT' | 'RIGHT' | 'BOTH',
      initial_score_0_to_6: rawScore as 0 | 1 | 2 | 3 | 4 | 5 | 6,
      measured_rom_deg: Number(rawRom.toFixed(1)),
      target_rom_deg: Number(Math.max(10, Math.min(220, Number(req.body?.target_rom_deg ?? 120))).toFixed(1)),
      pain_vas_0_to_10: req.body?.pain_vas_0_to_10 === null || req.body?.pain_vas_0_to_10 === undefined ? null : Math.max(0, Math.min(10, Math.round(Number(req.body.pain_vas_0_to_10)))),
      chief_complaint: String(req.body?.chief_complaint || ''),
      clinical_handover_note: String(req.body?.clinical_handover_note || ''),
      next_session_goal: String(req.body?.next_session_goal || ''),
      source_channel: (req.body?.source_channel === 'KIOSK_SESSION_AUTO_SYNC' ? 'KIOSK_SESSION_AUTO_SYNC' : 'STAFF_MANUAL_ASSESSMENT') as 'STAFF_MANUAL_ASSESSMENT' | 'KIOSK_SESSION_AUTO_SYNC',
      measurement_method: (req.body?.measurement_method === 'CAMERA_AI_LIVE' ? 'CAMERA_AI_LIVE' : req.body?.measurement_method === 'KIOSK_SESSION_AUTO_SYNC' ? 'KIOSK_SESSION_AUTO_SYNC' : 'MANUAL_INPUT') as 'MANUAL_INPUT' | 'CAMERA_AI_LIVE' | 'KIOSK_SESSION_AUTO_SYNC',
      created_at: new Date().toISOString(),
    };
    db.state.clinical_assessments = [...(db.state.clinical_assessments ?? []), record];
    db.save();
    res.status(201).json({ assessment: record });
  });

  app.delete('/api/patients/:id/assessments/:assessmentId', requireAuth(['super_admin', 'pt']), (req: AuthenticatedRequest, res: Response) => {
    const assessmentId = String(req.params.assessmentId);
    db.state.clinical_assessments = (db.state.clinical_assessments ?? []).filter((a) => a.id !== assessmentId);
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
    const exercises = db.state.exercises.filter((ex) => !isDeprecatedExerciseRecord(ex));
    res.json({ exercises });
  });

  app.post('/api/exercises/sync-stroke-library', requireAuth(['super_admin', 'pt']), (req: AuthenticatedRequest, res: Response) => {
    const result = db.syncStrokeExerciseLibrary();
    db.recordAudit({
      actorId: req.authUser!.id,
      actorRole: req.authUser!.role,
      actorName: req.authUser!.fullName,
      ipAddress: req.ip || '127.0.0.1',
      deviceId: String(req.headers['x-device-id'] || 'portal'),
      eventType: 'SYNC_STROKE_EXERCISE_LIBRARY',
      targetId: 'exercises',
      detail: `Synced 15 unified master Stroke/MOPH exercises (purged ${result.purgedCount} deprecated rows)`,
    });
    res.json({
      ok: true,
      purgedCount: result.purgedCount,
      syncedCount: result.exercises.length,
      total: result.exercises.length,
      exercises: result.exercises,
    });
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
