import fs from 'node:fs';
import path from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import type { SupabaseClient } from '@supabase/supabase-js';
import {
  l2Normalize,
  type GeometricAnthropometricRatios,
} from '@drtechcares/smart-biometric-core';
import { createCloudRouter } from '../../packages/api-client/src/cloudRouter';
import {
  persistWithTimeoutBestEffort,
  resetSupabaseCircuitForTests,
  resetUnsupportedColumnsForTests,
} from '../../packages/api-client/src/supabasePersistence';

/**
 * Generates a realistic non-negative 1024-D ReLU6 facial embedding with high variance
 * and distinct per-seed identity peaks so triple-gate cosine + centered cosine + L2 checks
 * match genuine same-person probes and strictly reject strangers.
 */
function makeRelu6Face1024(seed: number, noiseAmp = 0): number[] {
  const raw = new Array<number>(1024);
  for (let i = 0; i < 1024; i += 1) {
    const primary = Math.sin(i * 0.137 + seed * 2.31) + Math.cos(i * 0.053 - seed * 1.19);
    const fine = Math.sin(i * 0.419 + seed * 5.73) * 0.65;
    const base = primary + fine > 0.35 ? primary + fine : 0;
    const jitter = noiseAmp > 0 ? Math.sin(i * 1.7 + seed * 9.1) * noiseAmp : 0;
    raw[i] = Math.max(0, base + jitter);
  }
  return l2Normalize(raw);
}

function makeGeometricRatios(seed: number): GeometricAnthropometricRatios {
  const patch384 = new Array<number>(384);
  let sum = 0;
  for (let i = 0; i < 384; i += 1) {
    const v = Math.sin(i * 0.21 + seed * 1.7) + 0.5 * Math.cos(i * 0.09 - seed * 0.8);
    patch384[i] = v;
    sum += v;
  }
  const mean = sum / 384;
  let sq = 0;
  for (let i = 0; i < 384; i += 1) {
    patch384[i] -= mean;
    sq += patch384[i] * patch384[i];
  }
  const std = Math.sqrt(sq / 384) || 1;
  for (let i = 0; i < 384; i += 1) {
    patch384[i] = Number((patch384[i] / std).toFixed(4));
  }

  const ocular256 = new Array<number>(256);
  let oSum = 0;
  for (let i = 0; i < 256; i += 1) {
    const v = Math.cos(i * 0.19 + seed * 2.1) + 0.4 * Math.sin(i * 0.33 + seed);
    ocular256[i] = v;
    oSum += v;
  }
  const oMean = oSum / 256;
  let oSq = 0;
  for (let i = 0; i < 256; i += 1) {
    ocular256[i] -= oMean;
    oSq += ocular256[i] * ocular256[i];
  }
  const oStd = Math.sqrt(oSq / 256) || 1;
  for (let i = 0; i < 256; i += 1) {
    ocular256[i] = Number((ocular256[i] / oStd).toFixed(4));
  }

  return {
    eyeToNoseRatio: Number((0.92 + (seed % 5) * 0.04).toFixed(4)),
    faceWidthToUpperHeight: Number((1.36 + (seed % 4) * 0.05).toFixed(4)),
    mouthToEyeRatio: Number((0.78 + (seed % 3) * 0.03).toFixed(4)),
    noseToChinRatio: Number((0.68 + (seed % 6) * 0.02).toFixed(4)),
    appearancePatch384: patch384,
    craniofacialRatios12: Array.from({ length: 12 }, (_, idx) =>
      Number((0.55 + ((seed * 7 + idx * 13) % 40) * 0.02).toFixed(4)),
    ),
    ocularIrisPatch256: ocular256,
    irisOcularRatios8: Array.from({ length: 8 }, (_, idx) =>
      Number((0.42 + ((seed * 5 + idx * 11) % 30) * 0.015).toFixed(4)),
    ),
  };
}

interface SimulatedDbConfig {
  /** Allowed columns per table; any SELECT/INSERT/UPDATE/UPSERT referencing unlisted columns returns PGRST204 / 42703. */
  schema: Record<string, Set<string>>;
  /** In-memory table rows persisted across Worker cold restarts. */
  tables: Record<string, Record<string, unknown>[]>;
  /** Optional latency injected on writes/reads to test cold-start timeouts. */
  writeDelayMs?: number;
  readDelayMs?: number;
  /** Optional forced HTTP error status per table on write. */
  failWritesOnTable?: Record<string, { status: number; message: string }>;
}

/**
 * Authentic PostgREST HTTP fetch simulator for `@supabase/supabase-js` so we test the REAL
 * `createCloudRouter` + `supabasePersistence` + `selectExplicitWithFallback` + `upsertPatientBiometricsCompat`
 * across true cold Worker isolate destruction & recreation.
 */
function createSimulatedPostgrestFetch(cfg: SimulatedDbConfig): typeof fetch {
  const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

  const projectRow = (
    table: string,
    row: Record<string, unknown>,
    selectParam: string | null,
  ): { projected?: Record<string, unknown>; missingCol?: string } => {
    const allowed = cfg.schema[table];
    if (!selectParam || selectParam === '*') {
      return { projected: { ...row } };
    }
    const parts = selectParam
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean);
    const out: Record<string, unknown> = {};
    for (const part of parts) {
      const colonIdx = part.indexOf(':');
      const alias = colonIdx >= 0 ? part.slice(0, colonIdx).trim() : null;
      const expr = colonIdx >= 0 ? part.slice(colonIdx + 1).trim() : part;
      const jsonTextIdx = expr.indexOf('->>');
      const jsonObjIdx = expr.indexOf('->');
      const baseCol =
        jsonTextIdx >= 0
          ? expr.slice(0, jsonTextIdx).trim()
          : jsonObjIdx >= 0
            ? expr.slice(0, jsonObjIdx).trim()
            : expr;
      if (allowed && !allowed.has(baseCol)) {
        return { missingCol: baseCol };
      }
      if (jsonTextIdx >= 0 || jsonObjIdx >= 0) {
        const key =
          jsonTextIdx >= 0
            ? expr.slice(jsonTextIdx + 3).trim()
            : expr.slice(jsonObjIdx + 2).trim();
        const container = row[baseCol];
        const rawVal =
          container && typeof container === 'object'
            ? (container as Record<string, unknown>)[key]
            : null;
        const outKey = alias || key;
        out[outKey] =
          jsonTextIdx >= 0
            ? rawVal === null || rawVal === undefined
              ? null
              : String(rawVal)
            : rawVal ?? null;
      } else {
        out[alias || baseCol] = row[baseCol] ?? null;
      }
    }
    return { projected: out };
  };

  return async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const urlStr =
      typeof input === 'string'
        ? input
        : input instanceof URL
          ? input.toString()
          : input.url;
    const url = new URL(urlStr);
    const method = (init?.method || (input instanceof Request ? input.method : 'GET')).toUpperCase();

    if (!url.pathname.includes('/rest/v1/')) {
      return new Response(JSON.stringify({ data: null }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    const table = url.pathname.split('/rest/v1/')[1]?.split('/')[0] ?? '';
    const allowed = cfg.schema[table];
    if (!cfg.tables[table]) {
      cfg.tables[table] = [];
    }

    if (method === 'GET' && cfg.readDelayMs && cfg.readDelayMs > 0) {
      await sleep(cfg.readDelayMs);
    }
    if (method !== 'GET' && cfg.writeDelayMs && cfg.writeDelayMs > 0) {
      await sleep(cfg.writeDelayMs);
    }

    if (method !== 'GET' && cfg.failWritesOnTable?.[table]) {
      const f = cfg.failWritesOnTable[table];
      return new Response(
        JSON.stringify({ code: '42501', message: f.message }),
        { status: f.status, headers: { 'Content-Type': 'application/json' } },
      );
    }

    const filterRows = (rows: Record<string, unknown>[]): Record<string, unknown>[] => {
      let filtered = [...rows];
      for (const [paramKey, paramVal] of url.searchParams.entries()) {
        if (['select', 'limit', 'order', 'on_conflict', 'columns'].includes(paramKey)) continue;
        if (paramVal.startsWith('eq.')) {
          const target = paramVal.slice(3);
          filtered = filtered.filter((r) => String(r[paramKey] ?? '') === target);
        } else if (paramVal.startsWith('in.(') && paramVal.endsWith(')')) {
          const vals = new Set(
            paramVal
              .slice(4, -1)
              .split(',')
              .map((v) => v.replace(/^"|"$/g, '').trim())
              .filter(Boolean),
          );
          filtered = filtered.filter((r) => vals.has(String(r[paramKey] ?? '')));
        }
      }
      const limitParam = Number(url.searchParams.get('limit'));
      if (Number.isFinite(limitParam) && limitParam > 0) {
        filtered = filtered.slice(0, limitParam);
      }
      return filtered;
    };

    if (method === 'GET') {
      const selectParam = url.searchParams.get('select');
      // Validate columns even on empty tables just like real Postgres / PostgREST
      const colCheck = projectRow(table, {}, selectParam);
      if (colCheck.missingCol) {
        return new Response(
          JSON.stringify({
            code: '42703',
            message: `column ${table}.${colCheck.missingCol} does not exist`,
          }),
          { status: 400, headers: { 'Content-Type': 'application/json' } },
        );
      }
      const matched = filterRows(cfg.tables[table]);
      const projectedRows = matched.map((r) => projectRow(table, r, selectParam).projected!);
      const accept = new Headers(init?.headers).get('accept') || '';
      if (accept.includes('application/vnd.pgrst.object+json')) {
        return new Response(JSON.stringify(projectedRows[0] ?? null), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      return new Response(JSON.stringify(projectedRows), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    const rawBody = typeof init?.body === 'string' ? init.body : '';
    const parsedBody = rawBody ? (JSON.parse(rawBody) as unknown) : null;
    const incomingRows: Record<string, unknown>[] = Array.isArray(parsedBody)
      ? (parsedBody as Record<string, unknown>[])
      : parsedBody && typeof parsedBody === 'object'
        ? [parsedBody as Record<string, unknown>]
        : [];

    if (allowed) {
      for (const row of incomingRows) {
        for (const col of Object.keys(row)) {
          if (!allowed.has(col)) {
            return new Response(
              JSON.stringify({
                code: 'PGRST204',
                message: `Could not find the '${col}' column of '${table}' in the schema cache`,
              }),
              { status: 400, headers: { 'Content-Type': 'application/json' } },
            );
          }
        }
      }
    }

    if (method === 'POST') {
      const prefer = (new Headers(init?.headers).get('prefer') || '').toLowerCase();
      const onConflict = url.searchParams.get('on_conflict') || 'id';
      const isUpsert = prefer.includes('resolution=merge-duplicates');
      for (const row of incomingRows) {
        const keyVal = row[onConflict];
        const existingIdx =
          keyVal !== undefined && keyVal !== null
            ? cfg.tables[table].findIndex((r) => String(r[onConflict] ?? '') === String(keyVal))
            : -1;
        if (existingIdx >= 0) {
          if (isUpsert) {
            cfg.tables[table][existingIdx] = { ...cfg.tables[table][existingIdx], ...row };
          } else {
            return new Response(
              JSON.stringify({
                code: '23505',
                message: `duplicate key value violates unique constraint "${table}_${onConflict}_key"`,
              }),
              { status: 409, headers: { 'Content-Type': 'application/json' } },
            );
          }
        } else {
          cfg.tables[table].push({ ...row });
        }
      }
      return new Response(JSON.stringify(incomingRows), {
        status: 201,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    if (method === 'PATCH') {
      const patch = incomingRows[0] ?? {};
      const matched = filterRows(cfg.tables[table]);
      for (const target of matched) {
        Object.assign(target, patch);
      }
      return new Response(JSON.stringify(matched), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    if (method === 'DELETE') {
      const toRemove = new Set(filterRows(cfg.tables[table]));
      cfg.tables[table] = cfg.tables[table].filter((r) => !toRemove.has(r));
      return new Response(JSON.stringify([]), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    return new Response(JSON.stringify([]), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  };
}

function makeLegacyPartialSchema(): Record<string, Set<string>> {
  return {
    // Exact base 001 + 002 patients schema (MISSING master_embedding, angle_embeddings, face_embedding, biometric_data, biometric_updated_at, gender, stroke_type, etc.)
    patients: new Set([
      'id',
      'hn',
      'full_name',
      'age',
      'phone',
      'diagnosis',
      'registration_status',
      'assigned_pt_id',
      'next_appointment_date',
      'face_biometric_status',
      'face_frames_count',
      'face_image_url',
      'history',
      'created_at',
      'updated_at',
    ]),
    // Exact base 002_face_id_120.sql schema (MISSING geometric_ratios, frame_count, model_version, biometric_engine, embedding_dim, exemplar_embeddings!)
    patient_biometrics: new Set([
      'id',
      'patient_id',
      'hn',
      'master_embedding',
      'angle_embeddings',
      'frames_captured',
      'quality_score',
      'avatar_url',
      'updated_at',
    ]),
    face_templates: new Set(['id', 'patient_id', 'pose', 'embedding', 'quality_score']),
    face_enrollments: new Set(['id', 'patient_id', 'shot_count', 'liveness_verified', 'enrolled_by']),
    staff_users: new Set([
      'id',
      'username',
      'password_hash',
      'role',
      'full_name',
      'phone',
      'specialty',
      'license_no',
      'patient_id',
      'active',
      'force_logged_out_at',
      'created_at',
    ]),
    patient_courses: new Set([
      'id',
      'patient_id',
      'title',
      'weekly_schedule',
      'cognitive_modpack_id',
      'updated_at',
    ]),
    system_settings: new Set([
      'id',
      'ai_face_scan_enabled',
      'ai_face_threshold',
      'ai_pose_detection_enabled',
      'kiosk_auto_reset_sec',
    ]),
    banned_targets: new Set(['id', 'target_type', 'value', 'reason', 'banned_by', 'created_at']),
    active_sessions: new Set(['token', 'user_id', 'role', 'ip_address', 'device_id', 'created_at', 'revoked']),
    exercises: new Set(['id', 'code', 'category', 'name_i18n', 'instruction_i18n', 'detection_config', 'active']),
    modpacks: new Set(['id', 'name', 'categories', 'questions', 'active']),
    kiosks: new Set(['kiosk_id', 'location', 'online', 'current_state', 'last_heartbeat_at']),
    audit_logs: new Set(['id', 'event_type', 'actor_role', 'actor_id', 'actor_name', 'ip_address', 'device_id', 'target_id', 'detail', 'created_at']),
    medical_notes: new Set(['id', 'patient_id', 'author_id', 'author_name', 'content', 'created_at']),
    rehab_sessions: new Set(['id', 'patient_id', 'exercise_id', 'completed_sets', 'completed_reps', 'accuracy_percent', 'max_rom_deg', 'pain_stop_triggered', 'kinematics_replay', 'minigame_score', 'created_at']),
    treatment_reports: new Set(['id', 'patient_id', 'pt_id', 'pt_name', 'summary', 'ai_recommendation', 'score_summary', 'created_at', 'updated_at']),
    staff_calls: new Set(['id', 'kiosk_id', 'patient_id', 'reason', 'status', 'created_at']),
    rehab_session_history: new Set(['id', 'patient_id', 'hn', 'session_date', 'exercise_code', 'exercise_name', 'achieved_max_rom_deg', 'valid_reps', 'created_at']),
    patient_clinical_assessments: new Set(['id', 'patient_id', 'patient_hn', 'universal_link_code', 'visit_number', 'assessed_at', 'exercise_code', 'initial_score_0_to_6', 'measured_rom_deg', 'target_rom_deg']),
  };
}

async function loginAdminOnRouter(router: ReturnType<typeof createCloudRouter>): Promise<string> {
  const res = await router(
    new Request('https://kiosk.local/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: 'admin', password: 'Admin@2026' }),
    }),
  );
  const body = (await res.json()) as { token?: string };
  if (!body.token) throw new Error('Failed to obtain admin token in test');
  return body.token;
}

describe('Post-Deploy Biometric Persistence & Cold-Start Verification', () => {
  beforeEach(() => {
    resetUnsupportedColumnsForTests();
    resetSupabaseCircuitForTests();
  });

  afterEach(() => {
    resetUnsupportedColumnsForTests();
    resetSupabaseCircuitForTests();
  });

  it('persists enrolled 1024-D patient on legacy partial schema and matches immediately after simulated deploy (cold Worker restart)', async () => {
    const dbCfg: SimulatedDbConfig = {
      schema: makeLegacyPartialSchema(),
      tables: {},
    };
    const mockFetch = createSimulatedPostgrestFetch(dbCfg);

    // 1. Worker Isolate #1 (before deploy)
    const routerBeforeDeploy = createCloudRouter({
      supabaseUrl: 'https://test-post-deploy.supabase.co',
      supabaseAnonKey: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyb2xlIjoiYW5vbiJ9.signature1234567890',
      customSupabaseFetch: mockFetch,
    });
    const adminToken = await loginAdminOnRouter(routerBeforeDeploy);

    const masterVec = makeRelu6Face1024(7);
    const centerVec = makeRelu6Face1024(7, 0.015);
    const leftVec = makeRelu6Face1024(7, 0.03);
    const rightVec = makeRelu6Face1024(7, 0.035);
    const upVec = makeRelu6Face1024(7, 0.025);
    const downVec = makeRelu6Face1024(7, 0.028);
    const exemplars = [
      makeRelu6Face1024(7, 0.01),
      makeRelu6Face1024(7, 0.018),
      makeRelu6Face1024(7, 0.022),
      makeRelu6Face1024(7, 0.026),
    ];
    const geom = makeGeometricRatios(7);

    const enrollRes = await routerBeforeDeploy(
      new Request('https://kiosk.local/api/biometric/enroll', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${adminToken}`,
        },
        body: JSON.stringify({
          hn: 'HN-POSTDEPLOY-001',
          fullName: 'Somchai PostDeploy',
          age: 58,
          phone: '081-234-5678',
          diagnosis: 'Stroke Shoulder Rehab',
          masterEmbedding: masterVec,
          angleEmbeddings: {
            center: centerVec,
            left: leftVec,
            right: rightVec,
            up: upVec,
            down: downVec,
          },
          exemplarEmbeddings: exemplars,
          biometricEngine: 'human-faceres-1024d',
          geometricRatios: geom,
          framesCaptured: 120,
          qualityScore: 0.97,
        }),
      }),
    );

    expect(enrollRes.status).toBe(201);
    const enrollBody = (await enrollRes.json()) as {
      ok: boolean;
      persisted: boolean;
      patient: { id: string; hn: string; fullName: string };
    };
    expect(enrollBody.ok).toBe(true);
    expect(enrollBody.persisted).toBe(true);
    // Both `patients` AND `patient_biometrics` MUST actually have durable rows despite 6 missing columns!
    expect(dbCfg.tables.patients.length).toBe(1);
    expect(dbCfg.tables.patient_biometrics.length).toBe(1);

    // 2. SIMULATE CLOUDFLARE DEPLOY: clear all module caches and create a brand-new cold Worker isolate
    resetUnsupportedColumnsForTests();
    resetSupabaseCircuitForTests();

    const routerAfterDeploy = createCloudRouter({
      supabaseUrl: 'https://test-post-deploy.supabase.co',
      supabaseAnonKey: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyb2xlIjoiYW5vbiJ9.signature1234567890',
      customSupabaseFetch: mockFetch,
    });

    // 3. Stats on cold isolate MUST accurately report 1 compatible account (not 0!)
    const statsRes = await routerAfterDeploy(
      new Request('https://kiosk.local/api/biometrics/stats?refresh=1', { method: 'GET' }),
    );
    expect(statsRes.status).toBe(200);
    const statsBody = (await statsRes.json()) as {
      totalActivePatients: number;
      compatibleTemplatesCount: number;
      dbSynced: boolean;
    };
    expect(statsBody.dbSynced).toBe(true);
    expect(statsBody.totalActivePatients).toBe(1);
    expect(statsBody.compatibleTemplatesCount).toBe(1);

    // 4. Index on cold isolate MUST include patient + exemplars + geometricRatios
    const indexRes = await routerAfterDeploy(
      new Request('https://kiosk.local/api/biometrics/index', { method: 'GET' }),
    );
    expect(indexRes.status).toBe(200);
    const indexBody = (await indexRes.json()) as {
      index: Array<{
        patientId: string;
        masterEmbedding: number[];
        exemplarEmbeddings?: number[][];
        geometricRatios?: GeometricAnthropometricRatios;
      }>;
    };
    expect(indexBody.index.length).toBe(1);
    expect(indexBody.index[0].patientId).toBe(enrollBody.patient.id);
    expect(indexBody.index[0].masterEmbedding.length).toBe(1024);
    expect(indexBody.index[0].exemplarEmbeddings?.length).toBeGreaterThanOrEqual(3);
    expect(indexBody.index[0].geometricRatios?.appearancePatch384?.length).toBe(384);

    // 5. Face verify on cold isolate MUST match enrolled patient immediately
    const liveProbe = makeRelu6Face1024(7, 0.012);
    const verifyRes = await routerAfterDeploy(
      new Request('https://kiosk.local/api/biometric/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          probeVector: liveProbe,
          engine: 'human-faceres-1024d',
          geometricRatios: geom,
          yawDeg: 0,
          pitchDeg: 0,
          livenessEvidence: {
            biologicalVerified: true,
            blinkCount: 1,
            spoofRisk: 0.08,
            nonRigidScore: 0.45,
          },
        }),
      }),
    );
    expect(verifyRes.status).toBe(200);
    const verifyBody = (await verifyRes.json()) as {
      matched: boolean;
      patientId: string | null;
      reasonCode?: string;
    };
    expect(verifyBody.matched).toBe(true);
    expect(verifyBody.patientId).toBe(enrollBody.patient.id);
    expect(verifyBody.reasonCode).toBe('MATCHED');

    // 6. Strict security preserved: an unregistered stranger MUST STILL be rejected!
    const strangerProbe = makeRelu6Face1024(99);
    const strangerGeom = makeGeometricRatios(99);
    const strangerRes = await routerAfterDeploy(
      new Request('https://kiosk.local/api/biometric/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          probeVector: strangerProbe,
          engine: 'human-faceres-1024d',
          geometricRatios: strangerGeom,
          yawDeg: 0,
          pitchDeg: 0,
          livenessEvidence: {
            biologicalVerified: true,
            blinkCount: 1,
            spoofRisk: 0.08,
            nonRigidScore: 0.45,
          },
        }),
      }),
    );
    expect(strangerRes.status).toBe(401);
    const strangerBody = (await strangerRes.json()) as { matched: boolean };
    expect(strangerBody.matched).toBe(false);
  });

  it('recovers patient vectors stored ONLY inside patients.history JSONB when patient_biometrics is empty and top-level embedding columns do not exist', async () => {
    const masterVec = makeRelu6Face1024(15);
    const centerVec = makeRelu6Face1024(15, 0.01);
    const leftVec = makeRelu6Face1024(15, 0.02);
    const rightVec = makeRelu6Face1024(15, 0.025);
    const upVec = makeRelu6Face1024(15, 0.02);
    const downVec = makeRelu6Face1024(15, 0.02);
    const geom = makeGeometricRatios(15);
    const existingPid = '11111111-2222-4333-8444-555555555555';

    const dbCfg: SimulatedDbConfig = {
      schema: makeLegacyPartialSchema(),
      tables: {
        patients: [
          {
            id: existingPid,
            hn: 'HN-HIST-ONLY',
            full_name: 'Suda HistoryOnly',
            age: 62,
            phone: '089-111-2222',
            diagnosis: 'Knee Rehab',
            registration_status: 'active',
            assigned_pt_id: null,
            next_appointment_date: null,
            // Even with stale PENDING top-level status, vectors exist in history!
            face_biometric_status: 'PENDING',
            face_frames_count: 0,
            face_image_url: null,
            history: {
              faceBiometricStatus: 'VERIFIED',
              faceFramesCount: 120,
              biometricEngine: 'human-faceres-1024d',
              masterEmbedding: masterVec,
              angleEmbeddings: {
                center: centerVec,
                left: leftVec,
                right: rightVec,
                up: upVec,
                down: downVec,
              },
              exemplarEmbeddings: [
                makeRelu6Face1024(15, 0.008),
                makeRelu6Face1024(15, 0.014),
                makeRelu6Face1024(15, 0.019),
              ],
              geometricRatios: geom,
            },
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          },
        ],
        // Completely empty patient_biometrics table!
        patient_biometrics: [],
      },
    };
    const mockFetch = createSimulatedPostgrestFetch(dbCfg);

    const coldRouter = createCloudRouter({
      supabaseUrl: 'https://test-hist-only.supabase.co',
      supabaseAnonKey: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyb2xlIjoiYW5vbiJ9.signature1234567890',
      customSupabaseFetch: mockFetch,
    });

    const statsRes = await coldRouter(
      new Request('https://kiosk.local/api/biometrics/stats?refresh=1', { method: 'GET' }),
    );
    const statsBody = (await statsRes.json()) as {
      compatibleTemplatesCount: number;
      totalActivePatients: number;
    };
    expect(statsBody.totalActivePatients).toBe(1);
    expect(statsBody.compatibleTemplatesCount).toBe(1);

    const verifyRes = await coldRouter(
      new Request('https://kiosk.local/api/biometric/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          probeVector: makeRelu6Face1024(15, 0.01),
          engine: 'human-faceres-1024d',
          geometricRatios: geom,
          yawDeg: 0,
          pitchDeg: 0,
          livenessEvidence: {
            biologicalVerified: true,
            blinkCount: 1,
            spoofRisk: 0.05,
            nonRigidScore: 0.5,
          },
        }),
      }),
    );
    expect(verifyRes.status).toBe(200);
    const verifyBody = (await verifyRes.json()) as { matched: boolean; patientId: string | null };
    expect(verifyBody.matched).toBe(true);
    expect(verifyBody.patientId).toBe(existingPid);

    // Wait briefly for self-healing background backfill to populate patient_biometrics
    await new Promise((r) => setTimeout(r, 80));
    expect(dbCfg.tables.patient_biometrics.length).toBe(1);
  });

  it('does NOT abandon enrollment when Supabase write takes > 1500ms (cold DB wake) and preserves vectors across deploy', async () => {
    const dbCfg: SimulatedDbConfig = {
      schema: {
        ...makeLegacyPartialSchema(),
        // Full modern schema so 2 sequential phases take ~1900ms (> old 1500ms hard cap, < new 6000ms budget)
        patients: new Set([
          ...makeLegacyPartialSchema().patients,
          'master_embedding',
          'angle_embeddings',
          'face_embedding',
          'biometric_data',
          'biometric_updated_at',
          'gender',
          'stroke_type',
          'affected_side',
          'motor_assessment_score',
          'adl_dressing_score',
          'max_safe_rom_deg',
          'universal_health_link_code',
          'delegated_to_pt_id',
          'delegation_note',
        ]),
        patient_biometrics: new Set([
          'id',
          'patient_id',
          'hn',
          'master_embedding',
          'angle_embeddings',
          'exemplar_embeddings',
          'geometric_ratios',
          'biometric_engine',
          'model_version',
          'embedding_dim',
          'frame_count',
          'frames_captured',
          'quality_score',
          'avatar_url',
          'updated_at',
        ]),
      },
      tables: {},
      writeDelayMs: 950,
    };
    const mockFetch = createSimulatedPostgrestFetch(dbCfg);

    const router1 = createCloudRouter({
      supabaseUrl: 'https://test-slow-wake.supabase.co',
      supabaseAnonKey: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyb2xlIjoiYW5vbiJ9.signature1234567890',
      customSupabaseFetch: mockFetch,
    });
    const token = await loginAdminOnRouter(router1);

    const masterVec = makeRelu6Face1024(23);
    const geom = makeGeometricRatios(23);
    const enrollRes = await router1(
      new Request('https://kiosk.local/api/biometric/enroll', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({
          hn: 'HN-SLOW-WAKE-01',
          fullName: 'Narin SlowWake',
          age: 64,
          diagnosis: 'Shoulder Rehab',
          masterEmbedding: masterVec,
          angleEmbeddings: {
            center: makeRelu6Face1024(23, 0.01),
            left: makeRelu6Face1024(23, 0.02),
            right: makeRelu6Face1024(23, 0.02),
            up: makeRelu6Face1024(23, 0.02),
            down: makeRelu6Face1024(23, 0.02),
          },
          exemplarEmbeddings: [
            makeRelu6Face1024(23, 0.01),
            makeRelu6Face1024(23, 0.015),
            makeRelu6Face1024(23, 0.018),
          ],
          biometricEngine: 'human-faceres-1024d',
          geometricRatios: geom,
          framesCaptured: 120,
          qualityScore: 0.96,
        }),
      }),
    );

    expect(enrollRes.status).toBe(201);
    const enrollBody = (await enrollRes.json()) as { persisted: boolean; patient: { id: string } };
    expect(enrollBody.persisted).toBe(true);
    expect(dbCfg.tables.patient_biometrics.length).toBe(1);

    // Cold restart with fast reads
    dbCfg.writeDelayMs = 0;
    resetUnsupportedColumnsForTests();
    resetSupabaseCircuitForTests();
    const router2 = createCloudRouter({
      supabaseUrl: 'https://test-slow-wake.supabase.co',
      supabaseAnonKey: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyb2xlIjoiYW5vbiJ9.signature1234567890',
      customSupabaseFetch: mockFetch,
    });

    const verifyRes = await router2(
      new Request('https://kiosk.local/api/biometric/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          probeVector: makeRelu6Face1024(23, 0.01),
          engine: 'human-faceres-1024d',
          geometricRatios: geom,
          yawDeg: 0,
          pitchDeg: 0,
          livenessEvidence: {
            biologicalVerified: true,
            blinkCount: 1,
            spoofRisk: 0.05,
            nonRigidScore: 0.5,
          },
        }),
      }),
    );
    expect(verifyRes.status).toBe(200);
    const verifyBody = (await verifyRes.json()) as { matched: boolean; patientId: string | null };
    expect(verifyBody.matched).toBe(true);
    expect(verifyBody.patientId).toBe(enrollBody.patient.id);
  }, 15_000);

  it('preserves saved patients.history biometric vectors during non-biometric PUT /api/patients/:id updates on a cold isolate', async () => {
    const masterVec = makeRelu6Face1024(31);
    const geom = makeGeometricRatios(31);
    const pid = '22222222-3333-4444-8555-666666666666';

    const dbCfg: SimulatedDbConfig = {
      schema: makeLegacyPartialSchema(),
      tables: {
        patients: [
          {
            id: pid,
            hn: 'HN-PROTECT-01',
            full_name: 'Kanya Protected',
            age: 55,
            phone: '080-000-0001',
            diagnosis: 'Upper Limb Rehab',
            registration_status: 'active',
            assigned_pt_id: null,
            next_appointment_date: null,
            face_biometric_status: 'VERIFIED',
            face_frames_count: 120,
            face_image_url: null,
            history: {
              faceBiometricStatus: 'VERIFIED',
              faceFramesCount: 120,
              biometricEngine: 'human-faceres-1024d',
              masterEmbedding: masterVec,
              angleEmbeddings: {
                center: makeRelu6Face1024(31, 0.01),
                left: makeRelu6Face1024(31, 0.02),
                right: makeRelu6Face1024(31, 0.02),
                up: makeRelu6Face1024(31, 0.02),
                down: makeRelu6Face1024(31, 0.02),
              },
              exemplarEmbeddings: [
                makeRelu6Face1024(31, 0.01),
                makeRelu6Face1024(31, 0.014),
                makeRelu6Face1024(31, 0.018),
              ],
              geometricRatios: geom,
            },
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          },
        ],
        patient_biometrics: [],
      },
    };
    const mockFetch = createSimulatedPostgrestFetch(dbCfg);

    const router1 = createCloudRouter({
      supabaseUrl: 'https://test-protect.supabase.co',
      supabaseAnonKey: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyb2xlIjoiYW5vbiJ9.signature1234567890',
      customSupabaseFetch: mockFetch,
    });
    const token = await loginAdminOnRouter(router1);

    const putRes = await router1(
      new Request(`https://kiosk.local/api/patients/${pid}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ phone: '089-999-8888' }),
      }),
    );
    expect(putRes.status).toBe(200);

    // Cold restart #2 -> verify face STILL matches!
    resetUnsupportedColumnsForTests();
    resetSupabaseCircuitForTests();
    const router2 = createCloudRouter({
      supabaseUrl: 'https://test-protect.supabase.co',
      supabaseAnonKey: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyb2xlIjoiYW5vbiJ9.signature1234567890',
      customSupabaseFetch: mockFetch,
    });

    const verifyRes = await router2(
      new Request('https://kiosk.local/api/biometric/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          probeVector: makeRelu6Face1024(31, 0.01),
          engine: 'human-faceres-1024d',
          geometricRatios: geom,
          yawDeg: 0,
          pitchDeg: 0,
          livenessEvidence: {
            biologicalVerified: true,
            blinkCount: 1,
            spoofRisk: 0.05,
            nonRigidScore: 0.5,
          },
        }),
      }),
    );
    expect(verifyRes.status).toBe(200);
    const verifyBody = (await verifyRes.json()) as { matched: boolean; patientId: string | null };
    expect(verifyBody.matched).toBe(true);
    expect(verifyBody.patientId).toBe(pid);
  });

  it('honestly returns persisted: false when Supabase rejects patients writes instead of silently faking persisted: true', async () => {
    const dbCfg: SimulatedDbConfig = {
      schema: makeLegacyPartialSchema(),
      tables: {},
      failWritesOnTable: {
        patients: { status: 403, message: 'new row violates row-level security policy for table "patients"' },
      },
    };
    const mockFetch = createSimulatedPostgrestFetch(dbCfg);

    const router = createCloudRouter({
      supabaseUrl: 'https://test-rls-reject.supabase.co',
      supabaseAnonKey: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyb2xlIjoiYW5vbiJ9.signature1234567890',
      customSupabaseFetch: mockFetch,
    });
    const token = await loginAdminOnRouter(router);

    const enrollRes = await router(
      new Request('https://kiosk.local/api/biometric/enroll', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({
          hn: 'HN-RLS-FAIL',
          fullName: 'Blocked Patient',
          age: 50,
          diagnosis: 'Rehab',
          masterEmbedding: makeRelu6Face1024(41),
          angleEmbeddings: {
            center: makeRelu6Face1024(41, 0.01),
            left: makeRelu6Face1024(41, 0.02),
            right: makeRelu6Face1024(41, 0.02),
            up: makeRelu6Face1024(41, 0.02),
            down: makeRelu6Face1024(41, 0.02),
          },
          framesCaptured: 120,
        }),
      }),
    );
    expect(enrollRes.status).toBe(201);
    const body = (await enrollRes.json()) as { ok: boolean; persisted: boolean };
    expect(body.ok).toBe(true);
    expect(body.persisted).toBe(false);
  });

  it('registers slow writes in waitUntil so Cloudflare Workers never cancels in-flight Supabase writes', async () => {
    const fakeSb = {} as unknown as SupabaseClient;
    let writeCompleted = false;
    const backgroundPromises: Promise<unknown>[] = [];

    const res = await persistWithTimeoutBestEffort(
      fakeSb,
      async () => {
        await new Promise((r) => setTimeout(r, 650));
        writeCompleted = true;
        return { error: null };
      },
      500,
      'waituntil_test',
      (p) => {
        backgroundPromises.push(p);
      },
    );

    expect(res.persisted).toBe(false);
    expect(res.errorMessage).toBe('timeout');
    expect(backgroundPromises.length).toBe(1);
    expect(writeCompleted).toBe(false);

    await Promise.all(backgroundPromises);
    expect(writeCompleted).toBe(true);
  });

  it('includes idempotent migration 013 for post-deploy biometric schema completeness and backfill', () => {
    const sqlPath = path.resolve(
      process.cwd(),
      'supabase/migrations/013_permanent_biometric_post_deploy_sync.sql',
    );
    expect(fs.existsSync(sqlPath)).toBe(true);
    const sql = fs.readFileSync(sqlPath, 'utf8');
    expect(sql).toContain('ADD COLUMN IF NOT EXISTS geometric_ratios');
    expect(sql).toContain('ADD COLUMN IF NOT EXISTS frame_count');
    expect(sql).toContain('ADD COLUMN IF NOT EXISTS embedding_dim');
    expect(sql).toContain('ADD COLUMN IF NOT EXISTS exemplar_embeddings');
    expect(sql).toContain('INSERT INTO public.patient_biometrics');
    expect(sql).toContain("NOTIFY pgrst, 'reload schema'");
  });

  it('prevents Supabase egress bombs: lightweight /api/users, /api/staff-calls, /api/patients, cached /api/biometrics/stats and /api/health never download embedding vectors', async () => {
    const dbCfg: SimulatedDbConfig = {
      schema: makeLegacyPartialSchema(),
      tables: {
        patients: [
          {
            id: '33333333-4444-4555-8666-777777777777',
            hn: 'HN-EGRESS-01',
            full_name: 'Egress Guard Patient',
            age: 60,
            phone: '081-000-9999',
            diagnosis: 'Shoulder Rehab',
            registration_status: 'active',
            assigned_pt_id: null,
            next_appointment_date: null,
            face_biometric_status: 'VERIFIED',
            face_frames_count: 120,
            face_image_url: null,
            history: { faceBiometricStatus: 'VERIFIED', faceFramesCount: 120 },
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          },
        ],
        patient_biometrics: [
          {
            patient_id: '33333333-4444-4555-8666-777777777777',
            hn: 'HN-EGRESS-01',
            master_embedding: makeRelu6Face1024(5),
            angle_embeddings: { center: makeRelu6Face1024(5) },
            frames_captured: 120,
            quality_score: 0.96,
            avatar_url: null,
            updated_at: new Date().toISOString(),
          },
        ],
      },
    };

    const recordedQueries: Array<{ table: string; select: string }> = [];
    const baseFetch = createSimulatedPostgrestFetch(dbCfg);
    const trackingFetch: typeof fetch = async (input, init) => {
      const urlStr = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
      if (urlStr.includes('/rest/v1/')) {
        const u = new URL(urlStr);
        const table = u.pathname.split('/rest/v1/')[1]?.split('/')[0] ?? '';
        recordedQueries.push({ table, select: u.searchParams.get('select') ?? '' });
      }
      return baseFetch(input, init);
    };

    const router = createCloudRouter({
      supabaseUrl: 'https://test-egress-guard.supabase.co',
      supabaseAnonKey: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyb2xlIjoiYW5vbiJ9.signature1234567890',
      customSupabaseFetch: trackingFetch,
    });
    const token = await loginAdminOnRouter(router);

    // 1. GET /api/users?refresh=1 must NEVER touch patients or patient_biometrics
    recordedQueries.length = 0;
    const usersRes = await router(
      new Request('https://kiosk.local/api/users?refresh=1', {
        method: 'GET',
        headers: { Authorization: `Bearer ${token}` },
      }),
    );
    expect(usersRes.status).toBe(200);
    expect(recordedQueries.some((q) => q.table === 'patient_biometrics')).toBe(false);
    expect(recordedQueries.some((q) => q.table === 'patients' && q.select !== 'id')).toBe(false);

    // 2. GET /api/staff-calls must ONLY touch staff_calls (never 19-table hydration!)
    recordedQueries.length = 0;
    const callsRes = await router(
      new Request('https://kiosk.local/api/staff-calls', {
        method: 'GET',
        headers: { Authorization: `Bearer ${token}` },
      }),
    );
    expect(callsRes.status).toBe(200);
    expect(recordedQueries.some((q) => q.table === 'patient_biometrics')).toBe(false);
    expect(recordedQueries.some((q) => q.table === 'patients')).toBe(false);

    // 3. GET /api/patients must NEVER download master_embedding / angle_embeddings / SELECT *
    recordedQueries.length = 0;
    const patRes = await router(
      new Request('https://kiosk.local/api/patients?scope=all&refresh=1', {
        method: 'GET',
        headers: { Authorization: `Bearer ${token}` },
      }),
    );
    expect(patRes.status).toBe(200);
    for (const q of recordedQueries) {
      if (q.table === 'patient_biometrics') {
        expect(q.select).not.toContain('master_embedding');
        expect(q.select).not.toContain('angle_embeddings');
        expect(q.select).not.toContain('exemplar_embeddings');
      }
      if (q.table === 'patients') {
        expect(q.select).not.toBe('*');
      }
    }

    // 4. Repeated GET /api/biometrics/stats hits RAM cache (0 extra Supabase queries!)
    recordedQueries.length = 0;
    await router(new Request('https://kiosk.local/api/biometrics/stats', { method: 'GET' }));
    const firstStatsCount = recordedQueries.length;
    expect(firstStatsCount).toBeGreaterThan(0);
    await router(new Request('https://kiosk.local/api/biometrics/stats', { method: 'GET' }));
    await router(new Request('https://kiosk.local/api/biometrics/stats', { method: 'GET' }));
    expect(recordedQueries.length).toBe(firstStatsCount);

    // 5. KioskBiometricLoginView no longer subscribes to 30s full-vector GET /api/biometrics/index polling
    const kioskSrc = fs.readFileSync(
      path.resolve(process.cwd(), 'src/components/KioskBiometricLoginView.tsx'),
      'utf8',
    );
    expect(kioskSrc).not.toContain('subscribeBiometricIndex(');
    expect(kioskSrc).toContain('BIOMETRIC_INDEX_CHANGED_EVENT');
  });
});

