import { describe, expect, it } from 'vitest';
import {
  CANONICAL_HUMAN_FACE_BASELINE_512,
  FACE_EMBEDDING_DIM,
  l2Normalize,
} from '@drtechcares/smart-biometric-core';
import type { SupabaseClient } from '@supabase/supabase-js';
import { createCloudDelegatedRouter } from '../index';
import { hydrateStateFromSupabase, type CloudRouterState } from '../supabasePersistence';

function makeHybridEmbedding(seed: number): number[] {
  const raw = new Array<number>(FACE_EMBEDDING_DIM);
  for (let i = 0; i < FACE_EMBEDDING_DIM; i += 1) {
    const dctRipple = i < 128 ? Math.sin((i + 1) * (seed * 0.37)) * 0.28 : 0;
    const patchRipple = i >= 128 && i < 256 ? Math.cos((i - 127) * (seed * 0.53)) * 0.24 : 0;
    const ratioRipple = i >= 256 ? Math.sin((i - 255) * (seed * 0.71)) * 0.045 : 0;
    raw[i] = CANONICAL_HUMAN_FACE_BASELINE_512[i] + dctRipple + patchRipple + ratioRipple;
  }
  return l2Normalize(raw);
}

async function callRouter(
  router: (req: Request) => Promise<Response>,
  path: string,
  init: { method?: string; token?: string; deviceId?: string; body?: unknown } = {},
) {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (init.token) headers.Authorization = `Bearer ${init.token}`;
  if (init.deviceId) headers['x-device-id'] = init.deviceId;
  const req = new Request(`https://drtechcares.workers.dev${path}`, {
    method: init.method ?? 'GET',
    headers,
    body: init.body !== undefined ? JSON.stringify(init.body) : undefined,
  });
  return router(req);
}

describe('Cloudflare Worker Delegated API Router (Full Parity)', () => {
  const router = createCloudDelegatedRouter({});
  const ownerCentroid = makeHybridEmbedding(2.15);

  it('handles auth, RBAC, 10-shot biometric enroll + verify, clinical guards, security bans, and cascading purge', async () => {
    const health = await callRouter(router, '/api/health');
    expect(health.status).toBe(200);

    // Bad credentials
    const badLogin = await callRouter(router, '/api/auth/login', {
      method: 'POST',
      body: { username: 'admin', password: 'WrongPassword' },
    });
    expect(badLogin.status).toBe(401);

    // Valid logins
    const adminLogin = await callRouter(router, '/api/auth/login', {
      method: 'POST',
      body: { username: 'admin', password: 'Admin@2026' },
    });
    expect(adminLogin.status).toBe(200);
    const { token: adminToken } = (await adminLogin.json()) as { token: string };

    const pt1Login = await callRouter(router, '/api/auth/login', {
      method: 'POST',
      body: { username: 'pt_somchai', password: 'Pt@2026' },
    });
    expect(pt1Login.status).toBe(200);
    const { token: pt1Token } = (await pt1Login.json()) as { token: string };

    const pt2Login = await callRouter(router, '/api/auth/login', {
      method: 'POST',
      body: { username: 'pt_waranya', password: 'Pt@2026' },
    });
    expect(pt2Login.status).toBe(200);
    const { token: pt2Token } = (await pt2Login.json()) as { token: string };

    // RBAC checks
    const auditAsPt = await callRouter(router, '/api/audit-logs', { token: pt1Token });
    expect(auditAsPt.status).toBe(403);
    const auditAsAdmin = await callRouter(router, '/api/audit-logs', { token: adminToken });
    expect(auditAsAdmin.status).toBe(200);

    // 10-shot biometric enrollment
    const rawFrames = [
      ...Array.from({ length: 4 }, () => ({ pose: 'front', vector: ownerCentroid, qualityScore: 0.96 })),
      ...Array.from({ length: 2 }, () => ({ pose: 'left', vector: ownerCentroid, qualityScore: 0.91 })),
      ...Array.from({ length: 2 }, () => ({ pose: 'right', vector: ownerCentroid, qualityScore: 0.92 })),
      ...Array.from({ length: 2 }, () => ({ pose: 'nod', vector: ownerCentroid, qualityScore: 0.93 })),
    ];
    const enrollRes = await callRouter(router, '/api/biometric/enroll', {
      method: 'POST',
      token: pt1Token,
      body: {
        hn: 'HN-EDGE-01',
        fullName: 'คุณสมชาย EdgeTest',
        age: 65,
        diagnosis: 'Cognitive Rehab',
        frontalCentroid: ownerCentroid,
        overallCentroid: ownerCentroid,
        rawFrames,
      },
    });
    expect(enrollRes.status).toBe(201);
    const { patient } = (await enrollRes.json()) as { patient: { id: string } };

    // Verify genuine vs stranger
    const genuineRes = await callRouter(router, '/api/biometric/verify', {
      method: 'POST',
      body: { probeVector: ownerCentroid },
    });
    expect(genuineRes.status).toBe(200);
    const genuineJson = (await genuineRes.json()) as { matched: boolean; patient: { id: string } };
    expect(genuineJson.matched).toBe(true);
    expect(genuineJson.patient.id).toBe(patient.id);

    const strangerRes = await callRouter(router, '/api/biometric/verify', {
      method: 'POST',
      body: { probeVector: makeHybridEmbedding(9.12) },
    });
    expect(strangerRes.status).toBe(401);

    // Append-only medical notes
    const noteRes = await callRouter(router, `/api/patients/${patient.id}/notes`, {
      method: 'POST',
      token: pt1Token,
      body: { content: 'Good focus during matching exercise' },
    });
    expect(noteRes.status).toBe(201);
    const { note } = (await noteRes.json()) as { note: { id: string } };

    const editNoteAsPt = await callRouter(router, `/api/patients/${patient.id}/notes/${note.id}`, {
      method: 'PUT',
      token: pt1Token,
      body: { content: 'Edited' },
    });
    expect(editNoteAsPt.status).toBe(403);
    expect(((await editNoteAsPt.json()) as { error: string }).error).toBe('APPEND_ONLY_NOTES');

    // PT physio course lock
    const physioRes = await callRouter(router, `/api/courses/${patient.id}`, {
      method: 'PUT',
      token: pt1Token,
      body: {
        title: 'Blocked Physio',
        cognitiveModpackId: 'modpack-core-99',
        weeklySchedule: [0, 1, 2, 3, 4, 5, 6].map((dayIndex) => ({
          dayIndex,
          dayType: 'training',
          activityType: 'physio_exercise',
          difficulty: 'medium',
          label: 'Shoulder',
          exercises: [],
        })),
      },
    });
    expect(physioRes.status).toBe(403);
    expect(((await physioRes.json()) as { error: string }).error).toBe('PHYSIO_LOCKED_UNDER_DEVELOPMENT');

    // Delegation with date and note
    const delegateRes = await callRouter(router, `/api/patients/${patient.id}/delegate`, {
      method: 'POST',
      token: pt1Token,
      body: { targetPtId: 'user-pt-2', note: 'ฝากเคส', delegationDate: '2026-11-15' },
    });
    expect(delegateRes.status).toBe(200);
    const delegateJson = (await delegateRes.json()) as {
      patient: { delegatedToPtId: string | null; delegationNote: string | null; delegationDate?: string | null };
    };
    expect(delegateJson.patient.delegatedToPtId).toBe('user-pt-2');
    expect(delegateJson.patient.delegationNote).toBe('ฝากเคส');
    expect(delegateJson.patient.delegationDate).toBe('2026-11-15');

    const pt2Patients = await callRouter(router, '/api/patients', { token: pt2Token });
    const pt2List = ((await pt2Patients.json()) as { patients: { id: string }[] }).patients;
    expect(pt2List.some((p) => p.id === patient.id)).toBe(true);

    // Scoped global library
    const delGlobalEx = await callRouter(router, '/api/exercises/ex-sh-flex-left', {
      method: 'DELETE',
      token: pt1Token,
    });
    if (delGlobalEx.status !== 404) {
      expect(delGlobalEx.status).toBe(403);
    }

    // Inspect frame + biomechanics evaluate
    const inspectRes = await callRouter(router, '/api/biometric/inspect-frame', {
      method: 'POST',
      body: {
        faceCount: 1,
        requestedPose: 'TURN_LEFT',
        metrics: {
          faceHeightRatio: 0.55,
          brightness: 120,
          sharpness: 18,
          eyesVisible: true,
          mouthVisible: true,
          isLive: true,
          depthScore: 0.54,
          yawDeg: -24,
          pitchDeg: 0,
        },
      },
    });
    expect(inspectRes.status).toBe(200);
    expect(((await inspectRes.json()) as { scan_status: string }).scan_status).toBe('PASS');

    const bioEvalRes = await callRouter(router, '/api/biomechanics/evaluate', {
      method: 'POST',
      body: {
        auth_status: { verified: true, patient_id: patient.id, confidence: 0.95, liveness: true },
        spatial_framing: { distance_m: 2.0, full_body_visible: true },
        exercise_prescription: {
          exercise_name: 'Right Shoulder Abduction',
          target_rom_deg: 150,
          hold_sec: 3,
          target_reps: 10,
        },
        live_telemetry: {
          primary_angle_deg: 138.5,
          trunk_tilt_deg: 14.2,
          shoulder_elevation_cm: 1.1,
          angular_velocity_deg_s: 28,
          hold_elapsed_sec: 0,
          pain_score: 0.1,
        },
      },
    });
    expect(bioEvalRes.status).toBe(200);
    expect(((await bioEvalRes.json()) as { kiosk_state: string }).kiosk_state).toBe('ACTIVE_REP');

    // Staff Calls lifecycle: Create -> List -> Acknowledge -> Resolve -> Delete
    const createCallRes = await callRouter(router, '/api/staff-calls', {
      method: 'POST',
      body: {
        kioskId: 'KIOSK-OPD-01',
        patientId: 'HN-9988',
        reason: 'ทดสอบเรียกเจ้าหน้าที่ฉุกเฉินจากหน้าตู้ Kiosk',
      },
    });
    expect(createCallRes.status).toBe(201);
    const createdCall = ((await createCallRes.json()) as { call: { id: string; status: string; kioskId: string } }).call;
    expect(createdCall.status).toBe('open');
    expect(createdCall.kioskId).toBe('KIOSK-OPD-01');

    const listCallsRes = await callRouter(router, '/api/staff-calls', { token: pt1Token });
    expect(listCallsRes.status).toBe(200);
    const listedCalls = ((await listCallsRes.json()) as { calls: { id: string; status: string; createdAt: string }[] }).calls;
    expect(listedCalls.some((c) => c.id === createdCall.id && c.status === 'open')).toBe(true);
    for (let i = 1; i < listedCalls.length; i += 1) {
      expect(Date.parse(listedCalls[i - 1].createdAt)).toBeGreaterThanOrEqual(
        Date.parse(listedCalls[i].createdAt),
      );
    }

    const ackCallRes = await callRouter(router, `/api/staff-calls/${createdCall.id}/acknowledge`, {
      method: 'PATCH',
      token: pt1Token,
    });
    expect(ackCallRes.status).toBe(200);
    const ackedJson = (await ackCallRes.json()) as { ok: boolean; call: { status: string } };
    expect(ackedJson.ok).toBe(true);
    expect(ackedJson.call.status).toBe('acknowledged');

    const resolveCallRes = await callRouter(router, `/api/staff-calls/${createdCall.id}/resolve`, {
      method: 'PATCH',
      token: adminToken,
    });
    expect(resolveCallRes.status).toBe(200);
    const resolvedJson = (await resolveCallRes.json()) as { ok: boolean; call: { status: string } };
    expect(resolvedJson.ok).toBe(true);
    expect(resolvedJson.call.status).toBe('resolved');

    const deleteCallRes = await callRouter(router, `/api/staff-calls/${createdCall.id}`, {
      method: 'DELETE',
      token: adminToken,
    });
    expect(deleteCallRes.status).toBe(200);

    const listCallsAfterDelete = await callRouter(router, '/api/staff-calls', { token: adminToken });
    const afterDeleteJson = (await listCallsAfterDelete.json()) as { calls: { id: string }[] };
    expect(afterDeleteJson.calls.some((c) => c.id === createdCall.id)).toBe(false);

    const auditLogsRes = await callRouter(router, '/api/audit-logs', { token: adminToken });
    expect(auditLogsRes.status).toBe(200);
    const auditPayload = (await auditLogsRes.json()) as {
      logs: { eventType: string; createdAt: string }[];
      activeSessions: { token: string; createdAt: string }[];
    };
    const auditLogs = auditPayload.logs;
    expect(auditLogs.some((l) => l.eventType === 'KIOSK_STAFF_CALL')).toBe(true);
    expect(auditLogs.some((l) => l.eventType === 'STAFF_CALL_RESOLVED')).toBe(true);
    for (let i = 1; i < auditLogs.length; i += 1) {
      expect(Date.parse(auditLogs[i - 1].createdAt)).toBeGreaterThanOrEqual(
        Date.parse(auditLogs[i].createdAt),
      );
    }
    for (let i = 1; i < auditPayload.activeSessions.length; i += 1) {
      expect(Date.parse(auditPayload.activeSessions[i - 1].createdAt)).toBeGreaterThanOrEqual(
        Date.parse(auditPayload.activeSessions[i].createdAt),
      );
    }

    // Force logout + Device ban + Cascading patient purge
    const forceRes = await callRouter(router, '/api/security/force-logout/user-pt-2', {
      method: 'POST',
      token: adminToken,
    });
    expect(forceRes.status).toBe(200);
    const pt2Me = await callRouter(router, '/api/auth/me', { token: pt2Token });
    expect(pt2Me.status).toBe(401);

    const banRes = await callRouter(router, '/api/security/ban', {
      method: 'POST',
      token: adminToken,
      body: { targetType: 'device', value: 'DEV-BLOCKED', reason: 'Test ban' },
    });
    expect(banRes.status).toBe(201);
    const blocked = await callRouter(router, '/api/health', { deviceId: 'DEV-BLOCKED' });
    expect(blocked.status).toBe(403);

    const delPat = await callRouter(router, `/api/patients/${patient.id}`, {
      method: 'DELETE',
      token: adminToken,
    });
    expect(delPat.status).toBe(200);
    expect(((await delPat.json()) as { cascadingPurge: boolean }).cascadingPurge).toBe(true);

    const verifyDeleted = await callRouter(router, '/api/biometric/verify', {
      method: 'POST',
      body: { probeVector: ownerCentroid },
    });
    expect(verifyDeleted.status).toBe(401);
  });

  it('merges staff_calls, audit_logs, and banned_targets across Supabase hydration without wiping in-memory entries', async () => {
    const emptyResult = { data: [] as unknown[], error: null };
    const singleEmptyResult = { data: null, error: null };
    const makeQuery = () => {
      const q: Record<string, unknown> = {
        then: (resolve: (val: unknown) => void) => resolve(emptyResult),
      };
      q.select = () => q;
      q.order = () => q;
      q.limit = () => q;
      q.eq = () => q;
      q.maybeSingle = () => Promise.resolve(singleEmptyResult);
      return q;
    };
    const mockSupabase = {
      from: (table: string) => {
        const q = makeQuery();
        if (table === 'staff_calls') {
          q.then = (resolve: (val: unknown) => void) =>
            resolve({
              data: [
                {
                  id: 'call-shared',
                  kiosk_id: 'KIOSK-OPD-01',
                  patient_id: null,
                  reason: 'Old open copy from DB',
                  status: 'open',
                  created_at: '2026-11-14T10:00:00.000Z',
                },
                {
                  id: 'call-deleted',
                  kiosk_id: 'KIOSK-OPD-01',
                  patient_id: null,
                  reason: 'Deleted call should not return',
                  status: 'resolved',
                  created_at: '2026-11-14T09:00:00.000Z',
                },
              ],
              error: null,
            });
        }
        return q;
      },
    } as unknown as SupabaseClient;

    const state: CloudRouterState = {
      users: [],
      deleted_staff_call_ids: ['call-deleted'],
      patients: [],
      face_templates: [],
      medical_notes: [],
      patient_courses: [],
      rehab_sessions: [],
      treatment_reports: [],
      exercises: [],
      modpacks: [],
      staff_calls: [
        {
          id: 'call-local-only',
          kioskId: 'KIOSK-OPD-01',
          patientId: null,
          reason: 'คำเรียกเจ้าหน้าที่ในหน่วยความจำ',
          status: 'open',
          createdAt: '2026-11-14T12:00:00.000Z',
        },
        {
          id: 'call-shared',
          kioskId: 'KIOSK-OPD-01',
          patientId: null,
          reason: 'Resolved in memory',
          status: 'resolved',
          createdAt: '2026-11-14T10:00:00.000Z',
        },
      ],
      audit_logs: [
        {
          id: 'audit-local',
          eventType: 'KIOSK_STAFF_CALL',
          actorRole: 'patient_kiosk',
          actorId: 'KIOSK-OPD-01',
          actorName: 'KIOSK-OPD-01',
          ipAddress: '127.0.0.1',
          deviceId: 'kiosk',
          detail: 'Local audit entry',
          createdAt: '2026-11-14T12:00:00.000Z',
        },
      ],
      banned_targets: [],
      active_sessions: [
        {
          token: 'tok-older',
          userId: 'user-super-1',
          role: 'super_admin',
          ipAddress: '127.0.0.1',
          deviceId: 'DEV-1',
          createdAt: '2026-11-14T10:00:00.000Z',
          revoked: false,
        },
        {
          token: 'tok-newer',
          userId: 'user-super-1',
          role: 'super_admin',
          ipAddress: '127.0.0.1',
          deviceId: 'DEV-2',
          createdAt: '2026-11-14T14:26:00.000Z',
          revoked: false,
        },
      ],
      system_settings: {
        aiFaceScanEnabled: true,
        aiFaceThreshold: 0.925,
        aiPoseDetectionEnabled: false,
        kioskAutoResetSec: 6,
      },
      kiosks: [],
    };

    await hydrateStateFromSupabase(mockSupabase, state);

    expect(state.staff_calls.some((c) => c.id === 'call-local-only')).toBe(true);
    expect(state.staff_calls.find((c) => c.id === 'call-shared')?.status).toBe('resolved');
    expect(state.staff_calls.some((c) => c.id === 'call-deleted')).toBe(false);
    expect(state.audit_logs.some((l) => l.id === 'audit-local')).toBe(true);
    expect(state.active_sessions[0].token).toBe('tok-newer');
    expect(state.active_sessions[1].token).toBe('tok-older');
  });
});

