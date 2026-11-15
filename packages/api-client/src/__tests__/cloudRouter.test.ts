import { describe, expect, it } from 'vitest';
import {
  CANONICAL_HUMAN_FACE_BASELINE_512,
  FACE_EMBEDDING_DIM,
  l2Normalize,
} from '@drtechcares/smart-biometric-core';
import type { SupabaseClient } from '@supabase/supabase-js';
import {
  clampTtsText,
  createCloudDelegatedRouter,
  formatLocalIsoDate,
  normalizeTtsLang,
  resolveNextAppointmentDate,
  resolveTodayPatientSchedule,
  synthesizeTtsAudio,
} from '../index';
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
        if (table === 'patients') {
          q.then = (resolve: (val: unknown) => void) =>
            resolve({
              data: [
                {
                  id: 'pat-delegated-1',
                  hn: 'HN-61716',
                  full_name: 'Pdasdwa',
                  age: 45,
                  gender: 'ชาย',
                  phone: '0811112233',
                  diagnosis: 'เข่า (Knee)',
                  assigned_pt_id: '00000000-0000-4000-8000-000000000002',
                  delegated_to_pt_id: null,
                  delegation_note: null,
                  next_appointment_date: '2026-11-15',
                  registration_status: 'active',
                  face_biometric_status: 'VERIFIED',
                  history: {},
                  created_at: '2026-11-14T10:00:00.000Z',
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
      patients: [
        {
          id: 'pat-delegated-1',
          hn: 'HN-61716',
          fullName: 'Pdasdwa',
          age: 45,
          gender: 'ชาย',
          phone: '0811112233',
          diagnosis: 'เข่า (Knee)',
          assignedPtId: '00000000-0000-4000-8000-000000000002',
          delegatedToPtId: '00000000-0000-4000-8000-000000000003',
          delegationDate: '2026-11-14',
          delegationNote: 'ดูแลฝึกช่วงบ่าย',
          nextAppointmentDate: '2026-11-15',
          registrationStatus: 'active',
          enrolledAt: '2026-11-14T10:00:00.000Z',
          faceBiometricStatus: 'VERIFIED',
        },
      ],
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
    const hydratedPatient = state.patients.find((p) => p.id === 'pat-delegated-1');
    expect(hydratedPatient?.delegatedToPtId).toBe('00000000-0000-4000-8000-000000000003');
    expect(hydratedPatient?.delegationDate).toBe('2026-11-14');
    expect(hydratedPatient?.delegationNote).toBe('ดูแลฝึกช่วงบ่าย');

    // Verify explicit cancellation via override remains cleared across hydration
    state.delegation_overrides = {
      'pat-delegated-1': {
        delegatedToPtId: null,
        delegationDate: null,
        delegationNote: null,
      },
    };
    await hydrateStateFromSupabase(mockSupabase, state);
    expect(state.patients.find((p) => p.id === 'pat-delegated-1')?.delegatedToPtId).toBeNull();
  });

  it('validates /api/tts query parameters and synthesizes audio via fallback upstream fetch', async () => {
    const badTts = await callRouter(router, '/api/tts?lang=th');
    expect(badTts.status).toBe(400);

    expect(normalizeTtsLang('th-TH')).toBe('th');
    expect(normalizeTtsLang('en-US')).toBe('en');
    expect(normalizeTtsLang('zh-CN')).toBe('zh-CN');
    expect(normalizeTtsLang('cn')).toBe('zh-CN');
    expect(clampTtsText('  ค่อยๆ   เคลื่อนไหว   ไปยังเป้าหมาย  ')).toBe('ค่อยๆ เคลื่อนไหว ไปยังเป้าหมาย');

    const calls: string[] = [];
    const mockFetch: typeof fetch = async (input) => {
      const urlStr = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
      calls.push(urlStr);
      if (urlStr.includes('client=tw-ob')) {
        return new Response(null, { status: 429 });
      }
      const fakeMp3 = new Uint8Array(512).fill(0xff);
      return new Response(fakeMp3, {
        status: 200,
        headers: { 'Content-Type': 'audio/mpeg' },
      });
    };

    const result = await synthesizeTtsAudio('th', 'สวัสดีครับ', mockFetch);
    expect(calls.length).toBe(2);
    expect(calls[0]).toContain('tl=th');
    expect(calls[1]).toContain('client=gtx');
    expect(result?.provider).toBe('google-gtx');
    expect(result?.contentType).toBe('audio/mpeg');
    expect(result?.audio.byteLength).toBe(512);
  });

  it('enforces 1 rehab session per day by marking completedToday in resolveTodayPatientSchedule and /api/sessions', async () => {
    const todayIso = formatLocalIsoDate(new Date());
    const baseSchedule = {
      id: 'sched-1',
      patientId: 'pat-1',
      ptId: 'user-pt-1',
      scheduledDate: todayIso,
      exerciseId: 'ex-shoulder-flexion-r',
      sets: 3,
      reps: 10,
      status: 'ACTIVE' as const,
    };

    const beforeDone = resolveTodayPatientSchedule({
      rehabSchedules: [baseSchedule],
      rehabSessions: [
        {
          id: 'sess-checkin',
          patientId: 'pat-1',
          activityType: 'physio_exercise',
          exerciseId: 'CLINIC_CHECKIN',
          exerciseName: 'Check-in',
          completedSets: 0,
          completedReps: 0,
          accuracyPercent: 100,
          maxRomDeg: 0,
          painStopTriggered: false,
          aiFeedback: 'Checked in',
          kinematicsReplay: [],
          createdAt: `${todayIso}T08:00:00.000Z`,
        },
      ],
    });
    expect(beforeDone.hasAppointmentToday).toBe(true);
    expect(beforeDone.completedToday).toBe(false);
    expect(beforeDone.todayExercises[0]?.status).toBe('ACTIVE');

    const afterSession = resolveTodayPatientSchedule({
      rehabSchedules: [baseSchedule],
      rehabSessions: [
        {
          id: 'sess-physio-1',
          patientId: 'pat-1',
          activityType: 'physio_exercise',
          exerciseId: 'ex-shoulder-flexion-r',
          exerciseName: 'ยกแขนตรงไปด้านหน้า (ขวา)',
          completedSets: 3,
          completedReps: 10,
          accuracyPercent: 100,
          maxRomDeg: 145,
          painStopTriggered: false,
          aiFeedback: 'Completed',
          kinematicsReplay: [],
          createdAt: new Date().toISOString(),
        },
      ],
    });
    expect(afterSession.completedToday).toBe(true);
    expect(afterSession.todayExercises[0]?.status).toBe('COMPLETED');

    // Router end-to-end check with POST /api/sessions -> /api/biometric/index + /api/biometric/verify
    const dailyRouter = createCloudDelegatedRouter({});
    const ptLogin = await callRouter(dailyRouter, '/api/auth/login', {
      method: 'POST',
      body: { username: 'pt_somchai', password: 'Pt@2026' },
    });
    const { token: ptToken } = (await ptLogin.json()) as { token: string };
    const dailyCentroid = makeHybridEmbedding(5.42);
    const rawFrames = [
      ...Array.from({ length: 4 }, () => ({ pose: 'front', vector: dailyCentroid, qualityScore: 0.96 })),
      ...Array.from({ length: 2 }, () => ({ pose: 'left', vector: dailyCentroid, qualityScore: 0.91 })),
      ...Array.from({ length: 2 }, () => ({ pose: 'right', vector: dailyCentroid, qualityScore: 0.92 })),
      ...Array.from({ length: 2 }, () => ({ pose: 'nod', vector: dailyCentroid, qualityScore: 0.93 })),
    ];
    const enrollRes = await callRouter(dailyRouter, '/api/biometric/enroll', {
      method: 'POST',
      token: ptToken,
      body: {
        hn: 'HN-DAILY-01',
        fullName: 'คุณทดสอบ วันละหนึ่งครั้ง',
        age: 60,
        diagnosis: 'Shoulder Rehab',
        frontalCentroid: dailyCentroid,
        overallCentroid: dailyCentroid,
        rawFrames,
      },
    });
    expect(enrollRes.status).toBe(201);
    const { patient } = (await enrollRes.json()) as { patient: { id: string } };

    // Schedule today's rehab for the patient
    const updateSchedRes = await callRouter(dailyRouter, `/api/patients/${patient.id}`, {
      method: 'PUT',
      token: ptToken,
      body: {
        rehabSchedules: [
          {
            id: 'sched-daily-1',
            patientId: patient.id,
            ptId: 'user-pt-1',
            scheduledDate: todayIso,
            exerciseId: 'ex-shoulder-flexion-r',
            sets: 3,
            reps: 10,
            status: 'ACTIVE',
          },
        ],
      },
    });
    expect(updateSchedRes.status).toBe(200);

    const beforeVerify = await callRouter(dailyRouter, '/api/biometric/verify', {
      method: 'POST',
      body: { probeVector: dailyCentroid },
    });
    expect(beforeVerify.status).toBe(200);
    const beforeVerifyJson = (await beforeVerify.json()) as {
      matched: boolean;
      hasAppointmentToday: boolean;
      completedToday: boolean;
    };
    expect(beforeVerifyJson.matched).toBe(true);
    expect(beforeVerifyJson.hasAppointmentToday).toBe(true);
    expect(beforeVerifyJson.completedToday).toBe(false);

    const sessionRes = await callRouter(dailyRouter, '/api/sessions', {
      method: 'POST',
      body: {
        patientId: patient.id,
        activityType: 'physio_exercise',
        exerciseId: 'ex-shoulder-flexion-r',
        exerciseName: 'ยกแขนตรงไปด้านหน้า (ขวา)',
        completedSets: 3,
        completedReps: 10,
        accuracyPercent: 100,
        maxRomDeg: 145,
        painStopTriggered: false,
        scheduledDate: todayIso,
        markDailyCompleted: true,
      },
    });
    expect(sessionRes.status).toBe(201);

    const afterVerify = await callRouter(dailyRouter, '/api/biometric/verify', {
      method: 'POST',
      body: { probeVector: dailyCentroid },
    });
    expect(afterVerify.status).toBe(200);
    const afterVerifyJson = (await afterVerify.json()) as {
      completedToday: boolean;
      todayExercises: Array<{ status: string }>;
    };
    expect(afterVerifyJson.completedToday).toBe(true);
    expect(afterVerifyJson.todayExercises[0]?.status).toBe('COMPLETED');

    const indexRes = await callRouter(dailyRouter, '/api/biometrics/index');
    expect(indexRes.status).toBe(200);
    const indexJson = (await indexRes.json()) as {
      index: Array<{ patientId: string; completedToday?: boolean }>;
    };
    const indexEntry = indexJson.index.find((e) => e.patientId === patient.id);
    expect(indexEntry?.completedToday).toBe(true);
  });

  it('resolves nextAppointmentDate and nextAppointmentAfterToday directly from rehabSchedules before and after daily completion', async () => {
    const today = new Date();
    const todayIso = formatLocalIsoDate(today);
    const tomorrow = new Date(today.getFullYear(), today.getMonth(), today.getDate() + 1);
    const tomorrowIso = formatLocalIsoDate(tomorrow);
    const future = new Date(today.getFullYear(), today.getMonth(), today.getDate() + 3);
    const futureIso = formatLocalIsoDate(future);

    // 1. Unit checks: today (ACTIVE), tomorrow (REST), +3d (ACTIVE)
    const schedules = [
      {
        id: 'sched-today',
        patientId: 'pat-next-1',
        ptId: 'user-pt-1',
        scheduledDate: todayIso,
        exerciseId: 'ex-shoulder-flexion-r',
        sets: 3,
        reps: 10,
        status: 'ACTIVE' as const,
      },
      {
        id: 'sched-rest',
        patientId: 'pat-next-1',
        ptId: 'user-pt-1',
        scheduledDate: tomorrowIso,
        exerciseId: 'ex-shoulder-flexion-r',
        sets: 3,
        reps: 10,
        status: 'REST' as const,
      },
      {
        id: 'sched-future',
        patientId: 'pat-next-1',
        ptId: 'user-pt-1',
        scheduledDate: futureIso,
        exerciseId: 'ex-shoulder-flexion-l',
        sets: 3,
        reps: 10,
        status: 'ACTIVE' as const,
      },
    ];

    // Before today's completion: nextAppointmentDate = todayIso, nextAppointmentAfterToday = futureIso (skips REST tomorrowIso)
    const beforeSchedule = resolveTodayPatientSchedule({
      rehabSchedules: schedules,
      fallbackNextAppointmentDate: todayIso,
      now: today,
    });
    expect(beforeSchedule.nextAppointmentDate).toBe(todayIso);
    expect(beforeSchedule.nextAppointmentAfterToday).toBe(futureIso);

    // After today's completion: both advance to futureIso, never returning todayIso
    const completedSchedules = schedules.map((item) =>
      item.id === 'sched-today' ? { ...item, status: 'COMPLETED' as const } : item,
    );
    const afterSchedule = resolveTodayPatientSchedule({
      rehabSchedules: completedSchedules,
      fallbackNextAppointmentDate: todayIso,
      now: today,
    });
    expect(afterSchedule.completedToday).toBe(true);
    expect(afterSchedule.nextAppointmentDate).toBe(futureIso);
    expect(afterSchedule.nextAppointmentAfterToday).toBe(futureIso);

    // Even if only today was scheduled and completed, nextAppointmentAfterToday must never repeat todayIso
    expect(
      resolveNextAppointmentDate({
        rehabSchedules: [completedSchedules[0]],
        fallbackNextAppointmentDate: todayIso,
        todayIso,
        completedToday: true,
        afterTodayOnly: true,
      }),
    ).toBe('');

    // 2. End-to-end router check: completing today's session via POST /api/sessions advances patient.nextAppointmentDate to futureIso
    const router = createCloudDelegatedRouter({});
    const ptLogin = await callRouter(router, '/api/auth/login', {
      method: 'POST',
      body: { username: 'pt_somchai', password: 'Pt@2026' },
    });
    const { token: ptToken } = (await ptLogin.json()) as { token: string };
    const centroid = makeHybridEmbedding(6.19);
    const rawFrames = [
      ...Array.from({ length: 4 }, () => ({ pose: 'front', vector: centroid, qualityScore: 0.96 })),
      ...Array.from({ length: 2 }, () => ({ pose: 'left', vector: centroid, qualityScore: 0.91 })),
      ...Array.from({ length: 2 }, () => ({ pose: 'right', vector: centroid, qualityScore: 0.92 })),
      ...Array.from({ length: 2 }, () => ({ pose: 'nod', vector: centroid, qualityScore: 0.93 })),
    ];
    const enrollRes = await callRouter(router, '/api/biometric/enroll', {
      method: 'POST',
      token: ptToken,
      body: {
        hn: 'HN-NEXT-01',
        fullName: 'Pize Next Date Test',
        age: 34,
        diagnosis: 'Shoulder Flexion',
        frontalCentroid: centroid,
        overallCentroid: centroid,
        rawFrames,
      },
    });
    expect(enrollRes.status).toBe(201);
    const { patient } = (await enrollRes.json()) as { patient: { id: string } };

    // Deliberately pass stale nextAppointmentDate alongside rehabSchedules; router normalizes against schedule table
    const putSchedRes = await callRouter(router, `/api/patients/${patient.id}`, {
      method: 'PUT',
      token: ptToken,
      body: {
        nextAppointmentDate: todayIso,
        rehabSchedules: schedules.map((s) => ({ ...s, patientId: patient.id })),
      },
    });
    expect(putSchedRes.status).toBe(200);

    const beforeVerify = await callRouter(router, '/api/biometric/verify', {
      method: 'POST',
      body: { probeVector: centroid },
    });
    const beforeVerifyJson = (await beforeVerify.json()) as {
      patient: { nextAppointmentDate: string };
    };
    expect(beforeVerifyJson.patient.nextAppointmentDate).toBe(todayIso);

    const completeRes = await callRouter(router, '/api/sessions', {
      method: 'POST',
      body: {
        patientId: patient.id,
        activityType: 'physio_exercise',
        exerciseId: 'ex-shoulder-flexion-r',
        exerciseName: 'ยกแขนตรงไปด้านหน้า (ขวา)',
        completedSets: 3,
        completedReps: 10,
        accuracyPercent: 100,
        maxRomDeg: 145,
        painStopTriggered: false,
        scheduledDate: todayIso,
        markDailyCompleted: true,
      },
    });
    expect(completeRes.status).toBe(201);

    const afterVerify = await callRouter(router, '/api/biometric/verify', {
      method: 'POST',
      body: { probeVector: centroid },
    });
    const afterVerifyJson = (await afterVerify.json()) as {
      completedToday: boolean;
      patient: { nextAppointmentDate: string };
    };
    expect(afterVerifyJson.completedToday).toBe(true);
    expect(afterVerifyJson.patient.nextAppointmentDate).toBe(futureIso);

    const indexRes = await callRouter(router, '/api/biometrics/index');
    const indexJson = (await indexRes.json()) as {
      index: Array<{ patientId: string; nextAppointmentDate: string }>;
    };
    expect(indexJson.index.find((e) => e.patientId === patient.id)?.nextAppointmentDate).toBe(
      futureIso,
    );
  });
});

