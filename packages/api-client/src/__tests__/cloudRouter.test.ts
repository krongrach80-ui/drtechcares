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

  it('persists initial rehabSchedules and synced nextAppointmentDate during POST /api/biometric/enroll', async () => {
    const router = createCloudDelegatedRouter({});
    const ptLogin = await callRouter(router, '/api/auth/login', {
      method: 'POST',
      body: { username: 'pt_somchai', password: 'Pt@2026' },
    });
    const { token: ptToken } = (await ptLogin.json()) as { token: string };

    const today = new Date();
    const todayIso = formatLocalIsoDate(today);
    const plus2 = new Date(today.getFullYear(), today.getMonth(), today.getDate() + 2);
    const plus2Iso = formatLocalIsoDate(plus2);

    const centroid = makeHybridEmbedding(7.41);
    const rawFrames = [
      ...Array.from({ length: 4 }, () => ({ pose: 'front', vector: centroid, qualityScore: 0.97 })),
      ...Array.from({ length: 2 }, () => ({ pose: 'left', vector: centroid, qualityScore: 0.93 })),
      ...Array.from({ length: 2 }, () => ({ pose: 'right', vector: centroid, qualityScore: 0.94 })),
      ...Array.from({ length: 2 }, () => ({ pose: 'nod', vector: centroid, qualityScore: 0.95 })),
    ];

    const enrollRes = await callRouter(router, '/api/biometric/enroll', {
      method: 'POST',
      token: ptToken,
      body: {
        hn: 'HN-ENROLL-RX-01',
        fullName: 'คนไข้ใหม่ กำหนดท่ากายภาพ',
        age: 52,
        diagnosis: 'ข้อไหล่ติด',
        nextAppointmentDate: '',
        rehabSchedules: [
          {
            id: 'sched-init-2',
            patientId: 'pending',
            exerciseId: 'ex-03',
            scheduledDate: plus2Iso,
            sets: 3,
            reps: 10,
            status: 'ACTIVE',
            notes: 'ฝึกต่อที่คลินิก',
          },
          {
            id: 'sched-init-1',
            patientId: 'pending',
            exerciseId: 'ex-01',
            scheduledDate: todayIso,
            sets: 3,
            reps: 12,
            status: 'ACTIVE',
            notes: 'เริ่มฝึกวันแรกที่ตู้',
          },
        ],
        frontalCentroid: centroid,
        overallCentroid: centroid,
        rawFrames,
      },
    });
    expect(enrollRes.status).toBe(201);
    const { patient } = (await enrollRes.json()) as {
      patient: {
        id: string;
        nextAppointmentDate: string;
        rehabSchedules: Array<{ id: string; patientId: string; exerciseId?: string; scheduledDate: string }>;
      };
    };

    expect(patient.nextAppointmentDate).toBe(todayIso);
    expect(patient.rehabSchedules).toHaveLength(2);
    expect(patient.rehabSchedules.every((s) => s.patientId === patient.id)).toBe(true);
    expect(patient.rehabSchedules[0]?.exerciseId).toBe('ex-01');

    const verifyRes = await callRouter(router, '/api/biometric/verify', {
      method: 'POST',
      body: { probeVector: centroid },
    });
    expect(verifyRes.status).toBe(200);
    const verifyJson = (await verifyRes.json()) as {
      matched: boolean;
      hasAppointmentToday: boolean;
      todayExercises: Array<{ exerciseId: string; sets: number; reps: number }>;
      patient: { nextAppointmentDate: string };
    };
    expect(verifyJson.matched).toBe(true);
    expect(verifyJson.hasAppointmentToday).toBe(true);
    expect(verifyJson.patient.nextAppointmentDate).toBe(todayIso);
    expect(verifyJson.todayExercises.map((e) => e.exerciseId)).toEqual(['ex-01']);
    expect(verifyJson.todayExercises[0]?.sets).toBe(3);
    expect(verifyJson.todayExercises[0]?.reps).toBe(12);
  });

  it('rejects unregistered stranger sharing 3D craniofacial subband during POST /api/biometric/verify', async () => {
    const router = createCloudDelegatedRouter({});
    const ptLogin = await callRouter(router, '/api/auth/login', {
      method: 'POST',
      body: { username: 'pt_somchai', password: 'Pt@2026' },
    });
    const { token: ptToken } = (await ptLogin.json()) as { token: string };

    const ownerVec = makeHybridEmbedding(4.25);
    const rawFrames = [
      ...Array.from({ length: 4 }, () => ({ pose: 'front', vector: ownerVec, qualityScore: 0.97 })),
      ...Array.from({ length: 2 }, () => ({ pose: 'left', vector: ownerVec, qualityScore: 0.93 })),
      ...Array.from({ length: 2 }, () => ({ pose: 'right', vector: ownerVec, qualityScore: 0.94 })),
      ...Array.from({ length: 2 }, () => ({ pose: 'nod', vector: ownerVec, qualityScore: 0.95 })),
    ];

    const enrollRes = await callRouter(router, '/api/biometric/enroll', {
      method: 'POST',
      token: ptToken,
      body: {
        hn: 'HN-STRANGER-GUARD-01',
        fullName: 'พัทธดนย์ ทดสอบยืนยันตัวตน',
        age: 30,
        diagnosis: 'ฟื้นฟูข้อไหล่',
        frontalCentroid: ownerVec,
        overallCentroid: ownerVec,
        rawFrames,
      },
    });
    expect(enrollRes.status).toBe(201);

    const strangerShared3D = l2Normalize([
      ...makeHybridEmbedding(19.75).slice(0, 256),
      ...ownerVec.slice(256),
    ]);

    const strangerVerify = await callRouter(router, '/api/biometric/verify', {
      method: 'POST',
      body: { probeVector: strangerShared3D },
    });
    expect(strangerVerify.status).toBe(401);
    const strangerJson = (await strangerVerify.json()) as {
      matched: boolean;
      patient?: unknown;
    };
    expect(strangerJson.matched).toBe(false);
    expect(strangerJson.patient).toBeUndefined();

    const ownerVerify = await callRouter(router, '/api/biometric/verify', {
      method: 'POST',
      body: { probeVector: ownerVec },
    });
    expect(ownerVerify.status).toBe(200);
    const ownerJson = (await ownerVerify.json()) as {
      matched: boolean;
      confidencePercent: number;
    };
    expect(ownerJson.matched).toBe(true);
    expect(ownerJson.confidencePercent).toBeGreaterThanOrEqual(85);
  });

  it('persists 5-angle face photos and auto-heals VERIFIED rows with face_frames_count = 0 during hydration', async () => {
    const router = createCloudDelegatedRouter({});
    const ptLogin = await callRouter(router, '/api/auth/login', {
      method: 'POST',
      body: { username: 'pt_somchai', password: 'Pt@2026' },
    });
    const { token: ptToken } = (await ptLogin.json()) as { token: string };

    const vec = makeHybridEmbedding(7.5);
    const angleUrls = {
      center: 'https://cdn.example.test/center.jpg',
      left: 'https://cdn.example.test/left.jpg',
      right: 'https://cdn.example.test/right.jpg',
      up: 'https://cdn.example.test/up.jpg',
      down: 'https://cdn.example.test/down.jpg',
    };
    const compositeUrl = 'https://cdn.example.test/composite-5angles.jpg';

    const enrollRes = await callRouter(router, '/api/biometric/enroll', {
      method: 'POST',
      token: ptToken,
      body: {
        hn: 'HN-5ANGLE-01',
        fullName: 'ผู้ป่วย ทดสอบห้ามุมกล้อง',
        age: 29,
        diagnosis: 'ฟื้นฟูข้อเข่า',
        faceImageUrl: compositeUrl,
        angleImageUrls: angleUrls,
        faceImageUrls: [compositeUrl, ...Object.values(angleUrls)],
        masterEmbedding: vec,
        angleEmbeddings: {
          center: vec,
          left: vec,
          right: vec,
          up: vec,
          down: vec,
        },
        framesCaptured: 120,
        qualityScore: 0.96,
      },
    });
    expect(enrollRes.status).toBe(201);
    const enrolledJson = (await enrollRes.json()) as {
      patient: {
        faceFramesCount?: number;
        faceImageUrl?: string;
        faceImageUrls?: string[];
        angleImageUrls?: Record<string, string>;
      };
    };
    expect(enrolledJson.patient.faceFramesCount).toBe(120);
    expect(enrolledJson.patient.faceImageUrl).toBe(compositeUrl);
    expect(enrolledJson.patient.angleImageUrls).toEqual(angleUrls);
    expect(enrolledJson.patient.faceImageUrls).toHaveLength(6);

    const updatedPayloads: { id: string; payload: Record<string, unknown> }[] = [];
    const emptyResult = { data: [] as unknown[], error: null };
    const singleEmptyResult = { data: null, error: null };
    const makeQuery = (table: string) => {
      const q: Record<string, unknown> = {
        then: (resolve: (val: unknown) => void) => resolve(emptyResult),
      };
      q.select = () => q;
      q.order = () => q;
      q.limit = () => q;
      q.eq = (col: string, val: string) => {
        if (table === 'patients' && q._pendingUpdate && col === 'id') {
          updatedPayloads.push({
            id: val,
            payload: q._pendingUpdate as Record<string, unknown>,
          });
          return Promise.resolve({ error: null });
        }
        return q;
      };
      q.update = (payload: Record<string, unknown>) => {
        const uq = makeQuery(table);
        uq._pendingUpdate = payload;
        return uq;
      };
      q.maybeSingle = () => Promise.resolve(singleEmptyResult);
      return q;
    };

    const mockSupabase = {
      from: (table: string) => {
        const q = makeQuery(table);
        if (table === 'patients') {
          q.then = (resolve: (val: unknown) => void) =>
            resolve({
              data: [
                {
                  id: 'pat-heal-120',
                  hn: 'HN-65964',
                  full_name: 'พัทธดนย์ ทดสอบ',
                  age: 18,
                  phone: '0914448744',
                  diagnosis: 'ปวดข้อเข่า',
                  assigned_pt_id: '00000000-0000-4000-8000-000000000002',
                  registration_status: 'active',
                  face_biometric_status: 'VERIFIED',
                  face_frames_count: 0,
                  face_image_url: compositeUrl,
                  history: {
                    gender: 'ชาย',
                    faceBiometricStatus: 'VERIFIED',
                    faceFramesCount: 0,
                    faceImageUrl: compositeUrl,
                    faceImageUrls: [compositeUrl, ...Object.values(angleUrls)],
                    angleImageUrls: angleUrls,
                  },
                  created_at: '2026-11-15T14:00:00.000Z',
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
      patients: [],
      face_templates: [],
      medical_notes: [],
      patient_courses: [],
      rehab_sessions: [],
      treatment_reports: [],
      exercises: [],
      modpacks: [],
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
      kiosks: [],
    };

    await hydrateStateFromSupabase(mockSupabase, state);
    const healed = state.patients.find((p) => p.id === 'pat-heal-120');
    expect(healed?.faceFramesCount).toBe(120);
    expect(healed?.angleImageUrls).toEqual(angleUrls);
    expect(healed?.faceImageUrls).toHaveLength(6);
    expect(updatedPayloads.some((u) => u.id === 'pat-heal-120' && u.payload.face_frames_count === 120)).toBe(true);
  });

  it('resolves duplicate same-person enrollments without margin self-collision during POST /api/biometric/verify', async () => {
    const router = createCloudDelegatedRouter({});
    const ptLogin = await callRouter(router, '/api/auth/login', {
      method: 'POST',
      body: { username: 'pt_somchai', password: 'Pt@2026' },
    });
    const { token: ptToken } = (await ptLogin.json()) as { token: string };

    const olderVec = makeHybridEmbedding(5.25);
    const newerVec = l2Normalize(
      olderVec.map((v, i) => v * 0.85 + makeHybridEmbedding(6.15)[i] * 0.15),
    );
    const buildFrames = (vec: number[]) => [
      ...Array.from({ length: 4 }, () => ({ pose: 'front', vector: vec, qualityScore: 0.96 })),
      ...Array.from({ length: 2 }, () => ({ pose: 'left', vector: vec, qualityScore: 0.94 })),
      ...Array.from({ length: 2 }, () => ({ pose: 'right', vector: vec, qualityScore: 0.95 })),
      ...Array.from({ length: 2 }, () => ({ pose: 'nod', vector: vec, qualityScore: 0.94 })),
    ];

    const oldEnroll = await callRouter(router, '/api/biometric/enroll', {
      method: 'POST',
      token: ptToken,
      body: {
        hn: 'HN-40683',
        fullName: 'พัทธดนย์ ผ่องอารมย์',
        age: 28,
        diagnosis: 'ฟื้นฟูข้อไหล่',
        frontalCentroid: olderVec,
        overallCentroid: olderVec,
        rawFrames: buildFrames(olderVec),
      },
    });
    expect(oldEnroll.status).toBe(201);

    const newEnroll = await callRouter(router, '/api/biometric/enroll', {
      method: 'POST',
      token: ptToken,
      body: {
        hn: 'HN-54720',
        fullName: 'พัทธดนย์ ผ่องอารมย์',
        age: 28,
        diagnosis: 'ฟื้นฟูข้อไหล่',
        frontalCentroid: newerVec,
        overallCentroid: newerVec,
        rawFrames: buildFrames(newerVec),
      },
    });
    expect(newEnroll.status).toBe(201);

    // Probe near both duplicate enrollments (~0.83-0.96 similarity) must match the newer HN rather than colliding on margin
    const liveProbe = l2Normalize(olderVec.map((v, i) => v * 0.45 + newerVec[i] * 0.55));
    const verifyRes = await callRouter(router, '/api/biometric/verify', {
      method: 'POST',
      body: { probeVector: liveProbe },
    });
    expect(verifyRes.status).toBe(200);
    const verifyJson = (await verifyRes.json()) as {
      matched: boolean;
      confidencePercent: number;
      patient?: { hn: string };
    };
    expect(verifyJson.matched).toBe(true);
    expect(verifyJson.confidencePercent).toBeGreaterThanOrEqual(85);
    expect(verifyJson.patient?.hn).toBe('HN-54720');
  });

  it('persists Stroke Mahidol 0-6 score, computes ROM baseline deltas, and links Universal EHR records', async () => {
    const router = createCloudDelegatedRouter({});
    const loginRes = await callRouter(router, '/api/auth/login', {
      method: 'POST',
      body: { username: 'pt_somchai', password: 'Pt@2026' },
    });
    const ptToken = ((await loginRes.json()) as { token: string }).token;

    // 1. Enroll Stroke patient with Mahidol Grade 3 & safe max ROM 115 deg
    const enrollRes = await callRouter(router, '/api/biometric/enroll', {
      method: 'POST',
      token: ptToken,
      body: {
        hn: 'HN-STROKE-01',
        fullName: 'คุณวิชัย ฟื้นฟูสโตรก',
        age: 67,
        diagnosis: 'Ischemic Stroke แขนขวาอ่อนแรง',
        strokeType: 'ISCHEMIC',
        affectedSide: 'RIGHT',
        motorAssessmentScore: 3,
        adlDressingScore: 2,
        maxSafeRomDeg: 115,
        skipFaceScan: true,
      },
    });
    expect(enrollRes.status).toBe(201);
    const enrolledPatient = ((await enrollRes.json()) as {
      patient: { id: string; universalHealthLinkCode: string; motorAssessmentScore: number; maxSafeRomDeg: number };
    }).patient;
    expect(enrolledPatient.motorAssessmentScore).toBe(3);
    expect(enrolledPatient.maxSafeRomDeg).toBe(115);
    expect(enrolledPatient.universalHealthLinkCode.startsWith('DTC-EHR-')).toBe(true);

    // 2. Record Session 1 (Baseline 35°) then Session 2 (58° -> delta +23°)
    const s1 = await callRouter(router, '/api/sessions', {
      method: 'POST',
      token: ptToken,
      body: {
        patientId: enrolledPatient.id,
        activityType: 'physio_exercise',
        exerciseId: 'STROKE_SH_FLEX_R',
        exerciseName: '[Stroke ADL] ยกแขนขวาไปด้านหน้า',
        completedSets: 2,
        completedReps: 6,
        accuracyPercent: 88,
        maxRomDeg: 35,
      },
    });
    expect(s1.status).toBe(201);

    const s2 = await callRouter(router, '/api/sessions', {
      method: 'POST',
      token: ptToken,
      body: {
        patientId: enrolledPatient.id,
        activityType: 'physio_exercise',
        exerciseId: 'STROKE_SH_FLEX_R',
        exerciseName: '[Stroke ADL] ยกแขนขวาไปด้านหน้า',
        completedSets: 2,
        completedReps: 8,
        accuracyPercent: 94,
        maxRomDeg: 58,
      },
    });
    expect(s2.status).toBe(201);

    const histRes = await callRouter(router, `/api/patients/${enrolledPatient.id}/session-history`, {
      method: 'GET',
      token: ptToken,
    });
    expect(histRes.status).toBe(200);
    const histList = ((await histRes.json()) as {
      history: { baselineRomDeg: number; achievedMaxRomDeg: number; romDeltaDeg: number }[];
    }).history;
    expect(histList.length).toBeGreaterThanOrEqual(2);
    expect(histList[0].baselineRomDeg).toBe(35);
    expect(histList[0].achievedMaxRomDeg).toBe(58);
    expect(histList[0].romDeltaDeg).toBe(23);

    // 3. Universal EHR lookup + external cross-clinic merge into second clinic patient
    const targetEnroll = await callRouter(router, '/api/biometric/enroll', {
      method: 'POST',
      token: ptToken,
      body: {
        hn: 'HN-EXT-99',
        fullName: 'คุณวิชัย ฟื้นฟูสโตรก',
        age: 67,
        diagnosis: 'Refer from Hub Clinic',
        skipFaceScan: true,
      },
    });
    const targetId = ((await targetEnroll.json()) as { patient: { id: string } }).patient.id;

    const lookupRes = await callRouter(
      router,
      `/api/ehr/lookup?code=${encodeURIComponent(enrolledPatient.universalHealthLinkCode)}`,
      { method: 'GET', token: ptToken },
    );
    expect(lookupRes.status).toBe(200);
    const lookupJson = (await lookupRes.json()) as { ehrPackage: { sessionHistory: unknown[] } | null };
    expect(lookupJson.ehrPackage?.sessionHistory.length).toBeGreaterThanOrEqual(2);

    const linkRes = await callRouter(router, `/api/patients/${targetId}/link-ehr`, {
      method: 'POST',
      token: ptToken,
      body: { linkCode: enrolledPatient.universalHealthLinkCode },
    });
    expect(linkRes.status).toBe(200);
    const linkJson = (await linkRes.json()) as { importedCount: number; patient: { strokeType: string } };
    expect(linkJson.importedCount).toBeGreaterThanOrEqual(2);
    expect(linkJson.patient.strokeType).toBe('ISCHEMIC');
  });

  it('rejects /api/biometric/verify with 403 BIOMETRIC_SPOOF_DETECTED when biological liveness fails', async () => {
    const router = createCloudDelegatedRouter({});
    const loginRes = await callRouter(router, '/api/auth/login', {
      method: 'POST',
      body: { username: 'pt_somchai', password: 'Pt@2026' },
    });
    const ptToken = ((await loginRes.json()) as { token: string }).token;

    const ownerVec = makeHybridEmbedding(6.15);
    const rawFrames = [
      ...Array.from({ length: 4 }, () => ({ pose: 'front', vector: ownerVec, qualityScore: 0.97 })),
      ...Array.from({ length: 2 }, () => ({ pose: 'left', vector: ownerVec, qualityScore: 0.93 })),
      ...Array.from({ length: 2 }, () => ({ pose: 'right', vector: ownerVec, qualityScore: 0.94 })),
      ...Array.from({ length: 2 }, () => ({ pose: 'nod', vector: ownerVec, qualityScore: 0.95 })),
    ];
    const enrollRes = await callRouter(router, '/api/biometric/enroll', {
      method: 'POST',
      token: ptToken,
      body: {
        hn: 'HN-BIOSPOOF-01',
        fullName: 'คุณสมชาย ทดสอบชีววิทยา',
        age: 61,
        diagnosis: 'Stroke recovery',
        frontalCentroid: ownerVec,
        overallCentroid: ownerVec,
        rawFrames,
      },
    });
    expect(enrollRes.status).toBe(201);

    // Even with a 100% matching owner face vector, missing blink / phone screen spoof risk is blocked with 403
    const phoneSpoofRes = await callRouter(router, '/api/biometric/verify', {
      method: 'POST',
      body: {
        probeVector: ownerVec,
        livenessEvidence: {
          biologicalVerified: false,
          blinkCount: 0,
          spoofRisk: 0.78,
          nonRigidScore: 0.001,
        },
      },
    });
    expect(phoneSpoofRes.status).toBe(403);
    const phoneSpoofJson = (await phoneSpoofRes.json()) as { matched: boolean; error: string };
    expect(phoneSpoofJson.matched).toBe(false);
    expect(phoneSpoofJson.error).toBe('BIOMETRIC_SPOOF_DETECTED');

    // Genuine biological blink + clean risk passes normally
    const liveRes = await callRouter(router, '/api/biometric/verify', {
      method: 'POST',
      body: {
        probeVector: ownerVec,
        livenessEvidence: {
          biologicalVerified: true,
          blinkCount: 1,
          spoofRisk: 0.08,
          nonRigidScore: 0.14,
          rppgScore: 0.62,
        },
      },
    });
    expect(liveRes.status).toBe(200);
    const liveJson = (await liveRes.json()) as { matched: boolean };
    expect(liveJson.matched).toBe(true);

    // Long-range verified scan (faceHeightRatio = 0.09) with strong genuine core passes
    const slightDiff = makeHybridEmbedding(12.5);
    const longRangeProbe = l2Normalize(
      ownerVec.map((v, i) => v * 0.90 + slightDiff[i] * 0.10),
    );
    const lrRes = await callRouter(router, '/api/biometric/verify', {
      method: 'POST',
      body: {
        probeVector: longRangeProbe,
        livenessEvidence: {
          biologicalVerified: true,
          blinkCount: 0,
          spoofRisk: 0,
          nonRigidScore: 0.08,
          faceHeightRatio: 0.09,
        },
      },
    });
    expect(lrRes.status).toBe(200);
    const lrJson = (await lrRes.json()) as { matched: boolean; patient?: { hn: string } };
    expect(lrJson.matched).toBe(true);
    expect(lrJson.patient?.hn).toBe('HN-BIOSPOOF-01');

    // Enroll a close-looking second person (similarity within < 0.08 margin of owner) -> must return 401 AMBIGUOUS_CLOSE_CANDIDATES
    const closeLookAlikeVec = l2Normalize(
      ownerVec.map((v, i) => v * 0.96 + makeHybridEmbedding(44.2)[i] * 0.04),
    );
    const twinEnrollRes = await callRouter(router, '/api/biometric/enroll', {
      method: 'POST',
      token: ptToken,
      body: {
        hn: 'HN-BIOSPOOF-02',
        fullName: 'คุณแฝด คนละบัญชี',
        age: 62,
        phone: '0890000002',
        diagnosis: 'Knee Rehab',
        frontalCentroid: closeLookAlikeVec,
        overallCentroid: closeLookAlikeVec,
        rawFrames: [
          ...Array.from({ length: 4 }, () => ({ pose: 'front', vector: closeLookAlikeVec, qualityScore: 0.96 })),
          ...Array.from({ length: 2 }, () => ({ pose: 'left', vector: closeLookAlikeVec, qualityScore: 0.95 })),
          ...Array.from({ length: 2 }, () => ({ pose: 'right', vector: closeLookAlikeVec, qualityScore: 0.95 })),
          ...Array.from({ length: 2 }, () => ({ pose: 'nod', vector: closeLookAlikeVec, qualityScore: 0.95 })),
        ],
      },
    });
    expect(twinEnrollRes.status).toBe(201);
    const twinPatient = (await twinEnrollRes.json()) as { patient: { id: string } };

    // Ambiguous close candidates must reject instead of guessing between patient 1 & patient 2
    const ambiguousProbe = l2Normalize(
      ownerVec.map((v, i) => v * 0.985 + makeHybridEmbedding(44.2)[i] * 0.015),
    );
    const ambRes = await callRouter(router, '/api/biometric/verify', {
      method: 'POST',
      body: {
        probeVector: ambiguousProbe,
        livenessEvidence: {
          biologicalVerified: true,
          blinkCount: 1,
          spoofRisk: 0.05,
          nonRigidScore: 0.14,
        },
      },
    });
    expect(ambRes.status).toBe(401);
    const ambJson = (await ambRes.json()) as { matched: boolean; error?: string };
    expect(ambJson.matched).toBe(false);

    // Excluding twinPatient.patient.id ("ไม่ใช่ฉัน (สแกนใหม่)") restores clean unambiguous match to owner
    const excludedRes = await callRouter(router, '/api/biometric/verify', {
      method: 'POST',
      body: {
        probeVector: ambiguousProbe,
        excludedPatientIds: [twinPatient.patient.id],
        livenessEvidence: {
          biologicalVerified: true,
          blinkCount: 1,
          spoofRisk: 0.05,
          nonRigidScore: 0.14,
          excludedPatientIds: [twinPatient.patient.id],
        },
      },
    });
    expect(excludedRes.status).toBe(200);
    const excludedJson = (await excludedRes.json()) as { matched: boolean; patient?: { hn: string } };
    expect(excludedJson.matched).toBe(true);
    expect(excludedJson.patient?.hn).toBe('HN-BIOSPOOF-01');
  });

  it('matches Legacy V1 enrolled patients via legacyV1ProbeVector bridge while rejecting strangers at 401', async () => {
    const router = createCloudDelegatedRouter({});
    const ptLogin = await callRouter(router, '/api/auth/login', {
      method: 'POST',
      body: { username: 'pt_somchai', password: 'Pt@2026' },
    });
    const { token: ptToken } = (await ptLogin.json()) as { token: string };

    const makeLegacyV1Synthetic = (seed: number): number[] => {
      const raw = new Array<number>(FACE_EMBEDDING_DIM);
      for (let i = 0; i < FACE_EMBEDDING_DIM; i += 1) {
        const dctRipple = i < 128 ? Math.sin((i + 1) * (seed * 0.41)) * 0.25 : 0;
        const patchRipple = i >= 128 && i < 256 ? Math.cos((i - 127) * (seed * 0.59)) * 0.22 : 0;
        const ratioRipple = i >= 256 ? Math.sin((i - 255) * (seed * 0.67)) * 0.01 : 0;
        raw[i] =
          i >= 256
            ? CANONICAL_HUMAN_FACE_BASELINE_512[i] * 0.65 + ratioRipple
            : CANONICAL_HUMAN_FACE_BASELINE_512[i] * 0.5 + dctRipple + patchRipple;
      }
      return l2Normalize(raw);
    };

    const v1Owner = makeLegacyV1Synthetic(3.14);
    const v1Stranger = makeLegacyV1Synthetic(9.72);
    const unrelatedV2Probe = l2Normalize(
      makeHybridEmbedding(77.7).map((v, i) => (i >= 256 ? v * 0.14 : v)),
    );

    const enrollV1Res = await callRouter(router, '/api/biometric/enroll', {
      method: 'POST',
      token: ptToken,
      body: {
        hn: 'HN-LEGACY-V1',
        fullName: 'Phattadon LegacyV1',
        age: 29,
        phone: '0897776655',
        diagnosis: 'Knee Rehab',
        frontalCentroid: v1Owner,
        overallCentroid: v1Owner,
        rawFrames: [
          ...Array.from({ length: 4 }, () => ({ pose: 'front', vector: v1Owner, qualityScore: 0.96 })),
          ...Array.from({ length: 2 }, () => ({ pose: 'left', vector: v1Owner, qualityScore: 0.95 })),
          ...Array.from({ length: 2 }, () => ({ pose: 'right', vector: v1Owner, qualityScore: 0.95 })),
          ...Array.from({ length: 2 }, () => ({ pose: 'nod', vector: v1Owner, qualityScore: 0.95 })),
        ],
      },
    });
    expect(enrollV1Res.status).toBe(201);

    const genuineBridgeRes = await callRouter(router, '/api/biometric/verify', {
      method: 'POST',
      body: {
        probeVector: unrelatedV2Probe,
        legacyV1ProbeVector: v1Owner,
        livenessEvidence: {
          biologicalVerified: true,
          blinkCount: 1,
          spoofRisk: 0.05,
          nonRigidScore: 0.14,
          legacyV1ProbeVector: v1Owner,
        },
      },
    });
    expect(genuineBridgeRes.status).toBe(200);
    const genuineJson = (await genuineBridgeRes.json()) as {
      matched: boolean;
      patient?: { hn: string };
    };
    expect(genuineJson.matched).toBe(true);
    expect(genuineJson.patient?.hn).toBe('HN-LEGACY-V1');

    const strangerBridgeRes = await callRouter(router, '/api/biometric/verify', {
      method: 'POST',
      body: {
        probeVector: unrelatedV2Probe,
        legacyV1ProbeVector: v1Stranger,
        livenessEvidence: {
          biologicalVerified: true,
          blinkCount: 1,
          spoofRisk: 0.05,
          nonRigidScore: 0.14,
          legacyV1ProbeVector: v1Stranger,
        },
      },
    });
    expect(strangerBridgeRes.status).toBe(401);
    const strangerJson = (await strangerBridgeRes.json()) as { matched: boolean };
    expect(strangerJson.matched).toBe(false);
  });

  it('enforces engine-isolated ArcFace ResNet-50 1:N verification with zero hybrid cross-matching on POST /api/biometric/verify', async () => {
    const router = createCloudDelegatedRouter({});
    const ptLogin = await callRouter(router, '/api/auth/login', {
      method: 'POST',
      body: { username: 'pt_somchai', password: 'Pt@2026' },
    });
    const { token: ptToken } = (await ptLogin.json()) as { token: string };

    const mkArcVec = (seed: number): number[] => {
      const vec = new Array<number>(FACE_EMBEDDING_DIM);
      let s = seed >>> 0;
      for (let i = 0; i < FACE_EMBEDDING_DIM; i += 1) {
        s = (s * 1664525 + 1013904223 + i * 71) >>> 0;
        vec[i] = ((s % 2000) / 1000 - 1) * (i >= 256 ? 1.4 : 1.0);
      }
      return l2Normalize(vec);
    };

    const arcOwner = mkArcVec(9001);
    const arcOther = mkArcVec(9002);

    const enrollArcRes = await callRouter(router, '/api/biometric/enroll', {
      method: 'POST',
      token: ptToken,
      body: {
        hn: 'HN-ARC-API-01',
        fullName: 'คุณวิชัย อาร์คเฟซ',
        age: 34,
        phone: '0891230001',
        diagnosis: 'Shoulder Rehab',
        biometricEngine: 'arcface-onnx',
        masterEmbedding: arcOwner,
        angleEmbeddings: {
          center: arcOwner,
          left: arcOwner,
          right: arcOwner,
          up: arcOwner,
          down: arcOwner,
        },
        framesCaptured: 120,
        qualityScore: 0.96,
      },
    });
    expect(enrollArcRes.status).toBe(201);

    const arcVerifyOk = await callRouter(router, '/api/biometric/verify', {
      method: 'POST',
      body: {
        probeVector: l2Normalize(arcOwner.map((v, i) => v * 0.72 + mkArcVec(42)[i] * 0.36)),
        engine: 'arcface-onnx',
        livenessEvidence: {
          biologicalVerified: true,
          blinkCount: 1,
          spoofRisk: 0.04,
          nonRigidScore: 0.14,
          engine: 'arcface-onnx',
        },
      },
    });
    expect(arcVerifyOk.status).toBe(200);
    const okJson = (await arcVerifyOk.json()) as { matched: boolean; patient?: { hn: string } };
    expect(okJson.matched).toBe(true);
    expect(okJson.patient?.hn).toBe('HN-ARC-API-01');

    // Different person in ArcFace space rejected cleanly at 401
    const arcVerifyDiff = await callRouter(router, '/api/biometric/verify', {
      method: 'POST',
      body: {
        probeVector: arcOther,
        engine: 'arcface-onnx',
        livenessEvidence: {
          biologicalVerified: true,
          blinkCount: 1,
          spoofRisk: 0.04,
          nonRigidScore: 0.14,
          engine: 'arcface-onnx',
        },
      },
    });
    expect(arcVerifyDiff.status).toBe(401);

    // Hybrid probe MUST NEVER cross-match an ArcFace account
    const hybridCrossReject = await callRouter(router, '/api/biometric/verify', {
      method: 'POST',
      body: {
        probeVector: arcOwner,
        engine: 'hybrid-512d',
        livenessEvidence: {
          biologicalVerified: true,
          blinkCount: 1,
          spoofRisk: 0.04,
          nonRigidScore: 0.14,
          engine: 'hybrid-512d',
        },
      },
    });
    expect(hybridCrossReject.status).toBe(401);
  });
});

