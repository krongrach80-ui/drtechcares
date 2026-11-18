import { describe, expect, it } from 'vitest';
import {
  CANONICAL_HUMAN_FACE_BASELINE_512,
  FACE_EMBEDDING_DIM,
  l2Normalize,
} from '@drtechcares/smart-biometric-core';
import type { SupabaseClient } from '@supabase/supabase-js';
import {
  DrTechCaresApiClient,
  clampTtsText,
  createCloudDelegatedRouter,
  formatLocalIsoDate,
  isAdminRole,
  normalizeTtsLang,
  resolveNextAppointmentDate,
  resolveTodayPatientSchedule,
  synthesizeTtsAudio,
} from '../index';
import {
  HYDRATE_TTL_MS,
  hydrateStaffUsersOnly,
  hydrateStateFromSupabase,
  normalizeCompactExemplars,
  purgePatientInSupabaseBestEffort,
  purgeUserInSupabaseBestEffort,
  resetHydrationCacheForTests,
  resetSupabaseCircuitForTests,
  resetUnsupportedColumnsForTests,
  stripOversizedBase64ForDb,
  toSafeUuidOrNull,
  unsupportedColumnsByTable,
  type CloudRouterState,
} from '../supabasePersistence';

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
    await hydrateStateFromSupabase(mockSupabase, state, { force: true });
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
            exerciseId: 'STROKE_SH_ABD_R',
            scheduledDate: plus2Iso,
            sets: 3,
            reps: 10,
            status: 'ACTIVE',
            notes: 'ฝึกต่อที่คลินิก',
          },
          {
            id: 'sched-init-1',
            patientId: 'pending',
            exerciseId: 'STROKE_SH_FLEX_R',
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
    expect(patient.rehabSchedules[0]?.exerciseId).toBe('STROKE_SH_FLEX_R');

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
    expect(verifyJson.todayExercises.map((e) => e.exerciseId)).toEqual(['STROKE_SH_FLEX_R']);
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
    // Zero unnecessary background Postgres UPDATE writes triggered during read hydration
    expect(updatedPayloads).toHaveLength(0);
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

  it('enrolls and verifies @vladmandic/human faceres 1024-D KNN exemplar embeddings on /api/biometric/enroll, /api/biometrics/index, and /api/biometric/verify', async () => {
    const router = createCloudDelegatedRouter({});
    const ptLogin = await callRouter(router, '/api/auth/login', {
      method: 'POST',
      body: { username: 'pt_somchai', password: 'Pt@2026' },
    });
    const { token: ptToken } = (await ptLogin.json()) as { token: string };

    const mk1024Vec = (seed: number): number[] => {
      const vec = new Array<number>(1024);
      for (let i = 0; i < 1024; i += 1) {
        vec[i] = Math.sin((i + 1) * (seed * 0.91 + 0.37)) + 0.5 * Math.cos((i + 5) * seed);
      }
      return l2Normalize(vec);
    };

    const ownerMaster = mk1024Vec(12);
    const ownerExemplarLeft = l2Normalize(
      ownerMaster.map((v, i) => v * 0.8 + mk1024Vec(24)[i] * 0.25),
    );
    const otherUser = mk1024Vec(77);

    const enrollRes = await callRouter(router, '/api/biometric/enroll', {
      method: 'POST',
      token: ptToken,
      body: {
        hn: 'HN-HUMAN-1024-01',
        fullName: 'คุณพิมพ์พร ฮิวแมน',
        age: 29,
        phone: '0898881024',
        diagnosis: 'Upper Limb Rehab',
        biometricEngine: 'human-faceres-1024d',
        masterEmbedding: ownerMaster,
        angleEmbeddings: {
          center: ownerMaster,
          left: ownerExemplarLeft,
          right: ownerMaster,
          up: ownerMaster,
          down: ownerMaster,
        },
        exemplarEmbeddings: [ownerMaster, ownerExemplarLeft],
        framesCaptured: 120,
        qualityScore: 0.98,
      },
    });
    expect(enrollRes.status).toBe(201);

    // Biometric index should expose 1024-D vectors & exemplarEmbeddings
    const idxRes = await callRouter(router, '/api/biometrics/index', {});
    expect(idxRes.status).toBe(200);
    const idxJson = (await idxRes.json()) as {
      index: { hn: string; biometricEngine?: string; masterEmbedding: number[]; exemplarEmbeddings?: number[][] }[];
    };
    const found = idxJson.index.find((item) => item.hn === 'HN-HUMAN-1024-01');
    expect(found).toBeDefined();
    expect(found?.biometricEngine).toBe('human-faceres-1024d');
    expect(found?.masterEmbedding).toHaveLength(1024);
    expect(found?.exemplarEmbeddings?.length).toBeGreaterThanOrEqual(2);

    // Verify genuine 1024-D probe against left exemplar with noise -> matched!
    const liveProbe = l2Normalize(
      ownerExemplarLeft.map((v, i) => v * 0.78 + mk1024Vec(99)[i] * 0.22),
    );
    const verifyOk = await callRouter(router, '/api/biometric/verify', {
      method: 'POST',
      body: {
        probeVector: liveProbe,
        engine: 'human-faceres-1024d',
        livenessEvidence: {
          biologicalVerified: true,
          blinkCount: 1,
          spoofRisk: 0.03,
          nonRigidScore: 0.16,
          engine: 'human-faceres-1024d',
        },
      },
    });
    expect(verifyOk.status).toBe(200);
    const okJson = (await verifyOk.json()) as { matched: boolean; patient?: { hn: string } };
    expect(okJson.matched).toBe(true);
    expect(okJson.patient?.hn).toBe('HN-HUMAN-1024-01');

    // Stranger 1024-D probe must be rejected at 401
    const strangerRes = await callRouter(router, '/api/biometric/verify', {
      method: 'POST',
      body: {
        probeVector: otherUser,
        engine: 'human-faceres-1024d',
        livenessEvidence: {
          biologicalVerified: true,
          blinkCount: 1,
          spoofRisk: 0.03,
          nonRigidScore: 0.16,
          engine: 'human-faceres-1024d',
        },
      },
    });
    expect(strangerRes.status).toBe(401);
  });

  it('returns exact 15 unified master Stroke & MOPH exercises and purges legacy rows via POST /api/exercises/sync-stroke-library', async () => {
    const router = createCloudDelegatedRouter({});
    const ptLogin = await callRouter(router, '/api/auth/login', {
      method: 'POST',
      body: { username: 'pt_somchai', password: 'Pt@2026' },
    });
    const { token: ptToken } = (await ptLogin.json()) as { token: string };

    // Inject deprecated leg/ortho exercise into router and confirm GET strips it
    const badCreate = await callRouter(router, '/api/exercises', {
      method: 'POST',
      token: ptToken,
      body: {
        id: 'ex-26',
        code: 'SIT_TO_STAND',
        category: 'lower_limb',
        name: { th: 'ลุกนั่งเก้าอี้', en: 'Sit to Stand', cn: '坐站' },
        instruction: { th: '-', en: '-', cn: '-' },
        detection: {
          mode: 'angle_rep_hold',
          jointGroup: 'knee',
          side: 'both',
          targetMinDeg: 85,
          targetMaxDeg: 170,
          startRestMaxDeg: 100,
          holdSec: 2,
          defaultSets: 2,
          defaultReps: 10,
        },
      },
    });
    expect(badCreate.status).toBe(201);

    const listBeforeSync = await callRouter(router, '/api/exercises');
    expect(listBeforeSync.status).toBe(200);
    const listBeforeJson = (await listBeforeSync.json()) as { exercises: Array<{ id: string; code: string; category: string }> };
    expect(listBeforeJson.exercises).toHaveLength(15);
    expect(listBeforeJson.exercises.some((e) => e.category === 'lower_limb' || e.category === 'balance')).toBe(false);
    expect(listBeforeJson.exercises.some((e) => e.id === 'ex-26' || e.code === 'SIT_TO_STAND')).toBe(false);

    const syncRes = await callRouter(router, '/api/exercises/sync-stroke-library', {
      method: 'POST',
      token: ptToken,
    });
    expect(syncRes.status).toBe(200);
    const syncJson = (await syncRes.json()) as {
      ok: boolean;
      purgedCount: number;
      syncedCount: number;
      exercises: Array<{ id: string; code: string; category: string }>;
    };
    expect(syncJson.ok).toBe(true);
    expect(syncJson.purgedCount).toBeGreaterThanOrEqual(1);
    expect(syncJson.syncedCount).toBe(15);
    expect(syncJson.exercises).toHaveLength(15);
  });

  it('never resurrects deleted seeded PT accounts (pt_somchai / pt_waranya) during hydrateStateFromSupabase across fresh Worker isolates', async () => {
    const seededNow = new Date().toISOString();
    const freshIsolateState = {
      users: [
        {
          id: '00000000-0000-4000-8000-000000000001',
          username: 'admin',
          passwordHash: 'plain:Admin@2026',
          role: 'super_admin',
          fullName: 'Super Admin',
          phone: '02-000-0001',
          active: true,
          forceLoggedOutAt: null,
          createdAt: seededNow,
        },
        {
          id: '00000000-0000-4000-8000-000000000002',
          username: 'pt_somchai',
          passwordHash: 'plain:Pt@2026',
          role: 'pt',
          fullName: 'กภ.สมชาย ใจดี (PT)',
          phone: '081-111-2233',
          active: true,
          forceLoggedOutAt: null,
          createdAt: seededNow,
        },
        {
          id: '00000000-0000-4000-8000-000000000003',
          username: 'pt_waranya',
          passwordHash: 'plain:Pt@2026',
          role: 'pt',
          fullName: 'กภ.วรัญญา สุขสันต์ (PT)',
          phone: '089-222-3344',
          active: true,
          forceLoggedOutAt: null,
          createdAt: seededNow,
        },
      ],
      deleted_user_ids: [],
      deleted_patient_ids: [],
      patients: [],
      face_templates: [],
      medical_notes: [],
      patient_courses: [],
      rehab_sessions: [],
      rehab_session_history: [],
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
    } as unknown as CloudRouterState;

    const makeMockSupabaseWithUsers = (rows: Record<string, unknown>[]) =>
      ({
        from: (table: string) => ({
          select: () => ({
            eq: () => ({
              order: async () => ({ data: [], error: null }),
              maybeSingle: async () => ({ data: null, error: null }),
              then: (cb: (v: { data: unknown[]; error: null }) => void) =>
                Promise.resolve({ data: [], error: null }).then(cb),
            }),
            order: () => ({
              limit: async () => ({ data: [], error: null }),
              then: (cb: (v: { data: unknown[]; error: null }) => void) =>
                Promise.resolve({ data: [], error: null }).then(cb),
            }),
            then: (cb: (v: { data: Record<string, unknown>[]; error: null }) => void) =>
              Promise.resolve({ data: table === 'staff_users' ? rows : [], error: null }).then(cb),
          }),
        }),
      }) as unknown as SupabaseClient;

    // 1. Supabase has ONLY admin + 1 patient (pt_somchai & pt_waranya were deleted permanently)
    await hydrateStateFromSupabase(
      makeMockSupabaseWithUsers([
        {
          id: '00000000-0000-4000-8000-000000000001',
          username: 'admin',
          password_hash: 'plain:Admin@2026',
          role: 'super_admin',
          full_name: 'Super Admin',
          phone: '02-000-0001',
          active: true,
          created_at: seededNow,
        },
        {
          id: 'pat-u-1',
          username: 'hn29466',
          password_hash: 'hash',
          role: 'patient',
          full_name: 'lym',
          active: true,
          created_at: seededNow,
        },
      ]),
      freshIsolateState,
    );

    expect(freshIsolateState.users.map((u) => u.username)).toEqual(['admin', 'hn29466']);

    // 2. If staff_users is completely empty (brand-new DB), emergency admin bootstraps without pt_somchai/pt_waranya
    const emptyDbState = {
      ...freshIsolateState,
      users: [
        {
          id: '00000000-0000-4000-8000-000000000001',
          username: 'admin',
          passwordHash: 'plain:Admin@2026',
          role: 'super_admin',
          fullName: 'Super Admin',
          phone: '02-000-0001',
          active: true,
          forceLoggedOutAt: null,
          createdAt: seededNow,
        },
        {
          id: '00000000-0000-4000-8000-000000000002',
          username: 'pt_somchai',
          passwordHash: 'plain:Pt@2026',
          role: 'pt',
          fullName: 'กภ.สมชาย ใจดี (PT)',
          phone: '081-111-2233',
          active: true,
          forceLoggedOutAt: null,
          createdAt: seededNow,
        },
      ],
    } as unknown as CloudRouterState;

    await hydrateStateFromSupabase(makeMockSupabaseWithUsers([]), emptyDbState);
    expect(emptyDbState.users.map((u) => u.username)).toEqual(['admin']);
  });

  it('locks licenseNo against PT self-modification while allowing Admin updates via PUT/PATCH /api/users/:id', async () => {
    expect(isAdminRole('super_admin')).toBe(true);
    expect(isAdminRole('ADMIN')).toBe(true);
    expect(isAdminRole('SUPER_ADMIN')).toBe(true);
    expect(isAdminRole('ผู้ดูแลระบบ')).toBe(true);
    expect(isAdminRole('pt')).toBe(false);
    expect(isAdminRole('patient')).toBe(false);

    const router = createCloudDelegatedRouter({});
    const ptLogin = await callRouter(router, '/api/auth/login', {
      method: 'POST',
      body: { username: 'pt_somchai', password: 'Pt@2026' },
    });
    expect(ptLogin.status).toBe(200);
    const { token: ptToken, user: ptUser } = (await ptLogin.json()) as {
      token: string;
      user: { id: string; licenseNo?: string };
    };
    expect(ptUser.licenseNo).toBe('PT-44012');

    // PT updates specialty & phone while attempting to spoof licenseNo -> specialty/phone save, licenseNo MUST remain PT-44012
    const ptUpdateRes = await callRouter(router, `/api/users/${ptUser.id}`, {
      method: 'PUT',
      token: ptToken,
      body: {
        phone: '0899998888',
        specialty: 'Neurological Stroke Rehabilitation',
        licenseNo: 'PT-SPOOF-999',
      },
    });
    expect(ptUpdateRes.status).toBe(200);
    const ptUpdatedJson = (await ptUpdateRes.json()) as {
      user: { phone: string; specialty?: string; licenseNo?: string };
    };
    expect(ptUpdatedJson.user.phone).toBe('0899998888');
    expect(ptUpdatedJson.user.specialty).toBe('Neurological Stroke Rehabilitation');
    expect(ptUpdatedJson.user.licenseNo).toBe('PT-44012');

    // PATCH by PT also refuses to mutate licenseNo or license_number / license_no
    const ptPatchRes = await callRouter(router, `/api/users/${ptUser.id}`, {
      method: 'PATCH',
      token: ptToken,
      body: { licenseNo: 'PT-PATCH-HACK', license_number: 'PT-SNAKE-HACK', license_no: 'PT-SNAKE-HACK2' },
    });
    expect(ptPatchRes.status).toBe(200);
    expect(
      ((await ptPatchRes.json()) as { user: { licenseNo?: string } }).user.licenseNo,
    ).toBe('PT-44012');

    // Super Admin updates PT licenseNo (and snake_case license_number) -> allowed cleanly
    const adminLogin = await callRouter(router, '/api/auth/login', {
      method: 'POST',
      body: { username: 'admin', password: 'Admin@2026' },
    });
    const { token: adminToken } = (await adminLogin.json()) as { token: string };
    const adminUpdateRes = await callRouter(router, `/api/users/${ptUser.id}`, {
      method: 'PUT',
      token: adminToken,
      body: { license_number: 'PT-88776' },
    });
    expect(adminUpdateRes.status).toBe(200);
    expect(
      ((await adminUpdateRes.json()) as { user: { licenseNo?: string } }).user.licenseNo,
    ).toBe('PT-88776');
  });

  it('serves /api/health and /api/exercises in fast-path and bounds hydrateStaffUsersOnly timeout', async () => {
    const router = createCloudDelegatedRouter({});
    const t0 = Date.now();
    const healthRes = await callRouter(router, '/api/health');
    const healthJson = (await healthRes.json()) as { ok: boolean; status?: string; timestamp?: string };
    expect(healthRes.status).toBe(200);
    expect(healthJson.ok).toBe(true);
    expect(healthJson.status).toBe('healthy');
    expect(typeof healthJson.timestamp).toBe('string');

    const exRes = await callRouter(router, '/api/exercises');
    const exJson = (await exRes.json()) as { exercises: unknown[] };
    expect(exRes.status).toBe(200);
    expect(exJson.exercises).toHaveLength(15);
    expect(Date.now() - t0).toBeLessThan(200);

    // hydrateStaffUsersOnly queries ONLY staff_users and resolves false on timeout without throwing
    let queriedTables: string[] = [];
    const stalledSupabase = {
      from: (table: string) => {
        queriedTables.push(table);
        return {
          select: () => ({
            limit: () => new Promise(() => {}),
          }),
        };
      },
    } as unknown as SupabaseClient;
    const dummyState = { users: [], deleted_user_ids: [] } as unknown as CloudRouterState;
    const ok = await hydrateStaffUsersOnly(stalledSupabase, dummyState, 30);
    expect(ok).toBe(false);
    expect(queriedTables).toEqual(['staff_users']);

    // Staged hydration preserves Stage 1 patients even if Stage 2/3 biometric or log queries fail
    const queryOrder: string[] = [];
    const stagedState = {
      users: [],
      deleted_user_ids: [],
      deleted_patient_ids: [],
      deleted_ban_ids: [],
      deleted_staff_call_ids: [],
      delegation_overrides: {},
      patients: [],
      face_templates: [],
      medical_notes: [],
      patient_courses: [],
      rehab_sessions: [],
      rehab_session_history: [],
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
    } as unknown as CloudRouterState;
    const stagedSupabase = {
      from: (table: string) => {
        queryOrder.push(table);
        const failIfHeavy =
          table === 'patient_biometrics' || table === 'face_templates' || table === 'audit_logs';
        const q: Record<string, unknown> = {
          then: (resolve: (val: unknown) => void) =>
            resolve(
              failIfHeavy
                ? { data: null, error: { message: 'Upstream 522 simulated' } }
                : table === 'patients'
                  ? {
                      data: [
                        {
                          id: 'pat-stage1-1',
                          hn: 'HN-75127',
                          full_name: 'นายพงศกร จิตตเกตุ',
                          age: 62,
                          phone: '0811112233',
                          diagnosis: 'Right Hemiparesis',
                          registration_status: 'active',
                          face_biometric_status: 'VERIFIED',
                          history: {},
                          created_at: new Date().toISOString(),
                        },
                      ],
                      error: null,
                    }
                  : { data: [], error: null },
            ),
        };
        q.select = () => q;
        q.order = () => q;
        q.limit = () => q;
        q.eq = () => q;
        q.maybeSingle = () => Promise.resolve({ data: null, error: null });
        return q;
      },
    } as unknown as SupabaseClient;

    await hydrateStateFromSupabase(stagedSupabase, stagedState);
    expect(stagedState.patients.length).toBe(1);
    expect(stagedState.patients[0].hn).toBe('HN-75127');
    expect(queryOrder.indexOf('patients')).toBeLessThan(queryOrder.indexOf('patient_biometrics'));
    expect(queryOrder.indexOf('patients')).toBeLessThan(queryOrder.indexOf('face_templates'));
  });

  it('never returns 404 or 500 when deleting users or patients (UUID, non-UUID, missing, or DB FK failures)', async () => {
    const router = createCloudDelegatedRouter({});
    const adminLogin = await callRouter(router, '/api/auth/login', {
      method: 'POST',
      body: { username: 'admin', password: 'Admin@2026' },
    });
    const { token: adminToken } = (await adminLogin.json()) as { token: string };

    // Enroll a patient so we have both valid UUID-like/generated ID and HN-75127
    const enrollRes = await callRouter(router, '/api/biometric/enroll', {
      method: 'POST',
      token: adminToken,
      body: {
        hn: 'HN-75127',
        fullName: 'นายพงศกร จิตตเกตุ',
        age: 61,
        diagnosis: 'Right Hemiparesis',
        skipFaceScan: true,
      },
    });
    expect(enrollRes.status).toBe(201);
    const { patient } = (await enrollRes.json()) as { patient: { id: string; hn: string } };

    // 1. Delete existing non-UUID seeded PT (`user-pt-1`), delete by username via /api/staff/:id, and delete non-existent ID
    for (const url of [
      '/api/users/user-pt-1',
      '/api/staff/pt_waranya',
      '/api/physiotherapists/non-existent-pt',
      '/api/users?username=already_gone_user',
    ]) {
      const delRes = await callRouter(router, url, { method: 'DELETE', token: adminToken });
      expect(delRes.status).toBe(200);
      const delJson = (await delRes.json()) as { ok: boolean; deleted: boolean };
      expect(delJson.ok).toBe(true);
      expect(delJson.deleted).toBe(true);
    }

    // 2. Delete existing patient by id, re-delete by non-UUID HN-75127, delete missing patient, and delete biometrics
    for (const url of [
      `/api/patients/${patient.id}`,
      '/api/patients/HN-75127',
      '/api/patients?hn=ghost-patient-404',
      '/api/biometrics/HN-75127',
    ]) {
      const delRes = await callRouter(router, url, { method: 'DELETE', token: adminToken });
      expect(delRes.status).toBe(200);
      const delJson = (await delRes.json()) as { ok: boolean; deleted: boolean };
      expect(delJson.ok).toBe(true);
      expect(delJson.deleted).toBe(true);
    }

    // 3. Direct Supabase purge helpers validate UUID before querying uuid columns and swallow FK 23503 / 500 errors
    const eqCalls: { table: string; col: string; val: unknown }[] = [];
    const failingSupabase = {
      from: (table: string) => {
        const builder: Record<string, unknown> = {
          delete: () => builder,
          update: () => builder,
          eq: (col: string, val: unknown) => {
            eqCalls.push({ table, col, val });
            return Promise.reject(new Error('23503 foreign key violation / 500 simulated'));
          },
        };
        return builder;
      },
    } as unknown as SupabaseClient;

    await expect(
      purgeUserInSupabaseBestEffort(failingSupabase, {
        rawId: 'user-pt-1',
        matchedUserId: 'user-pt-1',
        username: 'pt_somchai',
      }),
    ).resolves.toBeUndefined();
    // Must NEVER pass non-UUID `user-pt-1` into `.eq('id', ...)` or `.eq('user_id', ...)`
    expect(eqCalls.some((c) => (c.col === 'id' || c.col === 'user_id') && c.val === 'user-pt-1')).toBe(false);
    expect(eqCalls.some((c) => c.col === 'username' && c.val === 'pt_somchai')).toBe(true);

    const uuidPid = '11111111-2222-4333-8444-555555555555';
    await expect(
      purgePatientInSupabaseBestEffort(failingSupabase, {
        rawId: uuidPid,
        patientId: uuidPid,
        hn: 'HN-75127',
      }),
    ).resolves.toBeUndefined();
    expect(eqCalls.some((c) => c.table === 'face_templates' && c.col === 'patient_id' && c.val === uuidPid)).toBe(true);
    expect(eqCalls.some((c) => c.table === 'patients' && c.col === 'id' && c.val === uuidPid)).toBe(true);

    // 4. Client deleteUser / deletePatientCascading / deleteStaff NEVER throw on 404, 500, 522, or network rejection
    const origFetch = globalThis.fetch;
    try {
      for (const status of [404, 500, 522]) {
        globalThis.fetch = (async () =>
          new Response(JSON.stringify({ error: `HTTP_${status}` }), { status })) as typeof fetch;
        const client = new DrTechCaresApiClient({});
        await expect(client.deleteUser('user-pt-1', 'pt_somchai')).resolves.toBeUndefined();
        await expect(client.deleteStaff('pt_waranya')).resolves.toBeUndefined();
        await expect(client.deletePatientCascading(uuidPid, 'HN-75127')).resolves.toBeUndefined();
      }
      globalThis.fetch = (async () => {
        throw new TypeError('Network failure');
      }) as typeof fetch;
      const offlineClient = new DrTechCaresApiClient({});
      await expect(offlineClient.deleteUser('user-pt-1')).resolves.toBeUndefined();
      await expect(offlineClient.deletePatientCascading('HN-75127')).resolves.toBeUndefined();
    } finally {
      globalThis.fetch = origFetch;
    }
  });

  it('prevents orphan zombie resurrection in hydrateStateFromSupabase and rejects 0.68 strangers / 1-outlier probes without leaking names', async () => {
    const mkUnit1024 = (seed: number): number[] => {
      const vec = Array.from({ length: 1024 }, (_, idx) => Math.sin((idx + 1) * (seed * 0.17 + 0.03)));
      return l2Normalize(vec);
    };
    const liveMaster = mkUnit1024(11);
    const deadVec = mkUnit1024(22);
    const ghostVec = mkUnit1024(33);
    const outlierLeft = mkUnit1024(99);
    const liveCenter = l2Normalize(liveMaster.map((v, i) => 0.96 * v + 0.28 * outlierLeft[i] * (i % 2 === 0 ? 1 : -1)));
    const liveRight = l2Normalize(liveMaster.map((v, i) => 0.95 * v + 0.31 * mkUnit1024(44)[i]));

    // 1. Orphaned / soft-deleted records in patient_biometrics & face_templates MUST NOT resurrect into state.patients or state.face_templates
    const hydrateState = {
      users: [],
      deleted_user_ids: [],
      deleted_patient_ids: [],
      deleted_ban_ids: [],
      deleted_staff_call_ids: [],
      delegation_overrides: {},
      patients: [],
      face_templates: [],
      medical_notes: [],
      patient_courses: [],
      rehab_sessions: [],
      rehab_session_history: [],
      treatment_reports: [],
      exercises: [],
      modpacks: [],
      staff_calls: [],
      audit_logs: [],
      banned_targets: [],
      active_sessions: [],
      system_settings: { aiFaceScanEnabled: true, aiFaceThreshold: 0.925, aiPoseDetectionEnabled: false, kioskAutoResetSec: 6 },
      kiosks: [],
    } as unknown as CloudRouterState;

    const zombieSupabase = {
      from: (table: string) => {
        const q: Record<string, unknown> = {
          then: (resolve: (val: unknown) => void) => {
            if (table === 'patients') {
              resolve({
                data: [
                  {
                    id: 'pat-live-1',
                    hn: 'HN-LIVE-01',
                    full_name: 'คุณวิไล ปัจจุบัน',
                    age: 58,
                    phone: '0810000001',
                    diagnosis: 'Stroke',
                    registration_status: 'active',
                    face_biometric_status: 'VERIFIED',
                    history: {},
                    created_at: new Date().toISOString(),
                  },
                  {
                    id: 'pat-dead-1',
                    hn: 'HN-DEAD-01',
                    full_name: 'สมชาย ที่ถูกลบแล้ว',
                    age: 64,
                    phone: '0810000002',
                    diagnosis: 'Stroke',
                    registration_status: 'deleted',
                    deleted_at: new Date().toISOString(),
                    face_biometric_status: 'NONE',
                    history: {},
                    created_at: new Date().toISOString(),
                  },
                ],
                error: null,
              });
              return;
            }
            if (table === 'patient_biometrics') {
              resolve({
                data: [
                  {
                    patient_id: 'pat-live-1',
                    patient_hn: 'HN-LIVE-01',
                    full_name: 'คุณวิไล ปัจจุบัน',
                    biometric_engine: 'human-faceres-1024d',
                    dimension: 1024,
                    master_embedding: liveMaster,
                    angle_embeddings: { center: liveCenter, left: outlierLeft, right: liveRight },
                    exemplar_embeddings: [liveMaster, liveCenter, liveRight],
                  },
                  {
                    patient_id: 'pat-dead-1',
                    patient_hn: 'HN-DEAD-01',
                    full_name: 'สมชาย ที่ถูกลบแล้ว',
                    biometric_engine: 'human-faceres-1024d',
                    dimension: 1024,
                    master_embedding: deadVec,
                    angle_embeddings: { center: deadVec },
                  },
                  {
                    patient_id: 'pat-ghost-999',
                    patient_hn: 'HN-GHOST-999',
                    full_name: 'ผีที่ไม่มีใน patients',
                    biometric_engine: 'human-faceres-1024d',
                    dimension: 1024,
                    master_embedding: ghostVec,
                    angle_embeddings: { center: ghostVec },
                  },
                ],
                error: null,
              });
              return;
            }
            if (table === 'face_templates') {
              resolve({
                data: [
                  {
                    patient_id: 'pat-ghost-999',
                    patient_hn: 'HN-GHOST-999',
                    full_name: 'ผีที่ไม่มีใน patients',
                    embedding_json: {
                      masterEmbedding: ghostVec,
                      angleEmbeddings: { center: ghostVec },
                      biometricEngine: 'human-faceres-1024d',
                    },
                  },
                ],
                error: null,
              });
              return;
            }
            resolve({ data: [], error: null });
          },
        };
        q.select = () => q;
        q.order = () => q;
        q.limit = () => q;
        q.eq = () => q;
        q.maybeSingle = () => Promise.resolve({ data: null, error: null });
        return q;
      },
    } as unknown as SupabaseClient;

    await hydrateStateFromSupabase(zombieSupabase, hydrateState);
    expect(hydrateState.patients.map((p) => p.id)).toEqual(['pat-live-1']);
    expect(Array.from(new Set(hydrateState.face_templates.map((t) => t.patientId)))).toEqual(['pat-live-1']);
    expect(hydrateState.deleted_patient_ids).toContain('pat-dead-1');

    // 2. Open-set /api/biometric/verify rejects 0.68 stranger and 1-outlier probe without leaking candidate names
    const router = createCloudDelegatedRouter({});
    const adminLogin = await callRouter(router, '/api/auth/login', {
      method: 'POST',
      body: { username: 'admin', password: 'Admin@2026' },
    });
    const { token } = (await adminLogin.json()) as { token: string };
    const eRes = await callRouter(router, '/api/biometric/enroll', {
      method: 'POST',
      token,
      body: {
        hn: 'HN-GATE-1024',
        fullName: 'นางสาวมัลลิกา เกตแน่น',
        age: 57,
        phone: '0897771024',
        diagnosis: 'Stroke',
        biometricEngine: 'human-faceres-1024d',
        masterEmbedding: liveMaster,
        angleEmbeddings: {
          center: liveCenter,
          left: outlierLeft,
          right: liveRight,
          up: liveMaster,
          down: liveCenter,
        },
        exemplarEmbeddings: [liveMaster, liveCenter, liveRight],
        framesCaptured: 120,
        qualityScore: 0.98,
      },
    });
    expect(eRes.status).toBe(201);

    // 2a. Stranger probe at ~0.50 similarity to liveMaster -> 401 + NO topCandidate name leaked!
    const rawO = mkUnit1024(77);
    const dotMO = liveMaster.reduce((acc, v, i) => acc + v * rawO[i], 0);
    const orthoVec = l2Normalize(rawO.map((v, i) => v - dotMO * liveMaster[i]));
    const probe050 = l2Normalize(
      liveMaster.map((v, i) => 0.50 * v + Math.sqrt(1 - 0.50 * 0.50) * orthoVec[i]),
    );
    const rej050 = await callRouter(router, '/api/biometric/verify', {
      method: 'POST',
      body: {
        probeVector: probe050,
        engine: 'human-faceres-1024d',
        livenessEvidence: { biologicalVerified: true, blinkCount: 1, spoofRisk: 0.02, nonRigidScore: 0.16, engine: 'human-faceres-1024d' },
      },
    });
    expect(rej050.status).toBe(401);
    const rej050Json = (await rej050.json()) as {
      matched: boolean;
      confidencePercent: number;
      topCandidate?: unknown;
      patient?: unknown;
    };
    expect(rej050Json.matched).toBe(false);
    expect(rej050Json.confidencePercent).toBeLessThan(60);
    expect(rej050Json.topCandidate).toBeNull();
    expect(rej050Json.patient).toBeUndefined();

    // 2b. Sub-threshold stranger probe at ~0.66 is STRICTLY rejected (401, matched: false)
    const probe066 = l2Normalize(
      liveMaster.map((v, i) => 0.66 * v + Math.sqrt(1 - 0.66 * 0.66) * orthoVec[i]),
    );
    const rej066 = await callRouter(router, '/api/biometric/verify', {
      method: 'POST',
      body: {
        probeVector: probe066,
        engine: 'human-faceres-1024d',
        livenessEvidence: { biologicalVerified: true, blinkCount: 1, spoofRisk: 0.02, nonRigidScore: 0.16, engine: 'human-faceres-1024d' },
      },
    });
    expect(rej066.status).toBe(401);
    const rej066Json = (await rej066.json()) as { matched: boolean; patientId: string | null };
    expect(rej066Json.matched).toBe(false);
    expect(rej066Json.patientId).toBeNull();

    // 2c. Genuine calibrated probe at ~0.92 (>= 0.84 raw, >= 0.80 centered, L2 <= 0.56) passes cleanly!
    const probe092 = l2Normalize(
      liveMaster.map((v, i) => 0.92 * v + Math.sqrt(1 - 0.92 * 0.92) * orthoVec[i]),
    );
    const accept092 = await callRouter(router, '/api/biometric/verify', {
      method: 'POST',
      body: {
        probeVector: probe092,
        engine: 'human-faceres-1024d',
        livenessEvidence: { biologicalVerified: true, blinkCount: 1, spoofRisk: 0.02, nonRigidScore: 0.16, engine: 'human-faceres-1024d' },
      },
    });
    expect(accept092.status).toBe(200);
    const accept092Json = (await accept092.json()) as { matched: boolean; patient?: { hn: string } };
    expect(accept092Json.matched).toBe(true);
    expect(accept092Json.patient?.hn).toBe('HN-GATE-1024');
  });

  it('sanitizes UUID/exemplar payloads and never returns HTTP 500 during enrollment/course saves even when Supabase fails', async () => {
    expect(toSafeUuidOrNull('user-pt-1')).toBe('00000000-0000-4000-8000-000000000002');
    expect(toSafeUuidOrNull('non-uuid-pt')).toBeNull();
    expect(toSafeUuidOrNull('11111111-1111-4111-8111-111111111111')).toBe(
      '11111111-1111-4111-8111-111111111111',
    );

    const mk1024 = (seed: number) =>
      l2Normalize(Array.from({ length: 1024 }, (_, idx) => Math.sin((idx + 1) * seed)));
    const exemplars18 = Array.from({ length: 18 }, (_, i) => mk1024(i + 1));
    expect(normalizeCompactExemplars(exemplars18, 1024, 9)).toHaveLength(9);

    const capturedRestCalls: { url: string; method: string; body: unknown }[] = [];
    const origFetch = globalThis.fetch;
    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
      const method = init?.method ?? (input instanceof Request ? input.method : 'GET');
      let parsedBody: unknown = null;
      const rawBodyText =
        typeof init?.body === 'string'
          ? init.body
          : input instanceof Request && method !== 'GET'
            ? await input.clone().text().catch(() => '')
            : '';
      if (rawBodyText) {
        try {
          parsedBody = JSON.parse(rawBodyText);
        } catch {
          parsedBody = rawBodyText;
        }
      }
      capturedRestCalls.push({ url, method, body: parsedBody });
      if (url.includes('/rest/v1/staff_users') && method === 'GET') {
        return new Response('[]', { status: 200, headers: { 'Content-Type': 'application/json' } });
      }
      // Simulate failing Supabase Postgres RPC / REST writes (400 + 22P02/23503)
      return new Response(
        JSON.stringify({
          code: '22P02',
          message: 'simulated postgres error on write',
        }),
        { status: 400, headers: { 'Content-Type': 'application/json' } },
      );
    }) as typeof fetch;

    try {
      const fakeJwt =
        'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyb2xlIjoic2VydmljZV9yb2xlIn0.signaturevaltestsafe1234567890';
      const resilientRouter = createCloudDelegatedRouter({
        supabaseUrl: 'https://vyjuwjhbdhknjdzyqctz.supabase.co',
        supabaseServiceKey: fakeJwt,
      });

      const adminLogin = await callRouter(resilientRouter, '/api/auth/login', {
        method: 'POST',
        body: { username: 'admin', password: 'Admin@2026' },
      });
      expect(adminLogin.status).toBe(200);
      const { token } = (await adminLogin.json()) as { token: string };

      const master1024 = mk1024(3.14);
      const enrollRes = await callRouter(resilientRouter, '/api/biometric/enroll', {
        method: 'POST',
        token,
        body: {
          hn: 'HN-43143',
          fullName: 'Sdas Test',
          age: 64,
          phone: '0812345678',
          diagnosis: 'Stroke Shoulder',
          assignedPtId: 'arbitrary-non-uuid-pt',
          nextAppointmentDate: '2026-11-20',
          rehabSchedules: [
            {
              id: 'sched-1',
              patientId: 'pending',
              exerciseId: 'STROKE_SH_FLEX_BOTH',
              scheduledDate: '2026-11-20',
              sets: 3,
              reps: 10,
              status: 'ACTIVE',
            },
          ],
          biometricEngine: 'human-faceres-1024d',
          masterEmbedding: master1024,
          angleEmbeddings: {
            center: master1024,
            left: mk1024(4.1),
            right: mk1024(5.2),
            up: mk1024(6.3),
            down: mk1024(7.4),
          },
          exemplarEmbeddings: exemplars18,
          rawFrames: Array.from({ length: 120 }, (_, idx) => ({
            pose: 'front' as const,
            vector: mk1024(idx + 1),
            qualityScore: 0.95,
          })),
          framesCaptured: 120,
          qualityScore: 0.96,
        },
      });
      expect(enrollRes.status).toBe(201);
      const enrollJson = (await enrollRes.json()) as {
        ok: boolean;
        patient: { id: string; hn: string };
      };
      expect(enrollJson.ok).toBe(true);
      expect(enrollJson.patient.hn).toBe('HN-43143');

      // Inspect outbound REST payloads sent to Supabase to prove sanitization & compact storage
      const patientWrite = capturedRestCalls.find(
        (c) => c.url.includes('/rest/v1/patients') && (c.method === 'POST' || c.method === 'PATCH'),
      );
      expect(patientWrite).toBeDefined();
      const patPayload = Array.isArray(patientWrite?.body)
        ? (patientWrite?.body[0] as Record<string, unknown>)
        : (patientWrite?.body as Record<string, unknown>);
      expect(patPayload.assigned_pt_id ?? null).toBeNull();

      const faceTplWrite = capturedRestCalls.find(
        (c) => c.url.includes('/rest/v1/face_templates') && c.method === 'POST',
      );
      if (faceTplWrite && Array.isArray(faceTplWrite.body)) {
        expect(faceTplWrite.body.length).toBeLessThanOrEqual(6);
      }

      // PUT /api/courses/:patientId and POST /api/patients also never return 500
      const courseRes = await callRouter(resilientRouter, `/api/courses/${enrollJson.patient.id}`, {
        method: 'PUT',
        token,
        body: {
          title: 'Stroke Shoulder Plan',
          cognitiveModpackId: 'missing-modpack-id',
          weeklySchedule: [],
        },
      });
      expect(courseRes.status).toBe(200);
      const createPatRes = await callRouter(resilientRouter, '/api/patients', {
        method: 'POST',
        token,
        body: {
          hn: 'HN-POST-500',
          fullName: 'Fallback Create',
          age: 58,
          assignedPtId: 'user-pt-1',
        },
      });
      expect(createPatRes.status).toBe(201);
    } finally {
      globalThis.fetch = origFetch;
    }

    // Frontend DrTechCaresApiClient falls back cleanly on HTTP 500 without throwing "Request failed (500)"
    const prevStorage = globalThis.localStorage;
    const memStore = new Map<string, string>();
    Object.defineProperty(globalThis, 'localStorage', {
      configurable: true,
      value: {
        getItem: (k: string) => memStore.get(k) ?? null,
        setItem: (k: string, v: string) => {
          memStore.set(k, String(v));
        },
        removeItem: (k: string) => {
          memStore.delete(k);
        },
      },
    });
    const prevFetch = globalThis.fetch;
    globalThis.fetch = (async () =>
      new Response(JSON.stringify({ message: 'Upstream 500' }), {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      })) as typeof fetch;

    try {
      const api = new DrTechCaresApiClient({ apiBaseUrl: 'https://drtechcares.workers.dev' });
      const enrolledFallback = await api.enrollPatient10Shot({
        hn: 'HN-OFFLINE-500',
        fullName: 'Offline Fallback Patient',
        age: 61,
        diagnosis: 'Stroke',
        assignedPtId: 'user-pt-1',
        masterEmbedding: mk1024(9.1),
      });
      expect(enrolledFallback.hn).toBe('HN-OFFLINE-500');
      expect(memStore.get('drtech_patients')).toContain('HN-OFFLINE-500');

      const courseFallback = await api.upsertPatientCourse({
        patientId: enrolledFallback.id,
        title: 'Offline Course',
        cognitiveModpackId: 'stroke_adl_basic',
        weeklySchedule: [],
      });
      expect(courseFallback.patientId).toBe(enrolledFallback.id);
      expect(memStore.get('drtech_patient_courses')).toContain(enrolledFallback.id);
    } finally {
      globalThis.fetch = prevFetch;
      Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: prevStorage });
    }
  });

  it('enrolls patient dddd (HN-37751) during Supabase HTTP 544 outage, trips 20s circuit breaker, strips oversized Base64, recovers immediately on /api/health, and logs dddd in on Kiosk scan', async () => {
    const oversizedBase64 = `data:image/jpeg;base64,${'A'.repeat(12000)}`;
    const sanitized = stripOversizedBase64ForDb({
      face_image_url: oversizedBase64,
      avatar_url: oversizedBase64,
      history: {
        faceImageUrl: oversizedBase64,
        faceImageUrls: [oversizedBase64, 'https://cdn.example.com/ok.jpg'],
        angleImageUrls: { center: oversizedBase64, left: 'https://cdn.example.com/left.jpg' },
      },
    });
    expect(sanitized.face_image_url).toBeNull();
    expect(sanitized.avatar_url).toBeNull();
    expect(sanitized.history.faceImageUrl).toBeNull();
    expect(sanitized.history.faceImageUrls).toEqual(['https://cdn.example.com/ok.jpg']);
    expect(sanitized.history.angleImageUrls.center).toBeNull();
    expect(sanitized.history.angleImageUrls.left).toBe('https://cdn.example.com/left.jpg');

    const mk1024 = (seed: number) =>
      l2Normalize(Array.from({ length: 1024 }, (_, idx) => Math.sin((idx + 1) * seed)));
    let outboundSupabaseCalls = 0;
    const origFetch = globalThis.fetch;
    globalThis.fetch = (async () => {
      outboundSupabaseCalls += 1;
      return new Response(
        JSON.stringify({
          code: '544',
          message: 'DbConnectionTimeout: Connection terminated due to connection timeout',
        }),
        { status: 544, headers: { 'Content-Type': 'application/json' } },
      );
    }) as typeof fetch;

    try {
      const fakeJwt =
        'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyb2xlIjoic2VydmljZV9yb2xlIn0.signaturevaltestsafe1234567890';
      const outageRouter = createCloudDelegatedRouter({
        supabaseUrl: 'https://vyjuwjhbdhknjdzyqctz.supabase.co',
        supabaseServiceKey: fakeJwt,
      });

      const loginRes = await callRouter(outageRouter, '/api/auth/login', {
        method: 'POST',
        body: { username: 'admin', password: 'Admin@2026' },
      });
      expect(loginRes.status).toBe(200);
      const { token } = (await loginRes.json()) as { token: string };

      // Realistic 5-angle captures of the SAME person (turning left/right/up/down shifts embedding ~10% vs center)
      const masterVec = mk1024(1.37);
      const mkSamePersonAngle = (seed: number) => {
        const rawA = mk1024(seed);
        const dot = masterVec.reduce((acc, v, idx) => acc + v * rawA[idx], 0);
        const ortho = l2Normalize(rawA.map((v, idx) => v - dot * masterVec[idx]));
        return l2Normalize(masterVec.map((v, idx) => 0.9 * v + Math.sqrt(1 - 0.9 * 0.9) * ortho[idx]));
      };
      const enrollRes = await callRouter(outageRouter, '/api/biometric/enroll', {
        method: 'POST',
        token,
        body: {
          hn: 'HN-37751',
          fullName: 'dddd',
          age: 60,
          phone: '0813775100',
          diagnosis: 'Stroke Rehab',
          biometricEngine: 'human-faceres-1024d',
          faceImageUrl: oversizedBase64,
          angleImageUrls: {
            center: oversizedBase64,
            left: oversizedBase64,
            right: oversizedBase64,
            up: oversizedBase64,
            down: oversizedBase64,
          },
          masterEmbedding: masterVec,
          angleEmbeddings: {
            center: masterVec,
            left: mkSamePersonAngle(2.11),
            right: mkSamePersonAngle(3.23),
            up: mkSamePersonAngle(4.37),
            down: mkSamePersonAngle(5.41),
          },
          exemplarEmbeddings: [masterVec],
          framesCaptured: 120,
          qualityScore: 0.97,
        },
      });
      expect(enrollRes.status).toBe(201);
      const enrollJson = (await enrollRes.json()) as {
        ok: boolean;
        patient: { id: string; hn: string; fullName: string };
      };
      expect(enrollJson.ok).toBe(true);
      expect(enrollJson.patient.hn).toBe('HN-37751');
      expect(enrollJson.patient.fullName).toBe('dddd');

      // Verify /api/health resets downUntil=0, tests SELECT id FROM patients LIMIT 1, and stays in 20s circuit breaker while HTTP 544 persists
      const healthRes = await callRouter(outageRouter, '/api/health');
      const healthJson = (await healthRes.json()) as {
        ok: boolean;
        supabaseConnected?: boolean;
        supabaseCircuitOpen?: boolean;
        supabaseCircuitRemainingSec?: number;
      };
      expect(healthJson.ok).toBe(true);
      expect(healthJson.supabaseConnected).toBe(false);
      expect(healthJson.supabaseCircuitOpen).toBe(true);
      expect((healthJson.supabaseCircuitRemainingSec ?? 0)).toBeGreaterThan(0);
      expect((healthJson.supabaseCircuitRemainingSec ?? 0)).toBeLessThanOrEqual(20);

      const callsAfterTrip = outboundSupabaseCalls;

      // Subsequent Kiosk index & patient list calls make ZERO extra outbound Supabase calls (0ms circuit fast-path)
      const idxRes = await callRouter(outageRouter, '/api/biometrics/index');
      expect(idxRes.status).toBe(200);
      const idxJson = (await idxRes.json()) as {
        index: Array<{ hn: string; fullName: string }>;
        biometrics: Array<{ hn: string; fullName: string }>;
      };
      expect(idxJson.index.some((e) => e.hn === 'HN-37751' && e.fullName === 'dddd')).toBe(true);
      expect(idxJson.biometrics.some((e) => e.hn === 'HN-37751')).toBe(true);

      // Unregistered stranger scan at ~0.65 cosine against single-patient dddd DB MUST fail 401 matched: false (zero single-patient loophole!)
      const rawO = mk1024(9.87);
      const dotMO = masterVec.reduce((acc, v, i) => acc + v * rawO[i], 0);
      const orthoVec = l2Normalize(rawO.map((v, i) => v - dotMO * masterVec[i]));
      const probe065Stranger = l2Normalize(
        masterVec.map((v, i) => 0.65 * v + Math.sqrt(1 - 0.65 * 0.65) * orthoVec[i]),
      );
      const strangerVerifyRes = await callRouter(outageRouter, '/api/biometric/verify', {
        method: 'POST',
        body: {
          probeVector: probe065Stranger,
          engine: 'human-faceres-1024d',
          livenessEvidence: {
            biologicalVerified: true,
            blinkCount: 1,
            spoofRisk: 0.02,
            nonRigidScore: 0.14,
            engine: 'human-faceres-1024d',
          },
        },
      });
      expect(strangerVerifyRes.status).toBe(401);
      const strangerJson = (await strangerVerifyRes.json()) as {
        matched: boolean;
        patientId: string | null;
      };
      expect(strangerJson.matched).toBe(false);
      expect(strangerJson.patientId).toBeNull();

      // Genuine Kiosk scan at ~0.92 cosine against dddd MUST succeed with matched: true!
      const probe092Genuine = l2Normalize(
        masterVec.map((v, i) => 0.92 * v + Math.sqrt(1 - 0.92 * 0.92) * orthoVec[i]),
      );
      const verifyRes = await callRouter(outageRouter, '/api/biometric/verify', {
        method: 'POST',
        body: {
          probeVector: probe092Genuine,
          engine: 'human-faceres-1024d',
          livenessEvidence: {
            biologicalVerified: true,
            blinkCount: 1,
            spoofRisk: 0.02,
            nonRigidScore: 0.14,
            engine: 'human-faceres-1024d',
          },
        },
      });
      expect(verifyRes.status).toBe(200);
      const verifyJson = (await verifyRes.json()) as {
        matched: boolean;
        patient?: { hn: string; fullName: string };
      };
      expect(verifyJson.matched).toBe(true);
      expect(verifyJson.patient?.hn).toBe('HN-37751');
      expect(verifyJson.patient?.fullName).toBe('dddd');

      // Also enroll a 2nd patient via POST /api/biometrics and confirm multi-patient matching with >= 0.08 margin works cleanly while stranger stays rejected!
      const bio2Res = await callRouter(outageRouter, '/api/biometrics', {
        method: 'POST',
        token,
        body: {
          hn: 'HN-99881',
          fullName: 'Another Patient',
          master_embedding: mk1024(19.5),
        },
      });
      expect(bio2Res.status).toBe(201);

      const multiStrangerRes = await callRouter(outageRouter, '/api/biometric/verify', {
        method: 'POST',
        body: {
          probeVector: probe065Stranger,
          engine: 'human-faceres-1024d',
          livenessEvidence: {
            biologicalVerified: true,
            blinkCount: 1,
            spoofRisk: 0.02,
            nonRigidScore: 0.14,
            engine: 'human-faceres-1024d',
          },
        },
      });
      expect(multiStrangerRes.status).toBe(401);

      const multiVerifyRes = await callRouter(outageRouter, '/api/biometric/verify', {
        method: 'POST',
        body: {
          probeVector: probe092Genuine,
          engine: 'human-faceres-1024d',
          livenessEvidence: {
            biologicalVerified: true,
            blinkCount: 1,
            spoofRisk: 0.02,
            nonRigidScore: 0.14,
            engine: 'human-faceres-1024d',
          },
        },
      });
      expect(multiVerifyRes.status).toBe(200);
      const multiJson = (await multiVerifyRes.json()) as {
        matched: boolean;
        patient?: { hn: string; fullName: string };
      };
      expect(multiJson.matched).toBe(true);
      expect(multiJson.patient?.fullName).toBe('dddd');

      // Confirm circuit breaker prevented any additional HTTP requests to Supabase while open!
      expect(outboundSupabaseCalls).toBe(callsAfterTrip);

      // Once Supabase recovers (HTTP 200 on SELECT id FROM patients LIMIT 1), /api/health resets supabaseDownUntil = 0 and immediately reports supabaseConnected: true
      globalThis.fetch = (async () =>
        new Response(JSON.stringify([]), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        })) as typeof fetch;

      const recoveredHealthRes = await callRouter(outageRouter, '/api/health');
      expect(recoveredHealthRes.status).toBe(200);
      const recoveredHealth = (await recoveredHealthRes.json()) as {
        ok: boolean;
        supabaseConnected: boolean;
        supabaseCircuitOpen: boolean;
        supabaseCircuitRemainingSec: number;
      };
      expect(recoveredHealth.ok).toBe(true);
      expect(recoveredHealth.supabaseConnected).toBe(true);
      expect(recoveredHealth.supabaseCircuitOpen).toBe(false);
      expect(recoveredHealth.supabaseCircuitRemainingSec).toBe(0);

      // Re-trip on simulated 503, then verify ?refresh=1 on regular API requests clears the open circuit immediately when Supabase returns 200
      globalThis.fetch = (async () =>
        new Response(JSON.stringify({ message: 'restarting' }), {
          status: 503,
          headers: { 'Content-Type': 'application/json' },
        })) as typeof fetch;
      const reTripHealth = (await (await callRouter(outageRouter, '/api/health')).json()) as {
        supabaseCircuitOpen: boolean;
      };
      expect(reTripHealth.supabaseCircuitOpen).toBe(true);

      globalThis.fetch = (async () =>
        new Response(JSON.stringify([]), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        })) as typeof fetch;
      const refreshPatientsRes = await callRouter(outageRouter, '/api/patients?refresh=1', { token });
      expect(refreshPatientsRes.status).toBe(200);
      const afterRefreshHealth = (await (await callRouter(outageRouter, '/api/health')).json()) as {
        supabaseConnected: boolean;
        supabaseCircuitOpen: boolean;
      };
      expect(afterRefreshHealth.supabaseConnected).toBe(true);
      expect(afterRefreshHealth.supabaseCircuitOpen).toBe(false);
    } finally {
      globalThis.fetch = origFetch;
      resetSupabaseCircuitForTests();
    }
  });

  it('uses safe base schema columns, caches 42703 missing columns permanently in unsupportedColumnsByTable, and deduplicates 60s hydration', async () => {
    resetUnsupportedColumnsForTests();
    const patientsSelectHistory: string[] = [];
    const seenColsByTable: Record<string, string[]> = {};
    let patientsError42703Count = 0;

    const createBaseSchemaMockSupabase = () =>
      ({
        from: (table: string) => ({
          select: (cols: string) => {
            seenColsByTable[table] ??= [];
            seenColsByTable[table].push(cols);
            if (table === 'patients') {
              patientsSelectHistory.push(cols);
            }
            const makeResult = () => {
              if (table === 'patients' && cols.split(',').includes('delegation_note')) {
                patientsError42703Count += 1;
                return {
                  data: null,
                  error: { code: '42703', message: 'column patients.delegation_note does not exist' },
                };
              }
              return { data: [], error: null };
            };
            const chain = {
              eq: () => chain,
              order: () => chain,
              limit: async () => makeResult(),
              maybeSingle: async () => ({ data: null, error: null }),
              then: (resolve: (val: unknown) => unknown) => Promise.resolve(makeResult()).then(resolve),
            };
            return chain;
          },
        }),
      }) as unknown as SupabaseClient;

    const mockSupabase = createBaseSchemaMockSupabase();
    const state = {
      users: [],
      patients: [],
      face_templates: [],
      medical_notes: [],
      patient_courses: [],
      rehab_sessions: [],
      rehab_session_history: [],
      treatment_reports: [],
      exercises: [],
      modpacks: [],
      staff_calls: [],
      audit_logs: [],
      banned_targets: [],
      active_sessions: [],
      system_settings: {
        aiFaceScanEnabled: true,
        aiFaceThreshold: 0.84,
        aiPoseDetectionEnabled: false,
        kioskAutoResetSec: 6,
      },
      kiosks: [],
    } as unknown as CloudRouterState;

    try {
      // 1. Concurrent initial calls deduplicate via hydrationInFlight and never spam unmigrated 42703 cols
      await Promise.all([
        hydrateStateFromSupabase(mockSupabase, state),
        hydrateStateFromSupabase(mockSupabase, state),
        hydrateStateFromSupabase(mockSupabase, state),
      ]);

      // Base schema column lists MUST NEVER request the 9 missing columns
      for (const unmigrated of [
        'gender',
        'stroke_type',
        'affected_side',
        'motor_assessment_score',
        'adl_dressing_score',
      ]) {
        expect(seenColsByTable.patients?.some((c) => c.split(',').includes(unmigrated))).toBe(false);
      }
      expect(seenColsByTable.exercises?.some((c) => c.split(',').includes('name_th'))).toBe(false);
      expect(seenColsByTable.modpacks?.some((c) => c.split(',').includes('category'))).toBe(false);
      expect(seenColsByTable.kiosks?.some((c) => c.split(',').includes('pairing_token'))).toBe(false);
      expect(seenColsByTable.audit_logs?.some((c) => c.split(',').includes('action'))).toBe(false);

      // Observed missing column is permanently cached after 1 fallback retry (never 5-retry loop)
      expect(patientsError42703Count).toBe(1);
      expect(unsupportedColumnsByTable.get('patients')?.has('delegation_note')).toBe(true);
      expect(patientsSelectHistory).toHaveLength(2);

      // 2. Subsequent call within HYDRATE_TTL_MS (60_000ms) skips Postgres completely
      expect(HYDRATE_TTL_MS).toBe(60_000);
      await hydrateStateFromSupabase(mockSupabase, state);
      expect(patientsSelectHistory).toHaveLength(2);

      // 3. Forced refresh strips cached unsupported column BEFORE querying -> 0 new 42703 errors!
      await hydrateStateFromSupabase(mockSupabase, state, { force: true });
      expect(patientsError42703Count).toBe(1);
      expect(patientsSelectHistory).toHaveLength(3);
      expect(patientsSelectHistory[2].split(',').includes('delegation_note')).toBe(false);
    } finally {
      resetHydrationCacheForTests(state);
      resetUnsupportedColumnsForTests();
    }
  });
});

