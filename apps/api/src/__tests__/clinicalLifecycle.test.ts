import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import type { AddressInfo } from 'node:net';
import type { Server } from 'node:http';
import {
  CANONICAL_HUMAN_FACE_BASELINE_512,
  FACE_EMBEDDING_DIM,
  l2Normalize,
} from '@drtechcares/smart-biometric-core';
import { createDrTechCaresApiApp } from '../app';
import { ClinicalDatabase } from '../db';

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

describe('DRTECHCARES Production API, Auth, RBAC & Biometric Lifecycle', () => {
  let server: Server;
  let baseUrl = '';
  let adminToken = '';
  let pt1Token = '';
  let pt2Token = '';
  let enrolledPatientId = '';
  const ownerCentroid = makeHybridEmbedding(2.15);

  beforeAll(async () => {
    const db = new ClinicalDatabase({ persistToDisk: false });
    const app = createDrTechCaresApiApp(db);
    await new Promise<void>((resolve) => {
      server = app.listen(0, () => {
        const addr = server.address() as AddressInfo;
        baseUrl = `http://127.0.0.1:${addr.port}`;
        resolve();
      });
    });
  });

  afterAll(async () => {
    await new Promise<void>((resolve, reject) => {
      server.close((err) => (err ? reject(err) : resolve()));
    });
  });

  it('authenticates seeded Super Admin and PT accounts and rejects bad credentials', async () => {
    const badRes = await fetch(`${baseUrl}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: 'admin', password: 'WrongPassword' }),
    });
    expect(badRes.status).toBe(401);

    const adminRes = await fetch(`${baseUrl}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: 'admin', password: 'Admin@2026' }),
    });
    expect(adminRes.status).toBe(200);
    const adminBody = (await adminRes.json()) as { token: string; user: { role: string } };
    expect(adminBody.user.role).toBe('super_admin');
    adminToken = adminBody.token;

    const pt1Res = await fetch(`${baseUrl}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: 'pt_somchai', password: 'Pt@2026' }),
    });
    expect(pt1Res.status).toBe(200);
    pt1Token = ((await pt1Res.json()) as { token: string }).token;

    const pt2Res = await fetch(`${baseUrl}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: 'pt_waranya', password: 'Pt@2026' }),
    });
    expect(pt2Res.status).toBe(200);
    pt2Token = ((await pt2Res.json()) as { token: string }).token;
  });

  it('enforces strict RBAC on user creation and audit logs (PT gets 403)', async () => {
    const createPtAsPt = await fetch(`${baseUrl}/api/users`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${pt1Token}`,
      },
      body: JSON.stringify({
        username: 'rogue_pt',
        password: 'Password123',
        role: 'pt',
        fullName: 'Rogue PT',
      }),
    });
    expect(createPtAsPt.status).toBe(403);

    const createAdminAsPt = await fetch(`${baseUrl}/api/users`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${pt1Token}`,
      },
      body: JSON.stringify({
        username: 'rogue_admin',
        password: 'Password123',
        role: 'super_admin',
        fullName: 'Rogue Admin',
      }),
    });
    expect(createAdminAsPt.status).toBe(403);

    const auditAsPt = await fetch(`${baseUrl}/api/audit-logs`, {
      headers: { Authorization: `Bearer ${pt1Token}` },
    });
    expect(auditAsPt.status).toBe(403);

    const auditAsAdmin = await fetch(`${baseUrl}/api/audit-logs`, {
      headers: { Authorization: `Bearer ${adminToken}` },
    });
    expect(auditAsAdmin.status).toBe(200);
    const auditBody = (await auditAsAdmin.json()) as { logs: unknown[] };
    expect(auditBody.logs.length).toBeGreaterThan(0);
  });

  it('enrolls 10-frame biometric patient, verifies genuine Kiosk face, and rejects stranger', async () => {
    const rawFrames = [
      ...Array.from({ length: 4 }, () => ({ pose: 'front' as const, vector: ownerCentroid, qualityScore: 0.96 })),
      ...Array.from({ length: 2 }, () => ({ pose: 'left' as const, vector: ownerCentroid, qualityScore: 0.91 })),
      ...Array.from({ length: 2 }, () => ({ pose: 'right' as const, vector: ownerCentroid, qualityScore: 0.92 })),
      ...Array.from({ length: 2 }, () => ({ pose: 'nod' as const, vector: ownerCentroid, qualityScore: 0.93 })),
    ];

    const enrollRes = await fetch(`${baseUrl}/api/biometric/enroll`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${pt1Token}`,
      },
      body: JSON.stringify({
        hn: 'HN-9001',
        fullName: 'คุณสมหมาย แข็งแรง',
        age: 68,
        phone: '081-999-0001',
        diagnosis: 'Post-Stroke Cognitive & Upper Limb Rehab',
        frontalCentroid: ownerCentroid,
        overallCentroid: ownerCentroid,
        rawFrames,
      }),
    });
    expect(enrollRes.status).toBe(201);
    const enrollBody = (await enrollRes.json()) as { patient: { id: string } };
    enrolledPatientId = enrollBody.patient.id;

    // Public Kiosk Face Verification for enrolled patient
    const genuineRes = await fetch(`${baseUrl}/api/biometric/verify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ probeVector: ownerCentroid }),
    });
    expect(genuineRes.status).toBe(200);
    const genuineBody = (await genuineRes.json()) as {
      matched: boolean;
      patient: { id: string; hn: string } | null;
      todayRoutine: { dayIndex: number; activityType: string } | null;
    };
    expect(genuineBody.matched).toBe(true);
    expect(genuineBody.patient?.id).toBe(enrolledPatientId);
    expect(genuineBody.todayRoutine).not.toBeNull();

    // Unregistered stranger must get 401 and matched: false
    const strangerVector = makeHybridEmbedding(8.91);
    const strangerRes = await fetch(`${baseUrl}/api/biometric/verify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ probeVector: strangerVector }),
    });
    expect(strangerRes.status).toBe(401);
    const strangerBody = (await strangerRes.json()) as { matched: boolean; confidencePercent: number };
    expect(strangerBody.matched).toBe(false);
    expect(strangerBody.confidencePercent).toBeLessThan(50);
  });

  it('enforces Append-Only medical notes, PT physio course lock, delegation [ฝากเคสชั่วคราว], and scoped library', async () => {
    // Append-only notes
    const noteRes = await fetch(`${baseUrl}/api/patients/${enrolledPatientId}/notes`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${pt1Token}`,
      },
      body: JSON.stringify({ content: 'Patient responds well to cognitive matching prompts.' }),
    });
    expect(noteRes.status).toBe(201);
    const { note } = (await noteRes.json()) as { note: { id: string } };

    const editNoteAsPt = await fetch(`${baseUrl}/api/patients/${enrolledPatientId}/notes/${note.id}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${pt1Token}`,
      },
      body: JSON.stringify({ content: 'Tampered note' }),
    });
    expect(editNoteAsPt.status).toBe(403);
    expect(((await editNoteAsPt.json()) as { error: string }).error).toBe('APPEND_ONLY_NOTES');

    const deleteNoteAsPt = await fetch(`${baseUrl}/api/patients/${enrolledPatientId}/notes/${note.id}`, {
      method: 'DELETE',
      headers: { Authorization: `Bearer ${pt1Token}` },
    });
    expect(deleteNoteAsPt.status).toBe(403);

    // PT cannot prescribe physio_exercise while AI Pose Detection is disabled
    const physioCourseRes = await fetch(`${baseUrl}/api/courses/${enrolledPatientId}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${pt1Token}`,
      },
      body: JSON.stringify({
        title: 'Blocked Physio Schedule',
        cognitiveModpackId: 'modpack-core-99',
        weeklySchedule: [0, 1, 2, 3, 4, 5, 6].map((dayIndex) => ({
          dayIndex,
          dayType: 'training',
          activityType: 'physio_exercise',
          difficulty: 'medium',
          label: 'Shoulder Flexion',
          exercises: [],
        })),
      }),
    });
    expect(physioCourseRes.status).toBe(403);
    expect(((await physioCourseRes.json()) as { error: string }).error).toBe(
      'PHYSIO_LOCKED_UNDER_DEVELOPMENT',
    );

    // PT can prescribe minigame schedule
    const minigameCourseRes = await fetch(`${baseUrl}/api/courses/${enrolledPatientId}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${pt1Token}`,
      },
      body: JSON.stringify({
        title: 'Cognitive Minigame Schedule',
        cognitiveModpackId: 'modpack-core-99',
        weeklySchedule: [0, 1, 2, 3, 4, 5, 6].map((dayIndex) => ({
          dayIndex,
          dayType: 'training',
          activityType: 'minigame',
          difficulty: 'medium',
          label: 'Cognitive Card Match',
          exercises: [],
        })),
      }),
    });
    expect(minigameCourseRes.status).toBe(200);

    // Temporary Case Handover [ฝากเคสชั่วคราว] from pt_somchai to pt_waranya (user-pt-2)
    const pt2ListBefore = await fetch(`${baseUrl}/api/patients`, {
      headers: { Authorization: `Bearer ${pt2Token}` },
    });
    const pt2PatientsBefore = ((await pt2ListBefore.json()) as { patients: { id: string }[] }).patients;
    expect(pt2PatientsBefore.some((p) => p.id === enrolledPatientId)).toBe(false);

    const delegateRes = await fetch(`${baseUrl}/api/patients/${enrolledPatientId}/delegate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${pt1Token}`,
      },
      body: JSON.stringify({ targetPtId: 'user-pt-2', note: 'ลาพักร้อน 2 วัน', delegationDate: '2026-11-15' }),
    });
    expect(delegateRes.status).toBe(200);
    const delegatedBody = (await delegateRes.json()) as {
      patient: { delegatedToPtId: string | null; delegationNote: string | null; delegationDate?: string | null };
    };
    expect(delegatedBody.patient.delegatedToPtId).toBe('user-pt-2');
    expect(delegatedBody.patient.delegationNote).toBe('ลาพักร้อน 2 วัน');
    expect(delegatedBody.patient.delegationDate).toBe('2026-11-15');

    const pt2ListAfter = await fetch(`${baseUrl}/api/patients`, {
      headers: { Authorization: `Bearer ${pt2Token}` },
    });
    const pt2PatientsAfter = ((await pt2ListAfter.json()) as { patients: { id: string }[] }).patients;
    expect(pt2PatientsAfter.some((p) => p.id === enrolledPatientId)).toBe(true);

    // Scoped exercise library: PT cannot delete hospital global exercise, can create & delete own
    const exListRes = await fetch(`${baseUrl}/api/exercises`);
    const { exercises } = (await exListRes.json()) as {
      exercises: { id: string; createdBy?: string }[];
    };
    const globalEx = exercises.find((e) => e.createdBy === 'system');
    expect(globalEx).toBeDefined();

    const delGlobalAsPt = await fetch(`${baseUrl}/api/exercises/${globalEx!.id}`, {
      method: 'DELETE',
      headers: { Authorization: `Bearer ${pt1Token}` },
    });
    expect(delGlobalAsPt.status).toBe(403);
    expect(((await delGlobalAsPt.json()) as { error: string }).error).toBe(
      'FORBIDDEN_GLOBAL_LIBRARY',
    );
  });

  it('supports Super Admin force logout, device ban, and cascading biometric patient purge', async () => {
    // Force logout pt_waranya (user-pt-2)
    const forceRes = await fetch(`${baseUrl}/api/security/force-logout/user-pt-2`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${adminToken}` },
    });
    expect(forceRes.status).toBe(200);

    const pt2AfterForce = await fetch(`${baseUrl}/api/auth/me`, {
      headers: { Authorization: `Bearer ${pt2Token}` },
    });
    expect(pt2AfterForce.status).toBe(401);

    // Device Ban
    const banRes = await fetch(`${baseUrl}/api/security/ban`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${adminToken}`,
      },
      body: JSON.stringify({ targetType: 'device', value: 'DEV-BAD', reason: 'Suspicious device' }),
    });
    expect(banRes.status).toBe(201);

    const blockedReq = await fetch(`${baseUrl}/api/health`, {
      headers: { 'x-device-id': 'DEV-BAD' },
    });
    expect(blockedReq.status).toBe(403);
    expect(((await blockedReq.json()) as { error: string }).error).toBe('BANNED_IP_OR_DEVICE');

    // Cascading delete patient removes biometric template too
    const delPatRes = await fetch(`${baseUrl}/api/patients/${enrolledPatientId}`, {
      method: 'DELETE',
      headers: { Authorization: `Bearer ${adminToken}` },
    });
    expect(delPatRes.status).toBe(200);
    expect(((await delPatRes.json()) as { cascadingPurge: boolean }).cascadingPurge).toBe(true);

    const verifyAfterDelete = await fetch(`${baseUrl}/api/biometric/verify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ probeVector: ownerCentroid }),
    });
    expect(verifyAfterDelete.status).toBe(401);
    expect(((await verifyAfterDelete.json()) as { matched: boolean }).matched).toBe(false);
  });

  it('evaluates live frame quality & anti-spoofing via /api/biometric/inspect-frame with strict JSON', async () => {
    const passRes = await fetch(`${baseUrl}/api/biometric/inspect-frame`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
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
      }),
    });
    expect(passRes.status).toBe(200);
    const passJson = (await passRes.json()) as {
      scan_status: string;
      detected_pose: string;
      error_code: string;
      checks: { framing_ok: boolean; lighting_ok: boolean; no_obstruction: boolean; liveness_ok: boolean };
    };
    expect(passJson.scan_status).toBe('PASS');
    expect(passJson.detected_pose).toBe('TURN_LEFT');
    expect(passJson.error_code).toBe('NONE');
    expect(passJson.checks.liveness_ok).toBe(true);

    const spoofRes = await fetch(`${baseUrl}/api/biometric/inspect-frame`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        faceCount: 1,
        metrics: {
          faceHeightRatio: 0.5,
          brightness: 120,
          sharpness: 16,
          eyesVisible: true,
          mouthVisible: true,
          moireOrBezelDetected: true,
          depthScore: 0.1,
        },
      }),
    });
    expect(spoofRes.status).toBe(200);
    const spoofJson = (await spoofRes.json()) as { scan_status: string; error_code: string };
    expect(spoofJson.scan_status).toBe('SPOOF_DETECTED');
    expect(spoofJson.error_code).toBe('SPOOF_ATTEMPT');
  });

  it('evaluates biomechanical telemetry & exact RTMPose/ZED SDK payload via /api/biomechanics/evaluate', async () => {
    const res = await fetch(`${baseUrl}/api/biomechanics/evaluate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        auth_status: { verified: true, patient_id: 'PT-8841', confidence: 0.94, liveness: true },
        spatial_framing: { distance_m: 2.05, full_body_visible: true },
        exercise_prescription: {
          exercise_name: 'Right Shoulder Abduction (กางแขนขวาด้านข้าง)',
          target_rom_deg: 150.0,
          hold_sec: 3.0,
          target_reps: 10,
        },
        live_telemetry: {
          primary_angle_deg: 138.5,
          trunk_tilt_deg: 14.2,
          shoulder_elevation_cm: 1.1,
          angular_velocity_deg_s: 28.0,
          hold_elapsed_sec: 0.0,
          pain_score: 0.1,
        },
      }),
    });
    expect(res.status).toBe(200);
    const json = (await res.json()) as {
      kiosk_state: string;
      telemetry_analysis: { rom_completion_pct: number; is_form_valid: boolean; compensation_detected: string[] };
      ui_directives: { skeleton_highlight_joint: string; highlight_color: string };
      tts_voice_th: string;
    };
    expect(Object.keys(json)).toEqual([
      'kiosk_state',
      'telemetry_analysis',
      'rep_counter',
      'ui_directives',
      'tts_voice_th',
    ]);
    expect(json.kiosk_state).toBe('ACTIVE_REP');
    expect(json.telemetry_analysis.rom_completion_pct).toBe(92);
    expect(json.telemetry_analysis.is_form_valid).toBe(false);
    expect(json.telemetry_analysis.compensation_detected).toEqual(['TRUNK_LEAN', 'INCOMPLETE_ROM']);
    expect(json.ui_directives.skeleton_highlight_joint).toBe('spine');
    expect(json.ui_directives.highlight_color).toBe('AMBER');
  });
});
