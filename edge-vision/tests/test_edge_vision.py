"""Comprehensive verification suite for Dr.Tech.Care Clinical 3D Edge Vision."""

from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path

os.environ["MOCK_VIDEO_PATH"] = "SYNTHETIC"

EDGE_ROOT = Path(__file__).resolve().parent.parent
if str(EDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(EDGE_ROOT))

import numpy as np
from fastapi.testclient import TestClient

from app.biomechanics.calibrator import StandingCalibrator
from app.biomechanics.exercises import ALL_26_EXERCISES, resolve_exercise_rule
from app.biomechanics.solver_3d import (
    Biomechanics3DSolver,
    compute_pelvic_tilt_deg,
    compute_shoulder_tilt_deg,
    compute_torso_plane,
    joint_angle_3d,
)
from app.biomechanics.state_machine import ClinicalRehabFSM
from app.config import EdgeVisionSettings
from app.main import create_app
from app.schemas.telemetry import ExercisePrescriptionSchema, HandRaiseTelemetry, PatientMatchSchema
from app.vision.face_120 import BUCKET_ORDER, CachedPatientBiometric, InsightFace120Manager, l2_normalize
from app.vision.pose_3d import (
    ClinicalPoseEstimator,
    MediaPipePoseEngine,
    OneEuroFilter,
    OneEuroFilter3D,
    RTMW3DPoseEstimator,
    TOTAL_MEDIAPIPE_KEYPOINTS,
    TOTAL_WHOLEBODY_KEYPOINTS,
    filter_landmarks_by_visibility,
    generate_synthetic_133_pose,
    wholebody133_to_mediapipe33,
    wholebody133_to_world_meters,
)


class TestClinicalEdgeVision(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = EdgeVisionSettings(
            MOCK_VIDEO_PATH="SYNTHETIC",
            SUPABASE_URL="",
            SUPABASE_ANON_KEY="",
            SUPABASE_SERVICE_ROLE_KEY="",
        )

    def test_01_mediapipe_pose_config_and_requirements(self) -> None:
        """Verify MediaPipe Pose clinical settings, complexity modes, and requirements alignment."""
        mp_cfg = self.settings.get_mediapipe_pose_config()
        self.assertEqual(mp_cfg["model_complexity"], 2)
        self.assertTrue(mp_cfg["smooth_landmarks"])
        self.assertGreaterEqual(mp_cfg["min_detection_confidence"], 0.7)
        self.assertGreaterEqual(mp_cfg["min_tracking_confidence"], 0.7)
        self.assertGreaterEqual(mp_cfg["min_visibility"], 0.7)

        # Kiosk precision mode overrides
        self.assertEqual(self.settings.resolve_model_complexity("clinical"), 2)
        self.assertEqual(self.settings.resolve_model_complexity("balanced"), 1)
        self.assertEqual(self.settings.resolve_model_complexity("lite"), 0)

        req_text = (EDGE_ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
        self.assertIn("mediapipe", req_text)
        self.assertIn("pytest", req_text)

        engine = MediaPipePoseEngine(self.settings)
        try:
            self.assertIsNotNone(engine.filter_2d)
            self.assertIsNotNone(engine.filter_3d)
            # When .task asset is bundled in workspace public/models, engine should initialize cleanly
            if self.settings.resolve_mediapipe_model_path() is not None:
                self.assertTrue(engine.is_ready)
        finally:
            engine.close()

    def test_02_one_euro_filter_jitter_suppression(self) -> None:
        """One-Euro Filter smooths micro-jitter during static hold across all 133 (X, Y, Z) points."""
        filt = OneEuroFilter3D(num_keypoints=TOTAL_WHOLEBODY_KEYPOINTS, min_cutoff=1.2, beta=0.08)
        rng = np.random.default_rng(42)
        base, _ = generate_synthetic_133_pose(0.0, is_calibrating=True)

        noisy_inputs = []
        filtered_outputs = []
        for i in range(40):
            ts = i * (1.0 / 60.0)
            noise = rng.normal(0.0, 0.015, size=(TOTAL_WHOLEBODY_KEYPOINTS, 3))
            noisy = base + noise
            noisy_inputs.append(noisy[10, 0])
            out = filt.filter(noisy, ts)
            if i >= 10:
                filtered_outputs.append(out[10, 0])

        self.assertLess(np.std(filtered_outputs), np.std(noisy_inputs[10:]) * 0.75)

    def test_03_two_second_standing_calibration(self) -> None:
        """2-Second Standing Calibration rejects tilted postures and locks 3D bone segments after 2s."""
        cal = StandingCalibrator(duration_sec=2.0, stability_max_tilt_deg=12.0)
        stand_pts, scores = generate_synthetic_133_pose(0.0, is_calibrating=True)

        # Tilted body should reset/reject
        tilted = stand_pts.copy()
        tilted[5, 0] += 0.22
        tilted[6, 0] += 0.22
        st_bad = cal.update(tilted, scores, 0.5)
        self.assertFalse(st_bad.is_calibrated)
        self.assertEqual(st_bad.progress, 0.0)

        # Upright standing for >= 2.0s completes calibration
        for step in range(25):
            st = cal.update(stand_pts, scores, step * 0.1)
        self.assertTrue(st.is_calibrated)
        self.assertIsNotNone(st.bone_lengths)
        self.assertGreater(st.bone_lengths.cm_per_unit, 80.0)  # type: ignore[union-attr]
        self.assertGreater(st.bone_lengths.torso_height, 0.15)  # type: ignore[union-attr]

    def test_04_all_26_clinical_exercises_and_aliases(self) -> None:
        """All 26 standardized Stroke ADL & MOPH exercises plus Kiosk aliases resolve cleanly."""
        self.assertEqual(len(ALL_26_EXERCISES), 26)
        ids = {r.id for r in ALL_26_EXERCISES}
        self.assertEqual(len(ids), 26)
        self.assertTrue(all(r.category in ("upper_limb", "neck_trunk") for r in ALL_26_EXERCISES))
        self.assertEqual(resolve_exercise_rule("right_shoulder_abduction").id, "STROKE_SH_ABD_R")
        self.assertEqual(resolve_exercise_rule("left_shoulder_abduction").id, "STROKE_SH_ABD_L")
        self.assertEqual(resolve_exercise_rule("STROKE_CROSS_BODY_R").id, "STROKE_CROSS_BODY_R")
        self.assertEqual(resolve_exercise_rule("MOPH_NECK_TURN_BILATERAL").id, "MOPH_NECK_TURN_BILATERAL")
        self.assertEqual(resolve_exercise_rule("ex-01").id, "STROKE_SH_FLEX_R")

    def test_05_3d_solver_angles_and_all_five_anti_cheat_checks(self) -> None:
        """Verify true 3D ROM + TRUNK_LEAN, SHOULDER_HIKE, ELBOW_FLEXION, TOO_FAST, and FALL."""
        solver = Biomechanics3DSolver(self.settings)
        cal = StandingCalibrator(duration_sec=2.0)
        stand_pts, _ = generate_synthetic_133_pose(0.0, is_calibrating=True)
        cal.force_calibrate_from_pose(stand_pts)

        # Verify pure 90 deg 3D angle
        ang90 = joint_angle_3d(
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, -1.0]),
        )
        self.assertAlmostEqual(ang90, 90.0, places=3)

        # 1. Clean right shoulder abduction (~150 deg)
        rule_sh = resolve_exercise_rule("STROKE_SH_ABD_R")
        good_sh, _ = generate_synthetic_133_pose(0.0, "STROKE_SH_ABD_R", mock_pose_override="raise_right")
        res_good, emerg = solver.analyze(good_sh, rule_sh, cal, timestamp_sec=1.0, target_rom_override=140.0)
        self.assertIsNone(emerg)
        self.assertTrue(res_good.is_form_valid)
        self.assertGreater(res_good.current_rom_deg, 130.0)

        # 2. TRUNK_LEAN (> 10 deg) & SHOULDER_HIKE (> 3 cm) & TOO_FAST (> 145 deg/s)
        cheat_sh = good_sh.copy()
        cheat_sh[5, 0] += 0.14
        cheat_sh[6, 0] += 0.14
        cheat_sh[6, 1] -= 0.04  # right shoulder hiked up ~8cm
        cheat_sh[8] = cheat_sh[6] + np.array([-0.02, 0.14, 0.0])  # instant drop creating huge ang_vel
        res_cheat, _ = solver.analyze(cheat_sh, rule_sh, cal, timestamp_sec=1.05, target_rom_override=140.0)
        self.assertFalse(res_cheat.is_form_valid)
        self.assertIn("TRUNK_LEAN", res_cheat.compensation_detected)
        self.assertIn("SHOULDER_HIKE", res_cheat.compensation_detected)
        self.assertIn("TOO_FAST", res_cheat.compensation_detected)

        # 3. Straight-arm guard (ELBOW_FLEXION) on STROKE_SH_FLEX_R when elbow bends excessively
        solver.reset()
        rule_flex = resolve_exercise_rule("STROKE_SH_FLEX_R")
        bent_arm = good_sh.copy()
        bent_arm[10] = bent_arm[8] + np.array([0.0, -0.13, 0.0])  # wrist folded sharp at elbow (~30-70 deg interior)
        res_bent, _ = solver.analyze(bent_arm, rule_flex, cal, timestamp_sec=2.0)
        self.assertIn("ELBOW_FLEXION", res_bent.compensation_detected)

        # 4. Emergency Fall Detection
        fall_pts, _ = generate_synthetic_133_pose(0.0, "STROKE_SH_ABD_R", mock_pose_override="fall")
        res_fall, fall_reason = solver.analyze(fall_pts, rule_sh, cal, timestamp_sec=2.1)
        self.assertIsNotNone(fall_reason)
        self.assertTrue(res_fall.pain_alert)

    def test_06_clinical_fsm_full_rep_cycle_and_emergency(self) -> None:
        """Verify FSM lifecycle: FACE_AUTH -> CALIBRATE_2S -> ACTIVE -> HOLD -> REP_DONE -> SET_FINISHED."""
        fsm = ClinicalRehabFSM(self.settings)
        self.assertEqual(fsm.fsm_state, "FACE_AUTH")

        rx = ExercisePrescriptionSchema(
            exercise_id="STROKE_SH_ABD_R",
            target_rom_deg=130.0,
            hold_sec=0.5,
            target_reps=1,
        )
        fsm.set_prescription(rx, start_calibration=True)
        self.assertEqual(fsm.fsm_state, "CALIBRATE_2S")

        stand_pts, scores = generate_synthetic_133_pose(0.0, is_calibrating=True)
        raised_pts, _ = generate_synthetic_133_pose(0.0, "STROKE_SH_ABD_R", mock_pose_override="raise_right")
        hand_empty = HandRaiseTelemetry()

        # Run 2.1s upright calibration
        t = 0.0
        for _ in range(22):
            payload = fsm.step(stand_pts, scores, t, 30.0, "synthetic_mock", hand_empty, [])
            t += 0.1
        self.assertEqual(payload.fsm_state, "ACTIVE")

        # Raise right arm physiologically (~1.2s raise so angular velocity < 145 deg/s)
        t += 1.2
        payload = fsm.step(raised_pts, scores, t, 30.0, "synthetic_mock", hand_empty, [])
        self.assertEqual(payload.fsm_state, "HOLD")

        # Hold for > 0.5s cleanly
        t += 0.6
        payload = fsm.step(raised_pts, scores, t, 30.0, "synthetic_mock", hand_empty, [])
        self.assertEqual(payload.fsm_state, "REP_DONE")
        self.assertEqual(payload.rep_counter.valid_reps, 1)

        # Return to rest -> SET_FINISHED (use gradual dt so ang_vel < 145 deg/s)
        t += 2.0
        payload = fsm.step(stand_pts, scores, t, 30.0, "synthetic_mock", hand_empty, [])
        self.assertEqual(payload.fsm_state, "SET_FINISHED")
        self.assertGreater(len(payload.tts_voice_th), 5)

        # Emergency trigger halts cleanly
        fsm.trigger_emergency("Unit test emergency")
        self.assertEqual(fsm.fsm_state, "EMERGENCY")

    def test_07_insightface_120_enrollment_and_sub_100ms_1_to_n_match(self) -> None:
        """Verify 120-frame 5-angle aggregation & <100ms cosine matching across 150 patients."""
        mgr = InsightFace120Manager(self.settings)
        sess = mgr.start_enrollment(patient_id="p-test-01", hn="HN-99001", full_name="Somchai Test")

        rng = np.random.default_rng(7)
        probe_frame = np.full((240, 240, 3), 128, dtype=np.uint8)
        probe_frame[60:180, 70:170, 0] = 210
        probe_frame[80:160, 90:150, 2] = 45

        # Enroll 120 frames (40 center + 20 each left/right/up/down)
        targets = {"center": 40, "left": 20, "right": 20, "up": 20, "down": 20}
        for bucket in BUCKET_ORDER:
            for _ in range(targets[bucket]):
                state = mgr.process_enrollment_frame(sess.session_id, probe_frame, forced_bucket=bucket)
        self.assertEqual(state.frames_captured, 120)
        self.assertEqual(state.phase, "completed")

        res = mgr.finalize_enrollment(
            sess.session_id, patient_id="p-test-01", hn="HN-99001", persist_to_supabase=False
        )
        self.assertEqual(res.frames_captured, 120)
        self.assertEqual(len(res.master_embedding), 512)
        self.assertEqual(set(res.angle_embeddings.keys()), set(BUCKET_ORDER))

        # Populate 149 additional synthetic enrolled patients -> 150 total 1:N candidates
        extra = []
        for idx in range(149):
            rand_vec = l2_normalize(rng.normal(size=512))
            extra.append(
                CachedPatientBiometric(
                    patient_id=f"p-mock-{idx}",
                    hn=f"HN-{idx:04d}",
                    full_name=f"Patient {idx}",
                    master_embedding=rand_vec,
                    angle_embeddings={},
                )
            )
        mgr._patients = mgr._patients + extra
        mgr._matrix = np.vstack([p.master_embedding for p in mgr._patients])

        feat = mgr.extract_face_features(probe_frame)
        self.assertIsNotNone(feat)
        t0 = time.perf_counter()
        match = mgr.match_1_to_n(feat["embedding"])  # type: ignore[index]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.assertLess(elapsed_ms, 100.0)
        self.assertLess(match.latency_ms, 100.0)
        self.assertEqual(match.id, "p-test-01")
        self.assertGreater(match.similarity, 0.90)

    def test_08_fastapi_rest_and_websocket_endpoints(self) -> None:
        """Verify FastAPI /health, /api/exercises, /api/enroll/*, and WebSocket /ws/kiosk."""
        app = create_app()
        with TestClient(app) as client:
            health = client.get("/health")
            self.assertEqual(health.status_code, 200)
            h_json = health.json()
            self.assertEqual(h_json["status"], "ok")
            self.assertEqual(h_json["wholebody_keypoints"], 133)

            exs = client.get("/api/exercises")
            self.assertEqual(exs.status_code, 200)
            self.assertEqual(exs.json()["count"], 26)

            # Start -> frame -> finalize enrollment
            s_res = client.post("/api/enroll/start", json={"patient_id": "p-ws-01", "hn": "HN-555"}).json()
            sid = s_res["session_id"]
            f_res = client.post("/api/enroll/frame", json={"session_id": sid}).json()
            self.assertGreaterEqual(f_res["frames_captured"], 1)
            fin = client.post(
                "/api/enroll/finalize",
                json={"session_id": sid, "patient_id": "p-ws-01", "hn": "HN-555", "persist_to_supabase": False},
            )
            self.assertEqual(fin.status_code, 200)
            self.assertEqual(len(fin.json()["master_embedding"]), 512)

            # WebSocket telemetry & commands
            with client.websocket_connect("/ws/kiosk") as ws:
                first = ws.receive_json()
                self.assertIn("keypoints_3d", first)
                self.assertEqual(len(first["keypoints_3d"]), 133)
                self.assertIn("hand_status", first)

                ws.send_json({"type": "prescription", "data": {"exercise_id": "STROKE_SH_ABD_R", "target_rom_deg": 140}})
                ack = ws.receive_json()
                while ack.get("type") != "ack":
                    ack = ws.receive_json()
                self.assertEqual(ack["fsm_state"], "CALIBRATE_2S")

                ws.send_json({"type": "emergency_stop", "reason": "Test WS Stop"})
                ack2 = ws.receive_json()
                while ack2.get("type") != "ack":
                    ack2 = ws.receive_json()
                self.assertEqual(ack2["fsm_state"], "EMERGENCY")

    def test_09_continuous_identity_guard_multi_person_and_auto_unlock(self) -> None:
        """Rule 1: Multi-person crowd detection freezes reps/hold and auto-unlocks after 3 single-person matches."""
        pose_est = RTMW3DPoseEstimator(self.settings)
        mgr = InsightFace120Manager(self.settings)
        fsm = ClinicalRehabFSM(self.settings)

        rng = np.random.default_rng(11)
        auth_vec = l2_normalize(rng.normal(size=512))
        mgr.lock_active_patient(
            patient_id="p-mp-01",
            hn="HN-701",
            full_name="Multi Person Patient",
            master_embedding=auth_vec.tolist(),
        )
        fsm.set_patient_matched(
            PatientMatchSchema(id="p-mp-01", hn="HN-701", full_name="Multi Person Patient", similarity=1.0),
            auto_start_calibration=False,
        )
        rx = ExercisePrescriptionSchema(exercise_id="STROKE_SH_ABD_R", target_rom_deg=110.0, hold_sec=0.5, target_reps=3)
        fsm.set_prescription(rx, start_calibration=True)

        scores = np.full((TOTAL_WHOLEBODY_KEYPOINTS,), 0.95, dtype=np.float64)
        stand_pts, _ = generate_synthetic_133_pose(0.0, is_calibrating=True)
        raised_pts, _ = generate_synthetic_133_pose(1.5, exercise_id="STROKE_SH_ABD_R", is_calibrating=False)
        hand_empty = HandRaiseTelemetry()

        t = 0.0
        for _ in range(22):
            payload = fsm.step(stand_pts, scores, t, 30.0, "synthetic_mock", hand_empty, [])
            t += 0.1
        self.assertEqual(payload.fsm_state, "ACTIVE")

        # Simulate 2+ people in frame
        xyz_mp, scores_mp, skel_cnt, skel_bboxes = pose_est.estimate_with_crowd(
            frame_bgr=None,
            timestamp_sec=t,
            exercise_id="STROKE_SH_ABD_R",
            mock_pose_override="multi_person",
            is_mock_mode=True,
        )
        self.assertGreaterEqual(skel_cnt, 2)
        self.assertGreaterEqual(len(skel_bboxes), 2)

        guard_mp = mgr.evaluate_identity_guard(
            frame_bgr=None,
            primary_xyz=xyz_mp,
            primary_scores=scores_mp,
            skeleton_count=skel_cnt,
            skeleton_bboxes=skel_bboxes,
            mock_override="multi_person",
        )
        self.assertEqual(guard_mp.status, "MULTI_PERSON")
        self.assertFalse(guard_mp.is_counting_allowed)

        # Even if raising arm past target and holding, valid_reps stays frozen at 0
        t += 1.2
        p1 = fsm.step(raised_pts, scores, t, 30.0, "synthetic_mock", hand_empty, [], identity_guard=guard_mp)
        self.assertEqual(p1.fsm_state, "PAUSED_MULTI_PERSON")
        self.assertEqual(p1.kiosk_state, "PAUSED_MULTI_PERSON")
        self.assertEqual(p1.ui_directives.highlight_color, "AMBER")
        self.assertIn("ตรวจพบมีบุคคลอยู่หลายคนในเฟรม", p1.tts_voice_th)
        self.assertEqual(p1.rep_counter.valid_reps, 0)
        self.assertEqual(p1.rep_counter.hold_remaining_sec, 0.5)

        t += 0.8
        p2 = fsm.step(raised_pts, scores, t, 30.0, "synthetic_mock", hand_empty, [], identity_guard=guard_mp)
        self.assertEqual(p2.rep_counter.valid_reps, 0)

        # Extra person leaves -> 1 person + 3 consecutive face matches unlock automatically
        auth_face = [
            {
                "embedding": auth_vec.tolist(),
                "det_score": 0.96,
                "bbox_1280": [540, 40, 740, 220],
                "bbox_norm": (0.42, 0.05, 0.58, 0.30),
            }
        ]
        g1 = mgr.evaluate_identity_guard(
            frame_bgr=None,
            primary_xyz=stand_pts,
            primary_scores=scores,
            skeleton_count=1,
            face_detections_override=auth_face,
            force_identity_check=True,
        )
        g2 = mgr.evaluate_identity_guard(
            frame_bgr=None,
            primary_xyz=stand_pts,
            primary_scores=scores,
            skeleton_count=1,
            face_detections_override=auth_face,
            force_identity_check=True,
        )
        self.assertFalse(g1.is_counting_allowed)
        self.assertFalse(g2.is_counting_allowed)

        g3 = mgr.evaluate_identity_guard(
            frame_bgr=None,
            primary_xyz=stand_pts,
            primary_scores=scores,
            skeleton_count=1,
            face_detections_override=auth_face,
            force_identity_check=True,
        )
        self.assertEqual(g3.status, "VERIFIED")
        self.assertTrue(g3.is_counting_allowed)

        t += 1.5
        unlocked = fsm.step(stand_pts, scores, t, 30.0, "synthetic_mock", hand_empty, [], identity_guard=g3)
        self.assertEqual(unlocked.fsm_state, "ACTIVE")
        self.assertIn("ยืนยันตัวตนคนไข้ถูกต้องแล้ว", unlocked.tts_voice_th)

    def test_10_continuous_identity_guard_impostor_and_spatial_binding(self) -> None:
        """Rule 2: Impostor / spatially unbound face locks counting in RED state and unlocks on 3 verified frames."""
        mgr = InsightFace120Manager(self.settings)
        fsm = ClinicalRehabFSM(self.settings)

        rng = np.random.default_rng(21)
        auth_vec = l2_normalize(rng.normal(size=512))
        raw_imp = rng.normal(size=512)
        impostor_vec = l2_normalize(raw_imp - np.dot(raw_imp, auth_vec) * auth_vec)

        mgr.lock_active_patient(
            patient_id="p-auth-1",
            hn="HN-101",
            full_name="Authorized Patient",
            master_embedding=auth_vec.tolist(),
        )
        rx = ExercisePrescriptionSchema(exercise_id="STROKE_SH_ABD_R", target_rom_deg=110.0, hold_sec=0.5, target_reps=3)
        fsm.set_prescription(rx, start_calibration=True)

        scores = np.full((TOTAL_WHOLEBODY_KEYPOINTS,), 0.95, dtype=np.float64)
        stand_pts, _ = generate_synthetic_133_pose(0.0, is_calibrating=True)
        raised_pts, _ = generate_synthetic_133_pose(1.5, exercise_id="STROKE_SH_ABD_R", is_calibrating=False)
        hand_empty = HandRaiseTelemetry()

        t = 0.0
        for _ in range(22):
            fsm.step(stand_pts, scores, t, 30.0, "synthetic_mock", hand_empty, [])
            t += 0.1
        self.assertEqual(fsm.fsm_state, "ACTIVE")

        # 5 consecutive impostor mismatches lock into IMPOSTOR
        imp_face = [
            {
                "embedding": impostor_vec.tolist(),
                "det_score": 0.92,
                "bbox_1280": [540, 40, 740, 220],
                "bbox_norm": (0.42, 0.05, 0.58, 0.30),
            }
        ]
        guard_imp = None
        for idx in range(5):
            guard_imp = mgr.evaluate_identity_guard(
                frame_bgr=None,
                primary_xyz=stand_pts,
                primary_scores=scores,
                skeleton_count=1,
                face_detections_override=imp_face,
                force_identity_check=True,
            )
            if idx < 4:
                self.assertEqual(guard_imp.status, "VERIFIED")
        assert guard_imp is not None
        self.assertEqual(guard_imp.status, "IMPOSTOR")
        self.assertFalse(guard_imp.is_counting_allowed)
        self.assertEqual(guard_imp.detected_boxes[0].label, "✕ ไม่ใช่คนไข้ที่ลงทะเบียน")

        t += 1.2
        p_imp = fsm.step(raised_pts, scores, t, 30.0, "synthetic_mock", hand_empty, [], identity_guard=guard_imp)
        self.assertEqual(p_imp.fsm_state, "PAUSED_IMPOSTOR")
        self.assertEqual(p_imp.ui_directives.highlight_color, "RED")
        self.assertEqual(p_imp.rep_counter.valid_reps, 0)

        # Skeleton-to-Face spatial binding check: authorized face far away from skeleton head fails binding
        mgr.reset_identity_guard(clear_patient=False)
        off_corner_box = (0.02, 0.02, 0.12, 0.15)
        self.assertFalse(mgr._is_face_bound_to_skeleton(off_corner_box, stand_pts, scores))
        unbound_face = [
            {
                "embedding": auth_vec.tolist(),
                "det_score": 0.95,
                "bbox_1280": [25, 14, 150, 110],
                "bbox_norm": off_corner_box,
            }
        ]
        guard_unbound = None
        for _ in range(5):
            guard_unbound = mgr.evaluate_identity_guard(
                frame_bgr=None,
                primary_xyz=stand_pts,
                primary_scores=scores,
                skeleton_count=1,
                face_detections_override=unbound_face,
                force_identity_check=True,
            )
        assert guard_unbound is not None
        self.assertEqual(guard_unbound.status, "IMPOSTOR")

        # WebSocket /ws/kiosk lock_patient + multi_person smoke test
        app = create_app()
        with TestClient(app) as client:
            with client.websocket_connect("/ws/kiosk") as ws:
                _ = ws.receive_json()
                ws.send_json(
                    {
                        "type": "lock_patient",
                        "patient": {"id": "p-ws-guard", "hn": "HN-999", "full_name": "WS Guard"},
                        "start_calibration": True,
                    }
                )
                ack = ws.receive_json()
                while ack.get("type") != "ack":
                    ack = ws.receive_json()
                self.assertEqual(ack["command"], "lock_patient")
                self.assertEqual(ack["patient_id"], "p-ws-guard")

                ws.send_json({"type": "mock_pose", "pose": "multi_person"})
                frame = ws.receive_json()
                while frame.get("type") == "ack" or frame.get("fsm_state") != "PAUSED_MULTI_PERSON":
                    frame = ws.receive_json()
                self.assertEqual(frame["identity_guard"]["status"], "MULTI_PERSON")
                self.assertFalse(frame["identity_guard"]["is_counting_allowed"])

    def test_11_dual_2d_3d_one_euro_filter_and_visibility_gating(self) -> None:
        """OneEuroFilter smooths 2D & 3D coordinates using real timestamps and freezes low-visibility outliers."""
        filter_2d = OneEuroFilter(min_cutoff=1.5, beta=0.08, d_cutoff=1.0, min_visibility=0.70)
        filter_3d = OneEuroFilter(min_cutoff=1.2, beta=0.08, d_cutoff=1.0, min_visibility=0.70)

        rng = np.random.default_rng(123)
        base_133, scores_133 = generate_synthetic_133_pose(0.0, is_calibrating=True)
        base_2d, vis_33 = wholebody133_to_mediapipe33(base_133, scores_133)
        base_3d = wholebody133_to_world_meters(base_133)[:TOTAL_MEDIAPIPE_KEYPOINTS]

        raw_samples = []
        smooth_samples = []
        for i in range(30):
            ts = i / 60.0
            n2 = base_2d + rng.normal(0.0, 0.012, size=base_2d.shape)
            n3 = base_3d + rng.normal(0.0, 0.015, size=base_3d.shape)
            s2 = filter_2d.filter(n2, timestamp_sec=ts, visibility=vis_33)
            s3 = filter_3d.filter(n3, timestamp_sec=ts, visibility=vis_33)
            raw_samples.append(n3[14, 0])
            if i >= 10:
                smooth_samples.append(s3[14, 0])
            self.assertEqual(s2.shape, (TOTAL_MEDIAPIPE_KEYPOINTS, 3))
            self.assertEqual(s3.shape, (TOTAL_MEDIAPIPE_KEYPOINTS, 3))

        self.assertLess(np.std(smooth_samples), np.std(raw_samples[10:]) * 0.75)

        # Low-visibility spike on landmark 14 (right elbow) should be rejected by filter & gating helper
        last_valid = filter_3d.filter(base_3d, timestamp_sec=0.60, visibility=vis_33)
        occluded_vis = vis_33.copy()
        occluded_vis[14] = 0.20
        spike_3d = base_3d.copy()
        spike_3d[14] += np.array([2.5, -2.5, 1.5])
        held = filter_3d.filter(spike_3d, timestamp_sec=0.62, visibility=occluded_vis)
        self.assertAlmostEqual(float(held[14, 0]), float(last_valid[14, 0]), places=6)

        cleaned, valid_mask, min_req = filter_landmarks_by_visibility(
            spike_3d,
            occluded_vis,
            min_visibility=0.70,
            fallback_xyz=last_valid,
            required_indices=(11, 12, 14, 16, 23, 24),
        )
        self.assertFalse(bool(valid_mask[14]))
        self.assertLess(min_req, 0.70)
        self.assertTrue(np.allclose(cleaned[14], last_valid[14]))

    def test_12_3d_world_yaw_invariance_torso_plane_shoulder_pelvic_tilt_and_elbow(self) -> None:
        """True 3D vectors from world landmarks remain invariant to yaw rotation and flag compensations."""
        solver = Biomechanics3DSolver(self.settings)
        cal = StandingCalibrator(duration_sec=2.0)
        stand_pts, scores = generate_synthetic_133_pose(0.0, is_calibrating=True)
        stand_world = wholebody133_to_world_meters(stand_pts)
        cal.force_calibrate_from_pose(stand_pts, world_pts=stand_world)

        raised_pts, _ = generate_synthetic_133_pose(0.0, "STROKE_SH_ABD_R", mock_pose_override="raise_right")
        raised_world = wholebody133_to_world_meters(raised_pts)
        rule = resolve_exercise_rule("STROKE_SH_ABD_R")

        res_front, _ = solver.analyze(
            raised_pts, rule, cal, timestamp_sec=1.0, target_rom_override=140.0, world_pts=raised_world, scores=scores
        )
        self.assertTrue(res_front.is_form_valid)
        self.assertGreaterEqual(res_front.stability_score, 85.0)

        # Rotate world landmarks 35 deg around vertical Y axis (patient standing angled to camera)
        theta = np.radians(35.0)
        rot_y = np.array(
            [
                [np.cos(theta), 0.0, np.sin(theta)],
                [0.0, 1.0, 0.0],
                [-np.sin(theta), 0.0, np.cos(theta)],
            ],
            dtype=np.float64,
        )
        rotated_world = raised_world @ rot_y.T
        plane = compute_torso_plane(rotated_world)
        self.assertGreater(abs(plane["yaw_deg"]), 25.0)

        solver.reset()
        res_rot, _ = solver.analyze(
            raised_pts, rule, cal, timestamp_sec=1.1, target_rom_override=140.0, world_pts=rotated_world, scores=scores
        )
        self.assertAlmostEqual(res_front.current_rom_deg, res_rot.current_rom_deg, delta=1.0)

        # Check Shoulder Tilt, Pelvic Tilt, Elbow Flexion, Low Visibility warnings
        bad_world = raised_world.copy()
        bad_world[6, 1] -= 0.12  # heavy right shoulder hike/tilt (> 12 deg)
        bad_world[12, 1] += 0.08  # pelvic tilt (> 10 deg)
        # Bend right elbow by folding wrist back toward shoulder (~90 deg < 150 deg)
        bad_world[10] = bad_world[8] + np.array([0.0, -0.25, 0.0])

        self.assertGreater(compute_shoulder_tilt_deg(bad_world), 12.0)
        self.assertGreater(compute_pelvic_tilt_deg(bad_world), 10.0)

        low_scores = scores.copy()
        low_scores[8] = 0.40  # right elbow occluded

        res_bad, _ = solver.analyze(
            raised_pts, rule, cal, timestamp_sec=1.6, target_rom_override=140.0, world_pts=bad_world, scores=low_scores
        )
        self.assertFalse(res_bad.is_form_valid)
        self.assertFalse(res_bad.visibility_ok)
        self.assertIn("SHOULDER_HIKE", res_bad.compensation_detected)
        self.assertIn("SHOULDER_TILT", res_bad.compensation_detected)
        self.assertIn("PELVIC_TILT", res_bad.compensation_detected)
        self.assertIn("ELBOW_FLEXION", res_bad.compensation_detected)
        self.assertIn("LOW_VISIBILITY", res_bad.compensation_detected)
        self.assertGreater(len(res_bad.compensation_warnings), 0)
        self.assertLess(res_bad.stability_score, 80.0)

    def test_13_calibrator_distance_framing_and_anthropometrics(self) -> None:
        """Calibrator rejects too-far, too-close, out-of-frame, or low-visibility poses before starting."""
        cal = StandingCalibrator(
            duration_sec=2.0,
            stability_max_tilt_deg=12.0,
            min_visibility=0.70,
            min_torso_height_ratio=0.16,
            max_torso_height_ratio=0.55,
        )
        stand_pts, scores = generate_synthetic_133_pose(0.0, is_calibrating=True)

        # Too far (shrink skeleton around center)
        too_far = 0.5 + (stand_pts - 0.5) * 0.35
        st_far = cal.update(too_far, scores, 0.2)
        self.assertFalse(st_far.is_calibrated)
        self.assertEqual(st_far.distance_status, "TOO_FAR")

        # Too close (expand torso height > 0.55 inside frame)
        too_close = stand_pts.copy()
        too_close[5, 1] = 0.15
        too_close[6, 1] = 0.15
        too_close[11, 1] = 0.80
        too_close[12, 1] = 0.80
        st_close = cal.update(too_close, scores, 0.4)
        self.assertFalse(st_close.is_calibrated)
        self.assertEqual(st_close.distance_status, "TOO_CLOSE")

        # Low visibility
        low_vis = scores.copy()
        low_vis[7] = 0.35
        st_occ = cal.update(stand_pts, low_vis, 0.6)
        self.assertFalse(st_occ.is_calibrated)
        self.assertEqual(st_occ.distance_status, "OCCLUDED")
        self.assertIn("left_elbow", st_occ.missing_joints)

        # Valid framing & anthropometric lock
        world_pts = wholebody133_to_world_meters(stand_pts)
        for step in range(25):
            st_ok = cal.update(stand_pts, scores, 1.0 + step * 0.1, world_pts=world_pts)
        self.assertTrue(st_ok.is_calibrated)
        self.assertEqual(st_ok.distance_status, "OK")
        assert st_ok.bone_lengths is not None
        self.assertGreater(st_ok.bone_lengths.arm_to_torso_ratio, 0.4)
        self.assertGreater(st_ok.bone_lengths.leg_to_torso_ratio, 0.8)

    def test_14_fsm_movement_phases_hysteresis_and_freeze_hold_timer(self) -> None:
        """Verify IDLE->READY->CONCENTRIC->HOLD->REP_COMPLETE->ECCENTRIC, hysteresis band, and Freeze Hold Timer."""
        fsm = ClinicalRehabFSM(self.settings)
        self.assertEqual(fsm.movement_phase, "IDLE")

        rx = ExercisePrescriptionSchema(
            exercise_id="STROKE_SH_ABD_R",
            target_rom_deg=110.0,
            hold_sec=2.0,
            target_reps=2,
        )
        fsm.set_prescription(rx, start_calibration=True)
        stand_pts, scores = generate_synthetic_133_pose(0.0, is_calibrating=True)
        hand_empty = HandRaiseTelemetry()

        # Calibrate over 2.1s
        t = 0.0
        for _ in range(22):
            p = fsm.step(stand_pts, scores, t, 30.0, "synthetic_mock", hand_empty, [])
            t += 0.1
        self.assertEqual(p.fsm_state, "ACTIVE")
        self.assertEqual(p.movement_phase, "READY")

        # Partial raise (~65 deg) -> CONCENTRIC
        pts_65, _ = generate_synthetic_133_pose(0.0, is_calibrating=True)
        upper = 0.145
        fore = 0.135
        ang65 = np.radians(65.0)
        pts_65[8] = [pts_65[6, 0] - np.sin(ang65) * upper, pts_65[6, 1] + np.cos(ang65) * upper, 0.0]
        pts_65[10] = [pts_65[8, 0] - np.sin(ang65) * fore, pts_65[8, 1] + np.cos(ang65) * fore, 0.0]
        t += 0.6
        p_conc = fsm.step(pts_65, scores, t, 30.0, "synthetic_mock", hand_empty, [])
        self.assertEqual(p_conc.fsm_state, "ACTIVE")
        self.assertEqual(p_conc.movement_phase, "CONCENTRIC")

        # Enter target zone (>= 110 - 6 = 104 deg) -> HOLD
        def make_arm_angle(deg: float) -> np.ndarray:
            arr = stand_pts.copy()
            r = np.radians(deg)
            arr[8] = [arr[6, 0] - np.sin(r) * upper, arr[6, 1] + np.cos(r) * upper, 0.0]
            arr[10] = [arr[8, 0] - np.sin(r) * fore, arr[8, 1] + np.cos(r) * fore, 0.0]
            return arr

        pts_108 = make_arm_angle(108.0)
        t += 0.5
        p_hold0 = fsm.step(pts_108, scores, t, 30.0, "synthetic_mock", hand_empty, [])
        self.assertEqual(p_hold0.fsm_state, "HOLD")
        self.assertEqual(p_hold0.movement_phase, "HOLD")

        # Hysteresis test: minor dip to 100 deg (< 104 enter threshold, > 96 exit threshold) stays in HOLD!
        pts_100 = make_arm_angle(100.0)
        t += 0.8
        p_hyst = fsm.step(pts_100, scores, t, 30.0, "synthetic_mock", hand_empty, [])
        self.assertEqual(p_hyst.fsm_state, "HOLD")
        self.assertAlmostEqual(p_hyst.hold_progress_pct, 40.0, delta=2.0)
        frozen_remaining = p_hyst.rep_counter.hold_remaining_sec

        # Compensation during HOLD (trunk lean > 10 deg) freezes Hold Timer immediately!
        pts_cheat = pts_108.copy()
        pts_cheat[5, 0] += 0.14
        pts_cheat[6, 0] += 0.14
        t += 0.7
        p_frozen = fsm.step(pts_cheat, scores, t, 30.0, "synthetic_mock", hand_empty, [])
        self.assertEqual(p_frozen.fsm_state, "HOLD")
        self.assertTrue(p_frozen.telemetry_analysis.hold_timer_frozen)
        self.assertAlmostEqual(p_frozen.rep_counter.hold_remaining_sec, frozen_remaining, places=2)
        self.assertAlmostEqual(p_frozen.hold_progress_pct, p_hyst.hold_progress_pct, places=1)

        # Resume valid hold for remaining 1.25s -> REP_COMPLETE
        t += 1.3
        p_done = fsm.step(pts_108, scores, t, 30.0, "synthetic_mock", hand_empty, [])
        self.assertEqual(p_done.fsm_state, "REP_DONE")
        self.assertEqual(p_done.movement_phase, "REP_COMPLETE")
        self.assertEqual(p_done.hold_progress_pct, 100.0)
        self.assertEqual(p_done.rep_counter.valid_reps, 1)

        # Mid-lowering -> ECCENTRIC, then full rest -> READY
        t += 0.6
        p_ecc = fsm.step(pts_65, scores, t, 30.0, "synthetic_mock", hand_empty, [])
        self.assertEqual(p_ecc.movement_phase, "ECCENTRIC")

        t += 0.8
        p_ready = fsm.step(stand_pts, scores, t, 30.0, "synthetic_mock", hand_empty, [])
        self.assertEqual(p_ready.fsm_state, "ACTIVE")
        self.assertEqual(p_ready.movement_phase, "READY")

    def test_15_telemetry_schema_precision_stability_and_http_endpoint(self) -> None:
        """Verify main.py /health and /telemetry emit 1-decimal ROM, hold_progress_pct, warnings, and stability_score."""
        est = ClinicalPoseEstimator(self.settings)
        self.addCleanup(est.close)
        app = create_app()
        with TestClient(app) as client:
            h = client.get("/health").json()
            self.assertIn("mediapipe_config", h)
            self.assertEqual(h["mediapipe_config"]["model_complexity"], 2)

            snap = client.get("/telemetry")
            self.assertEqual(snap.status_code, 200)
            body = snap.json()
            for key in (
                "current_rom_deg",
                "hold_progress_pct",
                "compensation_warnings",
                "stability_score",
                "movement_phase",
                "telemetry",
            ):
                self.assertIn(key, body)
            self.assertEqual(len(body["telemetry"]["pose_world_landmarks"]), TOTAL_MEDIAPIPE_KEYPOINTS)

    def test_16_seated_shoulder_flexion_vertical_reference_and_pna_headers(self) -> None:
        """Seated Right Shoulder Flexion (STROKE_SH_FLEX_R) with occluded/out-of-frame hips uses [0, 1, 0] vertical vector."""
        rule_flex = resolve_exercise_rule("STROKE_SH_FLEX_R")
        req_indices = rule_flex.required_landmark_indices()
        self.assertNotIn(11, req_indices)
        self.assertNotIn(12, req_indices)
        self.assertIn(6, req_indices)
        self.assertIn(8, req_indices)

        # Simulate seated desk posture: hips occluded (visibility=0.05) and out-of-frame (y=1.08),
        # while right arm reaches forward ~90 deg toward camera (-Z in MediaPipe world coords).
        pts_2d, scores = generate_synthetic_133_pose(0.0, is_calibrating=True)
        world_pts = wholebody133_to_world_meters(pts_2d)
        pts_2d[11, 1] = 1.08
        pts_2d[12, 1] = 1.08
        scores[11] = 0.05
        scores[12] = 0.05

        world_pts[8] = world_pts[6] + np.array([0.0, 0.0, -0.28])
        world_pts[10] = world_pts[8] + np.array([0.0, 0.0, -0.26])

        solver = Biomechanics3DSolver(self.settings)
        cal = StandingCalibrator(duration_sec=0.2)
        for step in range(4):
            st = cal.update(pts_2d, scores, step * 0.1, world_pts=world_pts, required_indices=req_indices)
        self.assertTrue(st.is_calibrated)

        res, emerg = solver.analyze(
            pts_2d,
            rule_flex,
            cal,
            timestamp_sec=1.0,
            target_rom_override=110.0,
            world_pts=world_pts,
            scores=scores,
        )
        self.assertIsNone(emerg)
        self.assertTrue(res.visibility_ok)
        self.assertGreater(res.current_rom_deg, 85.0)
        self.assertLess(res.trunk_tilt_deg, 10.0)
        self.assertNotIn("PELVIC_TILT", res.compensation_detected)

        # Verify PNA middleware header is present for cross-origin HTTPS -> localhost connections
        app = create_app()
        with TestClient(app) as client:
            resp = client.get("/health", headers={"Origin": "https://kiosk.example.workers.dev"})
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.headers.get("access-control-allow-private-network"), "true")

            with client.websocket_connect("/ws/kiosk") as ws:
                _ = ws.receive_json()
                ws.send_json({"type": "config", "stream_video": True})
                ack = ws.receive_json()
                while ack.get("type") != "ack":
                    ack = ws.receive_json()
                self.assertEqual(ack["command"], "config")
                nxt = ws.receive_json()
                self.assertIn("pose_landmarks", nxt)
                self.assertEqual(len(nxt["pose_landmarks"]), TOTAL_MEDIAPIPE_KEYPOINTS)

                # Bridge live 33-point browser MediaPipe pose/world landmarks -> Edge 3D evaluator
                mp_2d = [{"x": 0.5, "y": 0.5, "z": 0.0, "score": 0.99} for _ in range(TOTAL_MEDIAPIPE_KEYPOINTS)]
                mp_2d[11] = {"x": 0.62, "y": 0.78, "z": 0.0, "score": 0.99}
                mp_2d[12] = {"x": 0.38, "y": 0.78, "z": 0.0, "score": 0.99}
                mp_2d[14] = {"x": 0.38, "y": 0.78, "z": -0.18, "score": 0.99}
                mp_2d[16] = {"x": 0.38, "y": 0.78, "z": -0.35, "score": 0.99}
                mp_2d[23] = {"x": 0.58, "y": 1.15, "z": 0.0, "score": 0.05}
                mp_2d[24] = {"x": 0.42, "y": 1.15, "z": 0.0, "score": 0.05}
                mp_world = [{"x": 0.0, "y": 0.0, "z": 0.0, "score": 0.99} for _ in range(TOTAL_MEDIAPIPE_KEYPOINTS)]
                mp_world[11] = {"x": 0.19, "y": 0.0, "z": 0.0, "score": 0.99}
                mp_world[12] = {"x": -0.19, "y": 0.0, "z": 0.0, "score": 0.99}
                mp_world[14] = {"x": -0.19, "y": 0.0, "z": -0.28, "score": 0.99}
                mp_world[16] = {"x": -0.19, "y": 0.0, "z": -0.54, "score": 0.99}
                ws.send_json(
                    {
                        "type": "prescription",
                        "start_calibration": False,
                        "data": {
                            "exercise_id": "STROKE_SH_FLEX_R",
                            "exercise_name": "[Stroke ADL] ยกแขนขวาไปด้านหน้า (ฝึกเอื้อม/สอดแขนเสื้อ)",
                            "target_joint": "right_shoulder",
                            "target_rom_deg": 110,
                            "hold_sec": 3.0,
                            "target_reps": 10,
                        },
                    }
                )
                ws.send_json(
                    {
                        "type": "client_landmarks",
                        "person_count": 1,
                        "pose_landmarks": mp_2d,
                        "pose_world_landmarks": mp_world,
                    }
                )
                bridged = ws.receive_json()
                for _ in range(25):
                    if bridged.get("camera_mode") == "live_browser_bridge":
                        break
                    bridged = ws.receive_json()
                self.assertEqual(bridged.get("camera_mode"), "live_browser_bridge")
                self.assertGreater(bridged["telemetry_analysis"]["current_rom_deg"], 80.0)

    def test_17_live_mode_no_synthetic_fallback_on_missing_frame_or_camera_disconnect(self) -> None:
        """
        Verify live mode never falls back to procedural/synthetic skeleton coordinates
        when camera frames are absent (frame=None) or camera is disconnected, and returns
        explicit CAMERA_DISCONNECTED error metadata + zero/empty telemetry.
        """
        live_settings = EdgeVisionSettings(
            MOCK_VIDEO_PATH="",
            SUPABASE_URL="",
            SUPABASE_ANON_KEY="",
            SUPABASE_SERVICE_ROLE_KEY="",
        )
        estimator = RTMW3DPoseEstimator(live_settings)
        try:
            result = estimator.estimate_clinical_with_crowd(
                frame_bgr=None,
                timestamp_sec=1.0,
                is_mock_mode=False,
            )
            self.assertEqual(result.skeleton_count, 0)
            self.assertEqual(result.backend_used, "CAMERA_DISCONNECTED")
            self.assertEqual(result.pose_landmarks_2d.shape[0], 0)
            self.assertEqual(result.pose_world_landmarks_3d.shape[0], 0)
            self.assertTrue(bool(np.allclose(result.scores_133, 0.0)))
        finally:
            estimator.close()

        hand_zero = RTMW3DPoseEstimator.extract_hand_raise_telemetry(
            np.zeros((133, 3), dtype=np.float64),
            np.zeros((133,), dtype=np.float64),
        )
        self.assertFalse(hand_zero.left_raised)
        self.assertFalse(hand_zero.right_raised)

        fsm = ClinicalRehabFSM(live_settings)
        fsm.set_prescription(
            ExercisePrescriptionSchema(
                exercise_id="STROKE_SH_FLEX_R",
                exercise_name="[Stroke ADL] ยกแขนขวาไปด้านหน้า (ฝึกเอื้อม/สอดแขนเสื้อ)",
                target_joint="right_shoulder",
                target_rom_deg=145.0,
                hold_sec=3.0,
                target_reps=10,
            ),
            start_calibration=False,
        )
        payload = fsm.step(
            pts=np.zeros((133, 3), dtype=np.float64),
            scores=np.zeros((133,), dtype=np.float64),
            timestamp_sec=2.0,
            fps=30.0,
            camera_mode="camera_disconnected",
            hand_status=HandRaiseTelemetry(),
            keypoints_3d=[],
            pose_world_landmarks=[],
            pose_landmarks=[],
            error_code="CAMERA_DISCONNECTED",
            error_message="Camera unavailable",
        )
        self.assertEqual(payload.error_code, "CAMERA_DISCONNECTED")
        self.assertEqual(payload.error_message, "Camera unavailable")
        self.assertEqual(payload.pose_landmarks, [])
        self.assertEqual(payload.keypoints_3d, [])
        self.assertEqual(payload.telemetry_analysis.current_rom_deg, 0.0)
        self.assertEqual(payload.calibration.estimated_distance_m, 0.0)
        self.assertFalse(payload.hand_status.left_raised)
        self.assertFalse(payload.hand_status.right_raised)

        from app.api.ws_kiosk import KioskRuntimeCoordinator

        coordinator = KioskRuntimeCoordinator(live_settings)
        try:
            coordinator.camera._mode = "camera_disconnected"
            coord_payload = coordinator.step_once(include_frame=True)
            self.assertEqual(coord_payload.error_code, "CAMERA_DISCONNECTED")
            self.assertEqual(coord_payload.pose_landmarks, [])
            self.assertEqual(coord_payload.keypoints_3d, [])
            self.assertFalse(coord_payload.hand_status.left_raised)
            self.assertFalse(coord_payload.hand_status.right_raised)
        finally:
            coordinator.pose_estimator.close()

    def test_18_stroke_moph_exercises_and_safety_guards(self) -> None:
        """Stroke ADL & MOPH Elderly exercises (exact 26), 3D ROM calculation, and Safety Guards work accurately."""
        from app.biomechanics.exercises import list_all_exercises

        all_meta = list_all_exercises(include_stroke_moph=True)
        self.assertEqual(len(ALL_26_EXERCISES), 26)
        self.assertEqual(len(all_meta), 26)
        self.assertTrue(all(item["category"] in ("upper_limb", "neck_trunk") for item in all_meta))

        for code in (
            "STROKE_SH_FLEX",
            "STROKE_SH_ABD",
            "STROKE_CROSS_BODY",
            "STROKE_SCAP_PROTRACT",
            "STROKE_SH_ROT",
            "STROKE_BOBATH_ARM_RAISE",
            "STROKE_BOBATH_FORWARD_PUSH",
            "STROKE_ELB_FLEX_EXT",
            "STROKE_HAND_TO_HEAD",
            "STROKE_FOREARM_TURN",
            "STROKE_WRIST_FINGER",
            "STROKE_THUMB_PINCH",
            "MOPH_NECK_TILT",
            "MOPH_NECK_TURN_BILATERAL",
            "MOPH_SHOULDER_SHRUG_ROLL",
        ):
            resolved = resolve_exercise_rule(code)
            self.assertIsNotNone(resolved)

        solver = Biomechanics3DSolver(self.settings)
        cal = StandingCalibrator(duration_sec=0.2)
        stand_pts, scores = generate_synthetic_133_pose(0.0, is_calibrating=True)
        cal.force_calibrate_from_pose(stand_pts)

        # 1. Verify RTMW3D 133-pt hand landmarks drive STROKE_WRIST_FINGER_R and new Stroke/MOPH metrics
        wrist_rule = resolve_exercise_rule("STROKE_WRIST_FINGER")
        hand_pts = stand_pts.copy()
        hand_pts[10] = np.array([0.36, 0.48, -0.25], dtype=np.float64)
        hand_pts[112] = np.array([0.36, 0.48, -0.25], dtype=np.float64)
        hand_pts[124] = np.array([0.36, 0.36, -0.28], dtype=np.float64)
        wrist_res, wrist_emerg = solver.analyze(hand_pts, wrist_rule, cal, timestamp_sec=1.0)
        self.assertIsNone(wrist_emerg)
        self.assertGreater(wrist_res.current_rom_deg, 20.0)

        cross_pts = stand_pts.copy()
        cross_pts[10] = cross_pts[5] + np.array([0.02, 0.04, -0.05])
        cross_res, _ = solver.analyze(cross_pts, resolve_exercise_rule("STROKE_CROSS_BODY_R"), cal, timestamp_sec=1.2)
        self.assertGreater(cross_res.current_rom_deg, 65.0)

        neck_turn_pts = stand_pts.copy()
        neck_turn_pts[0] = np.array([0.54, 0.16, -0.04])
        neck_res, _ = solver.analyze(neck_turn_pts, resolve_exercise_rule("MOPH_NECK_TURN_BILATERAL"), cal, timestamp_sec=1.4)
        self.assertGreater(neck_res.current_rom_deg, 30.0)

        # 2. Verify CONTRAINDICATION_ROM_LIMIT + soft spasticity prompt
        solver.reset()
        flex_rule = resolve_exercise_rule("STROKE_SH_FLEX_R")
        over_pts, _ = generate_synthetic_133_pose(0.0, "STROKE_SH_ABD_R", mock_pose_override="raise_right")
        safe_res, safe_emerg = solver.analyze(
            over_pts,
            flex_rule,
            cal,
            timestamp_sec=2.0,
            target_rom_override=150.0,
            max_safe_rom_deg=95.0,
        )
        self.assertIsNone(safe_emerg)
        self.assertTrue(safe_res.contraindication_clamped)
        self.assertLessEqual(safe_res.target_rom_deg, 95.0)
        self.assertTrue(safe_res.spasticity_warning)
        self.assertIn("ช้าลงนิดนึงครับ", safe_res.safety_guard_banner_th or "")

        # 3. Verify POSTURAL_SWAY_AND_FALL_GUARD triggers EMERGENCY_HALT on trunk sway > 18 deg sustained 3 frames
        solver.reset()
        fsm = ClinicalRehabFSM(self.settings)
        fsm.set_prescription(
            ExercisePrescriptionSchema(
                exercise_id="STROKE_SH_FLEX_R",
                target_rom_deg=95.0,
                max_safe_rom_deg=95.0,
                hold_sec=2.0,
                target_reps=5,
            ),
            start_calibration=False,
        )
        fsm.calibrator.force_calibrate_from_pose(stand_pts)
        fsm.fsm_state = "ACTIVE"
        sway_pts = stand_pts.copy()
        sway_pts[5, 0] += 0.16
        sway_pts[6, 0] += 0.16
        hand_empty = HandRaiseTelemetry()
        p_sway = None
        for idx in range(3):
            p_sway = fsm.step(sway_pts, scores, 3.0 + idx * 0.25, 30.0, "synthetic_mock", hand_empty, [])
        self.assertIsNotNone(p_sway)
        self.assertEqual(p_sway.fsm_state, "EMERGENCY")
        self.assertTrue(p_sway.telemetry_analysis.pain_alert)
        self.assertEqual(p_sway.telemetry_analysis.safety_guard_alert, "POSTURAL_SWAY_AND_FALL_GUARD")
        self.assertIn("ตรวจพบการเสียสมดุล", p_sway.tts_voice_th)

    def test_19_face_matcher_strict_margin_pose_sync_and_exclusion(self) -> None:
        """Verify Edge Face 1:N matcher enforces >= 0.08 margin, pose-synchronized matching, and excluded IDs."""
        rng = np.random.default_rng(777)
        mgr = InsightFace120Manager(self.settings)
        base_a = l2_normalize(rng.normal(size=512))
        close_b = l2_normalize(0.96 * base_a + 0.04 * l2_normalize(rng.normal(size=512)))

        mgr._patients = [
            CachedPatientBiometric(
                patient_id="p-edge-a",
                hn="HN-E1",
                full_name="Patient Edge A",
                master_embedding=base_a,
                angle_embeddings={"center": base_a},
            ),
            CachedPatientBiometric(
                patient_id="p-edge-b",
                hn="HN-E2",
                full_name="Patient Edge B",
                master_embedding=close_b,
                angle_embeddings={"center": close_b},
            ),
        ]
        mgr._matrix = np.vstack([p.master_embedding for p in mgr._patients])

        # Close candidates within < 0.08 margin must be rejected (id is None)
        amb_match = mgr.match_1_to_n(base_a, yaw_deg=0.0, pitch_deg=0.0)
        self.assertIsNone(amb_match.id)

        # Excluding p-edge-b restores unambiguous match to p-edge-a
        clean_match = mgr.match_1_to_n(base_a, yaw_deg=0.0, pitch_deg=0.0, excluded_patient_ids=["p-edge-b"])
        self.assertEqual(clean_match.id, "p-edge-a")

    def test_20_stroke_biomechanics_upgrade_baseline_hand_and_hysteresis(self) -> None:
        """Verify 1.15/0.06 One-Euro config, auto bone baseline, ascending elbow, NECK_DROP, SHOULDER_HIKE >14%, and <45% hysteresis."""
        # 1. Config defaults match Stroke biomechanical specification
        self.assertAlmostEqual(self.settings.ONE_EURO_MIN_CUTOFF, 1.15, places=2)
        self.assertAlmostEqual(self.settings.ONE_EURO_BETA, 0.06, places=2)
        self.assertAlmostEqual(self.settings.ONE_EURO_2D_MIN_CUTOFF, 1.15, places=2)
        self.assertAlmostEqual(self.settings.ONE_EURO_2D_BETA, 0.06, places=2)
        self.assertAlmostEqual(self.settings.TRUNK_LEAN_MAX_DEG, 12.0, places=2)

        # 2. Auto bone baseline captures anatomical minimums and resting pose deltas
        cal = StandingCalibrator(duration_sec=0.2)
        stand_pts, scores = generate_synthetic_133_pose(0.0, is_calibrating=True)
        bones = cal.force_calibrate_from_pose(stand_pts)
        self.assertGreaterEqual(bones.upper_arm_len, bones.shoulder_width * 0.82 - 1e-5)
        self.assertAlmostEqual(bones.resting_shoulder_delta_y, float(stand_pts[5, 1] - stand_pts[6, 1]), places=4)

        # 3. Ascending STROKE_ELB_FLEX_EXT_R (0 -> 135 deg) & strict <45% targetRom rest hysteresis
        elb_rule = resolve_exercise_rule("STROKE_ELB_FLEX_EXT_R")
        self.assertFalse(elb_rule.is_descending_target)
        self.assertFalse(elb_rule.is_in_rest_zone(50.0, target_override=100.0))
        self.assertTrue(elb_rule.is_in_rest_zone(42.0, target_override=100.0))

        solver = Biomechanics3DSolver(self.settings)
        elb_rest_res, _ = solver.analyze(stand_pts, elb_rule, cal, timestamp_sec=1.0, scores=scores)
        self.assertLess(elb_rest_res.current_rom_deg, 25.0)

        elb_flex_pts = stand_pts.copy()
        elb_flex_pts[8] = elb_flex_pts[6] + np.array([0.0, 0.16, -0.02])
        elb_flex_pts[10] = elb_flex_pts[6] + np.array([0.0, 0.03, -0.06])
        elb_flex_res, _ = solver.analyze(elb_flex_pts, elb_rule, cal, timestamp_sec=2.2, scores=scores)
        self.assertGreater(elb_flex_res.current_rom_deg, 95.0)

        # 4. Hand-to-head detects NECK_DROP > 15 deg with exact Thai correction prompt
        solver.reset()
        hth_rule = resolve_exercise_rule("STROKE_HAND_TO_HEAD_R")
        drop_pts = stand_pts.copy()
        drop_pts[0] = drop_pts[3] * 0.5 + drop_pts[4] * 0.5 + np.array([0.0, 0.09, -0.05])
        drop_pts[10] = drop_pts[0] + np.array([-0.03, -0.02, -0.04])
        drop_res, _ = solver.analyze(drop_pts, hth_rule, cal, timestamp_sec=3.0, scores=scores)
        self.assertIn("NECK_DROP", drop_res.compensation_detected)
        self.assertIn("มองตรงข้างหน้าครับ", " ".join(drop_res.compensation_warnings))

        # 5. MOPH Shoulder Shrug Roll never triggers SHOULDER_HIKE anti-cheat banner
        shrug_rule = resolve_exercise_rule("MOPH_SHOULDER_SHRUG_ROLL")
        shrug_pts = stand_pts.copy()
        shrug_pts[5, 1] -= 0.04
        shrug_pts[6, 1] -= 0.04
        shrug_res, _ = solver.analyze(shrug_pts, shrug_rule, cal, timestamp_sec=4.0, scores=scores)
        self.assertNotIn("SHOULDER_HIKE", shrug_res.compensation_detected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
