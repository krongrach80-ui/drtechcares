import React, { useEffect, useRef, useState } from 'react';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import type { DrTechCaresApiClient, PatientRecord, UserAccount } from '@drtechcares/api-client';
import {
  FACE_ID_SCREEN_PROMPTS,
  TEN_SHOT_SEQUENCE,
  TenShotEnrollmentController,
  alignFaceTo112,
  evaluateLivenessAndPose,
  inspectBiometricFrame3D,
  resolveFaceIdScreenPrompt,
  selectClinicalCameraDeviceId,
  type BiometricVerificationJson,
} from '@drtechcares/face-client';
import { evaluateFrameQuality } from '@drtechcares/smart-biometric-core';
import {
  BIOMETRIC_ENGINE_ID,
  captureCanonicalFaceSnapshot,
  extractCanonicalFaceDescriptor,
  syncPatientBiometricEverywhere,
} from '../../../src/lib/biometricEngine';
import { Camera, CheckCircle2, UserPlus } from 'lucide-react';
import { FaceIdCircularOverlay } from '../../../src/components/FaceIdCircularOverlay';

interface SmartEnrollmentViewProps {
  apiClient: DrTechCaresApiClient;
  currentUserRole: 'super_admin' | 'pt';
  ptList?: UserAccount[];
  onEnrolled: (patient: PatientRecord) => void;
}

const POSE_PROMPT: Record<'front' | 'left' | 'right' | 'nod', string> = {
  front: 'มองตรงเข้ากล้อง (หน้าตรง 4 ช็อต)',
  left: 'หันหน้าไปทางซ้ายเล็กน้อย (2 ช็อต)',
  right: 'หันหน้าไปทางขวาเล็กน้อย (2 ช็อต)',
  nod: 'พยักหน้าก้มลงเล็กน้อย (2 ช็อต)',
};

export function SmartEnrollmentView({
  apiClient,
  currentUserRole,
  ptList = [],
  onEnrolled,
}: SmartEnrollmentViewProps) {
  const [hn, setHn] = useState(`HN-${Math.floor(100000 + Math.random() * 899999)}`);
  const [fullName, setFullName] = useState('');
  const [age, setAge] = useState(64);
  const [phone, setPhone] = useState('');
  const [diagnosis, setDiagnosis] = useState('ฟื้นฟูข้อไหล่และสมอง');
  const [assignedPtId, setAssignedPtId] = useState(ptList[0]?.id ?? '');
  const [nextAppointmentDate, setNextAppointmentDate] = useState(
    new Date(Date.now() + 2 * 24 * 60 * 60 * 1000).toISOString().slice(0, 10),
  );
  const [portalUsername, setPortalUsername] = useState('');
  const [portalPassword, setPortalPassword] = useState('Patient@2026');
  const [shotsCount, setShotsCount] = useState(0);
  const [capturing, setCapturing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [statusNote, setStatusNote] = useState('กด "เปิดกล้องสแกนใบหน้า 10 เฟรม" เพื่อเริ่มเก็บข้อมูลจริง');
  const [liveInspection, setLiveInspection] = useState<BiometricVerificationJson | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const enrollRef = useRef(new TenShotEnrollmentController());

  const expectedPose = enrollRef.current.expectedPose;

  useEffect(() => {
    if (!capturing) return;
    let active = true;
    let stream: MediaStream | null = null;
    let rafId = 0;
    let landmarker: FaceLandmarker | null = null;
    let prevLandmarks: { x: number; y: number; z?: number }[] | null = null;
    let lastShotAt = 0;

    async function runCapture() {
      try {
        const deviceId = await selectClinicalCameraDeviceId();
        stream = await navigator.mediaDevices.getUserMedia({
          video: deviceId ? { deviceId: { ideal: deviceId }, width: 640, height: 480 } : { width: 640, height: 480 },
          audio: false,
        });
        if (!active || !videoRef.current) {
          stream.getTracks().forEach((tr) => tr.stop());
          stream = null;
          return;
        }
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        if (!active) {
          stream.getTracks().forEach((tr) => tr.stop());
          stream = null;
          return;
        }

        const vision = await FilesetResolver.forVisionTasks('/mediapipe/wasm');
        if (!active) return;
        const createdLm = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: '/models/face_landmarker.task' },
          runningMode: 'VIDEO',
          numFaces: 1,
        });
        if (!active) {
          createdLm.close();
          return;
        }
        landmarker = createdLm;

        const tick = async () => {
          if (!active || !videoRef.current || !landmarker || enrollRef.current.isComplete) return;
          if (videoRef.current.readyState >= 2) {
            const now = performance.now();
            const res = landmarker.detectForVideo(videoRef.current, now);
            const faceCount = res.faceLandmarks?.length ?? 0;
            const landmarks = res.faceLandmarks?.[0] ?? [];
            const priorLandmarks = prevLandmarks;
            const liveness = evaluateLivenessAndPose(landmarks, priorLandmarks, faceCount);

            if (!canvasRef.current) canvasRef.current = document.createElement('canvas');
            const crop =
              landmarks.length >= 300 ? alignFaceTo112(videoRef.current, landmarks, canvasRef.current) : null;

            const poseNeeded = enrollRef.current.expectedPose;
            const requested3DPose = enrollRef.current.expected3DPose;
            const inspection = inspectBiometricFrame3D({
              landmarks3D: landmarks,
              prevLandmarks3D: priorLandmarks,
              faceCount,
              image: crop,
              requestedPose: requested3DPose,
            });
            prevLandmarks = landmarks;
            setLiveInspection(inspection);

            if (inspection.scan_status === 'SPOOF_DETECTED') {
              setStatusNote(inspection.user_instruction_th);
            } else if (!liveness.isLive || !inspection.checks.framing_ok || !inspection.checks.lighting_ok || !inspection.checks.no_obstruction) {
              setStatusNote(inspection.user_instruction_th);
            } else if (poseNeeded && now - lastShotAt > 600) {
              const poseOk =
                (poseNeeded === 'front' && Math.abs(liveness.yawDeg) <= 15 && Math.abs(liveness.pitchDeg) <= 15) ||
                (poseNeeded === 'left' && liveness.yawDeg <= -10) ||
                (poseNeeded === 'right' && liveness.yawDeg >= 10) ||
                (poseNeeded === 'nod' && (Math.abs(liveness.pitchDeg) >= 8 || inspection.detected_pose === 'TILT_UP' || inspection.detected_pose === 'TILT_DOWN'));

              setStatusNote(poseOk ? `กำลังบันทึกช็อต 3D (${poseNeeded})...` : inspection.user_instruction_th || POSE_PROMPT[poseNeeded]);

              if (poseOk && crop) {
                const q = evaluateFrameQuality(crop, liveness, 35, 35);
                if (q.sharpness >= 4.0) {
                  const snap = captureCanonicalFaceSnapshot(videoRef.current, null, undefined, landmarks);
                  const extracted = snap ? await extractCanonicalFaceDescriptor(snap) : null;
                  if (
                    extracted?.embedding &&
                    enrollRef.current.acceptShot(poseNeeded, extracted.embedding, q.sharpness)
                  ) {
                    lastShotAt = now;
                    setShotsCount(enrollRef.current.currentStepIndex);
                  }
                } else {
                  setStatusNote('ภาพไม่คมชัด กรุณาหยุดนิ่งสักครู่');
                }
              }
            }
          }
          rafId = requestAnimationFrame(tick);
        };
        rafId = requestAnimationFrame(tick);
      } catch (err) {
        setErrorMsg(err instanceof Error ? err.message : 'ไม่สามารถเปิดกล้องเว็บแคมได้');
        setCapturing(false);
      }
    }

    runCapture();
    return () => {
      active = false;
      cancelAnimationFrame(rafId);
      landmarker?.close();
      stream?.getTracks().forEach((tr) => tr.stop());
    };
  }, [capturing]);

  const handleSaveEnrollment = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!fullName.trim()) {
      setErrorMsg('กรุณากรอกชื่อ-นามสกุลผู้ป่วย');
      return;
    }
    if (!enrollRef.current.isComplete) {
      setErrorMsg(`กรุณาสแกนใบหน้าจริงให้ครบ 10 เฟรมก่อนบันทึก (${shotsCount}/10)`);
      return;
    }
    setSaving(true);
    setErrorMsg(null);
    try {
      const payload = enrollRef.current.buildPayload();
      const exemplarEmbeddings = payload.raw_frames.map((r) => r.vector);
      const centerAngle = payload.frontal_centroid;
      const patient = await apiClient.enrollPatient10Shot({
        hn,
        fullName: fullName.trim(),
        age,
        phone,
        diagnosis,
        assignedPtId: currentUserRole === 'super_admin' && assignedPtId ? assignedPtId : undefined,
        nextAppointmentDate,
        username: portalUsername.trim() || undefined,
        password: portalPassword,
        frontalCentroid: payload.frontal_centroid,
        overallCentroid: payload.overall_centroid,
        masterEmbedding: payload.overall_centroid,
        angleEmbeddings: {
          center: centerAngle,
          left: payload.raw_frames.find((r) => r.pose === 'left')?.vector ?? payload.overall_centroid,
          right: payload.raw_frames.find((r) => r.pose === 'right')?.vector ?? payload.overall_centroid,
          up: payload.raw_frames.find((r) => r.pose === 'nod')?.vector ?? payload.overall_centroid,
          down: payload.raw_frames.find((r) => r.pose === 'nod')?.vector ?? payload.overall_centroid,
        },
        exemplarEmbeddings,
        biometricEngine: BIOMETRIC_ENGINE_ID,
        rawFrames: payload.raw_frames.map((r) => ({
          pose: r.pose,
          vector: r.vector,
          qualityScore: r.qualityScore,
        })),
      });
      await syncPatientBiometricEverywhere({
        apiClient,
        patient,
        masterEmbedding: payload.overall_centroid,
        angleEmbeddings: {
          center: centerAngle,
          left: payload.raw_frames.find((r) => r.pose === 'left')?.vector ?? payload.overall_centroid,
          right: payload.raw_frames.find((r) => r.pose === 'right')?.vector ?? payload.overall_centroid,
          up: payload.raw_frames.find((r) => r.pose === 'nod')?.vector ?? payload.overall_centroid,
          down: payload.raw_frames.find((r) => r.pose === 'nod')?.vector ?? payload.overall_centroid,
        },
        exemplarEmbeddings,
        framesCaptured: payload.raw_frames.length,
        qualityScore: 0.95,
        requireSupabase: Boolean(apiClient.getSupabaseClient()),
      });
      enrollRef.current.reset();
      setShotsCount(0);
      setCapturing(false);
      setFullName('');
      setPhone('');
      setPortalUsername('');
      setHn(`HN-${Math.floor(100000 + Math.random() * 899999)}`);
      onEnrolled(patient);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'บันทึกข้อมูลผู้ป่วยไม่สำเร็จ');
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={handleSaveEnrollment} className="grid grid-cols-1 gap-6 lg:grid-cols-2 animate-view-enter">
      <div className="rounded-3xl bg-white p-6 shadow-md border border-slate-200 space-y-4 card-hover-lift animate-card-enter">
        <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2">
          <UserPlus className="h-5 w-5 text-teal-600" />
          ลงทะเบียนคนไข้ใหม่ + สแกนใบหน้า 10 เฟรม (กล้องจริงเท่านั้น)
        </h2>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-xs font-semibold text-slate-600">Hospital Number (HN)</label>
            <input
              value={hn}
              onChange={(e) => setHn(e.target.value)}
              className="mt-1 w-full rounded-xl border border-slate-300 px-3 py-2 text-sm"
              required
            />
          </div>
          <div>
            <label className="text-xs font-semibold text-slate-600">อายุ (ปี)</label>
            <input
              type="number"
              inputMode="numeric"
              value={age}
              onKeyDown={(e) => {
                if (['e', 'E', '+', '-', '.'].includes(e.key)) e.preventDefault();
              }}
              onChange={(e) => {
                const digits = e.target.value.replace(/\D/g, '').slice(0, 3);
                setAge(digits ? Math.min(120, Number(digits)) : 0);
              }}
              className="mt-1 w-full rounded-xl border border-slate-300 px-3 py-2 text-sm"
              min={1}
              max={120}
              required
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-xs font-semibold text-slate-600">ชื่อ-นามสกุลผู้ป่วย</label>
            <input
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              placeholder="เช่น คุณสมหญิง รักสุขภาพ"
              className="mt-1 w-full rounded-xl border border-slate-300 px-3 py-2 text-sm"
              required
            />
          </div>
          <div>
            <label className="text-xs font-semibold text-slate-600">เบอร์โทรศัพท์</label>
            <input
              type="tel"
              inputMode="numeric"
              pattern="[0-9]*"
              maxLength={10}
              value={phone}
              onChange={(e) => setPhone(e.target.value.replace(/\D/g, '').slice(0, 10))}
              placeholder="08xxxxxxxx"
              className="mt-1 w-full rounded-xl border border-slate-300 px-3 py-2 text-sm"
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-xs font-semibold text-slate-600">การวินิจฉัย / อาการ</label>
            <input
              value={diagnosis}
              onChange={(e) => setDiagnosis(e.target.value)}
              className="mt-1 w-full rounded-xl border border-slate-300 px-3 py-2 text-sm"
            />
          </div>
          <div>
            <label className="text-xs font-semibold text-slate-600">วันนัดหมายครั้งถัดไป</label>
            <input
              type="date"
              value={nextAppointmentDate}
              onChange={(e) => setNextAppointmentDate(e.target.value)}
              className="mt-1 w-full rounded-xl border border-slate-300 px-3 py-2 text-sm"
            />
          </div>
        </div>

        {currentUserRole === 'super_admin' && ptList.length > 0 && (
          <div>
            <label className="text-xs font-semibold text-slate-600">นักกายภาพบำบัดผู้รับผิดชอบ</label>
            <select
              value={assignedPtId}
              onChange={(e) => setAssignedPtId(e.target.value)}
              className="mt-1 w-full rounded-xl border border-slate-300 px-3 py-2 text-sm"
            >
              {ptList.map((pt) => (
                <option key={pt.id} value={pt.id}>
                  {pt.fullName}
                </option>
              ))}
            </select>
          </div>
        )}

        <div className="grid grid-cols-2 gap-3 rounded-2xl bg-slate-50 p-3 border border-slate-200">
          <div>
            <label className="text-xs font-semibold text-slate-600">ชื่อผู้ใช้ Patient Portal (ไม่ระบุใช้ HN)</label>
            <input
              value={portalUsername}
              onChange={(e) => setPortalUsername(e.target.value)}
              placeholder={hn.replace(/[^a-zA-Z0-9]/g, '').toLowerCase()}
              className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-xs"
            />
          </div>
          <div>
            <label className="text-xs font-semibold text-slate-600">รหัสผ่านเริ่มต้น Patient Portal</label>
            <input
              type="text"
              value={portalPassword}
              onChange={(e) => setPortalPassword(e.target.value)}
              className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-xs"
              required
            />
          </div>
        </div>

        <div className="rounded-2xl bg-slate-50 p-4 border border-slate-200">
          <div className="flex items-center justify-between text-xs font-bold text-slate-700">
            <span>ความคืบหน้าสแกนใบหน้า 10 เฟรม (หน้าตรง 4, ซ้าย 2, ขวา 2, พยักหน้า 2)</span>
            <span>{shotsCount} / 10</span>
          </div>
          <div className="mt-2 grid grid-cols-10 gap-1.5">
            {TEN_SHOT_SEQUENCE.map((pose, idx) => (
              <div
                key={`${pose}-${idx}`}
                className={`rounded-lg py-1.5 text-center text-[10px] font-bold uppercase ${
                  idx < shotsCount
                    ? 'bg-emerald-500 text-white'
                    : idx === shotsCount
                      ? 'bg-teal-600 text-white animate-pulse'
                      : 'bg-slate-200 text-slate-500'
                }`}
              >
                {pose}
              </div>
            ))}
          </div>
        </div>

        {errorMsg && (
          <div className="rounded-xl bg-rose-50 border border-rose-200 px-3 py-2 text-xs font-semibold text-rose-700">
            {errorMsg}
          </div>
        )}

        <button
          type="submit"
          disabled={saving || shotsCount < 10}
          className="w-full rounded-2xl bg-teal-600 py-3 text-sm font-bold text-white hover:bg-teal-700 disabled:opacity-50"
        >
          {saving ? 'กำลังบันทึกข้อมูลลงฐานข้อมูล...' : 'บันทึกผู้ป่วยและเวกเตอร์ใบหน้า 10 เฟรม'}
        </button>
      </div>

      {/* Real Camera Capture Panel with iPhone Face ID Circular Overlay */}
      <div className="rounded-3xl bg-slate-950 p-4 text-white shadow-md flex flex-col justify-between gap-3">
        <div className="relative overflow-hidden rounded-2xl bg-slate-900 aspect-[4/3]">
          <video ref={videoRef} playsInline muted className="h-full w-full object-cover -scale-x-100" />
          <FaceIdCircularOverlay
            progressRatio={shotsCount / 10}
            inspection={liveInspection}
            screenPrompt={resolveFaceIdScreenPrompt(liveInspection, enrollRef.current.scanStage)}
            compact
          />
          {enrollRef.current.isComplete && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-emerald-950/85 text-center p-4">
              <CheckCircle2 className="h-14 w-14 text-emerald-400" />
              <p className="mt-2 text-lg font-extrabold">{FACE_ID_SCREEN_PROMPTS[6].primaryPrompt}</p>
              <p className="mt-1 text-xs text-emerald-200">{FACE_ID_SCREEN_PROMPTS[6].subPrompt}</p>
            </div>
          )}
        </div>

        <div className="rounded-xl bg-slate-900 border border-slate-800 p-3 text-[11px] font-mono text-emerald-300 overflow-x-auto">
          <div className="text-[10px] font-sans font-bold text-slate-400 mb-1 uppercase tracking-wider">
            3D Biometric Assistant Output (Strict JSON) — {statusNote}
          </div>
          <pre className="whitespace-pre-wrap leading-relaxed">
            {JSON.stringify(
              liveInspection ?? {
                scan_status: 'RETRY',
                confidence_score: 0.0,
                detected_pose: 'UNKNOWN',
                checks: {
                  framing_ok: false,
                  lighting_ok: false,
                  no_obstruction: false,
                  liveness_ok: false,
                },
                error_code: 'TOO_FAR',
                user_instruction_th: 'วางใบหน้าของคุณให้อยู่ในกรอบ',
              },
              null,
              2,
            )}
          </pre>
        </div>

        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => setCapturing((v) => !v)}
            className="flex-1 rounded-xl bg-teal-600 py-2.5 text-xs font-bold text-white hover:bg-teal-500"
          >
            <Camera className="inline h-4 w-4 mr-1" />
            {capturing ? 'หยุดกล้องชั่วคราว' : 'เปิดกล้องสแกนใบหน้า 10 เฟรม'}
          </button>
          <button
            type="button"
            onClick={() => {
              enrollRef.current.reset();
              setShotsCount(0);
              setErrorMsg(null);
            }}
            className="rounded-xl border border-slate-700 px-4 py-2.5 text-xs font-semibold text-slate-300 hover:bg-slate-800"
          >
            เริ่มสแกนใหม่
          </button>
        </div>
      </div>
    </form>
  );
}
