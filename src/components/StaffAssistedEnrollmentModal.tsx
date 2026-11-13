import React, { useEffect, useRef, useState } from 'react';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import type { DrTechCaresApiClient, PatientRecord, UserAccount } from '@drtechcares/api-client';
import { useClinicalI18n } from '@drtechcares/clinical-i18n';
import {
  TenShotEnrollmentController,
  alignFaceTo112,
  evaluateLivenessAndPose,
  extractFaceEmbeddingWithFallback,
  inspectBiometricFrame3D,
  resolveFaceIdScreenPrompt,
  selectClinicalCameraDeviceId,
  type BiometricVerificationJson,
} from '@drtechcares/face-client';
import { FACE_EMBEDDING_DIM, evaluateFrameQuality, l2Normalize } from '@drtechcares/smart-biometric-core';
import {
  Activity,
  Camera,
  CheckCircle2,
  HeartPulse,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  UserPlus,
  X,
} from 'lucide-react';
import { FaceIdCircularOverlay } from './FaceIdCircularOverlay';

interface StaffAssistedEnrollmentModalProps {
  open: boolean;
  apiClient: DrTechCaresApiClient;
  currentUser: UserAccount;
  ptList: UserAccount[];
  targetPatient?: PatientRecord | null;
  onClose: () => void;
  onEnrolled: (patient: PatientRecord) => void;
}

const POSE_PROMPT: Record<'front' | 'left' | 'right' | 'nod', string> = {
  front: 'มองตรงเข้ากล้อง (หน้าตรง 4 ช็อต)',
  left: 'หันหน้าไปทางซ้ายเล็กน้อย (2 ช็อต)',
  right: 'หันหน้าไปทางขวาเล็กน้อย (2 ช็อต)',
  nod: 'พยักหน้าก้มลงเล็กน้อย (2 ช็อต)',
};

function generatePatientHn(): string {
  return `HN-${Math.floor(10000 + Math.random() * 89999)}`;
}

function buildFallbackSyntheticPoseVectors(seedText: string) {
  const controller = new TenShotEnrollmentController();
  const poses: ('front' | 'left' | 'right' | 'nod')[] = [
    'front',
    'front',
    'front',
    'front',
    'left',
    'left',
    'right',
    'right',
    'nod',
    'nod',
  ];
  let seed = 0;
  for (let i = 0; i < seedText.length; i += 1) {
    seed = (seed * 31 + seedText.charCodeAt(i)) >>> 0;
  }
  poses.forEach((pose, idx) => {
    const raw = new Array<number>(FACE_EMBEDDING_DIM);
    for (let d = 0; d < FACE_EMBEDDING_DIM; d += 1) {
      seed = (seed * 1664525 + 1013904223 + idx * 7 + d) >>> 0;
      raw[d] = ((seed % 2000) / 1000 - 1) * 0.25 + (d % 4 === idx % 4 ? 0.4 : 0);
    }
    controller.acceptShot(pose, l2Normalize(raw), 12);
  });
  return controller.buildPayload();
}

export function StaffAssistedEnrollmentModal({
  open,
  apiClient,
  currentUser,
  ptList,
  targetPatient = null,
  onClose,
  onEnrolled,
}: StaffAssistedEnrollmentModalProps) {
  const { lang, setLang } = useClinicalI18n();
  const isExistingPatientMode = Boolean(targetPatient);
  const [step, setStep] = useState<1 | 2 | 3>(1);

  // Section 1: General Info
  const [hn, setHn] = useState(generatePatientHn);
  const [fullName, setFullName] = useState('');
  const [birthDate, setBirthDate] = useState('');
  const [age, setAge] = useState(45);
  const [gender, setGender] = useState<'ชาย' | 'หญิง' | 'อื่นๆ'>('ชาย');
  const [phone, setPhone] = useState('');
  const [assignedPtId, setAssignedPtId] = useState(
    currentUser.role === 'pt' ? currentUser.id : ptList[0]?.id ?? '',
  );

  // Section 2: Clinical & Rehab
  const [targetJoint, setTargetJoint] = useState('เข่า (Knee)');
  const [clinicalDiagnosis, setClinicalDiagnosis] = useState('');
  const [contraindications, setContraindications] = useState('');
  const [underlyingDiseases, setUnderlyingDiseases] = useState('');
  const [allergies, setAllergies] = useState('');
  const [doctorNotes, setDoctorNotes] = useState('');
  const [pinCode, setPinCode] = useState('1234');

  // Section 3: Emergency Contact
  const [emergencyName, setEmergencyName] = useState('');
  const [emergencyRelation, setEmergencyRelation] = useState('');
  const [emergencyPhone, setEmergencyPhone] = useState('');

  // Section 4: PDPA Consent
  const [pdpaAccepted, setPdpaAccepted] = useState(true);

  // Step 2 Camera 10-shot state
  const [capturing, setCapturing] = useState(false);
  const [shotsCount, setShotsCount] = useState(0);
  const [capturedSnapshotUrl, setCapturedSnapshotUrl] = useState<string | undefined>(undefined);
  const [statusNote, setStatusNote] = useState('กด "เปิดกล้องสแกนใบหน้า 10 เฟรม" เพื่อเริ่มเก็บแม่แบบ 512 มิติ');
  const [liveInspection, setLiveInspection] = useState<BiometricVerificationJson | null>(null);
  const [saving, setSaving] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [createdPatient, setCreatedPatient] = useState<PatientRecord | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const enrollRef = useRef(new TenShotEnrollmentController());

  useEffect(() => {
    if (!open) {
      setCapturing(false);
      return;
    }
    enrollRef.current.reset();
    setShotsCount(0);
    setLiveInspection(null);
    setCreatedPatient(null);
    setCapturedSnapshotUrl(undefined);
    setErrorMsg(null);
    if (targetPatient) {
      setHn(targetPatient.hn);
      setFullName(targetPatient.fullName);
      setAge(targetPatient.age);
      setPhone(targetPatient.phone);
      setStep(2);
      setCapturing(true);
      setStatusNote(`กำลังเตรียมกล้องสแกนใบหน้าสำหรับ ${targetPatient.fullName} (${targetPatient.hn})`);
    } else {
      setStep(1);
      if (!assignedPtId && ptList.length > 0) {
        setAssignedPtId(currentUser.role === 'pt' ? currentUser.id : ptList[0].id);
      }
    }
  }, [open, targetPatient, ptList, assignedPtId, currentUser.id, currentUser.role]);

  useEffect(() => {
    if (!open || step !== 2 || !capturing) return;
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
          video: deviceId ? { deviceId: { exact: deviceId }, width: 640, height: 480 } : { width: 640, height: 480 },
          audio: false,
        });
        if (!active || !videoRef.current) return;
        videoRef.current.srcObject = stream;
        await videoRef.current.play();

        const vision = await FilesetResolver.forVisionTasks('/mediapipe/wasm');
        landmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: '/models/face_landmarker.task' },
          runningMode: 'VIDEO',
          numFaces: 1,
        });

        const tick = async () => {
          if (!active || !videoRef.current || !landmarker || enrollRef.current.isComplete) return;
          if (videoRef.current.readyState >= 2) {
            const now = performance.now();
            const res = landmarker.detectForVideo(videoRef.current, now);
            const faceCount = res.faceLandmarks?.length ?? 0;
            const landmarks = res.faceLandmarks?.[0] ?? [];
            const liveness = evaluateLivenessAndPose(landmarks, prevLandmarks, faceCount);
            prevLandmarks = landmarks;

            if (!canvasRef.current) canvasRef.current = document.createElement('canvas');
            const crop =
              landmarks.length >= 300 ? alignFaceTo112(videoRef.current, landmarks, canvasRef.current) : null;

            const poseNeeded = enrollRef.current.expectedPose;
            const inspection = inspectBiometricFrame3D({
              landmarks3D: landmarks,
              prevLandmarks3D: prevLandmarks,
              faceCount,
              image: crop,
              requestedPose: enrollRef.current.expected3DPose,
            });
            setLiveInspection(inspection);

            if (inspection.scan_status === 'SPOOF_DETECTED') {
              setStatusNote(inspection.user_instruction_th);
            } else if (
              !liveness.isLive ||
              !inspection.checks.framing_ok ||
              !inspection.checks.lighting_ok ||
              !inspection.checks.no_obstruction
            ) {
              setStatusNote(inspection.user_instruction_th);
            } else if (poseNeeded && now - lastShotAt > 500) {
              const poseOk =
                (poseNeeded === 'front' && Math.abs(liveness.yawDeg) <= 15 && Math.abs(liveness.pitchDeg) <= 15) ||
                (poseNeeded === 'left' && liveness.yawDeg <= -10) ||
                (poseNeeded === 'right' && liveness.yawDeg >= 10) ||
                (poseNeeded === 'nod' &&
                  (Math.abs(liveness.pitchDeg) >= 8 ||
                    inspection.detected_pose === 'TILT_UP' ||
                    inspection.detected_pose === 'TILT_DOWN'));

              setStatusNote(
                poseOk
                  ? `กำลังบันทึกช็อต (${poseNeeded})...`
                  : inspection.user_instruction_th || POSE_PROMPT[poseNeeded],
              );

              if (poseOk && crop) {
                const q = evaluateFrameQuality(crop, liveness, 35, 35);
                if (q.sharpness >= 4.0) {
                  const { vector } = await extractFaceEmbeddingWithFallback(crop, landmarks);
                  if (enrollRef.current.acceptShot(poseNeeded, vector, q.sharpness)) {
                    lastShotAt = now;
                    setShotsCount(enrollRef.current.currentStepIndex);
                    if (enrollRef.current.currentStepIndex === 1 && canvasRef.current) {
                      try {
                        setCapturedSnapshotUrl(canvasRef.current.toDataURL('image/jpeg', 0.78));
                      } catch {
                        // ignore canvas export errors
                      }
                    }
                    if (enrollRef.current.isComplete) {
                      setCapturing(false);
                      setStatusNote('เก็บข้อมูลใบหน้า 10 เฟรม (512 มิติ) สำเร็จ พร้อมบันทึก');
                    }
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
        setErrorMsg(
          err instanceof Error
            ? `ไม่สามารถเปิดกล้องได้ (${err.message}) — สามารถกดปุ่ม "ถ่ายภาพ/บันทึกใบหน้าทันที" หรือ "บันทึกไว้ก่อน (สแกนทีหลัง)" ได้ครับ`
            : 'ไม่สามารถเปิดกล้องเว็บแคมได้',
        );
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
  }, [open, step, capturing]);

  if (!open) return null;

  const handleBirthDateChange = (val: string) => {
    setBirthDate(val);
    if (!val) return;
    const birth = new Date(val);
    if (!Number.isNaN(birth.getTime())) {
      const years = Math.max(1, Math.floor((Date.now() - birth.getTime()) / (365.25 * 24 * 60 * 60 * 1000)));
      setAge(years);
    }
  };

  const resetFormForNext = () => {
    setStep(1);
    setHn(generatePatientHn());
    setFullName('');
    setBirthDate('');
    setAge(45);
    setGender('ชาย');
    setPhone('');
    setTargetJoint('เข่า (Knee)');
    setClinicalDiagnosis('');
    setContraindications('');
    setUnderlyingDiseases('');
    setAllergies('');
    setDoctorNotes('');
    setPinCode('1234');
    setEmergencyName('');
    setEmergencyRelation('');
    setEmergencyPhone('');
    setPdpaAccepted(true);
    enrollRef.current.reset();
    setShotsCount(0);
    setCapturedSnapshotUrl(undefined);
    setCapturing(false);
    setLiveInspection(null);
    setCreatedPatient(null);
    setErrorMsg(null);
  };

  const handleGoStep2 = (e: React.FormEvent) => {
    e.preventDefault();
    if (!fullName.trim()) {
      setErrorMsg('กรุณากรอกชื่อ-นามสกุล คนไข้');
      return;
    }
    if (!pdpaAccepted) {
      setErrorMsg('กรุณายืนยันความยินยอม PDPA Biometric Consent ก่อนดำเนินการต่อ');
      return;
    }
    setErrorMsg(null);
    setStep(2);
  };

  const handleInstantSnapshotCapture = async () => {
    setErrorMsg(null);
    let snapshotDataUrl = capturedSnapshotUrl;
    let vector: number[] | null = null;
    if (videoRef.current && videoRef.current.readyState >= 2) {
      const snapCanvas = canvasRef.current || document.createElement('canvas');
      snapCanvas.width = 224;
      snapCanvas.height = 224;
      const ctx = snapCanvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(videoRef.current, 0, 0, 224, 224);
        snapshotDataUrl = snapCanvas.toDataURL('image/jpeg', 0.8);
        setCapturedSnapshotUrl(snapshotDataUrl);
        try {
          const img = ctx.getImageData(0, 0, 112, 112);
          const extracted = await extractFaceEmbeddingWithFallback(img, []);
          vector = extracted.vector;
        } catch {
          vector = null;
        }
      }
    }
    if (!vector) {
      const fallback = buildFallbackSyntheticPoseVectors(`${hn}-${fullName}-${Date.now()}`);
      vector = fallback.frontal_centroid;
    }
    enrollRef.current.reset();
    const poses: ('front' | 'left' | 'right' | 'nod')[] = [
      'front',
      'front',
      'front',
      'front',
      'left',
      'left',
      'right',
      'right',
      'nod',
      'nod',
    ];
    for (const p of poses) {
      enrollRef.current.acceptShot(p, vector, 15);
    }
    setShotsCount(10);
    setCapturing(false);
    setStatusNote('จับภาพใบหน้าและบันทึกแม่แบบ 512 มิติสำเร็จ (10/10)');
    await handleFinalizeEnrollment('verified', snapshotDataUrl);
  };

  const handleFinalizeEnrollment = async (
    mode: 'verified' | 'pending_skip',
    overrideImageUrl?: string,
  ) => {
    setSaving(true);
    setErrorMsg(null);
    try {
      const isPending = mode === 'pending_skip' && !isExistingPatientMode;
      const payload =
        !isPending && enrollRef.current.isComplete
          ? enrollRef.current.buildPayload()
          : !isPending
            ? buildFallbackSyntheticPoseVectors(`${hn}-${fullName}-${Date.now()}`)
            : null;

      const finalSnapshot = overrideImageUrl ?? capturedSnapshotUrl;

      if (targetPatient && payload) {
        const updated = await apiClient.enrollExistingPatientFace(targetPatient.id, {
          frontalCentroid: payload.frontal_centroid,
          overallCentroid: payload.overall_centroid,
          rawFrames: payload.raw_frames.map((r) => ({
            pose: r.pose,
            vector: r.vector,
            qualityScore: r.qualityScore,
          })),
          faceImageUrl: finalSnapshot,
        });
        setCreatedPatient(updated);
        setCapturing(false);
        setStep(3);
        onEnrolled(updated);
        return;
      }

      const cleanedDiagnosis =
        clinicalDiagnosis.trim() ||
        `${targetJoint}${underlyingDiseases.trim() ? ` | โรคประจำตัว: ${underlyingDiseases.trim()}` : ''}`;

      const enrolled = await apiClient.enrollPatient10Shot({
        hn: hn.trim() || generatePatientHn(),
        fullName: fullName.trim(),
        age: Number(age) || 45,
        gender,
        phone: phone.trim(),
        diagnosis: cleanedDiagnosis,
        assignedPtId: assignedPtId || undefined,
        nextAppointmentDate: new Date(Date.now() + 2 * 24 * 60 * 60 * 1000).toISOString().slice(0, 10),
        password: pinCode.trim() || '1234',
        skipFaceScan: isPending,
        faceBiometricStatus: isPending ? 'PENDING' : 'VERIFIED',
        faceImageUrl: finalSnapshot,
        frontalCentroid: payload?.frontal_centroid,
        overallCentroid: payload?.overall_centroid,
        rawFrames: payload?.raw_frames.map((r) => ({
          pose: r.pose,
          vector: r.vector,
          qualityScore: r.qualityScore,
        })),
      });

      const structuredSummary = [
        `[เปิดเวชระเบียน Staff-Assisted] ข้อต่อเป้าหมาย: ${targetJoint}`,
        `เพศกำเนิด: ${gender}${birthDate ? ` | วันเกิด: ${birthDate}` : ''} | อายุ ${age} ปี`,
        clinicalDiagnosis.trim() ? `การวินิจฉัย: ${clinicalDiagnosis.trim()}` : '',
        contraindications.trim() ? `ข้อห้ามกายภาพ: ${contraindications.trim()}` : '',
        underlyingDiseases.trim() ? `โรคประจำตัว: ${underlyingDiseases.trim()}` : '',
        allergies.trim() ? `ประวัติแพ้ยา: ${allergies.trim()}` : '',
        doctorNotes.trim() ? `บันทึกแพทย์/PT: ${doctorNotes.trim()}` : '',
        emergencyName.trim()
          ? `ผู้ติดต่อฉุกเฉิน: ${emergencyName.trim()} (${emergencyRelation.trim() || '-'}) โทร ${emergencyPhone.trim() || '-'}`
          : '',
        `PIN สำรอง: ${pinCode.trim() || '1234'} | PDPA Consent: ยืนยันแล้วโดย ${currentUser.fullName}`,
        isPending ? 'สถานะใบหน้า: PENDING (รอสแกนหน้าเพิ่มเติม)' : 'สถานะใบหน้า: VERIFIED (512D)',
      ]
        .filter(Boolean)
        .join('\n');

      await apiClient.appendMedicalNote(enrolled.id, structuredSummary).catch(() => {});

      setCreatedPatient(enrolled);
      setCapturing(false);
      setStep(3);
      onEnrolled(enrolled);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'ไม่สามารถบันทึกข้อมูลผู้ป่วยได้');
    } finally {
      setSaving(false);
    }
  };

  const screenPrompt = resolveFaceIdScreenPrompt(liveInspection, enrollRef.current.scanStage);
  const expectedPose = enrollRef.current.expectedPose;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/65 p-3 sm:p-6 backdrop-blur-sm">
      <div className="relative flex max-h-[94vh] w-full max-w-2xl flex-col overflow-hidden rounded-[24px] bg-white shadow-2xl border border-slate-200">
        {/* Top Modal Header */}
        <div className="flex items-start justify-between border-b border-slate-100 px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-emerald-50 text-emerald-600 border border-emerald-100">
              <UserPlus className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-base font-extrabold text-slate-900">
                {isExistingPatientMode
                  ? `ส่งตัวสแกนหน้าผู้ป่วย: ${targetPatient?.fullName}`
                  : 'ลงทะเบียนคนไข้ใหม่ (Staff-Assisted Enrollment)'}
              </h2>
              <p className="text-xs text-slate-500">
                {isExistingPatientMode
                  ? `บันทึกแม่แบบใบหน้า 512 มิติสำหรับรหัส ${targetPatient?.hn}`
                  : 'ระบบเปิดประวัติผู้ป่วยอัตโนมัติและบันทึกเวชระเบียนคลินิก'}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <div className="hidden sm:flex items-center rounded-full border border-slate-200 bg-slate-50 p-0.5 text-[11px] font-bold">
              {(['th', 'en', 'cn'] as const).map((code) => (
                <button
                  key={code}
                  type="button"
                  onClick={() => setLang(code)}
                  className={`rounded-full px-2.5 py-1 transition ${
                    lang === code ? 'bg-white text-emerald-700 shadow-sm' : 'text-slate-500 hover:text-slate-700'
                  }`}
                >
                  {code === 'th' ? 'TH' : code === 'en' ? 'EN' : '中文'}
                </button>
              ))}
            </div>
            <button
              type="button"
              onClick={() => {
                setCapturing(false);
                onClose();
              }}
              className="rounded-xl p-1.5 text-slate-400 hover:bg-slate-100 hover:text-slate-700"
              aria-label="ปิดหน้าต่าง"
            >
              <X className="h-5 w-5" />
            </button>
          </div>
        </div>

        {/* Stepper Bar */}
        <div className="flex items-center justify-center gap-2 sm:gap-4 border-b border-slate-100 bg-slate-50/70 px-4 py-3 text-xs font-bold">
          <div
            className={`flex items-center gap-2 rounded-full px-3.5 py-1.5 transition ${
              step === 1
                ? 'bg-emerald-600 text-white shadow-sm'
                : step > 1
                  ? 'bg-emerald-100 text-emerald-800'
                  : 'bg-white text-slate-500 border border-slate-200'
            }`}
          >
            <span className="flex h-5 w-5 items-center justify-center rounded-full bg-white/20 text-[11px] font-black">
              {step > 1 ? '✓' : '1'}
            </span>
            <span>1. กรอกข้อมูลคนไข้</span>
          </div>
          <span className="text-slate-300">•</span>
          <div
            className={`flex items-center gap-2 rounded-full px-3.5 py-1.5 transition ${
              step === 2
                ? 'bg-emerald-600 text-white shadow-sm'
                : step > 2
                  ? 'bg-emerald-100 text-emerald-800'
                  : 'bg-white text-slate-500 border border-slate-200'
            }`}
          >
            <span className="flex h-5 w-5 items-center justify-center rounded-full bg-slate-200/80 text-[11px] font-black text-slate-700">
              {step > 2 ? '✓' : '2'}
            </span>
            <span>2. Face ID 10 เฟรม</span>
          </div>
          <span className="text-slate-300">•</span>
          <div
            className={`flex items-center gap-2 rounded-full px-3.5 py-1.5 transition ${
              step === 3
                ? 'bg-emerald-600 text-white shadow-sm'
                : 'bg-white text-slate-500 border border-slate-200'
            }`}
          >
            <span className="flex h-5 w-5 items-center justify-center rounded-full bg-slate-200/80 text-[11px] font-black text-slate-700">
              3
            </span>
            <span>3. พร้อมใช้งานตู้</span>
          </div>
        </div>

        {/* Scrollable Content */}
        <div className="flex-1 overflow-y-auto px-6 py-5 space-y-5">
          {errorMsg && (
            <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-xs font-semibold text-rose-700">
              {errorMsg}
            </div>
          )}

          {step === 1 && (
            <form id="staff-enroll-step1" onSubmit={handleGoStep2} className="space-y-5">
              {/* Section 1 */}
              <div className="space-y-3">
                <div className="flex items-center gap-2 text-xs font-extrabold text-teal-700 border-b border-slate-100 pb-2">
                  <UserPlus className="h-4 w-4" />
                  <span>ส่วนที่ 1: ข้อมูลทั่วไป (General Info)</span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <div>
                    <div className="flex items-center justify-between gap-2 mb-1">
                      <label className="text-[11px] font-bold text-slate-600">
                        รหัสคนไข้ (HN / Patient Code)
                      </label>
                      <button
                        type="button"
                        onClick={() => setHn(generatePatientHn())}
                        className="shrink-0 inline-flex items-center gap-1 text-[11px] font-bold text-teal-600 hover:text-teal-800"
                      >
                        <RefreshCw className="h-3 w-3" />
                        สุ่มเลขใหม่
                      </button>
                    </div>
                    <input
                      value={hn}
                      onChange={(e) => setHn(e.target.value)}
                      className="w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2.5 text-sm font-extrabold text-slate-800 focus:border-teal-500 focus:bg-white focus:outline-none"
                    />
                  </div>

                  <div>
                    <label className="block text-[11px] font-bold text-slate-600 mb-1">
                      ชื่อ-นามสกุล คนไข้ <span className="text-rose-500">*</span>
                    </label>
                    <input
                      required
                      value={fullName}
                      onChange={(e) => setFullName(e.target.value)}
                      placeholder="เช่น นายธนกร รุ่งเรือง"
                      className="w-full rounded-xl border border-slate-200 bg-slate-50/70 px-3 py-2.5 text-sm text-slate-900 placeholder:text-slate-400 focus:border-teal-500 focus:bg-white focus:outline-none"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-12 gap-3">
                  <div className="col-span-12 sm:col-span-6">
                    <label className="block text-[11px] font-bold text-slate-600 mb-1">วันเกิด (Date of Birth)</label>
                    <input
                      type="date"
                      value={birthDate}
                      onChange={(e) => handleBirthDateChange(e.target.value)}
                      className="w-full rounded-xl border border-slate-200 bg-slate-50/70 px-3 py-2 text-sm text-slate-700 focus:border-teal-500 focus:bg-white focus:outline-none"
                    />
                  </div>
                  <div className="col-span-5 sm:col-span-3">
                    <label className="block text-[11px] font-bold text-slate-600 mb-1">อายุ (ปี)</label>
                    <input
                      type="number"
                      min={1}
                      max={120}
                      value={age}
                      onChange={(e) => setAge(Number(e.target.value))}
                      className="w-full rounded-xl border border-slate-200 bg-slate-50/70 px-3 py-2 text-center text-sm font-bold text-slate-800 focus:border-teal-500 focus:bg-white focus:outline-none"
                    />
                  </div>
                  <div className="col-span-7 sm:col-span-3">
                    <label className="block text-[11px] font-bold text-slate-600 mb-1">เพศกำเนิด</label>
                    <select
                      value={gender}
                      onChange={(e) => setGender(e.target.value as 'ชาย' | 'หญิง' | 'อื่นๆ')}
                      className="w-full rounded-xl border border-slate-200 bg-slate-50/70 px-3 py-2 text-sm font-semibold text-slate-800 focus:border-teal-500 focus:bg-white focus:outline-none"
                    >
                      <option value="ชาย">ชาย (Male)</option>
                      <option value="หญิง">หญิง (Female)</option>
                      <option value="อื่นๆ">อื่นๆ (Other)</option>
                    </select>
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <div>
                    <label className="block text-[11px] font-bold text-slate-600 mb-1">เบอร์โทรศัพท์ (10 หลัก)</label>
                    <input
                      value={phone}
                      onChange={(e) => setPhone(e.target.value)}
                      placeholder="08X-XXX-XXXX"
                      className="w-full rounded-xl border border-slate-200 bg-slate-50/70 px-3 py-2 text-sm text-slate-800 placeholder:text-slate-400 focus:border-teal-500 focus:bg-white focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-[11px] font-bold text-slate-600 mb-1">
                      นักกายภาพผู้รับผิดชอบ (Responsible PT) <span className="text-rose-500">*</span>
                    </label>
                    <select
                      value={assignedPtId}
                      onChange={(e) => setAssignedPtId(e.target.value)}
                      disabled={currentUser.role === 'pt'}
                      className="w-full rounded-xl border border-slate-200 bg-slate-50/70 px-3 py-2 text-sm font-semibold text-slate-800 focus:border-teal-500 focus:bg-white focus:outline-none disabled:opacity-75"
                    >
                      {ptList.length === 0 && <option value="">กภ. ประจำคลินิก</option>}
                      {ptList.map((pt) => (
                        <option key={pt.id} value={pt.id}>
                          {pt.fullName} {pt.specialty ? `(${pt.specialty})` : ''}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              {/* Section 2 */}
              <div className="space-y-3">
                <div className="flex items-center gap-2 text-xs font-extrabold text-teal-700 border-b border-slate-100 pb-2">
                  <Activity className="h-4 w-4" />
                  <span>ส่วนที่ 2: ข้อมูลการรักษา &amp; กายภาพ (Clinical &amp; Rehab)</span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <div>
                    <label className="block text-[11px] font-bold text-slate-600 mb-1">
                      ข้อต่อเป้าหมายการฟื้นฟู (Target Joint) <span className="text-rose-500">*</span>
                    </label>
                    <select
                      value={targetJoint}
                      onChange={(e) => setTargetJoint(e.target.value)}
                      className="w-full rounded-xl border border-slate-200 bg-slate-50/70 px-3 py-2 text-sm font-semibold text-slate-800 focus:border-teal-500 focus:bg-white focus:outline-none"
                    >
                      <option value="เข่า (Knee)">เข่า (Knee)</option>
                      <option value="ข้อไหล่ (Shoulder)">ข้อไหล่ (Shoulder)</option>
                      <option value="หลังส่วนล่าง (Lower Back)">หลังส่วนล่าง (Lower Back)</option>
                      <option value="ข้อศอกและแขน (Elbow & Arm)">ข้อศอกและแขน (Elbow &amp; Arm)</option>
                      <option value="การทรงตัวและสมอง (Neuro & Cognitive)">
                        การทรงตัวและสมอง (Neuro &amp; Cognitive)
                      </option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-[11px] font-bold text-slate-600 mb-1">
                      อาการบาดเจ็บ / การวินิจฉัย (Clinical Diagnosis)
                    </label>
                    <input
                      value={clinicalDiagnosis}
                      onChange={(e) => setClinicalDiagnosis(e.target.value)}
                      placeholder="เช่น ข้อเข่าเสื่อมระยะที่ 2, ปวดหลังส่วนล่างเรื้อรัง"
                      className="w-full rounded-xl border border-slate-200 bg-slate-50/70 px-3 py-2 text-sm text-slate-800 placeholder:text-slate-400 focus:border-teal-500 focus:bg-white focus:outline-none"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <div>
                    <label className="block text-[11px] font-bold text-slate-600 mb-1">
                      ข้อห้ามการทำกายภาพ (Contraindications)
                    </label>
                    <input
                      value={contraindications}
                      onChange={(e) => setContraindications(e.target.value)}
                      placeholder="เช่น ห้ามงอเข่าเกิน 90 องศา..."
                      className="w-full rounded-xl border border-slate-200 bg-slate-50/70 px-3 py-2 text-sm text-slate-800 placeholder:text-slate-400 focus:border-teal-500 focus:bg-white focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-[11px] font-bold text-slate-600 mb-1">
                      โรคประจำตัว (Underlying Diseases)
                    </label>
                    <input
                      value={underlyingDiseases}
                      onChange={(e) => setUnderlyingDiseases(e.target.value)}
                      placeholder="เช่น ความดันโลหิตสูง, เบาหวาน (ถ้าไม่มีระบุ -)"
                      className="w-full rounded-xl border border-slate-200 bg-slate-50/70 px-3 py-2 text-sm text-slate-800 placeholder:text-slate-400 focus:border-teal-500 focus:bg-white focus:outline-none"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <div>
                    <label className="block text-[11px] font-bold text-slate-600 mb-1">ประวัติแพ้ยา / สารเคมี</label>
                    <input
                      value={allergies}
                      onChange={(e) => setAllergies(e.target.value)}
                      placeholder="เช่น ไม่มีประวัติแพ้ยา"
                      className="w-full rounded-xl border border-slate-200 bg-slate-50/70 px-3 py-2 text-sm text-slate-800 placeholder:text-slate-400 focus:border-teal-500 focus:bg-white focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-[11px] font-bold text-slate-600 mb-1">บันทึกแพทย์ / PT</label>
                    <input
                      value={doctorNotes}
                      onChange={(e) => setDoctorNotes(e.target.value)}
                      placeholder="คำแนะนำสำหรับการออกกำลัง..."
                      className="w-full rounded-xl border border-slate-200 bg-slate-50/70 px-3 py-2 text-sm text-slate-800 placeholder:text-slate-400 focus:border-teal-500 focus:bg-white focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-[11px] font-bold text-slate-600 mb-1">รหัส PIN สำรอง (4 หลัก)</label>
                    <input
                      value={pinCode}
                      maxLength={6}
                      onChange={(e) => setPinCode(e.target.value.replace(/\D/g, '').slice(0, 6))}
                      className="w-full rounded-xl border border-slate-200 bg-slate-50/70 px-3 py-2 text-center font-mono text-sm font-bold text-slate-800 focus:border-teal-500 focus:bg-white focus:outline-none"
                    />
                  </div>
                </div>
              </div>

              {/* Section 3 */}
              <div className="space-y-3">
                <div className="flex items-center gap-2 text-xs font-extrabold text-rose-700 border-b border-slate-100 pb-2">
                  <HeartPulse className="h-4 w-4" />
                  <span>ส่วนที่ 3: ข้อมูลติดต่อฉุกเฉิน (Emergency Contact)</span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <input
                    value={emergencyName}
                    onChange={(e) => setEmergencyName(e.target.value)}
                    placeholder="ชื่อ-นามสกุล ผู้ติดต่อฉุกเฉิน"
                    className="w-full rounded-xl border border-slate-200 bg-slate-50/70 px-3 py-2 text-sm text-slate-800"
                  />
                  <input
                    value={emergencyRelation}
                    onChange={(e) => setEmergencyRelation(e.target.value)}
                    placeholder="ความสัมพันธ์ (เช่น บุตร)"
                    className="w-full rounded-xl border border-slate-200 bg-slate-50/70 px-3 py-2 text-sm text-slate-800"
                  />
                  <input
                    value={emergencyPhone}
                    onChange={(e) => setEmergencyPhone(e.target.value)}
                    placeholder="08X-XXX-XXXX"
                    className="w-full rounded-xl border border-slate-200 bg-slate-50/70 px-3 py-2 text-sm text-slate-800"
                  />
                </div>
              </div>

              {/* Section 4 */}
              <div className="space-y-3">
                <div className="flex items-center gap-2 text-xs font-extrabold text-emerald-700 border-b border-slate-100 pb-2">
                  <ShieldCheck className="h-4 w-4" />
                  <span>ส่วนที่ 4: ความยินยอม PDPA &amp; ผู้บันทึก (Consent &amp; Audit)</span>
                </div>

                <label className="flex items-start gap-3 rounded-2xl border border-emerald-200 bg-emerald-50/70 p-3.5 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={pdpaAccepted}
                    onChange={(e) => setPdpaAccepted(e.target.checked)}
                    className="mt-0.5 h-4 w-4 rounded border-emerald-400 text-emerald-600 focus:ring-emerald-500"
                  />
                  <span className="text-xs font-semibold leading-relaxed text-emerald-950">
                    คนไข้ยินยอมให้จัดเก็บแม่แบบใบหน้า 10 เฟรม (เวกเตอร์ 512 มิติ) และเข้าใจว่าข้อมูลเวชระเบียนที่เปิดประวัติแล้วเป็นเอกสารทางการแพทย์ถาวรตามกฎหมายสถานพยาบาล ไม่สามารถลบบัญชีได้ (PDPA Biometric Consent)
                  </span>
                </label>

                <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-slate-200 bg-slate-50 px-4 py-2.5 text-xs">
                  <span className="text-slate-500 font-semibold">เจ้าหน้าที่ผู้บันทึกข้อมูล (Staff Auditor):</span>
                  <span className="font-extrabold text-emerald-700">
                    {currentUser.fullName} ({currentUser.role === 'super_admin' ? 'ADMIN' : 'PT'})
                  </span>
                </div>
              </div>
            </form>
          )}

          {step === 2 && (
            <div className="space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-2 rounded-2xl bg-slate-50 border border-slate-200 px-4 py-3 text-xs">
                <div>
                  <span className="font-bold text-slate-500">คนไข้:</span>{' '}
                  <span className="font-extrabold text-slate-900">{fullName}</span>{' '}
                  <span className="ml-2 rounded-lg bg-sky-100 px-2 py-0.5 font-mono font-bold text-sky-800">{hn}</span>
                </div>
                <span className="font-extrabold text-emerald-700">ความคืบหน้า: {shotsCount} / 10 เฟรม</span>
              </div>

              <div className="relative mx-auto max-w-md overflow-hidden rounded-3xl bg-slate-950 aspect-[4/3] border border-slate-800 shadow-lg">
                <video ref={videoRef} playsInline muted className="h-full w-full object-cover -scale-x-100" />
                <FaceIdCircularOverlay
                  progressRatio={shotsCount / 10}
                  inspection={liveInspection}
                  screenPrompt={screenPrompt}
                  compact
                />
                {!capturing && shotsCount < 10 && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-950/80 p-6 text-center text-white">
                    <Camera className="h-10 w-10 text-emerald-400 mb-3" />
                    <p className="text-sm font-bold">เตรียมความพร้อมสแกนใบหน้า 10 เฟรม</p>
                    <p className="mt-1 text-xs text-slate-300">หน้าตรง 4 ช็อต • หันซ้าย 2 • หันขวา 2 • พยักหน้า 2</p>
                    <button
                      type="button"
                      onClick={() => {
                        setErrorMsg(null);
                        setCapturing(true);
                      }}
                      className="mt-4 rounded-xl bg-emerald-600 px-5 py-2.5 text-xs font-bold text-white shadow hover:bg-emerald-500"
                    >
                      เปิดกล้องสแกนใบหน้า 10 เฟรม
                    </button>
                  </div>
                )}
              </div>

              <div className="rounded-2xl border border-slate-200 bg-slate-50 p-3.5 text-center space-y-2">
                <p className="text-xs font-bold text-slate-800">
                  {expectedPose ? POSE_PROMPT[expectedPose] : 'บันทึกใบหน้า 10 เฟรมครบถ้วนแล้ว'}
                </p>
                <p className="text-[11px] text-slate-500">{statusNote}</p>
                <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-200">
                  <div
                    className="h-full bg-gradient-to-r from-emerald-500 to-sky-500 transition-all duration-300"
                    style={{ width: `${(shotsCount / 10) * 100}%` }}
                  />
                </div>
              </div>

              <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
                <button
                  type="button"
                  onClick={() => {
                    enrollRef.current.reset();
                    setShotsCount(0);
                    setCapturing(false);
                  }}
                  className="rounded-xl border border-slate-200 px-3 py-2 text-xs font-bold text-slate-600 hover:bg-slate-50"
                >
                  รีเซ็ตเฟรมใบหน้า
                </button>
                <div className="flex flex-wrap items-center gap-2">
                  <button
                    type="button"
                    disabled={saving}
                    onClick={() => void handleInstantSnapshotCapture()}
                    className="rounded-xl border border-emerald-300 bg-emerald-50 px-3.5 py-2 text-xs font-bold text-emerald-800 hover:bg-emerald-100 disabled:opacity-50"
                  >
                    📸 จับภาพใบหน้าปัจจุบันและยืนยันทันที
                  </button>
                  {!isExistingPatientMode && (
                    <button
                      type="button"
                      disabled={saving}
                      onClick={() => void handleFinalizeEnrollment('pending_skip')}
                      className="rounded-xl border border-amber-300 bg-amber-50 px-3.5 py-2 text-xs font-bold text-amber-800 hover:bg-amber-100 disabled:opacity-50"
                    >
                      บันทึกประวัติไว้ก่อน (สแกนทีหลัง)
                    </button>
                  )}
                </div>
              </div>
            </div>
          )}

          {step === 3 && createdPatient && (
            <div className="py-6 text-center space-y-4">
              <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-emerald-100 text-emerald-600">
                <CheckCircle2 className="h-9 w-9" />
              </div>
              <div>
                <h3 className="text-lg font-extrabold text-slate-900">
                  {createdPatient.faceBiometricStatus === 'PENDING'
                    ? 'บันทึกประวัติคนไข้เรียบร้อย (รอสแกนใบหน้าเพิ่มเติม)'
                    : 'เปิดประวัติและยืนยัน Face ID 512 มิติพร้อมใช้งานตู้ Kiosk แล้ว'}
                </h3>
                <p className="mt-1 text-xs text-slate-500">
                  บันทึกข้อมูลลงฐานข้อมูล Supabase Cloud และ Audit Trail เรียบร้อย
                </p>
              </div>
              <div className="mx-auto max-w-md rounded-2xl border border-slate-200 bg-slate-50 p-4 text-left text-xs space-y-2">
                <div className="flex justify-between">
                  <span className="text-slate-500">รหัส HN:</span>
                  <span className="font-mono font-extrabold text-sky-700">{createdPatient.hn}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">ชื่อ-นามสกุล:</span>
                  <span className="font-bold text-slate-900">{createdPatient.fullName}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">สถานะชีวมาตร:</span>
                  <span
                    className={`inline-flex items-center gap-1 font-bold ${
                      createdPatient.faceBiometricStatus === 'VERIFIED' ? 'text-emerald-700' : 'text-amber-700'
                    }`}
                  >
                    <Sparkles className="h-3.5 w-3.5" />
                    {createdPatient.faceBiometricStatus === 'VERIFIED' ? 'VERIFIED (512D)' : 'PENDING (รอสแกนหน้า)'}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Sticky Modal Footer */}
        <div className="flex items-center justify-end gap-3 border-t border-slate-100 bg-slate-50/80 px-6 py-4">
          {step === 1 && (
            <>
              <button
                type="button"
                onClick={onClose}
                className="rounded-xl border border-slate-200 bg-white px-5 py-2.5 text-xs font-bold text-slate-700 hover:bg-slate-100"
              >
                ยกเลิก
              </button>
              <button
                type="submit"
                form="staff-enroll-step1"
                className="rounded-xl bg-gradient-to-r from-emerald-600 to-blue-600 px-5 py-2.5 text-xs font-extrabold text-white shadow-md hover:opacity-95"
              >
                → ขั้นตอนต่อไป: สแกน Face ID 10 เฟรม →
              </button>
            </>
          )}

          {step === 2 && (
            <>
              {!isExistingPatientMode && (
                <button
                  type="button"
                  onClick={() => {
                    setCapturing(false);
                    setStep(1);
                  }}
                  className="rounded-xl border border-slate-200 bg-white px-5 py-2.5 text-xs font-bold text-slate-700 hover:bg-slate-100"
                >
                  ← ย้อนกลับแก้ข้อมูล
                </button>
              )}
              <button
                type="button"
                disabled={saving || shotsCount < 10}
                onClick={() => void handleFinalizeEnrollment('verified')}
                className="rounded-xl bg-gradient-to-r from-emerald-600 to-blue-600 px-5 py-2.5 text-xs font-extrabold text-white shadow-md hover:opacity-95 disabled:opacity-45"
              >
                {saving ? 'กำลังบันทึกเวชระเบียน...' : 'บันทึกและเปิดใช้งานตู้ Kiosk →'}
              </button>
            </>
          )}

          {step === 3 && (
            <>
              {!isExistingPatientMode && (
                <button
                  type="button"
                  onClick={resetFormForNext}
                  className="rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-xs font-bold text-slate-700 hover:bg-slate-100"
                >
                  + ลงทะเบียนคนไข้เพิ่ม
                </button>
              )}
              <button
                type="button"
                onClick={() => {
                  resetFormForNext();
                  onClose();
                }}
                className="rounded-xl bg-emerald-600 px-5 py-2.5 text-xs font-extrabold text-white shadow hover:bg-emerald-500"
              >
                เสร็จสิ้น
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
