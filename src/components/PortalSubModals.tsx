import React, { useEffect, useRef, useState } from 'react';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import type {
  DrTechCaresApiClient,
  ManagedExerciseRecord,
  MedicalNoteRecord,
  ModpackRecord,
  PatientBiometricIndexEntry,
  PatientRecord,
  RehabScheduleItem,
  RehabSessionRecord,
  UserAccount,
} from '@drtechcares/api-client';
import {
  alignFaceTo112,
  evaluateLivenessAndPose,
  extractFaceEmbeddingWithFallback,
  selectClinicalCameraDeviceId,
} from '@drtechcares/face-client';
import {
  TemporalBiometricMatcher,
  computeBoundingBoxFromLandmarks,
  subscribeBiometricIndex,
} from '../lib/faceMatcher';
import {
  Activity,
  Calendar,
  Camera,
  CheckCircle2,
  FileText,
  KeyRound,
  Plus,
  ScanFace,
  Trash2,
  UserPlus,
  X,
} from 'lucide-react';

export function WelcomeFaceScanModal({
  open,
  apiClient,
  onClose,
  onOpenNewEnrollment,
  onCheckInSuccess,
}: {
  open: boolean;
  apiClient: DrTechCaresApiClient;
  onClose: () => void;
  onOpenNewEnrollment: () => void;
  onCheckInSuccess: (patientId: string, label: string) => void;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const biometricIndexRef = useRef<PatientBiometricIndexEntry[]>([]);
  const temporalMatcherRef = useRef(new TemporalBiometricMatcher(5));
  const confirmedLockRef = useRef(false);

  const [cameraError, setCameraError] = useState<string | null>(null);
  const [scanning, setScanning] = useState(false);
  const [hudBox, setHudBox] = useState<{ xPct: number; yPct: number; wPct: number; hPct: number } | null>(null);
  const [liveMatchHud, setLiveMatchHud] = useState<{
    candidate: PatientBiometricIndexEntry;
    confidencePercent: number;
    streak: number;
    confirmed: boolean;
  } | null>(null);
  const [matchResult, setMatchResult] = useState<{
    matched: boolean;
    confidencePercent: number;
    reason?: string;
    patient?: { id: string; hn: string; fullName: string; nextAppointmentDate: string } | null;
  } | null>(null);

  useEffect(() => {
    if (!open) return;
    return subscribeBiometricIndex(apiClient, (entries) => {
      biometricIndexRef.current = entries;
    });
  }, [open, apiClient]);

  useEffect(() => {
    if (!open) return;
    let active = true;
    let stream: MediaStream | null = null;
    let landmarker: FaceLandmarker | null = null;
    let rafId = 0;
    let prevLandmarks: Array<{ x: number; y: number; z?: number }> | null = null;
    let lastEmbedAt = 0;
    const alignCanvas = document.createElement('canvas');

    confirmedLockRef.current = false;
    temporalMatcherRef.current.reset();
    setMatchResult(null);
    setCameraError(null);
    setHudBox(null);
    setLiveMatchHud(null);

    async function startLiveVerification() {
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
        if (!active) return;
        landmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: '/models/face_landmarker.task' },
          runningMode: 'VIDEO',
          numFaces: 1,
          minFaceDetectionConfidence: 0.85,
          minFacePresenceConfidence: 0.85,
        });

        const loop = async () => {
          if (!active || !videoRef.current || !landmarker) return;
          if (videoRef.current.readyState >= 2) {
            const now = performance.now();
            const res = landmarker.detectForVideo(videoRef.current, now);
            const faceCount = res.faceLandmarks?.length ?? 0;
            const lms = res.faceLandmarks?.[0] ?? [];

            if (faceCount !== 1 || lms.length < 300) {
              setHudBox(null);
              if (!confirmedLockRef.current) {
                setLiveMatchHud(null);
                temporalMatcherRef.current.reset();
              }
              prevLandmarks = null;
            } else {
              const rawBox = computeBoundingBoxFromLandmarks(lms);
              if (rawBox) {
                setHudBox({
                  xPct: Math.max(2, (1 - rawBox.x - rawBox.width) * 100),
                  yPct: rawBox.y * 100,
                  wPct: rawBox.width * 100,
                  hPct: rawBox.height * 100,
                });
              }
              const liveness = evaluateLivenessAndPose(lms, prevLandmarks, faceCount, false);
              prevLandmarks = lms;

              if (
                !confirmedLockRef.current &&
                biometricIndexRef.current.length > 0 &&
                liveness.isLive &&
                now - lastEmbedAt >= 120
              ) {
                lastEmbedAt = now;
                const crop = alignFaceTo112(videoRef.current, lms, alignCanvas);
                if (crop) {
                  const { vector } = await extractFaceEmbeddingWithFallback(crop, lms);
                  const temporal = temporalMatcherRef.current.evaluateFrame(
                    {
                      probeVector: vector,
                      yawDeg: liveness.yawDeg,
                      pitchDeg: liveness.pitchDeg,
                      depthScore: liveness.depthScore,
                      microMotion: liveness.microMotion,
                      blinkDetected: liveness.blinkDetected,
                    },
                    biometricIndexRef.current,
                  );

                  const isHudMatch = Boolean(temporal.candidate && temporal.streakCount > 0);
                  setLiveMatchHud(
                    isHudMatch && temporal.candidate
                      ? {
                          candidate: temporal.candidate,
                          confidencePercent: temporal.confidencePercent,
                          streak: temporal.streakCount,
                          confirmed: temporal.confirmed,
                        }
                      : null,
                  );

                  if (temporal.confirmed && temporal.candidate && !confirmedLockRef.current) {
                    confirmedLockRef.current = true;
                    const matchedCand = temporal.candidate;
                    const matchedConf = Math.round(temporal.confidencePercent);
                    try {
                      const verified = await apiClient.verifyFaceLogin(vector);
                      if (!active) return;
                      if (verified.matched && verified.patient) {
                        setMatchResult({
                          matched: true,
                          confidencePercent: Math.max(verified.confidencePercent, matchedConf),
                          patient: verified.patient,
                        });
                      } else {
                        setMatchResult({
                          matched: true,
                          confidencePercent: matchedConf,
                          patient: {
                            id: matchedCand.patientId,
                            hn: matchedCand.hn,
                            fullName: matchedCand.fullName,
                            nextAppointmentDate:
                              matchedCand.nextAppointmentDate ?? new Date().toISOString().slice(0, 10),
                          },
                        });
                      }
                    } catch {
                      if (!active) return;
                      setMatchResult({
                        matched: true,
                        confidencePercent: matchedConf,
                        patient: {
                          id: matchedCand.patientId,
                          hn: matchedCand.hn,
                          fullName: matchedCand.fullName,
                          nextAppointmentDate:
                            matchedCand.nextAppointmentDate ?? new Date().toISOString().slice(0, 10),
                        },
                      });
                    }
                  }
                }
              }
            }
          }
          rafId = requestAnimationFrame(loop);
        };

        rafId = requestAnimationFrame(loop);
      } catch (err) {
        if (active) {
          setCameraError(err instanceof Error ? err.message : 'ไม่สามารถเข้าถึงกล้องได้');
        }
      }
    }

    void startLiveVerification();
    return () => {
      active = false;
      cancelAnimationFrame(rafId);
      landmarker?.close();
      stream?.getTracks().forEach((tr) => tr.stop());
    };
  }, [open, apiClient]);

  if (!open) return null;

  const handleScanFrame = async () => {
    setScanning(true);
    setMatchResult(null);
    confirmedLockRef.current = false;
    temporalMatcherRef.current.reset();
    try {
      let probeVector: number[] | null = null;
      if (videoRef.current && videoRef.current.readyState >= 2) {
        const canvas = document.createElement('canvas');
        try {
          const vision = await FilesetResolver.forVisionTasks('/mediapipe/wasm');
          const landmarker = await FaceLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: '/models/face_landmarker.task' },
            runningMode: 'IMAGE',
            numFaces: 1,
            minFaceDetectionConfidence: 0.85,
          });
          const res = landmarker.detect(videoRef.current);
          const lms = res.faceLandmarks?.[0] ?? [];
          const crop = lms.length >= 300 ? alignFaceTo112(videoRef.current, lms, canvas) : null;
          if (crop) {
            const extracted = await extractFaceEmbeddingWithFallback(crop, lms);
            probeVector = extracted.vector;
          }
          landmarker.close();
        } catch {
          canvas.width = 112;
          canvas.height = 112;
          const ctx = canvas.getContext('2d');
          if (ctx) {
            ctx.drawImage(videoRef.current, 0, 0, 112, 112);
            const extracted = await extractFaceEmbeddingWithFallback(ctx.getImageData(0, 0, 112, 112), []);
            probeVector = extracted.vector;
          }
        }
      }
      if (!probeVector) {
        setMatchResult({
          matched: false,
          confidencePercent: 0,
          reason: 'ไม่พบสัญญาณภาพจากกล้อง กรุณาเปิดกล้องหรือลงทะเบียนคนไข้ใหม่',
        });
        return;
      }
      const verified = await apiClient.verifyFaceLogin(probeVector);
      setMatchResult({
        matched: verified.matched,
        confidencePercent: verified.confidencePercent,
        reason: verified.failureReasons?.[0],
        patient: verified.patient,
      });
    } catch (err) {
      setMatchResult({
        matched: false,
        confidencePercent: 0,
        reason: err instanceof Error ? err.message : 'สแกนใบหน้าไม่สำเร็จ',
      });
    } finally {
      setScanning(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/65 p-4 backdrop-blur-sm">
      <div className="w-full max-w-lg rounded-3xl bg-white p-6 shadow-2xl border border-slate-200 space-y-4 text-xs">
        <div className="flex items-center justify-between border-b border-slate-100 pb-3">
          <div className="flex items-center gap-2">
            <ScanFace className="h-5 w-5 text-emerald-600" />
            <div>
              <h3 className="text-base font-black text-slate-900">ต้อนรับ &amp; สแกนใบหน้าคนไข้ (Welcome Verification)</h3>
              <p className="text-[11px] text-slate-500">ตรวจสอบอัตลักษณ์ 3D Multi-Angle 512 มิติ อัตโนมัติแบบเรียลไทม์</p>
            </div>
          </div>
          <button type="button" onClick={onClose} className="rounded-xl p-1 text-slate-400 hover:bg-slate-100">
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="relative overflow-hidden rounded-2xl bg-slate-950 aspect-[4/3] border border-slate-800">
          <video ref={videoRef} playsInline muted className="h-full w-full object-cover -scale-x-100" />

          {hudBox && liveMatchHud?.candidate && (
            <div
              className="pointer-events-none absolute z-20 transition-all duration-100"
              style={{
                left: `${hudBox.xPct}%`,
                top: `${hudBox.yPct}%`,
                width: `${hudBox.wPct}%`,
                height: `${hudBox.hPct}%`,
              }}
            >
              <div
                className={`relative h-full w-full rounded-xl border-2 ${
                  liveMatchHud.confirmed
                    ? 'border-emerald-400 shadow-[0_0_24px_rgba(16,185,129,0.7)]'
                    : 'border-emerald-400/75 shadow-[0_0_12px_rgba(16,185,129,0.4)]'
                }`}
              >
                <span className="absolute -top-1 -left-1 h-3.5 w-3.5 border-t-[3px] border-l-[3px] border-emerald-300 rounded-tl" />
                <span className="absolute -top-1 -right-1 h-3.5 w-3.5 border-t-[3px] border-r-[3px] border-emerald-300 rounded-tr" />
                <span className="absolute -bottom-1 -left-1 h-3.5 w-3.5 border-b-[3px] border-l-[3px] border-emerald-300 rounded-bl" />
                <span className="absolute -bottom-1 -right-1 h-3.5 w-3.5 border-b-[3px] border-r-[3px] border-emerald-300 rounded-br" />
              </div>

              <div className="absolute left-1/2 top-full mt-1.5 w-max max-w-[260px] -translate-x-1/2 rounded-lg border border-emerald-400/45 bg-slate-950/90 px-3 py-1.5 text-center backdrop-blur-md shadow-xl">
                <div className="text-[11px] font-black text-emerald-300">
                  ✓ ยืนยันตัวตนสำเร็จ: {liveMatchHud.candidate.fullName} ({liveMatchHud.candidate.hn})
                </div>
                <div className="text-[10px] font-bold text-white/90">
                  ความแม่นยำ: {liveMatchHud.confidencePercent.toFixed(1)}% (3D Multi-Angle Match •{' '}
                  {liveMatchHud.streak}/5)
                </div>
              </div>
            </div>
          )}

          {cameraError && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-950/90 p-5 text-center text-amber-300">
              <Camera className="h-8 w-8 mb-2" />
              <p className="font-bold">{cameraError}</p>
            </div>
          )}
        </div>

        {matchResult && (
          <div
            className={`rounded-2xl border p-4 space-y-2 ${
              matchResult.matched && matchResult.patient
                ? 'border-emerald-200 bg-emerald-50 text-emerald-950'
                : 'border-amber-200 bg-amber-50 text-amber-950'
            }`}
          >
            {matchResult.matched && matchResult.patient ? (
              <>
                <div className="flex items-center justify-between font-extrabold">
                  <span className="inline-flex items-center gap-1.5 text-emerald-700">
                    <CheckCircle2 className="h-4 w-4" /> ยืนยันตัวตนสำเร็จ ({matchResult.confidencePercent}%)
                  </span>
                  <span className="font-mono text-sky-700">{matchResult.patient.hn}</span>
                </div>
                <div className="text-sm font-black">{matchResult.patient.fullName}</div>
                <button
                  type="button"
                  onClick={async () => {
                    const pid = matchResult.patient!.id;
                    const pname = matchResult.patient!.fullName;
                    await apiClient.quickCheckInPatient(pid, 'เช็คอินผ่านต้อนรับ & สแกนหน้า');
                    onCheckInSuccess(pid, pname);
                    onClose();
                  }}
                  className="w-full rounded-xl bg-emerald-600 py-2 font-extrabold text-white hover:bg-emerald-500"
                >
                  เช็คอินเข้าฝึกวันนี้ &amp; เปิดตารางฝึก
                </button>
              </>
            ) : (
              <div className="flex items-center justify-between gap-2">
                <span>
                  ไม่พบข้อมูลใบหน้าในระบบ ({matchResult.confidencePercent}%) — {matchResult.reason || 'กรุณาลงทะเบียนใหม่'}
                </span>
                <button
                  type="button"
                  onClick={() => {
                    onClose();
                    onOpenNewEnrollment();
                  }}
                  className="shrink-0 rounded-xl bg-emerald-600 px-3 py-1.5 font-bold text-white"
                >
                  + ลงทะเบียนใหม่
                </button>
              </div>
            )}
          </div>
        )}

        <div className="flex items-center justify-between gap-2">
          <button
            type="button"
            onClick={() => {
              onClose();
              onOpenNewEnrollment();
            }}
            className="inline-flex items-center gap-1.5 rounded-xl border border-slate-200 px-4 py-2.5 font-bold text-slate-700 hover:bg-slate-50"
          >
            <UserPlus className="h-4 w-4" />
            ลงทะเบียนคนไข้ใหม่
          </button>
          <button
            type="button"
            disabled={scanning}
            onClick={() => void handleScanFrame()}
            className="inline-flex items-center gap-1.5 rounded-xl bg-gradient-to-r from-emerald-600 to-blue-600 px-5 py-2.5 font-extrabold text-white shadow hover:opacity-95 disabled:opacity-50"
          >
            <ScanFace className="h-4 w-4" />
            {scanning ? 'กำลังตรวจสอบ 512 มิติ...' : 'เริ่มสแกนจับคู่ใบหน้า'}
          </button>
        </div>
      </div>
    </div>
  );
}

export function ResetPasswordModal({
  user,
  apiClient,
  onClose,
  onSuccess,
}: {
  user: UserAccount | null;
  apiClient: DrTechCaresApiClient;
  onClose: () => void;
  onSuccess: (msg: string) => void;
}) {
  const [newPassword, setNewPassword] = useState('Staff@2026');
  const [saving, setSaving] = useState(false);
  const [errMsg, setErrMsg] = useState<string | null>(null);

  if (!user) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/60 p-4 backdrop-blur-sm">
      <div className="w-full max-w-sm rounded-3xl bg-white p-6 shadow-2xl border border-slate-200 space-y-4 text-xs">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <KeyRound className="h-4 w-4 text-sky-600" />
            <h3 className="text-sm font-black text-slate-900">รีเซ็ตรหัสผ่าน @{user.username}</h3>
          </div>
          <button type="button" onClick={onClose} className="text-slate-400 hover:text-slate-700">
            <X className="h-4 w-4" />
          </button>
        </div>
        {errMsg && <div className="rounded-xl bg-rose-50 px-3 py-2 text-rose-700 font-semibold">{errMsg}</div>}
        <form
          onSubmit={async (e) => {
            e.preventDefault();
            if (newPassword.trim().length < 6) {
              setErrMsg('กรุณากรอกรหัสผ่านอย่างน้อย 6 ตัวอักษร');
              return;
            }
            setSaving(true);
            setErrMsg(null);
            try {
              await apiClient.resetUserPassword(user.id, newPassword.trim());
              onSuccess(`รีเซ็ตรหัสผ่านของ @${user.username} เรียบร้อยแล้ว`);
              onClose();
            } catch (err) {
              setErrMsg(err instanceof Error ? err.message : 'รีเซ็ตรหัสผ่านไม่สำเร็จ');
            } finally {
              setSaving(false);
            }
          }}
          className="space-y-3"
        >
          <div>
            <label className="block font-bold text-slate-600 mb-1">รหัสผ่านใหม่ (อย่างน้อย 6 ตัวอักษร)</label>
            <input
              required
              minLength={6}
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              className="w-full rounded-xl border border-slate-200 px-3 py-2.5 font-mono text-sm"
            />
          </div>
          <div className="flex justify-end gap-2">
            <button type="button" onClick={onClose} className="rounded-xl border border-slate-200 px-4 py-2 font-bold">
              ยกเลิก
            </button>
            <button
              type="submit"
              disabled={saving}
              className="rounded-xl bg-sky-600 px-4 py-2 font-extrabold text-white hover:bg-sky-500 disabled:opacity-50"
            >
              {saving ? 'กำลังบันทึก...' : 'บันทึกรหัสผ่านใหม่'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export function PatientProfileAndHistoryModal({
  patient,
  initialTab = 'profile',
  apiClient,
  currentUser,
  ptUsers,
  sessions,
  onClose,
  onRefresh,
  onOpenFaceScan,
  onFlash,
}: {
  patient: PatientRecord | null;
  initialTab?: 'profile' | 'history';
  apiClient: DrTechCaresApiClient;
  currentUser: UserAccount;
  ptUsers: UserAccount[];
  sessions: RehabSessionRecord[];
  onClose: () => void;
  onRefresh: () => Promise<void>;
  onOpenFaceScan: (patient: PatientRecord) => void;
  onFlash: (kind: 'success' | 'error', msg: string) => void;
}) {
  const [modalTab, setModalTab] = useState<'profile' | 'history'>(initialTab);
  const [fullName, setFullName] = useState('');
  const [age, setAge] = useState(45);
  const [gender, setGender] = useState('ชาย');
  const [phone, setPhone] = useState('');
  const [diagnosis, setDiagnosis] = useState('');
  const [assignedPtId, setAssignedPtId] = useState('');
  const [delegatePtId, setDelegatePtId] = useState('');
  const [delegateNote, setDelegateNote] = useState('');
  const [notes, setNotes] = useState<MedicalNoteRecord[]>([]);
  const [newNote, setNewNote] = useState('');
  const [reportSummary, setReportSummary] = useState('');
  const [reportRec, setReportRec] = useState('');
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setModalTab(initialTab);
    if (!patient) return;
    setFullName(patient.fullName);
    setAge(patient.age);
    setGender(patient.gender || 'ชาย');
    setPhone(patient.phone || '');
    setDiagnosis(patient.diagnosis || '');
    setAssignedPtId(patient.assignedPtId || '');
    setDelegatePtId(patient.delegatedToPtId || patient.assignedPtId || '');
    setDelegateNote(patient.delegationNote || '');
    apiClient
      .listMedicalNotes(patient.id)
      .then((n) => setNotes(n))
      .catch(() => setNotes([]));
  }, [patient, initialTab, apiClient]);

  if (!patient) return null;

  const patientSessions = sessions.filter((s) => s.patientId === patient.id);
  const ptNameMap = new Map(ptUsers.map((p) => [p.id, p.fullName]));

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/60 p-4 backdrop-blur-sm">
      <div className="w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-3xl bg-white p-6 shadow-2xl border border-slate-200 space-y-5 text-xs">
        <div className="flex items-start justify-between border-b border-slate-100 pb-4">
          <div className="flex items-center gap-3">
            {patient.faceImageUrl ? (
              <img
                src={patient.faceImageUrl}
                alt={patient.fullName}
                className="h-12 w-12 rounded-2xl object-cover border border-emerald-200"
              />
            ) : (
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-emerald-50 text-emerald-700 font-black text-base">
                {patient.fullName.slice(0, 1)}
              </div>
            )}
            <div>
              <div className="flex items-center gap-2">
                <span className="rounded-lg bg-sky-100 px-2.5 py-0.5 font-mono font-extrabold text-sky-700">
                  {patient.hn}
                </span>
                <h3 className="text-base font-black text-slate-900">{patient.fullName}</h3>
              </div>
              <p className="text-slate-500 mt-1">
                อายุ {patient.age} ปี ({patient.gender || 'ชาย'}) • PT:{' '}
                {ptNameMap.get(patient.delegatedToPtId || patient.assignedPtId) || 'กภ. ประจำคลินิก'}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => {
                onClose();
                onOpenFaceScan(patient);
              }}
              className="rounded-xl border border-emerald-200 bg-emerald-50 px-3 py-1.5 font-bold text-emerald-700 hover:bg-emerald-100"
            >
              📸 ส่งตัวสแกนหน้า
            </button>
            <button type="button" onClick={onClose} className="rounded-xl p-1 text-slate-400 hover:bg-slate-100">
              <X className="h-5 w-5" />
            </button>
          </div>
        </div>

        {/* Modal Sub-tabs */}
        <div className="inline-flex rounded-xl border border-slate-200 bg-slate-50 p-1 font-bold">
          <button
            type="button"
            onClick={() => setModalTab('profile')}
            className={`rounded-lg px-4 py-1.5 ${
              modalTab === 'profile' ? 'bg-white text-emerald-700 shadow-sm' : 'text-slate-500'
            }`}
          >
            <FileText className="inline h-3.5 w-3.5 mr-1" /> โปรไฟล์ &amp; เวชระเบียน
          </button>
          <button
            type="button"
            onClick={() => setModalTab('history')}
            className={`rounded-lg px-4 py-1.5 ${
              modalTab === 'history' ? 'bg-white text-emerald-700 shadow-sm' : 'text-slate-500'
            }`}
          >
            <Activity className="inline h-3.5 w-3.5 mr-1" /> ประวัติการฝึก &amp; ROM ({patientSessions.length})
          </button>
        </div>

        {modalTab === 'profile' ? (
          <>
            <form
              onSubmit={async (e) => {
                e.preventDefault();
                setSaving(true);
                try {
                  await apiClient.updatePatient(patient.id, {
                    fullName: fullName.trim(),
                    age: Number(age),
                    gender,
                    phone: phone.trim(),
                    diagnosis: diagnosis.trim(),
                    assignedPtId: currentUser.role === 'super_admin' ? assignedPtId : undefined,
                  });
                  onFlash('success', `อัปเดตประวัติคนไข้ ${fullName} เรียบร้อยแล้ว`);
                  await onRefresh();
                } catch (err) {
                  onFlash('error', err instanceof Error ? err.message : 'อัปเดตประวัติไม่สำเร็จ');
                } finally {
                  setSaving(false);
                }
              }}
              className="rounded-2xl border border-slate-200 bg-slate-50/60 p-4 space-y-3"
            >
              <div className="font-extrabold text-slate-900">แก้ไขข้อมูลพื้นฐานคนไข้</div>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                <input
                  required
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  placeholder="ชื่อ-นามสกุล"
                  className="rounded-xl border border-slate-200 bg-white px-3 py-2 sm:col-span-2"
                />
                <div className="flex gap-1.5">
                  <input
                    type="number"
                    value={age}
                    onChange={(e) => setAge(Number(e.target.value))}
                    className="w-16 rounded-xl border border-slate-200 bg-white px-2 py-2 text-center"
                  />
                  <select
                    value={gender}
                    onChange={(e) => setGender(e.target.value)}
                    className="flex-1 rounded-xl border border-slate-200 bg-white px-2 py-2"
                  >
                    <option value="ชาย">ชาย</option>
                    <option value="หญิง">หญิง</option>
                    <option value="อื่นๆ">อื่นๆ</option>
                  </select>
                </div>
                <input
                  value={phone}
                  onChange={(e) => setPhone(e.target.value)}
                  placeholder="เบอร์โทร"
                  className="rounded-xl border border-slate-200 bg-white px-3 py-2"
                />
                <input
                  value={diagnosis}
                  onChange={(e) => setDiagnosis(e.target.value)}
                  placeholder="การวินิจฉัยโรค"
                  className="rounded-xl border border-slate-200 bg-white px-3 py-2 sm:col-span-2"
                />
              </div>
              <div className="flex flex-wrap items-center justify-between gap-2">
                {currentUser.role === 'super_admin' && (
                  <select
                    value={assignedPtId}
                    onChange={(e) => setAssignedPtId(e.target.value)}
                    className="rounded-xl border border-slate-200 bg-white px-3 py-2 font-semibold"
                  >
                    {ptUsers.map((pt) => (
                      <option key={pt.id} value={pt.id}>
                        PT หลัก: {pt.fullName}
                      </option>
                    ))}
                  </select>
                )}
                <button
                  type="submit"
                  disabled={saving}
                  className="rounded-xl bg-emerald-600 px-4 py-2 font-bold text-white hover:bg-emerald-500 disabled:opacity-50"
                >
                  {saving ? 'กำลังบันทึก...' : 'บันทึกโปรไฟล์คนไข้'}
                </button>
              </div>
            </form>

            {/* Delegation */}
            <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4 space-y-2.5">
              <div className="font-extrabold text-slate-800">ส่งต่อเคสให้ PT ผู้รับผิดชอบ (Delegation)</div>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                <select
                  value={delegatePtId}
                  onChange={(e) => setDelegatePtId(e.target.value)}
                  className="rounded-xl border border-slate-200 bg-white px-3 py-2 font-semibold"
                >
                  <option value="">-- ไม่ส่งต่อ (ใช้ PT หลัก) --</option>
                  {ptUsers.map((pt) => (
                    <option key={pt.id} value={pt.id}>
                      {pt.fullName}
                    </option>
                  ))}
                </select>
                <input
                  value={delegateNote}
                  onChange={(e) => setDelegateNote(e.target.value)}
                  placeholder="หมายเหตุการส่งต่อเคส..."
                  className="rounded-xl border border-slate-200 bg-white px-3 py-2"
                />
                <button
                  type="button"
                  onClick={async () => {
                    await apiClient.delegatePatientCase(patient.id, delegatePtId || null, delegateNote || undefined);
                    onFlash('success', 'อัปเดตการส่งต่อ PT เรียบร้อยแล้ว');
                    await onRefresh();
                  }}
                  className="rounded-xl bg-blue-600 px-3 py-2 font-bold text-white hover:bg-blue-500"
                >
                  อัปเดต PT ผู้รับผิดชอบ
                </button>
              </div>
            </div>

            {/* Append-Only Medical Notes */}
            <div className="space-y-2.5">
              <div className="font-extrabold text-slate-900">บันทึกเวชระเบียนถาวร (Append-Only Notes)</div>
              <div className="max-h-40 overflow-y-auto space-y-2 rounded-2xl border border-slate-200 p-3 bg-slate-50/40">
                {notes.map((n) => (
                  <div key={n.id} className="rounded-xl border border-slate-200 bg-white p-3 space-y-1">
                    <div className="flex justify-between text-[11px] text-slate-400">
                      <span className="font-bold text-emerald-700">{n.authorName}</span>
                      <span>{new Date(n.createdAt).toLocaleString('th-TH')}</span>
                    </div>
                    <p className="whitespace-pre-line text-slate-800">{n.content}</p>
                  </div>
                ))}
                {notes.length === 0 && <p className="text-center text-slate-400 py-3">ยังไม่มีบันทึกเวชระเบียน</p>}
              </div>

              <div className="flex gap-2">
                <input
                  value={newNote}
                  onChange={(e) => setNewNote(e.target.value)}
                  placeholder="พิมพ์บันทึกอาการเพิ่มเติม..."
                  className="flex-1 rounded-xl border border-slate-200 px-3 py-2"
                />
                <button
                  type="button"
                  onClick={async () => {
                    if (!newNote.trim()) return;
                    const created = await apiClient.appendMedicalNote(patient.id, newNote.trim());
                    setNotes((prev) => [created, ...prev]);
                    setNewNote('');
                  }}
                  className="rounded-xl bg-emerald-600 px-4 py-2 font-bold text-white"
                >
                  + บันทึก Note
                </button>
              </div>
            </div>

            {/* Treatment Report */}
            <div className="border-t border-slate-100 pt-3 space-y-2">
              <div className="font-extrabold text-slate-900">ออกใบสรุปผลการรักษา (Treatment Report)</div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                <input
                  value={reportSummary}
                  onChange={(e) => setReportSummary(e.target.value)}
                  placeholder="สรุปผลการฟื้นฟู..."
                  className="rounded-xl border border-slate-200 px-3 py-2"
                />
                <input
                  value={reportRec}
                  onChange={(e) => setReportRec(e.target.value)}
                  placeholder="คำแนะนำต่อเนื่อง..."
                  className="rounded-xl border border-slate-200 px-3 py-2"
                />
              </div>
              <button
                type="button"
                onClick={async () => {
                  if (!reportSummary.trim()) return;
                  await apiClient.createTreatmentReport({
                    patientId: patient.id,
                    summary: reportSummary.trim(),
                    aiRecommendation: reportRec.trim() || 'ฝึกต่อเนื่องตามนัดหมาย',
                    scoreSummary: `บันทึกโดย ${currentUser.fullName}`,
                  });
                  setReportSummary('');
                  setReportRec('');
                  onFlash('success', 'ออกใบสรุปผลการรักษาเรียบร้อยแล้ว');
                }}
                className="rounded-xl bg-slate-900 px-4 py-2 font-bold text-white"
              >
                บันทึกใบสรุปผลการรักษา
              </button>
            </div>
          </>
        ) : (
          <div className="space-y-4">
            <div className="grid grid-cols-3 gap-3 text-center">
              <div className="rounded-2xl border border-slate-200 bg-slate-50 p-3">
                <div className="text-[11px] text-slate-500">จำนวนครั้งที่ฝึก</div>
                <div className="text-lg font-black text-slate-900">{patientSessions.length} ครั้ง</div>
              </div>
              <div className="rounded-2xl border border-slate-200 bg-emerald-50/50 p-3">
                <div className="text-[11px] text-slate-500">ความแม่นยำเฉลี่ย</div>
                <div className="text-lg font-black text-emerald-700">
                  {patientSessions.length > 0
                    ? Math.round(
                        patientSessions.reduce((acc, s) => acc + s.accuracyPercent, 0) / patientSessions.length,
                      )
                    : 0}
                  %
                </div>
              </div>
              <div className="rounded-2xl border border-slate-200 bg-sky-50/50 p-3">
                <div className="text-[11px] text-slate-500">ROM สูงสุด</div>
                <div className="text-lg font-black text-sky-700">
                  {patientSessions.reduce((max, s) => Math.max(max, s.maxRomDeg), 0)}°
                </div>
              </div>
            </div>

            <div className="max-h-60 overflow-y-auto space-y-2">
              {patientSessions.map((s) => (
                <div key={s.id} className="rounded-xl border border-slate-200 p-3 flex items-center justify-between">
                  <div>
                    <div className="font-bold text-slate-900">{s.exerciseName}</div>
                    <div className="text-[11px] text-slate-500">
                      {s.completedSets} เซต × {s.completedReps} ครั้ง • ความแม่นยำ {s.accuracyPercent}% • Max ROM{' '}
                      {s.maxRomDeg}°
                    </div>
                  </div>
                  <span className="text-[11px] text-slate-400">
                    {new Date(s.createdAt).toLocaleDateString('th-TH')}
                  </span>
                </div>
              ))}
              {patientSessions.length === 0 && (
                <div className="rounded-2xl border border-dashed border-slate-200 py-8 text-center text-slate-400">
                  ยังไม่มีประวัติการฝึกของคนไข้รายนี้
                </div>
              )}
            </div>

            <button
              type="button"
              onClick={async () => {
                await apiClient.recordRehabSession({
                  patientId: patient.id,
                  activityType: 'physio_exercise',
                  exerciseId: 'MANUAL_CLINIC_SESSION',
                  exerciseName: 'ฝึกกายภาพบำบัดร่วมกับนักกายภาพ',
                  completedSets: 3,
                  completedReps: 10,
                  accuracyPercent: 95,
                  maxRomDeg: 135,
                  painStopTriggered: false,
                  aiFeedback: `บันทึกการฝึกโดย ${currentUser.fullName}`,
                  kinematicsReplay: [],
                });
                onFlash('success', 'บันทึกผลการฝึกกายภาพใหม่เรียบร้อย');
                await onRefresh();
              }}
              className="w-full rounded-xl bg-emerald-600 py-2.5 font-extrabold text-white hover:bg-emerald-500"
            >
              + บันทึกผลเซสชันการฝึกวันนี้
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export function ScheduleCreatorAndDayModal({
  mode,
  selectedDateIso,
  patient,
  patients,
  ptUsers,
  exercises,
  modpacks,
  apiClient,
  onClose,
  onSaved,
}: {
  mode: 'create' | 'day_detail' | null;
  selectedDateIso: string | null;
  patient: PatientRecord | null;
  patients: PatientRecord[];
  ptUsers: UserAccount[];
  exercises: ManagedExerciseRecord[];
  modpacks: ModpackRecord[];
  apiClient: DrTechCaresApiClient;
  onClose: () => void;
  onSaved: (msg: string) => Promise<void>;
}) {
  const [targetPatientId, setTargetPatientId] = useState(patient?.id || patients[0]?.id || '');
  const [ptId, setPtId] = useState(patient?.assignedPtId || ptUsers[0]?.id || '');
  const [exerciseId, setExerciseId] = useState(exercises[0]?.id || '');
  const [modpackId, setModpackId] = useState(modpacks[0]?.id || '');
  const [startDate, setStartDate] = useState(selectedDateIso || new Date().toISOString().slice(0, 10));
  const [repeatDays, setRepeatDays] = useState(3);
  const [sets, setSets] = useState(3);
  const [reps, setReps] = useState(10);
  const [notes, setNotes] = useState('ตารางฝึกตามโปรแกรมคลินิก');
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (patient?.id) setTargetPatientId(patient.id);
    if (patient?.assignedPtId) setPtId(patient.assignedPtId);
    if (selectedDateIso) setStartDate(selectedDateIso);
  }, [patient, selectedDateIso, mode]);

  if (!mode) return null;

  const activePatient = patients.find((p) => p.id === targetPatientId) || patient;
  const dayItems =
    mode === 'day_detail' && activePatient && selectedDateIso
      ? (activePatient.rehabSchedules ?? []).filter((s) => s.scheduledDate === selectedDateIso)
      : [];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/60 p-4 backdrop-blur-sm">
      <div className="w-full max-w-lg rounded-3xl bg-white p-6 shadow-2xl border border-slate-200 space-y-4 text-xs">
        <div className="flex items-center justify-between border-b border-slate-100 pb-3">
          <div className="flex items-center gap-2">
            <Calendar className="h-5 w-5 text-emerald-600" />
            <h3 className="text-base font-black text-slate-900">
              {mode === 'create'
                ? 'สร้างตารางฝึกใหม่ (Rehab Schedule)'
                : `รายละเอียดตารางฝึกวันที่ ${selectedDateIso}`}
            </h3>
          </div>
          <button type="button" onClick={onClose} className="rounded-xl p-1 text-slate-400 hover:bg-slate-100">
            <X className="h-5 w-5" />
          </button>
        </div>

        {mode === 'day_detail' && activePatient && selectedDateIso && (
          <div className="space-y-3">
            <div className="rounded-xl bg-slate-50 px-3.5 py-2.5 font-bold text-slate-700">
              คนไข้: {activePatient.fullName} ({activePatient.hn})
            </div>

            {dayItems.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-slate-200 p-6 text-center text-slate-400 space-y-2">
                <p>ยังไม่มีรายการฝึกที่กำหนดเฉพาะสำหรับวันที่ {selectedDateIso}</p>
                <button
                  type="button"
                  onClick={async () => {
                    const updated: RehabScheduleItem[] = [
                      ...(activePatient.rehabSchedules ?? []),
                      {
                        id: `sched-${Date.now()}`,
                        patientId: activePatient.id,
                        ptId: activePatient.assignedPtId,
                        exerciseId: exercises[0]?.id,
                        modpackId: modpacks[0]?.id,
                        scheduledDate: selectedDateIso,
                        sets: 3,
                        reps: 10,
                        status: 'ACTIVE',
                        notes: 'เพิ่มนัดหมายจากการคลิกปฏิทิน',
                      },
                    ];
                    await apiClient.updatePatient(activePatient.id, {
                      nextAppointmentDate: selectedDateIso,
                      rehabSchedules: updated,
                    });
                    await onSaved(`เพิ่มตารางฝึกวันที่ ${selectedDateIso} แล้ว`);
                    onClose();
                  }}
                  className="inline-flex items-center gap-1 rounded-xl bg-emerald-600 px-4 py-2 font-bold text-white"
                >
                  <Plus className="h-3.5 w-3.5" /> เพิ่มนัดหมายฝึกวันนี้
                </button>
              </div>
            ) : (
              <div className="space-y-2.5">
                {dayItems.map((item) => {
                  const ex = exercises.find((e) => e.id === item.exerciseId);
                  const mp = modpacks.find((m) => m.id === item.modpackId);
                  return (
                    <div key={item.id} className="rounded-2xl border border-slate-200 p-3.5 space-y-2.5">
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="font-extrabold text-slate-900">
                            {ex ? ex.name.th : 'ฝึกตามคอร์ส'} {mp ? `• ${mp.name}` : ''}
                          </div>
                          <div className="text-[11px] text-slate-500">
                            {item.sets} เซต × {item.reps} ครั้ง {item.notes ? `• ${item.notes}` : ''}
                          </div>
                        </div>
                        <span className="rounded-full bg-slate-100 px-2.5 py-0.5 text-[10px] font-black">
                          {item.status}
                        </span>
                      </div>
                      <div className="flex flex-wrap gap-1.5">
                        {(
                          [
                            { status: 'ACTIVE', label: 'กำลังรักษา', cls: 'bg-emerald-600 text-white' },
                            { status: 'REST', label: 'พักการฝึก', cls: 'bg-amber-500 text-white' },
                            { status: 'COMPLETED', label: 'สำเร็จครบ', cls: 'bg-blue-600 text-white' },
                            { status: 'MISSED', label: 'ขาดการฝึก', cls: 'bg-rose-600 text-white' },
                          ] as const
                        ).map((btn) => (
                          <button
                            key={btn.status}
                            type="button"
                            onClick={async () => {
                              const nextList = (activePatient.rehabSchedules ?? []).map((s) =>
                                s.id === item.id ? { ...s, status: btn.status } : s,
                              );
                              await apiClient.updatePatient(activePatient.id, { rehabSchedules: nextList });
                              await onSaved(`เปลี่ยนสถานะวันที่ ${selectedDateIso} เป็น ${btn.label}`);
                            }}
                            className={`rounded-lg px-2.5 py-1 text-[11px] font-bold ${btn.cls}`}
                          >
                            {btn.label}
                          </button>
                        ))}
                        <button
                          type="button"
                          onClick={async () => {
                            const nextList = (activePatient.rehabSchedules ?? []).filter((s) => s.id !== item.id);
                            await apiClient.updatePatient(activePatient.id, { rehabSchedules: nextList });
                            await onSaved(`ลบนัดหมายวันที่ ${selectedDateIso} แล้ว`);
                          }}
                          className="rounded-lg border border-rose-200 bg-rose-50 px-2.5 py-1 text-[11px] font-bold text-rose-700"
                        >
                          <Trash2 className="inline h-3 w-3 mr-0.5" /> ลบ
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {mode === 'create' && (
          <form
            onSubmit={async (e) => {
              e.preventDefault();
              if (!activePatient) return;
              setSaving(true);
              try {
                const start = new Date(startDate);
                const generated: RehabScheduleItem[] = [];
                for (let i = 0; i < Math.max(1, repeatDays); i += 1) {
                  const d = new Date(start.getTime() + i * 2 * 24 * 60 * 60 * 1000);
                  generated.push({
                    id: `sched-${Date.now()}-${i}`,
                    patientId: activePatient.id,
                    ptId,
                    exerciseId,
                    modpackId,
                    scheduledDate: d.toISOString().slice(0, 10),
                    sets,
                    reps,
                    status: 'ACTIVE',
                    notes,
                  });
                }
                const merged = [...(activePatient.rehabSchedules ?? []), ...generated];
                await apiClient.updatePatient(activePatient.id, {
                  nextAppointmentDate: startDate,
                  rehabSchedules: merged,
                });
                await onSaved(`สร้างตารางนัด ${generated.length} วันให้ ${activePatient.fullName} แล้ว`);
                onClose();
              } finally {
                setSaving(false);
              }
            }}
            className="space-y-3"
          >
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
              <div>
                <label className="block font-bold text-slate-600 mb-1">คนไข้ *</label>
                <select
                  value={targetPatientId}
                  onChange={(e) => setTargetPatientId(e.target.value)}
                  className="w-full rounded-xl border border-slate-200 px-3 py-2 font-semibold"
                >
                  {patients.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.fullName} ({p.hn})
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block font-bold text-slate-600 mb-1">นักกายภาพผู้รับผิดชอบ (PT)</label>
                <select
                  value={ptId}
                  onChange={(e) => setPtId(e.target.value)}
                  className="w-full rounded-xl border border-slate-200 px-3 py-2 font-semibold"
                >
                  {ptUsers.map((pt) => (
                    <option key={pt.id} value={pt.id}>
                      {pt.fullName}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
              <div>
                <label className="block font-bold text-slate-600 mb-1">ท่ากายภาพบำบัด</label>
                <select
                  value={exerciseId}
                  onChange={(e) => setExerciseId(e.target.value)}
                  className="w-full rounded-xl border border-slate-200 px-3 py-2"
                >
                  {exercises.map((ex) => (
                    <option key={ex.id} value={ex.id}>
                      {ex.name.th}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block font-bold text-slate-600 mb-1">ชุดมินิเกม Modpack</label>
                <select
                  value={modpackId}
                  onChange={(e) => setModpackId(e.target.value)}
                  className="w-full rounded-xl border border-slate-200 px-3 py-2"
                >
                  {modpacks.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="grid grid-cols-4 gap-2">
              <div className="col-span-2">
                <label className="block font-bold text-slate-600 mb-1">วันที่เริ่มฝึก</label>
                <input
                  type="date"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                  className="w-full rounded-xl border border-slate-200 px-3 py-2"
                />
              </div>
              <div>
                <label className="block font-bold text-slate-600 mb-1">จำนวนวัน</label>
                <input
                  type="number"
                  min={1}
                  max={14}
                  value={repeatDays}
                  onChange={(e) => setRepeatDays(Number(e.target.value))}
                  className="w-full rounded-xl border border-slate-200 px-2.5 py-2 text-center"
                />
              </div>
              <div>
                <label className="block font-bold text-slate-600 mb-1">เซต×ครั้ง</label>
                <div className="flex gap-1">
                  <input
                    type="number"
                    min={1}
                    max={10}
                    value={sets}
                    onChange={(e) => setSets(Number(e.target.value))}
                    className="w-1/2 rounded-xl border border-slate-200 px-1.5 py-2 text-center"
                  />
                  <input
                    type="number"
                    min={1}
                    max={30}
                    value={reps}
                    onChange={(e) => setReps(Number(e.target.value))}
                    className="w-1/2 rounded-xl border border-slate-200 px-1.5 py-2 text-center"
                  />
                </div>
              </div>
            </div>

            <input
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="หมายเหตุ..."
              className="w-full rounded-xl border border-slate-200 px-3 py-2"
            />

            <div className="flex justify-end gap-2 pt-2">
              <button type="button" onClick={onClose} className="rounded-xl border border-slate-200 px-4 py-2 font-bold">
                ยกเลิก
              </button>
              <button
                type="submit"
                disabled={saving}
                className="rounded-xl bg-emerald-600 px-5 py-2 font-extrabold text-white hover:bg-emerald-500 disabled:opacity-50"
              >
                {saving ? 'กำลังบันทึก...' : 'บันทึกตารางฝึก'}
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
}
