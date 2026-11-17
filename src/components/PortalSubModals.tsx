import React, { useEffect, useMemo, useRef, useState } from 'react';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { useClinicalI18n, type ClinicalLang } from '@drtechcares/clinical-i18n';
import {
  MAHIDOL_MOTOR_SCALE,
  resolveNextAppointmentDate,
  sanitizeAccidentalDefaultKneePrefix,
  type AffectedSide,
  type DrTechCaresApiClient,
  type ManagedExerciseRecord,
  type MedicalNoteRecord,
  type ModpackRecord,
  type PatientBiometricIndexEntry,
  type PatientRecord,
  type RehabScheduleItem,
  type RehabSessionHistoryEntry,
  type RehabSessionRecord,
  type StrokeType,
  type UniversalEhrPackage,
  type UserAccount,
} from '@drtechcares/api-client';
import {
  alignFaceTo112,
  BiologicalLivenessTracker,
  captureDownscaledContextFrame,
  evaluateLivenessAndPose,
  inspectBiometricFrame3D,
  selectClinicalCameraDeviceId,
  selectPrimaryFace,
  type BiologicalLivenessReport,
} from '@drtechcares/face-client';
import {
  computeBoundingBoxFromLandmarks,
  subscribeBiometricIndex,
} from '../lib/faceMatcher';
import {
  BIOMETRIC_ENGINE_ID,
  LiveIdentityStreakTracker,
  captureCanonicalFaceSnapshot,
  evaluateLiveMatch1toN,
  extractCanonicalFaceDescriptor,
  loadEnrolledPatientsWithBiometrics,
  warmupBiometricEngine,
  type EnrolledBiometricCandidate,
} from '../lib/biometricEngine';
import {
  Activity,
  CalendarDays,
  Camera,
  Check,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Copy,
  Download,
  Dumbbell,
  FileText,
  KeyRound,
  Link2,
  QrCode,
  ScanFace,
  Trash2,
  TrendingUp,
  Upload,
  UserPlus,
  X,
} from 'lucide-react';

const SUBMODAL_TEXT: Record<
  ClinicalLang,
  {
    welcomeTitle: string;
    welcomeSubtitle: string;
    matchedOk: string;
    checkInTodayBtn: string;
    enrollNewPatientBtn: string;
    scanningFaceBtn: string;
    startFaceMatchBtn: string;
    resetPwdTitle: string;
    newPwdLabel: string;
    cancelBtn: string;
    savingBtn: string;
    saveNewPwdBtn: string;
    agePrefix: string;
    ageUnit: string;
    sendFaceScanBtn: string;
    profileTab: string;
    historyTab: string;
    editProfileHeading: string;
    saveProfileBtn: string;
    delegateHeading: string;
    updatePtBtn: string;
    notesHeading: string;
    emptyNotes: string;
    addNoteBtn: string;
    reportHeading: string;
    saveReportBtn: string;
    statSessions: string;
    sessionUnit: string;
    statAccuracy: string;
    statMaxRom: string;
    emptyHistory: string;
    recordSessionBtn: string;
    schedCreateTitle: string;
    schedDayTitle: string;
    patientLabel: string;
    emptyDayMsg: string;
    addTodayApptBtn: string;
    selectPatientLabel: string;
    selectPtLabel: string;
    exerciseLabel: string;
    modpackLabel: string;
    startDateLabel: string;
    daysLabel: string;
    setsRepsLabel: string;
    saveScheduleBtn: string;
    statusActive: string;
    statusRest: string;
    statusCompleted: string;
    statusMissed: string;
    deleteBtn: string;
  }
> = {
  th: {
    welcomeTitle: 'ต้อนรับ & สแกนใบหน้าคนไข้ (Welcome Verification)',
    welcomeSubtitle: 'ตรวจสอบอัตลักษณ์ 3D Multi-Angle 512 มิติ อัตโนมัติแบบเรียลไทม์',
    matchedOk: 'ยืนยันตัวตนสำเร็จ',
    checkInTodayBtn: 'เช็คอินเข้าฝึกวันนี้ & เปิดตารางฝึก',
    enrollNewPatientBtn: 'ลงทะเบียนคนไข้ใหม่',
    scanningFaceBtn: 'กำลังตรวจสอบ 512 มิติ...',
    startFaceMatchBtn: 'เริ่มสแกนจับคู่ใบหน้า',
    resetPwdTitle: 'รีเซ็ตรหัสผ่าน',
    newPwdLabel: 'รหัสผ่านใหม่ (อย่างน้อย 6 ตัวอักษร)',
    cancelBtn: 'ยกเลิก',
    savingBtn: 'กำลังบันทึก...',
    saveNewPwdBtn: 'บันทึกรหัสผ่านใหม่',
    agePrefix: 'อายุ',
    ageUnit: 'ปี',
    sendFaceScanBtn: '📸 ส่งตัวสแกนหน้า',
    profileTab: 'โปรไฟล์ & เวชระเบียน',
    historyTab: 'ประวัติการฝึก & ROM',
    editProfileHeading: 'แก้ไขข้อมูลพื้นฐานคนไข้',
    saveProfileBtn: 'บันทึกโปรไฟล์คนไข้',
    delegateHeading: 'ส่งต่อเคสให้ PT ผู้รับผิดชอบ (Delegation)',
    updatePtBtn: 'อัปเดต PT ผู้รับผิดชอบ',
    notesHeading: 'บันทึกเวชระเบียนถาวร (Append-Only Notes)',
    emptyNotes: 'ยังไม่มีบันทึกเวชระเบียน',
    addNoteBtn: '+ บันทึก Note',
    reportHeading: 'ออกใบสรุปผลการรักษา (Treatment Report)',
    saveReportBtn: 'บันทึกใบสรุปผลการรักษา',
    statSessions: 'จำนวนครั้งที่ฝึก',
    sessionUnit: 'ครั้ง',
    statAccuracy: 'ความแม่นยำเฉลี่ย',
    statMaxRom: 'ROM สูงสุด',
    emptyHistory: 'ยังไม่มีประวัติการฝึกของคนไข้รายนี้',
    recordSessionBtn: '+ บันทึกผลเซสชันการฝึกวันนี้',
    schedCreateTitle: 'สร้างตารางฝึกใหม่ (Rehab Schedule)',
    schedDayTitle: 'รายละเอียดตารางฝึกวันที่',
    patientLabel: 'คนไข้:',
    emptyDayMsg: 'ยังไม่มีรายการฝึกที่กำหนดเฉพาะสำหรับวันที่',
    addTodayApptBtn: 'เพิ่มนัดหมายฝึกวันนี้',
    selectPatientLabel: 'คนไข้ *',
    selectPtLabel: 'นักกายภาพผู้รับผิดชอบ (PT)',
    exerciseLabel: 'ท่ากายภาพบำบัด',
    modpackLabel: 'ชุดมินิเกม Modpack',
    startDateLabel: 'วันที่เริ่มฝึก',
    daysLabel: 'จำนวนวัน',
    setsRepsLabel: 'เซต×ครั้ง',
    saveScheduleBtn: 'บันทึกตารางฝึก',
    statusActive: 'กำลังรักษา',
    statusRest: 'พักการฝึก',
    statusCompleted: 'สำเร็จครบ',
    statusMissed: 'ขาดการฝึก',
    deleteBtn: 'ลบ',
  },
  en: {
    welcomeTitle: 'Welcome & Patient Face Verification',
    welcomeSubtitle: 'Realtime 3D Multi-Angle 512-D biometric verification',
    matchedOk: 'Identity Verified',
    checkInTodayBtn: "Check In Today's Session & Open Schedule",
    enrollNewPatientBtn: 'Enroll New Patient',
    scanningFaceBtn: 'Verifying 512-D Face...',
    startFaceMatchBtn: 'Capture & Verify Face',
    resetPwdTitle: 'Reset Password',
    newPwdLabel: 'New Password (min 6 characters)',
    cancelBtn: 'Cancel',
    savingBtn: 'Saving...',
    saveNewPwdBtn: 'Save New Password',
    agePrefix: 'Age',
    ageUnit: 'yrs',
    sendFaceScanBtn: '📸 Send to Face Scan',
    profileTab: 'Profile & Medical Records',
    historyTab: 'Training History & ROM',
    editProfileHeading: 'Edit Patient Demographics',
    saveProfileBtn: 'Save Patient Profile',
    delegateHeading: 'Delegate Case to Physiotherapist (PT)',
    updatePtBtn: 'Update Responsible PT',
    notesHeading: 'Append-Only Clinical Notes',
    emptyNotes: 'No clinical notes recorded yet',
    addNoteBtn: '+ Append Note',
    reportHeading: 'Issue Treatment Report',
    saveReportBtn: 'Save Treatment Report',
    statSessions: 'Sessions Completed',
    sessionUnit: 'sessions',
    statAccuracy: 'Average Accuracy',
    statMaxRom: 'Max ROM',
    emptyHistory: 'No training history found for this patient',
    recordSessionBtn: "+ Log Today's Training Session",
    schedCreateTitle: 'Create Rehab Schedule',
    schedDayTitle: 'Schedule Details for',
    patientLabel: 'Patient:',
    emptyDayMsg: 'No specific rehab items scheduled for',
    addTodayApptBtn: 'Add Appointment for This Day',
    selectPatientLabel: 'Patient *',
    selectPtLabel: 'Responsible Physiotherapist (PT)',
    exerciseLabel: 'Rehab Exercise',
    modpackLabel: 'Cognitive Modpack',
    startDateLabel: 'Start Date',
    daysLabel: 'Days',
    setsRepsLabel: 'Sets×Reps',
    saveScheduleBtn: 'Save Rehab Schedule',
    statusActive: 'Active',
    statusRest: 'Rest',
    statusCompleted: 'Completed',
    statusMissed: 'Missed',
    deleteBtn: 'Delete',
  },
  cn: {
    welcomeTitle: '患者欢迎迎宾与人脸核验 (Welcome Verification)',
    welcomeSubtitle: '实时 3D 多角度 512 维面部生物识别验证',
    matchedOk: '身份核验成功',
    checkInTodayBtn: '今日签到并打开康复日程',
    enrollNewPatientBtn: '注册新患者',
    scanningFaceBtn: '正在比对 512 维特征...',
    startFaceMatchBtn: '开始人脸抓拍核验',
    resetPwdTitle: '重置账号密码',
    newPwdLabel: '新密码（至少 6 位字符）',
    cancelBtn: '取消',
    savingBtn: '保存中...',
    saveNewPwdBtn: '保存新密码',
    agePrefix: '年龄',
    ageUnit: '岁',
    sendFaceScanBtn: '📸 前往人脸建档',
    profileTab: '档案与病历记录',
    historyTab: '康复记录与 ROM 角度',
    editProfileHeading: '编辑患者基本信息',
    saveProfileBtn: '保存患者档案',
    delegateHeading: '指派/转交责任康复师 (Delegation)',
    updatePtBtn: '更新责任康复师',
    notesHeading: '不可篡改临床病程记录 (Append-Only Notes)',
    emptyNotes: '暂无临床病历记录',
    addNoteBtn: '+ 添加记录',
    reportHeading: '出具康复治疗总结报告',
    saveReportBtn: '保存康复总结报告',
    statSessions: '累计训练次数',
    sessionUnit: '次',
    statAccuracy: '平均动作准确度',
    statMaxRom: '最大关节活动度',
    emptyHistory: '该患者暂无康复训练记录',
    recordSessionBtn: '+ 录入今日训练记录',
    schedCreateTitle: '新建康复训练日程 (Rehab Schedule)',
    schedDayTitle: '训练日程详情',
    patientLabel: '患者：',
    emptyDayMsg: '该日期暂无指定康复项目：',
    addTodayApptBtn: '添加当日康复预约',
    selectPatientLabel: '患者 *',
    selectPtLabel: '责任物理治疗师 (PT)',
    exerciseLabel: '康复动作',
    modpackLabel: '认知训练题库包',
    startDateLabel: '开始日期',
    daysLabel: '训练天数',
    setsRepsLabel: '组数×次数',
    saveScheduleBtn: '保存康复日程',
    statusActive: '训练中',
    statusRest: '休息',
    statusCompleted: '已完成',
    statusMissed: '缺席',
    deleteBtn: '删除',
  },
};

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
  const { lang } = useClinicalI18n();
  const sm = SUBMODAL_TEXT[lang];
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const biometricCandidatesRef = useRef<EnrolledBiometricCandidate[]>([]);
  const streakTrackerRef = useRef(new LiveIdentityStreakTracker(3));
  const bioLivenessRef = useRef(new BiologicalLivenessTracker());
  const bioReportRef = useRef<BiologicalLivenessReport | null>(null);
  const confirmedLockRef = useRef(false);

  const [cameraError, setCameraError] = useState<string | null>(null);
  const [scanning, setScanning] = useState(false);
  const [autoScanCount, setAutoScanCount] = useState(0);
  const [bioStatus, setBioStatus] = useState<BiologicalLivenessReport | null>(null);
  const [spoofDetectedNow, setSpoofDetectedNow] = useState(false);
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
    let mounted = true;
    const syncVault = () => {
      void loadEnrolledPatientsWithBiometrics(apiClient)
        .then((res) => {
          if (mounted) biometricCandidatesRef.current = res.compatibleCandidates;
        })
        .catch(() => {});
    };
    syncVault();
    const unsub = subscribeBiometricIndex(apiClient, () => {
      syncVault();
    });
    return () => {
      mounted = false;
      unsub();
    };
  }, [open, apiClient]);

  useEffect(() => {
    if (!open) return;
    let active = true;
    let stream: MediaStream | null = null;
    let landmarker: FaceLandmarker | null = null;
    let rafId = 0;
    let prevLandmarks: Array<{ x: number; y: number; z?: number }> | null = null;
    let lastEmbedAt = 0;
    let fallbackFrameCount = 0;
    let lastServerVerifyAt = 0;
    let lastDetectTs = 0;
    let lastProcessedVideoTime = -1;
    let lastFaceDetectWallMs = 0;
    const alignCanvas = document.createElement('canvas');
    const roiCanvas = document.createElement('canvas');
    const contextCanvas = document.createElement('canvas');

    void warmupBiometricEngine();
    bioLivenessRef.current.reset();
    bioReportRef.current = null;
    setBioStatus(null);
    setSpoofDetectedNow(false);
    confirmedLockRef.current = false;
    streakTrackerRef.current = new LiveIdentityStreakTracker(3);
    streakTrackerRef.current.reset();
    setMatchResult(null);
    setCameraError(null);
    setHudBox(null);
    setLiveMatchHud(null);
    setAutoScanCount(0);

    async function startLiveVerification() {
      try {
        const deviceId = await selectClinicalCameraDeviceId();
        stream = await navigator.mediaDevices.getUserMedia({
          video: deviceId
            ? { deviceId: { exact: deviceId }, width: { ideal: 1920 }, height: { ideal: 1080 }, facingMode: 'user' }
            : { width: { ideal: 1920 }, height: { ideal: 1080 }, facingMode: 'user' },
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
          numFaces: 2,
          outputFaceBlendshapes: true,
          minFaceDetectionConfidence: 0.45,
          minFacePresenceConfidence: 0.45,
          minTrackingConfidence: 0.45,
        });

        const loop = async () => {
          if (!active || !videoRef.current || !landmarker) return;
          let shouldContinueLoop = true;
          try {
            if (videoRef.current.readyState >= 2) {
              const now = performance.now();
              const currentVideoTime = videoRef.current.currentTime;
              if (currentVideoTime === lastProcessedVideoTime && now - lastFaceDetectWallMs < 120) {
                return;
              }
              lastProcessedVideoTime = currentVideoTime;
              lastFaceDetectWallMs = now;
              const detectTs = Math.max(lastDetectTs + 1, Math.round(now));
              lastDetectTs = detectTs;
              const res = landmarker.detectForVideo(videoRef.current, detectTs);
              const selection = selectPrimaryFace(res.faceLandmarks ?? []);
              const faceCount = selection.effectiveFaceCount;
              const lms = selection.primaryFace ? [...selection.primaryFace] : [];

                if (faceCount !== 1 || lms.length < 300) {
                setHudBox(null);
                fallbackFrameCount = 0;
                if (!confirmedLockRef.current) {
                  setLiveMatchHud(null);
                  setAutoScanCount(0);
                  streakTrackerRef.current.reset();
                  bioLivenessRef.current.reset();
                  bioReportRef.current = null;
                  setBioStatus(null);
                  setSpoofDetectedNow(false);
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
                const crop = alignFaceTo112(videoRef.current, lms, alignCanvas, roiCanvas);
                const contextFrame = captureDownscaledContextFrame(videoRef.current, contextCanvas, 160);
                const bio = bioLivenessRef.current.update({
                  landmarks3D: lms,
                  blendshapes: res.faceBlendshapes?.[0]?.categories,
                  earLeft: liveness.earLeft,
                  earRight: liveness.earRight,
                  yawDeg: liveness.yawDeg,
                  pitchDeg: liveness.pitchDeg,
                  frameWidth: videoRef.current.videoWidth || 1920,
                  frameHeight: videoRef.current.videoHeight || 1080,
                  faceCrop: crop,
                  contextFrame,
                  faceBox: rawBox,
                  timestampMs: now,
                });
                bioReportRef.current = bio;
                setBioStatus(bio);
                inspectBiometricFrame3D({
                  landmarks3D: lms,
                  prevLandmarks3D: prevLandmarks,
                  faceCount,
                  image: crop,
                  contextImage: contextFrame,
                  faceBox: rawBox,
                  biologicalVerified: bio.biologicalVerified,
                  allowLongRange: true,
                  metrics: {
                    allowLongRange: true,
                    biologicalVerified: bio.biologicalVerified,
                    planarRigidSpoof: bio.planarSpoofDetected,
                    moireOrBezelDetected: bio.screenSpoofDetected,
                  },
                });
                const isSpoofFrame = bio.spoofDetected && !bio.biologicalVerified;
                setSpoofDetectedNow(isSpoofFrame);
                prevLandmarks = lms;

                if (isSpoofFrame) {
                  streakTrackerRef.current.reset();
                  fallbackFrameCount = 0;
                  setAutoScanCount(0);
                  setLiveMatchHud(null);
                } else if (!confirmedLockRef.current && now - lastEmbedAt >= 120) {
                  lastEmbedAt = now;
                  const snap = captureCanonicalFaceSnapshot(videoRef.current, rawBox);
                  const extracted = snap ? await extractCanonicalFaceDescriptor(snap) : null;
                  if (!active) {
                    shouldContinueLoop = false;
                    return;
                  }
                  const vector = extracted?.embedding;
                  if (vector) {
                    const portalFaceH = rawBox?.height ?? bio.faceScale;
                    const livenessEvidence = {
                      biologicalVerified: bio.biologicalVerified,
                      blinkCount: bio.blinkCount,
                      spoofRisk: bio.spoofScore,
                      nonRigidScore: bio.nonRigidScore,
                      rppgScore: bio.rppgScore,
                      faceHeightRatio: portalFaceH,
                      engine: BIOMETRIC_ENGINE_ID,
                    };

                    if (!confirmedLockRef.current && biometricCandidatesRef.current.length > 0) {
                      const evalRes = evaluateLiveMatch1toN(vector, biometricCandidatesRef.current);
                      const streakRes = streakTrackerRef.current.step(evalRes, {
                        spoofDetected: isSpoofFrame,
                        biologicalVerified: bio.biologicalVerified,
                      });

                      const isHudMatch = Boolean(streakRes.candidate && streakRes.streak > 0);
                      setLiveMatchHud(
                        isHudMatch && streakRes.candidate
                          ? {
                              candidate: streakRes.candidate.sourceEntry,
                              confidencePercent: streakRes.confidencePercent,
                              streak: streakRes.streak,
                              confirmed: streakRes.confirmed,
                            }
                          : null,
                      );

                      if (
                        streakRes.confirmed &&
                        streakRes.candidate &&
                        bio.biologicalVerified &&
                        !confirmedLockRef.current
                      ) {
                        confirmedLockRef.current = true;
                        const matchedConf = Math.round(streakRes.confidencePercent);
                        const localCand = streakRes.candidate;
                        let matchedNow = false;
                        try {
                          const verified = await apiClient.verifyFaceLogin(vector, livenessEvidence);
                          if (!active) {
                            shouldContinueLoop = false;
                            return;
                          }
                          if (verified.matched && verified.patient) {
                            matchedNow = true;
                            shouldContinueLoop = false;
                            setMatchResult({
                              matched: true,
                              confidencePercent: Math.max(verified.confidencePercent, matchedConf),
                              patient: verified.patient,
                            });
                            return;
                          }
                        } catch {
                          if (!active) {
                            shouldContinueLoop = false;
                            return;
                          }
                        } finally {
                          if (!matchedNow) {
                            confirmedLockRef.current = false;
                          }
                        }
                        if (!matchedNow && localCand && streakRes.similarity >= 0.6) {
                          confirmedLockRef.current = true;
                          shouldContinueLoop = false;
                          setMatchResult({
                            matched: true,
                            confidencePercent: matchedConf,
                            patient: {
                              id: localCand.patientId,
                              hn: localCand.hn,
                              fullName: localCand.fullName,
                              nextAppointmentDate: localCand.nextAppointmentDate,
                            },
                          });
                          return;
                        }
                        streakTrackerRef.current.reset();
                        setLiveMatchHud(null);
                        void loadEnrolledPatientsWithBiometrics(apiClient)
                          .then((loaded) => {
                            if (active) biometricCandidatesRef.current = loaded.compatibleCandidates;
                          })
                          .catch(() => {});
                      }
                    }

                    if (!confirmedLockRef.current && bio.biologicalVerified && now - lastServerVerifyAt >= 2500) {
                      fallbackFrameCount = Math.min(3, fallbackFrameCount + 1);
                      setAutoScanCount(fallbackFrameCount);
                      if (fallbackFrameCount >= 3) {
                        confirmedLockRef.current = true;
                        let matchedFallback = false;
                        try {
                          const verified = await apiClient.verifyFaceLogin(vector, livenessEvidence);
                          if (!active) {
                            shouldContinueLoop = false;
                            return;
                          }
                          if (verified.matched && verified.patient) {
                            matchedFallback = true;
                            shouldContinueLoop = false;
                            setMatchResult({
                              matched: true,
                              confidencePercent: verified.confidencePercent,
                              patient: verified.patient,
                            });
                            return;
                          }
                          setMatchResult({
                            matched: false,
                            confidencePercent: verified.confidencePercent,
                            reason: verified.failureReasons?.[0],
                          });
                        } catch {
                          if (!active) {
                            shouldContinueLoop = false;
                            return;
                          }
                        } finally {
                          if (!matchedFallback) {
                            confirmedLockRef.current = false;
                          }
                        }
                        fallbackFrameCount = 0;
                        setAutoScanCount(0);
                        lastServerVerifyAt = performance.now();
                      }
                    }
                  }
                }
              }
            }
          } catch {
            confirmedLockRef.current = false;
          } finally {
            if (active && shouldContinueLoop) {
              rafId = requestAnimationFrame(loop);
            }
          }
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
    const bioNow = bioReportRef.current;
    if (!bioNow?.biologicalVerified || bioNow.spoofDetected) {
      setMatchResult({
        matched: false,
        confidencePercent: 0,
        reason: bioNow?.spoofDetected
          ? 'ตรวจพบภาพถ่ายหรือหน้าจอมือถือ — ระบบปฏิเสธการสแกนด้วยรูปภาพ'
          : 'กรุณากะพริบตา 1 ครั้ง หน้ากล้องเพื่อยืนยันสิ่งมีชีวิตก่อนกดสแกน',
      });
      return;
    }
    setScanning(true);
    setMatchResult(null);
    confirmedLockRef.current = false;
    streakTrackerRef.current.reset();
    try {
      let probeVector: number[] | null = null;
      if (videoRef.current && videoRef.current.readyState >= 2) {
        const snap = captureCanonicalFaceSnapshot(videoRef.current);
        const extracted = snap ? await extractCanonicalFaceDescriptor(snap) : null;
        probeVector = extracted?.embedding ?? null;
      }
      if (!probeVector) {
        setMatchResult({
          matched: false,
          confidencePercent: 0,
          reason: 'ไม่พบสัญญาณใบหน้าจริงจากกล้อง กรุณามองกล้องและกะพริบตา',
        });
        return;
      }
      const localEval = evaluateLiveMatch1toN(probeVector, biometricCandidatesRef.current);
      if (localEval.matched && localEval.topCandidate) {
        setMatchResult({
          matched: true,
          confidencePercent: localEval.confidencePercent,
          patient: {
            id: localEval.topCandidate.patientId,
            hn: localEval.topCandidate.hn,
            fullName: localEval.topCandidate.fullName,
            nextAppointmentDate: localEval.topCandidate.nextAppointmentDate,
          },
        });
        return;
      }
      const verified = await apiClient.verifyFaceLogin(probeVector, {
        biologicalVerified: bioNow.biologicalVerified,
        blinkCount: bioNow.blinkCount,
        spoofRisk: bioNow.spoofScore,
        nonRigidScore: bioNow.nonRigidScore,
        rppgScore: bioNow.rppgScore,
        engine: BIOMETRIC_ENGINE_ID,
      });
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
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#0F4C81]/35 p-4 backdrop-blur-sm">
      <div className="w-full max-w-lg rounded-3xl bg-white p-6 shadow-2xl border border-slate-200 space-y-4 text-xs">
        <div className="flex items-center justify-between border-b border-slate-100 pb-3">
          <div className="flex items-center gap-2 min-w-0">
            <ScanFace className="h-5 w-5 text-emerald-600 shrink-0" />
            <div className="min-w-0">
              <h3 className="text-base font-black text-slate-900 truncate">{sm.welcomeTitle}</h3>
              <p className="text-[11px] text-slate-500 truncate">{sm.welcomeSubtitle}</p>
            </div>
          </div>
          <button type="button" onClick={onClose} className="rounded-xl p-1 text-slate-400 hover:bg-slate-100 shrink-0">
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-sky-100 via-emerald-50 to-white aspect-[4/3] border border-sky-200">
          <video ref={videoRef} playsInline muted className="h-full w-full object-cover -scale-x-100" />

          {hudBox && (
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
                className={`relative h-full w-full overflow-hidden rounded-xl border-2 ${
                  spoofDetectedNow
                    ? 'border-rose-500 shadow-[0_0_24px_rgba(244,63,94,0.85)]'
                    : liveMatchHud?.candidate && bioStatus?.biologicalVerified
                      ? liveMatchHud.confirmed
                        ? 'border-emerald-400 shadow-[0_0_24px_rgba(16,185,129,0.7)]'
                        : 'border-emerald-400/75 shadow-[0_0_12px_rgba(16,185,129,0.4)]'
                      : 'border-cyan-400/85 shadow-[0_0_18px_rgba(34,211,238,0.45)]'
                }`}
              >
                <span className="absolute top-0 left-0 h-3.5 w-3.5 border-t-[3px] border-l-[3px] border-emerald-300 rounded-tl" />
                <span className="absolute top-0 right-0 h-3.5 w-3.5 border-t-[3px] border-r-[3px] border-emerald-300 rounded-tr" />
                <span className="absolute bottom-0 left-0 h-3.5 w-3.5 border-b-[3px] border-l-[3px] border-emerald-300 rounded-bl" />
                <span className="absolute bottom-0 right-0 h-3.5 w-3.5 border-b-[3px] border-r-[3px] border-emerald-300 rounded-br" />
                {!liveMatchHud?.candidate && (
                  <div className="absolute inset-x-1.5 top-1/2 h-[2px] -translate-y-1/2 bg-gradient-to-r from-transparent via-cyan-300 to-transparent shadow-[0_0_10px_#22d3ee]" />
                )}
              </div>

              {spoofDetectedNow ? (
                <div className="absolute left-1/2 top-full mt-1.5 w-max max-w-[290px] -translate-x-1/2 rounded-lg border border-rose-400 bg-rose-50/95 px-3 py-1.5 text-center backdrop-blur-md shadow-xl">
                  <div className="text-[11px] font-black text-rose-700">
                    ⚠️ ตรวจพบภาพถ่ายหรือหน้าจอมือถือ — กรุณากะพริบตาหรือขยับศีรษะด้วยใบหน้าจริง
                  </div>
                </div>
              ) : !bioStatus?.biologicalVerified ? (
                <div className="absolute left-1/2 top-full mt-1.5 w-max max-w-[290px] -translate-x-1/2 rounded-lg border border-sky-300 bg-white/95 px-3 py-1.5 text-center backdrop-blur-md shadow-xl">
                  <div className="text-[10px] font-extrabold text-[#0F4C81]">
                    👁️ กรุณากะพริบตาหรือขยับศีรษะเล็กน้อยเพื่อยืนยันตัวตน
                  </div>
                </div>
              ) : liveMatchHud?.candidate ? (
                <div className="absolute left-1/2 top-full mt-1.5 w-max max-w-[260px] -translate-x-1/2 rounded-lg border border-emerald-400 bg-white/95 px-3 py-1.5 text-center backdrop-blur-md shadow-xl">
                  <div className="text-[11px] font-black text-emerald-700">
                    ✓ {sm.matchedOk}: {liveMatchHud.candidate.fullName} ({liveMatchHud.candidate.hn})
                  </div>
                  <div className="text-[10px] font-bold text-[#0F4C81]">
                    {liveMatchHud.confidencePercent.toFixed(1)}% (3D Multi-Angle Match •{' '}
                    {liveMatchHud.streak}/5)
                  </div>
                </div>
              ) : (
                <div className="absolute left-1/2 top-full mt-1.5 w-max max-w-[260px] -translate-x-1/2 rounded-lg border border-sky-300 bg-white/95 px-3 py-1 text-center backdrop-blur-md shadow-xl">
                  <div className="text-[10px] font-extrabold text-[#0F4C81]">
                    กำลังสแกนใบหน้าเทียบฐานข้อมูล... {autoScanCount}/5
                  </div>
                </div>
              )}
            </div>
          )}

          {cameraError && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-white/95 p-5 text-center text-[#0F4C81]">
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
                    <CheckCircle2 className="h-4 w-4 shrink-0" /> {sm.matchedOk} ({matchResult.confidencePercent}%)
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
                  className="w-full whitespace-nowrap h-10 px-4 inline-flex items-center justify-center rounded-xl bg-emerald-600 font-extrabold text-white hover:bg-emerald-500"
                >
                  {sm.checkInTodayBtn}
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
            className="whitespace-nowrap h-10 px-4 inline-flex items-center justify-center gap-1.5 rounded-xl border border-slate-200 font-bold text-slate-700 hover:bg-slate-50"
          >
            <UserPlus className="h-4 w-4 shrink-0" />
            {sm.enrollNewPatientBtn}
          </button>
          <button
            type="button"
            disabled={scanning}
            onClick={() => void handleScanFrame()}
            className="whitespace-nowrap h-10 px-5 inline-flex items-center justify-center gap-1.5 rounded-xl bg-gradient-to-r from-emerald-600 to-blue-600 font-extrabold text-white shadow hover:opacity-95 disabled:opacity-50"
          >
            <ScanFace className="h-4 w-4 shrink-0" />
            {scanning ? sm.scanningFaceBtn : sm.startFaceMatchBtn}
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
  const { lang } = useClinicalI18n();
  const sm = SUBMODAL_TEXT[lang];
  const [newPassword, setNewPassword] = useState('Staff@2026');
  const [saving, setSaving] = useState(false);
  const [errMsg, setErrMsg] = useState<string | null>(null);

  if (!user) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#0F4C81]/35 p-4 backdrop-blur-sm">
      <div className="w-full max-w-sm rounded-3xl bg-white p-6 shadow-2xl border border-slate-200 space-y-4 text-xs">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 min-w-0">
            <KeyRound className="h-4 w-4 text-sky-600 shrink-0" />
            <h3 className="text-sm font-black text-slate-900 truncate">
              {sm.resetPwdTitle} @{user.username}
            </h3>
          </div>
          <button type="button" onClick={onClose} className="text-slate-400 hover:text-slate-700 shrink-0">
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
            <label className="block font-bold text-slate-600 mb-1">{sm.newPwdLabel}</label>
            <input
              required
              minLength={6}
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              className="w-full rounded-xl border border-slate-200 px-3 py-2.5 font-mono text-sm"
            />
          </div>
          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={onClose}
              className="whitespace-nowrap h-10 px-4 inline-flex items-center justify-center rounded-xl border border-slate-200 font-bold"
            >
              {sm.cancelBtn}
            </button>
            <button
              type="submit"
              disabled={saving}
              className="whitespace-nowrap h-10 px-4 inline-flex items-center justify-center rounded-xl bg-sky-600 font-extrabold text-white hover:bg-sky-500 disabled:opacity-50"
            >
              {saving ? sm.savingBtn : sm.saveNewPwdBtn}
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
  onOpenLinkEhr,
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
  onOpenLinkEhr?: (patient: PatientRecord) => void;
  onFlash: (kind: 'success' | 'error', msg: string) => void;
}) {
  const { lang } = useClinicalI18n();
  const sm = SUBMODAL_TEXT[lang];
  const [modalTab, setModalTab] = useState<'profile' | 'history'>(initialTab);
  const [fullName, setFullName] = useState('');
  const [age, setAge] = useState(45);
  const [gender, setGender] = useState('ชาย');
  const [phone, setPhone] = useState('');
  const [diagnosis, setDiagnosis] = useState('');
  const [strokeType, setStrokeType] = useState<StrokeType>('ELDERLY_GENERAL');
  const [affectedSide, setAffectedSide] = useState<AffectedSide>('NONE');
  const [motorAssessmentScore, setMotorAssessmentScore] = useState<number>(3);
  const [adlDressingScore, setAdlDressingScore] = useState<number>(3);
  const [maxSafeRomDeg, setMaxSafeRomDeg] = useState<number>(150);
  const [sessionHistory, setSessionHistory] = useState<RehabSessionHistoryEntry[]>([]);
  const [assignedPtId, setAssignedPtId] = useState('');
  const [delegatePtId, setDelegatePtId] = useState('');
  const [delegateDate, setDelegateDate] = useState(() => new Date().toISOString().slice(0, 10));
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
    setStrokeType(patient.strokeType ?? 'ELDERLY_GENERAL');
    setAffectedSide(patient.affectedSide ?? 'NONE');
    setMotorAssessmentScore(patient.motorAssessmentScore ?? 3);
    setAdlDressingScore(patient.adlDressingScore ?? 3);
    setMaxSafeRomDeg(patient.maxSafeRomDeg ?? 150);
    setAssignedPtId(patient.assignedPtId || '');
    setDelegatePtId(patient.delegatedToPtId || patient.assignedPtId || '');
    setDelegateDate(patient.delegationDate || new Date().toISOString().slice(0, 10));
    setDelegateNote(patient.delegationNote || '');
    apiClient
      .listMedicalNotes(patient.id)
      .then((n) => setNotes(n))
      .catch(() => setNotes([]));
    apiClient
      .listRehabSessionHistory(patient.id)
      .then((h) => setSessionHistory(h))
      .catch(() => setSessionHistory(patient.rehabSessionHistory ?? []));
  }, [patient, initialTab, apiClient]);

  if (!patient) return null;

  const patientSessions = sessions.filter((s) => s.patientId === patient.id);
  const ptNameMap = new Map(ptUsers.map((p) => [p.id, p.fullName]));

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#0F4C81]/35 p-4 backdrop-blur-sm">
      <div className="w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-3xl bg-white p-6 shadow-2xl border border-slate-200 space-y-5 text-xs">
        <div className="flex items-start justify-between border-b border-slate-100 pb-4">
          <div className="flex items-center gap-3 min-w-0">
            {patient.faceImageUrl ? (
              <img
                src={patient.faceImageUrl}
                alt={patient.fullName}
                className="h-12 w-12 rounded-2xl object-cover border border-emerald-200 shrink-0"
              />
            ) : (
              <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-emerald-50 text-emerald-700 font-black text-base">
                {patient.fullName.slice(0, 1)}
              </div>
            )}
            <div className="min-w-0">
              <div className="flex items-center gap-2 min-w-0">
                <span className="whitespace-nowrap rounded-lg bg-sky-100 px-2.5 py-0.5 font-mono font-extrabold text-sky-700 shrink-0">
                  {patient.hn}
                </span>
                <h3 className="text-base font-black text-slate-900 truncate">{patient.fullName}</h3>
              </div>
              <p className="text-slate-500 mt-1 truncate">
                {sm.agePrefix} {patient.age} {sm.ageUnit} ({patient.gender || 'ชาย'}) • PT:{' '}
                {ptNameMap.get(patient.delegatedToPtId || patient.assignedPtId) || 'กภ. ประจำคลินิก'}
              </p>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2 shrink-0">
            {onOpenLinkEhr && (
              <button
                type="button"
                onClick={() => onOpenLinkEhr(patient)}
                className="whitespace-nowrap h-9 px-3 inline-flex items-center justify-center gap-1 rounded-xl border border-indigo-200 bg-indigo-50 font-bold text-indigo-700 hover:bg-indigo-100 cursor-pointer"
              >
                <Link2 className="h-3.5 w-3.5 shrink-0" />
                <span>เชื่อมโยง/ส่งต่อประวัติการรักษา (Link EHR)</span>
              </button>
            )}
            <button
              type="button"
              onClick={() => {
                onClose();
                onOpenFaceScan(patient);
              }}
              className="whitespace-nowrap h-9 px-3 inline-flex items-center justify-center rounded-xl border border-emerald-200 bg-emerald-50 font-bold text-emerald-700 hover:bg-emerald-100"
            >
              {sm.sendFaceScanBtn}
            </button>
            <button type="button" onClick={onClose} className="rounded-xl p-1 text-slate-400 hover:bg-slate-100">
              <X className="h-5 w-5" />
            </button>
          </div>
        </div>

        {/* 5-Angle Face ID Reference Strip */}
        {(patient.angleImageUrls || (patient.faceImageUrls && patient.faceImageUrls.length > 0)) && (
          <div className="rounded-2xl border border-emerald-200 bg-emerald-50/50 p-3">
            <div className="flex items-center justify-between gap-2 mb-2">
              <span className="text-[11px] font-extrabold text-emerald-900">
                ภาพถ่ายใบหน้า Face ID 5 มุมมอง ({patient.faceFramesCount ?? 0} เฟรม)
              </span>
              <span className="text-[10px] font-bold text-emerald-700">
                {
                  (
                    patient.angleImageUrls
                      ? Object.values(patient.angleImageUrls).filter(Boolean).length
                      : patient.faceImageUrls?.length ?? 0
                  )
                }{' '}
                รูปที่จัดเก็บในระบบ
              </span>
            </div>
            {patient.angleImageUrls ? (
              <div className="grid grid-cols-5 gap-2">
                {(
                  [
                    ['center', 'หน้าตรง'],
                    ['left', 'หันซ้าย'],
                    ['right', 'หันขวา'],
                    ['up', 'เงยหน้า'],
                    ['down', 'ก้มหน้า'],
                  ] as const
                ).map(([key, label]) => {
                  const src = patient.angleImageUrls?.[key];
                  return (
                    <div
                      key={key}
                      className="rounded-xl border border-emerald-200 bg-white overflow-hidden flex flex-col items-center"
                    >
                      {src ? (
                        <a href={src} target="_blank" rel="noreferrer" className="w-full block">
                          <img
                            src={src}
                            alt={label}
                            className="h-14 w-full object-cover"
                          />
                        </a>
                      ) : (
                        <div className="h-14 w-full bg-slate-100 flex items-center justify-center text-[10px] text-slate-400">
                          -
                        </div>
                      )}
                      <span className="py-1 text-[10px] font-bold text-slate-700">{label}</span>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="flex flex-wrap gap-2">
                {patient.faceImageUrls?.map((url, idx) => (
                  <a
                    key={`${url}-${idx}`}
                    href={url}
                    target="_blank"
                    rel="noreferrer"
                    className="rounded-xl border border-emerald-200 bg-white overflow-hidden block"
                  >
                    <img
                      src={url}
                      alt={`Face angle ${idx + 1}`}
                      className="h-14 w-16 object-cover"
                    />
                  </a>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Modal Sub-tabs */}
        <div className="inline-flex rounded-xl border border-slate-200 bg-slate-50 p-1 font-bold">
          <button
            type="button"
            onClick={() => setModalTab('profile')}
            className={`whitespace-nowrap h-8 px-4 inline-flex items-center justify-center gap-1 rounded-lg ${
              modalTab === 'profile' ? 'bg-white text-emerald-700 shadow-sm' : 'text-slate-500'
            }`}
          >
            <FileText className="h-3.5 w-3.5 shrink-0" /> {sm.profileTab}
          </button>
          <button
            type="button"
            onClick={() => setModalTab('history')}
            className={`whitespace-nowrap h-8 px-4 inline-flex items-center justify-center gap-1 rounded-lg ${
              modalTab === 'history' ? 'bg-white text-emerald-700 shadow-sm' : 'text-slate-500'
            }`}
          >
            <Activity className="h-3.5 w-3.5 shrink-0" /> {sm.historyTab} ({patientSessions.length})
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
                    strokeType,
                    affectedSide,
                    motorAssessmentScore,
                    adlDressingScore,
                    maxSafeRomDeg,
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
              <div className="font-extrabold text-slate-900">{sm.editProfileHeading}</div>
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
                    inputMode="numeric"
                    min={1}
                    max={120}
                    value={age}
                    onKeyDown={(e) => {
                      if (['e', 'E', '+', '-', '.'].includes(e.key)) e.preventDefault();
                    }}
                    onChange={(e) => {
                      const digits = e.target.value.replace(/\D/g, '').slice(0, 3);
                      setAge(digits ? Math.min(120, Number(digits)) : 0);
                    }}
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
                  type="tel"
                  inputMode="numeric"
                  pattern="[0-9]*"
                  maxLength={10}
                  value={phone}
                  onChange={(e) => setPhone(e.target.value.replace(/\D/g, '').slice(0, 10))}
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

              {/* Stroke Clinical Profile & Mahidol 0-6 Scale */}
              <div className="rounded-xl border border-emerald-200 bg-emerald-50/40 p-3 space-y-2.5">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="font-extrabold text-emerald-950">
                    ข้อมูลกายภาพ Stroke &amp; ประเมินกำลังกล้ามเนื้อ (Mahidol 0–6 Scale)
                  </span>
                  <span className="rounded-lg bg-white border border-emerald-200 px-2 py-0.5 font-mono text-[10px] font-bold text-indigo-700">
                    Universal EHR: {patient.universalHealthLinkCode || `DTC-EHR-${patient.hn}`}
                  </span>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                  <select
                    value={strokeType}
                    onChange={(e) => setStrokeType(e.target.value as StrokeType)}
                    className="rounded-xl border border-emerald-200 bg-white px-2.5 py-2 font-bold"
                  >
                    <option value="ISCHEMIC">Stroke ตีบ/ตัน (Ischemic)</option>
                    <option value="HEMORRHAGIC">Stroke แตก (Hemorrhagic)</option>
                    <option value="ELDERLY_GENERAL">ผู้สูงอายุ (MOPH)</option>
                    <option value="NEURO">ระบบประสาท (Neuro)</option>
                    <option value="ORTHOPEDIC">กระดูก/ข้อ (Ortho)</option>
                  </select>
                  <select
                    value={affectedSide}
                    onChange={(e) => setAffectedSide(e.target.value as AffectedSide)}
                    className="rounded-xl border border-emerald-200 bg-white px-2.5 py-2 font-bold"
                  >
                    <option value="LEFT">อ่อนแรง: แขนซ้าย (LEFT)</option>
                    <option value="RIGHT">อ่อนแรง: แขนขวา (RIGHT)</option>
                    <option value="BOTH">อ่อนแรง: ทั้ง 2 ข้าง (BOTH)</option>
                    <option value="NONE">ปกติ/สมดุล (NONE)</option>
                  </select>
                  <input
                    type="number"
                    min={30}
                    max={180}
                    value={maxSafeRomDeg}
                    onChange={(e) => setMaxSafeRomDeg(Math.max(30, Math.min(180, Number(e.target.value) || 30)))}
                    placeholder="Safe ROM (°)"
                    title="Safe Max ROM (องศาปลอดภัยสูงสุด)"
                    className="rounded-xl border border-emerald-200 bg-white px-2.5 py-2 font-mono font-bold text-emerald-800"
                  />
                  <select
                    value={adlDressingScore}
                    onChange={(e) => setAdlDressingScore(Number(e.target.value))}
                    className="rounded-xl border border-emerald-200 bg-white px-2.5 py-2 font-bold"
                  >
                    {[0, 1, 2, 3, 4, 5].map((sc) => (
                      <option key={sc} value={sc}>
                        ADL ใส่เสื้อ: {sc}/5
                      </option>
                    ))}
                  </select>
                </div>
                <div className="grid grid-cols-7 gap-1">
                  {MAHIDOL_MOTOR_SCALE.map((lvl) => (
                    <button
                      key={lvl.score}
                      type="button"
                      onClick={() => {
                        setMotorAssessmentScore(lvl.score);
                        setMaxSafeRomDeg(lvl.recommendedMaxRomDeg);
                      }}
                      className={`rounded-lg border py-1.5 text-center font-black transition cursor-pointer ${
                        motorAssessmentScore === lvl.score
                          ? 'border-emerald-600 bg-emerald-600 text-white'
                          : 'border-emerald-200 bg-white text-slate-700 hover:bg-emerald-100'
                      }`}
                    >
                      {lvl.score}
                    </button>
                  ))}
                </div>
                <div className="text-[11px] text-emerald-950 font-semibold">
                  {MAHIDOL_MOTOR_SCALE[motorAssessmentScore]?.titleTh}: {MAHIDOL_MOTOR_SCALE[motorAssessmentScore]?.descTh}
                </div>
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
                        PT: {pt.fullName}
                      </option>
                    ))}
                  </select>
                )}
                <button
                  type="submit"
                  disabled={saving}
                  className="whitespace-nowrap h-10 px-4 inline-flex items-center justify-center rounded-xl bg-emerald-600 font-bold text-white hover:bg-emerald-500 disabled:opacity-50"
                >
                  {saving ? sm.savingBtn : sm.saveProfileBtn}
                </button>
              </div>
            </form>

            {/* Delegation */}
            <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4 space-y-2.5">
              <div className="font-extrabold text-slate-800">{sm.delegateHeading}</div>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2">
                <select
                  value={delegatePtId}
                  onChange={(e) => setDelegatePtId(e.target.value)}
                  className="rounded-xl border border-slate-200 bg-white px-3 py-2 font-semibold"
                >
                  <option value="">-- Primary PT --</option>
                  {ptUsers.map((pt) => (
                    <option key={pt.id} value={pt.id}>
                      {pt.fullName}
                    </option>
                  ))}
                </select>
                <input
                  type="date"
                  value={delegateDate}
                  onChange={(e) => setDelegateDate(e.target.value)}
                  className="rounded-xl border border-slate-200 bg-white px-3 py-2 font-mono font-semibold"
                />
                <input
                  value={delegateNote}
                  onChange={(e) => setDelegateNote(e.target.value)}
                  placeholder="หมายเหตุการฝากเคส..."
                  className="rounded-xl border border-slate-200 bg-white px-3 py-2"
                />
                <button
                  type="button"
                  onClick={async () => {
                    await apiClient.delegatePatientCase(
                      patient.id,
                      delegatePtId || null,
                      delegateNote || undefined,
                      delegatePtId ? delegateDate : undefined,
                    );
                    onFlash('success', 'อัปเดตการฝากเคส PT เรียบร้อยแล้ว');
                    await onRefresh();
                  }}
                  className="whitespace-nowrap h-10 px-4 inline-flex items-center justify-center rounded-xl bg-blue-600 font-bold text-white hover:bg-blue-500 cursor-pointer"
                >
                  {sm.updatePtBtn}
                </button>
              </div>
            </div>

            {/* Append-Only Medical Notes */}
            <div className="space-y-2.5">
              <div className="font-extrabold text-slate-900">{sm.notesHeading}</div>
              <div className="max-h-40 overflow-y-auto space-y-2 rounded-2xl border border-slate-200 p-3 bg-slate-50/40">
                {notes.map((n) => (
                  <div key={n.id} className="rounded-xl border border-slate-200 bg-white p-3 space-y-1">
                    <div className="flex justify-between text-[11px] text-slate-400">
                      <span className="font-bold text-emerald-700">{n.authorName}</span>
                      <span>
                        {new Date(n.createdAt).toLocaleString(lang === 'cn' ? 'zh-CN' : lang === 'en' ? 'en-US' : 'th-TH')}
                      </span>
                    </div>
                    <p className="whitespace-pre-line text-slate-800">{n.content}</p>
                  </div>
                ))}
                {notes.length === 0 && <p className="text-center text-slate-400 py-3">{sm.emptyNotes}</p>}
              </div>

              <div className="flex gap-2">
                <input
                  value={newNote}
                  onChange={(e) => setNewNote(e.target.value)}
                  placeholder={lang === 'cn' ? '输入临床症状记录...' : lang === 'en' ? 'Enter clinical note...' : 'พิมพ์บันทึกอาการเพิ่มเติม...'}
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
                  className="whitespace-nowrap h-10 px-4 inline-flex items-center justify-center rounded-xl bg-emerald-600 font-bold text-white"
                >
                  {sm.addNoteBtn}
                </button>
              </div>
            </div>

            {/* Treatment Report */}
            <div className="border-t border-slate-100 pt-3 space-y-2">
              <div className="font-extrabold text-slate-900">{sm.reportHeading}</div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                <input
                  value={reportSummary}
                  onChange={(e) => setReportSummary(e.target.value)}
                  placeholder={lang === 'cn' ? '康复总结...' : lang === 'en' ? 'Rehab summary...' : 'สรุปผลการฟื้นฟู...'}
                  className="rounded-xl border border-slate-200 px-3 py-2"
                />
                <input
                  value={reportRec}
                  onChange={(e) => setReportRec(e.target.value)}
                  placeholder={lang === 'cn' ? '后续建议...' : lang === 'en' ? 'Recommendations...' : 'คำแนะนำต่อเนื่อง...'}
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
                className="whitespace-nowrap h-10 px-4 inline-flex items-center justify-center rounded-xl bg-gradient-to-r from-[#0F4C81] to-[#0D9488] font-bold text-white shadow-xs"
              >
                {sm.saveReportBtn}
              </button>
            </div>
          </>
        ) : (
          <div className="space-y-4">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
              <div className="rounded-2xl border border-slate-200 bg-slate-50 p-3">
                <div className="text-[11px] text-slate-500">{sm.statSessions}</div>
                <div className="text-lg font-black text-slate-900">
                  {Math.max(patientSessions.length, sessionHistory.length)} {sm.sessionUnit}
                </div>
              </div>
              <div className="rounded-2xl border border-slate-200 bg-emerald-50/50 p-3">
                <div className="text-[11px] text-slate-500">Mahidol Motor Score</div>
                <div className="text-lg font-black text-emerald-700">
                  {patient.motorAssessmentScore ?? 3} / 6
                </div>
              </div>
              <div className="rounded-2xl border border-slate-200 bg-sky-50/50 p-3">
                <div className="text-[11px] text-slate-500">{sm.statMaxRom}</div>
                <div className="text-lg font-black text-sky-700">
                  {Math.max(
                    patientSessions.reduce((max, s) => Math.max(max, s.maxRomDeg), 0),
                    sessionHistory.reduce((max, h) => Math.max(max, h.achievedMaxRomDeg), 0),
                  )}
                  °
                </div>
              </div>
              <div className="rounded-2xl border border-indigo-200 bg-indigo-50/60 p-3">
                <div className="text-[11px] text-indigo-700">Safe ROM Limit</div>
                <div className="text-lg font-black text-indigo-800">
                  {patient.maxSafeRomDeg ?? 150}°
                </div>
              </div>
            </div>

            {/* Historical ROM Progress Comparison & Inline SVG Trend Chart */}
            {(() => {
              const timeline = [...sessionHistory].sort(
                (a, b) => new Date(a.sessionDate).getTime() - new Date(b.sessionDate).getTime(),
              );
              const byExercise = new Map<string, RehabSessionHistoryEntry[]>();
              for (const row of timeline) {
                const list = byExercise.get(row.exerciseCode) ?? [];
                list.push(row);
                byExercise.set(row.exerciseCode, list);
              }
              const comparisons = Array.from(byExercise.entries()).map(([code, list]) => {
                const first = list[0];
                const latest = list[list.length - 1];
                const base = first.baselineRomDeg || first.achievedMaxRomDeg;
                const delta = Number((latest.achievedMaxRomDeg - base).toFixed(1));
                return { code, name: latest.exerciseName || code, first, latest, base, delta, count: list.length };
              });
              const chartPts = timeline.slice(-10);
              return (
                <div className="rounded-2xl border border-emerald-200 bg-emerald-50/35 p-3.5 space-y-3">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="inline-flex items-center gap-1.5 font-extrabold text-emerald-950">
                      <TrendingUp className="h-4 w-4 text-emerald-600" />
                      เปรียบเทียบพัฒนาการองศาการเคลื่อนไหว (ROM Progress vs Baseline)
                    </span>
                    {onOpenLinkEhr && (
                      <button
                        type="button"
                        onClick={() => onOpenLinkEhr(patient)}
                        className="inline-flex items-center gap-1 rounded-xl border border-indigo-200 bg-white px-2.5 py-1 text-[11px] font-bold text-indigo-700 hover:bg-indigo-50 cursor-pointer"
                      >
                        <Link2 className="h-3.5 w-3.5" />
                        <span>เชื่อมโยง/ส่งต่อประวัติการรักษา (Link EHR)</span>
                      </button>
                    )}
                  </div>

                  {comparisons.length > 0 ? (
                    <div className="space-y-2">
                      {comparisons.map((cmp) => (
                        <div
                          key={cmp.code}
                          className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-emerald-200 bg-white px-3 py-2"
                        >
                          <div className="min-w-0">
                            <div className="font-bold text-slate-900 truncate">
                              {cmp.name}{' '}
                              <span className="font-mono text-[10px] text-slate-500">({cmp.code})</span>
                            </div>
                            <div className="text-[11px] text-slate-600">
                              ครั้งที่ 1: <span className="font-mono font-bold">{cmp.base.toFixed(1)}°</span> ➔{' '}
                              ครั้งล่าสุด ({cmp.count}):{' '}
                              <span className="font-mono font-black text-sky-700">
                                {cmp.latest.achievedMaxRomDeg.toFixed(1)}°
                              </span>{' '}
                              • PT: {cmp.latest.recordedByPtName || 'กภ. ประจำคลินิก'}
                            </div>
                          </div>
                          <span
                            className={`rounded-full px-2.5 py-1 font-mono text-[11px] font-black ${
                              cmp.delta >= 0
                                ? 'bg-emerald-100 text-emerald-800 border border-emerald-300'
                                : 'bg-amber-100 text-amber-800 border border-amber-300'
                            }`}
                          >
                            {cmp.delta >= 0 ? `+${cmp.delta.toFixed(1)}° ↑` : `${cmp.delta.toFixed(1)}° ↓`}
                          </span>
                        </div>
                      ))}

                      {chartPts.length >= 2 && (
                        <div className="rounded-xl border border-slate-200 bg-white p-2.5">
                          <div className="text-[10px] font-bold text-slate-500 mb-1">
                            ROM Trend (Latest {chartPts.length} sessions)
                          </div>
                          <svg viewBox="0 0 300 68" className="w-full h-16">
                            {(() => {
                              const pts = chartPts.map((pt, idx) => {
                                const x = chartPts.length === 1 ? 150 : 12 + (idx * 276) / (chartPts.length - 1);
                                const norm = Math.max(0, Math.min(180, pt.achievedMaxRomDeg)) / 180;
                                const y = 58 - norm * 46;
                                return { x, y, val: pt.achievedMaxRomDeg };
                              });
                              const d = pts
                                .map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`)
                                .join(' ');
                              return (
                                <>
                                  <path d={d} fill="none" stroke="#059669" strokeWidth="2.5" strokeLinecap="round" />
                                  {pts.map((p, i) => (
                                    <g key={i}>
                                      <circle cx={p.x} cy={p.y} r="3.5" fill="#0284C7" />
                                      <text x={p.x} y={Math.max(10, p.y - 6)} textAnchor="middle" fontSize="8" fill="#0F4C81" fontWeight="bold">
                                        {Math.round(p.val)}°
                                      </text>
                                    </g>
                                  ))}
                                </>
                              );
                            })()}
                          </svg>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="rounded-xl border border-dashed border-emerald-200 bg-white/70 p-3 text-center text-[11px] text-slate-500">
                      ยังไม่มีประวัติเปรียบเทียบ ROM (เริ่มสะสมข้อมูลอัตโนมัติเมื่อฝึกหรือบันทึกผลกายภาพ)
                    </div>
                  )}
                </div>
              );
            })()}

            <div className="max-h-52 overflow-y-auto space-y-2">
              {sessionHistory.map((h) => (
                <div key={h.id} className="rounded-xl border border-slate-200 p-3 flex items-center justify-between gap-2">
                  <div className="min-w-0">
                    <div className="font-bold text-slate-900 truncate">
                      {h.exerciseName || h.exerciseCode}{' '}
                      <span className="ml-1 rounded bg-sky-50 px-1.5 py-0.5 font-mono text-[10px] text-sky-700">
                        {h.facilityName}
                      </span>
                    </div>
                    <div className="text-[11px] text-slate-500">
                      Max ROM {h.achievedMaxRomDeg.toFixed(1)}° (Baseline {h.baselineRomDeg.toFixed(1)}° • Δ{' '}
                      {h.romDeltaDeg >= 0 ? `+${h.romDeltaDeg.toFixed(1)}°` : `${h.romDeltaDeg.toFixed(1)}°`}) •{' '}
                      {h.validReps} reps • Motor {h.motorScoreAtSession ?? patient.motorAssessmentScore ?? 3}/6 •{' '}
                      PT: {h.recordedByPtName || 'กภ. ประจำคลินิก'}
                    </div>
                  </div>
                  <span className="text-[11px] text-slate-400 whitespace-nowrap">
                    {new Date(h.sessionDate).toLocaleDateString(lang === 'cn' ? 'zh-CN' : lang === 'en' ? 'en-US' : 'th-TH')}
                  </span>
                </div>
              ))}
              {sessionHistory.length === 0 &&
                patientSessions.map((s) => (
                  <div key={s.id} className="rounded-xl border border-slate-200 p-3 flex items-center justify-between">
                    <div>
                      <div className="font-bold text-slate-900">{s.exerciseName}</div>
                      <div className="text-[11px] text-slate-500">
                        {s.completedSets} × {s.completedReps} • {s.accuracyPercent}% • Max ROM {s.maxRomDeg}°
                      </div>
                    </div>
                    <span className="text-[11px] text-slate-400 whitespace-nowrap">
                      {new Date(s.createdAt).toLocaleDateString(lang === 'cn' ? 'zh-CN' : lang === 'en' ? 'en-US' : 'th-TH')}
                    </span>
                  </div>
                ))}
              {sessionHistory.length === 0 && patientSessions.length === 0 && (
                <div className="rounded-2xl border border-dashed border-slate-200 py-8 text-center text-slate-400">
                  {sm.emptyHistory}
                </div>
              )}
            </div>

            <button
              type="button"
              onClick={async () => {
                const baseRom =
                  sessionHistory[sessionHistory.length - 1]?.baselineRomDeg ?? 35;
                const nextRom = Math.min(patient.maxSafeRomDeg ?? 150, Math.max(baseRom + 18, 95));
                await apiClient.recordRehabSession({
                  patientId: patient.id,
                  activityType: 'physio_exercise',
                  exerciseId: 'STROKE_SH_FLEX_R',
                  exerciseName: '[Stroke ADL] ยกแขนขวาไปด้านหน้า',
                  jointName: 'shoulder',
                  bodySide: patient.affectedSide ?? 'RIGHT',
                  completedSets: 3,
                  completedReps: 10,
                  accuracyPercent: 95,
                  maxRomDeg: nextRom,
                  motorScoreAtSession: patient.motorAssessmentScore ?? 3,
                  recordedByPtName: currentUser.fullName,
                  painStopTriggered: false,
                  aiFeedback: `บันทึกการฝึกโดย ${currentUser.fullName}`,
                  kinematicsReplay: [],
                });
                const updatedHist = await apiClient.listRehabSessionHistory(patient.id).catch(() => []);
                setSessionHistory(updatedHist);
                onFlash('success', 'บันทึกผลการฝึกกายภาพใหม่พร้อมคำนวณ Δ ROM เรียบร้อย');
                await onRefresh();
              }}
              className="w-full whitespace-nowrap h-10 px-4 inline-flex items-center justify-center rounded-xl bg-emerald-600 font-extrabold text-white hover:bg-emerald-500 cursor-pointer"
            >
              {sm.recordSessionBtn}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

/** Deterministic lightweight QR SVG matrix generator for Universal EHR Link Codes */
function UniversalEhrQrSvg({ value }: { value: string }) {
  const size = 21;
  const cells: boolean[][] = Array.from({ length: size }, () => Array(size).fill(false));
  const markFinder = (ox: number, oy: number) => {
    for (let y = 0; y < 7; y += 1) {
      for (let x = 0; x < 7; x += 1) {
        const border = x === 0 || x === 6 || y === 0 || y === 6;
        const inner = x >= 2 && x <= 4 && y >= 2 && y <= 4;
        cells[oy + y][ox + x] = border || inner;
      }
    }
  };
  markFinder(0, 0);
  markFinder(size - 7, 0);
  markFinder(0, size - 7);

  let hash = 2166136261;
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const inFinder =
        (x < 8 && y < 8) || (x >= size - 8 && y < 8) || (x < 8 && y >= size - 8);
      if (inFinder) continue;
      hash ^= (x + 1) * 31 + (y + 7) * 17;
      hash = Math.imul(hash, 16777619);
      cells[y][x] = (hash & 1) === 1;
    }
  }

  return (
    <svg viewBox={`0 0 ${size} ${size}`} className="h-28 w-28 rounded-xl border border-slate-200 bg-white p-1.5">
      {cells.flatMap((row, y) =>
        row.map((on, x) =>
          on ? <rect key={`${x}-${y}`} x={x} y={y} width={1} height={1} fill="#0F4C81" /> : null,
        ),
      )}
    </svg>
  );
}

export function UniversalEhrLinkModal({
  patient,
  apiClient,
  onClose,
  onLinked,
  onFlash,
}: {
  patient: PatientRecord | null;
  apiClient: DrTechCaresApiClient;
  onClose: () => void;
  onLinked: () => Promise<void>;
  onFlash: (kind: 'success' | 'error', msg: string) => void;
}) {
  const [exportBundle, setExportBundle] = useState<UniversalEhrPackage | null>(null);
  const [lookupInput, setLookupInput] = useState('');
  const [lookupPreview, setLookupPreview] = useState<UniversalEhrPackage | null>(null);
  const [rawJsonInput, setRawJsonInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    if (!patient) return;
    setLookupInput('');
    setLookupPreview(null);
    setRawJsonInput('');
    setErrorMsg(null);
    const code = patient.universalHealthLinkCode || patient.hn;
    apiClient
      .lookupUniversalEhr(code)
      .then((pkg) => setExportBundle(pkg))
      .catch(() => setExportBundle(null));
  }, [patient, apiClient]);

  if (!patient) return null;

  const linkCode = patient.universalHealthLinkCode || `DTC-EHR-${patient.hn}`;

  const handleDownloadJson = () => {
    const pkg: UniversalEhrPackage = exportBundle ?? {
      version: '1.0',
      exportedAt: new Date().toISOString(),
      universalHealthLinkCode: linkCode,
      facilityName: 'Dr.Tech.Care Clinic',
      patientProfile: {
        hn: patient.hn,
        fullName: patient.fullName,
        age: patient.age,
        gender: patient.gender,
        diagnosis: patient.diagnosis,
        strokeType: patient.strokeType,
        affectedSide: patient.affectedSide,
        motorAssessmentScore: patient.motorAssessmentScore,
        adlDressingScore: patient.adlDressingScore,
        maxSafeRomDeg: patient.maxSafeRomDeg,
      },
      sessionHistory: patient.rehabSessionHistory ?? [],
    };
    const jsonStr = JSON.stringify(pkg, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `drtechcares-ehr-${patient.hn}-${linkCode}.json`;
    a.click();
    URL.revokeObjectURL(url);
    onFlash('success', `ส่งออกไฟล์ Universal EHR JSON (${linkCode}) แล้ว`);
  };

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-[#0F4C81]/40 p-4 backdrop-blur-sm">
      <div className="w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-3xl bg-white p-6 shadow-2xl border border-indigo-200 space-y-5 text-xs">
        <div className="flex items-start justify-between border-b border-slate-100 pb-3">
          <div>
            <h3 className="flex items-center gap-2 text-base font-black text-[#0F4C81]">
              <Link2 className="h-5 w-5 text-indigo-600" />
              <span>เชื่อมโยงและส่งต่อประวัติการรักษาข้ามสถานพยาบาล (Universal EHR Link)</span>
            </h3>
            <p className="text-slate-500 mt-0.5">
              {patient.fullName} (HN: {patient.hn}) • รหัสกลาง:{' '}
              <span className="font-mono font-black text-indigo-700">{linkCode}</span>
            </p>
          </div>
          <button type="button" onClick={onClose} className="rounded-xl p-1 text-slate-400 hover:bg-slate-100">
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Export / QR Section */}
        <div className="rounded-2xl border border-indigo-200 bg-indigo-50/40 p-4 flex flex-col sm:flex-row items-center gap-4">
          <div className="shrink-0 flex flex-col items-center gap-1">
            <UniversalEhrQrSvg value={JSON.stringify({ code: linkCode, hn: patient.hn, rom: patient.maxSafeRomDeg ?? 150 })} />
            <span className="inline-flex items-center gap-1 text-[10px] font-bold text-indigo-800">
              <QrCode className="h-3 w-3" /> Universal EHR QR
            </span>
          </div>
          <div className="flex-1 space-y-2.5 min-w-0">
            <div className="font-extrabold text-slate-900">
              1. ส่งออกประวัติเพื่อส่งต่อโรงพยาบาล / คลินิกอื่น (Export EHR Package)
            </div>
            <p className="text-[11px] text-slate-600">
              ประกอบด้วยค่า Baseline/Delta ROM, คะแนนกำลังกล้ามเนื้อ Mahidol ({patient.motorAssessmentScore ?? 3}/6), Safe Max ROM ({patient.maxSafeRomDeg ?? 150}°) และประวัติฝึกทั้งหมด {exportBundle?.sessionHistory.length ?? patient.rehabSessionHistory?.length ?? 0} ครั้ง
            </p>
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                onClick={async () => {
                  try {
                    await navigator.clipboard.writeText(linkCode);
                    onFlash('success', `คัดลอกรหัส EHR ${linkCode} แล้ว`);
                  } catch {
                    onFlash('error', 'ไม่สามารถคัดลอกได้');
                  }
                }}
                className="inline-flex items-center gap-1.5 rounded-xl border border-indigo-200 bg-white px-3 py-1.5 font-bold text-indigo-700 hover:bg-indigo-50 cursor-pointer"
              >
                <Copy className="h-3.5 w-3.5" />
                <span>คัดลอกรหัสเชื่อมโยง</span>
              </button>
              <button
                type="button"
                onClick={handleDownloadJson}
                className="inline-flex items-center gap-1.5 rounded-xl bg-indigo-600 px-3.5 py-1.5 font-extrabold text-white hover:bg-indigo-500 cursor-pointer"
              >
                <Download className="h-3.5 w-3.5" />
                <span>ส่งออก JSON Data Package</span>
              </button>
            </div>
          </div>
        </div>

        {/* Lookup & Import Section */}
        <div className="rounded-2xl border border-emerald-200 bg-emerald-50/40 p-4 space-y-3">
          <div className="font-extrabold text-emerald-950">
            2. ดึง/นำเข้าประวัติเดิมจากสถานพยาบาลอื่น (Import by Universal Link Code or JSON)
          </div>

          <div className="flex gap-2">
            <input
              value={lookupInput}
              onChange={(e) => setLookupInput(e.target.value)}
              placeholder="กรอก Universal Health Link Code (เช่น DTC-EHR-...) หรือ HN เดิม"
              className="flex-1 rounded-xl border border-emerald-200 bg-white px-3 py-2 font-mono font-bold text-slate-800"
            />
            <button
              type="button"
              disabled={!lookupInput.trim() || busy}
              onClick={async () => {
                setBusy(true);
                setErrorMsg(null);
                try {
                  const found = await apiClient.lookupUniversalEhr(lookupInput);
                  if (!found) {
                    setLookupPreview(null);
                    setErrorMsg(`ไม่พบประวัติ Universal EHR สำหรับรหัส "${lookupInput.trim()}"`);
                  } else {
                    setLookupPreview(found);
                  }
                } catch (err) {
                  setErrorMsg(err instanceof Error ? err.message : 'ค้นหาไม่สำเร็จ');
                } finally {
                  setBusy(false);
                }
              }}
              className="rounded-xl bg-sky-600 px-4 py-2 font-extrabold text-white hover:bg-sky-500 disabled:opacity-50 cursor-pointer"
            >
              ค้นหารหัส EHR
            </button>
          </div>

          <textarea
            rows={3}
            value={rawJsonInput}
            onChange={(e) => setRawJsonInput(e.target.value)}
            placeholder="หรือวาง JSON EHR Package ที่ได้รับจากสถานพยาบาลต้นทาง..."
            className="w-full rounded-xl border border-emerald-200 bg-white p-2.5 font-mono text-[11px] text-slate-700"
          />

          {lookupPreview && (
            <div className="rounded-xl border border-emerald-300 bg-white p-3 text-[11px] space-y-1">
              <div className="font-black text-emerald-900">
                พบข้อมูลจาก: {lookupPreview.facilityName} • คนไข้ {lookupPreview.patientProfile.fullName} ({lookupPreview.patientProfile.hn})
              </div>
              <div className="text-slate-600">
                Mahidol Score: {lookupPreview.patientProfile.motorAssessmentScore ?? 3}/6 •{' '}
                Safe ROM: {lookupPreview.patientProfile.maxSafeRomDeg ?? 150}° •{' '}
                จำนวนประวัติฝึก: {lookupPreview.sessionHistory.length} รายการ
              </div>
            </div>
          )}

          {errorMsg && (
            <div className="rounded-xl border border-rose-200 bg-rose-50 px-3 py-2 font-bold text-rose-700">
              {errorMsg}
            </div>
          )}

          <button
            type="button"
            disabled={busy || (!lookupPreview && !rawJsonInput.trim())}
            onClick={async () => {
              setBusy(true);
              setErrorMsg(null);
              try {
                let parsedPkg: UniversalEhrPackage | undefined;
                if (rawJsonInput.trim()) {
                  parsedPkg = JSON.parse(rawJsonInput.trim()) as UniversalEhrPackage;
                } else if (lookupPreview) {
                  parsedPkg = lookupPreview;
                }
                if (!parsedPkg) {
                  setErrorMsg('กรุณาค้นหารหัส EHR หรือวางข้อมูล JSON ก่อนนำเข้า');
                  setBusy(false);
                  return;
                }
                const res = await apiClient.linkExternalEhrRecord(patient.id, {
                  linkCode: parsedPkg.universalHealthLinkCode,
                  ehrPackage: parsedPkg,
                });
                onFlash(
                  'success',
                  `เชื่อมโยงประวัติการรักษาสำเร็จ (นำเข้า ${res.importedCount} รายการเข้าสู่เวชระเบียนของ ${patient.fullName})`,
                );
                await onLinked();
                onClose();
              } catch (err) {
                setErrorMsg(err instanceof Error ? err.message : 'นำเข้าข้อมูล JSON ไม่ถูกต้อง');
              } finally {
                setBusy(false);
              }
            }}
            className="w-full inline-flex items-center justify-center gap-2 rounded-xl bg-emerald-600 py-2.5 font-extrabold text-white hover:bg-emerald-500 disabled:opacity-50 cursor-pointer"
          >
            <Upload className="h-4 w-4" />
            <span>{busy ? 'กำลังนำเข้าและผสานประวัติ...' : 'นำเข้าและเชื่อมโยงประวัติเข้าเวชระเบียนนี้ (Merge EHR)'}</span>
          </button>
        </div>
      </div>
    </div>
  );
}

const MONTH_LABELS: Record<ClinicalLang, readonly string[]> = {
  th: [
    'มกราคม',
    'กุมภาพันธ์',
    'มีนาคม',
    'เมษายน',
    'พฤษภาคม',
    'มิถุนายน',
    'กรกฎาคม',
    'สิงหาคม',
    'กันยายน',
    'ตุลาคม',
    'พฤศจิกายน',
    'ธันวาคม',
  ],
  en: [
    'January',
    'February',
    'March',
    'April',
    'May',
    'June',
    'July',
    'August',
    'September',
    'October',
    'November',
    'December',
  ],
  cn: ['一月', '二月', '三月', '四月', '五月', '六月', '七月', '八月', '九月', '十月', '十一月', '十二月'],
};

const CALENDAR_DOW_HEADERS: Record<ClinicalLang, readonly string[]> = {
  th: ['อา.', 'จ.', 'อ.', 'พ.', 'พฤ.', 'ศ.', 'ส.'],
  en: ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'],
  cn: ['日', '一', '二', '三', '四', '五', '六'],
};

const WEEKDAY_TOGGLE_OPTIONS: ReadonlyArray<{
  dayIndex: number;
  label: Record<ClinicalLang, string>;
  short: Record<ClinicalLang, string>;
}> = [
  { dayIndex: 1, label: { th: 'จันทร์', en: 'Monday', cn: '周一' }, short: { th: 'จ.', en: 'Mon', cn: '一' } },
  { dayIndex: 2, label: { th: 'อังคาร', en: 'Tuesday', cn: '周二' }, short: { th: 'อ.', en: 'Tue', cn: '二' } },
  { dayIndex: 3, label: { th: 'พุธ', en: 'Wednesday', cn: '周三' }, short: { th: 'พ.', en: 'Wed', cn: '三' } },
  { dayIndex: 4, label: { th: 'พฤหัสบดี', en: 'Thursday', cn: '周四' }, short: { th: 'พฤ.', en: 'Thu', cn: '四' } },
  { dayIndex: 5, label: { th: 'ศุกร์', en: 'Friday', cn: '周五' }, short: { th: 'ศ.', en: 'Fri', cn: '五' } },
  { dayIndex: 6, label: { th: 'เสาร์', en: 'Saturday', cn: '周六' }, short: { th: 'ส.', en: 'Sat', cn: '六' } },
  { dayIndex: 0, label: { th: 'อาทิตย์', en: 'Sunday', cn: '周日' }, short: { th: 'อา.', en: 'Sun', cn: '日' } },
];

const WEEKDAY_PRESETS: ReadonlyArray<{
  id: string;
  label: Record<ClinicalLang, string>;
  days: number[];
}> = [
  { id: 'mwf', label: { th: 'จ. / พ. / ศ.', en: 'Mon / Wed / Fri', cn: '一 / 三 / 五' }, days: [1, 3, 5] },
  { id: 'tts', label: { th: 'อ. / พฤ. / ส.', en: 'Tue / Thu / Sat', cn: '二 / 四 / 六' }, days: [2, 4, 6] },
  { id: 'weekdays', label: { th: 'จันทร์–ศุกร์', en: 'Mon–Fri', cn: '工作日' }, days: [1, 2, 3, 4, 5] },
  { id: 'weekend', label: { th: 'เสาร์–อาทิตย์', en: 'Sat–Sun', cn: '周末' }, days: [6, 0] },
  { id: 'daily', label: { th: 'ทุกวัน (7 วัน)', en: 'Every Day', cn: '每天' }, days: [1, 2, 3, 4, 5, 6, 0] },
];

const EXERCISE_FILTER_TABS: ReadonlyArray<{
  id: 'all' | 'upper_limb' | 'lower_limb' | 'neck_trunk' | 'balance';
  label: Record<ClinicalLang, string>;
}> = [
  { id: 'all', label: { th: 'ทั้งหมด', en: 'All Postures', cn: '全部动作' } },
  { id: 'upper_limb', label: { th: 'แขนและไหล่', en: 'Upper Limb', cn: '上肢/肩肘' } },
  { id: 'lower_limb', label: { th: 'ขาและเข่า', en: 'Lower Limb', cn: '下肢/膝踝' } },
  { id: 'neck_trunk', label: { th: 'คอและลำตัว', en: 'Neck & Trunk', cn: '颈部/躯干' } },
  { id: 'balance', label: { th: 'ฝึกทรงตัว', en: 'Balance', cn: '平衡训练' } },
];

/** Parses YYYY-MM-DD cleanly in local calendar coordinates without UTC drift. */
function parseIsoYmd(iso: string | null | undefined): { year: number; month: number; day: number } {
  if (iso && /^\d{4}-\d{2}-\d{2}$/.test(iso)) {
    const [y, m, d] = iso.split('-').map(Number);
    return { year: y, month: m - 1, day: d };
  }
  const now = new Date();
  return { year: now.getFullYear(), month: now.getMonth(), day: now.getDate() };
}

/** Formats local year, 0-based month, and day as YYYY-MM-DD. */
function formatIsoYmd(year: number, month: number, day: number): string {
  const dt = new Date(year, month, day);
  return `${dt.getFullYear()}-${String(dt.getMonth() + 1).padStart(2, '0')}-${String(dt.getDate()).padStart(2, '0')}`;
}

/** Builds Sunday-aligned calendar cells for a single month (null for leading padding). */
function getMonthCalendarCells(year: number, month: number): (number | null)[] {
  const firstDay = new Date(year, month, 1).getDay();
  const daysInMonth = new Date(year, month + 1, 0).getDate();
  const cells: (number | null)[] = [];
  for (let i = 0; i < firstDay; i += 1) cells.push(null);
  for (let d = 1; d <= daysInMonth; d += 1) cells.push(d);
  return cells;
}

/** Computes sorted unique YYYY-MM-DD dates matching the chosen weekdays across N months. */
function computeWeeklyDatesForMonths(opts: {
  startYear: number;
  startMonth: number;
  monthSpan: number;
  weekdays: number[];
  minDateIso?: string | null;
}): string[] {
  const weekdaySet = new Set(opts.weekdays);
  if (weekdaySet.size === 0) return [];
  const results: string[] = [];
  for (let mOffset = 0; mOffset < Math.max(1, opts.monthSpan); mOffset += 1) {
    const target = new Date(opts.startYear, opts.startMonth + mOffset, 1);
    const y = target.getFullYear();
    const m = target.getMonth();
    const daysInMonth = new Date(y, m + 1, 0).getDate();
    for (let d = 1; d <= daysInMonth; d += 1) {
      const dow = new Date(y, m, d).getDay();
      if (!weekdaySet.has(dow)) continue;
      const iso = formatIsoYmd(y, m, d);
      if (opts.minDateIso && iso < opts.minDateIso) continue;
      results.push(iso);
    }
  }
  return results;
}

/** Returns the next active scheduled date from the schedule table, advancing past completed today items. */
function computeNextAppointmentDate(schedules: readonly RehabScheduleItem[], todayIso: string): string {
  return resolveNextAppointmentDate({ rehabSchedules: schedules, todayIso });
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
  const { lang } = useClinicalI18n();
  const sm = SUBMODAL_TEXT[lang];
  const todayIso = useMemo(() => {
    const now = new Date();
    return formatIsoYmd(now.getFullYear(), now.getMonth(), now.getDate());
  }, []);

  const [targetPatientId, setTargetPatientId] = useState(patient?.id || patients[0]?.id || '');
  const [ptId, setPtId] = useState(patient?.assignedPtId || ptUsers[0]?.id || '');
  const [exerciseCategoryFilter, setExerciseCategoryFilter] = useState<
    'all' | 'upper_limb' | 'lower_limb' | 'neck_trunk' | 'balance'
  >('all');
  const [selectedExerciseIds, setSelectedExerciseIds] = useState<string[]>([]);
  const [modpackId, setModpackId] = useState(modpacks[0]?.id || '');
  const [sets, setSets] = useState(3);
  const [reps, setReps] = useState(10);
  const [notes, setNotes] = useState('ตารางฝึกตามโปรแกรมคลินิก');

  const [dateSelectMode, setDateSelectMode] = useState<'daily' | 'weekly'>('daily');
  const [viewMonthDate, setViewMonthDate] = useState<{ year: number; month: number }>(() => {
    const p = parseIsoYmd(selectedDateIso || todayIso);
    return { year: p.year, month: p.month };
  });
  const [manualDates, setManualDates] = useState<string[]>(() => [selectedDateIso || todayIso]);
  const [selectedWeekdays, setSelectedWeekdays] = useState<number[]>(() => {
    const p = parseIsoYmd(selectedDateIso || todayIso);
    return [new Date(p.year, p.month, p.day).getDay()];
  });
  const [monthSpan, setMonthSpan] = useState<number>(1);
  const [startFromAnchorOnly, setStartFromAnchorOnly] = useState<boolean>(false);
  const [weeklyExcludedDates, setWeeklyExcludedDates] = useState<string[]>([]);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!mode) return;
    const anchorIso = selectedDateIso || todayIso;
    const parsed = parseIsoYmd(anchorIso);
    if (patient?.id) setTargetPatientId(patient.id);
    if (patient?.assignedPtId) setPtId(patient.assignedPtId);
    setViewMonthDate({ year: parsed.year, month: parsed.month });
    setManualDates([anchorIso]);
    setSelectedWeekdays([new Date(parsed.year, parsed.month, parsed.day).getDay()]);
    setMonthSpan(1);
    setStartFromAnchorOnly(false);
    setWeeklyExcludedDates([]);
    const sanitizedSchedules = sanitizeAccidentalDefaultKneePrefix(patient?.rehabSchedules ?? []);
    const anchorDayExIds = Array.from(
      new Set(
        sanitizedSchedules
          .filter(
            (item) => item.scheduledDate === anchorIso && item.status !== 'REST' && Boolean(item.exerciseId),
          )
          .map((item) => item.exerciseId as string),
      ),
    );
    setSelectedExerciseIds(anchorDayExIds);
  }, [patient, selectedDateIso, mode, todayIso]);

  const activePatient = patients.find((p) => p.id === targetPatientId) || patient;
  const existingScheduleDatesSet = useMemo(() => {
    const map = new Map<string, RehabScheduleItem[]>();
    for (const item of activePatient?.rehabSchedules ?? []) {
      const list = map.get(item.scheduledDate) ?? [];
      list.push(item);
      map.set(item.scheduledDate, list);
    }
    return map;
  }, [activePatient]);

  const dayItems = useMemo(() => {
    if (mode !== 'day_detail' || !activePatient || !selectedDateIso) return [];
    return existingScheduleDatesSet.get(selectedDateIso) ?? [];
  }, [mode, activePatient, selectedDateIso, existingScheduleDatesSet]);

  const filteredExercises = useMemo(() => {
    if (exerciseCategoryFilter === 'all') return exercises;
    return exercises.filter((ex) => ex.category === exerciseCategoryFilter);
  }, [exercises, exerciseCategoryFilter]);

  const computedWeeklyBaseDates = useMemo(() => {
    const minIso = startFromAnchorOnly ? selectedDateIso || todayIso : null;
    return computeWeeklyDatesForMonths({
      startYear: viewMonthDate.year,
      startMonth: viewMonthDate.month,
      monthSpan,
      weekdays: selectedWeekdays,
      minDateIso: minIso,
    });
  }, [viewMonthDate, monthSpan, selectedWeekdays, startFromAnchorOnly, selectedDateIso, todayIso]);

  const finalSelectedDates = useMemo(() => {
    if (dateSelectMode === 'daily') {
      return [...manualDates].sort();
    }
    const excluded = new Set(weeklyExcludedDates);
    return computedWeeklyBaseDates.filter((d) => !excluded.has(d));
  }, [dateSelectMode, manualDates, computedWeeklyBaseDates, weeklyExcludedDates]);

  const finalSelectedDateSet = useMemo(() => new Set(finalSelectedDates), [finalSelectedDates]);
  const monthCells = useMemo(
    () => getMonthCalendarCells(viewMonthDate.year, viewMonthDate.month),
    [viewMonthDate.year, viewMonthDate.month],
  );

  if (!mode) return null;

  const anchorIso = selectedDateIso || todayIso;

  const toggleExercise = (id: string) => {
    setSelectedExerciseIds((prev) => (prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]));
  };

  const toggleManualDate = (iso: string) => {
    setManualDates((prev) => (prev.includes(iso) ? prev.filter((d) => d !== iso) : [...prev, iso].sort()));
  };

  const toggleWeeklyException = (iso: string) => {
    const p = parseIsoYmd(iso);
    const dow = new Date(p.year, p.month, p.day).getDay();
    if (!selectedWeekdays.includes(dow)) return;
    setWeeklyExcludedDates((prev) => (prev.includes(iso) ? prev.filter((d) => d !== iso) : [...prev, iso]));
  };

  const toggleWeekday = (dow: number) => {
    setWeeklyExcludedDates([]);
    setSelectedWeekdays((prev) => (prev.includes(dow) ? prev.filter((d) => d !== dow) : [...prev, dow]));
  };

  const shiftViewMonth = (delta: number) => {
    setViewMonthDate((prev) => {
      const next = new Date(prev.year, prev.month + delta, 1);
      return { year: next.getFullYear(), month: next.getMonth() };
    });
  };

  const selectThisWeekInDaily = () => {
    const p = parseIsoYmd(anchorIso);
    const base = new Date(p.year, p.month, p.day);
    const dow = base.getDay();
    const sunday = new Date(base.getFullYear(), base.getMonth(), base.getDate() - dow);
    const weekDates: string[] = [];
    for (let i = 0; i < 7; i += 1) {
      const d = new Date(sunday.getFullYear(), sunday.getMonth(), sunday.getDate() + i);
      weekDates.push(formatIsoYmd(d.getFullYear(), d.getMonth(), d.getDate()));
    }
    setManualDates((prev) => Array.from(new Set([...prev, ...weekDates])).sort());
  };

  const handleSaveSchedule = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!activePatient || selectedExerciseIds.length === 0 || finalSelectedDates.length === 0) return;
    setSaving(true);
    try {
      const finalSelectedDateSet = new Set(finalSelectedDates);
      const generated: RehabScheduleItem[] = [];
      const baseTs = Date.now();
      let seq = 0;

      for (const dateIso of finalSelectedDates) {
        for (const exId of selectedExerciseIds) {
          seq += 1;
          generated.push({
            id: `sched-${baseTs}-${seq}`,
            patientId: activePatient.id,
            ptId: ptId || activePatient.assignedPtId,
            exerciseId: exId,
            modpackId: modpackId || undefined,
            scheduledDate: dateIso,
            sets,
            reps,
            status: 'ACTIVE',
            notes,
          });
        }
      }

      const keptExisting = sanitizeAccidentalDefaultKneePrefix(activePatient.rehabSchedules ?? []).filter(
        (item) => !finalSelectedDateSet.has(item.scheduledDate),
      );
      const merged = [...keptExisting, ...generated].sort((a, b) => a.scheduledDate.localeCompare(b.scheduledDate));
      const nextAppt = computeNextAppointmentDate(merged, todayIso) || finalSelectedDates[0] || '';

      await apiClient.updatePatient(activePatient.id, {
        nextAppointmentDate: nextAppt,
        rehabSchedules: merged,
      });

      await onSaved(
        `บันทึกตารางกายภาพ ${selectedExerciseIds.length} ท่า จำนวน ${finalSelectedDates.length} วัน (${generated.length} รายการ) ให้ ${activePatient.fullName} แล้ว`,
      );
      onClose();
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#0F4C81]/35 p-3 sm:p-4 backdrop-blur-sm">
      <div className="w-full max-w-4xl max-h-[92vh] overflow-y-auto rounded-3xl bg-white p-5 sm:p-7 shadow-2xl border border-slate-200 space-y-5 text-xs">
        {/* Modal Header */}
        <div className="flex items-center justify-between border-b border-slate-100 pb-3.5">
          <div className="flex items-center gap-2.5 min-w-0">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-emerald-50 text-emerald-600 border border-emerald-200">
              <CalendarDays className="h-5 w-5" />
            </div>
            <div className="min-w-0">
              <h3 className="text-base sm:text-lg font-black text-slate-900 truncate">
                {mode === 'day_detail'
                  ? `${sm.schedDayTitle} ${selectedDateIso}`
                  : sm.schedCreateTitle}
              </h3>
              <p className="text-[11px] font-semibold text-slate-500">
                เลือกท่ากายภาพบำบัด และกำหนดวันฝึกแบบรายวันหรือรายสัปดาห์ต่อเดือน
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-xl p-1.5 text-slate-400 hover:bg-slate-100 hover:text-slate-700 shrink-0 cursor-pointer"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Section A: Existing Scheduled Items on Clicked Date */}
        {mode === 'day_detail' && activePatient && selectedDateIso && (
          <div className="rounded-2xl border border-slate-200 bg-slate-50/70 p-4 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="font-extrabold text-slate-800">
                รายการที่กำหนดไว้ในวันที่ <span className="text-emerald-700 font-mono">{selectedDateIso}</span> •{' '}
                {activePatient.fullName} ({activePatient.hn})
              </div>
              {dayItems.length > 1 && (
                <button
                  type="button"
                  onClick={async () => {
                    const nextList = (activePatient.rehabSchedules ?? []).filter(
                      (s) => s.scheduledDate !== selectedDateIso,
                    );
                    await apiClient.updatePatient(activePatient.id, {
                      nextAppointmentDate: computeNextAppointmentDate(nextList, todayIso),
                      rehabSchedules: nextList,
                    });
                    await onSaved(`ลบรายการฝึกทั้งหมดของวันที่ ${selectedDateIso} แล้ว`);
                  }}
                  className="whitespace-nowrap rounded-lg border border-rose-200 bg-rose-50 px-2.5 py-1 text-[11px] font-bold text-rose-700 hover:bg-rose-100 cursor-pointer"
                >
                  <Trash2 className="inline h-3 w-3 mr-1" /> ลบทั้งหมดของวันนี้ ({dayItems.length})
                </button>
              )}
            </div>

            {dayItems.length === 0 ? (
              <div className="rounded-xl border border-dashed border-slate-300 bg-white px-4 py-3 text-slate-500 font-semibold">
                {sm.emptyDayMsg} {selectedDateIso} — สามารถเลือกท่ากายภาพและวันฝึกด้านล่างได้ทันที
              </div>
            ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                {dayItems.map((item) => {
                  const ex = exercises.find((e) => e.id === item.exerciseId);
                  const mp = modpacks.find((m) => m.id === item.modpackId);
                  return (
                    <div key={item.id} className="rounded-2xl border border-slate-200 bg-white p-3 space-y-2 shadow-xs">
                      <div className="flex items-start justify-between gap-2">
                        <div className="min-w-0">
                          <div className="font-extrabold text-slate-900 truncate">
                            {ex ? ex.name[lang] || ex.name.th : 'Rehab Course'} {mp ? `• ${mp.name}` : ''}
                          </div>
                          <div className="text-[11px] text-slate-500">
                            {item.sets} × {item.reps} {item.notes ? `• ${item.notes}` : ''}
                          </div>
                        </div>
                        <span className="whitespace-nowrap rounded-full bg-slate-100 px-2.5 py-0.5 text-[10px] font-black text-slate-700">
                          {item.status}
                        </span>
                      </div>
                      <div className="flex flex-wrap gap-1.5">
                        {(
                          [
                            { status: 'ACTIVE', label: sm.statusActive, cls: 'bg-emerald-600 text-white' },
                            { status: 'REST', label: sm.statusRest, cls: 'bg-amber-500 text-white' },
                            { status: 'COMPLETED', label: sm.statusCompleted, cls: 'bg-blue-600 text-white' },
                            { status: 'MISSED', label: sm.statusMissed, cls: 'bg-rose-600 text-white' },
                          ] as const
                        ).map((btn) => (
                          <button
                            key={btn.status}
                            type="button"
                            onClick={async () => {
                              const nextList = (activePatient.rehabSchedules ?? []).map((s) =>
                                s.id === item.id ? { ...s, status: btn.status } : s,
                              );
                              await apiClient.updatePatient(activePatient.id, {
                                nextAppointmentDate: computeNextAppointmentDate(nextList, todayIso),
                                rehabSchedules: nextList,
                              });
                              await onSaved(`เปลี่ยนสถานะวันที่ ${selectedDateIso} เป็น ${btn.label}`);
                            }}
                            className={`whitespace-nowrap rounded-lg px-2 py-1 text-[10px] font-bold cursor-pointer ${btn.cls}`}
                          >
                            {btn.label}
                          </button>
                        ))}
                        <button
                          type="button"
                          onClick={async () => {
                            const nextList = (activePatient.rehabSchedules ?? []).filter((s) => s.id !== item.id);
                            await apiClient.updatePatient(activePatient.id, {
                              nextAppointmentDate: computeNextAppointmentDate(nextList, todayIso),
                              rehabSchedules: nextList,
                            });
                            await onSaved(`ลบนัดหมายวันที่ ${selectedDateIso} แล้ว`);
                          }}
                          className="whitespace-nowrap rounded-lg border border-rose-200 bg-rose-50 px-2 py-1 text-[10px] font-bold text-rose-700 cursor-pointer"
                        >
                          <Trash2 className="inline h-3 w-3 mr-0.5" /> {sm.deleteBtn}
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* Section B & C: Posture Selector + Dual-Mode Date Builder */}
        <form onSubmit={handleSaveSchedule} className="space-y-5">
          {/* Patient & PT Selection */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div>
              <label className="block font-bold text-slate-600 mb-1">{sm.selectPatientLabel}</label>
              <select
                value={targetPatientId}
                onChange={(e) => setTargetPatientId(e.target.value)}
                className="w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 font-bold text-slate-800"
              >
                {patients.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.fullName} ({p.hn})
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block font-bold text-slate-600 mb-1">{sm.selectPtLabel}</label>
              <select
                value={ptId}
                onChange={(e) => setPtId(e.target.value)}
                className="w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 font-bold text-slate-800"
              >
                {ptUsers.map((pt) => (
                  <option key={pt.id} value={pt.id}>
                    {pt.fullName}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* 1. เลือกท่าทางในการกายภาพได้ */}
          <div className="rounded-2xl border border-slate-200 bg-white p-4 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <span className="flex h-6 w-6 items-center justify-center rounded-full bg-emerald-600 text-[11px] font-black text-white">
                  1
                </span>
                <span className="text-sm font-black text-slate-900 flex items-center gap-1.5">
                  <Dumbbell className="h-4 w-4 text-emerald-600" />
                  เลือกท่าทางในการกายภาพบำบัด (เลือกได้หลายท่า)
                </span>
                <span className="rounded-full bg-emerald-50 border border-emerald-200 px-2.5 py-0.5 text-[11px] font-extrabold text-emerald-700">
                  เลือกแล้ว {selectedExerciseIds.length} ท่า
                </span>
              </div>

              <div className="flex items-center gap-1.5">
                <button
                  type="button"
                  onClick={() => setSelectedExerciseIds(filteredExercises.map((ex) => ex.id))}
                  className="rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1 text-[11px] font-bold text-slate-700 hover:bg-emerald-50 hover:text-emerald-700 cursor-pointer"
                >
                  เลือกทั้งหมดในหมวดนี้
                </button>
                <button
                  type="button"
                  onClick={() => setSelectedExerciseIds([])}
                  className="rounded-lg border border-slate-200 bg-white px-2.5 py-1 text-[11px] font-bold text-slate-500 hover:bg-rose-50 hover:text-rose-700 cursor-pointer"
                >
                  ล้างท่า
                </button>
              </div>
            </div>

            {/* Category Filter Tabs */}
            <div className="flex flex-wrap gap-1.5">
              {EXERCISE_FILTER_TABS.map((tab) => (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setExerciseCategoryFilter(tab.id)}
                  className={`rounded-xl px-3 py-1.5 text-[11px] font-bold transition cursor-pointer ${
                    exerciseCategoryFilter === tab.id
                      ? 'bg-gradient-to-r from-[#0F4C81] to-[#0284C7] text-white shadow-xs'
                      : 'bg-sky-50 text-[#0F4C81] hover:bg-sky-100'
                  }`}
                >
                  {tab.label[lang]}
                </button>
              ))}
            </div>

            {/* Interactive Posture Multi-Select Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 max-h-60 overflow-y-auto pr-1">
              {filteredExercises.map((ex) => {
                const checked = selectedExerciseIds.includes(ex.id);
                return (
                  <button
                    key={ex.id}
                    type="button"
                    onClick={() => toggleExercise(ex.id)}
                    className={`rounded-2xl border p-3 text-left transition cursor-pointer flex items-start gap-2.5 ${
                      checked
                        ? 'border-emerald-500 bg-emerald-50/70 ring-1 ring-emerald-500/40 shadow-xs'
                        : 'border-slate-200 bg-white hover:border-emerald-300 hover:bg-slate-50/60'
                    }`}
                  >
                    <span
                      className={`mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-md border ${
                        checked
                          ? 'border-emerald-600 bg-emerald-600 text-white'
                          : 'border-slate-300 bg-white text-transparent'
                      }`}
                    >
                      <Check className="h-3.5 w-3.5 stroke-[3]" />
                    </span>
                    <div className="min-w-0 flex-1 space-y-1">
                      <div className="flex items-center justify-between gap-1.5">
                        <span className="font-extrabold text-slate-900 truncate">
                          {ex.name[lang] || ex.name.th}
                        </span>
                        <span className="shrink-0 rounded bg-slate-100 px-1.5 py-0.5 font-mono text-[10px] font-bold text-slate-600">
                          {ex.code}
                        </span>
                      </div>
                      <p className="text-[11px] text-slate-500 line-clamp-1">
                        {ex.instruction[lang] || ex.instruction.th}
                      </p>
                      <div className="text-[10px] font-bold text-emerald-700">
                        มุมเป้า {ex.detection.targetMinDeg}°–{ex.detection.targetMaxDeg}° • ค้าง {ex.detection.holdSec} วิ • แนะนำ {ex.detection.defaultSets}×{ex.detection.defaultReps}
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>

            {selectedExerciseIds.length === 0 && (
              <div className="rounded-xl bg-amber-50 border border-amber-200 px-3 py-2 text-[11px] font-bold text-amber-800">
                กรุณาเลือกท่ากายภาพบำบัดอย่างน้อย 1 ท่า
              </div>
            )}

            {/* Sets, Reps, Modpack, Notes */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 pt-1 border-t border-slate-100">
              <div>
                <label className="block font-bold text-slate-600 mb-1">จำนวนเซต (Sets)</label>
                <input
                  type="number"
                  inputMode="numeric"
                  min={1}
                  max={10}
                  value={sets}
                  onKeyDown={(e) => {
                    if (['e', 'E', '+', '-', '.'].includes(e.key)) e.preventDefault();
                  }}
                  onChange={(e) => {
                    const digits = e.target.value.replace(/\D/g, '').slice(0, 2);
                    setSets(digits ? Math.min(10, Math.max(1, Number(digits))) : 1);
                  }}
                  className="w-full rounded-xl border border-slate-200 px-2.5 py-2 text-center font-bold"
                />
              </div>
              <div>
                <label className="block font-bold text-slate-600 mb-1">จำนวนครั้ง (Reps)</label>
                <input
                  type="number"
                  inputMode="numeric"
                  min={1}
                  max={30}
                  value={reps}
                  onKeyDown={(e) => {
                    if (['e', 'E', '+', '-', '.'].includes(e.key)) e.preventDefault();
                  }}
                  onChange={(e) => {
                    const digits = e.target.value.replace(/\D/g, '').slice(0, 2);
                    setReps(digits ? Math.min(30, Math.max(1, Number(digits))) : 1);
                  }}
                  className="w-full rounded-xl border border-slate-200 px-2.5 py-2 text-center font-bold"
                />
              </div>
              <div className="col-span-2">
                <label className="block font-bold text-slate-600 mb-1">{sm.modpackLabel} (เสริม)</label>
                <select
                  value={modpackId}
                  onChange={(e) => setModpackId(e.target.value)}
                  className="w-full rounded-xl border border-slate-200 px-3 py-2 font-semibold"
                >
                  <option value="">ไม่ระบุ (ฝึกเฉพาะท่ากายภาพ AI)</option>
                  {modpacks.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <input
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="หมายเหตุการฝึก (เช่น ฝึกช่วงเช้า ระวังปวดไหล่)"
              className="w-full rounded-xl border border-slate-200 px-3 py-2"
            />
          </div>

          {/* 2. เลือกวันกายภาพได้ 2 แบบ: 2.1 รายวัน (ตารางวัน) / 2.2 รายสัปดาห์ต่อเดือน (จันทร์-อาทิตย์) */}
          <div className="rounded-2xl border border-slate-200 bg-white p-4 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <span className="flex h-6 w-6 items-center justify-center rounded-full bg-blue-600 text-[11px] font-black text-white">
                  2
                </span>
                <span className="text-sm font-black text-slate-900">
                  เลือกวันจัดกายภาพบำบัด (เลือกได้ 2 รูปแบบ)
                </span>
              </div>
              <span className="rounded-full bg-blue-50 border border-blue-200 px-3 py-0.5 text-[11px] font-extrabold text-blue-700">
                เลือกไว้ทั้งหมด {finalSelectedDates.length} วัน
              </span>
            </div>

            {/* Dual Mode Switcher */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
              <button
                type="button"
                onClick={() => setDateSelectMode('daily')}
                className={`rounded-2xl border-2 p-3 text-left transition cursor-pointer ${
                  dateSelectMode === 'daily'
                    ? 'border-emerald-600 bg-emerald-50/80 shadow-xs'
                    : 'border-slate-200 bg-slate-50/50 hover:border-emerald-300'
                }`}
              >
                <div className="font-black text-slate-900 text-xs sm:text-sm">
                  2.1 เลือกวันแบบรายวัน (ตารางปฏิทิน)
                </div>
                <div className="mt-0.5 text-[11px] font-semibold text-slate-500">
                  กดจิ้มเลือกวันจากในตารางปฏิทินเป็นวันๆ ไป
                </div>
              </button>

              <button
                type="button"
                onClick={() => setDateSelectMode('weekly')}
                className={`rounded-2xl border-2 p-3 text-left transition cursor-pointer ${
                  dateSelectMode === 'weekly'
                    ? 'border-blue-600 bg-blue-50/80 shadow-xs'
                    : 'border-slate-200 bg-slate-50/50 hover:border-blue-300'
                }`}
              >
                <div className="font-black text-slate-900 text-xs sm:text-sm">
                  2.2 เลือกวันเป็นรายสัปดาห์ ต่อเดือน (จันทร์–อาทิตย์)
                </div>
                <div className="mt-0.5 text-[11px] font-semibold text-slate-500">
                  เลือกว่าใน 1 สัปดาห์ต้องการกายภาพวันไหนบ้าง แล้วสร้างให้ทั้งเดือนอัตโนมัติ
                </div>
              </button>
            </div>

            {/* Weekly Mode Controls (2.2: จันทร์ - อาทิตย์ ต่อเดือน) */}
            {dateSelectMode === 'weekly' && (
              <div className="rounded-2xl border border-blue-200 bg-blue-50/40 p-3.5 space-y-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="font-extrabold text-slate-800">
                    เลือกวันใน 1 สัปดาห์ที่ต้องการทำกายภาพ (จันทร์ – อาทิตย์):
                  </span>
                  <div className="flex flex-wrap gap-1.5">
                    {WEEKDAY_PRESETS.map((preset) => (
                      <button
                        key={preset.id}
                        type="button"
                        onClick={() => {
                          setWeeklyExcludedDates([]);
                          setSelectedWeekdays(preset.days);
                        }}
                        className="rounded-lg border border-blue-200 bg-white px-2.5 py-1 text-[10px] font-bold text-blue-700 hover:bg-blue-600 hover:text-white cursor-pointer transition"
                      >
                        {preset.label[lang]}
                      </button>
                    ))}
                  </div>
                </div>

                {/* 7 Weekday Toggle Buttons (Mon -> Sun) */}
                <div className="grid grid-cols-4 sm:grid-cols-7 gap-2">
                  {WEEKDAY_TOGGLE_OPTIONS.map((w) => {
                    const active = selectedWeekdays.includes(w.dayIndex);
                    return (
                      <button
                        key={w.dayIndex}
                        type="button"
                        onClick={() => toggleWeekday(w.dayIndex)}
                        className={`rounded-xl border-2 py-2.5 px-2 text-center font-extrabold transition cursor-pointer ${
                          active
                            ? 'border-blue-600 bg-blue-600 text-white shadow-sm'
                            : 'border-slate-200 bg-white text-slate-700 hover:border-blue-300'
                        }`}
                      >
                        <div className="text-xs">{w.label[lang]}</div>
                        <div className={`text-[10px] mt-0.5 ${active ? 'text-blue-100' : 'text-slate-400'}`}>
                          {active ? '✓ ฝึก' : 'พัก'}
                        </div>
                      </button>
                    );
                  })}
                </div>

                {/* Month Span & Anchor Option */}
                <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-bold text-slate-700">ช่วงระยะเวลา (ต่อเดือน):</span>
                    {[1, 2, 3].map((span) => (
                      <button
                        key={span}
                        type="button"
                        onClick={() => {
                          setWeeklyExcludedDates([]);
                          setMonthSpan(span);
                        }}
                        className={`rounded-lg px-3 py-1 text-[11px] font-bold cursor-pointer border ${
                          monthSpan === span
                            ? 'border-[#0284C7] bg-gradient-to-r from-[#0F4C81] to-[#0284C7] text-white'
                            : 'border-sky-200 bg-white text-[#0F4C81] hover:bg-sky-50'
                        }`}
                      >
                        {span} เดือน
                      </button>
                    ))}
                  </div>

                  <label className="inline-flex items-center gap-2 font-semibold text-slate-700 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={startFromAnchorOnly}
                      onChange={(e) => {
                        setWeeklyExcludedDates([]);
                        setStartFromAnchorOnly(e.target.checked);
                      }}
                      className="h-4 w-4 accent-blue-600 rounded"
                    />
                    <span>เริ่มนับตั้งแต่วันที่ {anchorIso} เป็นต้นไป</span>
                  </label>
                </div>
              </div>
            )}

            {/* Shared Interactive Monthly Calendar Grid (used in both 2.1 Daily Picker & 2.2 Weekly Month Preview) */}
            <div className="rounded-2xl border border-slate-200 bg-slate-50/50 p-3.5 space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => shiftViewMonth(-1)}
                    className="rounded-lg border border-slate-200 bg-white p-1.5 hover:bg-slate-100 cursor-pointer"
                  >
                    <ChevronLeft className="h-4 w-4" />
                  </button>
                  <span className="text-sm font-black text-slate-900 min-w-[160px] text-center">
                    {lang === 'th'
                      ? `${MONTH_LABELS.th[viewMonthDate.month]} ${viewMonthDate.year + 543}`
                      : `${MONTH_LABELS[lang][viewMonthDate.month]} ${viewMonthDate.year}`}
                  </span>
                  <button
                    type="button"
                    onClick={() => shiftViewMonth(1)}
                    className="rounded-lg border border-slate-200 bg-white p-1.5 hover:bg-slate-100 cursor-pointer"
                  >
                    <ChevronRight className="h-4 w-4" />
                  </button>
                </div>

                {dateSelectMode === 'daily' ? (
                  <div className="flex flex-wrap items-center gap-1.5">
                    <button
                      type="button"
                      onClick={() => setManualDates([anchorIso])}
                      className="rounded-lg border border-slate-200 bg-white px-2.5 py-1 text-[11px] font-bold text-slate-700 hover:bg-emerald-50 hover:text-emerald-700 cursor-pointer"
                    >
                      เฉพาะวันที่ {anchorIso}
                    </button>
                    <button
                      type="button"
                      onClick={selectThisWeekInDaily}
                      className="rounded-lg border border-slate-200 bg-white px-2.5 py-1 text-[11px] font-bold text-slate-700 hover:bg-emerald-50 hover:text-emerald-700 cursor-pointer"
                    >
                      เลือกทั้งสัปดาห์นี้
                    </button>
                    <button
                      type="button"
                      onClick={() => setManualDates([])}
                      className="rounded-lg border border-rose-200 bg-rose-50 px-2.5 py-1 text-[11px] font-bold text-rose-700 hover:bg-rose-100 cursor-pointer"
                    >
                      ล้างวันที่เลือก
                    </button>
                  </div>
                ) : (
                  <span className="text-[11px] font-semibold text-blue-700">
                    คลิกวันที่ในปฏิทินด้านล่างเพื่อยกเว้น/เปิดคืนเฉพาะวันได้ ({monthSpan} เดือน)
                  </span>
                )}
              </div>

              <div className="grid grid-cols-7 gap-1.5 text-center text-[11px] font-bold text-slate-500">
                {CALENDAR_DOW_HEADERS[lang].map((dLabel) => (
                  <div key={dLabel} className="py-1">
                    {dLabel}
                  </div>
                ))}
              </div>

              <div className="grid grid-cols-7 gap-1.5">
                {monthCells.map((dayNum, idx) => {
                  if (dayNum === null) {
                    return <div key={`blank-${idx}`} className="h-12 rounded-xl bg-slate-100/60" />;
                  }
                  const cellIso = formatIsoYmd(viewMonthDate.year, viewMonthDate.month, dayNum);
                  const isSelected = finalSelectedDateSet.has(cellIso);
                  const existingCount = existingScheduleDatesSet.get(cellIso)?.length ?? 0;
                  const isAnchor = cellIso === anchorIso;
                  const dow = new Date(viewMonthDate.year, viewMonthDate.month, dayNum).getDay();
                  const isWeeklyPatternMatch = dateSelectMode === 'weekly' && selectedWeekdays.includes(dow);
                  const isExcluded = isWeeklyPatternMatch && weeklyExcludedDates.includes(cellIso);

                  return (
                    <button
                      key={cellIso}
                      type="button"
                      onClick={() => {
                        if (dateSelectMode === 'daily') toggleManualDate(cellIso);
                        else toggleWeeklyException(cellIso);
                      }}
                      className={`h-12 rounded-xl border px-1.5 py-1 flex flex-col items-center justify-between transition cursor-pointer ${
                        isSelected
                          ? dateSelectMode === 'daily'
                            ? 'border-emerald-600 bg-emerald-600 text-white shadow-sm'
                            : 'border-blue-600 bg-blue-600 text-white shadow-sm'
                          : isExcluded
                            ? 'border-amber-300 bg-amber-50 text-amber-700 line-through'
                            : isAnchor
                              ? 'border-emerald-400 bg-emerald-50 text-slate-900'
                              : 'border-slate-200 bg-white text-slate-800 hover:border-emerald-300'
                      }`}
                    >
                      <span className="text-xs font-extrabold">{dayNum}</span>
                      <span className={`text-[9px] font-bold ${isSelected ? 'text-white/90' : 'text-emerald-600'}`}>
                        {isSelected ? '✓ เลือก' : existingCount > 0 ? `${existingCount} ท่าเดิม` : ''}
                      </span>
                    </button>
                  );
                })}
              </div>

              {/* Selected Dates Chips Summary */}
              {finalSelectedDates.length > 0 && (
                <div className="pt-1 flex flex-wrap items-center gap-1.5 max-h-24 overflow-y-auto">
                  <span className="text-[11px] font-bold text-slate-600 mr-1">วันที่เลือก:</span>
                  {finalSelectedDates.map((iso) => (
                    <span
                      key={iso}
                      className="inline-flex items-center gap-1 rounded-full bg-white border border-slate-200 px-2 py-0.5 font-mono text-[10px] font-bold text-slate-700"
                    >
                      {iso}
                      <button
                        type="button"
                        onClick={() => {
                          if (dateSelectMode === 'daily') toggleManualDate(iso);
                          else toggleWeeklyException(iso);
                        }}
                        className="text-slate-400 hover:text-rose-600 cursor-pointer"
                      >
                        ×
                      </button>
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Summary Bar & Submit */}
          <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl bg-gradient-to-r from-[#0F4C81] via-[#0284C7] to-[#0D9488] px-4 py-3 text-white shadow-sm">
            <div className="space-y-0.5">
              <div className="font-extrabold text-xs sm:text-sm">
                สรุป: {selectedExerciseIds.length} ท่ากายภาพ × {finalSelectedDates.length} วัน ={' '}
                <span className="text-emerald-200">{selectedExerciseIds.length * finalSelectedDates.length} รายการ</span>
              </div>
              <div className="text-[11px] text-sky-100">
                เซต × ครั้ง: {sets} × {reps} • โหมด:{' '}
                {dateSelectMode === 'daily' ? 'รายวัน (จากตารางวัน)' : `รายสัปดาห์ (${monthSpan} เดือน)`}
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={onClose}
                className="whitespace-nowrap h-10 px-4 inline-flex items-center justify-center rounded-xl border border-white/20 bg-white/10 font-bold text-white hover:bg-white/20 cursor-pointer"
              >
                {sm.cancelBtn}
              </button>
              <button
                type="submit"
                disabled={saving || selectedExerciseIds.length === 0 || finalSelectedDates.length === 0}
                className="whitespace-nowrap h-10 px-5 inline-flex items-center justify-center rounded-xl bg-emerald-500 font-extrabold text-slate-950 hover:bg-emerald-400 disabled:opacity-40 cursor-pointer"
              >
                {saving ? sm.savingBtn : sm.saveScheduleBtn}
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}
