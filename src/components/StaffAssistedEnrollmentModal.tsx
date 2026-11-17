import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  formatLocalIsoDate,
  MAHIDOL_MOTOR_SCALE,
  resolveNextAppointmentDate,
  type AffectedSide,
  type DrTechCaresApiClient,
  type ManagedExerciseRecord,
  type ModpackRecord,
  type PatientRecord,
  type RehabScheduleItem,
  type StrokeType,
  type UserAccount,
} from '@drtechcares/api-client';
import { useClinicalI18n, type ClinicalLang } from '@drtechcares/clinical-i18n';
import { CLINICAL_EXERCISES, sortExercisesStrokeUpperFirst } from '@drtechcares/pose-engine';
import {
  Activity,
  CalendarDays,
  Camera,
  Check,
  CheckCircle2,
  Dumbbell,
  HeartPulse,
  Plus,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  UserPlus,
  X,
} from 'lucide-react';
import FaceID120Scanner, { type FaceID120Result } from './FaceID120Scanner';
import { syncPatientBiometricEverywhere } from '../lib/biometricEngine';

interface StaffAssistedEnrollmentModalProps {
  open: boolean;
  apiClient: DrTechCaresApiClient;
  currentUser: UserAccount;
  ptList: UserAccount[];
  exercises?: ManagedExerciseRecord[];
  modpacks?: ModpackRecord[];
  targetPatient?: PatientRecord | null;
  onClose: () => void;
  onEnrolled: (patient: PatientRecord) => void;
}

type ExerciseFilterCategory = 'all' | 'upper_limb' | 'lower_limb' | 'neck_trunk' | 'balance';

const ENROLLMENT_EXERCISE_FILTER_TABS: ReadonlyArray<{
  id: ExerciseFilterCategory;
  label: Record<ClinicalLang, string>;
}> = [
  { id: 'all', label: { th: 'ทั้งหมด', en: 'All Postures', cn: '全部动作' } },
  { id: 'upper_limb', label: { th: 'แขนและไหล่', en: 'Upper Limb', cn: '上肢/肩肘' } },
  { id: 'lower_limb', label: { th: 'ขาและเข่า', en: 'Lower Limb', cn: '下肢/膝踝' } },
  { id: 'neck_trunk', label: { th: 'คอและลำตัว', en: 'Neck & Trunk', cn: '颈部/躯干' } },
  { id: 'balance', label: { th: 'ฝึกทรงตัว', en: 'Balance', cn: '平衡训练' } },
];

const FALLBACK_MANAGED_EXERCISES: ManagedExerciseRecord[] = CLINICAL_EXERCISES.map((ex) => ({
  ...ex,
  createdBy: 'system',
}));

function generatePatientHn(): string {
  return `HN-${Math.floor(10000 + Math.random() * 89999)}`;
}

function addDaysToIso(baseIso: string, deltaDays: number): string {
  const [y, m, d] = baseIso.split('-').map(Number);
  if (!y || !m || !d) return formatLocalIsoDate();
  return formatLocalIsoDate(new Date(y, m - 1, d + deltaDays));
}

function buildTriWeeklyDates(startIso: string, count = 6): string[] {
  const [y, m, d] = startIso.split('-').map(Number);
  if (!y || !m || !d) return [startIso];
  const result: string[] = [];
  for (let offset = 0; offset < 28 && result.length < count; offset += 1) {
    const dt = new Date(y, m - 1, d + offset);
    const dow = dt.getDay();
    if (dow === 1 || dow === 3 || dow === 5) {
      result.push(formatLocalIsoDate(dt));
    }
  }
  return result.length > 0 ? result : [startIso];
}

function getRecommendedExerciseIdsForJoint(
  jointLabel: string,
  catalog: readonly ManagedExerciseRecord[],
): string[] {
  const preferredByJoint: Record<string, string[]> = {
    'ข้อไหล่และแขน Stroke ADL (Shoulder & Arm)': ['ex-27', 'ex-29', 'ex-31', 'ex-33', 'ex-35'],
    'คอ บ่า และแขน สธ. ผู้สูงอายุ (MOPH Upper Body)': ['ex-37', 'ex-38', 'ex-39', 'ex-40'],
    'ข้อไหล่ (Shoulder)': ['ex-27', 'ex-29', 'ex-01', 'ex-02'],
    'ข้อศอกและแขน (Elbow & Arm)': ['ex-31', 'ex-32', 'ex-35', 'ex-05', 'ex-06'],
    'หลังส่วนล่าง (Lower Back)': ['ex-23', 'ex-39', 'ex-15', 'ex-16'],
    'เข่า (Knee)': ['ex-13', 'ex-14', 'ex-26'],
    'การทรงตัวและสมอง (Neuro & Cognitive)': ['ex-21', 'ex-22', 'ex-39', 'ex-40'],
  };
  const preferred = preferredByJoint[jointLabel] ?? ['ex-27', 'ex-29', 'ex-37', 'ex-39'];
  const validSet = new Set(catalog.map((e) => e.id));
  const filtered = preferred.filter((id) => validSet.has(id));
  if (filtered.length > 0) return filtered;
  return catalog.slice(0, 3).map((e) => e.id);
}

export function StaffAssistedEnrollmentModal({
  open,
  apiClient,
  currentUser,
  ptList,
  exercises = [],
  modpacks = [],
  targetPatient = null,
  onClose,
  onEnrolled,
}: StaffAssistedEnrollmentModalProps) {
  const { lang, setLang } = useClinicalI18n();
  const isExistingPatientMode = Boolean(targetPatient);
  const [step, setStep] = useState<1 | 2 | 3 | 4>(1);
  const todayIso = useMemo(() => formatLocalIsoDate(), []);
  const catalogExercises = useMemo(
    () =>
      sortExercisesStrokeUpperFirst(
        exercises.length > 0 ? exercises : FALLBACK_MANAGED_EXERCISES,
      ),
    [exercises],
  );

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

  // Section 2: Clinical & Rehab (Stroke & Elderly Upper-Body Focus + Mahidol 0-6 Scale)
  const [targetJoint, setTargetJoint] = useState('ข้อไหล่และแขน Stroke ADL (Shoulder & Arm)');
  const [strokeType, setStrokeType] = useState<StrokeType>('ISCHEMIC');
  const [affectedSide, setAffectedSide] = useState<AffectedSide>('RIGHT');
  const [motorAssessmentScore, setMotorAssessmentScore] = useState<number>(3);
  const [adlDressingScore, setAdlDressingScore] = useState<number>(3);
  const [maxSafeRomDeg, setMaxSafeRomDeg] = useState<number>(120);
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

  // Step 2 Camera 120-frame 3D Face ID state
  const [capturing, setCapturing] = useState(false);
  const [allFiveAnglesReady, setAllFiveAnglesReady] = useState(false);
  const [biometric120Result, setBiometric120Result] = useState<FaceID120Result | null>(null);
  const [capturedSnapshotUrl, setCapturedSnapshotUrl] = useState<string | undefined>(undefined);
  const [pendingSkipFaceScan, setPendingSkipFaceScan] = useState(false);

  // Step 3 Initial Rehab Prescription state
  const [exerciseCategoryFilter, setExerciseCategoryFilter] = useState<ExerciseFilterCategory>('all');
  const [selectedExerciseIds, setSelectedExerciseIds] = useState<string[]>([]);
  const [prescriptionTouched, setPrescriptionTouched] = useState(false);
  const [prescriptionSets, setPrescriptionSets] = useState(3);
  const [prescriptionReps, setPrescriptionReps] = useState(10);
  const [selectedModpackId, setSelectedModpackId] = useState('');
  const [scheduleNotes, setScheduleNotes] = useState('');
  const [scheduledDates, setScheduledDates] = useState<string[]>(() => [todayIso]);
  const [customDateInput, setCustomDateInput] = useState(todayIso);

  const [saving, setSaving] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [toastMsg, setToastMsg] = useState<string | null>(null);
  const [createdPatient, setCreatedPatient] = useState<PatientRecord | null>(null);
  const triggerFinalizeRef = useRef<(() => Promise<FaceID120Result | null>) | null>(null);

  useEffect(() => {
    if (!open) {
      setCapturing(false);
      return;
    }
    setBiometric120Result(null);
    setAllFiveAnglesReady(false);
    setPendingSkipFaceScan(false);
    triggerFinalizeRef.current = null;
    setCreatedPatient(null);
    setCapturedSnapshotUrl(undefined);
    setErrorMsg(null);
    setToastMsg(null);
    setExerciseCategoryFilter('all');
    setPrescriptionTouched(false);
    setSelectedExerciseIds([]);
    setPrescriptionSets(3);
    setPrescriptionReps(10);
    setSelectedModpackId('');
    setScheduleNotes('');
    setScheduledDates([todayIso]);
    setCustomDateInput(todayIso);
    if (targetPatient) {
      setHn(targetPatient.hn);
      setFullName(targetPatient.fullName);
      setAge(targetPatient.age);
      setPhone(targetPatient.phone);
      setStep(2);
      setCapturing(true);
    } else {
      setStep(1);
    }
  }, [open, targetPatient, catalogExercises, todayIso]);

  useEffect(() => {
    if (!open || targetPatient) return;
    if (!assignedPtId && ptList.length > 0) {
      setAssignedPtId(currentUser.role === 'pt' ? currentUser.id : ptList[0].id);
    }
  }, [open, targetPatient, ptList, assignedPtId, currentUser.id, currentUser.role]);

  const filteredExercises = useMemo(() => {
    if (exerciseCategoryFilter === 'all') return catalogExercises;
    return catalogExercises.filter((ex) => ex.category === exerciseCategoryFilter);
  }, [catalogExercises, exerciseCategoryFilter]);

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
    setTargetJoint('ข้อไหล่และแขน Stroke ADL (Shoulder & Arm)');
    setStrokeType('ISCHEMIC');
    setAffectedSide('RIGHT');
    setMotorAssessmentScore(3);
    setAdlDressingScore(3);
    setMaxSafeRomDeg(120);
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
    setBiometric120Result(null);
    setAllFiveAnglesReady(false);
    setPendingSkipFaceScan(false);
    triggerFinalizeRef.current = null;
    setCapturedSnapshotUrl(undefined);
    setCapturing(false);
    setExerciseCategoryFilter('all');
    setPrescriptionTouched(false);
    setSelectedExerciseIds([]);
    setPrescriptionSets(3);
    setPrescriptionReps(10);
    setSelectedModpackId('');
    setScheduleNotes('');
    setScheduledDates([todayIso]);
    setCustomDateInput(todayIso);
    setCreatedPatient(null);
    setErrorMsg(null);
    setToastMsg(null);
  };

  const validateStep1 = (): boolean => {
    if (!fullName.trim()) {
      setErrorMsg('กรุณากรอกชื่อ-นามสกุล คนไข้');
      return false;
    }
    if (!pdpaAccepted) {
      setErrorMsg('กรุณายืนยันความยินยอม PDPA Biometric Consent ก่อนดำเนินการต่อ');
      return false;
    }
    return true;
  };

  const syncDefaultPrescriptionIfUntouched = () => {
    // Do not auto-preselect exercises behind the user's back; require explicit selection or quick-fill button.
  };

  const handleGoStep2 = (e: React.FormEvent) => {
    e.preventDefault();
    if (!validateStep1()) return;
    setErrorMsg(null);
    setPendingSkipFaceScan(false);
    setStep(2);
    setCapturing(true);
  };

  const handleSkipFaceToStep3 = () => {
    if (!validateStep1()) return;
    setErrorMsg(null);
    setPendingSkipFaceScan(true);
    setCapturing(false);
    syncDefaultPrescriptionIfUntouched();
    setStep(3);
  };

  const handleProceedFromStep2ToStep3 = async () => {
    setErrorMsg(null);
    let bio = biometric120Result;
    if (!bio && triggerFinalizeRef.current) {
      setSaving(true);
      try {
        const finalized = await triggerFinalizeRef.current();
        if (finalized) {
          bio = finalized;
          setBiometric120Result(finalized);
          setCapturedSnapshotUrl(finalized.avatarDataUrl);
        }
      } finally {
        setSaving(false);
      }
    }
    if (!bio) {
      setErrorMsg('กรุณาสแกนใบหน้าให้ครบ 5 มุมหรือครบ 120 เฟรมก่อนดำเนินการต่อ');
      return;
    }
    setPendingSkipFaceScan(false);
    setCapturing(false);
    syncDefaultPrescriptionIfUntouched();
    setStep(3);
  };

  const toggleExerciseSelection = (id: string) => {
    setPrescriptionTouched(true);
    setSelectedExerciseIds((prev) => (prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]));
  };

  const toggleScheduledDate = (iso: string) => {
    if (!iso) return;
    setScheduledDates((prev) => (prev.includes(iso) ? prev.filter((d) => d !== iso) : [...prev, iso].sort()));
  };

  const addScheduledDates = (datesToAdd: string[]) => {
    const valid = datesToAdd.filter(Boolean);
    if (valid.length === 0) return;
    setScheduledDates((prev) => Array.from(new Set([...prev, ...valid])).sort());
  };

  const buildPrescriptionSchedules = (): RehabScheduleItem[] => {
    if (selectedExerciseIds.length === 0 || scheduledDates.length === 0) return [];
    const sortedDates = [...scheduledDates].sort();
    const ptId = assignedPtId || currentUser.id;
    const baseTs = Date.now();
    const out: RehabScheduleItem[] = [];
    let seq = 0;
    for (const dateIso of sortedDates) {
      for (const exId of selectedExerciseIds) {
        seq += 1;
        out.push({
          id: `sched-${baseTs}-${seq}`,
          patientId: 'pending',
          ptId,
          exerciseId: exId,
          modpackId: selectedModpackId || undefined,
          scheduledDate: dateIso,
          sets: Math.max(1, Number(prescriptionSets) || 3),
          reps: Math.max(1, Number(prescriptionReps) || 10),
          status: 'ACTIVE',
          notes: scheduleNotes.trim() || undefined,
        });
      }
    }
    return out;
  };

  const handleFinalizeEnrollment = async (
    mode: 'verified' | 'pending_skip',
    overrideImageUrl?: string,
    overrideBiometric120?: FaceID120Result,
    customSchedules: RehabScheduleItem[] = [],
  ) => {
    setSaving(true);
    setErrorMsg(null);
    setToastMsg(null);
    try {
      const isPending = mode === 'pending_skip' && !isExistingPatientMode;
      let bio = overrideBiometric120 ?? biometric120Result;
      if (!isPending && !bio && triggerFinalizeRef.current) {
        const finalized = await triggerFinalizeRef.current();
        if (finalized) {
          bio = finalized;
          setBiometric120Result(finalized);
        }
      }
      const finalSnapshot =
        overrideImageUrl ?? bio?.compositeSheetDataUrl ?? bio?.avatarDataUrl ?? capturedSnapshotUrl;

      if (!isPending && !bio) {
        setErrorMsg('กรุณาสแกนใบหน้าให้ครบ 5 มุมหรือครบ 120 เฟรมก่อนบันทึกข้อมูล');
        setSaving(false);
        return;
      }

      if (targetPatient && bio) {
        const compatRawFrames: { pose: 'front' | 'left' | 'right' | 'nod'; vector: number[]; qualityScore: number }[] = [
          { pose: 'front', vector: bio.angleEmbeddings.center, qualityScore: bio.qualityScore },
          { pose: 'left', vector: bio.angleEmbeddings.left, qualityScore: bio.qualityScore },
          { pose: 'right', vector: bio.angleEmbeddings.right, qualityScore: bio.qualityScore },
          { pose: 'nod', vector: bio.angleEmbeddings.down, qualityScore: bio.qualityScore },
        ];
        const updated = await apiClient.enrollExistingPatientFace(targetPatient.id, {
          frontalCentroid: bio.angleEmbeddings.center,
          overallCentroid: bio.masterEmbedding,
          rawFrames: compatRawFrames,
          masterEmbedding: bio.masterEmbedding,
          angleEmbeddings: bio.angleEmbeddings,
          ...(bio.exemplarEmbeddings ? { exemplarEmbeddings: bio.exemplarEmbeddings } : {}),
          ...(bio.biometricEngine ? { biometricEngine: bio.biometricEngine } : {}),
          ...(bio.geometricRatios ? { geometricRatios: bio.geometricRatios } : {}),
          framesCaptured: bio.framesCaptured,
          qualityScore: bio.qualityScore,
          faceImageUrl: finalSnapshot,
          faceImageUrls: bio.faceImageUrls,
          angleImageUrls: bio.angleSnapshots,
        });
        await syncPatientBiometricEverywhere({
          apiClient,
          patient: updated,
          masterEmbedding: bio.masterEmbedding,
          angleEmbeddings: bio.angleEmbeddings,
          exemplarEmbeddings: bio.exemplarEmbeddings,
          geometricRatios: bio.geometricRatios,
          framesCaptured: bio.framesCaptured,
          qualityScore: bio.qualityScore,
        });
        setToastMsg('ลงทะเบียนคนไข้และบันทึก Face ID 120 เฟรมสำเร็จ!');
        setCreatedPatient(updated);
        setCapturing(false);
        setStep(4);
        onEnrolled(updated);
        return;
      }

      const cleanedDiagnosis =
        clinicalDiagnosis.trim() ||
        `${targetJoint}${underlyingDiseases.trim() ? ` | โรคประจำตัว: ${underlyingDiseases.trim()}` : ''}`;

      const effectiveNextAppointmentDate =
        customSchedules.length > 0
          ? resolveNextAppointmentDate({ rehabSchedules: customSchedules, todayIso })
          : '';

      let enrolled = await apiClient.enrollPatient10Shot({
        hn: hn.trim() || generatePatientHn(),
        fullName: fullName.trim(),
        age: Number(age) || 45,
        gender,
        phone: phone.trim(),
        diagnosis: cleanedDiagnosis,
        strokeType,
        affectedSide,
        motorAssessmentScore,
        adlDressingScore,
        maxSafeRomDeg,
        assignedPtId: assignedPtId || undefined,
        nextAppointmentDate: effectiveNextAppointmentDate,
        rehabSchedules: customSchedules,
        password: pinCode.trim() || '1234',
        skipFaceScan: isPending,
        faceBiometricStatus: isPending ? 'PENDING' : 'VERIFIED',
        faceImageUrl: finalSnapshot,
        faceImageUrls: !isPending && bio ? bio.faceImageUrls : undefined,
        angleImageUrls: !isPending && bio ? bio.angleSnapshots : undefined,
        frontalCentroid: !isPending && bio ? bio.angleEmbeddings.center : undefined,
        overallCentroid: !isPending && bio ? bio.masterEmbedding : undefined,
        masterEmbedding: !isPending && bio ? bio.masterEmbedding : undefined,
        angleEmbeddings: !isPending && bio ? bio.angleEmbeddings : undefined,
        ...(!isPending && bio?.exemplarEmbeddings ? { exemplarEmbeddings: bio.exemplarEmbeddings } : {}),
        ...(!isPending && bio?.biometricEngine ? { biometricEngine: bio.biometricEngine } : {}),
        ...(!isPending && bio?.geometricRatios ? { geometricRatios: bio.geometricRatios } : {}),
        framesCaptured: !isPending && bio ? bio.framesCaptured : 0,
        qualityScore: !isPending && bio ? bio.qualityScore : undefined,
      });

      if (!isPending && bio) {
        await syncPatientBiometricEverywhere({
          apiClient,
          patient: enrolled,
          masterEmbedding: bio.masterEmbedding,
          angleEmbeddings: bio.angleEmbeddings,
          exemplarEmbeddings: bio.exemplarEmbeddings,
          geometricRatios: bio.geometricRatios,
          framesCaptured: bio.framesCaptured,
          qualityScore: bio.qualityScore,
        });
      }

      if (customSchedules.length > 0 && (enrolled.rehabSchedules ?? []).length === 0) {
        const finalizedSchedules = customSchedules.map((s) => ({ ...s, patientId: enrolled.id }));
        const patched = await apiClient
          .updatePatient(enrolled.id, {
            nextAppointmentDate: effectiveNextAppointmentDate,
            rehabSchedules: finalizedSchedules,
          })
          .catch(() => null);
        enrolled = patched ?? {
          ...enrolled,
          nextAppointmentDate: effectiveNextAppointmentDate,
          rehabSchedules: finalizedSchedules,
        };
      }

      const prescribedUniqueCount = new Set(
        (enrolled.rehabSchedules ?? customSchedules).map((s) => s.exerciseId).filter(Boolean),
      ).size;

      const structuredSummary = [
        `[เปิดเวชระเบียน Staff-Assisted Stroke & Elderly Rehab] ข้อต่อเป้าหมาย: ${targetJoint}`,
        `กลุ่มโรค: ${strokeType} | ข้างที่อ่อนแรง/เน้น: ${affectedSide} | Mahidol Motor Score: ${motorAssessmentScore}/6 | ADL ใส่เสื้อผ้า: ${adlDressingScore}/5 | Max Safe ROM: ${maxSafeRomDeg}°`,
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
        isPending
          ? 'สถานะใบหน้า: PENDING (รอสแกนหน้าเพิ่มเติม 120 เฟรม)'
          : `สถานะใบหน้า: VERIFIED (3D Multi-Angle ${bio?.framesCaptured ?? 120} เฟรม)`,
        customSchedules.length > 0
          ? `ท่ากายภาพเริ่มต้น: ${prescribedUniqueCount} ท่า (${customSchedules.length} รายการ${
              effectiveNextAppointmentDate ? ` • นัดถัดไป ${effectiveNextAppointmentDate}` : ''
            })`
          : 'ท่ากายภาพเริ่มต้น: ยังไม่ได้กำหนด (สามารถเพิ่มภายหลังในตารางนัด)',
      ]
        .filter(Boolean)
        .join('\n');

      await apiClient.appendMedicalNote(enrolled.id, structuredSummary).catch(() => {});

      setToastMsg(
        isPending
          ? customSchedules.length > 0
            ? `บันทึกประวัติและท่ากายภาพ ${prescribedUniqueCount} ท่าเรียบร้อย (รอสแกน Face ID เพิ่มเติม)`
            : 'บันทึกประวัติคนไข้เรียบร้อย (รอสแกน Face ID เพิ่มเติม)'
          : customSchedules.length > 0
            ? `ลงทะเบียนคนไข้ บันทึก Face ID 120 เฟรม และกำหนดท่ากายภาพ ${prescribedUniqueCount} ท่าสำเร็จ!`
            : 'ลงทะเบียนคนไข้และบันทึก Face ID 120 เฟรมสำเร็จ!',
      );
      setCreatedPatient(enrolled);
      setCapturing(false);
      setStep(4);
      onEnrolled(enrolled);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'ไม่สามารถบันทึกข้อมูลผู้ป่วยได้');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#0F4C81]/35 p-3 sm:p-6 backdrop-blur-sm">
      <div
        className={`relative flex max-h-[94vh] w-full flex-col overflow-hidden rounded-[24px] bg-white shadow-2xl border border-slate-200 transition-all ${
          step === 3 ? 'max-w-3xl' : 'max-w-2xl'
        }`}
      >
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
            <div className="flex items-center rounded-full border border-slate-200 bg-slate-50 p-0.5 text-[11px] font-bold">
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
        <div className="flex flex-wrap items-center justify-center gap-1.5 sm:gap-2.5 border-b border-slate-100 bg-slate-50/70 px-4 py-3 text-[11px] sm:text-xs font-bold">
          {!isExistingPatientMode && (
            <>
              <div
                className={`flex items-center gap-1.5 rounded-full px-3 py-1.5 transition ${
                  step === 1
                    ? 'bg-emerald-600 text-white shadow-sm'
                    : step > 1
                      ? 'bg-emerald-100 text-emerald-800'
                      : 'bg-white text-slate-500 border border-slate-200'
                }`}
              >
                <span className="flex h-5 w-5 items-center justify-center rounded-full bg-white/25 text-[11px] font-black">
                  {step > 1 ? '✓' : '1'}
                </span>
                <span>1. กรอกข้อมูลคนไข้</span>
              </div>
              <span className="text-slate-300">•</span>
            </>
          )}

          <div
            className={`flex items-center gap-1.5 rounded-full px-3 py-1.5 transition ${
              step === 2
                ? 'bg-emerald-600 text-white shadow-sm'
                : step > 2
                  ? 'bg-emerald-100 text-emerald-800'
                  : 'bg-white text-slate-500 border border-slate-200'
            }`}
          >
            <span className="flex h-5 w-5 items-center justify-center rounded-full bg-slate-200/80 text-[11px] font-black text-slate-700">
              {step > 2 ? '✓' : isExistingPatientMode ? '1' : '2'}
            </span>
            <span>{isExistingPatientMode ? '1. Face ID 120 เฟรม (3D)' : '2. Face ID 120 เฟรม (3D)'}</span>
          </div>

          {!isExistingPatientMode && (
            <>
              <span className="text-slate-300">•</span>
              <div
                className={`flex items-center gap-1.5 rounded-full px-3 py-1.5 transition ${
                  step === 3
                    ? 'bg-emerald-600 text-white shadow-sm'
                    : step > 3
                      ? 'bg-emerald-100 text-emerald-800'
                      : 'bg-white text-slate-500 border border-slate-200'
                }`}
              >
                <span className="flex h-5 w-5 items-center justify-center rounded-full bg-slate-200/80 text-[11px] font-black text-slate-700">
                  {step > 3 ? '✓' : '3'}
                </span>
                <span>3. กำหนดท่ากายภาพบำบัด</span>
              </div>
            </>
          )}

          <span className="text-slate-300">•</span>
          <div
            className={`flex items-center gap-1.5 rounded-full px-3 py-1.5 transition ${
              step === 4
                ? 'bg-emerald-600 text-white shadow-sm'
                : 'bg-white text-slate-500 border border-slate-200'
            }`}
          >
            <span className="flex h-5 w-5 items-center justify-center rounded-full bg-slate-200/80 text-[11px] font-black text-slate-700">
              {isExistingPatientMode ? '2' : '4'}
            </span>
            <span>{isExistingPatientMode ? '2. พร้อมใช้งานตู้' : '4. พร้อมใช้งานตู้'}</span>
          </div>
        </div>

        {/* Scrollable Content */}
        <div className="flex-1 overflow-y-auto px-6 py-5 space-y-5">
          {errorMsg && (
            <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-xs font-semibold text-rose-700">
              {errorMsg}
            </div>
          )}
          {toastMsg && (
            <div className="rounded-xl border border-emerald-300 bg-emerald-50 px-4 py-3 text-xs font-bold text-emerald-800 flex items-center gap-2 shadow-xs">
              <CheckCircle2 className="h-4 w-4 text-emerald-600 shrink-0" />
              <span>{toastMsg}</span>
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
                      type="tel"
                      inputMode="numeric"
                      pattern="[0-9]*"
                      maxLength={10}
                      value={phone}
                      onChange={(e) => setPhone(e.target.value.replace(/\D/g, '').slice(0, 10))}
                      placeholder="08XXXXXXXX"
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
                      <option value="ข้อไหล่และแขน Stroke ADL (Shoulder & Arm)">
                        ข้อไหล่และแขน Stroke ADL (Shoulder &amp; Arm — แนะนำหลัก)
                      </option>
                      <option value="คอ บ่า และแขน สธ. ผู้สูงอายุ (MOPH Upper Body)">
                        คอ บ่า และแขน สธ. ผู้สูงอายุ (MOPH Upper Body)
                      </option>
                      <option value="ข้อไหล่ (Shoulder)">ข้อไหล่ (Shoulder)</option>
                      <option value="ข้อศอกและแขน (Elbow & Arm)">ข้อศอกและแขน (Elbow &amp; Arm)</option>
                      <option value="เข่า (Knee)">เข่า (Knee)</option>
                      <option value="หลังส่วนล่าง (Lower Back)">หลังส่วนล่าง (Lower Back)</option>
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
                      placeholder="เช่น Ischemic Stroke แขนขวาอ่อนแรง, ไหล่ติดผู้สูงอายุ"
                      className="w-full rounded-xl border border-slate-200 bg-slate-50/70 px-3 py-2 text-sm text-slate-800 placeholder:text-slate-400 focus:border-teal-500 focus:bg-white focus:outline-none"
                    />
                  </div>
                </div>

                {/* Stroke Classification, Affected Side, Mahidol 0-6 Motor Assessment Scale & Safe ROM */}
                <div className="rounded-2xl border border-emerald-200 bg-emerald-50/50 p-3.5 space-y-3">
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    <div>
                      <label className="block text-[11px] font-extrabold text-emerald-950 mb-1">
                        ประเภทกลุ่มผู้ป่วย (Stroke / Elderly Classification)
                      </label>
                      <select
                        value={strokeType}
                        onChange={(e) => setStrokeType(e.target.value as StrokeType)}
                        className="w-full rounded-xl border border-emerald-200 bg-white px-3 py-2 text-xs font-bold text-slate-800"
                      >
                        <option value="ISCHEMIC">Stroke เส้นเลือดตีบ/ตัน (Ischemic)</option>
                        <option value="HEMORRHAGIC">Stroke เส้นเลือดแตก (Hemorrhagic)</option>
                        <option value="ELDERLY_GENERAL">ผู้สูงอายุทั่วไป (MOPH / Active Aging)</option>
                        <option value="NEURO">ระบบประสาท (Neuro Rehab)</option>
                        <option value="ORTHOPEDIC">กระดูกและข้อ (Orthopedic)</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-[11px] font-extrabold text-emerald-950 mb-1">
                        จุดอ่อนแรงที่ต้องเน้น (Focus Weak Side)
                      </label>
                      <div className="grid grid-cols-4 gap-1">
                        {(
                          [
                            ['LEFT', 'ซ้าย'],
                            ['RIGHT', 'ขวา'],
                            ['BOTH', '2 ข้าง'],
                            ['NONE', 'ปกติ'],
                          ] as const
                        ).map(([val, lbl]) => (
                          <button
                            key={val}
                            type="button"
                            onClick={() => setAffectedSide(val)}
                            className={`rounded-lg py-2 text-[11px] font-extrabold border transition cursor-pointer ${
                              affectedSide === val
                                ? 'border-emerald-600 bg-emerald-600 text-white shadow-xs'
                                : 'border-emerald-200 bg-white text-slate-700 hover:bg-emerald-100'
                            }`}
                          >
                            {lbl}
                          </button>
                        ))}
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <label className="block text-[11px] font-extrabold text-emerald-950 mb-1">
                          Safe Max ROM (°)
                        </label>
                        <input
                          type="number"
                          min={30}
                          max={180}
                          value={maxSafeRomDeg}
                          onChange={(e) => setMaxSafeRomDeg(Math.max(30, Math.min(180, Number(e.target.value) || 30)))}
                          className="w-full rounded-xl border border-emerald-200 bg-white px-2.5 py-1.5 text-center font-mono text-sm font-black text-emerald-800"
                        />
                      </div>
                      <div>
                        <label className="block text-[11px] font-extrabold text-emerald-950 mb-1">
                          ADL ใส่เสื้อ (0–5)
                        </label>
                        <select
                          value={adlDressingScore}
                          onChange={(e) => setAdlDressingScore(Number(e.target.value))}
                          className="w-full rounded-xl border border-emerald-200 bg-white px-2 py-2 text-xs font-bold text-slate-800"
                        >
                          <option value={0}>0: ช่วยตัวเองไม่ได้</option>
                          <option value={1}>1: ช่วยมาก (พึ่งพาสูง)</option>
                          <option value={2}>2: สอดแขนได้บ้าง</option>
                          <option value={3}>3: ทำได้เมื่อมีผู้ช่วย</option>
                          <option value={4}>4: ใส่เองได้ช้าๆ</option>
                          <option value={5}>5: ใส่เสื้อผ้าเองได้คล่อง</option>
                        </select>
                      </div>
                    </div>
                  </div>

                  <div>
                    <div className="flex items-center justify-between mb-1.5">
                      <label className="text-[11px] font-extrabold text-emerald-950">
                        Motor Assessment Scale (กำลังกล้ามเนื้อ แนวทางมหิดล 0–6 คะแนน)
                      </label>
                      <span className="rounded-full bg-emerald-700 px-2.5 py-0.5 text-[10px] font-black text-white">
                        Grade {motorAssessmentScore} / 6
                      </span>
                    </div>
                    <div className="grid grid-cols-7 gap-1.5">
                      {MAHIDOL_MOTOR_SCALE.map((lvl) => {
                        const active = motorAssessmentScore === lvl.score;
                        return (
                          <button
                            key={lvl.score}
                            type="button"
                            onClick={() => {
                              setMotorAssessmentScore(lvl.score);
                              setMaxSafeRomDeg(lvl.recommendedMaxRomDeg);
                            }}
                            className={`rounded-xl border py-2 text-center transition cursor-pointer ${
                              active
                                ? 'border-emerald-600 bg-emerald-600 text-white shadow-sm'
                                : 'border-emerald-200 bg-white text-slate-700 hover:bg-emerald-100'
                            }`}
                          >
                            <div className="text-sm font-black">{lvl.score}</div>
                            <div className="text-[9px] font-bold opacity-85">{lvl.recommendedMaxRomDeg}°</div>
                          </button>
                        );
                      })}
                    </div>
                    <p className="mt-1.5 rounded-xl border border-emerald-200 bg-white px-3 py-1.5 text-[11px] font-semibold text-emerald-950">
                      <span className="font-black">
                        {MAHIDOL_MOTOR_SCALE[motorAssessmentScore]?.titleTh}:
                      </span>{' '}
                      {MAHIDOL_MOTOR_SCALE[motorAssessmentScore]?.descTh}
                    </p>
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
                      type="text"
                      inputMode="numeric"
                      pattern="[0-9]*"
                      value={pinCode}
                      maxLength={4}
                      onChange={(e) => setPinCode(e.target.value.replace(/\D/g, '').slice(0, 4))}
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
                    type="tel"
                    inputMode="numeric"
                    pattern="[0-9]*"
                    maxLength={10}
                    value={emergencyPhone}
                    onChange={(e) => setEmergencyPhone(e.target.value.replace(/\D/g, '').slice(0, 10))}
                    placeholder="08XXXXXXXX"
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
                    คนไข้ยินยอมให้จัดเก็บแม่แบบใบหน้า 120 เฟรม (เวกเตอร์ 3D Multi-Angle 512 มิติ) และเข้าใจว่าข้อมูลเวชระเบียนที่เปิดประวัติแล้วเป็นเอกสารทางการแพทย์ถาวรตามกฎหมายสถานพยาบาล ไม่สามารถลบบัญชีได้ (PDPA Biometric Consent)
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
                <span className="font-extrabold text-emerald-700">
                  {biometric120Result
                    ? `บันทึกครบ ${biometric120Result.framesCaptured} / 120 เฟรม (${Math.round(biometric120Result.qualityScore * 100)}%)`
                    : 'Face ID 120 เฟรม (3D Multi-Angle)'}
                </span>
              </div>

              {capturing ? (
                <FaceID120Scanner
                  active={open && step === 2 && capturing}
                  patientName={fullName}
                  hn={hn}
                  onComplete={(result) => {
                    setBiometric120Result(result);
                    setCapturedSnapshotUrl(result.avatarDataUrl);
                    setAllFiveAnglesReady(true);
                  }}
                  onReadyStateChange={(ready, triggerFinalize) => {
                    setAllFiveAnglesReady(ready);
                    triggerFinalizeRef.current = triggerFinalize;
                  }}
                  onCancel={() => setCapturing(false)}
                />
              ) : (
                <div className="flex flex-col items-center justify-center rounded-3xl border-2 border-emerald-200 bg-gradient-to-br from-emerald-50 via-sky-50 to-white p-8 text-center text-[#0F4C81] shadow-sm">
                  <Camera className="mb-3 h-10 w-10 text-emerald-600" />
                  <p className="text-sm font-extrabold">
                    {biometric120Result
                      ? 'สแกนใบหน้า 120 เฟรม (3D Multi-Angle) สำเร็จแล้ว'
                      : 'กล้องสแกนใบหน้า 120 เฟรมหยุดพักชั่วคราว'}
                  </p>
                  <p className="mt-1 text-xs text-slate-600">
                    หมุนศีรษะช้าๆ เพื่อสร้าง Master Embedding + Angle Embeddings (Center, Left, Right, Up, Down)
                  </p>
                  <button
                    type="button"
                    onClick={() => {
                      setErrorMsg(null);
                      setCapturing(true);
                    }}
                    className="mt-4 rounded-xl bg-emerald-600 px-5 py-2.5 text-xs font-extrabold text-white shadow hover:bg-emerald-500"
                  >
                    {biometric120Result ? 'สแกน 120 เฟรมใหม่อีกครั้ง' : 'เปิดกล้องสแกนใบหน้า 120 เฟรม (3D)'}
                  </button>
                </div>
              )}

              {!isExistingPatientMode && (
                <div className="flex flex-wrap items-center justify-end gap-2 pt-1">
                  <button
                    type="button"
                    disabled={saving}
                    onClick={handleSkipFaceToStep3}
                    className="rounded-xl border border-amber-300 bg-amber-50 px-3.5 py-2 text-xs font-bold text-amber-800 hover:bg-amber-100 disabled:opacity-50"
                  >
                    ข้ามสแกนใบหน้าไปกำหนดท่ากายภาพ (สแกนทีหลัง) →
                  </button>
                </div>
              )}
            </div>
          )}

          {step === 3 && !isExistingPatientMode && (
            <div className="space-y-4">
              {/* Patient & Biometric Status Banner */}
              <div className="flex flex-wrap items-center justify-between gap-2 rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-xs">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-bold text-slate-500">คนไข้:</span>
                  <span className="font-extrabold text-slate-900">{fullName}</span>
                  <span className="rounded-lg bg-sky-100 px-2 py-0.5 font-mono font-bold text-sky-800">{hn}</span>
                  <span className="rounded-lg bg-teal-50 border border-teal-200 px-2 py-0.5 font-bold text-teal-800">
                    {targetJoint}
                  </span>
                </div>
                <span
                  className={`inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-[11px] font-extrabold ${
                    !pendingSkipFaceScan && biometric120Result
                      ? 'bg-emerald-100 text-emerald-800'
                      : 'bg-amber-100 text-amber-800'
                  }`}
                >
                  {!pendingSkipFaceScan && biometric120Result
                    ? `Face ID 120 เฟรม พร้อมแล้ว ✓ (${Math.round(biometric120Result.qualityScore * 100)}%)`
                    : 'รอสแกน Face ID ภายหลัง'}
                </span>
              </div>

              {/* Section A: Multi-Select Rehab Exercises */}
              <div className="rounded-2xl border border-slate-200 bg-white p-4 space-y-3 text-xs">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <span className="flex h-6 w-6 items-center justify-center rounded-full bg-emerald-600 text-[11px] font-black text-white">
                      1
                    </span>
                    <span className="text-sm font-black text-slate-900 flex items-center gap-1.5">
                      <Dumbbell className="h-4 w-4 text-emerald-600" />
                      เลือกท่ากายภาพบำบัดสำหรับคนไข้รายนี้ (เลือกได้หลายท่า)
                    </span>
                    <span className="rounded-full bg-emerald-50 border border-emerald-200 px-2.5 py-0.5 text-[11px] font-extrabold text-emerald-700">
                      เลือกแล้ว {selectedExerciseIds.length} ท่า
                    </span>
                  </div>

                  <div className="flex flex-wrap items-center gap-1.5">
                    <button
                      type="button"
                      onClick={() => {
                        setPrescriptionTouched(true);
                        setSelectedExerciseIds(getRecommendedExerciseIdsForJoint(targetJoint, catalogExercises));
                      }}
                      className="rounded-lg border border-teal-200 bg-teal-50 px-2.5 py-1 text-[11px] font-bold text-teal-800 hover:bg-teal-100 cursor-pointer"
                    >
                      แนะนำตามข้อต่อ
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setPrescriptionTouched(true);
                        setSelectedExerciseIds((prev) =>
                          Array.from(new Set([...prev, ...filteredExercises.map((ex) => ex.id)])),
                        );
                      }}
                      className="rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1 text-[11px] font-bold text-slate-700 hover:bg-emerald-50 hover:text-emerald-700 cursor-pointer"
                    >
                      เลือกทั้งหมดในหมวด
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setPrescriptionTouched(true);
                        setSelectedExerciseIds([]);
                      }}
                      className="rounded-lg border border-slate-200 bg-white px-2.5 py-1 text-[11px] font-bold text-slate-500 hover:bg-rose-50 hover:text-rose-700 cursor-pointer"
                    >
                      ล้างที่เลือก
                    </button>
                  </div>
                </div>

                {selectedExerciseIds.length > 0 && (
                  <div className="rounded-xl border border-emerald-200 bg-emerald-50/70 p-2.5 space-y-1.5">
                    <div className="flex items-center justify-between text-[11px] font-bold text-emerald-800">
                      <span>ท่าที่เลือกไว้ทั้งหมด ({selectedExerciseIds.length} ท่า — ทุกหมวด)</span>
                      <span className="text-[10px] font-medium text-emerald-700">กดที่แท็กเพื่อเอาออกได้</span>
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {selectedExerciseIds.map((exId) => {
                        const found = catalogExercises.find((c) => c.id === exId);
                        const label = found ? found.name[lang] || found.name.th : exId;
                        return (
                          <button
                            key={exId}
                            type="button"
                            onClick={() => toggleExerciseSelection(exId)}
                            className="inline-flex items-center gap-1 rounded-full bg-white border border-emerald-300 px-2.5 py-0.5 text-[11px] font-bold text-emerald-900 hover:bg-rose-50 hover:border-rose-300 hover:text-rose-800 cursor-pointer"
                          >
                            <span>{label}</span>
                            <span aria-hidden="true">×</span>
                          </button>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Category Filter Tabs */}
                <div className="flex flex-wrap gap-1.5">
                  {ENROLLMENT_EXERCISE_FILTER_TABS.map((tab) => (
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

                {/* Interactive Multi-Select Exercise Grid */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 max-h-56 overflow-y-auto pr-1">
                  {filteredExercises.map((ex) => {
                    const checked = selectedExerciseIds.includes(ex.id);
                    return (
                      <button
                        key={ex.id}
                        type="button"
                        onClick={() => toggleExerciseSelection(ex.id)}
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
                    กรุณาเลือกท่ากายภาพอย่างน้อย 1 ท่า หรือกด “ข้ามการกำหนดท่า (บันทึกเฉพาะประวัติ)” ด้านล่าง
                  </div>
                )}

                {/* Sets, Reps, Modpack, Notes */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 pt-2 border-t border-slate-100">
                  <div>
                    <label className="block font-bold text-slate-600 mb-1">จำนวนเซต (Sets)</label>
                    <input
                      type="number"
                      inputMode="numeric"
                      min={1}
                      max={10}
                      value={prescriptionSets}
                      onKeyDown={(e) => {
                        if (['e', 'E', '+', '-', '.'].includes(e.key)) e.preventDefault();
                      }}
                      onChange={(e) => {
                        const digits = e.target.value.replace(/\D/g, '').slice(0, 2);
                        setPrescriptionSets(digits ? Math.min(10, Math.max(1, Number(digits))) : 1);
                      }}
                      className="w-full rounded-xl border border-slate-200 px-2.5 py-2 text-center font-bold text-slate-800"
                    />
                  </div>
                  <div>
                    <label className="block font-bold text-slate-600 mb-1">จำนวนครั้ง (Reps)</label>
                    <input
                      type="number"
                      inputMode="numeric"
                      min={1}
                      max={50}
                      value={prescriptionReps}
                      onKeyDown={(e) => {
                        if (['e', 'E', '+', '-', '.'].includes(e.key)) e.preventDefault();
                      }}
                      onChange={(e) => {
                        const digits = e.target.value.replace(/\D/g, '').slice(0, 2);
                        setPrescriptionReps(digits ? Math.min(50, Math.max(1, Number(digits))) : 1);
                      }}
                      className="w-full rounded-xl border border-slate-200 px-2.5 py-2 text-center font-bold text-slate-800"
                    />
                  </div>
                  <div className="col-span-2">
                    <label className="block font-bold text-slate-600 mb-1">มินิเกมฝึกสมองควบคู่ (เสริม)</label>
                    <select
                      value={selectedModpackId}
                      onChange={(e) => setSelectedModpackId(e.target.value)}
                      className="w-full rounded-xl border border-slate-200 px-3 py-2 font-semibold text-slate-800"
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
                  value={scheduleNotes}
                  onChange={(e) => setScheduleNotes(e.target.value)}
                  placeholder="หมายเหตุการฝึกสำหรับคนไข้รายนี้ (เช่น ยกไม่เกิน 140°, พักระหว่างเซต 30 วิ)"
                  className="w-full rounded-xl border border-slate-200 bg-slate-50/70 px-3 py-2 text-xs text-slate-800 placeholder:text-slate-400 focus:border-teal-500 focus:bg-white focus:outline-none"
                />
              </div>

              {/* Section B: Scheduled Rehab Dates */}
              <div className="rounded-2xl border border-slate-200 bg-white p-4 space-y-3 text-xs">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <span className="flex h-6 w-6 items-center justify-center rounded-full bg-blue-600 text-[11px] font-black text-white">
                      2
                    </span>
                    <span className="text-sm font-black text-slate-900 flex items-center gap-1.5">
                      <CalendarDays className="h-4 w-4 text-blue-600" />
                      กำหนดวันนัดฝึกกายภาพบำบัด
                    </span>
                  </div>
                  <span className="rounded-full bg-blue-50 border border-blue-200 px-2.5 py-0.5 text-[11px] font-extrabold text-blue-700">
                    เลือกไว้ {scheduledDates.length} วัน ({selectedExerciseIds.length * scheduledDates.length} รายการ)
                  </span>
                </div>

                <div className="flex flex-wrap items-center gap-1.5">
                  <button
                    type="button"
                    onClick={() => addScheduledDates([todayIso])}
                    className={`rounded-lg border px-2.5 py-1 text-[11px] font-bold transition cursor-pointer ${
                      scheduledDates.includes(todayIso)
                        ? 'border-emerald-300 bg-emerald-50 text-emerald-800'
                        : 'border-slate-200 bg-slate-50 text-slate-700 hover:bg-emerald-50'
                    }`}
                  >
                    + วันนี้ ({todayIso})
                  </button>
                  <button
                    type="button"
                    onClick={() => addScheduledDates([addDaysToIso(todayIso, 1)])}
                    className="rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1 text-[11px] font-bold text-slate-700 hover:bg-sky-50 cursor-pointer"
                  >
                    + พรุ่งนี้ (+1 วัน)
                  </button>
                  <button
                    type="button"
                    onClick={() =>
                      addScheduledDates([todayIso, addDaysToIso(todayIso, 1), addDaysToIso(todayIso, 2)])
                    }
                    className="rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1 text-[11px] font-bold text-slate-700 hover:bg-sky-50 cursor-pointer"
                  >
                    + ต่อเนื่อง 3 วัน
                  </button>
                  <button
                    type="button"
                    onClick={() => addScheduledDates(buildTriWeeklyDates(todayIso, 6))}
                    className="rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1 text-[11px] font-bold text-slate-700 hover:bg-sky-50 cursor-pointer"
                  >
                    + จันทร์-พุธ-ศุกร์ (2 สัปดาห์)
                  </button>

                  <div className="ml-auto flex items-center gap-1.5">
                    <input
                      type="date"
                      value={customDateInput}
                      onChange={(e) => setCustomDateInput(e.target.value)}
                      className="rounded-lg border border-slate-200 bg-slate-50 px-2 py-1 text-[11px] font-bold text-slate-800"
                    />
                    <button
                      type="button"
                      onClick={() => addScheduledDates([customDateInput])}
                      className="inline-flex items-center gap-1 rounded-lg bg-[#0F4C81] px-2.5 py-1 text-[11px] font-bold text-white hover:bg-[#0c3f6b] cursor-pointer"
                    >
                      <Plus className="h-3 w-3" />
                      เพิ่มวันนัด
                    </button>
                  </div>
                </div>

                {scheduledDates.length === 0 ? (
                  <div className="rounded-xl bg-amber-50 border border-amber-200 px-3 py-2 text-[11px] font-bold text-amber-800">
                    กรุณาเลือกวันนัดอย่างน้อย 1 วัน (แนะนำ “วันนี้” เพื่อให้คนไข้เข้าฝึกที่ตู้ Kiosk ได้ทันที)
                  </div>
                ) : (
                  <div className="flex flex-wrap gap-1.5">
                    {scheduledDates.map((d) => (
                      <span
                        key={d}
                        className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-[11px] font-bold ${
                          d === todayIso
                            ? 'border-emerald-300 bg-emerald-50 text-emerald-800'
                            : 'border-sky-200 bg-sky-50 text-sky-800'
                        }`}
                      >
                        <span className="font-mono">{d}</span>
                        {d === todayIso && <span className="text-[10px]">(วันนี้ • เข้าตู้ได้ทันที)</span>}
                        <button
                          type="button"
                          onClick={() => toggleScheduledDate(d)}
                          className="ml-0.5 rounded-full p-0.5 hover:bg-black/10 cursor-pointer"
                          aria-label={`ลบวันที่ ${d}`}
                        >
                          <X className="h-3 w-3" />
                        </button>
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {step === 4 && createdPatient && (
            <div className="py-5 text-center space-y-4">
              <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-emerald-100 text-emerald-600">
                <CheckCircle2 className="h-9 w-9" />
              </div>
              <div>
                <h3 className="text-lg font-extrabold text-slate-900">
                  {createdPatient.faceBiometricStatus === 'PENDING'
                    ? 'บันทึกประวัติคนไข้เรียบร้อย (รอสแกนใบหน้า 120 เฟรมเพิ่มเติม)'
                    : 'เปิดประวัติและยืนยัน Face ID 120 เฟรม (3D) พร้อมใช้งานตู้ Kiosk แล้ว'}
                </h3>
                <p className="mt-1 text-xs text-slate-500">
                  บันทึกข้อมูลลงฐานข้อมูล Supabase Cloud และ Audit Trail เรียบร้อย
                </p>
              </div>
              <div className="mx-auto max-w-lg rounded-2xl border border-slate-200 bg-slate-50 p-4 text-left text-xs space-y-2.5">
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
                    {createdPatient.faceBiometricStatus === 'VERIFIED'
                      ? `VERIFIED (${createdPatient.faceFramesCount ?? 120} เฟรม • 512D 3D)`
                      : 'PENDING (รอสแกนหน้า 120 เฟรม)'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">วันนัดหมายถัดไป:</span>
                  <span className="font-mono font-bold text-slate-800">
                    {createdPatient.nextAppointmentDate || '-'}
                  </span>
                </div>

                {(() => {
                  const previewAngles =
                    createdPatient.angleImageUrls ?? biometric120Result?.angleSnapshots;
                  if (!previewAngles) return null;
                  return (
                    <div className="border-t border-slate-200 pt-2.5 mt-2 space-y-1.5">
                      <div className="font-extrabold text-emerald-800">
                        ภาพใบหน้า Face ID 5 มุมมองที่จัดเก็บ
                      </div>
                      <div className="grid grid-cols-5 gap-1.5">
                        {(
                          [
                            ['center', 'หน้าตรง'],
                            ['left', 'ซ้าย'],
                            ['right', 'ขวา'],
                            ['up', 'เงย'],
                            ['down', 'ก้ม'],
                          ] as const
                        ).map(([angleKey, label]) => {
                          const url = previewAngles[angleKey];
                          return (
                            <div
                              key={angleKey}
                              className="rounded-lg border border-slate-200 bg-white overflow-hidden flex flex-col items-center"
                            >
                              {url ? (
                                <img
                                  src={url}
                                  alt={label}
                                  className="h-12 w-full object-cover"
                                />
                              ) : (
                                <div className="h-12 w-full bg-slate-100 flex items-center justify-center text-[10px] text-slate-400">
                                  -
                                </div>
                              )}
                              <span className="py-0.5 text-[10px] font-bold text-slate-600">
                                {label}
                              </span>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  );
                })()}

                {(createdPatient.rehabSchedules ?? []).length > 0 && (
                  <div className="border-t border-slate-200 pt-2.5 mt-2 space-y-1.5">
                    <div className="font-extrabold text-emerald-800 flex items-center gap-1.5">
                      <Dumbbell className="h-3.5 w-3.5" />
                      ท่ากายภาพบำบัดที่กำหนดไว้ ({createdPatient.rehabSchedules!.length} รายการ)
                    </div>
                    <div className="max-h-36 overflow-y-auto space-y-1 pr-1">
                      {createdPatient.rehabSchedules!.map((sched) => {
                        const exMeta = catalogExercises.find((e) => e.id === sched.exerciseId);
                        return (
                          <div
                            key={sched.id}
                            className="flex items-center justify-between gap-2 rounded-lg bg-white border border-slate-200 px-2.5 py-1.5 text-[11px]"
                          >
                            <span className="font-bold text-slate-800 truncate">
                              {exMeta ? exMeta.name[lang] || exMeta.name.th : sched.exerciseId || 'Rehab'}
                            </span>
                            <span className="shrink-0 font-mono text-slate-600">
                              {sched.scheduledDate} • {sched.sets}×{sched.reps}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Sticky Modal Footer */}
        <div className="flex flex-wrap items-center justify-end gap-2.5 border-t border-slate-100 bg-slate-50/80 px-6 py-4">
          {step === 1 && (
            <>
              <button
                type="button"
                onClick={onClose}
                className="rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-xs font-bold text-slate-700 hover:bg-slate-100"
              >
                ยกเลิก
              </button>
              <button
                type="button"
                onClick={handleSkipFaceToStep3}
                className="rounded-xl border border-amber-300 bg-amber-50 px-4 py-2.5 text-xs font-bold text-amber-800 hover:bg-amber-100"
              >
                ข้ามสแกนหน้า → กำหนดท่ากายภาพ
              </button>
              <button
                type="submit"
                form="staff-enroll-step1"
                className="rounded-xl bg-gradient-to-r from-emerald-600 to-blue-600 px-5 py-2.5 text-xs font-extrabold text-white shadow-md hover:opacity-95"
              >
                → ขั้นตอนต่อไป: สแกน Face ID 120 เฟรม →
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
              {isExistingPatientMode ? (
                <button
                  type="button"
                  disabled={saving || (!biometric120Result && !allFiveAnglesReady)}
                  onClick={() => void handleFinalizeEnrollment('verified')}
                  className={`rounded-xl px-5 py-2.5 text-xs font-extrabold transition ${
                    biometric120Result || allFiveAnglesReady
                      ? 'bg-emerald-500 hover:bg-emerald-600 text-white shadow-lg shadow-emerald-500/30 cursor-pointer'
                      : 'bg-slate-200 text-slate-400 cursor-not-allowed'
                  } disabled:opacity-60`}
                >
                  {saving ? 'กำลังบันทึกเวชระเบียน...' : 'บันทึกและเปิดใช้งานตู้ Kiosk →'}
                </button>
              ) : (
                <button
                  type="button"
                  disabled={saving || (!biometric120Result && !allFiveAnglesReady)}
                  onClick={() => void handleProceedFromStep2ToStep3()}
                  className={`rounded-xl px-5 py-2.5 text-xs font-extrabold transition ${
                    biometric120Result || allFiveAnglesReady
                      ? 'bg-gradient-to-r from-emerald-600 to-blue-600 hover:opacity-95 text-white shadow-lg shadow-emerald-500/25 cursor-pointer'
                      : 'bg-slate-200 text-slate-400 cursor-not-allowed'
                  } disabled:opacity-60`}
                >
                  {saving ? 'กำลังเตรียมข้อมูลใบหน้า...' : '→ ขั้นตอนต่อไป: กำหนดท่ากายภาพบำบัด →'}
                </button>
              )}
            </>
          )}

          {step === 3 && !isExistingPatientMode && (
            <>
              <button
                type="button"
                disabled={saving}
                onClick={() => {
                  setErrorMsg(null);
                  setStep(2);
                  if (!biometric120Result) {
                    setCapturing(true);
                  }
                }}
                className="rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-xs font-bold text-slate-700 hover:bg-slate-100 disabled:opacity-50"
              >
                ← ย้อนกลับสแกนใบหน้า
              </button>
              <button
                type="button"
                disabled={saving}
                onClick={() =>
                  void handleFinalizeEnrollment(
                    pendingSkipFaceScan ? 'pending_skip' : 'verified',
                    undefined,
                    undefined,
                    [],
                  )
                }
                className="rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-xs font-bold text-slate-600 hover:bg-slate-100 disabled:opacity-50"
              >
                ข้ามการกำหนดท่า (บันทึกเฉพาะประวัติ)
              </button>
              <button
                type="button"
                disabled={saving || selectedExerciseIds.length === 0 || scheduledDates.length === 0}
                onClick={() => {
                  const built = buildPrescriptionSchedules();
                  void handleFinalizeEnrollment(
                    pendingSkipFaceScan ? 'pending_skip' : 'verified',
                    undefined,
                    undefined,
                    built,
                  );
                }}
                className="rounded-xl bg-emerald-600 px-5 py-2.5 text-xs font-extrabold text-white shadow-lg shadow-emerald-600/25 hover:bg-emerald-500 disabled:bg-slate-200 disabled:text-slate-400 disabled:shadow-none disabled:cursor-not-allowed cursor-pointer"
              >
                {saving
                  ? 'กำลังบันทึกเวชระเบียนและท่ากายภาพ...'
                  : `บันทึกคนไข้พร้อมท่ากายภาพ (${selectedExerciseIds.length} ท่า) และเปิดใช้งานตู้ →`}
              </button>
            </>
          )}

          {step === 4 && (
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
