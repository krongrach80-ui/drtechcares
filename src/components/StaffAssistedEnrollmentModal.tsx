import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  ApiError,
  formatLocalIsoDate,
  MAHIDOL_MOTOR_SCALE,
  resolveNextAppointmentDate,
  type AffectedSide,
  type DrTechCaresApiClient,
  type DuplicateFaceCheckResponse,
  type ManagedExerciseRecord,
  type ModpackRecord,
  type PatientRecord,
  type RehabScheduleItem,
  type StrokeType,
  type UserAccount,
} from '@drtechcares/api-client';
import { useClinicalI18n, type ClinicalLang } from '@drtechcares/clinical-i18n';
import {
  CLINICAL_EXERCISES,
  formatEffectiveExerciseCode,
  isDeprecatedExerciseRecord,
  mergeDuplicateSideExercises,
  normalizeSideValue,
  parseExerciseCodeAndSide,
  resolveDefaultSideForExercise,
  resolveSideAdjustedExerciseCopy,
  sortExercisesStrokeUpperFirst,
  type ExerciseSideCode,
} from '@drtechcares/pose-engine';
import {
  BILATERAL_INHERENT_EXERCISE_CODES,
  ExerciseSidePills,
  GlobalQuickSideSelector,
  getSideBadgeMeta,
} from './ExerciseSideSelector';
import { upsertPatientStaffUserInSupabase } from '../utils/unifyPatientDirectory';
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BadgeCheck,
  CalendarDays,
  Camera,
  Check,
  CheckCircle2,
  Dumbbell,
  HeartPulse,
  KeyRound,
  Plus,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Stethoscope,
  UserPlus,
  X,
} from 'lucide-react';
import FaceID120Scanner, { type FaceID120Result } from './FaceID120Scanner';
import { syncPatientBiometricEverywhere } from '../lib/biometricEngine';
import { playGameSfx } from '../lib/gameUiSfx';
import { normalizePatientRecord } from '../lib/patientHydration';
import { GameDatePickerField } from './GameDatePickerField';
import { calculateExactAgeFromIso } from '../utils/gameDatePickerHelpers';
import {
  composeThaiFullName,
  computeEnrollmentReadiness,
  PATIENT_GENDER_OPTIONS,
  QUICK_CLINICAL_DIAGNOSIS_PRESETS,
  QUICK_EMERGENCY_RELATION_PRESETS,
  resolveGenderForPrefix,
  resolvePrefixForGender,
  splitThaiFullName,
  THAI_NAME_PREFIX_OPTIONS,
  type PatientGenderTh,
  type ThaiNamePrefix,
} from '../utils/patientNameHelpers';

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

type ExerciseFilterCategory = 'all' | 'upper_limb' | 'neck_trunk';

const ENROLLMENT_EXERCISE_FILTER_TABS: ReadonlyArray<{
  id: ExerciseFilterCategory;
  label: Record<ClinicalLang, string>;
}> = [
  { id: 'all', label: { th: 'ทั้งหมด', en: 'All Postures', cn: '全部动作' } },
  { id: 'upper_limb', label: { th: 'แขน ไหล่ ศอก และมือ', en: 'Upper Limb (Stroke)', cn: '上肢与肩肘手' } },
  { id: 'neck_trunk', label: { th: 'คอและบ่า (สธ.)', en: 'Neck & Shoulders (MOPH)', cn: '颈部与肩胛' } },
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
    'ข้อไหล่และแขน Stroke ADL (Shoulder & Arm)': [
      'STROKE_SH_FLEX',
      'STROKE_SH_ABD',
      'STROKE_CROSS_BODY',
      'STROKE_SCAP_PROTRACT',
      'STROKE_BOBATH_ARM_RAISE',
    ],
    'คอ บ่า และแขน สธ. ผู้สูงอายุ (MOPH Upper Body)': [
      'MOPH_NECK_TILT',
      'MOPH_NECK_TURN',
      'MOPH_SHOULDER_SHRUG_ROLL',
    ],
    'ข้อไหล่ (Shoulder)': [
      'STROKE_SH_FLEX',
      'STROKE_SH_ABD',
      'STROKE_SH_ROT',
      'STROKE_CROSS_BODY',
      'STROKE_SCAP_PROTRACT',
    ],
    'ข้อศอกและแขน (Elbow & Arm)': [
      'STROKE_ELB_FLEX_EXT',
      'STROKE_BOBATH_FORWARD_PUSH',
      'STROKE_HAND_TO_HEAD',
      'STROKE_FOREARM_TURN',
    ],
    'ข้อมือและกล้ามเนื้อมัดเล็ก (Wrist & Hand Fingers)': [
      'STROKE_WRIST_FINGER',
      'STROKE_THUMB_PINCH',
      'STROKE_FOREARM_TURN',
    ],
    'กล้ามเนื้อคอและบ่า สธ. (Cervical & Shoulder Girdle)': [
      'MOPH_NECK_TILT',
      'MOPH_NECK_TURN',
      'MOPH_SHOULDER_SHRUG_ROLL',
    ],
  };
  const preferred = preferredByJoint[jointLabel] ?? [
    'STROKE_SH_FLEX',
    'STROKE_SH_ABD',
    'STROKE_ELB_FLEX_EXT',
    'MOPH_SHOULDER_SHRUG_ROLL',
  ];
  const validSet = new Set(catalog.map((e) => parseExerciseCodeAndSide(e.code || e.id).baseCode));
  const filtered = preferred.filter((id) => validSet.has(id));
  if (filtered.length > 0) return filtered;
  return catalog.slice(0, 3).map((e) => parseExerciseCodeAndSide(e.code || e.id).baseCode);
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
  const catalogExercises = useMemo(() => {
    const cleanPassed = exercises.filter((ex) => !isDeprecatedExerciseRecord(ex));
    return mergeDuplicateSideExercises(
      sortExercisesStrokeUpperFirst(
        cleanPassed.length > 0 ? cleanPassed : FALLBACK_MANAGED_EXERCISES,
      ),
    );
  }, [exercises]);

  // Section 1: General Info (Split Thai Prefix + First Name + Last Name synced with Gender & Live Profile)
  const [hn, setHn] = useState(generatePatientHn);
  const [namePrefix, setNamePrefix] = useState<ThaiNamePrefix>('นาย');
  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [birthDate, setBirthDate] = useState('');
  const [age, setAge] = useState(45);
  const [ageSyncedPulse, setAgeSyncedPulse] = useState(false);
  const [genderSyncedPulse, setGenderSyncedPulse] = useState(false);
  const [prefixSyncedPulse, setPrefixSyncedPulse] = useState(false);
  const [romSyncedPulse, setRomSyncedPulse] = useState(false);
  const agePulseTimerRef = useRef<number | null>(null);
  const genderPulseTimerRef = useRef<number | null>(null);
  const prefixPulseTimerRef = useRef<number | null>(null);
  const romPulseTimerRef = useRef<number | null>(null);
  useEffect(() => {
    return () => {
      if (typeof window === 'undefined') return;
      if (agePulseTimerRef.current !== null) window.clearTimeout(agePulseTimerRef.current);
      if (genderPulseTimerRef.current !== null) window.clearTimeout(genderPulseTimerRef.current);
      if (prefixPulseTimerRef.current !== null) window.clearTimeout(prefixPulseTimerRef.current);
      if (romPulseTimerRef.current !== null) window.clearTimeout(romPulseTimerRef.current);
    };
  }, []);
  const [gender, setGender] = useState<PatientGenderTh>('ชาย');
  const fullName = useMemo(
    () => composeThaiFullName(namePrefix, firstName, lastName),
    [namePrefix, firstName, lastName],
  );
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
  const modalBodyRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    modalBodyRef.current?.scrollTo({ top: 0 });
  }, [step]);

  // Step 3 Initial Rehab Prescription state
  const [exerciseCategoryFilter, setExerciseCategoryFilter] = useState<ExerciseFilterCategory>('all');
  const [selectedSideByBaseCode, setSelectedSideByBaseCode] = useState<Record<string, ExerciseSideCode>>({});
  const [globalQuickSide, setGlobalQuickSide] = useState<ExerciseSideCode>('RIGHT');
  const [prescriptionTouched, setPrescriptionTouched] = useState(false);
  const [prescriptionSets, setPrescriptionSets] = useState(3);
  const [prescriptionReps, setPrescriptionReps] = useState(10);
  const [selectedModpackId, setSelectedModpackId] = useState('');
  const [scheduleNotes, setScheduleNotes] = useState('');
  const [scheduledDates, setScheduledDates] = useState<string[]>(() => [todayIso]);
  const [customDateInput, setCustomDateInput] = useState(todayIso);

  const [saving, setSaving] = useState(false);
  const [duplicateCheckPending, setDuplicateCheckPending] = useState(false);
  const [duplicateFaceWarning, setDuplicateFaceWarning] = useState<DuplicateFaceCheckResponse | null>(null);
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
    triggerFinalizeRef.current = null;
    setCreatedPatient(null);
    setCapturedSnapshotUrl(undefined);
    setDuplicateCheckPending(false);
    setDuplicateFaceWarning(null);
    setErrorMsg(null);
    setToastMsg(null);
    setExerciseCategoryFilter('all');
    setPrescriptionTouched(false);
    setSelectedSideByBaseCode({});
    setGlobalQuickSide('RIGHT');
    setPrescriptionSets(3);
    setPrescriptionReps(10);
    setSelectedModpackId('');
    setScheduleNotes('');
    setScheduledDates([todayIso]);
    setCustomDateInput(todayIso);
    if (targetPatient) {
      const parsedName = splitThaiFullName(targetPatient.fullName);
      setHn(targetPatient.hn);
      setNamePrefix(parsedName.prefix);
      setFirstName(parsedName.firstName);
      setLastName(parsedName.lastName);
      setAge(targetPatient.age);
      if (
        targetPatient.gender === 'ชาย' ||
        targetPatient.gender === 'หญิง' ||
        targetPatient.gender === 'อื่นๆ'
      ) {
        setGender(targetPatient.gender);
      }
      setPhone(targetPatient.phone);
      setStep(2);
      setCapturing(true);
    } else {
      setHn(`HN-${Math.floor(100000 + Math.random() * 899999)}`);
      setNamePrefix('นาย');
      setFirstName('');
      setLastName('');
      setBirthDate('');
      setAge(45);
      setGender('ชาย');
      setPhone('');
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
      setStep(1);
    }
  }, [open, targetPatient, todayIso]);

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

  const selectedEntries = useMemo(
    () => Object.entries(selectedSideByBaseCode) as [string, ExerciseSideCode][],
    [selectedSideByBaseCode],
  );

  useEffect(() => {
    if (!prescriptionTouched) {
      setGlobalQuickSide(normalizeSideValue(affectedSide) ?? 'RIGHT');
    }
  }, [affectedSide, prescriptionTouched]);

  const enrollmentReadiness = useMemo(
    () =>
      computeEnrollmentReadiness({
        hn,
        firstName,
        lastName,
        birthDate,
        age,
        phone,
        clinicalDiagnosis,
        emergencyName,
        pdpaAccepted,
      }),
    [hn, firstName, lastName, birthDate, age, phone, clinicalDiagnosis, emergencyName, pdpaAccepted],
  );

  if (!open) return null;

  const triggerSyncPulse = (
    setter: React.Dispatch<React.SetStateAction<boolean>>,
    timerRef: React.MutableRefObject<number | null>,
  ) => {
    setter(true);
    if (timerRef.current !== null && typeof window !== 'undefined') {
      window.clearTimeout(timerRef.current);
    }
    if (typeof window !== 'undefined') {
      timerRef.current = window.setTimeout(() => {
        timerRef.current = null;
        setter(false);
      }, 560);
    }
  };

  const handleSelectNamePrefix = (nextPrefix: ThaiNamePrefix) => {
    playGameSfx('tap');
    setNamePrefix(nextPrefix);
    const nextGender = resolveGenderForPrefix(nextPrefix, gender);
    if (nextGender !== gender) {
      setGender(nextGender);
      triggerSyncPulse(setGenderSyncedPulse, genderPulseTimerRef);
    }
  };

  const handleSelectGender = (nextGender: PatientGenderTh) => {
    playGameSfx('tap');
    setGender(nextGender);
    const nextPrefix = resolvePrefixForGender(nextGender, namePrefix);
    if (nextPrefix !== namePrefix) {
      setNamePrefix(nextPrefix);
      triggerSyncPulse(setPrefixSyncedPulse, prefixPulseTimerRef);
    }
  };

  const handleFirstNamePaste = (e: React.ClipboardEvent<HTMLInputElement>) => {
    if (lastName.trim().length > 0) return;
    const pasted = e.clipboardData.getData('text').trim();
    if (!pasted || !/\s/.test(pasted)) return;
    const parsed = splitThaiFullName(pasted);
    if (parsed.firstName && parsed.lastName) {
      e.preventDefault();
      if (parsed.prefix) {
        setNamePrefix(parsed.prefix);
        const syncedGender = resolveGenderForPrefix(parsed.prefix, gender);
        if (syncedGender !== gender) {
          setGender(syncedGender);
          triggerSyncPulse(setGenderSyncedPulse, genderPulseTimerRef);
        }
        triggerSyncPulse(setPrefixSyncedPulse, prefixPulseTimerRef);
      }
      setFirstName(parsed.firstName);
      setLastName(parsed.lastName);
      playGameSfx('correct');
    }
  };

  const handleSelectMotorScore = (score: number, recommendedMaxRomDeg: number) => {
    playGameSfx('correct');
    setMotorAssessmentScore(score);
    setMaxSafeRomDeg(recommendedMaxRomDeg);
    triggerSyncPulse(setRomSyncedPulse, romPulseTimerRef);
  };

  const handleBirthDateChange = (val: string) => {
    setBirthDate(val);
    if (!val) return;
    const exactYears = calculateExactAgeFromIso(val);
    if (exactYears !== null) {
      setAge(exactYears);
      triggerSyncPulse(setAgeSyncedPulse, agePulseTimerRef);
    }
  };

  const resetFormForNext = () => {
    setStep(1);
    setHn(generatePatientHn());
    setNamePrefix('นาย');
    setFirstName('');
    setLastName('');
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
    triggerFinalizeRef.current = null;
    setCapturedSnapshotUrl(undefined);
    setCapturing(false);
    setExerciseCategoryFilter('all');
    setPrescriptionTouched(false);
    setSelectedSideByBaseCode({});
    setGlobalQuickSide('RIGHT');
    setPrescriptionSets(3);
    setPrescriptionReps(10);
    setSelectedModpackId('');
    setScheduleNotes('');
    setScheduledDates([todayIso]);
    setCustomDateInput(todayIso);
    setCreatedPatient(null);
    setDuplicateCheckPending(false);
    setDuplicateFaceWarning(null);
    setErrorMsg(null);
    setToastMsg(null);
  };

  const validateStep1 = (): boolean => {
    if (!firstName.trim()) {
      playGameSfx('wrong');
      setErrorMsg('กรุณากรอกชื่อจริงคนไข้');
      return false;
    }
    if (!lastName.trim()) {
      playGameSfx('wrong');
      setErrorMsg('กรุณากรอกนามสกุลคนไข้');
      return false;
    }
    if (!pdpaAccepted) {
      playGameSfx('wrong');
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
    playGameSfx('unlock');
    setErrorMsg(null);
    setStep(2);
    setCapturing(true);
  };

  const verifyNoDuplicateFace = async (
    bio: FaceID120Result,
  ): Promise<DuplicateFaceCheckResponse> => {
    if (typeof apiClient.checkDuplicateFace !== 'function') {
      return { duplicate: false };
    }
    setDuplicateCheckPending(true);
    try {
      const checkRes = await apiClient.checkDuplicateFace({
        masterEmbedding: bio.masterEmbedding,
        centerEmbedding: bio.angleEmbeddings.center,
        ...(bio.geometricRatios ? { geometricRatios: bio.geometricRatios } : {}),
        ...(targetPatient ? { excludePatientId: targetPatient.id } : {}),
      });
      if (checkRes.duplicate) {
        setDuplicateFaceWarning(checkRes);
        setErrorMsg(null);
      } else {
        setDuplicateFaceWarning(null);
      }
      return checkRes;
    } catch {
      return { duplicate: false };
    } finally {
      setDuplicateCheckPending(false);
    }
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
      } catch (err) {
        setErrorMsg(
          err instanceof Error
            ? err.message
            : 'ไม่สามารถประมวลผลโปรไฟล์ใบหน้า 120 เฟรมได้ กรุณาสแกนใหม่อีกครั้ง',
        );
        return;
      } finally {
        setSaving(false);
      }
    }
    if (!bio) {
      setErrorMsg('กรุณาสแกนใบหน้าให้ครบ 5 มุมหรือครบ 120 เฟรมก่อนดำเนินการต่อ');
      return;
    }
    if (duplicateFaceWarning?.duplicate) {
      setCapturing(false);
      return;
    }
    const dupCheck = await verifyNoDuplicateFace(bio);
    if (dupCheck.duplicate) {
      setCapturing(false);
      return;
    }
    setCapturing(false);
    syncDefaultPrescriptionIfUntouched();
    setStep(3);
  };

  const toggleExerciseSelection = (ex: ManagedExerciseRecord) => {
    const baseCode = parseExerciseCodeAndSide(ex.code || ex.id).baseCode;
    setPrescriptionTouched(true);
    setSelectedSideByBaseCode((prev) => {
      if (prev[baseCode]) {
        const next = { ...prev };
        delete next[baseCode];
        return next;
      }
      const defaultSide = BILATERAL_INHERENT_EXERCISE_CODES.has(baseCode)
        ? 'BOTH'
        : globalQuickSide || resolveDefaultSideForExercise(ex, affectedSide);
      return { ...prev, [baseCode]: defaultSide };
    });
  };

  const setCardSide = (baseCode: string, side: ExerciseSideCode) => {
    setPrescriptionTouched(true);
    setSelectedSideByBaseCode((prev) => ({
      ...prev,
      [baseCode]: side,
    }));
  };

  const applyGlobalSideToSelected = (side: ExerciseSideCode) => {
    setPrescriptionTouched(true);
    setGlobalQuickSide(side);
    setSelectedSideByBaseCode((prev) => {
      const next: Record<string, ExerciseSideCode> = {};
      for (const [code] of Object.entries(prev)) {
        next[code] =
          side !== 'BOTH' && BILATERAL_INHERENT_EXERCISE_CODES.has(code)
            ? 'BOTH'
            : side;
      }
      return next;
    });
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
    if (selectedEntries.length === 0 || scheduledDates.length === 0) return [];
    const sortedDates = [...scheduledDates].sort();
    const ptId = assignedPtId || currentUser.id;
    const baseTs = Date.now();
    const out: RehabScheduleItem[] = [];
    let seq = 0;
    for (const dateIso of sortedDates) {
      for (const [baseCode, side] of selectedEntries) {
        seq += 1;
        const effectiveCode = formatEffectiveExerciseCode(baseCode, side);
        out.push({
          id: `sched-${baseTs}-${seq}`,
          patientId: 'pending',
          ptId,
          exerciseId: effectiveCode,
          exerciseCode: baseCode,
          side,
          effectiveCode,
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
    _mode: 'verified' = 'verified',
    overrideImageUrl?: string,
    overrideBiometric120?: FaceID120Result,
    customSchedules: RehabScheduleItem[] = [],
  ) => {
    setSaving(true);
    setErrorMsg(null);
    setToastMsg(null);
    try {
      let bio = overrideBiometric120 ?? biometric120Result;
      if (!bio && triggerFinalizeRef.current) {
        const finalized = await triggerFinalizeRef.current();
        if (finalized) {
          bio = finalized;
          setBiometric120Result(finalized);
        }
      }
      const finalSnapshot =
        overrideImageUrl ?? bio?.compositeSheetDataUrl ?? bio?.avatarDataUrl ?? capturedSnapshotUrl;

      if (!bio) {
        setErrorMsg('กรุณาสแกนใบหน้าให้ครบ 5 มุมหรือครบ 120 เฟรมก่อนบันทึกข้อมูล');
        setStep(2);
        setCapturing(true);
        setSaving(false);
        return;
      }

      if (duplicateFaceWarning?.duplicate) {
        setStep(2);
        setCapturing(false);
        setSaving(false);
        return;
      }

      const dupCheck = await verifyNoDuplicateFace(bio);
      if (dupCheck.duplicate) {
        setStep(2);
        setCapturing(false);
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
        const needsDirectSupabaseBackup =
          typeof apiClient.wasLastBiometricWritePersisted === 'function' &&
          !apiClient.wasLastBiometricWritePersisted() &&
          Boolean(apiClient.getSupabaseClient());
        await syncPatientBiometricEverywhere({
          apiClient,
          patient: updated,
          masterEmbedding: bio.masterEmbedding,
          angleEmbeddings: bio.angleEmbeddings,
          exemplarEmbeddings: bio.exemplarEmbeddings,
          geometricRatios: bio.geometricRatios,
          framesCaptured: bio.framesCaptured,
          qualityScore: bio.qualityScore,
          requireSupabase: needsDirectSupabaseBackup,
        }).catch(() => null);
        const normalizedUpdated = normalizePatientRecord(
          {
            ...updated,
            faceBiometricStatus: 'VERIFIED',
            faceFramesCount: updated.faceFramesCount || bio.framesCaptured || 120,
            faceImageUrl: updated.faceImageUrl || finalSnapshot,
            masterEmbedding: updated.masterEmbedding?.length ? updated.masterEmbedding : bio.masterEmbedding,
            angleEmbeddings: updated.angleEmbeddings ?? bio.angleEmbeddings,
            ...(updated.exemplarEmbeddings?.length
              ? { exemplarEmbeddings: updated.exemplarEmbeddings }
              : bio.exemplarEmbeddings?.length
                ? { exemplarEmbeddings: bio.exemplarEmbeddings }
                : {}),
            ...(updated.biometricEngine || bio.biometricEngine
              ? { biometricEngine: updated.biometricEngine ?? bio.biometricEngine }
              : {}),
            ...(updated.geometricRatios || bio.geometricRatios
              ? { geometricRatios: updated.geometricRatios ?? bio.geometricRatios }
              : {}),
          },
          {
            requireRealVectorsForVerified: false,
          },
        );
        void upsertPatientStaffUserInSupabase(
          apiClient.getSupabaseClient(),
          normalizedUpdated,
        );
        setErrorMsg(null);
        setToastMsg('บันทึกใบหน้าและข้อมูลคนไข้เรียบร้อยแล้ว!');
        setCreatedPatient(normalizedUpdated);
        setCapturing(false);
        setStep(4);
        onEnrolled(normalizedUpdated);
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
        skipFaceScan: false,
        faceBiometricStatus: 'VERIFIED',
        faceImageUrl: finalSnapshot,
        faceImageUrls: bio.faceImageUrls,
        angleImageUrls: bio.angleSnapshots,
        frontalCentroid: bio.angleEmbeddings.center,
        overallCentroid: bio.masterEmbedding,
        masterEmbedding: bio.masterEmbedding,
        angleEmbeddings: bio.angleEmbeddings,
        ...(bio.exemplarEmbeddings ? { exemplarEmbeddings: bio.exemplarEmbeddings } : {}),
        ...(bio.biometricEngine ? { biometricEngine: bio.biometricEngine } : {}),
        ...(bio.geometricRatios ? { geometricRatios: bio.geometricRatios } : {}),
        framesCaptured: bio.framesCaptured,
        qualityScore: bio.qualityScore,
      });

      const needsDirectSupabaseBackup =
        typeof apiClient.wasLastBiometricWritePersisted === 'function' &&
        !apiClient.wasLastBiometricWritePersisted() &&
        Boolean(apiClient.getSupabaseClient());
      await syncPatientBiometricEverywhere({
        apiClient,
        patient: enrolled,
        masterEmbedding: bio.masterEmbedding,
        angleEmbeddings: bio.angleEmbeddings,
        exemplarEmbeddings: bio.exemplarEmbeddings,
        geometricRatios: bio.geometricRatios,
        framesCaptured: bio.framesCaptured,
        qualityScore: bio.qualityScore,
        requireSupabase: needsDirectSupabaseBackup,
      }).catch(() => null);

      if (customSchedules.length > 0 && (enrolled.rehabSchedules ?? []).length === 0) {
        const finalizedSchedules = customSchedules.map((s) => ({ ...s, patientId: enrolled.id }));
        const patched = await apiClient
          .updatePatient(enrolled.id, {
            nextAppointmentDate: effectiveNextAppointmentDate,
            rehabSchedules: finalizedSchedules,
            faceBiometricStatus: 'VERIFIED',
          })
          .catch(() => null);
        enrolled = normalizePatientRecord(
          patched ?? {
            ...enrolled,
            nextAppointmentDate: effectiveNextAppointmentDate,
            rehabSchedules: finalizedSchedules,
            faceBiometricStatus: 'VERIFIED',
          },
          { requireRealVectorsForVerified: false },
        );
      }
      enrolled = normalizePatientRecord(
        {
          ...enrolled,
          faceBiometricStatus: 'VERIFIED',
          faceFramesCount: enrolled.faceFramesCount || bio.framesCaptured || 120,
          faceImageUrl: enrolled.faceImageUrl || finalSnapshot,
          masterEmbedding: enrolled.masterEmbedding?.length ? enrolled.masterEmbedding : bio.masterEmbedding,
          angleEmbeddings: enrolled.angleEmbeddings ?? bio.angleEmbeddings,
          ...(enrolled.exemplarEmbeddings?.length
            ? { exemplarEmbeddings: enrolled.exemplarEmbeddings }
            : bio.exemplarEmbeddings?.length
              ? { exemplarEmbeddings: bio.exemplarEmbeddings }
              : {}),
          ...(enrolled.biometricEngine || bio.biometricEngine
            ? { biometricEngine: enrolled.biometricEngine ?? bio.biometricEngine }
            : {}),
          ...(enrolled.geometricRatios || bio.geometricRatios
            ? { geometricRatios: enrolled.geometricRatios ?? bio.geometricRatios }
            : {}),
        },
        { requireRealVectorsForVerified: false },
      );

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
        `สถานะใบหน้า: VERIFIED (3D Multi-Angle 1024-D ${bio.framesCaptured ?? 120} เฟรม)`,
        customSchedules.length > 0
          ? `ท่ากายภาพเริ่มต้น: ${prescribedUniqueCount} ท่า (${customSchedules.length} รายการ${
              effectiveNextAppointmentDate ? ` • นัดถัดไป ${effectiveNextAppointmentDate}` : ''
            })`
          : 'ท่ากายภาพเริ่มต้น: ยังไม่ได้กำหนด (สามารถเพิ่มภายหลังในตารางนัด)',
      ]
        .filter(Boolean)
        .join('\n');

      await apiClient.appendMedicalNote(enrolled.id, structuredSummary).catch(() => {});
      void upsertPatientStaffUserInSupabase(apiClient.getSupabaseClient(), enrolled);

      setErrorMsg(null);
      setToastMsg(
        customSchedules.length > 0
          ? `ลงทะเบียนใบหน้าและกำหนดท่ากายภาพ ${prescribedUniqueCount} ท่าเรียบร้อย!`
          : 'ลงทะเบียนใบหน้าและบันทึกเวชระเบียนคนไข้เรียบร้อยแล้ว!',
      );
      setCreatedPatient(enrolled);
      setCapturing(false);
      setStep(4);
      onEnrolled(enrolled);
    } catch (err) {
      if (
        err instanceof ApiError &&
        (err.code === 'DUPLICATE_FACE_ENROLLMENT' || err.status === 409)
      ) {
        const rawExisting = err.details?.existingPatient as
          | { id?: string; hn?: string; fullName?: string }
          | undefined;
        const simVal =
          typeof err.details?.similarityPercent === 'number'
            ? err.details.similarityPercent
            : undefined;
        setDuplicateFaceWarning({
          duplicate: true,
          similarityPercent: simVal,
          ...(rawExisting?.id && rawExisting?.hn && rawExisting?.fullName
            ? {
                existingPatient: {
                  id: rawExisting.id,
                  hn: rawExisting.hn,
                  fullName: rawExisting.fullName,
                },
              }
            : {}),
          message: err.message,
        });
        setErrorMsg(null);
        setCapturing(false);
        setStep(2);
      } else {
        setErrorMsg(err instanceof Error ? err.message : 'ไม่สามารถบันทึกข้อมูลผู้ป่วยได้');
      }
    } finally {
      setSaving(false);
    }
  };

  const weakSideDisplayLabel =
    affectedSide === 'LEFT'
      ? 'ซ้าย'
      : affectedSide === 'RIGHT'
        ? 'ขวา'
        : affectedSide === 'BOTH'
          ? '2 ข้าง'
          : 'ปกติ';

  const phoneDigitCount = phone.replace(/\D/g, '').length;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#0F4C81]/40 p-2 sm:p-4 backdrop-blur-sm animate-backdrop-fade">
      <div
        className={`relative flex max-h-[96vh] w-full flex-col overflow-hidden rounded-[28px] bg-white shadow-[0_28px_80px_-14px_rgba(15,76,129,0.5)] border-2 border-sky-200/90 transition-all animate-popup-spring ${
          step === 1 ? 'max-w-[960px]' : step === 3 ? 'max-w-4xl' : 'max-w-2xl'
        }`}
      >
        {/* Large Readable Arcade-Clinical Header with Inline Live Synced Profile Chip */}
        <div className="relative overflow-hidden bg-gradient-to-r from-[#0F4C81] via-[#0B6699] to-[#059669] px-5 sm:px-6 py-3.5 text-white">
          <div className="relative flex items-center justify-between gap-3">
            <div className="flex items-center gap-3 min-w-0">
              <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl border border-white/30 bg-white/15 text-white shadow-inner">
                <UserPlus className="h-6 w-6" />
              </div>
              <div className="min-w-0 flex flex-wrap items-center gap-2.5">
                <h2 className="text-lg sm:text-xl font-black tracking-tight text-white shrink-0">
                  {isExistingPatientMode
                    ? `สแกนหน้าผู้ป่วย: ${targetPatient?.fullName}`
                    : 'ลงทะเบียนคนไข้ใหม่'}
                </h2>
                {!isExistingPatientMode && step === 1 && (
                  <div
                    data-testid="enroll-live-profile-card"
                    className="hidden sm:inline-flex items-center gap-2 rounded-full border border-white/25 bg-white/15 px-3.5 py-1 text-xs sm:text-sm font-bold text-white backdrop-blur-sm max-w-[460px]"
                  >
                    <span className="font-mono text-xs font-black text-emerald-200 shrink-0">
                      {hn}
                    </span>
                    <span className="text-white/50">•</span>
                    <span
                      data-testid="enroll-composed-name-preview"
                      className="truncate font-black text-white"
                    >
                      {fullName || 'รอชื่อ-นามสกุล'}
                    </span>
                    <span className="text-white/50">•</span>
                    <span className="shrink-0 text-sky-100">
                      {gender} {age > 0 ? `${age} ปี` : ''} • {weakSideDisplayLabel} Lv.{motorAssessmentScore}
                    </span>
                  </div>
                )}
              </div>
            </div>

            <div className="flex items-center gap-2 shrink-0">
              {!isExistingPatientMode && step === 1 && (
                <span
                  data-testid="enroll-readiness-bar"
                  className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs sm:text-sm font-black border ${
                    enrollmentReadiness.readyForFaceScan
                      ? 'border-emerald-300 bg-emerald-400 text-emerald-950'
                      : 'border-white/30 bg-white/15 text-white'
                  }`}
                >
                  <Sparkles className="h-3.5 w-3.5" />
                  <span>
                    {enrollmentReadiness.readyForFaceScan
                      ? `พร้อม ${enrollmentReadiness.percent}%`
                      : `${enrollmentReadiness.percent}%`}
                  </span>
                </span>
              )}
              <div className="flex items-center rounded-full border border-white/25 bg-white/15 p-0.5 text-xs font-extrabold">
                {(['th', 'en', 'cn'] as const).map((code) => (
                  <button
                    key={code}
                    type="button"
                    onClick={() => {
                      playGameSfx('tap');
                      setLang(code);
                    }}
                    className={`rounded-full px-2.5 py-1 transition cursor-pointer ${
                      lang === code
                        ? 'bg-white text-[#0F4C81] shadow-xs'
                        : 'text-white/85 hover:text-white'
                    }`}
                  >
                    {code === 'th' ? 'TH' : code === 'en' ? 'EN' : '中文'}
                  </button>
                ))}
              </div>
              <button
                type="button"
                onClick={() => {
                  playGameSfx('tap');
                  setCapturing(false);
                  onClose();
                }}
                className="flex h-9 w-9 items-center justify-center rounded-xl border border-white/25 bg-white/15 text-white hover:bg-white/25 transition cursor-pointer"
                aria-label="ปิดหน้าต่าง"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
          </div>
        </div>

        {/* Perfectly Level Single-Row Stepper Rail + Hairline Readiness Bar */}
        <div className="relative border-b border-slate-200/80 bg-slate-50/90 px-5 sm:px-6 py-2.5">
          <div className="flex items-center justify-between gap-2">
            {!isExistingPatientMode && (
              <>
                <div
                  className={`inline-flex h-9 shrink-0 items-center gap-2 rounded-full px-3.5 text-xs sm:text-sm font-extrabold leading-none transition-all ${
                    step === 1
                      ? 'bg-emerald-600 text-white shadow-xs ring-2 ring-emerald-200'
                      : step > 1
                        ? 'bg-emerald-50 text-emerald-800 border border-emerald-200'
                        : 'bg-white text-slate-500 border border-slate-200'
                  }`}
                >
                  <span
                    className={`flex h-5 w-5 items-center justify-center rounded-full text-xs font-black ${
                      step === 1
                        ? 'bg-white/25 text-white'
                        : step > 1
                          ? 'bg-emerald-600 text-white'
                          : 'bg-slate-100 text-slate-600'
                    }`}
                  >
                    {step > 1 ? <Check className="h-3 w-3" /> : '1'}
                  </span>
                  <span>ข้อมูลคนไข้</span>
                </div>
                <div
                  className={`h-0.5 flex-1 min-w-2 rounded-full ${
                    step > 1 ? 'bg-emerald-400' : 'bg-slate-200'
                  }`}
                />
              </>
            )}

            <div
              className={`inline-flex h-9 shrink-0 items-center gap-2 rounded-full px-3.5 text-xs sm:text-sm font-extrabold leading-none transition-all ${
                step === 2
                  ? 'bg-emerald-600 text-white shadow-xs ring-2 ring-emerald-200'
                  : step > 2
                    ? 'bg-emerald-50 text-emerald-800 border border-emerald-200'
                    : 'bg-white text-slate-500 border border-slate-200'
              }`}
            >
              <span
                className={`flex h-5 w-5 items-center justify-center rounded-full text-xs font-black ${
                  step === 2
                    ? 'bg-white/25 text-white'
                    : step > 2
                      ? 'bg-emerald-600 text-white'
                      : 'bg-slate-100 text-slate-600'
                }`}
              >
                {step > 2 ? <Check className="h-3 w-3" /> : isExistingPatientMode ? '1' : '2'}
              </span>
              <span>Face ID 120 เฟรม</span>
            </div>

            {!isExistingPatientMode && (
              <>
                <div
                  className={`h-0.5 flex-1 min-w-2 rounded-full ${
                    step > 2 ? 'bg-emerald-400' : 'bg-slate-200'
                  }`}
                />
                <div
                  className={`inline-flex h-9 shrink-0 items-center gap-2 rounded-full px-3.5 text-xs sm:text-sm font-extrabold leading-none transition-all ${
                    step === 3
                      ? 'bg-emerald-600 text-white shadow-xs ring-2 ring-emerald-200'
                      : step > 3
                        ? 'bg-emerald-50 text-emerald-800 border border-emerald-200'
                        : 'bg-white text-slate-500 border border-slate-200'
                  }`}
                >
                  <span
                    className={`flex h-5 w-5 items-center justify-center rounded-full text-xs font-black ${
                      step === 3
                        ? 'bg-white/25 text-white'
                        : step > 3
                          ? 'bg-emerald-600 text-white'
                          : 'bg-slate-100 text-slate-600'
                    }`}
                  >
                    {step > 3 ? <Check className="h-3 w-3" /> : '3'}
                  </span>
                  <span>ท่ากายภาพ</span>
                </div>
              </>
            )}

            <div
              className={`h-0.5 flex-1 min-w-2 rounded-full ${
                step > 3 ? 'bg-emerald-400' : 'bg-slate-200'
              }`}
            />
            <div
              className={`inline-flex h-9 shrink-0 items-center gap-2 rounded-full px-3.5 text-xs sm:text-sm font-extrabold leading-none transition-all ${
                step === 4
                  ? 'bg-emerald-600 text-white shadow-xs ring-2 ring-emerald-200'
                  : 'bg-white text-slate-500 border border-slate-200'
              }`}
            >
              <span
                className={`flex h-5 w-5 items-center justify-center rounded-full text-xs font-black ${
                  step === 4 ? 'bg-white/25 text-white' : 'bg-slate-100 text-slate-600'
                }`}
              >
                {isExistingPatientMode ? '2' : '4'}
              </span>
              <span>พร้อมใช้งาน</span>
            </div>
          </div>

          {!isExistingPatientMode && step === 1 && (
            <div className="absolute inset-x-0 bottom-0 h-1.5 bg-slate-200/70 overflow-hidden">
              <div
                className={`h-full transition-all duration-300 game-progress-stripes ${
                  enrollmentReadiness.readyForFaceScan
                    ? 'bg-gradient-to-r from-emerald-500 to-teal-500'
                    : 'bg-gradient-to-r from-[#0F4C81] to-sky-500'
                }`}
                style={{ width: `${Math.max(10, enrollmentReadiness.percent)}%` }}
              />
            </div>
          )}
        </div>

        {/* Content Body (Large Portrait Zero-Scroll on Step 1) */}
        <div
          ref={modalBodyRef}
          className="flex-1 overflow-y-auto bg-gradient-to-b from-slate-50/70 via-white to-sky-50/30 px-5 sm:px-6 py-3.5 space-y-3"
        >
          {errorMsg && (
            <div className="flex items-center gap-2 rounded-xl border-2 border-rose-200 bg-rose-50 px-4 py-2.5 text-sm font-extrabold text-rose-700 animate-toast-in animate-gentle-shake">
              <AlertTriangle className="h-4.5 w-4.5 text-rose-600 shrink-0" />
              <span>{errorMsg}</span>
            </div>
          )}
          {duplicateFaceWarning?.duplicate && (
            <div
              data-testid="duplicate-face-warning-banner"
              role="alert"
              className="rounded-2xl border-2 border-rose-300 bg-rose-50 px-4 py-3.5 text-xs text-rose-950 shadow-sm space-y-2.5 animate-toast-in"
            >
              <div className="flex items-start gap-2.5">
                <AlertTriangle className="h-5 w-5 text-rose-600 shrink-0 mt-0.5" />
                <div className="space-y-1">
                  <div className="font-extrabold text-sm text-rose-800">
                    {duplicateFaceWarning.existingPatient
                      ? 'ตรวจพบใบหน้าซ้ำกับบัญชีที่มีอยู่แล้วในระบบ'
                      : 'ตรวจพบใบหน้านี้ลงทะเบียนไว้ในระบบแล้ว'}
                  </div>
                  <p className="leading-relaxed text-rose-900">
                    {duplicateFaceWarning.existingPatient
                      ? `ใบหน้านี้ตรงกับบัญชี ${duplicateFaceWarning.existingPatient.fullName} (${duplicateFaceWarning.existingPatient.hn})${
                          typeof duplicateFaceWarning.similarityPercent === 'number'
                            ? ` ความคล้าย ${duplicateFaceWarning.similarityPercent.toFixed(1)}%`
                            : ''
                        } — ห้ามสร้างบัญชีซ้ำเพราะจะทำให้การสแกนหน้าเข้าสู่ระบบที่ตู้ Kiosk สับสน กรุณากลับไปอัปเดตใบหน้าในบัญชีเดิมแทน หรือสแกนใบหน้าผู้ป่วยคนใหม่`
                      : duplicateFaceWarning.message ??
                        'ใบหน้านี้มีบัญชีอยู่ในระบบแล้ว กรุณากลับไปสแกนหน้าเข้าสู่ระบบ หรือติดต่อเจ้าหน้าที่'}
                  </p>
                </div>
              </div>
              <div className="flex flex-wrap items-center gap-2 pl-7">
                <button
                  type="button"
                  onClick={() => {
                    setDuplicateFaceWarning(null);
                    setErrorMsg(null);
                    setBiometric120Result(null);
                    setCapturedSnapshotUrl(undefined);
                    setAllFiveAnglesReady(false);
                    triggerFinalizeRef.current = null;
                    setStep(2);
                    setCapturing(true);
                  }}
                  className="rounded-xl bg-rose-600 px-3.5 py-2 text-xs font-extrabold text-white shadow-xs hover:bg-rose-700 cursor-pointer"
                >
                  สแกนใบหน้าใหม่
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setCapturing(false);
                    onClose();
                  }}
                  className="rounded-xl border border-rose-300 bg-white px-3.5 py-2 text-xs font-bold text-rose-800 hover:bg-rose-100 cursor-pointer"
                >
                  {duplicateFaceWarning.existingPatient
                    ? 'ปิดหน้าต่างเพื่อกลับไปอัปเดตบัญชีเดิม'
                    : 'ปิดหน้าต่าง'}
                </button>
              </div>
            </div>
          )}
          {toastMsg && (
            <div className="rounded-xl border border-emerald-300 bg-emerald-50 px-4 py-2.5 text-sm font-bold text-emerald-800 flex items-center gap-2 shadow-xs animate-toast-in">
              <CheckCircle2 className="h-4.5 w-4.5 text-emerald-600 shrink-0" />
              <span>{toastMsg}</span>
            </div>
          )}

          {step === 1 && (
            <form
              id="staff-enroll-step1"
              onSubmit={handleGoStep2}
              className="space-y-3"
            >
              {/* Card 1: Patient Identity */}
              <div className="rounded-2xl border-2 border-sky-100 bg-white p-4 shadow-2xs space-y-3">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <span className="inline-flex h-7 w-7 items-center justify-center rounded-lg bg-[#0F4C81] text-sm font-black text-white">
                      1
                    </span>
                    <span className="text-base font-black text-slate-900">ข้อมูลผู้ป่วย</span>
                  </div>
                  <span
                    className={`rounded-full px-3 py-0.5 text-xs font-extrabold border transition-all ${
                      prefixSyncedPulse || genderSyncedPulse || ageSyncedPulse
                        ? 'border-emerald-400 bg-emerald-100 text-emerald-800 game-age-sync-glow'
                        : 'border-sky-200 bg-sky-50 text-[#0F4C81]'
                    }`}
                  >
                    {fullName || 'ซิงก์คำนำหน้า • เพศ • อายุอัตโนมัติ'}
                  </span>
                </div>

                {/* Prefix Bar */}
                <div>
                  <label className="block text-sm font-extrabold text-slate-700 mb-1.5">
                    คำนำหน้า <span className="text-rose-500">*</span>
                  </label>
                  <div
                    data-testid="enroll-prefix-bar"
                    className={`grid grid-cols-6 gap-2 rounded-xl transition-all ${
                      prefixSyncedPulse ? 'bg-emerald-50/80 game-age-sync-glow' : ''
                    }`}
                  >
                    {THAI_NAME_PREFIX_OPTIONS.map((option) => {
                      const active = namePrefix === option.value;
                      return (
                        <button
                          key={option.label}
                          type="button"
                          data-testid={`enroll-prefix-${option.value || 'none'}`}
                          aria-pressed={active}
                          onClick={() => handleSelectNamePrefix(option.value)}
                          className={`h-11 rounded-xl border-2 px-2 text-base font-black transition-all duration-150 cursor-pointer ${
                            active
                              ? 'border-[#0b3960] bg-[#0F4C81] text-white btn-game-3d-blue'
                              : 'border-slate-200 bg-white text-slate-700 btn-game-3d-white hover:border-sky-300 hover:bg-sky-50/70'
                          }`}
                        >
                          {option.label}
                        </button>
                      );
                    })}
                  </div>
                </div>

                {/* Split First Name & Last Name */}
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label
                      htmlFor="enroll-first-name-input"
                      className="block text-sm font-extrabold text-slate-700 mb-1"
                    >
                      ชื่อจริง <span className="text-rose-500">*</span>
                    </label>
                    <div className="relative">
                      <input
                        id="enroll-first-name-input"
                        data-testid="enroll-first-name-input"
                        required
                        value={firstName}
                        onPaste={handleFirstNamePaste}
                        onChange={(e) => {
                          setFirstName(e.target.value);
                          if (errorMsg) setErrorMsg(null);
                        }}
                        placeholder="เช่น ธนกร"
                        className={`h-12 w-full rounded-xl border-2 pl-4 pr-10 text-base font-bold text-slate-900 placeholder:text-slate-400 transition-all focus:outline-none ${
                          enrollmentReadiness.checks.hasFirstName
                            ? 'border-emerald-300 bg-white focus:border-emerald-500'
                            : 'border-slate-200 bg-slate-50/50 focus:bg-white focus:border-[#0F4C81]'
                        }`}
                      />
                      {enrollmentReadiness.checks.hasFirstName && (
                        <Check className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 h-5 w-5 text-emerald-600" />
                      )}
                    </div>
                  </div>

                  <div>
                    <label
                      htmlFor="enroll-last-name-input"
                      className="block text-sm font-extrabold text-slate-700 mb-1"
                    >
                      นามสกุล <span className="text-rose-500">*</span>
                    </label>
                    <div className="relative">
                      <input
                        id="enroll-last-name-input"
                        data-testid="enroll-last-name-input"
                        required
                        value={lastName}
                        onChange={(e) => {
                          setLastName(e.target.value);
                          if (errorMsg) setErrorMsg(null);
                        }}
                        placeholder="เช่น รุ่งเรือง"
                        className={`h-12 w-full rounded-xl border-2 pl-4 pr-10 text-base font-bold text-slate-900 placeholder:text-slate-400 transition-all focus:outline-none ${
                          enrollmentReadiness.checks.hasLastName
                            ? 'border-emerald-300 bg-white focus:border-emerald-500'
                            : 'border-slate-200 bg-slate-50/50 focus:bg-white focus:border-[#0F4C81]'
                        }`}
                      />
                      {enrollmentReadiness.checks.hasLastName && (
                        <Check className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 h-5 w-5 text-emerald-600" />
                      )}
                    </div>
                  </div>
                </div>

                {/* DOB + Age + Gender (Full-Width Aligned h-12 Row) */}
                <div className="grid grid-cols-12 gap-3 items-end">
                  <div className="col-span-5">
                    <label className="block text-sm font-extrabold text-slate-700 mb-1">
                      วันเกิด
                    </label>
                    <GameDatePickerField
                      mode="dob"
                      value={birthDate}
                      onChange={handleBirthDateChange}
                      ageHint={age}
                      maxDate={todayIso}
                      placeholder="แตะเลือกวันเกิด"
                      popupTitle="เลือกวันเดือนปีเกิดผู้ป่วย"
                      testId="enroll-dob-picker"
                    />
                  </div>

                  <div className="col-span-2">
                    <label className="block text-sm font-extrabold text-slate-700 mb-1">
                      อายุ (ปี)
                    </label>
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
                      className={`h-12 w-full rounded-xl border-2 px-2 text-center text-lg font-black transition-all focus:border-teal-500 focus:bg-white focus:outline-none ${
                        ageSyncedPulse
                          ? 'border-emerald-400 bg-emerald-50 text-emerald-900 game-age-sync-glow'
                          : birthDate
                            ? 'border-emerald-200 bg-emerald-50/40 text-emerald-900'
                            : 'border-slate-200 bg-slate-50/70 text-slate-800'
                      }`}
                    />
                  </div>

                  <div className="col-span-5">
                    <label className="block text-sm font-extrabold text-slate-700 mb-1">
                      เพศ
                    </label>
                    <div
                      data-testid="enroll-gender-bar"
                      className={`grid grid-cols-3 gap-2 rounded-xl transition-all ${
                        genderSyncedPulse ? 'bg-emerald-50/80 game-age-sync-glow' : ''
                      }`}
                    >
                      {PATIENT_GENDER_OPTIONS.map((gOpt) => {
                        const active = gender === gOpt.value;
                        const activeStyle =
                          gOpt.value === 'หญิง'
                            ? 'border-rose-500 bg-rose-500 text-white btn-game-3d-rose'
                            : gOpt.value === 'อื่นๆ'
                              ? 'border-teal-600 bg-teal-600 text-white btn-game-3d-emerald'
                              : 'border-[#0b3960] bg-[#0F4C81] text-white btn-game-3d-blue';
                        return (
                          <button
                            key={gOpt.value}
                            type="button"
                            data-testid={`enroll-gender-${gOpt.value}`}
                            aria-pressed={active}
                            onClick={() => handleSelectGender(gOpt.value)}
                            className={`h-12 rounded-xl border-2 text-base font-black transition-all duration-150 cursor-pointer ${
                              active
                                ? activeStyle
                                : 'border-slate-200 bg-white text-slate-700 btn-game-3d-white hover:border-sky-300'
                            }`}
                          >
                            {gOpt.label}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                </div>

                {/* HN + Phone + PT */}
                <div className="grid grid-cols-12 gap-3">
                  <div className="col-span-4">
                    <div className="flex items-center justify-between mb-1">
                      <label className="text-sm font-extrabold text-slate-700">รหัส HN</label>
                      <button
                        type="button"
                        onClick={() => {
                          playGameSfx('tap');
                          setHn(generatePatientHn());
                        }}
                        className="inline-flex items-center gap-1 rounded-full bg-sky-50 border border-sky-200 px-2 py-0.5 text-xs font-extrabold text-[#0F4C81] hover:bg-sky-100 cursor-pointer"
                      >
                        <RefreshCw className="h-3 w-3" />
                        สุ่ม
                      </button>
                    </div>
                    <input
                      value={hn}
                      onChange={(e) => setHn(e.target.value)}
                      className="h-11 w-full rounded-xl border-2 border-slate-200 bg-slate-50 px-3.5 font-mono text-base font-black text-slate-800 focus:border-[#0F4C81] focus:bg-white focus:outline-none"
                    />
                  </div>

                  <div className="col-span-4">
                    <div className="flex items-center justify-between mb-1">
                      <label className="text-sm font-extrabold text-slate-700">เบอร์โทร</label>
                      <span
                        className={`text-xs font-black ${
                          phoneDigitCount === 10 ? 'text-emerald-600' : 'text-slate-400'
                        }`}
                      >
                        {phoneDigitCount}/10
                      </span>
                    </div>
                    <input
                      type="tel"
                      inputMode="numeric"
                      pattern="[0-9]*"
                      maxLength={10}
                      value={phone}
                      onChange={(e) => setPhone(e.target.value.replace(/\D/g, '').slice(0, 10))}
                      placeholder="08XXXXXXXX"
                      className={`h-11 w-full rounded-xl border-2 px-3.5 text-base font-bold text-slate-800 placeholder:text-slate-400 focus:bg-white focus:outline-none ${
                        phoneDigitCount === 10
                          ? 'border-emerald-300 bg-emerald-50/30'
                          : 'border-slate-200 bg-slate-50/70 focus:border-teal-500'
                      }`}
                    />
                  </div>

                  <div className="col-span-4">
                    <label className="block text-sm font-extrabold text-slate-700 mb-1">
                      นักกายภาพ <span className="text-rose-500">*</span>
                    </label>
                    <select
                      value={assignedPtId}
                      onChange={(e) => setAssignedPtId(e.target.value)}
                      disabled={currentUser.role === 'pt'}
                      className="h-11 w-full rounded-xl border-2 border-slate-200 bg-slate-50/70 px-3 text-sm sm:text-base font-bold text-slate-800 focus:border-teal-500 focus:bg-white focus:outline-none disabled:opacity-75"
                    >
                      {ptList.length === 0 && <option value="">กภ. ประจำคลินิก</option>}
                      {ptList.map((pt) => (
                        <option key={pt.id} value={pt.id}>
                          {pt.fullName}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              {/* Card 2: Clinical & Rehab Arena */}
              <div className="rounded-2xl border-2 border-emerald-100 bg-white p-4 shadow-2xs space-y-3">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <span className="inline-flex h-7 w-7 items-center justify-center rounded-lg bg-emerald-600 text-sm font-black text-white">
                      2
                    </span>
                    <Stethoscope className="h-5 w-5 text-emerald-700" />
                    <span className="text-base font-black text-slate-900">
                      อาการ &amp; กำลังกล้ามเนื้อ
                    </span>
                  </div>
                  <span className="relative overflow-hidden rounded-full bg-emerald-700 px-3.5 py-1 text-xs sm:text-sm font-black text-white game-badge-shimmer">
                    Grade {motorAssessmentScore}/6 • {maxSafeRomDeg}°
                  </span>
                </div>

                {/* Target Joint & Patient Group */}
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-sm font-extrabold text-slate-700 mb-1">
                      ข้อต่อเป้าหมาย <span className="text-rose-500">*</span>
                    </label>
                    <select
                      value={targetJoint}
                      onChange={(e) => {
                        playGameSfx('tap');
                        setTargetJoint(e.target.value);
                      }}
                      className="h-11 w-full rounded-xl border-2 border-slate-200 bg-slate-50/70 px-3 text-sm sm:text-base font-bold text-slate-800 focus:border-emerald-500 focus:bg-white focus:outline-none"
                    >
                      <option value="ข้อไหล่และแขน Stroke ADL (Shoulder & Arm)">
                        ข้อไหล่และแขน Stroke ADL
                      </option>
                      <option value="คอ บ่า และแขน สธ. ผู้สูงอายุ (MOPH Upper Body)">
                        คอ บ่า และแขน สธ. ผู้สูงอายุ
                      </option>
                      <option value="ข้อไหล่ (Shoulder)">ข้อไหล่ (Shoulder)</option>
                      <option value="ข้อศอกและแขน (Elbow & Arm)">ข้อศอกและแขน</option>
                      <option value="ข้อมือและกล้ามเนื้อมัดเล็ก (Wrist & Hand Fingers)">
                        ข้อมือและนิ้วมือ
                      </option>
                      <option value="กล้ามเนื้อคอและบ่า สธ. (Cervical & Shoulder Girdle)">
                        กล้ามเนื้อคอและบ่า สธ.
                      </option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-extrabold text-slate-700 mb-1">
                      กลุ่มผู้ป่วย
                    </label>
                    <select
                      value={strokeType}
                      onChange={(e) => {
                        playGameSfx('tap');
                        setStrokeType(e.target.value as StrokeType);
                      }}
                      className="h-11 w-full rounded-xl border-2 border-slate-200 bg-slate-50/70 px-3 text-sm sm:text-base font-bold text-slate-800 focus:border-emerald-500 focus:bg-white focus:outline-none"
                    >
                      <option value="ISCHEMIC">Stroke เส้นเลือดตีบ/ตัน (Ischemic)</option>
                      <option value="HEMORRHAGIC">Stroke เส้นเลือดแตก (Hemorrhagic)</option>
                      <option value="ELDERLY_GENERAL">ผู้สูงอายุทั่วไป (MOPH)</option>
                      <option value="NEURO">ระบบประสาท (Neuro)</option>
                      <option value="ORTHOPEDIC">กระดูกและข้อ (Ortho)</option>
                    </select>
                  </div>
                </div>

                {/* Diagnosis + 1-Line Quick Presets */}
                <div className="space-y-2">
                  <input
                    value={clinicalDiagnosis}
                    onChange={(e) => setClinicalDiagnosis(e.target.value)}
                    placeholder="อาการ / การวินิจฉัย เช่น Ischemic Stroke แขนขวาอ่อนแรง"
                    className="h-11 w-full rounded-xl border-2 border-slate-200 bg-slate-50/70 px-3.5 text-sm sm:text-base font-semibold text-slate-800 placeholder:text-slate-400 focus:border-emerald-500 focus:bg-white focus:outline-none"
                  />
                  <div className="flex flex-wrap items-center gap-1.5">
                    {QUICK_CLINICAL_DIAGNOSIS_PRESETS.map((preset) => {
                      const active = clinicalDiagnosis === preset;
                      return (
                        <button
                          key={preset}
                          type="button"
                          onClick={() => {
                            playGameSfx('tap');
                            setClinicalDiagnosis(preset);
                          }}
                          className={`rounded-full border px-3 py-1 text-xs sm:text-sm font-bold transition cursor-pointer ${
                            active
                              ? 'border-emerald-600 bg-emerald-600 text-white'
                              : 'border-slate-200 bg-slate-50 text-slate-700 hover:border-emerald-300 hover:bg-emerald-50'
                          }`}
                        >
                          {preset}
                        </button>
                      );
                    })}
                  </div>
                </div>

                {/* Emerald Power Arena: Weak Side + ROM + ADL + Mahidol 0-6 */}
                <div className="rounded-2xl border-2 border-emerald-200/90 bg-gradient-to-br from-emerald-50/75 via-teal-50/35 to-white p-3.5 space-y-3">
                  <div className="grid grid-cols-12 gap-3 items-end">
                    <div className="col-span-6">
                      <label className="block text-xs sm:text-sm font-extrabold text-emerald-950 mb-1">
                        ข้างที่อ่อนแรง
                      </label>
                      <div className="grid grid-cols-4 gap-1.5">
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
                            onClick={() => {
                              playGameSfx('tap');
                              setAffectedSide(val);
                            }}
                            className={`h-11 rounded-xl text-sm sm:text-base font-black border-2 transition-all cursor-pointer ${
                              affectedSide === val
                                ? 'border-emerald-700 bg-emerald-600 text-white btn-game-3d-emerald'
                                : 'border-emerald-200 bg-white text-slate-700 btn-game-3d-white hover:bg-emerald-50'
                            }`}
                          >
                            {lbl}
                          </button>
                        ))}
                      </div>
                    </div>

                    <div className="col-span-2">
                      <label className="block text-xs sm:text-sm font-extrabold text-emerald-950 mb-1">
                        ROM (°)
                      </label>
                      <input
                        type="number"
                        min={30}
                        max={180}
                        value={maxSafeRomDeg}
                        onChange={(e) =>
                          setMaxSafeRomDeg(
                            Math.max(30, Math.min(180, Number(e.target.value) || 30)),
                          )
                        }
                        className={`h-11 w-full rounded-xl border-2 bg-white px-2 text-center font-mono text-base font-black text-emerald-800 focus:border-emerald-500 focus:outline-none ${
                          romSyncedPulse
                            ? 'border-emerald-400 bg-emerald-50 game-age-sync-glow'
                            : 'border-emerald-200'
                        }`}
                      />
                    </div>

                    <div className="col-span-4">
                      <label className="block text-xs sm:text-sm font-extrabold text-emerald-950 mb-1">
                        ADL ใส่เสื้อ (0–5)
                      </label>
                      <select
                        value={adlDressingScore}
                        onChange={(e) => {
                          playGameSfx('tap');
                          setAdlDressingScore(Number(e.target.value));
                        }}
                        className="h-11 w-full rounded-xl border-2 border-emerald-200 bg-white px-2.5 text-sm font-extrabold text-slate-800 focus:border-emerald-500 focus:outline-none"
                      >
                        <option value={0}>0: ช่วยตัวเองไม่ได้</option>
                        <option value={1}>1: ช่วยมาก (พึ่งพาสูง)</option>
                        <option value={2}>2: สอดแขนได้บ้าง</option>
                        <option value={3}>3: ทำได้เมื่อมีผู้ช่วย</option>
                        <option value={4}>4: ใส่เองได้ช้าๆ</option>
                        <option value={5}>5: ใส่เองได้คล่อง</option>
                      </select>
                    </div>
                  </div>

                  <div>
                    <div className="flex items-center justify-between gap-2 mb-1.5">
                      <label className="text-sm font-black text-emerald-950 inline-flex items-center gap-1.5">
                        <Activity className="h-4 w-4 text-emerald-600" />
                        กำลังกล้ามเนื้อ มหิดล (0–6)
                      </label>
                      <span className="text-xs sm:text-sm font-extrabold text-emerald-700 truncate max-w-[380px]">
                        {MAHIDOL_MOTOR_SCALE[motorAssessmentScore]?.titleTh}
                      </span>
                    </div>
                    <div className="grid grid-cols-7 gap-2">
                      {MAHIDOL_MOTOR_SCALE.map((lvl) => {
                        const active = motorAssessmentScore === lvl.score;
                        return (
                          <button
                            key={lvl.score}
                            type="button"
                            onClick={() =>
                              handleSelectMotorScore(lvl.score, lvl.recommendedMaxRomDeg)
                            }
                            className={`rounded-xl border-2 py-2 px-1 text-center transition-all duration-150 cursor-pointer ${
                              active
                                ? 'border-emerald-700 bg-emerald-600 text-white btn-game-3d-emerald'
                                : 'border-emerald-200 bg-white text-slate-700 btn-game-3d-white hover:border-emerald-400 hover:bg-emerald-50'
                            }`}
                          >
                            <div className="text-lg font-black leading-none">{lvl.score}</div>
                            <div
                              className={`mt-1 text-xs font-bold ${
                                active ? 'text-emerald-100' : 'text-emerald-700'
                              }`}
                            >
                              {lvl.recommendedMaxRomDeg}°
                            </div>
                          </button>
                        );
                      })}
                    </div>
                    <p className="mt-2 rounded-xl border border-emerald-200/80 bg-white px-3.5 py-1.5 text-xs sm:text-sm font-semibold text-emerald-950 truncate">
                      {MAHIDOL_MOTOR_SCALE[motorAssessmentScore]?.descTh}
                    </p>
                  </div>
                </div>

                {/* Additional Medical Details + Backup PIN */}
                <div className="space-y-2.5 border-t border-slate-100 pt-2">
                  <div className="grid grid-cols-2 gap-3">
                    <input
                      value={contraindications}
                      onChange={(e) => setContraindications(e.target.value)}
                      placeholder="ข้อห้ามกายภาพ (ถ้ามี)"
                      className="h-10 w-full rounded-xl border-2 border-slate-200 bg-slate-50/70 px-3.5 text-sm font-medium text-slate-800 placeholder:text-slate-400 focus:border-teal-500 focus:bg-white focus:outline-none"
                    />
                    <input
                      value={underlyingDiseases}
                      onChange={(e) => setUnderlyingDiseases(e.target.value)}
                      placeholder="โรคประจำตัว (ถ้าไม่มี -)"
                      className="h-10 w-full rounded-xl border-2 border-slate-200 bg-slate-50/70 px-3.5 text-sm font-medium text-slate-800 placeholder:text-slate-400 focus:border-teal-500 focus:bg-white focus:outline-none"
                    />
                  </div>

                  <div className="grid grid-cols-12 gap-3">
                    <input
                      value={allergies}
                      onChange={(e) => setAllergies(e.target.value)}
                      placeholder="ประวัติแพ้ยา"
                      className="col-span-4 h-10 w-full rounded-xl border-2 border-slate-200 bg-slate-50/70 px-3.5 text-sm font-medium text-slate-800 placeholder:text-slate-400 focus:border-teal-500 focus:bg-white focus:outline-none"
                    />
                    <input
                      value={doctorNotes}
                      onChange={(e) => setDoctorNotes(e.target.value)}
                      placeholder="บันทึกแพทย์ / PT"
                      className="col-span-5 h-10 w-full rounded-xl border-2 border-slate-200 bg-slate-50/70 px-3.5 text-sm font-medium text-slate-800 placeholder:text-slate-400 focus:border-teal-500 focus:bg-white focus:outline-none"
                    />
                    <div className="col-span-3 relative">
                      <KeyRound className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-sky-600" />
                      <input
                        type="text"
                        inputMode="numeric"
                        pattern="[0-9]*"
                        value={pinCode}
                        maxLength={4}
                        title="รหัส PIN สำรอง 4 หลัก"
                        onChange={(e) => setPinCode(e.target.value.replace(/\D/g, '').slice(0, 4))}
                        className="h-10 w-full rounded-xl border-2 border-sky-200 bg-white pl-9 pr-3 text-center font-mono text-base font-black tracking-widest text-[#0F4C81] focus:border-[#0F4C81] focus:outline-none"
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Card 3: Emergency Contact (Large Readable 1-Row) */}
              <div className="rounded-2xl border-2 border-rose-100 bg-gradient-to-br from-rose-50/35 to-white p-3.5 shadow-2xs space-y-2.5">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <span className="inline-flex h-7 w-7 items-center justify-center rounded-lg bg-rose-600 text-sm font-black text-white">
                      3
                    </span>
                    <HeartPulse className="h-5 w-5 text-rose-600" />
                    <span className="text-base font-black text-slate-900">ผู้ติดต่อฉุกเฉิน</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    {QUICK_EMERGENCY_RELATION_PRESETS.map((rel) => (
                      <button
                        key={rel}
                        type="button"
                        onClick={() => {
                          playGameSfx('tap');
                          setEmergencyRelation(rel);
                        }}
                        className={`rounded-full border px-3 py-1 text-xs font-bold transition cursor-pointer ${
                          emergencyRelation === rel
                            ? 'border-rose-600 bg-rose-600 text-white'
                            : 'border-rose-200 bg-white text-rose-700 hover:bg-rose-50'
                        }`}
                      >
                        {rel}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-3">
                  <input
                    value={emergencyName}
                    onChange={(e) => setEmergencyName(e.target.value)}
                    placeholder="ชื่อผู้ติดต่อ"
                    className="h-10 w-full rounded-xl border-2 border-slate-200 bg-white px-3.5 text-sm font-semibold text-slate-800 placeholder:text-slate-400 focus:border-rose-400 focus:outline-none"
                  />
                  <input
                    value={emergencyRelation}
                    onChange={(e) => setEmergencyRelation(e.target.value)}
                    placeholder="ความสัมพันธ์"
                    className="h-10 w-full rounded-xl border-2 border-slate-200 bg-white px-3.5 text-sm font-semibold text-slate-800 placeholder:text-slate-400 focus:border-rose-400 focus:outline-none"
                  />
                  <input
                    type="tel"
                    inputMode="numeric"
                    pattern="[0-9]*"
                    maxLength={10}
                    value={emergencyPhone}
                    onChange={(e) =>
                      setEmergencyPhone(e.target.value.replace(/\D/g, '').slice(0, 10))
                    }
                    placeholder="เบอร์ฉุกเฉิน"
                    className="h-10 w-full rounded-xl border-2 border-slate-200 bg-white px-3.5 text-sm font-semibold text-slate-800 placeholder:text-slate-400 focus:border-rose-400 focus:outline-none"
                  />
                </div>
              </div>

              {/* Card 4: PDPA Consent & Staff Auditor */}
              <div
                className={`rounded-2xl border-2 px-4 py-3 transition-all ${
                  pdpaAccepted
                    ? 'border-emerald-200 bg-gradient-to-r from-emerald-50/90 to-teal-50/40'
                    : 'border-amber-300 bg-amber-50/70'
                }`}
              >
                <label className="flex flex-wrap items-center justify-between gap-2 cursor-pointer">
                  <div className="flex items-center gap-2.5 min-w-0">
                    <input
                      type="checkbox"
                      checked={pdpaAccepted}
                      onChange={(e) => {
                        playGameSfx(e.target.checked ? 'correct' : 'tap');
                        setPdpaAccepted(e.target.checked);
                      }}
                      className="h-5 w-5 rounded border-emerald-400 text-emerald-600 focus:ring-emerald-500 cursor-pointer shrink-0"
                    />
                    <span className="inline-flex items-center gap-1.5 text-sm font-black text-emerald-950">
                      <ShieldCheck className="h-4.5 w-4.5 text-emerald-600 shrink-0" />
                      ยินยอม PDPA ใบหน้า 3D 120 เฟรม (เวชระเบียนถาวรตามกฎหมาย)
                    </span>
                  </div>
                  <span className="inline-flex items-center gap-1.5 rounded-full bg-white/90 border border-emerald-200 px-3 py-1 text-xs sm:text-sm font-extrabold text-emerald-800 shrink-0">
                    <BadgeCheck className="h-4 w-4 text-emerald-600" />
                    {currentUser.fullName}
                  </span>
                </label>
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
                    void verifyNoDuplicateFace(result);
                  }}
                  onReadyStateChange={(ready, triggerFinalize) => {
                    setAllFiveAnglesReady(ready);
                    triggerFinalizeRef.current = triggerFinalize;
                    if (!ready) {
                      setBiometric120Result(null);
                      setCapturedSnapshotUrl(undefined);
                      setDuplicateFaceWarning(null);
                    }
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
                      setDuplicateFaceWarning(null);
                      setBiometric120Result(null);
                      setCapturedSnapshotUrl(undefined);
                      setAllFiveAnglesReady(false);
                      triggerFinalizeRef.current = null;
                      setCapturing(true);
                    }}
                    className="mt-4 rounded-xl bg-emerald-600 px-5 py-2.5 text-xs font-extrabold text-white shadow hover:bg-emerald-500"
                  >
                    {biometric120Result ? 'สแกน 120 เฟรมใหม่อีกครั้ง' : 'เปิดกล้องสแกนใบหน้า 120 เฟรม (3D)'}
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
                <span className="inline-flex items-center gap-1 rounded-full bg-emerald-100 px-2.5 py-1 text-[11px] font-extrabold text-emerald-800">
                  {biometric120Result
                    ? `Face ID 120 เฟรม (1024-D 3D) พร้อมแล้ว (${Math.round(biometric120Result.qualityScore * 100)}%)`
                    : 'กรุณาสแกน Face ID 120 เฟรมก่อนบันทึก'}
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
                      เลือกท่ากายภาพบำบัดสำหรับคนไข้รายนี้ (15 ท่าหลัก — เลือกข้างในแต่ละการ์ด)
                    </span>
                    <span className="rounded-full bg-emerald-50 border border-emerald-200 px-2.5 py-0.5 text-[11px] font-extrabold text-emerald-700">
                      เลือกแล้ว {selectedEntries.length} ท่า
                    </span>
                  </div>

                  <div className="flex flex-wrap items-center gap-1.5">
                    <button
                      type="button"
                      onClick={() => {
                        setPrescriptionTouched(true);
                        const recIds = getRecommendedExerciseIdsForJoint(targetJoint, catalogExercises);
                        const nextMap: Record<string, ExerciseSideCode> = {};
                        for (const baseCode of recIds) {
                          nextMap[baseCode] = BILATERAL_INHERENT_EXERCISE_CODES.has(baseCode)
                            ? 'BOTH'
                            : globalQuickSide || normalizeSideValue(affectedSide) || 'RIGHT';
                        }
                        setSelectedSideByBaseCode(nextMap);
                      }}
                      className="rounded-lg border border-teal-200 bg-teal-50 px-2.5 py-1 text-[11px] font-bold text-teal-800 hover:bg-teal-100 cursor-pointer"
                    >
                      แนะนำตามข้อต่อ
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setPrescriptionTouched(true);
                        setSelectedSideByBaseCode((prev) => {
                          const next = { ...prev };
                          for (const ex of filteredExercises) {
                            const baseCode = parseExerciseCodeAndSide(ex.code || ex.id).baseCode;
                            if (!next[baseCode]) {
                              next[baseCode] = BILATERAL_INHERENT_EXERCISE_CODES.has(baseCode)
                                ? 'BOTH'
                                : globalQuickSide || resolveDefaultSideForExercise(ex, affectedSide);
                            }
                          }
                          return next;
                        });
                      }}
                      className="rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1 text-[11px] font-bold text-slate-700 hover:bg-emerald-50 hover:text-emerald-700 cursor-pointer"
                    >
                      เลือกทั้งหมดในหมวด
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setPrescriptionTouched(true);
                        setSelectedSideByBaseCode({});
                      }}
                      className="rounded-lg border border-slate-200 bg-white px-2.5 py-1 text-[11px] font-bold text-slate-500 hover:bg-rose-50 hover:text-rose-700 cursor-pointer"
                    >
                      ล้างที่เลือก
                    </button>
                  </div>
                </div>

                <GlobalQuickSideSelector
                  value={globalQuickSide}
                  selectedCount={selectedEntries.length}
                  onSelectGlobal={applyGlobalSideToSelected}
                  lang={lang}
                />

                {selectedEntries.length > 0 && (
                  <div className="rounded-xl border border-emerald-200 bg-emerald-50/70 p-2.5 space-y-1.5">
                    <div className="flex items-center justify-between text-[11px] font-bold text-emerald-800">
                      <span>ท่าที่เลือกไว้ทั้งหมด ({selectedEntries.length} ท่า — ทุกหมวด)</span>
                      <span className="text-[10px] font-medium text-emerald-700">กดที่แท็กเพื่อเอาออกได้</span>
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {selectedEntries.map(([baseCode, side]) => {
                        const found = catalogExercises.find(
                          (c) => parseExerciseCodeAndSide(c.code || c.id).baseCode === baseCode,
                        );
                        const label = found ? found.name[lang] || found.name.th : baseCode;
                        const sideMeta = getSideBadgeMeta(side);
                        return (
                          <button
                            key={baseCode}
                            type="button"
                            onClick={() =>
                              setSelectedSideByBaseCode((prev) => {
                                const next = { ...prev };
                                delete next[baseCode];
                                return next;
                              })
                            }
                            className="inline-flex items-center gap-1 rounded-full bg-white border border-emerald-300 px-2.5 py-0.5 text-[11px] font-bold text-emerald-900 hover:bg-rose-50 hover:border-rose-300 hover:text-rose-800 cursor-pointer"
                          >
                            <span>{label}</span>
                            <span
                              className={`rounded-full border px-1.5 py-0 text-[10px] font-extrabold ${sideMeta.badgeClass}`}
                            >
                              {sideMeta.shortLabel[lang]}
                            </span>
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

                {/* Interactive Unified Multi-Select Exercise Grid */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 max-h-60 overflow-y-auto pr-1">
                  {filteredExercises.map((ex) => {
                    const baseCode = parseExerciseCodeAndSide(ex.code || ex.id).baseCode;
                    const assignedSide = selectedSideByBaseCode[baseCode] ?? null;
                    const checked = assignedSide !== null;
                    const effectiveDisplay = assignedSide
                      ? formatEffectiveExerciseCode(baseCode, assignedSide)
                      : baseCode;
                    const displayCopy = assignedSide
                      ? resolveSideAdjustedExerciseCopy(ex, assignedSide)
                      : null;
                    return (
                      <div
                        key={baseCode}
                        role="button"
                        tabIndex={0}
                        onClick={() => toggleExerciseSelection(ex)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' || e.key === ' ') {
                            e.preventDefault();
                            toggleExerciseSelection(ex);
                          }
                        }}
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
                        <div className="min-w-0 flex-1 space-y-1.5">
                          <div className="flex items-center justify-between gap-1.5">
                            <span className="font-extrabold text-slate-900 truncate">
                              {displayCopy
                                ? (lang === 'en' ? displayCopy.nameEn : displayCopy.nameTh)
                                : ex.name[lang] || ex.name.th}
                            </span>
                            <span className="shrink-0 rounded bg-slate-100 px-1.5 py-0.5 font-mono text-[10px] font-bold text-slate-600">
                              {effectiveDisplay}
                            </span>
                          </div>
                          <p className="text-[11px] text-slate-500 line-clamp-1">
                            {ex.instruction[lang] || ex.instruction.th}
                          </p>
                          <div className="flex flex-wrap items-center justify-between gap-1.5 pt-0.5">
                            <ExerciseSidePills
                              value={assignedSide}
                              onSelect={(side) => setCardSide(baseCode, side)}
                              lang={lang}
                              compact
                            />
                            <div className="text-[10px] font-bold text-emerald-700">
                              เป้า {ex.detection.targetMinDeg}°–{ex.detection.targetMaxDeg}° • {ex.detection.defaultSets}×{ex.detection.defaultReps}
                            </div>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>

                {selectedEntries.length === 0 && (
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
                    เลือกไว้ {scheduledDates.length} วัน ({selectedEntries.length * scheduledDates.length} รายการ)
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
                    <GameDatePickerField
                      mode="appointment"
                      compact
                      value={customDateInput}
                      onChange={setCustomDateInput}
                      onConfirmDate={(iso) => {
                        setCustomDateInput(iso);
                        addScheduledDates([iso]);
                      }}
                      minDate={todayIso}
                      highlightedDates={scheduledDates}
                      placeholder="เลือกวันนัดเพิ่ม"
                      popupTitle="เลือกวันนัดฝึกกายภาพบำบัด"
                      testId="enroll-schedule-date-picker"
                    />
                    <button
                      type="button"
                      onClick={() => addScheduledDates([customDateInput])}
                      className="btn-game-3d-blue inline-flex items-center gap-1 rounded-xl bg-[#0F4C81] px-3 py-1.5 text-[11px] font-bold text-white hover:bg-[#0c3f6b] cursor-pointer"
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
                      ? `VERIFIED (${createdPatient.faceFramesCount ?? 120} เฟรม • 1024-D 3D)`
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
                        const schedBase = parseExerciseCodeAndSide(sched.exerciseId || '').baseCode;
                        const exMeta = catalogExercises.find(
                          (e) =>
                            e.id === sched.exerciseId ||
                            parseExerciseCodeAndSide(e.code || e.id).baseCode === schedBase,
                        );
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
        <div
          className={`flex flex-wrap items-center gap-3 border-t border-slate-200/80 bg-white px-5 sm:px-6 py-3.5 ${
            step === 1 ? 'justify-between' : 'justify-end'
          }`}
        >
          {step === 1 && (
            <>
              <div className="hidden sm:flex items-center gap-2 text-sm font-bold">
                <span
                  className={`inline-flex h-2.5 w-2.5 rounded-full ${
                    enrollmentReadiness.readyForFaceScan ? 'bg-emerald-500' : 'bg-sky-400'
                  }`}
                />
                <span
                  className={
                    enrollmentReadiness.readyForFaceScan ? 'text-emerald-700' : 'text-slate-600'
                  }
                >
                  {enrollmentReadiness.readyForFaceScan
                    ? `พร้อมลงทะเบียน: ${fullName}`
                    : 'กรอกชื่อจริงและนามสกุลเพื่อไปต่อ'}
                </span>
              </div>
              <div className="flex flex-1 sm:flex-initial items-center justify-end gap-2.5">
                <button
                  type="button"
                  onClick={() => {
                    playGameSfx('tap');
                    onClose();
                  }}
                  className="h-11 rounded-xl border-2 border-slate-200 bg-white px-5 text-sm font-extrabold text-slate-700 btn-game-3d-white hover:bg-slate-50 cursor-pointer"
                >
                  ยกเลิก
                </button>
                <button
                  type="submit"
                  form="staff-enroll-step1"
                  className="inline-flex h-11 items-center gap-2 rounded-xl bg-gradient-to-r from-emerald-600 via-teal-600 to-[#0F4C81] px-6 text-base font-black text-white btn-game-3d-emerald btn-primary-sheen hover:opacity-95 cursor-pointer"
                >
                  <span>ถัดไป: สแกน Face ID 120 เฟรม</span>
                  <ArrowRight className="h-5 w-5" />
                </button>
              </div>
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
                  disabled={
                    saving ||
                    duplicateCheckPending ||
                    Boolean(duplicateFaceWarning?.duplicate) ||
                    (!biometric120Result && !allFiveAnglesReady)
                  }
                  onClick={() => void handleFinalizeEnrollment('verified')}
                  className={`rounded-xl px-5 py-2.5 text-xs font-extrabold transition ${
                    (biometric120Result || allFiveAnglesReady) && !duplicateFaceWarning?.duplicate
                      ? 'bg-emerald-500 hover:bg-emerald-600 text-white shadow-lg shadow-emerald-500/30 cursor-pointer'
                      : 'bg-slate-200 text-slate-400 cursor-not-allowed'
                  } disabled:opacity-60`}
                >
                  {saving
                    ? 'กำลังบันทึกเวชระเบียน...'
                    : duplicateCheckPending
                      ? 'กำลังตรวจสอบใบหน้าซ้ำ...'
                      : 'บันทึกและเปิดใช้งานตู้ Kiosk →'}
                </button>
              ) : (
                <button
                  type="button"
                  disabled={
                    saving ||
                    duplicateCheckPending ||
                    Boolean(duplicateFaceWarning?.duplicate) ||
                    (!biometric120Result && !allFiveAnglesReady)
                  }
                  onClick={() => void handleProceedFromStep2ToStep3()}
                  className={`rounded-xl px-5 py-2.5 text-xs font-extrabold transition ${
                    (biometric120Result || allFiveAnglesReady) && !duplicateFaceWarning?.duplicate
                      ? 'bg-gradient-to-r from-emerald-600 to-blue-600 hover:opacity-95 text-white shadow-lg shadow-emerald-500/25 cursor-pointer'
                      : 'bg-slate-200 text-slate-400 cursor-not-allowed'
                  } disabled:opacity-60`}
                >
                  {saving
                    ? 'กำลังเตรียมข้อมูลใบหน้า...'
                    : duplicateCheckPending
                      ? 'กำลังตรวจสอบใบหน้าซ้ำ...'
                      : '→ ขั้นตอนต่อไป: กำหนดท่ากายภาพบำบัด →'}
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
                disabled={saving || duplicateCheckPending || Boolean(duplicateFaceWarning?.duplicate)}
                onClick={() =>
                  void handleFinalizeEnrollment('verified', undefined, undefined, [])
                }
                className="rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-xs font-bold text-slate-600 hover:bg-slate-100 disabled:opacity-50"
              >
                ข้ามการกำหนดท่า (บันทึกเฉพาะประวัติ)
              </button>
              <button
                type="button"
                disabled={
                  saving ||
                  duplicateCheckPending ||
                  Boolean(duplicateFaceWarning?.duplicate) ||
                  selectedEntries.length === 0 ||
                  scheduledDates.length === 0
                }
                onClick={() => {
                  const built = buildPrescriptionSchedules();
                  void handleFinalizeEnrollment('verified', undefined, undefined, built);
                }}
                className="rounded-xl bg-emerald-600 px-5 py-2.5 text-xs font-extrabold text-white shadow-lg shadow-emerald-600/25 hover:bg-emerald-500 disabled:bg-slate-200 disabled:text-slate-400 disabled:shadow-none disabled:cursor-not-allowed cursor-pointer"
              >
                {saving
                  ? 'กำลังบันทึกเวชระเบียนและท่ากายภาพ...'
                  : `บันทึกคนไข้พร้อมท่ากายภาพ (${selectedEntries.length} ท่า) และเปิดใช้งานตู้ →`}
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
