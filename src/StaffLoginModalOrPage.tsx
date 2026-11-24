import React, { useCallback, useEffect, useRef, useState } from 'react';
import type { DrTechCaresApiClient, UserAccount } from '@drtechcares/api-client';
import { useClinicalI18n, type ClinicalLang } from '@drtechcares/clinical-i18n';
import {
  AlertCircle,
  ArrowLeft,
  ArrowRight,
  BadgeCheck,
  Briefcase,
  Globe,
  Info,
  Lock,
  Phone,
  Shield,
  User,
  UserPlus,
} from 'lucide-react';
import {
  KINETIC_AUTH_HOLD_DELAY_MS,
  KineticAuthStatusBadge,
} from './components/KineticAuthStatusBadge';
import { playGameSfx } from './lib/gameUiSfx';

export type StaffLoginAnimStatus = 'idle' | 'checking' | 'success' | 'not_found';

export async function executeStaffAuthSubmission(params: {
  runRequest: () => Promise<UserAccount>;
  setAnimStatus: (next: StaffLoginAnimStatus) => void;
  setLoading: (loading: boolean) => void;
  setError: (msg: string | null) => void;
  onSuccess: (user: UserAccount) => void;
  scheduleTimer: (fn: () => void, ms: number) => number;
  redirectDelayMs?: number;
  treatErrorAsNotFound?: boolean;
  fallbackErrorMsg?: string;
}): Promise<StaffLoginAnimStatus> {
  params.setError(null);
  params.setLoading(true);
  params.setAnimStatus('checking');
  try {
    const user = await params.runRequest();
    params.setLoading(false);
    params.setAnimStatus('success');
    params.scheduleTimer(() => {
      params.onSuccess(user);
    }, params.redirectDelayMs ?? KINETIC_AUTH_HOLD_DELAY_MS);
    return 'success';
  } catch (err) {
    params.setLoading(false);
    if (params.treatErrorAsNotFound !== false) {
      params.setAnimStatus('not_found');
      return 'not_found';
    }
    params.setAnimStatus('idle');
    params.setError(
      err instanceof Error && err.message ? err.message : (params.fallbackErrorMsg ?? 'Auth error'),
    );
    return 'idle';
  }
}

export type StaffAuthIntent = 'general' | 'kiosk_patient_registration';
export type StaffSuccessStage = 'kinetic' | 'welcome';

export const STAFF_SUCCESS_WELCOME_SWITCH_MS = 2150;
export const STAFF_SUCCESS_TOTAL_REDIRECT_MS = 4650;

interface StaffLoginModalOrPageProps {
  apiClient: DrTechCaresApiClient;
  onSuccess: (user: UserAccount) => void;
  onReturnToKiosk?: () => void;
  authIntent?: StaffAuthIntent;
  asModal?: boolean;
  initialLoginAnimStatus?: StaffLoginAnimStatus;
  initialSuccessStage?: StaffSuccessStage;
}

const SUCCESS_FIREWORK_RAYS: ReadonlyArray<{ deg: string; color: string; delay: string }> = [
  { deg: '0deg', color: '#10b981', delay: '1040ms' },
  { deg: '30deg', color: '#0ea5e9', delay: '1070ms' },
  { deg: '60deg', color: '#14b8a6', delay: '1050ms' },
  { deg: '90deg', color: '#f59e0b', delay: '1080ms' },
  { deg: '120deg', color: '#10b981', delay: '1060ms' },
  { deg: '150deg', color: '#0284c7', delay: '1090ms' },
  { deg: '180deg', color: '#10b981', delay: '1045ms' },
  { deg: '210deg', color: '#38bdf8', delay: '1075ms' },
  { deg: '240deg', color: '#14b8a6', delay: '1055ms' },
  { deg: '270deg', color: '#f59e0b', delay: '1085ms' },
  { deg: '300deg', color: '#10b981', delay: '1065ms' },
  { deg: '330deg', color: '#0ea5e9', delay: '1095ms' },
];

const SUCCESS_CONFETTI_PIECES: ReadonlyArray<{
  cx: string;
  cy: string;
  crot: string;
  cdelay: string;
  color: string;
  shape: 'pill' | 'diamond' | 'ring' | 'dot';
}> = [
  // Wave 1: Initial joyful pop on check draw
  { cx: '-156px', cy: '-128px', crot: '-135deg', cdelay: '180ms', color: '#10b981', shape: 'pill' },
  { cx: '-96px', cy: '-162px', crot: '95deg', cdelay: '220ms', color: '#0ea5e9', shape: 'diamond' },
  { cx: '0px', cy: '-174px', crot: '140deg', cdelay: '200ms', color: '#f59e0b', shape: 'pill' },
  { cx: '104px', cy: '-158px', crot: '-115deg', cdelay: '240ms', color: '#14b8a6', shape: 'diamond' },
  { cx: '162px', cy: '-122px', crot: '135deg', cdelay: '190ms', color: '#38bdf8', shape: 'pill' },
  { cx: '184px', cy: '-38px', crot: '-90deg', cdelay: '280ms', color: '#34d399', shape: 'ring' },
  { cx: '172px', cy: '58px', crot: '125deg', cdelay: '310ms', color: '#0284c7', shape: 'pill' },
  { cx: '118px', cy: '128px', crot: '-145deg', cdelay: '340ms', color: '#fbbf24', shape: 'diamond' },
  { cx: '-122px', cy: '124px', crot: '155deg', cdelay: '330ms', color: '#10b981', shape: 'diamond' },
  { cx: '-174px', cy: '54px', crot: '-110deg', cdelay: '300ms', color: '#0ea5e9', shape: 'pill' },
  { cx: '-184px', cy: '-42px', crot: '85deg', cdelay: '270ms', color: '#2dd4bf', shape: 'ring' },
  { cx: '0px', cy: '148px', crot: '180deg', cdelay: '370ms', color: '#14b8a6', shape: 'pill' },
  // Wave 2: Grand finale celebration burst right when the outer ring locks
  { cx: '-196px', cy: '-92px', crot: '165deg', cdelay: '1080ms', color: '#0ea5e9', shape: 'pill' },
  { cx: '-136px', cy: '-178px', crot: '-125deg', cdelay: '1100ms', color: '#f59e0b', shape: 'ring' },
  { cx: '-46px', cy: '-188px', crot: '85deg', cdelay: '1120ms', color: '#10b981', shape: 'diamond' },
  { cx: '52px', cy: '-186px', crot: '-95deg', cdelay: '1110ms', color: '#38bdf8', shape: 'diamond' },
  { cx: '142px', cy: '-174px', crot: '145deg', cdelay: '1090ms', color: '#10b981', shape: 'ring' },
  { cx: '202px', cy: '-84px', crot: '-150deg', cdelay: '1130ms', color: '#fbbf24', shape: 'pill' },
  { cx: '206px', cy: '18px', crot: '120deg', cdelay: '1140ms', color: '#14b8a6', shape: 'dot' },
  { cx: '152px', cy: '116px', crot: '-130deg', cdelay: '1150ms', color: '#0ea5e9', shape: 'pill' },
  { cx: '-156px', cy: '114px', crot: '130deg', cdelay: '1145ms', color: '#34d399', shape: 'pill' },
  { cx: '-206px', cy: '16px', crot: '-120deg', cdelay: '1135ms', color: '#f59e0b', shape: 'dot' },
];

const WELCOME_CONFETTI_PIECES: ReadonlyArray<{
  cx: string;
  cy: string;
  crot: string;
  cdelay: string;
  color: string;
  shape: 'pill' | 'diamond' | 'ring' | 'dot';
}> = [
  { cx: '-168px', cy: '-116px', crot: '-120deg', cdelay: '120ms', color: '#0F4C81', shape: 'pill' },
  { cx: '-104px', cy: '-154px', crot: '105deg', cdelay: '160ms', color: '#10b981', shape: 'diamond' },
  { cx: '0px', cy: '-166px', crot: '145deg', cdelay: '140ms', color: '#0ea5e9', shape: 'ring' },
  { cx: '108px', cy: '-150px', crot: '-105deg', cdelay: '170ms', color: '#14b8a6', shape: 'diamond' },
  { cx: '170px', cy: '-112px', crot: '125deg', cdelay: '130ms', color: '#f59e0b', shape: 'pill' },
  { cx: '188px', cy: '-16px', crot: '-85deg', cdelay: '210ms', color: '#38bdf8', shape: 'dot' },
  { cx: '154px', cy: '94px', crot: '135deg', cdelay: '240ms', color: '#10b981', shape: 'pill' },
  { cx: '-154px', cy: '94px', crot: '-135deg', cdelay: '230ms', color: '#0284c7', shape: 'pill' },
  { cx: '-188px', cy: '-18px', crot: '85deg', cdelay: '200ms', color: '#34d399', shape: 'dot' },
  { cx: '-64px', cy: '-138px', crot: '-70deg', cdelay: '270ms', color: '#fbbf24', shape: 'dot' },
  { cx: '66px', cy: '-136px', crot: '70deg', cdelay: '280ms', color: '#0ea5e9', shape: 'dot' },
  { cx: '136px', cy: '-66px', crot: '160deg', cdelay: '250ms', color: '#10b981', shape: 'ring' },
];

const SUCCESS_TWINKLE_STARS: ReadonlyArray<{
  className: string;
  sizeClass: string;
  color: string;
  delay: string;
}> = [
  { className: 'top-8 left-10', sizeClass: 'w-6 h-6', color: '#10b981', delay: '0ms' },
  { className: 'top-12 right-12', sizeClass: 'w-5 h-5', color: '#0ea5e9', delay: '380ms' },
  { className: 'bottom-16 left-14', sizeClass: 'w-4 h-4', color: '#f59e0b', delay: '720ms' },
  { className: 'bottom-14 right-14', sizeClass: 'w-6 h-6', color: '#14b8a6', delay: '520ms' },
  { className: 'top-1/2 left-7 -translate-y-1/2', sizeClass: 'w-4 h-4', color: '#38bdf8', delay: '900ms' },
  { className: 'top-1/2 right-7 -translate-y-1/2', sizeClass: 'w-5 h-5', color: '#34d399', delay: '260ms' },
];

const LANG_OPTIONS: { code: ClinicalLang; prefix: string; label: string }[] = [
  { code: 'cn', prefix: 'CN', label: '中文' },
  { code: 'th', prefix: 'TH', label: 'ไทย' },
  { code: 'en', prefix: 'GB', label: 'EN' },
];

const STAFF_LOGIN_COPY: Record<
  ClinicalLang,
  {
    badge: string;
    registrationBadge: string;
    title: string;
    registrationTitle: string;
    subtitle: string;
    registrationSubtitle: string;
    usernameLabel: string;
    usernamePlaceholder: string;
    passwordLabel: string;
    passwordPlaceholder: string;
    roleNotice: string;
    loggingIn: string;
    loginBtn: string;
    registrationLoginBtn: string;
    returnToPatientMain: string;
    securityNotice: string;
    requiredFields: string;
    invalidCreds: string;
    tabLogin: string;
    tabRegister: string;
    registerTitle: string;
    registerSubtitle: string;
    roleLabel: string;
    rolePt: string;
    roleAdmin: string;
    fullNameLabel: string;
    fullNamePlaceholder: string;
    phoneLabel: string;
    phonePlaceholder: string;
    specialtyLabel: string;
    specialtyPlaceholder: string;
    licenseLabel: string;
    licensePlaceholder: string;
    registerBtn: string;
    registrationRegisterBtn: string;
    registering: string;
    registerRequiredFields: string;
    registerPasswordHint: string;
  }
> = {
  th: {
    badge: '+ Dr.Tech.Care • Hospital Staff Portal',
    registrationBadge: 'เจ้าหน้าที่ลงทะเบียนคนไข้ (STAFF-ASSISTED MODE)',
    title: 'เข้าสู่ระบบบุคลากร Dr.Tech.Care',
    registrationTitle: 'ยืนยันสิทธิ์เจ้าหน้าที่เพื่อสมัครคนไข้ใหม่',
    subtitle: 'เข้าถึงระบบบริหารเวชระเบียนคลินิกและตารางกายภาพบำบัด',
    registrationSubtitle: 'กรุณาเข้าสู่ระบบด้วยบัญชี Admin, ผอ.รพ. หรือ นักกายภาพบำบัด (PT)',
    usernameLabel: 'Username (ชื่อผู้ใช้) *',
    usernamePlaceholder: 'เช่น director หรือ pt_somchai',
    passwordLabel: 'Password (รหัสผ่าน) *',
    passwordPlaceholder: '••••••••••••',
    roleNotice: 'ระบบตรวจสอบสิทธิ์อัตโนมัติ: ผอ.รพ. / Super Admin หรือ นักกายภาพบำบัด PT',
    loggingIn: 'กำลังเข้าสู่ระบบ...',
    loginBtn: 'เข้าสู่ระบบบุคลากร',
    registrationLoginBtn: 'ยืนยันรหัสและเปิดหน้าสมัครคนไข้',
    returnToPatientMain: 'กลับหน้าหลัก (คนไข้)',
    securityNotice: 'Hospital Clinical Grade Security • HttpOnly Session Cookie • Zero-PII Leak',
    requiredFields: 'กรุณากรอกชื่อผู้ใช้และรหัสผ่านให้ครบถ้วน',
    invalidCreds: 'ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง',
    tabLogin: 'เข้าสู่ระบบ',
    tabRegister: 'สมัครสมาชิก (Admin / PT)',
    registerTitle: 'สมัครบัญชีบุคลากร Dr.Tech.Care',
    registerSubtitle: 'เปิดบัญชี Admin หรือ นักกายภาพบำบัด (PT) เพื่อเข้าใช้งานพอร์ทัลได้ทันที',
    roleLabel: 'ตำแหน่ง / บทบาท *',
    rolePt: 'นักกายภาพบำบัด (PT)',
    roleAdmin: 'ผู้ดูแลระบบ / ผอ. (Admin)',
    fullNameLabel: 'ชื่อ-นามสกุล (Full Name) *',
    fullNamePlaceholder: 'เช่น กภ.สมหญิง รักษาดี',
    phoneLabel: 'เบอร์โทรศัพท์',
    phonePlaceholder: '08X-XXX-XXXX',
    specialtyLabel: 'ความเชี่ยวชาญ / แผนก',
    specialtyPlaceholder: 'เช่น กายภาพบำบัดระบบกระดูกและกล้ามเนื้อ',
    licenseLabel: 'เลขที่ใบประกอบวิชาชีพ',
    licensePlaceholder: 'เช่น กภ. 99421 (ถ้ามี)',
    registerBtn: 'สมัครสมาชิกและเข้าใช้งานทันที',
    registrationRegisterBtn: 'สร้างบัญชีและเปิดหน้าสมัครคนไข้ทันที',
    registering: 'กำลังสร้างบัญชีบุคลากร...',
    registerRequiredFields: 'กรุณากรอกชื่อ-นามสกุล, Username และรหัสผ่านอย่างน้อย 4 ตัวอักษร',
    registerPasswordHint: 'ตั้งรหัสผ่านอย่างน้อย 4 ตัวอักษร',
  },
  en: {
    badge: '+ Dr.Tech.Care • Hospital Staff Portal',
    registrationBadge: 'Staff-Assisted Patient Registration Mode',
    title: 'Dr.Tech.Care Hospital Staff Login',
    registrationTitle: 'Staff Verification for Patient Enrollment',
    subtitle: 'Access clinical medical records and physical therapy schedule management',
    registrationSubtitle:
      'Please sign in with Admin, Hospital Director, or Physical Therapist (PT) credentials',
    usernameLabel: 'Username *',
    usernamePlaceholder: 'e.g. director or pt_somchai',
    passwordLabel: 'Password *',
    passwordPlaceholder: '••••••••••••',
    roleNotice: 'Automatic Role Detection: Hospital Director / Super Admin or PT Specialist',
    loggingIn: 'Logging in...',
    loginBtn: 'Sign In to Staff Portal',
    registrationLoginBtn: 'Verify & Open Patient Registration',
    returnToPatientMain: 'Back to Main (Patient)',
    securityNotice: 'Hospital Clinical Grade Security • HttpOnly Session Cookie • Zero-PII Leak',
    requiredFields: 'Please fill in both username and password',
    invalidCreds: 'Invalid username or password',
    tabLogin: 'Sign In',
    tabRegister: 'Register (Admin / PT)',
    registerTitle: 'Register Dr.Tech.Care Staff Account',
    registerSubtitle: 'Create an Admin or Physical Therapist (PT) account and start immediately',
    roleLabel: 'Role *',
    rolePt: 'Physical Therapist (PT)',
    roleAdmin: 'Hospital Admin / Director',
    fullNameLabel: 'Full Name *',
    fullNamePlaceholder: 'e.g. Dr. Somying Rehab',
    phoneLabel: 'Phone Number',
    phonePlaceholder: '08X-XXX-XXXX',
    specialtyLabel: 'Specialty / Department',
    specialtyPlaceholder: 'e.g. Orthopedic Rehabilitation',
    licenseLabel: 'License Number',
    licensePlaceholder: 'e.g. PT-99421 (Optional)',
    registerBtn: 'Create Account & Enter Portal',
    registrationRegisterBtn: 'Create Account & Open Patient Registration',
    registering: 'Creating staff account...',
    registerRequiredFields: 'Please enter full name, username, and a password of at least 4 characters',
    registerPasswordHint: 'Minimum 4 characters',
  },
  cn: {
    badge: '+ Dr.Tech.Care • Hospital Staff Portal',
    registrationBadge: '医护人员协助建档模式',
    title: 'Dr.Tech.Care 医院员工登录',
    registrationTitle: '医护人员验证 • 患者建档',
    subtitle: '访问临床病历管理系统与物理康复治疗日程',
    registrationSubtitle: '请使用管理员、院长或物理康复理疗师 (PT) 账号登录验证',
    usernameLabel: '用户名 (Username) *',
    usernamePlaceholder: '例如 director 或 pt_somchai',
    passwordLabel: '密码 (Password) *',
    passwordPlaceholder: '••••••••••••',
    roleNotice: '自动权限校验：医院院长 / 超级管理员 或 物理康复理疗师 PT',
    loggingIn: '正在登录...',
    loginBtn: '登录医护系统',
    registrationLoginBtn: '验证并打开患者注册弹窗',
    returnToPatientMain: '返回首页 (患者)',
    securityNotice: 'Hospital Clinical Grade Security • HttpOnly Session Cookie • Zero-PII Leak',
    requiredFields: '请填写完整的用户名和密码',
    invalidCreds: '用户名或密码错误',
    tabLogin: '员工登录',
    tabRegister: '注册账号 (Admin / PT)',
    registerTitle: '注册 Dr.Tech.Care 医护账号',
    registerSubtitle: '创建管理员 (Admin) 或理疗师 (PT) 账号并立即进入管理后台',
    roleLabel: '职位角色 *',
    rolePt: '物理康复理疗师 (PT)',
    roleAdmin: '医院管理员 / 院长 (Admin)',
    fullNameLabel: '姓名 (Full Name) *',
    fullNamePlaceholder: '例如 李医生 / PT 专家',
    phoneLabel: '联系电话',
    phonePlaceholder: '08X-XXX-XXXX',
    specialtyLabel: '专业科室',
    specialtyPlaceholder: '例如 骨科与运动康复',
    licenseLabel: '执业医师/理疗师证号',
    licensePlaceholder: '例如 PT-99421（选填）',
    registerBtn: '注册并立即进入后台',
    registrationRegisterBtn: '创建账号并立即打开患者注册',
    registering: '正在创建账号...',
    registerRequiredFields: '请填写姓名、用户名以及至少 4 位密码',
    registerPasswordHint: '至少 4 位字符',
  },
};

export function StaffLoginModalOrPage({
  apiClient,
  onSuccess,
  onReturnToKiosk,
  authIntent = 'general',
  asModal = false,
  initialLoginAnimStatus = 'idle',
  initialSuccessStage = 'kinetic',
}: StaffLoginModalOrPageProps) {
  const { lang, setLang } = useClinicalI18n();
  const copy = STAFF_LOGIN_COPY[lang];
  const isRegistrationMode = authIntent === 'kiosk_patient_registration';

  const [mode, setMode] = useState<'login' | 'register'>('login');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [regRole, setRegRole] = useState<'pt' | 'super_admin'>('pt');
  const [regFullName, setRegFullName] = useState('');
  const [regPhone, setRegPhone] = useState('');
  const [regSpecialty, setRegSpecialty] = useState('');
  const [regLicenseNo, setRegLicenseNo] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(initialLoginAnimStatus === 'checking');
  const [loginAnimStatus, setLoginAnimStatus] = useState<StaffLoginAnimStatus>(initialLoginAnimStatus);
  const [successStage, setSuccessStage] = useState<StaffSuccessStage>(initialSuccessStage);
  const [isLeavingKinetic, setIsLeavingKinetic] = useState(false);
  const [authenticatedStaff, setAuthenticatedStaff] = useState<UserAccount | null>(null);
  const successTimerRef = useRef<number | null>(null);
  const usernameInputRef = useRef<HTMLInputElement | null>(null);

  const clearPendingSuccessTimer = useCallback(() => {
    if (successTimerRef.current !== null) {
      window.clearTimeout(successTimerRef.current);
      successTimerRef.current = null;
    }
  }, []);

  useEffect(() => () => clearPendingSuccessTimer(), [clearPendingSuccessTimer]);

  useEffect(() => {
    if (loginAnimStatus === 'success') {
      setSuccessStage(initialSuccessStage);
      setIsLeavingKinetic(false);
      playGameSfx('unlock');
      if (typeof window !== 'undefined' && initialSuccessStage === 'kinetic') {
        const ringLockSfxId = window.setTimeout(() => {
          playGameSfx('combo', { comboCount: 4 });
        }, 1120);
        const startExitId = window.setTimeout(() => {
          setIsLeavingKinetic(true);
        }, STAFF_SUCCESS_WELCOME_SWITCH_MS - 260);
        const switchToWelcomeId = window.setTimeout(() => {
          setIsLeavingKinetic(false);
          setSuccessStage('welcome');
          playGameSfx('victory');
        }, STAFF_SUCCESS_WELCOME_SWITCH_MS);
        return () => {
          window.clearTimeout(ringLockSfxId);
          window.clearTimeout(startExitId);
          window.clearTimeout(switchToWelcomeId);
        };
      }
    } else if (loginAnimStatus === 'not_found') {
      playGameSfx('wrong');
    }
    return undefined;
  }, [loginAnimStatus, initialSuccessStage]);

  const handleReturn = useCallback(() => {
    clearPendingSuccessTimer();
    if (onReturnToKiosk) {
      onReturnToKiosk();
      return;
    }
    if (typeof window !== 'undefined') {
      window.history.pushState({}, '', '/');
      window.dispatchEvent(new PopStateEvent('popstate'));
    }
  }, [clearPendingSuccessTimer, onReturnToKiosk]);

  useEffect(() => {
    if (!asModal) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && loginAnimStatus !== 'checking' && loginAnimStatus !== 'success') {
        handleReturn();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [asModal, handleReturn, loginAnimStatus]);

  const switchMode = (nextMode: 'login' | 'register') => {
    clearPendingSuccessTimer();
    setMode(nextMode);
    setError(null);
    setLoginAnimStatus('idle');
    setLoading(false);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (loading || loginAnimStatus === 'checking' || loginAnimStatus === 'success') return;

    if (mode === 'register') {
      if (!regFullName.trim() || !username.trim() || password.length < 4) {
        setError(copy.registerRequiredFields);
        return;
      }
      await executeStaffAuthSubmission({
        runRequest: async () => {
          const res = await apiClient.registerStaff({
            username: username.trim(),
            password,
            role: regRole,
            fullName: regFullName.trim(),
            phone: regPhone.trim() || undefined,
            specialty: regSpecialty.trim() || undefined,
            licenseNo: regLicenseNo.trim() || undefined,
          });
          setAuthenticatedStaff(res.user);
          return res.user;
        },
        setAnimStatus: setLoginAnimStatus,
        setLoading,
        setError,
        onSuccess,
        redirectDelayMs: STAFF_SUCCESS_TOTAL_REDIRECT_MS,
        scheduleTimer: (fn, ms) => {
          clearPendingSuccessTimer();
          const tid = window.setTimeout(fn, ms);
          successTimerRef.current = tid;
          return tid;
        },
        treatErrorAsNotFound: false,
        fallbackErrorMsg: copy.invalidCreds,
      });
      return;
    }

    if (!username.trim() || !password) {
      setError(copy.requiredFields);
      return;
    }

    await executeStaffAuthSubmission({
      runRequest: async () => {
        const res = await apiClient.login(username.trim(), password);
        setAuthenticatedStaff(res.user);
        return res.user;
      },
      setAnimStatus: setLoginAnimStatus,
      setLoading,
      setError,
      onSuccess,
      redirectDelayMs: STAFF_SUCCESS_TOTAL_REDIRECT_MS,
      scheduleTimer: (fn, ms) => {
        clearPendingSuccessTimer();
        const tid = window.setTimeout(fn, ms);
        successTimerRef.current = tid;
        return tid;
      },
      treatErrorAsNotFound: true,
      fallbackErrorMsg: copy.invalidCreds,
    });
  };

  const isCheckingOrSuccess = loading || loginAnimStatus === 'checking' || loginAnimStatus === 'success';

  const wrapperClass = asModal
    ? 'fixed inset-0 z-[70] flex items-center justify-center p-2 sm:p-4 bg-[#0F4C81]/35 backdrop-blur-md animate-backdrop-fade'
    : 'animate-view-enter min-h-screen flex items-center justify-center p-2 sm:p-4 bg-gradient-to-br from-[#EAF8F5] via-[#EFF6FF] to-white';

  return (
    <div
      className={wrapperClass}
      onClick={(e) => {
        if (asModal && !isCheckingOrSuccess && e.target === e.currentTarget) handleReturn();
      }}
    >
      {loginAnimStatus === 'success' ? (
        <div
          data-testid="staff-login-success-stage"
          style={{
            boxShadow: '0 28px 72px -18px rgba(15, 23, 42, 0.22)',
          }}
          className="relative w-full max-w-lg sm:max-w-xl min-h-[430px] sm:min-h-[480px] rounded-[36px] border border-slate-200/90 bg-white px-8 py-12 shadow-2xl overflow-hidden flex flex-col items-center justify-center animate-popup-spring"
        >
          {/* Twinkling 4-Point Crisp Vector Sparkles on Pure Flat White */}
          {SUCCESS_TWINKLE_STARS.map((star, idx) => (
            <svg
              key={idx}
              aria-hidden="true"
              viewBox="0 0 24 24"
              fill="none"
              style={{ ['--sdelay' as string]: star.delay }}
              className={`auth-success-star pointer-events-none absolute ${star.className} ${star.sizeClass}`}
            >
              <path
                d="M12 1.5 L14.6 9.4 L22.5 12 L14.6 14.6 L12 22.5 L9.4 14.6 L1.5 12 L9.4 9.4 Z"
                fill={star.color}
              />
            </svg>
          ))}

          {successStage === 'kinetic' ? (
            /* ACT 1: Vibrant Multi-Wave Kinetic Success Celebration on Clean White */
            <div
              className={`relative flex flex-col items-center justify-center w-full ${
                isLeavingKinetic ? 'auth-success-stage-exit' : ''
              }`}
            >
              <div
                aria-hidden="true"
                className="pointer-events-none absolute top-[96px] sm:top-[112px] left-1/2 -translate-x-1/2 -translate-y-1/2 flex items-center justify-center"
              >
                {/* 3 Crisp Staggered Expanding Sonar Rings */}
                <div className="auth-success-shockwave-1 absolute h-44 w-44 sm:h-52 sm:w-52 rounded-full border-2 border-emerald-400/60" />
                <div className="auth-success-shockwave-2 absolute h-44 w-44 sm:h-52 sm:w-52 rounded-full border-2 border-sky-400/55" />
                <div className="auth-success-shockwave-3 absolute h-44 w-44 sm:h-52 sm:w-52 rounded-full border-2 border-teal-300/50" />

                {/* 12 Radial Starburst Firework Rays on Ring Closure */}
                {SUCCESS_FIREWORK_RAYS.map((ray, idx) => (
                  <span
                    key={idx}
                    style={{
                      ['--rdeg' as string]: ray.deg,
                      ['--rdelay' as string]: ray.delay,
                      backgroundColor: ray.color,
                    }}
                    className="auth-success-ray absolute h-6 w-1 rounded-full"
                  />
                ))}

                {/* Rotating Dual Precision Orbital Rings */}
                <svg
                  viewBox="0 0 260 260"
                  fill="none"
                  className="auth-success-orbit absolute h-64 w-64 sm:h-72 sm:w-72"
                >
                  <circle
                    cx="130"
                    cy="130"
                    r="108"
                    stroke="rgba(16, 185, 129, 0.24)"
                    strokeWidth="2"
                    strokeDasharray="10 14"
                  />
                  <circle cx="130" cy="22" r="6" fill="#10b981" />
                  <circle cx="224" cy="184" r="5" fill="#0ea5e9" />
                  <circle cx="36" cy="184" r="4.5" fill="#f59e0b" />
                </svg>

                <svg
                  viewBox="0 0 260 260"
                  fill="none"
                  className="auth-success-orbit-reverse absolute h-56 w-56 sm:h-64 sm:w-64"
                >
                  <circle
                    cx="130"
                    cy="130"
                    r="116"
                    stroke="rgba(14, 165, 233, 0.2)"
                    strokeWidth="1.5"
                    strokeDasharray="4 16"
                  />
                  <circle cx="246" cy="130" r="4" fill="#14b8a6" />
                  <circle cx="14" cy="130" r="4" fill="#38bdf8" />
                </svg>

                {/* 22-Piece Two-Wave Geometric Vector Confetti */}
                <div
                  data-testid="staff-login-confetti-burst"
                  className="absolute inset-0 flex items-center justify-center"
                >
                  {SUCCESS_CONFETTI_PIECES.map((piece, idx) => (
                    <span
                      key={idx}
                      style={{
                        ['--cx' as string]: piece.cx,
                        ['--cy' as string]: piece.cy,
                        ['--crot' as string]: piece.crot,
                        ['--cdelay' as string]: piece.cdelay,
                        backgroundColor: piece.shape === 'ring' ? 'transparent' : piece.color,
                        borderColor: piece.shape === 'ring' ? piece.color : undefined,
                      }}
                      className={`auth-confetti-burst absolute ${
                        piece.shape === 'pill'
                          ? 'h-2.5 w-6 rounded-full'
                          : piece.shape === 'diamond'
                            ? 'h-3.5 w-3.5 rotate-45 rounded-[3px]'
                            : piece.shape === 'ring'
                              ? 'h-4 w-4 rounded-full border-[2.5px]'
                              : 'h-3 w-3 rounded-full'
                      }`}
                    />
                  ))}
                </div>
              </div>

              {/* Giant Hero Kinetic Checkmark + Big Bold "Success" */}
              <div className="relative z-10">
                <KineticAuthStatusBadge
                  status="success"
                  testId="staff-login-success-panel"
                  subtitle=""
                  hero
                />
              </div>
            </div>
          ) : (
            /* ACT 2: Staff Brand Welcome Showcase ("ยินดีต้อนรับสู่บุคลากรของเรา" + Official Dr.Tech.Care Logo) */
            <div
              data-testid="staff-login-welcome-panel"
              className="auth-welcome-stage-in relative flex flex-col items-center justify-center text-center w-full"
            >
              {/* Logo Halo Rings + Celebratory Second Confetti Burst */}
              <div className="relative flex items-center justify-center mb-6">
                <div
                  aria-hidden="true"
                  className="auth-success-shockwave-1 pointer-events-none absolute h-32 w-32 sm:h-36 sm:w-36 rounded-full border-2 border-emerald-400/55"
                />
                <div
                  aria-hidden="true"
                  className="auth-success-shockwave-2 pointer-events-none absolute h-32 w-32 sm:h-36 sm:w-36 rounded-full border-2 border-sky-400/50"
                />

                <div
                  aria-hidden="true"
                  data-testid="staff-welcome-confetti-burst"
                  className="pointer-events-none absolute inset-0 flex items-center justify-center"
                >
                  {WELCOME_CONFETTI_PIECES.map((piece, idx) => (
                    <span
                      key={idx}
                      style={{
                        ['--cx' as string]: piece.cx,
                        ['--cy' as string]: piece.cy,
                        ['--crot' as string]: piece.crot,
                        ['--cdelay' as string]: piece.cdelay,
                        backgroundColor: piece.shape === 'ring' ? 'transparent' : piece.color,
                        borderColor: piece.shape === 'ring' ? piece.color : undefined,
                      }}
                      className={`auth-confetti-burst absolute ${
                        piece.shape === 'pill'
                          ? 'h-2.5 w-6 rounded-full'
                          : piece.shape === 'diamond'
                            ? 'h-3.5 w-3.5 rotate-45 rounded-[3px]'
                            : piece.shape === 'ring'
                              ? 'h-4 w-4 rounded-full border-[2.5px]'
                              : 'h-3 w-3 rounded-full'
                      }`}
                    />
                  ))}
                </div>

                {/* Official Dr.Tech.Care Heart-ECG Logo Emblem */}
                <div
                  data-testid="staff-welcome-project-logo"
                  className="auth-welcome-logo relative overflow-hidden flex h-28 w-28 sm:h-32 sm:w-32 items-center justify-center rounded-[30px] shadow-[0_22px_50px_-10px_rgba(14,165,233,0.48)] ring-4 ring-emerald-50"
                  style={{
                    background: 'linear-gradient(135deg, #0F4C81 0%, #0ea5e9 52%, #10b981 100%)',
                  }}
                >
                  <div
                    aria-hidden="true"
                    className="auth-success-sheen pointer-events-none absolute inset-y-0 -left-1/3 w-1/2 bg-gradient-to-r from-transparent via-white/45 to-transparent"
                  />
                  <svg
                    viewBox="0 0 48 48"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                    aria-hidden="true"
                    className="relative z-10 h-16 w-16 sm:h-18 sm:w-18"
                  >
                    <path
                      d="M24 40 C24 40 7.5 29.8 7.5 17.8 C7.5 12.2 11.9 8 17.2 8 C20.4 8 23.1 9.6 24 12.1 C24.9 9.6 27.6 8 30.8 8 C36.1 8 40.5 12.2 40.5 17.8 C40.5 29.8 24 40 24 40 Z"
                      stroke="#ffffff"
                      strokeWidth="2.6"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                    <path
                      className="standby-intro-ecg-line"
                      d="M10.5 23 H17.5 L20.5 16 L24.5 30 L28 20.5 L30.5 23 H37.5"
                      stroke="#ffffff"
                      strokeWidth="2.6"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                </div>
              </div>

              {/* Project Wordmark */}
              <div className="auth-welcome-text">
                <h2
                  data-testid="staff-welcome-project-name"
                  className="text-4xl sm:text-5xl font-black tracking-tight bg-gradient-to-r from-[#0F4C81] via-[#0284C7] to-[#059669] bg-clip-text text-transparent leading-tight"
                  style={{ fontFamily: "'Prompt', 'Inter', 'IBM Plex Sans Thai', sans-serif" }}
                >
                  Dr.Tech.Care
                </h2>
              </div>

              {/* Center Expanding Gradient Hairline Divider */}
              <div
                aria-hidden="true"
                className="auth-welcome-divider my-4 h-1 w-28 rounded-full bg-gradient-to-r from-[#0F4C81] via-[#0ea5e9] to-[#10b981]"
              />

              {/* Warm Staff Welcome Headline */}
              <p
                data-testid="staff-welcome-headline"
                className="auth-welcome-subtext text-xl sm:text-2xl font-extrabold text-slate-900 tracking-tight"
                style={{ fontFamily: "'Prompt', 'IBM Plex Sans Thai', sans-serif" }}
              >
                ยินดีต้อนรับสู่บุคลากรของเรา
              </p>

              {authenticatedStaff?.fullName && (
                <div className="auth-welcome-subtext mt-3 inline-flex items-center gap-1.5 rounded-full border border-emerald-200 bg-emerald-50 px-4 py-1 text-xs sm:text-sm font-extrabold text-emerald-800">
                  <BadgeCheck className="h-4 w-4 text-emerald-600 shrink-0" />
                  <span>{authenticatedStaff.fullName}</span>
                </div>
              )}

              {/* Sleek Portal Entry Progress Bar */}
              <div
                aria-hidden="true"
                className="mt-7 h-1.5 w-44 sm:w-52 overflow-hidden rounded-full bg-slate-100"
              >
                <div className="auth-welcome-bar-fill h-full w-full rounded-full bg-gradient-to-r from-[#0F4C81] via-[#0ea5e9] to-[#10b981] game-progress-stripes" />
              </div>
            </div>
          )}
        </div>
      ) : (
      <div
        style={{ boxShadow: '0 25px 60px -15px rgba(6, 57, 50, 0.35)' }}
        className="w-full max-w-lg sm:max-w-xl bg-white rounded-3xl border-2 border-emerald-500/25 overflow-hidden shadow-2xl relative max-h-[92dvh] overflow-y-auto animate-popup-spring"
      >
        {/* Top-Right Language Switcher */}
        <div className="sm:absolute sm:top-5 sm:right-5 z-10 flex justify-end px-6 pt-4 sm:p-0">
          <div
            className="inline-flex items-center gap-1 rounded-xl bg-white border border-slate-200/90 px-1.5 py-1 shadow-xs"
            data-testid="staff-login-language-switcher"
          >
            <Globe className="w-3.5 h-3.5 text-emerald-600 ml-1 mr-0.5 shrink-0" />
            {LANG_OPTIONS.map((opt) => {
              const active = lang === opt.code;
              return (
                <button
                  key={opt.code}
                  type="button"
                  onClick={() => setLang(opt.code)}
                  className={`px-2 py-1 rounded-lg text-[11px] font-bold transition cursor-pointer flex items-center gap-1 ${
                    active
                      ? 'bg-emerald-50 text-emerald-800 border border-emerald-300 shadow-xs'
                      : 'text-slate-500 hover:bg-slate-100 hover:text-slate-900'
                  }`}
                >
                  <span className="text-[10px] opacity-80">{opt.prefix}</span>
                  <span>{opt.label}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Header Section */}
        <div className="pt-6 sm:pt-9 pb-3 px-6 sm:px-10 text-center flex flex-col items-center">
          <div
            key={mode}
            className="animate-card-enter w-16 h-16 rounded-2xl bg-gradient-to-tr from-emerald-500 to-teal-400 flex items-center justify-center text-white shadow-xl shadow-emerald-950/20 mb-3.5"
          >
            {mode === 'register' ? <UserPlus className="w-8 h-8" /> : <Shield className="w-8 h-8" />}
          </div>

          <span className="inline-block px-3.5 py-1 rounded-full bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs font-bold tracking-wide mb-2.5">
            {isRegistrationMode ? copy.registrationBadge : copy.badge}
          </span>

          <h1 className="text-2xl sm:text-3xl font-black text-slate-800 tracking-normal">
            {mode === 'register'
              ? copy.registerTitle
              : isRegistrationMode
                ? copy.registrationTitle
                : copy.title}
          </h1>
          <p className="text-sm sm:text-base text-slate-600 mt-2 leading-relaxed max-w-md mx-auto">
            {mode === 'register'
              ? copy.registerSubtitle
              : isRegistrationMode
                ? copy.registrationSubtitle
                : copy.subtitle}
          </p>

          {/* Login / Register Mode Toggle */}
          <div className="mt-4 w-full grid grid-cols-2 gap-1.5 p-1.5 rounded-2xl bg-slate-100 border border-slate-200">
            <button
              type="button"
              data-testid="staff-mode-login"
              onClick={() => switchMode('login')}
              className={`h-11 px-3 rounded-xl text-xs sm:text-sm font-bold transition flex items-center justify-center gap-1.5 whitespace-nowrap cursor-pointer ${
                mode === 'login'
                  ? 'bg-white text-emerald-700 shadow-sm border border-emerald-200'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Shield className="w-4 h-4 shrink-0" />
              <span className="truncate">{copy.tabLogin}</span>
            </button>
            <button
              type="button"
              data-testid="staff-mode-register"
              onClick={() => switchMode('register')}
              className={`h-11 px-3 rounded-xl text-xs sm:text-sm font-bold transition flex items-center justify-center gap-1.5 whitespace-nowrap cursor-pointer ${
                mode === 'register'
                  ? 'bg-white text-emerald-700 shadow-sm border border-emerald-200'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <UserPlus className="w-4 h-4 shrink-0" />
              <span className="truncate">{copy.tabRegister}</span>
            </button>
          </div>
        </div>

        {/* Not Found Animated Panel */}
        {loginAnimStatus === 'not_found' && (
              <div className="mx-6 sm:mx-10 mt-3">
                <KineticAuthStatusBadge
                  status="not_found"
                  id="staff-login-error"
                  testId="staff-login-not-found-panel"
                  heading="ไม่พบบัญชี"
                  subtitle="ไม่พบชื่อผู้ใช้หรือรหัสผ่านนี้ในระบบคลินิก"
                  noAccountLabel="ไม่มีบัญชี"
                  secondaryLabel="สมัครสมาชิก (Admin / PT)"
                  onNoAccountClick={() => {
                    setLoginAnimStatus('idle');
                    setError(null);
                    usernameInputRef.current?.focus();
                  }}
                  onSecondaryClick={() => switchMode('register')}
                />
              </div>
            )}

            {/* Validation / Non-Credential Error Alert */}
            {error && loginAnimStatus !== 'not_found' && (
              <div
                id="staff-login-error"
                data-testid="staff-login-error"
                className="mx-6 sm:mx-10 mt-2 p-3.5 rounded-2xl bg-rose-50 border border-rose-200 text-rose-700 text-sm flex items-center gap-2.5 font-medium animate-toast-in animate-gentle-shake"
              >
                <AlertCircle className="w-5 h-5 shrink-0 text-rose-600" />
                <span>{error}</span>
              </div>
            )}

            {/* Form Body */}
            <form key={mode} onSubmit={handleSubmit} className="p-6 sm:p-10 sm:pt-5 space-y-4 animate-panel-switch">
          {mode === 'register' && (
            <>
              <div>
                <label className="block text-sm font-semibold text-slate-700 mb-1.5">
                  {copy.roleLabel}
                </label>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    data-testid="staff-register-role-pt"
                    onClick={() => setRegRole('pt')}
                    className={`h-11 px-3 rounded-xl border text-xs sm:text-sm font-bold transition cursor-pointer ${
                      regRole === 'pt'
                        ? 'bg-emerald-50 border-emerald-500 text-emerald-800 shadow-xs'
                        : 'bg-slate-50 border-slate-200 text-slate-600 hover:bg-white'
                    }`}
                  >
                    {copy.rolePt}
                  </button>
                  <button
                    type="button"
                    data-testid="staff-register-role-admin"
                    onClick={() => setRegRole('super_admin')}
                    className={`h-11 px-3 rounded-xl border text-xs sm:text-sm font-bold transition cursor-pointer ${
                      regRole === 'super_admin'
                        ? 'bg-emerald-50 border-emerald-500 text-emerald-800 shadow-xs'
                        : 'bg-slate-50 border-slate-200 text-slate-600 hover:bg-white'
                    }`}
                  >
                    {copy.roleAdmin}
                  </button>
                </div>
              </div>

              <div>
                <label
                  htmlFor="staff-register-fullname"
                  className="block text-sm font-semibold text-slate-700 mb-1.5"
                >
                  {copy.fullNameLabel}
                </label>
                <div className="relative flex items-center">
                  <div className="absolute left-3.5 top-1/2 -translate-y-1/2 pointer-events-none text-slate-400 z-10 flex items-center justify-center w-5 h-5">
                    <User className="w-5 h-5" />
                  </div>
                  <input
                    id="staff-register-fullname"
                    data-testid="staff-register-fullname"
                    name="fullName"
                    type="text"
                    required
                    placeholder={copy.fullNamePlaceholder}
                    value={regFullName}
                    onChange={(e) => setRegFullName(e.target.value)}
                    className="w-full h-12 pl-11 pr-4 py-2.5 bg-slate-50 border border-slate-200 rounded-2xl text-base text-slate-800 placeholder-slate-400 focus:bg-white focus:outline-none focus:border-emerald-500 focus:ring-4 focus:ring-emerald-500/15 transition shadow-xs"
                  />
                </div>
              </div>
            </>
          )}

          <div>
            <label
              htmlFor="staff-login-username"
              className="block text-sm sm:text-base font-semibold text-slate-700 mb-2"
            >
              {copy.usernameLabel}
            </label>
            <div className="relative flex items-center">
              <div className="absolute left-3.5 top-1/2 -translate-y-1/2 pointer-events-none text-slate-400 z-10 flex items-center justify-center w-5 h-5">
                <User className="w-5 h-5" />
              </div>
              <input
                ref={usernameInputRef}
                id="staff-login-username"
                data-testid="staff-login-username"
                name="username"
                type="text"
                required
                disabled={isCheckingOrSuccess}
                autoComplete="username"
                placeholder={copy.usernamePlaceholder}
                value={username}
                onChange={(e) => {
                  setUsername(e.target.value);
                  if (loginAnimStatus === 'not_found') {
                    setLoginAnimStatus('idle');
                    setError(null);
                  }
                }}
                className="w-full h-13 pl-11 pr-4 py-3 bg-slate-50 border border-slate-200 rounded-2xl text-base text-slate-800 placeholder-slate-400 focus:bg-white focus:outline-none focus:border-emerald-500 focus:ring-4 focus:ring-emerald-500/15 transition shadow-xs disabled:opacity-60"
              />
            </div>
          </div>

          <div>
            <label
              htmlFor="staff-login-password"
              className="block text-sm sm:text-base font-semibold text-slate-700 mb-2"
            >
              {copy.passwordLabel}
            </label>
            <div className="relative flex items-center">
              <div className="absolute left-3.5 top-1/2 -translate-y-1/2 pointer-events-none text-slate-400 z-10 flex items-center justify-center w-5 h-5">
                <Lock className="w-5 h-5" />
              </div>
              <input
                id="staff-login-password"
                data-testid="staff-login-password"
                name="password"
                type="password"
                required
                disabled={isCheckingOrSuccess}
                minLength={mode === 'register' ? 4 : undefined}
                autoComplete={mode === 'register' ? 'new-password' : 'current-password'}
                placeholder={mode === 'register' ? copy.registerPasswordHint : copy.passwordPlaceholder}
                value={password}
                onChange={(e) => {
                  setPassword(e.target.value);
                  if (loginAnimStatus === 'not_found') {
                    setLoginAnimStatus('idle');
                    setError(null);
                  }
                }}
                className="w-full h-13 pl-11 pr-4 py-3 bg-slate-50 border border-slate-200 rounded-2xl text-base text-slate-800 placeholder-slate-400 focus:bg-white focus:outline-none focus:border-emerald-500 focus:ring-4 focus:ring-emerald-500/15 transition shadow-xs disabled:opacity-60"
              />
            </div>
          </div>

          {mode === 'register' && (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label
                  htmlFor="staff-register-phone"
                  className="block text-xs sm:text-sm font-semibold text-slate-700 mb-1.5"
                >
                  {copy.phoneLabel}
                </label>
                <div className="relative flex items-center">
                  <div className="absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none text-slate-400 z-10">
                    <Phone className="w-4 h-4" />
                  </div>
                  <input
                    id="staff-register-phone"
                    data-testid="staff-register-phone"
                    name="phone"
                    type="tel"
                    inputMode="numeric"
                    pattern="[0-9]*"
                    maxLength={10}
                    placeholder={copy.phonePlaceholder}
                    value={regPhone}
                    onChange={(e) => setRegPhone(e.target.value.replace(/\D/g, '').slice(0, 10))}
                    className="w-full h-11 pl-9 pr-3 bg-slate-50 border border-slate-200 rounded-xl text-sm text-slate-800 placeholder-slate-400 focus:bg-white focus:outline-none focus:border-emerald-500"
                  />
                </div>
              </div>

              <div>
                <label
                  htmlFor="staff-register-specialty"
                  className="block text-xs sm:text-sm font-semibold text-slate-700 mb-1.5"
                >
                  {copy.specialtyLabel}
                </label>
                <div className="relative flex items-center">
                  <div className="absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none text-slate-400 z-10">
                    <Briefcase className="w-4 h-4" />
                  </div>
                  <input
                    id="staff-register-specialty"
                    data-testid="staff-register-specialty"
                    name="specialty"
                    type="text"
                    placeholder={copy.specialtyPlaceholder}
                    value={regSpecialty}
                    onChange={(e) => setRegSpecialty(e.target.value)}
                    className="w-full h-11 pl-9 pr-3 bg-slate-50 border border-slate-200 rounded-xl text-sm text-slate-800 placeholder-slate-400 focus:bg-white focus:outline-none focus:border-emerald-500"
                  />
                </div>
              </div>

              {regRole === 'pt' && (
                <div className="sm:col-span-2">
                  <label
                    htmlFor="staff-register-license"
                    className="block text-xs sm:text-sm font-semibold text-slate-700 mb-1.5"
                  >
                    {copy.licenseLabel}
                  </label>
                  <div className="relative flex items-center">
                    <div className="absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none text-slate-400 z-10">
                      <BadgeCheck className="w-4 h-4" />
                    </div>
                    <input
                      id="staff-register-license"
                      data-testid="staff-register-license"
                      name="licenseNo"
                      type="text"
                      placeholder={copy.licensePlaceholder}
                      value={regLicenseNo}
                      onChange={(e) => setRegLicenseNo(e.target.value)}
                      className="w-full h-11 pl-9 pr-3 bg-slate-50 border border-slate-200 rounded-xl text-sm text-slate-800 placeholder-slate-400 focus:bg-white focus:outline-none focus:border-emerald-500"
                    />
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Automatic Role Notice */}
          {mode === 'login' && (
            <div className="p-4 rounded-2xl bg-emerald-50/90 border border-emerald-200/70 text-emerald-900 text-xs sm:text-sm flex items-start gap-3">
              <Info className="w-5 h-5 shrink-0 mt-0.5 text-emerald-600" />
              <span className="leading-relaxed">{copy.roleNotice}</span>
            </div>
          )}

          {/* Submit Button */}
          <button
            id="staff-login-submit-btn"
            data-testid="staff-login-submit-btn"
            type="submit"
            disabled={isCheckingOrSuccess}
            className="btn-primary-sheen btn-game-3d-emerald w-full h-13 py-3.5 px-6 rounded-2xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-bold text-base sm:text-lg transition flex items-center justify-center gap-2.5 cursor-pointer disabled:opacity-60 disabled:cursor-not-allowed"
          >
            {isCheckingOrSuccess ? (
              <>
                <svg
                  viewBox="0 0 24 24"
                  fill="none"
                  className="w-5 h-5 animate-spin text-white shrink-0"
                  aria-hidden="true"
                >
                  <circle
                    cx="12"
                    cy="12"
                    r="9"
                    stroke="currentColor"
                    strokeWidth="2.8"
                    className="opacity-30"
                  />
                  <path
                    d="M21 12a9 9 0 0 0-9-9"
                    stroke="currentColor"
                    strokeWidth="2.8"
                    strokeLinecap="round"
                  />
                </svg>
                <span>{mode === 'register' ? copy.registering : 'กำลังตรวจสอบบัญชี...'}</span>
              </>
            ) : (
              <>
                {mode === 'register' ? (
                  <UserPlus className="w-5 h-5 shrink-0" />
                ) : (
                  <ArrowRight className="w-5 h-5 shrink-0" />
                )}
                <span>
                  {mode === 'register'
                    ? isRegistrationMode
                      ? copy.registrationRegisterBtn
                      : copy.registerBtn
                    : isRegistrationMode
                      ? copy.registrationLoginBtn
                      : copy.loginBtn}
                </span>
              </>
            )}
          </button>

          {/* Back to Kiosk Button */}
          <button
            id="staff-login-return-kiosk-btn"
            data-testid="staff-login-return-kiosk-btn"
            type="button"
            onClick={handleReturn}
            className="btn-game-3d-white w-full h-12 px-4 rounded-2xl bg-slate-100 hover:bg-slate-200 border border-slate-200 text-slate-700 text-sm sm:text-base font-semibold transition flex items-center justify-center gap-2 cursor-pointer"
          >
            <ArrowLeft className="w-4 h-4 shrink-0" />
            <span>{copy.returnToPatientMain}</span>
          </button>
        </form>

        {/* Bottom Security Notice */}
        <div className="py-3.5 px-8 bg-slate-50 border-t border-slate-100 text-center text-xs text-slate-500 font-medium">
          {copy.securityNotice}
        </div>
      </div>
      )}
    </div>
  );
}
