import React, { useCallback, useEffect, useState } from 'react';
import type { DrTechCaresApiClient, UserAccount } from '@drtechcares/api-client';
import { useClinicalI18n, type ClinicalLang } from '@drtechcares/clinical-i18n';
import {
  AlertCircle,
  ArrowLeft,
  ArrowRight,
  Globe,
  Info,
  Loader2,
  Lock,
  Shield,
  User,
} from 'lucide-react';

export type StaffAuthIntent = 'general' | 'kiosk_patient_registration';

interface StaffLoginModalOrPageProps {
  apiClient: DrTechCaresApiClient;
  onSuccess: (user: UserAccount) => void;
  onReturnToKiosk?: () => void;
  authIntent?: StaffAuthIntent;
  asModal?: boolean;
}

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
    subtitle: string;
    registrationSubtitle: string;
    usernameLabel: string;
    usernamePlaceholder: string;
    passwordLabel: string;
    passwordPlaceholder: string;
    roleNotice: string;
    loggingIn: string;
    loginBtn: string;
    returnToPatientMain: string;
    securityNotice: string;
    requiredFields: string;
    invalidCreds: string;
  }
> = {
  th: {
    badge: '+ Dr.Tech.Care • Hospital Staff Portal',
    registrationBadge: '👨‍⚕️ เจ้าหน้าที่ลงทะเบียนคนไข้ (STAFF-ASSISTED MODE)',
    title: 'เข้าสู่ระบบบุคลากร Dr.Tech.Care',
    subtitle: 'เข้าถึงระบบบริหารเวชระเบียนคลินิกและตารางกายภาพบำบัด',
    registrationSubtitle: 'กรุณาเข้าสู่ระบบด้วยบัญชี Admin, ผอ.รพ. หรือ นักกายภาพบำบัด (PT)',
    usernameLabel: 'Username (ชื่อผู้ใช้) *',
    usernamePlaceholder: 'เช่น director หรือ pt_somchai',
    passwordLabel: 'Password (รหัสผ่าน) *',
    passwordPlaceholder: '••••••••••••',
    roleNotice: 'ระบบตรวจสอบสิทธิ์อัตโนมัติ: ผอ.รพ. / Super Admin หรือ นักกายภาพบำบัด PT',
    loggingIn: 'กำลังเข้าสู่ระบบ...',
    loginBtn: 'เข้าสู่ระบบบุคลากร',
    returnToPatientMain: 'กลับหน้าหลัก (คนไข้)',
    securityNotice: 'Hospital Clinical Grade Security • HttpOnly Session Cookie • Zero-PII Leak',
    requiredFields: 'กรุณากรอกชื่อผู้ใช้และรหัสผ่านให้ครบถ้วน',
    invalidCreds: 'ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง',
  },
  en: {
    badge: '+ Dr.Tech.Care • Hospital Staff Portal',
    registrationBadge: '👨‍⚕️ Staff-Assisted Patient Registration Mode',
    title: 'Dr.Tech.Care Hospital Staff Login',
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
    returnToPatientMain: 'Back to Main (Patient)',
    securityNotice: 'Hospital Clinical Grade Security • HttpOnly Session Cookie • Zero-PII Leak',
    requiredFields: 'Please fill in both username and password',
    invalidCreds: 'Invalid username or password',
  },
  cn: {
    badge: '+ Dr.Tech.Care • Hospital Staff Portal',
    registrationBadge: '👨‍⚕️ 医护人员协助建档模式',
    title: 'Dr.Tech.Care 医院员工登录',
    subtitle: '访问临床病历管理系统与物理康复治疗日程',
    registrationSubtitle: '请使用管理员、院长或物理康复理疗师 (PT) 账号登录验证',
    usernameLabel: '用户名 (Username) *',
    usernamePlaceholder: '例如 director 或 pt_somchai',
    passwordLabel: '密码 (Password) *',
    passwordPlaceholder: '••••••••••••',
    roleNotice: '自动权限校验：医院院长 / 超级管理员 或 物理康复理疗师 PT',
    loggingIn: '正在登录...',
    loginBtn: '登录医护系统',
    returnToPatientMain: '返回首页 (患者)',
    securityNotice: 'Hospital Clinical Grade Security • HttpOnly Session Cookie • Zero-PII Leak',
    requiredFields: '请填写完整的用户名和密码',
    invalidCreds: '用户名或密码错误',
  },
};

export function StaffLoginModalOrPage({
  apiClient,
  onSuccess,
  onReturnToKiosk,
  authIntent = 'general',
  asModal = false,
}: StaffLoginModalOrPageProps) {
  const { lang, setLang } = useClinicalI18n();
  const copy = STAFF_LOGIN_COPY[lang];
  const isRegistrationMode = authIntent === 'kiosk_patient_registration';

  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleReturn = useCallback(() => {
    if (onReturnToKiosk) {
      onReturnToKiosk();
      return;
    }
    if (typeof window !== 'undefined') {
      window.history.pushState({}, '', '/');
      window.dispatchEvent(new PopStateEvent('popstate'));
    }
  }, [onReturnToKiosk]);

  useEffect(() => {
    if (!asModal) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') handleReturn();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [asModal, handleReturn]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!username.trim() || !password) {
      setError(copy.requiredFields);
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const res = await apiClient.login(username.trim(), password);
      onSuccess(res.user);
    } catch (err) {
      setError(err instanceof Error && err.message ? err.message : copy.invalidCreds);
    } finally {
      setLoading(false);
    }
  };

  const wrapperClass = asModal
    ? 'fixed inset-0 z-50 flex items-center justify-center p-2 sm:p-4 bg-slate-900/60 backdrop-blur-md'
    : 'min-h-screen flex items-center justify-center p-2 sm:p-4 bg-gradient-to-br from-[#0b2724] via-[#133834] to-[#0a1d24]';

  return (
    <div
      className={wrapperClass}
      onClick={(e) => {
        if (asModal && e.target === e.currentTarget) handleReturn();
      }}
    >
      <div
        style={{ boxShadow: '0 25px 60px -15px rgba(6, 57, 50, 0.35)' }}
        className="w-full max-w-lg sm:max-w-xl bg-white rounded-3xl border-2 border-emerald-500/25 overflow-hidden shadow-2xl relative max-h-[92dvh] overflow-y-auto"
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
          <div className="w-16 h-16 rounded-2xl bg-gradient-to-tr from-emerald-500 to-teal-400 flex items-center justify-center text-white shadow-xl shadow-emerald-950/20 mb-3.5">
            <Shield className="w-8 h-8" />
          </div>

          <span className="inline-block px-3.5 py-1 rounded-full bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs font-bold tracking-wide mb-2.5">
            {isRegistrationMode ? copy.registrationBadge : copy.badge}
          </span>

          <h1 className="text-2xl sm:text-3xl font-black text-slate-800 tracking-normal">
            {copy.title}
          </h1>
          <p className="text-sm sm:text-base text-slate-600 mt-2 leading-relaxed max-w-md mx-auto">
            {isRegistrationMode ? copy.registrationSubtitle : copy.subtitle}
          </p>
        </div>

        {/* Error Alert */}
        {error && (
          <div
            id="staff-login-error"
            data-testid="staff-login-error"
            className="mx-6 sm:mx-10 mt-2 p-3.5 rounded-2xl bg-rose-50 border border-rose-200 text-rose-700 text-sm flex items-center gap-2.5 font-medium"
          >
            <AlertCircle className="w-5 h-5 shrink-0 text-rose-600" />
            <span>{error}</span>
          </div>
        )}

        {/* Form Body */}
        <form onSubmit={handleSubmit} className="p-6 sm:p-10 sm:pt-6 space-y-5">
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
                id="staff-login-username"
                data-testid="staff-login-username"
                name="username"
                type="text"
                required
                autoComplete="username"
                placeholder={copy.usernamePlaceholder}
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="w-full h-13 pl-11 pr-4 py-3 bg-slate-50 border border-slate-200 rounded-2xl text-base text-slate-800 placeholder-slate-400 focus:bg-white focus:outline-none focus:border-emerald-500 focus:ring-4 focus:ring-emerald-500/15 transition shadow-xs"
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
                autoComplete="current-password"
                placeholder={copy.passwordPlaceholder}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full h-13 pl-11 pr-4 py-3 bg-slate-50 border border-slate-200 rounded-2xl text-base text-slate-800 placeholder-slate-400 focus:bg-white focus:outline-none focus:border-emerald-500 focus:ring-4 focus:ring-emerald-500/15 transition shadow-xs"
              />
            </div>
          </div>

          {/* Automatic Role Notice */}
          <div className="p-4 rounded-2xl bg-emerald-50/90 border border-emerald-200/70 text-emerald-900 text-xs sm:text-sm flex items-start gap-3">
            <Info className="w-5 h-5 shrink-0 mt-0.5 text-emerald-600" />
            <span className="leading-relaxed">{copy.roleNotice}</span>
          </div>

          {/* Submit Button */}
          <button
            id="staff-login-submit-btn"
            data-testid="staff-login-submit-btn"
            type="submit"
            disabled={loading}
            className="w-full h-13 py-3.5 px-6 rounded-2xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 active:scale-[0.98] text-white font-bold text-base sm:text-lg shadow-lg shadow-emerald-950/20 transition flex items-center justify-center gap-2.5 cursor-pointer disabled:opacity-60 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin text-white shrink-0" />
                <span>{copy.loggingIn}</span>
              </>
            ) : (
              <>
                <ArrowRight className="w-5 h-5 shrink-0" />
                <span>{copy.loginBtn}</span>
              </>
            )}
          </button>

          {/* Back to Kiosk Button */}
          <button
            id="staff-login-return-kiosk-btn"
            data-testid="staff-login-return-kiosk-btn"
            type="button"
            onClick={handleReturn}
            className="w-full h-12 px-4 rounded-2xl bg-slate-100 hover:bg-slate-200 text-slate-700 text-sm sm:text-base font-semibold transition flex items-center justify-center gap-2 active:scale-[0.98] cursor-pointer"
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
    </div>
  );
}
