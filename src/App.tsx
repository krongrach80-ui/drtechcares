import React, { useEffect, useMemo, useState } from 'react';
import { AdminPortalApp } from '@apps/admin';
import { KioskPortalApp, PhysicalCardGame, type PhysicalCardGameSummary } from '@apps/kiosk';
import { PtPortalApp } from '@apps/pt';
import {
  DrTechCaresApiClient,
  resolveTodayPatientSchedule,
  type UserAccount,
} from '@drtechcares/api-client';
import { PatientPersonalPortal } from './PatientPersonalPortal';
import { StaffLoginModalOrPage, type StaffAuthIntent } from './StaffLoginModalOrPage';
import {
  KioskBiometricLoginView,
  type KioskLoginSuccessPayload,
} from './components/KioskBiometricLoginView';
import { RandomHandStartPrompt } from './components/RandomHandStartPrompt';
import { SmartRehabKioskUI } from './components/SmartRehabKioskUI';

type AppRoute = '/' | '/login' | '/admin' | '/pt' | '/patient';
type KioskViewMode = 'login' | 'training' | 'rehab' | 'classic';

function resolveRoute(pathname: string): AppRoute {
  if (pathname.startsWith('/admin')) return '/admin';
  if (pathname.startsWith('/pt')) return '/pt';
  if (pathname.startsWith('/patient')) return '/patient';
  if (pathname.startsWith('/login')) return '/login';
  if (pathname.startsWith('/kiosk')) return '/';
  return '/';
}

export function App() {
  const [route, setRoute] = useState<AppRoute>(() =>
    typeof window !== 'undefined' ? resolveRoute(window.location.pathname) : '/',
  );
  const [kioskMode, setKioskMode] = useState<KioskViewMode>('login');
  const [staffModalIntent, setStaffModalIntent] = useState<StaffAuthIntent | null>(null);
  const [pendingAutoOpenEnroll, setPendingAutoOpenEnroll] = useState(false);
  const [authenticatedPatient, setAuthenticatedPatient] = useState<KioskLoginSuccessPayload | null>(null);
  const [allowRestDayPractice, setAllowRestDayPractice] = useState(false);
  const [startedCardGame, setStartedCardGame] = useState(false);

  useEffect(() => {
    setAllowRestDayPractice(false);
    setStartedCardGame(false);
  }, [authenticatedPatient]);

  const apiClient = useMemo(
    () =>
      new DrTechCaresApiClient({
        supabaseUrl: import.meta.env.VITE_SUPABASE_URL,
        supabaseAnonKey: import.meta.env.VITE_SUPABASE_ANON_KEY,
      }),
    [],
  );

  const [currentUser, setCurrentUser] = useState<UserAccount | null>(() => apiClient.getStoredUser());
  const [authChecked, setAuthChecked] = useState(false);

  const navigate = (next: AppRoute) => {
    setRoute(next);
    if (typeof window !== 'undefined' && window.location.pathname !== next) {
      window.history.pushState({}, '', next);
    }
  };

  useEffect(() => {
    const onPop = () => setRoute(resolveRoute(window.location.pathname));
    const onAuthExpired = () => {
      setCurrentUser(null);
      setAuthChecked(true);
    };
    window.addEventListener('popstate', onPop);
    window.addEventListener('drtech:auth-expired', onAuthExpired);
    return () => {
      window.removeEventListener('popstate', onPop);
      window.removeEventListener('drtech:auth-expired', onAuthExpired);
    };
  }, []);

  useEffect(() => {
    if (route === '/') {
      setAuthChecked(true);
      return;
    }
    let cancelled = false;
    apiClient.getMe().then((u) => {
      if (!cancelled) {
        setCurrentUser(u);
        setAuthChecked(true);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [route, apiClient]);

  // Locked Kiosk Mode (/) — Exact Kiosk Biometric Login UI by default, transitioning into Rehab / Classic Kiosk
  if (route === '/') {
    if (kioskMode === 'classic') {
      return (
        <div className="min-h-screen bg-slate-950 p-4">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <button
              type="button"
              onClick={() => {
                setAuthenticatedPatient(null);
                setKioskMode('login');
              }}
              className="rounded-xl border border-emerald-500/40 bg-emerald-500/15 px-4 py-1.5 text-xs font-bold text-emerald-300 hover:bg-emerald-500/25 cursor-pointer"
            >
              ← กลับหน้าสแกนใบหน้า (Kiosk Login)
            </button>
            <button
              type="button"
              onClick={() => setKioskMode('rehab')}
              className="rounded-xl border border-teal-500/40 bg-teal-500/10 px-4 py-1.5 text-xs font-bold text-teal-300 hover:bg-teal-500/20 cursor-pointer"
            >
              หน้าวิเคราะห์ชีวกลศาสตร์ (Smart Rehab UI)
            </button>
          </div>
          <KioskPortalApp apiClient={apiClient} />
        </div>
      );
    }

    if (kioskMode === 'training' && authenticatedPatient) {
      const todaySchedule = resolveTodayPatientSchedule({
        rehabSchedules: authenticatedPatient.rehabSchedules,
        todayRoutine: authenticatedPatient.todayRoutine,
        nextAppointmentDate: authenticatedPatient.patient.nextAppointmentDate,
        serverTodayExercises: authenticatedPatient.todayExercises,
      });
      const hasAppointmentToday =
        authenticatedPatient.hasAppointmentToday ?? todaySchedule.hasAppointmentToday;
      const todayExercises =
        authenticatedPatient.todayExercises && authenticatedPatient.todayExercises.length > 0
          ? authenticatedPatient.todayExercises
          : todaySchedule.todayExercises;

      const handleExitToLogin = () => {
        setAllowRestDayPractice(false);
        setStartedCardGame(false);
        setAuthenticatedPatient(null);
        setKioskMode('login');
      };

      const saveTrainingSession = async (summary: PhysicalCardGameSummary) => {
        const accuracy = Math.round((summary.correct / Math.max(1, summary.total)) * 100);
        try {
          await apiClient.recordRehabSession({
            patientId: authenticatedPatient.patient.id,
            activityType: 'minigame',
            exerciseId: 'cognitive-card-game',
            exerciseName: 'มินิเกมฝึกสมอง (การ์ดกายภาพ)',
            completedSets: 1,
            completedReps: summary.total,
            accuracyPercent: accuracy,
            maxRomDeg: 0,
            painStopTriggered: false,
            aiFeedback: `Cognitive Accuracy ${accuracy}% (${summary.correct}/${summary.total})`,
            kinematicsReplay: [],
            minigameScore: summary,
          });
        } catch {
          // Allow patient summary screen to remain responsive even if offline
        }
      };

      if (!hasAppointmentToday && !allowRestDayPractice) {
        return (
          <div className="min-h-screen w-full bg-slate-950 flex flex-col">
            <div className="bg-[#042924] px-4 py-2.5 text-white flex flex-wrap items-center justify-between gap-2 text-xs border-b border-emerald-500/20">
              <div className="flex items-center gap-2 font-bold">
                <span className="px-2.5 py-0.5 rounded-full bg-amber-500/25 text-amber-200">
                  Rest Day • วันพักผ่อน
                </span>
                <span>
                  ผู้ป่วย: {authenticatedPatient.patient.fullName} (HN: {authenticatedPatient.patient.hn})
                </span>
              </div>
              <button
                type="button"
                onClick={handleExitToLogin}
                className="rounded-xl border border-emerald-400/50 bg-emerald-600/25 px-3 py-1.5 font-bold text-emerald-100 hover:bg-emerald-600/40 cursor-pointer"
              >
                ← กลับหน้าสแกนใบหน้า (Kiosk Login)
              </button>
            </div>
            <div className="flex-1 flex items-center justify-center p-4 sm:p-8">
              <div className="w-full max-w-xl rounded-3xl bg-white border-2 border-amber-300 shadow-2xl p-6 sm:p-8 text-center">
                <div className="mx-auto w-16 h-16 rounded-full bg-amber-100 text-amber-700 flex items-center justify-center text-3xl mb-4">
                  🌿
                </div>
                <span className="inline-block px-3 py-1 rounded-full bg-amber-50 border border-amber-200 text-amber-800 text-xs font-extrabold mb-3">
                  ตรวจสอบตารางนัดหมายวันนี้ ({todaySchedule.todayIso})
                </span>
                <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900">
                  วันนี้คือวันพักผ่อนของคุณ ( {authenticatedPatient.patient.fullName} )
                </h1>
                <p className="text-sm text-slate-600 mt-3 leading-relaxed">
                  ระบบตรวจสอบตารางนัดหมายแล้ว วันนี้ ({todaySchedule.todayIso}) ไม่มีรายการนัดทำกายภาพบำบัดของคุณ
                  <br />
                  <span className="font-semibold text-emerald-700">
                    นัดถัดไป: {authenticatedPatient.patient.nextAppointmentDate || '-'}
                  </span>
                </p>
                <div className="mt-6 flex flex-col gap-3">
                  <button
                    type="button"
                    onClick={handleExitToLogin}
                    className="w-full py-3.5 px-5 rounded-2xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-extrabold text-sm shadow-lg cursor-pointer"
                  >
                    ← กลับหน้าสแกนใบหน้า (Kiosk Login)
                  </button>
                  <button
                    type="button"
                    onClick={() => setAllowRestDayPractice(true)}
                    className="w-full py-2.5 px-4 rounded-xl bg-slate-100 hover:bg-slate-200 border border-slate-300 text-slate-700 text-xs font-bold cursor-pointer"
                  >
                    ฝึกทบทวนเบาๆ ด้วยตัวเอง (ไม่บังคับ)
                  </button>
                </div>
              </div>
            </div>
          </div>
        );
      }

      if (hasAppointmentToday && !startedCardGame) {
        return (
          <div className="min-h-screen w-full bg-slate-950 flex flex-col">
            <div className="bg-[#042924] px-4 py-2.5 text-white flex flex-wrap items-center justify-between gap-2 text-xs border-b border-emerald-500/20">
              <div className="flex items-center gap-2 font-bold">
                <span className="px-2.5 py-0.5 rounded-full bg-emerald-500/25 text-emerald-300">
                  Scheduled Today • มีนัดกายภาพวันนี้
                </span>
                <span>
                  ผู้ป่วย: {authenticatedPatient.patient.fullName} (HN: {authenticatedPatient.patient.hn})
                </span>
              </div>
              <button
                type="button"
                onClick={handleExitToLogin}
                className="rounded-xl border border-emerald-400/50 bg-emerald-600/25 px-3 py-1.5 font-bold text-emerald-100 hover:bg-emerald-600/40 cursor-pointer"
              >
                ← กลับหน้าสแกนใบหน้า (Kiosk Login)
              </button>
            </div>
            <div className="flex-1 flex items-center justify-center p-4 sm:p-8">
              <div className="w-full max-w-2xl rounded-3xl bg-white border-2 border-emerald-400 shadow-2xl p-6 sm:p-8 text-center">
                <div className="mx-auto w-16 h-16 rounded-full bg-emerald-100 text-emerald-700 flex items-center justify-center text-3xl mb-4">
                  📅
                </div>
                <span className="inline-block px-3 py-1 rounded-full bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs font-extrabold mb-3">
                  ตรวจสอบตารางนัดหมายวันนี้ ({todaySchedule.todayIso}) • มีนัด {todayExercises.length} รายการ
                </span>
                <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900">
                  รายการกายภาพบำบัดวันนี้ของคุณ ( {authenticatedPatient.patient.fullName} )
                </h1>
                <p className="text-xs text-slate-500 mt-1.5 font-mono">
                  HN: {authenticatedPatient.patient.hn} • แผนฝึกกายภาพบำบัดประจำวันที่ {todaySchedule.todayIso}
                </p>

                <div className="mt-5 space-y-2.5 text-left max-h-72 overflow-y-auto pr-1">
                  {todayExercises.map((item, idx) => (
                    <div
                      key={item.id}
                      className="rounded-2xl border border-emerald-200 bg-emerald-50/70 px-4 py-3 flex items-center justify-between gap-3"
                    >
                      <div className="flex items-start gap-3 min-w-0">
                        <span className="shrink-0 w-7 h-7 rounded-full bg-emerald-600 text-white text-xs font-black flex items-center justify-center mt-0.5">
                          {idx + 1}
                        </span>
                        <div className="min-w-0">
                          <p className="text-sm font-extrabold text-slate-900">{item.nameTh}</p>
                          <p className="text-[11px] text-slate-500">{item.nameEn}</p>
                          <div className="flex flex-wrap items-center gap-1.5 mt-1">
                            <span className="inline-block px-2 py-0.5 rounded-md bg-teal-50 text-teal-800 border border-teal-200 text-[11px] font-semibold">
                              {item.targetJoint}
                            </span>
                            <span className="inline-block px-2 py-0.5 rounded-md bg-white text-slate-600 border border-slate-200 text-[11px] font-semibold">
                              ค้าง {item.holdSec} วินาที
                            </span>
                            {item.status === 'COMPLETED' && (
                              <span className="inline-block px-2 py-0.5 rounded-md bg-sky-50 text-sky-700 border border-sky-200 text-[11px] font-bold">
                                ทำครบแล้ว
                              </span>
                            )}
                          </div>
                        </div>
                      </div>
                      <span className="shrink-0 rounded-xl bg-white border border-emerald-300 px-3 py-1.5 text-xs font-extrabold text-emerald-700 shadow-sm">
                        {item.sets} เซต × {item.reps} ครั้ง
                      </span>
                    </div>
                  ))}
                </div>

                <RandomHandStartPrompt onConfirmStart={() => setStartedCardGame(true)} />

                <div className="mt-4 flex flex-col gap-2">
                  <button
                    type="button"
                    onClick={() => setKioskMode('rehab')}
                    className="w-full py-2.5 px-4 rounded-xl bg-sky-50 hover:bg-sky-100 border border-sky-200 text-sky-900 text-xs font-extrabold cursor-pointer"
                  >
                    เปิดหน้าจอวิเคราะห์ชีวกลศาสตร์ (Smart Rehab Telemetry)
                  </button>
                  <button
                    type="button"
                    onClick={handleExitToLogin}
                    className="w-full py-2.5 px-4 rounded-xl bg-slate-100 hover:bg-slate-200 border border-slate-300 text-slate-700 text-xs font-bold cursor-pointer"
                  >
                    ← กลับหน้าสแกนใบหน้า (Kiosk Login)
                  </button>
                </div>
              </div>
            </div>
          </div>
        );
      }

      return (
        <div className="min-h-screen w-full bg-slate-950 flex flex-col">
          <div className="bg-[#042924] px-4 py-2.5 text-white flex flex-wrap items-center justify-between gap-2 text-xs border-b border-emerald-500/20">
            <div className="flex items-center gap-2 font-bold">
              <span className="px-2.5 py-0.5 rounded-full bg-emerald-500/25 text-emerald-300">
                AI Kiosk Training
              </span>
              <span>
                ผู้ป่วย: {authenticatedPatient.patient.fullName} (HN: {authenticatedPatient.patient.hn})
              </span>
            </div>
            <div className="flex items-center gap-2">
              {hasAppointmentToday && (
                <button
                  type="button"
                  onClick={() => setStartedCardGame(false)}
                  className="rounded-xl border border-teal-400/40 bg-teal-500/15 px-3 py-1.5 font-bold text-teal-200 hover:bg-teal-500/30 cursor-pointer"
                >
                  ← กลับหน้ารายการกายภาพวันนี้
                </button>
              )}
              <button
                type="button"
                onClick={() => setKioskMode('rehab')}
                className="rounded-xl border border-sky-400/40 bg-sky-500/15 px-3 py-1.5 font-bold text-sky-200 hover:bg-sky-500/30 cursor-pointer"
              >
                Smart Rehab Telemetry
              </button>
              <button
                type="button"
                onClick={handleExitToLogin}
                className="rounded-xl border border-emerald-400/50 bg-emerald-600/25 px-3 py-1.5 font-bold text-emerald-100 hover:bg-emerald-600/40 cursor-pointer"
              >
                ← กลับหน้าสแกนใบหน้า (Kiosk Login)
              </button>
            </div>
          </div>
          <div className="flex-1 flex flex-col gap-4 p-3 sm:p-5">
            {todayExercises.length > 0 ? (
              <div className="rounded-2xl border border-emerald-500/30 bg-slate-900/90 p-4 text-white shadow-lg">
                <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
                  <div className="flex items-center gap-2">
                    <span className="text-base">📅</span>
                    <h2 className="text-sm sm:text-base font-extrabold text-emerald-300">
                      รายการท่าทางที่ต้องทำของวันนี้ ({todaySchedule.todayIso}) — {todayExercises.length} รายการ
                    </h2>
                  </div>
                  <button
                    type="button"
                    onClick={() => setKioskMode('rehab')}
                    className="rounded-xl bg-sky-500 hover:bg-sky-400 px-3 py-1.5 text-xs font-extrabold text-slate-950 cursor-pointer"
                  >
                    เริ่มฝึกกายภาพผ่านกล้อง AI (Smart Rehab Telemetry) ➔
                  </button>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2.5">
                  {todayExercises.map((item) => (
                    <div
                      key={item.id}
                      className="rounded-xl border border-emerald-500/25 bg-slate-950/70 p-3 flex items-center justify-between gap-3"
                    >
                      <div className="min-w-0">
                        <p className="text-xs sm:text-sm font-bold text-white truncate">{item.nameTh}</p>
                        <p className="text-[11px] text-emerald-300/90 mt-0.5">{item.targetJoint}</p>
                      </div>
                      <span className="shrink-0 rounded-lg bg-emerald-500/20 border border-emerald-400/30 px-2.5 py-1 text-xs font-extrabold text-emerald-200">
                        {item.sets} เซต × {item.reps} ครั้ง
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="rounded-2xl border border-amber-400/40 bg-amber-500/10 px-4 py-3 text-amber-200 text-xs flex flex-wrap items-center justify-between gap-2">
                <span>
                  🌿 วันนี้คือวันพักผ่อนของคุณ ( {authenticatedPatient.patient.fullName} ) — โหมดฝึกทบทวนด้วยตัวเอง (ไม่มีนัดบังคับวันนี้)
                </span>
                <button
                  type="button"
                  onClick={handleExitToLogin}
                  className="underline font-bold text-amber-100 cursor-pointer"
                >
                  กลับหน้าสแกนใบหน้า
                </button>
              </div>
            )}
            <PhysicalCardGame
              patientName={`${authenticatedPatient.patient.fullName} (${authenticatedPatient.patient.hn})`}
              onSessionFinished={saveTrainingSession}
              onComplete={handleExitToLogin}
            />
          </div>
        </div>
      );
    }

    if (kioskMode === 'rehab') {
      return (
        <div className="relative min-h-screen">
          <div className="bg-[#042924] px-4 py-2 text-white flex flex-wrap items-center justify-between gap-2 text-xs">
            <div className="flex items-center gap-2 font-bold">
              <span className="px-2 py-0.5 rounded-full bg-emerald-500/25 text-emerald-300">AI Kiosk Session</span>
              {authenticatedPatient ? (
                <span>
                  ผู้ป่วย: {authenticatedPatient.patient.fullName} (HN: {authenticatedPatient.patient.hn})
                </span>
              ) : (
                <span>โหมดวิเคราะห์ชีวกลศาสตร์กายภาพบำบัด</span>
              )}
            </div>
            <button
              type="button"
              onClick={() => {
                setAuthenticatedPatient(null);
                setKioskMode('login');
              }}
              className="rounded-xl border border-emerald-400/50 bg-emerald-600/25 px-3 py-1 font-bold text-emerald-100 hover:bg-emerald-600/40 cursor-pointer"
            >
              ← กลับหน้าสแกนใบหน้า (Kiosk Login)
            </button>
          </div>
          <SmartRehabKioskUI
            onOpenStaffPortal={() => navigate('/login')}
            onOpenLegacyKiosk={() => setKioskMode('classic')}
          />
        </div>
      );
    }

    return (
      <>
        <KioskBiometricLoginView
          apiClient={apiClient}
          kioskId="kiosk-01"
          staffModalOpen={staffModalIntent !== null}
          onPatientAuthenticated={(payload) => {
            setAuthenticatedPatient(payload);
            setKioskMode('training');
          }}
          onOpenStaffLogin={() => setStaffModalIntent('general')}
          onOpenStaffRegister={() => {
            setStaffModalIntent('kiosk_patient_registration');
          }}
          onOpenTelemetrySimulator={() => setKioskMode('rehab')}
        />
        {staffModalIntent !== null && (
          <StaffLoginModalOrPage
            apiClient={apiClient}
            asModal
            authIntent={staffModalIntent}
            onReturnToKiosk={() => setStaffModalIntent(null)}
            onSuccess={(user) => {
              const intent = staffModalIntent;
              setStaffModalIntent(null);
              setCurrentUser(user);
              if (intent === 'kiosk_patient_registration' && (user.role === 'pt' || user.role === 'super_admin')) {
                setPendingAutoOpenEnroll(true);
                navigate(user.role === 'super_admin' ? '/admin' : '/pt');
              } else if (user.role === 'super_admin') {
                navigate('/admin');
              } else if (user.role === 'pt') {
                navigate('/pt');
              } else {
                navigate('/patient');
              }
            }}
          />
        )}
      </>
    );
  }

  if (!authChecked) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-100 text-sm font-bold text-slate-600">
        กำลังตรวจสอบสิทธิ์การเข้าใช้งาน...
      </div>
    );
  }

  // Require login for /login, /admin, /pt, /patient
  if (!currentUser || route === '/login') {
    return (
      <StaffLoginModalOrPage
        apiClient={apiClient}
        onReturnToKiosk={() => navigate('/')}
        onSuccess={(user) => {
          setCurrentUser(user);
          if (user.role === 'super_admin') navigate('/admin');
          else if (user.role === 'pt') navigate('/pt');
          else navigate('/patient');
        }}
      />
    );
  }

  const handleLogout = async () => {
    await apiClient.logout();
    setCurrentUser(null);
    navigate('/login');
  };

  // Strict Role Routing Guard
  if (route === '/admin') {
    if (currentUser.role !== 'super_admin') {
      return (
        <div className="min-h-screen flex flex-col items-center justify-center bg-rose-50 p-6 text-center">
          <h1 className="text-3xl font-black text-rose-900">403 Forbidden</h1>
          <p className="mt-2 text-sm text-rose-700">เฉพาะผู้อำนวยการโรงพยาบาล (Super Admin) เท่านั้นที่เข้าถึงหน้านี้ได้</p>
          <button
            onClick={() => navigate(currentUser.role === 'pt' ? '/pt' : '/patient')}
            className="mt-4 rounded-xl bg-slate-900 px-4 py-2 text-xs font-bold text-white"
          >
            กลับพอร์ทัลของคุณ
          </button>
        </div>
      );
    }
    return (
      <AdminPortalApp
        apiClient={apiClient}
        currentUser={currentUser}
        onLogout={handleLogout}
        onReturnToKiosk={() => navigate('/')}
        autoOpenEnroll={pendingAutoOpenEnroll}
        onAutoOpenEnrollConsumed={() => setPendingAutoOpenEnroll(false)}
      />
    );
  }

  if (route === '/pt') {
    if (currentUser.role !== 'pt' && currentUser.role !== 'super_admin') {
      return (
        <div className="min-h-screen flex flex-col items-center justify-center bg-rose-50 p-6 text-center">
          <h1 className="text-3xl font-black text-rose-900">403 Forbidden</h1>
          <p className="mt-2 text-sm text-rose-700">ไม่มีสิทธิ์เข้าถึงพอร์ทัลนักกายภาพบำบัด</p>
        </div>
      );
    }
    return (
      <PtPortalApp
        apiClient={apiClient}
        currentUser={currentUser}
        onLogout={handleLogout}
        onReturnToKiosk={() => navigate('/')}
        autoOpenEnroll={pendingAutoOpenEnroll}
        onAutoOpenEnrollConsumed={() => setPendingAutoOpenEnroll(false)}
      />
    );
  }

  return (
    <div className="min-h-screen bg-slate-100">
      <PatientPersonalPortal apiClient={apiClient} currentUser={currentUser} onLogout={handleLogout} />
    </div>
  );
}
export default App;
