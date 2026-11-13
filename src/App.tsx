import React, { useEffect, useMemo, useState } from 'react';
import { AdminPortalApp } from '@apps/admin';
import { KioskPortalApp } from '@apps/kiosk';
import { PtPortalApp } from '@apps/pt';
import { DrTechCaresApiClient, type UserAccount } from '@drtechcares/api-client';
import { PatientPersonalPortal } from './PatientPersonalPortal';
import { StaffLoginModalOrPage, type StaffAuthIntent } from './StaffLoginModalOrPage';
import {
  KioskBiometricLoginView,
  type KioskLoginSuccessPayload,
} from './components/KioskBiometricLoginView';
import { SmartRehabKioskUI } from './components/SmartRehabKioskUI';

type AppRoute = '/' | '/login' | '/admin' | '/pt' | '/patient';
type KioskViewMode = 'login' | 'rehab' | 'classic';

function resolveRoute(pathname: string): AppRoute {
  if (pathname.startsWith('/admin')) return '/admin';
  if (pathname.startsWith('/pt')) return '/pt';
  if (pathname.startsWith('/patient')) return '/patient';
  if (pathname.startsWith('/login')) return '/login';
  return '/';
}

export function App() {
  const [route, setRoute] = useState<AppRoute>(() =>
    typeof window !== 'undefined' ? resolveRoute(window.location.pathname) : '/',
  );
  const [kioskMode, setKioskMode] = useState<KioskViewMode>('login');
  const [staffModalIntent, setStaffModalIntent] = useState<StaffAuthIntent | null>(null);
  const [authenticatedPatient, setAuthenticatedPatient] = useState<KioskLoginSuccessPayload | null>(null);
  const [currentUser, setCurrentUser] = useState<UserAccount | null>(null);
  const [authChecked, setAuthChecked] = useState(false);

  const apiClient = useMemo(
    () =>
      new DrTechCaresApiClient({
        supabaseUrl: import.meta.env.VITE_SUPABASE_URL,
        supabaseAnonKey: import.meta.env.VITE_SUPABASE_ANON_KEY,
      }),
    [],
  );

  const navigate = (next: AppRoute) => {
    setRoute(next);
    if (typeof window !== 'undefined' && window.location.pathname !== next) {
      window.history.pushState({}, '', next);
    }
  };

  useEffect(() => {
    const onPop = () => setRoute(resolveRoute(window.location.pathname));
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, []);

  useEffect(() => {
    if (route === '/') {
      setAuthChecked(true);
      return;
    }
    apiClient.getMe().then((u) => {
      setCurrentUser(u);
      setAuthChecked(true);
    });
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
          onPatientAuthenticated={(payload) => {
            setAuthenticatedPatient(payload);
            setKioskMode('rehab');
          }}
          onOpenStaffLogin={() => setStaffModalIntent('general')}
          onOpenStaffRegister={() => {
            if (currentUser && (currentUser.role === 'pt' || currentUser.role === 'super_admin')) {
              navigate('/pt');
            } else {
              setStaffModalIntent('kiosk_patient_registration');
            }
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
                navigate('/pt');
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
