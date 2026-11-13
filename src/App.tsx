import React, { useEffect, useMemo, useState } from 'react';
import { AdminPortalApp } from '@apps/admin';
import { KioskPortalApp } from '@apps/kiosk';
import { PtPortalApp } from '@apps/pt';
import { DrTechCaresApiClient, type UserAccount } from '@drtechcares/api-client';
import { PatientPersonalPortal } from './PatientPersonalPortal';
import { StaffLoginModalOrPage } from './StaffLoginModalOrPage';
import { SmartRehabKioskUI } from './components/SmartRehabKioskUI';

type AppRoute = '/' | '/login' | '/admin' | '/pt' | '/patient';

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
  const [useClassicKiosk, setUseClassicKiosk] = useState(false);
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

  // Locked Kiosk Mode (/) — Clinical Zen Bio-Tech UI by default, with optional classic kiosk toggle
  if (route === '/') {
    if (useClassicKiosk) {
      return (
        <div className="min-h-screen bg-slate-950 p-4">
          <div className="mb-3 flex justify-end">
            <button
              type="button"
              onClick={() => setUseClassicKiosk(false)}
              className="rounded-xl border border-teal-500/40 bg-teal-500/10 px-4 py-1.5 text-xs font-bold text-teal-300 hover:bg-teal-500/20"
            >
              กลับหน้า Clinical Zen Smart Kiosk UI
            </button>
          </div>
          <KioskPortalApp apiClient={apiClient} />
        </div>
      );
    }
    return (
      <SmartRehabKioskUI
        onOpenStaffPortal={() => navigate('/login')}
        onOpenLegacyKiosk={() => setUseClassicKiosk(true)}
      />
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
      <div className="min-h-screen bg-slate-100 p-4">
        <AdminPortalApp apiClient={apiClient} currentUser={currentUser} onLogout={handleLogout} />
      </div>
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
      <div className="min-h-screen bg-slate-100 p-4">
        <PtPortalApp apiClient={apiClient} currentUser={currentUser} onLogout={handleLogout} />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-100">
      <PatientPersonalPortal apiClient={apiClient} currentUser={currentUser} onLogout={handleLogout} />
    </div>
  );
}
export default App;
