import React, { useState } from 'react';
import type { DrTechCaresApiClient, UserAccount } from '@drtechcares/api-client';
import { HeartPulse, Lock, LogIn } from 'lucide-react';

interface StaffLoginModalOrPageProps {
  apiClient: DrTechCaresApiClient;
  onSuccess: (user: UserAccount) => void;
}

export function StaffLoginModalOrPage({ apiClient, onSuccess }: StaffLoginModalOrPageProps) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const res = await apiClient.login(username.trim(), password);
      onSuccess(res.user);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'เข้าสู่ระบบไม่สำเร็จ');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-100 px-4">
      <div className="w-full max-w-md rounded-3xl bg-white p-8 shadow-xl border border-slate-200 space-y-6">
        <div className="text-center">
          <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-teal-600 text-white shadow">
            <HeartPulse className="h-8 w-8" />
          </div>
          <h1 className="mt-4 text-2xl font-black text-slate-900">DRTECHCARES Portal Login</h1>
          <p className="mt-1 text-xs text-slate-500">
            ระบบยืนยันตัวตนส่วนกลางแยกสิทธิ์ (Super Admin / นักกายภาพบำบัด / คนไข้)
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="text-xs font-bold text-slate-700">ชื่อผู้ใช้ (Username)</label>
            <input
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="กรอกชื่อผู้ใช้"
              className="mt-1 w-full rounded-xl border border-slate-300 px-4 py-2.5 text-sm"
              required
            />
          </div>
          <div>
            <label className="text-xs font-bold text-slate-700">รหัสผ่าน (Password)</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              className="mt-1 w-full rounded-xl border border-slate-300 px-4 py-2.5 text-sm"
              required
            />
          </div>

          {error && (
            <div className="rounded-xl bg-rose-50 border border-rose-200 px-3 py-2 text-xs font-bold text-rose-700">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full inline-flex items-center justify-center gap-2 rounded-xl bg-teal-600 py-3 text-sm font-bold text-white hover:bg-teal-700 disabled:opacity-50"
          >
            <LogIn className="h-4 w-4" />
            {loading ? 'กำลังตรวจสอบสิทธิ์...' : 'เข้าสู่ระบบ'}
          </button>
        </form>

        <div className="rounded-2xl bg-slate-50 p-3 text-[11px] text-slate-500 flex items-start gap-2">
          <Lock className="h-4 w-4 shrink-0 text-slate-400 mt-0.5" />
          <span>ทุกการเข้าสู่ระบบจะถูกบันทึก IP, Device ID และสิทธิ์การเข้าถึงลงใน Audit Logs</span>
        </div>
      </div>
    </div>
  );
}

