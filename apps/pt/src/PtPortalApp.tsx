import React, { useEffect, useState } from 'react';
import type {
  DrTechCaresApiClient,
  ManagedExerciseRecord,
  ModpackRecord,
  PatientRecord,
  UserAccount,
} from '@drtechcares/api-client';
import {
  Activity,
  CalendarCheck2,
  FileText,
  KeyRound,
  Lock,
  LogOut,
  Plus,
  ShieldAlert,
  Trash2,
  TrendingUp,
  User,
  UserPlus,
} from 'lucide-react';
import { CoursePrescriptionView } from './CoursePrescriptionView';
import { PatientProgressRealtimeView } from './PatientProgressRealtimeView';
import { SmartEnrollmentView } from './SmartEnrollmentView';

interface PtPortalAppProps {
  apiClient: DrTechCaresApiClient;
  currentUser: UserAccount;
  onLogout: () => void;
}

type PtTab = 'patients' | 'workspace' | 'notes' | 'results' | 'library' | 'profile' | 'audit_403';

export function PtPortalApp({ apiClient, currentUser, onLogout }: PtPortalAppProps) {
  const [tab, setTab] = useState<PtTab>('patients');
  const [patients, setPatients] = useState<PatientRecord[]>([]);
  const [users, setUsers] = useState<UserAccount[]>([]);
  const [exercises, setExercises] = useState<ManagedExerciseRecord[]>([]);
  const [modpacks, setModpacks] = useState<ModpackRecord[]>([]);
  const [aiPoseEnabled, setAiPoseEnabled] = useState(false);
  const [audit403Error, setAudit403Error] = useState<string | null>(null);
  const [newModpackName, setNewModpackName] = useState('');
  const [profileName, setProfileName] = useState(currentUser.fullName);
  const [profilePhone, setProfilePhone] = useState(currentUser.phone);
  const [profileMsg, setProfileMsg] = useState<string | null>(null);

  const loadAll = async () => {
    const [p, u, ex, mp, cfg] = await Promise.all([
      apiClient.listPatients(),
      apiClient.listUsers(),
      apiClient.listExercises(),
      apiClient.listModpacks(),
      apiClient.getSystemSettings(),
    ]);
    setPatients(p);
    setUsers(u);
    setExercises(ex);
    setModpacks(mp);
    setAiPoseEnabled(cfg.settings.aiPoseDetectionEnabled);
  };

  useEffect(() => {
    loadAll();
  }, [apiClient]);

  useEffect(() => {
    if (tab === 'audit_403') {
      apiClient
        .listSecurityAudit()
        .then(() => setAudit403Error(null))
        .catch((err) => {
          setAudit403Error(
            err instanceof Error
              ? `403 Forbidden: ${err.message}`
              : '403 Forbidden: ปิดกั้นการเข้าถึง 100%',
          );
        });
    }
  }, [tab, apiClient]);

  const ptColleagues = users.filter((u) => u.role === 'pt');
  const patientUsers = users.filter((u) => u.role === 'patient');

  return (
    <div className="mx-auto max-w-7xl space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4 rounded-3xl bg-white p-5 shadow-sm border border-slate-200">
        <div>
          <span className="inline-block rounded-full bg-teal-50 px-3 py-1 text-xs font-bold text-teal-700">
            PHYSIOTHERAPIST PORTAL (นักกายภาพบำบัด)
          </span>
          <h1 className="mt-1 text-xl font-extrabold text-slate-900">{currentUser.fullName}</h1>
          <p className="text-xs text-slate-500">
            จัดการเฉพาะคนไข้ในความดูแล / เคสที่รับฝากชั่วคราว ({patients.length} เคส)
          </p>
        </div>
        <button
          onClick={onLogout}
          className="inline-flex items-center gap-1.5 rounded-xl border border-slate-300 px-4 py-2 text-xs font-bold text-slate-700 hover:bg-slate-50"
        >
          <LogOut className="h-4 w-4" /> ออกจากระบบ
        </button>
      </div>

      {/* PT Navigation Tabs */}
      <div className="flex flex-wrap gap-2 rounded-2xl bg-white p-2 shadow-sm border border-slate-200">
        {[
          { id: 'patients', label: 'จัดการคนไข้ & สแกนหน้า 10 เฟรม', icon: UserPlus },
          { id: 'workspace', label: 'ตารางนัด & [ฝากเคสชั่วคราว]', icon: CalendarCheck2 },
          { id: 'notes', label: 'เวชระเบียน (Append-Only)', icon: FileText },
          { id: 'results', label: 'ผลการรักษา & รายงาน', icon: TrendingUp },
          { id: 'library', label: 'คลังท่าฝึก & มินิเกมของฉัน', icon: Activity },
          { id: 'profile', label: 'ข้อมูลส่วนตัว PT', icon: User },
          { id: 'audit_403', label: 'ระบบความปลอดภัย (Audit)', icon: ShieldAlert },
        ].map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              onClick={() => setTab(item.id as PtTab)}
              className={`inline-flex items-center gap-1.5 rounded-xl px-3.5 py-2 text-xs font-bold transition ${
                tab === item.id ? 'bg-teal-600 text-white shadow' : 'text-slate-600 hover:bg-slate-100'
              }`}
            >
              <Icon className="h-4 w-4" /> {item.label}
            </button>
          );
        })}
      </div>

      {tab === 'patients' && (
        <div className="space-y-6">
          <SmartEnrollmentView
            apiClient={apiClient}
            currentUserRole="pt"
            onEnrolled={() => {
              loadAll();
              setTab('workspace');
            }}
          />

          <div className="rounded-3xl bg-white p-6 shadow-md border border-slate-200">
            <h3 className="text-base font-bold text-slate-900 mb-3">
              รายชื่อคนไข้ที่รับผิดชอบ (รีเซ็ตรหัสผ่าน / ลบเฉพาะคนไข้ที่ดูแล)
            </h3>
            <div className="space-y-2">
              {patients.length === 0 && <p className="text-xs text-slate-500">ยังไม่มีคนไข้ในความดูแล</p>}
              {patients.map((p) => {
                const account = patientUsers.find((u) => u.patientId === p.id);
                const isOwner = p.assignedPtId === currentUser.id;
                return (
                  <div
                    key={p.id}
                    className="flex flex-wrap items-center justify-between gap-2 rounded-2xl border border-slate-200 p-3 text-sm"
                  >
                    <div>
                      <span className="font-bold text-slate-900">{p.fullName}</span>{' '}
                      <span className="text-xs text-slate-500">({p.hn}) • {p.diagnosis}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      {account && (
                        <button
                          onClick={async () => {
                            await apiClient.resetUserPassword(account.id, 'Patient@2026');
                            alert(`รีเซ็ตรหัสผ่านของ ${p.fullName} เป็น Patient@2026 แล้ว`);
                          }}
                          className="inline-flex items-center gap-1 rounded-lg bg-slate-100 px-3 py-1.5 text-xs font-bold text-slate-700 hover:bg-slate-200"
                        >
                          <KeyRound className="h-3.5 w-3.5" /> รีเซ็ตรหัสผ่าน
                        </button>
                      )}
                      {isOwner && (
                        <button
                          onClick={() => apiClient.deletePatientCascading(p.id).then(loadAll)}
                          className="inline-flex items-center gap-1 rounded-lg bg-rose-50 px-3 py-1.5 text-xs font-bold text-rose-700 hover:bg-rose-100"
                        >
                          <Trash2 className="h-3.5 w-3.5" /> ลบคนไข้ถาวร
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {tab === 'workspace' && (
        <CoursePrescriptionView
          patients={patients}
          ptColleagues={ptColleagues}
          currentUser={currentUser}
          aiPoseDetectionEnabled={aiPoseEnabled}
          apiClient={apiClient}
          onRefresh={loadAll}
        />
      )}

      {tab === 'notes' && (
        <PatientProgressRealtimeView patients={patients} apiClient={apiClient} mode="notes" />
      )}

      {tab === 'results' && (
        <PatientProgressRealtimeView patients={patients} apiClient={apiClient} mode="results" />
      )}

      {tab === 'library' && (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <div className="rounded-3xl bg-white p-6 shadow-md border border-slate-200 space-y-3">
            <h3 className="text-base font-bold text-slate-900">คลังท่ากายภาพโรงพยาบาล (ดูทั้งหมด / ลบได้เฉพาะที่สร้างเอง)</h3>
            <div className="max-h-96 overflow-y-auto space-y-2">
              {exercises.map((ex) => {
                const isGlobal = ex.createdBy !== currentUser.id;
                return (
                  <div key={ex.id} className="flex items-center justify-between rounded-xl bg-slate-50 p-3 text-xs border border-slate-200">
                    <div>
                      <span className="font-bold text-slate-900">{ex.code} — {ex.name.th}</span>
                      <p className="text-slate-500">{isGlobal ? 'คลังส่วนกลางโรงพยาบาล (ห้ามแก้ไข/ลบ)' : 'สร้างโดยฉัน'}</p>
                    </div>
                    {isGlobal ? (
                      <Lock className="h-4 w-4 text-slate-400" />
                    ) : (
                      <button
                        onClick={() => apiClient.deleteExercise(ex.id).then(loadAll)}
                        className="text-rose-600 font-bold"
                      >
                        ลบ
                      </button>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          <div className="rounded-3xl bg-white p-6 shadow-md border border-slate-200 space-y-4">
            <h3 className="text-base font-bold text-slate-900">ชุดมินิเกมฝึกสมอง (สร้าง/จัดการแพ็กเกจของฉัน)</h3>
            <div className="flex gap-2">
              <input
                value={newModpackName}
                onChange={(e) => setNewModpackName(e.target.value)}
                placeholder="ชื่อแพ็กเกจมินิเกมใหม่..."
                className="flex-1 rounded-xl border border-slate-300 px-3 py-2 text-sm"
              />
              <button
                onClick={async () => {
                  if (!newModpackName.trim()) return;
                  await apiClient.createModpack({
                    name: newModpackName.trim(),
                    categories: ['safety_routine', 'counting'],
                    questionCount: 15,
                  });
                  setNewModpackName('');
                  loadAll();
                }}
                className="inline-flex items-center gap-1 rounded-xl bg-indigo-600 px-4 py-2 text-xs font-bold text-white"
              >
                <Plus className="h-4 w-4" /> สร้างแพ็กเกจ
              </button>
            </div>
            <div className="space-y-2">
              {modpacks.map((m) => {
                const isOwn = m.createdBy === currentUser.id;
                return (
                  <div key={m.id} className="flex items-center justify-between rounded-xl border border-indigo-100 bg-indigo-50/40 p-3 text-xs">
                    <div>
                      <p className="font-bold text-indigo-950">{m.name}</p>
                      <p className="text-indigo-700">{isOwn ? 'แพ็กเกจของฉัน' : 'แพ็กเกจส่วนกลางโรงพยาบาล'}</p>
                    </div>
                    {isOwn ? (
                      <button
                        onClick={() => apiClient.deleteModpack(m.id).then(loadAll)}
                        className="text-rose-600 font-bold"
                      >
                        ลบ
                      </button>
                    ) : (
                      <Lock className="h-4 w-4 text-indigo-300" />
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {tab === 'profile' && (
        <div className="max-w-lg rounded-3xl bg-white p-6 shadow-md border border-slate-200 space-y-4">
          <h3 className="text-base font-bold text-slate-900">ข้อมูลส่วนตัวนักกายภาพบำบัด (แก้ไขได้เฉพาะของตนเอง)</h3>
          <div>
            <label className="text-xs font-semibold text-slate-600">ชื่อ-นามสกุล</label>
            <input
              value={profileName}
              onChange={(e) => setProfileName(e.target.value)}
              className="mt-1 w-full rounded-xl border border-slate-300 px-3 py-2 text-sm"
            />
          </div>
          <div>
            <label className="text-xs font-semibold text-slate-600">เบอร์โทรศัพท์</label>
            <input
              value={profilePhone}
              onChange={(e) => setProfilePhone(e.target.value)}
              className="mt-1 w-full rounded-xl border border-slate-300 px-3 py-2 text-sm"
            />
          </div>
          {profileMsg && <p className="text-xs font-bold text-emerald-700">{profileMsg}</p>}
          <button
            onClick={async () => {
              await apiClient.updateUser(currentUser.id, { fullName: profileName, phone: profilePhone });
              setProfileMsg('อัปเดตข้อมูลส่วนตัวเรียบร้อย');
            }}
            className="rounded-xl bg-teal-600 px-4 py-2 text-sm font-bold text-white"
          >
            บันทึกข้อมูลส่วนตัว
          </button>
        </div>
      )}

      {tab === 'audit_403' && (
        <div className="rounded-3xl bg-rose-50 border-2 border-rose-300 p-10 text-center space-y-3">
          <ShieldAlert className="mx-auto h-14 w-14 text-rose-600" />
          <h2 className="text-2xl font-black text-rose-900">403 Forbidden</h2>
          <p className="text-sm font-semibold text-rose-700">
            {audit403Error || 'ปิดกั้นการเข้าถึงหน้า Audit Logs 100% สำหรับบทบาทนักกายภาพบำบัด'}
          </p>
        </div>
      )}
    </div>
  );
}
