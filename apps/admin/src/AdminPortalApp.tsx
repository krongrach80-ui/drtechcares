import React, { useEffect, useState } from 'react';
import { SmartEnrollmentView } from '@apps/pt';
import type {
  ActiveSessionRecord,
  AuditLogRecord,
  BannedTargetRecord,
  DrTechCaresApiClient,
  ManagedExerciseRecord,
  ModpackRecord,
  PatientRecord,
  RehabSessionRecord,
  SystemSettingsRecord,
  UserAccount,
  UserRole,
} from '@drtechcares/api-client';
import {
  Activity,
  Ban,
  BookOpen,
  KeyRound,
  LogOut,
  Plus,
  Settings,
  Shield,
  Stethoscope,
  Trash2,
  UserPlus,
  Users,
} from 'lucide-react';

interface AdminPortalAppProps {
  apiClient: DrTechCaresApiClient;
  currentUser: UserAccount;
  onLogout: () => void;
}

type AdminTab = 'users' | 'patients' | 'pts' | 'library' | 'security' | 'settings';

export function AdminPortalApp({ apiClient, currentUser, onLogout }: AdminPortalAppProps) {
  const [tab, setTab] = useState<AdminTab>('users');
  const [users, setUsers] = useState<UserAccount[]>([]);
  const [patients, setPatients] = useState<PatientRecord[]>([]);
  const [exercises, setExercises] = useState<ManagedExerciseRecord[]>([]);
  const [modpacks, setModpacks] = useState<ModpackRecord[]>([]);
  const [sessions, setSessions] = useState<RehabSessionRecord[]>([]);
  const [logs, setLogs] = useState<AuditLogRecord[]>([]);
  const [bannedTargets, setBannedTargets] = useState<BannedTargetRecord[]>([]);
  const [activeSessions, setActiveSessions] = useState<ActiveSessionRecord[]>([]);
  const [settings, setSettings] = useState<SystemSettingsRecord>({
    aiFaceScanEnabled: true,
    aiFaceThreshold: 0.925,
    aiPoseDetectionEnabled: false,
    kioskAutoResetSec: 6,
  });

  // Create Staff User Form
  const [newUsername, setNewUsername] = useState('');
  const [newPassword, setNewPassword] = useState('Staff@2026');
  const [newFullName, setNewFullName] = useState('');
  const [newRole, setNewRole] = useState<UserRole>('pt');
  const [newSpecialty, setNewSpecialty] = useState('Physical Therapy');

  // Create Exercise Form
  const [newExCode, setNewExCode] = useState('');
  const [newExName, setNewExName] = useState('');
  const [newModpackName, setNewModpackName] = useState('');

  // Ban Form
  const [banType, setBanType] = useState<'ip' | 'device'>('ip');
  const [banValue, setBanValue] = useState('');
  const [banReason, setBanReason] = useState('Suspicious activity');
  const [statusBanner, setStatusBanner] = useState<string | null>(null);

  const loadAll = async () => {
    const [u, p, ex, mp, s, sec, cfg] = await Promise.all([
      apiClient.listUsers(),
      apiClient.listPatients(),
      apiClient.listExercises(),
      apiClient.listModpacks(),
      apiClient.listRehabSessions(),
      apiClient.listSecurityAudit(),
      apiClient.getSystemSettings(),
    ]);
    setUsers(u);
    setPatients(p);
    setExercises(ex);
    setModpacks(mp);
    setSessions(s);
    setLogs(sec.logs);
    setBannedTargets(sec.bannedTargets);
    setActiveSessions(sec.activeSessions);
    setSettings(cfg.settings);
  };

  useEffect(() => {
    loadAll();
  }, [apiClient]);

  const ptUsers = users.filter((u) => u.role === 'pt');

  return (
    <div className="mx-auto max-w-7xl space-y-6">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 rounded-3xl bg-white p-5 shadow-sm border border-slate-200">
        <div>
          <span className="inline-block rounded-full bg-indigo-50 px-3 py-1 text-xs font-bold text-indigo-700">
            SUPER ADMIN / ผู้อำนวยการโรงพยาบาล (Full CRUD Access)
          </span>
          <h1 className="mt-1 text-xl font-extrabold text-slate-900">{currentUser.fullName}</h1>
          <p className="text-xs text-slate-500">
            ผู้ใช้ทั้งหมด {users.length} • คนไข้ {patients.length} • ท่าฝึก {exercises.length} • Active Sessions {activeSessions.length}
          </p>
        </div>
        <button
          onClick={onLogout}
          className="inline-flex items-center gap-1.5 rounded-xl border border-slate-300 px-4 py-2 text-xs font-bold text-slate-700 hover:bg-slate-50"
        >
          <LogOut className="h-4 w-4" /> ออกจากระบบ
        </button>
      </div>

      {statusBanner && (
        <div className="rounded-2xl bg-emerald-50 border border-emerald-200 px-4 py-2.5 text-xs font-bold text-emerald-800">
          {statusBanner}
        </div>
      )}

      {/* Navigation Tabs */}
      <div className="flex flex-wrap gap-2 rounded-2xl bg-white p-2 shadow-sm border border-slate-200">
        {[
          { id: 'users', label: 'จัดการผู้ใช้งาน & สแกนหน้า 10 เฟรม', icon: Users },
          { id: 'patients', label: 'ข้อมูลคนไข้ทั้งโรงพยาบาล', icon: UserPlus },
          { id: 'pts', label: 'ข้อมูลนักกายภาพบำบัด', icon: Stethoscope },
          { id: 'library', label: 'คลังท่ากายภาพ & มินิเกม', icon: BookOpen },
          { id: 'security', label: 'ประวัติการใช้งาน / แบน IP / Force Logout', icon: Shield },
          { id: 'settings', label: 'ตั้งค่าระบบ & AI', icon: Settings },
        ].map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              onClick={() => setTab(item.id as AdminTab)}
              className={`inline-flex items-center gap-1.5 rounded-xl px-3.5 py-2 text-xs font-bold transition ${
                tab === item.id ? 'bg-indigo-600 text-white shadow' : 'text-slate-600 hover:bg-slate-100'
              }`}
            >
              <Icon className="h-4 w-4" /> {item.label}
            </button>
          );
        })}
      </div>

      {/* TAB 1: USERS & 10-SHOT ENROLLMENT */}
      {tab === 'users' && (
        <div className="space-y-6">
          <div className="rounded-3xl bg-white p-6 shadow-md border border-slate-200 space-y-4">
            <h2 className="text-base font-bold text-slate-900">เพิ่มบัญชีผู้ใช้งานใหม่ (Admin / PT / Patient)</h2>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-5">
              <input
                value={newUsername}
                onChange={(e) => setNewUsername(e.target.value)}
                placeholder="Username"
                className="rounded-xl border border-slate-300 px-3 py-2 text-sm"
              />
              <input
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                placeholder="Password"
                className="rounded-xl border border-slate-300 px-3 py-2 text-sm"
              />
              <input
                value={newFullName}
                onChange={(e) => setNewFullName(e.target.value)}
                placeholder="ชื่อ-นามสกุล"
                className="rounded-xl border border-slate-300 px-3 py-2 text-sm"
              />
              <select
                value={newRole}
                onChange={(e) => setNewRole(e.target.value as UserRole)}
                className="rounded-xl border border-slate-300 px-3 py-2 text-sm font-bold"
              >
                <option value="super_admin">super_admin</option>
                <option value="pt">pt</option>
                <option value="patient">patient</option>
              </select>
              <button
                onClick={async () => {
                  if (!newUsername.trim() || !newFullName.trim()) return;
                  await apiClient.createUser({
                    username: newUsername.trim(),
                    password: newPassword,
                    role: newRole,
                    fullName: newFullName.trim(),
                    specialty: newSpecialty,
                  });
                  setNewUsername('');
                  setNewFullName('');
                  setStatusBanner('สร้างบัญชีผู้ใช้งานใหม่สำเร็จ');
                  loadAll();
                }}
                className="inline-flex items-center justify-center gap-1 rounded-xl bg-indigo-600 px-4 py-2 text-sm font-bold text-white hover:bg-indigo-700"
              >
                <Plus className="h-4 w-4" /> เพิ่มบัญชี
              </button>
            </div>

            <div className="space-y-2 pt-2">
              {users.map((u) => (
                <div key={u.id} className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-slate-200 p-3 text-sm">
                  <div>
                    <span className="font-bold text-slate-900">{u.fullName}</span>{' '}
                    <span className="text-xs text-slate-500">@{u.username} • {u.role}</span>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <select
                      value={u.role}
                      onChange={async (e) => {
                        await apiClient.updateUser(u.id, { role: e.target.value as UserRole });
                        loadAll();
                      }}
                      className="rounded-lg border border-slate-300 px-2.5 py-1 text-xs font-bold"
                    >
                      <option value="super_admin">super_admin</option>
                      <option value="pt">pt</option>
                      <option value="patient">patient</option>
                    </select>
                    <button
                      onClick={async () => {
                        await apiClient.resetUserPassword(u.id, 'Reset@2026');
                        setStatusBanner(`รีเซ็ตรหัสผ่าน @${u.username} เป็น Reset@2026 แล้ว`);
                      }}
                      className="inline-flex items-center gap-1 rounded-lg bg-slate-100 px-2.5 py-1 text-xs font-bold text-slate-700"
                    >
                      <KeyRound className="h-3.5 w-3.5" /> รีเซ็ตรหัสผ่าน
                    </button>
                    {u.id !== currentUser.id && (
                      <button
                        onClick={async () => {
                          await apiClient.deleteUser(u.id);
                          loadAll();
                        }}
                        className="inline-flex items-center gap-1 rounded-lg bg-rose-50 px-2.5 py-1 text-xs font-bold text-rose-700"
                      >
                        <Trash2 className="h-3.5 w-3.5" /> ลบ
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <SmartEnrollmentView
            apiClient={apiClient}
            currentUserRole="super_admin"
            ptList={ptUsers}
            onEnrolled={() => {
              setStatusBanner('ลงทะเบียนคนไข้และเวกเตอร์ใบหน้า 10 เฟรมสำเร็จ');
              loadAll();
            }}
          />
        </div>
      )}

      {/* TAB 2: ALL HOSPITAL PATIENTS */}
      {tab === 'patients' && (
        <div className="rounded-3xl bg-white p-6 shadow-md border border-slate-200 space-y-4">
          <h2 className="text-base font-bold text-slate-900">ข้อมูลคนไข้ & ประวัติการรักษาทั้งโรงพยาบาล ({patients.length})</h2>
          {patients.length === 0 && <p className="text-xs text-slate-500">ยังไม่มีข้อมูลผู้ป่วย</p>}
          <div className="space-y-3">
            {patients.map((p) => {
              const pSessions = sessions.filter((s) => s.patientId === p.id);
              const assignedPt = ptUsers.find((pt) => pt.id === p.assignedPtId);
              return (
                <div key={p.id} className="rounded-2xl border border-slate-200 p-4 text-sm space-y-2">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div>
                      <span className="font-bold text-slate-900">{p.fullName}</span>{' '}
                      <span className="text-xs text-slate-500">({p.hn} • อายุ {p.age} • นัดถัดไป {p.nextAppointmentDate})</span>
                      <p className="text-xs text-slate-600">
                        การวินิจฉัย: {p.diagnosis} • PT ผู้รับผิดชอบ: {assignedPt?.fullName ?? p.assignedPtId}
                      </p>
                    </div>
                    <button
                      onClick={() => apiClient.deletePatientCascading(p.id).then(loadAll)}
                      className="inline-flex items-center gap-1 rounded-xl bg-rose-50 px-3 py-1.5 text-xs font-bold text-rose-700 hover:bg-rose-100"
                    >
                      <Trash2 className="h-3.5 w-3.5" /> ลบคนไข้ (Cascading Purge)
                    </button>
                  </div>
                  <p className="text-xs text-teal-700 font-semibold">
                    ประวัติการฝึกสะสม: {pSessions.length} ครั้ง
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* TAB 3: PHYSIOTHERAPIST DIRECTORY */}
      {tab === 'pts' && (
        <div className="rounded-3xl bg-white p-6 shadow-md border border-slate-200 space-y-4">
          <h2 className="text-base font-bold text-slate-900">ข้อมูลนักกายภาพบำบัดและเคสที่รับผิดชอบ</h2>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            {ptUsers.map((pt) => {
              const owned = patients.filter((p) => p.assignedPtId === pt.id);
              const delegated = patients.filter((p) => p.delegatedToPtId === pt.id);
              return (
                <div key={pt.id} className="rounded-2xl border border-slate-200 p-4 space-y-2">
                  <h3 className="font-bold text-slate-900">{pt.fullName}</h3>
                  <p className="text-xs text-slate-500">
                    ใบประกอบวิชาชีพ: {pt.licenseNo || '-'} • ความเชี่ยวชาญ: {pt.specialty || '-'} • โทร: {pt.phone}
                  </p>
                  <p className="text-xs font-semibold text-teal-700">
                    เคสหลัก: {owned.length} คน • เคสรับฝากชั่วคราว: {delegated.length} คน
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* TAB 4: GLOBAL EXERCISE & MODPACK LIBRARY */}
      {tab === 'library' && (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <div className="rounded-3xl bg-white p-6 shadow-md border border-slate-200 space-y-4">
            <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
              <Activity className="h-5 w-5 text-teal-600" />
              คลังท่ากายภาพบำบัด ({exercises.length})
            </h2>
            <div className="flex gap-2">
              <input
                value={newExCode}
                onChange={(e) => setNewExCode(e.target.value)}
                placeholder="รหัสท่า (เช่น SH_EXT_R)"
                className="w-36 rounded-xl border border-slate-300 px-3 py-2 text-xs"
              />
              <input
                value={newExName}
                onChange={(e) => setNewExName(e.target.value)}
                placeholder="ชื่อท่ากายภาพใหม่"
                className="flex-1 rounded-xl border border-slate-300 px-3 py-2 text-xs"
              />
              <button
                onClick={async () => {
                  if (!newExCode.trim() || !newExName.trim()) return;
                  await apiClient.createExercise({
                    id: `ex-${Date.now()}`,
                    code: newExCode.trim(),
                    category: 'upper_limb',
                    name: { th: newExName.trim(), en: newExName.trim(), cn: newExName.trim() },
                    instruction: { th: newExName.trim(), en: newExName.trim(), cn: newExName.trim() },
                    detection: {
                      mode: 'angle_rep_hold',
                      jointGroup: 'shoulder',
                      side: 'right',
                      targetMinDeg: 100,
                      targetMaxDeg: 175,
                      startRestMaxDeg: 35,
                      holdSec: 3,
                      defaultSets: 2,
                      defaultReps: 5,
                      requireStraightElbow: true,
                    },
                  });
                  setNewExCode('');
                  setNewExName('');
                  loadAll();
                }}
                className="rounded-xl bg-teal-600 px-3 py-2 text-xs font-bold text-white"
              >
                เพิ่มท่า
              </button>
            </div>
            <div className="max-h-72 overflow-y-auto space-y-2">
              {exercises.map((ex) => (
                <div key={ex.id} className="flex items-center justify-between rounded-xl bg-slate-50 p-3 text-xs">
                  <span className="font-bold text-slate-900">{ex.code} — {ex.name.th}</span>
                  <button
                    onClick={() => apiClient.deleteExercise(ex.id).then(loadAll)}
                    className="font-bold text-rose-600 hover:underline"
                  >
                    ลบ
                  </button>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-3xl bg-white p-6 shadow-md border border-slate-200 space-y-4">
            <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
              <BookOpen className="h-5 w-5 text-indigo-600" />
              คลังชุดมินิเกม ({modpacks.length})
            </h2>
            <div className="flex gap-2">
              <input
                value={newModpackName}
                onChange={(e) => setNewModpackName(e.target.value)}
                placeholder="ชื่อชุดมินิเกมใหม่..."
                className="flex-1 rounded-xl border border-slate-300 px-3 py-2 text-xs"
              />
              <button
                onClick={async () => {
                  if (!newModpackName.trim()) return;
                  await apiClient.createModpack({
                    name: newModpackName.trim(),
                    categories: ['food_fruit', 'safety_routine'],
                    questionCount: 25,
                  });
                  setNewModpackName('');
                  loadAll();
                }}
                className="rounded-xl bg-indigo-600 px-3 py-2 text-xs font-bold text-white"
              >
                เพิ่มชุดเกม
              </button>
            </div>
            <div className="space-y-2">
              {modpacks.map((m) => (
                <div key={m.id} className="flex items-center justify-between rounded-xl bg-indigo-50/50 p-3 text-xs">
                  <span className="font-bold text-indigo-950">{m.name} ({m.questionCount} ข้อ)</span>
                  <button
                    onClick={() => apiClient.deleteModpack(m.id).then(loadAll)}
                    className="font-bold text-rose-600 hover:underline"
                  >
                    ลบ
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* TAB 5: SECURITY, AUDIT LOGS, IP/DEVICE BAN, FORCE LOGOUT */}
      {tab === 'security' && (
        <div className="space-y-6">
          {/* Ban IP / Device */}
          <div className="rounded-3xl bg-white p-6 shadow-md border border-slate-200 space-y-4">
            <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
              <Ban className="h-5 w-5 text-rose-600" />
              แบนอุปกรณ์ / ไอพี (IP & Device Ban)
            </h2>
            <div className="flex flex-wrap gap-2">
              <select
                value={banType}
                onChange={(e) => setBanType(e.target.value as 'ip' | 'device')}
                className="rounded-xl border border-slate-300 px-3 py-2 text-xs font-bold"
              >
                <option value="ip">IP Address</option>
                <option value="device">Device ID</option>
              </select>
              <input
                value={banValue}
                onChange={(e) => setBanValue(e.target.value)}
                placeholder="เช่น 192.168.1.50 หรือ DEV-XXXX"
                className="flex-1 rounded-xl border border-slate-300 px-3 py-2 text-xs"
              />
              <input
                value={banReason}
                onChange={(e) => setBanReason(e.target.value)}
                placeholder="เหตุผล"
                className="flex-1 rounded-xl border border-slate-300 px-3 py-2 text-xs"
              />
              <button
                onClick={async () => {
                  if (!banValue.trim()) return;
                  await apiClient.banTarget(banType, banValue.trim(), banReason);
                  setBanValue('');
                  loadAll();
                }}
                className="rounded-xl bg-rose-600 px-4 py-2 text-xs font-bold text-white"
              >
                สั่งแบนทันที
              </button>
            </div>
            <div className="space-y-1.5">
              {bannedTargets.map((b) => (
                <div key={b.id} className="flex items-center justify-between rounded-xl bg-rose-50 px-3 py-2 text-xs">
                  <span className="font-bold text-rose-900">
                    [{b.targetType.toUpperCase()}] {b.value} — {b.reason}
                  </span>
                  <button
                    onClick={() => apiClient.unbanTarget(b.id).then(loadAll)}
                    className="font-bold text-slate-700 hover:underline"
                  >
                    ปลดแบน
                  </button>
                </div>
              ))}
            </div>
          </div>

          {/* Active Sessions & Force Logout */}
          <div className="rounded-3xl bg-white p-6 shadow-md border border-slate-200 space-y-3">
            <h2 className="text-base font-bold text-slate-900">Active Sessions & บังคับ Force Logout</h2>
            {activeSessions.map((s) => (
              <div key={s.token} className="flex flex-wrap items-center justify-between gap-2 rounded-xl bg-slate-50 p-3 text-xs">
                <span>
                  User: <b>{s.userId}</b> ({s.role}) • IP: <b>{s.ipAddress}</b> • Device: <b>{s.deviceId}</b>
                </span>
                {s.userId !== currentUser.id && (
                  <button
                    onClick={async () => {
                      await apiClient.forceLogoutUser(s.userId);
                      setStatusBanner(`บังคับ Force Logout ผู้ใช้ ${s.userId} แล้ว`);
                      loadAll();
                    }}
                    className="rounded-lg bg-amber-600 px-3 py-1 font-bold text-white"
                  >
                    Force Logout
                  </button>
                )}
              </div>
            ))}
          </div>

          {/* Audit Logs */}
          <div className="rounded-3xl bg-white p-6 shadow-md border border-slate-200 space-y-3">
            <h2 className="text-base font-bold text-slate-900">ประวัติการเข้าใช้งาน & แก้ไขข้อมูล (Audit Logs)</h2>
            <div className="max-h-72 overflow-y-auto space-y-1.5 text-xs">
              {logs.map((log) => (
                <div key={log.id} className="flex flex-wrap items-center justify-between gap-2 rounded-lg bg-slate-50 px-3 py-2">
                  <div>
                    <span className="font-bold text-slate-900">{log.eventType}</span> • {log.detail}
                  </div>
                  <span className="text-slate-500">
                    {log.actorRole} ({log.ipAddress} / {log.deviceId}) • {new Date(log.createdAt).toLocaleString()}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* TAB 6: SYSTEM & AI SETTINGS */}
      {tab === 'settings' && (
        <div className="max-w-2xl rounded-3xl bg-white p-6 shadow-md border border-slate-200 space-y-5">
          <h2 className="text-base font-bold text-slate-900">ตั้งค่า AI สแกนใบหน้า & AI วิเคราะห์กายภาพ</h2>

          <label className="flex items-center justify-between rounded-2xl border border-slate-200 p-4">
            <div>
              <p className="text-sm font-bold text-slate-900">เปิดใช้งาน AI สแกนใบหน้า (Facial Auth)</p>
              <p className="text-xs text-slate-500">Zero False-Positive Multi-Subband 512-D Gate</p>
            </div>
            <input
              type="checkbox"
              checked={settings.aiFaceScanEnabled}
              onChange={(e) => setSettings({ ...settings, aiFaceScanEnabled: e.target.checked })}
              className="h-5 w-5 accent-indigo-600"
            />
          </label>

          <label className="flex items-center justify-between rounded-2xl border border-slate-200 p-4">
            <div>
              <p className="text-sm font-bold text-slate-900">เปิดใช้งาน AI วิเคราะห์ท่ากายภาพบำบัด</p>
              <p className="text-xs text-slate-500">
                หากปิดไว้ ตู้ Kiosk จะแสดงข้อความ "ยังไม่ได้เพิ่มระบบดีเทคกายภาพ โปรดรอการพัฒนา" (3–5 วินาที) ตามผังระบบ
              </p>
            </div>
            <input
              type="checkbox"
              checked={settings.aiPoseDetectionEnabled}
              onChange={(e) => setSettings({ ...settings, aiPoseDetectionEnabled: e.target.checked })}
              className="h-5 w-5 accent-indigo-600"
            />
          </label>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-xs font-semibold text-slate-600">Face Cosine Threshold (0.90 - 0.98)</label>
              <input
                type="number"
                step="0.005"
                min="0.90"
                max="0.98"
                value={settings.aiFaceThreshold}
                onChange={(e) => setSettings({ ...settings, aiFaceThreshold: Number(e.target.value) })}
                className="mt-1 w-full rounded-xl border border-slate-300 px-3 py-2 text-sm"
              />
            </div>
            <div>
              <label className="text-xs font-semibold text-slate-600">เวลา Auto-Reset ตู้ Kiosk (วินาที)</label>
              <input
                type="number"
                min="3"
                max="30"
                value={settings.kioskAutoResetSec}
                onChange={(e) => setSettings({ ...settings, kioskAutoResetSec: Number(e.target.value) })}
                className="mt-1 w-full rounded-xl border border-slate-300 px-3 py-2 text-sm"
              />
            </div>
          </div>

          <button
            onClick={async () => {
              const updated = await apiClient.updateSystemSettings(settings);
              setSettings(updated);
              setStatusBanner('บันทึกการตั้งค่าระบบและ AI เรียบร้อยแล้ว');
            }}
            className="rounded-xl bg-indigo-600 px-5 py-2.5 text-sm font-bold text-white hover:bg-indigo-700"
          >
            บันทึกการตั้งค่า
          </button>
        </div>
      )}
    </div>
  );
}
