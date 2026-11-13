import React, { useEffect, useState } from 'react';
import { DynamicExerciseSessionView } from '@apps/kiosk';
import type {
  DrTechCaresApiClient,
  PatientCourseRecord,
  PatientRecord,
  RehabSessionRecord,
  TreatmentReportRecord,
  UserAccount,
} from '@drtechcares/api-client';
import { Activity, Calendar, KeyRound, LogOut, Sparkles, User } from 'lucide-react';

interface PatientPersonalPortalProps {
  apiClient: DrTechCaresApiClient;
  currentUser: UserAccount;
  onLogout: () => void;
}

export function PatientPersonalPortal({ apiClient, currentUser, onLogout }: PatientPersonalPortalProps) {
  const [patient, setPatient] = useState<PatientRecord | null>(null);
  const [course, setCourse] = useState<PatientCourseRecord | null>(null);
  const [sessions, setSessions] = useState<RehabSessionRecord[]>([]);
  const [reports, setReports] = useState<TreatmentReportRecord[]>([]);
  const [phone, setPhone] = useState(currentUser.phone);
  const [currentPw, setCurrentPw] = useState('');
  const [newPw, setNewPw] = useState('');
  const [notice, setNotice] = useState<string | null>(null);
  const [cameraActive, setCameraActive] = useState(false);

  const loadOwnData = async () => {
    const [pts, sess, reps] = await Promise.all([
      apiClient.listPatients(),
      apiClient.listRehabSessions(),
      apiClient.listTreatmentReports(),
    ]);
    const own = pts[0] ?? null;
    setPatient(own);
    if (own) {
      setPhone(own.phone);
      const c = await apiClient.getPatientCourse(own.id);
      setCourse(c);
    }
    setSessions(sess);
    setReports(reps);
  };

  useEffect(() => {
    loadOwnData();
  }, [apiClient]);

  if (cameraActive && patient) {
    return (
      <div className="mx-auto max-w-6xl p-4">
        <DynamicExerciseSessionView
          patient={patient}
          apiClient={apiClient}
          onFinished={() => {
            setCameraActive(false);
            loadOwnData();
          }}
        />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-4">
      <div className="flex flex-wrap items-center justify-between gap-4 rounded-3xl bg-white p-5 shadow-sm border border-slate-200">
        <div>
          <span className="inline-block rounded-full bg-sky-50 px-3 py-1 text-xs font-bold text-sky-700">
            PATIENT PORTAL (ดูได้เฉพาะข้อมูลของตัวเอง)
          </span>
          <h1 className="mt-1 text-xl font-extrabold text-slate-900">{currentUser.fullName}</h1>
          {patient && (
            <p className="text-xs text-slate-500">
              HN: {patient.hn} • การวินิจฉัย: {patient.diagnosis} • นัดหมายครั้งต่อไป: {patient.nextAppointmentDate}
            </p>
          )}
        </div>
        <button
          onClick={onLogout}
          className="inline-flex items-center gap-1.5 rounded-xl border border-slate-300 px-4 py-2 text-xs font-bold text-slate-700 hover:bg-slate-50"
        >
          <LogOut className="h-4 w-4" /> ออกจากระบบ
        </button>
      </div>

      {notice && (
        <div className="rounded-xl bg-emerald-50 border border-emerald-200 px-4 py-2 text-xs font-bold text-emerald-800">
          {notice}
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* ข้อมูลส่วนตัว & เปลี่ยนรหัสผ่าน */}
        <div className="rounded-3xl bg-white p-6 shadow-md border border-slate-200 space-y-4">
          <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
            <User className="h-5 w-5 text-sky-600" />
            ข้อมูลส่วนตัว (แก้ไขเบอร์โทร / เปลี่ยนรหัสผ่าน)
          </h2>
          <div className="flex gap-2">
            <input
              value={phone}
              onChange={(e) => setPhone(e.target.value)}
              placeholder="เบอร์โทรศัพท์"
              className="flex-1 rounded-xl border border-slate-300 px-3 py-2 text-sm"
            />
            <button
              onClick={async () => {
                if (patient) await apiClient.updatePatient(patient.id, { phone });
                await apiClient.updateUser(currentUser.id, { phone });
                setNotice('บันทึกเบอร์โทรศัพท์เรียบร้อย');
              }}
              className="rounded-xl bg-sky-600 px-4 py-2 text-xs font-bold text-white"
            >
              บันทึก
            </button>
          </div>
          <div className="pt-2 border-t border-slate-100 space-y-2">
            <p className="text-xs font-bold text-slate-700 flex items-center gap-1">
              <KeyRound className="h-3.5 w-3.5" /> เปลี่ยนรหัสผ่านส่วนตัว
            </p>
            <div className="flex gap-2">
              <input
                type="password"
                value={currentPw}
                onChange={(e) => setCurrentPw(e.target.value)}
                placeholder="รหัสผ่านเดิม"
                className="flex-1 rounded-xl border border-slate-300 px-3 py-2 text-xs"
              />
              <input
                type="password"
                value={newPw}
                onChange={(e) => setNewPw(e.target.value)}
                placeholder="รหัสผ่านใหม่ (>=6 ตัว)"
                className="flex-1 rounded-xl border border-slate-300 px-3 py-2 text-xs"
              />
              <button
                onClick={async () => {
                  await apiClient.changePassword(currentPw, newPw);
                  setCurrentPw('');
                  setNewPw('');
                  setNotice('เปลี่ยนรหัสผ่านเรียบร้อย');
                }}
                className="rounded-xl bg-slate-800 px-3 py-2 text-xs font-bold text-white"
              >
                เปลี่ยนรหัส
              </button>
            </div>
          </div>
        </div>

        {/* แผนการรักษา & ฝึกผ่านกล้อง */}
        <div className="rounded-3xl bg-white p-6 shadow-md border border-slate-200 space-y-4">
          <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
            <Calendar className="h-5 w-5 text-teal-600" />
            ข้อมูลการรักษา & ตารางฝึกของฉัน
          </h2>
          <p className="text-sm text-slate-600">{course?.title ?? 'แผนฟื้นฟูสมรรถภาพ'}</p>
          <button
            onClick={() => setCameraActive(true)}
            disabled={!patient}
            className="w-full inline-flex items-center justify-center gap-2 rounded-2xl bg-teal-600 py-3 text-sm font-bold text-white hover:bg-teal-700"
          >
            <Activity className="h-4 w-4" /> ทำท่ากายภาพตามโปรแกรมผ่านกล้อง
          </button>
        </div>
      </div>

      {/* ผลการรักษา & ผลวิเคราะห์จาก AI */}
      <div className="rounded-3xl bg-white p-6 shadow-md border border-slate-200 space-y-3">
        <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
          <Sparkles className="h-5 w-5 text-indigo-600" />
          ผลการรักษา & ผลวิเคราะห์ท่าทางจาก AI
        </h2>
        {sessions.length === 0 && <p className="text-xs text-slate-500">ยังไม่มีผลคะแนนการฝึก</p>}
        <div className="space-y-2">
          {sessions.map((s) => (
            <div key={s.id} className="rounded-2xl border border-slate-200 p-4 text-xs">
              <div className="flex justify-between font-bold text-slate-900">
                <span>{s.exerciseName}</span>
                <span>{new Date(s.createdAt).toLocaleString()}</span>
              </div>
              <p className="mt-1 text-slate-700">
                ความแม่นยำ: <b>{s.accuracyPercent}%</b> • มุม ROM สูงสุด: <b>{s.maxRomDeg}°</b>
              </p>
              <p className="mt-1 font-semibold text-indigo-700">AI คำแนะนำ: {s.aiFeedback}</p>
            </div>
          ))}
        </div>
        {reports.length > 0 && (
          <div className="pt-3 border-t border-slate-100 space-y-2">
            <p className="text-xs font-bold text-slate-700">รายงานจากนักกายภาพบำบัด</p>
            {reports.map((r) => (
              <div key={r.id} className="rounded-xl bg-slate-50 p-3 text-xs">
                <p className="font-bold text-slate-900">{r.summary}</p>
                <p className="text-slate-600">{r.aiRecommendation}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

