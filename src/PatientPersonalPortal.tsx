import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { DynamicExerciseSessionView } from '@apps/kiosk';
import { CLINICAL_EXERCISES, parseExerciseCodeAndSide, type BodySide } from '@drtechcares/pose-engine';
import {
  formatLocalIsoDate,
  type DrTechCaresApiClient,
  type PatientCourseRecord,
  type PatientRecord,
  type RehabSessionRecord,
  type TreatmentReportRecord,
  type UserAccount,
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
  const [phone, setPhone] = useState(currentUser.phone ?? '');
  const [currentPw, setCurrentPw] = useState('');
  const [newPw, setNewPw] = useState('');
  const [notice, setNotice] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [savingPhone, setSavingPhone] = useState(false);
  const [savingPw, setSavingPw] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const mountedRef = useRef(true);

  const loadOwnData = useCallback(async () => {
    try {
      setErrorMsg(null);
      const [pts, sess, reps] = await Promise.all([
        apiClient.listPatients(),
        apiClient.listRehabSessions(),
        apiClient.listTreatmentReports(),
      ]);
      if (!mountedRef.current) return;
      const own =
        pts.find((p) => p.id === currentUser.patientId || p.hn === currentUser.username) ??
        pts[0] ??
        null;
      setPatient(own);
      setSessions(sess);
      setReports(reps);
      if (own) {
        setPhone(own.phone ?? '');
        const c = await apiClient.getPatientCourse(own.id).catch(() => null);
        if (!mountedRef.current) return;
        setCourse(c);
      } else {
        setCourse(null);
      }
    } catch (err) {
      if (mountedRef.current) {
        setErrorMsg(err instanceof Error ? err.message : 'ไม่สามารถโหลดข้อมูลส่วนตัวได้');
      }
    }
  }, [apiClient, currentUser.patientId, currentUser.username]);

  useEffect(() => {
    mountedRef.current = true;
    void loadOwnData();
    return () => {
      mountedRef.current = false;
    };
  }, [loadOwnData]);

  const prescribedExercise = useMemo(() => {
    const resolveExerciseWithSide = (rawId: string | undefined, explicitSide?: string | null) => {
      if (!rawId) return null;
      const { baseCode, side, hasExplicitSide } = parseExerciseCodeAndSide(rawId, explicitSide);
      const matched = CLINICAL_EXERCISES.find(
        (ex) =>
          ex.id === rawId ||
          ex.code === rawId ||
          ex.code === baseCode ||
          parseExerciseCodeAndSide(ex.code || ex.id).baseCode === baseCode,
      );
      if (!matched) return null;
      if (!hasExplicitSide && !explicitSide) return matched;
      const nextSide: BodySide | 'both' = side === 'LEFT' ? 'left' : side === 'RIGHT' ? 'right' : 'both';
      return {
        ...matched,
        detection: {
          ...matched.detection,
          side: nextSide,
        },
      };
    };

    const todayIso = formatLocalIsoDate(new Date());
    const schedules = patient?.rehabSchedules ?? [];
    const todayCustom =
      schedules.find((s) => s.scheduledDate === todayIso && s.status === 'ACTIVE') ??
      schedules.find((s) => s.scheduledDate === todayIso && s.status === 'COMPLETED') ??
      schedules.find((s) => s.status === 'ACTIVE');
    if (todayCustom) {
      const fromSchedule = resolveExerciseWithSide(
        todayCustom.effectiveCode || todayCustom.exerciseCode || todayCustom.exerciseId,
        todayCustom.side,
      );
      if (fromSchedule) return fromSchedule;
    }

    const todayIdx = new Date().getDay();
    const days = course?.weeklySchedule ?? [];
    const todayRoutine = days.find((d) => d.dayIndex === todayIdx && d.dayType === 'training' && d.exercises.length > 0);
    const anyRoutine = days.find((d) => d.dayType === 'training' && d.exercises.length > 0);
    const firstId = todayRoutine?.exercises[0]?.exerciseId ?? anyRoutine?.exercises[0]?.exerciseId;
    if (firstId) {
      const fromCourse = resolveExerciseWithSide(firstId);
      if (fromCourse) return fromCourse;
    }
    return CLINICAL_EXERCISES[0];
  }, [patient, course]);

  if (cameraActive && patient) {
    return (
      <div className="mx-auto max-w-6xl p-4 animate-view-enter">
        <DynamicExerciseSessionView
          patient={patient}
          exercise={prescribedExercise}
          apiClient={apiClient}
          onFinished={() => {
            setCameraActive(false);
            void loadOwnData();
          }}
        />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-4 animate-view-enter">
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
        <div className="rounded-xl bg-emerald-50 border border-emerald-200 px-4 py-2 text-xs font-bold text-emerald-800 animate-toast-in">
          {notice}
        </div>
      )}
      {errorMsg && (
        <div className="rounded-xl bg-rose-50 border border-rose-200 px-4 py-2 text-xs font-bold text-rose-700 animate-toast-in">
          {errorMsg}
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2 animate-stagger-children">
        {/* ข้อมูลส่วนตัว & เปลี่ยนรหัสผ่าน */}
        <div className="rounded-3xl bg-white p-6 shadow-md border border-slate-200 space-y-4 card-hover-lift animate-card-enter">
          <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
            <User className="h-5 w-5 text-sky-600" />
            ข้อมูลส่วนตัว (แก้ไขเบอร์โทร / เปลี่ยนรหัสผ่าน)
          </h2>
          <div className="flex gap-2">
            <input
              type="tel"
              inputMode="numeric"
              pattern="[0-9]*"
              maxLength={10}
              value={phone}
              onChange={(e) => setPhone(e.target.value.replace(/\D/g, '').slice(0, 10))}
              placeholder="เบอร์โทรศัพท์"
              className="flex-1 rounded-xl border border-slate-300 px-3 py-2 text-sm"
            />
            <button
              type="button"
              disabled={savingPhone}
              onClick={async () => {
                setErrorMsg(null);
                setNotice(null);
                setSavingPhone(true);
                try {
                  if (patient) await apiClient.updatePatient(patient.id, { phone });
                  await apiClient.updateUser(currentUser.id, { phone });
                  if (mountedRef.current) setNotice('บันทึกเบอร์โทรศัพท์เรียบร้อย');
                } catch (err) {
                  if (mountedRef.current) {
                    setErrorMsg(err instanceof Error ? err.message : 'ไม่สามารถบันทึกเบอร์โทรศัพท์ได้');
                  }
                } finally {
                  if (mountedRef.current) setSavingPhone(false);
                }
              }}
              className="rounded-xl bg-sky-600 px-4 py-2 text-xs font-bold text-white disabled:opacity-50"
            >
              {savingPhone ? 'กำลังบันทึก...' : 'บันทึก'}
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
                type="button"
                disabled={savingPw}
                onClick={async () => {
                  setErrorMsg(null);
                  setNotice(null);
                  if (!currentPw || newPw.length < 6) {
                    setErrorMsg('กรุณากรอกรหัสผ่านเดิมและรหัสผ่านใหม่อย่างน้อย 6 ตัวอักษร');
                    return;
                  }
                  setSavingPw(true);
                  try {
                    await apiClient.changePassword(currentPw, newPw);
                    if (mountedRef.current) {
                      setCurrentPw('');
                      setNewPw('');
                      setNotice('เปลี่ยนรหัสผ่านเรียบร้อย');
                    }
                  } catch (err) {
                    if (mountedRef.current) {
                      setErrorMsg(err instanceof Error ? err.message : 'เปลี่ยนรหัสผ่านไม่สำเร็จ');
                    }
                  } finally {
                    if (mountedRef.current) setSavingPw(false);
                  }
                }}
                className="btn-primary-sheen rounded-xl bg-gradient-to-r from-[#0F4C81] to-[#0D9488] px-3 py-2 text-xs font-bold text-white disabled:opacity-50"
              >
                {savingPw ? 'กำลังเปลี่ยน...' : 'เปลี่ยนรหัส'}
              </button>
            </div>
          </div>
        </div>

        {/* แผนการรักษา & ฝึกผ่านกล้อง */}
        <div
          style={{ animationDelay: '60ms' }}
          className="rounded-3xl bg-white p-6 shadow-md border border-slate-200 space-y-4 card-hover-lift animate-card-enter"
        >
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

