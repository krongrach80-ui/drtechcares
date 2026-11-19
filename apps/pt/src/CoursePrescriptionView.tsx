import React, { useEffect, useState } from 'react';
import type {
  ActivityType,
  DailyScheduleDay,
  DrTechCaresApiClient,
  ModpackRecord,
  PatientRecord,
  UserAccount,
} from '@drtechcares/api-client';
import { ArrowRightLeft, CalendarCheck2, Lock, Save } from 'lucide-react';

interface CoursePrescriptionViewProps {
  patients: PatientRecord[];
  ptColleagues: UserAccount[];
  currentUser: UserAccount;
  aiPoseDetectionEnabled: boolean;
  apiClient: DrTechCaresApiClient;
  onRefresh: () => void;
}

const DAYS = ['อาทิตย์', 'จันทร์', 'อังคาร', 'พุธ', 'พฤหัสบดี', 'ศุกร์', 'เสาร์'];

export function CoursePrescriptionView({
  patients,
  ptColleagues,
  currentUser,
  aiPoseDetectionEnabled,
  apiClient,
  onRefresh,
}: CoursePrescriptionViewProps) {
  const [selectedPatientId, setSelectedPatientId] = useState<string>(patients[0]?.id ?? '');
  const [modpacks, setModpacks] = useState<ModpackRecord[]>([]);
  const [selectedModpackId, setSelectedModpackId] = useState('modpack-core-99');
  const [schedule, setSchedule] = useState<DailyScheduleDay[]>([]);
  const [delegateTargetPtId, setDelegateTargetPtId] = useState('');
  const [delegateNote, setDelegateNote] = useState('');
  const [notice, setNotice] = useState<string | null>(null);

  const selectedPatient = patients.find((p) => p.id === selectedPatientId) ?? null;
  const isLockedForPt = currentUser.role === 'pt' && !aiPoseDetectionEnabled;

  useEffect(() => {
    apiClient.listModpacks().then(setModpacks);
  }, [apiClient]);

  useEffect(() => {
    if (!selectedPatientId && patients[0]) {
      setSelectedPatientId(patients[0].id);
      return;
    }
    if (!selectedPatientId) return;
    apiClient.getPatientCourse(selectedPatientId).then((course) => {
      if (course) {
        setSchedule(course.weeklySchedule);
        setSelectedModpackId(course.cognitiveModpackId);
      }
    });
  }, [selectedPatientId, patients, apiClient]);

  const updateDay = (dayIndex: number, patch: Partial<DailyScheduleDay>) => {
    setSchedule((prev) => prev.map((d) => (d.dayIndex === dayIndex ? { ...d, ...patch } : d)));
  };

  const handleSaveSchedule = async () => {
    if (!selectedPatientId) return;
    try {
      await apiClient.upsertPatientCourse({
        patientId: selectedPatientId,
        title: `${selectedPatient?.fullName ?? 'Patient'} — แผนการรักษา`,
        weeklySchedule: schedule,
        cognitiveModpackId: selectedModpackId,
      });
      setNotice('บันทึกตารางนัดหมายและชุดมินิเกมลงฐานข้อมูลเรียบร้อย');
    } catch (err) {
      setNotice(err instanceof Error ? err.message : 'บันทึกตารางไม่สำเร็จ');
    }
  };

  const handleDelegateCase = async () => {
    if (!selectedPatientId || !delegateTargetPtId) return;
    await apiClient.delegatePatientCase(selectedPatientId, delegateTargetPtId, delegateNote);
    setNotice('ฝากเคสชั่วคราวให้เพื่อนนักกายภาพบำบัดเรียบร้อยแล้ว');
    onRefresh();
  };

  const handleReclaimCase = async () => {
    if (!selectedPatientId) return;
    await apiClient.delegatePatientCase(selectedPatientId, null);
    setNotice('ดึงสิทธิ์ดูแลเคสกลับมาที่ตนเองเรียบร้อยแล้ว');
    onRefresh();
  };

  return (
    <div className="space-y-6 animate-view-enter">
      {/* Case Handover: [ฝากเคสชั่วคราว] */}
      <div className="rounded-3xl bg-white p-6 shadow-md border border-slate-200 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2">
            <CalendarCheck2 className="h-5 w-5 text-teal-600" />
            ตารางนัด & พื้นที่ทำงาน (สั่งจ่ายตาราง + ฝากเคสชั่วคราว)
          </h2>

          <div className="flex flex-wrap items-center gap-2">
            <select
              value={selectedPatientId}
              onChange={(e) => setSelectedPatientId(e.target.value)}
              className="rounded-xl border border-slate-300 px-3 py-2 text-sm font-medium"
            >
              {patients.length === 0 && <option value="">ยังไม่มีคนไข้ในความดูแล</option>}
              {patients.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.hn} — {p.fullName} {p.delegatedToPtId ? '(ฝากเคสชั่วคราว)' : ''}
                </option>
              ))}
            </select>

            <select
              value={selectedModpackId}
              onChange={(e) => setSelectedModpackId(e.target.value)}
              className="rounded-xl border border-slate-300 px-3 py-2 text-sm font-medium"
            >
              {modpacks.map((m) => (
                <option key={m.id} value={m.id}>
                  ชุดมินิเกม: {m.name}
                </option>
              ))}
            </select>

            <button
              onClick={handleSaveSchedule}
              disabled={!selectedPatientId}
              className="inline-flex items-center gap-1.5 rounded-xl bg-teal-600 px-4 py-2 text-sm font-bold text-white hover:bg-teal-700 disabled:opacity-50"
            >
              <Save className="h-4 w-4" /> บันทึกตารางนัด
            </button>
          </div>
        </div>

        {isLockedForPt && (
          <div className="flex items-center gap-2 rounded-2xl bg-amber-50 border border-amber-200 px-4 py-2.5 text-xs font-bold text-amber-900 animate-toast-in">
            <Lock className="h-4 w-4 shrink-0" />
            ข้อจำกัด PT: สั่งจ่ายตารางฝึกได้เฉพาะ "มินิเกม" เท่านั้น (ท่ากายภาพถูกล็อกสถานะ "อยู่ระหว่างพัฒนา" จนกว่า Super Admin จะเปิดใช้งาน)
          </div>
        )}

        {notice && (
          <div className="rounded-xl bg-emerald-50 border border-emerald-200 px-4 py-2 text-xs font-bold text-emerald-800 animate-toast-in">
            {notice}
          </div>
        )}

        <div className="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-4">
          {schedule.map((day, idx) => (
            <div
              key={day.dayIndex}
              style={{ animationDelay: `${idx * 40}ms` }}
              className={`rounded-2xl border p-4 space-y-2 card-hover-lift animate-card-enter ${
                day.dayType === 'training' ? 'border-teal-200 bg-teal-50/40' : 'border-slate-200 bg-slate-50'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-sm font-bold text-slate-900">วัน{DAYS[day.dayIndex]}</span>
                <button
                  type="button"
                  onClick={() =>
                    updateDay(day.dayIndex, {
                      dayType: day.dayType === 'training' ? 'rest' : 'training',
                    })
                  }
                  className={`rounded-full px-2.5 py-0.5 text-[11px] font-bold ${
                    day.dayType === 'training' ? 'bg-teal-600 text-white' : 'bg-slate-300 text-slate-800'
                  }`}
                >
                  {day.dayType === 'training' ? 'วันฝึก' : 'วันพักฟื้น'}
                </button>
              </div>

              {day.dayType === 'training' && (
                <div className="space-y-2 text-xs">
                  <select
                    value={day.activityType}
                    onChange={(e) => updateDay(day.dayIndex, { activityType: e.target.value as ActivityType })}
                    className="w-full rounded-lg border border-slate-300 bg-white px-2 py-1.5 font-semibold"
                  >
                    <option value="minigame">มินิเกมฝึกสมอง</option>
                    <option value="physio_exercise" disabled={isLockedForPt}>
                      {isLockedForPt ? 'ท่ากายภาพ (อยู่ระหว่างพัฒนา - ล็อก)' : 'ท่ากายภาพบำบัดหน้ากล้อง'}
                    </option>
                  </select>

                  <select
                    value={day.difficulty}
                    onChange={(e) =>
                      updateDay(day.dayIndex, { difficulty: e.target.value as 'easy' | 'medium' | 'hard' })
                    }
                    className="w-full rounded-lg border border-slate-300 bg-white px-2 py-1.5"
                  >
                    <option value="easy">ระดับความยาก: ง่าย</option>
                    <option value="medium">ระดับความยาก: ปานกลาง</option>
                    <option value="hard">ระดับความยาก: ยาก</option>
                  </select>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Special Permission: [ฝากเคสชั่วคราว] */}
      {selectedPatient && (
        <div className="rounded-3xl bg-white p-6 shadow-md border border-slate-200">
          <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
            <ArrowRightLeft className="h-5 w-5 text-indigo-600" />
            สิทธิ์พิเศษ: [ฝากเคสชั่วคราว] โอนสิทธิ์ดูแลให้เพื่อน PT คนอื่น
          </h3>
          <div className="mt-3 flex flex-wrap items-center gap-3">
            <select
              value={delegateTargetPtId}
              onChange={(e) => setDelegateTargetPtId(e.target.value)}
              className="rounded-xl border border-slate-300 px-3 py-2 text-sm"
            >
              <option value="">-- เลือกนักกายภาพบำบัดที่จะฝากเคส --</option>
              {ptColleagues
                .filter((pt) => pt.id !== currentUser.id)
                .map((pt) => (
                  <option key={pt.id} value={pt.id}>
                    {pt.fullName} ({pt.specialty})
                  </option>
                ))}
            </select>
            <input
              value={delegateNote}
              onChange={(e) => setDelegateNote(e.target.value)}
              placeholder="หมายเหตุการฝากเคส (เช่น ลาพักร้อน 3 วัน)"
              className="flex-1 min-w-[200px] rounded-xl border border-slate-300 px-3 py-2 text-sm"
            />
            <button
              onClick={handleDelegateCase}
              disabled={!delegateTargetPtId}
              className="rounded-xl bg-indigo-600 px-4 py-2 text-sm font-bold text-white hover:bg-indigo-700 disabled:opacity-50"
            >
              [ฝากเคสชั่วคราว]
            </button>
            {selectedPatient.delegatedToPtId && (
              <button
                onClick={handleReclaimCase}
                className="rounded-xl border border-slate-300 px-4 py-2 text-sm font-bold text-slate-700 hover:bg-slate-50"
              >
                ดึงเคสกลับ
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
