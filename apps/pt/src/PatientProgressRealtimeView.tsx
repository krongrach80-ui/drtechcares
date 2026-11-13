import React, { useEffect, useState } from 'react';
import type {
  DrTechCaresApiClient,
  MedicalNoteRecord,
  PatientRecord,
  RehabSessionRecord,
  StaffCallRecord,
  TreatmentReportRecord,
} from '@drtechcares/api-client';
import { Activity, BellRing, Check, FileText, Lock, Plus, TrendingUp } from 'lucide-react';

interface PatientProgressRealtimeViewProps {
  patients: PatientRecord[];
  apiClient: DrTechCaresApiClient;
  mode: 'notes' | 'results';
}

export function PatientProgressRealtimeView({
  patients,
  apiClient,
  mode,
}: PatientProgressRealtimeViewProps) {
  const [selectedPatientId, setSelectedPatientId] = useState(patients[0]?.id ?? '');
  const [notes, setNotes] = useState<MedicalNoteRecord[]>([]);
  const [newNote, setNewNote] = useState('');
  const [sessions, setSessions] = useState<RehabSessionRecord[]>([]);
  const [reports, setReports] = useState<TreatmentReportRecord[]>([]);
  const [calls, setCalls] = useState<StaffCallRecord[]>([]);
  const [reportSummary, setReportSummary] = useState('');
  const [reportAiRec, setReportAiRec] = useState('');

  useEffect(() => {
    if (!selectedPatientId && patients[0]) {
      setSelectedPatientId(patients[0].id);
    }
  }, [patients, selectedPatientId]);

  const loadData = async () => {
    const [s, r, c] = await Promise.all([
      apiClient.listRehabSessions(selectedPatientId || undefined),
      apiClient.listTreatmentReports(),
      apiClient.listStaffCalls(),
    ]);
    setSessions(s);
    setReports(r.filter((rep) => !selectedPatientId || rep.patientId === selectedPatientId));
    setCalls(c);
    if (selectedPatientId) {
      const n = await apiClient.listMedicalNotes(selectedPatientId);
      setNotes(n);
    }
  };

  useEffect(() => {
    loadData();
  }, [selectedPatientId]);

  const handleAppendNote = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedPatientId || !newNote.trim()) return;
    await apiClient.appendMedicalNote(selectedPatientId, newNote.trim());
    setNewNote('');
    loadData();
  };

  const handleCreateReport = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedPatientId || !reportSummary.trim()) return;
    await apiClient.createTreatmentReport({
      patientId: selectedPatientId,
      summary: reportSummary.trim(),
      aiRecommendation: reportAiRec.trim() || 'ติดตามฝึกต่อเนื่องตามตารางนัด',
      scoreSummary: `${sessions.length} sessions recorded`,
    });
    setReportSummary('');
    setReportAiRec('');
    loadData();
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl bg-white p-4 shadow-sm border border-slate-200">
        <span className="text-sm font-bold text-slate-700">เลือกคนไข้ในความดูแล:</span>
        <select
          value={selectedPatientId}
          onChange={(e) => setSelectedPatientId(e.target.value)}
          className="rounded-xl border border-slate-300 px-3 py-2 text-sm font-medium"
        >
          {patients.length === 0 && <option value="">ยังไม่มีคนไข้</option>}
          {patients.map((p) => (
            <option key={p.id} value={p.id}>
              {p.hn} — {p.fullName}
            </option>
          ))}
        </select>
      </div>

      {mode === 'notes' && (
        <div className="rounded-3xl bg-white p-6 shadow-md border border-slate-200 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
              <FileText className="h-5 w-5 text-teal-600" />
              เวชระเบียน (Append-Only Medical Notes)
            </h3>
            <span className="inline-flex items-center gap-1 rounded-full bg-slate-100 px-3 py-1 text-xs font-bold text-slate-600">
              <Lock className="h-3.5 w-3.5" /> ห้ามแก้ไขหรือลบโน้ตประวัติเก่า
            </span>
          </div>

          <form onSubmit={handleAppendNote} className="flex gap-2">
            <input
              value={newNote}
              onChange={(e) => setNewNote(e.target.value)}
              placeholder="เขียนบันทึกอาการ / โน้ตใหม่ต่อท้าย..."
              className="flex-1 rounded-xl border border-slate-300 px-4 py-2.5 text-sm"
              disabled={!selectedPatientId}
            />
            <button
              type="submit"
              disabled={!selectedPatientId || !newNote.trim()}
              className="inline-flex items-center gap-1.5 rounded-xl bg-teal-600 px-4 py-2.5 text-sm font-bold text-white hover:bg-teal-700 disabled:opacity-50"
            >
              <Plus className="h-4 w-4" /> เพิ่มโน้ตใหม่ต่อท้าย
            </button>
          </form>

          <div className="space-y-2.5">
            {notes.length === 0 && <p className="text-xs text-slate-500">ยังไม่มีบันทึกเวชระเบียน</p>}
            {notes.map((n) => (
              <div key={n.id} className="rounded-2xl border border-slate-200 bg-slate-50 p-4 text-sm">
                <div className="flex justify-between text-xs text-slate-500 mb-1">
                  <span className="font-bold text-slate-700">{n.authorName}</span>
                  <span>{new Date(n.createdAt).toLocaleString()}</span>
                </div>
                <p className="text-slate-900">{n.content}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {mode === 'results' && (
        <>
          {/* Live Kiosk Staff Calls */}
          <div className="rounded-3xl bg-white p-6 shadow-md border border-slate-200">
            <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
              <BellRing className="h-5 w-5 text-amber-500" />
              สัญญาณเรียกเจ้าหน้าที่จากตู้ Kiosk ({calls.filter((c) => c.status === 'open').length} Open)
            </h3>
            <div className="mt-3 space-y-2">
              {calls.length === 0 && <p className="text-xs text-slate-500">ไม่มีรายการเรียกเจ้าหน้าที่ค้างอยู่</p>}
              {calls.map((call) => (
                <div
                  key={call.id}
                  className="flex flex-wrap items-center justify-between gap-2 rounded-2xl border border-amber-200 bg-amber-50/70 px-4 py-2.5 text-xs"
                >
                  <div>
                    <span className="font-bold text-amber-900">{call.kioskId}</span> — {call.reason}
                  </div>
                  {call.status === 'open' ? (
                    <button
                      onClick={() => apiClient.resolveStaffCall(call.id).then(loadData)}
                      className="inline-flex items-center gap-1 rounded-lg bg-emerald-600 px-3 py-1 font-bold text-white"
                    >
                      <Check className="h-3.5 w-3.5" /> รับเรื่องแล้ว
                    </button>
                  ) : (
                    <span className="font-semibold text-emerald-700">ดำเนินการแล้ว</span>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Sessions & AI Results */}
          <div className="rounded-3xl bg-white p-6 shadow-md border border-slate-200">
            <h3 className="text-base font-bold text-slate-900 flex items-center gap-2 mb-3">
              <TrendingUp className="h-5 w-5 text-teal-600" />
              ผลการรักษา คะแนนความแม่นยำ & ผลวิเคราะห์จาก AI
            </h3>
            <div className="space-y-3">
              {sessions.length === 0 && <p className="text-xs text-slate-500">ยังไม่มีประวัติการฝึกของคนไข้รายนี้</p>}
              {sessions.map((s) => (
                <div key={s.id} className="rounded-2xl border border-slate-200 p-4 text-xs">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="font-bold text-slate-900 flex items-center gap-1.5">
                      <Activity className="h-4 w-4 text-teal-600" />
                      {s.exerciseName}
                    </span>
                    <span className="text-slate-500">{new Date(s.createdAt).toLocaleString()}</span>
                  </div>
                  <div className="mt-2 flex flex-wrap gap-4 text-slate-700">
                    <span>ความถูกต้อง: <b>{s.accuracyPercent}%</b></span>
                    <span>มุม ROM สูงสุด: <b>{s.maxRomDeg}°</b></span>
                    {s.minigameScore && (
                      <span>คะแนนมินิเกม: <b>{s.minigameScore.correct}/{s.minigameScore.total}</b></span>
                    )}
                  </div>
                  <p className="mt-1.5 text-teal-700 font-semibold">AI Analysis: {s.aiFeedback}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Create/Edit Reports */}
          <div className="rounded-3xl bg-white p-6 shadow-md border border-slate-200 space-y-4">
            <h3 className="text-base font-bold text-slate-900">รายงานผลการรักษา (สร้าง / แก้ไขสำหรับคนไข้ที่ดูแล)</h3>
            <form onSubmit={handleCreateReport} className="space-y-2.5">
              <input
                value={reportSummary}
                onChange={(e) => setReportSummary(e.target.value)}
                placeholder="สรุปผลการฟื้นฟู..."
                className="w-full rounded-xl border border-slate-300 px-3 py-2 text-sm"
                disabled={!selectedPatientId}
              />
              <input
                value={reportAiRec}
                onChange={(e) => setReportAiRec(e.target.value)}
                placeholder="คำแนะนำเพิ่มเติม..."
                className="w-full rounded-xl border border-slate-300 px-3 py-2 text-sm"
                disabled={!selectedPatientId}
              />
              <button
                type="submit"
                disabled={!selectedPatientId || !reportSummary.trim()}
                className="rounded-xl bg-teal-600 px-4 py-2 text-xs font-bold text-white hover:bg-teal-700 disabled:opacity-50"
              >
                บันทึกรายงานผล
              </button>
            </form>
            <div className="space-y-2">
              {reports.map((rep) => (
                <div key={rep.id} className="rounded-xl bg-slate-50 p-3 text-xs border border-slate-200">
                  <p className="font-bold text-slate-900">{rep.summary}</p>
                  <p className="text-slate-600 mt-1">คำแนะนำ: {rep.aiRecommendation}</p>
                  <p className="text-slate-400 mt-1">โดย {rep.ptName} • {new Date(rep.updatedAt).toLocaleString()}</p>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
