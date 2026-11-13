import React, { useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import {
  Activity,
  AlertOctagon,
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  HeartPulse,
  Pause,
  Play,
  RotateCcw,
  SlidersHorizontal,
  Sparkles,
  Timer,
  UserCheck,
  Wifi,
  WifiOff,
} from 'lucide-react';
import { FaceScanOverlay } from './FaceScanOverlay';
import { RomGauge } from './RomGauge';
import { SkeletonCanvas } from './SkeletonCanvas';
import { VoiceCoachBanner } from './VoiceCoachBanner';
import {
  EXERCISE_OPTIONS,
  useKioskSocket,
  type DemoPresetId,
} from '../hooks/useKioskSocket';

interface SmartRehabKioskUIProps {
  onOpenStaffPortal?: () => void;
  onOpenLegacyKiosk?: () => void;
}

const DEMO_PRESETS: { id: DemoPresetId; label: string; shortLabel: string }[] = [
  { id: 'FACE_SCAN', label: 'Face Scan Login', shortLabel: '1. Face Scan' },
  { id: 'ACTIVE_REP', label: 'Active Exercise', shortLabel: '2. Active' },
  { id: 'TRUNK_LEAN', label: 'Cheating / Trunk Lean', shortLabel: '3. Trunk Lean' },
  { id: 'HOLD_PHASE', label: 'Hold Phase', shortLabel: '4. Hold' },
  { id: 'REP_COMPLETED', label: 'Rep Completed', shortLabel: '5. Rep +' },
  { id: 'SET_FINISHED', label: 'Set Finished', shortLabel: '6. Finished' },
];

export function SmartRehabKioskUI({
  onOpenStaffPortal,
  onOpenLegacyKiosk,
}: SmartRehabKioskUIProps) {
  const {
    payload,
    connectionStatus,
    isDemoMode,
    toggleDemoMode,
    selectedExercise,
    setSelectedExercise,
    demoPreset,
    setDemoPreset,
    demoRomDeg,
    setDemoRomDeg,
    demoTrunkTiltDeg,
    setDemoTrunkTiltDeg,
    demoDistanceM,
    setDemoDistanceM,
    demoStreakFrames,
    demoFaceMatched,
    autoCycleDemo,
    setAutoCycleDemo,
    triggerEmergencyStop,
    resetDemoSession,
  } = useKioskSocket('ws://localhost:8000/ws/kiosk');

  const [voiceMuted, setVoiceMuted] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(true);

  const { kiosk_state, patient, telemetry_analysis, rep_counter, ui_directives, tts_voice_th } = payload;
  const compensations = telemetry_analysis.compensation_detected;
  const trunkLeanAlert = compensations.includes('TRUNK_LEAN');
  const shoulderHikeAlert = compensations.includes('SHOULDER_HIKE');
  const kneeValgusAlert = compensations.includes('KNEE_VALGUS');
  const tooFastAlert = compensations.includes('TOO_FAST');
  const isHoldPhase = kiosk_state === 'HOLD_PHASE';
  const isRepCelebration = kiosk_state === 'REP_COMPLETED' || kiosk_state === 'SET_FINISHED';
  const isEmergency = kiosk_state === 'EMERGENCY_HALT';

  const patientRevealed = Boolean(patient?.name && (kiosk_state !== 'AUTH_FRAMING' || demoFaceMatched));

  const checklistItems = [
    {
      label: 'ลำตัวตั้งตรง ไม่เอียงชดเชย',
      ok: !trunkLeanAlert && !isEmergency,
      warnText: 'เอียงลำตัว',
    },
    {
      label:
        selectedExercise.target_joint === 'right_knee'
          ? 'เข่าตรงแนว ไม่บิดเข้าด้านใน'
          : 'ผ่อนหัวไหล่ ไม่ยักไหล่ขึ้น',
      ok:
        selectedExercise.target_joint === 'right_knee'
          ? !kneeValgusAlert && !isEmergency
          : !shoulderHikeAlert && !isEmergency,
      warnText: selectedExercise.target_joint === 'right_knee' ? 'เข่าบิดใน' : 'ยักไหล่',
    },
    {
      label: 'ความเร็วเคลื่อนไหวสม่ำเสมอ',
      ok: !tooFastAlert && !isEmergency,
      warnText: 'เร็วเกินไป',
    },
  ];

  return (
    <div className="clinical-mist-bg relative min-h-screen w-full overflow-x-hidden text-[#0F4C81]">
      {/* 1. Ambient Breathing Aurora Background */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <motion.div
          animate={{ x: [0, 36, -18, 0], y: [0, -24, 20, 0], scale: [1, 1.08, 0.96, 1] }}
          transition={{ repeat: Infinity, duration: 18, ease: 'easeInOut' }}
          className="absolute -left-24 -top-24 h-96 w-96 rounded-full bg-[#A7F3D0]/55 blur-3xl"
        />
        <motion.div
          animate={{ x: [0, -40, 24, 0], y: [0, 28, -16, 0], scale: [1, 0.94, 1.07, 1] }}
          transition={{ repeat: Infinity, duration: 22, ease: 'easeInOut' }}
          className="absolute -bottom-24 -right-24 h-[440px] w-[440px] rounded-full bg-[#BAE6FD]/60 blur-3xl"
        />
        <motion.div
          animate={{ opacity: [0.25, 0.45, 0.25] }}
          transition={{ repeat: Infinity, duration: 10, ease: 'easeInOut' }}
          className="absolute left-1/2 top-1/3 h-80 w-80 -translate-x-1/2 rounded-full bg-[#E0F2FE]/70 blur-3xl"
        />
      </div>

      {/* Main Container */}
      <div className="relative z-10 mx-auto flex min-h-screen max-w-[1600px] flex-col justify-between gap-4 p-3 sm:p-5">
        {/* 2. Top Status Bar (Glass Header) */}
        <header className="glass-card flex flex-wrap items-center justify-between gap-3 rounded-3xl px-4 py-3 sm:px-6">
          {/* Left: Logo + Edge Vision Connection Status */}
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-2.5">
              <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-gradient-to-br from-[#0284C7] to-[#10B981] text-white shadow-md">
                <HeartPulse className="h-6 w-6" />
              </div>
              <div>
                <h1 className="text-lg font-extrabold tracking-tight text-[#0F4C81] sm:text-xl">
                  DRTECH<span className="text-[#0D9488]">CARES</span>
                </h1>
                <p className="text-[11px] font-semibold text-[#0284C7]">Smart Rehabilitation Kiosk</p>
              </div>
            </div>

            <div
              className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-bold ${
                !isDemoMode && connectionStatus === 'connected'
                  ? 'border-emerald-200 bg-emerald-50 text-[#0D9488]'
                  : 'border-sky-200 bg-sky-50 text-[#0284C7]'
              }`}
            >
              <span className="relative flex h-2.5 w-2.5">
                <span
                  className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-75 ${
                    !isDemoMode && connectionStatus === 'connected' ? 'bg-[#10B981]' : 'bg-[#38BDF8]'
                  }`}
                />
                <span
                  className={`relative inline-flex h-2.5 w-2.5 rounded-full ${
                    !isDemoMode && connectionStatus === 'connected' ? 'bg-[#10B981]' : 'bg-[#0284C7]'
                  }`}
                />
              </span>
              {!isDemoMode && connectionStatus === 'connected' ? (
                <>
                  <Wifi className="h-3.5 w-3.5" />
                  <span>Connected to Edge Vision</span>
                </>
              ) : (
                <>
                  <WifiOff className="h-3.5 w-3.5" />
                  <span>{connectionStatus === 'connected' ? 'Demo Simulator (Live Paused)' : 'Demo Simulator Mode'}</span>
                </>
              )}
            </div>
          </div>

          {/* Center: Patient Profile Badge */}
          <AnimatePresence>
            {patientRevealed && patient && (
              <motion.div
                initial={{ opacity: 0, y: -10, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -10 }}
                className="flex items-center gap-3 rounded-2xl border border-emerald-200/90 bg-white/90 px-4 py-1.5 shadow-sm"
              >
                <div className="flex h-8 w-8 items-center justify-center rounded-full bg-emerald-100 text-[#0D9488]">
                  <UserCheck className="h-4 w-4" />
                </div>
                <div className="leading-tight">
                  <p className="text-sm font-bold text-[#0F4C81]">{patient.name}</p>
                  <p className="tabular-mono text-[11px] font-semibold text-[#0284C7]">
                    {patient.id} • ความคืบหน้า {rep_counter.valid_reps}/{selectedExercise.target_reps} ครั้ง
                  </p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Right: Demo Mode Switch + Portal Switch + Emergency Stop */}
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => toggleDemoMode()}
              className={`inline-flex items-center gap-2 rounded-2xl border px-3.5 py-2 text-xs font-bold transition-all sm:text-sm ${
                isDemoMode
                  ? 'border-[#38BDF8] bg-[#E0F2FE] text-[#0F4C81]'
                  : 'border-emerald-300 bg-emerald-50 text-[#0D9488]'
              }`}
            >
              <SlidersHorizontal className="h-4 w-4" />
              <span>{isDemoMode ? 'Demo Simulator: ON' : 'Live Edge: ON'}</span>
            </button>

            {onOpenLegacyKiosk && (
              <button
                type="button"
                onClick={onOpenLegacyKiosk}
                className="rounded-2xl border border-sky-200 bg-white/80 px-3 py-2 text-xs font-bold text-[#0F4C81] hover:bg-sky-50"
              >
                โหมดสแกนเต็มระบบ
              </button>
            )}

            {onOpenStaffPortal && (
              <button
                type="button"
                onClick={onOpenStaffPortal}
                className="rounded-2xl border border-sky-200 bg-white/80 px-3 py-2 text-xs font-bold text-[#0284C7] hover:bg-sky-50"
              >
                Staff Portal
              </button>
            )}

            <button
              type="button"
              onClick={triggerEmergencyStop}
              className="inline-flex items-center gap-2 rounded-2xl bg-[#EF4444] px-4 py-2 text-xs font-extrabold text-white shadow-md transition-transform hover:bg-red-600 active:scale-95 sm:text-sm"
            >
              <AlertOctagon className="h-4 w-4" />
              <span>หยุดฉุกเฉิน</span>
            </button>
          </div>
        </header>

        {/* 3. Main Clinical Workspace: 16:9 Landscape (~65% Left / ~35% Right), 9:16 Portrait Responsive */}
        <main className="grid flex-1 grid-cols-1 gap-4 lg:grid-cols-12">
          {/* Left Stage (~65% width) */}
          <section className="lg:col-span-8">
            <SkeletonCanvas
              payload={payload}
              exercise={selectedExercise}
              isDemoMode={isDemoMode}
              distanceM={demoDistanceM}
              trunkTiltDeg={demoTrunkTiltDeg}
            >
              {kiosk_state === 'AUTH_FRAMING' && (
                <FaceScanOverlay
                  verified={demoFaceMatched || Boolean(patient?.id)}
                  streakFrames={demoStreakFrames}
                  requiredFrames={5}
                  distanceM={demoDistanceM}
                  patientName={patient?.name}
                />
              )}

              {isEmergency && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="absolute inset-0 z-30 flex flex-col items-center justify-center bg-rose-950/75 p-6 text-center text-white backdrop-blur-sm"
                >
                  <AlertOctagon className="h-20 w-20 text-rose-400" />
                  <h2 className="mt-4 text-3xl font-extrabold">หยุดพักฉุกเฉิน (EMERGENCY HALT)</h2>
                  <p className="mt-2 max-w-md text-base text-rose-100">
                    ระบบตรวจพบสัญญาณหยุดพักฉุกเฉินและได้แจ้งเตือนเจ้าหน้าที่พยาบาลเรียบร้อยแล้ว
                  </p>
                  <button
                    type="button"
                    onClick={resetDemoSession}
                    className="mt-6 inline-flex items-center gap-2 rounded-2xl bg-white px-6 py-3 text-sm font-bold text-rose-700 shadow-lg hover:bg-rose-50"
                  >
                    <RotateCcw className="h-4 w-4" />
                    รีเซ็ตสถานะเพื่อฝึกต่อ
                  </button>
                </motion.div>
              )}
            </SkeletonCanvas>
          </section>

          {/* Right Telemetry Panel (~35% width) */}
          <aside className="flex flex-col gap-4 lg:col-span-4">
            {/* Card 1: Exercise Prescription Header */}
            <div className="glass-card rounded-3xl p-5">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <span className="inline-flex items-center gap-1.5 rounded-full bg-sky-100/90 px-3 py-1 text-[11px] font-bold text-[#0284C7]">
                    <Sparkles className="h-3.5 w-3.5" />
                    ท่าฝึกกายภาพที่ได้รับมอบหมาย
                  </span>
                  <h2 className="mt-2 text-xl font-extrabold text-[#0F4C81]">{selectedExercise.name_th}</h2>
                  <p className="text-xs font-semibold text-[#0284C7]">{selectedExercise.name_en}</p>
                </div>

                {/* Animated Miniature Guide Illustration */}
                <div className="flex h-16 w-16 shrink-0 items-center justify-center rounded-2xl border border-sky-100 bg-gradient-to-br from-sky-50 to-emerald-50">
                  <svg viewBox="0 0 64 64" className="h-12 w-12">
                    <circle cx="32" cy="14" r="5" fill="#0F4C81" />
                    <line x1="32" y1="19" x2="32" y2="40" stroke="#0F4C81" strokeWidth="3.5" strokeLinecap="round" />
                    <line x1="32" y1="40" x2="24" y2="56" stroke="#0284C7" strokeWidth="3" strokeLinecap="round" />
                    <line x1="32" y1="40" x2="40" y2="56" stroke="#0284C7" strokeWidth="3" strokeLinecap="round" />
                    <motion.line
                      x1="32"
                      y1="24"
                      x2="14"
                      y2="34"
                      stroke="#10B981"
                      strokeWidth="3.5"
                      strokeLinecap="round"
                      animate={{ x2: [18, 12, 18], y2: [38, 10, 38] }}
                      transition={{ repeat: Infinity, duration: 3, ease: 'easeInOut' }}
                    />
                  </svg>
                </div>
              </div>

              <div className="mt-3 flex flex-wrap gap-1.5">
                {EXERCISE_OPTIONS.map((opt) => {
                  const active = opt.exercise_id === selectedExercise.exercise_id;
                  return (
                    <button
                      key={opt.exercise_id}
                      type="button"
                      onClick={() => setSelectedExercise(opt)}
                      className={`rounded-xl px-2.5 py-1.5 text-xs font-bold transition-colors ${
                        active
                          ? 'bg-[#0F4C81] text-white shadow-sm'
                          : 'bg-sky-50 text-[#0F4C81] hover:bg-sky-100'
                      }`}
                    >
                      {opt.name_th}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Card 2: Hero ROM Angle Gauge */}
            <RomGauge
              currentRomDeg={telemetry_analysis.current_rom_deg}
              targetRomDeg={telemetry_analysis.target_rom_deg}
              completionPct={telemetry_analysis.rom_completion_pct}
              kioskState={kiosk_state}
              highlightColor={ui_directives.highlight_color}
              holdRemainingSec={rep_counter.hold_remaining_sec}
              holdTargetSec={selectedExercise.hold_sec}
            />

            {/* Card 3: Repetition & Hold Counter */}
            <div className="glass-card relative overflow-hidden rounded-3xl p-5">
              {/* Emerald Shockwave Ripple on Rep Completion */}
              <AnimatePresence>
                {isRepCelebration && (
                  <motion.div
                    key={`ripple-${rep_counter.valid_reps}`}
                    initial={{ scale: 0.75, opacity: 0.65 }}
                    animate={{ scale: 1.45, opacity: 0 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.85, ease: 'easeOut' }}
                    className="pointer-events-none absolute inset-0 rounded-3xl border-4 border-[#10B981]"
                  />
                )}
              </AnimatePresence>

              <div className="grid grid-cols-2 gap-3">
                {/* Valid Reps */}
                <div className="rounded-2xl border border-emerald-100 bg-gradient-to-br from-white to-emerald-50/70 p-3.5">
                  <p className="text-xs font-bold text-[#0D9488]">จำนวนครั้งที่ถูกต้อง</p>
                  <div className="mt-1 flex items-baseline gap-1.5">
                    <motion.span
                      key={rep_counter.valid_reps}
                      initial={{ y: -10, scale: 1.25, opacity: 0.4 }}
                      animate={{ y: 0, scale: 1, opacity: 1 }}
                      transition={{ type: 'spring', stiffness: 320, damping: 18 }}
                      className="tabular-mono text-3xl font-extrabold text-[#0F4C81]"
                    >
                      {rep_counter.valid_reps}
                    </motion.span>
                    <span className="tabular-mono text-sm font-bold text-[#0F4C81]/60">
                      / {selectedExercise.target_reps} ครั้ง
                    </span>
                  </div>
                </div>

                {/* Live Hold Timer */}
                <div
                  className={`rounded-2xl border p-3.5 transition-colors ${
                    isHoldPhase
                      ? 'border-emerald-300 bg-emerald-500 text-white'
                      : 'border-sky-100 bg-gradient-to-br from-white to-sky-50/70 text-[#0F4C81]'
                  }`}
                >
                  <p className={`flex items-center gap-1 text-xs font-bold ${isHoldPhase ? 'text-emerald-50' : 'text-[#0284C7]'}`}>
                    <Timer className="h-3.5 w-3.5" />
                    ค้างท่าเป้าหมาย
                  </p>
                  <div className="tabular-mono mt-1 text-3xl font-extrabold">
                    {rep_counter.hold_remaining_sec.toFixed(1)}s
                  </div>
                </div>
              </div>

              {/* Rep Visual Dots */}
              <div className="mt-3.5 flex items-center gap-1.5">
                {Array.from({ length: selectedExercise.target_reps }).map((_, idx) => {
                  const done = idx < rep_counter.valid_reps;
                  return (
                    <motion.div
                      key={idx}
                      animate={{ scale: done ? [1, 1.15, 1] : 1 }}
                      transition={{ duration: 0.25 }}
                      className={`h-2.5 flex-1 rounded-full transition-colors ${
                        done ? 'bg-[#10B981]' : 'bg-sky-100'
                      }`}
                    />
                  );
                })}
              </div>
            </div>

            {/* Card 4: Biomechanical Form Checklist */}
            <div className="glass-card rounded-3xl p-5">
              <div className="mb-3 flex items-center justify-between">
                <h3 className="flex items-center gap-2 text-sm font-extrabold text-[#0F4C81]">
                  <Activity className="h-4 w-4 text-[#0D9488]" />
                  การประเมินท่าทางแบบเรียลไทม์
                </h3>
                <span className="tabular-mono text-xs font-bold text-[#0284C7]">
                  เอียงลำตัว {demoTrunkTiltDeg.toFixed(1)}°
                </span>
              </div>

              <div className="space-y-2">
                {checklistItems.map((item) => (
                  <div
                    key={item.label}
                    className={`flex items-center justify-between rounded-2xl border px-3.5 py-2.5 text-sm font-semibold transition-colors ${
                      item.ok
                        ? 'border-emerald-100 bg-emerald-50/60 text-[#0F4C81]'
                        : 'border-amber-300 bg-amber-50 text-amber-900'
                    }`}
                  >
                    <span className="flex items-center gap-2">
                      {item.ok ? (
                        <CheckCircle2 className="h-4 w-4 shrink-0 text-[#10B981]" />
                      ) : (
                        <AlertTriangle className="h-4 w-4 shrink-0 text-[#F59E0B]" />
                      )}
                      {item.label}
                    </span>
                    <span
                      className={`rounded-full px-2 py-0.5 text-[11px] font-bold ${
                        item.ok ? 'bg-emerald-100 text-[#0D9488]' : 'bg-amber-200 text-amber-900'
                      }`}
                    >
                      {item.ok ? 'ผ่าน' : item.warnText}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </aside>
        </main>

        {/* 4. Bottom Floating Dock: Thai AI Coach + Demo Simulator Drawer */}
        <footer className="space-y-3">
          <VoiceCoachBanner
            textTh={tts_voice_th}
            kioskState={kiosk_state}
            highlightColor={ui_directives.highlight_color}
            muted={voiceMuted}
            onToggleMute={() => setVoiceMuted((m) => !m)}
          />

          {/* Interactive Demo Simulator Floating Control Drawer */}
          {isDemoMode && (
            <div className="glass-card rounded-3xl px-4 py-3 sm:px-6">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <button
                  type="button"
                  onClick={() => setDrawerOpen((o) => !o)}
                  className="flex items-center gap-2 text-xs font-extrabold uppercase tracking-wider text-[#0F4C81] sm:text-sm"
                >
                  <SlidersHorizontal className="h-4 w-4 text-[#0284C7]" />
                  <span>Demo Simulator Controls (พรีวิวสถานะ &amp; มุมข้อต่อสด)</span>
                  {drawerOpen ? <ChevronDown className="h-4 w-4" /> : <ChevronUp className="h-4 w-4" />}
                </button>

                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => setAutoCycleDemo((a) => !a)}
                    className={`inline-flex items-center gap-1.5 rounded-full px-3.5 py-1.5 text-xs font-bold transition-colors ${
                      autoCycleDemo
                        ? 'bg-[#10B981] text-white'
                        : 'bg-sky-100 text-[#0F4C81] hover:bg-sky-200'
                    }`}
                  >
                    {autoCycleDemo ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
                    <span>{autoCycleDemo ? 'หยุดจำลองอัตโนมัติ' : 'เล่นจำลองต่อเนื่อง'}</span>
                  </button>
                  <button
                    type="button"
                    onClick={resetDemoSession}
                    className="inline-flex items-center gap-1 rounded-full border border-sky-200 bg-white px-3 py-1.5 text-xs font-bold text-[#0F4C81] hover:bg-sky-50"
                  >
                    <RotateCcw className="h-3.5 w-3.5" />
                    รีเซ็ต
                  </button>
                </div>
              </div>

              {drawerOpen && (
                <div className="mt-3 space-y-3 border-t border-sky-100 pt-3">
                  {/* State Switch Pills */}
                  <div className="flex flex-wrap gap-1.5">
                    {DEMO_PRESETS.map((p) => {
                      const active = demoPreset === p.id;
                      return (
                        <button
                          key={p.id}
                          type="button"
                          onClick={() => setDemoPreset(p.id)}
                          className={`rounded-xl px-3 py-1.5 text-xs font-bold transition-all ${
                            active
                              ? p.id === 'TRUNK_LEAN'
                                ? 'bg-[#F59E0B] text-white shadow-sm'
                                : 'bg-[#0284C7] text-white shadow-sm'
                              : 'bg-white/80 text-[#0F4C81] hover:bg-sky-50'
                          }`}
                        >
                          {p.label}
                        </button>
                      );
                    })}
                  </div>

                  {/* Sliders: current_rom_deg, trunk_tilt_deg, distance_m */}
                  <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
                    <label className="flex flex-col gap-1">
                      <div className="flex justify-between text-xs font-bold">
                        <span>มุมข้อต่อ (current_rom_deg)</span>
                        <span className="tabular-mono text-[#0284C7]">{demoRomDeg.toFixed(0)}°</span>
                      </div>
                      <input
                        type="range"
                        min={0}
                        max={180}
                        step={1}
                        value={demoRomDeg}
                        onChange={(e) => {
                          setAutoCycleDemo(false);
                          setDemoRomDeg(Number(e.target.value));
                        }}
                        className="accent-[#0284C7]"
                      />
                    </label>

                    <label className="flex flex-col gap-1">
                      <div className="flex justify-between text-xs font-bold">
                        <span>เอียงลำตัว (trunk_tilt_deg)</span>
                        <span
                          className={`tabular-mono ${
                            demoTrunkTiltDeg > 10 ? 'text-[#F59E0B]' : 'text-[#0D9488]'
                          }`}
                        >
                          {demoTrunkTiltDeg.toFixed(1)}°
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0}
                        max={25}
                        step={0.5}
                        value={demoTrunkTiltDeg}
                        onChange={(e) => {
                          setAutoCycleDemo(false);
                          setDemoTrunkTiltDeg(Number(e.target.value));
                        }}
                        className="accent-[#F59E0B]"
                      />
                    </label>

                    <label className="flex flex-col gap-1">
                      <div className="flex justify-between text-xs font-bold">
                        <span>ระยะยืนหน้าตู้ (distance_m)</span>
                        <span className="tabular-mono text-[#0D9488]">{demoDistanceM.toFixed(2)} ม.</span>
                      </div>
                      <input
                        type="range"
                        min={1.4}
                        max={2.6}
                        step={0.05}
                        value={demoDistanceM}
                        onChange={(e) => setDemoDistanceM(Number(e.target.value))}
                        className="accent-[#10B981]"
                      />
                    </label>
                  </div>
                </div>
              )}
            </div>
          )}
        </footer>
      </div>
    </div>
  );
}

