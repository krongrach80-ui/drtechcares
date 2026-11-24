import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import {
  CalendarDays,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  RotateCcw,
  Sparkles,
  X,
} from 'lucide-react';
import { playGameSfx } from '../lib/gameUiSfx';
import {
  THAI_MONTHS_FULL,
  THAI_MONTHS_SHORT,
  THAI_WEEKDAYS_SHORT,
  buildCalendarGridDays,
  buildYearPageRange,
  calculateExactAgeFromIso,
  clampDayToMonth,
  formatIsoDateParts,
  formatRelativeDaysFromToday,
  formatThaiDisplayDate,
  formatThaiFullDate,
  formatYearForEra,
  getLocalTodayIso,
  parseIsoDateParts,
  resolveInitialPickerYear,
  shiftIsoDateByDays,
  type DatePickerEra,
  type GameDatePickerMode,
} from '../utils/gameDatePickerHelpers';

export interface GameDatePickerFieldProps {
  /** ISO date string (`YYYY-MM-DD`) or empty string. */
  value: string;
  /** Fires immediately on date selection for live form sync (e.g., DOB <-> Age). */
  onChange: (nextIso: string) => void;
  /** Optional callback fired when the user confirms a date (useful for appointment quick-add). */
  onConfirmDate?: (confirmedIso: string) => void;
  /** Visual preset mode: `'dob'` (age quick jumps + Buddhist era) or `'appointment'` (today/+N presets). */
  mode?: GameDatePickerMode;
  /** Optional current age value to smartly pre-focus the matching birth year when `value` is empty. */
  ageHint?: number;
  /** Placeholder text shown when `value` is empty. */
  placeholder?: string;
  /** Optional minimum selectable ISO date (`YYYY-MM-DD`). */
  minDate?: string;
  /** Optional maximum selectable ISO date (`YYYY-MM-DD`). */
  maxDate?: string;
  /** Highlight already-scheduled dates with an emerald dot indicator (for appointment mode). */
  highlightedDates?: readonly string[];
  /** Compact trigger styling (for inline toolbars like Step 3 custom appointment bar). */
  compact?: boolean;
  /** Optional label shown in the popup header. */
  popupTitle?: string;
  /** Optional test id for automated verification. */
  testId?: string;
  disabled?: boolean;
}

type PickerViewMode = 'day' | 'month' | 'year';

const DOB_AGE_PRESETS: readonly number[] = [30, 45, 60, 70, 80];

const APPOINTMENT_DAY_PRESETS: ReadonlyArray<{ label: string; deltaDays: number }> = [
  { label: 'วันนี้', deltaDays: 0 },
  { label: 'พรุ่งนี้', deltaDays: 1 },
  { label: '+3 วัน', deltaDays: 3 },
  { label: '+7 วัน', deltaDays: 7 },
  { label: '+14 วัน', deltaDays: 14 },
];

export function GameDatePickerField({
  value,
  onChange,
  onConfirmDate,
  mode = 'dob',
  ageHint,
  placeholder,
  minDate,
  maxDate,
  highlightedDates = [],
  compact = false,
  popupTitle,
  testId,
  disabled = false,
}: GameDatePickerFieldProps) {
  const todayIso = useMemo(() => getLocalTodayIso(), []);
  const currentYear = useMemo(() => new Date().getFullYear(), []);
  const minYear = mode === 'dob' ? currentYear - 115 : currentYear - 2;
  const maxYear = mode === 'dob' ? currentYear : currentYear + 5;

  const [open, setOpen] = useState(false);
  const [era, setEra] = useState<DatePickerEra>('BE');
  const [viewMode, setViewMode] = useState<PickerViewMode>('day');
  const [viewYear, setViewYear] = useState<number>(() =>
    resolveInitialPickerYear(value, mode, ageHint),
  );
  const [viewMonth0, setViewMonth0] = useState<number>(() => {
    const parsed = parseIsoDateParts(value);
    return parsed ? parsed.month - 1 : new Date().getMonth();
  });
  const [yearPageAnchor, setYearPageAnchor] = useState<number>(() =>
    resolveInitialPickerYear(value, mode, ageHint),
  );
  const [draftIso, setDraftIso] = useState<string>(value);
  const [gridAnimKey, setGridAnimKey] = useState(0);
  const [justSyncedFlash, setJustSyncedFlash] = useState(false);

  const initialValueBeforeOpenRef = useRef<string>(value);
  const syncFlashTimerRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (syncFlashTimerRef.current !== null) {
        window.clearTimeout(syncFlashTimerRef.current);
      }
    };
  }, []);

  const triggerSyncFlash = useCallback(() => {
    setJustSyncedFlash(true);
    if (syncFlashTimerRef.current !== null) {
      window.clearTimeout(syncFlashTimerRef.current);
    }
    syncFlashTimerRef.current = window.setTimeout(() => {
      syncFlashTimerRef.current = null;
      setJustSyncedFlash(false);
    }, 520);
  }, []);

  const highlightedSet = useMemo(() => new Set(highlightedDates), [highlightedDates]);

  const isIsoOutOfRange = useCallback(
    (iso: string): boolean => {
      if (minDate && iso < minDate) return true;
      if (maxDate && iso > maxDate) return true;
      return false;
    },
    [minDate, maxDate],
  );

  const openPopup = useCallback(() => {
    if (disabled) return;
    playGameSfx('tap');
    initialValueBeforeOpenRef.current = value;
    const parsed = parseIsoDateParts(value);
    const startYear = resolveInitialPickerYear(value, mode, ageHint);
    const startMonth0 = parsed ? parsed.month - 1 : new Date().getMonth();
    setDraftIso(value);
    setViewYear(startYear);
    setViewMonth0(startMonth0);
    setYearPageAnchor(startYear);
    setViewMode('day');
    setGridAnimKey((k) => k + 1);
    setOpen(true);
  }, [disabled, value, mode, ageHint]);

  const handleCancel = useCallback(() => {
    playGameSfx('tap');
    const prev = initialValueBeforeOpenRef.current;
    if (prev !== draftIso) {
      onChange(prev);
    }
    setOpen(false);
  }, [draftIso, onChange]);

  const handleConfirm = useCallback(
    (explicitIso?: string) => {
      const targetIso = explicitIso ?? draftIso;
      if (targetIso && !isIsoOutOfRange(targetIso)) {
        if (targetIso !== value) {
          onChange(targetIso);
        }
        onConfirmDate?.(targetIso);
        playGameSfx('unlock');
        triggerSyncFlash();
      } else {
        playGameSfx('tap');
      }
      setOpen(false);
    },
    [draftIso, isIsoOutOfRange, value, onChange, onConfirmDate, triggerSyncFlash],
  );

  useEffect(() => {
    if (!open || typeof window === 'undefined') return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        handleCancel();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [open, handleCancel]);

  const selectDateLive = useCallback(
    (iso: string, closeImmediately = false) => {
      if (isIsoOutOfRange(iso)) return;
      const parsed = parseIsoDateParts(iso);
      if (!parsed) return;
      setDraftIso(iso);
      setViewYear(parsed.year);
      setViewMonth0(parsed.month - 1);
      setYearPageAnchor(parsed.year);
      onChange(iso);
      triggerSyncFlash();
      if (closeImmediately) {
        onConfirmDate?.(iso);
        playGameSfx('unlock');
        setOpen(false);
      } else {
        playGameSfx('correct');
      }
    },
    [isIsoOutOfRange, onChange, onConfirmDate, triggerSyncFlash],
  );

  const handleClear = useCallback(
    (e?: React.MouseEvent) => {
      e?.stopPropagation();
      playGameSfx('tap');
      setDraftIso('');
      onChange('');
      if (open) {
        initialValueBeforeOpenRef.current = '';
        setOpen(false);
      }
    },
    [onChange, open],
  );

  const stepPrev = useCallback(() => {
    playGameSfx('tap');
    setGridAnimKey((k) => k + 1);
    if (viewMode === 'year') {
      setYearPageAnchor((prev) => Math.max(minYear, prev - 12));
      return;
    }
    if (viewMode === 'month') {
      setViewYear((prev) => Math.max(minYear, prev - 1));
      return;
    }
    if (viewMonth0 === 0) {
      if (viewYear > minYear) {
        setViewYear((y) => y - 1);
        setViewMonth0(11);
      }
    } else {
      setViewMonth0((m) => m - 1);
    }
  }, [viewMode, minYear, viewMonth0, viewYear]);

  const stepNext = useCallback(() => {
    playGameSfx('tap');
    setGridAnimKey((k) => k + 1);
    if (viewMode === 'year') {
      setYearPageAnchor((prev) => Math.min(maxYear, prev + 12));
      return;
    }
    if (viewMode === 'month') {
      setViewYear((prev) => Math.min(maxYear, prev + 1));
      return;
    }
    if (viewMonth0 === 11) {
      if (viewYear < maxYear) {
        setViewYear((y) => y + 1);
        setViewMonth0(0);
      }
    } else {
      setViewMonth0((m) => m + 1);
    }
  }, [viewMode, maxYear, viewMonth0, viewYear]);

  const handleSelectMonth = useCallback(
    (monthIdx0: number) => {
      playGameSfx('tap');
      setViewMonth0(monthIdx0);
      setViewMode('day');
      setGridAnimKey((k) => k + 1);
      const parsed = parseIsoDateParts(draftIso);
      if (parsed) {
        const nextIso = formatIsoDateParts(viewYear, monthIdx0 + 1, parsed.day);
        if (!isIsoOutOfRange(nextIso)) {
          setDraftIso(nextIso);
          onChange(nextIso);
        }
      }
    },
    [draftIso, viewYear, isIsoOutOfRange, onChange],
  );

  const handleSelectYear = useCallback(
    (pickedYear: number) => {
      playGameSfx('tap');
      setViewYear(pickedYear);
      setYearPageAnchor(pickedYear);
      setViewMode('month');
      setGridAnimKey((k) => k + 1);
      const parsed = parseIsoDateParts(draftIso);
      if (parsed) {
        const clampedDay = clampDayToMonth(pickedYear, parsed.month, parsed.day);
        const nextIso = formatIsoDateParts(pickedYear, parsed.month, clampedDay);
        if (!isIsoOutOfRange(nextIso)) {
          setDraftIso(nextIso);
          onChange(nextIso);
        }
      }
    },
    [draftIso, isIsoOutOfRange, onChange],
  );

  const handleDobAgeQuickJump = useCallback(
    (targetAge: number) => {
      const targetYear = Math.max(minYear, Math.min(maxYear, currentYear - targetAge));
      const parsed = parseIsoDateParts(draftIso);
      const targetMonth = parsed ? parsed.month : new Date().getMonth() + 1;
      const targetDay = parsed ? clampDayToMonth(targetYear, targetMonth, parsed.day) : 15;
      const nextIso = formatIsoDateParts(targetYear, targetMonth, targetDay);
      setViewMode('day');
      setGridAnimKey((k) => k + 1);
      selectDateLive(nextIso, false);
    },
    [minYear, maxYear, currentYear, draftIso, selectDateLive],
  );

  const calendarCells = useMemo(
    () => buildCalendarGridDays(viewYear, viewMonth0),
    [viewYear, viewMonth0],
  );

  const yearChoices = useMemo(
    () => buildYearPageRange(yearPageAnchor, minYear, maxYear),
    [yearPageAnchor, minYear, maxYear],
  );

  const activeDisplayIso = open ? draftIso : value;
  const triggerThaiLabel = formatThaiDisplayDate(value, 'BE');
  const triggerParsed = parseIsoDateParts(value);
  const triggerAge = mode === 'dob' && value ? calculateExactAgeFromIso(value) : null;
  const heroFullThai = formatThaiFullDate(activeDisplayIso, era);
  const heroAge = mode === 'dob' && activeDisplayIso ? calculateExactAgeFromIso(activeDisplayIso) : null;
  const heroRelative =
    mode === 'appointment' && activeDisplayIso ? formatRelativeDaysFromToday(activeDisplayIso) : null;

  const headerTitle =
    popupTitle ?? (mode === 'dob' ? 'เลือกวันเดือนปีเกิดผู้ป่วย' : 'เลือกวันนัดฝึกกายภาพบำบัด');

  const resolvedPlaceholder =
    placeholder ?? (mode === 'dob' ? 'แตะเพื่อเลือกวันเกิด (พ.ศ. / ค.ศ.)' : 'เลือกวันนัดฝึก');

  const modalPortal =
    open && typeof document !== 'undefined'
      ? createPortal(
          <div
            className="game-datepicker-backdrop fixed inset-0 z-[120] flex items-center justify-center bg-slate-900/55 backdrop-blur-sm p-4 sm:p-6"
            role="dialog"
            aria-modal="true"
            aria-label={headerTitle}
            data-testid={testId ? `${testId}-popup` : 'game-datepicker-popup'}
            onMouseDown={(e) => {
              if (e.target === e.currentTarget) handleCancel();
            }}
          >
            <div className="game-datepicker-modal relative w-full max-w-[600px] sm:max-w-[620px] max-h-[94dvh] overflow-y-auto rounded-[32px] border-2 border-sky-200/90 bg-white shadow-[0_32px_84px_-18px_rgba(15,76,129,0.56)]">
              {/* Top Arcade-Clinical Header HUD */}
              <div className="relative overflow-hidden bg-gradient-to-br from-[#0F4C81] via-[#0284C7] to-[#059669] px-6 pt-5 pb-5 text-white">
                {/* Subtle animated diagonal stripe shimmer */}
                <div
                  aria-hidden="true"
                  className="pointer-events-none absolute inset-0 opacity-20 game-progress-stripes"
                />
                <div
                  aria-hidden="true"
                  className="pointer-events-none absolute -right-12 -top-12 h-40 w-40 rounded-full bg-emerald-300/25 blur-2xl"
                />

                <div className="relative flex items-start justify-between gap-3">
                  <div className="flex items-center gap-3 min-w-0">
                    <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-white/15 ring-1 ring-white/30 backdrop-blur-xs shadow-inner">
                      <CalendarDays className="h-6 w-6 text-white" />
                    </div>
                    <div className="min-w-0">
                      <div className="flex items-center gap-1.5">
                        <span className="inline-flex items-center gap-1 rounded-full bg-white/20 px-2.5 py-0.5 text-[11px] font-black tracking-wide uppercase text-emerald-100">
                          <Sparkles className="h-3 w-3" />
                          {mode === 'dob' ? 'DOB SYNC' : 'SCHEDULE PICKER'}
                        </span>
                      </div>
                      <h3 className="mt-1 truncate text-lg sm:text-xl font-black tracking-tight text-white">
                        {headerTitle}
                      </h3>
                    </div>
                  </div>

                  <div className="flex items-center gap-2 shrink-0">
                    {/* Instant Buddhist Era (พ.ศ.) / Gregorian (ค.ศ.) Toggle */}
                    <div
                      className="inline-flex rounded-2xl bg-slate-950/25 p-1 ring-1 ring-white/25"
                      role="group"
                      aria-label="สลับปฏิทิน พ.ศ. หรือ ค.ศ."
                    >
                      <button
                        type="button"
                        onClick={() => {
                          playGameSfx('tap');
                          setEra('BE');
                        }}
                        className={`rounded-xl px-3 py-1.5 text-xs font-black transition cursor-pointer ${
                          era === 'BE'
                            ? 'bg-white text-[#0F4C81] shadow-xs'
                            : 'text-sky-100 hover:text-white'
                        }`}
                      >
                        พ.ศ.
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          playGameSfx('tap');
                          setEra('CE');
                        }}
                        className={`rounded-xl px-3 py-1.5 text-xs font-black transition cursor-pointer ${
                          era === 'CE'
                            ? 'bg-white text-[#0F4C81] shadow-xs'
                            : 'text-sky-100 hover:text-white'
                        }`}
                      >
                        ค.ศ.
                      </button>
                    </div>

                    <button
                      type="button"
                      onClick={handleCancel}
                      aria-label="ปิดหน้าต่างเลือกวันที่"
                      className="flex h-10 w-10 items-center justify-center rounded-2xl bg-white/15 text-white transition hover:bg-white/25 active:scale-90 cursor-pointer"
                    >
                      <X className="h-5 w-5" />
                    </button>
                  </div>
                </div>

                {/* Live Synced Hero Date Preview */}
                <div className="relative mt-4 rounded-2xl border border-white/25 bg-slate-950/20 px-4 py-3.5 sm:px-5 backdrop-blur-xs">
                  <div className="flex items-center justify-between gap-3">
                    <div className="min-w-0">
                      <div className="text-xs font-bold uppercase tracking-wider text-sky-100/90">
                        {activeDisplayIso ? 'วันที่ที่เลือก (ซิงก์อัตโนมัติ)' : 'ยังไม่ได้เลือกวันที่'}
                      </div>
                      <div
                        key={activeDisplayIso || 'empty'}
                        className="mt-1 truncate text-xl sm:text-2xl font-black text-white game-combo-pop"
                      >
                        {heroFullThai || 'แตะวันที่ด้านล่างเพื่อเลือก'}
                      </div>
                    </div>
                    {heroAge !== null && (
                      <div className="shrink-0 rounded-2xl bg-emerald-400/95 px-4 py-2 text-center shadow-sm">
                        <div className="text-[10px] font-extrabold uppercase text-emerald-950/80 leading-none">
                          อายุจริง
                        </div>
                        <div className="mt-0.5 text-lg sm:text-xl font-black text-emerald-950 leading-tight">
                          {heroAge} ปี
                        </div>
                      </div>
                    )}
                    {heroRelative && (
                      <div className="shrink-0 rounded-2xl bg-emerald-400/95 px-4 py-2 text-sm sm:text-base font-black text-emerald-950 shadow-sm">
                        {heroRelative}
                      </div>
                    )}
                  </div>
                  {activeDisplayIso && (
                    <div className="mt-2 flex items-center gap-2.5 text-xs sm:text-sm font-semibold text-sky-100/95">
                      <span className="rounded-lg bg-white/15 px-2.5 py-0.5 font-mono text-xs font-bold text-white">
                        {activeDisplayIso}
                      </span>
                      <span>
                        {era === 'BE'
                          ? `ค.ศ. ${parseIsoDateParts(activeDisplayIso)?.year ?? ''}`
                          : `พ.ศ. ${(parseIsoDateParts(activeDisplayIso)?.year ?? 0) + 543}`}
                      </span>
                    </div>
                  )}
                </div>
              </div>

              {/* Body */}
              <div className="p-5 sm:p-6 space-y-4 bg-gradient-to-b from-slate-50/80 to-white">
                {/* Context-Aware Game Quick-Jump Chips */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs font-extrabold tracking-wide text-slate-500">
                    <span>{mode === 'dob' ? 'ทางลัดช่วงอายุ (แตะเพื่อกระโดดปี)' : 'ทางลัดวันนัดฝึก'}</span>
                    <span className="text-teal-600 font-bold">
                      {mode === 'dob' ? 'ซิงก์ช่องอายุทันที' : 'เลือกแล้วกดยืนยัน'}
                    </span>
                  </div>
                  <div className="grid grid-cols-5 gap-2">
                    {mode === 'dob'
                      ? DOB_AGE_PRESETS.map((presetAge) => {
                          const isNearAge = heroAge !== null && Math.abs(heroAge - presetAge) <= 1;
                          return (
                            <button
                              key={presetAge}
                              type="button"
                              onClick={() => handleDobAgeQuickJump(presetAge)}
                              className={`rounded-2xl border px-2 py-2 text-xs sm:text-sm font-extrabold transition cursor-pointer active:scale-95 ${
                                isNearAge
                                  ? 'border-emerald-500 bg-emerald-500 text-white shadow-[0_3px_0_#047857]'
                                  : 'border-sky-200 bg-white text-[#0F4C81] hover:border-emerald-400 hover:bg-emerald-50'
                              }`}
                            >
                              อายุ {presetAge} ปี
                            </button>
                          );
                        })
                      : APPOINTMENT_DAY_PRESETS.map((preset) => {
                          const iso = shiftIsoDateByDays(todayIso, preset.deltaDays);
                          const isSelected = draftIso === iso;
                          return (
                            <button
                              key={preset.label}
                              type="button"
                              onClick={() => selectDateLive(iso, false)}
                              className={`rounded-2xl border px-2 py-2 text-xs sm:text-sm font-extrabold transition cursor-pointer active:scale-95 ${
                                isSelected
                                  ? 'border-emerald-500 bg-emerald-500 text-white shadow-[0_3px_0_#047857]'
                                  : 'border-sky-200 bg-white text-[#0F4C81] hover:border-emerald-400 hover:bg-emerald-50'
                              }`}
                            >
                              {preset.label}
                            </button>
                          );
                        })}
                  </div>
                </div>

                {/* Month / Year Stage Navigator Bar */}
                <div className="flex items-center justify-between gap-2.5 rounded-2xl border border-slate-200/90 bg-white px-2.5 py-2 shadow-xs">
                  <button
                    type="button"
                    onClick={stepPrev}
                    aria-label="ก่อนหน้า"
                    className="btn-game-3d-white flex h-11 w-11 items-center justify-center rounded-2xl border border-slate-200 bg-white text-slate-700 transition hover:border-sky-300 hover:text-[#0F4C81] cursor-pointer"
                  >
                    <ChevronLeft className="h-5 w-5 stroke-[2.6]" />
                  </button>

                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => {
                        playGameSfx('tap');
                        setViewMode((prev) => (prev === 'month' ? 'day' : 'month'));
                        setGridAnimKey((k) => k + 1);
                      }}
                      className={`inline-flex items-center gap-1.5 rounded-2xl px-4 py-2 text-sm sm:text-base font-black transition cursor-pointer active:scale-95 ${
                        viewMode === 'month'
                          ? 'bg-[#0F4C81] text-white shadow-xs'
                          : 'bg-sky-50 text-[#0F4C81] hover:bg-sky-100'
                      }`}
                    >
                      <span>{THAI_MONTHS_FULL[viewMonth0]}</span>
                      <ChevronDown
                        className={`h-4 w-4 transition-transform ${
                          viewMode === 'month' ? 'rotate-180' : ''
                        }`}
                      />
                    </button>

                    <button
                      type="button"
                      onClick={() => {
                        playGameSfx('tap');
                        setYearPageAnchor(viewYear);
                        setViewMode((prev) => (prev === 'year' ? 'day' : 'year'));
                        setGridAnimKey((k) => k + 1);
                      }}
                      className={`inline-flex items-center gap-1.5 rounded-2xl px-4 py-2 text-sm sm:text-base font-black transition cursor-pointer active:scale-95 ${
                        viewMode === 'year'
                          ? 'bg-emerald-600 text-white shadow-xs'
                          : 'bg-emerald-50 text-emerald-800 hover:bg-emerald-100'
                      }`}
                    >
                      <span>
                        {era === 'BE' ? `พ.ศ. ${formatYearForEra(viewYear, 'BE')}` : `ค.ศ. ${viewYear}`}
                      </span>
                      <ChevronDown
                        className={`h-4 w-4 transition-transform ${
                          viewMode === 'year' ? 'rotate-180' : ''
                        }`}
                      />
                    </button>
                  </div>

                  <button
                    type="button"
                    onClick={stepNext}
                    aria-label="ถัดไป"
                    className="btn-game-3d-white flex h-11 w-11 items-center justify-center rounded-2xl border border-slate-200 bg-white text-slate-700 transition hover:border-sky-300 hover:text-[#0F4C81] cursor-pointer"
                  >
                    <ChevronRight className="h-5 w-5 stroke-[2.6]" />
                  </button>
                </div>

                {/* Animated Interactive Grid Area (Day / Month / Year) */}
                <div
                  key={`${viewMode}-${gridAnimKey}`}
                  className="game-datepicker-grid-enter min-h-[348px] sm:min-h-[388px]"
                >
                  {viewMode === 'day' && (
                    <div className="space-y-2">
                      <div className="grid grid-cols-7 gap-2 px-0.5">
                        {THAI_WEEKDAYS_SHORT.map((wk, idx) => (
                          <div
                            key={wk}
                            className={`py-1.5 text-center text-xs sm:text-sm font-black ${
                              idx === 0
                                ? 'text-rose-500'
                                : idx === 6
                                  ? 'text-sky-600'
                                  : 'text-slate-400'
                            }`}
                          >
                            {wk}
                          </div>
                        ))}
                      </div>

                      <div className="grid grid-cols-7 gap-2">
                        {calendarCells.map((cell) => {
                          const isSelected = draftIso === cell.iso;
                          const isDisabled = isIsoOutOfRange(cell.iso);
                          const isBooked = highlightedSet.has(cell.iso);
                          return (
                            <button
                              key={cell.iso}
                              type="button"
                              disabled={isDisabled}
                              onClick={() => selectDateLive(cell.iso, false)}
                              onDoubleClick={() => selectDateLive(cell.iso, true)}
                              aria-pressed={isSelected}
                              className={`relative flex h-12 sm:h-14 w-full flex-col items-center justify-center rounded-2xl text-base sm:text-lg font-black transition-all duration-150 ${
                                isDisabled
                                  ? 'cursor-not-allowed text-slate-300 bg-slate-50/50'
                                  : isSelected
                                    ? 'cursor-pointer bg-gradient-to-br from-emerald-500 to-teal-600 text-white shadow-[0_4px_0_#047857,0_10px_20px_-6px_rgba(5,150,105,0.55)] scale-[1.05] z-10'
                                    : cell.isToday
                                      ? 'cursor-pointer border-2 border-[#0284C7] bg-sky-50 text-[#0F4C81] hover:-translate-y-0.5 hover:bg-sky-100 active:scale-90'
                                      : cell.inCurrentMonth
                                        ? 'cursor-pointer border border-slate-200/80 bg-white text-slate-700 hover:-translate-y-0.5 hover:border-emerald-300 hover:bg-emerald-50/70 hover:text-emerald-900 active:scale-90'
                                        : 'cursor-pointer text-slate-400/65 hover:bg-slate-100 active:scale-90'
                              }`}
                            >
                              <span>{cell.day}</span>
                              {cell.isToday && !isSelected && (
                                <span
                                  aria-hidden="true"
                                  className="absolute bottom-1.5 h-1.5 w-1.5 rounded-full bg-[#0284C7]"
                                />
                              )}
                              {isBooked && !isSelected && (
                                <span
                                  aria-hidden="true"
                                  className="absolute right-1.5 top-1.5 h-2 w-2 rounded-full bg-emerald-500 ring-1 ring-white"
                                />
                              )}
                              {isSelected && (
                                <span
                                  aria-hidden="true"
                                  className="game-star-twinkle absolute -right-0.5 -top-0.5 h-2.5 w-2.5 rounded-full bg-amber-300 ring-2 ring-white"
                                />
                              )}
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {viewMode === 'month' && (
                    <div className="grid grid-cols-3 gap-2.5 pt-2">
                      {THAI_MONTHS_FULL.map((mFull, idx) => {
                        const isActiveMonth = idx === viewMonth0;
                        return (
                          <button
                            key={mFull}
                            type="button"
                            onClick={() => handleSelectMonth(idx)}
                            className={`rounded-2xl border px-3 py-4.5 text-center transition-all cursor-pointer active:scale-95 ${
                              isActiveMonth
                                ? 'border-emerald-600 bg-gradient-to-br from-emerald-500 to-teal-600 text-white shadow-[0_4px_0_#047857]'
                                : 'border-slate-200 bg-white text-slate-800 hover:-translate-y-0.5 hover:border-sky-300 hover:bg-sky-50'
                            }`}
                          >
                            <div className="text-base font-black">{THAI_MONTHS_SHORT[idx]}</div>
                            <div
                              className={`mt-1 text-xs font-semibold ${
                                isActiveMonth ? 'text-emerald-50' : 'text-slate-500'
                              }`}
                            >
                              {mFull}
                            </div>
                          </button>
                        );
                      })}
                    </div>
                  )}

                  {viewMode === 'year' && (
                    <div className="space-y-3 pt-1">
                      <div className="text-center text-xs sm:text-sm font-bold text-slate-500">
                        เลือกปี ({era === 'BE' ? 'พ.ศ.' : 'ค.ศ.'}) — กดลูกศรซ้าย/ขวาเพื่อเปลี่ยนช่วงปี
                      </div>
                      <div className="grid grid-cols-3 gap-2.5">
                        {yearChoices.map((yr) => {
                          const isActiveYear = yr === viewYear;
                          const approxAge = mode === 'dob' ? Math.max(0, currentYear - yr) : null;
                          return (
                            <button
                              key={yr}
                              type="button"
                              onClick={() => handleSelectYear(yr)}
                              className={`rounded-2xl border px-3 py-3.5 text-center transition-all cursor-pointer active:scale-95 ${
                                isActiveYear
                                  ? 'border-[#0F4C81] bg-gradient-to-br from-[#0F4C81] to-[#0284C7] text-white shadow-[0_4px_0_#0b3960]'
                                  : 'border-slate-200 bg-white text-slate-800 hover:-translate-y-0.5 hover:border-emerald-300 hover:bg-emerald-50'
                              }`}
                            >
                              <div className="text-lg font-black">
                                {formatYearForEra(yr, era)}
                              </div>
                              <div
                                className={`mt-0.5 text-xs font-semibold ${
                                  isActiveYear ? 'text-sky-100' : 'text-slate-400'
                                }`}
                              >
                                {mode === 'dob' && approxAge !== null
                                  ? `${era === 'BE' ? `ค.ศ. ${yr}` : `พ.ศ. ${yr + 543}`} • ~${approxAge} ปี`
                                  : era === 'BE'
                                    ? `ค.ศ. ${yr}`
                                    : `พ.ศ. ${yr + 543}`}
                              </div>
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>

                {/* Bottom Action Footer */}
                <div className="flex flex-wrap items-center justify-between gap-2.5 border-t border-slate-100 pt-4">
                  <div className="flex items-center gap-2">
                    {activeDisplayIso && (
                      <button
                        type="button"
                        onClick={() => handleClear()}
                        className="inline-flex items-center gap-1.5 rounded-2xl border border-slate-200 bg-white px-3.5 py-2.5 text-xs sm:text-sm font-bold text-slate-600 transition hover:border-rose-200 hover:bg-rose-50 hover:text-rose-700 active:scale-95 cursor-pointer"
                      >
                        <RotateCcw className="h-3.5 w-3.5" />
                        ล้างค่า
                      </button>
                    )}
                    {!isIsoOutOfRange(todayIso) && (
                      <button
                        type="button"
                        onClick={() => selectDateLive(todayIso, false)}
                        className="rounded-2xl border border-sky-200 bg-sky-50 px-3.5 py-2.5 text-xs sm:text-sm font-extrabold text-[#0F4C81] transition hover:bg-sky-100 active:scale-95 cursor-pointer"
                      >
                        วันนี้
                      </button>
                    )}
                  </div>

                  <div className="flex items-center gap-2.5">
                    <button
                      type="button"
                      onClick={handleCancel}
                      className="rounded-2xl border border-slate-200 bg-white px-4 py-2.5 text-sm font-bold text-slate-600 transition hover:bg-slate-50 active:scale-95 cursor-pointer"
                    >
                      ยกเลิก
                    </button>
                    <button
                      type="button"
                      disabled={!draftIso || isIsoOutOfRange(draftIso)}
                      onClick={() => handleConfirm()}
                      className="btn-game-3d-emerald relative overflow-hidden inline-flex items-center gap-2 rounded-2xl bg-gradient-to-r from-emerald-500 to-teal-600 px-6 py-3 text-sm sm:text-base font-black text-white transition hover:brightness-105 disabled:opacity-45 disabled:shadow-none cursor-pointer"
                    >
                      <Check className="h-4 w-4 stroke-[3]" />
                      <span>ยืนยันวันที่นี้</span>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>,
          document.body,
        )
      : null;

  if (compact) {
    return (
      <>
        <button
          type="button"
          disabled={disabled}
          data-testid={testId}
          onClick={openPopup}
          className={`btn-game-3d-white group inline-flex items-center gap-2 rounded-xl border border-sky-200 bg-white px-3 py-1.5 text-[11px] font-extrabold text-[#0F4C81] transition hover:border-emerald-400 hover:bg-emerald-50/40 cursor-pointer disabled:opacity-60 ${
            justSyncedFlash ? 'game-age-sync-glow' : ''
          }`}
        >
          <span className="flex h-5 w-5 items-center justify-center rounded-lg bg-gradient-to-br from-[#0F4C81] to-[#0284C7] text-white shadow-xs transition-transform group-hover:scale-110">
            <CalendarDays className="h-3 w-3" />
          </span>
          <span className="truncate">
            {triggerThaiLabel || resolvedPlaceholder}
          </span>
          <ChevronDown className="h-3 w-3 text-slate-400 transition-transform group-hover:translate-y-0.5" />
        </button>
        {modalPortal}
      </>
    );
  }

  return (
    <>
      <div className="relative">
        <button
          type="button"
          disabled={disabled}
          data-testid={testId}
          onClick={openPopup}
          className={`group relative flex w-full items-center justify-between gap-2 overflow-hidden border-2 text-left transition-all duration-200 cursor-pointer active:scale-[0.985] disabled:opacity-60 ${
            compact ? 'h-9 rounded-xl px-2.5 py-0.5' : 'h-12 rounded-xl px-3.5 py-2'
          } ${
            value
              ? 'border-emerald-300/90 bg-gradient-to-r from-white via-emerald-50/40 to-sky-50/50 shadow-[0_3px_0_rgba(16,185,129,0.16)] hover:border-emerald-400 hover:shadow-[0_4px_14px_-6px_rgba(16,185,129,0.45)]'
              : 'border-slate-200 bg-slate-50/80 shadow-[0_2px_0_rgba(148,163,184,0.2)] hover:border-sky-400 hover:bg-white'
          } ${justSyncedFlash ? 'game-age-sync-glow' : ''}`}
        >
          <div className="flex min-w-0 items-center gap-2.5">
            <span
              className={`flex shrink-0 items-center justify-center transition-transform duration-200 group-hover:scale-110 group-hover:-rotate-3 ${
                compact ? 'h-6 w-6 rounded-lg' : 'h-8 w-8 rounded-xl'
              } ${
                value
                  ? 'bg-gradient-to-br from-emerald-500 to-teal-600 text-white shadow-xs'
                  : 'bg-gradient-to-br from-[#0F4C81] to-[#0284C7] text-white shadow-xs'
              }`}
            >
              <CalendarDays className={compact ? 'h-3.5 w-3.5' : 'h-4.5 w-4.5'} />
            </span>

            <div className="min-w-0">
              {value ? (
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className={`truncate font-black text-slate-900 ${compact ? 'text-xs' : 'text-sm sm:text-base'}`}>
                    {triggerThaiLabel}
                  </span>
                  {!compact && triggerParsed && (
                    <span className="rounded-md bg-sky-100/90 px-1.5 py-0.5 font-mono text-[11px] font-bold text-[#0F4C81]">
                      ค.ศ. {triggerParsed.year}
                    </span>
                  )}
                </div>
              ) : (
                <span className={`block truncate font-bold text-slate-400 ${compact ? 'text-xs' : 'text-sm sm:text-base'}`}>
                  {resolvedPlaceholder}
                </span>
              )}
            </div>
          </div>

          <div className="flex shrink-0 items-center gap-1.5">
            {!compact && triggerAge !== null && (
              <span className="rounded-full bg-emerald-100 border border-emerald-200 px-2.5 py-0.5 text-xs font-black text-emerald-800">
                {triggerAge} ปี
              </span>
            )}
            <span
              className={`flex items-center justify-center rounded-lg bg-white/80 text-slate-500 ring-1 ring-slate-200/80 transition group-hover:bg-sky-50 group-hover:text-[#0F4C81] ${
                compact ? 'h-5 w-5' : 'h-7 w-7'
              }`}
            >
              <ChevronDown className={compact ? 'h-3 w-3' : 'h-4 w-4'} />
            </span>
          </div>
        </button>

        {value && !disabled && (
          <button
            type="button"
            onClick={handleClear}
            aria-label="ล้างวันที่"
            className="absolute -right-1.5 -top-1.5 flex h-5 w-5 items-center justify-center rounded-full border border-slate-200 bg-white text-slate-400 shadow-xs transition hover:border-rose-300 hover:bg-rose-50 hover:text-rose-600 active:scale-90 cursor-pointer"
          >
            <X className="h-3 w-3" />
          </button>
        )}
      </div>

      {modalPortal}
    </>
  );
}

