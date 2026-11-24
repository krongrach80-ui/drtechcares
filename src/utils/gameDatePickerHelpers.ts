/**
 * Pure date helpers for Dr.Tech.Care's Game-Style Interactive Date Picker Popup.
 * Supports Thai Buddhist Era (BE / พ.ศ.) + Gregorian (CE / ค.ศ.), exact calendar age sync,
 * smart DOB year jumping from an age hint, and 42-cell tactile calendar grids.
 * Zero external dependencies, zero Unicode emojis.
 */

export type DatePickerEra = 'BE' | 'CE';
export type GameDatePickerMode = 'dob' | 'appointment';

export interface ParsedIsoDate {
  year: number;
  month: number; // 1..12
  day: number; // 1..31
}

export interface CalendarGridCell {
  iso: string;
  year: number;
  month: number; // 1..12
  day: number;
  inCurrentMonth: boolean;
  isToday: boolean;
  isWeekend: boolean;
}

export const THAI_MONTHS_SHORT: readonly string[] = [
  'ม.ค.',
  'ก.พ.',
  'มี.ค.',
  'เม.ย.',
  'พ.ค.',
  'มิ.ย.',
  'ก.ค.',
  'ส.ค.',
  'ก.ย.',
  'ต.ค.',
  'พ.ย.',
  'ธ.ค.',
];

export const THAI_MONTHS_FULL: readonly string[] = [
  'มกราคม',
  'กุมภาพันธ์',
  'มีนาคม',
  'เมษายน',
  'พฤษภาคม',
  'มิถุนายน',
  'กรกฎาคม',
  'สิงหาคม',
  'กันยายน',
  'ตุลาคม',
  'พฤศจิกายน',
  'ธันวาคม',
];

export const THAI_WEEKDAYS_SHORT: readonly string[] = [
  'อา',
  'จ',
  'อ',
  'พ',
  'พฤ',
  'ศ',
  'ส',
];

export const THAI_WEEKDAYS_FULL: readonly string[] = [
  'วันอาทิตย์',
  'วันจันทร์',
  'วันอังคาร',
  'วันพุธ',
  'วันพฤหัสบดี',
  'วันศุกร์',
  'วันเสาร์',
];

export function pad2(n: number): string {
  return String(Math.max(0, Math.floor(n))).padStart(2, '0');
}

export function getDaysInMonth(year: number, month1To12: number): number {
  return new Date(year, month1To12, 0).getDate();
}

export function clampDayToMonth(year: number, month1To12: number, day: number): number {
  const maxDay = getDaysInMonth(year, month1To12);
  return Math.max(1, Math.min(maxDay, Math.floor(day)));
}

export function parseIsoDateParts(iso: string | null | undefined): ParsedIsoDate | null {
  if (!iso || typeof iso !== 'string') return null;
  const match = iso.trim().match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (!match) return null;
  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  if (
    !Number.isInteger(year) ||
    !Number.isInteger(month) ||
    !Number.isInteger(day) ||
    month < 1 ||
    month > 12 ||
    day < 1
  ) {
    return null;
  }
  const maxDay = getDaysInMonth(year, month);
  if (day > maxDay) return null;
  return { year, month, day };
}

export function formatIsoDateParts(year: number, month1To12: number, day: number): string {
  const safeMonth = Math.max(1, Math.min(12, Math.floor(month1To12)));
  const safeDay = clampDayToMonth(year, safeMonth, day);
  return `${String(year).padStart(4, '0')}-${pad2(safeMonth)}-${pad2(safeDay)}`;
}

export function getLocalTodayIso(referenceDate: Date = new Date()): string {
  return formatIsoDateParts(
    referenceDate.getFullYear(),
    referenceDate.getMonth() + 1,
    referenceDate.getDate(),
  );
}

export function shiftIsoDateByDays(baseIso: string, deltaDays: number): string {
  const parsed = parseIsoDateParts(baseIso);
  if (!parsed) return baseIso;
  const d = new Date(parsed.year, parsed.month - 1, parsed.day);
  d.setDate(d.getDate() + deltaDays);
  return formatIsoDateParts(d.getFullYear(), d.getMonth() + 1, d.getDate());
}

/**
 * Computes exact calendar age in years from an ISO birth date (`YYYY-MM-DD`).
 * Accounts accurately for whether the patient's birthday has already occurred this year.
 */
export function calculateExactAgeFromIso(
  birthIso: string,
  referenceDate: Date = new Date(),
): number | null {
  const parsed = parseIsoDateParts(birthIso);
  if (!parsed) return null;
  const refYear = referenceDate.getFullYear();
  const refMonth = referenceDate.getMonth() + 1;
  const refDay = referenceDate.getDate();

  let years = refYear - parsed.year;
  if (refMonth < parsed.month || (refMonth === parsed.month && refDay < parsed.day)) {
    years -= 1;
  }
  if (years < 0) return 0;
  return Math.min(120, Math.max(1, years));
}

export function formatYearForEra(yearCe: number, era: DatePickerEra = 'BE'): number {
  return era === 'BE' ? yearCe + 543 : yearCe;
}

export function formatThaiDisplayDate(
  iso: string | null | undefined,
  era: DatePickerEra = 'BE',
): string {
  const parsed = parseIsoDateParts(iso);
  if (!parsed) return '';
  const monthLabel = THAI_MONTHS_SHORT[parsed.month - 1] ?? '';
  const displayYear = formatYearForEra(parsed.year, era);
  return `${parsed.day} ${monthLabel} ${displayYear}`;
}

export function formatThaiFullDate(
  iso: string | null | undefined,
  era: DatePickerEra = 'BE',
): string {
  const parsed = parseIsoDateParts(iso);
  if (!parsed) return '';
  const dateObj = new Date(parsed.year, parsed.month - 1, parsed.day);
  const weekday = THAI_WEEKDAYS_FULL[dateObj.getDay()] ?? '';
  const monthFull = THAI_MONTHS_FULL[parsed.month - 1] ?? '';
  const displayYear = formatYearForEra(parsed.year, era);
  return `${weekday}ที่ ${parsed.day} ${monthFull} ${displayYear}`;
}

export function formatRelativeDaysFromToday(
  iso: string | null | undefined,
  referenceDate: Date = new Date(),
): string | null {
  const parsed = parseIsoDateParts(iso);
  if (!parsed) return null;
  const targetUtc = Date.UTC(parsed.year, parsed.month - 1, parsed.day);
  const todayUtc = Date.UTC(
    referenceDate.getFullYear(),
    referenceDate.getMonth(),
    referenceDate.getDate(),
  );
  const diffDays = Math.round((targetUtc - todayUtc) / (24 * 60 * 60 * 1000));
  if (diffDays === 0) return 'วันนี้';
  if (diffDays === 1) return 'พรุ่งนี้';
  if (diffDays === -1) return 'เมื่อวาน';
  if (diffDays > 1) return `อีก ${diffDays} วัน`;
  return `${Math.abs(diffDays)} วันที่แล้ว`;
}

/**
 * Picks the smartest initial year to display when opening the popup:
 * - If `valueIso` already has a date, opens directly to that date's year.
 * - In `dob` mode with an `ageHint` (e.g. `45` or `68`), jumps straight to `currentYear - ageHint`
 *   so clinicians never have to scroll back 60 years manually.
 */
export function resolveInitialPickerYear(
  valueIso: string | null | undefined,
  mode: GameDatePickerMode,
  ageHint?: number,
  referenceDate: Date = new Date(),
): number {
  const parsed = parseIsoDateParts(valueIso);
  if (parsed) return parsed.year;
  const currentYear = referenceDate.getFullYear();
  if (mode === 'dob' && typeof ageHint === 'number' && Number.isFinite(ageHint) && ageHint >= 1 && ageHint <= 120) {
    return currentYear - Math.round(ageHint);
  }
  return mode === 'dob' ? currentYear - 45 : currentYear;
}

/**
 * Builds a fixed 42-cell (6 weeks x 7 days) calendar matrix for smooth, zero-jump height
 * across every month transition.
 */
export function buildCalendarGridDays(
  year: number,
  monthIndex0: number,
  referenceDate: Date = new Date(),
): CalendarGridCell[] {
  const todayIso = getLocalTodayIso(referenceDate);
  const firstOfMonth = new Date(year, monthIndex0, 1);
  const leadingOffset = firstOfMonth.getDay(); // 0 = Sunday .. 6 = Saturday
  const startDate = new Date(year, monthIndex0, 1 - leadingOffset);

  const cells: CalendarGridCell[] = [];
  for (let i = 0; i < 42; i += 1) {
    const d = new Date(startDate.getFullYear(), startDate.getMonth(), startDate.getDate() + i);
    const cellYear = d.getFullYear();
    const cellMonth = d.getMonth() + 1;
    const cellDay = d.getDate();
    const iso = formatIsoDateParts(cellYear, cellMonth, cellDay);
    const dayOfWeek = d.getDay();
    cells.push({
      iso,
      year: cellYear,
      month: cellMonth,
      day: cellDay,
      inCurrentMonth: cellYear === year && cellMonth === monthIndex0 + 1,
      isToday: iso === todayIso,
      isWeekend: dayOfWeek === 0 || dayOfWeek === 6,
    });
  }
  return cells;
}

/**
 * Generates a 12-year page aligned around `anchorYear` clamped cleanly within `[minYear, maxYear]`.
 */
export function buildYearPageRange(
  anchorYear: number,
  minYear: number,
  maxYear: number,
): number[] {
  const safeMin = Math.min(minYear, maxYear);
  const safeMax = Math.max(minYear, maxYear);
  const clampedAnchor = Math.max(safeMin, Math.min(safeMax, anchorYear));
  const pageStart = Math.max(safeMin, Math.min(safeMax - 11, Math.floor(clampedAnchor / 12) * 12));
  const years: number[] = [];
  for (let y = pageStart; y < pageStart + 12 && y <= safeMax; y += 1) {
    years.push(y);
  }
  return years;
}

