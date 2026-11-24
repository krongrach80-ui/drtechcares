import fs from 'node:fs';
import path from 'node:path';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { GameDatePickerField } from '../components/GameDatePickerField';
import {
  buildCalendarGridDays,
  buildYearPageRange,
  calculateExactAgeFromIso,
  clampDayToMonth,
  formatIsoDateParts,
  formatRelativeDaysFromToday,
  formatThaiDisplayDate,
  formatThaiFullDate,
  parseIsoDateParts,
  resolveInitialPickerYear,
  shiftIsoDateByDays,
} from '../utils/gameDatePickerHelpers';

const EMOJI_RE = /[\p{Extended_Pictographic}\u{FE0F}\u{2600}-\u{27BF}\u{23E9}-\u{23FA}]/u;

describe('GameDatePickerField & Helpers', () => {
  it('calculates exact calendar age accurately before and after birthday in the current year', () => {
    const ref = new Date(2026, 10, 24); // 2026-11-24
    expect(calculateExactAgeFromIso('1981-11-24', ref)).toBe(45);
    expect(calculateExactAgeFromIso('1981-11-25', ref)).toBe(44);
    expect(calculateExactAgeFromIso('1981-11-23', ref)).toBe(45);
    expect(calculateExactAgeFromIso('1956-02-29', ref)).toBe(70);
    expect(calculateExactAgeFromIso('invalid', ref)).toBeNull();
  });

  it('formats Thai Buddhist Era (BE) and Gregorian (CE) labels cleanly with exact leap-year clamping', () => {
    expect(formatThaiDisplayDate('2026-11-24', 'BE')).toBe('24 พ.ย. 2569');
    expect(formatThaiDisplayDate('2026-11-24', 'CE')).toBe('24 พ.ย. 2026');
    expect(formatThaiFullDate('2026-11-24', 'BE')).toContain('24 พฤศจิกายน 2569');
    expect(clampDayToMonth(2025, 2, 31)).toBe(28);
    expect(clampDayToMonth(2024, 2, 31)).toBe(29);
    expect(formatIsoDateParts(2025, 2, 31)).toBe('2025-02-28');
    expect(parseIsoDateParts('2025-02-29')).toBeNull();
  });

  it('smartly jumps initial DOB year from ageHint and computes relative appointment labels', () => {
    const ref = new Date(2026, 10, 24);
    expect(resolveInitialPickerYear('', 'dob', 68, ref)).toBe(1958);
    expect(resolveInitialPickerYear('1975-06-15', 'dob', 68, ref)).toBe(1975);
    expect(resolveInitialPickerYear('', 'appointment', 68, ref)).toBe(2026);

    expect(formatRelativeDaysFromToday('2026-11-24', ref)).toBe('วันนี้');
    expect(formatRelativeDaysFromToday('2026-11-25', ref)).toBe('พรุ่งนี้');
    expect(formatRelativeDaysFromToday(shiftIsoDateByDays('2026-11-24', 7), ref)).toBe('อีก 7 วัน');
  });

  it('builds a fixed 42-cell calendar grid and 12-year pages with zero layout jump', () => {
    const ref = new Date(2026, 10, 24);
    const cells = buildCalendarGridDays(2026, 10, ref);
    expect(cells).toHaveLength(42);
    expect(cells.filter((c) => c.inCurrentMonth)).toHaveLength(30);
    expect(cells.filter((c) => c.isToday)).toHaveLength(1);
    expect(cells.find((c) => c.isToday)?.iso).toBe('2026-11-24');

    const years = buildYearPageRange(1981, 1911, 2026);
    expect(years).toHaveLength(12);
    expect(years).toContain(1981);
  });

  it('renders DOB and Appointment trigger cards cleanly via SSR with zero Unicode emojis', () => {
    const dobHtml = renderToStaticMarkup(
      React.createElement(GameDatePickerField, {
        mode: 'dob',
        value: '1981-11-24',
        onChange: () => {},
        testId: 'enroll-dob-picker',
      }),
    );
    expect(dobHtml).toContain('24 พ.ย. 2524');
    expect(dobHtml).toContain('ค.ศ. 1981');
    expect(dobHtml).toContain('data-testid="enroll-dob-picker"');
    expect(EMOJI_RE.test(dobHtml)).toBe(false);

    const apptHtml = renderToStaticMarkup(
      React.createElement(GameDatePickerField, {
        mode: 'appointment',
        compact: true,
        value: '2026-11-24',
        onChange: () => {},
        testId: 'enroll-schedule-date-picker',
      }),
    );
    expect(apptHtml).toContain('24 พ.ย. 2569');
    expect(EMOJI_RE.test(apptHtml)).toBe(false);

    for (const rel of [
      'src/utils/gameDatePickerHelpers.ts',
      'src/components/GameDatePickerField.tsx',
      'src/components/StaffAssistedEnrollmentModal.tsx',
      'src/__tests__/gameDatePicker.test.ts',
    ]) {
      const src = fs.readFileSync(path.resolve(process.cwd(), rel), 'utf8');
      expect(EMOJI_RE.test(src), `Unexpected emoji in ${rel}`).toBe(false);
    }
  });
});

