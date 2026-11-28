import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import type { DrTechCaresApiClient, UserAccount } from '@drtechcares/api-client';
import { ClinicalI18nProvider } from '@drtechcares/clinical-i18n';
import {
  composeThaiFullName,
  computeEnrollmentReadiness,
  deriveElderlyMusclePowerLevelFromRom,
  ELDERLY_MIN_AGE,
  ELDERLY_MUSCLE_POWER_5_LEVELS,
  ELDERLY_THAI_NAME_PREFIX_OPTIONS,
  getElderlyMusclePowerSpec,
  resolveGenderForPrefix,
  resolvePrefixForGender,
  splitThaiFullName,
  THAI_NAME_PREFIX_OPTIONS,
} from '../utils/patientNameHelpers';
import { StaffAssistedEnrollmentModal } from '../components/StaffAssistedEnrollmentModal';

const EMOJI_PATTERN = /[\p{Extended_Pictographic}\u{1F300}-\u{1FAFF}]/u;

describe('Thai Patient Name Prefix, Split/Compose & Gender Sync Helpers', () => {
  it('exposes นาย, นางสาว, นาง, ด.ช., ด.ญ., and ไม่ระบุ prefix options in clear order', () => {
    const labels = THAI_NAME_PREFIX_OPTIONS.map((opt) => opt.label);
    expect(labels).toEqual(['นาย', 'นางสาว', 'นาง', 'ด.ช.', 'ด.ญ.', 'ไม่ระบุ']);
  });

  it('composes canonical Thai hospital full names without duplicating prefixes', () => {
    expect(composeThaiFullName('นาย', 'ธนกร', 'รุ่งเรือง')).toBe('นายธนกร รุ่งเรือง');
    expect(composeThaiFullName('นางสาว', 'กานดา', 'แสงทอง')).toBe('นางสาวกานดา แสงทอง');
    expect(composeThaiFullName('นาง', 'สมศรี', 'สุขใจ')).toBe('นางสมศรี สุขใจ');
    expect(composeThaiFullName('ด.ช.', 'ภูผา', 'ใจดี')).toBe('ด.ช. ภูผา ใจดี');
    expect(composeThaiFullName('ด.ญ.', 'ใบบัว', 'มีสุข')).toBe('ด.ญ. ใบบัว มีสุข');
    expect(composeThaiFullName('', 'สมชาย', 'ใจดี')).toBe('สมชาย ใจดี');
    // Deduplicate when operator types title inside first name
    expect(composeThaiFullName('นาย', 'นายธนกร', 'รุ่งเรือง')).toBe('นายธนกร รุ่งเรือง');
    expect(composeThaiFullName('นางสาว', 'น.ส. กานดา', 'แสงทอง')).toBe('นางสาวกานดา แสงทอง');
    // Latin first name keeps space after Thai title
    expect(composeThaiFullName('นาย', 'John', 'Doe')).toBe('นาย John Doe');
    expect(composeThaiFullName('นาย', '  ', '  ')).toBe('');
  });

  it('splits stored Thai full names back into prefix, firstName, and lastName accurately', () => {
    expect(splitThaiFullName('นายธนกร รุ่งเรือง')).toEqual({
      prefix: 'นาย',
      firstName: 'ธนกร',
      lastName: 'รุ่งเรือง',
    });
    // Must match นางสาว before นาง
    expect(splitThaiFullName('นางสาวกานดา แสงทอง')).toEqual({
      prefix: 'นางสาว',
      firstName: 'กานดา',
      lastName: 'แสงทอง',
    });
    expect(splitThaiFullName('น.ส. กานดา แสงทอง')).toEqual({
      prefix: 'นางสาว',
      firstName: 'กานดา',
      lastName: 'แสงทอง',
    });
    expect(splitThaiFullName('นางสมศรี ณ อยุธยา')).toEqual({
      prefix: 'นาง',
      firstName: 'สมศรี',
      lastName: 'ณ อยุธยา',
    });
    expect(splitThaiFullName('ด.ช. ภูผา ใจดี')).toEqual({
      prefix: 'ด.ช.',
      firstName: 'ภูผา',
      lastName: 'ใจดี',
    });
    expect(splitThaiFullName('สมชาย ใจดี')).toEqual({
      prefix: '',
      firstName: 'สมชาย',
      lastName: 'ใจดี',
    });
  });

  it('keeps Thai title prefix and gender synced bi-directionally', () => {
    expect(resolveGenderForPrefix('นาย', 'หญิง')).toBe('ชาย');
    expect(resolveGenderForPrefix('ด.ช.', 'หญิง')).toBe('ชาย');
    expect(resolveGenderForPrefix('นางสาว', 'ชาย')).toBe('หญิง');
    expect(resolveGenderForPrefix('นาง', 'ชาย')).toBe('หญิง');
    expect(resolveGenderForPrefix('ด.ญ.', 'ชาย')).toBe('หญิง');
    expect(resolveGenderForPrefix('', 'อื่นๆ')).toBe('อื่นๆ');

    expect(resolvePrefixForGender('หญิง', 'นาย')).toBe('นางสาว');
    expect(resolvePrefixForGender('หญิง', 'ด.ช.')).toBe('ด.ญ.');
    expect(resolvePrefixForGender('หญิง', 'นาง')).toBe('นาง');
    expect(resolvePrefixForGender('ชาย', 'นางสาว')).toBe('นาย');
    expect(resolvePrefixForGender('ชาย', 'นาง')).toBe('นาย');
    expect(resolvePrefixForGender('ชาย', 'ด.ญ.')).toBe('ด.ช.');
    expect(resolvePrefixForGender('อื่นๆ', 'นาย')).toBe('');
    expect(resolvePrefixForGender('ชาย', '')).toBe('');
  });

  it('computes live Step 1 readiness progress accurately', () => {
    const empty = computeEnrollmentReadiness({
      hn: 'HN-12345',
      firstName: '',
      lastName: '',
      birthDate: '',
      age: 45,
      phone: '',
      clinicalDiagnosis: '',
      emergencyName: '',
      pdpaAccepted: true,
    });
    expect(empty.readyForFaceScan).toBe(false);
    expect(empty.percent).toBe(48);

    const readyBasic = computeEnrollmentReadiness({
      hn: 'HN-12345',
      firstName: 'ธนกร',
      lastName: 'รุ่งเรือง',
      birthDate: '',
      age: 45,
      phone: '',
      clinicalDiagnosis: '',
      emergencyName: '',
      pdpaAccepted: true,
    });
    expect(readyBasic.readyForFaceScan).toBe(true);
    expect(readyBasic.percent).toBe(80);

    const fullComplete = computeEnrollmentReadiness({
      hn: 'HN-12345',
      firstName: 'ธนกร',
      lastName: 'รุ่งเรือง',
      birthDate: '1981-05-12',
      age: 45,
      phone: '0812345678',
      clinicalDiagnosis: 'ผู้สูงอายุแขนอ่อนแรง',
      emergencyName: 'สมศรี รุ่งเรือง',
      pdpaAccepted: true,
    });
    expect(fullComplete.readyForFaceScan).toBe(true);
    expect(fullComplete.percent).toBe(100);
  });

  it('defines elderly-only prefixes and a 5-level muscle power scale ending at 180 degrees', () => {
    expect(ELDERLY_MIN_AGE).toBe(60);
    expect(ELDERLY_THAI_NAME_PREFIX_OPTIONS.map((o) => o.label)).toEqual([
      'นาย',
      'นางสาว',
      'นาง',
      'ไม่ระบุ',
    ]);
    expect(ELDERLY_MUSCLE_POWER_5_LEVELS).toHaveLength(5);
    expect(ELDERLY_MUSCLE_POWER_5_LEVELS.map((l) => l.level)).toEqual([1, 2, 3, 4, 5]);
    expect(ELDERLY_MUSCLE_POWER_5_LEVELS[4].maxRomDeg).toBe(180);
    expect(getElderlyMusclePowerSpec(5).maxRomDeg).toBe(180);
    expect(deriveElderlyMusclePowerLevelFromRom(45)).toBe(1);
    expect(deriveElderlyMusclePowerLevelFromRom(85)).toBe(2);
    expect(deriveElderlyMusclePowerLevelFromRom(115)).toBe(3);
    expect(deriveElderlyMusclePowerLevelFromRom(145)).toBe(4);
    expect(deriveElderlyMusclePowerLevelFromRom(180)).toBe(5);
  });

  it('renders elderly-only Step 1 enrollment modal with 5-level 180° muscle scale, Chief Complaint, Diagnosis, Surgery History, and AI Arm ROM button', () => {
    const mockUser: UserAccount = {
      id: 'staff-1',
      username: 'admin',
      fullName: 'นพ.สมชาย ใจดี',
      role: 'super_admin',
      specialty: 'แพทย์เวชศาสตร์ฟื้นฟู',
      phone: '0810000000',
      active: true,
      createdAt: '2026-01-01T00:00:00.000Z',
    };
    const mockApiClient = {
      getSupabaseClient: () => null,
    } as unknown as DrTechCaresApiClient;

    const html = renderToStaticMarkup(
      React.createElement(
        ClinicalI18nProvider,
        null,
        React.createElement(StaffAssistedEnrollmentModal, {
          open: true,
          apiClient: mockApiClient,
          currentUser: mockUser,
          ptList: [mockUser],
          onClose: () => {},
          onEnrolled: () => {},
        }),
      ),
    );

    expect(html).toContain('data-testid="enroll-live-profile-card"');
    expect(html).toContain('data-testid="enroll-elderly-scope-badge"');
    expect(html).toContain('เฉพาะผู้สูงอายุ (60 ปีขึ้นไป)');
    expect(html).toContain('data-testid="enroll-prefix-bar"');
    expect(html).toContain('data-testid="enroll-prefix-นาย"');
    expect(html).toContain('data-testid="enroll-prefix-นางสาว"');
    expect(html).toContain('data-testid="enroll-prefix-นาง"');
    // Strictly elderly scope — child prefixes must not appear in enrollment modal
    expect(html).not.toContain('data-testid="enroll-prefix-ด.ช."');
    expect(html).not.toContain('data-testid="enroll-prefix-ด.ญ."');
    expect(html).toContain('data-testid="enroll-first-name-input"');
    expect(html).toContain('data-testid="enroll-last-name-input"');
    expect(html).toContain('data-testid="enroll-gender-bar"');
    expect(html).toContain('data-testid="enroll-dob-picker"');
    expect(html).toContain('data-testid="enroll-composed-name-preview"');
    // Chief Complaint (CC) full form
    expect(html).toContain('data-testid="enroll-chief-complaint-card"');
    expect(html).toContain('data-testid="enroll-chief-complaint-input"');
    expect(html).toContain('data-testid="enroll-cc-duration-input"');
    expect(html).toContain('data-testid="enroll-cc-pain-scale"');
    expect(html).toContain('Chief Complaint (CC)');
    // Dedicated Diagnosis & Surgical History fields
    expect(html).toContain('data-testid="enroll-diagnosis-input"');
    expect(html).toContain('การวินิจฉัยโรค (Diagnosis)');
    expect(html).toContain('data-testid="enroll-surgical-history-section"');
    expect(html).toContain('data-testid="enroll-surgical-history-input"');
    expect(html).toContain('ประวัติการผ่าตัด (Surgical History)');
    // 5-Level Muscle Power scale with level 5 = 180 deg
    expect(html).toContain('data-testid="enroll-muscle-power-scale"');
    expect(html).toContain('data-testid="enroll-muscle-level-1"');
    expect(html).toContain('data-testid="enroll-muscle-level-5"');
    expect(html).not.toContain('data-testid="enroll-muscle-level-6"');
    expect(html).toContain('180°');
    // Live AI Arm ROM measurement button
    expect(html).toContain('data-testid="enroll-arm-rom-panel"');
    expect(html).toContain('data-testid="enroll-arm-rom-input"');
    expect(html).toContain('data-testid="enroll-measure-arm-rom-btn"');
    expect(html).toContain('วัดองศาแขน (กล้อง AI)');
    // Elderly care & history taking
    expect(html).toContain('ข้อไหล่และแขนผู้สูงอายุ');
    expect(html).toContain('ผู้สูงอายุทั่วไป');
    expect(html).toContain('ผู้สูงอายุข้อเสื่อม');
    expect(html).toContain('data-testid="enroll-history-taking-card"');
    expect(html).toContain('data-testid="enroll-history-taking-input"');
    expect(html).toContain('ซักประวัติผู้สูงอายุ');
    expect(html).not.toContain('กำลังกล้ามเนื้อ มหิดล');
    expect(html).not.toContain('Grade 3/6');
    expect(html).not.toContain('NEW PLAYER PROFILE');
    expect(html).not.toContain('(Player Identity)');
    expect(EMOJI_PATTERN.test(html)).toBe(false);
  });

  it('keeps patient name helpers and enrollment modal strictly zero-emoji', () => {
    const root = process.cwd();
    const files = [
      'src/utils/patientNameHelpers.ts',
      'src/components/StaffAssistedEnrollmentModal.tsx',
    ];
    for (const rel of files) {
      const src = fs.readFileSync(path.join(root, rel), 'utf8');
      expect(EMOJI_PATTERN.test(src), `Unexpected emoji found in ${rel}`).toBe(false);
    }
  });
});
