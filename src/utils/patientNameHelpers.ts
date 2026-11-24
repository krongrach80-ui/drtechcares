export type ThaiNamePrefix = 'นาย' | 'นางสาว' | 'นาง' | 'ด.ช.' | 'ด.ญ.' | '';

export type PatientGenderTh = 'ชาย' | 'หญิง' | 'อื่นๆ';

export interface ThaiNamePrefixOption {
  value: ThaiNamePrefix;
  label: string;
  subLabel: string;
  syncedGender: PatientGenderTh | null;
}

export const THAI_NAME_PREFIX_OPTIONS: readonly ThaiNamePrefixOption[] = [
  { value: 'นาย', label: 'นาย', subLabel: 'ชาย', syncedGender: 'ชาย' },
  { value: 'นางสาว', label: 'นางสาว', subLabel: 'หญิง', syncedGender: 'หญิง' },
  { value: 'นาง', label: 'นาง', subLabel: 'หญิง', syncedGender: 'หญิง' },
  { value: 'ด.ช.', label: 'ด.ช.', subLabel: 'เด็กชาย', syncedGender: 'ชาย' },
  { value: 'ด.ญ.', label: 'ด.ญ.', subLabel: 'เด็กหญิง', syncedGender: 'หญิง' },
  { value: '', label: 'ไม่ระบุ', subLabel: 'กำหนดเอง', syncedGender: null },
] as const;

export const PATIENT_GENDER_OPTIONS: ReadonlyArray<{
  value: PatientGenderTh;
  label: string;
  enLabel: string;
}> = [
  { value: 'ชาย', label: 'ชาย', enLabel: 'Male' },
  { value: 'หญิง', label: 'หญิง', enLabel: 'Female' },
  { value: 'อื่นๆ', label: 'อื่นๆ', enLabel: 'Other' },
];

export const QUICK_CLINICAL_DIAGNOSIS_PRESETS: readonly string[] = [
  'Stroke แขนอ่อนแรง',
  'ไหล่ติดผู้สูงอายุ',
  'ฟื้นฟูแขน-มือ ADL',
  'ปวดคอ-บ่า สธ.',
] as const;

export const QUICK_EMERGENCY_RELATION_PRESETS: readonly string[] = [
  'บุตร',
  'คู่สมรส',
  'พี่น้อง',
  'ผู้ดูแล',
] as const;

interface PrefixMatchRule {
  canonical: Exclude<ThaiNamePrefix, ''>;
  pattern: RegExp;
}

// Longest / most specific prefixes MUST appear before shorter ones (e.g. นางสาว before นาง).
const PREFIX_MATCH_RULES: readonly PrefixMatchRule[] = [
  { canonical: 'นางสาว', pattern: /^(?:นางสาว|น\.ส\.)\s*/u },
  { canonical: 'ด.ช.', pattern: /^(?:ด\.ช\.|เด็กชาย)\s*/u },
  { canonical: 'ด.ญ.', pattern: /^(?:ด\.ญ\.|เด็กหญิง)\s*/u },
  { canonical: 'นาย', pattern: /^นาย\s*/u },
  { canonical: 'นาง', pattern: /^นาง\s*/u },
];

function stripDuplicateLeadingPrefix(rawFirstName: string, activePrefix: ThaiNamePrefix): string {
  const trimmed = rawFirstName.trim();
  if (!trimmed || !activePrefix) return trimmed;
  for (const rule of PREFIX_MATCH_RULES) {
    if (rule.pattern.test(trimmed)) {
      const stripped = trimmed.replace(rule.pattern, '').trim();
      if (stripped.length > 0) {
        return stripped;
      }
    }
  }
  return trimmed;
}

function isThaiCharacterStart(text: string): boolean {
  return /^[\u0E00-\u0E7F]/.test(text);
}

/**
 * Composes canonical Thai hospital full name from separate prefix, first name, and last name.
 * - Attaches 'นาย', 'นาง', 'นางสาว' directly before Thai first names (e.g., "นายธนกร รุ่งเรือง").
 * - Adds a single space after 'ด.ช.' / 'ด.ญ.' or before Latin names (e.g., "ด.ช. ภูผา ใจดี").
 * - Deduplicates if the operator accidentally typed the title inside `firstName`.
 */
export function composeThaiFullName(
  prefix: ThaiNamePrefix,
  firstName: string,
  lastName: string,
): string {
  const cleanFirst = stripDuplicateLeadingPrefix(firstName, prefix).replace(/\s+/g, ' ');
  const cleanLast = lastName.trim().replace(/\s+/g, ' ');

  if (!cleanFirst && !cleanLast) {
    return '';
  }

  let givenPart = cleanFirst;
  if (prefix) {
    if (cleanFirst) {
      const needsSpace = prefix.endsWith('.') || !isThaiCharacterStart(cleanFirst);
      givenPart = needsSpace ? `${prefix} ${cleanFirst}` : `${prefix}${cleanFirst}`;
    } else {
      givenPart = prefix;
    }
  }

  if (givenPart && cleanLast) {
    return `${givenPart} ${cleanLast}`;
  }
  return givenPart || cleanLast;
}

export interface SplitThaiNameResult {
  prefix: ThaiNamePrefix;
  firstName: string;
  lastName: string;
}

/**
 * Splits a stored Thai full name back into `{ prefix, firstName, lastName }`
 * so editing or re-enrolling an existing patient pre-fills all three fields accurately.
 */
export function splitThaiFullName(fullName: string): SplitThaiNameResult {
  const normalized = fullName.trim().replace(/\s+/g, ' ');
  if (!normalized) {
    return { prefix: 'นาย', firstName: '', lastName: '' };
  }

  let prefix: ThaiNamePrefix = '';
  let remainder = normalized;

  for (const rule of PREFIX_MATCH_RULES) {
    if (rule.pattern.test(remainder)) {
      const candidate = remainder.replace(rule.pattern, '').trim();
      if (candidate.length > 0) {
        prefix = rule.canonical;
        remainder = candidate;
      }
      break;
    }
  }

  const tokens = remainder.split(' ').filter(Boolean);
  if (tokens.length === 0) {
    return { prefix, firstName: '', lastName: '' };
  }
  if (tokens.length === 1) {
    return { prefix, firstName: tokens[0], lastName: '' };
  }
  return {
    prefix,
    firstName: tokens[0],
    lastName: tokens.slice(1).join(' '),
  };
}

/**
 * Resolves the synchronized gender when the operator selects a Thai title prefix.
 */
export function resolveGenderForPrefix(
  prefix: ThaiNamePrefix,
  currentGender: PatientGenderTh,
): PatientGenderTh {
  const option = THAI_NAME_PREFIX_OPTIONS.find((item) => item.value === prefix);
  return option?.syncedGender ?? currentGender;
}

/**
 * Resolves the synchronized Thai title prefix when the operator manually changes gender.
 * Keeps `'ไม่ระบุ'` (`''`) untouched when toggling between `'ชาย'` and `'หญิง'` so custom names stay intact.
 */
export function resolvePrefixForGender(
  gender: PatientGenderTh,
  currentPrefix: ThaiNamePrefix,
): ThaiNamePrefix {
  if (gender === 'อื่นๆ') {
    return '';
  }
  if (gender === 'ชาย') {
    if (currentPrefix === 'ด.ญ.') return 'ด.ช.';
    if (currentPrefix === 'นาง' || currentPrefix === 'นางสาว') return 'นาย';
    return currentPrefix;
  }
  if (currentPrefix === 'ด.ช.') return 'ด.ญ.';
  if (currentPrefix === 'นาย') return 'นางสาว';
  return currentPrefix;
}

export interface EnrollmentReadinessInput {
  hn: string;
  firstName: string;
  lastName: string;
  birthDate: string;
  age: number;
  phone: string;
  clinicalDiagnosis: string;
  emergencyName: string;
  pdpaAccepted: boolean;
}

export interface EnrollmentReadinessSummary {
  percent: number;
  readyForFaceScan: boolean;
  completedRequired: number;
  totalRequired: number;
  bonusCompleted: number;
  bonusTotal: number;
  checks: {
    hasHn: boolean;
    hasFirstName: boolean;
    hasLastName: boolean;
    hasValidAge: boolean;
    hasPdpa: boolean;
    hasBirthDate: boolean;
    hasFullPhone: boolean;
    hasDiagnosis: boolean;
    hasEmergency: boolean;
  };
}

/**
 * Computes live Step 1 gamified readiness progress bar and checklist badges.
 */
export function computeEnrollmentReadiness(
  input: EnrollmentReadinessInput,
): EnrollmentReadinessSummary {
  const hasHn = input.hn.trim().length >= 3;
  const hasFirstName = input.firstName.trim().length > 0;
  const hasLastName = input.lastName.trim().length > 0;
  const hasValidAge = Number.isFinite(input.age) && input.age >= 1 && input.age <= 120;
  const hasPdpa = Boolean(input.pdpaAccepted);

  const hasBirthDate = Boolean(input.birthDate.trim());
  const hasFullPhone = input.phone.replace(/\D/g, '').length === 10;
  const hasDiagnosis = input.clinicalDiagnosis.trim().length > 0;
  const hasEmergency = input.emergencyName.trim().length > 0;

  const requiredFlags = [hasHn, hasFirstName, hasLastName, hasValidAge, hasPdpa];
  const completedRequired = requiredFlags.filter(Boolean).length;
  const totalRequired = requiredFlags.length;

  const bonusFlags = [hasBirthDate, hasFullPhone, hasDiagnosis, hasEmergency];
  const bonusCompleted = bonusFlags.filter(Boolean).length;
  const bonusTotal = bonusFlags.length;

  // Required items account for 80% of the readiness bar, optional enrichment items up to +20%.
  // When all required items are complete, the CTA unlocks immediately even if optional items are blank.
  const requiredPct = (completedRequired / totalRequired) * 80;
  const bonusPct = (bonusCompleted / bonusTotal) * 20;
  const percent = Math.round(requiredPct + bonusPct);

  return {
    percent,
    readyForFaceScan: hasFirstName && hasLastName && hasPdpa,
    completedRequired,
    totalRequired,
    bonusCompleted,
    bonusTotal,
    checks: {
      hasHn,
      hasFirstName,
      hasLastName,
      hasValidAge,
      hasPdpa,
      hasBirthDate,
      hasFullPhone,
      hasDiagnosis,
      hasEmergency,
    },
  };
}

