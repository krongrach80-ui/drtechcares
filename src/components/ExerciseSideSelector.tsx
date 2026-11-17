import React from 'react';
import type { ClinicalLang } from '@drtechcares/clinical-i18n';
import type { ExerciseSideCode } from '@drtechcares/pose-engine';

export interface SidePillMeta {
  code: ExerciseSideCode;
  label: Record<ClinicalLang, string>;
  shortLabel: Record<ClinicalLang, string>;
  activeClass: string;
  badgeClass: string;
}

export const EXERCISE_SIDE_OPTIONS: readonly SidePillMeta[] = [
  {
    code: 'LEFT',
    label: { th: 'ซ้าย (L)', en: 'Left (L)', cn: '左 (L)' },
    shortLabel: { th: 'ซ้าย', en: 'Left', cn: '左侧' },
    activeClass: 'bg-sky-600 text-white border-sky-600 shadow-xs',
    badgeClass: 'bg-sky-100 text-sky-800 border-sky-200',
  },
  {
    code: 'RIGHT',
    label: { th: 'ขวา (R)', en: 'Right (R)', cn: '右 (R)' },
    shortLabel: { th: 'ขวา', en: 'Right', cn: '右侧' },
    activeClass: 'bg-emerald-600 text-white border-emerald-600 shadow-xs',
    badgeClass: 'bg-emerald-100 text-emerald-800 border-emerald-200',
  },
  {
    code: 'BOTH',
    label: { th: 'ทั้งสองข้าง (Both)', en: 'Both Sides', cn: '双侧 (Both)' },
    shortLabel: { th: 'ทั้งสองข้าง', en: 'Both', cn: '双侧' },
    activeClass: 'bg-teal-600 text-white border-teal-600 shadow-xs',
    badgeClass: 'bg-teal-100 text-teal-800 border-teal-200',
  },
];

export const BILATERAL_INHERENT_EXERCISE_CODES = new Set([
  'STROKE_BOBATH_ARM_RAISE',
  'STROKE_BOBATH_FORWARD_PUSH',
  'MOPH_NECK_TURN',
  'MOPH_SHOULDER_SHRUG_ROLL',
]);

export function getSideBadgeMeta(side: ExerciseSideCode): SidePillMeta {
  return (
    EXERCISE_SIDE_OPTIONS.find((opt) => opt.code === side) ??
    EXERCISE_SIDE_OPTIONS[1]
  );
}

export function ExerciseSidePills({
  value,
  onSelect,
  lang = 'th',
  compact = false,
}: {
  value: ExerciseSideCode | null;
  onSelect: (side: ExerciseSideCode) => void;
  lang?: ClinicalLang;
  compact?: boolean;
}) {
  return (
    <div
      className="inline-flex flex-wrap items-center gap-1 rounded-xl bg-slate-100/90 p-0.5 border border-slate-200/80"
      role="group"
      aria-label="เลือกข้างที่จะฝึก"
    >
      {EXERCISE_SIDE_OPTIONS.map((opt) => {
        const isActive = value === opt.code;
        return (
          <button
            key={opt.code}
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onSelect(opt.code);
            }}
            className={`rounded-lg border font-bold transition cursor-pointer ${
              compact ? 'px-2 py-0.5 text-[10px]' : 'px-2.5 py-1 text-[11px]'
            } ${
              isActive
                ? opt.activeClass
                : 'border-transparent bg-transparent text-slate-600 hover:bg-white hover:text-slate-900'
            }`}
          >
            {opt.label[lang]}
          </button>
        );
      })}
    </div>
  );
}

export function GlobalQuickSideSelector({
  value,
  selectedCount,
  onSelectGlobal,
  lang = 'th',
}: {
  value: ExerciseSideCode;
  selectedCount: number;
  onSelectGlobal: (side: ExerciseSideCode) => void;
  lang?: ClinicalLang;
}) {
  const titleText: Record<ClinicalLang, string> = {
    th:
      selectedCount > 0
        ? `เปลี่ยนข้างทั้งหมดที่เลือก (${selectedCount} ท่า):`
        : 'ข้างเริ่มต้นสำหรับท่าที่จะเลือก:',
    en:
      selectedCount > 0
        ? `Set side for selected (${selectedCount}):`
        : 'Default side for next selection:',
    cn:
      selectedCount > 0
        ? `批量设置已选动作训练侧 (${selectedCount} 项):`
        : '默认选择训练侧:',
  };

  return (
    <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-sky-200 bg-sky-50/70 px-3 py-2">
      <span className="text-[11px] font-extrabold text-sky-900">
        {titleText[lang]}
      </span>
      <ExerciseSidePills
        value={value}
        onSelect={onSelectGlobal}
        lang={lang}
      />
    </div>
  );
}
