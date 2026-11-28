import React, { useEffect, useMemo, useRef, useState } from 'react';
import { CLINICAL_EXERCISES, parseExerciseCodeAndSide } from '@drtechcares/pose-engine';
import type {
  RehabStockDayPoint,
  RehabStockIndexSummary,
  StockScoreCalculationMode,
} from '../utils/rehabStockScoreEngine';
import type { DoctorClinicalBrief } from '../utils/doctorClinicalBrief';
import { GameRankBadge } from './GameJuiceEffects';

export interface RehabStockPerformanceCardProps {
  patientName: string;
  patientHn: string;
  stockIndex: RehabStockIndexSummary;
  compact?: boolean;
  fillHeight?: boolean;
  doctorBrief?: DoctorClinicalBrief;
}

type StockChartStyle = 'AREA' | 'CANDLE';
type StockVisualView = 'EXERCISE_BARS' | 'DAILY_TREND';

function formatPoints(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}

function formatFriendlyStockExerciseName(rawName: string, rawCode: string): string {
  const cleanRaw = String(rawName ?? '').trim();
  const isRawCodeLike = !cleanRaw || /^[A-Z0-9_]+$/.test(cleanRaw);
  let resolved = cleanRaw;
  if (isRawCodeLike || cleanRaw.includes('_')) {
    const parsed = parseExerciseCodeAndSide(rawCode || cleanRaw);
    const found = CLINICAL_EXERCISES.find(
      (ex) => ex.code.toUpperCase() === parsed.baseCode.toUpperCase(),
    );
    if (found) {
      resolved = found.name.th;
    }
  }
  // Strip verbose bracket tags and parenthetical descriptions so chart labels stay crisp & readable
  const concise = resolved
    .replace(/^\[[^\]]+\]\s*/g, '')
    .replace(/\s*\([^)]*\)\s*/g, ' ')
    .replace(/\s*—\s*.*$/g, '')
    .replace(/\s+/g, ' ')
    .trim();
  return concise || resolved.replace(/^\[[^\]]+\]\s*/g, '').trim() || rawCode;
}

export function RehabStockPerformanceCard({
  patientName,
  patientHn,
  stockIndex,
  compact = false,
  fillHeight = false,
  doctorBrief,
}: RehabStockPerformanceCardProps) {
  const [chartStyle, setChartStyle] = useState<StockChartStyle>('AREA');
  const [scoreMode, setScoreMode] = useState<StockScoreCalculationMode>('SMART_WEIGHTED');
  const [showMa3, setShowMa3] = useState<boolean>(true);
  const [showAdvancedControls, setShowAdvancedControls] = useState<boolean>(false);
  const [showBreakdownDrawer, setShowBreakdownDrawer] = useState<boolean>(false);
  const [activeDateKey, setActiveDateKey] = useState<string | null>(null);
  const [activeTrendKey, setActiveTrendKey] = useState<string | null>(null);
  const [manualVisualView, setManualVisualView] = useState<StockVisualView | null>(null);
  const [barChartPage, setBarChartPage] = useState<number>(0);
  const [breakdownPage, setBreakdownPage] = useState<number>(0);
  const [breakdownPerPage, setBreakdownPerPage] = useState<number>(6);
  const cardRootRef = useRef<HTMLDivElement | null>(null);

  // Always default to the real-data Stock Area/Line Trend chart (both 1-day intraday & multi-day);
  // PT/admin can switch to per-exercise horizontal bars anytime in 1 click.
  const effectiveVisualView: StockVisualView = manualVisualView ?? 'DAILY_TREND';

  useEffect(() => {
    setBreakdownPage(0);
    setBarChartPage(0);
    setActiveTrendKey(null);
  }, [activeDateKey, stockIndex.points.length]);

  useEffect(() => {
    if (!fillHeight || typeof ResizeObserver === 'undefined' || !cardRootRef.current) return;
    const updatePageSize = () => {
      const h = cardRootRef.current?.clientHeight ?? 0;
      const w = cardRootRef.current?.clientWidth ?? 0;
      const cols = w >= 720 ? 3 : 2;
      const rows = h >= 820 ? 3 : 2;
      setBreakdownPerPage(cols * rows);
    };
    updatePageSize();
    const ro = new ResizeObserver(updatePageSize);
    ro.observe(cardRootRef.current);
    return () => ro.disconnect();
  }, [fillHeight]);

  const isSmartMode = scoreMode === 'SMART_WEIGHTED';

  const activePoint: RehabStockDayPoint | null = useMemo(() => {
    if (stockIndex.points.length === 0) return null;
    if (activeDateKey) {
      const found = stockIndex.points.find((p) => p.dateKey === activeDateKey);
      if (found) return found;
    }
    return stockIndex.latestPoint;
  }, [stockIndex.points, stockIndex.latestPoint, activeDateKey]);

  const currentPct = activePoint
    ? isSmartMode
      ? activePoint.smartTotalPct
      : activePoint.rawTotalPct
    : 0;
  const currentEarned = activePoint
    ? isSmartMode
      ? activePoint.smartEarnedScore
      : activePoint.rawEarnedScore
    : 0;
  const currentDeltaPct = activePoint
    ? isSmartMode
      ? activePoint.deltaSmartPct
      : activePoint.deltaRawPct
    : null;
  const currentDeltaPts = activePoint
    ? isSmartMode
      ? activePoint.deltaSmartPoints
      : activePoint.deltaRawPoints
    : null;

  const headlineTrend: 'UP' | 'DOWN' | 'FLAT' =
    currentDeltaPct === null || Math.abs(currentDeltaPct) < 0.1
      ? stockIndex.overallTrend
      : currentDeltaPct > 0
        ? 'UP'
        : 'DOWN';

  const accentStroke = headlineTrend === 'DOWN' ? '#e11d48' : '#059669';

  const highStat = isSmartMode ? stockIndex.allTimeHighSmartPct : stockIndex.allTimeHighRawPct;
  const lowStat = isSmartMode ? stockIndex.allTimeLowSmartPct : stockIndex.allTimeLowRawPct;
  const avgStat = isSmartMode ? stockIndex.avgSmartPct : stockIndex.avgRawPct;
  const ma3Stat = activePoint
    ? isSmartMode
      ? activePoint.ma3SmartPct
      : activePoint.ma3RawPct
    : null;

  // Pure 60FPS Dual-Pane SVG Geometry (Upper Score/% Stock Area & Candle Pane + Lower Volume Reps Pane)
  // Works on 100% real data for BOTH multi-day history (>= 2 days) AND 1-day real cumulative intraday sessions!
  const svgGeom = useMemo(() => {
    interface RealTrendNodeInput {
      key: string;
      dateKey: string;
      labelShortTh: string;
      detailTh: string;
      pctVal: number;
      openVal: number;
      highVal: number;
      lowVal: number;
      closeVal: number;
      ma3Val: number;
      volumeReps: number;
      isStartBaseline?: boolean;
    }

    const rawDays = stockIndex.points.slice(-16);
    const isIntradayRealMode = rawDays.length === 1;
    const nodes: RealTrendNodeInput[] = [];

    if (isIntradayRealMode) {
      const onlyDay = rawDays[0];
      const safeMax = Math.max(1, onlyDay.maxTotalScore);
      const played = onlyDay.exerciseBreakdown.filter((ex) => ex.played);
      const dayFinalPct = isSmartMode ? onlyDay.smartTotalPct : onlyDay.rawTotalPct;

      nodes.push({
        key: `${onlyDay.dateKey}::start`,
        dateKey: onlyDay.dateKey,
        labelShortTh: 'เริ่มฝึก',
        detailTh: `${onlyDay.labelFullTh} • จุดเริ่มต้นก่อนฝึก (0 / ${onlyDay.maxTotalScore} คะแนน)`,
        pctVal: 0,
        openVal: 0,
        highVal: 0,
        lowVal: 0,
        closeVal: 0,
        ma3Val: 0,
        volumeReps: 0,
        isStartBaseline: true,
      });

      if (played.length > 0) {
        let cumPts = 0;
        let prevPct = 0;
        const pctWindow: number[] = [0];
        played.forEach((ex, idx) => {
          const addedPts = isSmartMode ? ex.smartEarnedPoints : ex.rawEarnedPoints;
          cumPts += addedPts;
          const isLast = idx === played.length - 1;
          const pctVal = isLast
            ? dayFinalPct
            : Number(Math.max(0, Math.min(100, (cumPts / safeMax) * 100)).toFixed(1));
          pctWindow.push(pctVal);
          const maSlice = pctWindow.slice(-3);
          const ma3Val = Number(
            (maSlice.reduce((sum, v) => sum + v, 0) / maSlice.length).toFixed(1),
          );
          const friendly = formatFriendlyStockExerciseName(ex.exerciseNameTh, ex.exerciseCode);
          nodes.push({
            key: `${onlyDay.dateKey}::ex-${idx + 1}-${ex.exerciseCode}`,
            dateKey: onlyDay.dateKey,
            labelShortTh: `ท่าที่ ${idx + 1}`,
            detailTh: `${onlyDay.labelShortTh} • ท่าที่ ${idx + 1}: ${friendly} (${ex.completedReps}/${ex.targetReps} ครั้ง${ex.achievedRomDeg !== null ? ` • ROM ${ex.achievedRomDeg.toFixed(0)}°` : ''}) • สะสม ${pctVal.toFixed(1)}%`,
            pctVal,
            openVal: prevPct,
            highVal: Math.max(prevPct, pctVal),
            lowVal: Math.min(prevPct, pctVal),
            closeVal: pctVal,
            ma3Val,
            volumeReps: ex.completedReps,
          });
          prevPct = pctVal;
        });
      } else {
        nodes.push({
          key: `${onlyDay.dateKey}::end`,
          dateKey: onlyDay.dateKey,
          labelShortTh: onlyDay.labelShortTh,
          detailTh: `${onlyDay.labelFullTh} • คะแนนรวม ${dayFinalPct.toFixed(1)}%`,
          pctVal: dayFinalPct,
          openVal: 0,
          highVal: dayFinalPct,
          lowVal: 0,
          closeVal: dayFinalPct,
          ma3Val: Number((dayFinalPct / 2).toFixed(1)),
          volumeReps: onlyDay.volumeReps,
        });
      }
    } else {
      for (const p of rawDays) {
        const pctVal = isSmartMode ? p.smartTotalPct : p.rawTotalPct;
        nodes.push({
          key: p.dateKey,
          dateKey: p.dateKey,
          labelShortTh: p.labelShortTh,
          detailTh: `วันที่ ${p.labelFullTh} • ฝึก ${p.completedExercisesCount}/${p.totalExercisesCount} ท่า (${p.volumeReps} ครั้ง) • ดัชนีรวม ${pctVal.toFixed(1)}%`,
          pctVal,
          openVal: isSmartMode ? p.openPct : p.rawOpenPct,
          highVal: isSmartMode ? p.highPct : p.rawHighPct,
          lowVal: isSmartMode ? p.lowPct : p.rawLowPct,
          closeVal: isSmartMode ? p.closePct : p.rawClosePct,
          ma3Val: isSmartMode ? p.ma3SmartPct : p.ma3RawPct,
          volumeReps: p.volumeReps,
        });
      }
    }

    const minW = compact ? 560 : 640;
    const width = Math.max(minW, nodes.length > 8 ? nodes.length * 72 + 136 : minW);
    const height = compact ? 250 : 298;
    const padL = 72;
    const padR = 64;
    const padT = compact ? 24 : 28;
    const priceH = compact ? 132 : 164;
    const gapBetweenPanes = compact ? 16 : 18;
    const volTop = padT + priceH + gapBetweenPanes;
    const volH = compact ? 34 : 42;
    const innerPadX = 36;
    const plotW = width - padL - padR;
    const usableW = Math.max(40, plotW - innerPadX * 2);

    const maxVol = Math.max(
      10,
      isIntradayRealMode ? 12 : stockIndex.maxTotalScore,
      ...nodes.map((n) => n.volumeReps),
    );

    const yForPct = (pct: number) =>
      padT + priceH - (Math.max(0, Math.min(100, pct)) / 100) * priceH;

    const mapped = nodes.map((n, idx) => {
      const x =
        nodes.length === 1
          ? padL + plotW / 2
          : padL + innerPadX + (idx / Math.max(1, nodes.length - 1)) * usableW;
      const y = yForPct(n.pctVal);
      const yOpen = yForPct(n.openVal);
      const yHigh = yForPct(n.highVal);
      const yLow = yForPct(n.lowVal);
      const yClose = yForPct(n.closeVal);
      const yMa3 = yForPct(n.ma3Val);
      const bullish = n.closeVal >= n.openVal;

      const volBarH = n.volumeReps > 0 ? Math.max(4, (n.volumeReps / maxVol) * volH) : 0;
      const volY = volTop + volH - volBarH;

      return {
        node: n,
        x,
        y,
        pctVal: n.pctVal,
        yOpen,
        yHigh,
        yLow,
        yClose,
        yMa3,
        bullish,
        volBarH,
        volY,
      };
    });

    const linePoints = mapped.map((m) => `${m.x.toFixed(1)},${m.y.toFixed(1)}`).join(' ');
    const ma3Points = mapped.map((m) => `${m.x.toFixed(1)},${m.yMa3.toFixed(1)}`).join(' ');
    const areaPoints =
      mapped.length > 0
        ? `${mapped[0].x.toFixed(1)},${(padT + priceH).toFixed(1)} ${linePoints} ${mapped[mapped.length - 1].x.toFixed(1)},${(padT + priceH).toFixed(1)}`
        : '';

    const candleW = Math.min(18, Math.max(8, usableW / Math.max(8, mapped.length * 2.2)));
    const volBarW = Math.min(22, Math.max(8, usableW / Math.max(8, mapped.length * 1.9)));

    return {
      isIntradayRealMode,
      width,
      height,
      padL,
      padR,
      padT,
      priceH,
      volTop,
      volH,
      plotW,
      maxVol,
      yForPct,
      mapped,
      linePoints,
      ma3Points,
      areaPoints,
      candleW,
      volBarW,
    };
  }, [stockIndex.points, stockIndex.maxTotalScore, isSmartMode, compact]);

  const activeMapped = useMemo(() => {
    if (svgGeom.mapped.length === 0) return null;
    if (activeTrendKey) {
      const foundKey = svgGeom.mapped.find((m) => m.node.key === activeTrendKey);
      if (foundKey) return foundKey;
    }
    if (activePoint) {
      const matching = svgGeom.mapped.filter((m) => m.node.dateKey === activePoint.dateKey);
      if (matching.length > 0) return matching[matching.length - 1];
    }
    return svgGeom.mapped[svgGeom.mapped.length - 1];
  }, [svgGeom.mapped, activeTrendKey, activePoint]);

  const totalBreakdownItems = activePoint?.exerciseBreakdown.length ?? 0;
  const effectivePerPage = fillHeight ? breakdownPerPage : totalBreakdownItems || 6;
  const totalBreakdownPages = Math.max(1, Math.ceil(totalBreakdownItems / Math.max(1, effectivePerPage)));
  const safeBreakdownPage = Math.min(breakdownPage, totalBreakdownPages - 1);
  const visibleBreakdownItems = activePoint
    ? fillHeight
      ? activePoint.exerciseBreakdown.slice(
          safeBreakdownPage * effectivePerPage,
          (safeBreakdownPage + 1) * effectivePerPage,
        )
      : activePoint.exerciseBreakdown
    : [];

  const BAR_CHART_PER_PAGE = 9;
  const totalBarPages = Math.max(1, Math.ceil(totalBreakdownItems / BAR_CHART_PER_PAGE));
  const safeBarPage = Math.min(barChartPage, totalBarPages - 1);
  const visibleBarItems = activePoint
    ? activePoint.exerciseBreakdown.slice(
        safeBarPage * BAR_CHART_PER_PAGE,
        (safeBarPage + 1) * BAR_CHART_PER_PAGE,
      )
    : [];

  return (
    <div
      ref={cardRootRef}
      data-testid="rehab-stock-performance-card"
      className={
        fillHeight
          ? 'flex-1 min-h-0 h-full flex flex-col gap-2.5 overflow-hidden rounded-2xl border border-slate-200/90 bg-white p-3.5 shadow-[0_2px_14px_rgba(15,76,129,0.05)] animate-view-enter'
          : compact
            ? 'rounded-2xl border border-slate-200/90 bg-white p-4 shadow-sm space-y-3 animate-view-enter'
            : 'rounded-3xl border border-slate-200/90 bg-white p-5 shadow-md space-y-4 animate-view-enter'
      }
    >
      {/* Clean Balanced Card Header */}
      <div className="shrink-0 flex flex-wrap items-center justify-between gap-2.5 border-b border-slate-100 pb-2.5">
        <div className="flex flex-wrap items-center gap-2 min-w-0">
          <span className="inline-flex items-center rounded-lg bg-[#0F4C81] px-2.5 py-1 text-[11px] font-black tracking-wide text-white shadow-2xs">
            REHAB INDEX • {patientHn || 'PATIENT'}
          </span>
          <span
            data-testid="stock-index-current-pct"
            className={`text-2xl sm:text-3xl font-black tracking-tight tabular-nums leading-none ${
              headlineTrend === 'DOWN' ? 'text-rose-600' : 'text-emerald-700'
            }`}
          >
            {activePoint ? `${currentPct.toFixed(1)}%` : '0.0%'}
          </span>
          <span
            data-testid="stock-index-score-fraction"
            className="rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-0.5 text-xs font-black text-slate-800 tabular-nums"
          >
            {formatPoints(currentEarned)} / {stockIndex.maxTotalScore} คะแนน
          </span>
          <span className="sr-only">
            คะแนนเต็มรวม {stockIndex.maxTotalScore} คะแนน ({stockIndex.totalPrescribedExercises} ท่าที่แพทย์/PT กำหนด)
          </span>
          <GameRankBadge pct={activePoint ? currentPct : 0} />
          <span className="rounded-lg bg-teal-50 border border-teal-200/80 px-2 py-0.5 text-[11px] font-extrabold text-teal-800">
            ฝึกแล้ว {activePoint ? `${activePoint.completedExercisesCount}/${activePoint.totalExercisesCount}` : `0/${stockIndex.totalPrescribedExercises}`} ท่า
          </span>
          {currentDeltaPct !== null && (
            <span
              className={`inline-flex items-center gap-1 rounded-lg border px-2 py-0.5 text-[11px] font-extrabold tabular-nums ${
                currentDeltaPct > 0
                  ? 'border-emerald-200 bg-emerald-50 text-emerald-700'
                  : currentDeltaPct < 0
                    ? 'border-rose-200 bg-rose-50 text-rose-700'
                    : 'border-slate-200 bg-slate-50 text-slate-600'
              }`}
            >
              {currentDeltaPct >= 0 ? '+' : ''}
              {currentDeltaPct.toFixed(1)}%
            </span>
          )}
        </div>

        {/* Sleek Segmented View Switch + Collapsible Chart Settings */}
        <div className="flex items-center gap-1.5 shrink-0">
          {stockIndex.points.length > 0 && (
            <div className="inline-flex rounded-xl border border-slate-200 bg-slate-100/90 p-0.5 text-xs font-black">
              <button
                type="button"
                data-testid="stock-view-daily-trend"
                onClick={() => setManualVisualView('DAILY_TREND')}
                className={`rounded-lg px-3 py-1 cursor-pointer transition-all ${
                  effectiveVisualView === 'DAILY_TREND'
                    ? 'bg-[#0F4C81] text-white shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                แนวโน้มรายวัน
              </button>
              <button
                type="button"
                data-testid="stock-view-exercise-bars"
                onClick={() => setManualVisualView('EXERCISE_BARS')}
                className={`rounded-lg px-3 py-1 cursor-pointer transition-all ${
                  effectiveVisualView === 'EXERCISE_BARS'
                    ? 'bg-emerald-600 text-white shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                แยกตามท่า
              </button>
            </div>
          )}

          <button
            type="button"
            onClick={() => setShowAdvancedControls((v) => !v)}
            className={`rounded-xl border px-2.5 py-1.5 text-xs font-bold cursor-pointer transition-colors ${
              showAdvancedControls
                ? 'border-sky-300 bg-sky-50 text-[#0F4C81]'
                : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
            }`}
          >
            ตั้งค่ากราฟ
          </button>
        </div>
      </div>

      {/* Collapsible Advanced Stock Options (Hidden by default to keep UI clutter-free; kept in DOM for SSR tests) */}
      <div
        hidden={!showAdvancedControls}
        className={
          showAdvancedControls
            ? 'shrink-0 flex flex-wrap items-center justify-between gap-2 rounded-xl border border-slate-200 bg-slate-50/80 px-2.5 py-1.5 text-xs font-extrabold'
            : 'hidden'
        }
      >
        <div className="flex flex-wrap items-center gap-1.5">
          <div className="inline-flex rounded-lg border border-slate-200 bg-white p-0.5">
            <button
              type="button"
              data-testid="stock-style-area"
              onClick={() => {
                setChartStyle('AREA');
                setManualVisualView('DAILY_TREND');
              }}
              className={`rounded-md px-2.5 py-1 cursor-pointer ${
                chartStyle === 'AREA' ? 'bg-[#0F4C81] text-white' : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              กราฟพื้นที่ (Area)
            </button>
            <button
              type="button"
              data-testid="stock-style-candle"
              onClick={() => {
                setChartStyle('CANDLE');
                setManualVisualView('DAILY_TREND');
              }}
              className={`rounded-md px-2.5 py-1 cursor-pointer ${
                chartStyle === 'CANDLE' ? 'bg-emerald-600 text-white' : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              แท่งเทียน (Candle)
            </button>
          </div>

          <div className="inline-flex rounded-lg border border-sky-200 bg-white p-0.5">
            <button
              type="button"
              data-testid="stock-mode-smart"
              onClick={() => setScoreMode('SMART_WEIGHTED')}
              className={`rounded-md px-2.5 py-1 cursor-pointer ${
                scoreMode === 'SMART_WEIGHTED' ? 'bg-emerald-600 text-white' : 'text-emerald-800'
              }`}
            >
              อัจฉริยะ (ROM+แม่นยำ)
            </button>
            <button
              type="button"
              data-testid="stock-mode-raw"
              onClick={() => setScoreMode('RAW_REPS')}
              className={`rounded-md px-2.5 py-1 cursor-pointer ${
                scoreMode === 'RAW_REPS' ? 'bg-[#0F4C81] text-white' : 'text-[#0F4C81]'
              }`}
            >
              นับครั้งตรง (1 ครั้ง=1 แต้ม)
            </button>
          </div>

          <button
            type="button"
            onClick={() => setShowMa3((v) => !v)}
            className={`rounded-lg border px-2.5 py-1 cursor-pointer ${
              showMa3 ? 'border-sky-300 bg-sky-50 text-[#0F4C81]' : 'border-slate-200 bg-white text-slate-500'
            }`}
          >
            เส้น MA3
          </button>
        </div>

        <span className="text-slate-500 font-bold">
          สูงสุด {highStat !== null ? `${highStat.toFixed(1)}%` : '-'} • เฉลี่ย {avgStat !== null ? `${avgStat.toFixed(1)}%` : '-'}
          {ma3Stat !== null ? ` • MA3 ${ma3Stat.toFixed(1)}%` : ''}
        </span>
      </div>

      {/* GIANT 100% FULL-WIDTH Chart Arena (Dominates the Viewport!) */}
      {svgGeom.mapped.length === 0 ? (
        <div className="flex-1 flex flex-col items-center justify-center rounded-2xl border border-dashed border-sky-200 bg-slate-50/60 p-6 text-center space-y-2">
          <div className="text-base font-black text-slate-800">
            {stockIndex.totalPrescribedExercises > 0
              ? `พร้อมเริ่มเก็บสถิติ (${stockIndex.totalPrescribedExercises} ท่า • เต็ม ${stockIndex.maxTotalScore} คะแนน)`
              : 'ยังไม่มีข้อมูลการฝึกสำหรับผู้ป่วยรายนี้'}
          </div>
          <p className="text-sm text-slate-600 max-w-md">
            เมื่อผู้ป่วยฝึกหน้าตู้ Kiosk กราฟผลฝึกจะแสดงที่นี่อัตโนมัติ
          </p>
        </div>
      ) : (
        <div
          className={
            fillHeight
              ? 'flex-1 min-h-0 flex flex-col gap-2 overflow-hidden'
              : compact
                ? 'space-y-2.5'
                : 'space-y-3.5'
          }
        >
          {/* FULL-WIDTH MAIN GRAPH BOX */}
          <div
            className={
              fillHeight
                ? 'flex-1 min-h-[260px] w-full overflow-hidden rounded-2xl border border-slate-200/80 bg-gradient-to-b from-[#FAFDFE] via-white to-[#F8FCFA] p-3.5 flex flex-col'
                : 'w-full overflow-x-auto rounded-2xl border border-slate-200/80 bg-gradient-to-b from-[#FAFDFE] via-white to-[#F8FCFA] p-3.5 flex flex-col'
            }
          >
            {effectiveVisualView === 'EXERCISE_BARS' && activePoint ? (
              <div
                data-testid="stock-single-day-exercise-bars"
                className="flex-1 min-h-0 flex flex-col gap-2.5 overflow-hidden"
              >
                <div className="shrink-0 flex flex-wrap items-center justify-between gap-2 pb-1.5 border-b border-slate-100">
                  <div className="flex items-center gap-2">
                    <span className="h-2.5 w-2.5 rounded-full bg-emerald-500" />
                    <span className="text-sm font-black text-slate-900">
                      ผลฝึกแต่ละท่า • {activePoint.labelFullTh}
                    </span>
                  </div>
                  <div className="flex items-center gap-3 text-[11px] font-extrabold">
                    <span className="inline-flex items-center gap-1.5 text-emerald-700">
                      <span className="h-2 w-2 rounded-full bg-emerald-500" />
                      ผ่านเกณฑ์ (80%+)
                    </span>
                    <span className="inline-flex items-center gap-1.5 text-slate-500">
                      <span className="h-2 w-2 rounded-full bg-slate-300" />
                      รอฝึก
                    </span>
                    {totalBarPages > 1 && (
                      <div className="inline-flex items-center rounded-lg border border-slate-200 bg-white p-0.5">
                        <button
                          type="button"
                          aria-label="ชุดท่าก่อนหน้า"
                          onClick={() => setBarChartPage((p) => (p - 1 + totalBarPages) % totalBarPages)}
                          className="px-2 py-0.5 text-xs font-black text-slate-700 hover:text-[#0F4C81] cursor-pointer"
                        >
                          &lt;
                        </button>
                        <span className="px-1.5 text-xs font-black text-slate-800 tabular-nums">
                          {safeBarPage + 1}/{totalBarPages}
                        </span>
                        <button
                          type="button"
                          aria-label="ชุดท่าถัดไป"
                          onClick={() => setBarChartPage((p) => (p + 1) % totalBarPages)}
                          className="px-2 py-0.5 text-xs font-black text-slate-700 hover:text-[#0F4C81] cursor-pointer"
                        >
                          &gt;
                        </button>
                      </div>
                    )}
                  </div>
                </div>

                {/* Structured Cohesive Exercise Cards — Zero Sparse Empty White Gaps */}
                <div className="flex-1 min-h-0 overflow-y-auto space-y-2 pr-0.5">
                  {visibleBarItems.map((ex, idx) => {
                    const rowIndex = safeBarPage * BAR_CHART_PER_PAGE + idx + 1;
                    const pctVal = isSmartMode ? ex.smartPct : ex.rawPct;
                    const clampedPct = Math.max(0, Math.min(100, pctVal));
                    const friendlyTitle = formatFriendlyStockExerciseName(
                      ex.exerciseNameTh,
                      ex.exerciseCode,
                    );
                    const barColor = !ex.played
                      ? 'bg-slate-200'
                      : pctVal >= 80
                        ? 'bg-gradient-to-r from-emerald-500 to-teal-400'
                        : pctVal >= 60
                          ? 'bg-gradient-to-r from-[#0F4C81] to-sky-400'
                          : 'bg-gradient-to-r from-amber-500 to-orange-400';
                    return (
                      <div
                        key={ex.exerciseCode}
                        className={`rounded-xl border px-3 py-2 flex items-center gap-3 min-w-0 transition-colors ${
                          ex.played
                            ? 'border-emerald-200/70 bg-white hover:border-emerald-300 shadow-[0_1px_3px_rgba(15,23,42,0.03)]'
                            : 'border-slate-200/70 bg-slate-50/70'
                        }`}
                      >
                        <span
                          className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-[11px] font-black tabular-nums ${
                            ex.played
                              ? 'bg-emerald-100 text-emerald-800'
                              : 'bg-slate-200/80 text-slate-500'
                          }`}
                        >
                          {rowIndex}
                        </span>

                        <div className="w-36 sm:w-48 shrink-0 text-xs sm:text-sm font-bold text-slate-900 truncate">
                          {friendlyTitle}
                        </div>

                        <div className="relative flex-1 h-7 rounded-full bg-slate-100 border border-slate-200/70 overflow-hidden">
                          <div
                            aria-hidden="true"
                            className="pointer-events-none absolute top-0 bottom-0 left-[80%] z-10 border-r-2 border-dashed border-sky-500/70"
                          />
                          <div
                            className={`h-full rounded-full transition-all duration-300 ${barColor}`}
                            style={{ width: `${ex.played ? Math.max(10, clampedPct) : 0}%` }}
                          />
                          <div className="absolute inset-0 flex items-center justify-between px-3 text-xs font-black">
                            <span
                              className={
                                ex.played && clampedPct >= 34 ? 'text-white drop-shadow-2xs' : 'text-slate-500'
                              }
                            >
                              {ex.played ? `${pctVal.toFixed(0)}% • ${ex.completedReps}/${ex.targetReps} ครั้ง` : 'รอฝึก'}
                            </span>
                          </div>
                        </div>

                        <div className="w-24 sm:w-28 shrink-0 text-right">
                          {ex.played ? (
                            <span className="inline-block rounded-lg bg-emerald-50 border border-emerald-200 px-2 py-0.5 text-xs font-black text-emerald-800 tabular-nums whitespace-nowrap">
                              {ex.achievedRomDeg !== null
                                ? `${ex.achievedRomDeg.toFixed(0)}° / ${Math.round(ex.targetRomDeg)}°`
                                : `${ex.accuracyPct}%`}
                            </span>
                          ) : (
                            <span className="inline-block rounded-lg bg-slate-100 px-2 py-0.5 text-[11px] font-bold text-slate-400">
                              รอฝึก
                            </span>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : (
            <div className="flex-1 min-h-0 flex flex-col gap-2">
              {/* Live Real-Data Context Strip (Clean Glass Pill Header) */}
              {activeMapped && (
                <div className="shrink-0 flex flex-wrap items-center justify-between gap-2 rounded-xl border border-sky-100 bg-sky-50/60 px-3 py-1.5 text-xs">
                  <span className="font-extrabold text-[#0F4C81] leading-snug">
                    {activeMapped.node.detailTh}
                  </span>
                  <span className="shrink-0 rounded-full bg-white border border-emerald-200 px-2.5 py-0.5 text-[11px] font-extrabold text-emerald-700 shadow-2xs">
                    {svgGeom.isIntradayRealMode
                      ? 'สะสมจริงรายท่าของวันนี้'
                      : `แนวโน้มจริง ${stockIndex.points.length} วันฝึก`}
                  </span>
                </div>
              )}
              <svg
                viewBox={`0 0 ${svgGeom.width} ${svgGeom.height}`}
                preserveAspectRatio="xMidYMid meet"
                className={fillHeight ? 'flex-1 w-full h-full max-h-full select-none' : 'w-full h-auto select-none'}
                role="img"
                aria-label="กราฟดัชนีเปอร์เซ็นต์รวมทุกท่ากายภาพแบบกราฟหุ้น"
              >
              <defs>
                <linearGradient id="stockUpAreaGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#059669" stopOpacity="0.30" />
                  <stop offset="75%" stopColor="#10b981" stopOpacity="0.06" />
                  <stop offset="100%" stopColor="#10b981" stopOpacity="0.0" />
                </linearGradient>
                <linearGradient id="stockDownAreaGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#e11d48" stopOpacity="0.25" />
                  <stop offset="75%" stopColor="#f43f5e" stopOpacity="0.05" />
                  <stop offset="100%" stopColor="#f43f5e" stopOpacity="0.0" />
                </linearGradient>
              </defs>

              <rect
                x={svgGeom.padL}
                y={svgGeom.yForPct(100)}
                width={svgGeom.plotW}
                height={svgGeom.yForPct(80) - svgGeom.yForPct(100)}
                fill="#10b981"
                fillOpacity="0.08"
              />
              <rect
                x={svgGeom.padL}
                y={svgGeom.yForPct(80)}
                width={svgGeom.plotW}
                height={svgGeom.yForPct(60) - svgGeom.yForPct(80)}
                fill="#0284c7"
                fillOpacity="0.05"
              />

              {[0, 25, 50, 75, 100].map((pctTick) => {
                const y = svgGeom.yForPct(pctTick);
                const ptsTick = (stockIndex.maxTotalScore * pctTick) / 100;
                return (
                  <g key={pctTick}>
                    <line
                      x1={svgGeom.padL}
                      y1={y}
                      x2={svgGeom.width - svgGeom.padR}
                      y2={y}
                      stroke={pctTick === 0 ? '#cbd5e1' : '#e2e8f0'}
                      strokeDasharray={pctTick === 0 ? undefined : '3 3'}
                      strokeWidth="1.2"
                    />
                    <text x={svgGeom.padL - 8} y={y + 4} textAnchor="end" fill="#475569" fontSize="12" fontWeight="800">
                      {formatPoints(ptsTick)} แต้ม
                    </text>
                    <text x={svgGeom.width - svgGeom.padR + 8} y={y + 4} textAnchor="start" fill="#334155" fontSize="12" fontWeight="800">
                      {pctTick}%
                    </text>
                  </g>
                );
              })}

              <g>
                <line
                  x1={svgGeom.padL}
                  y1={svgGeom.yForPct(80)}
                  x2={svgGeom.width - svgGeom.padR}
                  y2={svgGeom.yForPct(80)}
                  stroke="#0284c7"
                  strokeDasharray="6 4"
                  strokeWidth="1.6"
                  opacity="0.8"
                />
                <rect x={svgGeom.padL + 6} y={svgGeom.yForPct(80) - 20} width="138" height="17" rx="8" fill="#ffffff" fillOpacity="0.92" />
                <text x={svgGeom.padL + 12} y={svgGeom.yForPct(80) - 8} textAnchor="start" fill="#0369a1" fontSize="11.5" fontWeight="900">
                  เกณฑ์ฟื้นตัวดีเยี่ยม 80%
                </text>
              </g>

              {activeMapped && (
                <g pointerEvents="none">
                  <line x1={activeMapped.x} y1={svgGeom.padT} x2={activeMapped.x} y2={svgGeom.volTop + svgGeom.volH} stroke="#64748b" strokeDasharray="3 3" strokeWidth="1.2" opacity="0.65" />
                  <line x1={svgGeom.padL} y1={activeMapped.y} x2={svgGeom.width - svgGeom.padR} y2={activeMapped.y} stroke={accentStroke} strokeDasharray="3 3" strokeWidth="1.2" opacity="0.65" />
                  <rect x={svgGeom.width - svgGeom.padR + 3} y={activeMapped.y - 11} width="58" height="22" rx="6" fill={accentStroke} />
                  <text x={svgGeom.width - svgGeom.padR + 32} y={activeMapped.y + 4} textAnchor="middle" fill="#ffffff" fontSize="11.5" fontWeight="900">
                    {activeMapped.pctVal.toFixed(1)}%
                  </text>
                </g>
              )}

              {chartStyle === 'AREA' && (
                <>
                  {svgGeom.areaPoints && (
                    <polygon points={svgGeom.areaPoints} fill={headlineTrend === 'DOWN' ? 'url(#stockDownAreaGrad)' : 'url(#stockUpAreaGrad)'} />
                  )}
                  {svgGeom.mapped.length >= 2 && (
                    <polyline data-testid="stock-area-polyline" fill="none" stroke={accentStroke} strokeWidth="3.2" strokeLinecap="round" strokeLinejoin="round" points={svgGeom.linePoints} />
                  )}
                </>
              )}

              {chartStyle === 'CANDLE' &&
                svgGeom.mapped.map((m) => {
                  const color = m.bullish ? '#059669' : '#e11d48';
                  const bodyTop = Math.min(m.yOpen, m.yClose);
                  const bodyH = Math.max(5, Math.abs(m.yClose - m.yOpen));
                  return (
                    <g key={`candle-${m.node.key}`} data-testid={`stock-candle-${m.node.key}`}>
                      <line x1={m.x} y1={m.yHigh} x2={m.x} y2={m.yLow} stroke={color} strokeWidth="2.2" />
                      <rect x={m.x - svgGeom.candleW / 2} y={bodyTop} width={svgGeom.candleW} height={bodyH} rx="3" fill={color} stroke={color} strokeWidth="1" />
                    </g>
                  );
                })}

              {showMa3 && svgGeom.mapped.length >= 2 && (
                <polyline data-testid="stock-ma3-polyline" fill="none" stroke="#0284c7" strokeWidth="2.2" strokeDasharray="6 4" strokeLinecap="round" strokeLinejoin="round" opacity="0.88" points={svgGeom.ma3Points} />
              )}

              <line x1={svgGeom.padL} y1={svgGeom.volTop - 6} x2={svgGeom.width - svgGeom.padR} y2={svgGeom.volTop - 6} stroke="#cbd5e1" strokeWidth="1.2" />
              <text x={svgGeom.padL - 8} y={svgGeom.volTop + 16} textAnchor="end" fill="#475569" fontSize="11" fontWeight="800">
                จำนวนครั้ง
              </text>

              {svgGeom.mapped.map((m) => {
                const isSelected = activeMapped?.node.key === m.node.key;
                const volFill = m.bullish ? '#10b981' : '#f43f5e';
                const labelY = m.y < svgGeom.padT + 24 ? m.y + 24 : m.y - 14;
                return (
                  <g
                    key={m.node.key}
                    onMouseEnter={() => {
                      setActiveTrendKey(m.node.key);
                      if (!svgGeom.isIntradayRealMode) setActiveDateKey(m.node.dateKey);
                    }}
                    onClick={() => {
                      setActiveTrendKey(m.node.key);
                      if (!svgGeom.isIntradayRealMode) setActiveDateKey(m.node.dateKey);
                    }}
                    className="cursor-pointer"
                  >
                    <rect x={m.x - 26} y={svgGeom.padT} width="52" height={svgGeom.volTop + svgGeom.volH - svgGeom.padT} fill="transparent" />
                    {m.volBarH > 0 && (
                      <rect x={m.x - svgGeom.volBarW / 2} y={m.volY} width={svgGeom.volBarW} height={m.volBarH} rx="4" fill={volFill} fillOpacity={isSelected ? '0.9' : '0.42'} />
                    )}
                    {chartStyle === 'AREA' && (
                      <>
                        {isSelected && <circle cx={m.x} cy={m.y} r="12" fill={accentStroke} fillOpacity="0.16" stroke={accentStroke} strokeWidth="1.4" />}
                        <circle cx={m.x} cy={m.y} r={isSelected ? '7' : '5'} fill={isSelected ? accentStroke : '#ffffff'} stroke={accentStroke} strokeWidth="2.8" />
                        <rect x={m.x - 24} y={labelY - 12} width="48" height="18" rx="9" fill={isSelected ? accentStroke : '#ffffff'} stroke={accentStroke} strokeWidth="1.4" opacity="0.98" />
                        <text x={m.x} y={labelY + 1} textAnchor="middle" fill={isSelected ? '#ffffff' : '#0f172a'} fontSize="11.5" fontWeight="900">
                          {m.pctVal.toFixed(0)}%
                        </text>
                      </>
                    )}
                    <text x={m.x} y={svgGeom.volTop + svgGeom.volH + 14} textAnchor="middle" fill="#475569" fontSize="11" fontWeight="800">
                      {m.node.isStartBaseline ? '-' : `${m.node.volumeReps} ครั้ง`}
                    </text>
                    <text x={m.x} y={svgGeom.volTop + svgGeom.volH + 29} textAnchor="middle" fill={isSelected ? '#0F4C81' : '#334155'} fontSize="11.5" fontWeight="900">
                      {m.node.labelShortTh}
                    </text>
                  </g>
                );
              })}
              </svg>
            </div>
            )}
          </div>

          {/* Polished 3-Card Doctor Summary Strip (Readable 2-Line Wrapping — Zero Mid-Word Cutoff!) */}
          {doctorBrief && (
            <div
              data-testid="doctor-clinical-brief-card"
              className="shrink-0 grid grid-cols-1 md:grid-cols-3 gap-2"
            >
              <div className="rounded-xl border border-sky-200/80 border-l-4 border-l-[#0F4C81] bg-sky-50/40 px-3 py-2 space-y-1">
                <div className="flex items-center justify-between gap-1.5">
                  <span className="text-[11px] font-black text-[#0F4C81]">สรุปสำหรับแพทย์</span>
                  <span
                    className={`rounded-full border px-2 py-0.5 text-[10.5px] font-black whitespace-nowrap ${
                      doctorBrief.statusTone === 'emerald'
                        ? 'border-emerald-300 bg-emerald-50 text-emerald-800'
                        : doctorBrief.statusTone === 'sky'
                          ? 'border-sky-300 bg-white text-[#0F4C81]'
                          : doctorBrief.statusTone === 'amber'
                            ? 'border-amber-300 bg-amber-50 text-amber-900'
                            : 'border-slate-200 bg-slate-50 text-slate-600'
                    }`}
                  >
                    {doctorBrief.statusBadgeTh}
                  </span>
                </div>
                <p className="text-xs font-bold text-slate-700 leading-snug line-clamp-2">
                  {doctorBrief.executiveSummaryTh}
                </p>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div className="rounded-xl border border-emerald-200/80 bg-emerald-50/50 px-2.5 py-2 space-y-0.5">
                  <div className="flex items-center gap-1 text-[11px] font-black text-emerald-800">
                    <span className="h-2 w-2 rounded-full bg-emerald-500 shrink-0" />
                    <span>จุดเด่น</span>
                  </div>
                  <div className="text-xs font-bold text-slate-900 leading-snug line-clamp-2">
                    {doctorBrief.bestExercise
                      ? `${formatFriendlyStockExerciseName(doctorBrief.bestExercise.nameTh, '')} (${doctorBrief.bestExercise.romDeg ?? '-'}°)`
                      : 'รอข้อมูล'}
                  </div>
                </div>
                <div className="rounded-xl border border-amber-200/80 bg-amber-50/50 px-2.5 py-2 space-y-0.5">
                  <div className="flex items-center gap-1 text-[11px] font-black text-amber-900">
                    <span className="h-2 w-2 rounded-full bg-amber-500 shrink-0" />
                    <span>ควรเน้น</span>
                  </div>
                  <div className="text-xs font-bold text-slate-900 leading-snug line-clamp-2">
                    {doctorBrief.attentionExercise
                      ? formatFriendlyStockExerciseName(doctorBrief.attentionExercise.nameTh, '')
                      : 'ครบทุกท่าแล้ว'}
                  </div>
                </div>
              </div>

              <div className="rounded-xl border border-teal-200/80 border-l-4 border-l-teal-500 bg-teal-50/30 px-3 py-2 space-y-1">
                <div className="flex items-center justify-between gap-1.5">
                  <div className="text-[11px] font-black text-teal-800">ข้อเสนอแนะเชิงคลินิก</div>
                  {activePoint && fillHeight && (
                    <button
                      type="button"
                      onClick={() => setShowBreakdownDrawer((v) => !v)}
                      className="shrink-0 rounded-lg border border-sky-200 bg-white px-2 py-0.5 text-[10.5px] font-black text-[#0F4C81] hover:bg-sky-50 shadow-2xs cursor-pointer whitespace-nowrap transition-colors"
                    >
                      {showBreakdownDrawer ? 'ซ่อนรายท่า' : `ดูรายท่า (${activePoint.completedExercisesCount}/${activePoint.totalExercisesCount})`}
                    </button>
                  )}
                </div>
                <p className="text-xs font-semibold text-slate-700 leading-snug line-clamp-2">
                  {doctorBrief.doctorRecommendations[0] ?? 'ฝึกต่อเนื่องตามตารางที่กำหนด'}
                </p>
              </div>
            </div>
          )}

          {/* Optional Collapsible Per-Exercise Cards (Collapsed by default in full-viewport dashboard so chart stays huge!) */}
          {activePoint && (
            <div
              data-testid="stock-day-breakdown-inspector"
              hidden={fillHeight && !showBreakdownDrawer}
              className={
                fillHeight && !showBreakdownDrawer
                  ? 'hidden'
                  : 'shrink-0 rounded-xl border border-sky-200/80 bg-slate-50/70 p-2.5 space-y-2'
              }
            >
              <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                <span className="font-black text-slate-900">
                  รายละเอียดวันที่ {activePoint.labelFullTh} • {formatPoints(isSmartMode ? activePoint.smartEarnedScore : activePoint.rawEarnedScore)}/{activePoint.maxTotalScore} คะแนน ({currentPct.toFixed(1)}%)
                </span>
                {fillHeight && totalBreakdownPages > 1 && (
                  <div className="inline-flex items-center rounded-lg border border-slate-200 bg-white p-0.5">
                    <button
                      type="button"
                      aria-label="ท่าก่อนหน้า"
                      onClick={() => setBreakdownPage((p) => (p - 1 + totalBreakdownPages) % totalBreakdownPages)}
                      className="px-2 py-0.5 text-xs font-black text-slate-700 hover:text-[#0F4C81] cursor-pointer"
                    >
                      &lt;
                    </button>
                    <span className="px-1.5 text-xs font-black text-slate-800 tabular-nums">
                      {safeBreakdownPage + 1}/{totalBreakdownPages}
                    </span>
                    <button
                      type="button"
                      aria-label="ท่าถัดไป"
                      onClick={() => setBreakdownPage((p) => (p + 1) % totalBreakdownPages)}
                      className="px-2 py-0.5 text-xs font-black text-slate-700 hover:text-[#0F4C81] cursor-pointer"
                    >
                      &gt;
                    </button>
                  </div>
                )}
              </div>

              <div className="grid grid-cols-2 gap-1.5 md:grid-cols-3">
                {visibleBreakdownItems.map((ex) => {
                  const earnedVal = isSmartMode ? ex.smartEarnedPoints : ex.rawEarnedPoints;
                  const pctVal = isSmartMode ? ex.smartPct : ex.rawPct;
                  const friendlyTitle = formatFriendlyStockExerciseName(ex.exerciseNameTh, ex.exerciseCode);
                  return (
                    <div
                      key={ex.exerciseCode}
                      data-testid={`stock-ex-contrib-${ex.exerciseCode}`}
                      className="rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 flex items-center justify-between gap-2 text-xs"
                    >
                      <span className="font-black text-slate-900 truncate">{friendlyTitle}</span>
                      <span className="shrink-0 font-black text-emerald-700 tabular-nums">
                        {ex.played
                          ? `${ex.achievedRomDeg !== null ? `${ex.achievedRomDeg.toFixed(0)}°` : `${pctVal.toFixed(0)}%`} • ${formatPoints(earnedVal)}/${ex.maxPoints}`
                          : 'รอฝึก'}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

