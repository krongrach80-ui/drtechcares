import React, { useMemo, useState } from 'react';
import type {
  RehabStockDayPoint,
  RehabStockIndexSummary,
  StockScoreCalculationMode,
} from '../utils/rehabStockScoreEngine';
import { GameRankBadge } from './GameJuiceEffects';

export interface RehabStockPerformanceCardProps {
  patientName: string;
  patientHn: string;
  stockIndex: RehabStockIndexSummary;
}

type StockChartStyle = 'AREA' | 'CANDLE';

function formatPoints(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}

export function RehabStockPerformanceCard({
  patientName,
  patientHn,
  stockIndex,
}: RehabStockPerformanceCardProps) {
  const [chartStyle, setChartStyle] = useState<StockChartStyle>('AREA');
  const [scoreMode, setScoreMode] = useState<StockScoreCalculationMode>('SMART_WEIGHTED');
  const [showMa3, setShowMa3] = useState<boolean>(true);
  const [activeDateKey, setActiveDateKey] = useState<string | null>(null);

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

  // Pure 60FPS Dual-Pane SVG Geometry (Upper Price/% Pane + Lower Volume Reps Pane)
  const svgGeom = useMemo(() => {
    const pts = stockIndex.points.slice(-18);
    const width = Math.max(720, pts.length > 10 ? pts.length * 64 + 120 : 720);
    const height = 316;
    const padL = 62;
    const padR = 58;
    const padT = 24;
    const priceH = 182;
    const gapBetweenPanes = 16;
    const volTop = padT + priceH + gapBetweenPanes;
    const volH = 44;
    const innerPadX = 32;
    const plotW = width - padL - padR;
    const usableW = Math.max(40, plotW - innerPadX * 2);

    const maxVol = Math.max(
      10,
      stockIndex.maxTotalScore,
      ...pts.map((p) => p.volumeReps),
    );

    const yForPct = (pct: number) =>
      padT + priceH - (Math.max(0, Math.min(100, pct)) / 100) * priceH;

    const mapped = pts.map((p, idx) => {
      const x =
        pts.length === 1
          ? padL + plotW / 2
          : padL + innerPadX + (idx / Math.max(1, pts.length - 1)) * usableW;
      const pctVal = isSmartMode ? p.smartTotalPct : p.rawTotalPct;
      const openVal = isSmartMode ? p.openPct : p.rawOpenPct;
      const highVal = isSmartMode ? p.highPct : p.rawHighPct;
      const lowVal = isSmartMode ? p.lowPct : p.rawLowPct;
      const closeVal = isSmartMode ? p.closePct : p.rawClosePct;
      const ma3Val = isSmartMode ? p.ma3SmartPct : p.ma3RawPct;

      const y = yForPct(pctVal);
      const yOpen = yForPct(openVal);
      const yHigh = yForPct(highVal);
      const yLow = yForPct(lowVal);
      const yClose = yForPct(closeVal);
      const yMa3 = yForPct(ma3Val);
      const bullish = closeVal >= openVal;

      const volBarH = Math.max(3, (p.volumeReps / maxVol) * volH);
      const volY = volTop + volH - volBarH;

      return {
        point: p,
        x,
        y,
        pctVal,
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
  }, [stockIndex.points, stockIndex.maxTotalScore, isSmartMode]);

  const activeMapped = useMemo(() => {
    if (!activePoint || svgGeom.mapped.length === 0) return null;
    return (
      svgGeom.mapped.find((m) => m.point.dateKey === activePoint.dateKey) ??
      svgGeom.mapped[svgGeom.mapped.length - 1]
    );
  }, [svgGeom.mapped, activePoint]);

  return (
    <div
      data-testid="rehab-stock-performance-card"
      className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm space-y-4 animate-view-enter"
    >
      {/* Stock Ticker Header Strip */}
      <div className="flex flex-wrap items-start justify-between gap-3 border-b border-slate-100 pb-3.5">
        <div className="space-y-1.5 min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center gap-1.5 rounded-lg bg-[#0F4C81] px-2.5 py-0.5 text-[10px] font-black tracking-wider text-white">
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.4"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="h-3 w-3"
                aria-hidden="true"
              >
                <polyline points="22 7 13.5 15.5 8.5 10.5 2 17" />
                <polyline points="16 7 22 7 22 13" />
              </svg>
              <span>REHAB INDEX • {patientHn || 'PATIENT'}</span>
            </span>
            <span className="rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-0.5 text-[10px] font-extrabold text-emerald-800">
              คะแนนเต็มรวม {stockIndex.maxTotalScore} คะแนน ({stockIndex.totalPrescribedExercises} ท่าที่แพทย์/PT กำหนด)
            </span>
          </div>

          <div className="flex flex-wrap items-baseline gap-2.5">
            <span
              data-testid="stock-index-current-pct"
              className={`text-3xl font-black tracking-tight tabular-nums ${
                headlineTrend === 'DOWN' ? 'text-rose-600' : 'text-emerald-700'
              }`}
            >
              {activePoint ? `${currentPct.toFixed(1)}%` : '0.0%'}
            </span>

            <span
              data-testid="stock-index-score-fraction"
              className="rounded-xl border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs font-black text-slate-900 tabular-nums"
            >
              {formatPoints(currentEarned)} / {stockIndex.maxTotalScore} คะแนน
            </span>

            <GameRankBadge pct={activePoint ? currentPct : 0} />

            {currentDeltaPct !== null && (
              <span
                className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-[11px] font-extrabold tabular-nums ${
                  currentDeltaPct > 0
                    ? 'border-emerald-200 bg-emerald-50 text-emerald-700'
                    : currentDeltaPct < 0
                      ? 'border-rose-200 bg-rose-50 text-rose-700'
                      : 'border-slate-200 bg-slate-50 text-slate-600'
                }`}
              >
                <svg
                  viewBox="0 0 12 12"
                  fill="currentColor"
                  className="h-2.5 w-2.5"
                  aria-hidden="true"
                >
                  {currentDeltaPct < 0 ? (
                    <path d="M6 10L1 3h10L6 10z" />
                  ) : (
                    <path d="M6 2l5 7H1l5-7z" />
                  )}
                </svg>
                <span>
                  {currentDeltaPct >= 0 ? '+' : ''}
                  {currentDeltaPct.toFixed(1)}%
                  {currentDeltaPts !== null
                    ? ` (${currentDeltaPts >= 0 ? '+' : ''}${formatPoints(currentDeltaPts)} คะแนน)`
                    : ''}
                </span>
              </span>
            )}
          </div>

          <p className="text-[11px] font-semibold text-slate-500">
            ดัชนีเปอร์เซ็นต์รวมทุกท่ากายภาพของ {patientName} •{' '}
            <span className="font-bold text-[#0F4C81]">{stockIndex.momentumLabelTh}</span>
          </p>
        </div>

        {/* Stock Chart Controls */}
        <div className="flex flex-wrap items-center gap-1.5">
          <div className="inline-flex rounded-xl border border-slate-200 bg-slate-50 p-0.5 text-[10px] font-extrabold">
            <button
              type="button"
              data-testid="stock-style-area"
              onClick={() => setChartStyle('AREA')}
              className={`rounded-lg px-2.5 py-1 cursor-pointer transition-colors ${
                chartStyle === 'AREA'
                  ? 'bg-[#0F4C81] text-white shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              กราฟพื้นที่ (Area)
            </button>
            <button
              type="button"
              data-testid="stock-style-candle"
              onClick={() => setChartStyle('CANDLE')}
              className={`rounded-lg px-2.5 py-1 cursor-pointer transition-colors ${
                chartStyle === 'CANDLE'
                  ? 'bg-emerald-600 text-white shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              แท่งเทียน (Candle)
            </button>
          </div>

          <div className="inline-flex rounded-xl border border-emerald-200 bg-emerald-50/70 p-0.5 text-[10px] font-extrabold">
            <button
              type="button"
              data-testid="stock-mode-smart"
              onClick={() => setScoreMode('SMART_WEIGHTED')}
              className={`rounded-lg px-2.5 py-1 cursor-pointer transition-colors ${
                scoreMode === 'SMART_WEIGHTED'
                  ? 'bg-emerald-600 text-white shadow-xs'
                  : 'text-emerald-800 hover:bg-white/80'
              }`}
            >
              อัจฉริยะ (ROM+แม่นยำ)
            </button>
            <button
              type="button"
              data-testid="stock-mode-raw"
              onClick={() => setScoreMode('RAW_REPS')}
              className={`rounded-lg px-2.5 py-1 cursor-pointer transition-colors ${
                scoreMode === 'RAW_REPS'
                  ? 'bg-[#0F4C81] text-white shadow-xs'
                  : 'text-[#0F4C81] hover:bg-white/80'
              }`}
            >
              นับครั้งตรง (1 ครั้ง=1 แต้ม)
            </button>
          </div>

          <button
            type="button"
            onClick={() => setShowMa3((v) => !v)}
            className={`rounded-lg border px-2.5 py-1 text-[10px] font-extrabold cursor-pointer ${
              showMa3
                ? 'border-sky-300 bg-sky-50 text-[#0F4C81]'
                : 'border-slate-200 bg-white text-slate-500'
            }`}
          >
            เส้น MA3
          </button>
        </div>
      </div>

      {/* Stock Market Summary Metrics Strip */}
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">
        <div className="game-card-surface rounded-xl border border-slate-200/80 bg-slate-50/70 px-3 py-2">
          <div className="text-[10px] font-bold text-slate-500">คะแนนเต็มชุดฝึก</div>
          <div className="text-xs font-black text-slate-900 tabular-nums">
            {stockIndex.maxTotalScore} คะแนน ({stockIndex.totalPrescribedExercises} ท่า)
          </div>
        </div>
        <div className="game-card-surface rounded-xl border border-emerald-200/70 bg-emerald-50/40 px-3 py-2">
          <div className="text-[10px] font-bold text-slate-500">สูงสุด (High)</div>
          <div className="text-xs font-black text-emerald-700 tabular-nums">
            {highStat !== null ? `${highStat.toFixed(1)}%` : '-'}
          </div>
        </div>
        <div className="game-card-surface rounded-xl border border-rose-200/60 bg-rose-50/30 px-3 py-2">
          <div className="text-[10px] font-bold text-slate-500">ต่ำสุด (Low)</div>
          <div className="text-xs font-black text-rose-700 tabular-nums">
            {lowStat !== null ? `${lowStat.toFixed(1)}%` : '-'}
          </div>
        </div>
        <div className="game-card-surface rounded-xl border border-sky-200/70 bg-sky-50/40 px-3 py-2">
          <div className="text-[10px] font-bold text-slate-500">เฉลี่ยรวม / MA3</div>
          <div className="text-xs font-black text-[#0F4C81] tabular-nums">
            {avgStat !== null ? `${avgStat.toFixed(1)}%` : '-'} /{' '}
            {ma3Stat !== null ? `${ma3Stat.toFixed(1)}%` : '-'}
          </div>
        </div>
        <div className="game-card-surface rounded-xl border border-teal-200/70 bg-teal-50/40 px-3 py-2 col-span-2 sm:col-span-1">
          <div className="text-[10px] font-bold text-slate-500">ครบตามท่าล่าสุด</div>
          <div className="text-xs font-black text-teal-800 tabular-nums">
            {activePoint
              ? `${activePoint.completedExercisesCount}/${activePoint.totalExercisesCount} ท่า`
              : `0/${stockIndex.totalPrescribedExercises} ท่า`}
          </div>
        </div>
      </div>

      {/* Dual-Pane Stock Chart Canvas OR Zero-Session Prescribed Score Pool Blueprint */}
      {svgGeom.mapped.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-sky-200 bg-gradient-to-br from-slate-50 via-white to-emerald-50/40 p-5 space-y-3">
          <div className="text-center space-y-1">
            <div className="text-xs font-black text-slate-800">
              {stockIndex.totalPrescribedExercises > 0
                ? `โครงสร้างคะแนนเต็มพร้อมคำนวณแล้ว: ${stockIndex.maxTotalScore} คะแนน จาก ${stockIndex.totalPrescribedExercises} ท่าที่แพทย์/PT สั่ง`
                : 'ยังไม่พบท่ากายภาพที่แพทย์/นักกายภาพจัดให้หรือประวัติการฝึกสำหรับผู้ป่วยรายนี้'}
            </div>
            <p className="text-[11px] text-slate-500 max-w-lg mx-auto">
              {stockIndex.totalPrescribedExercises > 0
                ? 'เมื่อผู้ป่วยฝึกครบแต่ละท่า ระบบจะรวมคะแนนสะสมเทียบคะแนนเต็มเป็นเปอร์เซ็นต์อัจฉริยะและแสดงเป็นกราฟแนวโน้มคล้ายกราฟหุ้นทันที'
                : 'เพิ่มท่ากายภาพในตารางนัด/คอร์สของผู้ป่วย หรือกดปุ่ม "ดึงผลเล่นคอร์สและบันทึกอัตโนมัติ" เพื่อเริ่มสร้างกราฟคะแนนรวม'}
            </p>
          </div>
          {stockIndex.prescribedExercises.length > 0 && (
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
              {stockIndex.prescribedExercises.map((spec) => (
                <div
                  key={spec.exerciseCode}
                  className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-[11px] flex items-center justify-between gap-2"
                >
                  <span className="font-bold text-slate-800 truncate">{spec.exerciseNameTh}</span>
                  <span className="shrink-0 rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-black text-slate-700">
                    0 / {spec.targetReps} คะแนน
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      ) : (
        <div className="space-y-3">
          <div className="w-full overflow-x-auto rounded-2xl border border-slate-200/80 bg-gradient-to-b from-slate-50/50 via-white to-slate-50/40 p-2">
            <svg
              viewBox={`0 0 ${svgGeom.width} ${svgGeom.height}`}
              className="w-full min-w-[580px] h-auto select-none"
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

              {/* Price/% Gridlines: Left Axis = Earned Points, Right Axis = % (Authentic Stock Layout) */}
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
                      strokeWidth="1"
                    />
                    <text
                      x={svgGeom.padL - 8}
                      y={y + 3.5}
                      textAnchor="end"
                      fill="#64748b"
                      fontSize="9.5"
                      fontWeight="700"
                    >
                      {formatPoints(ptsTick)} แต้ม
                    </text>
                    <text
                      x={svgGeom.width - svgGeom.padR + 8}
                      y={y + 3.5}
                      textAnchor="start"
                      fill="#475569"
                      fontSize="9.5"
                      fontWeight="800"
                    >
                      {pctTick}%
                    </text>
                  </g>
                );
              })}

              {/* 80% Clinical Excellence Target Line */}
              <g>
                <line
                  x1={svgGeom.padL}
                  y1={svgGeom.yForPct(80)}
                  x2={svgGeom.width - svgGeom.padR}
                  y2={svgGeom.yForPct(80)}
                  stroke="#0284c7"
                  strokeDasharray="6 4"
                  strokeWidth="1.2"
                  opacity="0.7"
                />
                <text
                  x={svgGeom.padL + 4}
                  y={svgGeom.yForPct(80) - 5}
                  textAnchor="start"
                  fill="#0369a1"
                  fontSize="8.8"
                  fontWeight="800"
                >
                  เกณฑ์ฟื้นตัวดีเยี่ยม 80%
                </text>
              </g>

              {/* Active Crosshair (Horizontal + Vertical + Right Axis Price Pill) */}
              {activeMapped && (
                <g pointerEvents="none">
                  <line
                    x1={activeMapped.x}
                    y1={svgGeom.padT}
                    x2={activeMapped.x}
                    y2={svgGeom.volTop + svgGeom.volH}
                    stroke="#64748b"
                    strokeDasharray="3 3"
                    strokeWidth="1.1"
                    opacity="0.65"
                  />
                  <line
                    x1={svgGeom.padL}
                    y1={activeMapped.y}
                    x2={svgGeom.width - svgGeom.padR}
                    y2={activeMapped.y}
                    stroke={accentStroke}
                    strokeDasharray="3 3"
                    strokeWidth="1.1"
                    opacity="0.65"
                  />
                  <rect
                    x={svgGeom.width - svgGeom.padR + 2}
                    y={activeMapped.y - 9}
                    width="52"
                    height="18"
                    rx="5"
                    fill={accentStroke}
                  />
                  <text
                    x={svgGeom.width - svgGeom.padR + 28}
                    y={activeMapped.y + 3.5}
                    textAnchor="middle"
                    fill="#ffffff"
                    fontSize="9.5"
                    fontWeight="900"
                  >
                    {activeMapped.pctVal.toFixed(1)}%
                  </text>
                </g>
              )}

              {/* AREA Mode: Stock Area Fill + Line */}
              {chartStyle === 'AREA' && (
                <>
                  {svgGeom.areaPoints && (
                    <polygon
                      points={svgGeom.areaPoints}
                      fill={
                        headlineTrend === 'DOWN'
                          ? 'url(#stockDownAreaGrad)'
                          : 'url(#stockUpAreaGrad)'
                      }
                    />
                  )}
                  {svgGeom.mapped.length >= 2 && (
                    <polyline
                      data-testid="stock-area-polyline"
                      fill="none"
                      stroke={accentStroke}
                      strokeWidth="2.8"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      points={svgGeom.linePoints}
                    />
                  )}
                </>
              )}

              {/* CANDLE Mode: OHLC Stock Candlesticks */}
              {chartStyle === 'CANDLE' &&
                svgGeom.mapped.map((m) => {
                  const color = m.bullish ? '#059669' : '#e11d48';
                  const bodyTop = Math.min(m.yOpen, m.yClose);
                  const bodyH = Math.max(4, Math.abs(m.yClose - m.yOpen));
                  return (
                    <g key={`candle-${m.point.dateKey}`} data-testid={`stock-candle-${m.point.dateKey}`}>
                      <line
                        x1={m.x}
                        y1={m.yHigh}
                        x2={m.x}
                        y2={m.yLow}
                        stroke={color}
                        strokeWidth="2"
                      />
                      <rect
                        x={m.x - svgGeom.candleW / 2}
                        y={bodyTop}
                        width={svgGeom.candleW}
                        height={bodyH}
                        rx="3"
                        fill={color}
                        stroke={color}
                        strokeWidth="1"
                      />
                    </g>
                  );
                })}

              {/* MA3 Moving Average Dashed Trendline */}
              {showMa3 && svgGeom.mapped.length >= 2 && (
                <polyline
                  data-testid="stock-ma3-polyline"
                  fill="none"
                  stroke="#0284c7"
                  strokeWidth="1.8"
                  strokeDasharray="5 4"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  opacity="0.85"
                  points={svgGeom.ma3Points}
                />
              )}

              {/* Divider & Lower Pane Label: Volume Histogram (Reps Played Per Day) */}
              <line
                x1={svgGeom.padL}
                y1={svgGeom.volTop - 4}
                x2={svgGeom.width - svgGeom.padR}
                y2={svgGeom.volTop - 4}
                stroke="#e2e8f0"
                strokeWidth="1"
              />
              <text
                x={svgGeom.padL - 8}
                y={svgGeom.volTop + 14}
                textAnchor="end"
                fill="#64748b"
                fontSize="8.5"
                fontWeight="700"
              >
                ปริมาณฝึก
              </text>
              <text
                x={svgGeom.padL - 8}
                y={svgGeom.volTop + 26}
                textAnchor="end"
                fill="#94a3b8"
                fontSize="8"
                fontWeight="600"
              >
                (ครั้ง/วัน)
              </text>

              {/* Volume Bars + Interactive Click/Hover Targets + X-Axis Dates */}
              {svgGeom.mapped.map((m) => {
                const isSelected = activeMapped?.point.dateKey === m.point.dateKey;
                const volFill = m.bullish ? '#10b981' : '#f43f5e';
                const labelY = m.y < svgGeom.padT + 18 ? m.y + 18 : m.y - 10;
                return (
                  <g
                    key={m.point.dateKey}
                    onMouseEnter={() => setActiveDateKey(m.point.dateKey)}
                    onClick={() => setActiveDateKey(m.point.dateKey)}
                    className="cursor-pointer"
                  >
                    {/* Wide invisible hit column for effortless crosshair scrubbing */}
                    <rect
                      x={m.x - 22}
                      y={svgGeom.padT}
                      width="44"
                      height={svgGeom.volTop + svgGeom.volH - svgGeom.padT}
                      fill="transparent"
                    />

                    {/* Volume Bar */}
                    <rect
                      x={m.x - svgGeom.volBarW / 2}
                      y={m.volY}
                      width={svgGeom.volBarW}
                      height={m.volBarH}
                      rx="3"
                      fill={volFill}
                      fillOpacity={isSelected ? '0.85' : '0.35'}
                    />

                    {/* Node Marker (Area Mode or Selected Ring) */}
                    {chartStyle === 'AREA' && (
                      <>
                        {isSelected && (
                          <circle
                            cx={m.x}
                            cy={m.y}
                            r="11"
                            fill={accentStroke}
                            fillOpacity="0.16"
                            stroke={accentStroke}
                            strokeWidth="1.2"
                          />
                        )}
                        <circle
                          cx={m.x}
                          cy={m.y}
                          r={isSelected ? '6' : '4.2'}
                          fill={isSelected ? accentStroke : '#ffffff'}
                          stroke={accentStroke}
                          strokeWidth="2.5"
                        />
                        <text
                          x={m.x}
                          y={labelY}
                          textAnchor="middle"
                          fill={isSelected ? accentStroke : '#334155'}
                          fontSize={isSelected ? '10' : '9'}
                          fontWeight="800"
                        >
                          {m.pctVal.toFixed(0)}%
                        </text>
                      </>
                    )}

                    {/* Volume rep count */}
                    <text
                      x={m.x}
                      y={svgGeom.volTop + svgGeom.volH + 12}
                      textAnchor="middle"
                      fill="#64748b"
                      fontSize="8.2"
                      fontWeight="700"
                    >
                      {m.point.volumeReps} ครั้ง
                    </text>

                    {/* Thai Short Date */}
                    <text
                      x={m.x}
                      y={svgGeom.volTop + svgGeom.volH + 25}
                      textAnchor="middle"
                      fill={isSelected ? '#0F4C81' : '#475569'}
                      fontSize="8.8"
                      fontWeight={isSelected ? '900' : '700'}
                    >
                      {m.point.labelShortTh}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>

          {/* Interactive Selected Day Breakdown + Per-Exercise Contribution Grid */}
          {activePoint && (
            <div
              data-testid="stock-day-breakdown-inspector"
              className="rounded-2xl border border-sky-200/80 bg-gradient-to-br from-[#F8FAFC] via-white to-[#F0FDF9] p-4 space-y-3"
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <div className="text-xs font-black text-slate-900">
                    สรุปรายละเอียดวันที่ {activePoint.labelFullTh} • คะแนนรวม{' '}
                    <span className="text-emerald-700">
                      {formatPoints(
                        isSmartMode ? activePoint.smartEarnedScore : activePoint.rawEarnedScore,
                      )}{' '}
                      / {activePoint.maxTotalScore} คะแนน ({currentPct.toFixed(1)}%)
                    </span>
                  </div>
                  <div className="text-[11px] text-slate-600">
                    เล่นครบ {activePoint.completedExercisesCount}/{activePoint.totalExercisesCount} ท่า • ปริมาณรวม{' '}
                    {activePoint.volumeReps} ครั้ง • คุณภาพ ROM เฉลี่ย {activePoint.avgRomQualityPct}% • ความแม่นยำ{' '}
                    {activePoint.avgAccuracyPct}%
                  </div>
                </div>
                <div className="flex flex-wrap items-center gap-1.5 text-[10px] font-extrabold">
                  <span className="rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-0.5 text-emerald-800">
                    อัจฉริยะ {activePoint.smartTotalPct.toFixed(1)}% ({formatPoints(activePoint.smartEarnedScore)}/{activePoint.maxTotalScore})
                  </span>
                  <span className="rounded-full border border-sky-200 bg-sky-50 px-2.5 py-0.5 text-[#0F4C81]">
                    นับครั้งตรง {activePoint.rawTotalPct.toFixed(1)}% ({formatPoints(activePoint.rawEarnedScore)}/{activePoint.maxTotalScore})
                  </span>
                </div>
              </div>

              {/* Per-Exercise Score Contribution Cards (Shows All Doctor-Assigned Exercises) */}
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
                {activePoint.exerciseBreakdown.map((ex) => {
                  const earnedVal = isSmartMode ? ex.smartEarnedPoints : ex.rawEarnedPoints;
                  const pctVal = isSmartMode ? ex.smartPct : ex.rawPct;
                  const barTone = !ex.played
                    ? 'bg-slate-300'
                    : pctVal >= 85
                      ? 'bg-emerald-500'
                      : pctVal >= 60
                        ? 'bg-sky-500'
                        : 'bg-amber-500';
                  return (
                    <div
                      key={ex.exerciseCode}
                      data-testid={`stock-ex-contrib-${ex.exerciseCode}`}
                      className={`game-card-surface rounded-xl border p-2.5 space-y-1.5 ${
                        ex.played
                          ? 'border-slate-200 bg-white'
                          : 'border-dashed border-slate-200 bg-slate-50/70 opacity-80'
                      }`}
                    >
                      <div className="flex items-start justify-between gap-1.5">
                        <span className="text-[11px] font-black text-slate-900 leading-tight line-clamp-1">
                          {ex.exerciseNameTh}
                        </span>
                        <span
                          className={`shrink-0 rounded-full px-2 py-0.5 text-[10px] font-black tabular-nums ${
                            !ex.played
                              ? 'bg-slate-100 text-slate-500'
                              : pctVal >= 85
                                ? 'bg-emerald-100 text-emerald-800'
                                : 'bg-sky-100 text-[#0F4C81]'
                          }`}
                        >
                          {formatPoints(earnedVal)}/{ex.maxPoints}
                        </span>
                      </div>
                      <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100">
                        <div
                          className={`h-full rounded-full transition-all ${ex.played ? 'game-progress-stripes' : ''} ${barTone}`}
                          style={{ width: `${Math.min(100, pctVal)}%` }}
                        />
                      </div>
                      <div className="text-[10px] font-semibold text-slate-500 flex flex-wrap items-center justify-between gap-1">
                        {ex.played ? (
                          <>
                            <span>
                              ทำ {ex.completedReps}/{ex.targetReps} ครั้ง ({pctVal.toFixed(0)}%)
                            </span>
                            <span>
                              {ex.achievedRomDeg !== null
                                ? `ROM ${ex.achievedRomDeg.toFixed(0)}°/${Math.round(ex.targetRomDeg)}°`
                                : `แม่นยำ ${ex.accuracyPct}%`}
                            </span>
                          </>
                        ) : (
                          <span>ยังไม่เล่นในวันนี้ (0/{ex.targetReps} คะแนน)</span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Smart Clinical Insight Strip */}
              <div className="rounded-xl border border-slate-200/80 bg-white/90 px-3 py-2 text-[10.5px] text-slate-600 flex flex-wrap items-center justify-between gap-2">
                <span>
                  <strong className="text-slate-800">สูตรคำนวณอัจฉริยะ:</strong> รวมคะแนนเต็มจากทุกท่าที่แพทย์/PT กำหนด ({stockIndex.maxTotalScore} คะแนน = {stockIndex.totalPrescribedExercises} ท่า) ถ่วงน้ำหนักด้วยจำนวนครั้ง + ความลึกองศา ROM (65%) + ความแม่นยำท่าทาง (35%)
                </span>
                {stockIndex.weakestExercise && !stockIndex.weakestExercise.played && (
                  <span className="font-bold text-amber-700">
                    ท่าที่ยังไม่ครบ: {stockIndex.weakestExercise.exerciseNameTh}
                  </span>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

