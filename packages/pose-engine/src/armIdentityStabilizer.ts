import type { PoseLandmark } from './types';

/**
 * Prevents left/right elbow (13, 14), wrist (15, 16), and attached palm extremity
 * landmark identity swaps during transient crossings near midline by checking
 * visibility-gated anatomical continuity scaled dynamically to shoulder width.
 */
export function stabilizeArmIdentity(
  current: readonly PoseLandmark[],
  previous: readonly PoseLandmark[] | null,
): PoseLandmark[] {
  const cloned = current.map((lm) => ({ ...lm }));
  if (!previous || cloned.length < 17 || previous.length < 17) return cloned;

  const lSh = cloned[11];
  const rSh = cloned[12];
  const shSpan = lSh && rSh ? Math.hypot(rSh.x - lSh.x, rSh.y - lSh.y) : 0.22;
  const swapMargin = Math.max(0.05, Math.min(0.12, shSpan * 0.35));

  const isReliable = (lm: PoseLandmark | undefined): lm is PoseLandmark =>
    Boolean(lm && Number.isFinite(lm.x) && Number.isFinite(lm.y) && (lm.visibility ?? 1) >= 0.25);

  const swapCheck = (leftIdx: number, rightIdx: number): boolean => {
    const curL = cloned[leftIdx];
    const curR = cloned[rightIdx];
    const prevL = previous[leftIdx];
    const prevR = previous[rightIdx];
    if (!isReliable(curL) || !isReliable(curR) || !isReliable(prevL) || !isReliable(prevR)) {
      return false;
    }

    const normalCost =
      Math.hypot(curL.x - prevL.x, curL.y - prevL.y) +
      Math.hypot(curR.x - prevR.x, curR.y - prevR.y);
    const swappedCost =
      Math.hypot(curL.x - prevR.x, curL.y - prevR.y) +
      Math.hypot(curR.x - prevL.x, curR.y - prevL.y);

    if (swappedCost + swapMargin < normalCost) {
      cloned[leftIdx] = { ...curR };
      cloned[rightIdx] = { ...curL };
      return true;
    }
    return false;
  };

  swapCheck(13, 14);
  const swappedWrists = swapCheck(15, 16);
  if (swappedWrists && cloned.length >= 23) {
    for (const [lExt, rExt] of [
      [17, 18],
      [19, 20],
      [21, 22],
    ] as const) {
      const lmL = cloned[lExt];
      const lmR = cloned[rExt];
      if (lmL && lmR) {
        cloned[lExt] = { ...lmR };
        cloned[rExt] = { ...lmL };
      }
    }
  }
  return cloned;
}

