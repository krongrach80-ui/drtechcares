import type { PoseLandmark } from './types';

/**
 * Prevents left/right elbow (13, 14) and wrist (15, 16) landmark identity swaps
 * during transient crossings near midline by checking anatomical continuity against
 * the previous frame.
 */
export function stabilizeArmIdentity(
  current: readonly PoseLandmark[],
  previous: readonly PoseLandmark[] | null,
): PoseLandmark[] {
  const cloned = current.map((lm) => ({ ...lm }));
  if (!previous || cloned.length < 17 || previous.length < 17) return cloned;

  const swapCheck = (leftIdx: number, rightIdx: number) => {
    const curL = cloned[leftIdx];
    const curR = cloned[rightIdx];
    const prevL = previous[leftIdx];
    const prevR = previous[rightIdx];
    if (!curL || !curR || !prevL || !prevR) return;

    const normalCost =
      Math.hypot(curL.x - prevL.x, curL.y - prevL.y) +
      Math.hypot(curR.x - prevR.x, curR.y - prevR.y);
    const swappedCost =
      Math.hypot(curL.x - prevR.x, curL.y - prevR.y) +
      Math.hypot(curR.x - prevL.x, curR.y - prevL.y);

    if (swappedCost + 0.08 < normalCost) {
      cloned[leftIdx] = { ...curR };
      cloned[rightIdx] = { ...curL };
    }
  };

  swapCheck(13, 14);
  swapCheck(15, 16);
  return cloned;
}

