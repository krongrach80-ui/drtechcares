import React from 'react';

interface QrVersionSpec {
  version: number;
  size: number;
  dataCodewords: number;
  ecCodewords: number;
  alignmentCenters: readonly number[];
}

// ISO/IEC 18004 Error Correction Level L (all single-block RS versions 1..5, up to 106 UTF-8 bytes)
const QR_LEVEL_L_SPECS: readonly QrVersionSpec[] = [
  { version: 1, size: 21, dataCodewords: 19, ecCodewords: 7, alignmentCenters: [] },
  { version: 2, size: 25, dataCodewords: 34, ecCodewords: 10, alignmentCenters: [6, 18] },
  { version: 3, size: 29, dataCodewords: 55, ecCodewords: 15, alignmentCenters: [6, 22] },
  { version: 4, size: 33, dataCodewords: 80, ecCodewords: 20, alignmentCenters: [6, 26] },
  { version: 5, size: 37, dataCodewords: 108, ecCodewords: 26, alignmentCenters: [6, 30] },
];

const GF_EXP = new Uint8Array(512);
const GF_LOG = new Uint8Array(256);
(() => {
  let x = 1;
  for (let i = 0; i < 255; i++) {
    GF_EXP[i] = x;
    GF_LOG[x] = i;
    x <<= 1;
    if (x & 0x100) x ^= 0x11d;
  }
  for (let i = 255; i < 512; i++) {
    GF_EXP[i] = GF_EXP[i - 255];
  }
})();

function gfMul(a: number, b: number): number {
  if (a === 0 || b === 0) return 0;
  return GF_EXP[GF_LOG[a] + GF_LOG[b]];
}

function buildReedSolomonEcc(data: Uint8Array, ecCount: number): Uint8Array {
  let gen = new Uint8Array([1]);
  for (let i = 0; i < ecCount; i++) {
    const next = new Uint8Array(gen.length + 1);
    const root = GF_EXP[i];
    for (let j = 0; j < gen.length; j++) {
      next[j] ^= gen[j];
      next[j + 1] ^= gfMul(gen[j], root);
    }
    gen = next;
  }

  const remainder = new Uint8Array(ecCount);
  for (let i = 0; i < data.length; i++) {
    const factor = data[i] ^ remainder[0];
    remainder.copyWithin(0, 1);
    remainder[ecCount - 1] = 0;
    if (factor !== 0) {
      for (let j = 0; j < ecCount; j++) {
        remainder[j] ^= gfMul(gen[j + 1], factor);
      }
    }
  }
  return remainder;
}

function encodeUtf8Bytes(input: string): Uint8Array {
  if (typeof TextEncoder !== 'undefined') {
    return new TextEncoder().encode(input);
  }
  const bytes: number[] = [];
  for (let i = 0; i < input.length; i++) {
    const code = input.charCodeAt(i);
    if (code < 0x80) {
      bytes.push(code);
    } else if (code < 0x800) {
      bytes.push(0xc0 | (code >> 6), 0x80 | (code & 0x3f));
    } else {
      bytes.push(0xe0 | (code >> 12), 0x80 | ((code >> 6) & 0x3f), 0x80 | (code & 0x3f));
    }
  }
  return new Uint8Array(bytes);
}

function computeFormatBitsLevelL(mask: number): number {
  // Format data for ECC Level L (01) + 3-bit mask index
  const data = (0b01 << 3) | (mask & 0x07);
  let rem = data << 10;
  for (let bit = 14; bit >= 10; bit--) {
    if ((rem >> bit) & 1) {
      rem ^= 0x537 << (bit - 10);
    }
  }
  return ((data << 10) | rem) ^ 0x5412;
}

function maskBit(mask: number, r: number, c: number): boolean {
  switch (mask) {
    case 0:
      return (r + c) % 2 === 0;
    case 1:
      return r % 2 === 0;
    case 2:
      return c % 3 === 0;
    case 3:
      return (r + c) % 3 === 0;
    case 4:
      return (Math.floor(r / 2) + Math.floor(c / 3)) % 2 === 0;
    case 5:
      return ((r * c) % 2) + ((r * c) % 3) === 0;
    case 6:
      return (((r * c) % 2) + ((r * c) % 3)) % 2 === 0;
    default:
      return (((r + c) % 2) + ((r * c) % 3)) % 2 === 0;
  }
}

function computePenaltyScore(matrix: boolean[][]): number {
  const n = matrix.length;
  let penalty = 0;

  // Rule 1: adjacent modules in row/col with same color
  for (let r = 0; r < n; r++) {
    let runColor = matrix[r][0];
    let runLen = 1;
    for (let c = 1; c < n; c++) {
      if (matrix[r][c] === runColor) {
        runLen += 1;
        if (runLen === 5) penalty += 3;
        else if (runLen > 5) penalty += 1;
      } else {
        runColor = matrix[r][c];
        runLen = 1;
      }
    }
  }
  for (let c = 0; c < n; c++) {
    let runColor = matrix[0][c];
    let runLen = 1;
    for (let r = 1; r < n; r++) {
      if (matrix[r][c] === runColor) {
        runLen += 1;
        if (runLen === 5) penalty += 3;
        else if (runLen > 5) penalty += 1;
      } else {
        runColor = matrix[r][c];
        runLen = 1;
      }
    }
  }

  // Rule 2: 2x2 blocks of same color
  for (let r = 0; r < n - 1; r++) {
    for (let c = 0; c < n - 1; c++) {
      const v = matrix[r][c];
      if (v === matrix[r][c + 1] && v === matrix[r + 1][c] && v === matrix[r + 1][c + 1]) {
        penalty += 3;
      }
    }
  }

  // Rule 4: balance of dark modules
  let dark = 0;
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      if (matrix[r][c]) dark += 1;
    }
  }
  const ratioPct = (dark * 100) / (n * n);
  const deviation = Math.floor(Math.abs(ratioPct - 50) / 5);
  penalty += deviation * 10;

  return penalty;
}

/**
 * Generates a real, phone-scannable ISO/IEC 18004 QR Code matrix (Byte Mode, ECC Level L,
 * Versions 1..5 up to 106 UTF-8 bytes) with zero external dependencies.
 */
export function generateQrMatrix(rawText: string): boolean[][] {
  const safeText = rawText.trim() || 'https://drtechcares.app';
  let bytes = encodeUtf8Bytes(safeText);
  if (bytes.length > 106) {
    bytes = bytes.slice(0, 106);
  }

  // Byte mode overhead is 2 bytes (4-bit mode + 8-bit length + 4-bit terminator = 16 bits)
  const spec =
    QR_LEVEL_L_SPECS.find((s) => bytes.length <= s.dataCodewords - 2) ??
    QR_LEVEL_L_SPECS[QR_LEVEL_L_SPECS.length - 1];
  const size = spec.size;

  // Build bit stream -> data codewords
  const bits: number[] = [];
  const pushBits = (val: number, len: number) => {
    for (let i = len - 1; i >= 0; i--) {
      bits.push((val >> i) & 1);
    }
  };
  pushBits(0b0100, 4); // Byte mode
  pushBits(bytes.length, 8);
  for (let i = 0; i < bytes.length; i++) {
    pushBits(bytes[i], 8);
  }
  const capacityBits = spec.dataCodewords * 8;
  const terminatorLen = Math.min(4, capacityBits - bits.length);
  pushBits(0, terminatorLen);
  while (bits.length % 8 !== 0) bits.push(0);

  const dataCodewords = new Uint8Array(spec.dataCodewords);
  let byteIdx = 0;
  for (let i = 0; i < bits.length; i += 8) {
    let b = 0;
    for (let j = 0; j < 8; j++) b = (b << 1) | bits[i + j];
    dataCodewords[byteIdx++] = b;
  }
  const padBytes = [0xec, 0x11];
  let padIdx = 0;
  while (byteIdx < spec.dataCodewords) {
    dataCodewords[byteIdx++] = padBytes[padIdx++ % 2];
  }

  const eccCodewords = buildReedSolomonEcc(dataCodewords, spec.ecCodewords);
  const fullCodewords = new Uint8Array(dataCodewords.length + eccCodewords.length);
  fullCodewords.set(dataCodewords, 0);
  fullCodewords.set(eccCodewords, dataCodewords.length);

  // Function modules matrix + fixed reservation mask
  const baseMatrix: boolean[][] = Array.from({ length: size }, () => Array(size).fill(false));
  const isFunction: boolean[][] = Array.from({ length: size }, () => Array(size).fill(false));

  const setFunctionModule = (r: number, c: number, dark: boolean) => {
    if (r < 0 || c < 0 || r >= size || c >= size) return;
    baseMatrix[r][c] = dark;
    isFunction[r][c] = true;
  };

  const drawFinder = (topR: number, leftC: number) => {
    for (let dr = -1; dr <= 7; dr++) {
      for (let dc = -1; dc <= 7; dc++) {
        const r = topR + dr;
        const c = leftC + dc;
        if (r < 0 || c < 0 || r >= size || c >= size) continue;
        const dist = Math.max(Math.abs(dr - 3), Math.abs(dc - 3));
        const dark = dist !== 2 && dist !== 4;
        setFunctionModule(r, c, dark);
      }
    }
  };

  drawFinder(0, 0);
  drawFinder(0, size - 7);
  drawFinder(size - 7, 0);

  // Timing patterns
  for (let i = 8; i < size - 8; i++) {
    setFunctionModule(6, i, i % 2 === 0);
    setFunctionModule(i, 6, i % 2 === 0);
  }

  // Alignment patterns (Versions 2..5)
  for (const rCenter of spec.alignmentCenters) {
    for (const cCenter of spec.alignmentCenters) {
      const overlapsFinder =
        (rCenter <= 8 && cCenter <= 8) ||
        (rCenter <= 8 && cCenter >= size - 9) ||
        (rCenter >= size - 9 && cCenter <= 8);
      if (overlapsFinder) continue;
      for (let dr = -2; dr <= 2; dr++) {
        for (let dc = -2; dc <= 2; dc++) {
          const dist = Math.max(Math.abs(dr), Math.abs(dc));
          setFunctionModule(rCenter + dr, cCenter + dc, dist !== 1);
        }
      }
    }
  }

  // Reserve format info strips & dark module
  for (let i = 0; i <= 8; i++) {
    if (i !== 6) {
      setFunctionModule(8, i, false);
      setFunctionModule(i, 8, false);
    }
  }
  for (let i = size - 8; i < size; i++) {
    setFunctionModule(8, i, false);
  }
  for (let i = size - 7; i < size; i++) {
    setFunctionModule(i, 8, false);
  }
  setFunctionModule(4 * spec.version + 9, 8, true);

  // Place data + ECC bits in standard upward/downward zigzag
  let bitStreamIdx = 0;
  const totalBits = fullCodewords.length * 8;
  let goingUp = true;
  for (let rightCol = size - 1; rightCol >= 1; rightCol -= 2) {
    if (rightCol === 6) rightCol = 5;
    for (let step = 0; step < size; step++) {
      const r = goingUp ? size - 1 - step : step;
      for (let dx = 0; dx < 2; dx++) {
        const c = rightCol - dx;
        if (isFunction[r][c]) continue;
        let dark = false;
        if (bitStreamIdx < totalBits) {
          dark = ((fullCodewords[bitStreamIdx >> 3] >> (7 - (bitStreamIdx & 7))) & 1) === 1;
          bitStreamIdx += 1;
        }
        baseMatrix[r][c] = dark;
      }
    }
    goingUp = !goingUp;
  }

  // Evaluate all 8 masks and pick the lowest penalty
  let bestMatrix: boolean[][] | null = null;
  let bestPenalty = Number.POSITIVE_INFINITY;

  for (let mask = 0; mask < 8; mask++) {
    const candidate = baseMatrix.map((row, r) =>
      row.map((val, c) => (isFunction[r][c] ? val : maskBit(mask, r, c) ? !val : val)),
    );

    // Stamp 15-bit format info (MSB = bit 14 -> LSB = bit 0)
    const fmt = computeFormatBitsLevelL(mask);
    const fmtBit = (idx: number) => ((fmt >> idx) & 1) === 1;

    // Copy 1 (top-left)
    for (let c = 0; c <= 5; c++) candidate[8][c] = fmtBit(14 - c);
    candidate[8][7] = fmtBit(8);
    candidate[8][8] = fmtBit(7);
    candidate[7][8] = fmtBit(6);
    for (let r = 5; r >= 0; r--) candidate[r][8] = fmtBit(r);

    // Copy 2 (bottom-left & top-right)
    for (let i = 0; i < 7; i++) candidate[size - 1 - i][8] = fmtBit(14 - i);
    for (let i = 0; i < 8; i++) candidate[8][size - 8 + i] = fmtBit(7 - i);

    const score = computePenaltyScore(candidate);
    if (score < bestPenalty) {
      bestPenalty = score;
      bestMatrix = candidate;
    }
  }

  return bestMatrix ?? baseMatrix;
}

export interface ClinicalQrCodeSvgProps {
  value: string;
  size?: number;
  label?: string;
  className?: string;
}

/**
 * Pure-SVG scannable QR code component for Hospital Clinical PDF Reports & Universal EHR links.
 */
export function ClinicalQrCodeSvg({
  value,
  size = 112,
  label = 'QR Code ลิงก์เวชระเบียนอิเล็กทรอนิกส์ผู้ป่วย',
  className = '',
}: ClinicalQrCodeSvgProps) {
  const matrix = generateQrMatrix(value);
  const quiet = 3;
  const viewSize = matrix.length + quiet * 2;
  const pathParts: string[] = [];

  for (let r = 0; r < matrix.length; r++) {
    for (let c = 0; c < matrix.length; c++) {
      if (matrix[r][c]) {
        pathParts.push(`M${c + quiet},${r + quiet}h1v1h-1z`);
      }
    }
  }

  return React.createElement(
    'svg',
    {
      role: 'img',
      'aria-label': label,
      'data-testid': 'clinical-qr-code-svg',
      width: size,
      height: size,
      viewBox: `0 0 ${viewSize} ${viewSize}`,
      shapeRendering: 'crispEdges',
      className,
    },
    React.createElement('rect', { x: 0, y: 0, width: viewSize, height: viewSize, fill: '#ffffff' }),
    React.createElement('path', { d: pathParts.join(''), fill: '#0f172a' }),
  );
}
