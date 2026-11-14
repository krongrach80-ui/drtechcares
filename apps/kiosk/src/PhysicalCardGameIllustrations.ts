/**
 * 100 Bespoke Question-Matched Vector Illustrations for the Dr.Tech.Care Cognitive Rehab Card Bank.
 * Rendered as pure self-contained SVG data URIs (offline-ready, razor sharp at any kiosk DPI).
 */

export type IllustrationCategory =
  | 'food_fruit'
  | 'objects'
  | 'animals'
  | 'counting'
  | 'safety_routine'
  | 'color_shape_time'
  | 'position';

export type CardArtKey =
  // 001..020 food_fruit
  | 'apple'
  | 'banana'
  | 'broccoli'
  | 'milk'
  | 'watermelon'
  | 'carrot'
  | 'fish_food'
  | 'mango'
  | 'orange'
  | 'fried_egg'
  | 'rice_bowl'
  | 'papaya'
  | 'pineapple'
  | 'coconut'
  | 'corn'
  | 'strawberry'
  | 'grapes'
  | 'pumpkin'
  | 'soup_bowl'
  | 'water_glass'
  // 021..040 objects
  | 'key'
  | 'toothbrush'
  | 'eyeglasses'
  | 'umbrella'
  | 'alarm_clock'
  | 'chair'
  | 'stethoscope'
  | 'spoon_fork'
  | 'hair_comb'
  | 'scissors'
  | 'pillow'
  | 'walking_shoes'
  | 'telephone'
  | 'open_book'
  | 'teacup'
  | 'wristwatch'
  | 'flashlight'
  | 'table_fan'
  | 'sun_hat'
  | 'thermometer'
  // 041..055 animals
  | 'elephant'
  | 'dog'
  | 'bird'
  | 'turtle'
  | 'bee'
  | 'cat'
  | 'rabbit'
  | 'tropical_fish'
  | 'butterfly'
  | 'rooster'
  | 'cow'
  | 'duck'
  | 'horse'
  | 'frog'
  | 'crab'
  // 056..070 counting
  | 'count_apples_3'
  | 'count_stars_2plus2'
  | 'count_fingers_5'
  | 'count_car_4wheels'
  | 'count_balls_2'
  | 'count_oranges_6'
  | 'count_balloons_3'
  | 'count_flowers_5'
  | 'count_birds_4'
  | 'count_cups_2'
  | 'count_marbles_7'
  | 'count_blocks_3plus2'
  | 'count_octopus_8'
  | 'count_clock_12'
  | 'count_hearts_1plus1'
  // 071..082 safety_routine
  | 'medicine'
  | 'nonslip_shoes'
  | 'stove_off'
  | 'wash_hands'
  | 'stair_handrail'
  | 'walking_cane'
  | 'bp_monitor'
  | 'safety_helmet'
  | 'seatbelt'
  | 'crosswalk'
  | 'wet_floor_cone'
  | 'shoulder_stretch'
  // 083..092 color_shape_time
  | 'traffic_red'
  | 'traffic_green'
  | 'triangle'
  | 'blue_circle'
  | 'green_square'
  | 'yellow_star'
  | 'morning_sun'
  | 'night_moon'
  | 'morning_clock_8am'
  | 'rainbow'
  // 093..100 position
  | 'arrow_up'
  | 'arrow_down'
  | 'arrow_right'
  | 'arrow_left'
  | 'house_roof_top'
  | 'apple_on_table'
  | 'ball_under_table'
  | 'bird_on_branch';

export interface CardIllustrationSpec {
  cardNumber: number;
  category: IllustrationCategory;
  artKey: CardArtKey;
}

const CATEGORY_THEME: Record<
  IllustrationCategory,
  { bgTop: string; bgBottom: string; accent: string; pillBg: string }
> = {
  food_fruit: { bgTop: '#FFF7ED', bgBottom: '#FFEDD5', accent: '#EA580C', pillBg: '#FED7AA' },
  objects: { bgTop: '#EFF6FF', bgBottom: '#DBEAFE', accent: '#2563EB', pillBg: '#BFDBFE' },
  animals: { bgTop: '#ECFDF5', bgBottom: '#D1FAE5', accent: '#059669', pillBg: '#A7F3D0' },
  counting: { bgTop: '#FAF5FF', bgBottom: '#F3E8FF', accent: '#7C3AED', pillBg: '#DDD6FE' },
  safety_routine: { bgTop: '#FEF2F2', bgBottom: '#FEE2E2', accent: '#DC2626', pillBg: '#FECACA' },
  color_shape_time: { bgTop: '#F0F9FF', bgBottom: '#E0F2FE', accent: '#0284C7', pillBg: '#BAE6FD' },
  position: { bgTop: '#F0FDFA', bgBottom: '#CCFBF1', accent: '#0D9488', pillBg: '#99F6E4' },
};

function starPolygon(cx: number, cy: number, outerR: number, innerR: number): string {
  const pts: string[] = [];
  for (let i = 0; i < 10; i += 1) {
    const r = i % 2 === 0 ? outerR : innerR;
    const angle = (-90 + i * 36) * (Math.PI / 180);
    pts.push(`${(cx + r * Math.cos(angle)).toFixed(1)},${(cy + r * Math.sin(angle)).toFixed(1)}`);
  }
  return pts.join(' ');
}

function drawMiniApple(cx: number, cy: number, s = 1): string {
  return `<g transform="translate(${cx},${cy}) scale(${s})">
    <ellipse cx="0" cy="48" rx="34" ry="8" fill="#0F172A" opacity="0.1"/>
    <path d="M-4,-34 C-2,-52 6,-56 10,-52" stroke="#78350F" stroke-width="6" fill="none" stroke-linecap="round"/>
    <path d="M4,-44 C18,-62 42,-54 34,-34 C22,-26 8,-34 4,-44 Z" fill="#22C55E"/>
    <path d="M0,-32 C-28,-42 -48,-16 -44,18 C-40,48 -18,54 -4,44 C0,42 4,42 8,44 C22,54 42,46 46,16 C50,-18 28,-42 0,-32 Z" fill="#EF4444" stroke="#B91C1C" stroke-width="2.5"/>
    <ellipse cx="-20" cy="-6" rx="8" ry="18" transform="rotate(-18 -20 -6)" fill="#FCA5A5" opacity="0.55"/>
  </g>`;
}

function drawMiniOrange(cx: number, cy: number, r = 34): string {
  return `<g transform="translate(${cx},${cy})">
    <ellipse cx="0" cy="${r + 6}" rx="${r * 0.85}" ry="7" fill="#0F172A" opacity="0.1"/>
    <circle cx="0" cy="0" r="${r}" fill="#F97316" stroke="#C2410C" stroke-width="2.5"/>
    <circle cx="-10" cy="-12" r="${r * 0.28}" fill="#FDBA74" opacity="0.5"/>
    <path d="M0,-${r} C10,-${r + 14} 24,-${r + 10} 20,-${r - 2} Z" fill="#16A34A"/>
  </g>`;
}

function drawMiniFlower(cx: number, cy: number, petalColor = '#EC4899', r = 13): string {
  let petals = '';
  for (let i = 0; i < 6; i += 1) {
    const a = (i * 60 * Math.PI) / 180;
    const px = (Math.cos(a) * r * 1.15).toFixed(1);
    const py = (Math.sin(a) * r * 1.15).toFixed(1);
    petals += `<circle cx="${px}" cy="${py}" r="${r}" fill="${petalColor}"/>`;
  }
  return `<g transform="translate(${cx},${cy})">${petals}<circle cx="0" cy="0" r="${r * 0.85}" fill="#FACC15" stroke="#CA8A04" stroke-width="2"/></g>`;
}

function drawMiniBird(cx: number, cy: number, s = 1, wingColor = '#38BDF8'): string {
  return `<g transform="translate(${cx},${cy}) scale(${s})">
    <path d="M-32,8 Q-4,-32 30,-2 Q16,26 -16,24 Q-30,22 -32,8 Z" fill="#0284C7"/>
    <circle cx="24" cy="-8" r="14" fill="#0EA5E9"/>
    <path d="M36,-10 L50,-6 L37,-1 Z" fill="#F59E0B"/>
    <circle cx="27" cy="-11" r="3" fill="#0F172A"/>
    <path d="M-12,-4 Q6,-28 20,-4 Q4,10 -12,-4 Z" fill="${wingColor}"/>
    <path d="M-28,10 L-44,18 L-34,2 Z" fill="#0369A1"/>
    <path d="M2,24 L0,38 M12,24 L14,38" stroke="#D97706" stroke-width="3" stroke-linecap="round"/>
  </g>`;
}

function drawAnalogClockBase(cx: number, cy: number, r: number, hourAngleDeg: number, minAngleDeg: number, withNums = false): string {
  const hrRad = ((hourAngleDeg - 90) * Math.PI) / 180;
  const mnRad = ((minAngleDeg - 90) * Math.PI) / 180;
  let ticks = '';
  for (let i = 0; i < 12; i += 1) {
    const a = ((i * 30 - 90) * Math.PI) / 180;
    const x1 = (cx + (r - 12) * Math.cos(a)).toFixed(1);
    const y1 = (cy + (r - 12) * Math.sin(a)).toFixed(1);
    const x2 = (cx + (r - 6) * Math.cos(a)).toFixed(1);
    const y2 = (cy + (r - 6) * Math.sin(a)).toFixed(1);
    ticks += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#475569" stroke-width="3" stroke-linecap="round"/>`;
    if (withNums) {
      const tx = (cx + (r - 22) * Math.cos(a)).toFixed(1);
      const ty = (cy + (r - 22) * Math.sin(a) + 4).toFixed(1);
      ticks += `<text x="${tx}" y="${ty}" text-anchor="middle" font-family="Arial,sans-serif" font-size="11" font-weight="800" fill="#1E293B">${i === 0 ? 12 : i}</text>`;
    }
  }
  return `<circle cx="${cx}" cy="${cy}" r="${r + 8}" fill="#1E293B"/>
    <circle cx="${cx}" cy="${cy}" r="${r}" fill="#FFFFFF" stroke="#E2E8F0" stroke-width="2"/>
    ${ticks}
    <line x1="${cx}" y1="${cy}" x2="${(cx + r * 0.52 * Math.cos(hrRad)).toFixed(1)}" y2="${(cy + r * 0.52 * Math.sin(hrRad)).toFixed(1)}" stroke="#0F172A" stroke-width="5.5" stroke-linecap="round"/>
    <line x1="${cx}" y1="${cy}" x2="${(cx + r * 0.74 * Math.cos(mnRad)).toFixed(1)}" y2="${(cy + r * 0.74 * Math.sin(mnRad)).toFixed(1)}" stroke="#2563EB" stroke-width="4" stroke-linecap="round"/>
    <circle cx="${cx}" cy="${cy}" r="5.5" fill="#EF4444"/>`;
}

function drawShoe(cx: number, cy: number, mainColor: string, soleColor: string, tread = false): string {
  let cleats = '';
  if (tread) {
    for (let x = -52; x <= 48; x += 16) {
      cleats += `<rect x="${x}" y="22" width="10" height="7" rx="2" fill="#F59E0B"/>`;
    }
  }
  return `<g transform="translate(${cx},${cy})">
    <ellipse cx="0" cy="36" rx="70" ry="9" fill="#0F172A" opacity="0.12"/>
    <path d="M-62,14 L-62,-22 C-52,-14 -34,-12 -22,-26 L12,0 C38,2 62,6 64,18 Z" fill="${mainColor}" stroke="#1E293B" stroke-width="3" stroke-linejoin="round"/>
    <path d="M-64,14 L66,14 L64,25 L-62,25 Z" fill="${soleColor}" stroke="#1E293B" stroke-width="2.5"/>
    <line x1="-10" y1="-10" x2="6" y2="-2" stroke="#FFFFFF" stroke-width="4" stroke-linecap="round"/>
    <line x1="-2" y1="-3" x2="14" y2="5" stroke="#FFFFFF" stroke-width="4" stroke-linecap="round"/>
    <line x1="6" y1="4" x2="22" y2="10" stroke="#FFFFFF" stroke-width="4" stroke-linecap="round"/>
    ${cleats}
  </g>`;
}

function renderArtwork(key: CardArtKey): string {
  switch (key) {
    // ================= 001..020 FOOD & FRUIT =================
    case 'apple':
      return drawMiniApple(200, 145, 1.55);
    case 'banana':
      return `<g transform="translate(200,145)">
        <ellipse cx="0" cy="54" rx="76" ry="11" fill="#0F172A" opacity="0.1"/>
        <path d="M-74,-22 C-42,48 34,58 84,12 C62,-6 4,-16 -54,-38 Z" fill="#FACC15" stroke="#CA8A04" stroke-width="3.5" stroke-linejoin="round"/>
        <path d="M-56,-40 C-8,-22 50,-10 74,4 C30,18 -26,8 -62,-28 Z" fill="#FDE047" stroke="#EAB308" stroke-width="2.5"/>
        <path d="M-74,-24 L-82,-36 L-64,-38 Z" fill="#78350F"/>
        <circle cx="84" cy="12" r="5" fill="#78350F"/>
      </g>`;
    case 'broccoli':
      return `<g transform="translate(200,150)">
        <ellipse cx="0" cy="64" rx="46" ry="10" fill="#0F172A" opacity="0.1"/>
        <path d="M-20,6 L-14,58 Q0,66 14,58 L20,6 Z" fill="#86EFAC" stroke="#15803D" stroke-width="3"/>
        <path d="M-38,-2 L-16,24 M36,-2 L14,24" stroke="#16A34A" stroke-width="11" stroke-linecap="round"/>
        <circle cx="-42" cy="-24" r="30" fill="#16A34A" stroke="#14532D" stroke-width="2.5"/>
        <circle cx="42" cy="-24" r="30" fill="#15803D" stroke="#14532D" stroke-width="2.5"/>
        <circle cx="0" cy="-42" r="34" fill="#22C55E" stroke="#14532D" stroke-width="2.5"/>
        <circle cx="-18" cy="-16" r="24" fill="#4ADE80"/>
        <circle cx="20" cy="-18" r="22" fill="#22C55E"/>
      </g>`;
    case 'milk':
      return `<g transform="translate(200,145)">
        <ellipse cx="0" cy="68" rx="74" ry="11" fill="#0F172A" opacity="0.1"/>
        <polygon points="-58,-44 -20,-44 -4,-62 -42,-62" fill="#BFDBFE" stroke="#1D4ED8" stroke-width="2.5"/>
        <rect x="-58" y="-44" width="52" height="106" rx="4" fill="#FFFFFF" stroke="#1D4ED8" stroke-width="3"/>
        <rect x="-58" y="-8" width="52" height="44" fill="#3B82F6"/>
        <text x="-32" y="20" text-anchor="middle" font-family="Arial,sans-serif" font-size="14" font-weight="900" fill="#FFFFFF">MILK</text>
        <path d="M24,-10 L68,-10 L62,62 L30,62 Z" fill="#F8FAFC" stroke="#0284C7" stroke-width="3"/>
        <path d="M27,6 L65,6 L61,60 L31,60 Z" fill="#FFFFFF"/>
        <line x1="28" y1="6" x2="64" y2="6" stroke="#BAE6FD" stroke-width="3"/>
      </g>`;
    case 'watermelon':
      return `<g transform="translate(200,152)">
        <ellipse cx="0" cy="54" rx="76" ry="11" fill="#0F172A" opacity="0.1"/>
        <path d="M-86,-22 A88,88 0 0,0 86,-22 Z" fill="#15803D" stroke="#14532D" stroke-width="3"/>
        <path d="M-76,-22 A78,78 0 0,0 76,-22 Z" fill="#BBF7D0"/>
        <path d="M-68,-22 A70,70 0 0,0 68,-22 Z" fill="#EF4444"/>
        <ellipse cx="-26" cy="-4" rx="3.5" ry="6" transform="rotate(-20 -26 -4)" fill="#1E293B"/>
        <ellipse cx="0" cy="14" rx="3.5" ry="6" fill="#1E293B"/>
        <ellipse cx="26" cy="-4" rx="3.5" ry="6" transform="rotate(20 26 -4)" fill="#1E293B"/>
        <ellipse cx="-14" cy="24" rx="3.5" ry="6" fill="#1E293B"/>
        <ellipse cx="14" cy="24" rx="3.5" ry="6" fill="#1E293B"/>
      </g>`;
    case 'carrot':
      return `<g transform="translate(200,145) rotate(32)">
        <path d="M0,-62 C-24,-96 -6,-106 0,-74 C8,-108 26,-98 4,-74 C28,-90 36,-70 10,-62 Z" fill="#22C55E" stroke="#15803D" stroke-width="2.5"/>
        <path d="M-26,-56 C-30,-66 30,-66 26,-56 L6,66 C4,74 -4,74 -6,66 Z" fill="#F97316" stroke="#C2410C" stroke-width="3"/>
        <line x1="-16" y1="-26" x2="4" y2="-26" stroke="#C2410C" stroke-width="3" stroke-linecap="round"/>
        <line x1="-4" y1="6" x2="12" y2="6" stroke="#C2410C" stroke-width="3" stroke-linecap="round"/>
        <line x1="-10" y1="32" x2="2" y2="32" stroke="#C2410C" stroke-width="3" stroke-linecap="round"/>
      </g>`;
    case 'fish_food':
      return `<g transform="translate(200,145)">
        <ellipse cx="0" cy="54" rx="88" ry="12" fill="#E2E8F0" stroke="#94A3B8" stroke-width="2.5"/>
        <path d="M-64,0 C-30,-42 44,-38 72,0 C44,38 -30,42 -64,0 Z" fill="#38BDF8" stroke="#0369A1" stroke-width="3"/>
        <polygon points="-60,0 -92,-24 -92,24" fill="#0EA5E9" stroke="#0369A1" stroke-width="2.5"/>
        <path d="M22,-18 Q12,0 22,18" stroke="#0369A1" stroke-width="3" fill="none"/>
        <circle cx="46" cy="-7" r="5" fill="#0F172A"/>
        <circle cx="48" cy="-9" r="1.8" fill="#FFFFFF"/>
        <path d="M-12,-28 L4,-46 L16,-24 Z" fill="#7DD3FC" stroke="#0369A1" stroke-width="2"/>
      </g>`;
    case 'mango':
      return `<g transform="translate(200,145)">
        <ellipse cx="0" cy="58" rx="56" ry="10" fill="#0F172A" opacity="0.1"/>
        <path d="M-12,-54 C28,-62 60,-22 50,24 C42,58 -6,66 -32,48 C-60,30 -54,-42 -12,-54 Z" fill="#FBBF24" stroke="#D97706" stroke-width="3.5"/>
        <ellipse cx="-14" cy="-14" rx="14" ry="26" transform="rotate(-15 -14 -14)" fill="#FEF08A" opacity="0.55"/>
        <path d="M-10,-54 C6,-72 34,-66 26,-48 Z" fill="#22C55E" stroke="#15803D" stroke-width="2"/>
      </g>`;
    case 'orange':
      return drawMiniOrange(200, 148, 56);
    case 'fried_egg':
      return `<g transform="translate(200,142)">
        <ellipse cx="0" cy="48" rx="78" ry="14" fill="#0F172A" opacity="0.12"/>
        <circle cx="0" cy="0" r="74" fill="#334155" stroke="#0F172A" stroke-width="4"/>
        <circle cx="0" cy="0" r="62" fill="#1E293B"/>
        <rect x="66" y="-8" width="58" height="16" rx="8" fill="#475569" stroke="#0F172A" stroke-width="3"/>
        <path d="M-36,-14 C-44,-38 -4,-48 20,-34 C46,-30 50,10 34,30 C16,46 -28,44 -40,18 C-46,4 -34,-4 -36,-14 Z" fill="#FFFFFF"/>
        <circle cx="4" cy="2" r="22" fill="#FACC15" stroke="#EAB308" stroke-width="2.5"/>
        <circle cx="-4" cy="-6" r="6" fill="#FEF9C3"/>
      </g>`;
    case 'rice_bowl':
      return `<g transform="translate(200,155)">
        <ellipse cx="0" cy="52" rx="68" ry="10" fill="#0F172A" opacity="0.1"/>
        <path d="M-68,-10 C-48,-52 48,-52 68,-10 Z" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="3"/>
        <circle cx="-22" cy="-24" r="3" fill="#CBD5E1"/><circle cx="6" cy="-32" r="3" fill="#CBD5E1"/><circle cx="28" cy="-20" r="3" fill="#CBD5E1"/>
        <path d="M-74,-10 L74,-10 C70,40 34,50 0,50 C-34,50 -70,40 -74,-10 Z" fill="#0284C7" stroke="#075985" stroke-width="3"/>
        <line x1="-22" y1="-20" x2="48" y2="-66" stroke="#B45309" stroke-width="5" stroke-linecap="round"/>
        <line x1="-8" y1="-16" x2="62" y2="-56" stroke="#D97706" stroke-width="5" stroke-linecap="round"/>
      </g>`;
    case 'papaya':
      return `<g transform="translate(200,145)">
        <ellipse cx="0" cy="54" rx="74" ry="10" fill="#0F172A" opacity="0.1"/>
        <path d="M-64,18 C-62,-28 -16,-54 44,-44 C76,-38 78,24 44,44 C6,64 -66,52 -64,18 Z" fill="#16A34A" stroke="#14532D" stroke-width="3"/>
        <path d="M-50,14 C-48,-22 -12,-42 36,-34 C60,-28 60,16 32,32 C0,48 -52,40 -50,14 Z" fill="#FB923C"/>
        <circle cx="-6" cy="-4" r="4.5" fill="#1E293B"/><circle cx="8" cy="-10" r="4.5" fill="#1E293B"/>
        <circle cx="20" cy="2" r="4.5" fill="#1E293B"/><circle cx="4" cy="10" r="4.5" fill="#1E293B"/>
        <circle cx="-16" cy="8" r="4.5" fill="#1E293B"/><circle cx="18" cy="-14" r="4.5" fill="#1E293B"/>
      </g>`;
    case 'pineapple':
      return `<g transform="translate(200,162)">
        <ellipse cx="0" cy="54" rx="52" ry="9" fill="#0F172A" opacity="0.1"/>
        <path d="M0,-44 L-26,-88 L-6,-62 L0,-96 L10,-62 L28,-86 Z" fill="#16A34A" stroke="#14532D" stroke-width="2.5"/>
        <ellipse cx="0" cy="4" r="42" rx="42" ry="50" fill="#F59E0B" stroke="#B45309" stroke-width="3"/>
        <path d="M-26,-24 L26,30 M-36,2 L16,48 M-14,-40 L36,8 M26,-24 L-26,30 M36,2 L-16,48 M14,-40 L-36,8" stroke="#B45309" stroke-width="2.2" opacity="0.7"/>
      </g>`;
    case 'coconut':
      return `<g transform="translate(200,152)">
        <ellipse cx="0" cy="56" rx="56" ry="10" fill="#0F172A" opacity="0.1"/>
        <circle cx="0" cy="4" r="52" fill="#22C55E" stroke="#15803D" stroke-width="3.5"/>
        <ellipse cx="0" cy="-24" rx="34" ry="14" fill="#ECFDF5" stroke="#15803D" stroke-width="2.5"/>
        <path d="M10,-22 L28,-74 L48,-68" stroke="#EF4444" stroke-width="6.5" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
      </g>`;
    case 'corn':
      return `<g transform="translate(200,150)">
        <ellipse cx="0" cy="64" rx="52" ry="10" fill="#0F172A" opacity="0.1"/>
        <ellipse cx="0" cy="-10" rx="28" ry="60" fill="#FACC15" stroke="#CA8A04" stroke-width="3"/>
        <path d="M-24,-34 L24,-34 M-28,-12 L28,-12 M-26,10 L26,10 M-10,-66 L-10,40 M10,-66 L10,40" stroke="#EAB308" stroke-width="2"/>
        <path d="M-28,6 C-56,42 -36,68 0,58 C-14,32 -16,12 -28,6 Z" fill="#22C55E" stroke="#15803D" stroke-width="2.5"/>
        <path d="M28,6 C56,42 36,68 0,58 C14,32 16,12 28,6 Z" fill="#16A34A" stroke="#15803D" stroke-width="2.5"/>
      </g>`;
    case 'strawberry':
      return `<g transform="translate(200,152)">
        <ellipse cx="0" cy="58" rx="46" ry="9" fill="#0F172A" opacity="0.1"/>
        <path d="M0,-44 C-44,-44 -54,0 -28,36 C-10,60 10,60 28,36 C54,0 44,-44 0,-44 Z" fill="#EF4444" stroke="#991B1B" stroke-width="3"/>
        <path d="M-28,-44 L-10,-58 L0,-44 L10,-58 L28,-44 L8,-32 L-8,-32 Z" fill="#22C55E" stroke="#15803D" stroke-width="2"/>
        <circle cx="-18" cy="-14" r="3" fill="#FEF08A"/><circle cx="0" cy="-20" r="3" fill="#FEF08A"/><circle cx="18" cy="-14" r="3" fill="#FEF08A"/>
        <circle cx="-22" cy="8" r="3" fill="#FEF08A"/><circle cx="0" cy="4" r="3" fill="#FEF08A"/><circle cx="22" cy="8" r="3" fill="#FEF08A"/>
        <circle cx="-10" cy="28" r="3" fill="#FEF08A"/><circle cx="10" cy="28" r="3" fill="#FEF08A"/>
      </g>`;
    case 'grapes':
      return `<g transform="translate(200,145)">
        <path d="M0,-58 L0,-34" stroke="#78350F" stroke-width="6" stroke-linecap="round"/>
        <path d="M0,-48 C22,-66 44,-52 34,-34 Z" fill="#22C55E"/>
        <circle cx="-26" cy="-20" r="18" fill="#9333EA" stroke="#6B21A8" stroke-width="2.5"/>
        <circle cx="0" cy="-22" r="18" fill="#A855F7" stroke="#6B21A8" stroke-width="2.5"/>
        <circle cx="26" cy="-20" r="18" fill="#7E22CE" stroke="#6B21A8" stroke-width="2.5"/>
        <circle cx="-16" cy="8" r="18" fill="#A855F7" stroke="#6B21A8" stroke-width="2.5"/>
        <circle cx="14" cy="8" r="18" fill="#9333EA" stroke="#6B21A8" stroke-width="2.5"/>
        <circle cx="0" cy="36" r="18" fill="#7E22CE" stroke="#6B21A8" stroke-width="2.5"/>
      </g>`;
    case 'pumpkin':
      return `<g transform="translate(200,152)">
        <ellipse cx="0" cy="54" rx="74" ry="11" fill="#0F172A" opacity="0.1"/>
        <path d="M-6,-44 L-2,-66 L10,-64 L6,-42 Z" fill="#15803D"/>
        <ellipse cx="-38" cy="4" rx="34" ry="46" fill="#EA580C" stroke="#9A3412" stroke-width="2.5"/>
        <ellipse cx="38" cy="4" rx="34" ry="46" fill="#EA580C" stroke="#9A3412" stroke-width="2.5"/>
        <ellipse cx="0" cy="4" rx="32" ry="50" fill="#F97316" stroke="#9A3412" stroke-width="2.5"/>
        <ellipse cx="-10" cy="-10" rx="8" ry="20" fill="#FED7AA" opacity="0.45"/>
      </g>`;
    case 'soup_bowl':
      return `<g transform="translate(200,155)">
        <path d="M-28,-50 Q-34,-68 -24,-80 M0,-54 Q-6,-72 4,-84 M28,-50 Q22,-68 32,-80" stroke="#94A3B8" stroke-width="4" fill="none" stroke-linecap="round"/>
        <ellipse cx="0" cy="-16" rx="72" ry="18" fill="#FBBF24" stroke="#B45309" stroke-width="3"/>
        <circle cx="-22" cy="-18" r="6" fill="#EF4444"/><circle cx="14" cy="-14" r="6" fill="#22C55E"/><circle cx="30" cy="-22" r="5" fill="#FFFFFF"/>
        <path d="M-72,-16 C-68,36 -36,48 0,48 C36,48 68,36 72,-16 Z" fill="#FFFFFF" stroke="#64748B" stroke-width="3"/>
      </g>`;
    case 'water_glass':
      return `<g transform="translate(200,145)">
        <ellipse cx="0" cy="66" rx="52" ry="9" fill="#0F172A" opacity="0.1"/>
        <path d="M-44,-54 L44,-54 L34,60 L-34,60 Z" fill="#F0F9FF" stroke="#0284C7" stroke-width="3.5" stroke-linejoin="round"/>
        <path d="M-38,-14 C-14,-22 14,-6 38,-14 L32,56 L-32,56 Z" fill="#38BDF8" opacity="0.85"/>
        <line x1="16" y1="-70" x2="0" y2="36" stroke="#EC4899" stroke-width="6" stroke-linecap="round"/>
        <ellipse cx="0" cy="-54" rx="44" ry="8" fill="#E0F2FE" stroke="#0284C7" stroke-width="3"/>
      </g>`;

    // ================= 021..040 OBJECTS =================
    case 'key':
      return `<g transform="translate(200,145) rotate(-18)">
        <circle cx="-54" cy="0" r="34" fill="#FACC15" stroke="#B45309" stroke-width="4"/>
        <circle cx="-64" cy="0" r="11" fill="#EFF6FF" stroke="#B45309" stroke-width="3"/>
        <rect x="-22" y="-10" width="112" height="20" rx="6" fill="#FACC15" stroke="#B45309" stroke-width="3.5"/>
        <rect x="44" y="10" width="14" height="22" rx="3" fill="#F59E0B" stroke="#B45309" stroke-width="3"/>
        <rect x="68" y="10" width="16" height="28" rx="3" fill="#F59E0B" stroke="#B45309" stroke-width="3"/>
      </g>`;
    case 'toothbrush':
      return `<g transform="translate(200,145) rotate(-22)">
        <rect x="-86" y="-8" width="174" height="20" rx="10" fill="#EC4899" stroke="#9D174D" stroke-width="3"/>
        <rect x="-76" y="-2" width="60" height="8" rx="4" fill="#FBCFE8"/>
        <rect x="24" y="-34" width="54" height="26" rx="4" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="2.5"/>
        <line x1="36" y1="-34" x2="36" y2="-8" stroke="#94A3B8" stroke-width="2"/>
        <line x1="50" y1="-34" x2="50" y2="-8" stroke="#94A3B8" stroke-width="2"/>
        <line x1="64" y1="-34" x2="64" y2="-8" stroke="#94A3B8" stroke-width="2"/>
        <path d="M22,-36 C36,-54 68,-50 76,-34 C56,-28 38,-30 22,-36 Z" fill="#2DD4BF"/>
      </g>`;
    case 'eyeglasses':
      return `<g transform="translate(200,148)">
        <path d="M-86,-16 L-106,-44 M86,-16 L106,-44" stroke="#1E293B" stroke-width="6" stroke-linecap="round"/>
        <circle cx="-48" cy="4" r="38" fill="#E0F2FE" stroke="#1E293B" stroke-width="7"/>
        <circle cx="48" cy="4" r="38" fill="#E0F2FE" stroke="#1E293B" stroke-width="7"/>
        <path d="M-10,2 Q0,-12 10,2" stroke="#1E293B" stroke-width="7" fill="none"/>
        <line x1="-66" y1="-12" x2="-44" y2="-20" stroke="#FFFFFF" stroke-width="4" stroke-linecap="round"/>
        <line x1="30" y1="-12" x2="52" y2="-20" stroke="#FFFFFF" stroke-width="4" stroke-linecap="round"/>
      </g>`;
    case 'umbrella':
      return `<g transform="translate(200,152)">
        <line x1="0" y1="-64" x2="0" y2="54" stroke="#475569" stroke-width="6" stroke-linecap="round"/>
        <path d="M0,54 C0,72 22,72 22,56" stroke="#D97706" stroke-width="6" fill="none" stroke-linecap="round"/>
        <path d="M-86,-6 C-76,-64 76,-64 86,-6 Q60,-18 42,-6 Q20,-18 0,-6 Q-20,-18 -42,-6 Q-60,-18 -86,-6 Z" fill="#3B82F6" stroke="#1E40AF" stroke-width="3.5"/>
        <path d="M0,-58 C-26,-44 -42,-22 -42,-6 M0,-58 C26,-44 42,-22 42,-6" stroke="#1D4ED8" stroke-width="2.5" fill="none"/>
      </g>`;
    case 'alarm_clock':
      return `<g transform="translate(200,150)">
        <circle cx="-44" cy="-48" r="18" fill="#EF4444" stroke="#991B1B" stroke-width="3"/>
        <circle cx="44" cy="-48" r="18" fill="#EF4444" stroke="#991B1B" stroke-width="3"/>
        <line x1="-38" y1="42" x2="-50" y2="62" stroke="#475569" stroke-width="7" stroke-linecap="round"/>
        <line x1="38" y1="42" x2="50" y2="62" stroke="#475569" stroke-width="7" stroke-linecap="round"/>
        ${drawAnalogClockBase(0, 0, 52, 60, 0)}
      </g>`;
    case 'chair':
      return `<g transform="translate(200,154)">
        <ellipse cx="0" cy="64" rx="58" ry="9" fill="#0F172A" opacity="0.1"/>
        <rect x="-42" y="-68" width="12" height="72" rx="4" fill="#B45309" stroke="#78350F" stroke-width="2.5"/>
        <rect x="30" y="-68" width="12" height="72" rx="4" fill="#B45309" stroke="#78350F" stroke-width="2.5"/>
        <rect x="-42" y="-62" width="84" height="32" rx="6" fill="#D97706" stroke="#78350F" stroke-width="2.5"/>
        <rect x="-50" y="4" width="100" height="16" rx="5" fill="#F59E0B" stroke="#78350F" stroke-width="3"/>
        <line x1="-38" y1="20" x2="-44" y2="64" stroke="#78350F" stroke-width="9" stroke-linecap="round"/>
        <line x1="38" y1="20" x2="44" y2="64" stroke="#78350F" stroke-width="9" stroke-linecap="round"/>
      </g>`;
    case 'stethoscope':
      return `<g transform="translate(200,145)">
        <path d="M-42,-56 C-42,-6 42,-6 42,-56" stroke="#0F766E" stroke-width="9" fill="none" stroke-linecap="round"/>
        <circle cx="-42" cy="-58" r="8" fill="#334155"/><circle cx="42" cy="-58" r="8" fill="#334155"/>
        <path d="M0,-12 L0,24 C0,54 58,54 58,20" stroke="#0D9488" stroke-width="8" fill="none" stroke-linecap="round"/>
        <circle cx="58" cy="14" r="22" fill="#E2E8F0" stroke="#475569" stroke-width="4"/>
        <circle cx="58" cy="14" r="11" fill="#0284C7"/>
      </g>`;
    case 'spoon_fork':
      return `<g transform="translate(200,145)">
        <g transform="translate(-38,0)">
          <ellipse cx="0" cy="-34" rx="18" ry="26" fill="#CBD5E1" stroke="#475569" stroke-width="3"/>
          <rect x="-6" y="-8" width="12" height="70" rx="6" fill="#94A3B8" stroke="#475569" stroke-width="3"/>
        </g>
        <g transform="translate(38,0)">
          <path d="M-16,-58 L-16,-14 L16,-14 L16,-58 M-5,-58 L-5,-22 M5,-58 L5,-22" stroke="#475569" stroke-width="5" fill="none" stroke-linecap="round"/>
          <rect x="-6" y="-14" width="12" height="76" rx="6" fill="#94A3B8" stroke="#475569" stroke-width="3"/>
        </g>
      </g>`;
    case 'hair_comb':
      return `<g transform="translate(200,145) rotate(-15)">
        <rect x="-86" y="-26" width="172" height="22" rx="10" fill="#8B5CF6" stroke="#5B21B6" stroke-width="3"/>
        ${[-74,-58,-42,-26,-10,6,22,38,54,70].map((x) => `<rect x="${x}" y="-4" width="7" height="36" rx="3" fill="#A78BFA" stroke="#5B21B6" stroke-width="2"/>`).join('')}
      </g>`;
    case 'scissors':
      return `<g transform="translate(200,145)">
        <polygon points="0,-6 74,-34 78,-18 -4,4" fill="#CBD5E1" stroke="#475569" stroke-width="3"/>
        <polygon points="0,6 74,34 78,18 -4,-4" fill="#E2E8F0" stroke="#475569" stroke-width="3"/>
        <circle cx="-46" cy="-28" r="22" fill="none" stroke="#EF4444" stroke-width="10"/>
        <circle cx="-46" cy="28" r="22" fill="none" stroke="#EF4444" stroke-width="10"/>
        <circle cx="8" cy="0" r="5" fill="#1E293B"/>
      </g>`;
    case 'pillow':
      return `<g transform="translate(200,148)">
        <ellipse cx="0" cy="48" rx="84" ry="12" fill="#0F172A" opacity="0.1"/>
        <path d="M-82,-36 C-34,-26 34,-26 82,-36 C74,-12 74,12 82,36 C34,26 -34,26 -82,36 C-74,12 -74,-12 -82,-36 Z" fill="#FFFFFF" stroke="#60A5FA" stroke-width="4" stroke-linejoin="round"/>
        <path d="M-54,-8 Q0,4 54,-8" stroke="#DBEAFE" stroke-width="4" fill="none" stroke-linecap="round"/>
      </g>`;
    case 'walking_shoes':
      return `<g transform="translate(200,145)">
        ${drawShoe(-42, 0, '#3B82F6', '#FFFFFF', false).replace('scale(1)', 'scale(0.8)')}
        <g transform="scale(0.78) translate(-52,-12)">${drawShoe(0, 0, '#2563EB', '#FFFFFF', false)}</g>
        <g transform="scale(0.78) translate(62,14)">${drawShoe(0, 0, '#3B82F6', '#FFFFFF', false)}</g>
      </g>`;
    case 'telephone':
      return `<g transform="translate(200,152)">
        <path d="M-64,4 L64,4 L76,46 L-76,46 Z" fill="#0284C7" stroke="#075985" stroke-width="3.5" stroke-linejoin="round"/>
        <circle cx="0" cy="24" r="17" fill="#FFFFFF" stroke="#075985" stroke-width="2.5"/>
        <path d="M-74,-22 C-56,-54 56,-54 74,-22 L76,-4 L42,-4 L42,-16 L-42,-16 L-42,-4 L-76,-4 Z" fill="#0369A1" stroke="#0F172A" stroke-width="3" stroke-linejoin="round"/>
      </g>`;
    case 'open_book':
      return `<g transform="translate(200,152)">
        <path d="M-84,-38 C-48,-50 -14,-44 0,-30 C14,-44 48,-50 84,-38 L84,42 C48,30 14,34 0,46 C-14,34 -48,30 -84,42 Z" fill="#FFFFFF" stroke="#2563EB" stroke-width="4" stroke-linejoin="round"/>
        <line x1="0" y1="-30" x2="0" y2="46" stroke="#2563EB" stroke-width="3.5"/>
        <line x1="-66" y1="-16" x2="-18" y2="-16" stroke="#94A3B8" stroke-width="4" stroke-linecap="round"/>
        <line x1="-66" y1="0" x2="-18" y2="0" stroke="#94A3B8" stroke-width="4" stroke-linecap="round"/>
        <line x1="-66" y1="16" x2="-24" y2="16" stroke="#94A3B8" stroke-width="4" stroke-linecap="round"/>
        <line x1="18" y1="-16" x2="66" y2="-16" stroke="#94A3B8" stroke-width="4" stroke-linecap="round"/>
        <line x1="18" y1="0" x2="66" y2="0" stroke="#94A3B8" stroke-width="4" stroke-linecap="round"/>
      </g>`;
    case 'teacup':
      return `<g transform="translate(200,158)">
        <ellipse cx="0" cy="40" rx="76" ry="12" fill="#E2E8F0" stroke="#64748B" stroke-width="3"/>
        <path d="M38,-20 C68,-20 68,20 34,20" fill="none" stroke="#0284C7" stroke-width="9" stroke-linecap="round"/>
        <path d="M-44,-32 L44,-32 C42,26 24,38 0,38 C-24,38 -42,26 -44,-32 Z" fill="#FFFFFF" stroke="#0284C7" stroke-width="3.5"/>
        <ellipse cx="0" cy="-32" rx="44" ry="10" fill="#B45309" stroke="#0284C7" stroke-width="3"/>
        <path d="M-12,-50 Q-18,-66 -8,-78 M12,-50 Q6,-66 16,-78" stroke="#94A3B8" stroke-width="3.5" fill="none" stroke-linecap="round"/>
      </g>`;
    case 'wristwatch':
      return `<g transform="translate(200,145)">
        <rect x="-24" y="-82" width="48" height="164" rx="10" fill="#334155" stroke="#0F172A" stroke-width="3"/>
        ${drawAnalogClockBase(0, 0, 42, 300, 60)}
      </g>`;
    case 'flashlight':
      return `<g transform="translate(175,148)">
        <polygon points="48,-24 148,-58 148,58 48,24" fill="#FEF08A" opacity="0.72"/>
        <rect x="-86" y="-16" width="112" height="32" rx="8" fill="#EF4444" stroke="#991B1B" stroke-width="3"/>
        <polygon points="24,-18 52,-28 52,28 24,18" fill="#94A3B8" stroke="#334155" stroke-width="3"/>
        <rect x="-46" y="-22" width="22" height="10" rx="3" fill="#0F172A"/>
      </g>`;
    case 'table_fan':
      return `<g transform="translate(200,152)">
        <ellipse cx="0" cy="60" rx="46" ry="10" fill="#1D4ED8"/>
        <rect x="-7" y="18" width="14" height="44" fill="#475569"/>
        <circle cx="0" cy="-14" r="52" fill="#E0F2FE" stroke="#0284C7" stroke-width="4"/>
        <ellipse cx="0" cy="-38" rx="10" ry="22" fill="#38BDF8"/>
        <ellipse cx="-22" cy="0" rx="22" ry="10" transform="rotate(-24 -22 0)" fill="#38BDF8"/>
        <ellipse cx="22" cy="0" rx="22" ry="10" transform="rotate(24 22 0)" fill="#38BDF8"/>
        <circle cx="0" cy="-14" r="11" fill="#0369A1"/>
      </g>`;
    case 'sun_hat':
      return `<g transform="translate(200,152)">
        <ellipse cx="0" cy="20" rx="96" ry="26" fill="#FDE047" stroke="#CA8A04" stroke-width="3.5"/>
        <path d="M-48,14 C-46,-46 46,-46 48,14 Z" fill="#FACC15" stroke="#CA8A04" stroke-width="3"/>
        <path d="M-47,6 Q0,18 47,6 L48,16 Q0,28 -48,16 Z" fill="#EF4444"/>
      </g>`;
    case 'thermometer':
      return `<g transform="translate(200,145) rotate(22)">
        <rect x="-16" y="-76" width="32" height="118" rx="16" fill="#FFFFFF" stroke="#475569" stroke-width="3.5"/>
        <circle cx="0" cy="48" r="24" fill="#EF4444" stroke="#475569" stroke-width="3.5"/>
        <rect x="-6" y="-44" width="12" height="84" rx="6" fill="#EF4444"/>
        <line x1="16" y1="-48" x2="26" y2="-48" stroke="#475569" stroke-width="3"/>
        <line x1="16" y1="-26" x2="26" y2="-26" stroke="#475569" stroke-width="3"/>
        <line x1="16" y1="-4" x2="26" y2="-4" stroke="#475569" stroke-width="3"/>
      </g>`;

    // ================= 041..055 ANIMALS =================
    case 'elephant':
      return `<g transform="translate(200,150)">
        <ellipse cx="-10" cy="14" rx="52" ry="42" fill="#94A3B8" stroke="#475569" stroke-width="3"/>
        <rect x="-46" y="34" width="18" height="34" rx="6" fill="#64748B" stroke="#475569" stroke-width="2.5"/>
        <rect x="6" y="34" width="18" height="34" rx="6" fill="#94A3B8" stroke="#475569" stroke-width="2.5"/>
        <circle cx="38" cy="-14" r="32" fill="#94A3B8" stroke="#475569" stroke-width="3"/>
        <ellipse cx="14" cy="-16" rx="22" ry="28" fill="#64748B" stroke="#475569" stroke-width="2.5"/>
        <path d="M64,-10 C84,-4 86,28 70,38 C60,42 56,28 66,24" stroke="#94A3B8" stroke-width="14" fill="none" stroke-linecap="round"/>
        <path d="M56,10 Q74,20 78,6" stroke="#FEF9C3" stroke-width="6" fill="none" stroke-linecap="round"/>
        <circle cx="48" cy="-22" r="4.5" fill="#0F172A"/>
      </g>`;
    case 'dog':
      return `<g transform="translate(200,150)">
        <ellipse cx="-16" cy="16" rx="48" ry="28" fill="#D97706" stroke="#78350F" stroke-width="3"/>
        <path d="M-62,6 C-76,-12 -74,-26 -60,-18" stroke="#B45309" stroke-width="8" fill="none" stroke-linecap="round"/>
        <rect x="-48" y="30" width="14" height="34" rx="6" fill="#B45309" stroke="#78350F" stroke-width="2.5"/>
        <rect x="-12" y="30" width="14" height="34" rx="6" fill="#D97706" stroke="#78350F" stroke-width="2.5"/>
        <rect x="18" y="30" width="14" height="34" rx="6" fill="#D97706" stroke="#78350F" stroke-width="2.5"/>
        <circle cx="38" cy="-16" r="28" fill="#F59E0B" stroke="#78350F" stroke-width="3"/>
        <ellipse cx="22" cy="-10" rx="10" ry="22" transform="rotate(14 22 -10)" fill="#78350F"/>
        <circle cx="62" cy="-8" r="7" fill="#0F172A"/>
        <circle cx="46" cy="-22" r="4.5" fill="#0F172A"/>
      </g>`;
    case 'bird':
      return drawMiniBird(200, 145, 1.55, '#BAE6FD');
    case 'turtle':
      return `<g transform="translate(200,156)">
        <ellipse cx="-38" cy="24" rx="14" ry="12" fill="#4ADE80" stroke="#15803D" stroke-width="2.5"/>
        <ellipse cx="28" cy="24" rx="14" ry="12" fill="#4ADE80" stroke="#15803D" stroke-width="2.5"/>
        <circle cx="68" cy="2" r="18" fill="#4ADE80" stroke="#15803D" stroke-width="3"/>
        <circle cx="74" cy="-3" r="3.5" fill="#0F172A"/>
        <path d="M-60,16 C-54,-44 42,-44 48,16 Z" fill="#15803D" stroke="#14532D" stroke-width="3.5"/>
        <polygon points="-14,-18 4,-18 12,2 -6,12 -22,2" fill="#22C55E" stroke="#14532D" stroke-width="2"/>
      </g>`;
    case 'bee':
      return `<g transform="translate(200,150)">
        <ellipse cx="-16" cy="-42" rx="22" ry="30" transform="rotate(-18 -16 -42)" fill="#E0F2FE" stroke="#0284C7" stroke-width="2.5"/>
        <ellipse cx="20" cy="-42" rx="22" ry="30" transform="rotate(18 20 -42)" fill="#E0F2FE" stroke="#0284C7" stroke-width="2.5"/>
        <ellipse cx="0" cy="4" rx="54" ry="34" fill="#FACC15" stroke="#0F172A" stroke-width="3.5"/>
        <rect x="-22" y="-26" width="14" height="60" fill="#1E293B"/>
        <rect x="6" y="-26" width="14" height="60" fill="#1E293B"/>
        <circle cx="42" cy="-2" r="5" fill="#0F172A"/>
        <polygon points="-54,4 -70,-2 -54,-8" fill="#0F172A"/>
      </g>`;
    case 'cat':
      return `<g transform="translate(200,150)">
        <ellipse cx="-12" cy="20" rx="44" ry="26" fill="#FB923C" stroke="#C2410C" stroke-width="3"/>
        <path d="M-54,18 C-76,4 -74,-26 -54,-24" stroke="#FB923C" stroke-width="10" fill="none" stroke-linecap="round"/>
        <polygon points="10,-28 18,-56 32,-34" fill="#FB923C" stroke="#C2410C" stroke-width="2.5"/>
        <polygon points="40,-34 54,-56 60,-28" fill="#FB923C" stroke="#C2410C" stroke-width="2.5"/>
        <circle cx="36" cy="-14" r="26" fill="#FDBA74" stroke="#C2410C" stroke-width="3"/>
        <circle cx="26" cy="-18" r="4" fill="#0F172A"/><circle cx="46" cy="-18" r="4" fill="#0F172A"/>
        <polygon points="32,-10 40,-10 36,-4" fill="#EC4899"/>
        <line x1="8" y1="-8" x2="-8" y2="-12" stroke="#78350F" stroke-width="2"/><line x1="64" y1="-8" x2="80" y2="-12" stroke="#78350F" stroke-width="2"/>
      </g>`;
    case 'rabbit':
      return `<g transform="translate(200,156)">
        <ellipse cx="-14" cy="-54" rx="11" ry="34" transform="rotate(-10 -14 -54)" fill="#FFFFFF" stroke="#94A3B8" stroke-width="3"/>
        <ellipse cx="18" cy="-54" rx="11" ry="34" transform="rotate(10 18 -54)" fill="#FFFFFF" stroke="#94A3B8" stroke-width="3"/>
        <ellipse cx="-14" cy="-54" rx="5" ry="22" transform="rotate(-10 -14 -54)" fill="#FBCFE8"/>
        <ellipse cx="18" cy="-54" rx="5" ry="22" transform="rotate(10 18 -54)" fill="#FBCFE8"/>
        <circle cx="2" cy="-12" r="28" fill="#FFFFFF" stroke="#94A3B8" stroke-width="3"/>
        <ellipse cx="-12" cy="24" rx="34" ry="24" fill="#FFFFFF" stroke="#94A3B8" stroke-width="3"/>
        <circle cx="-48" cy="26" r="10" fill="#F1F5F9" stroke="#94A3B8" stroke-width="2"/>
        <circle cx="-6" cy="-18" r="4" fill="#0F172A"/><circle cx="14" cy="-18" r="4" fill="#0F172A"/>
        <circle cx="4" cy="-10" r="4" fill="#EC4899"/>
      </g>`;
    case 'tropical_fish':
      return `<g transform="translate(200,145)">
        <path d="M-46,0 C-18,-44 42,-40 68,0 C42,40 -18,44 -46,0 Z" fill="#F97316" stroke="#C2410C" stroke-width="3"/>
        <path d="M-2,-34 Q-10,0 -2,34 M20,-30 Q12,0 20,30" stroke="#FFFFFF" stroke-width="9" fill="none"/>
        <polygon points="-44,0 -78,-26 -78,26" fill="#FB923C" stroke="#C2410C" stroke-width="3"/>
        <circle cx="44" cy="-8" r="5" fill="#0F172A"/>
        <circle cx="88" cy="-24" r="6" fill="#BAE6FD"/><circle cx="102" cy="-42" r="8" fill="#BAE6FD"/>
      </g>`;
    case 'butterfly':
      return `<g transform="translate(200,148)">
        <ellipse cx="-38" cy="-18" rx="40" ry="30" transform="rotate(-22 -38 -18)" fill="#A855F7" stroke="#581C87" stroke-width="3"/>
        <ellipse cx="38" cy="-18" rx="40" ry="30" transform="rotate(22 38 -18)" fill="#A855F7" stroke="#581C87" stroke-width="3"/>
        <ellipse cx="-30" cy="24" rx="28" ry="24" fill="#EC4899" stroke="#581C87" stroke-width="3"/>
        <ellipse cx="30" cy="24" rx="28" ry="24" fill="#EC4899" stroke="#581C87" stroke-width="3"/>
        <circle cx="-42" cy="-20" r="10" fill="#FDE047"/><circle cx="42" cy="-20" r="10" fill="#FDE047"/>
        <ellipse cx="0" cy="6" rx="8" ry="38" fill="#1E293B"/>
        <path d="M-2,-30 Q-16,-56 -28,-50 M2,-30 Q16,-56 28,-50" stroke="#1E293B" stroke-width="3" fill="none" stroke-linecap="round"/>
      </g>`;
    case 'rooster':
      return `<g transform="translate(200,152)">
        <ellipse cx="-6" cy="14" rx="40" ry="30" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="3"/>
        <path d="M-42,8 C-66,-24 -76,-12 -52,18 C-74,-2 -76,18 -46,24" fill="#16A34A" stroke="#14532D" stroke-width="2.5"/>
        <circle cx="28" cy="-20" r="22" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="3"/>
        <path d="M16,-40 C16,-56 32,-56 32,-42 C36,-54 48,-46 40,-34 Z" fill="#EF4444"/>
        <polygon points="48,-22 64,-18 48,-12" fill="#F59E0B"/>
        <circle cx="34" cy="-24" r="4" fill="#0F172A"/>
        <path d="M-12,42 L-12,62 M8,42 L8,62" stroke="#D97706" stroke-width="4" stroke-linecap="round"/>
      </g>`;
    case 'cow':
      return `<g transform="translate(200,150)">
        <rect x="-52" y="-12" width="82" height="50" rx="18" fill="#FFFFFF" stroke="#334155" stroke-width="3"/>
        <circle cx="-26" cy="4" r="14" fill="#1E293B"/><circle cx="6" cy="18" r="11" fill="#1E293B"/>
        <rect x="-44" y="34" width="12" height="30" rx="4" fill="#FFFFFF" stroke="#334155" stroke-width="2.5"/>
        <rect x="8" y="34" width="12" height="30" rx="4" fill="#FFFFFF" stroke="#334155" stroke-width="2.5"/>
        <circle cx="38" cy="-16" r="26" fill="#FFFFFF" stroke="#334155" stroke-width="3"/>
        <ellipse cx="44" cy="-2" rx="18" ry="12" fill="#FBCFE8" stroke="#EC4899" stroke-width="2"/>
        <path d="M24,-38 Q16,-52 28,-48 M52,-38 Q60,-52 48,-48" stroke="#D97706" stroke-width="5" fill="none" stroke-linecap="round"/>
        <circle cx="32" cy="-22" r="3.5" fill="#0F172A"/><circle cx="48" cy="-22" r="3.5" fill="#0F172A"/>
      </g>`;
    case 'duck':
      return `<g transform="translate(200,154)">
        <path d="M-76,34 Q-38,24 0,34 Q38,44 76,34" stroke="#38BDF8" stroke-width="6" fill="none" stroke-linecap="round"/>
        <ellipse cx="-6" cy="14" rx="44" ry="26" fill="#FACC15" stroke="#CA8A04" stroke-width="3"/>
        <circle cx="32" cy="-22" r="22" fill="#FACC15" stroke="#CA8A04" stroke-width="3"/>
        <ellipse cx="56" cy="-18" rx="16" ry="7" fill="#F97316" stroke="#C2410C" stroke-width="2"/>
        <circle cx="38" cy="-26" r="4" fill="#0F172A"/>
      </g>`;
    case 'horse':
      return `<g transform="translate(200,150)">
        <ellipse cx="-14" cy="14" rx="48" ry="28" fill="#92400E" stroke="#451A03" stroke-width="3"/>
        <path d="M-60,4 C-78,-12 -74,20 -58,28" stroke="#451A03" stroke-width="8" fill="none" stroke-linecap="round"/>
        <rect x="-48" y="34" width="12" height="34" rx="4" fill="#78350F"/><rect x="16" y="34" width="12" height="34" rx="4" fill="#92400E"/>
        <path d="M18,-6 L38,-48 L66,-26 L56,-12 L26,-2 Z" fill="#92400E" stroke="#451A03" stroke-width="3" stroke-linejoin="round"/>
        <path d="M18,-10 L32,-48 L22,-42 L10,-12 Z" fill="#451A03"/>
        <circle cx="44" cy="-34" r="3.5" fill="#0F172A"/>
      </g>`;
    case 'frog':
      return `<g transform="translate(200,154)">
        <ellipse cx="0" cy="42" rx="76" ry="14" fill="#16A34A" stroke="#14532D" stroke-width="3"/>
        <ellipse cx="0" cy="14" rx="46" ry="30" fill="#4ADE80" stroke="#15803D" stroke-width="3"/>
        <circle cx="-22" cy="-18" r="14" fill="#4ADE80" stroke="#15803D" stroke-width="3"/>
        <circle cx="22" cy="-18" r="14" fill="#4ADE80" stroke="#15803D" stroke-width="3"/>
        <circle cx="-22" cy="-18" r="6" fill="#0F172A"/><circle cx="22" cy="-18" r="6" fill="#0F172A"/>
        <path d="M-20,12 Q0,26 20,12" stroke="#14532D" stroke-width="3.5" fill="none" stroke-linecap="round"/>
      </g>`;
    case 'crab':
      return `<g transform="translate(200,152)">
        <ellipse cx="0" cy="12" rx="46" ry="30" fill="#EF4444" stroke="#991B1B" stroke-width="3"/>
        <circle cx="-18" cy="-24" r="7" fill="#FFFFFF" stroke="#991B1B" stroke-width="2.5"/>
        <circle cx="18" cy="-24" r="7" fill="#FFFFFF" stroke="#991B1B" stroke-width="2.5"/>
        <circle cx="-18" cy="-24" r="3" fill="#0F172A"/><circle cx="18" cy="-24" r="3" fill="#0F172A"/>
        <path d="M-40,-4 C-66,-18 -64,-44 -46,-36 C-54,-22 -42,-16 -38,-28 Z" fill="#EF4444" stroke="#991B1B" stroke-width="2.5"/>
        <path d="M40,-4 C66,-18 64,-44 46,-36 C54,-22 42,-16 38,-28 Z" fill="#EF4444" stroke="#991B1B" stroke-width="2.5"/>
        <path d="M-44,18 L-68,36 M-40,28 L-58,48 M44,18 L68,36 M40,28 L58,48" stroke="#991B1B" stroke-width="5" stroke-linecap="round"/>
      </g>`;

    // ================= 056..070 COUNTING =================
    case 'count_apples_3':
      return `${drawMiniApple(108, 145, 0.88)}${drawMiniApple(200, 145, 0.88)}${drawMiniApple(292, 145, 0.88)}`;
    case 'count_stars_2plus2':
      return `<g>
        <polygon points="${starPolygon(100, 118, 28, 12)}" fill="#FACC15" stroke="#CA8A04" stroke-width="2.5"/>
        <polygon points="${starPolygon(154, 118, 28, 12)}" fill="#FACC15" stroke="#CA8A04" stroke-width="2.5"/>
        <text x="200" y="130" text-anchor="middle" font-family="Arial,sans-serif" font-size="36" font-weight="900" fill="#7C3AED">+</text>
        <polygon points="${starPolygon(246, 118, 28, 12)}" fill="#FACC15" stroke="#CA8A04" stroke-width="2.5"/>
        <polygon points="${starPolygon(300, 118, 28, 12)}" fill="#FACC15" stroke="#CA8A04" stroke-width="2.5"/>
        <text x="200" y="198" text-anchor="middle" font-family="Arial,sans-serif" font-size="28" font-weight="900" fill="#4C1D95">2 + 2 = ?</text>
      </g>`;
    case 'count_fingers_5':
      return `<g transform="translate(200,152)">
        <rect x="-34" y="-4" width="68" height="62" rx="22" fill="#FDE68A" stroke="#B45309" stroke-width="3"/>
        <rect x="-56" y="4" width="26" height="16" rx="8" transform="rotate(-28 -44 12)" fill="#FDE68A" stroke="#B45309" stroke-width="3"/>
        <rect x="-32" y="-56" width="14" height="56" rx="7" fill="#FDE68A" stroke="#B45309" stroke-width="3"/>
        <rect x="-14" y="-68" width="14" height="68" rx="7" fill="#FDE68A" stroke="#B45309" stroke-width="3"/>
        <rect x="4" y="-60" width="14" height="60" rx="7" fill="#FDE68A" stroke="#B45309" stroke-width="3"/>
        <rect x="22" y="-44" width="13" height="46" rx="6.5" fill="#FDE68A" stroke="#B45309" stroke-width="3"/>
      </g>`;
    case 'count_car_4wheels':
      return `<g transform="translate(200,136)">
        <path d="M-74,10 L-56,-34 L38,-34 L62,10 L82,14 L82,36 L-82,36 L-82,14 Z" fill="#EF4444" stroke="#991B1B" stroke-width="3.5" stroke-linejoin="round"/>
        <polygon points="-44,-26 -4,-26 -4,6 -52,6" fill="#E0F2FE" stroke="#991B1B" stroke-width="2"/>
        <polygon points="4,-26 32,-26 48,6 4,6" fill="#E0F2FE" stroke="#991B1B" stroke-width="2"/>
        <circle cx="-44" cy="36" r="18" fill="#1E293B"/><circle cx="-44" cy="36" r="8" fill="#CBD5E1"/>
        <circle cx="46" cy="36" r="18" fill="#1E293B"/><circle cx="46" cy="36" r="8" fill="#CBD5E1"/>
        <text x="0" y="84" text-anchor="middle" font-family="Arial,sans-serif" font-size="26" font-weight="900" fill="#4C1D95">10 - 4 = ?</text>
      </g>`;
    case 'count_balls_2':
      return `<g>
        <circle cx="144" cy="145" r="44" fill="#FFFFFF" stroke="#1E293B" stroke-width="3.5"/>
        <polygon points="144,124 162,137 155,158 133,158 126,137" fill="#1E293B"/>
        <circle cx="256" cy="145" r="44" fill="#FFFFFF" stroke="#1E293B" stroke-width="3.5"/>
        <polygon points="256,124 274,137 267,158 245,158 238,137" fill="#1E293B"/>
      </g>`;
    case 'count_oranges_6':
      return `${drawMiniOrange(120, 108, 28)}${drawMiniOrange(200, 108, 28)}${drawMiniOrange(280, 108, 28)}${drawMiniOrange(120, 184, 28)}${drawMiniOrange(200, 184, 28)}${drawMiniOrange(280, 184, 28)}`;
    case 'count_balloons_3':
      return `<g>
        <line x1="126" y1="138" x2="126" y2="216" stroke="#64748B" stroke-width="2.5"/>
        <line x1="200" y1="124" x2="200" y2="216" stroke="#64748B" stroke-width="2.5"/>
        <line x1="274" y1="138" x2="274" y2="216" stroke="#64748B" stroke-width="2.5"/>
        <ellipse cx="126" cy="108" rx="30" ry="38" fill="#EF4444" stroke="#B91C1C" stroke-width="2.5"/>
        <ellipse cx="200" cy="94" rx="30" ry="38" fill="#3B82F6" stroke="#1D4ED8" stroke-width="2.5"/>
        <ellipse cx="274" cy="108" rx="30" ry="38" fill="#10B981" stroke="#047857" stroke-width="2.5"/>
      </g>`;
    case 'count_flowers_5':
      return `${drawMiniFlower(106, 110, '#EC4899', 12)}${drawMiniFlower(200, 110, '#A855F7', 12)}${drawMiniFlower(294, 110, '#F43F5E', 12)}${drawMiniFlower(152, 184, '#38BDF8', 12)}${drawMiniFlower(248, 184, '#FB923C', 12)}`;
    case 'count_birds_4':
      return `${drawMiniBird(132, 108, 0.72)}${drawMiniBird(268, 108, 0.72)}${drawMiniBird(132, 184, 0.72)}${drawMiniBird(268, 184, 0.72)}`;
    case 'count_cups_2':
      return `<g>
        <g transform="translate(132,155) scale(0.8)">
          <path d="M36,-20 C60,-20 60,16 32,16" fill="none" stroke="#0284C7" stroke-width="8"/>
          <path d="M-38,-28 L38,-28 C34,26 18,36 0,36 C-18,36 -34,26 -38,-28 Z" fill="#FFFFFF" stroke="#0284C7" stroke-width="4"/>
        </g>
        <g transform="translate(268,155) scale(0.8)">
          <path d="M36,-20 C60,-20 60,16 32,16" fill="none" stroke="#0284C7" stroke-width="8"/>
          <path d="M-38,-28 L38,-28 C34,26 18,36 0,36 C-18,36 -34,26 -38,-28 Z" fill="#FFFFFF" stroke="#0284C7" stroke-width="4"/>
        </g>
      </g>`;
    case 'count_marbles_7': {
      const cols = ['#EF4444', '#F97316', '#FACC15', '#22C55E', '#38BDF8', '#6366F1', '#EC4899'];
      const coords = [
        [110, 112], [170, 112], [230, 112], [290, 112],
        [140, 178], [200, 178], [260, 178],
      ];
      return coords
        .map(([x, y], idx) => `<circle cx="${x}" cy="${y}" r="24" fill="${cols[idx]}" stroke="#1E293B" stroke-width="2.5"/><circle cx="${x - 7}" cy="${y - 7}" r="6" fill="#FFFFFF" opacity="0.65"/>`)
        .join('');
    }
    case 'count_blocks_3plus2':
      return `<g>
        <rect x="64" y="138" width="38" height="38" rx="6" fill="#EF4444" stroke="#991B1B" stroke-width="2.5"/>
        <rect x="108" y="138" width="38" height="38" rx="6" fill="#3B82F6" stroke="#1E40AF" stroke-width="2.5"/>
        <rect x="86" y="96" width="38" height="38" rx="6" fill="#FACC15" stroke="#A16207" stroke-width="2.5"/>
        <text x="196" y="148" text-anchor="middle" font-family="Arial,sans-serif" font-size="38" font-weight="900" fill="#7C3AED">+</text>
        <rect x="234" y="122" width="38" height="38" rx="6" fill="#10B981" stroke="#065F46" stroke-width="2.5"/>
        <rect x="280" y="122" width="38" height="38" rx="6" fill="#A855F7" stroke="#6B21A8" stroke-width="2.5"/>
        <text x="200" y="216" text-anchor="middle" font-family="Arial,sans-serif" font-size="26" font-weight="900" fill="#4C1D95">3 + 2 = ?</text>
      </g>`;
    case 'count_octopus_8':
      return `<g transform="translate(200,145)">
        <circle cx="0" cy="-22" r="38" fill="#EC4899" stroke="#9D174D" stroke-width="3"/>
        <circle cx="-12" cy="-24" r="5" fill="#0F172A"/><circle cx="12" cy="-24" r="5" fill="#0F172A"/>
        ${[-66,-48,-30,-10,10,30,48,66].map((tx, i) => `<path d="M${(tx * 0.4).toFixed(0)},12 Q${tx},36 ${(tx + (i % 2 ? -8 : 8)).toFixed(0)},64" stroke="#EC4899" stroke-width="9" fill="none" stroke-linecap="round"/>`).join('')}
      </g>`;
    case 'count_clock_12':
      return drawAnalogClockBase(200, 145, 76, 300, 0, true);
    case 'count_hearts_1plus1':
      return `<g>
        <path d="M126,158 C92,126 90,94 114,94 C124,94 126,104 126,104 C126,104 128,94 138,94 C162,94 160,126 126,158 Z" fill="#EF4444" stroke="#991B1B" stroke-width="2.5"/>
        <text x="200" y="138" text-anchor="middle" font-family="Arial,sans-serif" font-size="36" font-weight="900" fill="#7C3AED">+</text>
        <path d="M274,158 C240,126 238,94 262,94 C272,94 274,104 274,104 C274,104 276,94 286,94 C310,94 308,126 274,158 Z" fill="#EF4444" stroke="#991B1B" stroke-width="2.5"/>
        <text x="200" y="206" text-anchor="middle" font-family="Arial,sans-serif" font-size="26" font-weight="900" fill="#4C1D95">1 + 1 = ?</text>
      </g>`;

    // ================= 071..082 SAFETY & ROUTINE =================
    case 'medicine':
      return `<g transform="translate(200,148)">
        <rect x="-78" y="-44" width="76" height="92" rx="12" fill="#E2E8F0" stroke="#475569" stroke-width="3"/>
        <ellipse cx="-56" cy="-16" rx="12" ry="8" fill="#EF4444"/>
        <ellipse cx="-24" cy="-16" rx="12" ry="8" fill="#3B82F6"/>
        <ellipse cx="-56" cy="18" rx="12" ry="8" fill="#3B82F6"/>
        <ellipse cx="-24" cy="18" rx="12" ry="8" fill="#EF4444"/>
        <path d="M24,-22 L76,-22 L68,52 L32,52 Z" fill="#F0F9FF" stroke="#0284C7" stroke-width="3.5"/>
        <path d="M27,6 L73,6 L68,50 L32,50 Z" fill="#38BDF8"/>
      </g>`;
    case 'nonslip_shoes':
      return drawShoe(200, 145, '#0D9488', '#F8FAFC', true);
    case 'stove_off':
      return `<g transform="translate(200,150)">
        <rect x="-82" y="-36" width="164" height="84" rx="12" fill="#E2E8F0" stroke="#334155" stroke-width="3.5"/>
        <circle cx="-38" cy="-8" r="24" fill="#334155"/><circle cx="38" cy="-8" r="24" fill="#334155"/>
        <circle cx="0" cy="24" r="14" fill="#FFFFFF" stroke="#0F172A" stroke-width="3"/>
        <line x1="0" y1="12" x2="0" y2="24" stroke="#16A34A" stroke-width="5" stroke-linecap="round"/>
        <rect x="-24" y="-66" width="48" height="22" rx="8" fill="#16A34A"/>
        <text x="0" y="-50" text-anchor="middle" font-family="Arial,sans-serif" font-size="13" font-weight="900" fill="#FFFFFF">OFF</text>
      </g>`;
    case 'wash_hands':
      return `<g transform="translate(200,145)">
        <path d="M-24,-64 L-24,-40 L16,-40" stroke="#64748B" stroke-width="10" fill="none" stroke-linecap="round"/>
        <path d="M12,-40 Q12,-8 12,18" stroke="#38BDF8" stroke-width="10" stroke-dasharray="10 6" stroke-linecap="round"/>
        <ellipse cx="-18" cy="26" rx="34" ry="18" transform="rotate(-14 -18 26)" fill="#FDE68A" stroke="#B45309" stroke-width="2.5"/>
        <ellipse cx="22" cy="28" rx="34" ry="18" transform="rotate(14 22 28)" fill="#FDE68A" stroke="#B45309" stroke-width="2.5"/>
        <circle cx="-6" cy="8" r="11" fill="#FFFFFF" stroke="#38BDF8" stroke-width="2"/>
        <circle cx="14" cy="12" r="9" fill="#FFFFFF" stroke="#38BDF8" stroke-width="2"/>
        <circle cx="2" cy="24" r="10" fill="#FFFFFF" stroke="#38BDF8" stroke-width="2"/>
      </g>`;
    case 'stair_handrail':
      return `<g transform="translate(200,152)">
        <polygon points="-76,56 -76,34 -38,34 -38,10 0,10 0,-14 38,-14 38,-38 76,-38 76,56" fill="#CBD5E1" stroke="#475569" stroke-width="3"/>
        <line x1="-68" y1="-8" x2="72" y2="-68" stroke="#F97316" stroke-width="10" stroke-linecap="round"/>
        <line x1="-42" y1="-20" x2="-42" y2="34" stroke="#64748B" stroke-width="6"/>
        <line x1="28" y1="-50" x2="28" y2="-14" stroke="#64748B" stroke-width="6"/>
        <circle cx="0" cy="-38" r="10" fill="#FDE68A" stroke="#B45309" stroke-width="2.5"/>
      </g>`;
    case 'walking_cane':
      return `<g transform="translate(200,148)">
        <path d="M-24,-42 C-24,-74 20,-74 20,-42 L20,58" stroke="#92400E" stroke-width="14" fill="none" stroke-linecap="round"/>
        <rect x="10" y="52" width="20" height="16" rx="4" fill="#1E293B"/>
      </g>`;
    case 'bp_monitor':
      return `<g transform="translate(200,150)">
        <rect x="-78" y="-36" width="84" height="82" rx="14" fill="#FFFFFF" stroke="#0284C7" stroke-width="3.5"/>
        <rect x="-64" y="-22" width="56" height="36" rx="6" fill="#0F172A"/>
        <text x="-36" y="-4" text-anchor="middle" font-family="Arial,sans-serif" font-size="14" font-weight="900" fill="#4ADE80">120</text>
        <text x="-36" y="10" text-anchor="middle" font-family="Arial,sans-serif" font-size="11" font-weight="800" fill="#38BDF8">80</text>
        <rect x="30" y="-22" width="46" height="58" rx="10" fill="#1D4ED8" stroke="#1E3A8A" stroke-width="3"/>
        <path d="M6,26 L30,26" stroke="#334155" stroke-width="6"/>
      </g>`;
    case 'safety_helmet':
      return `<g transform="translate(200,152)">
        <path d="M-66,18 C-66,-50 66,-50 66,18 Z" fill="#F59E0B" stroke="#B45309" stroke-width="3.5"/>
        <rect x="-78" y="14" width="156" height="14" rx="7" fill="#D97706" stroke="#92400E" stroke-width="2.5"/>
        <rect x="-12" y="-36" width="24" height="50" rx="6" fill="#FDE047"/>
      </g>`;
    case 'seatbelt':
      return `<g transform="translate(200,148)">
        <circle cx="-18" cy="-42" r="22" fill="#BAE6FD" stroke="#0284C7" stroke-width="3"/>
        <path d="M-56,52 C-56,-4 22,-4 22,52 Z" fill="#38BDF8" stroke="#0284C7" stroke-width="3"/>
        <line x1="-54" y1="-26" x2="34" y2="52" stroke="#1E293B" stroke-width="14" stroke-linecap="round"/>
        <rect x="26" y="34" width="28" height="22" rx="4" fill="#EF4444" stroke="#991B1B" stroke-width="2.5"/>
      </g>`;
    case 'crosswalk':
      return `<g transform="translate(200,150)">
        <rect x="-86" y="-28" width="172" height="82" rx="8" fill="#334155"/>
        ${[-66,-34,-2,30,62].map((x) => `<rect x="${x}" y="-16" width="18" height="58" rx="2" fill="#FFFFFF"/>`).join('')}
        <circle cx="0" cy="-54" r="10" fill="#10B981"/>
        <line x1="0" y1="-44" x2="0" y2="-16" stroke="#10B981" stroke-width="6" stroke-linecap="round"/>
      </g>`;
    case 'wet_floor_cone':
      return `<g transform="translate(200,152)">
        <polygon points="0,-64 -44,48 44,48" fill="#FACC15" stroke="#CA8A04" stroke-width="3.5" stroke-linejoin="round"/>
        <rect x="-56" y="46" width="112" height="14" rx="6" fill="#EAB308" stroke="#A16207" stroke-width="2.5"/>
        <polygon points="0,-24 -20,24 20,24" fill="#FEF08A" stroke="#1E293B" stroke-width="3"/>
        <text x="0" y="18" text-anchor="middle" font-family="Arial,sans-serif" font-size="22" font-weight="900" fill="#0F172A">!</text>
      </g>`;
    case 'shoulder_stretch':
      return `<g transform="translate(200,152)">
        <circle cx="0" cy="-46" r="16" fill="#0284C7"/>
        <rect x="-14" y="-24" width="28" height="56" rx="10" fill="#0D9488"/>
        <line x1="-14" y1="-16" x2="-58" y2="-46" stroke="#10B981" stroke-width="11" stroke-linecap="round"/>
        <line x1="14" y1="-16" x2="58" y2="-46" stroke="#10B981" stroke-width="11" stroke-linecap="round"/>
        <path d="M-42,-64 A54,54 0 0,1 42,-64" stroke="#38BDF8" stroke-width="4" stroke-dasharray="6 4" fill="none"/>
      </g>`;

    // ================= 083..092 COLOR SHAPE TIME =================
    case 'traffic_red':
      return `<g transform="translate(200,145)">
        <rect x="-32" y="-74" width="64" height="148" rx="16" fill="#1E293B" stroke="#0F172A" stroke-width="4"/>
        <circle cx="0" cy="-42" r="20" fill="#EF4444" stroke="#FCA5A5" stroke-width="4"/>
        <circle cx="0" cy="0" r="17" fill="#475569"/>
        <circle cx="0" cy="42" r="17" fill="#475569"/>
      </g>`;
    case 'traffic_green':
      return `<g transform="translate(200,145)">
        <rect x="-32" y="-74" width="64" height="148" rx="16" fill="#1E293B" stroke="#0F172A" stroke-width="4"/>
        <circle cx="0" cy="-42" r="17" fill="#475569"/>
        <circle cx="0" cy="0" r="17" fill="#475569"/>
        <circle cx="0" cy="42" r="20" fill="#22C55E" stroke="#86EFAC" stroke-width="4"/>
      </g>`;
    case 'triangle':
      return `<polygon points="200,66 116,204 284,204" fill="#EF4444" stroke="#991B1B" stroke-width="5" stroke-linejoin="round"/>`;
    case 'blue_circle':
      return `<circle cx="200" cy="145" r="68" fill="#0EA5E9" stroke="#0369A1" stroke-width="5"/>`;
    case 'green_square':
      return `<rect x="132" y="77" width="136" height="136" rx="10" fill="#10B981" stroke="#047857" stroke-width="5"/>`;
    case 'yellow_star':
      return `<polygon points="${starPolygon(200, 145, 74, 32)}" fill="#FACC15" stroke="#CA8A04" stroke-width="4.5" stroke-linejoin="round"/>`;
    case 'morning_sun':
      return `<g transform="translate(200,162)">
        <circle cx="0" cy="-26" r="46" fill="#FACC15" stroke="#F97316" stroke-width="4"/>
        ${[0,45,90,135,180,225,270,315].map((d) => {
          const a = (d * Math.PI) / 180;
          return `<line x1="${(58 * Math.cos(a)).toFixed(1)}" y1="${(-26 + 58 * Math.sin(a)).toFixed(1)}" x2="${(76 * Math.cos(a)).toFixed(1)}" y2="${(-26 + 76 * Math.sin(a)).toFixed(1)}" stroke="#F97316" stroke-width="6" stroke-linecap="round"/>`;
        }).join('')}
        <path d="M-96,42 Q-38,14 14,34 Q56,12 96,42 Z" fill="#16A34A"/>
      </g>`;
    case 'night_moon':
      return `<g transform="translate(200,145)">
        <rect x="-110" y="-76" width="220" height="152" rx="18" fill="#0F172A"/>
        <path d="M-12,-46 A48,48 0 1,0 44,32 A36,36 0 1,1 -12,-46 Z" fill="#FDE047"/>
        <polygon points="${starPolygon(-56, -34, 10, 4)}" fill="#FEF08A"/>
        <polygon points="${starPolygon(64, -36, 12, 5)}" fill="#FEF08A"/>
        <polygon points="${starPolygon(72, 24, 8, 3)}" fill="#FEF08A"/>
      </g>`;
    case 'morning_clock_8am':
      return `<g>${drawAnalogClockBase(200, 138, 66, 240, 0, true)}<text x="200" y="232" text-anchor="middle" font-family="Arial,sans-serif" font-size="20" font-weight="900" fill="#0284C7">08:00 AM</text></g>`;
    case 'rainbow':
      return `<g transform="translate(200,188)">
        <path d="M-96,0 A96,96 0 0,1 96,0" stroke="#EF4444" stroke-width="12" fill="none"/>
        <path d="M-84,0 A84,84 0 0,1 84,0" stroke="#F97316" stroke-width="12" fill="none"/>
        <path d="M-72,0 A72,72 0 0,1 72,0" stroke="#FACC15" stroke-width="12" fill="none"/>
        <path d="M-60,0 A60,60 0 0,1 60,0" stroke="#22C55E" stroke-width="12" fill="none"/>
        <path d="M-48,0 A48,48 0 0,1 48,0" stroke="#3B82F6" stroke-width="12" fill="none"/>
        <circle cx="-90" cy="4" r="20" fill="#FFFFFF"/><circle cx="-72" cy="8" r="16" fill="#FFFFFF"/>
        <circle cx="90" cy="4" r="20" fill="#FFFFFF"/><circle cx="72" cy="8" r="16" fill="#FFFFFF"/>
      </g>`;

    // ================= 093..100 POSITION =================
    case 'arrow_up':
      return `<g transform="translate(200,145)">
        <polygon points="0,-72 -58,-4 -24,-4 -24,68 24,68 24,-4 58,-4" fill="#0D9488" stroke="#115E59" stroke-width="4" stroke-linejoin="round"/>
      </g>`;
    case 'arrow_down':
      return `<g transform="translate(200,145)">
        <polygon points="0,72 -58,4 -24,4 -24,-68 24,-68 24,4 58,4" fill="#2563EB" stroke="#1E40AF" stroke-width="4" stroke-linejoin="round"/>
      </g>`;
    case 'arrow_right':
      return `<g transform="translate(200,145)">
        <polygon points="78,0 10,-54 10,-22 -74,-22 -74,22 10,22 10,54" fill="#10B981" stroke="#047857" stroke-width="4" stroke-linejoin="round"/>
      </g>`;
    case 'arrow_left':
      return `<g transform="translate(200,145)">
        <polygon points="-78,0 -10,-54 -10,-22 74,-22 74,22 -10,22 -10,54" fill="#F97316" stroke="#C2410C" stroke-width="4" stroke-linejoin="round"/>
      </g>`;
    case 'house_roof_top':
      return `<g transform="translate(200,155)">
        <rect x="-64" y="-8" width="128" height="68" fill="#FEF3C7" stroke="#B45309" stroke-width="3.5"/>
        <rect x="-18" y="18" width="36" height="42" fill="#92400E"/>
        <rect x="-50" y="6" width="22" height="22" fill="#BAE6FD" stroke="#0284C7" stroke-width="2.5"/>
        <rect x="28" y="6" width="22" height="22" fill="#BAE6FD" stroke="#0284C7" stroke-width="2.5"/>
        <polygon points="0,-74 -84,-6 84,-6" fill="#EF4444" stroke="#991B1B" stroke-width="4" stroke-linejoin="round"/>
      </g>`;
    case 'apple_on_table':
      return `<g transform="translate(200,164)">
        <rect x="-86" y="-8" width="172" height="16" rx="4" fill="#D97706" stroke="#78350F" stroke-width="3"/>
        <rect x="-70" y="8" width="14" height="52" fill="#92400E"/><rect x="56" y="8" width="14" height="52" fill="#92400E"/>
        ${drawMiniApple(0, -48, 0.78)}
      </g>`;
    case 'ball_under_table':
      return `<g transform="translate(200,156)">
        <rect x="-86" y="-26" width="172" height="16" rx="4" fill="#D97706" stroke="#78350F" stroke-width="3"/>
        <rect x="-70" y="-10" width="14" height="68" fill="#92400E"/><rect x="56" y="-10" width="14" height="68" fill="#92400E"/>
        <circle cx="0" cy="30" r="28" fill="#FFFFFF" stroke="#1E293B" stroke-width="3"/>
        <polygon points="0,16 12,24 8,38 -8,38 -12,24" fill="#1E293B"/>
      </g>`;
    case 'bird_on_branch':
      return `<g transform="translate(200,162)">
        <line x1="-96" y1="24" x2="96" y2="10" stroke="#78350F" stroke-width="11" stroke-linecap="round"/>
        <ellipse cx="64" cy="4" rx="18" ry="9" transform="rotate(-18 64 4)" fill="#22C55E"/>
        ${drawMiniBird(-10, -24, 1.05, '#BAE6FD')}
      </g>`;
  }
}

export function buildCardIllustrationDataUri(spec: CardIllustrationSpec): string {
  const theme = CATEGORY_THEME[spec.category];
  const gid = `bg-${spec.cardNumber}`;
  const art = renderArtwork(spec.artKey);
  const badge = `#${String(spec.cardNumber).padStart(3, '0')}`;

  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 260" width="400" height="260">
    <defs>
      <linearGradient id="${gid}" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="${theme.bgTop}"/>
        <stop offset="100%" stop-color="${theme.bgBottom}"/>
      </linearGradient>
    </defs>
    <rect width="400" height="260" rx="26" fill="url(#${gid})"/>
    <circle cx="48" cy="44" r="56" fill="${theme.pillBg}" opacity="0.35"/>
    <circle cx="356" cy="222" r="64" fill="${theme.pillBg}" opacity="0.35"/>
    <rect x="318" y="12" width="68" height="28" rx="14" fill="${theme.pillBg}"/>
    <text x="352" y="31" text-anchor="middle" font-family="Arial,sans-serif" font-size="14" font-weight="900" fill="${theme.accent}">${badge}</text>
    <g transform="translate(200,142) scale(1.26) translate(-200,-142)">
      ${art}
    </g>
  </svg>`;

  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

