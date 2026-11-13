import type { TrilingualEntry } from '@drtechcares/clinical-i18n';

export type CognitiveCategory =
  | 'food_fruit'
  | 'objects'
  | 'animals'
  | 'counting'
  | 'safety_routine'
  | 'color_shape_time'
  | 'position';

export interface CognitiveCardQuestion {
  id: string;
  category: CognitiveCategory;
  icon: string;
  question: TrilingualEntry;
  correctAnswer: 'yes' | 'no';
  explanation: TrilingualEntry;
}

interface CardSeed {
  category: CognitiveCategory;
  icon: string;
  subject: TrilingualEntry;
  positiveClaim: TrilingualEntry;
  negativeClaim: TrilingualEntry;
}

const SEEDS: readonly CardSeed[] = [
  // food_fruit
  { category: 'food_fruit', icon: '🍎', subject: { th: 'แอปเปิ้ล', en: 'Apple', cn: '苹果' }, positiveClaim: { th: 'สิ่งนี้คือผลไม้ที่รับประทานได้ใช่ไหม?', en: 'Is this an edible fruit?', cn: '这是可以食用的水果吗？' }, negativeClaim: { th: 'สิ่งนี้เป็นอุปกรณ์อิเล็กทรอนิกส์ใช่ไหม?', en: 'Is this an electronic device?', cn: '这是电子设备吗？' } },
  { category: 'food_fruit', icon: '🍌', subject: { th: 'กล้วยหอม', en: 'Banana', cn: '香蕉' }, positiveClaim: { th: 'ผลไม้นี้มีสีเหลืองเมื่อสุกใช่ไหม?', en: 'Is this fruit yellow when ripe?', cn: '这种水果成熟后是黄色的吗？' }, negativeClaim: { th: 'ผลไม้นี้ต้องต้ม 1 ชั่วโมงก่อนปลอกเปลือกใช่ไหม?', en: 'Must you boil this for an hour before peeling?', cn: '这种水果必须煮一小时才能剥皮吗？' } },
  { category: 'food_fruit', icon: '🥦', subject: { th: 'บรอกโคลี', en: 'Broccoli', cn: '西兰花' }, positiveClaim: { th: 'สิ่งนี้คือผักสีเขียวที่มีประโยชน์ใช่ไหม?', en: 'Is this a healthy green vegetable?', cn: '这是有益健康的绿色蔬菜吗？' }, negativeClaim: { th: 'สิ่งนี้คือขนมหวานน้ำตาลสูงใช่ไหม?', en: 'Is this a sugary dessert?', cn: '这是高糖甜点吗？' } },
  { category: 'food_fruit', icon: '🥛', subject: { th: 'นมสด', en: 'Milk', cn: '牛奶' }, positiveClaim: { th: 'เครื่องดื่มนี้มีแคลเซียมช่วยบำรุงกระดูกใช่ไหม?', en: 'Does this drink contain calcium for bones?', cn: '这种饮品含有有益骨骼的钙质吗？' }, negativeClaim: { th: 'สิ่งนี้ใช้เติมน้ำมันรถยนต์ใช่ไหม?', en: 'Is this used as car fuel?', cn: '这是用来给汽车加油的吗？' } },
  { category: 'food_fruit', icon: '🍉', subject: { th: 'แตงโม', en: 'Watermelon', cn: '西瓜' }, positiveClaim: { th: 'ผลไม้นี้มีน้ำมากช่วยให้สดชื่นใช่ไหม?', en: 'Is this fruit juicy and refreshing?', cn: '这种水果多汁且解渴吗？' }, negativeClaim: { th: 'ผลไม้นี้มีขนาดเล็กกว่าเมล็ดถั่วใช่ไหม?', en: 'Is this fruit smaller than a pea?', cn: '这种水果比豌豆还小吗？' } },
  { category: 'food_fruit', icon: '🥕', subject: { th: 'แครอท', en: 'Carrot', cn: '胡萝卜' }, positiveClaim: { th: 'ผักชนิดนี้มีสีส้มและบำรุงสายตาใช่ไหม?', en: 'Is this orange vegetable good for vision?', cn: '这种橙色蔬菜有益视力吗？' }, negativeClaim: { th: 'สิ่งนี้เติบโตบนยอดต้นมะพร้าวใช่ไหม?', en: 'Does this grow at the top of coconut trees?', cn: '它长在椰子树顶上吗？' } },
  { category: 'food_fruit', icon: '🐟', subject: { th: 'ปลาทะเล', en: 'Fish', cn: '鱼' }, positiveClaim: { th: 'อาหารนี้เป็นแหล่งโปรตีนที่ย่อยง่ายใช่ไหม?', en: 'Is this an easily digestible protein source?', cn: '这是易于消化的蛋白质食物吗？' }, negativeClaim: { th: 'สิ่งนี้อาศัยอยู่ในทะเลทรายแห้งแล้งใช่ไหม?', en: 'Does this live in dry deserts?', cn: '它生活在干燥的沙漠里吗？' } },
  // objects
  { category: 'objects', icon: '🔑', subject: { th: 'กุญแจ', en: 'Key', cn: '钥匙' }, positiveClaim: { th: 'ของสิ่งนี้ใช้สำหรับไขเปิดประตูใช่ไหม?', en: 'Is this used to unlock a door?', cn: '这是用来开锁开门的吗？' }, negativeClaim: { th: 'ของสิ่งนี้ใช้ตักซุปรับประทานใช่ไหม?', en: 'Is this used to eat soup?', cn: '这是用来喝汤的吗？' } },
  { category: 'objects', icon: '🪥', subject: { th: 'แปรงสีฟัน', en: 'Toothbrush', cn: '牙刷' }, positiveClaim: { th: 'อุปกรณ์นี้ใช้ทำความสะอาดฟันทุกเช้า-เย็นใช่ไหม?', en: 'Is this used to clean teeth morning and night?', cn: '这是早晚用来刷牙的工具吗？' }, negativeClaim: { th: 'อุปกรณ์นี้ใช้กวาดลานบ้านใช่ไหม?', en: 'Is this used to sweep the yard?', cn: '这是用来扫院子的吗？' } },
  { category: 'objects', icon: '👓', subject: { th: 'แว่นตา', en: 'Eyeglasses', cn: '眼镜' }, positiveClaim: { th: 'สิ่งนี้ช่วยให้มองเห็นภาพและตัวหนังสือชัดขึ้นใช่ไหม?', en: 'Does this help you see text more clearly?', cn: '这能帮助您更清晰地看东西吗？' }, negativeClaim: { th: 'สิ่งนี้สวมใส่ที่เท้าเวลาวิ่งใช่ไหม?', en: 'Do you wear this on your feet to run?', cn: '跑步时把它穿在脚上吗？' } },
  { category: 'objects', icon: '☂️', subject: { th: 'ร่ม', en: 'Umbrella', cn: '雨伞' }, positiveClaim: { th: 'สิ่งนี้ใช้กันฝนและกันแดดเวลาออกนอกบ้านใช่ไหม?', en: 'Is this used for protection from rain or sun?', cn: '外出时它用来遮雨防晒吗？' }, negativeClaim: { th: 'สิ่งนี้ใช้ทอดไข่ในกระทะใช่ไหม?', en: 'Is this used to fry eggs in a pan?', cn: '这是用来煎鸡蛋的吗？' } },
  { category: 'objects', icon: '⏰', subject: { th: 'นาฬิกาปลุก', en: 'Alarm Clock', cn: '闹钟' }, positiveClaim: { th: 'อุปกรณ์นี้ใช้บอกเวลาและตั้งปลุกใช่ไหม?', en: 'Does this tell time and set alarms?', cn: '它用来报时和设置闹钟吗？' }, negativeClaim: { th: 'อุปกรณ์นี้ใช้ชั่งน้ำหนักตัวใช่ไหม?', en: 'Is this used to weigh body mass?', cn: '这是用来称体重的吗？' } },
  { category: 'objects', icon: '🪑', subject: { th: 'เก้าอี้', en: 'Chair', cn: '椅子' }, positiveClaim: { th: 'สิ่งนี้ใช้นั่งพักผ่อนหรือฝึกกายภาพท่านั่งใช่ไหม?', en: 'Is this used for sitting during rehab?', cn: '这是用来坐着休息或做坐姿训练的吗？' }, negativeClaim: { th: 'สิ่งนี้ใช้โทรออกหาญาติใช่ไหม?', en: 'Is this used to make phone calls?', cn: '这是用来打电话的吗？' } },
  { category: 'objects', icon: '🩺', subject: { th: 'หูฟังแพทย์', en: 'Stethoscope', cn: '听诊器' }, positiveClaim: { th: 'เครื่องมือนี้แพทย์ใช้ตรวจฟังเสียงหัวใจและปอดใช่ไหม?', en: 'Do doctors use this to listen to heart & lungs?', cn: '医生用它来听心肺声音吗？' }, negativeClaim: { th: 'เครื่องมือนี้ใช้ตัดกระดาษใช่ไหม?', en: 'Is this used for cutting paper?', cn: '这是用来剪纸的吗？' } },
  // animals
  { category: 'animals', icon: '🐘', subject: { th: 'ช้าง', en: 'Elephant', cn: '大象' }, positiveClaim: { th: 'สัตว์ชนิดนี้มีงวงยาวและขนาดตัวใหญ่ใช่ไหม?', en: 'Does this large animal have a long trunk?', cn: '这种大型动物有长长的鼻子吗？' }, negativeClaim: { th: 'สัตว์ชนิดนี้บินได้สูงเหมือนนกอินทรีใช่ไหม?', en: 'Can this animal fly high like an eagle?', cn: '这种动物能像老鹰一样高飞吗？' } },
  { category: 'animals', icon: '🐕', subject: { th: 'สุนัข', en: 'Dog', cn: '狗' }, positiveClaim: { th: 'สัตว์เลี้ยงชนิดนี้มี 4 ขาและเห่าได้ใช่ไหม?', en: 'Does this pet have 4 legs and bark?', cn: '这种宠物有四条腿并且会汪汪叫吗？' }, negativeClaim: { th: 'สัตว์ชนิดนี้หายใจด้วยเหงือกใต้ทะเลลึกใช่ไหม?', en: 'Does this breathe with gills deep underwater?', cn: '它在深海里用鳃呼吸吗？' } },
  { category: 'animals', icon: '🐦', subject: { th: 'นก', en: 'Bird', cn: '鸟' }, positiveClaim: { th: 'สัตว์ชนิดนี้มีปีกและขนสำหรับบินใช่ไหม?', en: 'Does this animal have wings and feathers to fly?', cn: '这种动物有翅膀和羽毛可以飞翔吗？' }, negativeClaim: { th: 'สัตว์ชนิดนี้มี 8 ขาเหมือนแมงมุมใช่ไหม?', en: 'Does this have 8 legs like a spider?', cn: '它像蜘蛛一样有8条腿吗？' } },
  { category: 'animals', icon: '🐢', subject: { th: 'เต่า', en: 'Turtle', cn: '乌龟' }, positiveClaim: { th: 'สัตว์ชนิดนี้มีกระดองแข็งบนหลังใช่ไหม?', en: 'Does this animal carry a hard shell on its back?', cn: '这种动物背上有坚硬的壳吗？' }, negativeClaim: { th: 'สัตว์ชนิดนี้วิ่งเร็วกว่าเสือชีตาห์ใช่ไหม?', en: 'Can this animal run faster than a cheetah?', cn: '它跑得比猎豹还快吗？' } },
  { category: 'animals', icon: '🐝', subject: { th: 'ผึ้ง', en: 'Bee', cn: '蜜蜂' }, positiveClaim: { th: 'แมลงชนิดนี้ช่วยผสมเกสรและผลิตน้ำผึ้งใช่ไหม?', en: 'Does this insect pollinate flowers and make honey?', cn: '这种昆虫会授粉并酿造蜂蜜吗？' }, negativeClaim: { th: 'แมลงชนิดนี้ตัวใหญ่กว่ารถบัสใช่ไหม?', en: 'Is this insect larger than a bus?', cn: '这种昆虫比公交车还大吗？' } },
  // counting
  { category: 'counting', icon: '🍎🍎🍎', subject: { th: 'แอปเปิ้ล 3 ลูก', en: '3 Apples', cn: '3个苹果' }, positiveClaim: { th: 'ในภาพมีแอปเปิ้ลทั้งหมด 3 ลูกใช่ไหม?', en: 'Are there exactly 3 apples shown here?', cn: '图中有3个苹果吗？' }, negativeClaim: { th: 'ในภาพมีแอปเปิ้ลทั้งหมด 7 ลูกใช่ไหม?', en: 'Are there 7 apples shown here?', cn: '图中有7个苹果吗？' } },
  { category: 'counting', icon: '⭐⭐', subject: { th: 'ดาว 2 ดวง', en: '2 Stars', cn: '2颗星星' }, positiveClaim: { th: '2 + 2 เท่ากับ 4 ใช่ไหม?', en: 'Does 2 + 2 equal 4?', cn: '2 + 2 等于 4 吗？' }, negativeClaim: { th: '2 + 2 เท่ากับ 9 ใช่ไหม?', en: 'Does 2 + 2 equal 9?', cn: '2 + 2 等于 9 吗？' } },
  { category: 'counting', icon: '🖐️', subject: { th: 'นิ้วมือ 5 นิ้ว', en: '5 Fingers', cn: '5根手指' }, positiveClaim: { th: 'มือหนึ่งข้างของคนปกติมี 5 นิ้วใช่ไหม?', en: 'Does one human hand normally have 5 fingers?', cn: '人类一只手通常有5根手指吗？' }, negativeClaim: { th: 'มือหนึ่งข้างมี 12 นิ้วใช่ไหม?', en: 'Does one hand have 12 fingers?', cn: '一只手有12根手指吗？' } },
  { category: 'counting', icon: '🚗', subject: { th: 'รถยนต์ 4 ล้อ', en: '4-Wheel Car', cn: '4轮汽车' }, positiveClaim: { th: '10 ลบ 4 เท่ากับ 6 ใช่ไหม?', en: 'Is 10 minus 4 equal to 6?', cn: '10 减 4 等于 6 吗？' }, negativeClaim: { th: '10 ลบ 4 เท่ากับ 2 ใช่ไหม?', en: 'Is 10 minus 4 equal to 2?', cn: '10 减 4 等于 2 吗？' } },
  // safety_routine
  { category: 'safety_routine', icon: '💊', subject: { th: 'รับประทานยาตามแพทย์สั่ง', en: 'Medication Safety', cn: '遵医嘱服药' }, positiveClaim: { th: 'ควรรับประทานยาตามขนาดและเวลาที่แพทย์กำหนดใช่ไหม?', en: 'Should you take medicine on the prescribed schedule?', cn: '应该按医生规定的剂量和时间服药吗？' }, negativeClaim: { th: 'หากลืมกินยาให้กินเบิ้ล 5 เท่าในมื้อเดียวใช่ไหม?', en: 'If you miss a dose, should you take 5x at once?', cn: '如果漏服一次，应该一次吃5倍药量吗？' } },
  { category: 'safety_routine', icon: '🥿', subject: { th: 'รองเท้ากันลื่น', en: 'Non-Slip Footwear', cn: '防滑鞋' }, positiveClaim: { th: 'การสวมรองเท้าพื้นกันลื่นช่วยลดความเสี่ยงหกล้มใช่ไหม?', en: 'Does non-slip footwear reduce fall risk?', cn: '穿防滑鞋有助于降低跌倒风险吗？' }, negativeClaim: { th: 'พื้นห้องน้ำที่เปียกลื่นปลอดภัยสำหรับการวิ่งใช่ไหม?', en: 'Is a wet slippery bathroom floor safe for running?', cn: '湿滑的浴室地面适合跑步吗？' } },
  { category: 'safety_routine', icon: '🔥', subject: { th: 'ปิดเตาแก๊สหลังใช้งาน', en: 'Turn Off Stove', cn: '用后关闭燃气灶' }, positiveClaim: { th: 'หลังทำอาหารเสร็จต้องปิดวาล์วเตาแก๊สทุกครั้งใช่ไหม?', en: 'Should you always turn off the stove after cooking?', cn: '做完饭后每次都要关闭燃气灶吗？' }, negativeClaim: { th: 'ควรเปิดไฟแก๊สทิ้งไว้ก่อนเข้านอนทั้งคืนใช่ไหม?', en: 'Should you leave the stove burner on all night?', cn: '睡前应该整晚开着燃气灶吗？' } },
  { category: 'safety_routine', icon: '🧼', subject: { th: 'ล้างมือฟอกสบู่', en: 'Handwashing', cn: '肥皂洗手' }, positiveClaim: { th: 'การล้างมือด้วยสบู่ก่อนกินอาหารช่วยป้องกันเชื้อโรคใช่ไหม?', en: 'Does washing hands with soap prevent germs?', cn: '饭前用肥皂洗手能预防病菌吗？' }, negativeClaim: { th: 'การหยิบอาหารด้วยมือเปื้อนดินดีต่อสุขภาพใช่ไหม?', en: 'Is eating with muddy hands healthy?', cn: '用沾满泥土的手拿食物健康吗？' } },
  // color_shape_time
  { category: 'color_shape_time', icon: '🚦', subject: { th: 'สัญญาณไฟจราจร', en: 'Traffic Light', cn: '交通信号灯' }, positiveClaim: { th: 'ไฟสีแดงหมายถึงให้หยุดรถใช่ไหม?', en: 'Does a red traffic light mean STOP?', cn: '红灯表示停止通行吗？' }, negativeClaim: { th: 'ไฟสีแดงหมายถึงให้เร่งความเร็วสูงสุดใช่ไหม?', en: 'Does a red traffic light mean accelerate fast?', cn: '红灯表示加速冲刺吗？' } },
  { category: 'color_shape_time', icon: '🔺', subject: { th: 'รูปสามเหลี่ยม', en: 'Triangle', cn: '三角形' }, positiveClaim: { th: 'รูปสามเหลี่ยมมีมุมทั้งหมด 3 มุมใช่ไหม?', en: 'Does a triangle have 3 corners?', cn: '三角形有3个角吗？' }, negativeClaim: { th: 'รูปสามเหลี่ยมมีด้านโค้งกลมเหมือนวงกลมใช่ไหม?', en: 'Is a triangle completely round like a circle?', cn: '三角形像圆形一样没有角吗？' } },
  { category: 'color_shape_time', icon: '☀️', subject: { th: 'ดวงอาทิตย์ยามเช้า', en: 'Morning Sun', cn: '清晨太阳' }, positiveClaim: { th: 'ดวงอาทิตย์ขึ้นทางทิศตะวันออกในตอนเช้าใช่ไหม?', en: 'Does the sun rise in the east in the morning?', cn: '太阳早晨从东方升起吗？' }, negativeClaim: { th: 'เวลาเที่ยงคืนท้องฟ้าสว่างจ้าเหมือนเที่ยงวันใช่ไหม?', en: 'Is the midnight sky as bright as noon?', cn: '午夜的天空像正午一样明亮吗？' } },
  // position
  { category: 'position', icon: '⬆️', subject: { th: 'ทิศทางด้านบน', en: 'Upward Direction', cn: '向上方向' }, positiveClaim: { th: 'ลูกศรนี้ชี้ขึ้นไปด้านบนใช่ไหม?', en: 'Is this arrow pointing UP?', cn: '这个箭头是指向上的吗？' }, negativeClaim: { th: 'ลูกศรนี้ชี้ลงไปด้านล่างใช่ไหม?', en: 'Is this arrow pointing DOWN?', cn: '这个箭头是指向下的吗？' } },
  { category: 'position', icon: '➡️', subject: { th: 'ทิศทางด้านขวา', en: 'Right Direction', cn: '向右方向' }, positiveClaim: { th: 'ลูกศรนี้ชี้ไปทางด้านขวาใช่ไหม?', en: 'Is this arrow pointing RIGHT?', cn: '这个箭头是指向右方的吗？' }, negativeClaim: { th: 'ลูกศรนี้ชี้ไปทางด้านซ้ายใช่ไหม?', en: 'Is this arrow pointing LEFT?', cn: '这个箭头是指向左方的吗？' } },
  { category: 'position', icon: '🏠', subject: { th: 'หลังคาอยู่ด้านบนบ้าน', en: 'Roof Position', cn: '屋顶位置' }, positiveClaim: { th: 'หลังคาบ้านอยู่ด้านบนสุดของตัวบ้านใช่ไหม?', en: 'Is the roof located at the top of a house?', cn: '屋顶位于房子的最上方吗？' }, negativeClaim: { th: 'หลังคาบ้านอยู่ใต้ฐานรากใต้ดินใช่ไหม?', en: 'Is the roof buried underneath the foundation?', cn: '屋顶埋在地下地基下面吗？' } },
];

/**
 * Generates the deterministic 99-card trilingual Cognitive Rehab Bank across all 7 categories.
 */
export const COGNITIVE_CARD_BANK_99: readonly CognitiveCardQuestion[] = (() => {
  const cards: CognitiveCardQuestion[] = [];
  for (let i = 0; i < 99; i += 1) {
    const seed = SEEDS[i % SEEDS.length];
    const isYes = i % 2 === 0;
    const roundTag = Math.floor(i / SEEDS.length) + 1;
    cards.push({
      id: `card-${String(i + 1).padStart(2, '0')}`,
      category: seed.category,
      icon: seed.icon,
      question: isYes ? seed.positiveClaim : seed.negativeClaim,
      correctAnswer: isYes ? 'yes' : 'no',
      explanation: isYes
        ? {
            th: `ถูกต้อง! ${seed.subject.th} (ชุดที่ ${roundTag})`,
            en: `Correct! ${seed.subject.en} (Set ${roundTag})`,
            cn: `回答正确！${seed.subject.cn}（第${roundTag}组）`,
          }
        : {
            th: `คำตอบที่ถูกต้องคือ "ไม่ใช่" เพราะเป็น${seed.subject.th}`,
            en: `The correct answer is "NO" — this is ${seed.subject.en}.`,
            cn: `正确答案是“否”，这是${seed.subject.cn}。`,
          },
    });
  }
  return cards;
})();

export function sampleTenCognitiveCards(categories?: readonly CognitiveCategory[]): CognitiveCardQuestion[] {
  const pool =
    categories && categories.length > 0
      ? COGNITIVE_CARD_BANK_99.filter((c) => categories.includes(c.category))
      : [...COGNITIVE_CARD_BANK_99];
  const copy = [...pool];
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy.slice(0, 10);
}

