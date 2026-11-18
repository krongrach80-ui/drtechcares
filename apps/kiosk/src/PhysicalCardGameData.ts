import type { TrilingualEntry } from '@drtechcares/clinical-i18n';
import {
  buildCardIllustrationDataUri,
  type CardArtKey,
} from './PhysicalCardGameIllustrations';

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
  artKey: CardArtKey;
  icon: string;
  imageUrl: string;
  subject: TrilingualEntry;
  question: TrilingualEntry;
  correctAnswer: 'yes' | 'no';
  explanation: TrilingualEntry;
}

interface CardSeed {
  category: CognitiveCategory;
  artKey: CardArtKey;
  icon: string;
  subject: TrilingualEntry;
  question: TrilingualEntry;
  correctAnswer: 'yes' | 'no';
  explanation: TrilingualEntry;
}

const SEEDS: readonly CardSeed[] = [
  // ================= 001..020 FOOD & FRUIT =================
  {
    category: 'food_fruit',
    artKey: 'apple',
    icon: 'apple',
    subject: { th: 'แอปเปิลแดง', en: 'Red Apple', cn: '红苹果' },
    question: {
      th: 'ภาพนี้คือผลไม้ที่มีสีแดงใช่ไหม?',
      en: 'Is this a red fruit in the picture?',
      cn: '图中的是红色水果吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'แอปเปิลสีแดงเป็นผลไม้ที่มีใยอาหารและวิตามิน',
      en: 'Yes, this is a red apple rich in fiber and vitamins.',
      cn: '是的，红苹果是富含膳食纤维和维生素的水果。',
    },
  },
  {
    category: 'food_fruit',
    artKey: 'banana',
    icon: 'banana',
    subject: { th: 'กล้วยสีเหลือง', en: 'Yellow Banana', cn: '黄香蕉' },
    question: {
      th: 'กล้วยในภาพนี้มีสีม่วงใช่ไหม?',
      en: 'Is the banana in this picture purple?',
      cn: '图中的香蕉是紫色的吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ กล้วยสุกในภาพมีสีเหลือง ไม่ใช่สีม่วง',
      en: 'No, the banana in this picture is yellow, not purple.',
      cn: '不是，图中的香蕉是黄色的，不是紫色的。',
    },
  },
  {
    category: 'food_fruit',
    artKey: 'broccoli',
    icon: 'broccoli',
    subject: { th: 'บรอกโคลี', en: 'Broccoli', cn: '西兰花' },
    question: {
      th: 'ภาพนี้คือผักสีเขียวที่มีประโยชน์ใช่ไหม?',
      en: 'Is this a healthy green vegetable?',
      cn: '这是健康的绿色蔬菜吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ บรอกโคลีเป็นผักสีเขียวที่ช่วยเสริมสร้างร่างกาย',
      en: 'Yes, broccoli is a healthy green vegetable.',
      cn: '是的，西兰花是有益健康的绿色蔬菜。',
    },
  },
  {
    category: 'food_fruit',
    artKey: 'milk',
    icon: 'milk',
    subject: { th: 'นมสด', en: 'Fresh Milk', cn: '鲜牛奶' },
    question: {
      th: 'นมในภาพนี้เป็นอาหารทอดกรอบใช่ไหม?',
      en: 'Is the milk in this picture a deep-fried snack?',
      cn: '图中的牛奶是油炸食品吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ นมสดเป็นเครื่องดื่มที่มีแคลเซียมช่วยบำรุงกระดูก',
      en: 'No, milk is a calcium-rich drink, not a fried snack.',
      cn: '不是，牛奶是富含钙质的饮品，不是油炸食品。',
    },
  },
  {
    category: 'food_fruit',
    artKey: 'watermelon',
    icon: 'watermelon',
    subject: { th: 'แตงโม', en: 'Watermelon', cn: '西瓜' },
    question: {
      th: 'ผลไม้ในภาพนี้มีเนื้อสีแดงและเมล็ดสีดำใช่ไหม?',
      en: 'Does this fruit have red flesh and black seeds?',
      cn: '这种水果有红瓤和黑籽吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ แตงโมมีเนื้อสีแดงฉ่ำน้ำ เปลือกเขียว และเมล็ดสีดำ',
      en: 'Yes, watermelon has sweet red flesh, green rind, and black seeds.',
      cn: '是的，西瓜有红色多汁的果肉和黑色瓜籽。',
    },
  },
  {
    category: 'food_fruit',
    artKey: 'carrot',
    icon: 'carrot',
    subject: { th: 'แครอท', en: 'Carrot', cn: '胡萝卜' },
    question: {
      th: 'แครอทในภาพนี้มีสีฟ้าใช่ไหม?',
      en: 'Is the carrot in this picture blue?',
      cn: '图中的胡萝卜是蓝色的吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ แครอทในภาพมีหัวสีส้มและใบสีเขียว',
      en: 'No, the carrot is orange with green leaves.',
      cn: '不是，图中的胡萝卜是橙色带绿叶的。',
    },
  },
  {
    category: 'food_fruit',
    artKey: 'fish_food',
    icon: 'fish_food',
    subject: { th: 'ปลาสด', en: 'Fresh Fish', cn: '鲜鱼' },
    question: {
      th: 'ปลาในภาพนี้เป็นอาหารที่ให้โปรตีนใช่ไหม?',
      en: 'Does the fish in this picture provide protein?',
      cn: '图中的鱼是提供蛋白质的食物吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ เนื้อปลาเป็นโปรตีนที่ย่อยง่ายและดีต่อสุขภาพ',
      en: 'Yes, fish is an easy-to-digest source of healthy protein.',
      cn: '是的，鱼肉是易消化的优质蛋白质来源。',
    },
  },
  {
    category: 'food_fruit',
    artKey: 'mango',
    icon: 'mango',
    subject: { th: 'มะม่วงสุก', en: 'Ripe Mango', cn: '芒果' },
    question: {
      th: 'ผลไม้สีเหลืองในภาพนี้ใช้สวมเท้าแทนรองเท้าได้ใช่ไหม?',
      en: 'Can you wear this yellow fruit on your feet like shoes?',
      cn: '图中的黄色水果可以当鞋子穿在脚上吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ มะม่วงเป็นผลไม้รับประทาน ไม่ใช่รองเท้า',
      en: 'No, a mango is a fruit to eat, not a shoe.',
      cn: '不是，芒果是用来吃的水果，不是鞋子。',
    },
  },
  {
    category: 'food_fruit',
    artKey: 'orange',
    icon: 'orange',
    subject: { th: 'ส้ม', en: 'Orange', cn: '橙子' },
    question: {
      th: 'ผลไม้สีส้มลูกนี้เป็นแหล่งวิตามินซีใช่ไหม?',
      en: 'Is this orange fruit a source of vitamin C?',
      cn: '这个橙子是维生素C的来源吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ ส้มอุดมไปด้วยวิตามินซี ช่วยเสริมภูมิคุ้มกัน',
      en: 'Yes, oranges are rich in vitamin C to support immunity.',
      cn: '是的，橙子富含维生素C，有助于增强免疫力。',
    },
  },
  {
    category: 'food_fruit',
    artKey: 'fried_egg',
    icon: 'fried_egg',
    subject: { th: 'ไข่ดาว', en: 'Fried Egg', cn: '煎蛋' },
    question: {
      th: 'ไข่ดาวในกระทะนี้เป็นขนมเค้กวันเกิดใช่ไหม?',
      en: 'Is the fried egg in this pan a birthday cake?',
      cn: '锅里的煎蛋是生日蛋糕吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ ภาพนี้คือไข่ดาวในกระทะ ให้สารอาหารโปรตีน',
      en: 'No, this is a fried egg in a pan, not a birthday cake.',
      cn: '不是，这是锅里的煎蛋，不是生日蛋糕。',
    },
  },
  {
    category: 'food_fruit',
    artKey: 'rice_bowl',
    icon: 'rice_bowl',
    subject: { th: 'ข้าวสวยในชาม', en: 'Bowl of Rice', cn: '一碗米饭' },
    question: {
      th: 'ชามสีฟ้านี้บรรจุข้าวสวยสำหรับรับประทานใช่ไหม?',
      en: 'Does this blue bowl contain rice for eating?',
      cn: '这个蓝碗里装的是食用的米饭吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ ข้าวสวยในชามให้พลังงานแก่ร่างกายในมื้ออาหาร',
      en: 'Yes, cooked rice in a bowl provides daily energy.',
      cn: '是的，碗里的米饭能为身体提供能量。',
    },
  },
  {
    category: 'food_fruit',
    artKey: 'papaya',
    icon: 'papaya',
    subject: { th: 'มะละกอสุก', en: 'Ripe Papaya', cn: '木瓜' },
    question: {
      th: 'มะละกอในภาพนี้เป็นเครื่องดนตรีใช่ไหม?',
      en: 'Is the papaya in this picture a musical instrument?',
      cn: '图中的木瓜是一种乐器吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ มะละกอสุกเนื้อสีส้มเป็นผลไม้ที่ช่วยเรื่องขับถ่าย',
      en: 'No, papaya is a nutritious fruit that helps digestion.',
      cn: '不是，木瓜是帮助消化的水果，不是乐器。',
    },
  },
  {
    category: 'food_fruit',
    artKey: 'pineapple',
    icon: 'pineapple',
    subject: { th: 'สับปะรด', en: 'Pineapple', cn: '菠萝' },
    question: {
      th: 'สับปะรดในภาพมีจุกใบสีเขียวอยู่ด้านบนใช่ไหม?',
      en: 'Does the pineapple in the picture have green leaves on top?',
      cn: '图中的菠萝顶部有绿色叶子吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ สับปะรดมีเปลือกตาสีเหลืองส้มและจุกใบสีเขียวด้านบน',
      en: 'Yes, pineapples have a crown of green leaves on top.',
      cn: '是的，菠萝顶部有一簇绿叶。',
    },
  },
  {
    category: 'food_fruit',
    artKey: 'coconut',
    icon: 'coconut',
    subject: { th: 'มะพร้าวน้ำหอม', en: 'Fresh Coconut', cn: '青椰子' },
    question: {
      th: 'น้ำในลูกมะพร้าวนี้เป็นน้ำมันเครื่องใช่ไหม?',
      en: 'Is the liquid inside this fresh coconut motor oil?',
      cn: '这颗椰子里面装的是机油吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ ข้างในเป็นน้ำมะพร้าวใสสะอาด ดื่มแล้วสดชื่น',
      en: 'No, it holds refreshing natural coconut water, not motor oil.',
      cn: '不是，里面是清甜解渴的天然椰子水。',
    },
  },
  {
    category: 'food_fruit',
    artKey: 'corn',
    icon: 'corn',
    subject: { th: 'ข้าวโพด', en: 'Sweet Corn', cn: '玉米' },
    question: {
      th: 'ข้าวโพดในภาพมีเมล็ดสีเหลืองห่อด้วยเปลือกสีเขียวใช่ไหม?',
      en: 'Does this corn have yellow kernels wrapped in green husks?',
      cn: '图中的玉米有黄色的玉米粒和绿色的外皮吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ ข้าวโพดมีฝักเมล็ดสีเหลืองและกาบหุ้มสีเขียว',
      en: 'Yes, sweet corn has yellow kernels with green husks.',
      cn: '是的，玉米有黄色玉米粒和绿色苞叶。',
    },
  },
  {
    category: 'food_fruit',
    artKey: 'strawberry',
    icon: 'strawberry',
    subject: { th: 'สตรอว์เบอร์รี', en: 'Strawberry', cn: '草莓' },
    question: {
      th: 'ผลไม้สีแดงลูกเล็กนี้คือลูกมะพร้าวเปลือกแข็งใช่ไหม?',
      en: 'Is this red berry a hard-shelled coconut?',
      cn: '这颗红色浆果是硬壳椰子吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ ภาพนี้คือสตรอว์เบอร์รีผลสีแดง ไม่ใช่มะพร้าว',
      en: 'No, this picture shows a red strawberry, not a coconut.',
      cn: '不是，这是红草莓，不是椰子。',
    },
  },
  {
    category: 'food_fruit',
    artKey: 'grapes',
    icon: 'grapes',
    subject: { th: 'องุ่นม่วง', en: 'Purple Grapes', cn: '紫葡萄' },
    question: {
      th: 'พวงผลไม้สีม่วงในภาพนี้คือองุ่นใช่ไหม?',
      en: 'Is this bunch of purple fruit grapes?',
      cn: '图中这一串紫色的水果是葡萄吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ องุ่นออกผลเป็นพวง มีรสหวานอร่อยและวิตามิน',
      en: 'Yes, grapes grow in bunches and contain antioxidants.',
      cn: '是的，葡萄成串生长，味道甘甜。',
    },
  },
  {
    category: 'food_fruit',
    artKey: 'pumpkin',
    icon: 'pumpkin',
    subject: { th: 'ฟักทอง', en: 'Pumpkin', cn: '南瓜' },
    question: {
      th: 'ฟักทองลูกสีส้มนี้ใช้อาบน้ำแทนสบู่ใช่ไหม?',
      en: 'Do we use this orange pumpkin to shower like soap?',
      cn: '这个橙色南瓜是用来代替肥皂洗澡的吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ ฟักทองใช้ทำอาหารคาวหวานที่มีประโยชน์',
      en: 'No, pumpkins are nutritious food ingredients, not soap.',
      cn: '不是，南瓜是营养丰富的食材，不是肥皂。',
    },
  },
  {
    category: 'food_fruit',
    artKey: 'soup_bowl',
    icon: 'soup_bowl',
    subject: { th: 'ชามซุปร้อน', en: 'Hot Soup Bowl', cn: '热汤碗' },
    question: {
      th: 'ภาพนี้แสดงชามซุปร้อนที่มีไอน้ำลอยขึ้นใช่ไหม?',
      en: 'Does this picture show a bowl of hot soup with steam?',
      cn: '图中的是冒着热气的汤碗吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ ซุปอุ่นๆ ช่วยให้กลืนง่ายและอบอุ่นร่างกาย',
      en: 'Yes, warm soup is easy to swallow and comforting.',
      cn: '是的，热汤有助于吞咽和暖胃。',
    },
  },
  {
    category: 'food_fruit',
    artKey: 'water_glass',
    icon: 'water_glass',
    subject: { th: 'แก้วน้ำดื่ม', en: 'Glass of Water', cn: '一杯水' },
    question: {
      th: 'เราควรเทน้ำดื่มทิ้งทั้งหมดเพื่อไม่ให้ร่างกายได้น้ำใช่ไหม?',
      en: 'Should we throw away drinking water so our body gets dehydrated?',
      cn: '我们应该把水倒掉让身体缺水吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ ร่างกายต้องการดื่มน้ำสะอาดให้เพียงพอทุกวัน',
      en: 'No, our body needs clean drinking water every day.',
      cn: '不是，身体每天都需要喝足干净的水。',
    },
  },

  // ================= 021..040 OBJECTS =================
  {
    category: 'objects',
    artKey: 'key',
    icon: 'key',
    subject: { th: 'กุญแจ', en: 'Key', cn: '钥匙' },
    question: {
      th: 'กุญแจสีทองในภาพนี้ใช้ไขเปิดประตูใช่ไหม?',
      en: 'Is this golden key used to unlock a door?',
      cn: '这把金钥匙是用来开门锁的吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ กุญแจใช้สำหรับล็อกและไขเปิดประตูบ้านหรือตู้',
      en: 'Yes, keys are used to lock and unlock doors.',
      cn: '是的，钥匙用来开锁和锁门。',
    },
  },
  {
    category: 'objects',
    artKey: 'toothbrush',
    icon: 'toothbrush',
    subject: { th: 'แปรงสีฟัน', en: 'Toothbrush', cn: '牙刷' },
    question: {
      th: 'แปรงสีฟันสีชมพูนี้ใช้กวาดลานบ้านใช่ไหม?',
      en: 'Is this pink toothbrush used to sweep the yard?',
      cn: '这把粉红色牙刷是用来扫院子的吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ แปรงสีฟันใช้แปรงทำความสะอาดฟันในช่องปาก',
      en: 'No, a toothbrush is for brushing teeth, not sweeping.',
      cn: '不是，牙刷是用来刷牙的，不是扫院子的。',
    },
  },
  {
    category: 'objects',
    artKey: 'eyeglasses',
    icon: 'eyeglasses',
    subject: { th: 'แว่นตา', en: 'Eyeglasses', cn: '眼镜' },
    question: {
      th: 'แว่นตาในภาพนี้ช่วยให้มองเห็นตัวหนังสือได้ชัดเจนขึ้นใช่ไหม?',
      en: 'Do eyeglasses help you see text more clearly?',
      cn: '眼镜能帮助更清晰地看到文字吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ แว่นสายตาช่วยปรับการมองเห็นให้อ่านหนังสือได้ชัดขึ้น',
      en: 'Yes, eyeglasses help correct vision for reading clearly.',
      cn: '是的，眼镜可以矫正视力，让文字更清楚。',
    },
  },
  {
    category: 'objects',
    artKey: 'umbrella',
    icon: 'umbrella',
    subject: { th: 'ร่มสีฟ้า', en: 'Blue Umbrella', cn: '蓝雨伞' },
    question: {
      th: 'ร่มสีฟ้าในภาพนี้ใช้ตักแกงรับประทานใช่ไหม?',
      en: 'Is this blue umbrella used as a spoon to eat soup?',
      cn: '这把蓝雨伞是用来舀汤喝的吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ ร่มใช้กางกันแดดและกันฝน',
      en: 'No, an umbrella protects against rain and sun.',
      cn: '不是，雨伞是用来遮阳挡雨的。',
    },
  },
  {
    category: 'objects',
    artKey: 'alarm_clock',
    icon: 'alarm_clock',
    subject: { th: 'นาฬิกาปลุก', en: 'Alarm Clock', cn: '闹钟' },
    question: {
      th: 'นาฬิกาปลุกกระดิ่งสีแดงนี้ใช้ดูเวลาและปลุกตอนเช้าใช่ไหม?',
      en: 'Is this alarm clock used to tell time and wake up?',
      cn: '这个红铃闹钟是用来看时间和早上叫醒的吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ นาฬิกาปลุกช่วยบอกเวลาและเตือนให้ตื่นตรงเวลา',
      en: 'Yes, an alarm clock shows time and wakes us up.',
      cn: '是的，闹钟可以显示时间并提醒起床。',
    },
  },
  {
    category: 'objects',
    artKey: 'chair',
    icon: 'chair',
    subject: { th: 'เก้าอี้ไม้', en: 'Wooden Chair', cn: '木椅' },
    question: {
      th: 'เก้าอี้ไม้ในภาพนี้มีไว้สวมบนศีรษะแทนหมวกใช่ไหม?',
      en: 'Is this wooden chair meant to be worn on your head like a hat?',
      cn: '这把木椅是像帽子一样戴在头上的吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ เก้าอี้มีไว้สำหรับนั่งพักผ่อนอย่างมั่นคง',
      en: 'No, a chair is for sitting comfortably, not wearing on your head.',
      cn: '不是，椅子是用来坐的，不是戴在头上的。',
    },
  },
  {
    category: 'objects',
    artKey: 'stethoscope',
    icon: 'stethoscope',
    subject: { th: 'หูฟังแพทย์', en: 'Stethoscope', cn: '听诊器' },
    question: {
      th: 'อุปกรณ์ในภาพนี้คือหูฟังของคุณหมอใช่ไหม?',
      en: 'Is this device a doctor’s stethoscope?',
      cn: '图中的仪器是医生的听诊器吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ หูฟังแพทย์ใช้ฟังเสียงหัวใจและปอดในการตรวจร่างกาย',
      en: 'Yes, clinicians use stethoscopes to listen to heart and lung sounds.',
      cn: '是的，医生用听诊器听心肺声音。',
    },
  },
  {
    category: 'objects',
    artKey: 'spoon_fork',
    icon: 'spoon_fork',
    subject: { th: 'ช้อนและส้อม', en: 'Spoon and Fork', cn: '勺子和叉子' },
    question: {
      th: 'ช้อนและส้อมคู่นี้ใช้ตัดกระดาษแทนกรรไกรใช่ไหม?',
      en: 'Are this spoon and fork used to cut paper like scissors?',
      cn: '这对勺子和叉子是像剪刀一样用来剪纸的吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ ช้อนและส้อมเป็นอุปกรณ์ตักอาหารรับประทาน',
      en: 'No, a spoon and fork are utensils for eating meals.',
      cn: '不是，勺子和叉子是用餐的餐具。',
    },
  },
  {
    category: 'objects',
    artKey: 'hair_comb',
    icon: 'hair_comb',
    subject: { th: 'หวีผม', en: 'Hair Comb', cn: '梳子' },
    question: {
      th: 'หวีสีม่วงในภาพนี้ใช้หวีจัดทรงผมใช่ไหม?',
      en: 'Is this purple comb used to comb hair?',
      cn: '这把紫色梳子是用来梳头发的吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ หวีมีซี่เรียงกันสำหรับสางและจัดแต่งเส้นผม',
      en: 'Yes, a comb is used to untangle and style hair.',
      cn: '是的，梳子用来梳理头发。',
    },
  },
  {
    category: 'objects',
    artKey: 'scissors',
    icon: 'scissors',
    subject: { th: 'กรรไกร', en: 'Scissors', cn: '剪刀' },
    question: {
      th: 'กรรไกรด้ามสีแดงนี้ใช้ชงกาแฟใช่ไหม?',
      en: 'Are these red-handled scissors used to stir coffee?',
      cn: '这把红柄剪刀是用来搅拌咖啡的吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ กรรไกรมีคมมีดสำหรับตัดกระดาษหรือผ้า',
      en: 'No, scissors are used to cut paper or cloth, not stir coffee.',
      cn: '不是，剪刀用来裁剪纸张或布料，不是搅拌咖啡的。',
    },
  },
  {
    category: 'objects',
    artKey: 'pillow',
    icon: 'pillow',
    subject: { th: 'หมอนหนุน', en: 'Pillow', cn: '枕头' },
    question: {
      th: 'หมอนสีขาวใบนี้อยู่คู่กับการนอนหลับพักผ่อนใช่ไหม?',
      en: 'Is this white pillow used for sleeping and resting?',
      cn: '这个白色枕头是用来睡觉休息的吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ หมอนนุ่มๆ ช่วยรองรับศีรษะและคอเวลานอนหลับ',
      en: 'Yes, a pillow supports your head and neck while sleeping.',
      cn: '是的，枕头可以在睡觉时支撑头部和颈部。',
    },
  },
  {
    category: 'objects',
    artKey: 'walking_shoes',
    icon: 'walking_shoes',
    subject: { th: 'รองเท้าเดิน', en: 'Walking Shoes', cn: '步行鞋' },
    question: {
      th: 'รองเท้าสีฟ้าในภาพนี้ใช้ถือเขียนหนังสือแทนปากกาใช่ไหม?',
      en: 'Are these blue shoes used for writing like a pen?',
      cn: '图中的蓝鞋是像钢笔一样用来写字的吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ รองเท้าใช้สวมที่เท้าเพื่อเดินอย่างปลอดภัย',
      en: 'No, shoes are worn on the feet for walking, not writing.',
      cn: '不是，鞋子是穿在脚上走路的，不是写字的。',
    },
  },
  {
    category: 'objects',
    artKey: 'telephone',
    icon: 'telephone',
    subject: { th: 'โทรศัพท์', en: 'Telephone', cn: '电话' },
    question: {
      th: 'โทรศัพท์ในภาพนี้ใช้โทรติดต่อพูดคุยกับผู้อื่นใช่ไหม?',
      en: 'Is this telephone used to call and talk to others?',
      cn: '图中的电话是用来拨打并与他人通话的吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ โทรศัพท์ช่วยให้สื่อสารกับครอบครัวหรือขอความช่วยเหลือได้',
      en: 'Yes, a telephone lets us call family or seek help.',
      cn: '是的，电话方便联系家人或求助。',
    },
  },
  {
    category: 'objects',
    artKey: 'open_book',
    icon: 'open_book',
    subject: { th: 'หนังสือเปิด', en: 'Open Book', cn: '打开的书' },
    question: {
      th: 'หนังสือที่เปิดอยู่นี้ใช้ตักน้ำดื่มแทนแก้วใช่ไหม?',
      en: 'Is this open book used to hold drinking water like a cup?',
      cn: '这本打开的书是像水杯一样用来盛水的吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ หนังสือมีหน้ากระดาษไว้สำหรับอ่านและจดบันทึก',
      en: 'No, a book is for reading and writing, not holding water.',
      cn: '不是，书是用来阅读和记笔记的，不能盛水。',
    },
  },
  {
    category: 'objects',
    artKey: 'teacup',
    icon: 'teacup',
    subject: { th: 'ถ้วยชาร้อน', en: 'Hot Teacup', cn: '热茶杯' },
    question: {
      th: 'ถ้วยที่มีจานรองและไอน้ำนี้ใช้ใส่ชาอุ่นๆ ใช่ไหม?',
      en: 'Is this cup with a saucer and steam used for hot tea?',
      cn: '这个带托盘且冒热气的杯子是用来泡热茶的吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ ถ้วยมีหูจับและจานรองช่วยให้ถือเครื่องดื่มร้อนได้ถนัด',
      en: 'Yes, a teacup and saucer hold warm tea comfortably.',
      cn: '是的，茶杯和茶托方便端取热饮。',
    },
  },
  {
    category: 'objects',
    artKey: 'wristwatch',
    icon: 'wristwatch',
    subject: { th: 'นาฬิกาข้อมือ', en: 'Wristwatch', cn: '手表' },
    question: {
      th: 'นาฬิกาข้อมือเรือนนี้ใช้ทอดไข่ดาวบนเตาใช่ไหม?',
      en: 'Is this wristwatch used to fry eggs on a stove?',
      cn: '这块手表是放在炉子上煎鸡蛋用的吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ นาฬิกาข้อมือใช้สวมใส่เพื่อดูเวลา',
      en: 'No, a wristwatch is worn on the wrist to tell time.',
      cn: '不是，手表是戴在手腕上看时间的。',
    },
  },
  {
    category: 'objects',
    artKey: 'flashlight',
    icon: 'flashlight',
    subject: { th: 'ไฟฉาย', en: 'Flashlight', cn: '手电筒' },
    question: {
      th: 'ไฟฉายสีแดงในภาพนี้ส่องแสงสว่างในที่มืดได้ใช่ไหม?',
      en: 'Does this red flashlight shine light in dark places?',
      cn: '这支红色手电筒能在黑暗中照明吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ ไฟฉายให้แสงสว่างช่วยให้มองเห็นทางเดินเวลาไฟดับ',
      en: 'Yes, a flashlight shines bright light in the dark.',
      cn: '是的，手电筒可以在黑暗或停电时照明。',
    },
  },
  {
    category: 'objects',
    artKey: 'table_fan',
    icon: 'table_fan',
    subject: { th: 'พัดลมตั้งโต๊ะ', en: 'Table Fan', cn: '台扇' },
    question: {
      th: 'พัดลมสีฟ้าในภาพนี้มีหน้าที่ต้มน้ำให้เดือดใช่ไหม?',
      en: 'Is this blue table fan meant to boil water?',
      cn: '图中的蓝色台扇是用来烧开水的吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ พัดลมหมุนใบพัดเพื่อให้ลมเย็นสบาย',
      en: 'No, a fan circulates cool air, not boiling water.',
      cn: '不是，风扇是用来吹风降温的，不能烧水。',
    },
  },
  {
    category: 'objects',
    artKey: 'sun_hat',
    icon: 'sun_hat',
    subject: { th: 'หมวกกันแดด', en: 'Sun Hat', cn: '遮阳帽' },
    question: {
      th: 'หมวกสีเหลืองปีกกว้างนี้ช่วยบังแดดเวลาออกนอกบ้านใช่ไหม?',
      en: 'Does this wide-brimmed yellow hat shade you from the sun?',
      cn: '这顶宽边黄帽子在户外能遮阳吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ หมวกปีกกว้างช่วยบังแสงแดดและปกป้องใบหน้า',
      en: 'Yes, a sun hat helps shield your face and head from bright sunlight.',
      cn: '是的，遮阳帽可以帮助挡住刺眼的阳光。',
    },
  },
  {
    category: 'objects',
    artKey: 'thermometer',
    icon: 'thermometer',
    subject: { th: 'ปรอทวัดไข้', en: 'Thermometer', cn: '体温计' },
    question: {
      th: 'ปรอทวัดอุณหภูมิในภาพนี้ใช้ตอกตะปูแทนค้อนใช่ไหม?',
      en: 'Is this thermometer used to hammer nails?',
      cn: '图中的体温计是像锤子一样用来钉钉子的吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ ปรอทวัดอุณหภูมิใช้วัดไข้หรืออุณหภูมิ',
      en: 'No, a thermometer measures temperature, not hammering nails.',
      cn: '不是，体温计用来测量体温，不是敲钉子的。',
    },
  },

  // ================= 041..055 ANIMALS =================
  {
    category: 'animals',
    artKey: 'elephant',
    icon: 'elephant',
    subject: { th: 'ช้าง', en: 'Elephant', cn: '大象' },
    question: {
      th: 'ช้างในภาพนี้มีงวงยาวและงาสีขาวใช่ไหม?',
      en: 'Does the elephant in this picture have a long trunk and tusks?',
      cn: '图中的大象有长鼻子和象牙吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ ช้างเป็นสัตว์บกตัวใหญ่ที่มีงวงยาวและงา',
      en: 'Yes, elephants have a long trunk, large ears, and tusks.',
      cn: '是的，大象有长鼻子、大耳朵和象牙。',
    },
  },
  {
    category: 'animals',
    artKey: 'dog',
    icon: 'dog',
    subject: { th: 'สุนัข', en: 'Dog', cn: '小狗' },
    question: {
      th: 'สุนัขในภาพนี้มีปีกบินอยู่บนก้อนเมฆใช่ไหม?',
      en: 'Does the dog in this picture have wings flying in the clouds?',
      cn: '图中的小狗长着翅膀在云端飞吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ สุนัขเป็นสัตว์สี่ขาเดินบนพื้นดิน ไม่มีปีก',
      en: 'No, dogs walk on four legs on the ground and do not have wings.',
      cn: '不是，小狗在地上用四条腿走路，没有翅膀。',
    },
  },
  {
    category: 'animals',
    artKey: 'bird',
    icon: 'bird',
    subject: { th: 'นกสีฟ้า', en: 'Blue Bird', cn: '蓝鸟' },
    question: {
      th: 'นกสีฟ้าตัวนี้มีจะงอยปากและปีกใช่ไหม?',
      en: 'Does this blue bird have a beak and wings?',
      cn: '这只蓝鸟有喙和翅膀吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ นกมีจะงอยปาก ปีก 2 ข้าง และขา 2 ข้าง',
      en: 'Yes, birds have a beak, wings, and two legs.',
      cn: '是的，鸟类有喙、翅膀和两条腿。',
    },
  },
  {
    category: 'animals',
    artKey: 'turtle',
    icon: 'turtle',
    subject: { th: 'เต่า', en: 'Turtle', cn: '乌龟' },
    question: {
      th: 'เต่ากระดองเขียวตัวนี้วิ่งเร็วกว่าเสือชีตาห์ใช่ไหม?',
      en: 'Can this green-shelled turtle run faster than a cheetah?',
      cn: '这只绿壳乌龟跑得比猎豹还快吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ เต่ามีกระดองแข็งและเดินคลานช้าๆ',
      en: 'No, turtles carry a hard shell and move slowly.',
      cn: '不是，乌龟背着硬壳，行动缓慢。',
    },
  },
  {
    category: 'animals',
    artKey: 'bee',
    icon: 'bee',
    subject: { th: 'ผึ้ง', en: 'Honey Bee', cn: '蜜蜂' },
    question: {
      th: 'ผึ้งลายเหลืองดำตัวนี้มีปีกช่วยบินหาน้ำหวานใช่ไหม?',
      en: 'Does this yellow-and-black bee have wings to fly for nectar?',
      cn: '这只黄黑相间的蜜蜂有翅膀飞去采蜜吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ ผึ้งบินเก็บน้ำหวานจากดอกไม้มาทำน้ำผึ้ง',
      en: 'Yes, bees use their wings to collect flower nectar.',
      cn: '是的，蜜蜂用翅膀飞去采集花蜜。',
    },
  },
  {
    category: 'animals',
    artKey: 'cat',
    icon: 'cat',
    subject: { th: 'แมวส้ม', en: 'Orange Cat', cn: '橘猫' },
    question: {
      th: 'แมวสีส้มในภาพนี้มีกระดองแข็งเหมือนเต่าใช่ไหม?',
      en: 'Does this orange cat have a hard shell like a turtle?',
      cn: '图中的橘猫像乌龟一样有硬壳吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ แมวมีขนนุ่มและหนวดยาว ไม่มีกระดองแข็ง',
      en: 'No, a cat has soft fur and whiskers, not a hard shell.',
      cn: '不是，猫有柔软的毛发和胡须，没有硬壳。',
    },
  },
  {
    category: 'animals',
    artKey: 'rabbit',
    icon: 'rabbit',
    subject: { th: 'กระต่ายขาว', en: 'White Rabbit', cn: '白兔' },
    question: {
      th: 'กระต่ายสีขาวในภาพนี้มีหูยาว 2 ข้างใช่ไหม?',
      en: 'Does this white rabbit have two long ears?',
      cn: '图中的白兔有两只长耳朵吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ กระต่ายมีหูยาวช่วยรับเสียงและหางปุยสั้น',
      en: 'Yes, rabbits are known for their two long ears.',
      cn: '是的，兔子以两只长耳朵闻名。',
    },
  },
  {
    category: 'animals',
    artKey: 'tropical_fish',
    icon: 'tropical_fish',
    subject: { th: 'ปลาทะเลสีส้ม', en: 'Orange Clownfish', cn: '小丑鱼' },
    question: {
      th: 'ปลาสีส้มขาวตัวนี้เดินสี่ขาบนต้นไม้สูงใช่ไหม?',
      en: 'Does this orange-and-white fish walk on tall trees?',
      cn: '这条橙白条纹的鱼在高树上行走吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ ปลาอาศัยและว่ายน้ำอยู่ในน้ำ',
      en: 'No, fish swim in water using fins and tails.',
      cn: '不是，鱼用鱼鳍和尾巴在水里游动。',
    },
  },
  {
    category: 'animals',
    artKey: 'butterfly',
    icon: 'butterfly',
    subject: { th: 'ผีเสื้อ', en: 'Butterfly', cn: '蝴蝶' },
    question: {
      th: 'ผีเสื้อตัวนี้มีปีกสีม่วงชมพูสวยงาม 2 ข้างใช่ไหม?',
      en: 'Does this butterfly have colorful purple-pink wings?',
      cn: '这只蝴蝶有漂亮的紫粉色翅膀吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ ผีเสื้อมีปีกคู่สีสันสดใสและหนวดเรียวยาว',
      en: 'Yes, butterflies have colorful paired wings.',
      cn: '是的，蝴蝶拥有对称的彩色翅膀。',
    },
  },
  {
    category: 'animals',
    artKey: 'rooster',
    icon: 'rooster',
    subject: { th: 'ไก่ตัวผู้', en: 'Rooster', cn: '公鸡' },
    question: {
      th: 'ไก่หงอนแดงตัวนี้ดำน้ำลึกกลางมหาสมุทรใช่ไหม?',
      en: 'Does this red-combed rooster live deep underwater in the ocean?',
      cn: '这只红冠公鸡生活在深海海底吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ ไก่เป็นสัตว์ปีกบนบก มักขันในตอนเช้า',
      en: 'No, roosters live on land and crow in the morning.',
      cn: '不是，公鸡生活在陆地上，早晨会打鸣。',
    },
  },
  {
    category: 'animals',
    artKey: 'cow',
    icon: 'cow',
    subject: { th: 'วัวลายขาวดำ', en: 'Cow', cn: '奶牛' },
    question: {
      th: 'วัวลายขาวดำในภาพนี้เป็นสัตว์ที่ให้นมสดใช่ไหม?',
      en: 'Is this black-and-white cow an animal that gives milk?',
      cn: '这头黑白相间的牛是产奶的动物吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ แม่วัวให้น้ำนมที่มีประโยชน์ต่อคนเรา',
      en: 'Yes, dairy cows produce nutritious milk.',
      cn: '是的，奶牛为我们提供营养丰富的牛奶。',
    },
  },
  {
    category: 'animals',
    artKey: 'duck',
    icon: 'duck',
    subject: { th: 'เป็ดเหลือง', en: 'Yellow Duck', cn: '黄鸭' },
    question: {
      th: 'เป็ดสีเหลืองที่ลอยน้ำนี้มีขาสี่ขาเหมือนม้าใช่ไหม?',
      en: 'Does this swimming yellow duck have four legs like a horse?',
      cn: '这只游泳的小黄鸭像马一样有四条腿吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ เป็ดเป็นสัตว์ปีกที่มี 2 ขาและพังผืดว่ายน้ำ',
      en: 'No, ducks are birds with two webbed legs, not four legs.',
      cn: '不是，鸭子只有两条带蹼的腿。',
    },
  },
  {
    category: 'animals',
    artKey: 'horse',
    icon: 'horse',
    subject: { th: 'ม้าสีน้ำตาล', en: 'Brown Horse', cn: '棕马' },
    question: {
      th: 'ม้าสีน้ำตาลในภาพนี้มีแผงคอและหางยาวใช่ไหม?',
      en: 'Does this brown horse have a mane and a long tail?',
      cn: '图中的棕色马有鬃毛和长尾巴吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ ม้ามีแผงคอบนลำคอ ขา 4 ข้างที่แข็งแรง และหางยาว',
      en: 'Yes, horses have a mane, strong legs, and a long tail.',
      cn: '是的，马有鬃毛、强壮的四肢和长尾巴。',
    },
  },
  {
    category: 'animals',
    artKey: 'frog',
    icon: 'frog',
    subject: { th: 'กบเขียว', en: 'Green Frog', cn: '绿青蛙' },
    question: {
      th: 'กบสีเขียวบนใบบัวนี้มีปีกเหมือนนกใช่ไหม?',
      en: 'Does this green frog on a lily pad have wings like a bird?',
      cn: '荷叶上的绿青蛙像鸟一样有翅膀吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ กบไม่มีปีก แต่ใช้ขาหลังที่แข็งแรงกระโดด',
      en: 'No, frogs have strong hind legs for jumping, not wings.',
      cn: '不是，青蛙没有翅膀，靠强壮的后腿跳跃。',
    },
  },
  {
    category: 'animals',
    artKey: 'crab',
    icon: 'crab',
    subject: { th: 'ปูแดง', en: 'Red Crab', cn: '红螃蟹' },
    question: {
      th: 'ปูสีแดงตัวนี้มีก้ามโตอยู่ด้านข้างทั้งสองข้างใช่ไหม?',
      en: 'Does this red crab have two large claws?',
      cn: '这只红螃蟹长着两只大钳子吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ ปูมีก้ามคู่ด้านหน้าและมักเดินเฉียงไปด้านข้าง',
      en: 'Yes, crabs have a pair of large claws and walk sideways.',
      cn: '是的，螃蟹有两只大蟹钳并且横着走。',
    },
  },

  // ================= 056..070 COUNTING =================
  {
    category: 'counting',
    artKey: 'count_apples_3',
    icon: 'count_apples_3',
    subject: { th: 'แอปเปิล 3 ผล', en: 'Three Apples', cn: '三个苹果' },
    question: {
      th: 'ในภาพนี้มีแอปเปิลทั้งหมด 5 ลูกใช่ไหม?',
      en: 'Are there 5 apples in this picture?',
      cn: '图中有5个苹果吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ นับในภาพมีแอปเปิลสีแดง 3 ลูกเท่านั้น',
      en: 'No, there are 3 red apples in this picture, not 5.',
      cn: '不是，图中有3个红苹果，不是5个。',
    },
  },
  {
    category: 'counting',
    artKey: 'count_stars_2plus2',
    icon: 'count_stars_2plus2',
    subject: { th: 'ดาว 2 + 2 ดวง', en: 'Stars 2 + 2', cn: '星星 2+2' },
    question: {
      th: 'จากภาพ ดาว 2 ดวง บวก 2 ดวง เท่ากับ 4 ดวงใช่ไหม?',
      en: 'In this picture, does 2 stars + 2 stars equal 4 stars?',
      cn: '图中2颗星加2颗星等于4颗星吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ 2 + 2 เท่ากับ 4 ดวงถูกต้อง',
      en: 'Yes, 2 + 2 equals 4 stars.',
      cn: '是的，2 + 2 等于 4 颗星星。',
    },
  },
  {
    category: 'counting',
    artKey: 'count_fingers_5',
    icon: 'count_fingers_5',
    subject: { th: 'มือ 1 ข้าง 5 นิ้ว', en: 'Five Fingers', cn: '五根手指' },
    question: {
      th: 'มือ 1 ข้างในภาพนี้มีทั้งหมด 8 นิ้วใช่ไหม?',
      en: 'Does this single hand have 8 fingers?',
      cn: '图中的这只手有8根手指吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ มือ 1 ข้างมี 5 นิ้ว (โป้ง ชี้ กลาง นาง ก้อย)',
      en: 'No, one human hand has 5 fingers, not 8.',
      cn: '不是，一只手有5根手指，不是8根。',
    },
  },
  {
    category: 'counting',
    artKey: 'count_car_4wheels',
    icon: 'count_car_4wheels',
    subject: { th: 'โจทย์เลข 10 - 4', en: 'Math 10 - 4', cn: '算术 10 - 4' },
    question: {
      th: 'จากโจทย์ใต้รูปรถยนต์ 10 ลบ 4 เท่ากับ 6 ใช่ไหม?',
      en: 'From the math problem under the car, does 10 minus 4 equal 6?',
      cn: '根据汽车下方的算式，10 减 4 等于 6 吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ 10 ลบ 4 เท่ากับ 6 ถูกต้อง',
      en: 'Yes, 10 - 4 = 6.',
      cn: '是的，10 - 4 等于 6。',
    },
  },
  {
    category: 'counting',
    artKey: 'count_balls_2',
    icon: 'count_balls_2',
    subject: { th: 'ลูกฟุตบอล 2 ลูก', en: 'Two Soccer Balls', cn: '两个足球' },
    question: {
      th: 'ในภาพนี้มีลูกฟุตบอลทั้งหมด 4 ลูกใช่ไหม?',
      en: 'Are there 4 soccer balls in this picture?',
      cn: '图中有4个足球吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ นับในภาพมีลูกฟุตบอลแค่ 2 ลูก',
      en: 'No, there are 2 soccer balls in this picture.',
      cn: '不是，图中只有2个足球。',
    },
  },
  {
    category: 'counting',
    artKey: 'count_oranges_6',
    icon: 'count_oranges_6',
    subject: { th: 'ส้ม 6 ผล', en: 'Six Oranges', cn: '六个橙子' },
    question: {
      th: 'ในภาพนี้มีผลส้มทั้งหมด 6 ลูกใช่ไหม?',
      en: 'Are there 6 oranges in this picture?',
      cn: '图中有6个橙子吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ แถวบน 3 ลูก รวมกับแถวล่าง 3 ลูก เป็น 6 ลูก',
      en: 'Yes, 3 in the top row plus 3 in the bottom row equals 6.',
      cn: '是的，上排3个加下排3个共6个橙子。',
    },
  },
  {
    category: 'counting',
    artKey: 'count_balloons_3',
    icon: 'count_balloons_3',
    subject: { th: 'ลูกโป่ง 3 ใบ', en: 'Three Balloons', cn: '三个气球' },
    question: {
      th: 'ในภาพนี้มีลูกโป่งทั้งหมด 6 ใบใช่ไหม?',
      en: 'Are there 6 balloons in this picture?',
      cn: '图中有6个气球吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ มีลูกโป่ง 3 ใบ คือสีแดง น้ำเงิน และเขียว',
      en: 'No, there are only 3 balloons (red, blue, and green).',
      cn: '不是，图中只有红、蓝、绿3个气球。',
    },
  },
  {
    category: 'counting',
    artKey: 'count_flowers_5',
    icon: 'count_flowers_5',
    subject: { th: 'ดอกไม้ 5 ดอก', en: 'Five Flowers', cn: '五朵花' },
    question: {
      th: 'นับดอกไม้ในภาพนี้ได้ทั้งหมด 5 ดอกใช่ไหม?',
      en: 'Are there 5 flowers in this picture?',
      cn: '图中一共能数出5朵花吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ แถวบนมี 3 ดอก และแถวล่างมี 2 ดอก รวมเป็น 5 ดอก',
      en: 'Yes, 3 flowers on top plus 2 below equals 5 flowers.',
      cn: '是的，上排3朵加下排2朵总共5朵花。',
    },
  },
  {
    category: 'counting',
    artKey: 'count_birds_4',
    icon: 'count_birds_4',
    subject: { th: 'นก 4 ตัว', en: 'Four Birds', cn: '四只鸟' },
    question: {
      th: 'ในภาพนี้มีนกสีฟ้าทั้งหมด 2 ตัวใช่ไหม?',
      en: 'Are there only 2 blue birds in this picture?',
      cn: '图中只有2只蓝鸟吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ มีนกสีฟ้าทั้งหมด 4 ตัว (2 ตัวบน และ 2 ตัวล่าง)',
      en: 'No, there are 4 blue birds shown in total.',
      cn: '不是，图中共有4只蓝鸟。',
    },
  },
  {
    category: 'counting',
    artKey: 'count_cups_2',
    icon: 'count_cups_2',
    subject: { th: 'แก้วกาแฟ 2 ใบ', en: 'Two Cups', cn: '两个杯子' },
    question: {
      th: 'ในภาพนี้มีแก้วหูสีฟ้าทั้งหมด 2 ใบใช่ไหม?',
      en: 'Are there 2 cups in this picture?',
      cn: '图中有2个带把手的杯子吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ ในภาพมีแก้ววางคู่กันทั้งหมด 2 ใบ',
      en: 'Yes, there are 2 cups shown side by side.',
      cn: '是的，图中有2个并排的杯子。',
    },
  },
  {
    category: 'counting',
    artKey: 'count_marbles_7',
    icon: 'count_marbles_7',
    subject: { th: 'ลูกแก้ว 7 ลูก', en: 'Seven Marbles', cn: '七颗弹珠' },
    question: {
      th: 'ในภาพนี้มีลูกแก้วสีสันสดใสทั้งหมด 10 ลูกใช่ไหม?',
      en: 'Are there 10 marbles in this picture?',
      cn: '图中一共有10颗彩色弹珠吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ แถวบนมี 4 ลูก แถวล่างมี 3 ลูก รวมเป็น 7 ลูก',
      en: 'No, 4 on top plus 3 on bottom makes 7 marbles.',
      cn: '不是，上排4颗加下排3颗一共是7颗弹珠。',
    },
  },
  {
    category: 'counting',
    artKey: 'count_blocks_3plus2',
    icon: 'count_blocks_3plus2',
    subject: { th: 'บล็อกไม้ 3 + 2', en: 'Blocks 3 + 2', cn: '积木 3+2' },
    question: {
      th: 'บล็อก 3 ก้อน บวกเพิ่มอีก 2 ก้อน เท่ากับ 5 ก้อนใช่ไหม?',
      en: 'Does 3 blocks plus 2 blocks equal 5 blocks?',
      cn: '3块积木加2块积木等于5块积木吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ 3 + 2 เท่ากับ 5 ถูกต้อง',
      en: 'Yes, 3 + 2 = 5 blocks.',
      cn: '是的，3 + 2 等于 5 块积木。',
    },
  },
  {
    category: 'counting',
    artKey: 'count_octopus_8',
    icon: 'count_octopus_8',
    subject: { th: 'ปลาหมึก 8 หนวด', en: 'Eight-Arm Octopus', cn: '八爪章鱼' },
    question: {
      th: 'ปลาหมึกสีชมพูในภาพนี้มีหนวดแค่ 2 เส้นใช่ไหม?',
      en: 'Does this pink octopus only have 2 tentacles?',
      cn: '这只粉红章鱼只有2条触手吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ ปลาหมึกยักษ์มีหนวดทั้งหมด 8 เส้น',
      en: 'No, the octopus has 8 tentacles.',
      cn: '不是，这只章鱼共有8条触手。',
    },
  },
  {
    category: 'counting',
    artKey: 'count_clock_12',
    icon: 'count_clock_12',
    subject: { th: 'ตัวเลขบนหน้าปัดนาฬิกา', en: '12 Clock Numbers', cn: '时钟12个数字' },
    question: {
      th: 'หน้าปัดนาฬิกาในภาพมีตัวเลข 1 ถึง 12 ครบ 12 ตัวใช่ไหม?',
      en: 'Does this clock face show numbers 1 through 12?',
      cn: '图中的表盘上显示了1到12共12个数字吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ หน้าปัดนาฬิกามาตรฐานมีตัวเลขบอกชั่วโมง 1 ถึง 12',
      en: 'Yes, an analog clock face shows hours 1 through 12.',
      cn: '是的，标准表盘上有1到12共12个小时数字。',
    },
  },
  {
    category: 'counting',
    artKey: 'count_hearts_1plus1',
    icon: 'count_hearts_1plus1',
    subject: { th: 'หัวใจ 1 + 1 ดวง', en: 'Hearts 1 + 1', cn: '爱心 1+1' },
    question: {
      th: 'หัวใจ 1 ดวง บวก 1 ดวง เท่ากับ 5 ดวงใช่ไหม?',
      en: 'Does 1 heart plus 1 heart equal 5 hearts?',
      cn: '1颗心加1颗心等于5颗心吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ หัวใจ 1 + 1 เท่ากับ 2 ดวงเท่านั้น',
      en: 'No, 1 + 1 equals 2 hearts, not 5.',
      cn: '不是，1 + 1 等于 2 颗心，不是5颗。',
    },
  },

  // ================= 071..082 SAFETY & ROUTINE =================
  {
    category: 'safety_routine',
    artKey: 'medicine',
    icon: 'medicine',
    subject: { th: 'ยาและน้ำเปล่า', en: 'Medicine & Water', cn: '药物和温水' },
    question: {
      th: 'ควรทานยาตามแพทย์สั่งพร้อมน้ำสะอาดใช่ไหม?',
      en: 'Should you take prescribed medicine with clean water?',
      cn: '应该按医嘱用干净的温水服药吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ ควรทานยาตรงเวลาและดื่มน้ำสะอาดตามคำแนะนำแพทย์',
      en: 'Yes, take prescribed medicine on schedule with clean water.',
      cn: '是的，按时用干净的白开水送服处方药最安全。',
    },
  },
  {
    category: 'safety_routine',
    artKey: 'nonslip_shoes',
    icon: 'nonslip_shoes',
    subject: { th: 'รองเท้ากันลื่น', en: 'Non-Slip Shoes', cn: '防滑鞋' },
    question: {
      th: 'พื้นรองเท้าที่มีดอกยางกันลื่นช่วยทำให้ลื่นล้มง่ายขึ้นใช่ไหม?',
      en: 'Do non-slip shoe treads make you slip and fall more easily?',
      cn: '带有防滑花纹的鞋底会让人更容易滑倒吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ ดอกยางกันลื่นช่วยยึดเกาะพื้น ลดความเสี่ยงหกล้ม',
      en: 'No, non-slip treads improve grip and prevent falls.',
      cn: '不是，防滑鞋底能增加摩擦力，预防跌倒。',
    },
  },
  {
    category: 'safety_routine',
    artKey: 'stove_off',
    icon: 'stove_off',
    subject: { th: 'ปิดเตาแก๊ส (OFF)', en: 'Stove Switched OFF', cn: '关闭炉灶' },
    question: {
      th: 'เมื่อทำอาหารเสร็จแล้วควรปิดสวิตช์เตาให้อยู่ตำแหน่ง OFF ใช่ไหม?',
      en: 'Should you switch the stove OFF when finished cooking?',
      cn: '做完饭后应该把炉灶开关转到 OFF 关闭位置吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ การปิดเตาให้อยู่ตำแหน่ง OFF ทุกครั้งช่วยป้องกันอัคคีภัย',
      en: 'Yes, always turning the stove OFF prevents fire hazards.',
      cn: '是的，每次用完后关闭炉灶可以防止火灾危险。',
    },
  },
  {
    category: 'safety_routine',
    artKey: 'wash_hands',
    icon: 'wash_hands',
    subject: { th: 'ล้างมือให้สะอาด', en: 'Wash Hands', cn: '洗手' },
    question: {
      th: 'ก่อนรับประทานอาหาร เราไม่จำเป็นต้องล้างมือให้สะอาดใช่ไหม?',
      en: 'Is washing your hands before eating unnecessary?',
      cn: '饭前洗手是没有必要的吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ การล้างมือด้วยน้ำและสบู่ช่วยขจัดเชื้อโรคก่อนมื้ออาหาร',
      en: 'No, washing hands with soap and water removes germs.',
      cn: '不是，饭前用肥皂和水洗手能有效去除病菌。',
    },
  },
  {
    category: 'safety_routine',
    artKey: 'stair_handrail',
    icon: 'stair_handrail',
    subject: { th: 'ราวจับบันได', en: 'Stair Handrail', cn: '楼梯扶手' },
    question: {
      th: 'เวลาเดินขึ้นลงบันได ควรจับราวจับเพื่อป้องกันการหกล้มใช่ไหม?',
      en: 'Should you hold the handrail when using stairs to prevent falls?',
      cn: '上下楼梯时抓紧扶手能预防跌倒吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ การจับราวบันไดช่วยพยุงตัวและทรงตัวได้ปลอดภัย',
      en: 'Yes, holding the handrail adds stability on stairs.',
      cn: '是的，扶好楼梯扶手能增强平衡，保障安全。',
    },
  },
  {
    category: 'safety_routine',
    artKey: 'walking_cane',
    icon: 'walking_cane',
    subject: { th: 'ไม้เท้าพยุงเดิน', en: 'Walking Cane', cn: '拐杖' },
    question: {
      th: 'ผู้ที่มีปัญหาการทรงตัวควรกระโดดวิ่งโดยไม่ใช้ไม้เท้าช่วยพยุงใช่ไหม?',
      en: 'Should someone with balance trouble jump instead of using a cane?',
      cn: '平衡能力较差的人应该不用拐杖而去跳跃吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ ไม้เท้าช่วยค้ำยันเพิ่มความมั่นคงและป้องกันอุบัติเหตุ',
      en: 'No, a walking cane provides safe support for balance.',
      cn: '不是，使用拐杖可以提供稳定支撑，防止摔伤。',
    },
  },
  {
    category: 'safety_routine',
    artKey: 'bp_monitor',
    icon: 'bp_monitor',
    subject: { th: 'เครื่องวัดความดันโลหิต', en: 'Blood Pressure Monitor', cn: '血压计' },
    question: {
      th: 'เครื่องในภาพนี้ใช้วัดความดันโลหิตใช่ไหม?',
      en: 'Is this device used to measure blood pressure?',
      cn: '图中的设备是用来测量血压的吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ เครื่องวัดความดันแสดงค่า 120/80 เพื่อติดตามสุขภาพหัวใจ',
      en: 'Yes, blood pressure monitors track systolic and diastolic pressure.',
      cn: '是的，血压计用于监测收缩压和舒张压。',
    },
  },
  {
    category: 'safety_routine',
    artKey: 'safety_helmet',
    icon: 'safety_helmet',
    subject: { th: 'หมวกนิรภัย', en: 'Safety Helmet', cn: '安全帽' },
    question: {
      th: 'หมวกกันน็อกสีเหลืองนี้ใช้ใส่นอนบนเตียงในห้องนอนใช่ไหม?',
      en: 'Is this yellow safety helmet meant for sleeping in bed?',
      cn: '这顶黄色安全帽是躺在床上睡觉时戴的吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ หมวกนิรภัยใส่ป้องกันศีรษะเวลาขี่รถหรือทำงาน',
      en: 'No, a safety helmet protects your head while riding or working.',
      cn: '不是，安全帽用于骑行或工作时保护头部。',
    },
  },
  {
    category: 'safety_routine',
    artKey: 'seatbelt',
    icon: 'seatbelt',
    subject: { th: 'เข็มขัดนิรภัย', en: 'Seatbelt', cn: '安全带' },
    question: {
      th: 'เวลาโดยสารรถยนต์ควรคาดเข็มขัดนิรภัยทุกครั้งใช่ไหม?',
      en: 'Should you always fasten your seatbelt when riding in a car?',
      cn: '乘车时每次都应该系好安全带吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ เข็มขัดนิรภัยช่วยลดแรงกระแทกและปกป้องชีวิต',
      en: 'Yes, seatbelts keep passengers safe during travel.',
      cn: '是的，系好安全带可以在行车中保护安全。',
    },
  },
  {
    category: 'safety_routine',
    artKey: 'crosswalk',
    icon: 'crosswalk',
    subject: { th: 'ทางม้าลาย', en: 'Crosswalk', cn: '斑马线' },
    question: {
      th: 'เราควรวิ่งข้ามถนนตรงทางโค้งมืดๆ แทนการข้ามทางม้าลายใช่ไหม?',
      en: 'Should we run across a dark curved road instead of a crosswalk?',
      cn: '我们应该在昏暗的弯道横穿马路而不是走斑马线吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ ควรข้ามถนนตรงทางม้าลายและมองรถซ้ายขวาให้ปลอดภัย',
      en: 'No, always cross safely at marked crosswalks.',
      cn: '不是，过马路应该走斑马线并注意左右来车。',
    },
  },
  {
    category: 'safety_routine',
    artKey: 'wet_floor_cone',
    icon: 'wet_floor_cone',
    subject: { th: 'ป้ายเตือนพื้นเปียก', en: 'Wet Floor Sign', cn: '小心地滑标志' },
    question: {
      th: 'ป้ายตั้งสีเหลืองเครื่องหมายตกใจเตือนให้ระวังพื้นลื่นใช่ไหม?',
      en: 'Does this yellow warning cone alert you to a slippery floor?',
      cn: '这个带感叹号的黄色标志是提醒注意地滑吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ เมื่อเห็นป้ายเตือนพื้นเปียกควรเดินช้าลงหรือหลีกเลี่ยง',
      en: 'Yes, slow down or walk around wet floor warning cones.',
      cn: '是的，看到地滑警示牌时应放慢脚步或绕行。',
    },
  },
  {
    category: 'safety_routine',
    artKey: 'shoulder_stretch',
    icon: 'shoulder_stretch',
    subject: { th: 'ท่ายืดเหยียดไหล่', en: 'Shoulder Stretch', cn: '肩部伸展' },
    question: {
      th: 'การฝึกยืดเหยียดไหล่เบาๆ ทำให้กล้ามเนื้อฉีกขาดทันทีทุกครั้งใช่ไหม?',
      en: 'Does gentle shoulder stretching always tear your muscles immediately?',
      cn: '轻柔地做肩部伸展每次都会立刻撕裂肌肉吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ การยืดเหยียดอย่างนุ่มนวลช่วยให้ข้อต่อและกล้ามเนื้อคลายตัว',
      en: 'No, gentle physical therapy stretches improve flexibility safely.',
      cn: '不是，柔和的康复伸展动作能安全缓解僵硬并增强活动度。',
    },
  },

  // ================= 083..092 COLOR, SHAPE & TIME =================
  {
    category: 'color_shape_time',
    artKey: 'traffic_red',
    icon: 'traffic_red',
    subject: { th: 'ไฟจราจรสีแดง', en: 'Red Traffic Light', cn: '红灯' },
    question: {
      th: 'สัญญาณไฟสีแดงด้านบนสุดหมายถึงให้รถหยุดใช่ไหม?',
      en: 'Does the glowing red light at the top mean stop?',
      cn: '最上方亮起的红灯表示停车吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ ไฟแดงหมายถึงต้องหยุดรถหลังเส้นหยุดเพื่อความปลอดภัย',
      en: 'Yes, a red traffic signal always means stop.',
      cn: '是的，交通信号红灯表示必须停车等待。',
    },
  },
  {
    category: 'color_shape_time',
    artKey: 'traffic_green',
    icon: 'traffic_green',
    subject: { th: 'ไฟจราจรสีเขียว', en: 'Green Traffic Light', cn: '绿灯' },
    question: {
      th: 'เมื่อไฟจราจรสีเขียวด้านล่างสว่างขึ้น หมายถึงห้ามขับผ่านใช่ไหม?',
      en: 'When the green light turns on, does it mean stop and do not go?',
      cn: '下方绿灯亮起时表示禁止通行吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ สัญญาณไฟสีเขียวหมายถึงไปได้เมื่อทางปลอดภัย',
      en: 'No, a green traffic light means proceed when safe.',
      cn: '不是，绿灯亮起表示可以安全通行。',
    },
  },
  {
    category: 'color_shape_time',
    artKey: 'triangle',
    icon: 'triangle',
    subject: { th: 'สามเหลี่ยมสีแดง', en: 'Red Triangle', cn: '红色三角形' },
    question: {
      th: 'รูปทรงสีแดงในภาพนี้เป็นรูปสามเหลี่ยมมี 3 มุมใช่ไหม?',
      en: 'Is this red shape a triangle with 3 corners?',
      cn: '图中的红色图形是有3个角的三角形吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ รูปสามเหลี่ยมมีด้าน 3 ด้านและมุม 3 มุม',
      en: 'Yes, a triangle has three sides and three corners.',
      cn: '是的，三角形有三条边和三个角。',
    },
  },
  {
    category: 'color_shape_time',
    artKey: 'blue_circle',
    icon: 'blue_circle',
    subject: { th: 'วงกลมสีฟ้า', en: 'Blue Circle', cn: '蓝色圆形' },
    question: {
      th: 'รูปวงกลมสีฟ้าในภาพนี้มีมุมแหลม 4 มุมใช่ไหม?',
      en: 'Does this blue circle have 4 sharp corners?',
      cn: '图中的蓝色圆形有4个尖角吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ วงกลมมีเส้นโค้งต่อเนื่อง ไม่มีมุมแหลม',
      en: 'No, a circle is completely round with zero corners.',
      cn: '不是，圆形是光滑闭合的曲线，没有角。',
    },
  },
  {
    category: 'color_shape_time',
    artKey: 'green_square',
    icon: 'green_square',
    subject: { th: 'สี่เหลี่ยมจัตุรัสสีเขียว', en: 'Green Square', cn: '绿色正方形' },
    question: {
      th: 'รูปสีเขียวในภาพนี้คือรูปสี่เหลี่ยมที่มี 4 ด้านเท่ากันใช่ไหม?',
      en: 'Is this green shape a square with 4 equal sides?',
      cn: '图中的绿色图形是四条边相等的正方形吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ สี่เหลี่ยมจัตุรัสมีด้านยาวเท่ากันทั้ง 4 ด้าน',
      en: 'Yes, a square has four sides of equal length.',
      cn: '是的，正方形的四条边长度完全相等。',
    },
  },
  {
    category: 'color_shape_time',
    artKey: 'yellow_star',
    icon: 'yellow_star',
    subject: { th: 'ดาวสีเหลือง', en: 'Yellow Star', cn: '黄色星星' },
    question: {
      th: 'รูปดาว 5 แฉกในภาพนี้เป็นสีดำสนิทใช่ไหม?',
      en: 'Is the five-pointed star in this picture solid black?',
      cn: '图中的五角星是纯黑色的吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ รูปดาวในภาพเป็นสีเหลืองสดใส',
      en: 'No, the five-pointed star in this picture is bright yellow.',
      cn: '不是，图中的五角星是亮黄色的。',
    },
  },
  {
    category: 'color_shape_time',
    artKey: 'morning_sun',
    icon: 'morning_sun',
    subject: { th: 'ดวงอาทิตย์ยามเช้า', en: 'Morning Sun', cn: '早晨太阳' },
    question: {
      th: 'ดวงอาทิตย์ส่องแสงเหนือภูเขาในภาพแสดงถึงเวลากลางวันใช่ไหม?',
      en: 'Is the shining sun above the hills showing daytime?',
      cn: '图中升起的太阳代表白天吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ ดวงอาทิตย์ขึ้นและส่องแสงสว่างในเวลากลางวัน',
      en: 'Yes, the shining sun illuminates the sky during daytime.',
      cn: '是的，白天天空中太阳会发出明亮的光芒。',
    },
  },
  {
    category: 'color_shape_time',
    artKey: 'night_moon',
    icon: 'night_moon',
    subject: { th: 'พระจันทร์และดาวกลางคืน', en: 'Moon and Stars at Night', cn: '夜晚月亮星星' },
    question: {
      th: 'ท้องฟ้าสีเข้มที่มีพระจันทร์และดวงดาวคือเวลาเที่ยงวันใช่ไหม?',
      en: 'Does this dark sky with a crescent moon and stars show noon?',
      cn: '有月亮和星星的深色夜空代表正午吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ ภาพพระจันทร์และดวงดาวเป็นสัญลักษณ์ของเวลากลางคืน',
      en: 'No, the moon and stars shine in the night sky.',
      cn: '不是，月亮和星星出现在夜晚的天空中。',
    },
  },
  {
    category: 'color_shape_time',
    artKey: 'morning_clock_8am',
    icon: 'morning_clock_8am',
    subject: { th: 'นาฬิกาเวลา 08:00 น.', en: '08:00 AM Clock', cn: '上午8点时钟' },
    question: {
      th: 'นาฬิกาในภาพนี้ชี้บอกเวลา 8 โมงเช้า (08:00 AM) ใช่ไหม?',
      en: 'Does this clock show 08:00 AM in the morning?',
      cn: '图中的时钟显示的是早上8点 (08:00 AM) 吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ เข็มสั้นชี้เลข 8 และเข็มยาวชี้เลข 12 คือ 08:00 น.',
      en: 'Yes, the short hand points to 8 and long hand to 12 for 08:00 AM.',
      cn: '是的，时针指向8、分针指向12表示上午8点。',
    },
  },
  {
    category: 'color_shape_time',
    artKey: 'rainbow',
    icon: 'rainbow',
    subject: { th: 'รุ้งกินน้ำ', en: 'Rainbow', cn: '彩虹' },
    question: {
      th: 'รุ้งกินน้ำโค้งในภาพมีเพียงสีเทาสีเดียวใช่ไหม?',
      en: 'Does this arched rainbow only have one gray color?',
      cn: '图中的彩虹只有灰白色一种颜色吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ รุ้งกินน้ำมีหลายสีสดใส ทั้งแดง ส้ม เหลือง เขียว และน้ำเงิน',
      en: 'No, a rainbow displays bright bands of red, orange, yellow, green, and blue.',
      cn: '不是，彩虹由红、橙、黄、绿、蓝等多种鲜艳色彩组成。',
    },
  },

  // ================= 093..100 POSITION =================
  {
    category: 'position',
    artKey: 'arrow_up',
    icon: 'arrow_up',
    subject: { th: 'ลูกศรชี้ขึ้น', en: 'Up Arrow', cn: '向上箭头' },
    question: {
      th: 'ลูกศรสีเขียวเข้มในภาพนี้ชี้ขึ้นด้านบนใช่ไหม?',
      en: 'Is this teal arrow pointing UP?',
      cn: '这个青色箭头指向上方吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ หัวลูกศรชี้พุ่งขึ้นด้านบน',
      en: 'Yes, the arrowhead points upward.',
      cn: '是的，箭头朝向正上方。',
    },
  },
  {
    category: 'position',
    artKey: 'arrow_down',
    icon: 'arrow_down',
    subject: { th: 'ลูกศรชี้ลง', en: 'Down Arrow', cn: '向下箭头' },
    question: {
      th: 'ลูกศรสีน้ำเงินในภาพนี้ชี้ขึ้นข้างบนฟ้าใช่ไหม?',
      en: 'Is this blue arrow pointing UP toward the sky?',
      cn: '这个蓝色箭头是向上指着天空的吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ หัวลูกศรสีน้ำเงินชี้ลงด้านล่าง',
      en: 'No, the blue arrow is pointing downward.',
      cn: '不是，这个蓝色箭头指向下方。',
    },
  },
  {
    category: 'position',
    artKey: 'arrow_right',
    icon: 'arrow_right',
    subject: { th: 'ลูกศรชี้ขวา', en: 'Right Arrow', cn: '向右箭头' },
    question: {
      th: 'ลูกศรสีเขียวในภาพนี้ชี้ไปทางด้านขวาใช่ไหม?',
      en: 'Is this green arrow pointing to the RIGHT?',
      cn: '这个绿色箭头指向右边吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ หัวลูกศรสีเขียวหันชี้ไปทางขวา',
      en: 'Yes, the green arrow points to the right.',
      cn: '是的，绿色箭头朝向右侧。',
    },
  },
  {
    category: 'position',
    artKey: 'arrow_left',
    icon: 'arrow_left',
    subject: { th: 'ลูกศรชี้ซ้าย', en: 'Left Arrow', cn: '向左箭头' },
    question: {
      th: 'ลูกศรสีส้มในภาพนี้ชี้ไปทางด้านขวาใช่ไหม?',
      en: 'Is this orange arrow pointing to the RIGHT?',
      cn: '这个橙色箭头是指向右边吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ ลูกศรสีส้มชี้ไปทางด้านซ้าย',
      en: 'No, the orange arrow is pointing to the left.',
      cn: '不是，橙色箭头指向左边。',
    },
  },
  {
    category: 'position',
    artKey: 'house_roof_top',
    icon: 'house_roof_top',
    subject: { th: 'หลังคาบ้านด้านบน', en: 'Roof on Top', cn: '屋顶在上面' },
    question: {
      th: 'หลังคาบ้านสีแดงตั้งอยู่ด้านบนสุดของตัวบ้านใช่ไหม?',
      en: 'Is the red roof located at the TOP of the house?',
      cn: '红色的屋顶位于房子的最上方吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ หลังคาสามเหลี่ยมสีแดงอยู่ด้านบนเพื่อกันแดดกันฝน',
      en: 'Yes, the red roof sits on top of the house.',
      cn: '是的，红色三角形屋顶在房子的最上方。',
    },
  },
  {
    category: 'position',
    artKey: 'apple_on_table',
    icon: 'apple_on_table',
    subject: { th: 'แอปเปิลบนโต๊ะ', en: 'Apple on Table', cn: '桌子上的苹果' },
    question: {
      th: 'ลูกแอปเปิลสีแดงในภาพนี้อยู่ใต้โต๊ะใช่ไหม?',
      en: 'Is the red apple in this picture UNDER the table?',
      cn: '图中的红苹果在桌子下面吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ แอปเปิลสีแดงวางอยู่ด้านบนโต๊ะ',
      en: 'No, the red apple is resting ON TOP of the table.',
      cn: '不是，红苹果放在桌子上面。',
    },
  },
  {
    category: 'position',
    artKey: 'ball_under_table',
    icon: 'ball_under_table',
    subject: { th: 'ลูกบอลใต้โต๊ะ', en: 'Ball Under Table', cn: '桌子下的球' },
    question: {
      th: 'ลูกฟุตบอลในภาพนี้วางอยู่ใต้โต๊ะใช่ไหม?',
      en: 'Is the soccer ball in this picture UNDER the table?',
      cn: '图中的足球在桌子下面吗？',
    },
    correctAnswer: 'yes',
    explanation: {
      th: 'ใช่ ลูกฟุตบอลอยู่ด้านล่างใต้แผ่นโต๊ะไม้',
      en: 'Yes, the soccer ball is directly under the table.',
      cn: '是的，足球正处于木桌下方。',
    },
  },
  {
    category: 'position',
    artKey: 'bird_on_branch',
    icon: 'bird_on_branch',
    subject: { th: 'นกเกาะบนกิ่งไม้', en: 'Bird on Branch', cn: '树枝上的鸟' },
    question: {
      th: 'นกสีฟ้าในภาพนี้เกาะอยู่ใต้กิ่งไม้ใช่ไหม?',
      en: 'Is the blue bird in this picture UNDER the branch?',
      cn: '图中的蓝鸟在树枝下方吗？',
    },
    correctAnswer: 'no',
    explanation: {
      th: 'ไม่ใช่ นกสีฟ้ากำลังยืนเกาะอยู่บนกิ่งไม้',
      en: 'No, the blue bird is perched ON TOP of the branch.',
      cn: '不是，蓝鸟正停在树枝上面。',
    },
  },
];

export const COGNITIVE_CARD_BANK_100: readonly CognitiveCardQuestion[] = SEEDS.map((seed, index) => {
  const cardNumber = index + 1;
  return {
    id: `card-${String(cardNumber).padStart(3, '0')}`,
    category: seed.category,
    artKey: seed.artKey,
    icon: seed.icon,
    imageUrl: buildCardIllustrationDataUri({
      cardNumber,
      category: seed.category,
      artKey: seed.artKey,
    }),
    subject: seed.subject,
    question: seed.question,
    correctAnswer: seed.correctAnswer,
    explanation: seed.explanation,
  };
});

/** Backwards-compatible alias for downstream modules referencing COGNITIVE_CARD_BANK_99. */
export const COGNITIVE_CARD_BANK_99 = COGNITIVE_CARD_BANK_100;

export function sampleTenCognitiveCards(categories?: readonly CognitiveCategory[]): CognitiveCardQuestion[] {
  const pool =
    categories && categories.length > 0
      ? COGNITIVE_CARD_BANK_100.filter((c) => categories.includes(c.category))
      : [...COGNITIVE_CARD_BANK_100];
  const copy = [...pool];
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy.slice(0, 10);
}
