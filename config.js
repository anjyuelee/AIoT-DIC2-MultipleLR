// 可調整的設定：資料來源、預測目標、測試集比例、亂數種子
const CONFIG = {
  dataUrl: 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv',
  target: 'medv',
  testRatio: 0.2,
  seed: 42,
  // 「選擇題目」下拉選單的選項；網址不在清單中時顯示為「自訂 CSV 網址」；source 顯示在頁尾的資料來源
  presets: [
    { name: '50 Startups', dataUrl: 'https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv', target: 'Profit', source: 'Avik Jain，100-Days-Of-ML-Code' },
    { name: 'Boston Housing', dataUrl: 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv', target: 'medv', source: 'Harrison & Rubinfeld（1978），selva86/datasets 整理' },
  ],
};
