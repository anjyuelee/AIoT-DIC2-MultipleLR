# AIoT-DIC2-MultipleLR

- **課程名稱**：AIoT 與數據分析（AIoT & Data Analytics, AIoT-DA）
- **上課作業**：DIC2-多元線性回歸 50-startup and boston housing
- **儲存庫網址**：https://github.com/anjyuelee/AIoT-DIC2-MultipleLR
- **線上網址**：https://anjyuelee.github.io/AIoT-DIC2-MultipleLR/

## 內容

在瀏覽器裡跑的多元線性回歸特徵選擇頁面，沒有後端。載入 CSV 後先做 one-hot 編碼並切分 train/test，再用 Forward SFS、RFE、Lasso、ElasticNet、Exhaustive 五種方法，替 k = 1..p 各挑一組特徵、用 OLS 擬合，比較 Test RMSE、R² 與 MAE。

| 題目 | 資料來源 | 預測目標 |
|---|---|---|
| 50 Startups | [Avik-Jain/100-Days-Of-ML-Code](https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv) | `Profit` |
| Boston Housing | [selva86/datasets](https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv) | `medv` |

頁面上可以切換題目，也可以貼上其他 CSV 網址、選擇預測目標、測試集比例與亂數種子。

## 本機執行

```bash
python3 -m http.server 8765
```

開啟 http://127.0.0.1:8765/ 。預設資料來源與題目清單在 `config.js`。
