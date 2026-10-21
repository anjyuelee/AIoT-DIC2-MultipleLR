---
name: mlr-feature-selection
description: Forward Selection 多元線性回歸：給 CSV URL 與 target，做 one-hot、train/test 切分、k=1..p 的 OLS test RMSE/R²，產出純前端頁面（折線圖、表格、回歸方程式，可放 GitHub Pages）。用於 forward/stepwise 特徵選擇、依特徵數 k 比較回歸模型、或換資料集重跑這個頁面。
---

# MLR Forward Selection

產出一個純前端頁面（HTML + JS + Chart.js，無後端）：讀 CSV → 前處理 → Forward Selection → 每個 k 算 test RMSE/R² → 畫圖、表格、回歸方程式。所有計算在瀏覽器內完成。

## 參考實作

本 repo 根目錄就是完整實作，動手前先讀：

- `config.js`：設定（`dataUrl`、`target`、`testRatio`、`seed`）
- `regression.js`：純計算，不碰 DOM（`parseCSV`、`inspect`、`buildDataset`、`trainTestSplit`、`lstsq`、`forwardSelection`）
- `app.js`、`index.html`、`style.css`：畫面

**換資料集**：上述檔案都在時，只改 `config.js` 的 `dataUrl`、`target`，然後跳到步驟 6。程式不寫死任何欄位名稱。
**從零開始**：照步驟 1–6 建立同樣分工的五個檔案。

## 步驟

### 1. 輸入

- 必要：CSV URL、target 欄位，缺一項就問使用者。選填：`testRatio`（預設 0.2）、`seed`（預設 42）。
- 寫進 `config.js`；頁面上也要能改：URL 輸入框、target 下拉（只列數值欄）、測試比例、seed，後三者一改就重算，不重新下載。
- 純前端 `fetch` 需要 CORS：用 `curl -sI` 確認回應有 `access-control-allow-origin`（raw.githubusercontent.com 有）。

完成條件：取得 CSV，header 含 target，且 target 欄全為數值。

### 2. 前處理

- 略過含缺值的列，並在頁面顯示略過筆數。
- 欄型別：整欄都能轉成有限數字 → 數值；否則 → 類別。
- **One-hot drop first**：類別值依字母排序，丟掉第一個作為 baseline，dummy 命名 `{欄名}_{值}`，與 pandas `get_dummies(drop_first=True)` 相同。每個 dummy 是獨立的候選特徵。
- **切分**：`mulberry32(seed)` + Fisher–Yates 洗牌，測試筆數 `ceil(n × testRatio)`（與 sklearn 相同）。切出來的資料與 sklearn `random_state` 不同，數值不能直接拿來對照 Python notebook。

完成條件：特徵名稱與 pandas `get_dummies(drop_first=True)` 的欄位逐一相同。

### 3. Forward Selection（k = 1..p）

- 從只有截距的模型開始，每一步在剩下的特徵中挑「加入後**訓練集 RSS** 最小」者。選特徵只看訓練集，測試集只用來評估；用測試集挑特徵是 leakage。
- OLS 一律含截距，用 Householder QR 求解。候選特徵造成欄位線性相依（`|R_jj| ≤ 1e-10 × ‖原欄‖`）就跳過該候選；剩下的全部被跳過時提前結束，並在頁面註明未納入幾個。
- 每個 k 記錄：已選特徵（依加入順序）、係數、test RMSE、test R²。

完成條件：每個 k 各一筆紀錄，涵蓋 1..p（扣除線性相依者）。

### 4. 指標、最佳 k、方程式

- RMSE = `sqrt(mean((y − ŷ)²))`，R² = `1 − SS_res / SS_tot`，兩者都在測試集計算，SS_tot 用測試集平均（同 sklearn `r2_score`）。
- 最佳 k = test RMSE 最小者（同分取較小的 k）。在同一測試集上這也是 R² 最大者，所以兩張圖的最佳點必定一致。
- 回歸方程式用選取 k 在訓練集擬合的係數：`target = b0 + b1 × x1 − b2 × x2 …`。|係數| ≥ 1000 取兩位小數加千分位，否則取 4 位有效數字；截距與各項的負號統一用 `−`。

### 5. 畫面

- **兩張折線圖**（Chart.js）：Test RMSE vs k、Test R² vs k。單一序列不放圖例；最佳點用橘色，加「最佳 k = n」文字，標籤放在離開折線的那一側（RMSE 圖標在點下方、R² 圖標在點上方）。Hover 時顯示垂直線和 tooltip，數值為主、該 k 的特徵列表為輔。
- **表格**：k、選到的特徵（標出該步新加入者）、Test RMSE、較上一步的 ΔRMSE（▼ 綠 / ▲ 紅）、Test R²；最佳列加標記。表格同時是圖表的無障礙替代。
- **模型卡**：選取 k 的 RMSE、R²、特徵數、相較全部特徵的 RMSE 變化、回歸方程式（附複製按鈕）。
- **選取連動**：預設選最佳 k；點圖上的點或表格列可切換 k；滑過表格列時兩張圖同步顯示該 k 的 tooltip。
- **主題**：顏色定義成 `:root` 上的 CSS 變數，並在 `prefers-color-scheme: dark` 和 `[data-theme]` 下各定義一次深色值。Chart.js 的顏色從 CSS 變數讀取，切換主題時重畫。
- 375px 寬沒有橫向捲動，表格在自己的容器內捲動。

### 6. 驗證

1. **數值**：用 Node 的 `vm` 載入 `config.js` + `regression.js` 跑一次，輸出 train/test 索引和每個 k 的結果。再用 pandas `get_dummies(drop_first=True)` + `numpy.linalg.lstsq`，在**同一組索引**上重做 forward selection 並比對。
   完成條件：每個 k 的特徵順序、係數（rtol 1e-9）、RMSE、R² 全部一致。
2. **畫面**：`python3 -m http.server` 開頁面，逐項確認：console 無錯誤、light/dark 兩種主題、375px 寬、點選與 hover 連動、錯誤網址與錯誤 target 都有提示。

預設資料（50_Startups.csv、target `Profit`、seed 42、測試 20%）應得到：最佳 k = 2（R&D Spend、Marketing Spend），test RMSE 9,304.12，R² 0.9464。結果不同代表切分或選擇邏輯被改動過。

## 實作約定

- 用 classic `<script>`，不用 ES module，這樣雙擊 `index.html` 用 `file://` 開也能跑。
- Chart.js 從 cdnjs 載入，鎖定確切版本（目前 4.4.1）。
- 來自 CSV 的字串（欄名、類別值）一律用 `textContent` / `createElement` 放進 DOM，避免 XSS。
- 部署：push 到 GitHub → Settings → Pages → 選 `main` branch、`/ (root)`。
