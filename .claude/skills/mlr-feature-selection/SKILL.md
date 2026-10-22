---
name: mlr-feature-selection
description: Feature-selection multiple linear regression page: given a CSV URL and a target column, one-hot encode, split train/test, pick k=1..p feature subsets with Forward SFS, RFE, Lasso, ElasticNet, or Exhaustive search, fit OLS, and report test RMSE/R²/MAE, OLS inference (SE, t, p, CI, VIF, F), test residuals, and a live predictor as a static front-end page (GitHub Pages ready). Use for forward/backward/subset feature selection, comparing selection methods or models by feature count k, or re-running this page on a new dataset.
---

# MLR Feature Selection

Build a static front-end page (HTML + JS + Chart.js, no backend): load a CSV → preprocess → five feature-selection methods → test metrics for every (method, k) → charts, tables, coefficient inference, residuals, live predictor. All computation runs in the browser.

## Reference implementation

The repo root holds the complete implementation; read it before changing anything:

- `config.js`: settings (`dataUrl`, `target`, `testRatio`, `seed`) and `presets`, the datasets offered in the 選擇題目 dropdown (`{ name, dataUrl, target, source }`; `source` is the credit shown in the footer)
- `regression.js`: pure computation, no DOM: parsing and encoding (`parseCSV`, `inspect`, `buildDataset`, `trainTestSplit`), OLS (`lstsq` returns `{ coef, R, rss }`), selection (`independentFeatures`, `SELECTORS`, `evaluateSubsets`), inference (`olsSummary`, `tPValue`, `tQuantile`, `fPValue`)
- `app.js`, `index.html`, `style.css`: the UI

**New dataset**: when these files exist, add `{ name, dataUrl, target, source }` to `presets` in `config.js`; to also make it the page's default, set `dataUrl` and `target` to the same values. Then go to step 6. No column name is hard-coded anywhere.
**From scratch**: follow steps 1–6 to build five files with the same split of responsibilities.

## Steps

### 1. Input

- Required: CSV URL and target column; ask the user for whichever is missing. Optional: `testRatio` (default 0.2), `seed` (default 42).
- Put them in `config.js`, and make them editable on the page too: a URL field, a target dropdown listing numeric columns only, a 特徵選擇方法 dropdown (Forward SFS, RFE（Backward）, Lasso, ElasticNet, Exhaustive, 比較全部方法), test ratio, and seed. Changing target, ratio, or seed recomputes without re-fetching; changing the method only re-renders, because `analyze()` runs every method at once.
- A 題目 dropdown, first in the settings bar, lists every preset plus 自訂 CSV 網址. Choosing a preset fills in its URL, reloads, and selects its target. The dropdown follows the URL field: it shows the preset whose URL matches, or 自訂 when none does.
- A browser `fetch` needs CORS: confirm with `curl -sI` that the response carries `access-control-allow-origin` (raw.githubusercontent.com does).

Done when: the CSV is fetched, its header contains the target, and the target column is entirely numeric.

### 2. Preprocessing

- Drop rows with any empty value, and show the dropped count on the page. Only empty strings count as missing (`NA` is a category level, unlike pandas). Keep each kept row's position in the CSV (`ids`) for the residual table.
- Column types: a column whose every value parses to a finite number is numeric; anything else is categorical.
- **One-hot, drop first**: sort a categorical column's levels alphabetically, drop the first as the baseline, and name dummies `{column}_{level}`, matching pandas `get_dummies(drop_first=True)`. Each dummy is an independent candidate feature; record each feature's source column and level (`sources`) so the predictor can map back.
- **Split**: shuffle with `mulberry32(seed)` + Fisher–Yates; test size is `ceil(n × testRatio)` (same as sklearn). The resulting split differs from sklearn's `random_state`, so numbers are not comparable to a Python notebook.
- **Independent features**: walk the features in column order and keep one only if the training design (intercept + kept features + it) stays full rank under the QR test in step 3. Constant columns, duplicates, and all-zero training dummies drop out here; state on the page how many were left out. Every method runs on this kept set.

Done when: the feature names match the columns of pandas `get_dummies(drop_first=True)` one for one.

### 3. Feature selection (k = 1..p, training set only)

Selection sees only the training set and the test set is for evaluation only; choosing features by test error is leakage. Each method returns one feature subset per k:

- **Forward SFS**: start from the intercept-only model; at each step add the remaining feature giving the lowest training RSS. Subsets list features in order of addition.
- **RFE（Backward）**: start from all features, fit OLS, drop the feature with the smallest `|b_j × sd_j|` (training population sd), repeat. OLS is invariant to affine rescaling, so this equals sklearn `RFE(LinearRegression())` on `StandardScaler`-transformed training X.
- **Lasso / ElasticNet** (`l1_ratio` 1 / 0.5): standardize training X **and y** (mean, population sd; standardizing y keeps the ElasticNet path independent of the target's units). Solve the regularization path by coordinate descent with warm starts over 200 log-spaced λ from `λ_max = max|zᵀy| / (n × l1_ratio)` down to `1e-4 × λ_max` (sklearn's `lasso_path` / `enet_path` grid), stopping each λ when `max|Δw| < 1e-13` or after 1000 sweeps. Rank features by the first grid index where their coefficient is nonzero, ties by larger `|w|` at that index, then by column index; features that never enter go last. The k-subset is the top k.
- **Exhaustive**: for each k, the subset of all `C(p, k)` with the lowest training RSS (ties to the lexicographically first). Enumerate depth-first with an incrementally updated Cholesky factor on centered, unit-norm columns (`RSS = Σ(y − ȳ)² − ‖L⁻¹ Xᵀy‖²`); Boston's 8,191 subsets take ~5 ms. Disable the method (dropdown option and comparison) when more than `EXHAUSTIVE_MAX_P = 15` features are kept.

Then refit every subset by OLS with an intercept, solved with Householder QR (a column is linearly dependent when `|R_jj| ≤ 1e-10 × ‖original column‖`), and record per k: features, coefficients, test RMSE, test R².

Done when: every enabled method has one record per k covering 1..p.

### 4. Metrics, best model, inference, equation

- Test metrics: RMSE = `sqrt(mean((y − ŷ)²))`, R² = `1 − SS_res / SS_tot` with SS_tot around the test-set mean (same as sklearn `r2_score`), MAE = `mean|y − ŷ|`.
- **Best model**: per method, the lowest test RMSE (ties go to the smaller k); on the same test set this is also the highest R², so both charts agree. In 比較全部方法, the lowest across methods; RMSEs within `1e-9` relative are the same subset reached by different methods, so they tie and go to the earlier method in the dropdown order.
- **Inference** for the selected model, on the training set, matching statsmodels `OLS`: `df = n − k − 1`, `σ̂² = RSS / df`, `SE_j = σ̂ × ‖row j of R⁻¹‖`, `t = b / SE`, two-sided p from the t distribution, 95% CI `b ± t₀.₉₇₅,df × SE`, Adjusted R² `1 − (1 − R²_train)(n − 1)/df`, F `((SST − RSS)/k) / σ̂²` with its p-value, and VIF_j = `1 / (1 − R²_j)` from regressing feature j on the other selected features with an intercept (1 when k = 1). Distribution functions come from the regularized incomplete beta (continued fraction) and Lanczos ln Γ; the t quantile by bisection. When `df ≤ 0` every inferential value shows `—`.
- The regression equation uses the selected model's coefficients: `target = b0 + b1 × x1 − b2 × x2 …`. Format |coefficient| ≥ 1000 with two decimals and thousands separators, otherwise 4 significant digits; write every negative sign, intercept and tables included, as `−` (U+2212).

### 5. UI

The page follows the QClay "Finance Management Dashboard" shot on Dribbble (`dribbble.com/shots/23245345`): a rounded app canvas on a gray frame, a light-gray header band, and a bento grid of large-radius gray cards holding white inner panels; Manrope (Google Fonts) with tabular figures; coral as the one accent (primary buttons, the chart line, dots, warnings), black pills for the selected state, white pill dropdowns and circular icon buttons, no borders or shadows on cards.

- **Header band**: brand (black circle logo, two-line title); a pill 題目 select, a search-style pill holding the CSV URL with a coral 載入資料 → button, and a circular theme toggle; then the row count in a big circle with the candidate-feature count, pill fields for target, method, test ratio, and seed, and on the right a hero headline naming the selected model with a gray line under it (features used, Test RMSE) plus a coral 回到最佳 → button when it is not the best.
- **Bento grid**, with breakpoints set from the Boston table widths (cards included: feature matrix ~630px, method comparison ~680px, coefficients ~760px) so no table scrolls from 1000px up:
  - ≥ 1800px, 12 columns: KPI row (test 3 · train 3 · donut 2 · features 2 · split 2), charts across the full width, matrix 5 + residuals 7, coefficients 7 + predictor 5.
  - ≥ 1440px, 6 columns: KPIs in two rows, charts across the full width, matrix 3 + residuals 3, coefficients 4 + predictor 2.
  - ≥ 1000px, 6 columns: KPIs in two rows, every other card full width. Below 1000px, 2 columns with only the donut and feature-count cards side by side.
  - The chart card has its own row; the cards whose height changes with k sit below, so clicking a model never resizes the charts.
- **KPI cards**: 測試集 (Test RMSE large, R², MAE, n; footer with RMSE change vs the all-features model); 訓練集 OLS (Adjusted R² large, n, residual df, σ̂; footer with F and a Prob (F) pill); a black donut with a coral arc for Test R² (drawn clamped to 0–100%, labeled with the actual value); used features as `k / p` with one dot per candidate feature, the first k coral; train / test counts with a split bar and the ratio and seed.
- **Charts** (Chart.js): Test RMSE vs k and Test R² vs k, both always shown side by side in one card (stacked below 760px), each in its own white panel with a title and a 越低越好 / 越高越好 note; x axis k = 1..max k, horizontal grid lines only, monotone-smoothed lines, a black tooltip.
  - Single method: a coral line with coral markers; the best point is a larger black dot with a white ring and a black pill label "最佳 k = n" on the side away from the line (below on RMSE, above on R²), clamped inside the plot area. The selected k is a rounded coral band.
  - 比較全部方法: one series per method, colored `--series-1..5` (coral first) by the fixed method order so a disabled Exhaustive never repaints the others, each with its own marker shape (● ▲ ■ ✕ +) as secondary encoding; pill toggles above the charts (visually hidden checkboxes) show or hide each method on both charts, and hidden methods are kept when the charts redraw. The best point is the same black dot, labeled with the method or "n 個方法相同".
  - Hover shows a dashed vertical crosshair and the tooltip: the value first, then that k's feature list (single) or every method's value keyed by a line in its color (compare).
- **Table** (matrix card), the accessible equivalent of the charts:
  - Single method: a feature-inclusion matrix. Each row is a k with Test RMSE, ΔRMSE vs the previous k (▼ / ▲), and Test R², then one column per feature (ordered by first entry, names set vertically) holding a gray dot when excluded, a light-coral dot when included, or a coral one when new versus the previous k, and a last filler column taking the spare width; mark the best row with ◆ and give each row a visually hidden feature list for screen readers. Below 640px hide the ΔRMSE and R² columns.
  - 比較全部方法: rows are k, columns are methods, each cell a one-line pill button showing RMSE then R²; bold each method's lowest RMSE and mark every cell tied with the overall best with ◆.
- **Linked selection**: the mode's best model is selected by default; clicking a chart point, a table row (or Enter on a focused row), or a comparison cell selects that (method, k), turns that row or cell into a black pill, moves both charts' selected-k band, and updates the hero, KPI cards, and the cards below; hovering a table row shows that k's tooltip on both charts.
- **參數估計**: the equation (target in coral) with a copy button, then intercept plus each feature with estimate, SE, t, p-value (`< 0.0001` floor, coral `*** / ** / *` at 0.001 / 0.01 / 0.05), 95% CI, VIF (a coral 偏高 pill at ≥ 5).
- **即時預測**: the prediction (coral ŷ) on top, then one field per original column the selected model uses, in header order, flowing into several columns when the card is wide. Numeric: a coral slider over the training min..max with step `10^floor(log10(range / 200))` (at least 1 for integer columns), synced with a pill number box, defaulting to the training mean. Categorical: a pill dropdown of every level, the baseline marked （基準）. The prediction updates on every input and notes extrapolation when a number lies outside the training range. Input values persist across model switches and reset when the data, target, or split changes.
- **測試集殘差**: a barcode strip with one bar per test row in test-set order (up for positive, down for negative, coral when high) between dashed ±2σ̂ lines; then rank, data row (`第 n 筆`, the 1-based CSV data row), actual, predicted, residual (actual − predicted), % error (`|residual| / |actual|`, `—` when actual is 0), status; sort by |residual| descending, tag 高殘差 with a coral pill when `|residual| > 2σ̂` (training residual standard error), show 10 rows with a toggle for all. The card header carries the 高殘差 count.
- **Footer cards**: 欄位與型別 (row counts, dropped rows, dependent features left out, then one pill per column with its type, dummy count, and baseline; the target as a black pill); 方法說明; the six CRISP-DM phases (商業理解, 資料理解, 資料準備, 建模, 評估, 部署) as a timeline of numbered coral dots, each mapped to what this page does; 資料來源 with the current data source link, its preset `source` credit, and the reference page link.
- **Theme**: colors are CSS variables on `:root`, with dark values defined under both `prefers-color-scheme: dark` and `[data-theme]`. Chart.js reads its colors from those variables and redraws when the theme changes.
- No horizontal page scrolling at 375px width; tables scroll inside their own containers.

### 6. Verification

1. **Numbers**: run `config.js` + `regression.js` in Node via `vm`, dumping the train/test indices, `ids`, the kept features, and for every method and k: features, coefficients, test predictions and metrics, and `olsSummary`. In Python (pandas, numpy, scikit-learn, statsmodels) redo on **the same indices**: Forward by `numpy.linalg.lstsq` RSS, RFE by `sklearn.feature_selection.RFE` on `StandardScaler` X, Lasso / ElasticNet by `lasso_path` / `enet_path` (`eps=1e-4`, `alphas=200`, standardized X and y, tight `tol`) with the same ranking rule, Exhaustive by `itertools.combinations`, and inference by `statsmodels` `OLS` + `variance_inflation_factor`.
   Done when: every method's subsets match for every k (in order for Forward, Lasso, ElasticNet), and coefficients (rtol 1e-8: raw-scale features leave the design ill-conditioned, so QR and SVD differ by ~1e-9), RMSE, R², MAE (1e-9), SE, t, CI, VIF, F (1e-7), and p-values (1e-6) all match, at seed 42 / 20% and at a few other seeds and ratios.
2. **UI**: serve with `python3 -m http.server` and check each of: no console errors, light and dark themes, 1800px, 1440px, and 1000px (no table scrolls for Boston, in single or compare mode, and the chart height stays put when selecting k = 1 vs the best k) and 375px widths, click (on either chart), Enter, and hover linking in both single and compare modes, the method pill toggles hiding a line on both charts, the predictor with a categorical column, switching presets, loading a custom URL (one with more than 15 features disables Exhaustive), and visible errors for a bad URL and a bad target.

With seed 42 and 20% test, the presets must give:

- 50 Startups (target `Profit`): every method's best is k = 2 (R&D Spend, Marketing Spend), test RMSE 9,304.12, R² 0.9464, MAE 7,736.85, Adjusted R² 0.9477, F 354.50.
- Boston Housing (target `medv`): every method's best is k = 11, test RMSE 4.05, R² 0.7462, Adjusted R² 0.7297, F 99.89. At k = 3 RFE picks rm, dis, lstat (RMSE 4.82) while the others pick rm, ptratio, lstat (4.33). Lasso ranks lstat, rm, ptratio, b, chas, dis, nox, zn, crim, rad, tax, indus, age; ElasticNet ranks lstat, rm, ptratio, b, chas, nox, dis, crim, zn, rad, tax, indus, age.

A different result means the split, selection, or ranking logic has changed.

## Conventions

- Use classic `<script>` tags rather than ES modules, so opening `index.html` via `file://` still works.
- Load Chart.js from cdnjs pinned to an exact version (currently 4.4.1), and Manrope from Google Fonts; redraw the charts once `document.fonts.ready` resolves so canvas text uses it.
- Insert CSV-derived strings (column names, category levels) into the DOM only via `textContent` / `createElement` / `new Option`, to prevent XSS.
- UI copy and code comments are in Traditional Chinese.
- Deploy: push to GitHub → Settings → Pages → `main` branch, `/ (root)`.
