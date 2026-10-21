---
name: mlr-feature-selection
description: Forward Selection multiple linear regression: given a CSV URL and a target column, one-hot encode, split train/test, fit OLS for k=1..p and report test RMSE/R² as a static front-end page (line charts, table, regression equation; GitHub Pages ready). Use for forward/stepwise feature selection, comparing regression models by feature count k, or re-running this page on a new dataset.
---

# MLR Forward Selection

Build a static front-end page (HTML + JS + Chart.js, no backend): load a CSV → preprocess → Forward Selection → test RMSE/R² for every k → charts, table, regression equation. All computation runs in the browser.

## Reference implementation

The repo root holds the complete implementation; read it before changing anything:

- `config.js`: settings (`dataUrl`, `target`, `testRatio`, `seed`) and `presets`, the datasets offered in the 選擇題目 dropdown
- `regression.js`: pure computation, no DOM (`parseCSV`, `inspect`, `buildDataset`, `trainTestSplit`, `lstsq`, `forwardSelection`)
- `app.js`, `index.html`, `style.css`: the UI

**New dataset**: when these files exist, add `{ name, dataUrl, target }` to `presets` in `config.js`; to also make it the page's default, set `dataUrl` and `target` to the same values. Then go to step 6. No column name is hard-coded anywhere.
**From scratch**: follow steps 1–6 to build five files with the same split of responsibilities.

## Steps

### 1. Input

- Required: CSV URL and target column; ask the user for whichever is missing. Optional: `testRatio` (default 0.2), `seed` (default 42).
- Put them in `config.js`, and make them editable on the page too: a URL field, a target dropdown listing numeric columns only, test ratio, and seed. Changing any of the last three recomputes immediately without re-fetching.
- A 選擇題目 dropdown at the very top of the page lists every preset plus 自訂 CSV 網址. Choosing a preset fills in its URL, reloads, and selects its target. The dropdown follows the URL field: it shows the preset whose URL matches, or 自訂 when none does.
- A browser `fetch` needs CORS: confirm with `curl -sI` that the response carries `access-control-allow-origin` (raw.githubusercontent.com does).

Done when: the CSV is fetched, its header contains the target, and the target column is entirely numeric.

### 2. Preprocessing

- Drop rows with any missing value, and show the dropped count on the page.
- Column types: a column whose every value parses to a finite number is numeric; anything else is categorical.
- **One-hot, drop first**: sort a categorical column's levels alphabetically, drop the first as the baseline, and name dummies `{column}_{level}`, matching pandas `get_dummies(drop_first=True)`. Each dummy is an independent candidate feature.
- **Split**: shuffle with `mulberry32(seed)` + Fisher–Yates; test size is `ceil(n × testRatio)` (same as sklearn). The resulting split differs from sklearn's `random_state`, so numbers are not comparable to a Python notebook.

Done when: the feature names match the columns of pandas `get_dummies(drop_first=True)` one for one.

### 3. Forward Selection (k = 1..p)

- Start from the intercept-only model. At each step, add the remaining feature that gives the lowest **training RSS**. Selection sees only the training set and the test set is for evaluation only; choosing features by test error is leakage.
- OLS always includes an intercept and is solved with Householder QR. Skip a candidate that makes the columns linearly dependent (`|R_jj| ≤ 1e-10 × ‖original column‖`); if every remaining candidate is skipped, stop early and state on the page how many features were left out.
- For each k, record: selected features (in order of addition), coefficients, test RMSE, test R².

Done when: there is one record per k covering 1..p (minus linearly dependent features).

### 4. Metrics, best k, equation

- RMSE = `sqrt(mean((y − ŷ)²))` and R² = `1 − SS_res / SS_tot`, both on the test set, with SS_tot around the test-set mean (same as sklearn `r2_score`).
- Best k = the lowest test RMSE (ties go to the smaller k). On the same test set this is also the highest R², so both charts always agree on the best point.
- The regression equation uses the selected k's coefficients fitted on the training set: `target = b0 + b1 × x1 − b2 × x2 …`. Format |coefficient| ≥ 1000 with two decimals and thousands separators, otherwise 4 significant digits; write every negative sign, intercept included, as `−` (U+2212).

### 5. UI

- **Two line charts** (Chart.js): Test RMSE vs k and Test R² vs k. A single series needs no legend. Mark the best point in orange with a "best k = n" label placed on the side away from the line (below the point on the RMSE chart, above it on the R² chart). Hover shows a vertical crosshair and a tooltip with the value first and that k's feature list second.
- **Table**: k, selected features (highlighting the one added at that step), Test RMSE, ΔRMSE vs the previous step (▼ green / ▲ red), Test R²; tag the best row. The table doubles as the accessible equivalent of the charts.
- **Model card**: the selected k's RMSE, R², feature count, RMSE change vs the all-features model, and the regression equation with a copy button.
- **Linked selection**: the best k is selected by default; clicking a chart point or a table row switches k; hovering a table row shows that k's tooltip on both charts.
- **Theme**: colors are CSS variables on `:root`, with dark values defined under both `prefers-color-scheme: dark` and `[data-theme]`. Chart.js reads its colors from those variables and redraws when the theme changes.
- No horizontal scrolling at 375px width; the table scrolls inside its own container.

### 6. Verification

1. **Numbers**: run `config.js` + `regression.js` once in Node via `vm`, printing the train/test indices and every k's result. Redo forward selection with pandas `get_dummies(drop_first=True)` + `numpy.linalg.lstsq` on **the same indices** and compare.
   Done when: every k's feature order, coefficients (rtol 1e-9), RMSE, and R² all match.
2. **UI**: serve with `python3 -m http.server` and check each of: no console errors, light and dark themes, 375px width, click and hover linking, switching presets, loading a custom URL, and visible errors for a bad URL and a bad target.

With seed 42 and 20% test, the presets must give: 50 Startups (target `Profit`) best k = 2 (R&D Spend, Marketing Spend), test RMSE 9,304.12, R² 0.9464; Boston Housing (target `medv`) best k = 11, test RMSE 4.05, R² 0.7462. A different result means the split or selection logic has changed.

## Conventions

- Use classic `<script>` tags rather than ES modules, so opening `index.html` via `file://` still works.
- Load Chart.js from cdnjs pinned to an exact version (currently 4.4.1).
- Insert CSV-derived strings (column names, category levels) into the DOM only via `textContent` / `createElement`, to prevent XSS.
- UI copy and code comments are in Traditional Chinese.
- Deploy: push to GitHub → Settings → Pages → `main` branch, `/ (root)`.
