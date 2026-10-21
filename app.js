// 畫面：讀設定 → 抓資料 → 計算 → 畫圖與表格；選取的模型（方法 + k）會同步到模型卡、圖表、表格與下方細節
const $ = id => document.getElementById(id);
const charts = {};
const state = { header: null, rows: null, result: null, sel: null, model: null, inputs: {}, showAllResiduals: false };
const darkQuery = matchMedia('(prefers-color-scheme: dark)');
// 圖表顏色依這裡的順序對應 --series-1..5，不隨顯示的方法數改變
const METHODS = [
  { key: 'forward', name: 'Forward SFS' },
  { key: 'rfe', name: 'RFE（Backward）' },
  { key: 'lasso', name: 'Lasso' },
  { key: 'enet', name: 'ElasticNet' },
  { key: 'exhaustive', name: 'Exhaustive' },
];
const RESIDUAL_ROWS = 10;

Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;

$('data-url').value = CONFIG.dataUrl;
$('test-ratio').value = Math.round(CONFIG.testRatio * 100);
$('seed').value = CONFIG.seed;
$('preset').replaceChildren(
  ...CONFIG.presets.map((p, i) => new Option(p.name, i)),
  new Option('自訂 CSV 網址', 'custom'),
);
$('method').replaceChildren(
  ...METHODS.map(m => new Option(m.name, m.key)),
  new Option('比較全部方法', 'all'),
);

$('load-form').addEventListener('submit', e => {
  e.preventDefault();
  load();
});
$('preset').addEventListener('change', onPresetChange);
$('data-url').addEventListener('input', syncPreset);
for (const id of ['target', 'test-ratio', 'seed']) $(id).addEventListener('change', analyze);
$('method').addEventListener('change', onMethodChange);
$('back-to-best').addEventListener('click', () => {
  const b = modeBest();
  select(b.method, b.k);
});
$('copy-eq').addEventListener('click', copyEquation);
$('predict-inputs').addEventListener('input', onPredictInput);
$('resid-toggle').addEventListener('click', () => {
  state.showAllResiduals = !state.showAllResiduals;
  renderResiduals();
});
$('theme-toggle').addEventListener('click', toggleTheme);
darkQuery.addEventListener('change', onThemeChange);
updateThemeIcon();
load();

// ---------- 資料流程 ----------

// preferTarget：切換題目時指定的 target，優先於目前的選擇
async function load(preferTarget) {
  const url = $('data-url').value.trim();
  syncPreset();
  setBusy(true);
  try {
    const res = await fetch(url).catch(() => {
      throw new Error('無法下載資料。請確認網址正確，且伺服器允許跨來源（CORS）存取。');
    });
    if (!res.ok) throw new Error(`下載失敗（HTTP ${res.status}），請確認網址是否正確。`);
    const { header, rows } = parseCSV(await res.text());
    fillTargets(header, rows, preferTarget);
    Object.assign(state, { header, rows });
    const preset = CONFIG.presets.find(p => p.dataUrl === url);
    $('source-link').href = url;
    $('source-link').textContent = decodeURIComponent(url.split('/').pop()) || url;
    $('source-credit').textContent = preset?.source ? `（${preset.source}）` : '';
    analyze();
  } catch (err) {
    showError(err.message);
  } finally {
    setBusy(false);
  }
}

// 選預設題目：帶入它的網址與 target 後重新載入；選「自訂」則交給使用者輸入網址
function onPresetChange() {
  const p = CONFIG.presets[$('preset').value];
  if (!p) {
    $('data-url').focus();
    $('data-url').select();
    return;
  }
  $('data-url').value = p.dataUrl;
  load(p.target);
}

// 下拉選單跟著網址欄：網址屬於某個題目就選它，否則顯示「自訂」
function syncPreset() {
  const i = CONFIG.presets.findIndex(p => p.dataUrl === $('data-url').value.trim());
  $('preset').value = i < 0 ? 'custom' : i;
}

// target 下拉只列數值欄；優先順序：題目指定的 target → 目前選擇 → CONFIG.target → 最後一個數值欄
function fillTargets(header, rows, preferTarget) {
  const { isNum } = inspect(header, rows);
  const numeric = header.filter((_, c) => isNum[c]);
  if (!numeric.length) throw new Error('資料中沒有數值欄位可當預測目標。');
  const current = [preferTarget, $('target').value, CONFIG.target].find(n => numeric.includes(n)) ?? numeric[numeric.length - 1];
  $('target').replaceChildren(...numeric.map(n => new Option(n, n, false, n === current)));
  $('target').disabled = false;
}

// 五種方法一次算完：切換方法或比較全部時不必重算
function analyze() {
  if (!state.rows) return;
  const ratio = readInt('test-ratio', 5, 50, Math.round(CONFIG.testRatio * 100)) / 100;
  const seed = readInt('seed', 0, 4294967295, CONFIG.seed);
  const target = $('target').value;
  try {
    const data = buildDataset(state.header, state.rows, target);
    const split = trainTestSplit(data.y.length, ratio, seed);
    const pick = (a, idx) => idx.map(i => a[i]);
    const Xtr = pick(data.X, split.train), ytr = pick(data.y, split.train);
    const Xte = pick(data.X, split.test), yte = pick(data.y, split.test);
    const cols = independentFeatures(Xtr, ytr, data.names.length);
    if (!cols.length) throw new Error('訓練資料太少，無法擬合任何模型。');
    const methods = METHODS.map(m => m.key).filter(m => m !== 'exhaustive' || cols.length <= EXHAUSTIVE_MAX_P);
    const runs = {};
    for (const m of methods) {
      const steps = evaluateSubsets(SELECTORS[m](Xtr, ytr, cols), Xtr, ytr, Xte, yte);
      runs[m] = { steps, best: steps.reduce((b, s) => (s.rmse < b.rmse ? s : b)) };
    }
    state.result = { data, target, split, Xtr, ytr, Xte, yte, cols, methods, runs };
    state.inputs = {};
    syncMethodOptions();
    setStatus();
    render();
  } catch (err) {
    showError(err.message);
  }
}

// 特徵太多時 Exhaustive 停用；原本選它就改回 Forward SFS
function syncMethodOptions() {
  const ok = state.result.methods.includes('exhaustive');
  const opt = $('method').querySelector('option[value="exhaustive"]');
  opt.disabled = !ok;
  opt.textContent = ok ? 'Exhaustive' : `Exhaustive（特徵超過 ${EXHAUSTIVE_MAX_P} 個時停用）`;
  if (!ok && $('method').value === 'exhaustive') $('method').value = 'forward';
}

// 讀取整數輸入框，超出範圍就夾回並寫回輸入框
function readInt(id, min, max, fallback) {
  const v = Math.round(Number($(id).value));
  const clamped = Number.isFinite(v) && $(id).value !== '' ? Math.min(max, Math.max(min, v)) : fallback;
  $(id).value = clamped;
  return clamped;
}

function onMethodChange() {
  if (!state.result || $('results').hidden) return;
  const b = modeBest();
  showModel(b.method, b.k);
  renderTable();
  drawCharts();
}

const isCompare = () => $('method').value === 'all';
const modeMethods = () => (isCompare() ? state.result.methods : [$('method').value]);
const methodName = key => METHODS.find(m => m.key === key).name;
const stepOf = (method, k) => state.result.runs[method].steps.find(s => s.k === k);
// 不同方法選到同一組特徵時，RMSE 只差在浮點誤差，視為相同
const sameRmse = (a, b) => Math.abs(a - b) <= 1e-9 * Math.abs(b);

// 目前模式的最佳模型：單一方法取該方法 test RMSE 最低的 k；比較全部時取所有方法中最低者（相同時取排前面的方法）
function modeBest() {
  let best = null;
  for (const m of modeMethods()) {
    const b = state.result.runs[m].best;
    if (!best || (b.rmse < best.rmse && !sameRmse(b.rmse, best.rmse))) best = { method: m, k: b.k, rmse: b.rmse };
  }
  return best;
}

function select(method, k) {
  showModel(method, k);
  markSelected();
  Object.values(charts).forEach(c => c.draw());
}

// 切換選取的模型：算出它的統計量與測試集預測，更新模型卡與下方細節
function showModel(method, k) {
  const { Xtr, ytr, Xte } = state.result;
  const s = stepOf(method, k);
  state.sel = { method, k };
  state.model = { ...s, method, stats: olsSummary(Xtr, ytr, s.features), pred: predict(Xte, s.features, s.coef) };
  renderModel();
  renderCoefs();
  renderPredictor();
  renderResiduals();
}

// ---------- 狀態 ----------

function setBusy(busy) {
  $('load-btn').disabled = busy;
  $('preset').disabled = busy;
  $('load-btn').textContent = busy ? '載入中…' : '載入資料';
  $('results').classList.toggle('is-loading', busy);
  if (busy && $('results').hidden) setStatus('loading', '正在載入資料…');
}

function setStatus(kind = '', msg = '') {
  $('status').className = `status ${kind}`;
  $('status').textContent = msg;
  $('status').hidden = !kind;
}

function showError(msg) {
  setStatus('error', msg);
  $('results').hidden = true;
  $('data-info').hidden = true;
}

// ---------- 畫面 ----------

function render() {
  renderDataInfo();
  const b = modeBest();
  showModel(b.method, b.k);
  renderTable();
  $('results').hidden = false;
  drawCharts();
}

function renderDataInfo() {
  const { data, target, split, cols } = state.result;
  const parts = [
    `${data.y.length} 筆有效資料`,
    `訓練 ${split.train.length} 筆`,
    `測試 ${split.test.length} 筆`,
    `候選特徵 ${data.names.length} 個`,
  ];
  if (data.dropped) parts.push(`略過 ${data.dropped} 筆缺值`);
  if (cols.length < data.names.length) parts.push(`${data.names.length - cols.length} 個特徵因線性相依未納入`);
  $('data-counts').textContent = parts.join(' · ');

  const encoded = new Map(data.encoded.map(e => [e.column, e]));
  $('column-list').replaceChildren(...state.header.map(h => {
    const e = encoded.get(h);
    const type = h === target ? '預測目標'
      : e ? `類別 → ${e.levels.length - 1} 個 dummy，基準 ${e.baseline}`
      : '數值';
    return el('li', h === target ? 'chip target' : 'chip', el('b', '', h), el('span', '', type));
  }));
  $('data-info').hidden = false;
}

function renderModel() {
  const { data, target, yte, runs } = state.result;
  const s = state.model, { stats } = s;
  const best = modeBest();
  const isBest = best.method === s.method && best.k === s.k;
  const steps = runs[s.method].steps, full = steps[steps.length - 1];
  const label = `${methodName(s.method)}，k = ${s.k}`;

  $('model-title').textContent = isBest ? `最佳模型：${label}` : `${label} 的模型`;
  $('model-title').classList.toggle('is-best', isBest);
  $('back-to-best').hidden = isBest;
  $('back-to-best').textContent = `回到最佳：${methodName(best.method)}，k = ${best.k}`;

  $('m-rmse').textContent = fmtNum(s.rmse);
  $('m-r2').textContent = s.r2.toFixed(4);
  $('m-mae').textContent = fmtNum(mae(yte, s.pred));
  $('m-adj').textContent = dash(stats.adjR2, v => v.toFixed(4));
  $('m-f').replaceChildren(dash(stats.F, fmtNum), el('span', 'dd-sub', fmtP(stats.Fp, 'p ')));
  $('m-k').textContent = `${s.k} / ${data.names.length}`;
  $('m-vs-all-label').textContent = `RMSE 較全部 ${full.k} 個特徵`;
  if (s.k === full.k) {
    $('m-vs-all').className = '';
    $('m-vs-all').textContent = '—';
  } else {
    setDelta($('m-vs-all'), ((s.rmse - full.rmse) / full.rmse) * 100, v => `${Math.abs(v).toFixed(1)}%`);
  }

  const [b0, ...bs] = s.coef;
  const terms = bs.map((b, i) => el('span', 'term',
    `${b < 0 ? '−' : '+'} `, el('b', '', fmtCoef(Math.abs(b))), ` × ${data.names[s.features[i]]}`));
  $('equation').replaceChildren(
    el('span', 'eq-target', target), ' = ', el('span', 'term', b0 < 0 ? '−' : '', el('b', '', fmtCoef(Math.abs(b0)))),
    ...terms.flatMap(t => [' ', t]),
  );
}

function renderTable() {
  const compare = isCompare();
  $('table-title').textContent = compare ? '各方法比較' : '每個 k 的模型';
  $('table-hint').textContent = compare
    ? '每格為 Test RMSE 與 R²，點選即可查看該模型'
    : '外框標籤為比上一個 k 新加入的特徵';
  $('steps-body').closest('table').classList.toggle('compare', compare);
  if (compare) renderCompareTable();
  else renderStepsTable();
  markSelected();
}

function setHead(...cols) {
  $('steps-head').replaceChildren(el('tr', '', ...cols.map(([text, cls]) => {
    const th = el('th', cls, text);
    th.scope = 'col';
    return th;
  })));
}

function renderStepsTable() {
  const { data, runs } = state.result;
  const method = $('method').value;
  const { steps, best } = runs[method];
  setHead(['k'], ['選到的特徵'], ['Test RMSE', 'num'], ['較上一步', 'num'], ['Test R²', 'num']);
  $('steps-body').replaceChildren(...steps.map((s, i) => {
    const k = el('td', 'k-cell', String(s.k));
    if (s === best) k.append(el('span', 'best-tag', '最佳'));
    const prev = new Set(i ? steps[i - 1].features : []);
    const feats = el('td', 'features', ...s.features.map(f =>
      el('span', prev.has(f) ? 'feat' : 'feat new', data.names[f])));
    const delta = el('td', 'num', '—');
    if (i) setDelta(delta, s.rmse - steps[i - 1].rmse, v => fmtNum(Math.abs(v)));

    const tr = el('tr', '', k, feats, el('td', 'num', fmtNum(s.rmse)), delta, el('td', 'num', s.r2.toFixed(4)));
    tr.tabIndex = 0;
    tr.dataset.k = s.k;
    tr.addEventListener('click', () => select(method, s.k));
    tr.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        select(method, s.k);
      }
    });
    tr.addEventListener('mouseenter', () => peek(s.k));
    tr.addEventListener('focus', () => peek(s.k));
    tr.addEventListener('mouseleave', () => peek(null));
    tr.addEventListener('blur', () => peek(null));
    return tr;
  }));
}

// 比較表：列為 k、欄為方法；每個方法 RMSE 最低的格子加粗，全部方法中最低者標「最佳」
function renderCompareTable() {
  const { runs, methods } = state.result;
  const best = modeBest();
  const maxK = Math.max(...methods.map(m => runs[m].steps[runs[m].steps.length - 1].k));
  setHead(['k'], ...methods.map(m => [methodName(m), 'num']));
  $('steps-body').replaceChildren(...Array.from({ length: maxK }, (_, i) => {
    const k = i + 1;
    const cells = methods.map(m => {
      const s = stepOf(m, k);
      if (!s) return el('td', 'num', '—');
      const btn = el('button', 'cell-btn', el('b', '', fmtNum(s.rmse)), el('span', '', `R² ${s.r2.toFixed(4)}`));
      const isBest = sameRmse(s.rmse, best.rmse);
      if (isBest) btn.append(el('span', 'best-tag', '最佳'));
      btn.type = 'button';
      btn.setAttribute('aria-label', `${methodName(m)}，k = ${k}：Test RMSE ${fmtNum(s.rmse)}，R² ${s.r2.toFixed(4)}${isBest ? '，最佳' : ''}`);
      btn.addEventListener('click', () => select(m, k));
      const td = el('td', s === runs[m].best ? 'num cell method-best' : 'num cell', btn);
      td.dataset.method = m;
      td.dataset.k = k;
      return td;
    });
    const tr = el('tr', '', el('td', 'k-cell', String(k)), ...cells);
    tr.addEventListener('mouseenter', () => peek(k));
    tr.addEventListener('focusin', () => peek(k));
    tr.addEventListener('mouseleave', () => peek(null));
    tr.addEventListener('focusout', () => peek(null));
    return tr;
  }));
}

function markSelected() {
  const { method, k } = state.sel;
  for (const tr of $('steps-body').rows) {
    if (isCompare()) {
      for (const td of tr.cells) {
        const on = td.dataset.method === method && Number(td.dataset.k) === k;
        td.classList.toggle('selected', on);
        const btn = td.querySelector('button');
        if (on) btn.setAttribute('aria-current', 'true');
        else btn?.removeAttribute('aria-current');
      }
      continue;
    }
    const on = Number(tr.dataset.k) === k;
    tr.classList.toggle('selected', on);
    if (on) tr.setAttribute('aria-current', 'true');
    else tr.removeAttribute('aria-current');
  }
}

// 滑過表格列時，在兩張圖上同步顯示該 k 的 tooltip（比較模式會列出每個方法）
function peek(k) {
  for (const c of Object.values(charts)) {
    const active = k == null ? [] : c.data.datasets
      .map((ds, d) => ({ datasetIndex: d, index: k - 1 }))
      .filter(a => c.isDatasetVisible(a.datasetIndex) && c.data.datasets[a.datasetIndex].data[a.index] != null);
    c.setActiveElements(active);
    c.tooltip.setActiveElements(active, { x: 0, y: 0 });
    c.update('none');
  }
}

function renderCoefs() {
  const { data, ytr } = state.result;
  const { features, stats } = state.model;
  $('coef-note').textContent = `訓練集 n = ${ytr.length}，殘差自由度 ${stats.df}，殘差標準誤 σ̂ = ${dash(stats.sigma, fmtCoef)}`;
  $('coef-body').replaceChildren(...stats.terms.map((t, i) => {
    const name = el('th', '', i ? data.names[features[i - 1]] : '截距（Intercept）');
    name.scope = 'row';
    const p = el('td', 'num', fmtP(t.p));
    const stars = t.p < 0.001 ? '***' : t.p < 0.01 ? '**' : t.p < 0.05 ? '*' : '';
    p.append(el('span', 'sig', stars));
    const vif = el('td', 'num', t.vif == null ? '—' : dash(t.vif, v => v.toFixed(2)));
    if (t.vif >= 5) vif.prepend(el('span', 'flag', '偏高'));
    return el('tr', '', name,
      el('td', 'num', fmtSigned(t.coef)),
      el('td', 'num', dash(t.se, fmtCoef)),
      el('td', 'num', dash(t.t, v => fmtSigned(v, x => x.toFixed(3)))),
      p,
      el('td', 'num', Number.isFinite(t.lo) ? `[${fmtSigned(t.lo)}, ${fmtSigned(t.hi)}]` : '—'),
      vif);
  }));
}

// 即時預測：模型用到的每個原始欄位一個輸入（數值用滑桿 + 數字框、類別用下拉）；輸入值在切換模型時保留
function renderPredictor() {
  const { data, target, Xtr } = state.result;
  const used = new Set(state.model.features.map(f => data.sources[f].column));
  const encoded = new Map(data.encoded.map(e => [e.column, e]));
  $('predict-inputs').replaceChildren(...state.header.filter(h => used.has(h)).map((h, n) => {
    const id = `predict-${n}`;
    const label = el('label', 'field-label', h);
    label.htmlFor = id;
    const enc = encoded.get(h);
    if (enc) {
      if (!enc.levels.includes(state.inputs[h])) state.inputs[h] = enc.baseline;
      const select = el('select');
      select.id = id;
      select.dataset.column = h;
      select.append(...enc.levels.map(lv =>
        new Option(lv === enc.baseline ? `${lv}（基準）` : lv, lv, false, lv === state.inputs[h])));
      return el('div', 'predict-field', label, select);
    }
    const vals = column(Xtr, data.sources.findIndex(s => s.column === h));
    const lo = Math.min(...vals), hi = Math.max(...vals);
    const step = sliderStep(vals, lo, hi);
    if (!(h in state.inputs)) state.inputs[h] = roundTo(mean(vals), step);
    const range = el('input');
    Object.assign(range, { type: 'range', min: roundTo(Math.floor(lo / step) * step, step), max: roundTo(Math.ceil(hi / step) * step, step), step, value: state.inputs[h] });
    range.dataset.column = h;
    range.setAttribute('aria-label', `${h} 滑桿`);
    const num = el('input');
    Object.assign(num, { type: 'number', id, step: 'any', value: state.inputs[h] });
    Object.assign(num.dataset, { column: h, lo, hi });
    return el('div', 'predict-field', label, el('div', 'predict-row', range, num));
  }));
  $('predict-label').textContent = `${target} 預測值`;
  updatePrediction();
}

// 滑桿約分成 200 段，步長取 10 的次方；整數欄步長至少為 1
function sliderStep(vals, lo, hi) {
  const step = hi > lo ? 10 ** Math.floor(Math.log10((hi - lo) / 200)) : 1;
  return vals.every(Number.isInteger) ? Math.max(1, step) : step;
}

function roundTo(v, step) {
  return Number((Math.round(v / step) * step).toPrecision(12));
}

// 同一欄的滑桿與數字框互相同步
function onPredictInput(e) {
  const h = e.target.dataset.column;
  if (h == null) return;
  state.inputs[h] = e.target.value;
  if (e.target.value !== '') {
    for (const node of $('predict-inputs').querySelectorAll('input')) {
      if (node !== e.target && node.dataset.column === h) node.value = e.target.value;
    }
  }
  updatePrediction();
}

function updatePrediction() {
  const { data } = state.result;
  const { method, k, features, coef } = state.model;
  let y = coef[0];
  features.forEach((f, i) => {
    const { column: h, level } = data.sources[f];
    const v = state.inputs[h];
    y += coef[i + 1] * (level == null ? (v === '' ? NaN : Number(v)) : v === level ? 1 : 0);
  });
  const outside = [...$('predict-inputs').querySelectorAll('input[type="number"]')]
    .some(n => n.value !== '' && (Number(n.value) < Number(n.dataset.lo) || Number(n.value) > Number(n.dataset.hi)));
  $('predict-value').textContent = dash(y, v => fmtSigned(v, fmtNum));
  $('predict-note').textContent = `以「${methodName(method)}，k = ${k}」的模型計算`
    + (outside ? '；有輸入超出訓練資料範圍，屬於外插，預測較不可靠' : '');
}

// 測試集殘差（實際 − 預測）依 |殘差| 排序；|殘差| > 2σ̂（訓練集殘差標準誤）標為高殘差
function renderResiduals() {
  const { data, split, yte } = state.result;
  const { pred, stats } = state.model;
  const limit = 2 * stats.sigma;
  const rows = yte.map((y, i) => ({ id: data.ids[split.test[i]] + 1, y, p: pred[i], r: y - pred[i] }))
    .sort((a, b) => Math.abs(b.r) - Math.abs(a.r));
  const high = rows.filter(r => Math.abs(r.r) > limit).length;
  $('resid-note').textContent = Number.isFinite(limit)
    ? `依 |殘差| 由大到小排序；|殘差| > 2σ̂ = ${fmtCoef(limit)} 標為高殘差，共 ${high} 筆`
    : '依 |殘差| 由大到小排序';
  const shown = state.showAllResiduals ? rows : rows.slice(0, RESIDUAL_ROWS);
  $('resid-body').replaceChildren(...shown.map((r, i) => {
    const isHigh = Math.abs(r.r) > limit;
    return el('tr', isHigh ? 'high' : '',
      el('td', 'num', String(i + 1)),
      el('td', '', `第 ${r.id} 筆`),
      el('td', 'num', fmtSigned(r.y, fmtNum)),
      el('td', 'num', fmtSigned(r.p, fmtNum)),
      el('td', 'num', `${r.r > 0 ? '+' : ''}${fmtSigned(r.r, fmtNum)}`),
      el('td', 'num', r.y ? `${Math.abs((r.r / r.y) * 100).toFixed(1)}%` : '—'),
      el('td', '', isHigh ? el('span', 'flag', '高殘差') : el('span', 'muted', '正常')));
  }));
  $('resid-toggle').hidden = rows.length <= RESIDUAL_ROWS;
  $('resid-toggle').textContent = state.showAllResiduals ? `只顯示前 ${RESIDUAL_ROWS} 筆` : `顯示全部 ${rows.length} 筆`;
}

// RMSE 變化：下降為好（▼ 綠），上升為差（▲ 紅）
function setDelta(node, d, fmt) {
  node.classList.remove('good', 'bad');
  if (d) node.classList.add(d < 0 ? 'good' : 'bad');
  node.textContent = `${d < 0 ? '▼ ' : d > 0 ? '▲ ' : ''}${fmt(d)}`;
}

async function copyEquation() {
  const btn = $('copy-eq');
  try {
    await navigator.clipboard.writeText($('equation').textContent);
    btn.textContent = '已複製';
  } catch {
    btn.textContent = '複製失敗';
  }
  setTimeout(() => (btn.textContent = '複製'), 1500);
}

// ---------- 圖表 ----------

function drawCharts() {
  const { data, runs } = state.result;
  const css = getComputedStyle(document.documentElement);
  const color = v => css.getPropertyValue(v).trim();
  const compare = isCompare();
  const series = modeMethods();
  const maxK = Math.max(...series.map(m => runs[m].steps[runs[m].steps.length - 1].k));
  const labels = Array.from({ length: maxK }, (_, i) => i + 1);
  const best = modeBest();
  const bestSeries = series.indexOf(best.method), bestIdx = best.k - 1;
  const ties = series.filter(m => sameRmse(runs[m].best.rmse, best.rmse)).length;
  const bestText = !compare ? `最佳 k = ${best.k}`
    : ties > 1 ? `最佳 k = ${best.k}（${ties} 個方法相同）`
    : `最佳：${methodName(best.method)} k = ${best.k}`;
  const specs = [
    { key: 'rmse', label: 'Test RMSE', fmt: fmtNum, below: true },
    { key: 'r2', label: 'Test R²', fmt: v => v.toFixed(4), below: false },
  ];
  $('charts').classList.toggle('compare', compare);

  for (const { key, label, fmt, below } of specs) {
    charts[key]?.destroy();
    const datasets = series.map((m, d) => {
      const c = color(compare ? `--series-${METHODS.findIndex(x => x.key === m) + 1}` : '--series-1');
      const isBest = i => d === bestSeries && i === bestIdx;
      return {
        label: compare ? methodName(m) : label,
        data: labels.map(k => stepOf(m, k)?.[key] ?? null),
        borderColor: c,
        backgroundColor: c,
        borderWidth: 2,
        borderCapStyle: 'round',
        borderJoinStyle: 'round',
        pointRadius: labels.map((_, i) => (isBest(i) ? 6 : compare ? 3 : 4)),
        pointHoverRadius: 7,
        pointHitRadius: 12,
        // 比較模式的系列色已用到橘色，最佳點改用文字色外框標示
        pointBackgroundColor: labels.map((_, i) => (isBest(i) && !compare ? color('--best') : c)),
        pointBorderColor: labels.map((_, i) => color(isBest(i) && compare ? '--text' : '--surface')),
        pointBorderWidth: 2,
        pointHoverBorderWidth: 2,
      };
    });
    charts[key] = new Chart($(`${key}-chart`), {
      type: 'line',
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        layout: { padding: { top: 28, right: 8 } },
        interaction: { mode: 'index', intersect: false },
        onClick(evt, _, chart) {
          if (compare && evt.y >= chart.legend.top) return; // 點的是圖例
          const mode = compare ? 'nearest' : 'index';
          const [hit] = chart.getElementsAtEventForMode(evt, mode, { intersect: false }, false);
          if (hit) select(series[hit.datasetIndex], hit.index + 1);
        },
        plugins: {
          legend: {
            display: compare,
            position: 'bottom',
            labels: {
              color: color('--text-2'),
              usePointStyle: true,
              pointStyle: 'line',
              boxWidth: 20,
              padding: 12,
              // 預設會拿點的外框色（背景色）畫圖例線條，改用系列色
              generateLabels: chart => Chart.defaults.plugins.legend.labels.generateLabels(chart)
                .map(l => ({ ...l, strokeStyle: chart.data.datasets[l.datasetIndex].borderColor, lineWidth: 2 })),
            },
            // 兩張圖一起切換顯示的方法
            onClick(_, item) {
              for (const c of Object.values(charts)) {
                c.setDatasetVisibility(item.datasetIndex, !c.isDatasetVisible(item.datasetIndex));
                c.update();
              }
            },
          },
          tooltip: {
            backgroundColor: color('--surface'),
            borderColor: color('--border-strong'),
            borderWidth: 1,
            cornerRadius: 8,
            padding: 10,
            displayColors: compare,
            usePointStyle: true,
            titleColor: color('--text-2'),
            titleFont: { weight: 'normal' },
            bodyColor: color('--text'),
            bodyFont: { weight: '600', size: compare ? 13 : 14 },
            footerColor: color('--text-2'),
            footerFont: { weight: 'normal' },
            callbacks: {
              title: items => `k = ${items[0].label}`,
              label: item => (compare ? `${fmt(item.parsed.y)}  ${item.dataset.label}` : `${label}  ${fmt(item.parsed.y)}`),
              labelColor: item => ({ borderColor: item.dataset.borderColor, backgroundColor: item.dataset.borderColor, borderWidth: 2 }),
              labelPointStyle: () => ({ pointStyle: 'line', rotation: 0 }),
              footer: items => (compare ? '' : stepOf(series[0], items[0].dataIndex + 1).features.map(i => data.names[i]).join(', ')),
            },
          },
        },
        scales: {
          x: {
            title: { display: true, text: '特徵數 k', color: color('--text-2') },
            grid: { color: color('--grid') },
            border: { color: color('--axis') },
            ticks: { color: color('--muted') },
          },
          y: {
            grace: '15%',
            grid: { color: color('--grid') },
            border: { color: color('--axis') },
            ticks: { color: color('--muted') },
          },
        },
      },
      plugins: [markers(bestSeries, bestIdx, bestText, below, {
        band: color('--accent-wash'),
        crosshair: color('--axis'),
        text: color('--text'),
      })],
    });
  }
}

// 選取的 k 畫底色帶、滑鼠所在的 k 畫垂直線、最佳點加文字標籤
// （最佳點在 RMSE 圖是最低點、在 R² 圖是最高點，below 讓標籤放在離開折線的那一側）
function markers(bestSeries, bestIdx, text, below, colors) {
  const xOf = (chart, i) => chart.scales.x.getPixelForValue(i);
  return {
    id: 'markers',
    beforeDatasetsDraw(chart) {
      const { ctx, chartArea: a } = chart;
      const sel = state.sel.k - 1;
      const w = Math.min(40, ((a.right - a.left) / chart.data.labels.length) * 0.6);
      ctx.save();
      const x0 = Math.max(a.left, xOf(chart, sel) - w / 2);
      const x1 = Math.min(a.right, xOf(chart, sel) + w / 2);
      ctx.fillStyle = colors.band;
      ctx.fillRect(x0, a.top, x1 - x0, a.bottom - a.top);
      const [hover] = chart.tooltip.getActiveElements();
      if (hover) {
        const x = Math.round(xOf(chart, hover.index)) + 0.5;
        ctx.strokeStyle = colors.crosshair;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, a.top);
        ctx.lineTo(x, a.bottom);
        ctx.stroke();
      }
      ctx.restore();
    },
    afterDatasetsDraw(chart) {
      if (!chart.isDatasetVisible(bestSeries)) return;
      const pt = chart.getDatasetMeta(bestSeries).data[bestIdx];
      const { ctx, chartArea: a } = chart;
      ctx.save();
      ctx.fillStyle = colors.text;
      ctx.font = `600 12px ${Chart.defaults.font.family}`;
      ctx.textAlign = 'center';
      // 標籤靠近左右邊緣時往內移，避免被裁掉
      const half = ctx.measureText(text).width / 2;
      const x = Math.min(Math.max(pt.x, a.left + half), a.right - half);
      ctx.fillText(text, x, below ? pt.y + 24 : pt.y - 14);
      ctx.restore();
    },
  };
}

// ---------- 主題 ----------

function effectiveTheme() {
  return document.documentElement.dataset.theme || (darkQuery.matches ? 'dark' : 'light');
}

function toggleTheme() {
  const next = effectiveTheme() === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  try {
    localStorage.setItem('theme', next);
  } catch {}
  onThemeChange();
}

function onThemeChange() {
  updateThemeIcon();
  if (state.result && !$('results').hidden) drawCharts();
}

function updateThemeIcon() {
  const dark = effectiveTheme() === 'dark';
  $('theme-toggle').dataset.mode = dark ? 'dark' : 'light';
  $('theme-toggle').setAttribute('aria-label', dark ? '切換為淺色模式' : '切換為深色模式');
}

// ---------- 小工具 ----------

function el(tag, className, ...children) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  node.append(...children);
  return node;
}

function fmtNum(v) {
  return v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function fmtCoef(v) {
  return Math.abs(v) >= 1000 ? fmtNum(v) : String(Number(v.toPrecision(4)));
}

// 負號一律用 −（U+2212）
function fmtSigned(v, fmt = fmtCoef) {
  return `${v < 0 ? '−' : ''}${fmt(Math.abs(v))}`;
}

// 無法計算（例如殘差自由度為 0）時顯示 —
function dash(v, fmt) {
  return Number.isFinite(v) ? fmt(v) : '—';
}

function fmtP(p, prefix = '') {
  if (!Number.isFinite(p)) return '—';
  return p < 0.0001 ? `${prefix}< 0.0001` : `${prefix}${prefix ? '= ' : ''}${p.toFixed(4)}`;
}
