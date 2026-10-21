// 畫面：讀設定 → 抓資料 → 計算 → 畫圖與表格；選取的 k 會同步到模型卡、圖表與表格
const $ = id => document.getElementById(id);
const charts = {};
const state = { header: null, rows: null, result: null, selectedK: null };
const darkQuery = matchMedia('(prefers-color-scheme: dark)');

Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;

$('data-url').value = CONFIG.dataUrl;
$('test-ratio').value = Math.round(CONFIG.testRatio * 100);
$('seed').value = CONFIG.seed;

$('load-form').addEventListener('submit', e => {
  e.preventDefault();
  load();
});
for (const id of ['target', 'test-ratio', 'seed']) $(id).addEventListener('change', analyze);
$('back-to-best').addEventListener('click', () => select(state.result.best.k));
$('copy-eq').addEventListener('click', copyEquation);
$('theme-toggle').addEventListener('click', toggleTheme);
darkQuery.addEventListener('change', onThemeChange);
updateThemeIcon();
load();

// ---------- 資料流程 ----------

async function load() {
  const url = $('data-url').value.trim();
  setBusy(true);
  try {
    const res = await fetch(url).catch(() => {
      throw new Error('無法下載資料。請確認網址正確，且伺服器允許跨來源（CORS）存取。');
    });
    if (!res.ok) throw new Error(`下載失敗（HTTP ${res.status}），請確認網址是否正確。`);
    const { header, rows } = parseCSV(await res.text());
    fillTargets(header, rows);
    Object.assign(state, { header, rows });
    $('source-link').href = url;
    $('source-link').textContent = decodeURIComponent(url.split('/').pop()) || url;
    analyze();
  } catch (err) {
    showError(err.message);
  } finally {
    setBusy(false);
  }
}

// target 下拉只列數值欄；優先保留目前選擇，其次 CONFIG.target，否則取最後一個數值欄
function fillTargets(header, rows) {
  const { isNum } = inspect(header, rows);
  const numeric = header.filter((_, c) => isNum[c]);
  if (!numeric.length) throw new Error('資料中沒有數值欄位可當預測目標。');
  const current = [$('target').value, CONFIG.target].find(n => numeric.includes(n)) ?? numeric[numeric.length - 1];
  $('target').replaceChildren(...numeric.map(n => new Option(n, n, false, n === current)));
  $('target').disabled = false;
}

function analyze() {
  if (!state.rows) return;
  const ratio = readInt('test-ratio', 5, 50, Math.round(CONFIG.testRatio * 100)) / 100;
  const seed = readInt('seed', 0, 4294967295, CONFIG.seed);
  const target = $('target').value;
  try {
    const data = buildDataset(state.header, state.rows, target);
    const split = trainTestSplit(data.y.length, ratio, seed);
    const pick = (a, idx) => idx.map(i => a[i]);
    const steps = forwardSelection(
      pick(data.X, split.train), pick(data.y, split.train),
      pick(data.X, split.test), pick(data.y, split.test),
      data.names.length,
    );
    if (!steps.length) throw new Error('訓練資料太少，無法擬合任何模型。');
    const best = steps.reduce((b, s) => (s.rmse < b.rmse ? s : b));
    state.result = { data, target, split, steps, best };
    state.selectedK = best.k;
    setStatus();
    render();
  } catch (err) {
    showError(err.message);
  }
}

// 讀取整數輸入框，超出範圍就夾回並寫回輸入框
function readInt(id, min, max, fallback) {
  const v = Math.round(Number($(id).value));
  const clamped = Number.isFinite(v) && $(id).value !== '' ? Math.min(max, Math.max(min, v)) : fallback;
  $(id).value = clamped;
  return clamped;
}

function select(k) {
  state.selectedK = k;
  renderModel();
  markSelectedRow();
  Object.values(charts).forEach(c => c.draw());
}

// ---------- 狀態 ----------

function setBusy(busy) {
  $('load-btn').disabled = busy;
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
  renderTable();
  renderModel();
  $('results').hidden = false;
  drawCharts();
}

function renderDataInfo() {
  const { data, target, split, steps } = state.result;
  const parts = [
    `${data.y.length} 筆有效資料`,
    `訓練 ${split.train.length} 筆`,
    `測試 ${split.test.length} 筆`,
    `候選特徵 ${data.names.length} 個`,
  ];
  if (data.dropped) parts.push(`略過 ${data.dropped} 筆缺值`);
  if (steps.length < data.names.length) parts.push(`${data.names.length - steps.length} 個特徵因線性相依未納入`);
  $('data-counts').textContent = parts.join(' · ');

  const encoded = new Map(data.encoded.map(e => [e.column, e]));
  $('column-list').replaceChildren(...state.header.map(h => {
    const e = encoded.get(h);
    const type = h === target ? '預測目標'
      : e ? `類別 → ${e.levels - 1} 個 dummy，基準 ${e.baseline}`
      : '數值';
    return el('li', h === target ? 'chip target' : 'chip', el('b', '', h), el('span', '', type));
  }));
  $('data-info').hidden = false;
}

function renderModel() {
  const { data, target, steps, best } = state.result;
  const s = steps.find(x => x.k === state.selectedK);
  const full = steps[steps.length - 1];

  $('model-title').textContent = s === best ? `最佳模型：k = ${s.k}` : `k = ${s.k} 的模型`;
  $('model-title').classList.toggle('is-best', s === best);
  $('back-to-best').hidden = s === best;
  $('back-to-best').textContent = `回到最佳 k = ${best.k}`;

  $('m-rmse').textContent = fmtNum(s.rmse);
  $('m-r2').textContent = s.r2.toFixed(4);
  $('m-k').textContent = `${s.k} / ${data.names.length}`;
  $('m-vs-all-label').textContent = `RMSE 較全部 ${full.k} 個特徵`;
  if (s === full) {
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
  const { data, steps, best } = state.result;
  $('steps-body').replaceChildren(...steps.map((s, i) => {
    const k = el('td', 'k-cell', String(s.k));
    if (s === best) k.append(el('span', 'best-tag', '最佳'));
    const feats = el('td', 'features', ...s.features.map((f, j) =>
      el('span', j === s.features.length - 1 ? 'feat new' : 'feat', data.names[f])));
    const delta = el('td', 'num', '—');
    if (i) setDelta(delta, s.rmse - steps[i - 1].rmse, v => fmtNum(Math.abs(v)));

    const tr = el('tr', '', k, feats, el('td', 'num', fmtNum(s.rmse)), delta, el('td', 'num', s.r2.toFixed(4)));
    tr.tabIndex = 0;
    tr.dataset.k = s.k;
    tr.addEventListener('click', () => select(s.k));
    tr.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        select(s.k);
      }
    });
    tr.addEventListener('mouseenter', () => peek(i));
    tr.addEventListener('focus', () => peek(i));
    tr.addEventListener('mouseleave', () => peek(null));
    tr.addEventListener('blur', () => peek(null));
    return tr;
  }));
  markSelectedRow();
}

function markSelectedRow() {
  for (const tr of $('steps-body').rows) {
    const on = Number(tr.dataset.k) === state.selectedK;
    tr.classList.toggle('selected', on);
    if (on) tr.setAttribute('aria-current', 'true');
    else tr.removeAttribute('aria-current');
  }
}

// 滑過表格列時，在兩張圖上同步顯示該 k 的 tooltip
function peek(index) {
  const active = index == null ? [] : [{ datasetIndex: 0, index }];
  for (const c of Object.values(charts)) {
    c.setActiveElements(active);
    c.tooltip.setActiveElements(active, { x: 0, y: 0 });
    c.update('none');
  }
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
  const { data, steps, best } = state.result;
  const css = getComputedStyle(document.documentElement);
  const color = v => css.getPropertyValue(v).trim();
  const bestIdx = steps.indexOf(best);
  const specs = [
    { key: 'rmse', label: 'Test RMSE', fmt: fmtNum, below: true },
    { key: 'r2', label: 'Test R²', fmt: v => v.toFixed(4), below: false },
  ];

  for (const { key, label, fmt, below } of specs) {
    charts[key]?.destroy();
    charts[key] = new Chart($(`${key}-chart`), {
      type: 'line',
      data: {
        labels: steps.map(s => s.k),
        datasets: [{
          label,
          data: steps.map(s => s[key]),
          borderColor: color('--series-1'),
          borderWidth: 2,
          borderCapStyle: 'round',
          borderJoinStyle: 'round',
          pointRadius: steps.map((_, i) => (i === bestIdx ? 6 : 4)),
          pointHoverRadius: 7,
          pointHitRadius: 12,
          pointBackgroundColor: steps.map((_, i) => color(i === bestIdx ? '--best' : '--series-1')),
          pointBorderColor: color('--surface'),
          pointBorderWidth: 2,
          pointHoverBorderWidth: 2,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        layout: { padding: { top: 28, right: 8 } },
        interaction: { mode: 'index', intersect: false },
        onClick(evt, _, chart) {
          const [hit] = chart.getElementsAtEventForMode(evt, 'index', { intersect: false }, false);
          if (hit) select(steps[hit.index].k);
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: color('--surface'),
            borderColor: color('--border-strong'),
            borderWidth: 1,
            cornerRadius: 8,
            padding: 10,
            displayColors: false,
            titleColor: color('--text-2'),
            titleFont: { weight: 'normal' },
            bodyColor: color('--text'),
            bodyFont: { weight: '600', size: 14 },
            footerColor: color('--text-2'),
            footerFont: { weight: 'normal' },
            callbacks: {
              title: items => `k = ${items[0].label}`,
              label: item => `${label}  ${fmt(item.parsed.y)}`,
              footer: items => steps[items[0].dataIndex].features.map(i => data.names[i]).join(', '),
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
      plugins: [markers(bestIdx, `最佳 k = ${best.k}`, below, {
        band: color('--accent-wash'),
        crosshair: color('--axis'),
        text: color('--text'),
      })],
    });
  }
}

// 選取的 k 畫底色帶、滑鼠所在的 k 畫垂直線、最佳 k 加文字標籤
// （最佳點在 RMSE 圖是最低點、在 R² 圖是最高點，below 讓標籤放在離開折線的那一側）
function markers(bestIdx, text, below, colors) {
  const xOf = (chart, i) => chart.getDatasetMeta(0).data[i].x;
  return {
    id: 'markers',
    beforeDatasetsDraw(chart) {
      const { ctx, chartArea: a } = chart;
      const sel = state.result.steps.findIndex(s => s.k === state.selectedK);
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
      const pt = chart.getDatasetMeta(0).data[bestIdx];
      const { ctx } = chart;
      ctx.save();
      ctx.fillStyle = colors.text;
      ctx.font = `600 12px ${Chart.defaults.font.family}`;
      ctx.textAlign = bestIdx === 0 ? 'left' : bestIdx === chart.data.labels.length - 1 ? 'right' : 'center';
      ctx.fillText(text, pt.x, below ? pt.y + 24 : pt.y - 14);
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
