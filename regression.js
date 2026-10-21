// 純計算：CSV 解析、one-hot、資料切分、OLS、Forward Selection（不碰 DOM）

// 解析 CSV（支援雙引號欄位），第一列為欄位名稱
function parseCSV(text) {
  text = text.replace(/^﻿/, '');
  const rows = [];
  let row = [], field = '', inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inQuotes) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (c === '"') inQuotes = false;
      else field += c;
    } else if (c === '"') inQuotes = true;
    else if (c === ',') { row.push(field); field = ''; }
    else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++;
      row.push(field); rows.push(row); row = []; field = '';
    } else field += c;
  }
  if (field !== '' || row.length) { row.push(field); rows.push(row); }
  const header = rows.shift().map(h => h.trim());
  const body = rows.filter(r => r.some(v => v.trim() !== ''));
  return { header, rows: body.map(r => r.map(v => v.trim())) };
}

// 去掉有缺值的列，並判斷每一欄是否全為數值
function inspect(header, rows) {
  const complete = rows.filter(r => r.length === header.length && r.every(v => v !== ''));
  const isNum = header.map((_, c) => complete.every(r => Number.isFinite(Number(r[c]))));
  return { complete, isNum };
}

// 數值欄直接當特徵；其餘欄位視為類別，依字母排序後 one-hot 並 drop first
function buildDataset(header, rows, target) {
  const t = header.indexOf(target);
  if (t < 0) throw new Error(`找不到 target 欄位「${target}」，可用欄位：${header.join(', ')}`);
  const { complete, isNum } = inspect(header, rows);
  if (!isNum[t]) throw new Error(`target 欄位「${target}」不是數值欄`);

  const names = [], cols = [], encoded = [];
  header.forEach((h, c) => {
    if (c === t) return;
    if (isNum[c]) {
      names.push(h);
      cols.push(complete.map(r => Number(r[c])));
      return;
    }
    const levels = [...new Set(complete.map(r => r[c]))].sort();
    encoded.push({ column: h, baseline: levels[0], levels: levels.length });
    for (const lv of levels.slice(1)) {
      names.push(`${h}_${lv}`);
      cols.push(complete.map(r => (r[c] === lv ? 1 : 0)));
    }
  });
  if (!names.length) throw new Error('除了 target 之外沒有可用的特徵欄位');

  return {
    names,
    X: complete.map((_, i) => cols.map(col => col[i])),
    y: complete.map(r => Number(r[t])),
    dropped: rows.length - complete.length,
    encoded,
  };
}

// mulberry32：可重現的 32-bit 亂數產生器
function mulberry32(seed) {
  return function () {
    seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// 固定 seed 洗牌後切出測試集（測試筆數無條件進位，同 sklearn）
function trainTestSplit(n, testRatio, seed) {
  const rand = mulberry32(seed);
  const idx = [...Array(n).keys()];
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  const nTest = Math.ceil(n * testRatio);
  return { train: idx.slice(nTest), test: idx.slice(0, nTest) };
}

// Householder QR 解最小平方 min‖Ab − y‖；欄位線性相依時回傳 null
function lstsq(A, y) {
  const m = A.length, n = A[0].length;
  const R = A.map((r, i) => [...r, y[i]]); // 最後一欄是 y，跟著一起做反射
  for (let j = 0; j < n; j++) {
    let colNorm = 0, norm = 0;
    for (let i = 0; i < m; i++) colNorm += A[i][j] ** 2;
    for (let i = j; i < m; i++) norm += R[i][j] ** 2;
    norm = Math.sqrt(norm);
    if (norm <= 1e-10 * Math.sqrt(colNorm)) return null;

    const alpha = R[j][j] > 0 ? -norm : norm;
    const v = [];
    for (let i = j; i < m; i++) v.push(R[i][j]);
    v[0] -= alpha;
    const vv = v.reduce((s, x) => s + x * x, 0);
    for (let k = j; k <= n; k++) {
      let dot = 0;
      for (let i = j; i < m; i++) dot += v[i - j] * R[i][k];
      const f = (2 * dot) / vv;
      for (let i = j; i < m; i++) R[i][k] -= f * v[i - j];
    }
  }
  const coef = new Array(n);
  for (let j = n - 1; j >= 0; j--) {
    let s = R[j][n];
    for (let k = j + 1; k < n; k++) s -= R[j][k] * coef[k];
    coef[j] = s / R[j][j];
  }
  return coef;
}

// 設計矩陣：截距 + 指定的特徵欄
function design(X, cols) {
  return X.map(r => [1, ...cols.map(c => r[c])]);
}

function predict(X, cols, coef) {
  return design(X, cols).map(r => r.reduce((s, v, i) => s + v * coef[i], 0));
}

function sse(y, p) {
  return y.reduce((s, v, i) => s + (v - p[i]) ** 2, 0);
}

function rmse(y, p) {
  return Math.sqrt(sse(y, p) / y.length);
}

function r2(y, p) {
  const mean = y.reduce((s, v) => s + v, 0) / y.length;
  return 1 - sse(y, p) / y.reduce((s, v) => s + (v - mean) ** 2, 0);
}

// 每一步加入「讓訓練集 RSS 最小」的特徵；測試集只用來算 RMSE、R²
function forwardSelection(Xtr, ytr, Xte, yte, p) {
  const selected = [], remaining = [...Array(p).keys()], steps = [];
  while (remaining.length) {
    let best = null;
    for (const c of remaining) {
      const cols = [...selected, c];
      const coef = lstsq(design(Xtr, cols), ytr);
      if (!coef) continue;
      const rss = sse(ytr, predict(Xtr, cols, coef));
      if (!best || rss < best.rss) best = { c, coef, rss };
    }
    if (!best) break; // 剩下的特徵都與已選特徵線性相依
    selected.push(best.c);
    remaining.splice(remaining.indexOf(best.c), 1);
    const pred = predict(Xte, selected, best.coef);
    steps.push({
      k: selected.length,
      features: [...selected],
      coef: best.coef,
      rmse: rmse(yte, pred),
      r2: r2(yte, pred),
    });
  }
  return steps;
}
