// 純計算：CSV 解析、one-hot、資料切分、OLS 與統計檢定、五種特徵選擇方法（不碰 DOM）

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

// 去掉有缺值的列，並判斷每一欄是否全為數值；ids 為保留列在原始資料中的位置
function inspect(header, rows) {
  const ids = rows.map((_, i) => i).filter(i => rows[i].length === header.length && rows[i].every(v => v !== ''));
  const complete = ids.map(i => rows[i]);
  const isNum = header.map((_, c) => complete.every(r => Number.isFinite(Number(r[c]))));
  return { complete, isNum, ids };
}

// 數值欄直接當特徵；其餘欄位視為類別，依字母排序後 one-hot 並 drop first
function buildDataset(header, rows, target) {
  const t = header.indexOf(target);
  if (t < 0) throw new Error(`找不到 target 欄位「${target}」，可用欄位：${header.join(', ')}`);
  const { complete, isNum, ids } = inspect(header, rows);
  if (!isNum[t]) throw new Error(`target 欄位「${target}」不是數值欄`);

  // sources[i]：第 i 個特徵來自哪個原始欄位（dummy 另記類別值）
  const names = [], cols = [], sources = [], encoded = [];
  header.forEach((h, c) => {
    if (c === t) return;
    if (isNum[c]) {
      names.push(h);
      cols.push(complete.map(r => Number(r[c])));
      sources.push({ column: h, level: null });
      return;
    }
    const levels = [...new Set(complete.map(r => r[c]))].sort();
    encoded.push({ column: h, baseline: levels[0], levels });
    for (const lv of levels.slice(1)) {
      names.push(`${h}_${lv}`);
      cols.push(complete.map(r => (r[c] === lv ? 1 : 0)));
      sources.push({ column: h, level: lv });
    }
  });
  if (!names.length) throw new Error('除了 target 之外沒有可用的特徵欄位');

  return {
    names,
    X: complete.map((_, i) => cols.map(col => col[i])),
    y: complete.map(r => Number(r[t])),
    ids,
    dropped: rows.length - complete.length,
    sources,
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
// 回傳 coef、R（左上 n×n 為 QR 的上三角 R）與殘差平方和 rss
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
  let rss = 0;
  for (let i = n; i < m; i++) rss += R[i][n] ** 2;
  return { coef, R, rss };
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

function mae(y, p) {
  return y.reduce((s, v, i) => s + Math.abs(v - p[i]), 0) / y.length;
}

function mean(a) {
  return a.reduce((s, v) => s + v, 0) / a.length;
}

// 母體標準差（同 sklearn StandardScaler）
function std(a) {
  const m = mean(a);
  return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length);
}

function dot(a, b) {
  return a.reduce((s, v, i) => s + v * b[i], 0);
}

function column(X, c) {
  return X.map(r => r[c]);
}

// 訓練集 RSS；特徵線性相依時回傳 null
function trainRss(X, y, cols) {
  const fit = lstsq(design(X, cols), y);
  return fit && fit.rss;
}

// ---------- 特徵選擇：只看訓練集，回傳 k = 1..p 各自的特徵子集 ----------

// 依欄位順序保留與前面欄位（含截距）線性獨立的特徵，剔除常數欄、重複欄等
function independentFeatures(X, y, p) {
  const kept = [];
  for (let c = 0; c < p; c++) if (lstsq(design(X, [...kept, c]), y)) kept.push(c);
  return kept;
}

// Forward SFS：從只有截距開始，每一步加入「讓訓練集 RSS 最小」的特徵
function forwardSelection(X, y, cols) {
  const selected = [], remaining = [...cols], subsets = [];
  while (remaining.length) {
    let best = null;
    for (const c of remaining) {
      const rss = trainRss(X, y, [...selected, c]);
      if (rss != null && (!best || rss < best.rss)) best = { c, rss };
    }
    if (!best) break; // 剩下的特徵都與已選特徵線性相依
    selected.push(best.c);
    remaining.splice(remaining.indexOf(best.c), 1);
    subsets.push([...selected]);
  }
  return subsets;
}

// RFE（Backward）：從全部特徵開始，每次刪掉標準化係數 |b × sd| 最小的特徵
// （OLS 對平移縮放不變，等同 sklearn RFE 在 StandardScaler 後的資料上用 |coef| 排名）
function rfeSelection(X, y, cols) {
  const sd = new Map(cols.map(c => [c, std(column(X, c))]));
  const current = [...cols], subsets = [];
  while (current.length) {
    const fit = lstsq(design(X, current), y);
    if (!fit) break;
    subsets.unshift([...current]);
    const size = current.map((c, i) => Math.abs(fit.coef[i + 1] * sd.get(c)));
    current.splice(size.indexOf(Math.min(...size)), 1);
  }
  return subsets;
}

// Lasso（l1Ratio = 1）/ ElasticNet（l1Ratio = 0.5）：X 與 y 都標準化後，以座標下降法解正則化路徑
// λ 從 λ_max（全部係數為 0）到 1e-4 × λ_max 取 200 個對數等距點（同 sklearn lasso_path / enet_path），
// 每個 λ 最多迭代 1000 輪（同 sklearn 預設 max_iter；n 接近 p 時收斂很慢）；
// 特徵依「第一次出現非零係數」的先後排名，同一點才進入的以係數絕對值大者優先，前 k 名即 k 個特徵的子集
function pathSelection(X, y, cols, l1Ratio) {
  const n = X.length, p = cols.length;
  const z = cols.map(c => {
    const a = column(X, c), m = mean(a), s = std(a);
    return a.map(v => (v - m) / s);
  });
  const ym = mean(y), ys = std(y) || 1;
  const yz = y.map(v => (v - ym) / ys);
  const G = z.map(a => z.map(b => dot(a, b) / n));
  const g = z.map(a => dot(a, yz) / n);
  const lambdaMax = Math.max(...g.map(Math.abs)) / l1Ratio;
  const w = new Array(p).fill(0), entry = new Array(p).fill(Infinity), size = new Array(p).fill(0);
  for (let i = 0; i < 200; i++) {
    const lambda = lambdaMax * 1e-4 ** (i / 199);
    const l1 = lambda * l1Ratio, l2 = lambda * (1 - l1Ratio);
    for (let iter = 0; iter < 1000; iter++) {
      let maxDelta = 0;
      for (let j = 0; j < p; j++) {
        let rho = g[j];
        for (let k = 0; k < p; k++) if (k !== j) rho -= G[j][k] * w[k];
        const next = (Math.sign(rho) * Math.max(Math.abs(rho) - l1, 0)) / (G[j][j] + l2);
        maxDelta = Math.max(maxDelta, Math.abs(next - w[j]));
        w[j] = next;
      }
      if (maxDelta < 1e-13) break;
    }
    w.forEach((v, j) => {
      if (v !== 0 && entry[j] === Infinity) {
        entry[j] = i;
        size[j] = Math.abs(v);
      }
    });
  }
  const order = [...Array(p).keys()].sort((a, b) => (entry[a] - entry[b]) || (size[b] - size[a]) || (a - b));
  return order.map((_, k) => order.slice(0, k + 1).map(j => cols[j]));
}

// Exhaustive（Best Subset）：每個 k 列舉全部 C(p, k) 種組合，取訓練集 RSS 最小者
// 以深度優先列舉，子集每多一欄就增量更新 Cholesky 分解：RSS = Σ(y − ȳ)² − ‖L⁻¹ Xᵀy‖²
// （各欄先中心化並縮放成單位長度，Gram 矩陣即相關係數矩陣，數值較穩定）
const EXHAUSTIVE_MAX_P = 15;
function exhaustiveSelection(X, y, cols) {
  const unit = a => {
    const m = mean(a), c = a.map(v => v - m), norm = Math.sqrt(dot(c, c));
    return c.map(v => v / norm);
  };
  const z = cols.map(c => unit(column(X, c)));
  const ym = mean(y), yc = y.map(v => v - ym), syy = dot(yc, yc);
  const G = z.map(a => z.map(b => dot(a, b))), g = z.map(a => dot(a, yc));
  const best = [], subset = [], L = [], u = []; // L：目前子集的 Cholesky 下三角；u = L⁻¹ g_S
  (function visit(start, explained) {
    for (let j = start; j < cols.length; j++) {
      const m = subset.length, l = new Array(m);
      for (let i = 0; i < m; i++) {
        let s = G[subset[i]][j];
        for (let t = 0; t < i; t++) s -= L[i][t] * l[t];
        l[i] = s / L[i][i];
      }
      const d2 = G[j][j] - dot(l, l);
      if (d2 <= 1e-20) continue; // 與子集線性相依
      const d = Math.sqrt(d2);
      const uj = (g[j] - dot(l, u)) / d;
      const e = explained + uj * uj;
      subset.push(j);
      L.push([...l, d]);
      u.push(uj);
      if (!best[m] || syy - e < best[m].rss) best[m] = { rss: syy - e, cols: subset.map(i => cols[i]) };
      visit(j + 1, e);
      subset.pop();
      L.pop();
      u.pop();
    }
  })(0, 0);
  return best.map(b => b.cols);
}

const SELECTORS = {
  forward: forwardSelection,
  rfe: rfeSelection,
  lasso: (X, y, cols) => pathSelection(X, y, cols, 1),
  enet: (X, y, cols) => pathSelection(X, y, cols, 0.5),
  exhaustive: exhaustiveSelection,
};

// 每個子集以 OLS（含截距）在訓練集擬合，於測試集計算 RMSE、R²
function evaluateSubsets(subsets, Xtr, ytr, Xte, yte) {
  const steps = [];
  for (const features of subsets) {
    const fit = lstsq(design(Xtr, features), ytr);
    if (!fit) break;
    const pred = predict(Xte, features, fit.coef);
    steps.push({ k: features.length, features, coef: fit.coef, rmse: rmse(yte, pred), r2: r2(yte, pred) });
  }
  return steps;
}

// ---------- 統計推論（同 statsmodels OLS） ----------

// 所選模型在訓練集上的係數標準誤、t、p、95% 信賴區間、VIF，以及 Adjusted R²、F 檢定
function olsSummary(X, y, cols) {
  const n = X.length, k = cols.length, df = n - k - 1;
  const { coef, R, rss } = lstsq(design(X, cols), y);
  const ym = mean(y);
  const sst = y.reduce((s, v) => s + (v - ym) ** 2, 0);
  const sigma2 = df > 0 ? rss / df : NaN;
  const tCrit = df > 0 ? tQuantile(0.975, df) : NaN;
  // (AᵀA)⁻¹ = R⁻¹R⁻ᵀ，對角線即 R⁻¹ 每一列的平方和
  const Rinv = invUpper(R, k + 1);
  const terms = coef.map((b, i) => {
    const se = Math.sqrt(sigma2 * Rinv[i].reduce((s, v) => s + v * v, 0));
    const t = b / se;
    return { coef: b, se, t, p: tPValue(t, df), lo: b - tCrit * se, hi: b + tCrit * se, vif: i ? vif(X, cols, i - 1) : null };
  });
  const r2Train = 1 - rss / sst;
  const F = (sst - rss) / k / sigma2;
  return {
    terms, df,
    sigma: Math.sqrt(sigma2),
    adjR2: 1 - ((1 - r2Train) * (n - 1)) / df,
    F,
    Fp: fPValue(F, k, df),
  };
}

// VIF：把第 j 個特徵對其餘特徵（含截距）做回歸，VIF = 1 / (1 − R²_j)
function vif(X, cols, j) {
  const xj = column(X, cols[j]);
  const { rss } = lstsq(design(X, cols.filter((_, i) => i !== j)), xj);
  const m = mean(xj);
  return xj.reduce((s, v) => s + (v - m) ** 2, 0) / rss;
}

// 上三角矩陣 R（取左上 n×n）的反矩陣
function invUpper(R, n) {
  const inv = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let j = n - 1; j >= 0; j--) {
    inv[j][j] = 1 / R[j][j];
    for (let i = j - 1; i >= 0; i--) {
      let s = 0;
      for (let k = i + 1; k <= j; k++) s += R[i][k] * inv[k][j];
      inv[i][j] = -s / R[i][i];
    }
  }
  return inv;
}

// ln Γ(x)：Lanczos 近似（g = 7）
function lnGamma(x) {
  const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lnGamma(1 - x);
  x -= 1;
  let a = c[0];
  for (let i = 1; i < 9; i++) a += c[i] / (x + i);
  const t = x + 7.5;
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

// 正規化不完全 beta 函數 I_x(a, b)：連分數展開（Lentz 法）
function betaInc(x, a, b) {
  if (!(x > 0)) return x === 0 ? 0 : NaN;
  if (x >= 1) return 1;
  if (x > (a + 1) / (a + b + 2)) return 1 - betaInc(1 - x, b, a);
  const front = Math.exp(a * Math.log(x) + b * Math.log1p(-x) - lnGamma(a) - lnGamma(b) + lnGamma(a + b)) / a;
  let f = 1, c = 1, d = 0;
  for (let i = 0; i <= 500; i++) {
    const m = i >> 1;
    const num = i === 0 ? 1
      : i % 2 ? -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
      : (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m));
    d = 1 + num * d;
    d = 1 / (Math.abs(d) < 1e-300 ? 1e-300 : d);
    c = 1 + num / c;
    if (Math.abs(c) < 1e-300) c = 1e-300;
    f *= c * d;
    if (Math.abs(1 - c * d) < 1e-15) break;
  }
  return front * (f - 1);
}

// t 分佈雙尾 p-value
function tPValue(t, df) {
  return df > 0 ? betaInc(df / (df + t * t), df / 2, 0.5) : NaN;
}

// t 分佈分位數：在 [0, hi] 以二分法找雙尾 p = 2(1 − q) 的 t
function tQuantile(q, df) {
  const target = 2 * (1 - q);
  let lo = 0, hi = 1;
  while (tPValue(hi, df) > target) hi *= 2;
  for (let i = 0; i < 100; i++) {
    const mid = (lo + hi) / 2;
    if (tPValue(mid, df) > target) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

// F 分佈右尾 p-value
function fPValue(F, d1, d2) {
  return d2 > 0 ? betaInc(d2 / (d2 + d1 * F), d2 / 2, d1 / 2) : NaN;
}
