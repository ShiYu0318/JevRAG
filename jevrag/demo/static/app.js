"use strict";

const VERDICTS = ["sufficient", "partial", "conflicting", "insufficient"];
const VERDICT_ZH = { sufficient: "充足", partial: "部分", conflicting: "衝突", insufficient: "不足" };
const ACTION_ZH = {
  answer: "直接作答",
  answer_with_both_sides: "並列兩方說法",
  answer_partial_and_flag_gap: "回答已知部分並說明缺口",
  abstain_or_search_web: "拒答或改查網路",
  correct_premise: "指出錯誤前提",
};
const ROLE_ZH = {
  gold: "gold", hard_negative: "困難負例", easy_negative: "簡單負例",
  perturbed_gold: "擾動後的 gold", injection: "含注入",
};
const METHOD_ZH = {
  "pointwise:noul": "逐段：是否含答案",
  "pointwise:score": "逐段：相關度等級",
  "pointwise:noul_then_score": "逐段：先看含答案再看相關度",
  "packed:noul_then_score": "打包成一次請求",
};
const BACKEND_LABEL = { jev: "Jev", "laya-ml": "Laya 多語", "laya-en": "Laya 英文", kev4b: "Kev-4B", mock: "Mock" };

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x) => (x == null || Number.isNaN(x) ? "–" : x.toFixed(3));

function backendColor(name) {
  const n = String(name).toLowerCase();
  if (n.startsWith("jev")) return "var(--b-jev)";
  if (n.startsWith("laya-ml")) return "var(--b-laya-ml)";
  if (n.startsWith("laya-en")) return "var(--b-laya-en)";
  if (n.startsWith("kev")) return "var(--b-kev)";
  if (n.startsWith("bge")) return "var(--b-bge)";
  return "var(--b-other)";
}

async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}

/* ---------------------------------------------------------------- tabs */
function selectTab(which) {
  for (const t of ["results", "judge"]) {
    $(`#tab-${t}`).setAttribute("aria-selected", String(t === which));
    $(`#panel-${t}`).hidden = t !== which;
  }
}
$("#tab-results").addEventListener("click", () => selectTab("results"));
$("#tab-judge").addEventListener("click", () => selectTab("judge"));

/* ---------------------------------------------------------------- charts */
function bars(rows, { max = 1 } = {}) {
  // rows: [{label, value, lo, hi, color}]
  return `<div class="bars">${rows.map((r) => {
    const w = Math.max(0, Math.min(1, r.value / max)) * 100;
    const ci = r.lo != null ? `<span class="ci" style="left:${(r.lo / max) * 100}%;width:${((r.hi - r.lo) / max) * 100}%"></span>` : "";
    return `<span class="label"><span class="swatch" style="background:${r.color}"></span>${esc(r.label)}</span>
      <span class="track"><span class="fill" style="width:${w}%;background:${r.color}"></span>${ci}</span>
      <span class="val">${pct(r.value)}</span>`;
  }).join("")}</div>`;
}

function reliability(points, { title, color, ece }) {
  const s = 150, pad = 22;
  const x = (v) => pad + v * (s - pad - 6);
  const y = (v) => s - pad - v * (s - pad - 6);
  const total = points.reduce((a, p) => a + p.n, 0) || 1;
  const dots = points.map((p) => {
    const r = 2 + 8 * Math.sqrt(p.n / total);
    return `<circle cx="${x(p.conf)}" cy="${y(p.acc)}" r="${r}" fill="${color}" fill-opacity=".75"><title>信心 ${p.conf.toFixed(2)}，實際 ${p.acc.toFixed(2)}，n=${p.n}</title></circle>`;
  }).join("");
  return `<figure>
    <svg width="${s}" height="${s}" role="img" aria-label="${esc(title)} reliability diagram">
      <line x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}" stroke="currentColor" stroke-opacity=".25" stroke-dasharray="3 3"/>
      <line x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(0)}" stroke="currentColor" stroke-opacity=".35"/>
      <line x1="${x(0)}" y1="${y(0)}" x2="${x(0)}" y2="${y(1)}" stroke="currentColor" stroke-opacity=".35"/>
      ${dots}
      <text x="${x(0)}" y="${s - 6}">0</text><text x="${x(1) - 6}" y="${s - 6}">1</text>
      <text x="2" y="${y(1) + 4}">1</text>
      <text x="${x(0.5) - 12}" y="${s - 6}">信心</text>
      <text x="2" y="${y(0.5)}" transform="rotate(-90 8 ${y(0.5)})">實際比例</text>
    </svg>
    <figcaption>${esc(title)}<br>ECE ${pct(ece)}</figcaption>
  </figure>`;
}

function table(head, rows) {
  return `<div class="table-wrap"><table class="grid"><thead><tr>${head.map((h) => `<th>${esc(h)}</th>`).join("")}</tr></thead>
    <tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${typeof c === "number" ? pct(c) : esc(c)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}

function runMeta(runs) {
  return Object.entries(runs).map(([key, r]) => {
    const versions = Object.values(r.model_versions || {}).flatMap((v) => Object.keys(v)).join(", ");
    return `${esc(key)}：${esc(versions)}，${r.calls} 次呼叫，$${(r.cost_usd || 0).toFixed(4)}`;
  }).join("<br>");
}

/* ---------------------------------------------------------------- results */
const SECTIONS = [
  {
    id: "E1", title: "Passage 排序", ask: "在同一組 20 段候選中，各方法把含答案的段落排得多前面（nDCG@10，附 95% 信賴區間）？",
    render(runs) {
      const seen = new Map();
      for (const [key, r] of Object.entries(runs)) {
        for (const [m, v] of Object.entries(r.result.methods)) {
          const backend = key.split(" / ")[0];
          const tail = m.replace(/^[^:]+:/, "");
          const label = m === "rrf" ? "BM25 原始順序" : m === "bge-reranker" ? "bge-reranker"
            : `${BACKEND_LABEL[backend] || backend}：${METHOD_ZH[tail] || tail}`;
          if (!seen.has(label)) seen.set(label, { label, value: v["ndcg@10"].value, lo: v["ndcg@10"].lo, hi: v["ndcg@10"].hi, color: m === "rrf" ? "var(--b-other)" : m === "bge-reranker" ? "var(--b-bge)" : backendColor(backend) });
        }
      }
      return bars([...seen.values()].sort((a, b) => b.value - a.value));
    },
  },
  {
    id: "E2", title: "機率準不準", ask: "模型說「含答案的機率是 0.8」時，實際上是不是約八成真的含答案？點越貼近虛線越準；點的大小代表樣本數。",
    render(runs) {
      const figs = [];
      for (const [key, r] of Object.entries(runs)) {
        const backend = key.split(" / ")[0];
        const name = BACKEND_LABEL[backend] || backend;
        const raw = r.result.raw?.all_passages;
        if (raw) figs.push(reliability(raw.reliability, { title: `${name}：原始`, color: backendColor(backend), ece: raw.ece_width }));
        const pl = r.result.platt_scaled?.all_passages;
        if (pl) figs.push(reliability(pl.reliability, { title: `${name}：Platt 校準後`, color: backendColor(backend), ece: pl.ece_width }));
        const bge = r.result.baselines?.["bge-reranker"];
        if (bge && !figs.some((f) => f.includes("bge-reranker："))) {
          const b = bge.raw_sigmoid.all_passages;
          figs.push(reliability(b.reliability, { title: "bge-reranker：原始", color: "var(--b-bge)", ece: b.ece_width }));
        }
      }
      const rows = Object.entries(runs).map(([key, r]) => {
        const a = r.result.raw.all_passages, p = r.result.platt_scaled?.all_passages;
        return [key, a.auroc, a.ece_width, p ? p.ece_width : NaN, p ? p["coverage@risk0.05"] : NaN];
      });
      return `<div class="multiples">${figs.join("")}</div>` +
        table(["後端", "AUROC", "ECE 原始", "ECE Platt 後", "5% 風險下可作答比例"], rows);
    },
  },
  {
    id: "E3", title: "證據夠不夠", ask: "給一個問題和 3 段文字，模型能不能分出充足、部分、衝突、不足四種情況（macro-F1）？",
    render(runs) {
      const rows = [];
      let nli = null;
      for (const [key, r] of Object.entries(runs)) {
        const backend = key.split(" / ")[0];
        const name = BACKEND_LABEL[backend] || backend;
        const res = r.result;
        rows.push({ label: `${name}：整組一次判斷`, ...ci(res.set.macro_f1), color: backendColor(backend) });
        if (res.two_stage?.noisy_or) rows.push({ label: `${name}：兩階段`, ...ci(res.two_stage.noisy_or.macro_f1), color: backendColor(backend) });
        if (res.nli && !nli) nli = { label: "NLI（有給 gold 答案）", ...ci(res.nli.macro_f1), color: "var(--b-other)" };
        if (!rows.some((x) => x.label === "多數類別")) rows.push({ label: "多數類別", ...ci(res.majority.macro_f1), color: "var(--muted)" });
      }
      if (nli) rows.push(nli);
      const conds = ["S", "I-hard", "I-easy", "P", "C"];
      const acc = Object.entries(runs).map(([key, r]) => [key, ...conds.map((c) => r.result.set.accuracy_by_condition[c] ?? NaN)]);
      return bars(rows.sort((a, b) => b.value - a.value)) +
        `<p class="meta">整組一次判斷，各情境的準確率</p>` + table(["後端", ...conds], acc);
    },
  },
  {
    id: "E4", title: "繁簡與指令語言", ask: "同一題改成簡體、或把指令改成中文，判斷會不會變（翻轉率越低越穩）？",
    render(runs) {
      const rows = Object.entries(runs).map(([key, r]) => {
        const s = r.result.settings;
        return [key, s["zh-Hant/en"].macro_f1, s["zh-Hans/en"].flip_rate_vs_base.value, s["zh-Hant/zh"].flip_rate_vs_base.value, s["zh-Hant/en"].mean_confidence];
      });
      return table(["後端", "macro-F1（繁體／英文指令）", "繁→簡翻轉率", "英→中指令翻轉率", "平均信心"], rows);
    },
  },
  {
    id: "E0", title: "後端特性", ask: "每個後端多快、多穩定？同一題重問會不會變，換一種問法會不會變？",
    render(runs) {
      const rows = Object.entries(runs).map(([key, r]) => {
        const lat = r.result.latency.filter((x) => x.n_questions === 1);
        const short = lat.find((x) => x.state_chars === 384) || lat[0];
        const c = r.result.consistency;
        const para = Object.values(c.paraphrase_flip_rates);
        return [key, short ? `${Math.round(short.p50_ms)} ms` : "–", c.repeat_flip_rate_max, Math.max(...para),
          r.result.packed_vs_separate.pointwise.gold_top1, r.result.packed_vs_separate.packed.gold_top1];
      });
      return table(["後端", "延遲 p50（384 字、1 題）", "重問翻轉率", "換說法翻轉率", "gold 排第一（逐段）", "gold 排第一（打包）"], rows);
    },
  },
];

function ci(x) { return { value: x.value, lo: x.lo, hi: x.hi }; }

async function loadResults() {
  const panel = $("#panel-results");
  let data;
  try { data = await getJSON("/api/results"); } catch (e) {
    panel.innerHTML = `<p class="error">讀不到結果：${esc(e.message)}</p>`;
    return;
  }
  panel.innerHTML = SECTIONS.map((s) => {
    const runs = data.runs[s.id];
    let body;
    if (!runs) body = `<p class="empty">還沒有 ${s.id} 的結果。跑完 <code>python -m jevrag run ${s.id} ...</code> 之後重新整理這頁。</p>`;
    else {
      try { body = s.render(runs) + `<p class="meta">${runMeta(runs)}</p>`; }
      catch (e) { body = `<p class="error">這個實驗的結果格式和介面不相符：${esc(e.message)}</p>`; }
    }
    return `<section class="exp" id="sec-${s.id}"><h2>${s.id} ${esc(s.title)}</h2><p class="ask">${esc(s.ask)}</p>${body}</section>`;
  }).join("");

  const run = $("#running");
  if (data.running.length) {
    run.hidden = false;
    run.innerHTML = `<p><span class="pulse"></span>執行中</p>` + data.running.map((r) =>
      `<p>${esc(r.run_id.replace(/-\d{8}T\d{6}$/, ""))}：${r.calls.toLocaleString()} 次呼叫</p>`).join("");
  } else run.hidden = true;
}

async function loadMilestones() {
  const list = $("#milestones");
  const items = await getJSON("/api/milestones").catch(() => []);
  if (!items.length) {
    list.innerHTML = `<li><span class="t">還沒有里程碑檔</span><div class="d">在 docs/milestones.json 列出各階段後，這裡會顯示進度。</div></li>`;
    return;
  }
  list.innerHTML = items.map((m) => `<li class="${esc(m.status)}">
    <div class="t">${esc(m.title)}</div>
    <div class="d">${esc(m.when || "")}${m.note ? `<br>${esc(m.note)}` : ""}</div></li>`).join("");
}

/* ---------------------------------------------------------------- judge */
let current = null;
let available = {};

async function loadBackends() {
  available = await getJSON("/api/backends").catch(() => ({}));
  $("#backends").innerHTML = Object.entries(available).map(([name, a]) => `
    <label class="${a.ok ? "" : "off"}" title="${esc(a.reason || (a.remote ? "遠端 API，會產生費用" : "本機"))}">
      <input type="checkbox" value="${esc(name)}" ${a.ok && name !== "mock" ? "checked" : ""} ${a.ok ? "" : "disabled"}>
      <span class="dot" style="background:${backendColor(name)}"></span>${esc(BACKEND_LABEL[name] || name)}${a.remote ? "（付費）" : ""}
    </label>`).join("");
}

function highlight(text, inst, reveal) {
  let html = esc(text);
  if (!reveal) return { html, marked: false };
  const marks = [];
  for (const a of (inst.answers || []).flatMap((x) => x.split("；"))) if (a) marks.push([a, ""]);
  if (inst.perturbation?.to) marks.push([inst.perturbation.to, "changed"]);
  let marked = false;
  for (const [needle, cls] of marks.sort((x, y) => y[0].length - x[0].length)) {
    const e = esc(needle);
    if (html.includes(e)) {
      html = html.split(e).join(`<mark class="${cls}">${e}</mark>`);
      marked = true;
    }
  }
  return { html, marked };
}

function renderInstance(inst, results) {
  const reveal = $("#reveal").checked;
  const answer = reveal && inst.answers ? `<p class="answer">答案：${esc(inst.answers.join("、"))}　標籤：${esc(VERDICT_ZH[inst.label] || inst.label)}（${esc(inst.condition)}）</p>` : `<p class="answer">${inst.id ? esc(inst.id) : "自訂內容"}</p>`;
  const slips = inst.contexts.map((c, i) => {
    const probs = (results || []).filter((r) => !r.error).map((r) => {
      const p = r.passages[i];
      return `<div class="prob"><span>${esc(BACKEND_LABEL[r.backend] || r.backend)}</span>
        <span class="track"><span class="fill" style="width:${p.answers * 100}%;background:${backendColor(r.backend)}"></span></span>
        <span>${pct(p.answers)}</span></div>
        ${p.injection >= 0.5 ? `<div class="prob"><span></span><span class="flag">疑似指令注入 ${pct(p.injection)}</span><span></span></div>` : ""}`;
    }).join("");
    const role = reveal && c.role ? `<b>${esc(ROLE_ZH[c.role] || c.role)}</b>` : `段落 ${i + 1}`;
    const { html, marked } = highlight(c.text, inst, reveal);
    return `<div class="slip">
      <div><div class="role">${role}</div><div class="text${marked ? "" : " clamp"}" id="t${i}">${html}</div>
        ${marked ? "" : `<button class="more" data-i="${i}">展開全文</button>`}</div>
      <div class="probs">${probs ? `<div class="role">含答案的機率</div>${probs}` : `<div class="role">按「開始判斷」後顯示各模型的機率</div>`}</div>
    </div>`;
  }).join("");
  $("#instance").innerHTML = `<p class="question">${esc(inst.question)}</p>${answer}<div class="evidence">${slips}</div>`;
  for (const b of document.querySelectorAll(".more")) {
    b.addEventListener("click", () => {
      const t = $(`#t${b.dataset.i}`);
      const open = t.classList.toggle("clamp");
      b.textContent = open ? "展開全文" : "收合";
    });
  }
  renderVerdicts(results);
}

function renderVerdicts(results) {
  const box = $("#verdicts");
  if (!results) { box.innerHTML = ""; return; }
  box.innerHTML = `<h2 style="margin-top:2rem">整組段落的判定</h2><div class="verdicts">${results.map((r) => {
    if (r.error) return `<div class="verdict-row"><strong>${esc(BACKEND_LABEL[r.backend] || r.backend)}</strong><span class="error">${esc(r.error)}</span></div>`;
    const segs = VERDICTS.map((v) => `<span style="width:${(r.verdict[v] || 0) * 100}%;background:var(--${v})" title="${VERDICT_ZH[v]} ${pct(r.verdict[v] || 0)}"></span>`).join("");
    const top = VERDICTS.reduce((a, b) => ((r.verdict[a] || 0) >= (r.verdict[b] || 0) ? a : b));
    return `<div class="verdict-row"><strong>${esc(BACKEND_LABEL[r.backend] || r.backend)}</strong><div class="stack">${segs}</div>
      <div class="verdict-meta">最可能：<strong>${VERDICT_ZH[top]} ${pct(r.verdict[top])}</strong>，決策：<strong>${esc(ACTION_ZH[r.action] || r.action)}</strong>，${r.cached ? "來自快取" : `${r.latency_ms} ms`}${r.cost_usd ? `，$${r.cost_usd.toFixed(5)}` : ""}</div></div>`;
  }).join("")}</div>
  <div class="legend">${VERDICTS.map((v) => `<span><span class="swatch" style="background:var(--${v})"></span>${VERDICT_ZH[v]}</span>`).join("")}</div>`;
}

async function draw(seed) {
  $("#judge-note").textContent = "";
  try {
    const q = new URLSearchParams({ split: $("#split").value, condition: $("#condition").value, n: "1" });
    if (seed != null && seed !== "" && typeof seed !== "object") q.set("seed", seed);
    const [inst] = await getJSON(`/api/instances?${q}`);
    if (!inst) { $("#instance").innerHTML = `<p class="empty">這個切分沒有資料。先用 python -m jevrag build 建好資料集。</p>`; return; }
    current = { inst, results: null };
    renderInstance(inst, null);
    $("#judge").disabled = false;
  } catch (e) { $("#instance").innerHTML = `<p class="error">抽題失敗：${esc(e.message)}</p>`; }
}

async function judge() {
  if (!current) return;
  const backends = [...document.querySelectorAll("#backends input:checked")].map((i) => i.value);
  if (!backends.length) { $("#judge-note").textContent = "至少勾選一個模型。"; return; }
  $("#judge").disabled = true;
  $("#judge-note").textContent = "判斷中…";
  try {
    const results = await getJSON("/api/judge", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ question: current.inst.question, passages: current.inst.contexts.map((c) => c.text), backends, lang: $("#lang").value }),
    });
    current.results = results;
    renderInstance(current.inst, results);
    const cost = results.reduce((a, r) => a + (r.cost_usd || 0), 0);
    $("#judge-note").textContent = cost ? `這次花費 $${cost.toFixed(5)}` : "";
  } catch (e) { $("#judge-note").textContent = `判斷失敗：${e.message}`; }
  $("#judge").disabled = false;
}

$("#draw").addEventListener("click", () => draw());
$("#judge").addEventListener("click", judge);
$("#reveal").addEventListener("change", () => current && renderInstance(current.inst, current.results));
$("#toggle-custom").addEventListener("click", () => { $("#custom").hidden = !$("#custom").hidden; });
$("#use-custom").addEventListener("click", () => {
  const question = $("#custom-q").value.trim();
  const passages = $("#custom-p").value.split(/\n\s*\n/).map((s) => s.trim()).filter(Boolean);
  if (!question || !passages.length) { $("#judge-note").textContent = "填入問題和至少一段文字。"; return; }
  current = { inst: { question, contexts: passages.map((text) => ({ text })) }, results: null };
  renderInstance(current.inst, null);
  $("#judge").disabled = false;
});

/* Deep links for preparing a demo, e.g. #judge?condition=C&seed=3&backends=laya-ml,jev&run=1 */
async function openFromHash() {
  if (!location.hash.startsWith("#judge")) return;
  selectTab("judge");
  const p = new URLSearchParams(location.hash.split("?")[1] || "");
  if (p.get("split")) $("#split").value = p.get("split");
  if (p.get("condition") !== null) $("#condition").value = p.get("condition") || "";
  if (p.get("lang")) $("#lang").value = p.get("lang");
  if (p.get("reveal") === "1") $("#reveal").checked = true;
  await backendsReady;
  if (p.get("backends")) {
    const want = new Set(p.get("backends").split(","));
    for (const i of document.querySelectorAll("#backends input")) i.checked = !i.disabled && want.has(i.value);
  }
  await draw(p.get("seed"));
  if (p.get("run") === "1") await judge();
}

loadMilestones();
loadResults();
const backendsReady = loadBackends();
openFromHash();
setInterval(loadResults, 30000);
