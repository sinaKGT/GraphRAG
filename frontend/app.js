/* GraphRAG UI — upload, graph view, Q&A with animated traversal replay. No build step. */
"use strict";

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch { /* not json */ }
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return r.json();
}

/* ------------------------------------------------------------------ colours */
const PALETTE = ["#60a5fa", "#f472b6", "#34d399", "#fbbf24", "#a78bfa", "#f87171",
                 "#2dd4bf", "#fb923c", "#c084fc", "#a3e635", "#38bdf8", "#fda4af"];
function typeColor(t) {
  let h = 0;
  for (const c of t || "") h = (h * 31 + c.charCodeAt(0)) >>> 0;
  return PALETTE[h % PALETTE.length];
}
const CSS = getComputedStyle(document.documentElement);
const C = (v) => CSS.getPropertyValue(v).trim();

/* ------------------------------------------------------------------ cytoscape */
const HL = ["faded", "hl-scored", "hl-accepted", "hl-rejected", "hl-picked", "hl-cited", "hl-edge"];

const cy = cytoscape({
  container: $("#cy"),
  minZoom: 0.05,
  maxZoom: 3,
  style: [
    { selector: "node", style: {
        "label": "data(label)", "color": "#cbd5e1", "font-size": 9, "min-zoomed-font-size": 7,
        "text-valign": "bottom", "text-margin-y": 3, "text-wrap": "ellipsis", "text-max-width": 140,
        "transition-property": "background-color, border-color, border-width, opacity, underlay-opacity",
        "transition-duration": "0.25s" } },
    { selector: "node[kind='entity']", style: {
        "background-color": "data(color)", "width": "mapData(degree, 0, 15, 12, 32)",
        "height": "mapData(degree, 0, 15, 12, 32)", "border-width": 0 } },
    { selector: "node[kind='community']", style: {
        "shape": "round-rectangle", "background-color": "#1e293b", "border-color": "#64748b",
        "border-width": 1.5, "width": "data(w)", "height": "data(w)", "font-size": 10,
        "text-valign": "center", "text-halign": "center", "text-margin-y": 0, "text-wrap": "wrap",
        "text-max-width": "data(tw)", "color": "#e2e8f0", "font-weight": 600 } },
    { selector: "edge", style: { "curve-style": "haystack", "opacity": 0.55,
        "transition-property": "line-color, width, opacity", "transition-duration": "0.25s" } },
    { selector: "edge[kind='related']", style: { "line-color": "#475569", "width": "mapData(weight, 0, 3, 0.6, 3)" } },
    { selector: "edge[kind='member']", style: { "line-color": "#334155", "width": 0.6, "line-style": "dashed", "opacity": 0.35 } },
    { selector: "edge[kind='child']", style: { "line-color": "#64748b", "width": 1.6, "line-style": "dashed" } },
    { selector: ".hidden-comm", style: { "display": "none" } },

    // ---- highlighting ----
    { selector: ".faded", style: { "opacity": 0.12, "text-opacity": 0 } },
    { selector: "node.hl-scored", style: { "opacity": 1, "text-opacity": 1, "border-width": 4, "border-color": C("--scored") } },
    { selector: "node.hl-rejected", style: { "opacity": 0.55, "text-opacity": 1, "border-width": 3, "border-color": C("--rejected"), "background-color": "#374151" } },
    { selector: "node.hl-accepted", style: { "opacity": 1, "text-opacity": 1, "border-width": 4, "border-color": C("--accepted"), "background-color": "#14532d" } },
    { selector: "node.hl-picked", style: { "opacity": 1, "text-opacity": 1, "border-width": 4, "border-color": C("--picked") } },
    { selector: "node.hl-cited", style: { "opacity": 1, "text-opacity": 1, "border-width": 5, "border-color": C("--cited"),
        "underlay-color": C("--cited"), "underlay-padding": 8, "underlay-opacity": 0.35, "underlay-shape": "ellipse" } },
    { selector: "edge.hl-edge", style: { "opacity": 1, "line-color": C("--picked"), "width": 2.5 } },
    { selector: "node:selected", style: { "border-width": 3, "border-color": "#fff" } },
  ],
});

/** Fit to a set of nodes without zooming in absurdly close on small selections. */
function focus(eles, padding = 70) {
  if (!eles || eles.empty()) return;
  const bb = eles.boundingBox();
  const w = cy.width() - 2 * padding, h = cy.height() - 2 * padding;
  const zoom = Math.min(1.4, Math.max(cy.minZoom(), Math.min(w / Math.max(bb.w, 1), h / Math.max(bb.h, 1))));
  cy.animate({ zoom, center: { eles }, duration: 550, easing: "ease-in-out-cubic" });
}

let graphData = { entities: [], communities: [] };
const baseLabel = new Map();

async function loadGraph() {
  const g = await api("/api/graph");
  graphData = g;
  const els = [];
  baseLabel.clear();
  for (const c of g.communities) {
    const lvl = c.level || 0;
    baseLabel.set(c.id, c.label);
    els.push({ data: { id: c.id, kind: "community", label: c.label, level: lvl, size: c.size,
                       w: 34 + lvl * 18 + Math.min(30, Math.sqrt(c.size || 1) * 4), tw: 90 + lvl * 30 } });
    if (c.parent) els.push({ data: { id: `ch:${c.id}`, source: c.id, target: c.parent, kind: "child" } });
  }
  for (const e of g.entities) {
    baseLabel.set(e.id, e.label);
    els.push({ data: { id: e.id, kind: "entity", label: e.label, type: e.type, degree: e.degree, color: typeColor(e.type) } });
    if (e.community) els.push({ data: { id: `m:${e.id}`, source: e.id, target: e.community, kind: "member" } });
  }
  for (const r of g.related) {
    els.push({ data: { id: `r:${r.source}|${r.type}|${r.target}`, source: r.source, target: r.target,
                       kind: "related", type: r.type, weight: r.weight } });
  }
  cy.elements().remove();
  cy.add(els);
  $("#empty").classList.toggle("hidden", g.entities.length > 0);
  applyCommunityToggle();
  if (!g.entities.length) return;
  cy.layout({
    name: "fcose", quality: "default", randomize: true, animate: false, nodeSeparation: 60,
    nodeRepulsion: () => 7000,
    idealEdgeLength: (e) => ({ member: 55, child: 110, related: 80 }[e.data("kind")] || 80),
    edgeElasticity: (e) => (e.data("kind") === "member" ? 0.6 : 0.45),
  }).run();
  cy.fit(undefined, 30);
}

function applyCommunityToggle() {
  const on = $("#show-comm").checked;
  cy.$("node[kind='community'], edge[kind='member'], edge[kind='child']").toggleClass("hidden-comm", !on);
}
$("#show-comm").addEventListener("change", applyCommunityToggle);
$("#fit").addEventListener("click", () => cy.fit(cy.elements(":visible"), 30));
$("#clear-hl").addEventListener("click", clearHighlight);

function clearHighlight() {
  replayToken++;
  cy.elements().removeClass(HL.join(" "));
  cy.nodes().forEach((n) => n.data("label", baseLabel.get(n.id()) ?? n.data("label")));
}

/* ------------------------------------------------------------------ node details */
cy.on("tap", "node", (ev) => showDetails(ev.target.id()));
cy.on("tap", (ev) => { if (ev.target === cy) $("#details").classList.add("hidden"); });

function showDetails(id) {
  const e = graphData.entities.find((x) => x.id === id);
  const c = graphData.communities.find((x) => x.id === id);
  let html = `<span class="close" onclick="this.parentElement.classList.add('hidden')">✕</span>`;
  if (e) {
    const comm = graphData.communities.find((x) => x.id === e.community);
    html += `<div class="k">Entity · ${esc(e.type)}</div><h3 style="color:var(--text);font-size:15px;margin:4px 0">${esc(e.label)}</h3>
             <div>${esc(e.description || "")}</div>
             ${comm ? `<div class="k" style="margin-top:8px">Community</div><div>${esc(comm.label)}</div>` : ""}
             <div class="muted" style="margin-top:6px">${e.degree} relationships</div>`;
  } else if (c) {
    html += `<div class="k">Community · level ${c.level}${c.is_top ? " · top" : ""} · ${c.size} entities</div>
             <h3 style="color:var(--text);font-size:15px;margin:4px 0">${esc(c.label)}</h3><div>${esc(c.summary || "")}</div>`;
  } else return;
  $("#details").innerHTML = html;
  $("#details").classList.remove("hidden");
}

/* ------------------------------------------------------------------ status / docs */
async function refreshStats() {
  try {
    const s = await api("/api/stats");
    $("#stats").textContent = `${s.documents} docs · ${s.entities} entities · ${s.relationships} relationships · ` +
      `${s.communities} communities (${s.top_communities} top, ${s.max_level + 1} levels)`;
  } catch { $("#stats").textContent = ""; }
}

async function refreshDocs() {
  const docs = await api("/api/documents");
  $("#docs").innerHTML = docs.map((d) =>
    `<li title="${esc(d.error || "")}"><span>${esc(d.name)}</span><span class="badge ${esc(d.status)}">${esc(d.status)}</span></li>`).join("");
}

async function checkHealth() {
  const el = $("#health");
  let ok = false;
  try {
    const h = await api("/api/health");
    ok = h.status === "ok";
    const name = (m) => (m.model || "").replace(/^models\//, "").replace(/\.gguf$/, "");
    const bad = ["neo4j", "llm", "embeddings", "embedding_space"].find((k) => h[k] && !h[k].ok);
    el.className = "health " + (ok ? "ok" : "bad");
    el.querySelector(".txt").textContent = ok
      ? `${h.llm.provider}: ${name(h.llm)} · ${h.embeddings.provider} embeddings`
      : `${bad}: ${(h[bad].error || "not ready").slice(0, 90)}`;
    el.title = JSON.stringify(h, null, 2);
  } catch (e) {
    el.className = "health bad";
    el.querySelector(".txt").textContent = "backend unreachable";
  }
  // keep polling until everything is up (e.g. local model still downloading/loading)
  setTimeout(checkHealth, ok ? 60000 : 8000);
}

/* ------------------------------------------------------------------ upload + job polling */
const drop = $("#drop");
$("#file").addEventListener("change", (e) => e.target.files[0] && upload(e.target.files[0]));
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && upload(e.dataTransfer.files[0]));

async function upload(file) {
  const fd = new FormData();
  fd.append("file", file);
  showJob({ filename: file.name, status: "uploading", stage: "uploading…" });
  try {
    const res = await api("/api/documents", { method: "POST", body: fd });
    if (!res.job) { showJob({ filename: file.name, status: "done", stage: res.message, done: 1, total: 1 }); return; }
    pollJob(res.job.id);
  } catch (e) {
    showJob({ filename: file.name, status: "failed", stage: e.message });
  } finally {
    $("#file").value = "";
  }
}

function showJob(j) {
  $("#job").classList.remove("hidden");
  $("#job-name").textContent = j.filename || (j.kind === "rebuild" ? "rebuild communities" : "");
  $("#job-status").innerHTML = `<span class="badge ${esc(j.status)}">${esc(j.status)}</span>`;
  $("#job-stage").textContent = j.error ? j.error : `${j.stage || ""}${j.total ? ` (${j.done}/${j.total})` : ""}`;
  const bar = $("#job-bar");
  const running = j.status === "running" || j.status === "queued" || j.status === "uploading";
  bar.parentElement.classList.toggle("indeterminate", running && !j.total);
  bar.style.width = j.status === "done" ? "100%" : j.total ? `${(100 * j.done) / j.total}%` : running ? "" : "0";
}

async function pollJob(id) {
  for (;;) {
    let j;
    try { j = await api(`/api/jobs/${id}`); } catch { await sleep(3000); continue; }
    showJob(j);
    if (j.status === "done" || j.status === "failed") break;
    await sleep(1500);
  }
  await Promise.all([refreshDocs(), refreshStats()]);
  await loadGraph();
}

$("#reset").addEventListener("click", async () => {
  if (!confirm("Delete the whole graph and all uploaded files?")) return;
  try {
    await api("/api/graph", { method: "DELETE" });
    $("#job").classList.add("hidden");
    await Promise.all([refreshDocs(), refreshStats(), loadGraph()]);
  } catch (e) { alert(e.message); }
});

/* ------------------------------------------------------------------ Q&A */
let lastResult = null;
let replayToken = 0;

$("#ask").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("#question").value.trim();
  if (q.length < 2) return;
  const btn = $("#ask-btn");
  btn.disabled = true;
  $("#ask-status").textContent = "thinking… (rerank → answer → fact-check)";
  ["#answer", "#factcheck", "#trace-box", "#replay"].forEach((s) => $(s).classList.add("hidden"));
  clearHighlight();
  const t0 = performance.now();
  try {
    const res = await api("/api/query", { method: "POST", headers: { "Content-Type": "application/json" },
                                          body: JSON.stringify({ question: q }) });
    lastResult = res;
    $("#ask-status").textContent = `${((performance.now() - t0) / 1000).toFixed(1)}s`;
    renderAnswer(res);
    renderFactCheck(res);
    renderTrace(res);
    $("#replay").classList.remove("hidden");
    replay(res);
  } catch (err) {
    $("#ask-status").textContent = "";
    $("#answer").innerHTML = `<span style="color:var(--danger)">${esc(err.message)}</span>`;
    $("#answer").classList.remove("hidden");
  } finally {
    btn.disabled = false;
  }
});
$("#question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) $("#ask").requestSubmit();
});
$("#replay").addEventListener("click", () => lastResult && replay(lastResult));

function renderAnswer(res) {
  const inline = (s) => esc(s)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\[([CERS]\d+)\]/g, (m, l) => res.citations[l] ? `<span class="cite" data-l="${l}">${l}</span>` : "");
  const blocks = res.answer.trim().split(/\n{2,}/).map((b) => {
    const lines = b.split("\n");
    if (lines.every((l) => /^\s*([-*•]|\d+\.)\s+/.test(l))) {
      return `<ul>${lines.map((l) => `<li>${inline(l.replace(/^\s*([-*•]|\d+\.)\s+/, ""))}</li>`).join("")}</ul>`;
    }
    return `<p>${lines.map(inline).join("<br>")}</p>`;
  });
  $("#answer").innerHTML = blocks.join("") + `<div id="cite-pop" class="cite-pop hidden"></div>`;
  $("#answer").classList.remove("hidden");
  $("#answer").querySelectorAll(".cite").forEach((el) => el.addEventListener("click", () => showCitation(res, el.dataset.l)));
}

function showCitation(res, label) {
  const c = res.citations[label];
  if (!c) return;
  const pop = $("#cite-pop");
  pop.textContent = `${label} · ${c.kind} · ${c.title}\n\n${c.text.replace(/^\[\w+\]\s*/, "")}`;
  pop.classList.remove("hidden");
  replayToken++;
  cy.elements().removeClass(HL.join(" ")).addClass("faded");
  const nodes = cy.collection(c.node_ids.map((id) => cy.getElementById(id)).filter((n) => n.nonempty()));
  nodes.removeClass("faded").addClass("hl-cited");
  focus(nodes, 90);
}

function renderFactCheck(res) {
  const fc = res.fact_check || [];
  if (!fc.length) return;
  const counts = fc.reduce((a, v) => ((a[v.verdict] = (a[v.verdict] || 0) + 1), a), {});
  $("#factcheck").innerHTML = `<details><summary>Fact check · ${Object.entries(counts).map(([k, v]) => `${v} ${k.toLowerCase()}`).join(", ")}</summary>
    ${fc.map((v) => `<div class="fc"><span class="v ${v.verdict}">${v.verdict}</span><div>${esc(v.claim)}<div class="muted">${esc(v.note)}</div></div></div>`).join("")}</details>`;
  $("#factcheck").classList.remove("hidden");
}

function renderTrace(res) {
  const items = [];
  for (const ev of res.trace) {
    if (ev.step === "top_level") {
      items.push(`<li class="muted">Top-level scoring (threshold ${ev.threshold})</li>`);
      for (const c of ev.candidates)
        items.push(`<li data-id="${c.id}"><span class="sc ${c.selected ? "ok" : "no"}">${c.score.toFixed(3)}</span> ${esc(c.title)}</li>`);
    } else if (ev.step === "visit") {
      const cls = ev.accepted ? (ev.forced ? "forced" : "ok") : "no";
      const tag = ev.accepted ? (ev.forced ? "forced" : "accepted") : "rejected";
      items.push(`<li data-id="${ev.id}">visit <span class="sc ${cls}">${ev.score.toFixed(3)}</span> ${esc(ev.title)} <span class="muted">${tag}</span></li>`);
    } else if (ev.step === "leaf") {
      items.push(`<li data-id="${ev.id}" class="muted">↳ leaf: ${ev.entity_ids.length} entities, ${ev.relationship_keys.length} relationships, ${ev.chunk_ids.length} source chunks</li>`);
    } else if (ev.step === "rerank") {
      items.push(`<li class="muted">Rerank: kept ${ev.kept.length}, dropped ${ev.dropped.length}</li>`);
    } else if (ev.step === "context") {
      items.push(`<li class="muted">Context: ${ev.items.length} items, ${ev.chars} chars</li>`);
    }
  }
  $("#trace").innerHTML = items.join("");
  $("#trace").querySelectorAll("li[data-id]").forEach((li) => li.addEventListener("click", () => {
    const n = cy.getElementById(li.dataset.id);
    if (n.nonempty()) { cy.animate({ center: { eles: n }, zoom: Math.max(cy.zoom(), 1), duration: 400 }); showDetails(n.id()); }
  }));
  const visits = res.trace.filter((e) => e.step === "visit");
  $("#trace-meta").textContent = `· ${visits.length} visited, ${visits.filter((v) => v.accepted).length} accepted · ${res.timings.fact_check ?? res.timings.rerank ?? ""}s`;
  $("#trace-box").classList.remove("hidden");
}

/* ------------------------------------------------------------------ traversal replay */
async function replay(res) {
  const token = ++replayToken;
  const alive = () => token === replayToken;
  const node = (id) => cy.getElementById(id);
  const scoreLabel = (id, score) => { const n = node(id); if (n.nonempty()) n.data("label", `${baseLabel.get(id)} · ${score.toFixed(2)}`); };
  const set = (id, cls) => { const n = node(id); if (n.nonempty()) n.removeClass(HL.join(" ")).addClass(cls); };

  cy.elements().removeClass(HL.join(" ")).addClass("faded");
  cy.nodes().forEach((n) => n.data("label", baseLabel.get(n.id()) ?? n.data("label")));
  await sleep(300);

  for (const ev of res.trace) {
    if (!alive()) return;
    if (ev.step === "top_level") {
      ev.candidates.forEach((c) => { set(c.id, "hl-scored"); scoreLabel(c.id, c.score); });
      await sleep(700);
      ev.candidates.filter((c) => !c.selected).forEach((c) => set(c.id, "hl-rejected"));
      await sleep(400);
    } else if (ev.step === "visit") {
      set(ev.id, "hl-scored");
      scoreLabel(ev.id, ev.score);
      node(ev.id).connectedEdges("[kind='child']").removeClass("faded").addClass("hl-edge");
      await sleep(250);
      set(ev.id, ev.accepted ? "hl-accepted" : "hl-rejected");
      await sleep(250);
    } else if (ev.step === "leaf") {
      for (const id of ev.entity_ids) {
        set(id, "hl-picked");
        node(id).connectedEdges("[kind='member']").removeClass("faded").addClass("hl-edge");
      }
      for (const k of ev.relationship_keys) {
        const e = cy.getElementById(`r:${k}`);
        if (e.nonempty()) e.removeClass("faded").addClass("hl-edge");
        const [a, , b] = k.split("|");
        [a, b].forEach((id) => { const n = node(id); if (n.nonempty() && n.hasClass("faded")) n.removeClass("faded").addClass("hl-picked"); });
      }
      await sleep(450);
    } else if (ev.step === "rerank") {
      ev.dropped.forEach((id) => { const n = node(id); if (n.nonempty() && n.data("kind") === "community") set(id, "hl-rejected"); });
      await sleep(400);
    } else if (ev.step === "cited") {
      const cited = cy.collection(ev.node_ids.map(node).filter((n) => n.nonempty()));
      cited.removeClass("faded hl-scored hl-rejected").addClass("hl-cited");
      focus(cy.nodes(".hl-cited, .hl-accepted, .hl-picked"));
    }
  }
}

/* ------------------------------------------------------------------ boot */
(async function boot() {
  checkHealth();
  await Promise.all([refreshStats(), refreshDocs().catch(() => {})]);
  await loadGraph().catch((e) => console.error(e));
  // resume progress display if a job is running (e.g. after a page reload)
  try {
    const running = (await api("/api/jobs")).find((j) => j.status === "running" || j.status === "queued");
    if (running) pollJob(running.id);
  } catch { /* ignore */ }
})();
