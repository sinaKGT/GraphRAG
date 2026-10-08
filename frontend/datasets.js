/* GraphRAG · Datasets — upload Excel/CSV, describe dataset + columns, see the catalog graph. No build step. */
"use strict";

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const fmt = (n) => (n == null ? "–" : Number(n).toLocaleString());

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try {
      const d = (await r.json()).detail;
      msg = Array.isArray(d) ? d.map((x) => `${x.loc?.slice(-1)[0]}: ${x.msg}`).join("; ") : d || msg;
    } catch { /* not json */ }
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return r.json();
}
const jsonOpts = (method, body) => ({ method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

function toast(msg, err = false) {
  const t = document.createElement("div");
  t.className = "toast" + (err ? " err" : "");
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), err ? 6000 : 3000);
}

/* ------------------------------------------------------------------ colours */
const TYPE_COLOR = { text: "#60a5fa", integer: "#34d399", float: "#2dd4bf", datetime: "#fbbf24", time: "#fb923c",
                     boolean: "#c084fc", duration: "#f472b6", mixed: "#94a3b8", empty: "#475569" };
const CSS = getComputedStyle(document.documentElement);
const C = (v) => CSS.getPropertyValue(v).trim();

/* ------------------------------------------------------------------ cytoscape */
const cy = cytoscape({
  container: $("#cy"),
  minZoom: 0.05,
  maxZoom: 3,
  style: [
    { selector: "node", style: {
        "label": "data(label)", "color": "#cbd5e1", "font-size": 9, "min-zoomed-font-size": 6,
        "text-valign": "bottom", "text-margin-y": 3, "text-wrap": "ellipsis", "text-max-width": 120 } },
    { selector: "node[kind='dataset']", style: {
        "shape": "round-rectangle", "width": "data(w)", "height": "data(w)", "background-color": "#14532d",
        "border-color": C("--accepted"), "border-width": 2, "font-size": 12, "font-weight": 700, "color": "#e2e8f0",
        "text-valign": "center", "text-margin-y": 0, "text-wrap": "wrap", "text-max-width": "data(tw)" } },
    { selector: "node[kind='dataset'][status!='ready']", style: {
        "background-color": "#3b2a0a", "border-color": C("--scored"), "border-style": "dashed" } },
    { selector: "node[kind='column']", style: {
        "width": 14, "height": 14, "background-color": "data(color)", "border-width": 0 } },
    { selector: "node[kind='column'][?key]", style: { "border-width": 2.5, "border-color": "#ffffff" } },
    { selector: "node[kind='column'][?undescribed]", style: { "opacity": 0.55 } },
    { selector: "edge", style: { "curve-style": "haystack", "line-color": "#334155", "width": 1, "opacity": 0.6 } },
    { selector: ".hidden-col", style: { "display": "none" } },
    { selector: "node:selected", style: { "border-width": 3, "border-color": "#fff" } },
  ],
});

let graphData = { datasets: [], columns: [] };

async function loadGraph() {
  const g = await api("/api/datasets/graph");
  graphData = g;
  const els = [];
  for (const d of g.datasets) {
    const n = d.column_count || 1;
    els.push({ data: { id: d.id, kind: "dataset", label: d.label, status: d.status,
                       w: 60 + Math.min(50, Math.sqrt(n) * 8), tw: 70 + Math.min(50, Math.sqrt(n) * 8) } });
  }
  for (const c of g.columns) {
    els.push({ data: { id: c.id, kind: "column", label: c.label, color: TYPE_COLOR[c.dtype] || "#94a3b8",
                       key: !!c.is_unique, undescribed: !c.description } });
    els.push({ data: { id: `h:${c.id}`, source: c.dataset, target: c.id, kind: "has" } });
  }
  cy.elements().remove();
  cy.add(els);
  $("#empty").classList.toggle("hidden", g.datasets.length > 0);
  applyColToggle();
  if (!g.datasets.length) return;
  cy.layout({ name: "fcose", quality: "default", randomize: true, animate: false, nodeSeparation: 50,
              nodeRepulsion: () => 6000, idealEdgeLength: () => 55, edgeElasticity: () => 0.5 }).run();
  cy.fit(undefined, 40);
}

function applyColToggle() {
  cy.$("node[kind='column'], edge").toggleClass("hidden-col", !$("#show-cols").checked);
}
$("#show-cols").addEventListener("change", applyColToggle);
$("#fit").addEventListener("click", () => cy.fit(cy.elements(":visible"), 40));

cy.on("tap", "node[kind='dataset']", (ev) => openEditor(ev.target.id()));
cy.on("tap", "node[kind='column']", (ev) => showColumn(ev.target.id()));
cy.on("tap", (ev) => { if (ev.target === cy) $("#details").classList.add("hidden"); });

function showColumn(id) {
  const c = graphData.columns.find((x) => x.id === id);
  const d = c && graphData.datasets.find((x) => x.id === c.dataset);
  if (!c) return;
  $("#details").innerHTML = `<span class="close" onclick="this.parentElement.classList.add('hidden')">✕</span>
    <div class="k">Column · ${esc(c.dtype)}${c.is_unique ? " · unique" : ""} · ${fmt(c.distinct)}${c.distinct_capped ? "+" : ""} distinct</div>
    <h3 style="color:var(--text);font-size:15px;margin:4px 0">${esc(c.label)}</h3>
    <div>${c.description ? esc(c.description) : '<span class="muted">No description yet</span>'}</div>
    <div class="muted" style="margin-top:6px">e.g. ${esc((c.examples || []).slice(0, 5).join(" · "))}</div>
    <div class="k" style="margin-top:8px">Dataset</div><div><a href="#" data-ds="${esc(d?.id)}">${esc(d?.label)}</a></div>`;
  $("#details").querySelector("a[data-ds]").addEventListener("click", (e) => { e.preventDefault(); openEditor(d.id); });
  $("#details").classList.remove("hidden");
}

/* ------------------------------------------------------------------ status / list */
async function refreshStats() {
  try {
    const s = await api("/api/datasets/stats");
    $("#stats").textContent = `${s.datasets} datasets (${s.ready} described) · ${s.columns} columns`;
  } catch { $("#stats").textContent = ""; }
}

const STATUS_LABEL = { needs_description: "describe", ready: "ready", queued: "queued", profiling: "profiling", failed: "failed" };
let datasets = [];

async function refreshList() {
  datasets = await api("/api/datasets");
  $("#datasets").innerHTML = datasets.map((d) =>
    `<li data-id="${esc(d.id)}" class="${d.id === current?.id ? "sel" : ""}" title="${esc(d.error || d.file_name)}">
       <span>${esc(d.name)}<span class="sub">${d.row_count != null ? `${fmt(d.row_count)} rows · ${d.column_count} cols` : esc(d.file_name)}</span></span>
       <span class="badge ${esc(d.status)}">${esc(STATUS_LABEL[d.status] || d.status)}</span></li>`).join("");
  $("#datasets").querySelectorAll("li").forEach((li) => li.addEventListener("click", () => openEditor(li.dataset.id)));
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
  } catch {
    el.className = "health bad";
    el.querySelector(".txt").textContent = "backend unreachable";
  }
  setTimeout(checkHealth, ok ? 60000 : 8000);
}

/* ------------------------------------------------------------------ upload + profiling job */
const drop = $("#drop");
$("#file").addEventListener("change", (e) => e.target.files[0] && upload(e.target.files[0]));
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && upload(e.dataTransfer.files[0]));

async function upload(file) {
  const fd = new FormData();
  fd.append("file", file);
  showJob({ filename: file.name, status: "uploading", stage: `uploading ${(file.size / 1048576).toFixed(1)} MB…` });
  try {
    const res = await api("/api/datasets", { method: "POST", body: fd });
    await refreshList();
    if (!res.job) {
      showJob({ filename: file.name, status: "done", stage: res.message, done: 1, total: 1 });
      openEditor(res.dataset.id);
      return;
    }
    await pollJob(res.job.id);
  } catch (e) {
    showJob({ filename: file.name, status: "failed", stage: e.message });
  } finally {
    $("#file").value = "";
  }
}

function showJob(j) {
  $("#job").classList.remove("hidden");
  $("#job-name").textContent = j.filename || "";
  $("#job-status").innerHTML = `<span class="badge ${esc(j.status)}">${esc(j.status)}</span>`;
  $("#job-stage").textContent = j.error ? j.error
    : `${j.stage || ""}${j.total ? ` (${fmt(j.done)}/${fmt(j.total)})` : j.done ? ` (${fmt(j.done)})` : ""}`;
  const bar = $("#job-bar");
  const running = ["running", "queued", "uploading"].includes(j.status);
  bar.parentElement.classList.toggle("indeterminate", running && !j.total);
  bar.style.width = j.status === "done" ? "100%" : j.total ? `${(100 * j.done) / j.total}%` : running ? "" : "0";
}

async function pollJob(id) {
  let j;
  for (;;) {
    try { j = await api(`/api/jobs/${id}`); }
    catch (e) {
      if (/not found/i.test(e.message)) { $("#job").classList.add("hidden"); break; }
      await sleep(3000); continue;
    }
    showJob(j);
    if (j.status === "done" || j.status === "failed") break;
    await sleep(1200);
  }
  await Promise.all([refreshList(), refreshStats(), loadGraph()]);
  if (j && j.status === "done" && j.dataset_id) openEditor(j.dataset_id);   // next step: describe it
}

$("#reset").addEventListener("click", async () => {
  if (!confirm("Delete all datasets (catalog + uploaded Excel/CSV files)? The document graph is kept.")) return;
  try {
    await api("/api/datasets", { method: "DELETE" });
    closeEditor(true);
    $("#job").classList.add("hidden");
    await Promise.all([refreshList(), refreshStats(), loadGraph()]);
  } catch (e) { toast(e.message, true); }
});

/* ------------------------------------------------------------------ editor */
let current = null;   // dataset being edited
let dirty = false;

function markDirty() { dirty = true; $("#ed-msg").textContent = "unsaved changes"; }

function autosize(t) { t.style.height = "auto"; t.style.height = `${Math.min(220, t.scrollHeight + 2)}px`; }

function colRange(c) {
  if (c.min != null && c.max != null && ["integer", "float", "datetime", "time"].includes(c.dtype))
    return `${c.min} … ${c.max}${c.mean != null ? ` · mean ${Number(c.mean).toLocaleString(undefined, { maximumFractionDigits: 2 })}` : ""}`;
  const top = (c.top_values || []).slice(0, 3);
  return top.length ? top.join(" · ") : (c.examples || []).slice(0, 4).join(" · ");
}

async function openEditor(id) {
  if (dirty && current && current.id !== id && !confirm("Discard unsaved changes?")) return;
  let ds;
  try { ds = await api(`/api/datasets/${id}`); } catch (e) { toast(e.message, true); return; }
  current = ds;
  dirty = false;
  $("#details").classList.add("hidden");
  $("#datasets").querySelectorAll("li").forEach((li) => li.classList.toggle("sel", li.dataset.id === id));

  $("#ed-file").textContent = ds.file_name;
  $("#ed-meta").textContent = ds.row_count != null
    ? `${fmt(ds.row_count)} rows · ${ds.column_count} columns${ds.sheet ? ` · sheet "${ds.sheet}"` : ""}`
    : "not profiled yet";
  $("#ed-status").innerHTML = `<span class="badge ${esc(ds.status)}">${esc(STATUS_LABEL[ds.status] || ds.status)}</span>`;
  const warn = [];
  if (ds.other_sheets?.length) warn.push(`Only the first sheet with data ("${ds.sheet}") was read. Other sheets ignored: ${ds.other_sheets.join(", ")}.`);
  if (ds.status === "failed") warn.push(`Profiling failed: ${ds.error || "unknown error"}`);
  $("#ed-warn").textContent = warn.join(" ");
  $("#ed-warn").classList.toggle("hidden", !warn.length);

  $("#ed-name").value = ds.name || "";
  $("#ed-desc").value = ds.description || "";
  [$("#ed-name"), $("#ed-desc")].forEach((el) => el.classList.remove("ai"));
  $("#ed-rows").innerHTML = ds.columns.map((c) => `
    <tr data-pos="${c.position}">
      <td class="num">${c.position + 1}</td>
      <td class="cname">${esc(c.name)}${c.is_unique ? '<span class="key" title="Every value is different: possible ID / join key">unique</span>' : ""}</td>
      <td><span class="dtype" style="color:${TYPE_COLOR[c.dtype] || "#94a3b8"}">${esc(c.dtype)}</span></td>
      <td class="num">${(100 - c.null_pct).toFixed(c.null_pct % 1 ? 1 : 0)}%</td>
      <td class="num">${fmt(c.distinct)}${c.distinct_capped ? "+" : ""}</td>
      <td class="ex">${esc(colRange(c))}</td>
      <td><textarea rows="1" maxlength="2000" placeholder="meaning, unit, how to use it…">${esc(c.description || "")}</textarea></td>
    </tr>`).join("");
  $("#ed-rows").querySelectorAll("textarea").forEach((t) => {
    autosize(t);
    t.addEventListener("input", () => { autosize(t); t.classList.remove("ai"); markDirty(); updateCount(); });
  });
  updateCount();
  const busy = ["queued", "profiling"].includes(ds.status);
  $("#ed-ai").disabled = $("#ed-save").disabled = busy || !ds.columns.length;
  $("#ed-msg").textContent = ds.status === "needs_description" ? "Describe the dataset, then Save & index." : "";
  $("#editor").classList.remove("hidden");
  $("#editor").scrollTop = 0;
}

function updateCount() {
  const ts = [...$("#ed-rows").querySelectorAll("textarea")];
  const n = ts.filter((t) => t.value.trim()).length;
  $("#ed-count").textContent = `${n}/${ts.length} described`;
}

function closeEditor(force = false) {
  if (!force && dirty && !confirm("Discard unsaved changes?")) return;
  current = null;
  dirty = false;
  $("#editor").classList.add("hidden");
  $("#datasets").querySelectorAll("li").forEach((li) => li.classList.remove("sel"));
}
$("#ed-back").addEventListener("click", () => closeEditor());
[$("#ed-name"), $("#ed-desc")].forEach((el) => el.addEventListener("input", () => { el.classList.remove("ai"); markDirty(); }));
window.addEventListener("beforeunload", (e) => { if (dirty) { e.preventDefault(); e.returnValue = ""; } });

$("#ed-ai").addEventListener("click", async () => {
  if (!current) return;
  const btn = $("#ed-ai");
  btn.disabled = true;
  $("#ed-msg").textContent = "AI is drafting descriptions… (local model: up to a minute)";
  try {
    const d = await api(`/api/datasets/${current.id}/draft`, jsonOpts("POST", { description: $("#ed-desc").value.trim() || null }));
    let filled = 0;
    const fill = (el, val) => { if (!el.value.trim() && val) { el.value = val; el.classList.add("ai"); filled++; } };
    if ($("#ed-name").value.trim() === current.file_name.replace(/\.[^.]+$/, "")) $("#ed-name").value = "";
    fill($("#ed-name"), d.name);
    fill($("#ed-desc"), d.description);
    const byPos = new Map(d.columns.map((c) => [c.position, c.description]));
    $("#ed-rows").querySelectorAll("tr").forEach((tr) => {
      const t = tr.querySelector("textarea");
      fill(t, byPos.get(Number(tr.dataset.pos)));
      autosize(t);
    });
    updateCount();
    if (filled) markDirty();
    $("#ed-msg").textContent = filled ? `AI filled ${filled} empty field(s) (purple). Check them, then Save & index.` : "Nothing to fill: every field already has text.";
  } catch (e) {
    $("#ed-msg").textContent = "";
    toast(`AI draft failed: ${e.message}`, true);
  } finally {
    btn.disabled = false;
  }
});

$("#ed-save").addEventListener("click", async () => {
  if (!current) return;
  const description = $("#ed-desc").value.trim();
  if (description.length < 10) { toast("Please describe the dataset first (at least a sentence).", true); $("#ed-desc").focus(); return; }
  const columns = [...$("#ed-rows").querySelectorAll("tr")].map((tr) => ({
    position: Number(tr.dataset.pos), description: tr.querySelector("textarea").value.trim() }));
  const btn = $("#ed-save");
  btn.disabled = true;
  $("#ed-msg").textContent = "saving + embedding…";
  try {
    const ds = await api(`/api/datasets/${current.id}`, jsonOpts("PUT", { name: $("#ed-name").value.trim(), description, columns }));
    dirty = false;
    toast(`Saved "${ds.name}" — dataset + ${ds.columns.length} columns indexed`);
    await Promise.all([refreshList(), refreshStats(), loadGraph()]);
    await openEditor(ds.id);
    $("#ed-msg").textContent = "Saved and indexed.";
  } catch (e) {
    $("#ed-msg").textContent = "";
    toast(`Save failed: ${e.message}`, true);
  } finally {
    btn.disabled = false;
  }
});

$("#ed-delete").addEventListener("click", async () => {
  if (!current || !confirm(`Delete "${current.name}" and its uploaded file?`)) return;
  try {
    await api(`/api/datasets/${current.id}`, { method: "DELETE" });
    closeEditor(true);
    await Promise.all([refreshList(), refreshStats(), loadGraph()]);
  } catch (e) { toast(e.message, true); }
});

/* ------------------------------------------------------------------ boot */
(async function boot() {
  checkHealth();
  await Promise.all([refreshStats(), refreshList().catch(() => {})]);
  await loadGraph().catch((e) => console.error(e));
  try {
    const running = (await api("/api/jobs")).find((j) => j.kind === "profile" && (j.status === "running" || j.status === "queued"));
    if (running) pollJob(running.id);
  } catch { /* ignore */ }
})();
