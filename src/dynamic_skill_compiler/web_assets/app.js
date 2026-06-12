const state = {
  defaultSkillsDir: "",
  lastResponse: null,
};

const $ = (id) => document.getElementById(id);

document.addEventListener("DOMContentLoaded", async () => {
  bindTabs();
  bindActions();
  await loadHealth();
  await loadSkills();
});

function bindTabs() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
      document.querySelectorAll(".view").forEach((item) => item.classList.remove("active"));
      tab.classList.add("active");
      $(tab.dataset.view).classList.add("active");
    });
  });
}

function bindActions() {
  $("loadSkills").addEventListener("click", loadSkills);
  $("compile").addEventListener("click", compile);
  $("copyJson").addEventListener("click", () => navigator.clipboard.writeText($("rawJson").textContent));
}

async function loadHealth() {
  const response = await fetch("/api/health");
  const data = await response.json();
  state.defaultSkillsDir = data.default_skills_dir || "";
  $("skillsDir").value = state.defaultSkillsDir;
}

async function loadSkills() {
  setStatus("Loading skills", "busy");
  try {
    const response = await fetch(`/api/skills?skills_dir=${encodeURIComponent($("skillsDir").value)}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Failed to load skills");
    renderCandidateSkills(data.skills || []);
    $("candidateCount").textContent = data.skill_count || 0;
    $("selectedCount").textContent = "0";
    $("coverageScore").textContent = "0.00";
    $("beforeMeta").textContent = `${data.skill_count || 0} local skills`;
    setStatus("Skills loaded");
  } catch (error) {
    setStatus(error.message, "error");
  }
}

async function compile() {
  setStatus("Compiling", "busy");
  const payload = {
    query: $("query").value,
    skills_dir: $("skillsDir").value,
    min_relevance: Number($("minRelevance").value),
    preserve_top_k: Number($("preserveTopK").value),
    max_selected_skills: Number($("maxSelected").value),
  };

  try {
    const response = await fetch("/api/compile", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Compile failed");
    state.lastResponse = data;
    renderCompile(data);
    setStatus("Compile complete");
  } catch (error) {
    setStatus(error.message, "error");
  }
}

function renderCompile(data) {
  const metrics = data.metrics || {};
  const candidates = data.candidate_skills || [];
  const selected = data.selected_skills || [];
  $("candidateCount").textContent = metrics.candidate_count ?? candidates.length;
  $("selectedCount").textContent = metrics.selected_count ?? selected.length;
  $("coverageScore").textContent = formatNumber(metrics.coverage_score ?? 0);
  $("rawJson").textContent = JSON.stringify(data, null, 2);

  renderFlow(data);
  renderMetrics(metrics);
  renderExecutionOrder(data.execution_order || []);
  renderCandidateSkills(candidates);
  renderSelectedSkills(selected);
  renderPassTrace(data.pass_trace || []);

  $("flowMeta").textContent = `${data.relations?.length || 0} graph edges`;
  $("orderMeta").textContent = `${data.execution_order?.length || 0} skills`;
  $("beforeMeta").textContent = `${candidates.length} candidates`;
  $("afterMeta").textContent = `${selected.length} selected`;
  $("passMeta").textContent = `${data.pass_trace?.length || 0} passes`;
}

function renderFlow(data) {
  const metrics = data.metrics || {};
  const steps = [
    ["Query", 1, "Normalized and decomposed"],
    ["Candidates", metrics.candidate_count ?? data.candidate_skills?.length ?? 0, "Skills before pruning"],
    ["Passes", data.pass_trace?.length || 0, "Selection and repair"],
    ["Package", metrics.selected_count ?? data.selected_skills?.length ?? 0, "Skills after compile"],
  ];
  $("flow").innerHTML = steps
    .map(
      ([label, value, detail]) => `
        <article class="flow-step">
          <strong>${escapeHtml(value)}</strong>
          <span>${escapeHtml(label)}</span>
          <p class="description">${escapeHtml(detail)}</p>
        </article>
      `
    )
    .join("");
}

function renderMetrics(metrics) {
  const rows = [
    ["Coverage", metrics.coverage_score],
    ["Redundancy cut", metrics.redundancy_reduction],
    ["Token before", metrics.estimated_token_cost_before],
    ["Token after", metrics.estimated_token_cost_after],
    ["Edges before", metrics.edge_count_before],
    ["Edges after", metrics.edge_count_after],
    ["Fragments before", metrics.fragment_count_before],
    ["Fragments after", metrics.fragment_count_after],
  ];
  $("metrics").innerHTML = rows
    .map(
      ([label, value]) => `
        <div class="metric">
          <strong>${formatNumber(value ?? 0)}</strong>
          <span>${escapeHtml(label)}</span>
        </div>
      `
    )
    .join("");
}

function renderExecutionOrder(order) {
  $("executionOrder").innerHTML = order.length
    ? order.map((skill) => `<li>${escapeHtml(skill)}</li>`).join("")
    : `<div class="empty">Run compile to see execution order.</div>`;
}

function renderCandidateSkills(skills) {
  $("candidateSkills").innerHTML = skills.length
    ? skills.map((skill) => renderSkillCard(skill, "candidate")).join("")
    : `<div class="empty">No skills loaded.</div>`;
}

function renderSelectedSkills(skills) {
  $("selectedSkills").innerHTML = skills.length
    ? skills.map((skill) => renderSkillCard({ ...skill, selected: true }, "selected")).join("")
    : `<div class="empty">Run compile to see selected skills.</div>`;
}

function renderSkillCard(skill, mode) {
  const selected = skill.selected || mode === "selected";
  const dropped = Boolean(skill.dropped_reason);
  const badge = selected ? "Selected" : dropped ? "Dropped" : "Candidate";
  const className = selected ? "selected" : dropped ? "dropped" : "";
  const caps = (skill.capabilities || []).slice(0, 10);
  const score = typeof skill.utility_score === "number" ? ` · ${formatNumber(skill.utility_score)}` : "";
  return `
    <article class="skill-card ${className}">
      <div class="skill-title">
        <strong>${escapeHtml(skill.name || skill.skill_id)}${score}</strong>
        <span class="badge ${className}">${badge}</span>
      </div>
      <p class="description">${escapeHtml(skill.description || skill.selected_reason || "")}</p>
      ${skill.dropped_reason ? `<p class="description">Dropped: ${escapeHtml(skill.dropped_reason)}</p>` : ""}
      ${
        skill.selected_reason && mode === "selected"
          ? `<p class="description">${escapeHtml(skill.selected_reason)}</p>`
          : ""
      }
      <div class="capabilities">${caps.map((cap) => `<span>${escapeHtml(cap)}</span>`).join("")}</div>
    </article>
  `;
}

function renderPassTrace(trace) {
  $("passTrace").innerHTML = trace.length
    ? trace
        .map(
          (pass, index) => `
            <article class="pass-item">
              <header>
                <strong>${index + 1}. ${escapeHtml(pass.pass_name)}</strong>
                <span class="badge">${(pass.after_selected || []).length} selected</span>
              </header>
              <div class="pass-delta">
                <div><strong>Added</strong>${listInline(pass.added)}</div>
                <div><strong>Removed</strong>${listInline(pass.removed)}</div>
                <div><strong>Dropped</strong>${listDropped(pass.dropped_delta)}</div>
              </div>
            </article>
          `
        )
        .join("")
    : `<div class="empty">No pass trace yet.</div>`;
}

function listInline(items) {
  return items && items.length ? items.map((item) => escapeHtml(item)).join(", ") : "None";
}

function listDropped(delta) {
  const entries = Object.entries(delta || {});
  if (!entries.length) return "None";
  return entries.map(([skill, reason]) => `${escapeHtml(skill)}: ${escapeHtml(reason)}`).join("<br>");
}

function setStatus(message, mode = "") {
  $("status").textContent = message;
  $("status").className = `status ${mode}`.trim();
}

function formatNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "0.00";
  return number >= 10 ? number.toFixed(1) : number.toFixed(2);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
