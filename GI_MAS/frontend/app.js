const API_BASE = "";

const userPromptEl = document.getElementById("userPrompt");
const institutionNameEl = document.getElementById("institutionName");
const logoImageEl = document.getElementById("logoImage");
const brandColorsEl = document.getElementById("brandColors");
const statusTextEl = document.getElementById("statusText");
const generateBtn = document.getElementById("generateBtn");
const galleryEl = document.getElementById("gallery");
const metaSectionEl = document.getElementById("metaSection");
const metaContentEl = document.getElementById("metaContent");
const agentProgressSectionEl = document.getElementById("agentProgressSection");
const agentGraphEl = document.getElementById("agentGraph");
const agentLegendEl = document.getElementById("agentLegend");

let pollingHandle = null;
const AGENT_WORKING_GIF =
  "data:image/svg+xml;utf8," +
  encodeURIComponent(`
    <svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24'>
      <defs>
        <linearGradient id='g' x1='0%' y1='0%' x2='100%' y2='100%'>
          <stop offset='0%' stop-color='#0f8c95'/>
          <stop offset='100%' stop-color='#0d6c86'/>
        </linearGradient>
      </defs>
      <circle cx='12' cy='12' r='10' fill='none' stroke='url(#g)' stroke-width='3' stroke-linecap='round' stroke-dasharray='40 18'>
        <animateTransform attributeName='transform' type='rotate' from='0 12 12' to='360 12 12' dur='0.9s' repeatCount='indefinite'/>
      </circle>
      <circle cx='12' cy='12' r='3' fill='#0f8c95'>
        <animate attributeName='r' values='2.4;3.4;2.4' dur='0.9s' repeatCount='indefinite'/>
      </circle>
    </svg>
  `);

function splitColors(raw) {
  return raw
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function statusBadge(status) {
  if (status === "completed") {
    return `<span class="badge badge-ok">Done</span>`;
  }
  if (status === "in_progress") {
    return `<span class="badge badge-run"><img class="agent-gif" src="${AGENT_WORKING_GIF}" alt="Working" /> Working</span>`;
  }
  if (status === "failed") {
    return `<span class="badge badge-fail">Error</span>`;
  }
  return `<span class="badge badge-pending">Pending</span>`;
}

function statusColor(status) {
  if (status === "completed") return "#2d8f4d";
  if (status === "in_progress") return "#0f8c95";
  if (status === "failed") return "#b73428";
  return "#9aa6b2";
}

function nodeHtml(id, label, status, x, y) {
  return `
    <div class="graph-node" id="node-${id}" style="left:${x}%;top:${y}%;border-color:${statusColor(status)};">
      <div class="node-title">${label}</div>
      <div class="node-state">${statusBadge(status)}</div>
    </div>
  `;
}

function renderAgentTree(tree) {
  agentProgressSectionEl.classList.remove("hidden");

  const t = tree || {};
  const reviewer = t.reviewer?.status || "pending";
  const planner = t.planner?.status || "pending";
  const brand = t.brand_agent?.status || "pending";
  const prompt = t.prompt_designer?.status || "pending";
  const critic = t.critic?.status || "pending";
  const segmenter = t.segmenter?.status || "pending";
  const renderer = t.renderer?.status || "pending";
  const composer = t.composer?.status || "pending";

  const html = `
    <div class="graph-canvas">
      <svg class="graph-lines" viewBox="0 0 100 100" preserveAspectRatio="none">
        <line x1="50" y1="16" x2="30" y2="33" />
        <line x1="50" y1="16" x2="70" y2="33" />
        <line x1="30" y1="33" x2="50" y2="52" />
        <line x1="70" y1="33" x2="50" y2="52" />
        <line x1="50" y1="52" x2="50" y2="66" />
        <line x1="50" y1="66" x2="50" y2="80" />
        <line x1="50" y1="80" x2="28" y2="92" />
        <line x1="50" y1="80" x2="72" y2="92" />
      </svg>
      ${nodeHtml("reviewer", "Reviewer", reviewer, 50, 10)}
      ${nodeHtml("planner", "Planner", planner, 30, 30)}
      ${nodeHtml("brand", "Brand", brand, 70, 30)}
      ${nodeHtml("prompt", "Prompt", prompt, 50, 50)}
      ${nodeHtml("critic", "Critic", critic, 50, 64)}
      ${nodeHtml("segmenter", "Segmenter", segmenter, 50, 78)}
      ${nodeHtml("renderer", "Renderer", renderer, 28, 92)}
      ${nodeHtml("composer", "Composer", composer, 72, 92)}
    </div>
  `;
  agentGraphEl.innerHTML = html;

  const renderChildren = Object.values(t.renderer?.children || {});
  const compChildren = Object.values(t.composer?.children || {});
  const segChildren = Object.values(t.segmenter?.children || {});
  const renderDone = renderChildren.filter((x) => x.status === "completed").length;
  const compDone = compChildren.filter((x) => x.status === "completed").length;
  const segDone = segChildren.filter((x) => x.status === "completed").length;
  const renderFail = renderChildren.filter((x) => x.status === "failed").length;
  const compFail = compChildren.filter((x) => x.status === "failed").length;
  const segFail = segChildren.filter((x) => x.status === "failed").length;

  agentLegendEl.innerHTML = `
    <div class="legend-item">Segmenter: ${segDone}/5 completed, ${segFail} errors</div>
    <div class="legend-item">Renderer: ${renderDone}/5 completed, ${renderFail} errors</div>
    <div class="legend-item">Composer: ${compDone}/5 completed, ${compFail} errors</div>
  `;
}

function setStatus(text, isError = false) {
  statusTextEl.textContent = text;
  statusTextEl.style.color = isError ? "#b73428" : "#3a4e69";
}

function renderMeta(data) {
  metaSectionEl.classList.remove("hidden");
  const planner = data.agent_outputs?.planner || "-";
  const brandAgent = data.agent_outputs?.brand_agent || "-";
  const critic = data.agent_outputs?.critic_summary || "-";
  const warnings = Array.isArray(data.warnings) ? data.warnings : [];
  const warningHtml = warnings.length
    ? `<details><summary><strong>Warnings (${warnings.length})</strong></summary><ul>${warnings
        .map((w) => `<li>${w}</li>`)
        .join("")}</ul></details>`
    : "";

  metaContentEl.innerHTML = `
    <p><strong>Groq Model:</strong> ${data.groq?.model || "-"}</p>
    <p><strong>Groq Note:</strong> ${data.groq?.note || "-"}</p>
    <p><strong>Image Service:</strong> ${data.image_generation?.provider || "-"}</p>
    <p><strong>Planner Plan:</strong> ${planner}</p>
    <p><strong>Brand Direction:</strong> ${brandAgent}</p>
    <p><strong>Critic Review:</strong> ${critic}</p>
    ${warningHtml}
  `;
}

function renderGallery(images) {
  galleryEl.innerHTML = "";
  images.forEach((item, idx) => {
    const card = document.createElement("article");
    card.className = "card";
    card.style.animationDelay = `${idx * 0.08}s`;

    card.innerHTML = `
      <img src="${item.image_url}" alt="${item.title}" loading="lazy" />
      <div class="card-content">
        <h3>${item.title}</h3>
        <p>${item.style_note}</p>
        <p class="prompt">${item.prompt}</p>
      </div>
    `;
    galleryEl.appendChild(card);
  });
}

async function pollJob(jobId) {
  const response = await fetch(`${API_BASE}/jobs/${jobId}`);
  if (!response.ok) {
    throw new Error("Job status could not be retrieved.");
  }

  const data = await response.json();
  renderAgentTree(data.agent_tree || {});

  if (data.status === "completed") {
    if (pollingHandle) {
      clearInterval(pollingHandle);
      pollingHandle = null;
    }
    renderMeta(data.result || {});
    renderGallery(data.result?.images || []);
    setStatus("5 designs ready and saved to images folder.");
    generateBtn.disabled = false;
  }

  if (data.status === "failed") {
    if (pollingHandle) {
      clearInterval(pollingHandle);
      pollingHandle = null;
    }
    throw new Error(data.error || "An error occurred during generation.");
  }
}

async function generate() {
  const userPrompt = userPromptEl.value.trim();
  const institutionName = institutionNameEl.value.trim();
  const colors = splitColors(brandColorsEl.value);
  const logoImage = logoImageEl.files?.[0] || null;

  if (!userPrompt || !institutionName || colors.length === 0) {
    setStatus("Prompt, institution name, and brand colors are required.", true);
    return;
  }

  generateBtn.disabled = true;
  metaSectionEl.classList.add("hidden");
  galleryEl.innerHTML = "";
  setStatus("Starting process...");

  try {
    const formData = new FormData();
    formData.append("user_prompt", userPrompt);
    formData.append("institution_name", institutionName);
    formData.append("brand_colors", colors.join(","));
    if (logoImage) {
      formData.append("logo_image", logoImage);
    }

    const response = await fetch(`${API_BASE}/generate-images`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errData = await response.json();
      throw new Error(errData?.detail || "Request failed.");
    }

    const data = await response.json();
    setStatus("Agents are working. You can see the stages live below.");

    if (pollingHandle) {
      clearInterval(pollingHandle);
    }

    await pollJob(data.job_id);
    pollingHandle = setInterval(async () => {
      try {
        await pollJob(data.job_id);
      } catch (error) {
        setStatus(`Error: ${error.message}`, true);
        generateBtn.disabled = false;
        if (pollingHandle) {
          clearInterval(pollingHandle);
          pollingHandle = null;
        }
      }
    }, 1200);
  } catch (error) {
    setStatus(`Error: ${error.message}`, true);
    generateBtn.disabled = false;
  }
}

generateBtn.addEventListener("click", generate);
