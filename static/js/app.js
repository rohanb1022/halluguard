/**
 * HalluGuard Client-Side Application Controller
 * Handles live verification, presets, benchmarks, Colab NLI sandbox, and KB explorer.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Global state
  let currentPresets = [];
  let lastReport = null;

  // Initialize components
  initTabs();
  loadPresets();
  loadKnowledgeBase();
  loadHaluEvalResearch();
  initThresholdSlider();
  initEventListeners();
});

/* ==========================================================================
   Tab Navigation
   ========================================================================== */
function initTabs() {
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-tab");

      tabButtons.forEach(b => b.classList.remove("active"));
      tabPanes.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const targetPane = document.getElementById(`tab-${targetId}`);
      if (targetPane) {
        targetPane.classList.add("active");
      }

      // If switching to benchmark tab for the first time, auto-run benchmark if empty
      if (targetId === "benchmark" && document.getElementById("table-81b-body").children.length === 0) {
        runBenchmark();
      }
    });
  });
}

/* ==========================================================================
   Threshold Slider & Settings
   ========================================================================== */
function initThresholdSlider() {
  const slider = document.getElementById("slider-threshold");
  const valDisplay = document.getElementById("thresh-val");
  const customCtxCheck = document.getElementById("check-custom-ctx");
  const customCtxGroup = document.getElementById("custom-ctx-group");

  if (slider && valDisplay) {
    slider.addEventListener("input", (e) => {
      valDisplay.textContent = parseFloat(e.target.value).toFixed(2);
    });
  }

  if (customCtxCheck && customCtxGroup) {
    customCtxCheck.addEventListener("change", (e) => {
      customCtxGroup.style.display = e.target.checked ? "block" : "none";
    });
  }
}

/* ==========================================================================
   Presets Loader & Selection
   ========================================================================== */
async function loadPresets() {
  try {
    const res = await fetch("/api/presets");
    const data = await res.json();
    currentPresets = data.presets || [];

    const select = document.getElementById("sample-preset");
    if (!select) return;

    select.innerHTML = '<option value="">-- Choose a Pre-configured Sample (T1 to T10) --</option>';
    currentPresets.forEach(p => {
      const opt = document.createElement("option");
      opt.value = p.id;
      opt.textContent = `${p.id} — ${p.question} (${p.claim_count} claims, ${p.expected_hallucinations} hal.)`;
      select.appendChild(opt);
    });

    // Auto-select T4 (Give key milestones in AI) by default for instant demonstration!
    select.value = "T4";
    applyPreset("T4");

  } catch (err) {
    console.error("Failed to load presets:", err);
  }
}

function applyPreset(presetId) {
  const preset = currentPresets.find(p => p.id === presetId);
  if (!preset) return;

  document.getElementById("input-question").value = preset.question;
  document.getElementById("input-answer").value = preset.answer;
}

/* ==========================================================================
   Verification Studio Request & Rendering
   ========================================================================== */
async function performAskAndGuard() {
  const question = document.getElementById("input-question").value.trim();
  const genMode = document.getElementById("select-gen-mode").value;
  const engine = document.getElementById("select-engine").value;
  const threshold = parseFloat(document.getElementById("slider-threshold").value);
  const useCustomCtx = document.getElementById("check-custom-ctx").checked;
  const customContext = document.getElementById("input-custom-ctx").value.trim();

  if (!question) {
    showToast("Please enter a question or click one of the quick question buttons.");
    return;
  }

  const btn = document.getElementById("btn-ask-and-guard");
  btn.disabled = true;
  btn.innerHTML = '<span class="btn-icon">⏳</span> Generating &amp; Guarding...';

  try {
    const res = await fetch("/api/ask_and_guard", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: question,
        gen_mode: genMode,
        engine: engine,
        threshold: threshold,
        use_custom_context: useCustomCtx,
        custom_context: customContext
      })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || "Generation/guarding failed");
    }

    const report = await res.json();
    lastReport = report;

    // Put the generated answer into the answer box
    document.getElementById("input-answer").value = report.generated_answer || "";

    // Render results
    renderVerificationResults(report);

    const supCount = report.supported_claims !== undefined ? report.supported_claims : (report.total_claims - report.flagged_claims);
    showToast(`AI answered! ${supCount} supported, ${report.flagged_claims} flagged (${report.hallucination_rate_pct}% hallucination)`);

  } catch (err) {
    console.error("Ask and guard error:", err);
    showToast(`Error: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span class="btn-icon">🚀</span> Ask AI &amp; Guard Answer';
  }
}

async function performGenerateOnly() {
  const question = document.getElementById("input-question").value.trim();
  const genMode = document.getElementById("select-gen-mode").value;

  if (!question) {
    showToast("Please enter a question to generate an answer.");
    return;
  }

  const btn = document.getElementById("btn-gen-only");
  btn.disabled = true;
  btn.textContent = "Generating...";

  try {
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question, mode: genMode })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || "Generation failed");
    }

    const data = await res.json();
    document.getElementById("input-answer").value = data.answer || "";
    showToast("Answer generated! You can review/edit it, then click 'Verify Answer Box'.");

  } catch (err) {
    console.error("Generate error:", err);
    showToast(`Error: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = "✨ Generate Answer Only";
  }
}

async function performVerification() {
  const question = document.getElementById("input-question").value.trim();
  const answer = document.getElementById("input-answer").value.trim();
  const engine = document.getElementById("select-engine").value;
  const threshold = parseFloat(document.getElementById("slider-threshold").value);
  const useCustomCtx = document.getElementById("check-custom-ctx").checked;
  const customContext = document.getElementById("input-custom-ctx").value.trim();

  if (!answer) {
    showToast("Please enter or select an answer text to verify.");
    return;
  }

  const verifyBtn = document.getElementById("btn-verify");
  verifyBtn.disabled = true;
  verifyBtn.innerHTML = '<span class="btn-icon">⏳</span> Verifying...';

  try {
    const res = await fetch("/api/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: question,
        answer: answer,
        engine: engine,
        threshold: threshold,
        use_custom_context: useCustomCtx,
        custom_context: customContext
      })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || "Verification failed");
    }

    const report = await res.json();
    lastReport = report;
    renderVerificationResults(report);
    showToast(`Verification completed in ${report.response_time_ms} ms!`);

  } catch (err) {
    console.error("Verification error:", err);
    showToast(`Error: ${err.message}`);
  } finally {
    verifyBtn.disabled = false;
    verifyBtn.innerHTML = '<span class="btn-icon">🔍</span> Verify Answer';
  }
}

function renderVerificationResults(report) {
  // Update Engine Status
  const engineTag = document.getElementById("engine-status-tag");
  if (engineTag) {
    engineTag.textContent = `Engine: ${report.engine || "Fast Lexical & Rule-Based"}`;
  }

  // Update Metric Cards
  document.getElementById("metric-total").textContent = report.total_claims;
  const supElem = document.getElementById("metric-supported");
  if (supElem) {
    supElem.textContent = report.supported_claims !== undefined ? report.supported_claims : (report.total_claims - report.flagged_claims);
  }
  document.getElementById("metric-flagged").textContent = report.flagged_claims;
  document.getElementById("metric-rate").textContent = `${report.hallucination_rate_pct}%`;
  const timeElem = document.getElementById("metric-time");
  if (timeElem) {
    timeElem.textContent = `${report.response_time_ms} ms`;
  }

  // Update Annotated Answer
  const annotatedBox = document.getElementById("annotated-output");
  annotatedBox.innerHTML = report.annotated_html || "<p class='placeholder-text'>No claims detected.</p>";

  // Update Guarded Answer
  const guardedBox = document.getElementById("guarded-output");
  guardedBox.innerHTML = `<p>${escapeHtml(report.guarded_answer)}</p>`;

  // Update Claim-by-Claim Table
  const tableBody = document.getElementById("claim-table-body");
  tableBody.innerHTML = "";

  if (!report.claims || report.claims.length === 0) {
    tableBody.innerHTML = '<tr><td colspan="7" class="empty-table-row">No claims extracted.</td></tr>';
    return;
  }

  report.claims.forEach(c => {
    const tr = document.createElement("tr");

    let verdictClass = "verdict-supported";
    if (c.label === "CONTRADICTED") verdictClass = "verdict-contradicted";
    if (c.label === "UNSUPPORTED") verdictClass = "verdict-unsupported";

    const mismatchHtml = c.mismatches && c.mismatches.length > 0
      ? `<div style="font-size:11.5px; color:#b91c1c; margin-top:4px;"><strong>Mismatches:</strong> ${c.mismatches.join("; ")}</div>`
      : "";

    tr.innerHTML = `
      <td><strong>${c.index}</strong></td>
      <td>${escapeHtml(c.text)}</td>
      <td>
        <div>${escapeHtml(c.evidence)}</div>
        <span class="source-tag">📄 [${c.source}]</span>
      </td>
      <td><code>${c.similarity.toFixed(2)}</code></td>
      <td><code>${c.coverage.toFixed(2)}</code></td>
      <td><span class="verdict-tag ${verdictClass}">${c.label}</span></td>
      <td>
        <div>${escapeHtml(c.reason)}</div>
        ${mismatchHtml}
      </td>
    `;
    tableBody.appendChild(tr);
  });
}

/* ==========================================================================
   Benchmark Runner (Table 8.1b)
   ========================================================================== */
async function runBenchmark() {
  const btn = document.getElementById("btn-run-benchmark");
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<span>⏳</span> Evaluating 39 Claims...';
  }

  try {
    const res = await fetch("/api/benchmark");
    const data = await res.json();

    const s = data.summary;
    document.getElementById("bm-acc").textContent = `${(s.accuracy * 100).toFixed(1)}%`;
    document.getElementById("bm-prec").textContent = s.precision.toFixed(4);
    document.getElementById("bm-rec").textContent = s.recall.toFixed(4);
    document.getElementById("bm-f1").textContent = s.f1_score.toFixed(4);
    document.getElementById("bm-reduction").textContent = `${s.raw_hallucination_rate_pct}% → ${s.residual_hallucination_rate_pct}%`;
    document.getElementById("bm-latency").textContent = `${s.avg_ms_per_answer} ms`;

    // Confusion Matrix cells
    const cm = data.confusion_matrix;
    document.getElementById("cm-tn").querySelector(".cm-num").textContent = cm.tn;
    document.getElementById("cm-fp").querySelector(".cm-num").textContent = cm.fp;
    document.getElementById("cm-fn").querySelector(".cm-num").textContent = cm.fn;
    document.getElementById("cm-tp").querySelector(".cm-num").textContent = cm.tp;

    // Per-Answer Breakdown Table (Table 8.1b)
    const tbody = document.getElementById("table-81b-body");
    tbody.innerHTML = "";

    data.per_answer.forEach(row => {
      const tr = document.createElement("tr");
      const hasErrors = row.false_alarms > 0 || row.false_negatives > 0;
      const statusBadge = hasErrors
        ? `<span class="badge-tag" style="background:#fef3c7; color:#92400e;">FP:${row.false_alarms} FN:${row.false_negatives}</span>`
        : `<span class="badge-tag" style="background:#dcfce7; color:#15803d;">100% Pass</span>`;

      tr.innerHTML = `
        <td><strong>${row.id}</strong></td>
        <td>${escapeHtml(row.question)}</td>
        <td>${row.total_claims}</td>
        <td><strong>${row.gold_hallucinated}</strong></td>
        <td>${row.detected} / ${row.false_alarms}</td>
        <td>${statusBadge}</td>
      `;
      tbody.appendChild(tr);
    });

    showToast("PoC Benchmark evaluation complete!");

  } catch (err) {
    console.error("Benchmark error:", err);
    showToast(`Benchmark error: ${err.message}`);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = '<span>🔄</span> Run Full Benchmark';
    }
  }
}

/* ==========================================================================
   Colab Research Hub & Interactive Test
   ========================================================================== */
async function loadHaluEvalResearch() {
  try {
    const res = await fetch("/api/halueval");
    const data = await res.json();

    // Fill Threshold Sweep Table
    const sweepBody = document.getElementById("halueval-sweep-body");
    if (sweepBody) {
      sweepBody.innerHTML = "";
      data.threshold_sweep.forEach(row => {
        const tr = document.createElement("tr");
        if (row.selected) {
          tr.style.backgroundColor = "#eff6ff";
          tr.style.fontWeight = "700";
        }
        tr.innerHTML = `
          <td>${row.threshold.toFixed(2)} ${row.selected ? '<span class="badge-tag" style="font-size:10px;">Best</span>' : ''}</td>
          <td>${row.precision.toFixed(4)}</td>
          <td>${row.recall.toFixed(4)}</td>
          <td>${row.f1_score.toFixed(4)}</td>
        `;
        sweepBody.appendChild(tr);
      });
    }

    // Fill HaluEval Samples Table
    const samplesBody = document.getElementById("halueval-samples-body");
    if (samplesBody) {
      samplesBody.innerHTML = "";
      data.samples.forEach(s => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><strong>${s.id}</strong></td>
          <td><small>${escapeHtml(s.knowledge)}</small></td>
          <td><strong>${escapeHtml(s.question)}</strong></td>
          <td><span style="color:#059669;">${escapeHtml(s.right_answer)}</span></td>
          <td><span style="color:#dc2626;"><del>${escapeHtml(s.hallucinated_answer)}</del></span></td>
        `;
        samplesBody.appendChild(tr);
      });
    }

  } catch (err) {
    console.error("Failed to load HaluEval research:", err);
  }
}

async function runColabVerify() {
  const context = document.getElementById("colab-context").value.trim();
  const answer = document.getElementById("colab-answer").value.trim();

  if (!context || !answer) {
    showToast("Please provide both context and answer.");
    return;
  }

  const btn = document.getElementById("btn-colab-run");
  btn.disabled = true;
  btn.textContent = "Scoring NLI...";

  try {
    const res = await fetch("/api/colab/verify_answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ context: context, answer: answer, threshold: 0.10 })
    });

    const data = await res.json();
    const box = document.getElementById("colab-result-box");
    box.style.display = "block";

    const statusBadge = document.getElementById("colab-res-status");
    statusBadge.textContent = data.prediction;
    if (data.is_hallucination) {
      statusBadge.style.backgroundColor = "#fee2e2";
      statusBadge.style.color = "#991b1b";
    } else {
      statusBadge.style.backgroundColor = "#dcfce7";
      statusBadge.style.color = "#15803d";
    }

    document.getElementById("colab-res-score").textContent = data.hallucination_score;
    document.getElementById("colab-res-entail").textContent = data.entailment_probability;
    document.getElementById("colab-res-thresh").textContent = data.threshold;

  } catch (err) {
    console.error("Colab verify error:", err);
    showToast("Error running Colab verify");
  } finally {
    btn.disabled = false;
    btn.textContent = "Run Colab verify_answer()";
  }
}

/* ==========================================================================
   Knowledge Base Explorer
   ========================================================================== */
async function loadKnowledgeBase() {
  try {
    const res = await fetch("/api/kb");
    const data = await res.json();

    const container = document.getElementById("kb-domains-container");
    if (!container) return;
    container.innerHTML = "";

    data.files.forEach(f => {
      const card = document.createElement("div");
      card.className = "kb-file-card";

      const listItems = f.sentences.map(s => `<li>${escapeHtml(s)}</li>`).join("");

      card.innerHTML = `
        <h4>📄 ${f.filename}</h4>
        <div class="kb-file-meta">${f.sentence_count} factual grounding sentences</div>
        <ul class="kb-sentence-list">
          ${listItems}
        </ul>
      `;
      container.appendChild(card);
    });

  } catch (err) {
    console.error("Failed to load KB:", err);
  }
}

async function submitNewFact() {
  const factText = document.getElementById("new-fact-text").value.trim();
  const targetFile = document.getElementById("new-fact-file").value;

  if (!factText) {
    showToast("Please enter a fact sentence.");
    return;
  }

  try {
    const res = await fetch("/api/kb/add", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fact: factText, source_file: targetFile })
    });

    const data = await res.json();
    if (data.success) {
      showToast("Fact successfully added and indexed!");
      document.getElementById("new-fact-text").value = "";
      document.getElementById("add-fact-container").style.display = "none";
      document.getElementById("header-kb-count").textContent = data.stats.total_sentences;
      loadKnowledgeBase();
    }
  } catch (err) {
    console.error("Error adding fact:", err);
    showToast("Error adding fact.");
  }
}

/* ==========================================================================
   Event Listeners
   ========================================================================== */
function initEventListeners() {
  // Ask AI & Guard Button
  const askAndGuardBtn = document.getElementById("btn-ask-and-guard");
  if (askAndGuardBtn) {
    askAndGuardBtn.addEventListener("click", performAskAndGuard);
  }

  // Generate Answer Only Button
  const genOnlyBtn = document.getElementById("btn-gen-only");
  if (genOnlyBtn) {
    genOnlyBtn.addEventListener("click", performGenerateOnly);
  }

  // Quick Question Prompts
  document.querySelectorAll(".quick-samples button[data-q]").forEach(btn => {
    btn.addEventListener("click", () => {
      const q = btn.getAttribute("data-q");
      document.getElementById("input-question").value = q;
      performAskAndGuard();
    });
  });

  // Preset selector change
  const presetSelect = document.getElementById("sample-preset");
  if (presetSelect) {
    presetSelect.addEventListener("change", (e) => {
      applyPreset(e.target.value);
    });
  }

  // Verify button
  const verifyBtn = document.getElementById("btn-verify");
  if (verifyBtn) {
    verifyBtn.addEventListener("click", performVerification);
  }

  // Clear button
  const clearBtn = document.getElementById("btn-clear");
  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      document.getElementById("input-question").value = "";
      document.getElementById("input-answer").value = "";
      document.getElementById("sample-preset").value = "";
      document.getElementById("annotated-output").innerHTML = '<p class="placeholder-text">Submit an answer above to see claim-by-claim color highlighting and audit details.</p>';
      document.getElementById("guarded-output").innerHTML = '<p class="placeholder-text">[Guarded output will appear here once verified]</p>';
      document.getElementById("claim-table-body").innerHTML = '<tr><td colspan="7" class="empty-table-row">No claims verified yet. Choose a sample preset or enter text above.</td></tr>';
      document.getElementById("metric-total").textContent = "0";
      const supElem = document.getElementById("metric-supported");
      if (supElem) supElem.textContent = "0";
      document.getElementById("metric-flagged").textContent = "0";
      document.getElementById("metric-rate").textContent = "0%";
      document.getElementById("metric-time").textContent = "0.0 ms";
    });
  }

  // Copy Guarded Answer
  const copyBtn = document.getElementById("btn-copy-guarded");
  if (copyBtn) {
    copyBtn.addEventListener("click", () => {
      if (!lastReport || !lastReport.guarded_answer) {
        showToast("No guarded answer available to copy.");
        return;
      }
      navigator.clipboard.writeText(lastReport.guarded_answer).then(() => {
        showToast("Guarded answer copied to clipboard!");
      });
    });
  }

  // Export JSON Report
  const exportBtn = document.getElementById("btn-export-json");
  if (exportBtn) {
    exportBtn.addEventListener("click", () => {
      if (!lastReport) {
        showToast("No verification report available to export.");
        return;
      }
      const blob = new Blob([JSON.stringify(lastReport, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `halluguard_report_${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);
    });
  }

  // Benchmark Run Button
  const bmBtn = document.getElementById("btn-run-benchmark");
  if (bmBtn) {
    bmBtn.addEventListener("click", runBenchmark);
  }

  // Colab verify button
  const colabBtn = document.getElementById("btn-colab-run");
  if (colabBtn) {
    colabBtn.addEventListener("click", runColabVerify);
  }

  // Colab quick sample chips
  document.querySelectorAll(".quick-samples .btn-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const type = chip.getAttribute("data-sample");
      if (type === "eiffel-paris") {
        document.getElementById("colab-context").value = "The Eiffel Tower is located in Paris, France.";
        document.getElementById("colab-answer").value = "The Eiffel Tower is in Paris.";
      } else if (type === "eiffel-berlin") {
        document.getElementById("colab-context").value = "The Eiffel Tower is located in Paris, France.";
        document.getElementById("colab-answer").value = "The Eiffel Tower is in Berlin.";
      } else if (type === "sun") {
        document.getElementById("colab-context").value = "The Earth revolves around the Sun. One complete revolution takes approximately 365.25 days.";
        document.getElementById("colab-answer").value = "The Earth revolves around the Sun.";
      } else if (type === "mars") {
        document.getElementById("colab-context").value = "The Earth revolves around the Sun. One complete revolution takes approximately 365.25 days.";
        document.getElementById("colab-answer").value = "The Earth revolves around Mars.";
      }
      runColabVerify();
    });
  });

  // KB Add Fact toggles
  const openAddFactBtn = document.getElementById("btn-open-add-fact");
  const cancelFactBtn = document.getElementById("btn-cancel-fact");
  const addFactContainer = document.getElementById("add-fact-container");
  const submitFactBtn = document.getElementById("btn-submit-fact");

  if (openAddFactBtn && addFactContainer) {
    openAddFactBtn.addEventListener("click", () => {
      addFactContainer.style.display = "block";
    });
  }
  if (cancelFactBtn && addFactContainer) {
    cancelFactBtn.addEventListener("click", () => {
      addFactContainer.style.display = "none";
    });
  }
  if (submitFactBtn) {
    submitFactBtn.addEventListener("click", submitNewFact);
  }

  // Audit log button
  const auditBtn = document.getElementById("btn-audit-log");
  if (auditBtn) {
    auditBtn.addEventListener("click", () => {
      if (!lastReport) {
        showToast("Please run verification first.");
        return;
      }
      alert(`Audit Log:\n- Total claims checked: ${lastReport.total_claims}\n- Supported: ${lastReport.supported_claims}\n- Contradicted: ${lastReport.contradicted_claims}\n- Unsupported: ${lastReport.unsupported_claims}\n- Hallucination Rate: ${lastReport.hallucination_rate_pct}%\n- Model latency: ${lastReport.response_time_ms} ms`);
    });
  }
}

/* ==========================================================================
   Utilities
   ========================================================================== */
function escapeHtml(text) {
  if (!text) return "";
  const map = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;"
  };
  return text.replace(/[&<>"']/g, m => map[m]);
}

function showToast(msg) {
  const toast = document.getElementById("toast");
  if (!toast) return;
  toast.textContent = msg;
  toast.classList.add("show");
  setTimeout(() => {
    toast.classList.remove("show");
  }, 3200);
}

