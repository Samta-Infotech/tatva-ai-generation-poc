"""Browser UI HTML and legacy launcher for the TATVA AI Generation POC.

The active backend is FastAPI in ``api.py``. This module keeps the HTML and
artifact helpers in one place, and ``python ui.py`` now launches FastAPI for
backward-compatible local demos.
"""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from config.runtime_profiles import all_runtime_profiles
from config.settings import OUTPUT_DIR, ROOT_DIR
from main import PBQ_BENCHMARK_PROMPTS
from mcq.graph import run_mcq_graph
from mcq.schemas import MCQRequest
from pbq.graph import run_pbq_graph
from pbq.schemas import PBQRequest
from shared.logging import read_json


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TATVA AI Generation POC</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #18202a;
      --muted: #647184;
      --line: #dce2ea;
      --accent: #126b5a;
      --accent-dark: #0d5749;
      --danger: #a33b35;
      --warn: #946200;
      --ok: #1f7a4d;
      --code: #101820;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      position: sticky;
      top: 0;
      z-index: 5;
    }
    h1 { margin: 0; font-size: 20px; letter-spacing: 0; }
    main {
      display: grid;
      grid-template-columns: minmax(320px, 420px) minmax(0, 1fr);
      gap: 18px;
      padding: 18px;
      max-width: 1440px;
      margin: 0 auto;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }
    h2 { margin: 0 0 12px; font-size: 16px; }
    label { display: grid; gap: 6px; margin: 10px 0; color: var(--muted); font-weight: 600; }
    input, select, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      font: inherit;
      color: var(--text);
      background: #fff;
    }
    textarea { min-height: 86px; resize: vertical; }
    button {
      border: 0;
      border-radius: 6px;
      padding: 10px 12px;
      font-weight: 700;
      cursor: pointer;
      background: var(--accent);
      color: white;
    }
    button:hover { background: var(--accent-dark); }
    button.secondary { background: #27313d; }
    button.ghost {
      background: transparent;
      color: var(--accent);
      border: 1px solid var(--line);
    }
    .row { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
    .tabs { display: flex; gap: 8px; margin-bottom: 14px; }
    .tab[aria-selected="true"] { background: #27313d; color: white; }
    .tab[aria-selected="false"] { background: #eef2f6; color: #27313d; }
    .hidden { display: none; }
    .status {
      min-height: 38px;
      padding: 10px 12px;
      border-radius: 6px;
      background: #eef2f6;
      color: var(--muted);
      margin-top: 12px;
    }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fbfcfd;
    }
    .metric strong { display: block; font-size: 22px; color: var(--text); }
    .metric span { color: var(--muted); }
    table { width: 100%; border-collapse: collapse; margin-top: 10px; }
    th, td { text-align: left; padding: 9px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
    th { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
    .pill {
      display: inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      background: #eef2f6;
      color: #27313d;
      font-weight: 700;
      white-space: nowrap;
    }
    .pill.passed { color: var(--ok); background: #e8f5ee; }
    .pill.killed { color: var(--ok); background: #e8f5ee; }
    .pill.survived { color: var(--danger); background: #faecea; }
    .pill.failed, .pill.reference_failed, .pill.weak_tests { color: var(--danger); background: #faecea; }
    .pill.needs_repair { color: var(--warn); background: #fff4d7; }
    pre {
      min-height: 360px;
      max-height: 640px;
      overflow: auto;
      background: var(--code);
      color: #e7edf4;
      padding: 14px;
      border-radius: 8px;
      font-size: 12px;
      white-space: pre-wrap;
    }
    .artifact-list { display: grid; gap: 6px; margin-top: 8px; }
    .artifact-list button { text-align: left; background: #eef2f6; color: #27313d; font-weight: 600; }
    .tree-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
      margin-top: 10px;
    }
    .tree-box {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fbfcfd;
    }
    .tree-box h3 { margin: 0 0 8px; font-size: 14px; }
    .tree-output {
      min-height: 120px;
      max-height: 320px;
      overflow: auto;
      background: #111820;
      color: #e7edf4;
      padding: 10px;
      border-radius: 6px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      white-space: pre;
    }
    .explain {
      display: grid;
      gap: 8px;
      color: var(--muted);
      margin-top: 10px;
    }
    .explain p { margin: 0; }
    .review-actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      margin-bottom: 10px;
    }
    .review-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 12px;
      margin-top: 10px;
    }
    .review-block h3 {
      margin: 0 0 8px;
      font-size: 14px;
      color: var(--text);
    }
    .review-block pre {
      min-height: 160px;
      max-height: 380px;
      margin: 0;
    }
    .workspace-toolbar {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      padding-bottom: 10px;
      border-bottom: 1px solid var(--line);
      margin-bottom: 12px;
    }
    .workspace-layout {
      display: grid;
      grid-template-columns: minmax(220px, 320px) minmax(320px, 1fr) minmax(220px, 300px);
      border: 1px solid var(--line);
      border-radius: 8px;
      min-height: 460px;
      overflow: hidden;
      background: #fff;
    }
    .workspace-pane {
      min-width: 0;
      border-right: 1px solid var(--line);
      padding: 12px;
      overflow: auto;
    }
    .workspace-pane:last-child { border-right: 0; }
    .workspace-title {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      margin-bottom: 10px;
      color: var(--muted);
      font-weight: 700;
      text-transform: uppercase;
      font-size: 12px;
      letter-spacing: .04em;
    }
    .file-list { display: grid; gap: 6px; }
    .file-button {
      border: 1px solid transparent;
      background: #eef4fb;
      color: #27313d;
      text-align: left;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      padding: 8px 10px;
      font-weight: 600;
    }
    .file-button.active { border-color: var(--accent); background: #e7f6f2; }
    .file-tag {
      color: #6a7a90;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
    }
    .workspace-editor {
      width: 100%;
      min-height: 320px;
      resize: vertical;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.45;
      color: #12202f;
    }
    .dependency-list {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 8px;
    }
    .dependency-list span {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 10px;
      background: #fbfcfd;
      color: #334155;
      font-weight: 600;
    }
    @media (max-width: 1100px) {
      .workspace-layout { grid-template-columns: 1fr; }
      .workspace-pane { border-right: 0; border-bottom: 1px solid var(--line); }
    }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; padding: 12px; }
      header { align-items: flex-start; flex-direction: column; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>TATVA AI Generation POC</h1>
      <div>Run PBQ/MCQ generation and inspect real benchmark artifacts.</div>
    </div>
    <div class="row">
      <button class="ghost" onclick="refreshResults()">Refresh Results</button>
      <button class="ghost" onclick="testSlm()">Test SLM</button>
      <button class="secondary" onclick="runBenchmark('mcq')">Run MCQ Benchmark</button>
      <button class="secondary" onclick="runBenchmark('pbq')">Run PBQ Benchmark</button>
    </div>
  </header>
  <main>
    <div>
      <section>
        <div class="tabs" role="tablist">
          <button class="tab" id="pbqTab" aria-selected="true" onclick="showTab('pbq')">PBQ</button>
          <button class="tab" id="mcqTab" aria-selected="false" onclick="showTab('mcq')">MCQ</button>
        </div>
        <form id="pbqForm">
          <h2>Generate PBQ</h2>
          <label>Runtime Profile<select name="runtime" id="runtimeSelect"></select></label>
          <label>Difficulty<select name="difficulty"><option>easy</option><option>medium</option><option selected>hard</option></select></label>
          <label>Experience<input name="experience" value="4-6 years" /></label>
          <label>Duration Minutes<input name="duration" type="number" value="90" min="15" /></label>
          <label>Prompt<textarea name="prompt">Build an e-commerce cart management system with validation, discounting, and tax calculation.</textarea></label>
          <div class="row">
            <button type="submit">Run PBQ</button>
            <button type="button" class="secondary" onclick="designOnlyPbq()">Design Only</button>
            <button type="button" class="ghost" onclick="validatePbq()">Validate PBQ</button>
            <button type="button" class="ghost" onclick="regeneratePbqFromValidation()">Fix With AI</button>
          </div>
        </form>
        <form id="mcqForm" class="hidden">
          <h2>Generate MCQ</h2>
          <label>Skill<input name="skill" value="Python" /></label>
          <label>Topic<input name="topic" value="concurrency" /></label>
          <label>Difficulty<select name="difficulty"><option>easy</option><option>medium</option><option selected>hard</option></select></label>
          <label>Experience<input name="experience" value="4-6 years" /></label>
          <div class="row">
            <button type="submit">Run MCQ</button>
            <button type="button" class="ghost" onclick="validateMcq()">Validate MCQ</button>
          </div>
        </form>
        <div class="status" id="status">Ready.</div>
      </section>
      <section style="margin-top: 18px;">
        <h2>Artifacts</h2>
        <div id="artifacts" class="artifact-list"></div>
      </section>
    </div>
    <div>
      <section>
        <h2>Benchmark Summary</h2>
        <div id="summary" class="summary-grid"></div>
        <div id="tables"></div>
        <div id="treeSection"></div>
        <div class="explain">
          <h2 style="margin-top:18px;">How Benchmarking Works</h2>
          <p><strong>PBQ benchmark:</strong> runs one framework-appropriate PBQ through each runtime profile, materializes candidate/reference workspaces, runs the reference validation/test commands, runs private tests against intentional mutations, then records status, command output, mutation score, repairs, and timing.</p>
          <p><strong>MCQ benchmark:</strong> runs easy, medium, and hard MCQ generation, validates structure, asks an independent solver for the answer, checks distractors, judges difficulty, and records agreement plus metrics.</p>
          <p><strong>Infrastructure failures:</strong> missing tools such as <code>vite</code>, <code>vitest</code>, <code>jest</code>, or <code>mvn</code> are shown as real failures. They mean the local runtime dependencies are unavailable, not that the generated reference solution is logically wrong.</p>
        </div>
      </section>
      <section style="margin-top: 18px;">
        <h2>Selected JSON</h2>
        <pre id="jsonView">Select an artifact or run a graph.</pre>
      </section>
      <section style="margin-top: 18px;">
        <div class="workspace-toolbar">
          <div>
            <h2>PBQ Project Workspace</h2>
            <div style="color: var(--muted);">Edit generated question/files, validate, then regenerate with AI from errors.</div>
          </div>
          <div class="row">
            <button type="button" class="ghost" onclick="loadActiveWorkspace()">Load Workspace</button>
            <button type="button" class="ghost" onclick="saveWorkspaceDesign()">Save Question</button>
            <button type="button" class="ghost" onclick="saveWorkspaceFile()">Save File</button>
            <button type="button" onclick="validatePbq()">Validate</button>
            <button type="button" class="secondary" onclick="regeneratePbqFromValidation()">Regenerate With AI</button>
          </div>
        </div>
        <label>Question Title<input id="workspaceTitle" placeholder="Generated PBQ title" /></label>
        <label>Question Prompt<textarea id="workspacePrompt" placeholder="Prompt used to generate or repair the PBQ"></textarea></label>
        <div class="workspace-layout">
          <div class="workspace-pane">
            <div class="workspace-title"><span>Files</span><span id="workspaceFileCount">0</span></div>
            <div id="workspaceFiles" class="file-list"></div>
          </div>
          <div class="workspace-pane">
            <div class="workspace-title"><span id="workspaceSelectedPath">Select a file</span><span id="workspaceSelectedSection"></span></div>
            <textarea id="workspaceEditor" class="workspace-editor" placeholder="Select a file from the tree to edit it."></textarea>
          </div>
          <div class="workspace-pane">
            <div class="workspace-title"><span>Validation</span></div>
            <div id="workspaceStatus" class="explain"><p>No workspace loaded yet.</p></div>
            <div class="workspace-title" style="margin-top:16px;"><span>Approved Dependencies</span></div>
            <div id="workspaceDependencies" class="dependency-list"></div>
          </div>
        </div>
      </section>
      <section style="margin-top: 18px;">
        <h2>Validation Output</h2>
        <div id="validationOutput" class="explain">
          <p>Run or validate a PBQ/MCQ to see testcase output here.</p>
        </div>
      </section>
      <section style="margin-top: 18px;">
        <h2>Solution / Review Packet</h2>
        <div class="review-actions">
          <button type="button" class="ghost" onclick="showReviewPacket()">Show Solution Below</button>
          <button type="button" class="ghost" onclick="openReviewPacketWindow()">Open Separate Window</button>
        </div>
        <div id="reviewPacketOutput" class="explain">
          <p>Generate or validate a PBQ, then open the review packet to see the AI reference solution, folder trees, question, and testcase output.</p>
        </div>
      </section>
    </div>
  </main>
  <script>
    const statusEl = document.getElementById('status');
    const jsonView = document.getElementById('jsonView');
    const validationOutput = document.getElementById('validationOutput');
    const reviewPacketOutput = document.getElementById('reviewPacketOutput');
    let latestReviewPacket = null;
    let workspaceState = null;
    let selectedWorkspaceFileIndex = -1;

    function setStatus(text) { statusEl.textContent = text; }

    async function api(path, options = {}) {
      const response = await fetch(path, {
        headers: { 'Content-Type': 'application/json' },
        ...options
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || payload.detail || response.statusText);
      return payload;
    }

    function showTab(name) {
      document.getElementById('pbqForm').classList.toggle('hidden', name !== 'pbq');
      document.getElementById('mcqForm').classList.toggle('hidden', name !== 'mcq');
      document.getElementById('pbqTab').setAttribute('aria-selected', name === 'pbq');
      document.getElementById('mcqTab').setAttribute('aria-selected', name === 'mcq');
    }

    function formData(form) {
      return Object.fromEntries(new FormData(form).entries());
    }

    async function loadProfiles() {
      const payload = await api('/api/profiles');
      const select = document.getElementById('runtimeSelect');
      select.innerHTML = payload.profiles.map(p => `<option value="${p.id}">${p.id} - ${p.framework}</option>`).join('');
    }

    async function runGraph(kind, payload) {
      setStatus(`Running ${kind.toUpperCase()}...`);
      jsonView.textContent = 'Working...';
      const result = await api(`/api/${kind}`, { method: 'POST', body: JSON.stringify(payload) });
      jsonView.textContent = JSON.stringify(result.result, null, 2);
      renderValidationOutput(result.result);
      setStatus(`${kind.toUpperCase()} finished with status: ${result.result.final_status}`);
      if (kind === 'pbq') {
        await loadReviewPacket(payload.runtime, false);
        await loadPbqWorkspace(payload.runtime, false);
      }
      await refreshResults();
    }

    async function validatePbq() {
      const payload = formData(document.getElementById('pbqForm'));
      setStatus(`Validating PBQ for ${payload.runtime}...`);
      jsonView.textContent = 'Validating saved PBQ bundle...';
      const result = await api('/api/validate-pbq', { method: 'POST', body: JSON.stringify({ runtime: payload.runtime }) });
      jsonView.textContent = JSON.stringify(result.result, null, 2);
      renderValidationOutput(result.result);
      setStatus(`PBQ validation finished with status: ${result.result.final_status}`);
      await loadReviewPacket(payload.runtime, false);
      await loadPbqWorkspace(payload.runtime, false);
      await refreshResults();
    }

    async function regeneratePbqFromValidation() {
      const payload = formData(document.getElementById('pbqForm'));
      setStatus(`Sending validation errors for ${payload.runtime} back to AI...`);
      jsonView.textContent = 'Repairing PBQ with latest validation feedback...';
      const result = await api('/api/regenerate-pbq-from-validation', {
        method: 'POST',
        body: JSON.stringify({
          runtime: payload.runtime,
          reason: 'Use the latest validation output, failed testcase names, stdout, and stderr to fix the question bundle, reference solution, and tests.'
        })
      });
      jsonView.textContent = JSON.stringify(result.result, null, 2);
      renderValidationOutput(result.result);
      setStatus(`AI repair finished with status: ${result.result.final_status}`);
      await loadReviewPacket(payload.runtime, false);
      await loadPbqWorkspace(payload.runtime, false);
      await refreshResults();
    }

    async function loadActiveWorkspace() {
      const payload = formData(document.getElementById('pbqForm'));
      await loadPbqWorkspace(payload.runtime, true);
    }

    async function loadPbqWorkspace(runtime, showErrors = true) {
      try {
        workspaceState = await api(`/api/pbq-workspace?runtime=${encodeURIComponent(runtime)}`);
        selectedWorkspaceFileIndex = workspaceState.files?.length ? 0 : -1;
        renderPbqWorkspace();
        return workspaceState;
      } catch (err) {
        if (showErrors) {
          document.getElementById('workspaceStatus').innerHTML = `<p>${escapeHtml(err.message)}</p>`;
        }
        return null;
      }
    }

    function renderPbqWorkspace() {
      if (!workspaceState) return;
      document.getElementById('workspaceTitle').value = workspaceState.design?.title || '';
      document.getElementById('workspacePrompt').value = workspaceState.prompt || '';
      document.getElementById('workspaceFileCount').textContent = `${workspaceState.files?.length || 0} files`;
      document.getElementById('workspaceDependencies').innerHTML = (workspaceState.approved_dependencies || [])
        .map(item => `<span>${escapeHtml(item)}</span>`).join('');
      document.getElementById('workspaceFiles').innerHTML = (workspaceState.files || []).map((file, index) => `
        <button type="button" class="file-button ${index === selectedWorkspaceFileIndex ? 'active' : ''}" onclick="selectWorkspaceFile(${index})">
          <span>${escapeHtml(file.path)}</span>
          <span class="file-tag">${escapeHtml(sectionLabel(file.section))}</span>
        </button>
      `).join('');
      renderWorkspaceSelectedFile();
      renderWorkspaceStatus(workspaceState.result);
    }

    function sectionLabel(section) {
      return ({
        starter_files: 'starter',
        public_tests: 'public',
        private_tests: 'private',
        reference_solution: 'reference'
      })[section] || section || '';
    }

    function selectWorkspaceFile(index) {
      selectedWorkspaceFileIndex = index;
      renderPbqWorkspace();
    }

    function renderWorkspaceSelectedFile() {
      const file = workspaceState?.files?.[selectedWorkspaceFileIndex];
      document.getElementById('workspaceSelectedPath').textContent = file?.path || 'Select a file';
      document.getElementById('workspaceSelectedSection').textContent = file ? sectionLabel(file.section) : '';
      document.getElementById('workspaceEditor').value = file?.content || '';
    }

    function renderWorkspaceStatus(result) {
      const statusEl = document.getElementById('workspaceStatus');
      if (!result) {
        statusEl.innerHTML = '<p>Generated workspace loaded. Validate to see testcase output.</p>';
        return;
      }
      const failed = result.testcase_summary?.failed_count || 0;
      statusEl.innerHTML = `
        <p>Status: <span class="pill ${escapeHtml(result.final_status || 'unknown')}">${escapeHtml(result.final_status || 'unknown')}</span></p>
        <p>Reference passed: <strong>${result.reference_execution_success}</strong></p>
        <p>Failed commands/tests: <strong>${failed}</strong></p>
        <p>Mutation score: <strong>${result.mutation_score ?? 'n/a'}</strong></p>
      `;
    }

    async function saveWorkspaceFile() {
      if (!workspaceState || selectedWorkspaceFileIndex < 0) {
        setStatus('Load a PBQ workspace and select a file first.');
        return;
      }
      const file = workspaceState.files[selectedWorkspaceFileIndex];
      setStatus(`Saving ${file.path}...`);
      workspaceState = await api('/api/pbq-workspace/file', {
        method: 'POST',
        body: JSON.stringify({
          runtime: workspaceState.runtime,
          section: file.section,
          path: file.path,
          content: document.getElementById('workspaceEditor').value
        })
      });
      const sameIndex = workspaceState.files.findIndex(item => item.section === file.section && item.path === file.path);
      selectedWorkspaceFileIndex = sameIndex >= 0 ? sameIndex : 0;
      renderPbqWorkspace();
      setStatus(`Saved ${file.path}.`);
    }

    async function saveWorkspaceDesign() {
      if (!workspaceState) {
        await loadActiveWorkspace();
        if (!workspaceState) return;
      }
      setStatus('Saving question text...');
      workspaceState = await api('/api/pbq-workspace/design', {
        method: 'POST',
        body: JSON.stringify({
          runtime: workspaceState.runtime,
          title: document.getElementById('workspaceTitle').value,
          prompt: document.getElementById('workspacePrompt').value,
          behavioral_contract: workspaceState.design?.behavioral_contract || [],
          scaffolding_strategy: workspaceState.design?.scaffolding_strategy || '',
          candidate_freedom: workspaceState.design?.candidate_freedom || [],
          difficulty_notes: workspaceState.design?.difficulty_notes || []
        })
      });
      renderPbqWorkspace();
      setStatus('Question saved.');
    }

    async function validateMcq() {
      const payload = formData(document.getElementById('mcqForm'));
      setStatus(`Validating ${payload.difficulty} MCQ...`);
      jsonView.textContent = 'Validating saved MCQ...';
      const result = await api('/api/validate-mcq', { method: 'POST', body: JSON.stringify({ difficulty: payload.difficulty }) });
      jsonView.textContent = JSON.stringify(result.result, null, 2);
      renderValidationOutput(result.result);
      setStatus(`MCQ validation finished with status: ${result.result.final_status}`);
      await refreshResults();
    }

    async function runBenchmark(kind) {
      setStatus(`Running ${kind.toUpperCase()} benchmark...`);
      const result = await api(`/api/benchmark-${kind}`, { method: 'POST', body: '{}' });
      jsonView.textContent = JSON.stringify(result.summary, null, 2);
      setStatus(`${kind.toUpperCase()} benchmark finished.`);
      await refreshResults();
    }

    function commandBlock(item) {
      return `
        <div class="tree-box">
          <h3>${escapeHtml(item.command || 'command')}</h3>
          <p>exit_code: <strong>${item.exit_code}</strong> · timed_out: ${item.timed_out} · skipped: ${item.skipped}</p>
          <strong>stdout</strong>
          <pre class="tree-output">${escapeHtml(item.stdout || '')}</pre>
          <strong>stderr</strong>
          <pre class="tree-output">${escapeHtml(item.stderr || item.skip_reason || '')}</pre>
        </div>
      `;
    }

    function renderValidationOutput(result) {
      if (!result) {
        validationOutput.innerHTML = '<p>No validation output yet.</p>';
        return;
      }
      if (Array.isArray(result.execution_results)) {
        const commands = result.execution_results.map(commandBlock).join('');
        const testcaseSummary = renderTestcaseSummary(result.testcase_summary);
        const mutations = (result.mutation_results?.results || []).map(item => `
          <div class="tree-box">
            <h3>Mutation: ${escapeHtml(item.name)} · killed: ${item.killed} · inconclusive: ${item.inconclusive}</h3>
            ${commandBlock(item.execution || {})}
          </div>
        `).join('');
        validationOutput.innerHTML = `
          <p>Review packet folder: <code>${escapeHtml(result.review_packet_path || 'not written')}</code></p>
          <p>Final status: <span class="pill ${escapeHtml(result.final_status || 'unknown')}">${escapeHtml(result.final_status || 'unknown')}</span></p>
          ${testcaseSummary}
          ${(result.validation_repairs || []).length ? `<h3>AI Repair Notes</h3><pre class="tree-output">${escapeHtml((result.validation_repairs || []).join('\\n'))}</pre>` : ''}
          <h3>Reference/Test Commands</h3>
          ${commands || '<p>No command output recorded.</p>'}
          <h3>Mutation Testcases</h3>
          ${mutations || '<p>No mutation output recorded.</p>'}
        `;
        return;
      }
      if ('answer_agreement' in result) {
        validationOutput.innerHTML = `
          <div class="tree-box">
            <h3>MCQ Validation</h3>
            <p>final_status: <strong>${escapeHtml(result.final_status)}</strong></p>
            <p>generator_answer: <strong>${escapeHtml(result.generator_answer)}</strong></p>
            <p>independent_solver_answer: <strong>${escapeHtml(result.independent_solver_answer)}</strong></p>
            <p>answer_agreement: <strong>${result.answer_agreement}</strong></p>
            <p>distractor_score: <strong>${result.distractor_score}</strong></p>
            <p>predicted_difficulty: <strong>${escapeHtml(result.predicted_difficulty)}</strong></p>
          </div>
        `;
        return;
      }
      validationOutput.innerHTML = '<p>No validation output available for this artifact.</p>';
    }

    function renderTestcaseSummary(summary) {
      if (!summary) return '';
      const commandRows = (summary.commands || []).map(item => testcaseRow(item)).join('');
      const mutationRows = (summary.mutations || []).map(item => testcaseRow(item)).join('');
      return `
        <h3>Testcase Summary</h3>
        <table>
          <thead><tr><th>Command</th><th>Status</th><th>Passed</th><th>Failed Tests / Reason</th></tr></thead>
          <tbody>${commandRows}${mutationRows}</tbody>
        </table>
      `;
    }

    function testcaseRow(item) {
      const failed = (item.failed_tests || []).join(', ');
      const reason = failed || item.reason || 'ok';
      const label = item.mutation ? `${item.command || ''} (${item.mutation})` : item.command;
      return `
        <tr>
          <td>${escapeHtml(label || 'command')}</td>
          <td><span class="pill ${item.status}">${escapeHtml(item.status)}</span></td>
          <td>${item.passed_count || 0}</td>
          <td><pre class="tree-output">${escapeHtml(reason)}</pre></td>
        </tr>
      `;
    }

    async function testSlm() {
      setStatus('Testing SLM gateway...');
      const status = await api('/api/slm/status');
      const result = await api('/api/slm/test', { method: 'POST', body: '{}' });
      jsonView.textContent = JSON.stringify({ status, test: result }, null, 2);
      setStatus(`SLM test ${result.status}.`);
    }

    function activeRuntime() {
      return document.getElementById('runtimeSelect').value || 'django-sqlite-py312';
    }

    async function showReviewPacket() {
      const runtime = activeRuntime();
      setStatus(`Loading review packet for ${runtime}...`);
      await loadReviewPacket(runtime, true);
      setStatus(`Review packet loaded for ${runtime}.`);
    }

    async function loadReviewPacket(runtime, showErrors = true) {
      reviewPacketOutput.innerHTML = '<p>Loading solution and workspace tree...</p>';
      try {
        const packet = await api(`/api/review-packet?runtime=${encodeURIComponent(runtime)}`);
        latestReviewPacket = packet;
        renderReviewPacket(packet);
        return packet;
      } catch (err) {
        if (showErrors) {
          reviewPacketOutput.innerHTML = `<p>${escapeHtml(err.message)}</p>`;
        } else {
          reviewPacketOutput.innerHTML = '<p>Review packet not available yet. Validate this PBQ once to create it.</p>';
        }
        return null;
      }
    }

    async function openReviewPacketWindow() {
      const popup = window.open('', '_blank');
      if (!popup) {
        setStatus('Popup blocked. Use Show Solution Below.');
        return;
      }
      popup.document.write('<p>Loading review packet...</p>');
      const packet = latestReviewPacket || await loadReviewPacket(activeRuntime(), true);
      if (!packet) {
        popup.document.body.innerHTML = '<p>No review packet available.</p>';
        return;
      }
      popup.document.open();
      popup.document.write(reviewPacketWindowHtml(packet));
      popup.document.close();
    }

    function fileText(packet, name) {
      return packet?.files?.[name] || '';
    }

    function renderReviewPacket(packet) {
      const result = packet.result || {};
      reviewPacketOutput.innerHTML = `
        <p>Folder: <code>${escapeHtml(packet.review_packet_path)}</code></p>
        <p>Status: <span class="pill ${escapeHtml(result.final_status || 'unknown')}">${escapeHtml(result.final_status || 'unknown')}</span></p>
        <p>Tree check: candidate has <strong>${(result.candidate_workspace_tree || []).length}</strong> entries, reference has <strong>${(result.reference_workspace_tree || []).length}</strong> entries. If status is <strong>passed</strong>, the reference solution and tests ran successfully.</p>
        ${packet.missing?.length ? `<p>Missing files: ${escapeHtml(packet.missing.join(', '))}</p>` : ''}
        <div class="review-grid">
          ${reviewBlock('Question', fileText(packet, 'QUESTION.md'))}
          ${reviewBlock('Reference Solution', fileText(packet, 'REFERENCE_SOLUTION.md'))}
          ${reviewBlock('Candidate Folder Tree', fileText(packet, 'CANDIDATE_TREE.txt'))}
          ${reviewBlock('Reference Folder Tree', fileText(packet, 'REFERENCE_TREE.txt'))}
          ${reviewBlock('Public Tests', fileText(packet, 'PUBLIC_TESTS.md'))}
          ${reviewBlock('Private Tests', fileText(packet, 'PRIVATE_TESTS.md'))}
          ${reviewBlock('Validation Output', fileText(packet, 'VALIDATION_OUTPUT.md'))}
        </div>
      `;
    }

    function reviewBlock(title, content) {
      return `
        <div class="review-block">
          <h3>${escapeHtml(title)}</h3>
          <pre>${escapeHtml(content || '(not available)')}</pre>
        </div>
      `;
    }

    function reviewPacketWindowHtml(packet) {
      const result = packet.result || {};
      return `<!doctype html>
        <html>
        <head>
          <meta charset="utf-8" />
          <title>PBQ Review Packet - ${escapeHtml(packet.runtime)}</title>
          <style>
            body { margin: 0; padding: 20px; font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7f9; color: #18202a; }
            h1 { margin: 0 0 6px; font-size: 22px; }
            h2 { margin: 18px 0 8px; font-size: 16px; }
            .meta { color: #647184; margin-bottom: 16px; }
            pre { white-space: pre-wrap; overflow: auto; background: #101820; color: #e7edf4; border-radius: 8px; padding: 14px; max-height: 520px; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 14px; }
          </style>
        </head>
        <body>
          <h1>PBQ Review Packet</h1>
          <div class="meta">Runtime: ${escapeHtml(packet.runtime)} · Status: ${escapeHtml(result.final_status || 'unknown')} · Folder: ${escapeHtml(packet.review_packet_path)}</div>
          <div class="grid">
            ${windowReviewBlock('Question', fileText(packet, 'QUESTION.md'))}
            ${windowReviewBlock('Reference Solution', fileText(packet, 'REFERENCE_SOLUTION.md'))}
            ${windowReviewBlock('Candidate Folder Tree', fileText(packet, 'CANDIDATE_TREE.txt'))}
            ${windowReviewBlock('Reference Folder Tree', fileText(packet, 'REFERENCE_TREE.txt'))}
            ${windowReviewBlock('Public Tests', fileText(packet, 'PUBLIC_TESTS.md'))}
            ${windowReviewBlock('Private Tests', fileText(packet, 'PRIVATE_TESTS.md'))}
            ${windowReviewBlock('Validation Output', fileText(packet, 'VALIDATION_OUTPUT.md'))}
          </div>
        </body>
        </html>`;
    }

    function windowReviewBlock(title, content) {
      return `<section><h2>${escapeHtml(title)}</h2><pre>${escapeHtml(content || '(not available)')}</pre></section>`;
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, char => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'
      }[char]));
    }

    function renderPbqRows(results) {
      return `<table><thead><tr><th>Profile</th><th>Status</th><th>Reference</th><th>Mutation</th><th>Why</th></tr></thead><tbody>` +
        results.map(item => {
          const failed = (item.execution_results || []).find(r => r.exit_code !== 0 || r.skipped || r.timed_out);
          const why = failed ? (failed.skip_reason || failed.stderr || failed.command || '').slice(0, 160) : 'ok';
          const inconclusive = item.mutation_results?.inconclusive_mutations || 0;
          const mutationText = inconclusive ? `${item.mutation_score} (${inconclusive} inconclusive)` : item.mutation_score;
          const title = item.question_design?.title || item.runtime_profile.framework;
          return `<tr>
            <td><strong>${item.runtime_profile.id}</strong><br><span style="color: var(--muted);">${escapeHtml(title)}</span></td>
            <td><span class="pill ${item.final_status}">${item.final_status}</span></td>
            <td>${item.reference_execution_success}</td>
            <td>${mutationText}</td>
            <td>${escapeHtml(why)}</td>
          </tr>`;
        }).join('') + `</tbody></table>`;
    }

    function renderMcqRows(results) {
      return `<table><thead><tr><th>Difficulty</th><th>Status</th><th>Answer Agreement</th><th>Distractors</th><th>Reasoning</th></tr></thead><tbody>` +
        results.map(item => `<tr>
          <td>${item.requested_difficulty}</td>
          <td><span class="pill ${item.final_status}">${item.final_status}</span></td>
          <td>${item.answer_agreement}</td>
          <td>${item.distractor_score}</td>
          <td>${item.reasoning_depth}</td>
        </tr>`).join('') + `</tbody></table>`;
    }

    function makeTree(paths) {
      const root = {};
      for (const raw of paths || []) {
        const isDir = raw.endsWith('/');
        const parts = raw.replace(/\\/$/, '').split('/').filter(Boolean);
        let node = root;
        for (let index = 0; index < parts.length; index += 1) {
          const part = parts[index] + (index === parts.length - 1 && isDir ? '/' : '');
          node[part] = node[part] || {};
          node = node[part];
        }
      }
      function lines(node, prefix = '') {
        const entries = Object.keys(node).sort((a, b) => a.localeCompare(b));
        return entries.flatMap((entry, index) => {
          const last = index === entries.length - 1;
          const branch = last ? '└── ' : '├── ';
          const nextPrefix = prefix + (last ? '    ' : '│   ');
          return [`${prefix}${branch}${entry}`, ...lines(node[entry], nextPrefix)];
        });
      }
      return lines(root).join('\\n') || '(empty)';
    }

    function renderTreeCards(results) {
      if (!results || !results.length) return '';
      return `<h2 style="margin-top:18px;">Workspace Trees</h2>` + results.map(item => `
        <div class="tree-box">
          <h3>${item.runtime_profile.id} · ${item.final_status}</h3>
          <div class="tree-grid">
            <div>
              <strong>Candidate Workspace</strong>
              <pre class="tree-output">${escapeHtml(makeTree(item.candidate_workspace_tree || []))}</pre>
            </div>
            <div>
              <strong>Reference Workspace</strong>
              <pre class="tree-output">${escapeHtml(makeTree(item.reference_workspace_tree || []))}</pre>
            </div>
          </div>
        </div>
      `).join('');
    }

    async function refreshResults() {
      const data = await api('/api/results');
      const pbqCount = data.pbq_summary?.results?.length || 0;
      const mcqCount = data.mcq_summary?.results?.length || 0;
      document.getElementById('summary').innerHTML = `
        <div class="metric"><strong>${data.pbq_summary?.passed ?? 0}/${pbqCount}</strong><span>PBQ profiles passed</span></div>
        <div class="metric"><strong>${data.mcq_summary?.passed ?? 0}/${mcqCount}</strong><span>MCQ difficulties passed</span></div>
        <div class="metric"><strong>${data.artifacts.length}</strong><span>JSON artifacts</span></div>
      `;
      document.getElementById('tables').innerHTML = `
        <h2>PBQ</h2>${data.pbq_summary ? renderPbqRows(data.pbq_summary.results || []) : '<p>No PBQ summary yet.</p>'}
        <h2 style="margin-top:18px;">MCQ</h2>${data.mcq_summary ? renderMcqRows(data.mcq_summary.results || []) : '<p>No MCQ summary yet.</p>'}
      `;
      document.getElementById('treeSection').innerHTML = data.pbq_summary ? renderTreeCards(data.pbq_summary.results || []) : '';
      document.getElementById('artifacts').innerHTML = data.artifacts.map(path =>
        `<button type="button" onclick="loadArtifact('${path}')">${path}</button>`
      ).join('');
    }

    async function loadArtifact(path) {
      const data = await api(`/api/artifact?path=${encodeURIComponent(path)}`);
      jsonView.textContent = JSON.stringify(data.content, null, 2);
      renderValidationOutput(data.content);
    }

    document.getElementById('pbqForm').addEventListener('submit', event => {
      event.preventDefault();
      const payload = formData(event.currentTarget);
      payload.duration = Number(payload.duration || 90);
      runGraph('pbq', payload).catch(err => setStatus(err.message));
    });
    document.getElementById('mcqForm').addEventListener('submit', event => {
      event.preventDefault();
      runGraph('mcq', formData(event.currentTarget)).catch(err => setStatus(err.message));
    });

    loadProfiles().then(refreshResults).catch(err => setStatus(err.message));
  </script>
</body>
</html>
"""

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TATVA PBQ Generator</title>
  <style>
    :root {
      --bg: #f5f7fa;
      --panel: #fff;
      --text: #17202a;
      --muted: #667085;
      --line: #d8dee8;
      --accent: #0f766e;
      --accent-dark: #115e59;
      --danger: #b42318;
      --ok: #157347;
      --warn: #b7791f;
      --code: #101820;
    }
    /* ── Progress steps ── */
    .progress-wrap { margin-top: 12px; }
    .progress-bar-track {
      height: 6px; background: #e2e8f0; border-radius: 999px; overflow: hidden; margin-bottom: 10px;
    }
    .progress-bar-fill {
      height: 100%; background: var(--accent); border-radius: 999px;
      transition: width 0.4s ease;
    }
    .steps-list { display: grid; gap: 5px; }
    .step-row {
      display: grid; grid-template-columns: 22px 1fr auto;
      gap: 8px; align-items: start; padding: 7px 10px;
      border-radius: 6px; background: #f8f9fb;
      border: 1px solid transparent; font-size: 13px;
    }
    .step-row.running { background: #eff6ff; border-color: #bfdbfe; }
    .step-row.done { background: #f0fdf4; }
    .step-row.error { background: #fef2f2; border-color: #fecaca; }
    .step-row.skipped { opacity: .6; }
    .step-icon {
      width: 20px; height: 20px; border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-size: 11px; font-weight: 800; flex-shrink: 0; margin-top: 1px;
    }
    .step-icon.pending { background: #e2e8f0; color: #94a3b8; }
    .step-icon.running {
      background: #dbeafe; color: #2563eb;
      border: 2px solid #bfdbfe;
      border-top-color: #2563eb;
      animation: spin .8s linear infinite;
    }
    .step-icon.done { background: #dcfce7; color: #16a34a; }
    .step-icon.error { background: #fee2e2; color: #dc2626; }
    .step-icon.skipped { background: #f1f5f9; color: #94a3b8; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .step-body { min-width: 0; }
    .step-label { font-weight: 700; color: var(--text); }
    .step-msg { color: var(--muted); font-size: 11px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .step-ai-hint { color: #6366f1; font-size: 10px; font-style: italic; margin-top: 2px; }
    .step-elapsed { color: var(--muted); font-size: 11px; white-space: nowrap; align-self: center; }
    /* ── Template preview ── */
    .tpl-toggle {
      background: none; border: 0; color: var(--accent); font-size: 12px;
      font-weight: 700; cursor: pointer; padding: 0; text-decoration: underline;
    }
    .tpl-files { margin-top: 8px; display: grid; gap: 4px; max-height: 200px; overflow: auto; }
    .tpl-file-btn {
      display: grid; grid-template-columns: 1fr auto; gap: 8px;
      text-align: left; background: #eef4fb; color: #27313d;
      font-weight: 700; font-size: 12px; padding: 6px 8px; border-radius: 4px; border: 0; cursor: pointer;
    }
    .tpl-file-btn:hover { background: #dbeafe; }
    .tpl-file-btn.active { outline: 2px solid var(--accent); background: #e7f6f2; }
    .tpl-preview { margin-top: 8px; }
    .tpl-locked-badge {
      font-size: 10px; font-weight: 800; text-transform: uppercase;
      background: #fef3c7; color: #92400e; padding: 2px 5px; border-radius: 4px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      padding: 16px 22px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 4;
    }
    h1 { margin: 0; font-size: 20px; }
    h2 { margin: 0 0 12px; font-size: 16px; }
    h3 { margin: 0 0 8px; font-size: 14px; }
    main {
      display: grid;
      grid-template-columns: minmax(320px, 420px) minmax(0, 1fr);
      gap: 16px;
      max-width: 1440px;
      margin: 0 auto;
      padding: 16px;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }
    label {
      display: grid;
      gap: 6px;
      margin: 10px 0;
      color: var(--muted);
      font-weight: 700;
    }
    input, select, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      color: var(--text);
      background: #fff;
      font: inherit;
    }
    textarea { min-height: 140px; resize: vertical; }
    button {
      border: 0;
      border-radius: 6px;
      padding: 10px 12px;
      background: var(--accent);
      color: #fff;
      font-weight: 800;
      cursor: pointer;
    }
    button:hover { background: var(--accent-dark); }
    button.secondary { background: #2f3a46; }
    button.ghost {
      background: #fff;
      color: var(--accent);
      border: 1px solid var(--line);
    }
    .row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    .status {
      padding: 10px 12px;
      border-radius: 6px;
      background: #eef4f7;
      color: var(--muted);
      margin-top: 12px;
      min-height: 40px;
    }
    .stack { display: grid; gap: 16px; }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fbfcfd;
    }
    .metric strong { display: block; font-size: 20px; }
    .metric span { color: var(--muted); }
    .pill {
      display: inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      background: #eef2f6;
      color: #344054;
      font-weight: 800;
      white-space: nowrap;
    }
    .pill.passed, .pill.killed { color: var(--ok); background: #e8f5ee; }
    .pill.failed, .pill.reference_failed, .pill.weak_tests, .pill.survived { color: var(--danger); background: #faecea; }
    .pill.difficulty_mismatch { color: var(--warn); background: #fff4d7; }
    .tabs { display: flex; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; }
    .tabs button { background: #eef2f6; color: #344054; }
    .tabs button.active { background: #2f3a46; color: #fff; }
    .two-col {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px;
    }
    .preview-grid {
      display: grid;
      grid-template-columns: minmax(280px, .9fr) minmax(360px, 1.1fr);
      gap: 12px;
      align-items: start;
    }
    .tree-stack {
      display: grid;
      gap: 12px;
    }
    .box {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
      padding: 12px;
      min-width: 0;
    }
    pre {
      margin: 0;
      background: var(--code);
      color: #e7edf4;
      border-radius: 8px;
      padding: 12px;
      max-height: 420px;
      overflow: auto;
      white-space: pre-wrap;
      font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .tree-pre { white-space: pre; min-height: 160px; }
    .file-list {
      display: grid;
      gap: 6px;
      max-height: 280px;
      overflow: auto;
    }
    .file-button {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 8px;
      text-align: left;
      background: #eef4fb;
      color: #27313d;
      font-weight: 700;
    }
    .file-button.active { outline: 2px solid var(--accent); background: #e7f6f2; }
    .tag { color: #667085; font-size: 11px; text-transform: uppercase; font-weight: 800; }
    table { width: 100%; border-collapse: collapse; }
    th, td { text-align: left; padding: 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
    th { color: var(--muted); font-size: 12px; text-transform: uppercase; }
    @media (max-width: 920px) {
      main { grid-template-columns: 1fr; }
      .preview-grid { grid-template-columns: 1fr; }
      header { align-items: flex-start; flex-direction: column; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>TATVA PBQ Generator</h1>
      <div style="color: var(--muted);">Generate, validate, inspect tree, and repair with AI.</div>
    </div>
    <button class="ghost" type="button" onclick="loadWorkspace()">Reload Workspace</button>
  </header>

  <main>
    <aside class="stack">
      <section>
        <h2>Create Question</h2>
        <form id="pbqForm">
          <label>Template
            <select id="templateSelect" name="template"></select>
          </label>
          <div id="templatePreviewWrap" style="margin: -4px 0 8px;">
            <button type="button" class="tpl-toggle" onclick="toggleTemplatePreview()">▶ Preview template files</button>
            <div id="templatePreviewPanel" class="hidden">
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:6px;">
                <div>
                  <div class="tpl-files" id="tplFileList"></div>
                </div>
                <div class="tpl-preview">
                  <pre id="tplFileContent" style="min-height:120px;max-height:200px;font-size:11px;">(select a file)</pre>
                </div>
              </div>
              <p style="color:var(--muted);font-size:11px;margin:4px 0 0;">Locked files are copied as-is. Unlocked files can be modified per-question after generation.</p>
            </div>
          </div>
          <label>Framework / Runtime<select id="runtimeSelect" name="runtime"></select></label>
          <label>Difficulty
            <select name="difficulty">
              <option>easy</option>
              <option selected>medium</option>
              <option>hard</option>
            </select>
          </label>
          <label>Experience
            <select name="experience">
              <option>0-2 years</option>
              <option selected>2-4 years</option>
              <option>4-6 years</option>
              <option>6+ years</option>
            </select>
          </label>
          <label>Expected Testcases
            <select name="testcaseCount">
              <option>10</option>
              <option selected>15</option>
              <option>20</option>
            </select>
          </label>
          <label>Duration Minutes<input name="duration" type="number" value="60" min="15" /></label>
          <label>Question Prompt<textarea name="prompt" placeholder="Example: Build an authentication system with registration, login, logout, and token validation."></textarea></label>
          <div class="row">
            <button type="submit" data-action-button>Generate</button>
            <button type="button" class="ghost" onclick="validatePbq()" data-action-button>Validate</button>
            <button type="button" class="secondary" onclick="regeneratePbq()" data-action-button>Regenerate With AI</button>
          </div>
        </form>
        <div id="status" class="status">Ready.</div>
        <div id="progressWrap" class="progress-wrap hidden">
          <div class="progress-bar-track">
            <div class="progress-bar-fill" id="progressBarFill" style="width:0%"></div>
          </div>
          <div class="steps-list" id="stepsList"></div>
        </div>
      </section>

      <section>
        <h2>Current Result</h2>
        <div class="metric-grid">
          <div class="metric"><strong id="statusMetric">-</strong><span>Status</span></div>
          <div class="metric"><strong id="testsMetric">-</strong><span>Tests passed</span></div>
          <div class="metric"><strong id="failedMetric">-</strong><span>Failures</span></div>
          <div class="metric"><strong id="mutationMetric">-</strong><span>Mutation score</span></div>
        </div>
      </section>
    </aside>

    <div class="stack">
      <section id="questionTreeScreen">
        <h2>Question & Tree Preview</h2>
        <div class="preview-grid">
          <div class="box">
            <h3>Question</h3>
            <div id="questionView">No question generated yet.</div>
          </div>
          <div class="tree-stack">
            <div class="box">
              <h3>Default Template Tree</h3>
              <pre id="defaultTemplateTree" class="tree-pre">(select a template)</pre>
            </div>
            <div class="box">
              <h3>Candidate Workspace Tree</h3>
              <pre id="candidateTree" class="tree-pre">(empty)</pre>
            </div>
            <div class="box">
              <h3>Reference Workspace Tree</h3>
              <pre id="referenceTree" class="tree-pre">(empty)</pre>
            </div>
          </div>
        </div>
      </section>

      <section>
        <h2>Generated Files</h2>
        <div class="two-col">
          <div class="box">
            <div id="fileList" class="file-list">No files loaded.</div>
          </div>
          <div class="box">
            <h3 id="selectedFileTitle">Select a file</h3>
            <pre id="fileContent">(no file selected)</pre>
          </div>
        </div>
      </section>

      <section>
        <h2>Validation Output</h2>
        <div id="validationView" class="box">Run validation to see testcase output.</div>
      </section>
    </div>
  </main>

  <script>
    const state = {
      workspace: null, packet: null, result: null,
      selectedFile: 0, pending: null, templates: [],
      tplFiles: [], selectedTplFile: -1, tplPreviewOpen: false,
      pollTimer: null,
    };
    const statusEl = document.getElementById('status');

    function setStatus(message) { statusEl.textContent = message; }
    function setBusy(isBusy) {
      document.querySelectorAll('[data-action-button]').forEach(btn => { btn.disabled = isBusy; });
    }
    function activeRuntime() { return document.getElementById('runtimeSelect').value || 'django-sqlite-py312'; }

    function formPayload() {
      const raw = Object.fromEntries(new FormData(document.getElementById('pbqForm')).entries());
      const userPrompt = String(raw.prompt || '').trim();
      const template = String(raw.template || '');
      const selectedTemplate = state.templates.find(t => t.slug === template);
      const testcaseCount = String(raw.testcaseCount || '15');
      const prompt = [
        `Template: ${selectedTemplate?.name || template}`,
        `Template description: ${selectedTemplate?.description || ''}`,
        `Stack labels: ${(selectedTemplate?.stack_labels || []).join(', ')}`,
        `User prompt: ${userPrompt}`,
        `Expected testcase coverage: around ${testcaseCount} public/private testcases total.`,
      ].join('\\n');
      return {
        runtime: selectedTemplate?.runtime_profile || raw.runtime,
        difficulty: raw.difficulty,
        experience: raw.experience,
        duration: Number(raw.duration || 60),
        prompt,
        template,
        templateName: selectedTemplate?.name || template,
        userPrompt,
        testcaseCount,
        design_only: true,
      };
    }

    async function api(path, options = {}) {
      const response = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...options });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || payload.detail || response.statusText);
      return payload;
    }

    function escapeHtml(value) {
      return String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }[c]));
    }
    function sectionLabel(section) {
      return ({ starter_files: 'starter', reference_solution: 'reference', public_tests: 'public', private_tests: 'private' })[section] || section || '';
    }
    function makeTree(paths) {
      const root = {};
      for (const raw of paths || []) {
        const isDir = raw.endsWith('/');
        const parts = raw.replace(/\\/$/, '').split('/').filter(Boolean);
        let node = root;
        for (let i = 0; i < parts.length; i++) {
          const part = parts[i] + (i === parts.length - 1 && isDir ? '/' : '');
          node[part] = node[part] || {};
          node = node[part];
        }
      }
      function lines(node, prefix = '') {
        return Object.keys(node).sort((a, b) => a.localeCompare(b)).flatMap((entry, i, arr) => {
          const last = i === arr.length - 1;
          return [`${prefix}${last ? '`-- ' : '|-- '}${entry}`, ...lines(node[entry], prefix + (last ? '    ' : '|   '))];
        });
      }
      return lines(root).join('\\n') || '(empty)';
    }

    // ── Progress bar ──────────────────────────────────────────────────────────

    // Icon text only — no HTML tags, so escaping is safe
    const STEP_ICONS = { pending: '○', running: '', done: '✓', error: '✗', skipped: '─' };

    function showProgressBar(show) {
      document.getElementById('progressWrap').classList.toggle('hidden', !show);
    }

    // Track when each step started so we can compute client-side elapsed for running steps
    const _stepStartTimes = {};

    function renderProgressSteps(task) {
      if (!task) { showProgressBar(false); return; }
      showProgressBar(true);
      const pct = task.total_steps ? Math.round((task.done_steps / task.total_steps) * 100) : 0;
      document.getElementById('progressBarFill').style.width = `${pct}%`;
      const now = Date.now();
      document.getElementById('stepsList').innerHTML = task.steps.map(step => {
        let elapsed = '';
        if (step.status === 'running') {
          // Record start time locally the first time we see this step running
          if (!_stepStartTimes[step.key]) _stepStartTimes[step.key] = now;
          const secs = Math.round((now - _stepStartTimes[step.key]) / 1000);
          elapsed = `${secs}s`;
        } else if (step.elapsed != null) {
          elapsed = `${step.elapsed}s`;
          // Clear local timer once the step is done
          delete _stepStartTimes[step.key];
        }

        // Hint shown for AI steps running >15s
        const isLongAiStep = step.status === 'running' && step.key === 'design' && parseInt(elapsed) > 15;
        const hint = isLongAiStep ? '<div class="step-ai-hint">AI is generating code — typically 1–5 min</div>' : '';

        // running icon is empty — CSS spinner draws the ring; all others are plain text ✓/✗/○/─
        const iconText = step.status === 'running' ? '' : (STEP_ICONS[step.status] || '');
        const statusClass = escapeHtml(step.status);
        return `
          <div class="step-row ${statusClass}">
            <div class="step-icon ${statusClass}">${escapeHtml(iconText)}</div>
            <div class="step-body">
              <div class="step-label">${escapeHtml(step.label)}</div>
              ${step.message ? `<div class="step-msg">${escapeHtml(step.message)}</div>` : ''}
              ${hint}
            </div>
            <div class="step-elapsed">${escapeHtml(elapsed)}</div>
          </div>`;
      }).join('');
    }

    function stopPolling() {
      if (state.pollTimer) { clearTimeout(state.pollTimer); state.pollTimer = null; }
    }

    // How many consecutive 404s before we give up (handles brief server restarts)
    let _poll404Count = 0;
    const POLL_MAX_404 = 3;

    async function pollTask(taskId, runtime, onDone) {
      stopPolling();
      _poll404Count = 0;
      const tick = async () => {
        try {
          const task = await api(`/api/task/${taskId}`);
          _poll404Count = 0;
          renderProgressSteps(task);
          if (task.overall_status === 'done') {
            showProgressBar(false);
            setBusy(false);
            await onDone(task.result, runtime);
          } else if (task.overall_status === 'error') {
            showProgressBar(false);
            setBusy(false);
            setStatus(`Generation error — check server logs. ${task.error || ''}`);
            state.pending = { status: 'Error', error: task.error };
            renderAll();
          } else {
            const running = task.steps.find(s => s.status === 'running');
            if (running) setStatus(`Step ${task.done_steps + 1}/${task.total_steps}: ${running.label}...`);
            state.pollTimer = setTimeout(tick, 1200);
          }
        } catch (err) {
          if (err.message && err.message.includes('not found')) {
            _poll404Count++;
            if (_poll404Count >= POLL_MAX_404) {
              showProgressBar(false);
              setBusy(false);
              setStatus('Server restarted while task was running — please try again.');
              state.pending = { status: 'Server restarted', error: 'The server was restarted during generation. Please click Generate again.' };
              renderAll();
              return;
            }
            // Retry a few times in case of transient issue
            state.pollTimer = setTimeout(tick, 2000);
          } else {
            showProgressBar(false);
            setBusy(false);
            setStatus(`Polling error: ${err.message}`);
          }
        }
      };
      state.pollTimer = setTimeout(tick, 800);
    }

    async function onTaskDone(result, runtime) {
      state.pending = null;
      state.result = result;
      if (result?.final_status === 'design_only') {
        state.workspace = { design: result.question_design, files: result.files || [] };
        renderAll();
        renderDesignOnlyResult(result);
        setStatus('Design complete — question tree ready');
        setBusy(false);
        return;
      }
      await loadWorkspace(runtime, false);
      await loadReviewPacket(runtime, false);
      renderAll();
      focusQuestionTreeScreen();
      setStatus(`Finished: ${result?.final_status || 'done'}`);
    }

    function renderDesignOnlyResult(result) {
      const q = result.question || {};
      const plan = result.feature_plan || {};
      const entities = (plan.entities || []).map(e => typeof e === 'string' ? e : (e.name || JSON.stringify(e)));
      const features = (plan.features || []).map(f => typeof f === 'string' ? f : (f.name || f.description || JSON.stringify(f)));
      const contract = (q.behavioral_contract || []).map(c => typeof c === 'string' ? c : JSON.stringify(c));
      const refFiles = (result.reference_files || []).join('\\n');
      const starterFiles = (result.starter_files || []).join('\\n');
      const publicTests = (result.public_tests || []).join('\\n');
      const privateTests = (result.private_tests || []).join('\\n');
      const manifest = (result.file_manifest || []).map(f =>
        `  [${f.role || '?'}] ${f.path || '?'} — ${f.description || ''}`
      ).join('\\n');

      document.getElementById('questionTreeScreen').innerHTML = `
        <h2>Question Design (Design Only)</h2>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px">
          <div>
            <h3>${escapeHtml(q.title || 'Untitled')}</h3>
            <p><strong>Runtime:</strong> ${escapeHtml(q.runtime_profile || '')}</p>
            <p><strong>Difficulty:</strong> ${escapeHtml(q.difficulty || '')} &middot; <strong>Experience:</strong> ${escapeHtml(q.experience || '')}</p>

            <h4>Behavioral Contract</h4>
            <ol>${contract.map(c => '<li>' + escapeHtml(c) + '</li>').join('')}</ol>

            <h4>Feature Plan</h4>
            <p><strong>Domain:</strong> ${escapeHtml(plan.domain_summary || '')}</p>
            <p><strong>Entities:</strong> ${escapeHtml(entities.join(', '))}</p>
            <ul>${features.map(f => '<li>' + escapeHtml(f) + '</li>').join('')}</ul>
          </div>
          <div>
            <h4>File Manifest</h4>
            <pre style="font-size:12px;background:var(--bg-alt);padding:12px;border-radius:8px;overflow-x:auto">${escapeHtml(manifest || '(no manifest)')}</pre>

            <h4>Reference Files</h4>
            <pre style="font-size:12px;background:var(--bg-alt);padding:12px;border-radius:8px">${escapeHtml(refFiles || '(none)')}</pre>

            <h4>Starter Files</h4>
            <pre style="font-size:12px;background:var(--bg-alt);padding:12px;border-radius:8px">${escapeHtml(starterFiles || '(none)')}</pre>

            <h4>Tests</h4>
            <pre style="font-size:12px;background:var(--bg-alt);padding:12px;border-radius:8px">Public:\n${escapeHtml(publicTests || '(none)')}\nPrivate:\n${escapeHtml(privateTests || '(none)')}</pre>
          </div>
        </div>
      `;
      focusQuestionTreeScreen();
    }

    // ── Template preview ──────────────────────────────────────────────────────

    function toggleTemplatePreview() {
      state.tplPreviewOpen = !state.tplPreviewOpen;
      document.getElementById('templatePreviewPanel').classList.toggle('hidden', !state.tplPreviewOpen);
      document.querySelector('.tpl-toggle').textContent =
        (state.tplPreviewOpen ? '▼' : '▶') + ' Preview template files';
      if (state.tplPreviewOpen && !state.tplFiles.length) loadTemplateFiles();
    }

    async function loadTemplateFiles() {
      const slug = document.getElementById('templateSelect').value;
      if (!slug) return;
      try {
        const data = await api(`/api/template-files/${encodeURIComponent(slug)}`);
        state.tplFiles = data.files || [];
        state.selectedTplFile = state.tplFiles.length ? 0 : -1;
        renderTemplatePreview();
      } catch (err) {
        document.getElementById('tplFileList').innerHTML = `<p style="color:var(--danger);font-size:12px;">${escapeHtml(err.message)}</p>`;
      }
    }

    function renderTemplatePreview() {
      const files = state.tplFiles;
      const lockedSet = new Set((state.templates.find(t => t.slug === document.getElementById('templateSelect').value)?.locked_files) || []);
      document.getElementById('tplFileList').innerHTML = files.map((f, i) => `
        <button type="button" class="tpl-file-btn ${i === state.selectedTplFile ? 'active' : ''}" onclick="selectTplFile(${i})">
          <span>${escapeHtml(f.path)}</span>
          ${lockedSet.has(f.path) ? '<span class="tpl-locked-badge">locked</span>' : ''}
        </button>`).join('');
      const sel = files[state.selectedTplFile];
      document.getElementById('tplFileContent').textContent = sel?.content || '(no file selected)';
    }

    function selectTplFile(index) {
      state.selectedTplFile = index;
      renderTemplatePreview();
    }

    // ── Profiles & Templates loading ──────────────────────────────────────────

    async function loadProfiles() {
      const payload = await api('/api/profiles');
      document.getElementById('runtimeSelect').innerHTML = payload.profiles
        .map(p => `<option value="${p.id}">${p.id}</option>`).join('');
      document.getElementById('runtimeSelect').addEventListener('change', () => {
        clearPreview(`Selected ${activeRuntime()}. Click Generate to create a question or reload workspace.`);
      });
    }

    async function loadTemplates() {
      const payload = await api('/api/templates');
      state.templates = payload.templates || [];
      const sel = document.getElementById('templateSelect');
      sel.innerHTML = state.templates.map(t => `
        <option value="${escapeHtml(t.slug)}">${escapeHtml(t.name)} (${escapeHtml(t.runtime_profile)})</option>
      `).join('');
      sel.addEventListener('change', () => {
        state.tplFiles = [];
        state.selectedTplFile = -1;
        if (state.tplPreviewOpen) loadTemplateFiles();
        syncRuntimeFromTemplate();
      });
      syncRuntimeFromTemplate();
    }

    function syncRuntimeFromTemplate() {
      const selected = state.templates.find(t => t.slug === document.getElementById('templateSelect').value);
      if (!selected) return;
      document.getElementById('runtimeSelect').value = selected.runtime_profile;
      clearPreview(`Selected ${selected.name} (${selected.runtime_profile}). Enter prompt, then click Generate.`);
    }

    // ── Main operations ───────────────────────────────────────────────────────

    async function generatePbq(payload) {
      setBusy(true);
      showPendingGeneration(payload);
      try {
        const resp = await api('/api/pbq', {
          method: 'POST',
          body: JSON.stringify({
            template: payload.template,
            runtime: payload.runtime,
            difficulty: payload.difficulty,
            experience: payload.experience,
            duration: payload.duration,
            prompt: payload.prompt,
            design_only: payload.design_only || false,
          }),
        });
        setStatus('Generating PBQ — watching progress...');
        await pollTask(resp.task_id, resp.runtime, onTaskDone);
      } catch (err) {
        showGenerationError(payload, err);
        setBusy(false);
      }
    }

    async function designOnlyPbq() {
      const payload = formPayload();
      payload.design_only = true;
      setBusy(true);
      showPendingGeneration(payload);
      try {
        const resp = await api('/api/pbq', {
          method: 'POST',
          body: JSON.stringify({
            template: payload.template,
            runtime: payload.runtime,
            difficulty: payload.difficulty,
            experience: payload.experience,
            duration: payload.duration,
            prompt: payload.prompt,
            design_only: true,
          }),
        });
        setStatus('Design Only — generating question tree...');
        await pollTask(resp.task_id, resp.runtime, onTaskDone);
      } catch (err) {
        showGenerationError(payload, err);
        setBusy(false);
      }
    }

    async function validatePbq() {
      const runtime = activeRuntime();
      setBusy(true);
      setStatus(`Starting validation for ${runtime}...`);
      state.pending = { status: 'Validating...', runtime };
      renderAll();
      try {
        const resp = await api('/api/validate-pbq', { method: 'POST', body: JSON.stringify({ runtime }) });
        setStatus('Validating — watching progress...');
        await pollTask(resp.task_id, resp.runtime, onTaskDone);
      } catch (err) {
        setStatus(err.message);
        state.pending = null;
        setBusy(false);
      }
    }

    async function regeneratePbq() {
      const runtime = activeRuntime();
      setBusy(true);
      setStatus(`Starting AI regeneration for ${runtime}...`);
      state.pending = { status: 'Regenerating with AI...', runtime };
      renderAll();
      try {
        const resp = await api('/api/regenerate-pbq-from-validation', {
          method: 'POST',
          body: JSON.stringify({
            runtime,
            reason: 'Fix the PBQ using the latest validation failure output. Keep prompt domain, reference solution, and tests aligned. Add tests covering success, validation, and edge cases.',
          }),
        });
        setStatus('Regenerating — watching progress...');
        await pollTask(resp.task_id, resp.runtime, onTaskDone);
      } catch (err) {
        setStatus(err.message);
        state.pending = null;
        setBusy(false);
      }
    }

    async function loadWorkspace(runtime = activeRuntime(), showStatus = true) {
      try {
        state.workspace = await api(`/api/pbq-workspace?runtime=${encodeURIComponent(runtime)}`);
        state.selectedFile = 0;
        if (state.workspace.result) state.result = state.workspace.result;
        renderAll();
        if (showStatus) setStatus(`Workspace loaded for ${runtime}.`);
      } catch (err) {
        if (showStatus) setStatus(err.message);
      }
    }

    async function loadReviewPacket(runtime = activeRuntime(), showStatus = true) {
      try {
        state.packet = await api(`/api/review-packet?runtime=${encodeURIComponent(runtime)}`);
        if (state.packet.result) state.result = state.packet.result;
        renderAll();
        if (showStatus) setStatus(`Review packet loaded for ${runtime}.`);
      } catch (err) {
        state.packet = null;
        if (showStatus) setStatus(err.message);
      }
    }

    function renderAll() {
      renderMetrics();
      renderQuestion();
      renderTrees();
      renderFiles();
      renderValidation();
    }

    function focusQuestionTreeScreen() {
      document.getElementById('questionTreeScreen')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function clearPreview(message) {
      state.workspace = null;
      state.packet = null;
      state.result = null;
      state.pending = { message };
      state.selectedFile = 0;
      renderAll();
      setStatus(message);
    }

    function resetQuestionTreeScreen() {
      document.getElementById('questionTreeScreen').innerHTML = `
        <h2>Question & Tree Preview</h2>
        <div class="preview-grid">
          <div class="box">
            <h3>Question</h3>
            <div id="questionView">No question generated yet.</div>
          </div>
          <div class="tree-stack">
            <div class="box">
              <h3>Default Template Tree</h3>
              <pre id="defaultTemplateTree" class="tree-pre">(select a template)</pre>
            </div>
            <div class="box">
              <h3>Candidate Workspace Tree</h3>
              <pre id="candidateTree" class="tree-pre">(empty)</pre>
            </div>
            <div class="box">
              <h3>Reference Workspace Tree</h3>
              <pre id="referenceTree" class="tree-pre">(empty)</pre>
            </div>
          </div>
        </div>
      `;
    }

    function showPendingGeneration(payload) {
      resetQuestionTreeScreen();
      state.workspace = null;
      state.packet = null;
      state.result = null;
      state.selectedFile = 0;
      state.pending = {
        template: payload.templateName || payload.template,
        runtime: payload.runtime,
        difficulty: payload.difficulty,
        experience: payload.experience,
        duration: payload.duration,
        testcaseCount: payload.testcaseCount,
        userPrompt: payload.userPrompt,
        prompt: payload.prompt,
        status: 'Generating PBQ with AI. Waiting for API response...',
      };
      renderAll();
      focusQuestionTreeScreen();
      setStatus('Generating PBQ with AI...');
    }

    function showGenerationError(payload, err) {
      state.workspace = null;
      state.packet = null;
      state.result = null;
      state.pending = {
        ...payload,
        status: 'Generation failed.',
        error: err.message,
      };
      renderAll();
      focusQuestionTreeScreen();
      setStatus(err.message);
    }

    function renderMetrics() {
      if (state.pending) {
        document.getElementById('statusMetric').innerHTML = '<span class="pill">running</span>';
        document.getElementById('testsMetric').textContent = '-';
        document.getElementById('failedMetric').textContent = '-';
        document.getElementById('mutationMetric').textContent = '-';
        return;
      }
      const result = state.result || {};
      const summary = result.testcase_summary || {};
      const passed = [...(summary.commands || []), ...(summary.mutations || [])]
        .reduce((total, item) => total + (item.passed_count || 0), 0);
      document.getElementById('statusMetric').innerHTML = result.final_status
        ? `<span class="pill ${escapeHtml(result.final_status)}">${escapeHtml(result.final_status)}</span>`
        : '-';
      document.getElementById('testsMetric').textContent = passed || '-';
      document.getElementById('failedMetric').textContent = summary.failed_count ?? '-';
      document.getElementById('mutationMetric').textContent = result.mutation_score ?? '-';
    }

    function renderQuestion() {
      if (state.pending) {
        const pending = state.pending;
        document.getElementById('questionView').innerHTML = `
          <h3>${escapeHtml(pending.status || pending.message || 'Preparing question')}</h3>
          ${pending.error ? `<p><strong>Error</strong><br>${escapeHtml(pending.error)}</p>` : ''}
          ${pending.runtime ? `<p><strong>Framework</strong><br>${escapeHtml(pending.runtime)}</p>` : ''}
          ${pending.template ? `<p><strong>Template</strong><br>${escapeHtml(pending.template)}</p>` : ''}
          ${pending.difficulty ? `<p><strong>Difficulty / Experience</strong><br>${escapeHtml(pending.difficulty)} · ${escapeHtml(pending.experience)}</p>` : ''}
          ${pending.testcaseCount ? `<p><strong>Expected Testcases</strong><br>${escapeHtml(pending.testcaseCount)}</p>` : ''}
          ${pending.userPrompt ? `<p><strong>Your Prompt</strong><br>${escapeHtml(pending.userPrompt)}</p>` : ''}
          ${pending.prompt ? `<p><strong>Prompt Sent To AI</strong></p><pre>${escapeHtml(pending.prompt)}</pre>` : ''}
        `;
        return;
      }
      const design = state.workspace?.design || state.result?.question_design;
      if (!design) {
        document.getElementById('questionView').innerHTML = 'No question generated yet.';
        return;
      }
      document.getElementById('questionView').innerHTML = `
        <h3>${escapeHtml(design.title || 'Untitled PBQ')}</h3>
        <p><strong>Behavioral Contract</strong></p>
        <ul>${(design.behavioral_contract || []).map(item => `<li>${escapeHtml(item)}</li>`).join('')}</ul>
        <p><strong>Scaffolding</strong><br>${escapeHtml(design.scaffolding_strategy || '')}</p>
      `;
    }

    function renderTrees() {
      const selectedTemplate = state.templates.find(item => item.slug === document.getElementById('templateSelect')?.value);
      document.getElementById('defaultTemplateTree').textContent = makeTree(selectedTemplate?.default_tree || []);
      if (state.pending) {
        document.getElementById('candidateTree').textContent = state.pending.error ? '(not generated)' : '(generating candidate workspace...)';
        document.getElementById('referenceTree').textContent = state.pending.error ? '(not generated)' : '(generating reference solution workspace...)';
        return;
      }
      const result = state.result || {};
      document.getElementById('candidateTree').textContent = makeTree(result.candidate_workspace_tree || []);
      document.getElementById('referenceTree').textContent = makeTree(result.reference_workspace_tree || []);
    }

    function renderFiles() {
      if (state.pending) {
        document.getElementById('fileList').textContent = state.pending.error ? 'No files generated because the API request failed.' : 'Waiting for AI generated files...';
        document.getElementById('selectedFileTitle').textContent = state.pending.error ? 'Generation error' : 'Generating files';
        document.getElementById('fileContent').textContent = state.pending.error || '(files will appear after the API response returns)';
        return;
      }
      const files = state.workspace?.files || [];
      const list = document.getElementById('fileList');
      if (!files.length) {
        list.textContent = 'No files loaded.';
        document.getElementById('selectedFileTitle').textContent = 'Select a file';
        document.getElementById('fileContent').textContent = '(no file selected)';
        return;
      }
      list.innerHTML = files.map((file, index) => `
        <button type="button" class="file-button ${index === state.selectedFile ? 'active' : ''}" onclick="selectFile(${index})">
          <span>${escapeHtml(file.path)}</span>
          <span class="tag">${escapeHtml(sectionLabel(file.section))}</span>
        </button>
      `).join('');
      const selected = files[state.selectedFile] || files[0];
      document.getElementById('selectedFileTitle').textContent = `${selected.path} (${sectionLabel(selected.section)})`;
      document.getElementById('fileContent').textContent = selected.content || '';
    }

    function selectFile(index) {
      state.selectedFile = index;
      renderFiles();
    }

    function renderValidation() {
      if (state.pending) {
        document.getElementById('validationView').innerHTML = state.pending.error
          ? `<p><strong>Generation failed</strong></p><pre>${escapeHtml(state.pending.error)}</pre>`
          : 'Validation will appear after generation finishes and you click Validate.';
        return;
      }
      const result = state.result;
      if (!result) {
        document.getElementById('validationView').innerHTML = 'Run validation to see testcase output.';
        return;
      }
      const summary = result.testcase_summary || {};
      const rows = [...(summary.commands || []), ...(summary.mutations || [])].map(item => `
        <tr>
          <td>${escapeHtml(item.mutation ? `${item.command || ''} (${item.mutation})` : item.command || 'command')}</td>
          <td><span class="pill ${escapeHtml(item.status)}">${escapeHtml(item.status)}</span></td>
          <td>${item.passed_count || 0}</td>
          <td><pre>${escapeHtml((item.failed_tests || []).join(', ') || item.reason || 'ok')}</pre></td>
        </tr>
      `).join('');
      const repairNotes = (result.validation_repairs || []).length
        ? `<h3>AI Repair Notes</h3><pre>${escapeHtml(result.validation_repairs.join('\\n'))}</pre>`
        : '';
      document.getElementById('validationView').innerHTML = `
        <p>Review packet: <code>${escapeHtml(result.review_packet_path || 'not written')}</code></p>
        <table>
          <thead><tr><th>Command/Test</th><th>Status</th><th>Passed</th><th>Reason</th></tr></thead>
          <tbody>${rows || '<tr><td colspan="4">No testcase output recorded.</td></tr>'}</tbody>
        </table>
        ${repairNotes}
      `;
    }

    document.getElementById('pbqForm').addEventListener('submit', event => {
      event.preventDefault();
      generatePbq(formPayload()).catch(err => setStatus(err.message));
    });

    loadProfiles()
      .then(loadTemplates)
      .then(() => clearPreview('Select template/framework, enter prompt, then click Generate.'))
      .then(renderAll)
      .catch(err => setStatus(err.message));
  </script>
</body>
</html>
"""


class POCRequestHandler(BaseHTTPRequestHandler):
    server_version = "TatvaPOCUI/1.0"

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")

    def _send_json(self, payload: object, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        if not raw.strip():
            return {}
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object")
        return payload

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(INDEX_HTML)
                return
            if parsed.path == "/api/profiles":
                self._send_json({"profiles": [profile.to_dict() for profile in all_runtime_profiles()]})
                return
            if parsed.path == "/api/results":
                self._send_json(_results_payload())
                return
            if parsed.path == "/api/artifact":
                query = parse_qs(parsed.query)
                path = query.get("path", [""])[0]
                self._send_json({"path": path, "content": _load_artifact(path)})
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def do_POST(self) -> None:
        try:
            parsed = urlparse(self.path)
            payload = self._read_json_body()
            if parsed.path == "/api/pbq":
                state = run_pbq_graph(
                    PBQRequest(
                        runtime_profile=str(payload.get("runtime") or "django-sqlite-py312"),
                        difficulty=str(payload.get("difficulty") or "medium"),
                        experience=str(payload.get("experience") or "4-6 years"),
                        duration=int(payload.get("duration") or 90),
                        prompt=str(payload.get("prompt") or PBQ_BENCHMARK_PROMPTS["django-sqlite-py312"]),
                    )
                )
                self._send_json({"result": state.final, "output_dir": str(state.output_dir.relative_to(ROOT_DIR))})
                return
            if parsed.path == "/api/mcq":
                state = run_mcq_graph(
                    MCQRequest(
                        skill=str(payload.get("skill") or "Python"),
                        topic=str(payload.get("topic") or "concurrency"),
                        difficulty=str(payload.get("difficulty") or "medium"),
                        experience=str(payload.get("experience") or "4-6 years"),
                    )
                )
                self._send_json({"result": state.final, "output_dir": str(state.output_dir.relative_to(ROOT_DIR))})
                return
            if parsed.path == "/api/benchmark-pbq":
                self._send_json({"summary": _run_pbq_benchmark()})
                return
            if parsed.path == "/api/benchmark-mcq":
                self._send_json({"summary": _run_mcq_benchmark()})
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)


def _json_or_none(path: Path) -> object | None:
    return read_json(path) if path.is_file() else None


def _artifact_paths() -> list[str]:
    if not OUTPUT_DIR.exists():
        return []
    allowed_names = {"pbq_result.json", "pbq_bundle.json", "mcq_result.json", "pbq_summary.json", "mcq_summary.json"}
    return [
        path.relative_to(ROOT_DIR).as_posix()
        for path in sorted(OUTPUT_DIR.rglob("*.json"))
        if path.is_file() and path.name in allowed_names
    ]


def _load_artifact(relative_path: str) -> object:
    target = (ROOT_DIR / relative_path).resolve()
    output_root = OUTPUT_DIR.resolve()
    if not target.is_relative_to(output_root):
        raise ValueError("Artifact path must be inside outputs/")
    if not target.is_file():
        raise FileNotFoundError(relative_path)
    return read_json(target)


def _results_payload() -> dict:
    return {
        "pbq_summary": _json_or_none(OUTPUT_DIR / "benchmarks" / "pbq_summary.json"),
        "mcq_summary": _json_or_none(OUTPUT_DIR / "benchmarks" / "mcq_summary.json"),
        "artifacts": _artifact_paths(),
    }


def _run_pbq_benchmark() -> dict:
    results = []
    for profile in all_runtime_profiles():
        state = run_pbq_graph(
            PBQRequest(
                runtime_profile=profile.id,
                difficulty="hard",
                experience="4-6 years",
                duration=90,
                prompt=PBQ_BENCHMARK_PROMPTS[profile.id],
            ),
            output_dir=OUTPUT_DIR / "benchmarks" / "pbq" / profile.id,
        )
        results.append(state.final)
    summary = {
        "profiles": [item["runtime_profile"]["id"] for item in results],
        "passed": sum(1 for item in results if item["final_status"] == "passed"),
        "results": results,
    }
    path = OUTPUT_DIR / "benchmarks" / "pbq_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _run_mcq_benchmark() -> dict:
    results = []
    for difficulty in ("easy", "medium", "hard"):
        state = run_mcq_graph(
            MCQRequest(
                skill="Python",
                topic="concurrency",
                difficulty=difficulty,
                experience={"easy": "0-2 years", "medium": "2-4 years", "hard": "4-6 years"}[difficulty],
            ),
            output_dir=OUTPUT_DIR / "benchmarks" / "mcq" / difficulty,
        )
        results.append(state.final)
    summary = {
        "difficulties": [item["requested_difficulty"] for item in results],
        "passed": sum(1 for item in results if item["final_status"] == "passed"),
        "results": results,
    }
    path = OUTPUT_DIR / "benchmarks" / "mcq_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local TATVA AI Generation POC UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run("api:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
