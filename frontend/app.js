// ─── State Management ────────────────────────────────────────────────────────
const state = {
  activeDocumentId: null,
  activeDocumentName: '',
  activeDocumentPages: 0,
  documents: [],
  activeTab: 'chat', // 'chat' | 'inspector' | 'pipeline'
  topK: 5,
  chatHistory: []
};

// ─── DOM Elements ─────────────────────────────────────────────────────────────
const uploadState       = document.getElementById('uploadState');
const indexingState     = document.getElementById('indexingState');
const chatState         = document.getElementById('chatState');
const inspectorState    = document.getElementById('inspectorState');
const statusBanner      = document.getElementById('statusBanner');
const docSelect         = document.getElementById('docSelect');
const sidebarDocList    = document.getElementById('sidebarDocList');
const docCountBadge     = document.getElementById('docCountBadge');
const deleteDocBtn      = document.getElementById('deleteDocBtn');
const pdfFileInput      = document.getElementById('pdfFileInput');
const dropZone          = document.getElementById('dropZone');
const progressBar       = document.getElementById('progressBar');
const indexingTitle     = document.getElementById('indexingTitle');
const askForm           = document.getElementById('askForm');
const questionInput     = document.getElementById('questionInput');
const askSubmitBtn      = document.getElementById('askSubmitBtn');
const questionLoader    = document.getElementById('questionLoader');
const answerSection     = document.getElementById('answerSection');
const answerText        = document.getElementById('answerText');
const citationsContainer= document.getElementById('citationsContainer');
const sourceCount       = document.getElementById('sourceCount');
const retrievalTimePill = document.getElementById('retrievalTimePill');
const generationTimePill= document.getElementById('generationTimePill');
const activeDocName     = document.getElementById('activeDocName');
const activeDocPages    = document.getElementById('activeDocPages');
const activeDocBadge    = document.getElementById('activeDocBadge');
const noDocBadge        = document.getElementById('noDocBadge');
const sidebarUploadBtn  = document.getElementById('sidebarUploadBtn');
const uploadAnotherBtn  = document.getElementById('uploadAnotherBtn');
const navTabs           = document.getElementById('navTabs');
const topKSlider        = document.getElementById('topKSlider');
const topKVal           = document.getElementById('topKVal');
const chunkInspectorList= document.getElementById('chunkInspectorList');
const chatThread        = document.getElementById('chatThread');

// ─── Notification & Toast Banner ──────────────────────────────────────────────
function showBanner(msg, type = 'error') {
  statusBanner.textContent = msg;
  statusBanner.className = `status-banner ${type}`;
  statusBanner.classList.remove('hidden');
  if (type === 'success') {
    setTimeout(() => statusBanner.classList.add('hidden'), 4000);
  }
}

function hideBanner() {
  statusBanner.classList.add('hidden');
}

// ─── Navigation & Workspace Tab Switching ─────────────────────────────────────
function switchTab(tabName) {
  state.activeTab = tabName;
  
  // Highlight Tab Button
  document.querySelectorAll('.nav-tab').forEach(tab => {
    if (tab.dataset.tab === tabName) {
      tab.classList.add('active');
    } else {
      tab.classList.remove('active');
    }
  });

  // Toggle Visibility of Workspace Panels
  if (state.activeDocumentId || tabName === 'pipeline') {
    if (tabName === 'chat') {
      showState('chat');
    } else if (tabName === 'inspector') {
      showState('inspector');
      renderChunkInspector();
    } else if (tabName === 'pipeline') {
      showState('pipeline');
    }
  } else {
    showState('upload');
  }
}

navTabs?.addEventListener('click', e => {
  const tabBtn = e.target.closest('.nav-tab');
  if (tabBtn && tabBtn.dataset.tab) {
    switchTab(tabBtn.dataset.tab);
  }
});

topKSlider?.addEventListener('input', e => {
  state.topK = parseInt(e.target.value, 10);
  topKVal.textContent = state.topK;
});

function showState(name) {
  uploadState.classList.add('hidden');
  indexingState.classList.add('hidden');
  chatState.classList.add('hidden');
  if (inspectorState) inspectorState.classList.add('hidden');

  if (name === 'upload')    uploadState.classList.remove('hidden');
  if (name === 'indexing' || name === 'pipeline') indexingState.classList.remove('hidden');
  if (name === 'chat')      chatState.classList.remove('hidden');
  if (name === 'inspector' && inspectorState) inspectorState.classList.remove('hidden');
}

function setStepState(stepId, status) {
  const el = document.getElementById(stepId);
  if (!el) return;
  
  el.className = 'pipeline-node ' + status;
  const badge = el.querySelector('.node-status-badge');
  if (badge) {
    if (status === 'active') badge.textContent = 'Processing...';
    else if (status === 'completed') badge.textContent = 'Completed';
    else badge.textContent = 'Waiting';
  }
}

function setProgress(pct) {
  if (progressBar) progressBar.style.width = pct + '%';
}

function setQuestion(q) {
  questionInput.value = q;
  questionInput.focus();
}

// ─── Sidebar Document Library Manager ─────────────────────────────────────────
async function loadDocuments() {
  try {
    const res = await fetch('/api/documents/');
    const data = await res.json();
    state.documents = data.documents || [];
    renderSidebarDocuments();
  } catch (e) {
    console.error('Failed to load documents', e);
  }
}

function renderSidebarDocuments() {
  if (docCountBadge) docCountBadge.textContent = state.documents.length;
  
  // Rebuild hidden select options for backward compatibility
  docSelect.innerHTML = '<option value="">-- Select Active Document --</option>';
  sidebarDocList.innerHTML = '';

  if (state.documents.length === 0) {
    sidebarDocList.innerHTML = '<div class="empty-docs-notice">No documents indexed yet. Upload a PDF to begin.</div>';
    return;
  }

  state.documents.forEach(doc => {
    // Dropdown option
    const opt = document.createElement('option');
    opt.value = doc.id;
    opt.textContent = `${doc.filename} (${doc.page_count} pages)`;
    docSelect.appendChild(opt);

    // Sidebar Item Card
    const item = document.createElement('div');
    item.className = `sidebar-doc-item ${state.activeDocumentId === doc.id ? 'active' : ''}`;
    item.innerHTML = `
      <div class="doc-item-meta">
        <svg class="doc-item-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
          <polyline points="14 2 14 8 20 8"></polyline>
        </svg>
        <div class="doc-item-details">
          <span class="doc-item-title" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</span>
          <span class="doc-item-sub">${doc.page_count} pages &bull; Ready</span>
        </div>
      </div>
      <span class="doc-item-delete" title="Delete ${escapeAttr(doc.filename)}" role="button">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="3 6 5 6 21 6"></polyline>
          <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
        </svg>
      </span>
    `;

    item.addEventListener('click', () => selectDocument(doc));

    const deleteBtn = item.querySelector('.doc-item-delete');
    deleteBtn.addEventListener('click', async (e) => {
      e.stopPropagation();
      if (confirm(`Delete document "${doc.filename}" and all associated vector chunks?`)) {
        await deleteDocumentById(doc.id, doc.filename);
      }
    });

    sidebarDocList.appendChild(item);
  });

  if (state.activeDocumentId) {
    docSelect.value = state.activeDocumentId;
    deleteDocBtn.style.display = 'flex';
  }
}

async function deleteDocumentById(docId, docName) {
  try {
    const res = await fetch(`/api/documents/${docId}/`, { method: 'DELETE' });
    const data = await res.json();
    showBanner(data.message || `Document "${docName}" deleted successfully.`, 'success');
    
    if (state.activeDocumentId === docId) {
      state.activeDocumentId = null;
      state.activeDocumentName = '';
      state.activeDocumentPages = 0;
      
      if (activeDocBadge) activeDocBadge.style.display = 'none';
      if (noDocBadge) noDocBadge.style.display = 'block';
      deleteDocBtn.style.display = 'none';
      showState('upload');
    }

    await loadDocuments();
  } catch (e) {
    showBanner('Failed to delete document.');
  }
}

function selectDocument(doc) {
  state.activeDocumentId = doc.id;
  state.activeDocumentName = doc.filename;
  state.activeDocumentPages = doc.page_count;
  
  activeDocName.textContent = doc.filename;
  activeDocPages.textContent = `${doc.page_count} pages`;
  
  if (activeDocBadge) activeDocBadge.style.display = 'flex';
  if (noDocBadge) noDocBadge.style.display = 'none';
  deleteDocBtn.style.display = 'flex';
  
  renderSidebarDocuments();
  switchTab('chat');
}

docSelect.addEventListener('change', () => {
  const docId = parseInt(docSelect.value, 10);
  if (!docId) {
    state.activeDocumentId = null;
    deleteDocBtn.style.display = 'none';
    if (activeDocBadge) activeDocBadge.style.display = 'none';
    if (noDocBadge) noDocBadge.style.display = 'block';
    showState('upload');
    return;
  }
  const doc = state.documents.find(d => d.id === docId);
  if (doc) selectDocument(doc);
});

deleteDocBtn.addEventListener('click', async () => {
  if (!state.activeDocumentId) return;
  if (!confirm(`Delete document "${state.activeDocumentName}" and all associated vector index chunks?`)) return;
  await deleteDocumentById(state.activeDocumentId, state.activeDocumentName);
});

// ─── File Upload & Indexing Pipeline ──────────────────────────────────────────
sidebarUploadBtn?.addEventListener('click', () => {
  switchTab('upload');
  showState('upload');
  hideBanner();
});

uploadAnotherBtn?.addEventListener('click', () => {
  switchTab('upload');
  showState('upload');
  hideBanner();
});

pdfFileInput.addEventListener('change', e => {
  if (e.target.files[0]) handleFileSelected(e.target.files[0]);
});

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));

dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && file.name.toLowerCase().endsWith('.pdf')) {
    handleFileSelected(file);
  } else {
    showBanner('Please drop a valid PDF file.');
  }
});

async function handleFileSelected(file) {
  hideBanner();
  showState('indexing');
  indexingTitle.textContent = `Processing Document: ${file.name}`;
  setProgress(0);

  const steps = ['stepUpload', 'stepExtract', 'stepChunk', 'stepEmbed', 'stepSave'];
  steps.forEach(s => setStepState(s, ''));

  // Step 1: Upload
  setStepState('stepUpload', 'active');
  setProgress(15);
  let docId;
  
  try {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch('/api/documents/upload/', { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) {
      showBanner(data.error || 'Upload failed.');
      showState('upload');
      return;
    }
    docId = data.document_id;
    setStepState('stepUpload', 'completed');
    setProgress(30);
  } catch (e) {
    showBanner('Network error during upload.');
    showState('upload');
    return;
  }

  // Steps 2-5: Server-side Indexing Simulation
  setStepState('stepExtract', 'active'); setProgress(45);
  await sleep(400);
  setStepState('stepExtract', 'completed');

  setStepState('stepChunk', 'active'); setProgress(60);
  await sleep(300);
  setStepState('stepChunk', 'completed');

  setStepState('stepEmbed', 'active'); setProgress(75);

  try {
    const res = await fetch(`/api/documents/${docId}/index/`, { method: 'POST' });
    const data = await res.json();
    if (!res.ok) {
      showBanner(data.error || 'Indexing failed.');
      showState('upload');
      return;
    }
    setStepState('stepEmbed', 'completed');
    setStepState('stepSave', 'active'); setProgress(90);
    await sleep(300);
    setStepState('stepSave', 'completed'); setProgress(100);
    await sleep(400);

    await loadDocuments();
    const doc = state.documents.find(d => d.id === docId) || { id: docId, filename: file.name, page_count: data.page_count || 1 };
    selectDocument(doc);
    showBanner(`Indexed ${data.total_chunks} chunks across ${data.page_count} pages in ${data.indexing_time_sec}s`, 'success');
  } catch (e) {
    showBanner('Network error during vector indexing.');
    showState('upload');
  }
}

// ─── AI Ask & Question Answering Flow ─────────────────────────────────────────
askForm.addEventListener('submit', async e => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;
  if (!state.activeDocumentId) {
    showBanner('Please select or index a PDF document first.');
    return;
  }

  hideBanner();
  answerSection.classList.add('hidden');
  questionLoader.classList.remove('hidden');
  askSubmitBtn.disabled = true;

  try {
    const res = await fetch('/api/ask/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        document_id: state.activeDocumentId,
        top_k: state.topK
      })
    });

    const data = await res.json();
    if (!res.ok) {
      showBanner(data.error || 'Question query failed.');
      return;
    }

    renderAnswer(data);
  } catch (e) {
    showBanner('Network error while requesting answer.');
  } finally {
    questionLoader.classList.add('hidden');
    askSubmitBtn.disabled = false;
  }
});

function renderAnswer(data) {
  answerText.innerHTML = formatMarkdown(data.answer) || '(No answer text returned)';
  
  if (data.cached) {
    retrievalTimePill.innerHTML  = `<span style="color: #34d399; font-weight: 600;">⚡ Memory Cache (0ms)</span>`;
    generationTimePill.innerHTML = `<span style="color: #34d399; font-weight: 600;">⚡ Instant Retrieval</span>`;
  } else {
    retrievalTimePill.textContent  = `Retrieval: ${data.metrics?.retrieval_ms ?? '--'} ms`;
    generationTimePill.textContent = `Generation: ${data.metrics?.generation_ms ?? '--'} ms`;
  }

  if (data.is_fallback) {
    generationTimePill.innerHTML += ` &bull; <span style="color: #fbbf24; font-weight: 600;">🛡️ Offline Fallback</span>`;
  }

  const sources = data.sources || [];
  sourceCount.textContent = sources.length;
  citationsContainer.innerHTML = '';

  sources.forEach(src => {
    const scorePct = (src.score * 100).toFixed(1);
    const acc = document.createElement('div');
    acc.className = 'source-accordion';
    acc.innerHTML = `
      <button class="source-toggle" type="button">
        <span class="source-title">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
            <polyline points="14 2 14 8 20 8"></polyline>
          </svg>
          Page ${src.page} &mdash; ${escapeHtml(src.filename)}
          <span class="source-score">${scorePct}% Match</span>
        </span>
        <svg class="source-icon-arrow" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="6 9 12 15 18 9"></polyline>
        </svg>
      </button>
      <div class="source-content">${escapeHtml(src.text)}
        <div class="source-actions">
          <button class="btn btn-secondary btn-sm copy-btn" data-text="${escapeAttr(src.text)}">Copy Citation</button>
        </div>
      </div>`;

    acc.querySelector('.source-toggle').addEventListener('click', () => {
      acc.classList.toggle('open');
    });

    acc.querySelector('.copy-btn').addEventListener('click', e => {
      e.stopPropagation();
      const txt = e.target.getAttribute('data-text');
      navigator.clipboard.writeText(`[Page ${src.page}] ${txt}`).then(() => {
        e.target.textContent = 'Copied!';
        setTimeout(() => e.target.textContent = 'Copy Citation', 1500);
      });
    });

    citationsContainer.appendChild(acc);
  });

  answerSection.classList.remove('hidden');
}

// ─── Document Text & Chunks Inspector ─────────────────────────────────────────
function renderChunkInspector() {
  if (!chunkInspectorList) return;
  chunkInspectorList.innerHTML = `
    <div class="chunk-card">
      <div class="chunk-card-header">
        <span>Active Document: ${escapeHtml(state.activeDocumentName || 'None')}</span>
        <span>Pages: ${state.activeDocumentPages}</span>
      </div>
      <p class="chunk-text">Document text chunks are persisted as SQLite BLOB embeddings. Select AI Assistant tab to query grounded answers.</p>
    </div>
  `;
}

// ─── Utilities ────────────────────────────────────────────────────────────────
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function escapeHtml(text) {
  if (!text) return '';
  return text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function escapeAttr(text) {
  if (!text) return '';
  return text.replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

function formatMarkdown(text) {
  if (!text) return '';
  let html = escapeHtml(text);
  
  // Bold & Italics
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
  
  // Headings
  html = html.replace(/^### (.*$)/gim, '<h4 style="margin: 0.8rem 0 0.4rem; color: #60a5fa; font-weight: 600;">$1</h4>');
  html = html.replace(/^## (.*$)/gim, '<h3 style="margin: 1rem 0 0.5rem; color: #93c5fd; font-weight: 600;">$1</h3>');
  
  // Bullet Points
  html = html.replace(/^[\*\-\•]\s+(.*$)/gim, '<li style="margin-left: 1.2rem; list-style-type: disc; margin-bottom: 0.4rem; line-height: 1.6;">$1</li>');
  html = html.replace(/((?:<li style="margin-left: 1.2rem; list-style-type: disc; margin-bottom: 0.4rem; line-height: 1.6;">.*<\/li>\n?)+)/g, '<ul style="margin: 0.6rem 0 0.8rem; padding-left: 0.5rem;">$1</ul>');
  
  // Line Breaks
  html = html.replace(/\n\n/g, '<div style="margin-bottom: 0.7rem;"></div>');
  html = html.replace(/\n/g, '<br/>');
  return html;
}

// ─── Initialization ───────────────────────────────────────────────────────────
(async () => {
  await loadDocuments();
  if (state.documents.length > 0) {
    selectDocument(state.documents[0]);
  } else {
    showState('upload');
  }
  initVectorConstellationCanvas();
})();

// ─── Neural Vector Constellation Canvas Background Component ───────────────────
function initVectorConstellationCanvas() {
  const canvas = document.getElementById('bgCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  
  let width, height;
  let particles = [];
  const particleCount = 65;
  const maxDistance = 145;
  
  const colors = ['#6366f1', '#38bdf8', '#34d399', '#8b5cf6'];

  function resize() {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
  }

  window.addEventListener('resize', resize);
  resize();

  class Particle {
    constructor() {
      this.reset();
    }
    reset() {
      this.x = Math.random() * width;
      this.y = Math.random() * height;
      this.vx = (Math.random() - 0.5) * 0.45;
      this.vy = (Math.random() - 0.5) * 0.45;
      this.radius = Math.random() * 1.8 + 1;
      this.color = colors[Math.floor(Math.random() * colors.length)];
      this.alpha = Math.random() * 0.5 + 0.3;
    }
    update() {
      this.x += this.vx;
      this.y += this.vy;

      if (this.x < 0 || this.x > width) this.vx *= -1;
      if (this.y < 0 || this.y > height) this.vy *= -1;
    }
    draw() {
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
      ctx.fillStyle = this.color;
      ctx.globalAlpha = this.alpha;
      ctx.fill();
    }
  }

  for (let i = 0; i < particleCount; i++) {
    particles.push(new Particle());
  }

  function animate() {
    ctx.clearRect(0, 0, width, height);

    for (let i = 0; i < particles.length; i++) {
      particles[i].update();
      particles[i].draw();

      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < maxDistance) {
          const alpha = (1 - dist / maxDistance) * 0.22;
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = '#38bdf8';
          ctx.globalAlpha = alpha;
          ctx.lineWidth = 0.8;
          ctx.stroke();
        }
      }
    }

    requestAnimationFrame(animate);
  }

  animate();
}

