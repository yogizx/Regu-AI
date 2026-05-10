// static/js/app.js — ReguAI Frontend Logic
// All API calls go to Flask backend at localhost:5000

const API = '';  // Empty = same origin (Flask serves both)

// ── Navigation ──
function showSection(id) {
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.module-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  const btn = document.getElementById('btn-' + id);
  if (btn) btn.classList.add('active');
  if (id === 'history') loadHistory();
}

// ── Sample loading ──
function loadSample(section, idx) {
  const map = { anon:'anon-input', sum:'sum-input', comp:'comp-input', class:'class-input', pipe:'pipe-input' };
  const key = section;
  if (SAMPLES[key] && SAMPLES[key][idx] !== undefined) {
    document.getElementById(map[section]).value = SAMPLES[key][idx];
  }
}

function clearSec(s) {
  const inputs = { anon:'anon-input', sum:'sum-input', comp:'comp-input', class:'class-input', pipe:'pipe-input' };
  if (inputs[s]) document.getElementById(inputs[s]).value = '';
  if (s==='anon') { resetOutput('anon-output','Anonymised text will appear here…'); document.getElementById('anon-entities-card').style.display='none'; }
  if (s==='sum')  { resetOutput('sum-output','Summary will appear here…'); document.getElementById('sum-bullets-card').style.display='none'; }
  if (s==='comp') { document.getElementById('comp-results').style.display='none'; }
  if (s==='class'){ document.getElementById('class-results').style.display='none'; }
  if (s==='pipe') { document.getElementById('pipe-results').style.display='none'; document.getElementById('pipe-steps').style.display='none'; }
}

function resetOutput(id, msg) {
  document.getElementById(id).innerHTML = `<span class="output-placeholder">${msg}</span>`;
}

function showLoader(id, show) {
  document.getElementById(id + '-loader').classList.toggle('show', show);
}

// ── Generic POST to Flask API ──
async function callAPI(endpoint, text) {
  const res = await fetch(`${API}/api/${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `Server error ${res.status}`);
  }
  return res.json();
}

// ── Helper: render masked text ──
function renderMasked(text) {
  const masks = ['[PATIENT_NAME]','[PERSON_NAME]','[DOCTOR_NAME]','[HOSPITAL_NAME]',
                 '[PHONE]','[EMAIL]','[DATE]','[PINCODE]','[REPORT_ID]',
                 '[TRIAL_ID]','[PATIENT_ID]','[SUBJECT_ID]','[REG_NUMBER]'];
  masks.forEach(m => {
    text = text.split(m).join(`<span class="highlight-mask">${m}</span>`);
  });
  return text;
}

// ── Helper: score bar HTML ──
function scoreBar(label, value, color, labelWidth) {
  const w = labelWidth || '130px';
  return `<div class="score-row">
    <span class="score-label" style="width:${w}">${label}</span>
    <div class="score-bar"><div class="score-fill" style="width:${value}%;background:${color}"></div></div>
    <span class="score-pct">${value}%</span>
  </div>`;
}

// ── Helper: severity badge ──
function severityBadge(label) {
  const cls  = { Death:'sev-death', Disability:'sev-disability', Hospitalization:'sev-hospitalization', Other:'sev-other' };
  const icon = { Death:'💀', Disability:'♿', Hospitalization:'🏥', Other:'ℹ️' };
  return `<div class="severity-badge ${cls[label]||'sev-other'}">${icon[label]||'•'} ${label}</div>`;
}


// ════════════════════════════════════════
//  MODULE A — ANONYMISATION
// ════════════════════════════════════════
async function runModule(type) {
  if (type === 'anon') await runAnon();
  else if (type === 'sum') await runSum();
  else if (type === 'comp') await runComp();
  else if (type === 'class') await runClass();
}

async function runAnon() {
  const text = document.getElementById('anon-input').value.trim();
  if (!text) return alert('Please enter document text first.');
  showLoader('anon', true);
  resetOutput('anon-output', 'Processing with NER…');
  document.getElementById('anon-entities-card').style.display = 'none';
  try {
    const data = await callAPI('anonymise', text);
    document.getElementById('anon-output').innerHTML = renderMasked(data.anonymised_text || '');
    if (data.entities && data.entities.length > 0) {
      document.getElementById('anon-entities-card').style.display = 'block';
      let html = '<div style="display:flex;flex-wrap:wrap;gap:0.4rem">';
      data.entities.forEach(e => {
        html += `<div class="chip chip-missing" style="flex-direction:column;align-items:flex-start;padding:6px 10px">
          <span style="color:var(--muted2);font-size:0.65rem;margin-bottom:2px">${e.type}</span>
          <span style="color:#c4b5fd">${e.original || '—'}</span>
        </div>`;
      });
      html += '</div>';
      document.getElementById('anon-entities').innerHTML = html;
    }
  } catch(e) {
    resetOutput('anon-output', `❌ Error: ${e.message}. Is the Flask server running? (python app.py)`);
  }
  showLoader('anon', false);
}


// ════════════════════════════════════════
//  MODULE B — SUMMARISATION
// ════════════════════════════════════════
async function runSum() {
  const text = document.getElementById('sum-input').value.trim();
  if (!text) return alert('Please enter document text first.');
  showLoader('sum', true);
  resetOutput('sum-output', 'Running TF-IDF sentence scoring…');
  document.getElementById('sum-bullets-card').style.display = 'none';
  try {
    const data = await callAPI('summarise', text);
    const urgColor = data.urgency === 'Immediate' ? 'var(--red)' : data.urgency === 'Routine' ? 'var(--gold)' : 'var(--green)';
    document.getElementById('sum-output').innerHTML =
      `<div style="margin-bottom:0.5rem">
        <span style="font-size:0.72rem;font-family:var(--mono);color:var(--muted2)">${data.document_type || ''}</span> &nbsp;
        <span style="font-size:0.72rem;font-family:var(--mono);color:${urgColor};padding:2px 6px;background:rgba(255,255,255,0.04);border-radius:4px">${data.urgency || ''}</span>
        <span style="font-size:0.72rem;font-family:var(--mono);color:var(--muted);margin-left:8px">${data.sentence_count || ''} sentences processed</span>
      </div>${data.summary || ''}`;

    if (data.key_points && data.key_points.length > 0) {
      document.getElementById('sum-bullets-card').style.display = 'block';
      let html = '<ul style="padding-left:1.1rem;color:var(--muted2);font-size:0.85rem;line-height:2">';
      data.key_points.forEach(p => { html += `<li>${p}</li>`; });
      html += '</ul>';
      document.getElementById('sum-bullets').innerHTML = html;
    }
  } catch(e) {
    resetOutput('sum-output', `❌ Error: ${e.message}`);
  }
  showLoader('sum', false);
}


// ════════════════════════════════════════
//  MODULE C — COMPLETENESS CHECK
// ════════════════════════════════════════
async function runComp() {
  const text = document.getElementById('comp-input').value.trim();
  if (!text) return alert('Please enter document text first.');
  showLoader('comp', true);
  document.getElementById('comp-results').style.display = 'none';
  try {
    const data = await callAPI('completeness', text);
    document.getElementById('comp-results').style.display = 'block';

    // Score bar
    const score = data.score || 0;
    const barColor = score >= 80 ? 'var(--green)' : score >= 60 ? 'var(--gold)' : 'var(--red)';
    document.getElementById('comp-score-bar').innerHTML = scoreBar('Overall Score', score, barColor);

    // Field chips
    let chips = '';
    (data.present || []).forEach(f => { chips += `<span class="chip chip-ok">✓ ${f}</span>`; });
    (data.missing || []).forEach(f => { chips += `<span class="chip chip-missing">✗ ${f}</span>`; });
    document.getElementById('comp-chips').innerHTML = chips;

    // Notes
    const recColor = (data.recommendation||'').includes('Approve') ? 'var(--green)' :
                     (data.recommendation||'').includes('Return')  ? 'var(--gold)' : 'var(--red)';
    document.getElementById('comp-notes').innerHTML =
      `<span style="color:${recColor};font-weight:500;font-family:var(--body)">▶ ${data.recommendation || ''}</span>\n\n${data.notes || ''}`;
  } catch(e) {
    document.getElementById('comp-results').style.display = 'block';
    document.getElementById('comp-notes').textContent = `❌ Error: ${e.message}`;
  }
  showLoader('comp', false);
}


// ════════════════════════════════════════
//  MODULE D — SEVERITY CLASSIFICATION
// ════════════════════════════════════════
async function runClass() {
  const text = document.getElementById('class-input').value.trim();
  if (!text) return alert('Please enter document text first.');
  showLoader('class', true);
  document.getElementById('class-results').style.display = 'none';
  try {
    const data = await callAPI('classify', text);
    document.getElementById('class-results').style.display = 'block';

    const label = data.label || 'Other';
    const prioColor = {'P1-Critical':'var(--red)','P2-High':'var(--gold)','P3-Medium':'var(--accent)','P4-Low':'var(--green)'};
    document.getElementById('class-badge').innerHTML =
      `${severityBadge(label)} <span style="font-size:0.78rem;font-family:var(--mono);color:${prioColor[data.priority]||'var(--muted2)'};margin-left:10px">${data.priority||''}</span>`;

    const conf = data.confidence || {};
    const barColors = { Death:'var(--red)', Disability:'var(--gold)', Hospitalization:'var(--accent)', Other:'var(--green)' };
    let bars = '';
    ['Death','Disability','Hospitalization','Other'].forEach(k => {
      bars += scoreBar(k, conf[k] || 0, barColors[k]);
    });
    document.getElementById('class-score-bars').innerHTML = bars;
    document.getElementById('class-reason').textContent = data.reasoning || '';
  } catch(e) {
    document.getElementById('class-results').style.display = 'block';
    document.getElementById('class-reason').textContent = `❌ Error: ${e.message}`;
  }
  showLoader('class', false);
}


// ════════════════════════════════════════
//  FULL PIPELINE
// ════════════════════════════════════════
function setStep(id, state) {
  const el = document.getElementById('step-' + id);
  if (!el) return;
  el.className = 'step-item ' + state;
  if (state === 'done') el.querySelector('.step-circle').innerHTML = '✓';
}

async function runPipeline() {
  const text = document.getElementById('pipe-input').value.trim();
  if (!text) return alert('Please enter document text first.');

  const btn = document.getElementById('pipe-btn');
  btn.disabled = true;
  document.getElementById('pipe-steps').style.display = 'flex';
  document.getElementById('pipe-results').style.display = 'none';
  ['anon','sum','comp','class'].forEach(s => {
    setStep(s, '');
    document.getElementById('step-'+s).querySelector('.step-circle').textContent = ['anon','sum','comp','class'].indexOf(s)+1;
  });

  try {
    setStep('anon','active');
    setStep('sum','active');
    setStep('comp','active');
    setStep('class','active');

    const data = await callAPI('pipeline', text);

    setStep('anon','done'); setStep('sum','done'); setStep('comp','done'); setStep('class','done');

    document.getElementById('pipe-results').style.display = 'block';

    // Anon
    document.getElementById('pipe-anon-out').innerHTML = renderMasked(data.anonymisation?.anonymised_text || '');

    // Summary
    const sum = data.summarisation || {};
    const urgColor = sum.urgency === 'Immediate' ? 'var(--red)' : sum.urgency === 'Routine' ? 'var(--gold)' : 'var(--green)';
    let sumHtml = `<span style="font-size:0.72rem;font-family:var(--mono);color:${urgColor};display:block;margin-bottom:4px">${sum.urgency||''}</span>${sum.summary||''}`;
    if (sum.key_points?.length) {
      sumHtml += '<ul style="margin-top:8px;padding-left:1rem;color:var(--muted2);font-size:0.8rem;line-height:1.8">';
      sum.key_points.forEach(p => { sumHtml += `<li>${p}</li>`; });
      sumHtml += '</ul>';
    }
    document.getElementById('pipe-sum-out').innerHTML = sumHtml;

    // Completeness
    const comp = data.completeness || {};
    let chips = '';
    (comp.missing||[]).forEach(f => { chips += `<span class="chip chip-missing">✗ ${f}</span>`; });
    (comp.present||[]).slice(0,4).forEach(f => { chips += `<span class="chip chip-ok">✓ ${f}</span>`; });
    document.getElementById('pipe-comp-chips').innerHTML = chips;
    const recColor = (comp.recommendation||'').includes('Approve') ? 'var(--green)' : (comp.recommendation||'').includes('Return') ? 'var(--gold)' : 'var(--red)';
    document.getElementById('pipe-comp-score').innerHTML =
      `Score: <span style="color:${recColor};font-family:var(--head);font-weight:700">${comp.score||0}%</span>  ·  ${comp.recommendation||''}`;

    // Classification
    const cls = data.classification || {};
    const label = cls.label || 'Other';
    document.getElementById('pipe-sev-badge').innerHTML = severityBadge(label);
    const conf = cls.confidence || {};
    const barColors = { Death:'var(--red)', Disability:'var(--gold)', Hospitalization:'var(--accent)', Other:'var(--green)' };
    let sevBars = '';
    ['Death','Disability','Hospitalization','Other'].forEach(k => {
      sevBars += scoreBar(k, conf[k]||0, barColors[k], '110px');
    });
    document.getElementById('pipe-sev-bars').innerHTML = sevBars;

    // Action
    document.getElementById('pipe-action').textContent = data.action_summary || '';

  } catch(e) {
    document.getElementById('pipe-results').style.display = 'block';
    document.getElementById('pipe-action').textContent = `❌ Error: ${e.message}\n\nMake sure the Flask server is running: python app.py`;
  }
  btn.disabled = false;
}


// ════════════════════════════════════════
//  HISTORY
// ════════════════════════════════════════
async function loadHistory() {
  document.getElementById('history-table').innerHTML = '<p style="color:var(--muted2);font-size:0.85rem">Loading…</p>';
  try {
    const res = await fetch('/api/history');
    const rows = await res.json();
    if (!rows.length) {
      document.getElementById('history-table').innerHTML = '<p style="color:var(--muted2);font-size:0.85rem">No documents processed yet.</p>';
      return;
    }
    const sev = { Death:'#fca5a5', Disability:'#fcd34d', Hospitalization:'#93c5fd', Other:'#6ee7b7' };
    let html = `<table class="history-table"><thead><tr>
      <th>ID</th><th>File</th><th>Time</th><th>Severity</th><th>Priority</th><th>Compliance %</th>
    </tr></thead><tbody>`;
    rows.forEach(r => {
      const color = sev[r.label] || 'var(--muted2)';
      html += `<tr>
        <td style="font-family:var(--mono);color:var(--muted)">#${r.id}</td>
        <td>${r.filename || '—'}</td>
        <td style="font-family:var(--mono);font-size:0.75rem">${(r.time||'').slice(0,16)}</td>
        <td style="color:${color};font-weight:600">${r.label || '—'}</td>
        <td style="font-family:var(--mono);font-size:0.75rem">${r.priority || '—'}</td>
        <td>${r.score != null ? r.score + '%' : '—'}</td>
      </tr>`;
    });
    html += '</tbody></table>';
    document.getElementById('history-table').innerHTML = html;
  } catch(e) {
    document.getElementById('history-table').innerHTML = `<p style="color:var(--red);font-size:0.85rem">❌ ${e.message}</p>`;
  }
}
