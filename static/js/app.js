// Gmail-style prototype frontend. All classification happens server-side
// (Naive Bayes in classifier.py); this file renders what the API returns
// and wires up folders, search, star, compose/send, and the detector tab.

const state = {
  inbox: [],       // from /api/inbox -- is_starred/is_read are server-authoritative
  sent: [],        // from /api/sent
  presets: [],
  folder: "inbox", // inbox | starred | sent | spam
  searchTerm: "",
};

// ---------------------------------------------------------------- helpers
function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function initials(name) {
  return (name || "?").trim().charAt(0).toUpperCase();
}

function showToast(msg) {
  const toast = document.getElementById("toast");
  toast.textContent = msg;
  toast.style.display = "block";
  clearTimeout(window._toastTimer);
  window._toastTimer = setTimeout(() => (toast.style.display = "none"), 2500);
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `Request failed (${res.status})`);
  }
  return res.json();
}

function updateNavCounts() {
  document.getElementById("count-inbox").textContent = state.inbox.length;
  document.getElementById("count-starred").textContent = state.inbox.filter((e) => e.is_starred).length || "";
  document.getElementById("count-sent").textContent = state.sent.length || "";
  const spamCount = state.inbox.filter((e) => e.is_spam).length;
  document.getElementById("count-spam").textContent = spamCount || "";
}

// ---------------------------------------------------------------- views
function switchView(viewName) {
  document.querySelectorAll(".view").forEach((v) => (v.style.display = "none"));
  document.getElementById(`view-${viewName}`).style.display = "block";
  document.querySelectorAll(".nav-item").forEach((item) => {
    const matches = viewName === "folder"
      ? item.dataset.folder === state.folder
      : item.dataset.view === viewName;
    item.classList.toggle("active", matches);
  });
  if (viewName === "about") loadAbout();
}

document.querySelectorAll(".nav-item[data-folder]").forEach((item) => {
  item.addEventListener("click", () => {
    state.folder = item.dataset.folder;
    switchView("folder");
    renderFolder();
  });
});
document.querySelectorAll(".nav-item[data-view]").forEach((item) => {
  item.addEventListener("click", () => switchView(item.dataset.view));
});

// ---------------------------------------------------------------- hamburger + search
document.getElementById("menuToggle").addEventListener("click", () => {
  document.body.classList.toggle("sidebar-hidden");
});

document.getElementById("searchInput").addEventListener("input", (e) => {
  state.searchTerm = e.target.value.trim().toLowerCase();
  if (document.getElementById("view-folder").style.display !== "none") {
    renderFolder();
  }
});

// ---------------------------------------------------------------- data loading
async function loadAll() {
  const listEl = document.getElementById("emailList");
  try {
    [state.inbox, state.sent] = await Promise.all([api("/api/inbox"), api("/api/sent")]);
  } catch (e) {
    listEl.innerHTML = `<div class="loading">Couldn't load mail: ${escapeHtml(e.message)}</div>`;
    return;
  }
  updateNavCounts();
  renderFolder();

  const spamCount = state.inbox.filter((e) => e.is_spam).length;
  if (spamCount > 0) {
    showToast(`${spamCount} of ${state.inbox.length} inbox messages flagged as spam by the model.`);
  }
}

const FOLDER_META = {
  inbox: { title: "Inbox", sub: "30 real emails, randomly sampled from a held-out set the model never trained on. Spam is highlighted in red.", icon: "\u{1F4E5}", empty: "Your inbox is empty." },
  starred: { title: "Starred", sub: "Emails you've starred, across your inbox.", icon: "★", empty: "No starred emails yet — click the star on any email to add one here." },
  sent: { title: "Sent", sub: "Everything you've sent from Compose this session. Compose never runs the spam check — use the Spam Detector tab for that.", icon: "\u{1F4E4}", empty: "Nothing sent yet — click Compose to write your first message." },
  spam: { title: "Spam", sub: "Inbox emails the model classified as spam — same model, same highlighting, just filtered to this folder.", icon: "⚠", empty: "No spam in your inbox right now." },
};

function matchesSearch(email) {
  if (!state.searchTerm) return true;
  const haystack = `${email.sender_name || ""} ${email.to || ""} ${email.subject} ${email.snippet}`.toLowerCase();
  return haystack.includes(state.searchTerm);
}

function renderFolder() {
  const meta = FOLDER_META[state.folder];
  document.getElementById("folderTitle").textContent = meta.title;
  document.getElementById("folderSub").textContent = meta.sub;

  const listEl = document.getElementById("emailList");
  let items;
  if (state.folder === "sent") {
    items = state.sent.filter(matchesSearch);
  } else {
    items = state.inbox
      .filter((e) => (state.folder === "starred" ? e.is_starred : state.folder === "spam" ? e.is_spam : true))
      .filter(matchesSearch);
  }

  if (items.length === 0) {
    listEl.innerHTML = `<div class="empty-folder"><span class="empty-icon">${meta.icon}</span>${escapeHtml(
      state.searchTerm ? "No emails match your search." : meta.empty
    )}</div>`;
    return;
  }

  if (state.folder === "sent") {
    listEl.innerHTML = items
      .map(
        (email) => `
        <div class="email-row read" data-sent-id="${email.id}">
          <span class="star-icon" style="visibility:hidden;">&#9733;</span>
          <span class="sender-col">To: ${escapeHtml(email.to)}</span>
          <span class="subject-col">
            <span class="subject-text">${escapeHtml(email.subject)}</span>
            <span class="snippet-text">&mdash; ${escapeHtml(email.snippet || "(no body)")}</span>
          </span>
          <span class="date-col">${escapeHtml(email.date)}</span>
        </div>`
      )
      .join("");
    listEl.querySelectorAll(".email-row").forEach((row) => {
      row.addEventListener("click", () => openSentEmail(row.dataset.sentId));
    });
    return;
  }

  listEl.innerHTML = items
    .map((email) => {
      const rowClasses = [
        "email-row",
        email.is_read ? "read" : "unread",
        email.is_spam ? "spam-row" : "",
      ].join(" ");
      return `
        <div class="${rowClasses}" data-id="${email.id}">
          <span class="star-icon ${email.is_starred ? "starred" : ""}" data-star-id="${email.id}">&#9733;</span>
          <span class="sender-col">${escapeHtml(email.sender_name)}</span>
          <span class="subject-col">
            ${email.is_spam ? '<span class="spam-tag">Spam</span>' : ""}
            <span class="subject-text">${escapeHtml(email.subject)}</span>
            <span class="snippet-text">&mdash; ${escapeHtml(email.snippet)}</span>
          </span>
          <span class="date-col">${escapeHtml(email.date)}</span>
        </div>`;
    })
    .join("");

  listEl.querySelectorAll(".email-row").forEach((row) => {
    row.addEventListener("click", (evt) => {
      if (evt.target.closest(".star-icon")) return; // handled separately
      openEmail(row.dataset.id);
    });
  });
  listEl.querySelectorAll(".star-icon").forEach((star) => {
    star.addEventListener("click", (evt) => {
      evt.stopPropagation();
      toggleStar(star.dataset.starId);
    });
  });
}

async function toggleStar(id) {
  const email = state.inbox.find((e) => e.id === id);
  if (!email) return;
  const next = !email.is_starred;
  try {
    await api(`/api/email/${id}/star`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ starred: next }),
    });
    email.is_starred = next;
    updateNavCounts();
    renderFolder();
  } catch (e) {
    showToast(e.message);
  }
}

async function openEmail(id) {
  switchView("detail");
  const container = document.getElementById("emailDetailContent");
  container.innerHTML = '<div class="loading">Loading&hellip;</div>';
  let email;
  try {
    email = await api(`/api/email/${id}`);
    await api(`/api/email/${id}/read`, { method: "POST" }).catch(() => {});
    const local = state.inbox.find((e) => e.id === id);
    if (local) local.is_read = true;
  } catch (e) {
    container.innerHTML = `<div class="loading">${escapeHtml(e.message)}</div>`;
    return;
  }
  const c = email.classification;
  const isSpam = c.is_spam;

  container.innerHTML = `
    <div class="verdict-banner ${isSpam ? "spam" : "ham"}">
      <span>${isSpam ? "&#9888; Classified as SPAM" : "&#10003; Classified as not spam"}</span>
      <span class="pct">${isSpam ? c.spam_probability : c.ham_probability}%</span>
    </div>
    <div class="detail-card ${isSpam ? "spam" : ""}">
      <h1 class="detail-subject">${escapeHtml(email.subject)}</h1>
      <div class="detail-meta">
        <span class="avatar">${initials(email.sender_name)}</span>
        <div>
          <div><strong>${escapeHtml(email.sender_name)}</strong> &lt;${escapeHtml(email.sender_email)}&gt;</div>
          <div>${escapeHtml(email.date)}</div>
        </div>
      </div>
      <div class="detail-body">${c.highlighted_html}</div>
    </div>
  `;
}

function openSentEmail(id) {
  const email = state.sent.find((e) => e.id === id);
  if (!email) return;
  switchView("detail");
  document.getElementById("emailDetailContent").innerHTML = `
    <div class="detail-card">
      <h1 class="detail-subject">${escapeHtml(email.subject)}</h1>
      <div class="detail-meta">
        <span class="avatar">Me</span>
        <div>
          <div>To: <strong>${escapeHtml(email.to)}</strong></div>
          <div>${escapeHtml(email.date)}</div>
        </div>
      </div>
      <div class="detail-body">${escapeHtml(email.body || "(no body)")}</div>
    </div>
  `;
}

document.getElementById("backToFolder").addEventListener("click", () => {
  switchView("folder");
  renderFolder();
});

// ---------------------------------------------------------------- compose (no spam check)
const composeOverlay = document.getElementById("composeOverlay");
document.getElementById("composeBtn").addEventListener("click", () => {
  document.getElementById("composeTo").value = "";
  document.getElementById("composeSubject").value = "";
  document.getElementById("composeBody").value = "";
  composeOverlay.style.display = "flex";
});
document.getElementById("closeCompose").addEventListener("click", () => {
  composeOverlay.style.display = "none";
});
document.getElementById("sendComposeBtn").addEventListener("click", async () => {
  const to = document.getElementById("composeTo").value.trim();
  const subject = document.getElementById("composeSubject").value.trim();
  const body = document.getElementById("composeBody").value.trim();
  if (!to) {
    showToast("Add a recipient first.");
    return;
  }
  try {
    const result = await api("/api/compose/send", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ to, subject, body }),
    });
    state.sent.unshift(result.email);
    updateNavCounts();
    composeOverlay.style.display = "none";
    showToast(`Sent to ${to}. Check the Sent folder.`);
    if (state.folder === "sent") renderFolder();
  } catch (e) {
    showToast(e.message);
  }
});

// ---------------------------------------------------------------- spam detector tab
async function loadPresets() {
  try {
    state.presets = await api("/api/detector/presets");
  } catch (e) {
    return;
  }
  const row = document.getElementById("presetRow");
  row.innerHTML = state.presets
    .map(
      (p) => `
      <button class="preset-chip ${p.true_label === "spam" ? "spam-preset" : ""}" data-id="${p.id}">
        ${escapeHtml(p.title)}
      </button>`
    )
    .join("");
  row.querySelectorAll(".preset-chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      const preset = state.presets.find((p) => p.id === btn.dataset.id);
      document.getElementById("detectorSubject").value = preset.subject;
      document.getElementById("detectorBody").value = preset.body;
      runDetectorCheck();
    });
  });
}

function renderWordTable(words, kind) {
  if (!words || words.length === 0) return "<p style='color:#999;font-size:12.5px;'>None of this email's words were strong indicators.</p>";
  const rows = words
    .slice(0, 8)
    .map(
      (w) => `
      <tr>
        <td class="${kind === "spam" ? "word-spam" : "word-ham"}">${escapeHtml(w.word)}</td>
        <td>${w.spam_count}</td>
        <td>${w.ham_count}</td>
        <td>${(w.spamminess * 100).toFixed(1)}%</td>
        <td>${w.likelihood_ratio.toFixed(2)}&times;</td>
      </tr>`
    )
    .join("");
  return `
    <table class="word-table">
      <thead><tr><th>Word</th><th>Seen in spam</th><th>Seen in ham</th><th>P(Spam|word)</th><th>Likelihood ratio</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

let _lastDetectorResult = null;

function renderDetectorResult(result) {
  _lastDetectorResult = result;
  const resultEl = document.getElementById("detectorResult");
  const isSpam = result.is_spam;
  resultEl.innerHTML = `
    <div class="result-verdict ${isSpam ? "spam" : "ham"}">
      <div class="big-pct">${isSpam ? result.spam_probability : result.ham_probability}%</div>
      <div>
        <div class="verdict-label">${isSpam ? "Classified as SPAM" : "Classified as NOT spam"}</div>
        <div class="verdict-sub">${result.known_token_count} of ${result.token_count} words were in the model's vocabulary</div>
      </div>
    </div>

    <div class="explain-section">
      <h4>How Naive Bayes reached this result</h4>
      <div class="math-steps">
        <div class="math-step">
          <div class="step-head"><span class="step-num">Step 1</span> Priors (how common each class was in training)</div>
          <div class="formula">P(Spam) = <span class="hl">${result.p_spam_prior}</span> &nbsp;&nbsp; P(Ham) = <span class="hl">${result.p_ham_prior}</span></div>
        </div>
        <div class="math-step">
          <div class="step-head"><span class="step-num">Step 2</span> For every word w in this email, multiply in P(w|Spam) and P(w|Ham), learned from training counts with Laplace smoothing</div>
          <div class="formula">P(w|Spam) = (count(w,Spam) + 1) / (total spam words + vocab size)</div>
        </div>
        <div class="math-step">
          <div class="step-head"><span class="step-num">Step 3</span> Multiplying thousands of tiny probabilities underflows, so we sum their logs instead (equivalent math, done in log-space)</div>
          <div class="formula">log P(Spam|words) &prop; log P(Spam) + &Sigma; log P(w|Spam) = <span class="hl">${result.log_prob_spam}</span></div>
          <div class="formula">log P(Ham|words) &nbsp;&prop; log P(Ham) &nbsp;+ &Sigma; log P(w|Ham) &nbsp;= <span class="hl">${result.log_prob_ham}</span></div>
        </div>
        <div class="math-step">
          <div class="step-head"><span class="step-num">Step 4</span> Convert the log-odds difference back to a 0&ndash;100% probability with a sigmoid</div>
          <div class="formula">P(Spam | this email) = <span class="hl">${result.spam_probability}%</span></div>
        </div>
      </div>
    </div>

    <div class="explain-section">
      <h4>Words pushing toward SPAM</h4>
      ${renderWordTable(result.top_spam_words, "spam")}
    </div>

    <div class="explain-section">
      <h4>Words pushing toward HAM</h4>
      ${renderWordTable(result.top_ham_words, "ham")}
    </div>

    <div class="explain-section">
      <h4>Highlighted email</h4>
      <div class="highlighted-preview">${result.highlighted_html}</div>
    </div>

    <div class="feedback-row" id="feedbackRow">
      <span class="feedback-label">Was this verdict right?</span>
      <button class="feedback-btn agree" id="feedbackAgree">&#10003; Correct</button>
      <button class="feedback-btn disagree" id="feedbackDisagree">&#10007; Wrong, it's actually ${isSpam ? "not spam" : "spam"}</button>
    </div>
    <div id="feedbackNote"></div>
  `;

  document.getElementById("feedbackAgree").addEventListener("click", () => submitFeedback(true));
  document.getElementById("feedbackDisagree").addEventListener("click", () => submitFeedback(false));
}

async function submitFeedback(agree) {
  const row = document.getElementById("feedbackRow");
  const note = document.getElementById("feedbackNote");
  let counts;
  try {
    counts = await api("/api/detector/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ agree }),
    });
  } catch (e) {
    showToast(e.message);
    return;
  }
  row.innerHTML = `<span class="feedback-thanks">Thanks, feedback recorded (${counts.agree_count} agreed, ${counts.disagree_count} disagreed this session).</span>`;
  if (!agree) {
    note.innerHTML = `
      <div class="feedback-note">
        This Naive Bayes model trains once, in a single batch, when the server starts &mdash; it has no
        online-learning step, so this correction can't change its answer right now. To actually fix this
        case: add this email (with the correct label) to <code>data/spam_dataset.csv</code> and restart the
        server so it retrains on the corrected data, then rerun <code>evaluate.py</code> to confirm accuracy
        didn't regress elsewhere.
      </div>`;
  } else {
    note.innerHTML = "";
  }
}

async function runDetectorCheck() {
  const subject = document.getElementById("detectorSubject").value;
  const body = document.getElementById("detectorBody").value;
  const resultEl = document.getElementById("detectorResult");
  if (!body.trim() && !subject.trim()) {
    showToast("Write or paste some email text first.");
    return;
  }
  resultEl.innerHTML = '<div class="loading">Classifying&hellip;</div>';
  try {
    const result = await api("/api/detector/check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ subject, body }),
    });
    renderDetectorResult(result);
  } catch (e) {
    resultEl.innerHTML = `<div class="loading">${escapeHtml(e.message)}</div>`;
  }
}

document.getElementById("checkBtn").addEventListener("click", runDetectorCheck);

// ---------------------------------------------------------------- about tab
async function loadAbout() {
  const el = document.getElementById("aboutContent");
  let stats;
  try {
    stats = await api("/api/stats");
  } catch (e) {
    el.innerHTML = `<div class="loading">${escapeHtml(e.message)}</div>`;
    return;
  }
  document.getElementById("modelBadge").textContent =
    `Naive Bayes · ${stats.held_out_accuracy}% held-out accuracy`;

  const cm = stats.confusion_matrix;

  el.innerHTML = `
    <div class="stat-grid">
      <div class="stat-card"><div class="stat-value">${stats.training_examples.toLocaleString()}</div><div class="stat-label">Training emails</div></div>
      <div class="stat-card"><div class="stat-value">${stats.vocabulary_size.toLocaleString()}</div><div class="stat-label">Vocabulary size</div></div>
      <div class="stat-card"><div class="stat-value">${stats.held_out_accuracy}%</div><div class="stat-label">Held-out accuracy</div></div>
      <div class="stat-card"><div class="stat-value">${stats.held_out_f1}%</div><div class="stat-label">Held-out F1 score</div></div>
    </div>

    <h3>Dataset</h3>
    <p>${escapeHtml(stats.dataset_source)}. This is a <strong>real, publicly released, labeled corpus</strong> &mdash;
    no synthetic, hand-written, or AI-generated emails were used anywhere in training. A fixed set of 35 real emails
    was set aside <em>before</em> training and never trained on; those are what populate the inbox (30 random samples)
    and the Spam Detector presets (5 fixed examples), so every "demo" email you see is genuinely out-of-sample.</p>

    <h3>Held-out accuracy &mdash; confusion matrix</h3>
    <p>${stats.held_out_train_count.toLocaleString()} emails trained on, ${stats.held_out_test_count.toLocaleString()}
    tested on (never seen during training), computed fresh every time this server starts:</p>
    <div class="confusion-wrap">
      <div class="confusion-grid">
        <div></div>
        <div class="cm-head">Predicted Spam</div>
        <div class="cm-head">Predicted Ham</div>
        <div class="cm-row-label">Actual<br>Spam</div>
        <div class="cm-cell tp">${cm.tp}<span class="cm-sub">True Positive</span></div>
        <div class="cm-cell fn">${cm.fn}<span class="cm-sub">False Negative</span></div>
        <div class="cm-row-label">Actual<br>Ham</div>
        <div class="cm-cell fp">${cm.fp}<span class="cm-sub">False Positive</span></div>
        <div class="cm-cell tn">${cm.tn}<span class="cm-sub">True Negative</span></div>
      </div>
      <div class="confusion-legend">
        <div><span style="background:#188038"></span>Correct calls (${cm.tp + cm.tn} of ${cm.tp + cm.tn + cm.fp + cm.fn})</div>
        <div><span style="background:#d93025"></span>Mistakes (${cm.fp + cm.fn} of ${cm.tp + cm.tn + cm.fp + cm.fn})</div>
        <div>Precision: ${stats.held_out_precision}% &middot; Recall: ${stats.held_out_recall}%</div>
      </div>
    </div>

    <h3>How the model works &mdash; Multinomial Naive Bayes</h3>
    <p>For an email with words w&#8321;&hellip;w&#8345;, Bayes' theorem gives:</p>
    <p><code>P(Spam | words) &prop; P(Spam) &times; P(w&#8321;|Spam) &times; P(w&#8322;|Spam) &times; &hellip; &times; P(w&#8345;|Spam)</code></p>
    <p>The same product is computed for Ham, and whichever is larger wins. "Naive" refers to the simplifying
    assumption that, given the class, each word is independent of the others &mdash; untrue in real language, but
    effective for spam filtering in practice. Each <code>P(word | class)</code> is learned directly from word
    frequencies in the training data, with Laplace smoothing (&alpha; = ${stats.alpha}) so a word never seen in one
    class during training doesn't force that class's probability to exactly zero.</p>
    <p>Because this reasons about <em>which words co-occur and how strongly each one is statistically associated with
    spam</em> &mdash; not a fixed blocklist &mdash; it generalizes to spam phrased in new ways, as long as it uses
    words the model has seen before in a similar statistical pattern. It is still fundamentally a word-frequency
    model, not a deep contextual/sequence model, so unusual phrasing built entirely from words that look equally at
    home in either class can still fool it &mdash; see the mistakes in the confusion matrix above.</p>

    ${stats.feedback.agree + stats.feedback.disagree > 0 ? `
    <h3>Spam Detector feedback this session</h3>
    <p>${stats.feedback.agree} agreed with the model's verdict, ${stats.feedback.disagree} disagreed, on the
    Spam Detector tab. This model doesn't retrain live (see the note there), but this is where that feedback
    would feed back into a real deployment's retraining pipeline.</p>` : ""}
  `;
}

// ---------------------------------------------------------------- boot
loadAll();
loadPresets();
