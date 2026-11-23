
// =====================================================================================
// INTERPRETATIONS
// =====================================================================================

const INTERPRETATIONS = {
  mean_green: "Write your interpretation here. Compare the two curves: the further apart they are, the more this feature helps to tell infected from healthy cells.",
  contrast: "Write your interpretation here. Contrast is how much the pixel values vary inside one image. What does a higher or lower contrast say about a stained parasite?",
  area_px: "Write your interpretation here. If image size differs between classes, the model could learn the size instead of the parasite. Is that a risk here?"
};

const FEATURES = [
  { key: "mean_green", title: "Feature 1: average brightness (green channel)", xlabel: "Average green value (0-255)", unit: "" },
  { key: "contrast", title: "Feature 2: contrast (variation of pixel values)", xlabel: "Contrast (std of gray levels)", unit: "" },
  { key: "area_px", title: "Feature 3: image size", xlabel: "Image area (pixels)", unit: " px" }
];

const CLASSES = ["Parasitized", "Uninfected"];

const API = (new URLSearchParams(location.search).get("api") || "").replace(/\/$/, "");

const $ = (s) => document.querySelector(s);

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;"
}[c]));

async function api(path, opts = {}) {
  const r = await fetch(API + path, opts);

  let data = null;

  try {
    data = await r.json();
  } catch (e) {}

  if (!r.ok) {
    const d = data && data.detail;

    throw new Error(
      d ? (typeof d === "string" ? d : JSON.stringify(d))
        : `${r.status} ${r.statusText}`
    );
  }

  return data;
}

function showMsg(id, kind, text) {
  const el = $(id);
  el.className = "msg show " + kind;
  el.textContent = text;
}

function hideMsg(id) {
  $(id).className = "msg";
}

function fmtSeconds(s) {
  const d = Math.floor(s / 86400);
  const h = Math.floor(s % 86400 / 3600);
  const m = Math.floor(s % 3600 / 60);
  const sec = s % 60;

  return [
    d ? d + "d" : "",
    (d || h) ? h + "h" : "",
    m + "m",
    sec + "s"
  ].filter(Boolean).join(" ");
}

const pct = (x) => (x * 100).toFixed(1) + "%";


// =====================================================================================
// TABS
// =====================================================================================

let currentTab = "dash";

document.querySelectorAll("nav.tabs button").forEach((b) => {
  b.addEventListener("click", () => {
    currentTab = b.dataset.tab;

    document.querySelectorAll("nav.tabs button").forEach((x) => {
      x.setAttribute("aria-selected", x === b);
    });

    document.querySelectorAll(".panel").forEach((p) => {
      p.classList.toggle(
        "active",
        p.id === "tab-" + currentTab
      );
    });

    if (currentTab === "viz") loadViz();
    if (currentTab === "data") refreshStatus();
    if (currentTab === "dash") refreshDashboard();
  });
});


// =====================================================================================
// DASHBOARD
// =====================================================================================

function metricsHTML(m) {

  if (!m || (!m.accuracy && m.accuracy !== 0)) {
    return '<p class="muted">No metrics available (is there any image in data/test?).</p>';
  }

  const card = (l, v) =>
    `<div class="card metric"><div class="label">${l}</div><div class="value">${v}</div></div>`;

  let html = `
    <div class="grid g4">
      ${card("Accuracy", pct(m.accuracy))}
      ${card("Precision", pct(m.precision))}
      ${card("Recall", pct(m.recall))}
      ${card("F1-score", pct(m.f1))}
    </div>
  `;

  if (m.confusion_matrix) {

    const cm = m.confusion_matrix;

    const total = cm.flat().reduce((a, b) => a + b, 0) || 1;

    const cell = (v) =>
      `<td style="background: color-mix(in srgb, var(--unin) ${Math.round(v / total * 100 * 1.6)}%, transparent)">${v}</td>`;

    html += `
      <p class="muted" style="margin-top:16px">
        Confusion matrix on ${m.test_images ?? total} test images
        (rows = real class, columns = prediction)
      </p>

      <table class="cm">
        <tr>
          <th></th>
          <th>Predicted Parasitized</th>
          <th>Predicted Uninfected</th>
        </tr>

        <tr>
          <th>Real Parasitized</th>
          ${cell(cm[0][0])}
          ${cell(cm[0][1])}
        </tr>

        <tr>
          <th>Real Uninfected</th>
          ${cell(cm[1][0])}
          ${cell(cm[1][1])}
        </tr>
      </table>
    `;
  }

  return html;
}


async function refreshHealth() {

  const pill = $("#pill");

  try {

    const h = await api("/health");

    pill.classList.add("on");

    $("#pillText").textContent =
      h.model_loaded ? "API online, model loaded" : "API online, no model";

    $("#mStatus").textContent = "Online";

    $("#mLoaded").textContent =
      h.model_loaded ? "Yes" : "No";

    $("#mUptime").textContent =
      fmtSeconds(h.uptime_seconds);

    $("#startedAt").textContent =
      `Started at ${h.started_at}. Refreshes automatically every 3 seconds.`;

  } catch (e) {

    pill.classList.remove("on");

    $("#pillText").textContent = "API offline";

    $("#mStatus").textContent = "Offline";
    $("#mLoaded").textContent = "-";
    $("#mUptime").textContent = "-";
  }
}


async function refreshDashboard() {

  await refreshHealth();

  try {

    const st = await api("/status");

    const info = st.model_info || {};

    const ev = info.last_evaluation || info.test_metrics;

    $("#evalBox").innerHTML =
      (info.last_evaluation
        ? `<p class="muted">Last evaluation: ${esc(info.last_evaluation.evaluated_at || "")}</p>`
        : "") +
      (ev
        ? metricsHTML(ev)
        : '<p class="muted">No evaluation yet. Click the button above.</p>');

    $("#lastTrain").innerHTML = info.last_trained
      ? `
        <h2>Last training</h2>
        <p class="muted">Trained on ${esc(info.last_trained)}</p>

        <div class="grid g3">
          <div class="card metric">
            <div class="label">Type</div>
            <div class="value" style="font-size:1.2rem">${esc(info.mode || "-")}</div>
          </div>

          <div class="card metric">
            <div class="label">Training images</div>
            <div class="value">${esc(info.training_images ?? "-")}</div>
          </div>

          <div class="card metric">
            <div class="label">Duration</div>
            <div class="value">${esc(info.training_seconds ?? "-")} s</div>
          </div>
        </div>
      `
      : "";

  } catch (e) {}
}


// =====================================================================================
// ASYNCHRONOUS MODEL EVALUATION
// =====================================================================================

let evalPollTimer = null;


async function checkEvaluationStatus() {

  const b = $("#btnEval");

  try {

    const ev = await api("/evaluation-status");

    if (ev.status === "running") {

      b.disabled = true;
      b.textContent = "Evaluating...";

      $("#evalBox").innerHTML = `
        <div class="msg show info">
          Evaluation in progress... Please wait.
          The server is processing the test images.
        </div>
      `;

      return;
    }

    if (evalPollTimer !== null) {
      clearInterval(evalPollTimer);
      evalPollTimer = null;
    }

    b.disabled = false;

    b.textContent = "Evaluate model on the test set";

    if (ev.status === "done") {

      showMsg(
        "#evalMsg",
        "ok",
        "Evaluation completed successfully!"
      );

      await refreshDashboard();

    } else if (ev.status === "failed") {

      showMsg(
        "#evalMsg",
        "err",
        ev.message || "Evaluation failed."
      );

      $("#evalBox").innerHTML =
        `<div class="msg show err">${esc(ev.message || "Evaluation failed.")}</div>`;
    }

  } catch (e) {

    console.error("Evaluation status check failed:", e);
  }
}


function startEvaluationPolling() {

  if (evalPollTimer !== null) {
    clearInterval(evalPollTimer);
  }

  checkEvaluationStatus();

  evalPollTimer = setInterval(
    checkEvaluationStatus,
    3000
  );
}


$("#btnEval").addEventListener("click", async () => {

  const b = $("#btnEval");

  b.disabled = true;

  b.textContent = "Starting evaluation...";

  hideMsg("#evalMsg");

  try {

    await api("/evaluate", {
      method: "POST"
    });

    showMsg(
      "#evalMsg",
      "info",
      "Evaluation started. Please wait..."
    );

    startEvaluationPolling();

  } catch (e) {

    b.disabled = false;

    b.textContent = "Evaluate model on the test set";

    showMsg(
      "#evalMsg",
      "err",
      e.message
    );
  }
});


// =====================================================================================
// PREDICTION
// =====================================================================================

let oneFile = null;


function setOneFile(f) {

  if (!f) return;

  oneFile = f;

  $("#preview").src = URL.createObjectURL(f);

  $("#fileName").textContent = f.name;

  $("#predictArea").style.display = "block";

  $("#predResult").innerHTML = "";

  hideMsg("#predMsg");
}


function wireDrop(dropId, inputId, handler) {

  const drop = $(dropId);
  const input = $(inputId);

  input.addEventListener("change", () => {
    handler(input.files);
  });

  ["dragenter", "dragover"].forEach((ev) => {

    drop.addEventListener(ev, (e) => {

      e.preventDefault();

      drop.classList.add("over");
    });
  });

  ["dragleave", "drop"].forEach((ev) => {

    drop.addEventListener(ev, (e) => {

      e.preventDefault();

      drop.classList.remove("over");
    });
  });

  drop.addEventListener("drop", (e) => {
    handler(e.dataTransfer.files);
  });
}


wireDrop(
  "#dropOne",
  "#fileOne",
  (files) => setOneFile(files[0])
);


$("#btnPredict").addEventListener("click", async () => {

  if (!oneFile) return;

  const b = $("#btnPredict");

  b.disabled = true;

  hideMsg("#predMsg");

  const fd = new FormData();

  fd.append("file", oneFile, oneFile.name);

  const t0 = performance.now();

  try {

    const r = await api("/predict", {
      method: "POST",
      body: fd
    });

    const rt = Math.round(performance.now() - t0);

    const infected = r.label === "Parasitized";

    const bar = (c) => `
      <div>${c}: ${pct(r.probabilities[c])}</div>
      <div class="bar">
        <span style="width:${r.probabilities[c] * 100}%; background: var(--${c === "Parasitized" ? "para" : "unin"})"></span>
      </div>
    `;

    $("#predResult").innerHTML = `
      <div class="card result ${r.label}" style="margin-top:12px">

        <div class="big">
          ${infected ? "Parasitized (infected)" : "Uninfected (healthy)"}
        </div>

        <p class="muted">Confidence ${pct(r.confidence)}</p>

        ${bar("Parasitized")}
        ${bar("Uninfected")}

        <p class="muted">
          Model time: ${r.latency_ms} ms |
          Total response time: ${rt} ms
        </p>

      </div>
    `;

  } catch (e) {

    showMsg("#predMsg", "err", e.message);
  }

  b.disabled = false;
});


// =====================================================================================
// CHARTS
// =====================================================================================

let vizData = null;

const charts = {};


function palette() {

  const cs = getComputedStyle(document.documentElement);

  const v = (n) => cs.getPropertyValue(n).trim();

  return {
    text: v("--text2"),
    line: v("--line"),
    Parasitized: v("--para"),
    Uninfected: v("--unin")
  };
}


const valueLabels = {
  id: "valueLabels",

  afterDatasetsDraw(chart) {

    const { ctx } = chart;

    const p = palette();

    ctx.save();

    ctx.fillStyle = p.text;

    ctx.font = "12px system-ui";

    ctx.textAlign = "center";

    chart.data.datasets.forEach((ds, i) => {

      chart.getDatasetMeta(i).data.forEach((bar, j) => {

        ctx.fillText(
          ds.data[j],
          bar.x,
          bar.y - 4
        );
      });
    });

    ctx.restore();
  }
};


function baseOptions(p) {

  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,

    plugins: {

      legend: {
        labels: {
          color: p.text,
          usePointStyle: true,
          boxWidth: 8,

          filter: (item) =>
            !item.text.startsWith("Median")
        }
      },

      tooltip: {
        filter: (item) =>
          !item.dataset.label.startsWith("Median")
      }
    }
  };
}


function drawClasses(stats) {

  const p = palette();

  charts.classes && charts.classes.destroy();

  const keys = [
    ["train", "Train"],
    ["test", "Test"],
    ["uploads", "Pending uploads"]
  ];

  charts.classes = new Chart($("#chClasses"), {

    type: "bar",

    plugins: [valueLabels],

    data: {

      labels: keys.map((k) => k[1]),

      datasets: CLASSES.map((c) => ({

        label: c,

        backgroundColor: p[c],

        data: keys.map((k) => stats[k[0]][c]),

        borderRadius: 4,

        categoryPercentage: 0.7,

        barPercentage: 0.95
      }))
    },

    options: {

      ...baseOptions(p),

      layout: {
        padding: { top: 18 }
      },

      scales: {

        x: {
          grid: { display: false },
          ticks: { color: p.text },
          border: { color: p.line }
        },

        y: {
          beginAtZero: true,
          grid: { color: p.line },
          ticks: { color: p.text },
          border: { display: false },

          title: {
            display: true,
            text: "Number of images",
            color: p.text
          }
        }
      }
    }
  });
}


function median(a) {

  const s = [...a].sort((x, y) => x - y);

  const m = s.length >> 1;

  return s.length % 2
    ? s[m]
    : (s[m - 1] + s[m]) / 2;
}


function drawFeature(f, rows) {

  const p = palette();

  const vals = rows.map((r) => r[f.key]);

  const lo = Math.min(...vals);
  const hi = Math.max(...vals);

  const bins = 30;

  const w = (hi - lo) / bins || 1;

  const edges = (i) => lo + i * w;

  const datasets = [];

  let maxY = 0;

  CLASSES.forEach((c) => {

    const counts = new Array(bins).fill(0);

    const v = rows
      .filter((r) => r.class === c)
      .map((r) => r[f.key]);

    v.forEach((x) => {

      counts[
        Math.min(Math.floor((x - lo) / w), bins - 1)
      ]++;
    });

    maxY = Math.max(maxY, ...counts);

    const pts = [
      { x: edges(0), y: 0 },

      ...counts.map((n, i) => ({
        x: edges(i) + w / 2,
        y: n
      })),

      { x: edges(bins), y: 0 }
    ];

    datasets.push({

      label: c,

      data: pts,

      stepped: "middle",

      borderColor: p[c],

      backgroundColor: p[c] + "55",

      borderWidth: 2,

      fill: true,

      pointRadius: 0,

      pointHoverRadius: 4
    });
  });


  CLASSES.forEach((c) => {

    const v = rows
      .filter((r) => r.class === c)
      .map((r) => r[f.key]);

    if (v.length) {

      datasets.push({

        label: "Median " + c,

        data: [
          { x: median(v), y: 0 },
          { x: median(v), y: maxY * 1.05 }
        ],

        borderColor: p[c],

        borderDash: [6, 4],

        borderWidth: 1.5,

        pointRadius: 0,

        fill: false
      });
    }
  });


  charts[f.key] && charts[f.key].destroy();

  charts[f.key] = new Chart($("#ch_" + f.key), {

    type: "line",

    data: { datasets },

    options: {

      ...baseOptions(p),

      interaction: {
        mode: "index",
        intersect: false
      },

      scales: {

        x: {
          type: "linear",
          min: lo,
          max: hi,
          grid: { display: false },
          ticks: { color: p.text },
          border: { color: p.line },

          title: {
            display: true,
            text: f.xlabel,
            color: p.text
          }
        },

        y: {
          beginAtZero: true,
          grid: { color: p.line },
          ticks: { color: p.text },
          border: { display: false },

          title: {
            display: true,
            text: "Number of images",
            color: p.text
          }
        }
      }
    }
  });
}


function medianSentence(rows, f) {

  const m = (c) => {

    const v = rows
      .filter((r) => r.class === c)
      .map((r) => r[f.key]);

    return v.length ? median(v) : null;
  };

  const a = m("Parasitized");
  const b = m("Uninfected");

  if (a === null || b === null || b === 0) return "";

  const fmt = (x) =>
    Math.abs(x) >= 1000
      ? Math.round(x).toLocaleString("en")
      : x.toFixed(1);

  return `
    <strong>Median:</strong>
    Parasitized ${fmt(a)}${f.unit}
    vs Uninfected ${fmt(b)}${f.unit}
    (${((a - b) / b * 100) >= 0 ? "+" : ""}
    ${((a - b) / b * 100).toFixed(1)}% for infected cells).
    Dashed line = median.
  `;
}


function renderFeatureCards(rows) {

  $("#featureCards").innerHTML = FEATURES.map((f) => {

    const text = INTERPRETATIONS[f.key] || "";

    const placeholder =
      text.startsWith("Write your interpretation");

    return `
      <h2>${f.title}</h2>

      <div class="card">

        <div class="viz-row">

          <div>
            <div class="chart-box">
              <canvas
                id="ch_${f.key}"
                role="img"
                aria-label="${esc(f.title)}: distribution per class">
              </canvas>
            </div>
          </div>

          <div>

            <p>${medianSentence(rows, f)}</p>

            <h3>Interpretation</h3>

            <p class="interp ${placeholder ? "placeholder" : ""}">
              ${esc(text)}
            </p>

          </div>

        </div>

      </div>
    `;

  }).join("");

  FEATURES.forEach((f) => drawFeature(f, rows));
}


async function loadViz() {

  hideMsg("#vizMsg");

  try {

    const [stats, samples, feat] = await Promise.all([

      api("/dataset-stats"),

      api("/dataset-samples?n=5"),

      api("/dataset-features?n=" + $("#nPer").value)

    ]);

    vizData = {
      stats,
      rows: feat.rows
    };

    const t = (k) =>
      Object.values(stats[k]).reduce((a, b) => a + b, 0);

    $("#balanceText").innerHTML = `
      <strong>Train:</strong> ${t("train")} images<br>
      <strong>Test:</strong> ${t("test")} images<br>
      <strong>Pending uploads:</strong> ${t("uploads")}
    `;

    drawClasses(stats);

    $("#samples").innerHTML = CLASSES.flatMap((c) =>

      samples[c].map((u) => `
        <figure>
          <img
            src="${API + u}"
            alt="${c} cell"
            loading="lazy">
          <figcaption>${c}</figcaption>
        </figure>
      `)

    ).join("");


    if (!feat.rows.length) {

      showMsg(
        "#vizMsg",
        "warn",
        "No images found in data/train. Add the sample images to see the charts."
      );

      $("#featureCards").innerHTML = "";

      return;
    }

    renderFeatureCards(feat.rows);

  } catch (e) {

    showMsg(
      "#vizMsg",
      "err",
      "Cannot load the data: " + e.message
    );
  }
}


$("#nPer").addEventListener("change", loadViz);


matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {

  if (!vizData) return;

  drawClasses(vizData.stats);

  FEATURES.forEach((f) => {

    if ($("#ch_" + f.key)) {
      drawFeature(f, vizData.rows);
    }
  });
});


// =====================================================================================
// UPLOAD AND RETRAIN
// =====================================================================================

let manyFiles = [];


function setManyFiles(files) {

  manyFiles = [...files].filter(
    (f) => /\.(png|jpe?g)$/i.test(f.name)
  );

  $("#manyInfo").textContent = manyFiles.length
    ? `${manyFiles.length} image(s) selected.`
    : "No file selected.";

  $("#btnUpload").disabled = !manyFiles.length;
}


wireDrop(
  "#dropMany",
  "#fileMany",
  setManyFiles
);


$("#btnUpload").addEventListener("click", async () => {

  const label = $("#label").value;

  const bar = $("#upProgress");

  const b = $("#btnUpload");

  b.disabled = true;

  hideMsg("#upMsg");

  bar.style.display = "block";

  bar.value = 0;

  let saved = 0;
  let rejected = 0;

  try {

    for (let i = 0; i < manyFiles.length; i += 50) {

      const fd = new FormData();

      fd.append("label", label);

      manyFiles
        .slice(i, i + 50)
        .forEach((f) => fd.append("files", f, f.name));

      const r = await api("/upload", {
        method: "POST",
        body: fd
      });

      saved += r.saved;

      rejected += r.rejected;

      bar.value = Math.min(
        1,
        (i + 50) / manyFiles.length
      );
    }

    showMsg(
      "#upMsg",
      "ok",
      `${saved} images saved as ${label}` +
      (rejected
        ? `, ${rejected} rejected (not valid images).`
        : ".")
    );

    $("#fileMany").value = "";

    setManyFiles([]);

  } catch (e) {

    showMsg("#upMsg", "err", e.message);

    b.disabled = false;
  }

  bar.style.display = "none";

  refreshStatus();
});


function trainingHTML(t) {

  const kind =
    t.kind === "train" ? "Training" : "Retraining";

  if (t.status === "running") {

    return `
      <div class="msg show info">
        ${kind} in progress since ${esc(t.started_at)}.
        This block refreshes by itself.
      </div>
    `;
  }

  if (t.status === "failed") {

    return `
      <div class="msg show err">
        ${kind} failed: ${esc(t.message)}
      </div>
    `;
  }

  if (t.status === "done") {

    const r = t.result || {};

    return `
      <div class="msg show ok">
        ${kind} finished at ${esc(t.finished_at)}.
        The new model is now serving predictions.
      </div>

      <p class="muted">
        New images used: ${r.new_images_used ?? 0}
        | Epochs: ${r.epochs ?? "-"}
        | Duration: ${r.training_seconds ?? "-"} s
      </p>

      ${metricsHTML(r.test_metrics)}
    `;
  }

  return `
    <p class="muted">
      No training job has run since the API started.
    </p>
  `;
}


let lastTrainHTML = "";


async function refreshStatus() {

  try {

    const st = await api("/status");

    $("#mPending").textContent = st.pending_uploads;

    $("#mThreshold").textContent = st.retrain_threshold;

    const tm = $("#triggerMsg");

    tm.className =
      "msg show " + (st.retrain_needed ? "warn" : "");

    tm.textContent = st.retrain_needed
      ? "Enough new images have been uploaded: retraining is recommended."
      : "Retraining is recommended once the number of waiting images reaches the threshold.";

    if (!st.retrain_needed) {

      tm.style.cssText =
        "background:none;color:var(--text2);padding:0";

    } else {

      tm.style.cssText = "";
    }

    const running =
      st.training.status === "running";

    $("#btnRetrain").disabled =
      st.pending_uploads === 0 || running;

    $("#btnTrain").disabled =
      !$("#sure").checked || running;

    const html = trainingHTML(st.training);

    if (html !== lastTrainHTML) {

      $("#trainBox").innerHTML = html;

      lastTrainHTML = html;
    }

  } catch (e) {

    $("#trainBox").innerHTML =
      '<div class="msg show warn">API not reachable.</div>';

    lastTrainHTML = "";
  }
}


$("#sure").addEventListener(
  "change",
  refreshStatus
);


$("#btnRetrain").addEventListener("click", async () => {

  hideMsg("#retrainMsg");

  try {

    await api("/retrain", {
      method: "POST"
    });

    showMsg(
      "#retrainMsg",
      "ok",
      "Retraining started."
    );

  } catch (e) {

    showMsg("#retrainMsg", "err", e.message);
  }

  refreshStatus();
});


$("#btnTrain").addEventListener("click", async () => {

  hideMsg("#retrainMsg");

  try {

    await api("/train", {
      method: "POST"
    });

    showMsg(
      "#retrainMsg",
      "ok",
      "Training started."
    );

  } catch (e) {

    showMsg("#retrainMsg", "err", e.message);
  }

  refreshStatus();
});


// =====================================================================================
// AUTOMATIC REFRESH
// =====================================================================================

refreshDashboard();

setInterval(() => {

  refreshHealth();

  if (currentTab === "data") {
    refreshStatus();
  }

}, 3000);
