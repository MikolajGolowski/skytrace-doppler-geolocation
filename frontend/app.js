const state = {
  spec: null,
  tracedPoints: [],
  idealCurve1: null,
  idealCurve2: null,
  groundTruth: null,
  report: null,
  map: null,
  mapLayers: null,
};

const $ = (id) => document.getElementById(id);

function setStatus(msg, isError = false) {
  const el = $("statusText");
  el.textContent = msg;
  el.style.color = isError ? "#fb7185" : "#94a3b8";
}

function infernoColor(v) {
  // v in [0, 255] -> smooth dark-navy -> purple -> orange -> bright yellow-white
  const t = v / 255;
  let r, g, b;
  if (t < 0.25) {
    const k = t / 0.25;
    r = Math.round(8 + 58 * k);
    g = Math.round(12 + 12 * k);
    b = Math.round(28 + 85 * k);
  } else if (t < 0.55) {
    const k = (t - 0.25) / 0.3;
    r = Math.round(66 + 135 * k);
    g = Math.round(24 + 35 * k);
    b = Math.round(113 - 25 * k);
  } else if (t < 0.85) {
    const k = (t - 0.55) / 0.3;
    r = Math.round(201 + 50 * k);
    g = Math.round(59 + 130 * k);
    b = Math.round(88 - 65 * k);
  } else {
    const k = (t - 0.85) / 0.15;
    r = Math.round(251 + 4 * k);
    g = Math.round(189 + 64 * k);
    b = Math.round(23 + 190 * k);
  }
  return [r, g, b];
}

function drawSpectrogramCanvas(canvas, specObj, overlays = []) {
  if (!specObj || !specObj.width || !specObj.height) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  const padL = 44, padB = 26, padT = 10, padR = 12;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  ctx.fillStyle = "#05070d";
  ctx.fillRect(0, 0, W, H);

  // Decode base64 uint8 spectrogram into offscreen canvas
  const raw = atob(specObj.data_b64);
  const off = document.createElement("canvas");
  off.width = specObj.width;
  off.height = specObj.height;
  const octx = off.getContext("2d");
  const img = octx.createImageData(specObj.width, specObj.height);
  const d = img.data;
  for (let i = 0; i < raw.length; i++) {
    const [r, g, b] = infernoColor(raw.charCodeAt(i));
    const j = i * 4;
    d[j] = r;
    d[j + 1] = g;
    d[j + 2] = b;
    d[j + 3] = 255;
  }
  octx.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(off, padL, padT, plotW, plotH);

  // Coordinate helpers
  const xOfT = (t) => padL + ((t - specObj.t_min) / Math.max(specObj.t_max - specObj.t_min, 1e-6)) * plotW;
  const yOfF = (f) => padT + (1 - (f - specObj.f_min) / Math.max(specObj.f_max - specObj.f_min, 1e-6)) * plotH;

  // Grid & axes
  ctx.strokeStyle = "rgba(148,163,184,0.18)";
  ctx.fillStyle = "#94a3b8";
  ctx.font = "11px Inter, sans-serif";
  ctx.lineWidth = 1;

  const fSteps = 5;
  for (let i = 0; i <= fSteps; i++) {
    const f = specObj.f_min + ((specObj.f_max - specObj.f_min) * i) / fSteps;
    const y = yOfF(f);
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + plotW, y);
    ctx.stroke();
    ctx.fillText(Math.round(f) + "Hz", 4, y + 4);
  }

  const tSteps = 6;
  for (let i = 0; i <= tSteps; i++) {
    const t = specObj.t_min + ((specObj.t_max - specObj.t_min) * i) / tSteps;
    const x = xOfT(t);
    ctx.beginPath();
    ctx.moveTo(x, padT);
    ctx.lineTo(x, padT + plotH);
    ctx.stroke();
    ctx.fillText(t.toFixed(0) + "s", x - 10, H - 7);
  }

  // Overlays (curves & traced points)
  ctx.save();
  ctx.beginPath();
  ctx.rect(padL, padT, plotW, plotH);
  ctx.clip();

  for (const ov of overlays) {
    if (ov.type === "points") {
      ctx.fillStyle = ov.color || "#34d399";
      for (const [t, f] of ov.pts) {
        ctx.beginPath();
        ctx.arc(xOfT(t), yOfF(f), 4, 0, Math.PI * 2);
        ctx.fill();
      }
      continue;
    }
    if (!ov.t || !ov.f || !ov.t.length) continue;
    ctx.strokeStyle = ov.color || "#22d3ee";
    ctx.lineWidth = ov.width || 2;
    ctx.setLineDash(ov.dash || []);
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < ov.t.length; i++) {
      if (ov.f[i] == null) continue;
      const x = xOfT(ov.t[i]);
      const y = yOfF(ov.f[i]);
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
  }
  ctx.restore();
  return { padL, padT, plotW, plotH };
}

function redrawMainSpec() {
  if (!state.spec) return;
  const overlays = [];
  const showH = $("showHarmonic").checked;
  if (state.idealCurve1) {
    overlays.push({ t: state.idealCurve1.t, f: state.idealCurve1.f, color: "#22d3ee", width: 2 });
    if (showH) {
      overlays.push({ t: state.idealCurve1.t, f: state.idealCurve1.f.map((v) => v * 2), color: "rgba(34,211,238,0.45)", width: 1.5, dash: [4, 4] });
    }
  }
  if (state.idealCurve2 && $("locMode").value === "two_event") {
    overlays.push({ t: state.idealCurve2.t, f: state.idealCurve2.f, color: "#f59e0b", width: 2 });
    if (showH) {
      overlays.push({ t: state.idealCurve2.t, f: state.idealCurve2.f.map((v) => v * 2), color: "rgba(245,158,11,0.45)", width: 1.5, dash: [4, 4] });
    }
  }
  if (state.tracedPoints.length) {
    overlays.push({ type: "points", pts: state.tracedPoints, color: "#34d399" });
  }
  drawSpectrogramCanvas($("specCanvas"), state.spec, overlays);
}

async function updateIdealPreview(which) {
  const prefix = which === 1 ? "e1" : "e2";
  const tcpa = parseFloat($(`${prefix}_tcpa`).value);
  const f0 = parseFloat($(`${prefix}_f0`).value);
  const v = parseFloat($(`${prefix}_v_slider`).value);
  const d = parseFloat($(`${prefix}_d_slider`).value);
  $(`${prefix}_v_val`).textContent = v;
  $(`${prefix}_d_val`).textContent = d;

  const res = await fetch("/api/events/ideal_curve", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ t_cpa_s: tcpa, f0_hz: f0, speed_ms: v, d_cpa_m: d }),
  });
  const data = await res.json();
  if (which === 1) state.idealCurve1 = data;
  else state.idealCurve2 = data;
  redrawMainSpec();
}

function populateEventInputs(events) {
  if (events[0]) {
    const e = events[0];
    $("e1_tcpa").value = e.t_cpa_s;
    $("e1_f0").value = e.f0_hz;
    $("e1_vmin").value = Math.round(e.speed_min_ms);
    $("e1_vmax").value = Math.round(e.speed_max_ms);
    $("e1_dmin").value = Math.round(e.d_min_m);
    $("e1_dmax").value = Math.round(e.d_max_m);
    $("e1_ttol").value = e.t_tolerance_s;
    $("e1_ftol").value = e.f0_tolerance_hz;
    $("e1_v_slider").value = Math.round(e.speed_ms);
    $("e1_d_slider").value = Math.round(e.d_cpa_m);
  }
  if (events[1]) {
    const e = events[1];
    $("e2_tcpa").value = e.t_cpa_s;
    $("e2_f0").value = e.f0_hz;
    $("e2_vmin").value = Math.round(e.speed_min_ms);
    $("e2_vmax").value = Math.round(e.speed_max_ms);
    $("e2_dmin").value = Math.round(e.d_min_m);
    $("e2_dmax").value = Math.round(e.d_max_m);
    $("e2_ttol").value = e.t_tolerance_s;
    $("e2_ftol").value = e.f0_tolerance_hz;
    $("e2_v_slider").value = Math.round(e.speed_ms);
    $("e2_d_slider").value = Math.round(e.d_cpa_m);
  }
}

async function loadDemo() {
  const preset = $("demoPreset").value;
  setStatus(`Generating & analyzing scenario "${preset}"...`);
  $("btnLoadDemo").disabled = true;
  try {
    const res = await fetch("/api/demo/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ preset }),
    });
    const data = await res.json();
    state.spec = data.spectrogram;
    state.groundTruth = data.ground_truth;
    state.tracedPoints = [];
    $("traceCount").textContent = "0";
    $("btnBoostAudio").disabled = false;

    $("locMode").value = data.mode;
    syncModeUI();
    if (data.known_utc_cpa1) {
      $("knownUtcInput").value = Math.round(data.known_utc_cpa1);
    }
    populateEventInputs(data.detected_events || []);
    await updateIdealPreview(1);
    if (data.mode === "two_event") await updateIdealPreview(2);

    $("metaBadges").innerHTML = `
      <span class="pill pill-cyan">${data.duration_s}s audio</span>
      <span class="pill pill-amber">${data.track_count} ADS-B tracks</span>
    `;
    setStatus(`Loaded ${data.name}. Detected ${data.detected_events.length} Doppler event(s). Switch to tab 2 to geolocate!`);
  } catch (err) {
    setStatus("Error loading demo: " + err.message, true);
  } finally {
    $("btnLoadDemo").disabled = false;
  }
}

function syncModeUI() {
  const isTwo = $("locMode").value === "two_event";
  $("event2Card").classList.toggle("hidden", !isTwo);
  $("utcTimeWrap").classList.toggle("hidden", isTwo);
  $("overlayBox2").classList.toggle("hidden", !isTwo);
  $("scatterTitle").textContent = isTwo
    ? "Flight-Pair Peak Scores (Event 1 vs Event 2)"
    : "Candidate Flight Score Distribution";
}

function drawScatter(report) {
  const canvas = $("scatterCanvas");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  const padL = 48, padR = 20, padT = 20, padB = 38;
  const pw = W - padL - padR, ph = H - padT - padB;

  ctx.fillStyle = "#070b14";
  ctx.fillRect(0, 0, W, H);

  const peaks = report.pair_peaks || [];
  if (!peaks.length) {
    ctx.fillStyle = "#94a3b8";
    ctx.font = "13px Inter, sans-serif";
    ctx.fillText("No surviving candidate flight pairs.", 140, H / 2);
    return;
  }

  if (report.mode === "two_event") {
    const all = report.all_scored_sample || [];
    const max1 = Math.max(...peaks.map((p) => p.score1), 1.0) * 1.1;
    const max2 = Math.max(...peaks.map((p) => p.score2), 1.0) * 1.1;
    const xOf = (v) => padL + (v / max1) * pw;
    const yOf = (v) => padT + (1 - v / max2) * ph;

    ctx.strokeStyle = "rgba(148,163,184,0.16)";
    ctx.strokeRect(padL, padT, pw, ph);

    // Individual points (dim)
    ctx.fillStyle = "rgba(148,163,184,0.28)";
    for (const r of all) {
      ctx.beginPath();
      ctx.arc(xOf(r.score1), yOf(r.score2), 2.5, 0, Math.PI * 2);
      ctx.fill();
    }

    // Flight pair peaks
    peaks.forEach((p, idx) => {
      const x = xOf(p.score1);
      const y = yOf(p.score2);
      ctx.fillStyle = idx === 0 ? "#34d399" : idx < 3 ? "#f59e0b" : "#22d3ee";
      ctx.beginPath();
      ctx.arc(x, y, idx === 0 ? 7 : 5, 0, Math.PI * 2);
      ctx.fill();
      if (idx < 3) {
        ctx.fillStyle = "#f8fafc";
        ctx.font = "11px Inter, sans-serif";
        ctx.fillText(`#${idx + 1} ${p.flight1_callsign}+${p.flight2_callsign}`, Math.min(x + 9, W - 150), y + 4);
      }
    });

    ctx.fillStyle = "#94a3b8";
    ctx.font = "11px Inter, sans-serif";
    ctx.fillText("Event 1 Score →", padL + pw / 2 - 40, H - 10);
    ctx.fillText("Ev 2", 6, padT + 14);
  } else {
    // Bar chart of top flight peaks for single-event mode
    const top = peaks.slice(0, 10);
    const maxS = Math.max(...top.map((p) => p.total_score), 1.0);
    const barH = Math.min(24, ph / top.length - 6);
    top.forEach((p, i) => {
      const y = padT + i * (barH + 6);
      const bw = (p.total_score / maxS) * (pw - 90);
      ctx.fillStyle = i === 0 ? "#34d399" : "#22d3ee";
      ctx.fillRect(padL + 60, y, bw, barH);
      ctx.fillStyle = "#e2e8f0";
      ctx.font = "11px Inter, sans-serif";
      ctx.fillText(`${p.flight1_callsign}`, 10, y + barH * 0.7);
      ctx.fillText(`${p.total_score.toFixed(1)}`, padL + 66 + bw, y + barH * 0.7);
    });
  }
}

function renderMap(report) {
  const mapEl = $("mapContainer");
  const fbCanvas = $("fallbackMapCanvas");
  const top = report.top_candidates || [];
  if (!top.length) return;

  if (typeof L !== "undefined") {
    mapEl.classList.remove("hidden");
    fbCanvas.classList.add("hidden");
    if (!state.map) {
      state.map = L.map("mapContainer").setView([top[0].lat, top[0].lon], 13);
      L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
        attribution: "&copy; OpenStreetMap &copy; CARTO",
        maxZoom: 19,
      }).addTo(state.map);
      state.mapLayers = L.layerGroup().addTo(state.map);
    }
    state.mapLayers.clearLayers();
    state.map.setView([top[0].lat, top[0].lon], 13);

    // Flight tracks
    for (const fp of report.flight_paths || []) {
      const isWinner = fp.icao24 === top[0].flight1_id || fp.icao24 === top[0].flight2_id;
      L.polyline(fp.coords, {
        color: isWinner ? "#22d3ee" : "#475569",
        weight: isWinner ? 3 : 1.3,
        opacity: isWinner ? 0.9 : 0.45,
      })
        .bindPopup(`${fp.callsign} (${fp.type_code})`)
        .addTo(state.mapLayers);
    }

    // Fine 25m heatmap
    const hm = report.fine_heatmap || [];
    if (hm.length) {
      const scores = hm.map((h) => h.score);
      const minS = Math.min(...scores);
      const maxS = Math.max(...scores, minS + 1e-4);
      for (const h of hm) {
        const norm = (h.score - minS) / (maxS - minS);
        if (norm < 0.35) continue;
        L.circleMarker([h.lat, h.lon], {
          radius: 4,
          stroke: false,
          fillColor: norm > 0.85 ? "#34d399" : norm > 0.65 ? "#f59e0b" : "#06b6d4",
          fillOpacity: 0.25 + 0.55 * norm,
        }).addTo(state.mapLayers);
      }
    }

    // Top 3 markers
    top.slice(0, 3).forEach((c, idx) => {
      L.circleMarker([c.lat, c.lon], {
        radius: idx === 0 ? 8 : 6,
        color: "#fff",
        weight: 2,
        fillColor: idx === 0 ? "#34d399" : "#f59e0b",
        fillOpacity: 1,
      })
        .bindPopup(`<b>Rank #${idx + 1}</b><br>${c.lat.toFixed(5)}, ${c.lon.toFixed(5)}<br>Score: ${c.total_score}`)
        .addTo(state.mapLayers);
    });

    if (report.ground_truth_eval) {
      const gt = report.ground_truth_eval;
      L.circleMarker([gt.true_lat, gt.true_lon], {
        radius: 6,
        color: "#fb7185",
        weight: 2,
        fillColor: "#fb7185",
        fillOpacity: 0.5,
      })
        .bindPopup(`<b>Ground Truth</b><br>Error: ${gt.error_m} m`)
        .addTo(state.mapLayers);
    }
    setTimeout(() => state.map.invalidateSize(), 120);
  } else {
    mapEl.classList.add("hidden");
    fbCanvas.classList.remove("hidden");
  }
}

function renderOverlayInspector() {
  const rep = state.report;
  if (!rep || !rep.curve_overlays || !rep.curve_overlays.length) return;
  const idx = parseInt($("overlayRankSelect").value, 10) || 0;
  const ov = rep.curve_overlays[idx] || rep.curve_overlays[0];
  const showNaive = $("showNaiveToggle").checked;

  if (rep.spec_slices?.event1 && ov.event1) {
    const lines = [{ t: ov.event1.t_audio, f: ov.event1.f_retarded, color: "#22d3ee", width: 2.2 }];
    if (showNaive && ov.event1.f_naive) {
      lines.push({ t: ov.event1.t_audio, f: ov.event1.f_naive, color: "#fb7185", width: 1.8, dash: [5, 4] });
    }
    drawSpectrogramCanvas($("overlayCanvas1"), rep.spec_slices.event1, lines);
  }
  if (rep.spec_slices?.event2 && ov.event2) {
    const lines2 = [{ t: ov.event2.t_audio, f: ov.event2.f_retarded, color: "#f59e0b", width: 2.2 }];
    if (showNaive && ov.event2.f_naive) {
      lines2.push({ t: ov.event2.t_audio, f: ov.event2.f_naive, color: "#fb7185", width: 1.8, dash: [5, 4] });
    }
    drawSpectrogramCanvas($("overlayCanvas2"), rep.spec_slices.event2, lines2);
  }
}

async function runLocate() {
  if (!state.spec) {
    setStatus("Load a demo or upload audio + ADS-B first.", true);
    return;
  }
  const mode = $("locMode").value;
  const payload = {
    mode,
    aircraft_filter_mode: $("icaoFilter").value,
    dem_mode: $("demMode").value,
    enable_silence_filter: $("silenceEnabled").checked,
    known_utc_cpa1: mode === "single_event" && $("knownUtcInput").value ? parseFloat($("knownUtcInput").value) : null,
    event1: {
      t_cpa_s: parseFloat($("e1_tcpa").value),
      f0_hz: parseFloat($("e1_f0").value),
      speed_min_ms: parseFloat($("e1_vmin").value),
      speed_max_ms: parseFloat($("e1_vmax").value),
      d_min_m: parseFloat($("e1_dmin").value),
      d_max_m: parseFloat($("e1_dmax").value),
      t_tolerance_s: parseFloat($("e1_ttol").value),
      f0_tolerance_hz: parseFloat($("e1_ftol").value),
      traced_points: state.tracedPoints,
    },
    event2:
      mode === "two_event"
        ? {
            t_cpa_s: parseFloat($("e2_tcpa").value),
            f0_hz: parseFloat($("e2_f0").value),
            speed_min_ms: parseFloat($("e2_vmin").value),
            speed_max_ms: parseFloat($("e2_vmax").value),
            d_min_m: parseFloat($("e2_dmin").value),
            d_max_m: parseFloat($("e2_dmax").value),
            t_tolerance_s: parseFloat($("e2_ttol").value),
            f0_tolerance_hz: parseFloat($("e2_ftol").value),
          }
        : null,
  };

  setStatus("Running multi-stage 200m -> 100m -> 25m retarded-time geolocation...");
  $("btnRunLocate").disabled = true;
  try {
    const res = await fetch("/api/locate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const rep = await res.json();
    if (!res.ok) throw new Error(rep.detail || "Geolocation failed");
    state.report = rep;

    const fn = rep.funnel || {};
    $("fn_total").textContent = fn.total_tracks ?? "-";
    $("fn_nonjet").textContent = fn.non_jet_tracks ?? "-";
    $("fn_grid").textContent = fn.grid_200m_event1 ?? "-";
    $("fn_offset").textContent = fn.after_time_offset ?? "-";
    $("fn_silence").textContent = fn.after_silence_filter ?? "-";
    $("fn_scored").textContent = fn.refined_100m_scored ?? "-";

    const tbody = $("resultsTable").querySelector("tbody");
    tbody.innerHTML = "";
    (rep.top_candidates || []).forEach((c, idx) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>#${idx + 1}</td>
        <td>${c.lat.toFixed(5)}, ${c.lon.toFixed(5)}</td>
        <td>${c.elevation_m.toFixed(0)}m</td>
        <td>${c.flight1_callsign} (${c.flight1_type})</td>
        <td>${c.flight2_callsign ? `${c.flight2_callsign} (${c.flight2_type})` : "—"}</td>
        <td>${c.score1.toFixed(2)}</td>
        <td>${c.score2.toFixed(2)}</td>
        <td><strong>${c.total_score.toFixed(2)}</strong></td>
      `;
      tbody.appendChild(tr);
    });

    if (rep.top_candidates?.length) {
      const top = rep.top_candidates[0];
      const link = $("gmapsLink");
      link.href = `https://www.google.com/maps/@?api=1&map_action=pano&viewpoint=${top.lat},${top.lon}`;
      link.classList.remove("hidden");
    }

    const gtBadge = $("gtEvalBadge");
    if (rep.ground_truth_eval) {
      gtBadge.textContent = `Ground Truth Error: ${rep.ground_truth_eval.error_m} m`;
      gtBadge.classList.remove("hidden");
    } else {
      gtBadge.classList.add("hidden");
    }

    drawScatter(rep);
    renderMap(rep);
    renderOverlayInspector();
    setStatus("Geolocation complete.");
  } catch (err) {
    setStatus("Geolocation error: " + err.message, true);
  } finally {
    $("btnRunLocate").disabled = false;
  }
}

// Event Wiring
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $(btn.dataset.tab).classList.add("active");
    if (btn.dataset.tab === "tab-locate" && state.map) {
      setTimeout(() => state.map.invalidateSize(), 120);
    }
  });
});

$("btnLoadDemo").addEventListener("click", loadDemo);
$("locMode").addEventListener("change", syncModeUI);
$("btnRunLocate").addEventListener("click", runLocate);
$("overlayRankSelect").addEventListener("change", renderOverlayInspector);
$("showNaiveToggle").addEventListener("change", renderOverlayInspector);
$("showHarmonic").addEventListener("change", redrawMainSpec);

$("e1_v_slider").addEventListener("input", () => updateIdealPreview(1));
$("e1_d_slider").addEventListener("input", () => updateIdealPreview(1));
$("e2_v_slider").addEventListener("input", () => updateIdealPreview(2));
$("e2_d_slider").addEventListener("input", () => updateIdealPreview(2));

$("btnClearTrace").addEventListener("click", () => {
  state.tracedPoints = [];
  $("traceCount").textContent = "0";
  redrawMainSpec();
});

$("specCanvas").addEventListener("click", async (ev) => {
  if (!state.spec) return;
  const rect = ev.target.getBoundingClientRect();
  const scaleX = ev.target.width / rect.width;
  const scaleY = ev.target.height / rect.height;
  const cx = (ev.clientX - rect.left) * scaleX;
  const cy = (ev.clientY - rect.top) * scaleY;
  const padL = 44, padR = 12, padT = 10, padB = 26;
  const plotW = ev.target.width - padL - padR;
  const plotH = ev.target.height - padT - padB;
  if (cx < padL || cx > padL + plotW || cy < padT || cy > padT + plotH) return;

  const t = state.spec.t_min + ((cx - padL) / plotW) * (state.spec.t_max - state.spec.t_min);
  const f = state.spec.f_max - ((cy - padT) / plotH) * (state.spec.f_max - state.spec.f_min);
  const mode = document.querySelector('input[name="clickTarget"]:checked').value;

  if (mode === "trace") {
    state.tracedPoints.push([Number(t.toFixed(2)), Number(f.toFixed(2))]);
    $("traceCount").textContent = String(state.tracedPoints.length);
    redrawMainSpec();
    return;
  }

  setStatus(`Fitting ideal Doppler curve around t=${t.toFixed(1)}s...`);
  const res = await fetch("/api/events/fit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ t_center_s: t }),
  });
  const data = await res.json();
  if (data.event) {
    if (mode === "ev1") {
      populateEventInputs([data.event, null]);
      await updateIdealPreview(1);
    } else {
      populateEventInputs([null, data.event]);
      await updateIdealPreview(2);
    }
    setStatus(`Fitted ${mode.toUpperCase()} at t=${data.event.t_cpa_s}s, f₀=${data.event.f0_hz}Hz, v=${data.event.speed_ms}m/s`);
  }
});

$("btnRecomputeSpec").addEventListener("click", async () => {
  if (!state.spec) return;
  const res = await fetch("/api/spectrogram/recompute", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      fmin: parseFloat($("specFmin").value),
      fmax: parseFloat($("specFmax").value),
      win_sec: parseFloat($("specWin").value),
      use_whitened: $("specWhiten").checked,
    }),
  });
  const data = await res.json();
  state.spec = data.spectrogram;
  redrawMainSpec();
});

$("btnBoostAudio").addEventListener("click", () => {
  const player = $("boostAudioPlayer");
  player.src = `/api/audio/boosted?t=${Date.now()}`;
  player.classList.remove("hidden");
  player.play();
});

$("audioUpload").addEventListener("change", async (ev) => {
  const file = ev.target.files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append("file", file);
  setStatus(`Uploading and computing spectrogram for ${file.name}...`);
  const res = await fetch("/api/audio/upload", { method: "POST", body: fd });
  const data = await res.json();
  if (!res.ok) return setStatus(data.detail || "Upload error", true);
  state.spec = data.spectrogram;
  state.groundTruth = null;
  $("btnBoostAudio").disabled = false;
  populateEventInputs(data.detected_events || []);
  await updateIdealPreview(1);
  if ((data.detected_events || []).length >= 2) await updateIdealPreview(2);
  setStatus(`Loaded ${data.filename} (${data.duration_s}s). Detected ${data.detected_events.length} Doppler event(s).`);
});

$("adsbUpload").addEventListener("change", async (ev) => {
  const file = ev.target.files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append("file", file);
  setStatus(`Loading ADS-B tracks from ${file.name}...`);
  const res = await fetch("/api/adsb/upload", { method: "POST", body: fd });
  const data = await res.json();
  if (!res.ok) return setStatus(data.detail || "ADS-B load error", true);
  setStatus(`Loaded ${data.track_count} ADS-B track segments.`);
});

$("btnRunCm").addEventListener("click", async () => {
  if (!state.spec) return setStatus("Load audio or demo first.", true);
  setStatus("Running acoustic countermeasure simulation...");
  const res = await fetch("/api/countermeasures/test", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      t_cpa_s: parseFloat($("e1_tcpa").value),
      f0_hz: parseFloat($("e1_f0").value),
      speed_ms: parseFloat($("e1_v_slider").value),
      d_cpa_m: parseFloat($("e1_d_slider").value),
      method: $("cmMethod").value,
      param: parseFloat($("cmParam").value),
    }),
  });
  const data = await res.json();
  drawSpectrogramCanvas($("cmCanvas"), data.spectrogram, []);
  $("cmMetrics").innerHTML = `
    <div class="metric-box"><span>Original Curve SNR</span><strong>${data.original_curve_snr}</strong></div>
    <div class="metric-box"><span>Post-Attack Curve SNR</span><strong>${data.processed_curve_snr}</strong></div>
    <div class="metric-box"><span>Doppler Curve Retention</span><strong>${data.curve_retention_pct}%</strong></div>
    <div class="metric-box"><span>Speech Band Preserved</span><strong>${data.speech_preserved_pct}%</strong></div>
    <div class="metric-box" style="grid-column: span 2;">
      <span>Geolocation Verdict</span>
      <strong style="color:${data.curve_still_detectable ? "#fb7185" : "#34d399"}">
        ${data.curve_still_detectable ? "VULNERABLE — Doppler signature recovered after whitening!" : "MITIGATED — Signature disrupted"}
      </strong>
    </div>
  `;
  setStatus("Countermeasure evaluation complete.");
});

// Auto-load quick demo on initial open for immediate showcase
window.addEventListener("DOMContentLoaded", loadDemo);
