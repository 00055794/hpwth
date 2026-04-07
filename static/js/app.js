/* ═══════════════════════════════════════════════════════════
   KZ Real Estate Price Estimator  –  app.js
   ═══════════════════════════════════════════════════════════ */

"use strict";

/* ── Constants ────────────────────────────────────────────── */
const DEFAULT_LAT = 43.2567;
const DEFAULT_LON = 76.9286;

const DIST_META = {
  dist_to_school_km:       { icon: "🏫", label: "School" },
  dist_to_kindergarten_km: { icon: "🎒", label: "Kindergarten" },
  dist_to_hospital_km:     { icon: "🏥", label: "Hospital" },
  dist_to_healthcare_km:   { icon: "⚕️", label: "Healthcare centre" },
  dist_to_pharmacy_km:     { icon: "💊", label: "Pharmacy" },
  dist_to_main_road_km:    { icon: "🛣️", label: "Main road" },
};

/* ── State ────────────────────────────────────────────────── */
let ymapObj  = null;   // ymaps.Map instance
let placemark = null;  // ymaps.Placemark instance

/* ── Yandex Maps init ─────────────────────────────────────── */
ymaps.ready(function () {
  const mapDiv = document.getElementById("ymap");
  if (!mapDiv) return;

  ymapObj = new ymaps.Map("ymap", {
    center: [DEFAULT_LAT, DEFAULT_LON],
    zoom:   11,
    controls: ["zoomControl", "typeSelector"],
  });

  // Add search control
  try {
    const searchCtrl = new ymaps.control.SearchControl({
      options: { float: "right", floatIndex: 100, noPlacemark: true, size: "small" },
    });
    ymapObj.controls.add(searchCtrl);
    searchCtrl.events.add("resultselect", function (e) {
      const idx    = searchCtrl.getSelectedIndex();
      const result = searchCtrl.getResultsArray()[idx];
      const coords = result.geometry.getCoordinates();
      setLocation(coords[0], coords[1], true);
    });
  } catch (_) { /* search unavailable */ }

  // Initial placemark
  placemark = new ymaps.Placemark(
    [DEFAULT_LAT, DEFAULT_LON],
    { balloonContent: fmtCoords(DEFAULT_LAT, DEFAULT_LON) },
    { preset: "islands#redCircleDotIcon", draggable: true }
  );
  ymapObj.geoObjects.add(placemark);

  // Drag placemark → update inputs
  placemark.events.add("dragend", function () {
    const c = placemark.geometry.getCoordinates();
    setLocation(c[0], c[1], false);
  });

  // Click map → move placemark + update inputs
  ymapObj.events.add("click", function (e) {
    const coords = e.get("coords");
    setLocation(coords[0], coords[1], true);
  });
});

/* ── Location helpers ─────────────────────────────────────── */
function setLocation(lat, lon, movePlacemark) {
  document.getElementById("LATITUDE").value  = lat.toFixed(6);
  document.getElementById("LONGITUDE").value = lon.toFixed(6);

  if (ymapObj && placemark) {
    if (movePlacemark) {
      placemark.geometry.setCoordinates([lat, lon]);
    }
    placemark.properties.set("balloonContent", fmtCoords(lat, lon));
  }
}

function onCoordsInput() {
  const lat = parseFloat(document.getElementById("LATITUDE").value);
  const lon = parseFloat(document.getElementById("LONGITUDE").value);
  if (!isNaN(lat) && !isNaN(lon) && ymapObj && placemark) {
    placemark.geometry.setCoordinates([lat, lon]);
    ymapObj.setCenter([lat, lon]);
  }
}

function fmtCoords(lat, lon) {
  return lat.toFixed(6) + ", " + lon.toFixed(6);
}

/* ── Predict ──────────────────────────────────────────────── */
async function runPredict() {
  // Collect inputs
  const payload = {
    ROOMS:        parseInt(document.getElementById("ROOMS").value,       10),
    LONGITUDE:    parseFloat(document.getElementById("LONGITUDE").value),
    LATITUDE:     parseFloat(document.getElementById("LATITUDE").value),
    TOTAL_AREA:   parseFloat(document.getElementById("TOTAL_AREA").value),
    FLOOR:        parseInt(document.getElementById("FLOOR").value,       10),
    TOTAL_FLOORS: parseInt(document.getElementById("TOTAL_FLOORS").value,10),
    FURNITURE:    parseInt(document.getElementById("FURNITURE").value,   10),
    CONDITION:    parseInt(document.getElementById("CONDITION").value,   10),
    CEILING:      parseFloat(document.getElementById("CEILING").value),
    MATERIAL:     parseInt(document.getElementById("MATERIAL").value,    10),
    YEAR:         parseInt(document.getElementById("YEAR").value,        10),
  };

  // Basic validation
  if (isNaN(payload.LATITUDE) || isNaN(payload.LONGITUDE)) {
    showError("Please enter valid coordinates or click on the map.");
    return;
  }

  setUIState("loading");

  try {
    const resp = await fetch("/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || "Server error");
    }

    const data = await resp.json();
    renderResult(data);
    setUIState("result");
  } catch (err) {
    showError(err.message);
    setUIState("error");
  }
}

/* ── UI state machine ─────────────────────────────────────── */
function setUIState(state) {
  const btn = document.getElementById("predict-btn");
  hide("placeholder-card");
  hide("loading-card");
  hide("result-section");
  hide("error-card");

  if (state === "loading") {
    btn.disabled    = true;
    btn.textContent = "Calculating …";
    show("loading-card");
  } else if (state === "result") {
    btn.disabled    = false;
    btn.textContent = "Estimate Price";
    show("result-section");
  } else if (state === "error") {
    btn.disabled    = false;
    btn.textContent = "Estimate Price";
    show("error-card");
    show("result-section");   // keep previous result visible below error
  }
}

/* ── Render result ────────────────────────────────────────── */
function renderResult(data) {
  // Price
  document.getElementById("price-display").textContent =
    "₸ " + fmt(data.price_kzt);
  document.getElementById("price-sqm").textContent =
    fmt(data.price_per_sqm) + " ₸ / m²";
  document.getElementById("region-code").textContent = data.region_grid;
  document.getElementById("segment-code").textContent = data.segment_code;

  // Distances
  const dtDiv = document.getElementById("distances-table");
  dtDiv.innerHTML = "";
  for (const [key, meta] of Object.entries(DIST_META)) {
    const km  = data.distances[key];
    if (km === undefined) continue;
    const row = document.createElement("div");
    row.className = "info-row";

    const cls = km < 1 ? "val-near" : km > 5 ? "val-far" : "";
    row.innerHTML =
      `<span class="info-icon">${meta.icon}</span>` +
      `<span class="info-label">${meta.label}</span>` +
      `<span class="info-value ${cls}">${km.toFixed(3)} km</span>`;
    dtDiv.appendChild(row);
  }

  // Stats
  const stDiv = document.getElementById("stats-table");
  stDiv.innerHTML = "";
  for (const [label, val] of Object.entries(data.stat)) {
    if (val === null || val === undefined) continue;
    const row = document.createElement("div");
    row.className = "info-row";
    const display = typeof val === "number" ? fmtNum(val) : val;
    row.innerHTML =
      `<span class="info-label">${label}</span>` +
      `<span class="info-value">${display}</span>`;
    stDiv.appendChild(row);
  }
}

/* ── Helpers ──────────────────────────────────────────────── */
function fmt(n)    { return Math.round(n).toLocaleString("ru-RU"); }
function fmtNum(n) {
  if (Number.isInteger(n)) return n.toLocaleString("ru-RU");
  return parseFloat(n.toFixed(2)).toLocaleString("ru-RU");
}
function show(id)  { document.getElementById(id).classList.remove("hidden"); }
function hide(id)  { document.getElementById(id).classList.add("hidden"); }

function showError(msg) {
  document.getElementById("error-msg").textContent = msg;
}
