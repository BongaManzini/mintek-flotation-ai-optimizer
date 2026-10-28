// ==========================================================================
// Mintek Flotation Plant AI Optimizer - Interactive Digital Twin Engine
// Author: Bonga Manzini (UNISA BSc Informatics / Mintek SCI Hackathon)
// ==========================================================================

document.addEventListener('DOMContentLoaded', () => {
  console.log("🚀 Initializing Mintek AI Flotation Twin Dashboard...");
  initCharts();
  runSimulation();
  calculateROI();
  setupSmoothScrolling();
});

// Global state
let timelineChartInstance = null;
let featureChartInstance = null;

// Chemical pricing constants
const COST_AMINA_M3 = 3.20;  // USD per m³
const COST_STARCH_M3 = 0.85; // USD per m³

// ==========================================================================
// 1. Simulation Engine (AI Soft Sensor Prediction)
// ==========================================================================
function runSimulation() {
  // Read inputs
  const silicaFeed = parseFloat(document.getElementById('input-silica-feed').value);
  const ironFeed = parseFloat(document.getElementById('input-iron-feed').value);
  const pulpPH = parseFloat(document.getElementById('input-ph').value);
  const pulpDensity = parseFloat(document.getElementById('input-density').value);
  const airFlow = parseFloat(document.getElementById('input-air').value);
  const aminaFlow = parseFloat(document.getElementById('input-amina').value);
  const starchFlow = parseFloat(document.getElementById('input-starch').value);

  // Update slider badges
  document.getElementById('lbl-silica-feed').textContent = `${silicaFeed.toFixed(2)}%`;
  document.getElementById('lbl-iron-feed').textContent = `${ironFeed.toFixed(2)}%`;
  document.getElementById('lbl-ph').textContent = pulpPH.toFixed(2);
  document.getElementById('lbl-density').textContent = `${pulpDensity.toFixed(2)} g/cm³`;
  document.getElementById('lbl-air').textContent = `${airFlow.toFixed(0)} Nm³/h`;
  document.getElementById('lbl-amina').textContent = `${aminaFlow.toFixed(1)} m³/h`;
  document.getElementById('lbl-starch').textContent = `${starchFlow.toFixed(0)} m³/h`;

  // --- Metallurgical AI Surrogate Model Calculation ---
  // In reverse cationic flotation:
  // - Silica concentrate increases with higher feed silica
  // - Amina collector suppresses silica by floating it into tailings
  // - pH peak selectivity is between 10.0 and 10.4
  // - Aeration provides bubble surface area flux
  
  const phSelectivity = Math.max(0.45, 1.0 - Math.abs(pulpPH - 10.15) * 0.18);
  const airFactor = Math.max(0.7, airFlow / 280.0);
  const collectorEfficiency = (aminaFlow / 490.0) * phSelectivity * Math.pow(airFactor, 0.4);

  // Predicted silica concentrate impurity
  let predictedSilica = (silicaFeed * 0.155) / Math.max(0.35, collectorEfficiency);
  
  // Physical bounding
  predictedSilica = Math.max(0.60, Math.min(5.50, predictedSilica));

  // Predicted iron concentrate grade (% Fe)
  // Inverse relationship: every unit of silica removed elevates iron grade
  let predictedIron = 68.2 - (predictedSilica * 0.88) + ((ironFeed - 56.0) * 0.12);
  predictedIron = Math.max(60.0, Math.min(67.8, predictedIron));

  // Metallurgical Recovery Rate Calculation via Two-Product Mass Balance:
  // R = [c * (f - t)] / [f * (c - t)] * 100% (assuming typical tailings grade t ≈ 12% Fe)
  const tailingsIron = 12.5;
  const num = predictedIron * (ironFeed - tailingsIron);
  const den = ironFeed * (predictedIron - tailingsIron);
  let feRecovery = (num / Math.max(1e-4, den)) * 100.0;
  feRecovery = Math.max(65.0, Math.min(94.0, feRecovery));

  // Hourly chemical reagent cost
  const hourlyCost = (aminaFlow * COST_AMINA_M3) + (starchFlow * COST_STARCH_M3);

  // Update UI Elements
  document.getElementById('disp-silica-conc').textContent = `${predictedSilica.toFixed(2)}%`;
  document.getElementById('disp-iron-conc').textContent = `${predictedIron.toFixed(1)}% Fe`;
  document.getElementById('disp-fe-recovery').textContent = `${feRecovery.toFixed(1)}%`;
  document.getElementById('disp-chem-cost').textContent = `$${Math.round(hourlyCost).toLocaleString()}/h`;

  // Update Gauge Needle
  // Map silica from 0.5% (approx -80 deg) to 4.5% (approx +80 deg)
  const deg = -80 + ((predictedSilica - 0.5) / (4.5 - 0.5)) * 160;
  const clampedDeg = Math.max(-80, Math.min(80, deg));
  const needleEl = document.getElementById('gauge-needle');
  if (needleEl) {
    needleEl.style.transform = `rotate(${clampedDeg}deg)`;
  }

  // Update Quality Badge & Risk Status
  const badgeEl = document.getElementById('disp-quality-badge');
  const riskStatusEl = document.getElementById('disp-risk-status');

  badgeEl.className = 'quality-badge';
  if (predictedSilica <= 2.00) {
    badgeEl.classList.add('compliant');
    badgeEl.innerHTML = `<i class="fa-solid fa-circle-check"></i> <span>COMPLIANT (Under 2.0% Penalty Limit)</span>`;
    riskStatusEl.textContent = 'LOW RISK';
    riskStatusEl.style.color = 'var(--accent-emerald)';
  } else if (predictedSilica <= 2.50) {
    badgeEl.classList.add('warning');
    badgeEl.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> <span>ELEVATED IMPURITY (Caution Limit)</span>`;
    riskStatusEl.textContent = 'MODERATE';
    riskStatusEl.style.color = 'var(--accent-amber)';
  } else {
    badgeEl.classList.add('penalty');
    badgeEl.innerHTML = `<i class="fa-solid fa-circle-xmark"></i> <span>NON-COMPLIANT (Penalty Fines Incurred)</span>`;
    riskStatusEl.textContent = 'HIGH PENALTY';
    riskStatusEl.style.color = 'var(--accent-rose)';
  }
}

// ==========================================================================
// 2. Prescriptive Closed-Loop Optimizer
// ==========================================================================
function triggerOptimizer() {
  const silicaFeed = parseFloat(document.getElementById('input-silica-feed').value);
  const pulpPH = parseFloat(document.getElementById('input-ph').value);
  const airFlow = parseFloat(document.getElementById('input-air').value);
  const currentAmina = parseFloat(document.getElementById('input-amina').value);
  const currentStarch = parseFloat(document.getElementById('input-starch').value);

  const currentCost = (currentAmina * COST_AMINA_M3) + (currentStarch * COST_STARCH_M3);

  // Search optimal amina collector flow to hit target silica ≤ 1.85% (safe margin under 2.0%)
  const phSelectivity = Math.max(0.45, 1.0 - Math.abs(pulpPH - 10.15) * 0.18);
  const airFactor = Math.max(0.7, airFlow / 280.0);

  // Invert the surrogate equation: targetSilica = (silicaFeed * 0.155) / ((amina / 490) * phSelectivity * airFactor^0.4)
  const targetSilica = 1.85;
  let requiredAmina = ((silicaFeed * 0.155) / targetSilica) * (490.0 / (phSelectivity * Math.pow(airFactor, 0.4)));
  requiredAmina = Math.max(340.0, Math.min(680.0, requiredAmina));

  // Calculate corresponding required starch (balances iron depression based on feed)
  let requiredStarch = 2400.0 + (silicaFeed * 25.0);
  requiredStarch = Math.max(2000.0, Math.min(4200.0, requiredStarch));

  const newCost = (requiredAmina * COST_AMINA_M3) + (requiredStarch * COST_STARCH_M3);
  const costDiff = currentCost - newCost;

  // Animate slider updates
  animateSlider('input-amina', requiredAmina, 350);
  animateSlider('input-starch', requiredStarch, 350);

  // Trigger alert
  const alertEl = document.getElementById('opt-alert');
  const alertTextEl = document.getElementById('opt-alert-text');
  
  let msg = `Prescribed Amina: <strong>${requiredAmina.toFixed(1)} m³/h</strong>, Starch: <strong>${requiredStarch.toFixed(0)} m³/h</strong>. `;
  if (costDiff > 0) {
    msg += `Safely drives silica to ~${targetSilica}% while saving <strong>$${Math.round(costDiff)}/hour</strong> in chemicals!`;
  } else {
    msg += `Increased reagent dosing to prevent severe steel mill penalty fines!`;
  }
  
  alertTextEl.innerHTML = msg;
  alertEl.style.display = 'block';

  // Run simulation after slider animation
  setTimeout(() => {
    runSimulation();
  }, 380);
}

function animateSlider(sliderId, targetVal, durationMs) {
  const slider = document.getElementById(sliderId);
  const startVal = parseFloat(slider.value);
  const startTime = performance.now();

  function step(curTime) {
    const elapsed = curTime - startTime;
    const progress = Math.min(1.0, elapsed / durationMs);
    const eased = 0.5 - Math.cos(progress * Math.PI) / 2; // Ease in-out
    slider.value = startVal + (targetVal - startVal) * eased;
    if (progress < 1.0) {
      requestAnimationFrame(step);
    } else {
      slider.value = targetVal;
    }
  }
  requestAnimationFrame(step);
}

// ==========================================================================
// 3. Preset Scenarios
// ==========================================================================
function applyPreset(preset) {
  const alertEl = document.getElementById('opt-alert');
  alertEl.style.display = 'none';

  if (preset === 'nominal') {
    setInputs(14.65, 56.29, 9.77, 1.68, 280, 488.1, 2869);
  } else if (preset === 'highSilica') {
    setInputs(23.50, 49.20, 10.10, 1.72, 310, 560.0, 3200);
  } else if (preset === 'lowSilica') {
    setInputs(8.50, 61.20, 9.85, 1.65, 260, 410.0, 2400);
  } else if (preset === 'starvedAmina') {
    setInputs(16.50, 54.00, 9.40, 1.68, 250, 310.0, 2100);
  }
  runSimulation();
}

function setInputs(siFeed, feFeed, ph, density, air, amina, starch) {
  document.getElementById('input-silica-feed').value = siFeed;
  document.getElementById('input-iron-feed').value = feFeed;
  document.getElementById('input-ph').value = ph;
  document.getElementById('input-density').value = density;
  document.getElementById('input-air').value = air;
  document.getElementById('input-amina').value = amina;
  document.getElementById('input-starch').value = starch;
}

// ==========================================================================
// 4. Interactive Circuit Node Selection
// ==========================================================================
const circuitNodes = {
  cond: {
    title: "Conditioning Tank & Reagent Inlets",
    desc: "Raw crushed iron ore slurry is mixed with Corn Starch depressant (to shield iron oxide) and Ether Amine collector (to hydrophobize quartz). Operating pH is closely monitored at ~10.2 for optimal collector ionization.",
    badge: "pH ~10.2 | Amina 488 m³/h | Starch 2869 m³/h"
  },
  col1: {
    title: "Flotation Column 01 (Primary Rougher Bank)",
    desc: "First separation column receiving conditioned pulp. Operates with high slurry level (520 mm) and high air injection (280 Nm³/h) to float the primary bulk quartz gangue into the top overflow froth.",
    badge: "Level: 520 mm | Air: 280 Nm³/h | Bulk Float"
  },
  col2: {
    title: "Flotation Column 02 (Secondary Rougher Bank)",
    desc: "Secondary rougher column maintaining stable bubble swarm velocity. Provides continuous residence time to prevent uncollected coarse silica from escaping into downstream circuits.",
    badge: "Level: 522 mm | Air: 277 Nm³/h | Residence Stage"
  },
  col3: {
    title: "Flotation Column 03 (Tertiary Rougher Bank)",
    desc: "Final column in the rougher bank. Air injection (~281 Nm³/h) is dynamically trimmed to maximize quartz bubble loading before slurry passes to the scavengers.",
    badge: "Level: 531 mm | Air: 281 Nm³/h | Bubble Loading"
  },
  col4: {
    title: "Flotation Column 04 (Scavenger Bank)",
    desc: "Scavenger cells operate at lower pulp levels (~420 mm) and deeper froth zones to catch remaining fine quartz particles that missed rougher collection.",
    badge: "Level: 420 mm | Air: 299 Nm³/h | Deep Froth"
  },
  col5: {
    title: "Flotation Column 05 (Scavenger Bank)",
    desc: "Secondary scavenger stage. Telemetry shows strong correlation between Column 05 air flow and final concentrate grade.",
    badge: "Level: 425 mm | Air: 300 Nm³/h | High Selectivity"
  },
  col6: {
    title: "Flotation Column 06 & 07 (Cleaner Bank)",
    desc: "Final polishing stage before underflow discharge. Produces ultra-clean iron concentrate (Fe > 65%, SiO₂ < 2.0%) while sending rejected quartz froth to the tailings dam.",
    badge: "Level: 430 mm | Air: 292 Nm³/h | Final Product"
  }
};

function selectCircuitNode(nodeKey) {
  const node = circuitNodes[nodeKey];
  if (!node) return;

  // Highlight SVG node
  document.querySelectorAll('.node-group').forEach(el => el.classList.remove('active'));
  const activeEl = document.getElementById(`node-${nodeKey}`);
  if (activeEl) activeEl.classList.add('active');

  // Update info card
  document.getElementById('node-title').textContent = node.title;
  document.getElementById('node-desc').textContent = node.desc;
  document.getElementById('node-badge').textContent = node.badge;
}

// ==========================================================================
// 5. Chart.js Initializations
// ==========================================================================
function initCharts() {
  const data = window.MINTEK_DATA;
  if (!data || !data.timeline) {
    console.warn("⚠️ Telemetry data not loaded.");
    return;
  }

  // --- Chart 1: Actual vs Predicted Timeline ---
  const ctxTimeline = document.getElementById('timelineChart');
  if (ctxTimeline) {
    // Format timestamp labels nicely
    const rawLabels = data.timeline.timestamps.slice(0, 60);
    const shortLabels = rawLabels.map(t => {
      const parts = t.split(' ');
      return parts.length > 1 ? parts[1].substring(0, 5) : t;
    });

    timelineChartInstance = new Chart(ctxTimeline, {
      type: 'line',
      data: {
        labels: shortLabels,
        datasets: [
          {
            label: 'Actual Lab Assay (% Silica)',
            data: data.timeline.actual_silica.slice(0, 60),
            borderColor: '#94a3b8',
            backgroundColor: 'rgba(148, 163, 184, 0.1)',
            borderWidth: 2,
            pointRadius: 2,
            stepped: true
          },
          {
            label: 'AI Soft Sensor Prediction',
            data: data.timeline.predicted_silica.slice(0, 60),
            borderColor: '#06b6d4',
            backgroundColor: 'rgba(6, 182, 212, 0.15)',
            borderWidth: 2.5,
            pointRadius: 0,
            tension: 0.3
          },
          {
            label: 'Penalty Threshold (2.0%)',
            data: Array(60).fill(2.0),
            borderColor: '#f43f5e',
            borderWidth: 1.5,
            borderDash: [5, 5],
            pointRadius: 0,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
            labels: { color: '#94a3b8', boxWidth: 12, font: { family: 'Inter', size: 11 } }
          },
          tooltip: {
            mode: 'index',
            intersect: false,
            backgroundColor: '#0f172a',
            borderColor: '#24344d',
            borderWidth: 1
          }
        },
        scales: {
          x: {
            grid: { color: 'rgba(36, 52, 77, 0.4)' },
            ticks: { color: '#64748b', maxTicksLimit: 10, font: { family: 'JetBrains Mono', size: 10 } }
          },
          y: {
            min: 0.5,
            max: 4.5,
            grid: { color: 'rgba(36, 52, 77, 0.4)' },
            ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 10 }, callback: v => `${v}%` }
          }
        }
      }
    });
  }

  // --- Chart 2: Permutation Feature Importance ---
  const ctxFeature = document.getElementById('featureChart');
  if (ctxFeature && data.featureImportance) {
    const topFeats = data.featureImportance.slice(0, 6);
    const labels = topFeats.map(f => f.feature.replace(/Flotation Column 0/g, 'Col ').replace(/_avg/g, ' Avg'));
    const vals = topFeats.map(f => (f.importance * 100).toFixed(2));

    featureChartInstance = new Chart(ctxFeature, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Relative Permutation Influence (%)',
          data: vals,
          backgroundColor: [
            'rgba(59, 130, 246, 0.85)',
            'rgba(6, 182, 212, 0.85)',
            'rgba(16, 185, 129, 0.85)',
            'rgba(139, 92, 246, 0.85)',
            'rgba(245, 158, 11, 0.85)',
            'rgba(56, 189, 248, 0.85)'
          ],
          borderColor: '#1e293b',
          borderWidth: 1,
          borderRadius: 4
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false }
        },
        scales: {
          x: {
            grid: { color: 'rgba(36, 52, 77, 0.4)' },
            ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 10 }, callback: v => `${v}%` }
          },
          y: {
            grid: { display: false },
            ticks: { color: '#f8fafc', font: { family: 'Inter', size: 11, weight: '600' } }
          }
        }
      }
    });
  }
}

// ==========================================================================
// 6. Plant Economics & Sustainability (ROI Calculator)
// ==========================================================================
function calculateROI() {
  const capacity = parseFloat(document.getElementById('roi-cap').value);
  const penaltyPerTon = parseFloat(document.getElementById('roi-penalty').value);
  const savingsPct = parseFloat(document.getElementById('roi-savings').value);

  // Update slider labels
  document.getElementById('lbl-roi-cap').textContent = `${capacity.toLocaleString()} t/day`;
  document.getElementById('lbl-roi-penalty').textContent = `$${penaltyPerTon.toFixed(2)} / t`;
  document.getElementById('lbl-roi-savings').textContent = `${savingsPct.toFixed(1)}%`;

  // Constants: 345 operating days, ~72% concentrate mass pull
  const OPERATING_DAYS = 345;
  const annualConcentrateTons = capacity * 0.72 * OPERATING_DAYS;

  // 1. Reagent Cost Savings (Average Amina flow 488 m³/h, $3.20/m³)
  const baselineAnnualAminaM3 = 488.0 * 24 * OPERATING_DAYS;
  const aminaSavedM3 = baselineAnnualAminaM3 * (savingsPct / 100.0) * (capacity / 10000.0);
  const chemicalSavingsUSD = aminaSavedM3 * COST_AMINA_M3;

  // 2. Avoided Penalty Fines
  // Without AI soft sensor, ~14.2% of concentrate hours produce off-spec (> 2.0% silica).
  // With AI soft sensor, off-spec hours drop to ~3.5%.
  const offSpecAvoidedTons = annualConcentrateTons * (0.142 - 0.035);
  const penaltySavingsUSD = offSpecAvoidedTons * penaltyPerTon;

  const totalNetUSD = chemicalSavingsUSD + penaltySavingsUSD;

  // Update DOM
  document.getElementById('roi-net-savings').textContent = `$${Math.round(totalNetUSD).toLocaleString()}`;
  document.getElementById('roi-chem-savings').textContent = `$${Math.round(chemicalSavingsUSD).toLocaleString()}`;
  document.getElementById('roi-penalty-savings').textContent = `$${Math.round(penaltySavingsUSD).toLocaleString()}`;
  document.getElementById('roi-toxic-effluent').textContent = `${Math.round(aminaSavedM3).toLocaleString()} m³`;
}

// ==========================================================================
// 7. Navigation & Smooth Scrolling
// ==========================================================================
function setupSmoothScrolling() {
  const links = document.querySelectorAll('.nav-links a');
  window.addEventListener('scroll', () => {
    let current = '';
    const sections = document.querySelectorAll('header, section');
    sections.forEach(sec => {
      const top = sec.offsetTop - 120;
      if (window.scrollY >= top) {
        current = sec.getAttribute('id');
      }
    });

    links.forEach(link => {
      link.classList.remove('active');
      if (current && link.getAttribute('href') === `#${current}`) {
        link.classList.add('active');
      }
    });
  });
}
