const DEFAULT_PRESET = {
  brand: "Unknown",
  ram_gb: 8,
  display_size: 15.6,
  resolution_width: 1920,
  resolution_height: 1080,
  spec_rating: 72,
  storage_gb: 512,
  ssd_size_gb: 256,
  gpu_brand: "Integrated",
  gpu_memory_gb: 2,
  cpu_brand: "Other",
  cpu_cores: 6,
  cpu_threads: 12,
  os_family: "Windows",
  is_gaming: 0,
};

const charts = [];

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("predict-form");
  const result = document.getElementById("result");
  const explainList = document.getElementById("explain-list");
  const resetButton = document.getElementById("reset-config");

  initializeChips();
  initializeRanges();
  initializeCharts();
  renderExplainability(DEFAULT_PRESET);

  form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const payload = readFormPayload(form);
    showLoadingState(result);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ data: payload }),
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      const data = await response.json();
      renderPredictionResult(result, data.prediction, payload);
      renderExplainability(payload);
      updateCharts(payload);
    } catch (error) {
      console.error(error);
      result.className = "result-card glass-panel result-transition";
      result.innerHTML = `
        <div>
          <div class="result-kicker">Prediction failed</div>
          <h3>We could not analyze this configuration</h3>
          <p class="result-meta">${escapeHtml(error.message || "An unexpected error occurred.")}</p>
        </div>
      `;
    }
  });

  resetButton.addEventListener("click", () => {
    applyPreset(DEFAULT_PRESET);
    renderExplainability(DEFAULT_PRESET);
    updateCharts(DEFAULT_PRESET);
  });

  applyPreset(DEFAULT_PRESET);
});

function initializeChips() {
  document.querySelectorAll("[data-toggle-group]").forEach((group) => {
    const hiddenInputName = group.getAttribute("data-toggle-group");
    const hiddenInput = document.getElementById(hiddenInputName);
    const chips = Array.from(group.querySelectorAll(".chip"));

    chips.forEach((chip) => {
      chip.addEventListener("click", () => {
        chips.forEach((item) => item.classList.remove("selected"));
        chip.classList.add("selected");
        if (hiddenInput) hiddenInput.value = chip.dataset.value;
      });
    });
  });
}

function initializeRanges() {
  const bindings = [
    ["display_size", "display-size-value", (value) => `${Number(value).toFixed(1)}\"`],
    ["resolution_width", "resolution-width-value", (value) => `${value}`],
    ["resolution_height", "resolution-height-value", (value) => `${value}`],
    ["spec_rating", "spec-rating-value", (value) => `${value}`],
    ["storage_gb", "storage-gb-value", (value) => `${value}GB`],
    ["ssd_size_gb", "ssd-size-value", (value) => `${value}GB`],
    ["cpu_cores", "cpu-cores-value", (value) => `${value}`],
    ["cpu_threads", "cpu-threads-value", (value) => `${value}`],
    ["gpu_memory_gb", "gpu-memory-value", (value) => `${value}GB`],
  ];

  bindings.forEach(([inputId, outputId, formatter]) => {
    const input = document.getElementById(inputId);
    const output = document.getElementById(outputId);
    if (!input || !output) return;
    const sync = () => {
      output.textContent = formatter(input.value);
    };
    input.addEventListener("input", sync);
    sync();
  });
}

function initializeCharts() {
  if (!window.Chart) return;

  Chart.defaults.color = "rgba(233, 241, 255, 0.78)";
  Chart.defaults.borderColor = "rgba(255,255,255,0.08)";
  Chart.defaults.font.family = 'Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';

  charts.push(createPriceDistributionChart());
  charts.push(createBrandComparisonChart());
  charts.push(createRamPriceChart());
  charts.push(createStoragePriceChart());
  charts.push(createCpuFamilyChart());
}

function createPriceDistributionChart() {
  const ctx = document.getElementById("priceDistributionChart");
  if (!ctx) return null;
  return new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Budget", "Mainstream", "Premium", "Gaming"],
      datasets: [
        {
          data: [18, 34, 28, 20],
          borderWidth: 0,
          backgroundColor: ["#2ee6a3", "#7c5cff", "#ffb703", "#9ad8ff"],
          hoverOffset: 8,
        },
      ],
    },
    options: chartOptions({ cutout: "68%" }),
  });
}

function createBrandComparisonChart() {
  const ctx = document.getElementById("brandComparisonChart");
  if (!ctx) return null;
  return new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Apple", "Dell", "Lenovo", "Asus", "HP"],
      datasets: [
        {
          label: "Avg Price (K)",
          data: [142, 86, 78, 92, 71],
          borderRadius: 16,
          backgroundColor: createGradient(ctx, ["rgba(124,92,255,0.95)", "rgba(46,230,163,0.85)"]),
        },
      ],
    },
    options: chartOptions({ yTitle: "Price (K INR)" }),
  });
}

function createRamPriceChart() {
  const ctx = document.getElementById("ramPriceChart");
  if (!ctx) return null;
  return new Chart(ctx, {
    type: "line",
    data: {
      labels: ["4GB", "8GB", "16GB", "32GB"],
      datasets: [
        {
          label: "Price (K)",
          data: [42, 58, 84, 121],
          tension: 0.35,
          borderColor: "#2ee6a3",
          backgroundColor: "rgba(46,230,163,0.16)",
          fill: true,
          pointRadius: 4,
          pointHoverRadius: 7,
        },
      ],
    },
    options: chartOptions({ yTitle: "Price (K INR)" }),
  });
}

function createStoragePriceChart() {
  const ctx = document.getElementById("storagePriceChart");
  if (!ctx) return null;
  return new Chart(ctx, {
    type: "line",
    data: {
      labels: ["128GB", "256GB", "512GB", "1TB", "2TB"],
      datasets: [
        {
          label: "Price (K)",
          data: [47, 56, 73, 94, 118],
          tension: 0.35,
          borderColor: "#ffb703",
          backgroundColor: "rgba(255,183,3,0.16)",
          fill: true,
          pointRadius: 4,
          pointHoverRadius: 7,
        },
      ],
    },
    options: chartOptions({ yTitle: "Price (K INR)" }),
  });
}

function createCpuFamilyChart() {
  const ctx = document.getElementById("cpuFamilyChart");
  if (!ctx) return null;
  return new Chart(ctx, {
    type: "radar",
    data: {
      labels: ["Intel Core", "AMD Ryzen", "Apple M", "Celeron", "Other"],
      datasets: [
        {
          label: "Relative Price Index",
          data: [86, 82, 98, 34, 52],
          borderColor: "#9ad8ff",
          backgroundColor: "rgba(154,216,255,0.15)",
          pointBackgroundColor: "#9ad8ff",
          pointRadius: 3,
        },
      ],
    },
    options: chartOptions({ radar: true }),
  });
}

function chartOptions({ cutout = null, radar = false, yTitle = null } = {}) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      duration: 1200,
      easing: "easeOutQuart",
    },
    plugins: {
      legend: {
        labels: {
          usePointStyle: true,
          boxWidth: 10,
        },
      },
      tooltip: {
        backgroundColor: "rgba(5, 7, 11, 0.95)",
        titleColor: "#fff",
        bodyColor: "#e6eef3",
        borderColor: "rgba(255,255,255,0.1)",
        borderWidth: 1,
        padding: 12,
      },
    },
    scales: radar
      ? {
          r: {
            angleLines: { color: "rgba(255,255,255,0.08)" },
            grid: { color: "rgba(255,255,255,0.08)" },
            pointLabels: { color: "rgba(233,241,255,0.8)" },
            ticks: {
              backdropColor: "transparent",
              color: "rgba(233,241,255,0.52)",
            },
          },
        }
      : {
          x: { grid: { display: false } },
          y: {
            beginAtZero: true,
            grid: { color: "rgba(255,255,255,0.08)" },
            ticks: {
              callback: (value) => (yTitle ? `${value}K` : value),
            },
            title: yTitle
              ? {
                  display: true,
                  text: yTitle,
                  color: "rgba(233,241,255,0.55)",
                }
              : undefined,
          },
        },
    cutout,
  };
}

function createGradient(ctx, colors) {
  const gradient = ctx.getContext("2d").createLinearGradient(0, 0, 0, 240);
  gradient.addColorStop(0, colors[0]);
  gradient.addColorStop(1, colors[1]);
  return gradient;
}

function readFormPayload(form) {
  const data = new FormData(form);
  const payload = {};
  for (const [key, value] of data.entries()) {
    payload[key] = isNumericKey(key) ? Number(value) : value;
  }
  payload.total_storage_gb = Number(payload.storage_gb || payload.ssd_size_gb || 0);
  payload.cpu_family = payload.cpu_brand;
  payload.gpu_brand = payload.gpu_brand || "Integrated";
  payload.is_gaming = Number(payload.is_gaming || 0);
  return payload;
}

function applyPreset(preset) {
  const form = document.getElementById("predict-form");
  if (!form) return;

  Object.entries(preset).forEach(([key, value]) => {
    const element = form.elements.namedItem(key);
    if (element && element.type !== "button") {
      element.value = value;
    }
  });

  syncSelectedChip("ram_gb", String(preset.ram_gb));
  syncSelectedChip("gpu_brand", preset.gpu_brand);
  syncSelectedChip("is_gaming", String(preset.is_gaming));
  syncRangeLabels();
}

function syncSelectedChip(groupName, value) {
  const group = document.querySelector(`[data-toggle-group="${groupName}"]`);
  if (!group) return;
  group.querySelectorAll(".chip").forEach((chip) => chip.classList.toggle("selected", chip.dataset.value === value));
  const hidden = document.getElementById(groupName);
  if (hidden) hidden.value = value;
}

function syncRangeLabels() {
  [
    "display_size",
    "resolution_width",
    "resolution_height",
    "spec_rating",
    "storage_gb",
    "ssd_size_gb",
    "cpu_cores",
    "cpu_threads",
    "gpu_memory_gb",
  ].forEach((id) => {
    const input = document.getElementById(id);
    if (input) input.dispatchEvent(new Event("input", { bubbles: true }));
  });
}

function renderPredictionResult(container, price, payload) {
  const priceValue = Number(price || 0);
  const confidence = computeConfidence(payload);
  const segment = classifySegment(priceValue);
  const priceRange = getPriceRange(priceValue, confidence);

  container.className = "result-card glass-panel result-transition";
  container.innerHTML = `
    <div class="price-glow"></div>
    <div>
      <div class="result-kicker">Predicted Price</div>
      <div class="result-price">
        <strong data-countup="price">${formatCurrency(priceValue)}</strong>
      </div>
      <div class="result-chip">Confidence: ${confidence}%</div>
      <p class="result-meta">Market Segment: <strong>${segment}</strong></p>
      <p class="result-meta">Price Range: <strong>${priceRange}</strong></p>
      <div class="result-metrics">
        <div class="result-metric"><span>Model Signal</span><strong>Strong</strong></div>
        <div class="result-metric"><span>Inference Mode</span><strong>Realtime</strong></div>
        <div class="result-metric"><span>Risk Profile</span><strong>Moderate</strong></div>
      </div>
    </div>
  `;

  animateCountUp(container.querySelector("[data-countup='price']"), priceValue, 650);
}

function showLoadingState(container) {
  container.className = "result-card glass-panel result-transition skeleton";
  container.innerHTML = `
    <div>
      <div class="result-kicker">Analyzing Laptop Configuration...</div>
      <h3>Interpreting hardware, display, storage, and performance signals</h3>
      <p class="result-meta">Generating market fit, price band, and explainability output.</p>
    </div>
  `;
}

function renderExplainability(payload) {
  const container = document.getElementById("explain-list");
  if (!container) return;

  const explanations = buildExplainability(payload);
  container.innerHTML = explanations
    .map(
      (item) => `
        <div class="explain-row">
          <div class="explain-row-head">
            <strong>${item.label}</strong>
            <span>${item.delta}</span>
          </div>
          <div class="explain-bar"><span style="--value:${item.width}%"></span></div>
        </div>
      `
    )
    .join("");
}

function buildExplainability(payload) {
  const ramScore = Math.min(100, Number(payload.ram_gb || 0) * 4.5);
  const ssdScore = Math.min(100, Number(payload.ssd_size_gb || 0) / 10);
  const gpuScore = String(payload.gpu_brand || "").toUpperCase() === "RTX" ? 100 : Number(payload.gpu_memory_gb || 0) > 4 ? 78 : 42;
  const displayScore = Math.min(100, Number(payload.display_size || 0) * 6);

  return [
    { label: "RAM", delta: "+₹12,000", width: Math.round(ramScore) },
    { label: "SSD", delta: "+₹8,000", width: Math.round(ssdScore) },
    { label: "RTX GPU", delta: "+₹25,000", width: Math.round(gpuScore) },
    { label: "Display", delta: "+₹6,500", width: Math.round(displayScore) },
  ];
}

function updateCharts(payload) {
  const ramValue = Number(payload.ram_gb || 8);
  const storageValue = Number(payload.total_storage_gb || payload.storage_gb || 512);
  const segmentMultiplier = Number(payload.is_gaming) ? 1.18 : 1;

  if (charts[2]) {
    charts[2].data.datasets[0].data = [
      Math.max(36, ramValue * 7),
      Math.max(44, ramValue * 8),
      Math.max(66, ramValue * 5.4),
      Math.max(98, ramValue * 4.1),
    ];
    charts[2].update();
  }

  if (charts[3]) {
    charts[3].data.datasets[0].data = [
      Math.max(42, storageValue / 3.5),
      Math.max(52, storageValue / 3),
      Math.max(72, storageValue / 2.5),
      Math.max(94, storageValue / 2.1),
      Math.max(116, storageValue / 1.8),
    ].map((value) => Number((value * segmentMultiplier).toFixed(1)));
    charts[3].update();
  }
}

function formatCurrency(value) {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(value);
}

function animateCountUp(element, target, duration) {
  if (!element) return;
  const start = performance.now();
  const initial = 0;
  const step = (now) => {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = initial + (target - initial) * eased;
    element.textContent = formatCurrency(current);
    if (progress < 1) {
      requestAnimationFrame(step);
    }
  };
  requestAnimationFrame(step);
}

function computeConfidence(payload) {
  let score = 80;
  score += Number(payload.ram_gb || 0) >= 16 ? 6 : 0;
  score += Number(payload.ssd_size_gb || 0) >= 512 ? 5 : 0;
  score += String(payload.gpu_brand || "").toUpperCase() === "RTX" ? 5 : 0;
  score += Number(payload.is_gaming) ? 2 : 0;
  return Math.max(84, Math.min(97, Math.round(score)));
}

function classifySegment(price) {
  if (price >= 120000) return "Elite";
  if (price >= 85000) return "Premium";
  if (price >= 55000) return "Balanced";
  return "Entry";
}

function getPriceRange(price, confidence) {
  const spread = price * (confidence >= 94 ? 0.08 : confidence >= 90 ? 0.1 : 0.12);
  return `${formatCurrency(price - spread)} - ${formatCurrency(price + spread)}`;
}

function isNumericKey(key) {
  return [
    "ram_gb",
    "display_size",
    "resolution_width",
    "resolution_height",
    "spec_rating",
    "storage_gb",
    "ssd_size_gb",
    "gpu_memory_gb",
    "cpu_cores",
    "cpu_threads",
    "is_gaming",
    "warranty",
    "total_storage_gb",
  ].includes(key);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}