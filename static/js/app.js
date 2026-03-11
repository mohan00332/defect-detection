const API_BASE = (window.API_BASE || '').replace(/\/$/, '');
const apiUrl = (path) => (API_BASE ? `${API_BASE}${path}` : path);
const normalizeUrl = (url) => {
  if (!url) return url;
  if (/^(https?:|data:|blob:)/i.test(url)) return url;
  if (url.startsWith('/')) return apiUrl(url);
  return apiUrl(`/${url}`);
};
const LIVE_SUPPORTED = window.LIVE_SUPPORTED !== false;
const REPORTS_SUPPORTED = window.REPORTS_SUPPORTED !== false;
const navItems = document.querySelectorAll('.nav-item');
const panels = document.querySelectorAll('.panel');
const categoryChips = document.querySelectorAll('.chip');

const expectedInput = document.getElementById('expectedCount');
const systemStatus = document.getElementById('systemStatus');

const previewPlaceholder = document.getElementById('previewPlaceholder');
const imagePreview = document.getElementById('imagePreview');
const videoPreview = document.getElementById('videoPreview');
const liveFeed = document.getElementById('liveFeed');
const liveFeedDashboard = document.getElementById('liveFeedDashboard');
const previewDefectCount = document.getElementById('previewDefectCount');
const previewGoodCount = document.getElementById('previewGoodCount');
const previewCorner = document.getElementById('previewCorner');

const startCameraBtn = document.getElementById('startCamera');
const stopCameraBtn = document.getElementById('stopCamera');
const startCameraDashboardBtn = document.getElementById('startCameraDashboard');
const stopCameraDashboardBtn = document.getElementById('stopCameraDashboard');
const detectImageBtn = document.getElementById('detectImage');
const detectVideoBtn = document.getElementById('detectVideo');
const uploadDrop = document.getElementById('uploadDrop');
const sampleImages = document.querySelectorAll('[data-sample-url]');
const uploadFileInput = document.getElementById('uploadFileInput');
const uploadFileName = document.getElementById('uploadFileName');
const uploadProgressBar = document.getElementById('uploadProgressBar');
const uploadStatus = document.getElementById('uploadStatus');
const detectImageUrlBtn = document.getElementById('detectImageUrl');
const imageUrlInput = document.getElementById('imageUrlInput');
const confidenceSlider = document.getElementById('confidenceSlider');
const overlapSlider = document.getElementById('overlapSlider');
const confidenceValue = document.getElementById('confidenceValue');
const overlapValue = document.getElementById('overlapValue');
const labelModeSelect = document.getElementById('labelModeSelect');
const downloadReport = document.getElementById('downloadReport');
const downloadReportSecondary = document.getElementById('downloadReportSecondary');
const downloadDailyExcel = document.getElementById('downloadDailyExcel');
const downloadDailyPdf = document.getElementById('downloadDailyPdf');
const reportDate = document.getElementById('reportDate');
const analyticsTotalDetected = document.getElementById('analyticsTotalDetected');
const analyticsTotalDefect = document.getElementById('analyticsTotalDefect');
const analyticsGoodRate = document.getElementById('analyticsGoodRate');
const trendChart = document.getElementById('trendChart');
const categoryChart = document.getElementById('categoryChart');
const liveChart = document.getElementById('liveChart');

const currentCategory = document.getElementById('currentCategory');
const currentDetected = document.getElementById('currentDetected');
const currentDefect = document.getElementById('currentDefect');
const currentGood = document.getElementById('currentGood');
const expectedLine = document.getElementById('expectedLine');
const goodRate = document.getElementById('goodRate');
const lastUpdate = document.getElementById('lastUpdate');

const nutStats = document.getElementById('nutStats');
const boltStats = document.getElementById('boltStats');
const gearStats = document.getElementById('gearStats');
const nutDetected = document.getElementById('nutDetected');
const nutDefect = document.getElementById('nutDefect');
const nutGood = document.getElementById('nutGood');
const boltDetected = document.getElementById('boltDetected');
const boltDefect = document.getElementById('boltDefect');
const boltGood = document.getElementById('boltGood');
const gearDetected = document.getElementById('gearDetected');
const gearDefect = document.getElementById('gearDefect');
const gearGood = document.getElementById('gearGood');
const resetCategoryTotals = document.getElementById('resetCategoryTotals');

document.addEventListener('click', (event) => {
  const clickable = event.target.closest('button, .chip, .nav-item, .upload-btn, .stack-link');
  if (!clickable) return;
  clickable.classList.remove('click-glow');
  void clickable.offsetWidth;
  clickable.classList.add('click-glow');
});

if (!LIVE_SUPPORTED) {
  [startCameraBtn, startCameraDashboardBtn, stopCameraBtn, stopCameraDashboardBtn].forEach(btn => {
    if (!btn) return;
    btn.disabled = true;
    btn.classList.add('disabled');
    btn.title = 'Live camera not supported in this backend';
  });
}

if (!REPORTS_SUPPORTED) {
  [downloadReport, downloadReportSecondary, downloadDailyExcel, downloadDailyPdf].forEach(btn => {
    if (!btn) return;
    btn.disabled = true;
    btn.classList.add('disabled');
    btn.title = 'Reports not supported in this backend';
  });
}

let selectedCategory = 'Nut';
let livePolling = null;
let isLive = false;
const liveSeries = [];

function setActivePanel(panelName) {
  panels.forEach(panel => {
    panel.classList.toggle('active', panel.id === `panel-${panelName}`);
  });
  navItems.forEach(item => {
    item.classList.toggle('active', item.dataset.panel === panelName);
  });
}

function setCategory(category) {
  selectedCategory = category;
  categoryChips.forEach(chip => {
    chip.classList.toggle('active', chip.dataset.category === category);
  });
  if (isLive) {
    startLive();
  }
}

function showPreviewImage(url) {
  if (!imagePreview) return;
  if (previewPlaceholder) previewPlaceholder.classList.add('hidden');
  if (liveFeedDashboard) liveFeedDashboard.classList.add('hidden');
  if (videoPreview) videoPreview.classList.add('hidden');
  imagePreview.classList.remove('hidden');
  imagePreview.src = normalizeUrl(url);
}

function showPreviewVideo(url) {
  if (!videoPreview) return;
  if (previewPlaceholder) previewPlaceholder.classList.add('hidden');
  if (liveFeedDashboard) liveFeedDashboard.classList.add('hidden');
  if (imagePreview) imagePreview.classList.add('hidden');
  videoPreview.classList.remove('hidden');
  videoPreview.src = normalizeUrl(url);
}

function showLivePreview() {
  if (!liveFeedDashboard) return;
  if (previewPlaceholder) previewPlaceholder.classList.add('hidden');
  if (imagePreview) imagePreview.classList.add('hidden');
  if (videoPreview) videoPreview.classList.add('hidden');
  liveFeedDashboard.classList.remove('hidden');
}

function renderLiveChart() {
  if (!liveChart) return;
  liveChart.innerHTML = '';
  const maxVal = Math.max(1, ...liveSeries.map(item => item.defect));
  liveSeries.forEach(item => {
    const group = document.createElement('div');
    group.className = 'bar-group';

    const defectBar = document.createElement('div');
    defectBar.className = 'bar defect';
    defectBar.style.height = `${(item.defect / maxVal) * 100}%`;

    const label = document.createElement('div');
    label.className = 'bar-label';
    label.textContent = item.label;

    group.appendChild(defectBar);
    group.appendChild(label);
    liveChart.appendChild(group);
  });
}

function getThresholdSettings() {
  const conf = confidenceSlider ? Number(confidenceSlider.value || 50) / 100 : null;
  const iou = overlapSlider ? Number(overlapSlider.value || 50) / 100 : null;
  const labelMode = labelModeSelect ? labelModeSelect.value : 'confidence';
  return { conf, iou, labelMode };
}

function updateCurrentStats(category, detected, defect, good, expected) {
  currentCategory.textContent = `Category: ${category}`;
  currentDetected.textContent = `Detected Count: ${detected}`;
  currentDefect.textContent = `Defect Count: ${defect}`;
  currentGood.textContent = `Non-Defect Count: ${good}`;
  if (previewDefectCount) previewDefectCount.textContent = defect;
  if (previewGoodCount) previewGoodCount.textContent = good;
  if (previewCorner) {
    const isDefect = defect > 0;
    previewCorner.textContent = isDefect ? 'Defect' : 'Non-Defect';
    previewCorner.classList.toggle('defect', isDefect);
    previewCorner.classList.toggle('good', !isDefect);
  }
  if (expectedLine) expectedLine.textContent = `Expected Count: ${expected}`;

  if (expected > 0) {
    const rate = ((good / expected) * 100).toFixed(1);
    if (goodRate) goodRate.textContent = `Good Rate: ${rate}%`;
  } else {
    if (goodRate) goodRate.textContent = 'Good Rate: -';
  }

  lastUpdate.textContent = `Updated: ${new Date().toLocaleTimeString()}`;
}

async function refreshTotals() {
  const res = await fetch(apiUrl('/stats'));
  const data = await res.json();
  if (!data.ok) return;
  const stats = data.stats;
  if (nutDetected) nutDetected.textContent = stats.Nut.detected;
  if (nutDefect) nutDefect.textContent = stats.Nut.defect;
  if (nutGood) nutGood.textContent = stats.Nut.good;
  if (boltDetected) boltDetected.textContent = stats.Bolt.detected;
  if (boltDefect) boltDefect.textContent = stats.Bolt.defect;
  if (boltGood) boltGood.textContent = stats.Bolt.good;
  if (gearDetected) gearDetected.textContent = stats.Gear.detected;
  if (gearDefect) gearDefect.textContent = stats.Gear.defect;
  if (gearGood) gearGood.textContent = stats.Gear.good;
  refreshAnalytics();
}

async function resetTotals() {
  const res = await fetch(apiUrl('/reset_stats'), { method: 'POST' });
  const data = await res.json();
  if (!data.ok) return;
  await refreshTotals();
}

if (resetCategoryTotals) {
  resetCategoryTotals.addEventListener('click', () => {
    resetTotals();
  });
}

function clearChart(container) {
  if (!container) return;
  container.innerHTML = '';
}

function renderTrendChart(data) {
  if (!trendChart) return;
  clearChart(trendChart);
  const maxVal = Math.max(1, ...data.map(d => Math.max(d.defect, d.good)));
  data.forEach(item => {
    const group = document.createElement('div');
    group.className = 'bar-group';

    const defectBar = document.createElement('div');
    defectBar.className = 'bar defect';
    defectBar.style.height = `${(item.defect / maxVal) * 100}%`;

    const goodBar = document.createElement('div');
    goodBar.className = 'bar good';
    goodBar.style.height = `${(item.good / maxVal) * 100}%`;

    const label = document.createElement('div');
    label.className = 'bar-label';
    label.textContent = item.date.slice(5);

    group.appendChild(defectBar);
    group.appendChild(goodBar);
    group.appendChild(label);
    trendChart.appendChild(group);
  });
}

function renderCategoryChart(categories) {
  if (!categoryChart) return;
  clearChart(categoryChart);
  const items = Object.keys(categories).map(key => ({
    name: key,
    defect: categories[key].defect || 0
  }));
  const maxVal = Math.max(1, ...items.map(i => i.defect));
  items.forEach(item => {
    const group = document.createElement('div');
    group.className = 'bar-group';

    const defectBar = document.createElement('div');
    defectBar.className = 'bar defect';
    defectBar.style.height = `${(item.defect / maxVal) * 100}%`;

    const label = document.createElement('div');
    label.className = 'bar-label';
    label.textContent = item.name;

    group.appendChild(defectBar);
    group.appendChild(label);
    categoryChart.appendChild(group);
  });
}

async function refreshAnalytics() {
  if (!trendChart && !categoryChart) return;
  const res = await fetch(apiUrl('/analytics/summary'));
  if (!res.ok) return;
  const data = await res.json();
  if (!data.ok) return;

  if (analyticsTotalDetected) analyticsTotalDetected.textContent = data.totals.detected;
  if (analyticsTotalDefect) analyticsTotalDefect.textContent = data.totals.defect;
  if (analyticsGoodRate) {
    const rate = data.totals.detected > 0
      ? ((data.totals.good / data.totals.detected) * 100).toFixed(1)
      : '-';
    analyticsGoodRate.textContent = rate === '-' ? '-' : `${rate}%`;
  }

  renderTrendChart(data.trend || []);
  renderCategoryChart(data.categories || {});
}

navItems.forEach(item => {
  item.addEventListener('click', () => setActivePanel(item.dataset.panel));
});

categoryChips.forEach(chip => {
  chip.addEventListener('click', () => setCategory(chip.dataset.category));
});

async function startLive() {
  if (!LIVE_SUPPORTED) {
    alert('Live camera is not supported in this backend. Use image or video detection.');
    return;
  }
  systemStatus.textContent = 'Live';
  liveSeries.length = 0;
  renderLiveChart();
  const thresholds = getThresholdSettings();
  await fetch(apiUrl('/start_camera'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      category: selectedCategory,
      conf: thresholds.conf,
      iou: thresholds.iou,
      label_mode: thresholds.labelMode
    })
  });
  const src = apiUrl(`/video_feed?category=${encodeURIComponent(selectedCategory)}&ts=${Date.now()}`);
  liveFeed.src = src;
  liveFeedDashboard.src = src;
  showLivePreview();
  if (livePolling) clearInterval(livePolling);
  livePolling = setInterval(async () => {
    const res = await fetch(apiUrl('/live_stats'));
    if (!res.ok) return;
    const data = await res.json();
    if (!data.ok) return;
    const expected = Number(expectedInput.value || 0);
    updateCurrentStats(data.category, data.detected, data.defect, data.good, expected);
    const now = new Date();
    liveSeries.push({
      defect: Number(data.defect || 0),
      label: now.toLocaleTimeString().slice(3, 8)
    });
    if (liveSeries.length > 60) liveSeries.shift();
    renderLiveChart();
  }, 1000);
  isLive = true;
}

async function stopLive() {
  if (!LIVE_SUPPORTED) {
    return;
  }
  systemStatus.textContent = 'Idle';
  liveFeed.src = '';
  liveFeedDashboard.src = '';
  if (liveFeedDashboard) liveFeedDashboard.classList.add('hidden');
  if (livePolling) {
    clearInterval(livePolling);
    livePolling = null;
  }
  isLive = false;
  const expected = Number(expectedInput.value || 0);
  const res = await fetch(apiUrl('/stop_camera'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ expected_count: expected })
  });
  const data = await res.json();
  if (data.ok && data.output_url) {
    showPreviewImage(data.output_url);
    updateCurrentStats(data.category, data.detected, data.defect, data.good, expected);
    await refreshTotals();
    setActivePanel('dashboard');
  }
}

startCameraBtn.addEventListener('click', async () => {
  await startLive();
  setActivePanel('live');
});

stopCameraBtn.addEventListener('click', async () => {
  await stopLive();
});

startCameraDashboardBtn.addEventListener('click', async () => {
  await startLive();
  setActivePanel('dashboard');
});

stopCameraDashboardBtn.addEventListener('click', async () => {
  await stopLive();
});

detectImageBtn.addEventListener('click', async () => {
  const fileInput = document.getElementById('imageFile');
  if (!fileInput.files.length) return;

  systemStatus.textContent = 'Processing';
  const expected = Number(expectedInput.value || 0);
  const form = new FormData();
  form.append('image', fileInput.files[0]);
  form.append('category', selectedCategory);
  form.append('expected_count', expected);
  const thresholds = getThresholdSettings();
  if (thresholds.conf !== null) form.append('conf', thresholds.conf);
  if (thresholds.iou !== null) form.append('iou', thresholds.iou);
  form.append('label_mode', thresholds.labelMode);

  const res = await fetch(apiUrl('/detect_image'), { method: 'POST', body: form });
  const data = await res.json();
  systemStatus.textContent = 'Idle';

  if (data.ok) {
    showPreviewImage(data.output_url);
    updateCurrentStats(data.category, data.detected, data.defect, data.good, expected);
    await refreshTotals();
    setActivePanel('dashboard');
  }
});

detectVideoBtn.addEventListener('click', async () => {
  const fileInput = document.getElementById('videoFile');
  if (!fileInput.files.length) return;

  systemStatus.textContent = 'Processing';
  const expected = Number(expectedInput.value || 0);
  const form = new FormData();
  form.append('video', fileInput.files[0]);
  form.append('category', selectedCategory);
  form.append('expected_count', expected);

  const res = await fetch(apiUrl('/detect_video'), { method: 'POST', body: form });
  const data = await res.json();
  systemStatus.textContent = 'Idle';

  if (data.ok) {
    showPreviewVideo(data.output_url);
    updateCurrentStats(data.category, data.detected, data.defect, data.good, expected);
    await refreshTotals();
    setActivePanel('dashboard');
  }
});

function handleUploadFile(file) {
  if (!file) return;
  uploadFileName.textContent = file.name;
  uploadProgressBar.style.width = '0%';
  uploadStatus.textContent = 'Uploading...';
  systemStatus.textContent = 'Processing';

  const expected = Number(expectedInput.value || 0);
  const form = new FormData();
  form.append('file', file);
  form.append('category', selectedCategory);
  form.append('expected_count', expected);
  const thresholds = getThresholdSettings();
  if (thresholds.conf !== null) form.append('conf', thresholds.conf);
  if (thresholds.iou !== null) form.append('iou', thresholds.iou);
  form.append('label_mode', thresholds.labelMode);

  const xhr = new XMLHttpRequest();
  xhr.open('POST', apiUrl('/upload'), true);

  xhr.upload.onprogress = (event) => {
    if (event.lengthComputable) {
      const percent = Math.round((event.loaded / event.total) * 100);
      uploadProgressBar.style.width = `${percent}%`;
      uploadStatus.textContent = `Uploading... ${percent}%`;
    }
  };

  xhr.onload = async () => {
    systemStatus.textContent = 'Idle';
    if (xhr.status !== 200) {
      uploadStatus.textContent = 'Upload failed';
      return;
    }
    const data = JSON.parse(xhr.responseText || '{}');
    if (!data.ok) {
      uploadStatus.textContent = data.error || 'Upload failed';
      return;
    }
    uploadStatus.textContent = 'Upload complete';
    if (data.output_url.endsWith('.mp4')) {
      showPreviewVideo(data.output_url);
    } else {
      showPreviewImage(data.output_url);
    }
    updateCurrentStats(data.category, data.detected, data.defect, data.good, expected);
    await refreshTotals();
    setActivePanel('dashboard');
  };

  xhr.onerror = () => {
    systemStatus.textContent = 'Idle';
    uploadStatus.textContent = 'Upload failed';
  };

  xhr.send(form);
}

async function handleSampleClick(url) {
  if (!url) return;
  uploadStatus.textContent = 'Loading sample...';
  try {
    const res = await fetch(normalizeUrl(url));
    if (!res.ok) {
      uploadStatus.textContent = 'Failed to load sample';
      return;
    }
    const blob = await res.blob();
    const name = url.split('/').pop() || 'sample.jpg';
    const file = new File([blob], name, { type: blob.type || 'image/jpeg' });
    handleUploadFile(file);
  } catch {
    uploadStatus.textContent = 'Failed to load sample';
  }
}

uploadFileInput.addEventListener('change', () => {
  handleUploadFile(uploadFileInput.files[0]);
});

['dragenter', 'dragover'].forEach(eventName => {
  uploadDrop.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    uploadDrop.classList.add('dragover');
  });
});

['dragleave', 'drop'].forEach(eventName => {
  uploadDrop.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    uploadDrop.classList.remove('dragover');
  });
});

uploadDrop.addEventListener('drop', (event) => {
  const file = event.dataTransfer.files[0];
  handleUploadFile(file);
});

detectImageUrlBtn.addEventListener('click', async () => {
  const url = imageUrlInput.value.trim();
  if (!url) return;
  systemStatus.textContent = 'Processing';
  const expected = Number(expectedInput.value || 0);

  const res = await fetch(apiUrl('/detect_image_url'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      url,
      category: selectedCategory,
      expected_count: expected,
      conf: (confidenceSlider ? Number(confidenceSlider.value || 50) / 100 : null),
      iou: (overlapSlider ? Number(overlapSlider.value || 50) / 100 : null),
      label_mode: (labelModeSelect ? labelModeSelect.value : 'confidence')
    })
  });
  const data = await res.json();
  systemStatus.textContent = 'Idle';
  if (data.ok) {
    showPreviewImage(data.output_url);
    updateCurrentStats(data.category, data.detected, data.defect, data.good, expected);
    await refreshTotals();
    setActivePanel('dashboard');
  }
});

const sampleRow = document.querySelector('.sample-row');
const bindSampleImageClicks = (images) => {
  images.forEach(img => {
    img.addEventListener('click', () => handleSampleClick(img.dataset.sampleUrl || img.src));
  });
};

if (sampleImages.length) {
  bindSampleImageClicks(Array.from(sampleImages));
} else if (sampleRow) {
  fetch(apiUrl('/samples-json'))
    .then(res => res.json())
    .then(data => {
      if (!data.ok || !Array.isArray(data.images) || data.images.length === 0) {
        sampleRow.innerHTML = '<div class="sample-empty">No samples found in images/</div>';
        return;
      }
      sampleRow.innerHTML = '';
      data.images.slice(0, 10).forEach(item => {
        const img = document.createElement('img');
        img.src = normalizeUrl(item.url);
        img.alt = item.name || 'sample';
        img.dataset.sampleUrl = item.url;
        sampleRow.appendChild(img);
      });
      bindSampleImageClicks(Array.from(sampleRow.querySelectorAll('img')));
    })
    .catch(() => {
      sampleRow.innerHTML = '<div class="sample-empty">No samples found in images/</div>';
    });
}

downloadReport.addEventListener('click', () => {
  window.open(apiUrl('/report/latest'), '_blank');
});

downloadReportSecondary.addEventListener('click', () => {
  window.open(apiUrl('/report/latest'), '_blank');
});

function buildReportDateQuery() {
  if (!reportDate || !reportDate.value) return '';
  return `?date=${encodeURIComponent(reportDate.value)}`;
}

if (downloadDailyExcel) {
  downloadDailyExcel.addEventListener('click', () => {
    window.open(apiUrl(`/report/daily.xlsx${buildReportDateQuery()}`), '_blank');
  });
}

if (downloadDailyPdf) {
  downloadDailyPdf.addEventListener('click', () => {
    window.open(apiUrl(`/report/daily.pdf${buildReportDateQuery()}`), '_blank');
  });
}

setActivePanel('dashboard');
refreshTotals();

if (confidenceSlider && confidenceValue) {
  confidenceSlider.addEventListener('input', () => {
    confidenceValue.textContent = `${confidenceSlider.value}%`;
  });
  confidenceValue.textContent = `${confidenceSlider.value}%`;
}

if (overlapSlider && overlapValue) {
  overlapSlider.addEventListener('input', () => {
    overlapValue.textContent = `${overlapSlider.value}%`;
  });
  overlapValue.textContent = `${overlapSlider.value}%`;
}

async function initSupabaseClient() {
  if (!window.SUPABASE_URL || !window.SUPABASE_ANON_KEY) return;
  try {
    const { createClient } = await import('https://cdn.jsdelivr.net/npm/@supabase/supabase-js/+esm');
    window.supabase = createClient(window.SUPABASE_URL, window.SUPABASE_ANON_KEY, {
      auth: { persistSession: false }
    });
    const { error } = await window.supabase
      .from('detections')
      .select('id')
      .limit(1);
    if (error) {
      console.warn('Supabase read failed:', error.message);
    }
  } catch (err) {
    console.warn('Supabase init failed:', err.message);
  }
}

initSupabaseClient();

const urlParams = new URLSearchParams(window.location.search);
const sampleParam = urlParams.get('sample');
if (sampleParam) {
  handleSampleClick(sampleParam);
}


