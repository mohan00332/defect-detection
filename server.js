import express from 'express';
import multer from 'multer';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';
import crypto from 'crypto';
import { Readable } from 'stream';
import { createClient } from '@supabase/supabase-js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

const UPLOAD_DIR = path.join(__dirname, 'uploads');
const OUTPUT_DIR = path.join(__dirname, 'outputs');
const REPORT_DIR = path.join(__dirname, 'reports');
const IMAGES_DIR = path.join(__dirname, 'images');
const STATS_PATH = path.join(__dirname, 'stats.json');
const LOG_PATH = path.join(REPORT_DIR, 'detections_log.csv');
const LIVE_PROXY_URL = process.env.LIVE_PROXY_URL || 'http://localhost:5000';
const SUPABASE_URL = process.env.SUPABASE_URL || null;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || null;
const SUPABASE_ANON_KEY = process.env.SUPABASE_ANON_KEY || null;
const SUPABASE_TABLE = process.env.SUPABASE_TABLE || 'detections';

const supabaseKey = SUPABASE_SERVICE_ROLE_KEY || SUPABASE_ANON_KEY || null;
const supabase = SUPABASE_URL && supabaseKey
  ? createClient(SUPABASE_URL, supabaseKey, { auth: { persistSession: false } })
  : null;

function ensureLogHeader() {
  if (fs.existsSync(LOG_PATH)) return;
  fs.writeFileSync(LOG_PATH, 'timestamp,category,detected,defect,good\n');
}

function appendLog({ category, detected, defect, good, timestamp }) {
  ensureLogHeader();
  const ts = timestamp || new Date().toISOString();
  const line = `${ts},${category},${detected},${defect},${good}\n`;
  fs.appendFileSync(LOG_PATH, line);
  return ts;
}

async function logToSupabase(entry) {
  if (!supabase) return;
  try {
    const { error } = await supabase.from(SUPABASE_TABLE).insert([entry]);
    if (error) {
      console.warn('Supabase insert failed:', error.message);
    }
  } catch (err) {
    console.warn('Supabase insert error:', err.message);
  }
}

function recordDetection({ category, detected, defect, good, mode, expected }) {
  const timestamp = new Date().toISOString();
  appendLog({ category, detected, defect, good, timestamp });
  logToSupabase({
    timestamp,
    category,
    detected,
    defect,
    good,
    mode: mode || null,
    expected_count: Number.isFinite(expected) ? expected : null
  });
}

for (const dir of [UPLOAD_DIR, OUTPUT_DIR, REPORT_DIR, IMAGES_DIR]) {
  fs.mkdirSync(dir, { recursive: true });
}

app.use(express.json({ limit: '5mb' }));
app.use(express.urlencoded({ extended: true }));

app.use('/static', express.static(path.join(__dirname, 'static')));
app.use('/images', express.static(IMAGES_DIR));
app.use('/outputs', express.static(OUTPUT_DIR));
app.use('/reports', express.static(REPORT_DIR));

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

const upload = multer({ dest: UPLOAD_DIR });

const emptyStats = () => ({
  Nut: { detected: 0, defect: 0, good: 0 },
  Bolt: { detected: 0, defect: 0, good: 0 },
  Gear: { detected: 0, defect: 0, good: 0 }
});

let stats = emptyStats();
let statsMeta = { last_reset: null };

function loadStats() {
  if (!fs.existsSync(STATS_PATH)) return;
  try {
    const data = JSON.parse(fs.readFileSync(STATS_PATH, 'utf-8'));
    if (data && typeof data === 'object') {
      if (data._meta) statsMeta.last_reset = data._meta.last_reset || null;
      for (const key of Object.keys(stats)) {
        if (data[key]) {
          stats[key].detected = Number(data[key].detected || 0);
          stats[key].defect = Number(data[key].defect || 0);
          stats[key].good = Number(data[key].good || 0);
        }
      }
    }
  } catch {
    // ignore
  }
}

function saveStats() {
  const payload = { _meta: statsMeta, ...stats };
  fs.writeFileSync(STATS_PATH, JSON.stringify(payload));
}

function todayIso() {
  return new Date().toISOString().slice(0, 10);
}

function maybeResetStats() {
  const today = todayIso();
  if (statsMeta.last_reset === today) return;
  stats = emptyStats();
  statsMeta.last_reset = today;
  saveStats();
}

function updateStats(category, detected, defect) {
  maybeResetStats();
  const good = Math.max(detected - defect, 0);
  if (!stats[category]) return good;
  stats[category].detected += detected;
  stats[category].defect += defect;
  stats[category].good += good;
  saveStats();
  return good;
}

function updateStatsMany(perCategory) {
  for (const [cat, vals] of Object.entries(perCategory)) {
    updateStats(cat, vals.detected, vals.defect);
  }
}

loadStats();
if (!statsMeta.last_reset) {
  statsMeta.last_reset = todayIso();
  saveStats();
} else {
  maybeResetStats();
}

function runWorker(payload) {
  return new Promise((resolve, reject) => {
    const args = [
      path.join(__dirname, 'worker.py'),
      '--payload',
      JSON.stringify(payload)
    ];
    const proc = spawn('python', args, { cwd: __dirname });
    let out = '';
    let err = '';
    proc.stdout.on('data', (d) => { out += d.toString(); });
    proc.stderr.on('data', (d) => { err += d.toString(); });
    proc.on('close', (code) => {
      if (code !== 0) {
        return reject(new Error(err || `Worker failed with code ${code}`));
      }
      try {
        const data = JSON.parse(out.trim());
        resolve(data);
      } catch (e) {
        reject(new Error(`Invalid worker output: ${out}`));
      }
    });
  });
}

function normalizeOutputUrl(relPath) {
  if (!relPath) return null;
  if (relPath.startsWith('/')) return relPath;
  return `/${relPath}`;
}

app.get('/api/health', (req, res) => {
  res.json({ ok: true });
});

app.get('/api/stats', (req, res) => {
  maybeResetStats();
  res.json({ ok: true, stats });
});

app.post('/api/reset_stats', (req, res) => {
  stats = emptyStats();
  statsMeta.last_reset = todayIso();
  saveStats();
  res.json({ ok: true, stats });
});

app.get('/api/analytics/summary', (req, res) => {
  maybeResetStats();
  const categories = stats;
  const totals = {
    detected: Object.values(categories).reduce((a, v) => a + v.detected, 0),
    defect: Object.values(categories).reduce((a, v) => a + v.defect, 0),
    good: Object.values(categories).reduce((a, v) => a + v.good, 0)
  };
  const trend = [];
  res.json({ ok: true, totals, categories, trend });
});

app.get('/api/samples-json', (req, res) => {
  const images = [];
  try {
    const files = fs.readdirSync(IMAGES_DIR).sort();
    for (const name of files) {
      const ext = path.extname(name).toLowerCase();
      if (['.jpg', '.jpeg', '.png'].includes(ext)) {
        images.push({ name, url: `/images/${name}` });
      }
    }
  } catch {
    // ignore
  }
  res.json({ ok: true, images });
});

app.get('/api/samples-view', (req, res) => {
  const images = [];
  try {
    const files = fs.readdirSync(IMAGES_DIR).sort();
    for (const name of files) {
      const ext = path.extname(name).toLowerCase();
      if (['.jpg', '.jpeg', '.png'].includes(ext)) {
        images.push(name);
      }
    }
  } catch {
    // ignore
  }
  const cards = images.map(name => `
    <div class="samples-card">
      <img src="/images/${name}" alt="${name}" />
      <div class="samples-name">${name}</div>
    </div>
  `).join('');
  res.send(`
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <title>Samples</title>
      <link rel="stylesheet" href="/static/css/styles.css" />
    </head>
    <body>
      <div class="samples-page">
        <div class="samples-header">
          <div class="samples-title">Sample Images</div>
        </div>
        <div class="samples-grid">
          ${cards || '<div class="sample-empty">No samples found in images/</div>'}
        </div>
      </div>
    </body>
    </html>
  `);
});

app.post('/api/detect_image', upload.single('image'), async (req, res) => {
  try {
    const category = req.body.category || 'Nut';
    const expected = Number(req.body.expected_count || 0);
    const conf = req.body.conf ? Number(req.body.conf) : null;
    const iou = req.body.iou ? Number(req.body.iou) : null;
    const labelMode = req.body.label_mode || 'confidence';

    if (!req.file) return res.status(400).json({ ok: false, error: 'No image uploaded' });

    const payload = {
      mode: 'image',
      path: req.file.path,
      category,
      expected,
      conf,
      iou,
      label_mode: labelMode,
      outputs_dir: OUTPUT_DIR
    };

    const result = await runWorker(payload);
    if (!result.ok) return res.status(400).json(result);

    if (category === 'All' && result.per_category) {
      updateStatsMany(result.per_category);
    } else {
      updateStats(category, result.detected, result.defect);
    }

    recordDetection({
      category,
      detected: result.detected,
      defect: result.defect,
      good: result.good,
      mode: 'image',
      expected: Number(req.body.expected_count || 0)
    });
    res.json({
      ok: true,
      category,
      detected: result.detected,
      defect: result.defect,
      good: result.good,
      output_url: normalizeOutputUrl(result.output_url)
    });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

app.post('/api/detect_video', upload.single('video'), async (req, res) => {
  try {
    const category = req.body.category || 'Nut';
    const expected = Number(req.body.expected_count || 0);

    if (!req.file) return res.status(400).json({ ok: false, error: 'No video uploaded' });

    const payload = {
      mode: 'video',
      path: req.file.path,
      category,
      expected,
      outputs_dir: OUTPUT_DIR
    };

    const result = await runWorker(payload);
    if (!result.ok) return res.status(400).json(result);

    if (category === 'All' && result.per_category) {
      updateStatsMany(result.per_category);
    } else {
      updateStats(category, result.detected, result.defect);
    }

    recordDetection({
      category,
      detected: result.detected,
      defect: result.defect,
      good: result.good,
      mode: 'video',
      expected: Number(req.body.expected_count || 0)
    });
    res.json({
      ok: true,
      category,
      detected: result.detected,
      defect: result.defect,
      good: result.good,
      output_url: normalizeOutputUrl(result.output_url)
    });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

app.post('/api/upload', upload.single('file'), async (req, res) => {
  try {
    const category = req.body.category || 'Nut';
    const expected = Number(req.body.expected_count || 0);
    const conf = req.body.conf ? Number(req.body.conf) : null;
    const iou = req.body.iou ? Number(req.body.iou) : null;
    const labelMode = req.body.label_mode || 'confidence';

    if (!req.file) return res.status(400).json({ ok: false, error: 'No file uploaded' });

    const ext = path.extname(req.file.originalname || '').toLowerCase();
    const mime = req.file.mimetype || '';
    const isVideo = ['.mp4', '.avi', '.mov', '.mkv'].includes(ext) || mime.startsWith('video/');

    const payload = {
      mode: isVideo ? 'video' : 'image',
      path: req.file.path,
      category,
      expected,
      conf,
      iou,
      label_mode: labelMode,
      outputs_dir: OUTPUT_DIR
    };

    const result = await runWorker(payload);
    if (!result.ok) return res.status(400).json(result);

    if (category === 'All' && result.per_category) {
      updateStatsMany(result.per_category);
    } else {
      updateStats(category, result.detected, result.defect);
    }

    recordDetection({
      category,
      detected: result.detected,
      defect: result.defect,
      good: result.good,
      mode: 'upload',
      expected: Number(req.body.expected_count || 0)
    });
    res.json({
      ok: true,
      category,
      detected: result.detected,
      defect: result.defect,
      good: result.good,
      output_url: normalizeOutputUrl(result.output_url)
    });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

app.post('/api/detect_image_url', async (req, res) => {
  try {
    const { url, category = 'Nut', expected_count, conf, iou, label_mode } = req.body || {};
    if (!url) return res.status(400).json({ ok: false, error: 'No URL provided' });

    const resp = await fetch(url);
    if (!resp.ok) return res.status(400).json({ ok: false, error: 'Failed to fetch URL' });
    const buffer = Buffer.from(await resp.arrayBuffer());
    const ext = path.extname(new URL(url).pathname) || '.jpg';
    const filename = `url_${crypto.randomUUID()}${ext}`;
    const filePath = path.join(UPLOAD_DIR, filename);
    fs.writeFileSync(filePath, buffer);

    const payload = {
      mode: 'image',
      path: filePath,
      category,
      expected: Number(expected_count || 0),
      conf: conf != null ? Number(conf) : null,
      iou: iou != null ? Number(iou) : null,
      label_mode: label_mode || 'confidence',
      outputs_dir: OUTPUT_DIR
    };

    const result = await runWorker(payload);
    if (!result.ok) return res.status(400).json(result);

    if (category === 'All' && result.per_category) {
      updateStatsMany(result.per_category);
    } else {
      updateStats(category, result.detected, result.defect);
    }

    recordDetection({
      category,
      detected: result.detected,
      defect: result.defect,
      good: result.good,
      mode: 'url',
      expected: Number(req.body.expected_count || 0)
    });
    res.json({
      ok: true,
      category,
      detected: result.detected,
      defect: result.defect,
      good: result.good,
      output_url: normalizeOutputUrl(result.output_url)
    });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

async function proxyLiveJson(req, res, path, method = 'POST') {
  if (!LIVE_PROXY_URL) {
    res.status(501).json({ ok: false, error: 'Live camera proxy not configured' });
    return;
  }
  try {
    const url = new URL(path, LIVE_PROXY_URL);
    const upstream = await fetch(url, {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: method === 'GET' ? undefined : JSON.stringify(req.body || {})
    });
    const text = await upstream.text();
    const contentType = upstream.headers.get('content-type') || 'application/json';
    res.status(upstream.status).type(contentType).send(text);
  } catch (err) {
    res.status(502).json({ ok: false, error: 'Live camera backend not reachable', detail: err.message });
  }
}

app.post('/api/start_camera', (req, res) => {
  proxyLiveJson(req, res, '/start_camera', 'POST');
});

app.post('/api/stop_camera', (req, res) => {
  proxyLiveJson(req, res, '/stop_camera', 'POST');
});

app.get('/api/live_stats', (req, res) => {
  proxyLiveJson(req, res, '/live_stats', 'GET');
});

app.get('/api/video_feed', async (req, res) => {
  if (!LIVE_PROXY_URL) {
    res.status(501).send('Live camera proxy not configured');
    return;
  }
  try {
    const url = new URL('/video_feed', LIVE_PROXY_URL);
    for (const [key, value] of Object.entries(req.query || {})) {
      if (value !== undefined) url.searchParams.set(key, String(value));
    }
    const upstream = await fetch(url, { method: 'GET' });
    res.status(upstream.status);
    upstream.headers.forEach((value, key) => {
      res.setHeader(key, value);
    });
    if (!upstream.body) {
      res.end();
      return;
    }
    const nodeStream = Readable.fromWeb(upstream.body);
    nodeStream.pipe(res);
  } catch (err) {
    res.status(502).send(`Live camera backend not reachable: ${err.message}`);
  }
});

app.get('/api/report/latest', async (req, res) => {
  try {
    const payload = {
      mode: 'report_latest',
      log_path: LOG_PATH,
      reports_dir: REPORT_DIR
    };
    const result = await runWorker(payload);
    if (!result.ok) return res.status(404).json(result);
    return res.sendFile(result.path);
  } catch (err) {
    return res.status(500).json({ ok: false, error: err.message });
  }
});

app.get('/api/report/daily.xlsx', async (req, res) => {
  try {
    const payload = {
      mode: 'report_daily_xlsx',
      log_path: LOG_PATH,
      reports_dir: REPORT_DIR,
      date: req.query.date || null
    };
    const result = await runWorker(payload);
    if (!result.ok) return res.status(404).json(result);
    return res.sendFile(result.path);
  } catch (err) {
    return res.status(500).json({ ok: false, error: err.message });
  }
});

app.get('/api/report/daily.pdf', async (req, res) => {
  try {
    const payload = {
      mode: 'report_daily_pdf',
      log_path: LOG_PATH,
      reports_dir: REPORT_DIR,
      date: req.query.date || null
    };
    const result = await runWorker(payload);
    if (!result.ok) return res.status(404).json(result);
    return res.sendFile(result.path);
  } catch (err) {
    return res.status(500).json({ ok: false, error: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});




