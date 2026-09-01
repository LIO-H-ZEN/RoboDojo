#!/usr/bin/env python3
"""Review UI for object_attributes.json.

Extracts thumbnails from USDZ files lazily and serves a web page where you
can view each object's image, its extracted attributes, and edit them.

Usage:
  python3 scripts/internal/review_attributes_ui.py [--port PORT]

Then open http://localhost:PORT in your browser.
"""

import argparse
import json
import os
import shutil
import zipfile
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
RIGID_ROOT = Path(".cache/robodojo_assets_repo/Assets/Object/RoboDojo/Rigid")
ATTRIBUTES_PATH = Path("task/RoboDojo/config/object_attributes.json")
CACHE_DIR = Path("/tmp/robodojo_review_thumbnails")

# ---------------------------------------------------------------------------
# Extract thumbnail from USDZ (lazy)
# ---------------------------------------------------------------------------
def extract_thumbnail(category: str, variant: str = "00000") -> bytes | None:
    """Extract the largest texture image from a USDZ as JPEG bytes."""
    cache_path = CACHE_DIR / f"{category}_{variant}.jpg"
    if cache_path.exists():
        return cache_path.read_bytes()

    usdz_path = RIGID_ROOT / category / variant / "object.usdz"
    if not usdz_path.exists():
        return None

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(usdz_path) as z:
            # Find the largest JPEG texture
            textures = []
            for name in z.namelist():
                if name.startswith("SubUSDs/textures/"):
                    ext = name.rsplit(".", 1)[-1].lower()
                    if ext in ("jpg", "jpeg", "png"):
                        textures.append((name, z.getinfo(name).file_size))

            if not textures:
                return None

            textures.sort(key=lambda x: -x[1])
            best_name = textures[0][0]
            data = z.read(best_name)

            cache_path.write_bytes(data)
            return data
    except Exception as e:
        print(f"  [WARN] Failed to extract {category}/{variant}: {e}")
        return None


# ---------------------------------------------------------------------------
# Load / save attributes
# ---------------------------------------------------------------------------
def load_attrs() -> dict:
    return json.loads(ATTRIBUTES_PATH.read_text())

def save_attrs(attrs: dict):
    ATTRIBUTES_PATH.write_text(json.dumps(attrs, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# HTML page (single-page app, all inline)
# ---------------------------------------------------------------------------
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RoboDojo Object Attributes Review</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f5f5f7; color: #1d1d1f; padding: 20px; }
  h1 { font-size: 28px; font-weight: 600; margin-bottom: 8px; }
  .subtitle { color: #6e6e73; margin-bottom: 24px; font-size: 14px; }
  .stats { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
  .stat-card { background: white; border-radius: 12px; padding: 12px 20px;
               box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
  .stat-card .num { font-size: 24px; font-weight: 700; }
  .stat-card .label { font-size: 12px; color: #6e6e73; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
          gap: 16px; }
  .card { background: white; border-radius: 14px; overflow: hidden;
           box-shadow: 0 2px 8px rgba(0,0,0,0.08); transition: box-shadow 0.2s; }
  .card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.12); }
  .card-header { padding: 12px 16px; display: flex; justify-content: space-between;
                 align-items: center; border-bottom: 1px solid #f0f0f0; }
  .card-header .cat-name { font-weight: 600; font-size: 15px; }
  .card-header .badge { font-size: 11px; padding: 2px 8px; border-radius: 10px;
                        background: #e8e8ed; color: #6e6e73; }
  .card-body { display: flex; gap: 12px; padding: 12px; }
  .thumb { width: 120px; height: 120px; flex-shrink: 0; border-radius: 8px;
           overflow: hidden; background: #f0f0f0; display: flex;
           align-items: center; justify-content: center; }
  .thumb img { width: 100%; height: 100%; object-fit: cover; }
  .thumb .no-img { font-size: 11px; color: #999; text-align: center; padding: 8px; }
  .thumb .loading { font-size: 11px; color: #999; }
  .info { flex: 1; min-width: 0; }
  .info-row { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
  .info-row label { font-size: 12px; color: #6e6e73; width: 60px; flex-shrink: 0; }
  .info-row input, .info-row select { flex: 1; padding: 4px 8px; border: 1px solid #d2d2d7;
    border-radius: 6px; font-size: 13px; outline: none; }
  .info-row input:focus, .info-row select:focus { border-color: #007aff; box-shadow: 0 0 0 2px rgba(0,122,255,0.2); }
  .info-row .orig { font-size: 11px; color: #999; }
  .card-actions { padding: 8px 12px; border-top: 1px solid #f0f0f0;
                  display: flex; gap: 8px; justify-content: flex-end; }
  .btn { padding: 4px 14px; border-radius: 8px; border: none; font-size: 12px;
         font-weight: 500; cursor: pointer; transition: background 0.15s; }
  .btn-primary { background: #007aff; color: white; }
  .btn-primary:hover { background: #0062cc; }
  .btn-primary:disabled { background: #a0c4f0; cursor: default; }
  .btn-secondary { background: #e8e8ed; color: #1d1d1f; }
  .btn-secondary:hover { background: #d2d2d7; }
  .btn-success { background: #34c759; color: white; }
  .btn-success:hover { background: #28a745; }
  .btn-danger { background: #ff3b30; color: white; }
  .btn-danger:hover { background: #dc3545; }
  .toast { position: fixed; bottom: 20px; right: 20px; padding: 10px 20px;
           border-radius: 10px; color: white; font-size: 13px; font-weight: 500;
           opacity: 0; transition: opacity 0.3s; z-index: 1000; }
  .toast.show { opacity: 1; }
  .toast.success { background: #34c759; }
  .toast.error { background: #ff3b30; }
  .filter-bar { display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }
  .filter-bar input { padding: 6px 12px; border: 1px solid #d2d2d7;
    border-radius: 8px; font-size: 13px; flex: 1; min-width: 200px; }
  .filter-bar select { padding: 6px 12px; border: 1px solid #d2d2d7;
    border-radius: 8px; font-size: 13px; background: white; }
  .unsaved-badge { background: #ff9500; color: white; font-size: 10px;
                   padding: 1px 6px; border-radius: 8px; margin-left: 6px; }
  .status-ok { color: #34c759; }
  .status-needs-review { color: #ff9500; }
  .save-all-btn { margin-left: auto; }
  .color-dot { display: inline-block; width: 12px; height: 12px; border-radius: 50%;
               margin-right: 4px; vertical-align: middle; border: 1px solid #ddd; }
</style>
</head>
<body>

<h1> RoboDojo Object Attributes Review</h1>
<p class="subtitle" id="subtitle">Loading...</p>

<div class="stats" id="stats"></div>

<div class="filter-bar">
  <input type="text" id="search" placeholder="Search category or common name..." oninput="render()">
  <select id="filterColor" onchange="render()">
    <option value="">All colors</option>
  </select>
  <select id="filterShape" onchange="render()">
    <option value="">All shapes</option>
  </select>
  <select id="filterStatus" onchange="render()">
    <option value="">All status</option>
    <option value="needs_review">Needs review</option>
    <option value="ok">Reviewed OK</option>
  </select>
  <button class="btn btn-success save-all-btn" onclick="saveAll()">Save All Changes</button>
</div>

<div class="grid" id="grid"></div>
<div id="toast" class="toast"></div>

<script>
// =========================================================================
// Data
// =========================================================================
let allAttrs = {};
let originalAttrs = {};
let reviewStatus = {};

// =========================================================================
// Init
// =========================================================================
async function init() {
  const attrsResp = await fetch('/api/attrs');
  allAttrs = await attrsResp.json();
  originalAttrs = JSON.parse(JSON.stringify(allAttrs));

  // Load review status from localStorage
  try {
    const saved = localStorage.getItem('reviewStatus');
    if (saved) reviewStatus = JSON.parse(saved);
  } catch(e) {}

  // Populate filter dropdowns
  const colors = new Set();
  const shapes = new Set();
  for (const attr of Object.values(allAttrs)) {
    if (attr.color) colors.add(attr.color);
    if (attr.shape) shapes.add(attr.shape);
  }
  const colorOrder = ['black','white','gray','brown','red','green','blue','yellow','orange','purple','pink','gold','silver','beige','teal','multicolor','unknown'];
  const selColor = document.getElementById('filterColor');
  for (const c of colorOrder) {
    if (colors.has(c)) {
      const opt = document.createElement('option');
      opt.value = c; opt.textContent = c;
      selColor.appendChild(opt);
    }
  }
  const shapeOrder = ['rectangular','cylindrical','round','cube','irregular','figurine','bell-shaped','cup-shaped','curved','donut-shaped','L-shaped','triangular','oval','teardrop','bottle-shaped','spiral','crown-shaped','origami','beaded','pouch','shoe','slim','letter-shaped','bowl','hexagonal','square','hourglass','box-shaped','trapezoid','unknown'];
  const selShape = document.getElementById('filterShape');
  for (const s of shapeOrder) {
    if (shapes.has(s)) {
      const opt = document.createElement('option');
      opt.value = s; opt.textContent = s;
      selShape.appendChild(opt);
    }
  }

  updateStats();
  render();
}

// =========================================================================
// Color swatch helper
// =========================================================================
function colorSwatch(c) {
  const map = {
    'black': '#222', 'white': '#f5f5f5', 'gray': '#999', 'brown': '#8B4513',
    'red': '#ff3b30', 'green': '#34c759', 'blue': '#007aff', 'yellow': '#ffcc00',
    'orange': '#ff9500', 'purple': '#af52de', 'pink': '#ff2d55',
    'gold': '#d4a017', 'silver': '#c0c0c0', 'beige': '#f5f5dc',
    'teal': '#30b0c7', 'multicolor': 'linear-gradient(90deg,red,orange,yellow,green,blue,purple)',
    'unknown': '#eee'
  };
  return map[c] || '#eee';
}

// =========================================================================
// Render
// =========================================================================
function render() {
  const search = document.getElementById('search').value.toLowerCase();
  const filterColor = document.getElementById('filterColor').value;
  const filterShape = document.getElementById('filterShape').value;
  const filterStatus = document.getElementById('filterStatus').value;

  const entries = Object.entries(allAttrs).filter(([cat, attr]) => {
    if (search && !cat.toLowerCase().includes(search) &&
        !attr.common_name.toLowerCase().includes(search)) return false;
    if (filterColor && attr.color !== filterColor) return false;
    if (filterShape && attr.shape !== filterShape) return false;
    if (filterStatus === 'needs_review' && reviewStatus[cat] === 'ok') return false;
    if (filterStatus === 'ok' && reviewStatus[cat] !== 'ok') return false;
    return true;
  });

  const grid = document.getElementById('grid');
  grid.innerHTML = entries.map(([cat, attr]) => renderCard(cat, attr)).join('');
  updateStats();
}

function renderCard(cat, attr) {
  const orig = originalAttrs[cat] || {};
  const hasChanges = JSON.stringify(attr) !== JSON.stringify(orig);
  const status = reviewStatus[cat];

  const colorOptions = ['black','white','gray','brown','red','green','blue','yellow','orange','purple','pink','gold','silver','beige','teal','multicolor','unknown'];
  const shapeOptions = ['rectangular','cylindrical','round','cube','irregular','figurine','bell-shaped','cup-shaped','curved','donut-shaped','L-shaped','triangular','oval','teardrop','bottle-shaped','spiral','crown-shaped','origami','beaded','pouch','shoe','slim','letter-shaped','bowl','hexagonal','square','hourglass','box-shaped','trapezoid','unknown'];

  return `
    <div class="card" data-category="${cat}">
      <div class="card-header">
        <span class="cat-name">${cat} <span class="badge">${escHtml(attr.common_name)}</span></span>
        <span>
          ${status === 'ok' ? '<span class="status-ok">Reviewed</span>' : '<span class="status-needs-review">Needs review</span>'}
          ${hasChanges ? '<span class="unsaved-badge">unsaved</span>' : ''}
        </span>
      </div>
      <div class="card-body">
        <div class="thumb">
          <img src="/api/thumb/${cat}" alt="${cat}"
               onerror="this.parentNode.innerHTML='<div class=\\'no-img\\'>No preview</div>'"
               loading="lazy">
        </div>
        <div class="info">
          <div class="info-row">
            <label>Color</label>
            <span class="color-dot" style="background:${colorSwatch(attr.color)}"></span>
            <select onchange="updateAttr('${cat}','color',this.value)">
              ${colorOptions.map(c => `<option value="${c}" ${attr.color === c ? 'selected' : ''}>${c}</option>`).join('')}
            </select>
            ${orig.color !== attr.color ? `<span class="orig">was: ${orig.color}</span>` : ''}
          </div>
          <div class="info-row">
            <label>Shape</label>
            <select onchange="updateAttr('${cat}','shape',this.value)">
              ${shapeOptions.map(s => `<option value="${s}" ${attr.shape === s ? 'selected' : ''}>${s}</option>`).join('')}
            </select>
            ${orig.shape !== attr.shape ? `<span class="orig">was: ${orig.shape}</span>` : ''}
          </div>
          <div class="info-row">
            <label>Name</label>
            <input type="text" value="${escHtml(attr.common_name)}"
                   onchange="updateAttr('${cat}','common_name',this.value)">
          </div>
        </div>
      </div>
      <div class="card-actions">
        <button class="btn btn-primary" onclick="markReviewed('${cat}')">Mark OK</button>
        <button class="btn btn-secondary" onclick="resetCard('${cat}')">Reset</button>
      </div>
    </div>
  `;
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// =========================================================================
// Actions
// =========================================================================
function updateAttr(cat, key, value) {
  allAttrs[cat][key] = value;
  if (reviewStatus[cat] === 'ok') {
    delete reviewStatus[cat];
    localStorage.setItem('reviewStatus', JSON.stringify(reviewStatus));
  }
  render();
}

function resetCard(cat) {
  allAttrs[cat] = JSON.parse(JSON.stringify(originalAttrs[cat]));
  render();
}

function markReviewed(cat) {
  reviewStatus[cat] = 'ok';
  localStorage.setItem('reviewStatus', JSON.stringify(reviewStatus));
  render();
  showToast('Marked as reviewed', 'success');
}

async function saveAll() {
  const btn = document.querySelector('.save-all-btn');
  btn.disabled = true;
  btn.textContent = 'Saving...';

  try {
    const resp = await fetch('/api/save', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(allAttrs),
    });
    const result = await resp.json();
    if (result.ok) {
      originalAttrs = JSON.parse(JSON.stringify(allAttrs));
      showToast('Saved successfully!', 'success');
      render();
    } else {
      showToast('Error: ' + (result.error || 'unknown'), 'error');
    }
  } catch(e) {
    showToast('Network error: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Save All Changes';
  }
}

function updateStats() {
  const total = Object.keys(allAttrs).length;
  const reviewed = Object.values(reviewStatus).filter(s => s === 'ok').length;
  const modified = Object.keys(allAttrs).filter(c =>
    JSON.stringify(allAttrs[c]) !== JSON.stringify(originalAttrs[c])
  ).length;

  document.getElementById('subtitle').textContent =
    `${total} categories - ${reviewed} reviewed - ${modified} unsaved changes`;

  document.getElementById('stats').innerHTML = `
    <div class="stat-card"><div class="num">${total}</div><div class="label">Total</div></div>
    <div class="stat-card"><div class="num">${reviewed}</div><div class="label">Reviewed</div></div>
    <div class="stat-card"><div class="num">${total - reviewed}</div><div class="label">Needs Review</div></div>
    <div class="stat-card"><div class="num">${modified}</div><div class="label">Unsaved Changes</div></div>
  `;
}

function showToast(msg, type) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'toast ' + type + ' show';
  setTimeout(() => t.classList.remove('show'), 2500);
}

// =========================================================================
// Start
// =========================================================================
init();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class ReviewHandler(SimpleHTTPRequestHandler):
    attrs = None
    thumbnail_cache = {}  # category -> bool (has thumbnail)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif path == "/api/attrs":
            self.send_json(self.attrs)

        elif path.startswith("/api/thumb/"):
            cat = path[len("/api/thumb/"):]
            data = extract_thumbnail(cat)
            if data:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "max-age=86400")
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_error(404)

        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/save":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                new_attrs = json.loads(body)
                save_attrs(new_attrs)
                self.attrs = new_attrs
                self.send_json({"ok": True})
            except Exception as e:
                self.send_json({"ok": False, "error": str(e)}, status=500)
        else:
            self.send_error(404)

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args()

    # Clean cache on start
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)

    print("=" * 60)
    print("  RoboDojo - Object Attributes Review UI")
    print("=" * 60)

    # Load attributes
    attrs = load_attrs()
    ReviewHandler.attrs = attrs
    print(f"  Loaded {len(attrs)} categories from {ATTRIBUTES_PATH}")

    # Start server (thumbnails extracted lazily on request)
    server = HTTPServer(("0.0.0.0", args.port), ReviewHandler)
    print(f"\n  Open http://localhost:{args.port} in your browser")
    print(f"  Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down...")
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
        server.server_close()


if __name__ == "__main__":
    main()
