#!/usr/bin/env python3
"""Generate a standalone review HTML file (no server needed).

Usage:
  python3 scripts/internal/generate_review_html.py

Output:  scripts/internal/object_attributes_review.html
"""

import base64
import io
import json
import zipfile
from pathlib import Path

RIGID_ROOT = Path(".cache/robodojo_assets_repo/Assets/Object/RoboDojo/Rigid")
ATTRIBUTES_PATH = Path("task/RoboDojo/config/object_attributes.json")
OUTPUT_PATH = Path("scripts/internal/object_attributes_review.html")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def extract_thumbnail_b64(category, variant="00000"):
    usdz_path = RIGID_ROOT / category / variant / "object.usdz"
    if not usdz_path.exists():
        return None
    try:
        with zipfile.ZipFile(usdz_path) as z:
            textures = [
                (name, z.getinfo(name).file_size)
                for name in z.namelist()
                if name.startswith("SubUSDs/textures/")
                and name.rsplit(".", 1)[-1].lower() in ("jpg", "jpeg", "png")
            ]
            if not textures:
                return None
            textures.sort(key=lambda x: -x[1])
            best_name = textures[0][0]
            data = z.read(best_name)
            if HAS_PIL:
                img = Image.open(io.BytesIO(data))
                img.thumbnail((120, 120), Image.LANCZOS)
                buf = io.BytesIO()
                img.convert("RGB").save(buf, format="JPEG", quality=80)
                data = buf.getvalue()
            b64 = base64.b64encode(data).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


def generate():
    print("Loading attributes...")
    attrs = json.loads(ATTRIBUTES_PATH.read_text())
    categories = sorted(attrs.keys())

    print(f"Extracting thumbnails for {len(categories)} categories...")
    thumbnails = {}
    ok_count = 0
    for i, cat in enumerate(categories):
        b64 = extract_thumbnail_b64(cat)
        if b64:
            thumbnails[cat] = b64
            ok_count += 1
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(categories)}] {ok_count} thumbnails")
    print(f"  Done: {ok_count}/{len(categories)} have thumbnails")

    data_json = json.dumps({"attrs": attrs, "thumbnails": thumbnails})

    # Build the HTML template
    html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RoboDojo Object Attributes Review</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f5f5f7;color:#1d1d1f;padding:20px}
h1{font-size:28px;font-weight:600;margin-bottom:8px}
.subtitle{color:#6e6e73;margin-bottom:24px;font-size:14px}
.stats{display:flex;gap:16px;margin-bottom:24px;flex-wrap:wrap}
.stat-card{background:#fff;border-radius:12px;padding:12px 20px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.stat-card .num{font-size:24px;font-weight:700}
.stat-card .label{font-size:12px;color:#6e6e73}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px}
.card{background:#fff;border-radius:14px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,.08)}
.card-header{padding:12px 16px;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #f0f0f0}
.card-header .cat-name{font-weight:600;font-size:15px}
.card-header .badge{font-size:11px;padding:2px 8px;border-radius:10px;background:#e8e8ed;color:#6e6e73}
.card-body{display:flex;gap:12px;padding:12px}
.thumb{width:120px;height:120px;flex-shrink:0;border-radius:8px;overflow:hidden;background:#f0f0f0;display:flex;align-items:center;justify-content:center}
.thumb img{width:100%;height:100%;object-fit:cover}
.thumb .no-img{font-size:11px;color:#999;text-align:center;padding:8px}
.info{flex:1;min-width:0}
.info-row{display:flex;align-items:center;gap:8px;margin-bottom:6px}
.info-row label{font-size:12px;color:#6e6e73;width:60px;flex-shrink:0}
.info-row select,.info-row input{flex:1;padding:4px 8px;border:1px solid #d2d2d7;border-radius:6px;font-size:13px;outline:none}
.info-row select:focus,.info-row input:focus{border-color:#007aff;box-shadow:0 0 0 2px rgba(0,122,255,.2)}
.info-row .orig{font-size:11px;color:#999}
.card-actions{padding:8px 12px;border-top:1px solid #f0f0f0;display:flex;gap:8px;justify-content:flex-end}
.btn{padding:4px 14px;border-radius:8px;border:none;font-size:12px;font-weight:500;cursor:pointer}
.btn-primary{background:#007aff;color:#fff}
.btn-primary:hover{background:#0062cc}
.btn-secondary{background:#e8e8ed;color:#1d1d1f}
.btn-secondary:hover{background:#d2d2d7}
.btn-success{background:#34c759;color:#fff}
.btn-success:hover{background:#28a745}
.toast{position:fixed;bottom:20px;right:20px;padding:10px 20px;border-radius:10px;color:#fff;font-size:13px;font-weight:500;opacity:0;transition:opacity .3s;z-index:1000}
.toast.show{opacity:1}
.toast.success{background:#34c759}
.toast.error{background:#ff3b30}
.filter-bar{display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap}
.filter-bar input{padding:6px 12px;border:1px solid #d2d2d7;border-radius:8px;font-size:13px;flex:1;min-width:200px}
.filter-bar select{padding:6px 12px;border:1px solid #d2d2d7;border-radius:8px;font-size:13px;background:#fff}
.unsaved-badge{background:#ff9500;color:#fff;font-size:10px;padding:1px 6px;border-radius:8px;margin-left:6px}
.status-ok{color:#34c759}
.status-needs-review{color:#ff9500}
.color-dot{display:inline-block;width:14px;height:14px;border-radius:50%;margin-right:4px;vertical-align:middle;border:1px solid #ddd;flex-shrink:0}
.color-row{display:flex;align-items:center;gap:4px;flex:1}
</style>
</head>
<body>

<h1> &#x1F50D; RoboDojo Object Attributes Review</h1>
<p class="subtitle" id="subtitle">Loading...</p>

<div class="stats" id="stats"></div>

<div class="filter-bar">
  <input type="text" id="search" placeholder="Search..." oninput="render()">
  <select id="filterColor" onchange="render()"><option value="">All colors</option></select>
  <select id="filterShape" onchange="render()"><option value="">All shapes</option></select>
  <select id="filterStatus" onchange="render()">
    <option value="">All status</option>
    <option value="needs_review">Needs review</option>
    <option value="ok">Reviewed OK</option>
  </select>
  <button class="btn btn-success" onclick="saveAll()">&#x1F4E5; Download JSON</button>
</div>

<div class="grid" id="grid">Loading...</div>
<div id="toast" class="toast"></div>

<script>
var DATA = """ + data_json + r""";

var A = DATA.attrs;
var OA = JSON.parse(JSON.stringify(A));
var T = DATA.thumbnails;
var RS = {};

var COLORS = ["black","white","gray","brown","red","green","blue","yellow","orange","purple","pink","gold","silver","beige","teal","multicolor","unknown"];
var SHAPES = ["rectangular","cylindrical","round","cube","irregular","figurine","bell-shaped","cup-shaped","curved","donut-shaped","L-shaped","triangular","oval","teardrop","bottle-shaped","spiral","crown-shaped","origami","beaded","pouch","shoe","slim","letter-shaped","bowl","hexagonal","square","hourglass","box-shaped","trapezoid","unknown"];

var CMAP = {"black":"#222","white":"#f5f5f5","gray":"#999","brown":"#8B4513","red":"#ff3b30","green":"#34c759","blue":"#007aff","yellow":"#ffcc00","orange":"#ff9500","purple":"#af52de","pink":"#ff2d55","gold":"#d4a017","silver":"#c0c0c0","beige":"#f5f5dc","teal":"#30b0c7","multicolor":"linear-gradient(90deg,red,orange,yellow,green,blue,purple)","unknown":"#eee"};

try { var s = localStorage.getItem('rs'); if(s) RS = JSON.parse(s); } catch(e) {}

function esc(s) { return s.replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;'); }

function init() {
  var sc = document.getElementById('filterColor');
  var ss = document.getElementById('filterShape');
  var cs = new Set(), ss2 = new Set();
  for(var k in A) { if(A[k].color) cs.add(A[k].color); if(A[k].shape) ss2.add(A[k].shape); }
  COLORS.forEach(function(c){ if(cs.has(c)){ var o=document.createElement('option'); o.value=c; o.textContent=c; sc.appendChild(o); } });
  SHAPES.forEach(function(s){ if(ss2.has(s)){ var o=document.createElement('option'); o.value=s; o.textContent=s; ss.appendChild(o); } });
  updateStats();
  render();
}

function render() {
  var q = document.getElementById('search').value.toLowerCase();
  var fc = document.getElementById('filterColor').value;
  var fs = document.getElementById('filterShape').value;
  var fst = document.getElementById('filterStatus').value;

  var keys = [];
  for(var k in A) {
    var a = A[k];
    if(q && k.indexOf(q)===-1 && a.common_name.toLowerCase().indexOf(q)===-1) continue;
    if(fc && a.color!==fc) continue;
    if(fs && a.shape!==fs) continue;
    if(fst==='needs_review' && RS[k]==='ok') continue;
    if(fst==='ok' && RS[k]!=='ok') continue;
    keys.push(k);
  }
  keys.sort();

  var html = '';
  for(var i=0;i<keys.length;i++) { html += makeCard(keys[i], A[keys[i]]); }
  document.getElementById('grid').innerHTML = html || '<p style="color:#999;padding:20px">No matching categories</p>';
  updateStats();
}

function makeCard(cat, attr) {
  var orig = OA[cat] || {};
  var changed = JSON.stringify(attr)!==JSON.stringify(orig);
  var st = RS[cat];
  var thumb = T[cat];

  var colorOpts = '';
  for(var i=0;i<COLORS.length;i++) {
    var sel = (attr.color === COLORS[i]) ? ' selected' : '';
    colorOpts += '<option value="'+COLORS[i]+'"'+sel+'>'+COLORS[i]+'</option>';
  }
  var shapeOpts = '';
  for(var i=0;i<SHAPES.length;i++) {
    var sel = (attr.shape === SHAPES[i]) ? ' selected' : '';
    shapeOpts += '<option value="'+SHAPES[i]+'"'+sel+'>'+SHAPES[i]+'</option>';
  }

  var swatch = CMAP[attr.color] || '#eee';

  var statusBadge = (st === 'ok')
    ? '<span class="status-ok">&#x2705; Reviewed</span>'
    : '<span class="status-needs-review">&#x26A0;&#xFE0F; Needs review</span>';

  var unsaved = changed ? '<span class="unsaved-badge">unsaved</span>' : '';

  var colorDiff = (orig.color !== attr.color) ? '<span class="orig">was: '+orig.color+'</span>' : '';
  var shapeDiff = (orig.shape !== attr.shape) ? '<span class="orig">was: '+orig.shape+'</span>' : '';

  var thumbHtml = thumb
    ? '<img src="'+thumb+'" alt="'+esc(cat)+'">'
    : '<div class="no-img">No preview</div>';

  return '<div class="card">'
    + '<div class="card-header">'
    + '<span class="cat-name">'+esc(cat)+' <span class="badge">'+esc(attr.common_name)+'</span></span>'
    + '<span>'+statusBadge+unsaved+'</span></div>'
    + '<div class="card-body">'
    + '<div class="thumb">'+thumbHtml+'</div>'
    + '<div class="info">'
    + '<div class="info-row"><label>Color</label>'
    + '<div class="color-row"><span class="color-dot" style="background:'+swatch+'"></span>'
    + '<select onchange="update(\''+cat+'\',\'color\',this.value)">'+colorOpts+'</select></div>'
    + colorDiff+'</div>'
    + '<div class="info-row"><label>Shape</label>'
    + '<select onchange="update(\''+cat+'\',\'shape\',this.value)">'+shapeOpts+'</select>'
    + shapeDiff+'</div>'
    + '<div class="info-row"><label>Name</label>'
    + '<input type="text" value="'+esc(attr.common_name)+'" onchange="update(\''+cat+'\',\'common_name\',this.value)">'
    + '</div></div></div>'
    + '<div class="card-actions">'
    + '<button class="btn btn-primary" onclick="markOK(\''+cat+'\')">&#x2705; Mark OK</button>'
    + '<button class="btn btn-secondary" onclick="resetCard(\''+cat+'\')">&#x21A9; Reset</button>'
    + '</div></div>';
}

function update(cat, key, value) {
  A[cat][key] = value;
  if(RS[cat]==='ok') { delete RS[cat]; localStorage.setItem('rs',JSON.stringify(RS)); }
  render();
}

function resetCard(cat) {
  A[cat] = JSON.parse(JSON.stringify(OA[cat]));
  render();
}

function markOK(cat) {
  RS[cat] = 'ok';
  localStorage.setItem('rs',JSON.stringify(RS));
  render();
  showToast('Marked as reviewed','success');
}

function saveAll() {
  var blob = new Blob([JSON.stringify(A,null,2)], {type:'application/json'});
  var url = URL.createObjectURL(blob);
  var a = document.createElement('a');
  a.href = url; a.download = 'object_attributes.json';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  OA = JSON.parse(JSON.stringify(A));
  showToast('Downloaded! Replace task/RoboDojo/config/object_attributes.json','success');
  render();
}

function updateStats() {
  var total=0, reviewed=0, modified=0;
  for(var k in A) {
    total++;
    if(RS[k]==='ok') reviewed++;
    if(JSON.stringify(A[k])!==JSON.stringify(OA[k])) modified++;
  }
  document.getElementById('subtitle').textContent = total+' categories - '+reviewed+' reviewed - '+modified+' unsaved changes';
  document.getElementById('stats').innerHTML = ''
    + '<div class="stat-card"><div class="num">'+total+'</div><div class="label">Total</div></div>'
    + '<div class="stat-card"><div class="num">'+reviewed+'</div><div class="label">Reviewed</div></div>'
    + '<div class="stat-card"><div class="num">'+(total-reviewed)+'</div><div class="label">Needs Review</div></div>'
    + '<div class="stat-card"><div class="num">'+modified+'</div><div class="label">Unsaved Changes</div></div>';
}

function showToast(msg, type) {
  var t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'toast '+type+' show';
  setTimeout(function(){ t.classList.remove('show'); }, 3000);
}

init();
</script>
</body>
</html>"""

    OUTPUT_PATH.write_text(html)
    file_size = len(html.encode("utf-8"))
    print(f"\nWritten to {OUTPUT_PATH}")
    print(f"File size: {file_size / 1024:.0f} KB")
    print("Open it directly in your browser (double-click the file).")


if __name__ == "__main__":
    generate()
