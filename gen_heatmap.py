#!/usr/bin/env /usr/local/bin/python3
"""
速度計測 時間帯 × 日別 ヒートマップ 生成スクリプト
使い方: python3 gen_heatmap.py
出力:   ~/gazo-ai/speed_heatmap_YYYYMM.html
"""
import os, re, calendar
from datetime import date

# ============================================================
# ★ 変更するのはここだけ
YEAR  = 2026
MONTH = 7
MISSING = {4}    # システム停止日（欠測）
SPECIAL = {15}   # 特記日（台数が極端に少ない等）
# ============================================================

DIR = '/Users/fujiiritsuo/track-photo'
_, DAYS_IN_MONTH = calendar.monthrange(YEAR, MONTH)
ALL_DAYS  = list(range(1, DAYS_IN_MONTH + 1))
ALL_HOURS = list(range(0, 24))

# 当月1日の曜日を自動計算
# date.weekday(): 0=月 1=火 2=水 3=木 4=金 5=土 6=日
_JP_DOW = ['月','火','水','木','金','土','日']
_JP_EN  = ['mon','tue','wed','thu','fri','sat','sun']
_day1_wd = date(YEAR, MONTH, 1).weekday()   # 0=月〜6=日
DOW_LABELS = [_JP_DOW[(_day1_wd + i) % 7] for i in range(7)]
DOW_EN     = [_JP_EN [(_day1_wd + i) % 7] for i in range(7)]

COLORS = {
    '月':'#3d78d0','火':'#22b894','水':'#7060c8',
    '木':'#dc7a10','金':'#cc2a18','土':'#b050a0','日':'#7898b4'
}

# ---- ファイル読み込み ----
YM2  = f'{YEAR}{MONTH:02d}'
PAT  = re.compile(
    rf'^{YM2}(\d{{2}})_(\d{{2}})\d{{4}}_\d+_'
    r'(?:car|truck|bus|motion_vehicle|\d+)_\d+kmh\.jpg$'
)

hourly = {}
daily  = {}

for f in os.listdir(DIR):
    if '_line1.' in f or '_line2.' in f:
        continue
    m = PAT.match(f)
    if not m:
        continue
    day  = int(m.group(1))
    hour = int(m.group(2))
    daily[day]  = daily.get(day, 0) + 1
    hourly.setdefault(day, {})
    hourly[day][hour] = hourly[day].get(hour, 0) + 1

active_days  = sorted(daily.keys())
total_all    = sum(daily.values())
active_count = len(active_days)
avg_per_day  = round(total_all / active_count, 1) if active_count else 0

# ---- 統計 ----
hour_total = {}
for hmap in hourly.values():
    for h, cnt in hmap.items():
        hour_total[h] = hour_total.get(h, 0) + cnt

peak_hour   = max(hour_total, key=hour_total.get) if hour_total else 0
peak_hour_c = hour_total.get(peak_hour, 0)

if active_days:
    max_cell = max(
        hourly.get(d, {}).get(h, 0)
        for d in active_days for h in ALL_HOURS
    )
    max_cell_day, max_cell_hour = max(
        ((d, h) for d in active_days for h in ALL_HOURS),
        key=lambda dh: hourly.get(dh[0], {}).get(dh[1], 0)
    )
    max_cell_dow = DOW_LABELS[(max_cell_day - 1) % 7]
else:
    max_cell = max_cell_day = max_cell_hour = 0
    max_cell_dow = ''

# 曜日別最多台数日
dow_max = {}
for day in active_days:
    dow = DOW_LABELS[(day - 1) % 7]
    cnt = daily[day]
    if dow not in dow_max or cnt > dow_max[dow][0]:
        dow_max[dow] = (cnt, day)
star_days = {info[1] for info in dow_max.values()}

# ---- セル色（7段階カテゴリ） ----
LEVELS = [
    (0,   0,   'transparent', '0'),
    (1,   3,   '#FFF0D6',     '1–3'),
    (4,   8,   '#FECB7A',     '4–8'),
    (9,   13,  '#FBA337',     '9–13'),
    (14,  18,  '#E87012',     '14–18'),
    (19,  23,  '#C44400',     '19–23'),
    (24,  999, '#8C1A00',     '24以上'),
]

def cell_color(count):
    for lo, hi, color, _ in LEVELS:
        if lo <= count <= hi:
            return color
    return 'transparent'

# ---- 期間文字列 ----
m1_dow = DOW_LABELS[0]
last_day = DAYS_IN_MONTH
ml_dow  = DOW_LABELS[(last_day - 1) % 7]
period_str = f'{MONTH}/1 ({m1_dow}) 〜 {MONTH}/{last_day} ({ml_dow})'
today_str  = date.today().strftime('%Y-%m-%d')
OUTPUT = f'/Users/fujiiritsuo/gazo-ai/speed_heatmap_{YM2}.html'

# ---- CSS ----
css = '''
:root {
  --bg-page:#f2f4f8; --bg-surface:#ffffff; --bg-row:#f7f8fc;
  --text-pri:#1b2236; --text-sec:#4e5d7a; --text-mute:#7d8da8;
  --border:#cdd3e6; --border-lt:#e4e8f2;
  --shadow:0 1px 4px rgba(27,34,54,.07),0 4px 16px rgba(27,34,54,.05);
}
@media (prefers-color-scheme:dark) {
  :root {
    --bg-page:#111520; --bg-surface:#1a1e2c; --bg-row:#1e2335;
    --text-pri:#dde2f0; --text-sec:#8a96b4; --text-mute:#5a647e;
    --border:#252d44; --border-lt:#1f2640;
    --shadow:0 1px 4px rgba(0,0,0,.3),0 4px 16px rgba(0,0,0,.2);
  }
}
:root[data-theme="dark"] {
  --bg-page:#111520; --bg-surface:#1a1e2c; --bg-row:#1e2335;
  --text-pri:#dde2f0; --text-sec:#8a96b4; --text-mute:#5a647e;
  --border:#252d44; --border-lt:#1f2640;
  --shadow:0 1px 4px rgba(0,0,0,.3),0 4px 16px rgba(0,0,0,.2);
}
:root[data-theme="light"] {
  --bg-page:#f2f4f8; --bg-surface:#ffffff; --bg-row:#f7f8fc;
  --text-pri:#1b2236; --text-sec:#4e5d7a; --text-mute:#7d8da8;
  --border:#cdd3e6; --border-lt:#e4e8f2;
  --shadow:0 1px 4px rgba(27,34,54,.07),0 4px 16px rgba(27,34,54,.05);
}
*,*::before,*::after { box-sizing:border-box; margin:0; padding:0; }
body {
  background:var(--bg-page); color:var(--text-pri);
  font-family:-apple-system,'Hiragino Sans','Yu Gothic UI','Meiryo',sans-serif;
  font-size:14px; line-height:1.6; -webkit-font-smoothing:antialiased;
}
.page { max-width:1280px; margin:0 auto; padding:28px 24px 60px; display:flex; flex-direction:column; gap:22px; }
.breadcrumb { font-size:11px; color:var(--text-mute); letter-spacing:.03em; }
.breadcrumb span { color:var(--text-sec); }
.title-block { display:flex; align-items:flex-start; justify-content:space-between; flex-wrap:wrap; gap:12px; }
.title-block h1 { font-size:26px; font-weight:700; letter-spacing:-.02em; line-height:1.2; }
.meta { font-size:11px; color:var(--text-mute); margin-top:5px; line-height:1.8; }
.btn { background:var(--bg-surface); border:1px solid var(--border); border-radius:6px;
       padding:6px 14px; font-size:12px; font-family:inherit; color:var(--text-sec); cursor:pointer; }
.kpi-row { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }
.kpi-card {
  background:var(--bg-surface); border:1px solid var(--border-lt);
  border-radius:10px; padding:20px 22px 18px; box-shadow:var(--shadow);
  position:relative; overflow:hidden;
}
.kpi-card::before {
  content:''; position:absolute; top:0; left:0; right:0;
  height:4px; border-radius:10px 10px 0 0;
}
.kpi-card:nth-child(1)::before { background:#3d78d0; }
.kpi-card:nth-child(2)::before { background:#dc7a10; }
.kpi-card:nth-child(3)::before { background:#cc2a18; }
.kpi-card:nth-child(4)::before { background:#22b894; }
.kpi-lbl { font-size:11px; color:var(--text-mute); letter-spacing:.04em; text-transform:uppercase; margin-bottom:6px; }
.kpi-val { font-size:32px; font-weight:700; font-variant-numeric:tabular-nums;
           font-family:'SF Mono','Courier New',monospace; line-height:1.1; letter-spacing:-.02em; }
.kpi-unit { font-size:16px; font-weight:400; color:var(--text-sec); }
.kpi-sub { font-size:12px; color:var(--text-sec); margin-top:6px; }
.wmax-panel {
  background:var(--bg-surface); border:1px solid var(--border-lt);
  border-radius:10px; padding:20px 22px; box-shadow:var(--shadow);
}
.panel-title {
  font-size:12px; font-weight:700; letter-spacing:.06em;
  text-transform:uppercase; color:var(--text-sec); margin-bottom:14px;
}
.wmax-grid { display:grid; grid-template-columns:repeat(7,1fr); gap:10px; }
.wmax-card {
  background:var(--bg-row); border:1px solid var(--border-lt);
  border-radius:8px; padding:14px 10px 12px; text-align:center;
  position:relative; overflow:hidden;
}
.wmax-top { height:3px; position:absolute; top:0; left:0; right:0; border-radius:8px 8px 0 0; }
.wmax-dow { font-size:11px; font-weight:700; margin-bottom:5px; }
.wmax-date { font-size:22px; font-weight:700; font-variant-numeric:tabular-nums; line-height:1.1; }
.wmax-cnt { font-size:11px; color:var(--text-sec); margin-top:4px; }
.hm-panel {
  background:var(--bg-surface); border:1px solid var(--border-lt);
  border-radius:10px; padding:20px 22px 18px; box-shadow:var(--shadow); overflow-x:auto;
}
.hm-subtitle { font-size:12px; color:var(--text-sec); margin-bottom:16px; }
table.hm {
  border-collapse:collapse; font-size:11px; font-variant-numeric:tabular-nums;
  font-family:'SF Mono','Courier New',monospace; min-width:100%;
}
table.hm th, table.hm td { padding:2px 4px; }
table.hm thead th {
  text-align:center; font-size:10px; font-weight:700;
  color:var(--text-mute); border-bottom:1px solid var(--border);
  min-width:26px; white-space:nowrap;
}
table.hm thead th.th-hour { text-align:right; padding-right:8px; min-width:36px; }
.dn { font-size:11px; font-weight:700; display:block; }
.dw { font-size:9px; color:var(--text-mute); display:block; }
.mx { color:gold; font-size:8px; display:block; }
.wed .dn,.wed .dw{color:#7060c8;} .thu .dn,.thu .dw{color:#dc7a10;}
.fri .dn,.fri .dw{color:#cc2a18;} .sat .dn,.sat .dw{color:#b050a0;}
.sun .dn,.sun .dw{color:#7898b4;} .mon .dn,.mon .dw{color:#3d78d0;}
.tue .dn,.tue .dw{color:#22b894;}
.missing .dn,.missing .dw{color:var(--text-mute)!important;opacity:.4;}
table.hm tbody th {
  text-align:right; padding-right:8px;
  color:var(--text-mute); font-size:10px; font-weight:400; white-space:nowrap;
}
table.hm tbody td {
  text-align:center; height:22px; min-width:26px;
  border-radius:2px; font-size:10px; font-weight:600; color:rgba(0,0,0,.75);
}
:root[data-theme="dark"] table.hm tbody td { color:rgba(255,255,255,.85); }
table.hm tbody td.zero { color:var(--border); }
table.hm tbody td.missing-cell { background:var(--bg-row)!important; color:var(--border-lt); }
table.hm tfoot td {
  text-align:center; font-weight:700; font-size:11px;
  color:var(--text-pri); border-top:2px solid var(--border); padding-top:4px;
}
table.hm tfoot td.th-hour { text-align:right; padding-right:8px; color:var(--text-sec); font-weight:400; }
table.hm tfoot td.total-sum { color:#3d78d0; }
.legend-row {
  display:flex; align-items:center; gap:6px; flex-wrap:wrap;
  font-size:11px; color:var(--text-sec); margin-top:14px; padding-top:12px;
  border-top:1px solid var(--border-lt);
}
.leg-item { display:flex; align-items:center; gap:4px; }
.leg-swatch { width:14px; height:14px; border-radius:2px; border:1px solid rgba(0,0,0,.08); flex-shrink:0; }
.legend-sep { color:var(--border); margin:0 4px; }
.footer-note { font-size:11px; color:var(--text-mute); line-height:1.8; }
@media (max-width:800px) {
  .kpi-row { grid-template-columns:1fr 1fr; }
  .wmax-grid { grid-template-columns:repeat(4,1fr); }
  .kpi-val { font-size:24px; }
}
@media (prefers-reduced-motion:reduce) { * { transition:none!important; } }
'''

# ---- HTML生成 ----
L = []
W = L.append

W('<!DOCTYPE html>')
W('<html lang="ja" data-theme="">')
W('<head>')
W('<meta charset="UTF-8">')
W('<meta name="viewport" content="width=device-width, initial-scale=1.0">')
W(f'<title>速度計測ヒートマップ {YEAR}年{MONTH}月</title>')
W(f'<style>{css}</style>')
W('</head><body><div class="page">')

W('<div class="breadcrumb">速度計測システム <span>/</span> 時間帯 × 日別 ヒートマップ</div>')

W('<div class="title-block"><div>')
W(f'  <h1>{YEAR}年{MONTH}月 — 時間帯別 通過台数</h1>')
W(f'  <p class="meta">期間: {period_str}　／　記録: 0時〜23時　／　データ: {DIR}　／　✗ line1.jpg 除外　／　生成: {today_str}</p>')
W('</div><button class="btn" onclick="toggleTheme()">テーマ切替</button></div>')

W('<div class="kpi-row">')
W(f'<div class="kpi-card"><div class="kpi-lbl">総台数</div>'
  f'<div class="kpi-val">{total_all}<span class="kpi-unit">台</span></div>'
  f'<div class="kpi-sub">{MONTH}/1〜{MONTH}/{DAYS_IN_MONTH}（{active_count}稼働日）</div></div>')
W(f'<div class="kpi-card"><div class="kpi-lbl">ピーク時間帯</div>'
  f'<div class="kpi-val">{peak_hour}<span class="kpi-unit">時台</span></div>'
  f'<div class="kpi-sub">月間計 {peak_hour_c}台</div></div>')
W(f'<div class="kpi-card"><div class="kpi-lbl">最多台数</div>'
  f'<div class="kpi-val">{max_cell}<span class="kpi-unit">台</span></div>'
  f'<div class="kpi-sub">{MONTH}/{max_cell_day}({max_cell_dow}) {max_cell_hour}時台</div></div>')
W(f'<div class="kpi-card"><div class="kpi-lbl">稼働日数</div>'
  f'<div class="kpi-val">{active_count}<span class="kpi-unit">日</span></div>'
  f'<div class="kpi-sub">欠測: {", ".join(str(d) for d in sorted(MISSING))}日</div></div>')
W('</div>')

W('<div class="wmax-panel">')
W('  <div class="panel-title">曜日別 最多台数日　'
  '<span style="font-weight:400;color:var(--text-mute);">― 各曜日で台数が最も多かった日</span></div>')
W('  <div class="wmax-grid">')
for dow_l, dow_en in zip(DOW_LABELS, DOW_EN):
    if dow_l in dow_max:
        cnt, day = dow_max[dow_l]
        color = COLORS[dow_l]
        W(f'<div class="wmax-card">'
          f'<div class="wmax-top" style="background:{color};"></div>'
          f'<div class="wmax-dow" style="color:{color};">{dow_l}曜日</div>'
          f'<div class="wmax-date" style="color:{color};">{MONTH}/{day}</div>'
          f'<div class="wmax-cnt">{cnt}台</div></div>')
W('  </div></div>')

W('<div class="hm-panel">')
W('<div class="panel-title">時間帯 × 日別 台数ヒートマップ</div>')
W('<div class="hm-subtitle">各セルは当該時間帯に通過した台数。セル色は台数レベルを示す（凡例は表下）。</div>')
W('<table class="hm"><thead><tr><th class="th-hour">時刻</th>')
for day in ALL_DAYS:
    dow    = DOW_LABELS[(day - 1) % 7]
    dow_en = _JP_EN[(_day1_wd + day - 1) % 7]
    star   = '<span class="mx">★</span>' if day in star_days else ''
    cls    = 'missing' if day in MISSING else dow_en
    W(f'<th class="{cls}"><span class="dn">{day}</span><span class="dw">{dow}</span>{star}</th>')
W('</tr></thead><tbody>')

for h in ALL_HOURS:
    W(f'<tr><th>{h:02d}時</th>')
    for day in ALL_DAYS:
        if day in MISSING:
            W('<td class="missing-cell">—</td>')
            continue
        cnt   = hourly.get(day, {}).get(h, 0)
        bg    = cell_color(cnt)
        title = f'{MONTH}/{day} {h}時台: {cnt}台'
        cls   = 'zero' if cnt == 0 else ''
        W(f'<td class="{cls}" style="background:{bg}" title="{title}">{cnt if cnt else ""}</td>')
    W('</tr>')

W('</tbody><tfoot><tr><td class="th-hour">合計</td>')
for day in ALL_DAYS:
    if day in MISSING:
        W('<td class="missing-cell">✗</td>')
    else:
        cnt = daily.get(day, 0)
        W(f'<td class="total-sum" title="{MONTH}/{day} 日計: {cnt}台">{cnt}</td>')
W('</tr></tfoot></table>')

W('<div class="legend-row">')
W('<span style="color:var(--text-mute);font-weight:600;margin-right:4px;">合計</span>')
for lo, hi, color, label in LEVELS:
    border = '1px solid rgba(0,0,0,.1)' if color == 'transparent' else '1px solid rgba(0,0,0,.08)'
    W(f'<div class="leg-item"><div class="leg-swatch" style="background:{color};border:{border};"></div>{label}</div>')
W('<span class="legend-sep">│</span>'
  '<span style="color:var(--text-mute);">文字 ★ 時間帯（同色）</span>'
  '<span class="legend-sep">│</span>'
  '<span style="color:gold;">★</span>'
  '<span style="color:var(--text-mute);margin-left:3px;">曜日別最多台数日</span>')
W('</div>')
W('</div>')  # hm-panel

sp_txt = '、'.join(f'{MONTH}/{d}は合計{daily.get(d,0)}台のみ' for d in sorted(SPECIAL) if d in daily)
miss_txt = '、'.join(f'{MONTH}/{d}はシステム停止（欠測）' for d in sorted(MISSING))
W(f'<div class="footer-note">▲ {miss_txt}。{sp_txt}（計測中断あり）。</div>')

W('''<script>
function toggleTheme(){
  const r=document.documentElement;
  const dk=r.dataset.theme==='dark'||(r.dataset.theme===''&&window.matchMedia('(prefers-color-scheme:dark)').matches);
  r.dataset.theme=dk?'light':'dark';
}
</script>''')

W('</div></body></html>')

with open(OUTPUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))

print(f'Generated : {OUTPUT}')
print(f'Total     : {total_all}台 / {active_count}稼働日 / avg {avg_per_day}台/日')
print(f'Peak      : {peak_hour}時台 {peak_hour_c}台')
print(f'Max cell  : {MONTH}/{max_cell_day}({max_cell_dow}) {max_cell_hour}時 = {max_cell}台')
print(f'Star days : {sorted(star_days)}')
