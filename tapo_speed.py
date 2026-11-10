"""
Tapo C530WS 速度計測プログラム（Raspberry Pi 5 / Python / OpenCV / YOLO）

計測の考え方（これまで手作業で行ってきた方法をそのまま自動化）
  - 路面上の2点（丸マンホール = Gate A、四角マンホール = Gate B、間隔10m）に
    小さな監視枠（ROI）を置く
  - 車の前端が枠を覆った瞬間を「進入」とし、2つの枠の進入時刻の差から速度を計算
  - 時刻はフレームの時刻情報（PTS）を使う。Tapoはコマ間隔が一定でないため、15fps固定で割らない

トラックの判定
  - 昼間：AI（YOLO）でトラックを判定
  - 夜間など AI で判定できないとき：画面中央の縦の列（10マス）のうち、
    同時に8マス以上を覆った車をトラックとする（車高の高さで判定）

夜間対策
  - 枠の判定は「明るさ」ではなく「路面の模様との相関（ZNCC）」で行い、
    ゆっくりした明るさの変化は取り除く
  - 背景（車のいない路面）は毎コマ素早く更新するので、ヘッドライトが近づいて
    路面の見え方が変わっても、車の前端が来た「急な変化」だけを捉える
  - 昼夜の切り替え（IRモード）で画面全体が変化したら背景を自動で学習し直す

使い方
  リアルタイム（Tapo）:  python3 tapo_speed.py
  録画ファイルで検証:    python3 tapo_speed.py --source video.mp4
  YOLOなしで動作確認:     python3 tapo_speed.py --source video.mp4 --no-yolo
  画面なし（自動運用）:   python3 tapo_speed.py --no-display
  距離・補正を指定:       python3 tapo_speed.py --distance 10.0 --correction 0.97
  高さ判定の調整:         python3 tapo_speed.py --height-cells 10 --height-min 8

キー操作（画面表示中）
  q : 終了
  c : 計測位置の設定（A奥 → A手前 → B奥 → B手前 の順に4点クリック）
      A = 丸マンホールの位置で道路を横切る線、B = 四角マンホールの位置で道路を横切る線
  b : 背景の再学習（車がいないときに押す）
"""

import argparse
import csv
import json
import os
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2
import numpy as np

# ========================================
# 設定
# ========================================

CAMERA_IP = "192.168.11.12"
STREAM = "stream2"           # stream2 = 低解像度（ラズパイ向き）、stream1 = 高解像度

GATE_DISTANCE_M = 10.0       # Gate A ～ Gate B の実測距離 [m]
SPEED_CORRECTION = 1.00      # 速度の補正係数（全車両共通。例：表示が1割高い → 0.91）
TRUCK_CORRECTION_DAY = 1.00  # 昼間（カラー映像）のトラック補正
TRUCK_CORRECTION = 1.00      # トラックだけに追加でかける補正係数（例：20km/hが17km/hと出る → 20/17 ≒ 1.15）

# 計測位置（画面に対する割合 0～1）。'c' キーでクリック設定すると gates.json に保存される
# 各ゲートは「道路を横切る縦の帯」。奥側の車線(far)～手前側の車線(near)の2点で指定する。
# どちらの車線を走る車でも、前端が帯にかかった瞬間に反応する。
DEFAULT_GATES = {
    "A": {"far": [0.943, 0.560], "near": [0.935, 0.900], "name": "Round MH"},   # 右：丸マンホールの位置
    "B": {"far": [0.047, 0.320], "near": [0.047, 0.710], "name": "Square MH"},  # 左：四角マンホールの位置
    "roi_w": 0.060,   # 帯の幅（画面幅に対する割合）
    "cells": 5,       # 帯を何マスに分けて判定するか
}
GATES_FILE = "gates.json"

# 判定のしきい値（相関 ZNCC：1 = 背景と同じ、0 = 無関係）
COVER_TH = 0.45        # 背景との相関がこれ未満になったら「前端が来た」
BG_ALPHA = 0.30        # 車がいない間の背景更新の速さ（夜間のライト変化に追従）
STABLE_TH = 0.80       # 前コマとの相関がこれ以上 = 枠内が静止
STABLE_FRAMES = 3      # 静止が何コマ続いたら「車が抜けた」とするか
# マスが「覆われたまま」戻らないときの自動復帰。
# 木の葉の影のゆれ・水たまり・止まっている物などでマスが変化し続けると、
# 帯がずっと「覆われている」扱いになり、次の車の前端に反応できなくなる（後ろの部分で反応する）。
CELL_STUCK_SEC = 8.0   # この秒数以上覆われたままのマスは、今の見え方を背景として学習し直す
CELL_MUTE_SEC = 120.0  # 学習し直したマスは、この秒数のあいだ判定から外す（ゆれ続ける影などへの対策）
BLOCK_SEC = 0.5        # 計測直後、全体で反応を無視する短い時間
# 計測後は「その車が帯から抜けるまで」だけ、その帯の反応を無視する（同じ車体の再反応防止）。
# 帯が空いたら、すぐ次の車を受け付ける（トラックが続けて来ても取りこぼさない）。
BLOCK_MAX_SEC = 15.0   # 帯が空かない場合（停車など）でも、この時間で解除
TEX_MIN = 1.5          # 模様の少ないマス（夜の暗い路面など）は誤反応しやすいので判定に使わない
SECOND_WIN = 0.6       # 2つ目の帯の反応後、この時間待ってから各マスの反応時刻をまとめる
# 帯の通過時刻は「最初に反応したマス」ではなく「反応したマスの時刻の中央値」を使う。
# 背の高い車のミラーや運転席の角は、カメラから見ると実際より先（帯の上の方）に写るため、
# 上のマスだけが早く反応して速度が高く出ることがある。中央値ならその影響を受けにくい。
USE_MEDIAN = True
MEDIAN_MIN_CELLS = 2

MIN_SPEED_KMH = 3.0
MAX_SPEED_KMH = 70.0             # これを超える値は計測の誤りとして捨てる（この道路では現実的でない）
MAX_GAP_SEC = GATE_DISTANCE_M / (MIN_SPEED_KMH / 3.6)

# YOLO
MODEL_PATH = "yolo26n.pt"
CONF = 0.20                      # 夜間も拾えるよう少し低め
IMG_SIZE = 416
DETECT_INTERVAL_SEC = 0.20       # AI認識の間隔
VEHICLE_CLASSES = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck
# 箱型トラックはCOCOで "bus" と判定されることがあるので、bus も trucks として扱う
TRUCK_LABELS = {"truck", "bus"}
# 車種の判定（計測区間を通る間のAI判定を集計して多数決）
TRUCK_MIN_AREA = 0.25            # トラック判定でも、枠の面積が画面の25%未満なら対象外（遠くの車・軽自動車など）
TRUCK_FORCE_AREA = 0.60
# 高さ判定（夜間・薄暗いとき用）
#   画面の中央に、上から下まで縦に並んだマス（高さ判定の列）を置く。
#   トラックは車高が高く、通過中に列のほぼ全部（上の建物の部分まで）を覆う。
#   乗用車は道路の部分しか覆わないので、上のマスは覆われない。
#   AIでトラックと判定できなかった車でも、同時に HEIGHT_MIN 個以上のマスを覆えばトラックとする。
HEIGHT_COL = {"x": 0.50, "top": 0.10, "bottom": 0.95, "w": 0.06}   # 列の位置（画面に対する割合）
HEIGHT_CELLS = 10                # 列のマスの数
HEIGHT_TOP_CELLS = 1             # 一番上からこの数のマスが覆われることもトラックの条件
HEIGHT_MIN = 9                   # この数以上のマスを同時に覆えばトラック
HEIGHT_DIFF_TH = 0.50            # 背景との相関がこれ未満なら「覆われている」
HEIGHT_BRIGHT_TH = 999           # 明るさの変化での判定（999=使わない。カメラの露出変化で誤反応しやすいため）
HEIGHT_BG_ALPHA = 0.10           # 覆われていないマスの背景の更新の速さ

# 参考表示用（判定には使わない）
LEN_OFFSET = 5.0
EXIT_DEBOUNCE = 0.6
EXIT_LATENCY = 0.3
EXIT_WAIT_MAX = 10.0

# 順序チェック：車は A → 中央の列 → B（または逆）の順に通るはず。
# 2つ目の帯が反応した時点で、中央の列がまだ覆われていなければ
# ヘッドライトの光やカメラの明るさ調整による誤反応とみなし、無視して待ち直す。
ORDER_CHECK = True
# 実測（録画6台）では「中央→2つ目」は「1つ目→中央」の約0.5～0.9倍（平均0.7倍）
A_OVER_B = 1.4                   # 1つ目→中央 ÷ 中央→2つ目 の標準的な比（推定に使う）
MIN_B_RATIO = 0.45               # 中央→2つ目 がこの倍率より短い = 2つ目の帯の反応が早すぎる（誤反応）
BALANCE_RATIO = 2.5              # 1つ目→中央 の時間が 中央→2つ目 のこの倍を超えたら「古い反応」
# 夜は、ヘッドライトの光やカメラの明るさ調整で、車が来ていない帯が反応することがある。
# 夜は「中央の列の一番上（建物の高さ）が車体で隠れた後」でないと、2つ目の帯の反応を認めない。
# （光は路面にしか当たらないので、一番上のマスは本物の車体でしか隠れない）
NIGHT_REQUIRE_TOP = True
ORDER_MIN_CELLS = 3              # 中央の列がこの数以上覆われていれば「車が中央を通過中」

# 画面全体の変化の見張り（高さ判定用）：左上と右上の建物の部分。
# 車のライトやカメラの明るさ調整・ブレで画面全体の見え方が変わったときは、
# 両方とも見え方が変わる → そのコマの高さ判定は使わない。
# 本物のトラックは片側（来た側）しか隠さないので、もう片方は変わらない。
CONTROL_REGIONS = [(0.02, 0.10, 0.14, 0.28), (0.86, 0.10, 0.98, 0.28)]  # (x1, y1, x2, y2)
CONTROL_TH = 0.60                # 相関がこれ以上なら「変わっていない」

# 記録する進行方向。"A->B" = 丸マンホール → 四角マンホール（トラックの通常の方向）
#   "B->A" の車はライトがカメラ側を照らして誤判定しやすいため、既定では記録しない
DIRECTION = "A->B"               # 夜（白黒映像）に記録する方向 "A->B" / "B->A" / "both"
DIRECTION_DAY = "both"           # 昼（カラー映像）に記録する方向

# 夜は、トラックの荷台（赤外線で明るく写る）が一番上のマスを通ると、そのマスが明るくなる。
# 乗用車はライトや画面のブレで上のマスの模様が変わっても、明るくはならない。
NIGHT_TOP_BRIGHT = 8             # 夜のトラック判定：一番上のマスがこれ以上明るくなること

HEIGHT_JUDGE = True              # False にすると高さ判定を使わない（AIでトラックと判定できた車だけ）
TRUCK_ONLY = True                # True = トラックだけ記録（--all で全車両）
CLASSIFY_WAIT_SEC = 1.0          # 計測後、車種判定のためにAI結果を待つ時間
SAVE_CLIP = True                 # 計測した車ごとに前後数秒の動画（mp4）を debug フォルダに保存（原因調査用）
SAVE_DEBUG = True                # 2つのマンホールで反応した瞬間の画像も保存（調整用）

OUT_DIR = Path("speed_captures")
CSV_PATH = OUT_DIR / "speed_log.csv"


# ========================================
# 計測ゲート（マンホール監視枠）
# ========================================

GATE_WORK_WIDTH = 640    # 判定は常にこの幅に縮小して行う（stream1でもstream2でも同じ動作）


def prep(patch):
    """模様（細かい凹凸）だけを取り出す。ゆっくりした明るさの変化（ヘッドライトの照り返し等）は除去"""
    g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    g = g.astype(np.float32)
    s = max(1.0, g.shape[0] / 30)
    return cv2.GaussianBlur(g, (0, 0), s) - cv2.GaussianBlur(g, (0, 0), s * 5)


def zncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    if den < 1e-6:
        return 0.0
    return float((a * b).sum() / den)


class Cell:
    """帯の中の1マス。車の前端が来た急な変化（進入）と、車が抜けた静止（退出）を判定"""

    def __init__(self):
        self.bg = None
        self.prev = None
        self.covered = False
        self.stable = 0
        self.score = 1.0

    def reset(self, p):
        self.bg = p
        self.prev = p
        self.covered = False
        self.stable = 0

    def reliable(self):
        return self.bg is not None and float(np.sqrt((self.bg ** 2).mean())) >= TEX_MIN

    def update(self, p, ts=None):
        if self.covered and ts is not None:
            if getattr(self, "cov_since", None) is None:
                self.cov_since = ts
            elif ts - self.cov_since > CELL_STUCK_SEC:
                self.reset(p)
                self.cov_since = None
                self.stuck_resets = getattr(self, "stuck_resets", 0) + 1
                was_muted = ts < getattr(self, "mute_until", -1)
                self.mute_until = ts + CELL_MUTE_SEC
                if was_muted:
                    self.stuck_resets -= 1      # 判定から外している間の再学習は表示しない
                return "exit"
        else:
            self.cov_since = None
        if self.bg is None or self.bg.shape != p.shape:
            self.reset(p)
            return None
        event = None
        if not self.covered:
            self.score = zncc(p, self.bg)
            if self.score < COVER_TH and self.reliable():
                self.covered = True
                self.stable = 0
                event = "enter"
            else:
                self.bg = (1 - BG_ALPHA) * self.bg + BG_ALPHA * p
        else:
            self.score = zncc(p, self.prev)
            if self.score > STABLE_TH:
                self.stable += 1
                if self.stable >= STABLE_FRAMES:
                    self.covered = False
                    self.bg = p
                    event = "exit"
            else:
                self.stable = 0
        self.prev = p
        return event


class Gate:
    """
    道路を横切る縦の帯（奥の車線～手前の車線）。帯をいくつかのマスに分け、
    どれか1マスでも車の前端で覆われたら「進入」、全マスが空いたら「退出」。
    """

    def __init__(self, key, cfg):
        self.key = key
        self.name = cfg[key].get("name", key)
        self.set_points(cfg[key]["far"], cfg[key]["near"])
        self.nw = cfg.get("roi_w", 0.06)
        self.n = int(cfg.get("cells", 5))
        self.cells = [Cell() for _ in range(self.n)]
        self.covered = False
        self.score = 1.0
        self.hits = []          # 今回の進入で反応したマス [(マス番号, 時刻)]
        self.max_cov = 0        # 同時に覆われたマスの最大数（車高・車幅の目安）

    def set_points(self, far, near):
        self.far = tuple(far)
        self.near = tuple(near)
        self.nx = (far[0] + near[0]) / 2
        self.ny = (far[1] + near[1]) / 2

    def cell_rects(self, w, h):
        rects = []
        (fx, fy), (nx, ny) = self.far, self.near
        rw = int(self.nw * w)
        seg = abs(ny - fy) * h / self.n
        rh = max(8, int(seg))
        for i in range(self.n):
            t = (i + 0.5) / self.n
            cx = int((fx + (nx - fx) * t) * w)
            cy = int((fy + (ny - fy) * t) * h)
            x1 = max(0, cx - rw // 2)
            y1 = max(0, cy - rh // 2)
            rects.append((x1, y1, min(w, x1 + rw), min(h, y1 + rh)))
        return rects

    def rect(self, w, h):
        rs = self.cell_rects(w, h)
        return (min(r[0] for r in rs), min(r[1] for r in rs),
                max(r[2] for r in rs), max(r[3] for r in rs))

    def patches(self, small):
        h, w = small.shape[:2]
        return [prep(small[y1:y2, x1:x2]) for x1, y1, x2, y2 in self.cell_rects(w, h)]

    def reset_bg(self, small):
        for c, p in zip(self.cells, self.patches(small)):
            c.reset(p)
        self.covered = False

    def update(self, small, ts=0.0):
        """戻り値: 'enter'（前端が帯にかかった）/ 'exit'（車が抜けた）/ None"""
        before = [getattr(c, "stuck_resets", 0) for c in self.cells]
        evs = [c.update(p, ts) for c, p in zip(self.cells, self.patches(small))]
        for i, c in enumerate(self.cells):
            if getattr(c, "stuck_resets", 0) > before[i]:
                print(f"[自動復帰] 帯{self.key}のマス{i + 1}が{CELL_STUCK_SEC:.0f}秒以上覆われたままのため、"
                      f"背景を学習し直し、{CELL_MUTE_SEC:.0f}秒間判定から外します")
        # 判定から外しているマスは、反応も「覆われている」状態も数えない
        muted = [ts < getattr(c, "mute_until", -1) for c in self.cells]
        evs = [None if mu else e for e, mu in zip(evs, muted)]
        self.score = min(c.score for c in self.cells)
        any_cov = any(c.covered for c, mu in zip(self.cells, muted) if not mu)
        done = {i for i, _ in self.hits}
        for i, e in enumerate(evs):
            if e == "enter" and i not in done:
                self.hits.append((i, ts))
        ncov = sum(c.covered for c, mu in zip(self.cells, muted) if not mu)
        if not self.covered and "enter" in evs:
            self.covered = True
            self.max_cov = ncov
            self.hits = [(i, ts) for i, e in enumerate(evs) if e == "enter"]
            return "enter"
        if self.covered:
            self.max_cov = max(getattr(self, "max_cov", 0), ncov)
        if self.covered and not any_cov:
            self.covered = False
            return "exit"
        return None


class HeightColumn:
    """画面中央の縦の列。いくつのマスが同時に車体で覆われているかを数える"""

    def __init__(self, n=None):
        self.n = n = n or HEIGHT_CELLS
        self.bg = [None] * n
        self.bgm = [0.0] * n
        self.occ = [False] * n
        self.occ_since = [0.0] * n
        self.count = 0
        self.history = deque(maxlen=400)    # (時刻, 覆われたマスの数, 最上段)
        self.bright_hist = deque(maxlen=400)  # (時刻, 一番上のマスの明るさの変化)
        self.ctl_bg = [None] * len(CONTROL_REGIONS)
        self.ctl_z = [1.0] * len(CONTROL_REGIONS)
        self.global_change = False
        self.n_global = 0

    def rects(self, w, h):
        c = HEIGHT_COL
        rw = int(c["w"] * w)
        cx = int(c["x"] * w)
        y0, y1 = c["top"] * h, c["bottom"] * h
        step = (y1 - y0) / self.n
        return [(cx - rw // 2, int(y0 + i * step), cx + rw // 2, int(y0 + (i + 1) * step))
                for i in range(self.n)]

    def reset(self, small):
        h, w = small.shape[:2]
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        for i, (x1, y1, x2, y2) in enumerate(self.rects(w, h)):
            patch = g[y1:y2, x1:x2]
            self.bg[i] = prep(patch)
            self.bgm[i] = float(patch.mean())
            self.occ[i] = False
        self.ctl_bg = [None] * len(CONTROL_REGIONS)

    def update(self, small, ts):
        h, w = small.shape[:2]
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        for i, (x1, y1, x2, y2) in enumerate(self.rects(w, h)):
            patch = g[y1:y2, x1:x2]
            p = prep(patch)
            m = float(patch.mean())
            if self.bg[i] is None or self.bg[i].shape != p.shape:
                self.bg[i], self.bgm[i] = p, m
                continue
            occ = zncc(p, self.bg[i]) < HEIGHT_DIFF_TH or abs(m - self.bgm[i]) > HEIGHT_BRIGHT_TH
            if i == 0:
                self.top_dm = m - self.bgm[i]      # 一番上のマスの明るさの変化（夜のトラック判定用）
            if occ and not self.occ[i]:
                self.occ_since[i] = ts
            # 20秒以上覆われっぱなし（停車・照明の変化）なら背景として学習し直す
            if not occ or ts - self.occ_since[i] > 20:
                self.bg[i] = (1 - HEIGHT_BG_ALPHA) * self.bg[i] + HEIGHT_BG_ALPHA * p
                self.bgm[i] = (1 - HEIGHT_BG_ALPHA) * self.bgm[i] + HEIGHT_BG_ALPHA * m
            self.occ[i] = occ
        self.count = sum(self.occ)
        # 一番上のマス（建物の高さ）まで覆われているか。背の高いワンボックス車は届かない
        top = all(self.occ[:HEIGHT_TOP_CELLS])
        # 画面全体が変わったコマ（左上・右上の両方が変化）は高さの判定に使わない
        self.global_change = not self.control_ok(g)
        if self.global_change:
            self.n_global += 1
            self.history.append((ts, 0, False))
            self.bright_hist.append((ts, -999.0))
        else:
            self.history.append((ts, self.count, top))
            # 一番上のマスが「覆われている」コマだけ明るさの変化を記録
            self.bright_hist.append((ts, getattr(self, "top_dm", 0.0) if self.occ[0] else -999.0))

    def control_ok(self, g):
        h, w = g.shape[:2]
        zs = []
        for i, (a, b, c, d) in enumerate(CONTROL_REGIONS):
            p = prep(g[int(b * h):int(d * h), int(a * w):int(c * w)])
            if self.ctl_bg[i] is None or self.ctl_bg[i].shape != p.shape:
                self.ctl_bg[i] = p
            z = zncc(p, self.ctl_bg[i])
            if z > 0.7:
                self.ctl_bg[i] = 0.9 * self.ctl_bg[i] + 0.1 * p
            zs.append(z)
        self.ctl_z = zs
        return max(zs) >= CONTROL_TH

    def first_reach(self, t0, t1, n):
        for t, c, _ in self.history:
            if t0 <= t <= t1 and c >= n:
                return t
        return None

    def top_bright_between(self, t0, t1):
        """一番上のマスが覆われている間の明るさの変化の、2番目に大きい値（1コマだけの光を除く）"""
        vals = sorted((d for t, d in self.bright_hist if t0 <= t <= t1), reverse=True)
        return vals[1] if len(vals) >= 2 else -999.0

    def top_between(self, t0, t1):
        return any(top for t, c, top in self.history if t0 <= t <= t1)

    def max_between(self, t0, t1):
        vals = [c for t, c, _ in self.history if t0 <= t <= t1]
        return max(vals) if vals else 0


# ========================================
# 速度計算
# ========================================

class SpeedTracker:
    """
    2つの帯に車の前端がかかった時刻の差から速度を計算する。

    どちらの帯も、帯のどこか（奥の車線でも手前の車線でも）に車の前端がかかった
    最初の時刻を使う。模様の少ないマスは誤反応しやすいので判定に使わない。
    """

    def __init__(self, distance_m, gates):
        self.d = distance_m
        self.gates = gates
        self.first = None          # 先に反応した帯 {"gate", "ts", "img"}
        self.pending = None        # 2つ目の帯の反応待ち
        self.block_until = -1e9    # 計測直後は同じ車体の反応を無視
        self.blocked = {}          # 計測済みの車体がまだいる帯 {帯: 計測時刻}
        self.height = None         # 中央の列（FrameReader が設定）
        self.rejected = 0
        self.cand = None
        self.night = False

    def on_event(self, gate_key, event, ts, wall, small=None):
        # 1つ目の帯が「抜けた」時刻を記録（古い反応を見分けるため）
        if event == "exit":
            # 計測済みの車が帯を抜けた → その帯の受付を再開
            if gate_key in self.blocked:
                del self.blocked[gate_key]
            if self.first is not None and self.first["gate"] == gate_key:
                self.first["exit"] = ts
            return None
        if event != "enter" or ts < self.block_until or self.pending is not None:
            return None
        bt = self.blocked.get(gate_key)
        if bt is not None:
            if ts - bt < BLOCK_MAX_SEC:
                return None            # 計測済みの車体がまだこの帯にいる
            del self.blocked[gate_key]
        if self.first is not None and ts - self.first["ts"] > MAX_GAP_SEC:
            self.first = None
        if self.first is None:
            self.first = {"gate": gate_key, "ts": ts, "img": small}
            return None
        if self.first["gate"] == gate_key:
            # 同じ帯の再反応：いったん帯が空いてから再び覆われた場合は別の車（前の反応は古い）
            ex = self.first.get("exit")
            if ex is not None and ts - ex >= EXIT_DEBOUNCE:
                self.first = {"gate": gate_key, "ts": ts, "img": small}
            return None            # 同じ車体の再反応なら最初の時刻を優先
        if ORDER_CHECK and self.height is not None:
            # 1つ目の反応から4秒以上たっても中央を何も通っていない → 1つ目は誤反応。捨てて今の反応を1つ目にする
            if ts - self.first["ts"] > 4.0 and self.height.max_between(self.first["ts"], ts) < ORDER_MIN_CELLS:
                self.first = {"gate": gate_key, "ts": ts, "img": small}
                return None
            if NIGHT_REQUIRE_TOP and self.night and not self.height.top_between(self.first["ts"], ts):
                print(f"[無視] 帯{gate_key}の反応（{ts - self.first['ts']:.2f}秒後）：夜間で、車体がまだ中央を通っていません（ライト等による誤反応）")
                if self.cand is None:
                    self.cand = {"gate": gate_key, "ts": ts, "wall": wall}
                return "reject"
            if self.height.max_between(self.first["ts"], ts) < ORDER_MIN_CELLS:
                self.rejected += 1
                print(f"[無視] 帯{gate_key}の反応（{ts - self.first['ts']:.2f}秒後）は中央を車が通る前なので誤反応とみなします")
                return "reject"
            # 車の前端が中央に来た直後に2つ目の帯が反応した場合も誤反応
            # （昼間、白い大きな車体が画面に入るとカメラの明るさ調整・画質が変わり、
            #   車がまだ来ていない帯の路面の見え方が変わって反応してしまう）
            t_col = self.height.first_reach(self.first["ts"], ts, ORDER_MIN_CELLS)
            if t_col is not None:
                a, b = t_col - self.first["ts"], ts - t_col
                if a > 0.2 and b < MIN_B_RATIO * a:
                    print(f"[無視] 帯{gate_key}の反応が早すぎます（1つ目→中央 {a:.2f}秒 / 中央→2つ目 {b:.2f}秒）。車の到着を待ちます")
                    # 2秒待っても次の反応がなければ、この反応を使う（取りこぼし防止）
                    if self.cand is None:
                        self.cand = {"gate": gate_key, "ts": ts, "wall": wall}
                    return "reject"
        self.pending = {"gate": gate_key, "ts0": ts, "wall": wall}
        self.cand = None
        return None

    def poll(self, ts, small=None):
        """毎コマ呼ぶ。2つ目の帯の判定が確定したら結果を返す"""
        if self.first is not None and self.pending is None and ts - self.first["ts"] > MAX_GAP_SEC:
            self.first = None
        if self.pending is None and self.cand is not None and self.first is not None \
                and ts - self.cand["ts"] > 2.0:
            c = self.cand
            self.cand = None
            print("[注意] 2つ目の帯にその後の反応がないため、早めの反応で計測します（速度は高めの可能性）")
            self.pending = {"gate": c["gate"], "ts0": c["ts"], "wall": c["wall"], "early": True}
        p = self.pending
        if p is None:
            return None
        g = self.gates[p["gate"]]
        if p.get("img") is None and small is not None:
            p["img"] = small
        if ts < p["ts0"] + SECOND_WIN:
            return None
        t2 = p["ts0"]
        cell = g.hits[0][0] if g.hits else -1
        if USE_MEDIAN:
            h2 = [t for i, t in g.hits if p["ts0"] <= t <= p["ts0"] + SECOND_WIN]
            if len(h2) >= MEDIAN_MIN_CELLS:
                t2 = float(np.median(h2))
            g1 = self.gates[self.first["gate"]]
            h1 = [t for i, t in g1.hits if self.first["ts"] <= t <= self.first["ts"] + SECOND_WIN]
            if len(h1) >= MEDIAN_MIN_CELLS:
                self.first["ts"] = float(np.median(h1))
        # 時間の釣り合いチェック：中央の列は A と B のほぼ中間にあるので、
        # 「1つ目の帯 → 中央」と「中央 → 2つ目の帯」の時間はおおよそ同じになるはず。
        # 1つ目が極端に長ければ、1つ目の反応は前の車や光によるもの（古い）とみなし、
        # 中央 → 2つ目の時間から速度を推定する。
        estimated = False
        if self.height is not None:
            t_col = self.height.first_reach(self.first["ts"], t2, ORDER_MIN_CELLS)
            if t_col is not None:
                a, b = t_col - self.first["ts"], t2 - t_col
                if b > 0.05 and a > BALANCE_RATIO * b + 0.8:
                    print(f"[補正] 1つ目の反応が古いため推定（1つ目→中央 {a:.2f}秒 / 中央→2つ目 {b:.2f}秒）")
                    self.first["ts"] = t_col - b * A_OVER_B
                    estimated = True
        self.pending = None
        first = self.first
        self.first = None
        dt = t2 - first["ts"]
        kmh = self.d / dt * 3.6 * SPEED_CORRECTION if dt > 0 else 0
        if not (MIN_SPEED_KMH <= kmh <= MAX_SPEED_KMH):
            print(f"[破棄] 計算値 {kmh:.1f} km/h（{dt:.2f}秒）は範囲外のため記録しません")
            return None
        self.block_until = t2 + BLOCK_SEC
        # 両方の帯を「その車が抜けるまで」受付停止（すでに抜けている帯は停止しない）
        for k in (first["gate"], p["gate"]):
            if self.gates[k].covered:
                self.blocked[k] = t2
        other = first["gate"]
        return {
            "first_gate": other,
            "second_gate": p["gate"],
            "t_enter_first": first["ts"],
            "t_enter_second": t2,
            "second_cell": cell,
            "img_first": first.get("img"),
            "img_second": p.get("img"),
            "wall": p["wall"],
            "dt": dt,
            "front_kmh": kmh,
            "direction": f"{other}->{p['gate']}" + ("(est)" if estimated else "") + ("(early)" if p.get("early") else ""),
        }

    def timeout(self, ts):
        return None


# ========================================
# 映像入力（別スレッドで全コマ読み取り・ゲート判定）
# ========================================

class FrameReader(threading.Thread):
    def __init__(self, source, is_file, gates, tracker, on_result, realtime_file=False):
        super().__init__(daemon=True)
        self.source = source
        self.is_file = is_file
        self.gates = gates
        self.tracker = tracker
        self.on_result = on_result
        self.realtime_file = realtime_file
        self.lock = threading.Lock()
        self.latest = None          # (ts, wall, frame)
        self.running = True
        self.finished = False
        self.request_bg = True
        self.fps = 0.0
        self.last_mean = None
        self.start_wall = datetime.now()
        self.ts_source = "-"
        self.ts_ratio = None
        self.chk0 = None
        self.chk_done = False
        self.use_arrival = False
        self.recent = deque(maxlen=15)
        self.clipbuf = deque(maxlen=150)    # 動画保存用：直近約10秒分の縮小画像
        self.awaiting = []
        self.dt_hist = deque(maxlen=50)
        self.n_frames = 0
        self.n_stuck = 0
        self.ts_log = None
        self.height = HeightColumn()
        tracker.height = self.height

    def track_exit(self, k, e, ts):
        if e == "exit":
            self.gates[k].last_exit = ts
        for res, _ in self.awaiting:
            if k not in res["exit"] or res["exit"][k] is not None:
                continue
            if e == "exit":
                res["exit_cand"][k] = ts
            elif e == "enter":
                res["exit_cand"][k] = None      # すぐ覆われた → まだ同じ車体

    def flush_awaiting(self, ts, force=False):
        keep = []
        for res, snap in self.awaiting:
            for k, c in res["exit_cand"].items():
                if res["exit"][k] is None and c is not None and (ts - c >= EXIT_DEBOUNCE or force):
                    res["exit"][k] = c
            done = all(v is not None for v in res["exit"].values())
            if done or force or ts - res["t_enter_second"] > EXIT_WAIT_MAX:
                v = res["front_kmh"] / 3.6
                durs = []
                e1, e2 = res["exit"][res["first_gate"]], res["exit"][res["second_gate"]]
                if e1:
                    durs.append(e1 - res["t_enter_first"])
                if e2:
                    durs.append(e2 - res["t_enter_second"])
                res["len_est"] = max(0.0, (min(durs) - EXIT_LATENCY) * v - LEN_OFFSET) if durs else None
                res["cells_max"] = max(self.gates[k].max_cov for k in res["exit"])
                t_end = max([res["t_enter_second"] + 1.0] + [v for v in res["exit"].values() if v])
                res["height_max"] = self.height.max_between(res["t_enter_first"] - 0.5, t_end)
                res["height_top"] = self.height.top_between(res["t_enter_first"] - 0.5, t_end)
                res["top_bright"] = self.height.top_bright_between(res["t_enter_first"] - 0.5, t_end)
                if SAVE_CLIP:
                    t0, t1 = res["t_enter_first"] - 1.5, res["t_enter_second"] + 1.5
                    res["clip"] = [(t, f) for t, f in self.clipbuf if t0 <= t <= t1]
                self.on_result(res, snap)
            else:
                keep.append((res, snap))
        self.awaiting = keep

    def open(self):
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        return cap if cap.isOpened() else None

    def run(self):
        cap = self.open()
        if cap is None:
            print("カメラ／ファイルを開けません:", self.source)
            self.finished = True
            return
        last_pts = None
        last_ts = None
        t_prev = None
        mono0 = time.monotonic()
        while self.running:
            ok, frame = cap.read()
            if not ok:
                if self.is_file:
                    break
                print("映像取得エラー → 再接続します")
                cap.release()
                time.sleep(2)
                cap = self.open()
                last_pts = None
                last_ts = None
                self.request_bg = True
                if cap is None:
                    continue
                continue

            # --- 時刻：PTS（フレームの時刻情報）を優先 ---
            pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            mono = time.monotonic() - mono0
            # Tapo のライブ映像の PTS は、1秒ごとに約0.4秒戻ったり飛んだりする（のこぎり状）。
            # そのため PTS の値そのものではなく「コマ間の差」を使い、
            # 差が普通の範囲（0.02～0.2秒）ならそのまま、外れたら通常のコマ間隔で足していく。
            # （録画ファイルのように PTS が正しい場合は、PTS とまったく同じ結果になる）
            self.n_frames += 1
            if last_ts is None or pts <= 0 and last_pts is None:
                ts = pts if pts > 0 else mono
                self.ts_source = "PTS"
            else:
                d = pts - last_pts if last_pts is not None else -1
                if 0.02 <= d <= 0.2:
                    self.dt_hist.append(d)
                    step = d
                    self.ts_source = "PTS"
                else:
                    self.n_stuck += 1
                    step = float(np.mean(self.dt_hist)) if len(self.dt_hist) >= 5 else 1 / 15
                    self.ts_source = "PTS補完"
                ts = last_ts + step
            last_pts = pts
            last_ts = ts
            if self.ts_log is not None:
                self.ts_log.write(f"{self.n_frames},{pts:.4f},{mono:.4f},{ts:.4f}\n")
            # PTSの進み方と実時間を比べて、時刻情報が正しいか確認（ライブ時のみ）
            # 起動直後はカメラ側にたまった映像がまとめて届くので、60秒たってから
            # 5分ごとに「映像時刻の進み ÷ 実時間」を表示（1.00前後なら正常）
            if not self.is_file and mono >= 60:
                if self.chk0 is None:
                    self.chk0 = (ts, mono)
                elif mono - self.chk0[1] >= 300:
                    ratio = (ts - self.chk0[0]) / (mono - self.chk0[1])
                    self.ts_ratio = ratio
                    stuck = 100.0 * self.n_stuck / max(1, self.n_frames)
                    print(f"[時刻チェック] 映像時刻の進み/実時間 = {ratio:.3f}  "
                          f"PTSが飛んだコマ（補完）= {stuck:.1f}%  実fps = {self.n_frames / max(1e-6, mono - self.chk0[1]):.1f}"
                          + ("" if 0.97 <= ratio <= 1.03 else "  ★ずれています"))
                    self.n_frames = 0
                    self.n_stuck = 0
                    self.chk0 = (ts, mono)
            if self.use_arrival and not self.is_file:
                ts = mono
                self.ts_source = "arrival"
            wall = (self.start_wall + timedelta(seconds=ts)) if self.is_file else datetime.now()

            if t_prev is not None and mono > t_prev:
                inst = 1.0 / max(mono - t_prev, 0.02)
                self.fps = 0.95 * self.fps + 0.05 * inst if self.fps else inst
            t_prev = mono

            # 判定用の縮小画像（解像度に関係なく同じ動作にする）
            h0, w0 = frame.shape[:2]
            small = cv2.resize(frame, (GATE_WORK_WIDTH, int(GATE_WORK_WIDTH * h0 / w0)),
                               interpolation=cv2.INTER_AREA)

            # --- 昼夜の判定（夜は白黒映像＝色がない）。約1秒ごと ---
            if self.n_frames % 15 == 1:
                sat = float(cv2.cvtColor(cv2.resize(small, (80, 45)), cv2.COLOR_BGR2HSV)[:, :, 1].mean())
                self.tracker.night = sat < 12

            # --- 画面全体の急な明るさ変化（IR切替など）→ 背景を学習し直す ---
            mean = float(small.mean())
            if self.last_mean is not None and abs(mean - self.last_mean) > 25:
                self.request_bg = True
            self.last_mean = mean

            with self.lock:
                self.latest = (ts, wall, frame)

            self.recent.append((ts, frame))
            if SAVE_CLIP:
                self.clipbuf.append((ts, small))

            if self.request_bg:
                for g in self.gates.values():
                    g.reset_bg(small)
                self.height.reset(small)
                self.request_bg = False
            else:
                self.height.update(small, ts)
                events = [(k, g.update(small, ts)) for k, g in self.gates.items()]
                events = [(k, e) for k, e in events if e]
                # 2つの帯が同じコマで同時に「進入」→ 車ではなく画面全体の変化とみなす
                if len(events) == 2 and all(e == "enter" for _, e in events):
                    self.request_bg = True
                    events = []
                for k, e in events:
                    r = self.tracker.on_event(k, e, ts, wall, small.copy() if SAVE_DEBUG else None)
                    if r == "reject":
                        # 光で覆われたと判定された帯を、今の状態を背景として学習し直す
                        # （車本体が来たときに改めて反応させるため）
                        self.gates[k].reset_bg(small)
                        continue
                    self.track_exit(k, e, ts)
                self.flush_awaiting(ts)
                res = self.tracker.poll(ts, small.copy() if SAVE_DEBUG else None)
                if res:
                    res["ts_source"] = self.ts_source
                    # 前端が2つ目の帯に達した瞬間のコマ（車体が大きく写る）
                    snap = min(self.recent, key=lambda x: abs(x[0] - res["t_enter_second"]))[1]
                    # 車が帯を抜けるまで待って「長さ」を推定してから確定する
                    res["exit"] = {res["first_gate"]: None, res["second_gate"]: None}
                    res["exit_cand"] = {res["first_gate"]: None, res["second_gate"]: None}
                    # 1つ目の帯がすでに空いていれば、その時刻を候補にする
                    g1 = self.gates[res["first_gate"]]
                    if not g1.covered and getattr(g1, "last_exit", None):
                        res["exit_cand"][res["first_gate"]] = g1.last_exit
                    self.awaiting.append((res, snap.copy()))

            if self.is_file and self.realtime_file:
                # 録画ファイルは実際の速さで再生（ライブと同じ条件でAI判定させる）
                lag = ts - (time.monotonic() - mono0)
                if lag > 0:
                    time.sleep(lag)

        self.flush_awaiting(1e9, force=True)
        cap.release()
        self.finished = True


# ========================================
# YOLO（車種の確認）
# ========================================

class Detector:
    def __init__(self, enabled):
        self.model = None
        self.history = deque(maxlen=200)   # (ts, detections)
        self.last_info = {}
        if enabled:
            from ultralytics import YOLO
            print("YOLOモデルを読み込みます...")
            self.model = YOLO(MODEL_PATH)
            print("YOLO準備完了")

    def detect(self, ts, frame):
        if self.model is None:
            return []
        h, w = frame.shape[:2]
        # 夜間（暗い・白黒）のときはコントラストを補正してから認識
        det_img = frame
        if frame.mean() < 70:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            det_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        r = self.model.predict(det_img, conf=CONF, imgsz=IMG_SIZE,
                               classes=VEHICLE_CLASSES, verbose=False)[0]
        dets = []
        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                dets.append({
                    "label": self.model.names[cls_id],
                    "conf": float(box.conf[0]),
                    "box": (x1, y1, x2, y2),
                    "wratio": (x2 - x1) / w,
                    "area": (x2 - x1) * (y2 - y1) / (w * h),
                })
        self.history.append((ts, dets))
        return dets

    def classify(self, t0, t1):
        """計測区間中に見えた車両のAI判定を集計して車種を決める"""
        votes = {}
        max_area = 0.0
        best_conf = 0.0
        for ts, dets in self.history:
            if t0 - 0.5 <= ts <= t1 + 1.0:
                for d in dets:
                    lab = "truck" if d["label"] in TRUCK_LABELS else d["label"]
                    votes[lab] = votes.get(lab, 0.0) + d["conf"] * d["area"]
                    max_area = max(max_area, d["area"])
                    best_conf = max(best_conf, d["conf"])
        self.last_info = {k: round(v, 3) for k, v in votes.items()}
        self.last_info["max_area"] = round(max_area, 2)
        if not votes:
            return "unknown", 0.0
        top = max(votes, key=votes.get)
        if max_area >= TRUCK_FORCE_AREA:
            return "TRUCK(large)", best_conf
        if top == "truck" and max_area >= TRUCK_MIN_AREA:
            return "TRUCK", best_conf
        return top, best_conf


# ========================================
# 描画・保存
# ========================================

def draw_height(img, occ=None):
    h, w = img.shape[:2]
    hc = HeightColumn()
    for a, b, c, d in CONTROL_REGIONS:
        cv2.rectangle(img, (int(a * w), int(b * h)), (int(c * w), int(d * h)), (200, 0, 200), 1)
    for i, (x1, y1, x2, y2) in enumerate(hc.rects(w, h)):
        col = (0, 0, 255) if occ and occ[i] else (255, 200, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, max(1, int(w / 640)))


def draw_gates(img, gates):
    h, w = img.shape[:2]
    lw = max(1, int(2 * w / 640))
    for g in gates.values():
        for c, (x1, y1, x2, y2) in zip(g.cells, g.cell_rects(w, h)):
            col = (0, 0, 255) if c.covered else (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, lw)
        x1, y1, _, _ = g.rect(w, h)
        col = (0, 0, 255) if g.covered else (0, 255, 0)
        cv2.putText(img, f"{g.key}", (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * w / 640, col, lw)


def draw_result_panel(img, res, kind):
    h, w = img.shape[:2]
    s = w / 640
    ph = int(70 * s)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - ph), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    line1 = f"SPEED {res['front_kmh']:.1f} km/h"
    hm = res.get("height_max")
    hs = f"  H {hm}/{HEIGHT_CELLS}" if hm is not None else ""
    line2 = (f"{res['wall']:%Y-%m-%d %H:%M:%S}  {kind}  dir {res['direction']}  "
             f"{res['dt']:.2f}s / {GATE_DISTANCE_M:.0f}m{hs}")
    cv2.putText(img, line1, (int(10 * s), h - int(38 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1 * s, (0, 255, 255), max(2, int(3 * s)))
    cv2.putText(img, line2, (int(10 * s), h - int(12 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s, (255, 255, 255), max(1, int(1.5 * s)))


def save_result(res, frame, kind, conf, gates):
    OUT_DIR.mkdir(exist_ok=True)
    day_dir = OUT_DIR / f"{res['wall']:%Y%m%d}"
    day_dir.mkdir(exist_ok=True)
    fname = f"{res['wall']:%Y%m%d_%H%M%S}_{res['front_kmh']:.1f}kmh_{kind.split('(')[0]}.jpg"
    path = day_dir / fname
    if frame is not None:
        img = frame.copy()
        draw_gates(img, gates)
        draw_height(img)
        draw_result_panel(img, res, kind)
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    new = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(["日時", "速度(km/h)", "補正前(km/h)", "通過時間(秒)",
                         "距離(m)", "方向", "車種", "信頼度", "長さ推定(m)", "高さ(マス)", "昼夜", "画像"])
        wr.writerow([f"{res['wall']:%Y-%m-%d %H:%M:%S}", f"{res['front_kmh']:.1f}",
                     f"{res.get('raw_kmh', res['front_kmh']):.1f}",
                     f"{res['dt']:.3f}", GATE_DISTANCE_M, res["direction"], kind,
                     f"{conf:.2f}", f"{res['len_est']:.1f}" if res.get("len_est") is not None else "",
                     res.get("height_max", ""), res.get("daynight", ""),
                     str(path) if frame is not None else ""])
    return path


def save_clip(res, path, gates=None):
    """計測した車の前後の動画を保存。各コマに時刻と、反応した瞬間の印を入れる"""
    clip = res.get("clip")
    if not clip:
        return
    dbg = path.parent / "debug"
    dbg.mkdir(exist_ok=True)
    h, w = clip[0][1].shape[:2]
    out = cv2.VideoWriter(str(dbg / (path.stem + ".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
    for t, f in clip:
        im = f.copy()
        if gates:
            for g in gates.values():
                for x1, y1, x2, y2 in g.cell_rects(w, h):
                    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 255), 1)
        draw_height(im)
        mark = ""
        if abs(t - res["t_enter_first"]) < 0.04:
            mark = f"  <== Gate {res['first_gate']} enter"
        if abs(t - res["t_enter_second"]) < 0.04:
            mark = f"  <== Gate {res['second_gate']} enter"
        cv2.putText(im, f"t={t - res['t_enter_first']:+.3f}s{mark}", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        out.write(im)
    out.release()


def save_debug(res, path, gates=None):
    """2つのマンホールで反応した瞬間を並べた画像（調整・検証用）"""
    a, b = res.get("img_first"), res.get("img_second")
    if a is None or b is None:
        return
    tiles = []
    for img, t, label in ((a, res["t_enter_first"], res["first_gate"]),
                          (b, res["t_enter_second"], res["second_gate"])):
        im = img.copy()
        h, w = im.shape[:2]
        if gates:
            for g in gates.values():
                for x1, y1, x2, y2 in g.cell_rects(w, h):
                    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 255), 1)
        cv2.putText(im, f"Gate {label} enter  t={t:.3f}s", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(im)
    out = np.hstack(tiles)
    dbg = path.parent / "debug"
    dbg.mkdir(exist_ok=True)
    cv2.imwrite(str(dbg / path.name), out, [cv2.IMWRITE_JPEG_QUALITY, 85])


# ========================================
# 計測位置のクリック設定
# ========================================

def load_gate_cfg():
    if Path(GATES_FILE).exists():
        with open(GATES_FILE, encoding="utf-8") as f:
            cfg = json.load(f)
        if "far" in cfg.get("A", {}):
            return cfg
        print("古い形式の gates.json です。既定値を使います（c キーで設定し直してください）")
    return json.loads(json.dumps(DEFAULT_GATES))


def calibrate(frame, cfg):
    """A奥 → A手前 → B奥 → B手前 の順に4点クリック（帯の上端と下端）"""
    names = ["A far", "A near", "B far", "B near"]
    pts = []
    win = "Calibrate: A-far, A-near, B-far, B-near (ESC=cancel)"
    h, w = frame.shape[:2]

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x / w, y / h))

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while len(pts) < 4:
        img = frame.copy()
        cv2.putText(img, f"click: {names[len(pts)]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        for i, (px, py) in enumerate(pts):
            cv2.circle(img, (int(px * w), int(py * h)), 6, (0, 0, 255), 2)
            cv2.putText(img, names[i], (int(px * w) + 8, int(py * h)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow(win, img)
        if cv2.waitKey(30) & 0xFF == 27:
            cv2.destroyWindow(win)
            return cfg
    cv2.destroyWindow(win)
    cfg["A"]["far"], cfg["A"]["near"] = list(pts[0]), list(pts[1])
    cfg["B"]["far"], cfg["B"]["near"] = list(pts[2]), list(pts[3])
    with open(GATES_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print("計測位置を保存しました:", GATES_FILE)
    return cfg


# ========================================
# メイン
# ========================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="録画ファイル（省略時はTapoのRTSP）")
    ap.add_argument("--no-yolo", action="store_true", help="YOLOを使わない")
    ap.add_argument("--no-display", action="store_true", help="画面表示しない")
    ap.add_argument("--gates", help="計測位置のJSONファイル")
    ap.add_argument("--all", action="store_true", help="トラック以外も記録する")
    ap.add_argument("--fast", action="store_true", help="録画ファイルを最高速で処理（AI判定は間引かれる）")
    ap.add_argument("--use-arrival", action="store_true",
                    help="映像の時刻情報(PTS)を使わず、受信時刻で計算する")
    ap.add_argument("--distance", type=float, help="マンホール間の距離 [m]（既定 10.0）")
    ap.add_argument("--correction", type=float, help="速度の補正係数（既定 1.00）")
    ap.add_argument("--truck-correction", type=float, help="夜間のトラック補正係数（既定 1.15）")
    ap.add_argument("--truck-correction-day", type=float, help="昼間のトラック補正係数（既定 1.15）")
    ap.add_argument("--height-cells", type=int, help="高さ判定の列のマスの数（既定 10）")
    ap.add_argument("--height-min", type=int, help="この数以上のマスを覆えばトラック（既定 9）")
    ap.add_argument("--direction", choices=["A->B", "B->A", "both"],
                    help="夜に記録する進行方向（既定 A->B = 丸→四角）")
    ap.add_argument("--direction-day", choices=["A->B", "B->A", "both"],
                    help="昼に記録する進行方向（既定 both = 両方向）")
    ap.add_argument("--ts-log", help="各コマの時刻情報をCSVに書き出す（調査用）")
    ap.add_argument("--no-order-check", action="store_true", help="順序チェック（光による誤反応の除外）を使わない")
    ap.add_argument("--no-height", action="store_true", help="高さ判定を使わない（AIでトラックと判定できた車だけ記録）")
    # 旧版のオプション（指定してもエラーにならないよう受け付けるが使わない）
    ap.add_argument("--night-len", type=float, help=argparse.SUPPRESS)
    ap.add_argument("--night-cells", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--no-night", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--cover-th", type=float, help="進入判定のしきい値（既定 0.45。上げると敏感、下げると鈍感）")
    args = ap.parse_args()

    global GATES_FILE, GATE_DISTANCE_M, SPEED_CORRECTION, COVER_TH, MAX_GAP_SEC, TRUCK_CORRECTION
    global TRUCK_CORRECTION_DAY
    global HEIGHT_CELLS, HEIGHT_MIN, HEIGHT_JUDGE, ORDER_CHECK, DIRECTION, DIRECTION_DAY
    if args.gates:
        GATES_FILE = args.gates
    if args.distance:
        GATE_DISTANCE_M = args.distance
        MAX_GAP_SEC = GATE_DISTANCE_M / (MIN_SPEED_KMH / 3.6)
    if args.correction:
        SPEED_CORRECTION = args.correction
    if args.truck_correction:
        TRUCK_CORRECTION = args.truck_correction
    if args.truck_correction_day:
        TRUCK_CORRECTION_DAY = args.truck_correction_day
    if args.direction:
        DIRECTION = args.direction
    if args.direction_day:
        DIRECTION_DAY = args.direction_day
    if args.no_order_check:
        ORDER_CHECK = False
    if args.height_cells:
        HEIGHT_CELLS = args.height_cells
    if args.height_min:
        HEIGHT_MIN = args.height_min
    if args.no_height or args.no_night:
        HEIGHT_JUDGE = False
    if args.night_len or args.night_cells:
        print("※ --night-len / --night-cells はこの版では使いません（高さ判定に変わりました）")
    if args.cover_th:
        COVER_TH = args.cover_th
    print(f"区間 {GATE_DISTANCE_M} m / 補正係数 {SPEED_CORRECTION} / トラック補正 昼 {TRUCK_CORRECTION_DAY}・夜 {TRUCK_CORRECTION} / 判定しきい値 {COVER_TH}")
    print(f"記録する方向: 昼 {DIRECTION_DAY} / 夜 {DIRECTION}（A=丸マンホール、B=四角マンホール）")
    print(f"高さ判定 {'あり' if HEIGHT_JUDGE else 'なし'}（中央の列 {HEIGHT_CELLS} マス中 {HEIGHT_MIN} マス以上でトラック）")

    if args.source:
        source, is_file = args.source, True
    else:
        user, pw = os.environ["TAPO_USER"], os.environ["TAPO_PASS"]
        source = f"rtsp://{quote(user, safe='')}:{quote(pw, safe='')}@{CAMERA_IP}:554/{STREAM}"
        is_file = False

    cfg = load_gate_cfg()
    gates = {k: Gate(k, cfg) for k in ("A", "B")}
    tracker = SpeedTracker(GATE_DISTANCE_M, gates)
    detector = Detector(enabled=not args.no_yolo)

    results_q = deque()
    last_result = {"res": None, "kind": "", "until": 0}

    def on_result(res, snap):
        results_q.append((res, snap))

    def finalize(res, snap):
        kind, conf = detector.classify(res["t_enter_first"], res["t_enter_second"])
        info = detector.last_info
        L = res.get("len_est")
        car_votes = info.get("car", 0.0)
        H = res.get("height_max", 0)
        # トラックかどうかは「高さ」で決める（昼夜共通）。
        # 近くを通る乗用車は大きく写るため、AIや画面上の大きさだけではトラックと間違える。
        if HEIGHT_JUDGE:
            top = res.get("height_top", False)
            if snap is not None:
                _sat = float(cv2.cvtColor(cv2.resize(snap, (160, 90)), cv2.COLOR_BGR2HSV)[:, :, 1].mean())
                if _sat < 12 and res.get("top_bright", 999) < NIGHT_TOP_BRIGHT:
                    top = False         # 夜：上のマスが明るくならない＝荷台ではない（乗用車）
            if H >= HEIGHT_MIN and top:
                kind = "TRUCK(AI+height)" if kind.startswith("TRUCK") else "TRUCK(height)"
            else:
                kind = f"low(H{H}{'' if top else ',top-'})"   # 車高が足りない（乗用車・ワンボックス車）
        is_truck = kind.startswith("TRUCK")
        # 昼夜の判定：夜はTapoが赤外線の白黒映像になるので、色の濃さ（彩度）で見分ける
        night = True
        if snap is not None:
            sat = float(cv2.cvtColor(cv2.resize(snap, (160, 90)), cv2.COLOR_BGR2HSV)[:, :, 1].mean())
            night = sat < 12
        res["daynight"] = "night" if night else "day"
        corr = TRUCK_CORRECTION if night else TRUCK_CORRECTION_DAY
        if is_truck and corr != 1.0:
            res["raw_kmh"] = res["front_kmh"]
            res["front_kmh"] *= corr
        want = DIRECTION if night else DIRECTION_DAY
        dir_ok = want == "both" or res["direction"].startswith(want)
        if not dir_ok:
            print(f"[対象外] {res['wall']:%H:%M:%S}  {res['front_kmh']:.1f} km/h  方向 {res['direction']}（記録しない方向）"
                  f"  高さ={H}/{HEIGHT_CELLS}")
            return None
        if truck_only and not is_truck:
            Ls = f"{L:.1f}m" if L is not None else "-"
            print(f"[対象外] {res['wall']:%H:%M:%S}  {res['front_kmh']:.1f} km/h  {kind}（記録しません） "
                  f"高さ={H}/{HEIGHT_CELLS}{'(最上段○)' if res.get('height_top') else '(最上段×)'}  上段の明るさ{res.get('top_bright', 0):+.0f}  長さ推定={Ls}  判定内訳={detector.last_info}")
            return None
        path = save_result(res, snap, kind, conf, gates)
        if SAVE_DEBUG:
            save_debug(res, path, gates)
        if SAVE_CLIP:
            save_clip(res, path, gates)
        print(f"[計測] {res['wall']:%Y-%m-%d %H:%M:%S}  {res['front_kmh']:.1f} km/h  "
              f"{res['dt']:.2f}秒  {res['direction']}  {kind}  {res.get('daynight')}  時刻源={res.get('ts_source')}  → {path}")
        Ls = f"{L:.1f}m" if L is not None else "-"
        print(f"        高さ={H}/{HEIGHT_CELLS}{'(最上段○)' if res.get('height_top') else '(最上段×)'}  上段の明るさ{res.get('top_bright', 0):+.0f}  長さ推定={Ls}  判定内訳={detector.last_info}")
        return kind

    truck_only = TRUCK_ONLY and not args.all and detector.model is not None
    if TRUCK_ONLY and detector.model is None and not args.all:
        print("※ YOLOなしのため、車種で絞り込まず全て記録します")
    reader = FrameReader(source, is_file, gates, tracker, on_result,
                         realtime_file=is_file and not args.fast)
    reader.use_arrival = args.use_arrival
    if args.ts_log:
        reader.ts_log = open(args.ts_log, "w", buffering=1)
        reader.ts_log.write("frame,pts,arrival,ts\n")
    reader.start()
    print("計測開始（q で終了、c で計測位置設定、b で背景再学習）")

    last_det_ts = -1e9
    dets = []
    try:
        while True:
            with reader.lock:
                latest = reader.latest
            if latest is None or (reader.finished and not results_q and args.no_display):
                if reader.finished:
                    while results_q:
                        finalize(*results_q.popleft())
                    break
                time.sleep(0.01)
                continue
            ts, wall, frame = latest

            # --- AI認識（一定間隔） ---
            if detector.model is not None and ts - last_det_ts >= DETECT_INTERVAL_SEC:
                dets = detector.detect(ts, frame)
                last_det_ts = ts

            # --- 計測結果の確定・保存（車種判定のため少し待ってから） ---
            while results_q:
                res, snap = results_q.popleft()
                r = finalize(res, snap)
                if r:
                    last_result = {"res": res, "kind": r, "until": ts + 8}

            # --- 画面表示 ---
            if not args.no_display:
                img = frame.copy()
                h, w = img.shape[:2]
                for d in dets:
                    x1, y1, x2, y2 = d["box"]
                    col = (0, 165, 255) if d["label"] in TRUCK_LABELS else (0, 255, 0)
                    cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
                    cv2.putText(img, f"{d['label']} {d['conf']:.2f}", (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                draw_gates(img, gates)
                draw_height(img, reader.height.occ)
                cv2.putText(img, f"FPS {reader.fps:.1f}  time={reader.ts_source}", (10, int(55 * h / 360)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6 * w / 640, (0, 255, 255), 2)
                if last_result["res"] is not None and ts < last_result["until"]:
                    draw_result_panel(img, last_result["res"], last_result["kind"])
                cv2.imshow("Tapo Speed Meter", img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    cfg = calibrate(frame, cfg)
                    for k in ("A", "B"):
                        gates[k].set_points(cfg[k]["far"], cfg[k]["near"])
                    reader.request_bg = True
                if key == ord("b"):
                    reader.request_bg = True
            else:
                if reader.finished and not results_q:
                    break
                time.sleep(0.02)

            if reader.finished and not results_q and args.no_display:
                break
    finally:
        reader.running = False
        reader.join(timeout=3)
        cv2.destroyAllWindows()
    print("終了")


if __name__ == "__main__":
    main()