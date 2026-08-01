#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
speed_line_motion_v8.py

固定2ライン・画像変化方式 速度測定 v8
L1 → L2 専用 / Raspberry Pi + Picamera2 + YOLOv8

v8方針 (v7からの変更点):
- v7最大の問題「YOLOを毎フレーム同期実行 → 実測6fps」を解消する。
  YOLOを専用スレッドに分離し、速度測定ループ（画像差分）はカメラレートで回す。
  これにより実効fpsが大幅に上がり、速い車でもL1/L2を別フレームで捉えられる。
- last_vehicle_* (YOLO結果) はメインスレッドと共有するためロックで保護する。
- YOLOの推論時刻は「推論終了時刻」ではなく「フレーム取得時刻(cap_t)」を使う。
  これでbbox補助判定の経過時間が正確になる。
- 同一フレームを二重処理しないよう、新フレーム到着時のみ処理する。
- 高速車対応として、min_dt / ignore_l2_after_l1_sec のデフォルトを小さくする。
  （fpsが上がったので短いdtも信頼できる）

測定方向・しきい値・bbox補助などの基本ロジックは v7 を踏襲する。
"""

import cv2
import time
import argparse
import threading
import os
from picamera2 import Picamera2


CONFIG = {
    "frame_width": 640,
    "frame_height": 360,

    # v8: メインループからYOLOを外したので高fpsを狙える
    "target_fps": 40,

    # YOLO
    "model_name": "yolov8n.pt",
    "target_classes": [2, 5, 7],  # car, bus, truck
    "confidence_threshold": 0.15,

    # v8: YOLOスレッドの最小実行間隔(秒)。0なら推論が終わり次第すぐ次へ。
    # CPUを他処理に回したい場合のみ 0.1 などに上げる。
    "yolo_min_interval": 0.0,

    # 測定方向
    # L1 = 開始ライン / L2 = 終了ライン
    "line1_x": 150,
    "line2_x": 500,

    # 画像変化を見る範囲
    "motion_y_top": 60,
    "motion_y_bot": 300,

    # strip は広げすぎると誤測定が増えるため標準12
    "strip_half_width": 12,

    # 現場補正後の距離
    "real_distance_m": 2.5,

    # 速度の妥当範囲
    "min_valid_speed": 2,
    "max_valid_speed": 120,

    # 通常の画像変化しきい値
    "motion_threshold": 9.0,
    "motion_ratio_threshold": 0.035,

    # YOLOなしでも、強い動きなら車両ありとみなす
    "allow_motion_without_yolo": True,
    "strong_motion_threshold": 18.0,
    "strong_ratio_threshold": 0.18,

    # YOLO検出後、この秒数は車両ありとして保持
    "vehicle_hold_sec": 3.0,

    # v8.3: 車種ラベル記録
    # 測定がmotionで成立し車種が motion_vehicle のとき、
    # 成立後この秒数だけYOLOの車種検出(car/bus/truck)を待ち、
    # 来たら写真ファイル名とCSVを後から車種名に書き換える。
    # （YOLOは別スレッドで遅れて検出するため、後追いで反映する）
    "label_from_yolo": True,
    "label_grace_sec": 1.2,

    # L1からL2までの許容時間
    # v8: fpsが上がったので min_dt を小さくして高速車も測れるようにする
    # min_dt=0.08, dist=3.0 のとき上限は約135km/h
    "min_dt": 0.08,
    "max_dt": 8.00,

    # L1直後のL2誤反応を無視
    # v8: 高速車の正しい反応を捨てないよう小さめ
    "ignore_l2_after_l1_sec": 0.06,

    # 測定後クールダウン
    # v8.2: スマートクールダウン導入により、これは「最低待ち時間(floor)」になった。
    # 実際の受付再開は「ラインが空く(車が抜ける)まで」を優先する。
    "cooldown_sec": 0.5,

    # v8.2: スマートクールダウン
    # 測定後、L1/L2の動きが収まる(車が完全に抜ける)まで次の測定を開始しない。
    # 長尺車の二重カウントを防ぎつつ、車間が空いた連続車はすぐ分離できる。
    "smart_cooldown": True,
    # ラインが空かないまま(渋滞・連続通行など)この秒数を超えたら強制的に受付再開
    "clear_timeout": 6.0,
    # v8.5: 「空いた」と判定するには、この秒数だけ連続でラインが静かである必要がある。
    # 一様な側面(銀箱シャッター等)の一瞬の無反応で受付再開してしまうのを防ぐ。
    "clear_quiet_sec": 0.3,

    # v8.5: L1開始時にL2が既に反応していたら測定を開始しない。
    # = 車体が両ラインをまたいでいる(長尺車の二重カウント) or 逆走。
    # クリーンなL1→L2計測は「L1反応時にL2は静か」が前提なので弾く。
    "reject_l2_active_at_l1": True,

    # LINE1の誤反応抑制
    "line1_min_ratio": 0.015,

    # L2 bbox補助判定
    # v8.4: 既定OFFに変更。32fps化以降は害（巨大bboxで両ライン同時カバー→偽の高速値）
    # しかない実績のため。使いたい時だけ --l2-bbox-assist で有効化する。
    "l2_bbox_assist": False,
    "l2_bbox_min_area": 20000,
    "l2_bbox_min_elapsed": 0.06,

    # 保存
    "photo_dir": "photos",
    "save_log": True,
    "log_file": "speed_line_motion_v8_log.csv",

    # 表示・ログ
    "headless": False,
    "debug": True,
}


COCO_NAMES = {
    2: "car",
    5: "bus",
    7: "truck",
}


class Picamera2Cap:
    """
    v6で正常動作していた Picamera2 方式。
    ここは変更しない。
    """

    def __init__(self, width=640, height=360, fps=15):
        self.picam = Picamera2()

        config = self.picam.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameRate": float(fps)}
        )

        self.picam.configure(config)
        self.picam.start()

        self._frame = None
        self._ts = 0.0
        self._lock = threading.Lock()
        self._running = True

        threading.Thread(target=self._loop, daemon=True).start()
        time.sleep(1)

    def _loop(self):
        while self._running:
            frame = self.picam.capture_array("main")
            ts = time.time()

            with self._lock:
                self._frame = frame
                self._ts = ts

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None, 0.0

            return True, self._frame.copy(), self._ts

    def release(self):
        self._running = False
        self.picam.stop()


class LineMotionSpeedV8:
    def __init__(self, cfg):
        self.cfg = cfg

        os.makedirs(cfg["photo_dir"], exist_ok=True)

        print("[INFO] Loading YOLOv8 model...")
        from ultralytics import YOLO
        self.model = YOLO(cfg["model_name"])
        print("[INFO] Model loaded!")

        self.prev_gray = None

        # 最後にYOLOで見た車両（YOLOスレッドとメインスレッドで共有）
        self._vehicle_lock = threading.Lock()
        self.last_vehicle_t = 0.0
        self.last_vehicle_label = "vehicle"
        self.last_vehicle_bbox = None
        self.last_vehicle_source = "none"

        # v8: YOLOスレッドに渡す最新フレーム
        self._yolo_frame_lock = threading.Lock()
        self._yolo_frame = None
        self._yolo_frame_ts = 0.0
        self._yolo_running = True

        # 測定状態 (L1 → L2)
        self.state = "WAIT_L1"
        self.l1_time = None
        self.l2_time = None
        self.cooldown_until = 0.0

        # v8.2: スマートクールダウン用
        # require_clear=True の間は、ラインが空くまで次の測定を始めない
        self.require_clear = False
        self.require_clear_since = 0.0
        # v8.5: ラインが静かになり始めた時刻(継続判定用)。Noneは「今は静かでない」
        self.quiet_since = None

        # v8.3: 車種ラベル後追い反映用
        # ログ行・ファイルはYOLOスレッドからも書き換えるのでロックで保護
        self._log_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self.pending_label = None  # {until, photo, log_idx}

        # 立ち上がり検出用
        self.last_l1_high = False
        self.last_l2_high = False

        # 同一フレーム二重処理防止
        self.last_cap_t = 0.0

        self.recent_speeds = []

        self.log_lines = [
            "timestamp,vehicle,source,l2_source,speed_kmh,dt,l1_motion,l2_motion,l1_ratio,l2_ratio,photo\n"
        ]

        # heartbeat
        self.hb_t = time.time()
        self.hb_frames = 0
        self.hb_yolo_dets = 0

        print("======================================")
        print("[INFO] speed_line_motion_v8.py")
        print("[INFO] 測定方向: L1 → L2  / YOLO別スレッド化(高fps)")
        print("======================================")
        print(f"[CFG] L1(start)={cfg['line1_x']} L2(end)={cfg['line2_x']} dist={cfg['real_distance_m']}m")
        print(f"[CFG] target_fps={cfg['target_fps']} yolo_min_interval={cfg['yolo_min_interval']}")
        print(f"[CFG] motion_y={cfg['motion_y_top']}..{cfg['motion_y_bot']} strip={cfg['strip_half_width']}")
        print(f"[CFG] speed_range={cfg['min_valid_speed']}..{cfg['max_valid_speed']} km/h")
        print(f"[CFG] min_dt={cfg['min_dt']} max_dt={cfg['max_dt']}")
        print(f"[CFG] ignore_l2_after_l1={cfg['ignore_l2_after_l1_sec']}")
        print(f"[CFG] cooldown(floor)={cfg['cooldown_sec']} line1_min_ratio={cfg['line1_min_ratio']}")
        print(f"[CFG] smart_cooldown={cfg['smart_cooldown']} clear_timeout={cfg['clear_timeout']} clear_quiet={cfg['clear_quiet_sec']}")
        print(f"[CFG] reject_l2_active_at_l1={cfg['reject_l2_active_at_l1']}")
        print(f"[CFG] label_from_yolo={cfg['label_from_yolo']} label_grace={cfg['label_grace_sec']}")
        print(
            f"[CFG] l2_bbox_assist={cfg['l2_bbox_assist']} "
            f"min_area={cfg['l2_bbox_min_area']} "
            f"min_elapsed={cfg['l2_bbox_min_elapsed']}"
        )
        # min_dtから理論上の測定上限速度を表示
        if cfg["min_dt"] > 0:
            v_cap = cfg["real_distance_m"] / cfg["min_dt"] * 3.6
            print(f"[CFG] 測定上限(min_dt由来) ≈ {v_cap:.0f} km/h")
        print("======================================")

    # =========================
    # YOLO（別スレッド）
    # =========================

    def detect_vehicle_yolo(self, frame, now):
        """
        YOLOは車両あり判定と、L2補助判定用bbox取得に使う。
        速度計算の主軸には使わない。
        v8: YOLOスレッドから呼ばれる。
        """

        results = self.model(
            frame,
            classes=self.cfg["target_classes"],
            conf=self.cfg["confidence_threshold"],
            verbose=False,
        )[0]

        best = None
        best_area = 0

        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                label = COCO_NAMES.get(cls_id, "vehicle")
                area = (x2 - x1) * (y2 - y1)

                if area > best_area:
                    best_area = area
                    best = {
                        "bbox": (x1, y1, x2, y2),
                        "label": label,
                        "conf": conf,
                        "area": area,
                    }

        if best is not None:
            with self._vehicle_lock:
                self.last_vehicle_t = now
                self.last_vehicle_label = best["label"]
                self.last_vehicle_bbox = best["bbox"]
                self.last_vehicle_source = "yolo"
                self.hb_yolo_dets += 1

            if self.cfg["debug"]:
                x1, y1, x2, y2 = best["bbox"]
                print(
                    f"[YOLO] {best['label']} "
                    f"conf={best['conf']:.2f} "
                    f"bbox=({x1},{y1},{x2},{y2}) "
                    f"area={best['area']}"
                )

            # v8.3: 直前にmotionで成立した測定の車種ラベルを後追い反映
            self.try_apply_pending_label(best["label"], now)

            return best

        return None

    def try_apply_pending_label(self, label, now):
        """
        v8.3:
        motionで成立し車種が motion_vehicle になった測定について、
        成立後の猶予時間内にYOLOが car/bus/truck を検出したら、
        写真ファイル名とCSV行をその車種名に後追いで書き換える。
        （YOLOスレッドから呼ばれる）
        """
        if not self.cfg["label_from_yolo"]:
            return

        if label not in ("car", "bus", "truck"):
            return

        with self._pending_lock:
            p = self.pending_label
            if p is None:
                return

            if now > p["until"]:
                self.pending_label = None
                return

            old_photo = p["photo"]
            new_photo = old_photo.replace("motion_vehicle", label)

            # 写真ファイルをリネーム
            try:
                if old_photo != new_photo and os.path.exists(old_photo):
                    os.rename(old_photo, new_photo)
            except OSError:
                new_photo = old_photo

            # CSV行を書き換え（車種カラムと写真パスの両方の motion_vehicle を置換）
            with self._log_lock:
                idx = p["log_idx"]
                if 0 <= idx < len(self.log_lines):
                    self.log_lines[idx] = self.log_lines[idx].replace(
                        "motion_vehicle", label
                    )
                if self.cfg["save_log"]:
                    try:
                        with open(self.cfg["log_file"], "w") as f:
                            f.writelines(self.log_lines)
                    except OSError:
                        pass

            self.pending_label = None

            if self.cfg["debug"]:
                print(f"[LABEL-UPDATE] motion_vehicle → {label} : {new_photo}")

    def yolo_worker(self):
        """
        v8: 別スレッドで最新フレームに対してYOLOを回し続ける。
        メインの測定ループ(画像差分)を待たせない。
        """
        last_run = 0.0

        while self._yolo_running:
            frame = None
            ts = 0.0

            with self._yolo_frame_lock:
                if self._yolo_frame is not None:
                    frame = self._yolo_frame
                    ts = self._yolo_frame_ts
                    self._yolo_frame = None

            if frame is None:
                time.sleep(0.005)
                continue

            interval = self.cfg["yolo_min_interval"]
            if interval > 0.0:
                wait = last_run + interval - time.time()
                if wait > 0:
                    time.sleep(wait)

            last_run = time.time()

            try:
                self.detect_vehicle_yolo(frame, ts)
            except Exception as e:
                print(f"[YOLO-ERR] {e}")

    def submit_frame_for_yolo(self, frame, cap_t):
        with self._yolo_frame_lock:
            # 常に最新フレームだけを残す（古いものは破棄）
            self._yolo_frame = frame
            self._yolo_frame_ts = cap_t

    def yolo_active(self, now):
        with self._vehicle_lock:
            return (now - self.last_vehicle_t) <= self.cfg["vehicle_hold_sec"]

    def clear_vehicle_active(self):
        """
        測定成功後に同じ車両の残りを拾わないように、
        YOLO active状態を切る。
        """
        with self._vehicle_lock:
            self.last_vehicle_t = 0.0
            self.last_vehicle_label = "vehicle"
            self.last_vehicle_bbox = None
            self.last_vehicle_source = "none"

    def get_vehicle_snapshot(self):
        """共有車両状態をロックして一括取得。"""
        with self._vehicle_lock:
            return (
                self.last_vehicle_t,
                self.last_vehicle_label,
                self.last_vehicle_bbox,
                self.last_vehicle_source,
            )

    def bbox_assist_l2(self, now):
        """
        LINE1通過後、L2の画像差分が弱い場合に、
        YOLO bbox が L2 を含んでいれば L2通過補助とする。
        """

        if not self.cfg["l2_bbox_assist"]:
            return False

        if self.l1_time is None:
            return False

        with self._vehicle_lock:
            bbox = self.last_vehicle_bbox

        if bbox is None:
            return False

        elapsed = now - self.l1_time

        # 早すぎるbbox判定は誤反応になりやすいので無視
        if elapsed < self.cfg["l2_bbox_min_elapsed"]:
            return False

        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)

        if area < self.cfg["l2_bbox_min_area"]:
            return False

        l2 = self.cfg["line2_x"]

        if x1 <= l2 <= x2:
            if self.cfg["debug"]:
                print(
                    f"[L2-BBOX-ASSIST] "
                    f"bbox=({x1},{y1},{x2},{y2}) "
                    f"area={area} "
                    f"elapsed={elapsed:.3f}s"
                )
            return True

        return False

    # =========================
    # Motion
    # =========================

    def get_strip_motion(self, diff, line_x):
        h, w = diff.shape[:2]

        y1 = max(0, self.cfg["motion_y_top"])
        y2 = min(h, self.cfg["motion_y_bot"])

        half = self.cfg["strip_half_width"]

        x1 = max(0, line_x - half)
        x2 = min(w, line_x + half)

        strip = diff[y1:y2, x1:x2]

        if strip.size == 0:
            return 0.0, 0.0

        mean_motion = float(strip.mean())
        changed = strip > 25
        ratio_motion = float(changed.mean())

        return mean_motion, ratio_motion

    def motion_is_high(self, mean_motion, ratio_motion):
        return (
            mean_motion >= self.cfg["motion_threshold"] or
            ratio_motion >= self.cfg["motion_ratio_threshold"]
        )

    def motion_is_strong(self, mean_motion, ratio_motion):
        return (
            mean_motion >= self.cfg["strong_motion_threshold"] or
            ratio_motion >= self.cfg["strong_ratio_threshold"]
        )

    # =========================
    # 状態管理
    # =========================

    def update_motion_state(
        self,
        now,
        frame,
        l1_motion,
        l1_ratio,
        l2_motion,
        l2_ratio,
    ):
        if now < self.cooldown_until:
            return

        yolo_active = self.yolo_active(now)

        normal_l1_motion = self.motion_is_high(l1_motion, l1_ratio)
        normal_l2_motion = self.motion_is_high(l2_motion, l2_ratio)

        # v8.2: スマートクールダウン
        # 直前の測定後、車がまだ通過中ならライン上に動きが残る。
        # L1/L2が静かになる(車が抜ける)まで次の測定を開始しない
        #   → 長尺車の二重カウントを防ぐ。
        # 静かになった瞬間に受付再開
        #   → 車間が空いた連続車はすぐ分離できる。
        if self.require_clear:
            lines_quiet = (not normal_l1_motion) and (not normal_l2_motion)

            # v8.5: 一瞬の無反応では解除しない。clear_quiet_sec 連続で静かなときだけ解除。
            # （一様な側面の長尺車が、平らな面の通過中に「空いた」と誤認されるのを防ぐ）
            if lines_quiet:
                if self.quiet_since is None:
                    self.quiet_since = now
                quiet_enough = (now - self.quiet_since) >= self.cfg["clear_quiet_sec"]
            else:
                self.quiet_since = None
                quiet_enough = False

            timed_out = (now - self.require_clear_since) > self.cfg["clear_timeout"]

            if quiet_enough or timed_out:
                if timed_out and not quiet_enough and self.cfg["debug"]:
                    print("[CLEAR-TIMEOUT] ラインが空かないまま時間切れ → 受付再開")
                self.require_clear = False
                self.quiet_since = None
                self.last_l1_high = False
                self.last_l2_high = False
            else:
                # まだ車が通過中。新しい測定は始めない。
                return

        strong_l1_motion = self.motion_is_strong(l1_motion, l1_ratio)
        strong_l2_motion = self.motion_is_strong(l2_motion, l2_ratio)

        if self.cfg["allow_motion_without_yolo"]:
            active = yolo_active or strong_l1_motion or strong_l2_motion
        else:
            active = yolo_active

        if active and not yolo_active:
            with self._vehicle_lock:
                self.last_vehicle_label = "motion_vehicle"
                self.last_vehicle_source = "motion"

        # -------------------------
        # L1 → L2
        # -------------------------
        if self.state == "WAIT_L1":
            # LINE1は新しい車両の入口なので少し厳しめ
            l1_high = (
                active
                and normal_l1_motion
                and l1_ratio >= self.cfg["line1_min_ratio"]
            )
            l2_high = False

        elif self.state == "WAIT_L2":
            # LINE1後は active が切れても LINE2 を見る
            l1_high = False
            l2_high = normal_l2_motion

        else:
            l1_high = False
            l2_high = False

        l1_rise = l1_high and not self.last_l1_high
        l2_rise = l2_high and not self.last_l2_high

        self.last_l1_high = l1_high
        self.last_l2_high = l2_high

        # -------------------------
        # L1待ち 開始
        # -------------------------
        if self.state == "WAIT_L1":
            if l1_rise:
                # v8.5: L1開始時にL2が既に反応していたら測定を開始しない。
                # = 車体が両ラインをまたいでいる(長尺車の二重カウント) or 逆走。
                # クリーンなL1→L2計測は「L1反応時にL2は静か」が前提なので弾く。
                if self.cfg["reject_l2_active_at_l1"] and normal_l2_motion:
                    print(
                        "[REJECT] L1開始時にL2が反応中 "
                        f"(L2={l2_motion:.1f}/{l2_ratio:.3f}) "
                        "→ 長尺車またぎ/逆走の可能性。測定開始せず車が抜けるまで待機"
                    )
                    # require_clear で「車が抜けるまで」次の開始を抑止する
                    self.start_cooldown(now, floor=0.3)
                    return

                self.l1_time = now
                self.state = "WAIT_L2"

                # 念のため L1開始時のL2状態を初期値にしておく
                # （reject無効時でも継続反応を立ち上がりと誤認しないため）
                self.last_l2_high = normal_l2_motion

                _, veh_label, _, veh_source = self.get_vehicle_snapshot()

                print(
                    f"[LINE1-START] "
                    f"motion={l1_motion:.1f} "
                    f"ratio={l1_ratio:.3f} "
                    f"vehicle={veh_label} "
                    f"source={veh_source}"
                )

                self.save_debug_photo(
                    "line1",
                    frame,
                    l1_motion,
                    l2_motion,
                    l1_ratio,
                    l2_ratio,
                )

        # -------------------------
        # L2待ち 終了
        # -------------------------
        elif self.state == "WAIT_L2":
            elapsed = now - self.l1_time if self.l1_time is not None else 0.0

            if elapsed > self.cfg["max_dt"]:
                print(
                    f"[RESET] L2が来ないためリセット "
                    f"elapsed={elapsed:.3f}s"
                )
                self.start_cooldown(now, floor=0.5)
                self.reset_measurement()
                return

            # L1直後のL2誤反応を無視
            if elapsed < self.cfg["ignore_l2_after_l1_sec"]:
                if self.cfg["debug"] and (l2_rise or normal_l2_motion):
                    print(
                        f"[IGNORE] L1直後のL2反応を無視 "
                        f"elapsed={elapsed:.3f}s "
                        f"L2={l2_motion:.1f}/{l2_ratio:.3f}"
                    )
                # v8.1: 無視窓の間は L2 の「実際の状態」を追従させる。
                # 以前は False に固定していたため、長尺車で L2 が継続反応している
                # と、無視窓が明けた瞬間に偽の立ち上がりになって
                # dt≈無視窓時間 の異常な高速値が出ていた。
                # 実状態を保持すれば、継続反応は立ち上がりとみなされない。
                self.last_l2_high = normal_l2_motion

                return

            # 基本は画像差分、補助でYOLO bbox
            l2_by_motion = l2_rise
            l2_by_bbox = self.bbox_assist_l2(now)

            if l2_by_motion or l2_by_bbox:
                self.l2_time = now
                dt = self.l2_time - self.l1_time

                l2_source = "motion" if l2_by_motion else "bbox_assist"

                # 早すぎる測定は破棄せず、WAIT_L2のまま待つ
                if dt < self.cfg["min_dt"]:
                    print(
                        f"[IGNORE] L2が早すぎる "
                        f"dt={dt:.3f}s "
                        f"source={l2_source} "
                        f"L2={l2_motion:.1f}/{l2_ratio:.3f}"
                    )
                    return

                if dt > self.cfg["max_dt"]:
                    print(
                        f"[SKIP] dt範囲外 "
                        f"dt={dt:.3f}s source={l2_source}"
                    )
                    self.start_cooldown(now, floor=0.5)
                    self.reset_measurement()
                    return

                speed_kmh = self.cfg["real_distance_m"] / dt * 3.6

                print(
                    f"[LINE2-END] source={l2_source} "
                    f"motion={l2_motion:.1f} "
                    f"ratio={l2_ratio:.3f} "
                    f"dt={dt:.3f}s "
                    f"speed={speed_kmh:.1f}km/h"
                )

                _, veh_label, _, _ = self.get_vehicle_snapshot()

                if (
                    self.cfg["min_valid_speed"]
                    <= speed_kmh
                    <= self.cfg["max_valid_speed"]
                ):
                    print(
                        f"[SPEED] "
                        f"{veh_label.upper()} "
                        f"{speed_kmh:.1f} km/h "
                        f"(dt={dt:.3f}s)"
                    )

                    self.recent_speeds.append(speed_kmh)

                    if len(self.recent_speeds) > 5:
                        self.recent_speeds.pop(0)

                    photo = self.save_speed_photo(
                        frame,
                        speed_kmh,
                        dt,
                        l1_motion,
                        l2_motion,
                        l1_ratio,
                        l2_ratio,
                        l2_source,
                    )

                    log_idx = self.write_log(
                        speed_kmh,
                        dt,
                        l1_motion,
                        l2_motion,
                        l1_ratio,
                        l2_ratio,
                        l2_source,
                        photo,
                    )

                    # v8.3: 車種が motion_vehicle のままなら、
                    # 成立後の猶予時間内にYOLO車種が来たら後追いで反映する
                    if (
                        self.cfg["label_from_yolo"]
                        and veh_label == "motion_vehicle"
                        and "motion_vehicle" in photo
                    ):
                        with self._pending_lock:
                            self.pending_label = {
                                "until": now + self.cfg["label_grace_sec"],
                                "photo": photo,
                                "log_idx": log_idx,
                            }

                    self.start_cooldown(now)
                    self.clear_vehicle_active()
                    self.reset_measurement()

                else:
                    print(
                        f"[SKIP] speed範囲外 "
                        f"{speed_kmh:.1f} km/h "
                        f"(dt={dt:.3f}s source={l2_source})"
                    )
                    self.start_cooldown(now)
                    self.clear_vehicle_active()
                    self.reset_measurement()

    def start_cooldown(self, now, floor=None):
        """
        v8.2: 測定後のクールダウン開始。
        floor秒は最低でも待つ。さらに smart_cooldown が有効なら、
        ライン上の動きが収まる(車が抜ける)まで次の測定を始めない。
        """
        if floor is None:
            floor = self.cfg["cooldown_sec"]

        self.cooldown_until = now + floor

        if self.cfg["smart_cooldown"]:
            self.require_clear = True
            self.require_clear_since = now
            self.quiet_since = None

    def reset_measurement(self):
        self.state = "WAIT_L1"
        self.l1_time = None
        self.l2_time = None
        self.last_l1_high = False
        self.last_l2_high = False

    # =========================
    # 保存
    # =========================

    def make_timestamp(self):
        return time.strftime("%Y%m%d_%H%M%S") + f"_{int((time.time() % 1) * 1000):03d}"

    def save_debug_photo(
        self,
        prefix,
        frame,
        l1_motion,
        l2_motion,
        l1_ratio,
        l2_ratio,
    ):
        ts = self.make_timestamp()
        filename = f"{self.cfg['photo_dir']}/{ts}_{prefix}.jpg"

        _, veh_label, veh_bbox, veh_source = self.get_vehicle_snapshot()

        img = frame.copy()
        self.draw_guides(img)

        cv2.putText(
            img,
            f"{prefix} L1={l1_motion:.1f}/{l1_ratio:.3f} "
            f"L2={l2_motion:.1f}/{l2_ratio:.3f}",
            (5, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            img,
            f"vehicle={veh_label} source={veh_source}",
            (5, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
        )

        if veh_bbox:
            x1, y1, x2, y2 = veh_bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

        cv2.imwrite(filename, img)

        print(f"[DEBUG-PHOTO] 保存: {filename}")

        return filename

    def save_speed_photo(
        self,
        frame,
        speed_kmh,
        dt,
        l1_motion,
        l2_motion,
        l1_ratio,
        l2_ratio,
        l2_source,
    ):
        ts = self.make_timestamp()

        _, veh_label, veh_bbox, veh_source = self.get_vehicle_snapshot()

        filename = (
            f"{self.cfg['photo_dir']}/"
            f"{ts}_{veh_label}_{speed_kmh:.0f}kmh.jpg"
        )

        img = frame.copy()
        self.draw_guides(img)

        cv2.putText(
            img,
            time.strftime("%Y-%m-%d %H:%M:%S"),
            (5, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        cv2.putText(
            img,
            f"{veh_label.upper()} {speed_kmh:.1f} km/h",
            (5, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            img,
            f"direction=L1->L2 source={veh_source} L2={l2_source} dt={dt:.3f}s",
            (5, 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
        )

        cv2.putText(
            img,
            f"L1={l1_motion:.1f}/{l1_ratio:.3f} "
            f"L2={l2_motion:.1f}/{l2_ratio:.3f}",
            (5, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 0),
            1,
        )

        if veh_bbox:
            x1, y1, x2, y2 = veh_bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imwrite(filename, img)

        print(f"[PHOTO] 保存: {filename}")

        return filename

    def write_log(
        self,
        speed_kmh,
        dt,
        l1_motion,
        l2_motion,
        l1_ratio,
        l2_ratio,
        l2_source,
        photo,
    ):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")

        _, veh_label, _, veh_source = self.get_vehicle_snapshot()

        with self._log_lock:
            self.log_lines.append(
                f"{ts},"
                f"{veh_label},"
                f"{veh_source},"
                f"{l2_source},"
                f"{speed_kmh:.1f},"
                f"{dt:.3f},"
                f"{l1_motion:.1f},"
                f"{l2_motion:.1f},"
                f"{l1_ratio:.3f},"
                f"{l2_ratio:.3f},"
                f"{photo}\n"
            )
            log_idx = len(self.log_lines) - 1

            if self.cfg["save_log"]:
                with open(self.cfg["log_file"], "w") as f:
                    f.writelines(self.log_lines)

        return log_idx

    # =========================
    # 表示
    # =========================

    def heartbeat(self, now, l1_motion, l2_motion, active):
        self.hb_frames += 1

        if now - self.hb_t >= 5.0:
            elapsed = now - self.hb_t
            fps = self.hb_frames / elapsed if elapsed > 0 else 0.0

            with self._vehicle_lock:
                yolo_dets = self.hb_yolo_dets
                self.hb_yolo_dets = 0

            print(
                f"[HB] {self.hb_frames}フレーム({fps:.1f}fps) / "
                f"YOLO検出{yolo_dets}回 / "
                f"active={active} / "
                f"L1={l1_motion:.1f} L2={l2_motion:.1f} / "
                f"state={self.state}"
            )

            self.hb_t = now
            self.hb_frames = 0

    def draw_guides(self, frame):
        h, w = frame.shape[:2]

        l1 = self.cfg["line1_x"]
        l2 = self.cfg["line2_x"]

        y_top = self.cfg["motion_y_top"]
        y_bot = self.cfg["motion_y_bot"]

        half = self.cfg["strip_half_width"]

        # ROI
        cv2.line(frame, (0, y_top), (w, y_top), (0, 200, 80), 1)
        cv2.line(frame, (0, y_bot), (w, y_bot), (0, 200, 80), 1)

        # L1 start strip
        cv2.rectangle(
            frame,
            (max(0, l1 - half), y_top),
            (min(w - 1, l1 + half), y_bot),
            (255, 100, 0),
            1,
        )
        cv2.line(frame, (l1, 0), (l1, h), (255, 100, 0), 2)
        cv2.putText(
            frame,
            "L1 START",
            (l1 + 3, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 100, 0),
            1
        )

        # L2 end strip
        cv2.rectangle(
            frame,
            (max(0, l2 - half), y_top),
            (min(w - 1, l2 + half), y_bot),
            (0, 60, 255),
            1,
        )
        cv2.line(frame, (l2, 0), (l2, h), (0, 60, 255), 2)
        cv2.putText(
            frame,
            "L2 END",
            (l2 + 3, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 60, 255),
            1
        )

    def draw(
        self,
        frame,
        l1_motion,
        l1_ratio,
        l2_motion,
        l2_ratio,
        active,
    ):
        self.draw_guides(frame)

        h, w = frame.shape[:2]

        _, veh_label, veh_bbox, veh_source = self.get_vehicle_snapshot()

        cv2.putText(
            frame,
            f"V8 Direction:L1->L2 State:{self.state} Active:{active}",
            (5, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 255, 0),
            1,
        )

        cv2.putText(
            frame,
            f"L1:{l1_motion:.1f}/{l1_ratio:.3f} "
            f"L2:{l2_motion:.1f}/{l2_ratio:.3f}",
            (5, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 255, 0),
            1,
        )

        cv2.putText(
            frame,
            f"vehicle:{veh_label} source:{veh_source}",
            (5, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (200, 255, 0),
            1,
        )

        if veh_bbox and active:
            x1, y1, x2, y2 = veh_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

        if self.recent_speeds:
            cv2.putText(
                frame,
                f"Last:{self.recent_speeds[-1]:.1f} km/h",
                (5, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 100),
                2,
            )

        return frame

    # =========================
    # Run
    # =========================

    def run(self):
        print("[INFO] Starting camera...")

        cap = Picamera2Cap(
            self.cfg["frame_width"],
            self.cfg["frame_height"],
            self.cfg["target_fps"],
        )

        print("[INFO] Camera ready!")

        # v8: YOLOワーカースレッド開始
        yolo_thread = threading.Thread(target=self.yolo_worker, daemon=True)
        yolo_thread.start()
        print("[INFO] YOLO worker thread started")

        print("[INFO] v8 L1→L2 固定ライン画像変化 + L2 bbox補助方式で測定開始")
        print("[INFO] [q]終了 [s]snapshot [r]reset")

        snap_count = 0

        try:
            while True:
                ret, frame, cap_t = cap.read()

                if not ret:
                    time.sleep(0.002)
                    continue

                # 同一フレームの二重処理を避ける（新フレームのみ処理）
                if cap_t == self.last_cap_t:
                    time.sleep(0.002)
                    continue
                self.last_cap_t = cap_t

                now = cap_t

                # YOLOには最新フレームを渡すだけ（推論は別スレッド）
                self.submit_frame_for_yolo(frame, cap_t)
                yolo_active = self.yolo_active(now)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)

                if self.prev_gray is None:
                    self.prev_gray = gray
                    continue

                diff = cv2.absdiff(self.prev_gray, gray)
                self.prev_gray = gray

                l1_motion, l1_ratio = self.get_strip_motion(diff, self.cfg["line1_x"])
                l2_motion, l2_ratio = self.get_strip_motion(diff, self.cfg["line2_x"])

                strong_l1 = self.motion_is_strong(l1_motion, l1_ratio)
                strong_l2 = self.motion_is_strong(l2_motion, l2_ratio)

                if self.cfg["allow_motion_without_yolo"]:
                    active = yolo_active or strong_l1 or strong_l2
                else:
                    active = yolo_active

                self.update_motion_state(
                    now,
                    frame,
                    l1_motion,
                    l1_ratio,
                    l2_motion,
                    l2_ratio,
                )

                self.heartbeat(now, l1_motion, l2_motion, active)

                if not self.cfg["headless"]:
                    disp = self.draw(
                        frame.copy(),
                        l1_motion,
                        l1_ratio,
                        l2_motion,
                        l2_ratio,
                        active,
                    )

                    cv2.imshow("Speed Line Motion V8 L1 to L2", disp)

                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        break

                    elif key == ord("s"):
                        fn = f"snapshot_v8_{snap_count:04d}.jpg"
                        cv2.imwrite(fn, disp)
                        print(f"[SNAP] {fn}")
                        snap_count += 1

                    elif key == ord("r"):
                        self.reset_measurement()
                        self.clear_vehicle_active()
                        self.require_clear = False
                        print("[INFO] Reset")

        except KeyboardInterrupt:
            print("\n[INFO] Stopped.")

        finally:
            self._yolo_running = False
            cap.release()

            if not self.cfg["headless"]:
                cv2.destroyAllWindows()

            if self.cfg["save_log"]:
                with open(self.cfg["log_file"], "w") as f:
                    f.writelines(self.log_lines)

                print(f"[INFO] Log saved: {self.cfg['log_file']}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dist", type=float, default=CONFIG["real_distance_m"])
    parser.add_argument("--line1", type=int, default=CONFIG["line1_x"])
    parser.add_argument("--line2", type=int, default=CONFIG["line2_x"])

    parser.add_argument("--y-top", type=int, default=CONFIG["motion_y_top"])
    parser.add_argument("--y-bot", type=int, default=CONFIG["motion_y_bot"])

    parser.add_argument("--strip", type=int, default=CONFIG["strip_half_width"])

    parser.add_argument("--motion-th", type=float, default=CONFIG["motion_threshold"])
    parser.add_argument("--ratio-th", type=float, default=CONFIG["motion_ratio_threshold"])

    parser.add_argument("--strong-motion-th", type=float, default=CONFIG["strong_motion_threshold"])
    parser.add_argument("--strong-ratio-th", type=float, default=CONFIG["strong_ratio_threshold"])

    parser.add_argument("--conf", type=float, default=CONFIG["confidence_threshold"])
    parser.add_argument("--model", type=str, default=CONFIG["model_name"])

    parser.add_argument("--fps", type=int, default=CONFIG["target_fps"])
    parser.add_argument("--yolo-min-interval", type=float, default=CONFIG["yolo_min_interval"])

    parser.add_argument("--hold", type=float, default=CONFIG["vehicle_hold_sec"])
    parser.add_argument("--label-grace", type=float, default=CONFIG["label_grace_sec"])
    parser.add_argument("--no-yolo-label", action="store_true")

    parser.add_argument("--min-dt", type=float, default=CONFIG["min_dt"])
    parser.add_argument("--max-dt", type=float, default=CONFIG["max_dt"])
    parser.add_argument("--ignore-l2-after-l1", type=float, default=CONFIG["ignore_l2_after_l1_sec"])

    parser.add_argument("--cooldown", type=float, default=CONFIG["cooldown_sec"])
    parser.add_argument("--clear-timeout", type=float, default=CONFIG["clear_timeout"])
    parser.add_argument("--clear-quiet", type=float, default=CONFIG["clear_quiet_sec"])
    parser.add_argument("--no-smart-cooldown", action="store_true")
    parser.add_argument("--no-reject-l2-active", action="store_true")
    parser.add_argument("--line1-min-ratio", type=float, default=CONFIG["line1_min_ratio"])

    # v8.4: bbox補助は既定OFF。使いたい時だけ --l2-bbox-assist で有効化。
    # --no-l2-bbox-assist は後方互換のため残置(指定すると強制OFF)。
    parser.add_argument("--l2-bbox-assist", action="store_true")
    parser.add_argument("--no-l2-bbox-assist", action="store_true")
    parser.add_argument("--l2-bbox-min-area", type=int, default=CONFIG["l2_bbox_min_area"])
    parser.add_argument("--l2-bbox-min-elapsed", type=float, default=CONFIG["l2_bbox_min_elapsed"])

    parser.add_argument("--no-motion-without-yolo", action="store_true")

    parser.add_argument("--min-speed", type=float, default=CONFIG["min_valid_speed"])
    parser.add_argument("--max-speed", type=float, default=CONFIG["max_valid_speed"])

    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--no-debug", action="store_true")

    args = parser.parse_args()

    cfg = CONFIG.copy()

    cfg["real_distance_m"] = args.dist
    cfg["line1_x"] = args.line1
    cfg["line2_x"] = args.line2

    cfg["motion_y_top"] = args.y_top
    cfg["motion_y_bot"] = args.y_bot

    cfg["strip_half_width"] = args.strip

    cfg["motion_threshold"] = args.motion_th
    cfg["motion_ratio_threshold"] = args.ratio_th

    cfg["strong_motion_threshold"] = args.strong_motion_th
    cfg["strong_ratio_threshold"] = args.strong_ratio_th

    cfg["confidence_threshold"] = args.conf
    cfg["model_name"] = args.model

    cfg["target_fps"] = args.fps
    cfg["yolo_min_interval"] = args.yolo_min_interval

    cfg["vehicle_hold_sec"] = args.hold
    cfg["label_grace_sec"] = args.label_grace
    cfg["label_from_yolo"] = not args.no_yolo_label

    cfg["min_dt"] = args.min_dt
    cfg["max_dt"] = args.max_dt
    cfg["ignore_l2_after_l1_sec"] = args.ignore_l2_after_l1

    cfg["cooldown_sec"] = args.cooldown
    cfg["clear_timeout"] = args.clear_timeout
    cfg["clear_quiet_sec"] = args.clear_quiet
    cfg["smart_cooldown"] = not args.no_smart_cooldown
    cfg["reject_l2_active_at_l1"] = not args.no_reject_l2_active
    cfg["line1_min_ratio"] = args.line1_min_ratio

    # 既定OFF。--l2-bbox-assist で有効化、--no-l2-bbox-assist が優先(強制OFF)
    cfg["l2_bbox_assist"] = args.l2_bbox_assist and not args.no_l2_bbox_assist
    cfg["l2_bbox_min_area"] = args.l2_bbox_min_area
    cfg["l2_bbox_min_elapsed"] = args.l2_bbox_min_elapsed

    cfg["allow_motion_without_yolo"] = not args.no_motion_without_yolo

    cfg["min_valid_speed"] = args.min_speed
    cfg["max_valid_speed"] = args.max_speed

    cfg["headless"] = args.headless
    cfg["save_log"] = not args.no_log
    cfg["debug"] = not args.no_debug

    monitor = LineMotionSpeedV8(cfg)
    monitor.run()


if __name__ == "__main__":
    main()
