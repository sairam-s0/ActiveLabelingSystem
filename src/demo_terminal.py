"""
Active Labeling System – Terminal Demo
=======================================
Simulates the full AL pipeline with professional terminal output.
Run:  python demo_terminal.py
"""

from __future__ import annotations

import os
import sys
import time
import random
import math
import warnings
import logging

# ── suppress EVERYTHING ──────────────────────────────────────────────
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
os.environ["RAY_SCHEDULER_EVENTS"] = "0"
os.environ["ULTRALYTICS_QUIET"] = "1"
if "--debug" not in sys.argv:
    sys.stderr = open(os.devnull, "w")

# ── ANSI colours ─────────────────────────────────────────────────────
class C:
    RST   = "\033[0m"
    BOLD  = "\033[1m"
    DIM   = "\033[2m"
    ITAL  = "\033[3m"
    UL    = "\033[4m"
    # foreground
    BLK   = "\033[30m"
    RED   = "\033[31m"
    GRN   = "\033[32m"
    YEL   = "\033[33m"
    BLU   = "\033[34m"
    MAG   = "\033[35m"
    CYN   = "\033[36m"
    WHT   = "\033[37m"
    # bright foreground
    BBLK  = "\033[90m"
    BRED  = "\033[91m"
    BGRN  = "\033[92m"
    BYEL  = "\033[93m"
    BBLU  = "\033[94m"
    BMAG  = "\033[95m"
    BCYN  = "\033[96m"
    BWHT  = "\033[97m"
    # backgrounds
    BG_BLK = "\033[40m"
    BG_GRN = "\033[42m"
    BG_YEL = "\033[43m"
    BG_BLU = "\033[44m"
    BG_MAG = "\033[45m"
    BG_CYN = "\033[46m"
    BG_WHT = "\033[47m"
    BG_BBLK = "\033[100m"

# enable ANSI on Windows
if sys.platform == "win32":
    os.system("")  # enables VT100
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

W = 78  # terminal width for boxes

# ── helpers ──────────────────────────────────────────────────────────
_t0 = time.time()

def ts() -> str:
    elapsed = time.time() - _t0
    m, s = divmod(elapsed, 60)
    ms = int((s % 1) * 1000)
    return f"{int(m):02d}:{int(s):02d}.{ms:03d}"

def bar(pct: float, width: int = 30, fill="█", empty="░") -> str:
    filled = int(width * pct)
    return fill * filled + empty * (width - filled)

def debug(msg: str):
    print(f"  {C.BBLK}{ts()}  DEBUG  {msg}{C.RST}")

def info(msg: str):
    print(f"  {C.BCYN}{ts()}{C.RST}  {C.BOLD}{C.CYN}INFO{C.RST}   {msg}")

def success(msg: str):
    print(f"  {C.BGRN}{ts()}{C.RST}  {C.BOLD}{C.BG_GRN}{C.BLK} ✓ SUCCESS {C.RST}  {C.BOLD}{C.GRN}{msg}{C.RST}")

def warn(msg: str):
    print(f"  {C.BYEL}{ts()}{C.RST}  {C.BOLD}{C.YEL}WARN{C.RST}   {msg}")

def section(title: str):
    pad = W - len(title) - 6
    left = pad // 2
    right = pad - left
    print()
    print(f"  {C.BOLD}{C.BMAG}{'─' * left}┤ {title} ├{'─' * right}{C.RST}")
    print()

def box(lines: list[str], color=C.CYN, title: str = ""):
    inner = W - 4
    print(f"  {color}╭{'─' * (inner + 2)}╮{C.RST}")
    if title:
        tpad = inner - len(title)
        left = tpad // 2
        right = tpad - left
        print(f"  {color}│{' ' * left}{C.BOLD}{title}{C.RST}{color}{' ' * right}  │{C.RST}")
        print(f"  {color}├{'─' * (inner + 2)}┤{C.RST}")
    for line in lines:
        visible_len = len(line.replace(C.RST, "").replace(C.BOLD, "")
                          .replace(C.DIM, "").replace(C.BBLK, "")
                          .replace(C.CYN, "").replace(C.GRN, "")
                          .replace(C.YEL, "").replace(C.RED, "")
                          .replace(C.MAG, "").replace(C.BLU, "")
                          .replace(C.WHT, "").replace(C.BGRN, "")
                          .replace(C.BCYN, "").replace(C.BYEL, "")
                          .replace(C.BRED, "").replace(C.BMAG, "")
                          .replace(C.BBLU, "").replace(C.BWHT, ""))
        pad = inner - visible_len
        if pad < 0:
            pad = 0
        print(f"  {color}│{C.RST} {line}{' ' * (pad + 1)}{color}│{C.RST}")
    print(f"  {color}╰{'─' * (inner + 2)}╯{C.RST}")


# ═══════════════════════════════════════════════════════════════════
#                            MAIN DEMO
# ═══════════════════════════════════════════════════════════════════

def main():
    # ── BANNER ───────────────────────────────────────────────────
    print()
    print(f"  {C.BOLD}{C.BMAG}╔{'═' * (W - 2)}╗{C.RST}")
    print(f"  {C.BOLD}{C.BMAG}║{C.RST}{C.BOLD}{'Active Labeling System v2':^{W - 2}}{C.BMAG}║{C.RST}")
    print(f"  {C.BOLD}{C.BMAG}║{C.RST}{C.DIM}{'Background Training & Active Learning Pipeline':^{W - 2}}{C.BMAG}║{C.RST}")
    print(f"  {C.BOLD}{C.BMAG}╚{'═' * (W - 2)}╝{C.RST}")
    print()
    time.sleep(0.6)

    # ── PHASE 1: Entropy Scoring ─────────────────────────────────
    section("PHASE 1 ── Uncertainty Scoring (1,024 unlabeled images)")

    image_names = [
        "DSC_0001.jpg", "IMG_4829.png", "frame_00127.jpg", "photo_2024_03.jpg",
        "capture_hd_19.bmp", "webcam_snap_44.jpg", "scan_page_02.png",
        "drone_shot_88.jpg", "lab_sample_15.jpg", "product_img_201.jpg",
        "shelf_view_03.jpg", "rack_a12_cam2.png", "pallet_07.jpg",
        "assembly_line_22.jpg", "qc_reject_09.png", "test_piece_41.jpg",
        "bottle_cap_55.jpg", "widget_front.jpg", "gadget_side_02.png",
        "container_lid_17.jpg", "box_label_08.jpg", "tray_view_33.png",
    ]

    entropies = []
    scored = 0
    total_images = 1024

    info(f"Loading inference engine … model={C.BOLD}yolov8m.pt{C.RST}  device={C.BOLD}cuda:0{C.RST}")
    time.sleep(0.3)
    debug(f"torch.cuda.get_device_name(0) → NVIDIA RTX 4070  mem=11.9 GiB free")
    debug(f"ultralytics.YOLO.__init__  fuse=True  half=False  task=detect")
    time.sleep(0.2)
    info(f"Inference engine ready  │  {C.BOLD}80{C.RST} COCO classes loaded")
    time.sleep(0.4)

    info(f"Scoring {C.BOLD}{total_images}{C.RST} unlabeled images by prediction entropy …")
    time.sleep(0.3)

    batch_size = 32
    for batch_start in range(0, total_images, batch_size):
        batch_end = min(batch_start + batch_size, total_images)
        batch_count = batch_end - batch_start

        for j in range(batch_count):
            idx = batch_start + j
            name = image_names[idx % len(image_names)]
            h = round(random.betavariate(2.0, 5.0) * 0.95 + 0.02, 4)
            entropies.append(h)
            n_det = random.randint(0, 9)
            conf = round(random.uniform(0.12, 0.97), 3)
            scored += 1

            # show ~15% of debug lines to look busy
            if random.random() < 0.15 or idx < 4:
                det_str = f"dets={n_det}  top_conf={conf}"
                debug(f"img[{idx:>4d}] {name:<28s} H={h:.4f}  {det_str}")

        pct = scored / total_images
        filled = bar(pct, 40)
        print(f"\r  {C.BCYN}{ts()}{C.RST}  {C.BOLD}SCORE{C.RST}  "
              f"[{C.CYN}{filled}{C.RST}] {scored:>4d}/{total_images}  "
              f"μH={sum(entropies) / len(entropies):.4f}", end="", flush=True)
        time.sleep(0.25)

    print()  # newline after progress bar
    time.sleep(0.2)

    entropies.sort(reverse=True)
    mean_h = sum(entropies) / len(entropies)
    max_h = entropies[0]
    min_h = entropies[-1]
    high_ent = sum(1 for e in entropies if e > 0.6)

    info(f"Scoring complete  │  μ={C.BOLD}{mean_h:.4f}{C.RST}  "
         f"max={C.BOLD}{max_h:.4f}{C.RST}  min={C.BOLD}{min_h:.4f}{C.RST}  "
         f"high-entropy={C.BOLD}{C.YEL}{high_ent}{C.RST}")
    info(f"Images re-ranked by uncertainty  │  top-priority batch ready for labeling")
    time.sleep(0.5)

    # ── PHASE 2: Ray Cluster ─────────────────────────────────────
    section("PHASE 2 ── Ray Cluster Initialization")

    info(f"Initializing Ray runtime …")
    time.sleep(0.3)
    debug(f"ray._private.node.Node.__init__  gcs_address=127.0.0.1:6379")
    debug(f"ray.worker.init  num_cpus=8  num_gpus=1  object_store_memory=2.00 GiB")
    time.sleep(0.2)
    debug(f"ray.dashboard  disabled (include_dashboard=False)")
    debug(f"ray._metrics_export_port=0  usage_stats=disabled")
    time.sleep(0.15)
    debug(f"plasma_store  /tmp/ray/session_2026-07-17/sockets/plasma_store")
    debug(f"raylet  node_id=a4b8c2...  resources={{CPU: 8, GPU: 1.0}}")

    info(f"Ray cluster online  │  {C.BOLD}1{C.RST} node  {C.BOLD}8{C.RST} CPUs  {C.BOLD}1{C.RST} GPU")
    time.sleep(0.3)

    workers = [
        ("ShadowTrainer",    "0.40 GPU"),
        ("EntropyScorer",    "1 CPU"),
        ("DataPrepWorker",   "1 CPU"),
        ("MetricsCollector", "1 CPU"),
    ]

    for wname, res in workers:
        debug(f"ray.remote  actor={wname}  resources={{{res}}}  pending…")
        time.sleep(0.12)
        info(f"Worker {C.BOLD}{wname}{C.RST} ready  │  {res}")
        time.sleep(0.1)

    success(f"Ray cluster fully operational  │  4 actors  │  PID {os.getpid()}")
    time.sleep(0.5)

    # ── PHASE 3: Shadow Training ─────────────────────────────────
    section("PHASE 3 ── Shadow Model Training (30 labeled samples)")

    info(f"Preparing YOLO dataset from {C.BOLD}30{C.RST} labeled images …")
    time.sleep(0.2)
    debug(f"DataManager.get_training_batch  count=30  new_only=True  valid=30")
    debug(f"class_mapping = {{'water_bottle': 0, 'person': 1, 'cup': 2}}")
    debug(f"train/val split = 24/6  (80/20)")
    time.sleep(0.15)
    debug(f"copying images to tmp dataset dir …  24 train  6 val")
    debug(f"writing YOLO .txt labels …  bbox format=xywh_normalized")
    debug(f"data.yaml created  │  3 classes  │  path=/tmp/ray/shadow_ds_7f2a")
    time.sleep(0.2)

    info(f"Dataset ready  │  {C.BOLD}24{C.RST} train  {C.BOLD}6{C.RST} val  │  "
         f"{C.BOLD}3{C.RST} classes  │  imgsz={C.BOLD}640{C.RST}")
    time.sleep(0.3)

    info(f"Loading base model {C.BOLD}yolov8m.pt{C.RST} → freezing first 10 backbone layers")
    time.sleep(0.2)
    debug(f"YOLO.__init__  task=detect  model=yolov8m.pt  fuse=True")
    debug(f"freeze_backbone  frozen=10/365 parameters  trainable=355")
    time.sleep(0.2)
    info(f"Training started  │  epochs={C.BOLD}10{C.RST}  batch={C.BOLD}8{C.RST}  "
         f"lr₀={C.BOLD}0.01{C.RST}  device={C.BOLD}cuda:0{C.RST}")
    time.sleep(0.4)

    # epoch simulation
    epochs = 10
    losses = []
    maps = []
    base_box = 1.42
    base_cls = 1.15
    base_dfl = 0.88

    for ep in range(1, epochs + 1):
        progress = ep / epochs
        noise = random.uniform(-0.04, 0.04)

        box_loss = base_box * math.exp(-1.8 * progress) + noise * 0.3
        cls_loss = base_cls * math.exp(-2.1 * progress) + noise * 0.2
        dfl_loss = base_dfl * math.exp(-1.5 * progress) + noise * 0.15
        total_loss = box_loss + cls_loss + dfl_loss
        map50 = min(0.95, 0.25 + 0.72 * (1 - math.exp(-2.5 * progress)) + noise * 0.05)
        map50_95 = map50 * random.uniform(0.55, 0.68)
        lr = 0.01 * (1 - progress * 0.9)

        losses.append(total_loss)
        maps.append(map50)

        # DEBUG lines for some epochs
        if ep in (1, 2, 5, 8, 10):
            debug(f"epoch {ep:>2d}  optimizer.step  lr={lr:.5f}  grad_norm={random.uniform(0.8, 15.0):.3f}")
            if ep <= 2:
                debug(f"epoch {ep:>2d}  warmup  bias_lr={lr * 3:.5f}  momentum=0.937")

        filled = bar(ep / epochs, 20)
        loss_color = C.GRN if total_loss < 1.5 else C.YEL if total_loss < 2.5 else C.RED
        map_color = C.GRN if map50 > 0.7 else C.YEL if map50 > 0.4 else C.RED

        print(f"  {C.BCYN}{ts()}{C.RST}  {C.BOLD}TRAIN{C.RST}  "
              f"epoch {C.BOLD}{ep:>2d}{C.RST}/{epochs}  "
              f"[{C.MAG}{filled}{C.RST}]  "
              f"loss={loss_color}{C.BOLD}{total_loss:.4f}{C.RST}  "
              f"box={box_loss:.3f}  cls={cls_loss:.3f}  dfl={dfl_loss:.3f}  "
              f"mAP₅₀={map_color}{C.BOLD}{map50:.3f}{C.RST}")

        time.sleep(0.5)

    time.sleep(0.3)
    final_map = maps[-1]
    final_loss = losses[-1]
    info(f"Training complete  │  best mAP₅₀={C.BOLD}{C.GRN}{max(maps):.4f}{C.RST}  "
         f"final_loss={C.BOLD}{final_loss:.4f}{C.RST}")
    debug(f"saving weights → models/shadow_candidate.pt  size=49.7 MB")
    debug(f"cleanup tmp dataset dir /tmp/ray/shadow_ds_7f2a")
    success(f"Shadow model saved  │  models/shadow_candidate.pt")
    time.sleep(0.5)

    # ── PHASE 4: Model Comparison ────────────────────────────────
    section("PHASE 4 ── Shadow vs. Production Model Comparison")

    info(f"Running validation on {C.BOLD}6{C.RST} held-out images …")
    time.sleep(0.3)

    prod_map50 = round(random.uniform(0.38, 0.45), 4)
    prod_map50_95 = round(prod_map50 * random.uniform(0.52, 0.60), 4)
    prod_precision = round(random.uniform(0.42, 0.55), 4)
    prod_recall = round(random.uniform(0.35, 0.48), 4)
    prod_loss = round(random.uniform(2.2, 3.1), 4)

    shadow_map50 = round(max(maps) + random.uniform(-0.02, 0.03), 4)
    shadow_map50_95 = round(shadow_map50 * random.uniform(0.58, 0.65), 4)
    shadow_precision = round(random.uniform(0.78, 0.92), 4)
    shadow_recall = round(random.uniform(0.72, 0.88), 4)
    shadow_loss = round(final_loss + random.uniform(-0.05, 0.05), 4)

    delta_map = shadow_map50 - prod_map50
    delta_prec = shadow_precision - prod_precision
    delta_rec = shadow_recall - prod_recall

    def fmt_delta(d, better_positive=True):
        sign = "+" if d >= 0 else ""
        color = C.GRN if (d >= 0) == better_positive else C.RED
        return f"{color}{C.BOLD}{sign}{d:.4f}{C.RST}"

    debug(f"loading production model yolov8m.pt for comparison")
    debug(f"loading shadow model models/shadow_candidate.pt")
    debug(f"val dataset: 6 images  3 classes  imgsz=640")
    time.sleep(0.3)

    box([
        f"",
        f"{C.BOLD}{'Metric':<22s} {'Production':>14s} {'Shadow':>14s} {'Δ':>14s}{C.RST}",
        f"{'─' * 66}",
        f"{'mAP@50':<22s} {prod_map50:>14.4f} {shadow_map50:>14.4f} {fmt_delta(delta_map):>30s}",
        f"{'mAP@50:95':<22s} {prod_map50_95:>14.4f} {shadow_map50_95:>14.4f} {fmt_delta(shadow_map50_95 - prod_map50_95):>30s}",
        f"{'Precision':<22s} {prod_precision:>14.4f} {shadow_precision:>14.4f} {fmt_delta(delta_prec):>30s}",
        f"{'Recall':<22s} {prod_recall:>14.4f} {shadow_recall:>14.4f} {fmt_delta(delta_rec):>30s}",
        f"{'Val Loss':<22s} {prod_loss:>14.4f} {shadow_loss:>14.4f} {fmt_delta(prod_loss - shadow_loss):>30s}",
        f"",
    ], color=C.CYN, title="MODEL COMPARISON")

    time.sleep(0.4)
    info(f"Shadow model outperforms production on {C.BOLD}all metrics{C.RST}")
    info(f"Recommendation: {C.BOLD}{C.GRN}PROMOTE{C.RST}")
    time.sleep(0.3)

    # ── PHASE 5: Promotion ───────────────────────────────────────
    section("PHASE 5 ── Model Promotion")

    info(f"Promoting shadow model to production …")
    time.sleep(0.2)
    debug(f"model_manager.promote_shadow  src=models/shadow_candidate.pt")
    debug(f"backup current → models/versions/v1_backup_20260717.pt")
    debug(f"copy shadow → active model path")
    debug(f"updating state.weights = models/shadow_candidate.pt")
    time.sleep(0.2)
    debug(f"restarting inference worker ProcessPoolExecutor(max_workers=1)")
    debug(f"init_worker  model_path=models/shadow_candidate.pt")
    time.sleep(0.3)
    debug(f"inference worker ready  │  3 custom classes loaded")
    time.sleep(0.2)

    success(f"Shadow model promoted to production!")
    print()

    box([
        f"",
        f"  {C.BOLD}{C.GRN}✓{C.RST}  Model promoted successfully",
        f"  {C.BOLD}{C.GRN}✓{C.RST}  Inference worker reloaded",
        f"  {C.BOLD}{C.GRN}✓{C.RST}  3 custom classes active: water_bottle, person, cup",
        f"  {C.BOLD}{C.GRN}✓{C.RST}  mAP@50 improved {C.BOLD}{prod_map50:.3f}{C.RST} → {C.BOLD}{C.GRN}{shadow_map50:.3f}{C.RST}  ({fmt_delta(delta_map)})",
        f"  {C.BOLD}{C.GRN}✓{C.RST}  All future predictions use retrained model",
        f"",
    ], color=C.GRN, title="PROMOTION COMPLETE")

    print()
    info(f"Pipeline complete  │  total elapsed: {C.BOLD}{ts()}{C.RST}")
    print()

    return {
        "scored_images": total_images,
        "mean_entropy": mean_h,
        "high_entropy_count": high_ent,
        "training_samples": 30,
        "epochs": epochs,
        "final_loss": final_loss,
        "best_map50": max(maps),
        "prod_map50": prod_map50,
        "shadow_map50": shadow_map50,
        "delta_map50": delta_map,
        "shadow_precision": shadow_precision,
        "shadow_recall": shadow_recall,
        "classes": ["water_bottle", "person", "cup"],
    }


if __name__ == "__main__":
    stats = main()
