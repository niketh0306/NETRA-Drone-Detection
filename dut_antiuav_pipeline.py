# ============================================================
#  DUT Anti-UAV — RT-DETR + CBAM Pipeline
#  Kaggle-ready notebook (all compatibility issues patched)
# ============================================================


# ── CELL 1 ── Install & Imports ─────────────────────────────
# No changes needed — ultralytics is available on Kaggle.
# Make sure "Internet" is ON in Notebook Settings (needed for
# model weight download in Cell 6).

!pip install -U ultralytics

import os
import glob
import zipfile
import random
import shutil
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from PIL import Image
import cv2
import matplotlib.pyplot as plt

from ultralytics import RTDETR
from ultralytics.utils.metrics import DetMetrics

print("✅ All imports successful")
print(f"   PyTorch  : {torch.__version__}")
print(f"   CUDA GPU : {torch.cuda.is_available()} — "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")


# ── CELL 2 ── Locate Dataset & Extract Zips ─────────────────
# FIX 1: Auto-detect the correct Kaggle input slug instead of
#         hard-coding it, so it works regardless of the exact
#         dataset name the user added via "Add Data".
# FIX 2: Handle BOTH cases:
#         (a) zips exist  → extract them
#         (b) already extracted (many Kaggle datasets) → use as-is

YOLO_BASE = "/kaggle/working/yolo_data"
RAW_BASE  = "/kaggle/working/dut_raw"

# ── auto-detect input folder ──────────────────────────────────
def find_dataset_root():
    """Walk /kaggle/input and find the folder that contains
    train.zip / a train/ sub-folder with images."""
    kaggle_input = "/kaggle/input"
    for dataset in os.listdir(kaggle_input):
        d = os.path.join(kaggle_input, dataset)
        if not os.path.isdir(d):
            continue
        contents = os.listdir(d)
        # has zip files?
        if any(f in contents for f in ["train.zip", "val.zip"]):
            print(f"✅ Dataset found (zip): {d}")
            return d, "zip"
        # already extracted?
        if any(f in contents for f in ["train", "val", "test"]):
            print(f"✅ Dataset found (extracted): {d}")
            return d, "extracted"
    raise FileNotFoundError(
        "Could not find DUT Anti-UAV dataset under /kaggle/input.\n"
        "Please click 'Add Data' in the Kaggle notebook and add the dataset."
    )

DATASET_ROOT, DATASET_FORMAT = find_dataset_root()
SPLITS = ["train", "val", "test"]

# ── extract or point to raw folders ──────────────────────────
for split in SPLITS:
    dest = os.path.join(RAW_BASE, split)
    os.makedirs(dest, exist_ok=True)

    if DATASET_FORMAT == "zip":
        zip_path = os.path.join(DATASET_ROOT, f"{split}.zip")
        if os.path.exists(zip_path):
            print(f"Extracting {split}.zip …")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(dest)
            print(f"  ✅ Done → {dest}")
        else:
            print(f"  ⚠️  {split}.zip not found — skipping")
    else:
        # Already extracted: symlink or copy pointer
        src = os.path.join(DATASET_ROOT, split)
        if os.path.isdir(src):
            # Use a symlink so we don't duplicate ~10 GB
            if not os.path.exists(dest):
                os.symlink(src, dest)
            else:
                shutil.rmtree(dest)
                os.symlink(src, dest)
            print(f"  ✅ Linked {split}/ → {dest}")

# ── Inspect structure ─────────────────────────────────────────
print("\n── Extracted directory structure (first 3 levels) ──")
for split in SPLITS:
    base = os.path.join(RAW_BASE, split)
    if not os.path.isdir(base):
        print(f"  [{split}] — NOT FOUND")
        continue
    for root, dirs, files in os.walk(base):
        depth = root.replace(base, "").count(os.sep)
        if depth > 2:
            continue
        indent = "  " * depth
        print(f"{indent}{os.path.basename(root)}/")
        if depth == 2:
            exts = set(os.path.splitext(f)[1].lower() for f in files)
            print(f"{indent}  [{len(files)} files, types: {exts}]")


# ── CELL 3 ── Auto-detect Format & Convert to YOLO ──────────
# FIX 3: Extended image extension support (.bmp, .tif, .tiff)
#         to cover all formats used in anti-UAV datasets.
# FIX 4: .txt filter in label scan to skip hidden system files.

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

for p in [f"images/{s}" for s in SPLITS] + [f"labels/{s}" for s in SPLITS]:
    os.makedirs(os.path.join(YOLO_BASE, p), exist_ok=True)


def coco_bbox_to_yolo(bbox, img_w, img_h):
    """COCO [x_min, y_min, w, h] → YOLO normalised [xc, yc, w, h]"""
    x_min, y_min, bw, bh = bbox
    xc = (x_min + bw / 2) / img_w
    yc = (y_min + bh / 2) / img_h
    # clamp — critical for occlusion-clipped boxes touching frame edges
    return (
        min(max(xc,        0.0), 1.0),
        min(max(yc,        0.0), 1.0),
        min(max(bw/img_w,  0.0), 1.0),
        min(max(bh/img_h,  0.0), 1.0),
    )


def find_images_dir(split_root):
    """Robustly locate the images sub-folder."""
    for candidate in ["images", "JPEGImages", "imgs", "image", ""]:
        p = os.path.join(split_root, candidate) if candidate else split_root
        if os.path.isdir(p) and any(
            f.lower().endswith(IMAGE_EXTS) for f in os.listdir(p)
        ):
            return p
    return split_root   # fallback: images are directly in split root


def find_labels_dir(split_root):
    """Robustly locate the labels sub-folder (YOLO .txt)."""
    for candidate in ["labels", "label", "annotations", "Annotations"]:
        p = os.path.join(split_root, candidate)
        if os.path.isdir(p) and any(
            f.endswith(".txt") for f in os.listdir(p)
        ):
            return p
    return None


def find_coco_json(split_root, split):
    """Locate a COCO-style JSON annotation file."""
    candidates = [
        f"annotations.json",
        f"{split}.json",
        os.path.join("annotations", f"{split}.json"),
        os.path.join("annotations", "instances_default.json"),
    ]
    for c in candidates:
        p = os.path.join(split_root, c)
        if os.path.exists(p):
            return p
    return None


def process_split(split):
    raw_split = os.path.join(RAW_BASE, split)
    if not os.path.isdir(raw_split):
        print(f"[{split}] ⚠️  Directory not found — skipping")
        return

    img_src  = find_images_dir(raw_split)
    ann_dir  = find_labels_dir(raw_split)
    ann_json = find_coco_json(raw_split, split)

    img_dst = os.path.join(YOLO_BASE, "images", split)
    lbl_dst = os.path.join(YOLO_BASE, "labels", split)

    # ── CASE A: YOLO .txt labels ──────────────────────────────
    if ann_dir:
        print(f"[{split}] Format: YOLO .txt → copying")
        copied = 0
        for img_file in tqdm(os.listdir(img_src), desc=split):
            if not img_file.lower().endswith(IMAGE_EXTS):
                continue
            stem = os.path.splitext(img_file)[0]
            lbl_src_path = os.path.join(ann_dir, stem + ".txt")
            if not os.path.exists(lbl_src_path):
                continue          # skip unannotated images
            shutil.copy(os.path.join(img_src, img_file),
                        os.path.join(img_dst, img_file))
            shutil.copy(lbl_src_path,
                        os.path.join(lbl_dst, stem + ".txt"))
            copied += 1
        print(f"  ✅ {copied} pairs copied")

    # ── CASE B: COCO JSON ─────────────────────────────────────
    elif ann_json:
        print(f"[{split}] Format: COCO JSON → converting ({ann_json})")
        with open(ann_json) as f:
            coco = json.load(f)

        id2info = {img["id"]: img for img in coco["images"]}
        ann_by_img = defaultdict(list)
        for ann in coco["annotations"]:
            if ann.get("iscrowd", 0):
                continue
            ann_by_img[ann["image_id"]].append(ann)

        converted = 0
        for img_id, anns in tqdm(ann_by_img.items(), desc=split):
            info     = id2info[img_id]
            img_file = os.path.basename(info["file_name"])
            img_w, img_h = info["width"], info["height"]
            stem     = os.path.splitext(img_file)[0]
            src_path = os.path.join(img_src, img_file)
            if not os.path.exists(src_path):
                continue
            shutil.copy(src_path, os.path.join(img_dst, img_file))
            with open(os.path.join(lbl_dst, stem + ".txt"), "w") as lf:
                for ann in anns:
                    xc, yc, w, h = coco_bbox_to_yolo(
                        ann["bbox"], img_w, img_h)
                    lf.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
            converted += 1
        print(f"  ✅ {converted} images converted")

    else:
        print(f"[{split}] ❌ Could not detect annotation format.")
        print(f"         Inspect {raw_split} manually.")


for split in SPLITS:
    process_split(split)

# Sanity: confirm image counts
print("\nImage counts after conversion:")
for split in SPLITS:
    d = os.path.join(YOLO_BASE, "images", split)
    n = len([f for f in os.listdir(d) if f.lower().endswith(IMAGE_EXTS)])
    print(f"  {split}: {n} images")


# ── CELL 4 ── Dataset YAML ───────────────────────────────────
# YOLO_BASE is already an absolute path — no issue here.

data_yaml = {
    "path":  YOLO_BASE,
    "train": "images/train",
    "val":   "images/val",
    "test":  "images/test",
    "nc":    1,
    "names": ["drone"]
}

YAML_PATH = "/kaggle/working/dut_antiuav.yaml"
with open(YAML_PATH, "w") as f:
    yaml.dump(data_yaml, f)

print("✅ dut_antiuav.yaml written")
print(yaml.dump(data_yaml))


# ── CELL 5 ── CBAM Definition ────────────────────────────────
# Completely unchanged from VisioDECT pipeline.

class CBAM(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // r),
            nn.ReLU(),
            nn.Linear(channels // r, channels)
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        # Channel attention
        avg = torch.mean(x, dim=(2, 3))
        mx  = torch.amax(x, dim=(2, 3))
        ca  = torch.sigmoid(
            self.mlp(avg) + self.mlp(mx)
        ).unsqueeze(-1).unsqueeze(-1)
        x = x * ca

        # Spatial attention
        avg_map    = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x,  dim=1, keepdim=True)
        sa = torch.sigmoid(
            self.spatial(torch.cat([avg_map, max_map], dim=1))
        )
        return x * sa


class ConvWithCBAM(nn.Module):
    def __init__(self, conv_block):
        super().__init__()
        self.conv = conv_block
        self.cbam = CBAM(conv_block.conv.out_channels)

    def forward(self, x):
        return self.cbam(self.conv(x))

print("✅ CBAM classes defined")


# ── CELL 6 ── Model Init + CBAM Injection ───────────────────
# FIX 5: "Internet" must be ON in Kaggle Notebook Settings
#         for rtdetr-l.pt to download (~130 MB, one-time only).
#         Once downloaded it is cached in /root/.config/Ultralytics/

print("⚠️  Ensure 'Internet' is enabled in Notebook Settings")
print("   (right panel → Settings → Internet ON)")
print("   Required to download rtdetr-l.pt weights (~130 MB)\n")

model = RTDETR("rtdetr-l.pt")

# Inject CBAM into the last FPN layer only — unchanged logic
for layer in reversed(model.model.model):
    if hasattr(layer, "cv2"):
        layer.cv2 = ConvWithCBAM(layer.cv2)
        print("✅ CBAM injected into last FPN layer")
        break


# ── CELL 7 ── Training ───────────────────────────────────────
# FIX 6: epochs=50 may hit the 9h Kaggle timeout on T4.
#         Set to 30 for a safe first run; increase if you have P100.
#         copy_paste + scale are key for occlusion robustness.

# Detect GPU type and adjust epochs
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
EPOCHS = 50 if "P100" in gpu_name or "V100" in gpu_name else 30
print(f"GPU: {gpu_name}  →  Using epochs={EPOCHS}")

model.train(
    data=YAML_PATH,
    epochs=EPOCHS,
    imgsz=640,
    patience=10,
    batch=8,
    device=0,
    project="runs/rtdetr",
    name="dut_antiuav_attention",

    # Occlusion-aware augmentation
    mosaic=1.0,        # always on — handles scale variation
    copy_paste=0.3,    # paste-augment drones to simulate occlusion
    degrees=10.0,      # mild rotation
    scale=0.6,         # scale jitter for small/distant drones
    fliplr=0.5,
)


# ── CELL 8 ── Quick Validation After Training ────────────────

metrics = model.val(data=YAML_PATH, imgsz=640)

print(f"mAP@50:      {metrics.box.map50:.4f}")
print(f"mAP@50-95:   {metrics.box.map:.4f}")
print(f"Precision:   {metrics.box.mp:.4f}")
print(f"Recall:      {metrics.box.mr:.4f}")


# ── CELL 9 ── Load Best Weights ──────────────────────────────
# FIX 7: Guard against empty glob (if training was interrupted).
# FIX 8: Use glob instead of hard-coded path — Ultralytics may
#         suffix the run folder (e.g. dut_antiuav_attention2).

candidates = glob.glob(
    "/kaggle/working/runs/**/best.pt", recursive=True
)

if not candidates:
    raise FileNotFoundError(
        "best.pt not found. Training may have been interrupted.\n"
        "Check /kaggle/working/runs/ for partial checkpoints (last.pt)."
    )

MODEL_PATH = candidates[0]
model = RTDETR(MODEL_PATH)
print(f"✅ Loaded: {MODEL_PATH}")


# ── CELL 10 ── Full Metrics + Per-class Report ───────────────
# FIX 9: metrics.box.f1 may be scalar (not array) in some
#         Ultralytics versions — wrapped in try/except.

metrics = model.val(
    data=YAML_PATH,
    imgsz=640,
    device=0
)

print(f"mAP@50:         {metrics.box.map50:.4f}")
print(f"mAP@50-95:      {metrics.box.map:.4f}")
print(f"Mean Precision: {metrics.box.mp:.4f}")
print(f"Mean Recall:    {metrics.box.mr:.4f}")

try:
    f1 = metrics.box.f1
    mean_f1 = f1.mean() if hasattr(f1, "mean") else float(f1)
    print(f"Mean F1-score:  {mean_f1:.4f}")
except Exception:
    # Fallback: compute from P and R
    p, r = metrics.box.mp, metrics.box.mr
    mean_f1 = 2 * p * r / (p + r + 1e-8)
    print(f"Mean F1-score:  {mean_f1:.4f}  (computed from P/R)")

for i, name in metrics.names.items():
    try:
        f1_val = float(metrics.box.f1[i]) if hasattr(metrics.box.f1, "__len__") \
                 else float(metrics.box.f1)
    except Exception:
        pi, ri = metrics.box.p[i], metrics.box.r[i]
        f1_val = 2 * pi * ri / (pi + ri + 1e-8)

    print(f"\nClass : {name}")
    print(f"  Precision : {metrics.box.p[i]:.4f}")
    print(f"  Recall    : {metrics.box.r[i]:.4f}")
    print(f"  F1-score  : {f1_val:.4f}")
    print(f"  AP@50     : {metrics.box.ap50[i]:.4f}")
    print(f"  AP@50-95  : {metrics.box.ap[i]:.4f}")


# ── CELL 11 ── GT vs Prediction Visualisation ────────────────
# Unchanged — paths use YOLO_BASE dynamically.

def visualize_gt_vs_pred(img_path, label_path, conf=0.25):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read: {img_path}")
        return
    h, w, _ = img.shape

    # Ground truth (GREEN)
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                _, xc, yc, bw, bh = map(float, parts)
                x1 = int((xc - bw/2) * w)
                y1 = int((yc - bh/2) * h)
                x2 = int((xc + bw/2) * w)
                y2 = int((yc + bh/2) * h)
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)

    # Predictions (RED)
    res = model(img_path, conf=conf)[0]
    for b in res.boxes.xyxy.cpu().numpy():
        x1,y1,x2,y2 = map(int, b)
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)

    plt.figure(figsize=(8,6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("GT (Green) vs Prediction (Red)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


val_img_dir = os.path.join(YOLO_BASE, "images/val")
val_lbl_dir = os.path.join(YOLO_BASE, "labels/val")

val_images = [f for f in os.listdir(val_img_dir)
              if f.lower().endswith(IMAGE_EXTS)]
sample = random.choice(val_images)

visualize_gt_vs_pred(
    os.path.join(val_img_dir, sample),
    os.path.join(val_lbl_dir, os.path.splitext(sample)[0] + ".txt")
)


# ── CELL 12 ── PR / F1 Curves ────────────────────────────────
# FIX 10: Dynamically resolve the run directory with glob
#          instead of hard-coding, in case Ultralytics appends
#          a suffix like dut_antiuav_attention2.

run_dirs = glob.glob(
    "/kaggle/working/runs/rtdetr/dut_antiuav_attention*",
    recursive=False
)

if not run_dirs:
    print("⚠️  Run directory not found — run training first")
else:
    BASE = sorted(run_dirs)[-1]   # pick the most recent if multiple
    print(f"Using run dir: {BASE}")

    curve_files = {
        "PR Curve":        "BoxPR_curve.png",
        "F1 Curve":        "BoxF1_curve.png",
        "Precision Curve": "BoxP_curve.png",
        "Recall Curve":    "BoxR_curve.png",
    }

    for title, fname in curve_files.items():
        full = os.path.join(BASE, fname)
        if os.path.exists(full):
            img = Image.open(full)
            plt.figure(figsize=(6,6))
            plt.imshow(img)
            plt.title(title)
            plt.axis("off")
            plt.tight_layout()
            plt.show()
        else:
            print(f"⚠️  {fname} not found yet — run model.val() first")


# ── CELL 13 ── Attention Heatmap ────────────────────────────
# Unchanged.

def attention_heatmap(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read: {img_path}")
        return
    res = model(img_path, conf=0.25)[0]

    heatmap = np.zeros(img.shape[:2], dtype=np.float32)
    for box, conf in zip(
        res.boxes.xyxy.cpu().numpy(),
        res.boxes.conf.cpu().numpy()
    ):
        x1,y1,x2,y2 = map(int, box)
        heatmap[y1:y2, x1:x2] += float(conf)

    heatmap = cv2.GaussianBlur(heatmap, (31,31), 0)
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)

    plt.figure(figsize=(8,6))
    plt.imshow(heatmap, cmap="jet")
    plt.colorbar(label="Detection confidence")
    plt.title("Attention / Activation Heatmap")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

attention_heatmap(os.path.join(val_img_dir, sample))


# ── CELL 14 ── Box Size Distribution ────────────────────────
# FIX 11: Added .txt filter to skip non-label files
#          (.DS_Store, .gitkeep, etc.) — prevents ValueError on split.

def plot_box_size_distribution(label_dir):
    sizes = []
    for f in os.listdir(label_dir):
        if not f.endswith(".txt"):          # ← guard added
            continue
        with open(os.path.join(label_dir, f)) as fp:
            for line in fp:
                parts = line.strip().split()
                if len(parts) == 5:
                    _, _, _, w, h = map(float, parts)
                    sizes.append(w * h)

    if not sizes:
        print("No labels found — skipping distribution plot")
        return

    sizes = np.array(sizes)
    plt.figure(figsize=(9,5))
    plt.hist(sizes, bins=60, color="steelblue", edgecolor="black", alpha=0.85)
    plt.axvline(x=0.005, color="red", linestyle="--",
                label="Small object threshold (0.5%)")
    plt.xlabel("Normalised Bounding Box Area  (w × h)")
    plt.ylabel("Count")
    plt.title("Drone Size Distribution — DUT Anti-UAV  (Occlusion / Scale Variance)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    pct_small = 100 * np.mean(sizes < 0.005)
    print(f"Total boxes   : {len(sizes)}")
    print(f"Median area   : {np.median(sizes):.5f}")
    print(f"Small objects : {pct_small:.1f}%  (area < 0.005)")

plot_box_size_distribution(val_lbl_dir)


# ── CELL 15 ── False Positive Analysis ──────────────────────
# Unchanged.

def show_false_positives(img_path, conf=0.5):
    res = model(img_path, conf=conf)[0]

    if len(res.boxes) == 0:
        print(f"No predictions → background suppressed ({os.path.basename(img_path)})")
        return

    img = cv2.imread(img_path)
    for b in res.boxes.xyxy.cpu().numpy():
        x1,y1,x2,y2 = map(int, b)
        cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)

    plt.figure(figsize=(8,6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("False Positive Analysis")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

sample_imgs = random.sample(val_images, min(3, len(val_images)))
for img_name in sample_imgs:
    show_false_positives(os.path.join(val_img_dir, img_name))


# ── CELL 16 ── [BONUS] Test-set Evaluation ──────────────────
# DUT Anti-UAV has a dedicated test set — evaluate it separately.

test_metrics = model.val(
    data=YAML_PATH,
    split="test",
    imgsz=640,
    device=0
)

print("── TEST SET RESULTS ──────────────────────────")
print(f"mAP@50:         {test_metrics.box.map50:.4f}")
print(f"mAP@50-95:      {test_metrics.box.map:.4f}")
print(f"Mean Precision: {test_metrics.box.mp:.4f}")
print(f"Mean Recall:    {test_metrics.box.mr:.4f}")
