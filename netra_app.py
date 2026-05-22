"""
NETRA v1.3 — Drone Detection System
Neural Enhanced Threat Recognition & Analysis
RT-DETR-L + Multi-Scale CBAM + Agent (ReAct) + Dual-Model Inference
GITAM University, Hyderabad | Final Year Project | 2026
"""

import os
import csv
import cv2
import json
import time
import torch
import tempfile
import numpy as np
import gradio as gr
import ollama
from PIL import Image
from datetime import datetime
from ultralytics import RTDETR
import torch.nn as nn

# ─────────────────────────────────────────────
# CBAM
# ─────────────────────────────────────────────

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
        avg = torch.mean(x, dim=(2, 3))
        mx  = torch.amax(x, dim=(2, 3))
        ca  = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).unsqueeze(-1).unsqueeze(-1)
        x   = x * ca
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        sa  = torch.sigmoid(self.spatial(torch.cat([avg_map, max_map], dim=1)))
        return x * sa


# ─────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────

MODEL_PATHS = {
    "DUT Anti-UAV (Occlusion)":   os.getenv("NETRA_MODEL_DUT",       "best_dut.pt"),
    "VisioDECT (Multi-Scenario)": os.getenv("NETRA_MODEL_VISIODECT", "best_visiodect.pt"),
}
CONF_THRESHOLD = 0.25
loaded_models  = {}

def get_device():
    if torch.cuda.is_available():        return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def get_model(model_name):
    if model_name not in loaded_models:
        path = MODEL_PATHS[model_name]
        try:
            m = RTDETR(path)
            m.to(get_device())
            loaded_models[model_name] = m
        except FileNotFoundError:
            return None, f"Model file not found: {path}"
        except Exception as e:
            return None, str(e)
    return loaded_models[model_name], None


# ─────────────────────────────────────────────
# Session state  (single-detection only)
# ─────────────────────────────────────────────

session_history = []

def add_to_history(detected, confidence, threat, quadrant, model_used, source="single"):
    entry = {
        "time":       datetime.now().strftime("%H:%M:%S"),
        "detected":   detected,
        "confidence": round(confidence, 3),
        "threat":     threat,
        "quadrant":   quadrant,
        "model":      model_used,
        "source":     source,
    }
    session_history.append(entry)
    if len(session_history) > 50:
        session_history.pop(0)

def log_escalation(status, alert, threat, conf):
    log_path = os.path.join(tempfile.gettempdir(), "netra_alerts.txt")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"THREAT={threat} CONF={conf:.2f}\n"
            f"STATUS : {status}\nALERT  : {alert}\n{'─'*48}\n"
        )

def get_history_text():
    if not session_history:
        return "No detections yet."
    return "\n".join([
        f"[{e['time']}] [{e.get('source','single').upper()[:1]}] "
        f"{'DRONE' if e['detected'] else 'CLEAR'} | "
        f"Conf:{e['confidence']:.2f} | Threat:{e['threat']} | {e['model'][:12]}"
        for e in reversed(session_history)
    ])

def get_stats_text():
    if not session_history:
        return "No data yet. Run detections first."
    total    = len(session_history)
    dets     = sum(1 for e in session_history if e["detected"])
    avg_conf = (sum(e["confidence"] for e in session_history if e["detected"]) / dets
                if dets > 0 else 0.0)
    tc = {}
    for e in session_history:
        tc[e["threat"]] = tc.get(e["threat"], 0) + 1
    return (
        f"Session Statistics\n──────────────────────\n"
        f"Total scans      : {total}\n"
        f"Detections       : {dets}\n"
        f"Detection rate   : {dets/total*100:.1f}%\n"
        f"Avg confidence   : {avg_conf:.1%}\n"
        f"Dominant threat  : {max(tc, key=tc.get)}"
    )

def export_session_csv():
    if not session_history:
        return None
    path = os.path.join(tempfile.gettempdir(), f"netra_session_{datetime.now().strftime('%H%M%S')}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["time","detected","confidence","threat","quadrant","model","source"])
        w.writeheader()
        w.writerows(session_history)
    return path

def clear_session():
    session_history.clear()
    return "No detections yet.", "No data yet. Run detections first."

def refresh_dashboard():
    return get_history_text(), get_stats_text()


# ─────────────────────────────────────────────
# Threat level  (3 categories)
# ─────────────────────────────────────────────

def compute_threat_level(detected, confidence):
    if not detected:            return "LOW"
    return "CRITICAL" if confidence >= 0.60 else "MEDIUM"

def get_quadrant(cx, cy, w, h):
    return f"{'upper' if cy < h/2 else 'lower'}-{'left' if cx < w/2 else 'right'}"


# ─────────────────────────────────────────────
# Single-model inference helper
# ─────────────────────────────────────────────

def _run_single_model(image, model_name):
    model, err = get_model(model_name)
    if err:
        return 0.0, "centre", False, image.copy(), 0.0, 0, err

    img_h, img_w = image.shape[:2]
    annotated    = image.copy()
    start        = time.time()
    results      = model.predict(source=image, conf=CONF_THRESHOLD, imgsz=640, verbose=False)
    ms           = (time.time() - start) * 1000

    boxes    = results[0].boxes
    detected = boxes is not None and len(boxes) > 0
    best_conf, best_quad, num_boxes = 0.0, "centre", 0

    if detected:
        num_boxes = len(boxes)
        for box in boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cv2.rectangle(annotated, (x1,y1),(x2,y2),(0,255,80),2)
            label = f"Drone {conf:.2f}"
            (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated,(x1,y1-th-8),(x1+tw+4,y1),(0,255,80),-1)
            cv2.putText(annotated,label,(x1+2,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,0),2)
            if conf > best_conf:
                best_conf = conf
                best_quad = get_quadrant((x1+x2)//2,(y1+y2)//2,img_w,img_h)
        cv2.rectangle(annotated,(0,0),(img_w,38),(8,18,40),-1)
        cv2.putText(annotated,f"DRONE DETECTED  |  Conf:{best_conf:.2f}  |  {ms:.0f}ms",
                    (10,26),cv2.FONT_HERSHEY_SIMPLEX,0.65,(56,189,248),2)
    else:
        cv2.rectangle(annotated,(0,0),(img_w,38),(8,14,28),-1)
        cv2.putText(annotated,f"NO DRONE DETECTED  |  {ms:.0f}ms",
                    (10,26),cv2.FONT_HERSHEY_SIMPLEX,0.65,(148,163,184),2)

    return best_conf, best_quad, detected, annotated, ms, num_boxes, None


# ─────────────────────────────────────────────
# TASK 4: Improved Agent — ReAct 5-step (dual-model aware)
# ─────────────────────────────────────────────

AGENT_SYSTEM = """You are SENTINEL, the AI reasoning agent embedded in NETRA — a real-time drone
detection system built on RT-DETR-L with multi-scale CBAM attention (P3/P4/P5).

You receive raw outputs from TWO independently trained detection models that analysed the same image:
  - Model A (DUT Anti-UAV): Specialised for occluded, partially hidden, or cluttered-background drones
  - Model B (VisioDECT): Trained across 6 drone types under sunny, cloudy, and evening conditions

Your job is to reason like an experienced ISR (Intelligence, Surveillance & Reconnaissance) analyst.

CHAIN-OF-THOUGHT REASONING PROCESS — follow these steps internally before writing output:
  Step 1 — Model Agreement: Do both models agree? Disagreement lowers reliability.
  Step 2 — Confidence Calibration: Is confidence > 60% (high)? 40–60% (moderate)? <40% (low)?
  Step 3 — Spatial Context: What does the quadrant location tell us about threat vectors?
  Step 4 — Session Pattern: Is this an isolated detection or part of a repeating pattern?
  Step 5 — Risk Classification: What is the realistic operational risk given all factors above?

OUTPUT SCHEMA — respond ONLY in valid JSON, no markdown fences, no extra keys:
{
  "threat_level": "LOW" | "MEDIUM" | "CRITICAL",
  "reasoning": "<3–4 sentences covering: what each model found, why one was selected, what the confidence level means operationally, and any reliability caveat based on model agreement>",
  "alert": "<one direct sentence addressed TO the operator — name the quadrant, confidence, and urgency; for clear scans say so plainly>",
  "actions": ["<specific operator action 1>", "<specific operator action 2>", "<specific operator action 3 if MEDIUM/CRITICAL>"],
  "escalate": true | false
}

REASONING QUALITY RULES:
- Be specific: use actual numbers from the input (confidence %, inference ms, box count)
- Explain WHY one model was preferred — not just that it had higher confidence
- For LOW threat: explain why the scene is clear and what would change this assessment
- For MEDIUM: explain what the uncertain confidence means for operator decision-making
- For CRITICAL: describe the realistic operational implications (perimeter breach, proximity, etc.)
- If models disagree: always flag this explicitly as a reliability concern
- Never repeat the prompt back. Never use placeholder text. Never say "Detection processed."

ACTIONS must be concrete operator directives — not generic:
  Good: "Dispatch ground unit to upper-left perimeter for visual verification"
  Bad:  "Take action" / "Monitor the situation" / "action1"

SESSION PATTERN AWARENESS:
- If history shows 3+ consecutive detections: flag as persistent threat pattern
- If history shows alternating detect/clear: flag as intermittent, possibly false positives
- If first detection this session: note this as an isolated event requiring confirmation"""


def run_agent(dut_detected, dut_conf, dut_ms, dut_boxes,
              vis_detected, vis_conf, vis_ms, vis_boxes,
              chosen_model, final_conf, threat_level, quadrant):
    if session_history:
        recent = session_history[-5:]
        hist   = "\n".join([
            f"  [{e['time']}] {'DETECTED' if e['detected'] else 'CLEAR'} "
            f"conf={e['confidence']:.2f} threat={e['threat']}"
            for e in recent
        ])
        consecutive_detections = sum(1 for e in reversed(session_history[-3:]) if e["detected"])
        pattern_note = (
            "WARNING: 3 consecutive detections — persistent threat pattern active."
            if consecutive_detections == 3 else
            "Pattern: Mixed detect/clear — treat with caution for false positives."
            if any(e["detected"] for e in session_history[-3:]) and any(not e["detected"] for e in session_history[-3:])
            else "Pattern: Stable clear history."
        )
        ctx = (f"SESSION HISTORY (last {len(recent)} scans):\n{hist}\n"
               f"Total detections this session: "
               f"{sum(1 for e in session_history if e['detected'])}/{len(session_history)}\n"
               f"{pattern_note}")
    else:
        ctx = "SESSION HISTORY: No prior scans this session. Treat as isolated event requiring confirmation."

    selection_rationale = (
        f"DUT selected — occlusion specialist outperformed scenario model ({dut_conf:.1%} vs {vis_conf:.1%})"
        if chosen_model.startswith("DUT")
        else f"VisioDECT selected — scenario-generalised model outperformed occlusion specialist ({vis_conf:.1%} vs {dut_conf:.1%})"
    )

    model_agreement = (
        "AGREEMENT: Both models concur on detection status — result reliability is HIGH."
        if dut_detected == vis_detected
        else "DISAGREEMENT: Models diverge on detection — reliability is REDUCED, treat with caution."
    )

    msg = (
        f"=== DUAL MODEL SCAN RESULTS ===\n\n"
        f"Model A — DUT Anti-UAV (Occlusion Specialist):\n"
        f"  Detection  : {'DRONE DETECTED' if dut_detected else 'NO DRONE'}\n"
        f"  Confidence : {dut_conf:.1%}  ({dut_conf:.3f} raw)\n"
        f"  Bounding boxes : {dut_boxes}\n"
        f"  Inference time : {dut_ms:.0f}ms\n\n"
        f"Model B — VisioDECT (Multi-Scenario):\n"
        f"  Detection  : {'DRONE DETECTED' if vis_detected else 'NO DRONE'}\n"
        f"  Confidence : {vis_conf:.1%}  ({vis_conf:.3f} raw)\n"
        f"  Bounding boxes : {vis_boxes}\n"
        f"  Inference time : {vis_ms:.0f}ms\n\n"
        f"=== ARBITRATION ===\n"
        f"Selected   : {chosen_model}\n"
        f"Rationale  : {selection_rationale}\n"
        f"{model_agreement}\n\n"
        f"=== SCENE CONTEXT ===\n"
        f"Final confidence : {final_conf:.1%}\n"
        f"Threat class     : {threat_level}\n"
        f"Spatial location : {quadrant} quadrant\n"
        f"Escalate flag    : {'YES — confidence exceeds 60% threshold' if final_conf >= 0.60 and (dut_detected or vis_detected) else 'NO'}\n\n"
        f"=== OPERATOR SESSION ===\n"
        f"{ctx}\n\n"
        f"Apply your 5-step reasoning chain and produce the JSON output now."
    )

    try:
        _client = ollama.Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        resp = _client.chat(
            model=os.getenv("NETRA_LLM_MODEL", "llama3.2:3b"),
            messages=[
                {"role": "system", "content": AGENT_SYSTEM},
                {"role": "user",   "content": msg},
            ],
            options={"temperature": 0.35},
        )
        raw = resp["message"]["content"].strip()

        # Strip markdown fences if present
        if "```" in raw:
            for part in raw.split("```"):
                cleaned = part.strip().lstrip("json").strip()
                if cleaned.startswith("{"):
                    raw = cleaned
                    break

        # Extract outermost JSON object
        s, e = raw.find("{"), raw.rfind("}") + 1
        if s != -1 and e > s:
            raw = raw[s:e]

        result    = json.loads(raw)
        reasoning = result.get("reasoning", "").strip()
        alert     = result.get("alert",     "").strip()
        actions   = result.get("actions",   [])

        any_detected = dut_detected or vis_detected

        # Reject placeholder actions the LLM sometimes copies from the prompt
        _placeholders = {"action1", "action2", "action3",
                         "concrete action verb phrase 1", "concrete action verb phrase 2",
                         "concrete action verb phrase 3", "specific operator action 1",
                         "specific operator action 2", "specific operator action 3 if medium/critical"}
        if not actions or all(a.strip().lower() in _placeholders for a in actions):
            actions = _default_actions(any_detected, threat_level, quadrant)

        if not reasoning:
            reasoning = _build_reasoning(
                dut_detected, dut_conf, vis_detected, vis_conf,
                chosen_model, final_conf, threat_level, quadrant
            )
        if not alert or alert.lower() in ("detection processed.", ""):
            alert = _build_alert(any_detected, final_conf, threat_level, quadrant)

        return (
            result.get("threat_level", threat_level),
            reasoning,
            alert,
            actions,
            result.get("escalate", final_conf >= 0.60 and any_detected),
        )

    except Exception:
        any_detected = dut_detected or vis_detected
        return (
            threat_level,
            _build_reasoning(dut_detected, dut_conf, vis_detected, vis_conf,
                             chosen_model, final_conf, threat_level, quadrant),
            _build_alert(any_detected, final_conf, threat_level, quadrant),
            _default_actions(any_detected, threat_level, quadrant),
            final_conf >= 0.60 and any_detected,
        )


def _build_reasoning(dut_det, dut_conf, vis_det, vis_conf,
                     chosen, final_conf, threat, quadrant):
    dut_s = f"detected a drone ({dut_conf:.0%})" if dut_det else f"found no drone ({dut_conf:.0%})"
    vis_s = f"detected a drone ({vis_conf:.0%})" if vis_det else f"found no drone ({vis_conf:.0%})"
    agreement = "Both models agree — detection confidence is high." if dut_det == vis_det else \
                "The models disagree — treat this result with caution and seek visual confirmation."
    return (
        f"DUT Anti-UAV {dut_s} while VisioDECT {vis_s}. "
        f"{chosen} was selected as the authoritative result due to its higher confidence output. "
        f"{'A drone signature is present in the ' + quadrant + ' quadrant' if (dut_det or vis_det) else 'No drone was detected in this frame'}, "
        f"placing the threat at {threat} "
        f"({'above escalation threshold' if final_conf >= 0.60 else 'below escalation threshold — monitoring recommended'}). "
        f"{agreement}"
    )


def _build_alert(detected, conf, threat, quadrant):
    if detected:
        urgency = "CRITICAL ALERT" if threat == "CRITICAL" else "ALERT"
        return (
            f"{urgency}: Drone detected in the {quadrant} sector at {conf:.0%} confidence — "
            f"initiate visual confirmation and notify security personnel immediately."
        )
    return (
        f"All clear — no drone detected in this frame. "
        f"Continue standard monitoring protocols."
    )


def _default_actions(detected, threat, quadrant="unknown"):
    if not detected:
        return ["Continue standard patrol sweep", "Log clear result to session history"]
    if threat == "CRITICAL":
        return [
            f"Dispatch ground unit to {quadrant} perimeter for visual verification",
            "Notify security operations centre and initiate alert protocol",
            "Log detection with timestamp, confidence, and quadrant for incident report"
        ]
    return [
        f"Increase sensor coverage of {quadrant} sector",
        "Log detection and flag for follow-up if repeated",
        "Prepare escalation if next scan confirms presence"
    ]


# ─────────────────────────────────────────────
# Core inference — dual model, auto best-pick
# ─────────────────────────────────────────────

def run_detection(image):
    if image is None:
        return (None, "No image uploaded.", "", "", "LOW", "")

    dut_conf, dut_quad, dut_det, dut_ann, dut_ms, dut_boxes, dut_err = \
        _run_single_model(image, "DUT Anti-UAV (Occlusion)")
    vis_conf, vis_quad, vis_det, vis_ann, vis_ms, vis_boxes, vis_err = \
        _run_single_model(image, "VisioDECT (Multi-Scenario)")

    if dut_err and vis_err:
        return (None, f"Both models failed.\n{dut_err}\n{vis_err}", "", "", "LOW", "")

    if dut_conf >= vis_conf:
        chosen_model = "DUT Anti-UAV (Occlusion)"
        best_conf, best_quad, detected = dut_conf, dut_quad, dut_det
        annotated, best_ms, best_boxes = dut_ann, dut_ms, dut_boxes
    else:
        chosen_model = "VisioDECT (Multi-Scenario)"
        best_conf, best_quad, detected = vis_conf, vis_quad, vis_det
        annotated, best_ms, best_boxes = vis_ann, vis_ms, vis_boxes

    threat_level = compute_threat_level(detected, best_conf)

    agent_threat, reasoning, alert, actions, escalate = run_agent(
        dut_det, dut_conf, dut_ms, dut_boxes,
        vis_det, vis_conf, vis_ms, vis_boxes,
        chosen_model, best_conf, threat_level, best_quad
    )

    add_to_history(detected, best_conf, agent_threat, best_quad, chosen_model)

    if escalate:
        log_escalation(
            f"{'DRONE' if detected else 'CLEAR'} | Threat:{agent_threat} | Quad:{best_quad}",
            alert, agent_threat, best_conf
        )

    annotated_pil = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    status = (
        f"{'DRONE DETECTED' if detected else 'AREA CLEAR'}\n"
        f"Confidence  : {best_conf:.1%}\n"
        f"Threat      : {agent_threat}\n"
        f"Quadrant    : {best_quad}\n"
        f"Detections  : {best_boxes}\n"
        f"Model used  : {chosen_model}\n"
        f"Inference   : {best_ms:.0f}ms\n"
        f"Escalated   : {'YES' if escalate else 'No'}"
    )
    actions_text = "\n".join([f"  {a}" for a in actions])
    return (annotated_pil, status, reasoning, alert, agent_threat, actions_text)


# ─────────────────────────────────────────────
# Frame tracker (tracking always ON)
# ─────────────────────────────────────────────

_track_registry   = {}
_next_track_id    = [0]
_track_trails     = {}
_annotated_frames = []

def _assign_track(cx, cy, max_dist=80):
    best_id, best_d = None, max_dist
    for tid,(tx,ty) in _track_registry.items():
        d = ((cx-tx)**2+(cy-ty)**2)**0.5
        if d < best_d:
            best_id, best_d = tid, d
    if best_id is None:
        best_id = _next_track_id[0]
        _next_track_id[0] += 1
    _track_registry[best_id] = (cx, cy)
    return best_id

def reset_tracker():
    _track_registry.clear()
    _track_trails.clear()
    _next_track_id[0] = 0

_TRACK_COLORS = [
    (0,255,80),(0,220,255),(180,0,255),(0,180,255),(255,0,180),(100,255,200),
]

def run_frame_batch(files):
    global _annotated_frames
    if not files:
        _annotated_frames = []
        return None, "No frames loaded."

    reset_tracker()
    annotated_gallery = []

    for f in files:
        img     = np.array(Image.open(f.name).convert("RGB"))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_h, img_w = img_bgr.shape[:2]

        dut_conf,_,dut_det,_,dut_ms,dut_boxes,_ = _run_single_model(img_bgr,"DUT Anti-UAV (Occlusion)")
        vis_conf,_,vis_det,_,vis_ms,vis_boxes,_ = _run_single_model(img_bgr,"VisioDECT (Multi-Scenario)")

        chosen = "DUT Anti-UAV (Occlusion)" if dut_conf >= vis_conf else "VisioDECT (Multi-Scenario)"
        model, _ = get_model(chosen)
        results  = model.predict(source=img_bgr, conf=CONF_THRESHOLD, imgsz=640, verbose=False)

        frame_out = img_bgr.copy()
        boxes     = results[0].boxes
        detected  = boxes is not None and len(boxes) > 0
        best_conf = max((float(b.conf[0]) for b in boxes), default=0.0) if detected else 0.0
        threat    = compute_threat_level(detected, best_conf)

        if detected:
            for box in boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cx,cy = (x1+x2)//2,(y1+y2)//2
                tid   = _assign_track(cx, cy)
                color = _TRACK_COLORS[tid % len(_TRACK_COLORS)]
                if tid not in _track_trails:
                    _track_trails[tid] = []
                _track_trails[tid].append((cx,cy))
                trail = _track_trails[tid]
                overlay = frame_out.copy()
                for i in range(1, len(trail)):
                    alpha     = 0.15 + 0.85 * (i / len(trail))
                    thickness = max(1, int(3 * (i / len(trail))))
                    cv2.line(overlay, trail[i-1], trail[i], color, thickness, cv2.LINE_AA)
                    cv2.addWeighted(overlay, alpha, frame_out, 1-alpha, 0, frame_out)
                    overlay = frame_out.copy()
                cv2.circle(frame_out,(cx,cy),7,color,-1,cv2.LINE_AA)
                cv2.circle(frame_out,(cx,cy),7,(255,255,255),1,cv2.LINE_AA)
                label = f"#{tid}  {conf:.2f}"
                cv2.rectangle(frame_out,(x1,y1),(x2,y2),color,2)
                (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.55,2)
                cv2.rectangle(frame_out,(x1,y1-th-8),(x1+tw+4,y1),color,-1)
                cv2.putText(frame_out,label,(x1+2,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,0),2)
            cv2.rectangle(frame_out,(0,0),(img_w,38),(8,18,40),-1)
            cv2.putText(frame_out,f"DETECTED  Conf:{best_conf:.2f}  {threat}  [{chosen[:3]}]",
                        (10,26),cv2.FONT_HERSHEY_SIMPLEX,0.6,(56,189,248),2)
        else:
            cv2.rectangle(frame_out,(0,0),(img_w,38),(8,14,28),-1)
            cv2.putText(frame_out,f"NO DRONE  {threat}",
                        (10,26),cv2.FONT_HERSHEY_SIMPLEX,0.6,(148,163,184),2)

        annotated_gallery.append(Image.fromarray(cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)))
        add_to_history(detected, best_conf, threat,
                       get_quadrant(img_w//2, img_h//2, img_w, img_h),
                       chosen, source="batch")

    _annotated_frames = annotated_gallery
    n = len(_annotated_frames)
    return _annotated_frames[0], f"Frame  1 / {n}"


def _nav_frame(current_label, delta):
    if not _annotated_frames:
        return None, "No frames loaded."
    n = len(_annotated_frames)
    try:
        idx = int(current_label.split()[1]) - 1
    except Exception:
        idx = 0
    idx = (idx + delta) % n
    return _annotated_frames[idx], f"Frame  {idx+1} / {n}"


# ─────────────────────────────────────────────
# TASK 3: Project Details HTML — Rewritten
# ─────────────────────────────────────────────

PROJECT_DETAILS_HTML = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

.pd-root {
    font-family: 'Space Grotesk', 'Inter', system-ui, sans-serif;
    background: #070b14;
    color: #cbd5e1;
    padding: 32px 40px;
    border-radius: 12px;
    border: 1px solid #1e293b;
    line-height: 1.7;
}

/* ── Page title ── */
.pd-hero {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 16px;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 24px;
    margin-bottom: 28px;
}
.pd-hero-left {}
.pd-title {
    font-size: 36px;
    font-weight: 700;
    letter-spacing: 6px;
    background: linear-gradient(120deg, #38bdf8 0%, #7dd3fc 40%, #e2e8f0 70%, #38bdf8 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: pdShim 5s linear infinite;
    margin: 0 0 4px 0;
    line-height: 1.1;
}
@keyframes pdShim {
    0%   { background-position: 0% center }
    100% { background-position: 200% center }
}
.pd-subtitle {
    font-size: 11px;
    color: #334155;
    letter-spacing: 0.25em;
    font-weight: 500;
    text-transform: uppercase;
}
.pd-affil {
    font-size: 11px;
    color: #475569;
    letter-spacing: 0.08em;
    margin-top: 6px;
}
.pd-hero-right {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 6px;
}
.pd-status-pill {
    display: flex;
    align-items: center;
    gap: 7px;
    background: rgba(52,211,153,0.07);
    border: 1px solid rgba(52,211,153,0.25);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 11px;
    color: #34d399;
    font-weight: 600;
    letter-spacing: 0.1em;
}
.pd-status-dot {
    width: 7px; height: 7px;
    background: #34d399;
    border-radius: 50%;
    box-shadow: 0 0 6px #34d399;
    animation: pdBlink 2s ease-in-out infinite;
}
@keyframes pdBlink { 0%,100%{opacity:1} 50%{opacity:.3} }
.pd-version-badge {
    font-size: 11px;
    color: #475569;
    letter-spacing: 0.1em;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Section heading ── */
.pd-section-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.28em;
    color: #38bdf8;
    text-transform: uppercase;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 7px;
    margin: 32px 0 16px 0;
}

/* ── Card grid ── */
.pd-card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 14px;
    margin: 0 0 10px 0;
}
.pd-card {
    background: #0d1424;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 18px 20px;
    transition: border-color 0.22s, box-shadow 0.22s, transform 0.22s;
    cursor: default;
}
.pd-card:hover {
    border-color: rgba(56,189,248,0.38);
    box-shadow: 0 0 24px rgba(56,189,248,0.07), 0 4px 20px rgba(0,0,0,0.3);
    transform: translateY(-2px);
}
.pd-card-icon {
    font-size: 22px;
    margin-bottom: 10px;
    display: block;
}
.pd-card-title {
    font-size: 12px;
    font-weight: 700;
    color: #38bdf8;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.pd-card-body {
    font-size: 12px;
    color: #64748b;
    line-height: 1.6;
}
.pd-card-body strong {
    color: #94a3b8;
    font-weight: 600;
}

/* ── Objective strip ── */
.pd-objective {
    background: linear-gradient(135deg, #0d1a30, #0a1220);
    border: 1px solid rgba(56,189,248,0.18);
    border-left: 3px solid #38bdf8;
    border-radius: 8px;
    padding: 14px 20px;
    font-size: 13px;
    color: #94a3b8;
    line-height: 1.7;
}

/* ── Tech stack pills ── */
.pd-stack-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 8px 0;
}
.pd-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.05em;
    transition: box-shadow 0.18s, transform 0.18s;
    cursor: default;
}
.pd-pill:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
.pill-blue   { background:rgba(56,189,248,0.09);  border:1px solid rgba(56,189,248,0.28); color:#38bdf8; }
.pill-amber  { background:rgba(232,184,75,0.09);  border:1px solid rgba(232,184,75,0.28); color:#e8b84b; }
.pill-green  { background:rgba(52,211,153,0.09);  border:1px solid rgba(52,211,153,0.28); color:#34d399; }
.pill-slate  { background:rgba(100,116,139,0.09); border:1px solid rgba(100,116,139,0.28); color:#94a3b8; }
.pill-violet { background:rgba(139,92,246,0.09);  border:1px solid rgba(139,92,246,0.28); color:#a78bfa; }

/* ── Architecture flow ── */
.pd-flow-wrap {
    background: #080c14;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 20px 24px;
    overflow-x: auto;
}
.pd-flow {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: nowrap;
    min-width: max-content;
}
.pd-flow-node {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 5px 13px;
    font-size: 11px;
    color: #7dd3fc;
    white-space: nowrap;
    transition: border-color 0.18s, color 0.18s;
    font-family: 'JetBrains Mono', monospace;
}
.pd-flow-node:hover { border-color: rgba(56,189,248,0.5); color: #38bdf8; }
.pd-flow-node.amber { color: #e8b84b; border-color: rgba(232,184,75,0.2); }
.pd-flow-node.green { color: #34d399; border-color: rgba(52,211,153,0.2); }
.pd-flow-arrow { color: #1e293b; font-size: 16px; flex-shrink: 0; }

/* ── Metrics bar ── */
.pd-metrics {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
    margin: 8px 0;
}
.pd-metric {
    background: #0d1424;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 14px 16px;
    text-align: center;
    transition: border-color 0.2s, transform 0.2s;
}
.pd-metric:hover { border-color: rgba(56,189,248,0.3); transform: translateY(-1px); }
.pd-metric-val {
    font-size: 22px;
    font-weight: 700;
    color: #38bdf8;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
    margin-bottom: 4px;
}
.pd-metric-val.amber { color: #e8b84b; }
.pd-metric-val.green { color: #34d399; }
.pd-metric-label {
    font-size: 10px;
    color: #475569;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 600;
}

/* ── Highlight list ── */
.pd-list {
    padding-left: 0;
    list-style: none;
    margin: 8px 0;
}
.pd-list li {
    font-size: 12px;
    color: #64748b;
    padding: 5px 0 5px 20px;
    position: relative;
    border-bottom: 1px solid #0d1424;
    transition: color 0.18s;
}
.pd-list li:hover { color: #94a3b8; }
.pd-list li::before {
    content: '▸';
    position: absolute;
    left: 0;
    color: #38bdf8;
    font-size: 11px;
}
.pd-list li strong { color: #7dd3fc; font-weight: 600; }
code.pd-code {
    background: #111827;
    color: #38bdf8;
    padding: 1px 7px;
    border-radius: 4px;
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    border: 1px solid #1e293b;
}

/* ── Print button ── */
.pd-print-btn {
    margin-top: 32px;
    padding: 8px 24px;
    background: transparent;
    border: 1px solid #38bdf8;
    border-radius: 6px;
    color: #38bdf8;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.15em;
    cursor: pointer;
    text-transform: uppercase;
    transition: background 0.18s, box-shadow 0.18s;
    font-family: 'Space Grotesk', sans-serif;
}
.pd-print-btn:hover {
    background: rgba(56,189,248,0.08);
    box-shadow: 0 0 14px rgba(56,189,248,0.18);
}

@media print {
    .pd-root { background:#fff!important; color:#000!important; }
    .pd-title { -webkit-text-fill-color:#000!important; }
    .pd-card,.pd-metric,.pd-flow-node { background:#f9fafb!important; border-color:#ddd!important; }
    .pd-print-btn { display:none; }
}
</style>

<div class="pd-root" id="netra-project-details">

    <!-- ── Hero ── -->
    <div class="pd-hero">
        <div class="pd-hero-left">
            <div class="pd-title">NETRA</div>
            <div class="pd-subtitle">Neural Enhanced Threat Recognition &amp; Analysis</div>
            <div class="pd-affil">GITAM University, Hyderabad &nbsp;·&nbsp; Final Year Project &nbsp;·&nbsp; 2026</div>
        </div>
        <div class="pd-hero-right">
            <div class="pd-status-pill">
                <div class="pd-status-dot"></div>
                SYSTEM ONLINE
            </div>
            <div class="pd-version-badge">v1.3 &nbsp;·&nbsp; Dual Model &nbsp;·&nbsp; Offline-First</div>
        </div>
    </div>

    <!-- ── Objective ── -->
    <div class="pd-section-label">01 &nbsp;·&nbsp; Objective</div>
    <div class="pd-objective">
        Design and deploy an end-to-end, offline-capable drone detection and threat assessment system
        that combines dual RT-DETR-L inference with multi-scale CBAM attention — achieving
        high detection recall across occluded, small, and multi-scenario drone targets — and bridges
        the gap between raw model output and operator-ready intelligence through a locally deployed LLM agent.
    </div>

    <!-- ── Key Metrics ── -->
    <div class="pd-section-label">02 &nbsp;·&nbsp; Key Results</div>
    <div class="pd-metrics">
        <div class="pd-metric">
            <div class="pd-metric-val">90.1%</div>
            <div class="pd-metric-label">mAP@0.5 (VisioDECT)</div>
        </div>
        <div class="pd-metric">
            <div class="pd-metric-val">90.9%</div>
            <div class="pd-metric-label">F1-Score</div>
        </div>
        <div class="pd-metric">
            <div class="pd-metric-val amber">3</div>
            <div class="pd-metric-label">CBAM Injection Layers (P3/P4/P5)</div>
        </div>
        <div class="pd-metric">
            <div class="pd-metric-val">2</div>
            <div class="pd-metric-label">Specialised Detection Models</div>
        </div>
        <div class="pd-metric">
            <div class="pd-metric-val green">0</div>
            <div class="pd-metric-label">Cloud Dependencies</div>
        </div>
        <div class="pd-metric">
            <div class="pd-metric-val amber">10K</div>
            <div class="pd-metric-label">DUT Anti-UAV Training Images</div>
        </div>
    </div>

    <!-- ── What It Does ── -->
    <div class="pd-section-label">03 &nbsp;·&nbsp; Core Modules</div>
    <div class="pd-card-grid">
        <div class="pd-card">
            <span class="pd-card-icon">🎯</span>
            <div class="pd-card-title">Dual-Model Inference</div>
            <div class="pd-card-body">
                Two RT-DETR-L models run in parallel on every image.
                <strong>DUT Anti-UAV</strong> handles occluded &amp; partial drones.
                <strong>VisioDECT</strong> covers 6 drone types across weather conditions.
                Best-confidence result is selected automatically.
            </div>
        </div>
        <div class="pd-card">
            <span class="pd-card-icon">🧠</span>
            <div class="pd-card-title">SENTINEL Agent</div>
            <div class="pd-card-body">
                Local LLM agent (llama3.2:3b via Ollama) applies 5-step ISR-style
                reasoning across both model outputs, session history, and spatial context —
                producing structured operator alerts with recommended actions.
                <strong>Fully offline.</strong>
            </div>
        </div>
        <div class="pd-card">
            <span class="pd-card-icon">📡</span>
            <div class="pd-card-title">Multi-Scale CBAM</div>
            <div class="pd-card-body">
                Convolutional Block Attention injected at FPN layers
                <strong>P3 (stride 8), P4 (stride 16), P5 (stride 32)</strong> —
                the key innovation over prior single-scale approaches.
                Enables simultaneous attention refinement at small, medium, and large drone scales.
            </div>
        </div>
        <div class="pd-card">
            <span class="pd-card-icon">🎞️</span>
            <div class="pd-card-title">Frame Tracker</div>
            <div class="pd-card-body">
                Multi-frame centroid tracking with persistent colour-coded IDs and
                fading motion trails. Upload video frames in sequence and step
                through each annotated frame with <strong>◀ / ▶</strong> navigation.
            </div>
        </div>
        <div class="pd-card">
            <span class="pd-card-icon">📊</span>
            <div class="pd-card-title">Session Dashboard</div>
            <div class="pd-card-body">
                Live detection history, session statistics, threat distribution,
                and CSV export — with source tagging for <strong>single [S]</strong>
                and <strong>batch [B]</strong> detections.
            </div>
        </div>
        <div class="pd-card">
            <span class="pd-card-icon">⚡</span>
            <div class="pd-card-title">Escalation Engine</div>
            <div class="pd-card-body">
                Auto-escalation when confidence exceeds <strong>60%</strong>.
                Escalation events are written to a persistent local alert log
                with timestamp, threat level, and operator alert text.
            </div>
        </div>
    </div>

    <!-- ── Tech Stack ── -->
    <div class="pd-section-label">04 &nbsp;·&nbsp; Tech Stack</div>
    <div class="pd-stack-row">
        <span class="pd-pill pill-blue">⚙ RT-DETR-L</span>
        <span class="pd-pill pill-blue">📐 CBAM P3/P4/P5</span>
        <span class="pd-pill pill-blue">🔬 PyTorch</span>
        <span class="pd-pill pill-blue">🏃 Ultralytics</span>
    </div>
    <div class="pd-stack-row">
        <span class="pd-pill pill-amber">🤖 llama3.2:3b</span>
        <span class="pd-pill pill-amber">🖥 Ollama (Local)</span>
        <span class="pd-pill pill-amber">⚡ ReAct Agent</span>
    </div>
    <div class="pd-stack-row">
        <span class="pd-pill pill-slate">🎨 Gradio 4.x</span>
        <span class="pd-pill pill-slate">🖼 OpenCV</span>
        <span class="pd-pill pill-slate">🐍 Python 3.12</span>
    </div>
    <div class="pd-stack-row">
        <span class="pd-pill pill-green">🏋 VisioDECT (19.5K images)</span>
        <span class="pd-pill pill-green">🌫 DUT Anti-UAV (10K images)</span>
    </div>

    <!-- ── System Architecture ── -->
    <div class="pd-section-label">05 &nbsp;·&nbsp; System Architecture</div>
    <div class="pd-flow-wrap">
        <div style="font-size:10px;color:#334155;letter-spacing:0.15em;margin-bottom:12px;">SINGLE DETECTION FLOW</div>
        <div class="pd-flow">
            <div class="pd-flow-node">📷 Upload Image</div>
            <div class="pd-flow-arrow">→</div>
            <div class="pd-flow-node">DUT Anti-UAV</div>
            <div class="pd-flow-arrow">⟩</div>
            <div class="pd-flow-node">VisioDECT</div>
            <div class="pd-flow-arrow">→</div>
            <div class="pd-flow-node">Best-Pick Arbitration</div>
            <div class="pd-flow-arrow">→</div>
            <div class="pd-flow-node amber">SENTINEL Agent</div>
            <div class="pd-flow-arrow">→</div>
            <div class="pd-flow-node green">Operator Output</div>
        </div>
        <div style="margin-top:16px;font-size:10px;color:#334155;letter-spacing:0.15em;margin-bottom:12px;">FRAME TRACKER FLOW</div>
        <div class="pd-flow">
            <div class="pd-flow-node">🎞 Upload Frames</div>
            <div class="pd-flow-arrow">→</div>
            <div class="pd-flow-node">Both Models per Frame</div>
            <div class="pd-flow-arrow">→</div>
            <div class="pd-flow-node">Centroid Tracker</div>
            <div class="pd-flow-arrow">→</div>
            <div class="pd-flow-node">Fading Trail Render</div>
            <div class="pd-flow-arrow">→</div>
            <div class="pd-flow-node green">Slide Viewer + Log</div>
        </div>
    </div>

    <!-- ── Innovation ── -->
    <div class="pd-section-label">06 &nbsp;·&nbsp; Innovation &amp; Highlights</div>
    <ul class="pd-list">
        <li><strong>Multi-Scale CBAM Injection:</strong> CBAM applied at all three FPN levels (P3/P4/P5) — the primary technical contribution over prior single-scale approaches (+0.57% mAP, +0.95% precision over single-scale baseline)</li>
        <li><strong>Occlusion-Robust Training:</strong> DUT Anti-UAV dataset (10K occlusion-heavy images) + copy-paste augmentation (p=0.3) targets the most critical real-world failure mode</li>
        <li><strong>ISR-Style Agent Reasoning:</strong> SENTINEL applies a 5-step chain-of-thought (model agreement → confidence calibration → spatial context → session pattern → risk classification) producing auditable, plain-English operator intelligence</li>
        <li><strong>Fully Offline Architecture:</strong> LLM runs via Ollama locally — no API keys, no cloud calls, zero data exfiltration risk — suitable for security-sensitive operational environments</li>
        <li><strong>Persistent Session Memory:</strong> Agent receives last 5 detections and flags persistent threat patterns or alternating detect/clear instability</li>
        <li><strong>Dual-Model Auto-Arbitration:</strong> Operator never selects a model manually — the higher-confidence result from two specialised models is chosen transparently</li>
    </ul>

    <!-- ── Setup Notes ── -->
    <div class="pd-section-label">07 &nbsp;·&nbsp; Setup Requirements</div>
    <ul class="pd-list">
        <li>Model weights <code class="pd-code">best_dut.pt</code> and <code class="pd-code">best_visiodect.pt</code> must be present in the working directory</li>
        <li>Run <code class="pd-code">ollama serve</code> before launching — LLM agent falls back to rule-based output if Ollama is unavailable</li>
        <li>Device auto-selected: CUDA → MPS (Apple Silicon) → CPU</li>
        <li>Alert escalation log written to <code class="pd-code">$TMPDIR/netra_alerts.txt</code> for cross-platform compatibility</li>
        <li>Environment variables: <code class="pd-code">NETRA_MODEL_DUT</code>, <code class="pd-code">NETRA_MODEL_VISIODECT</code>, <code class="pd-code">OLLAMA_HOST</code>, <code class="pd-code">NETRA_LLM_MODEL</code></li>
    </ul>

    <button class="pd-print-btn" onclick="window.print()">⎙ &nbsp; PRINT / SAVE AS PDF</button>
</div>
"""


# ─────────────────────────────────────────────
# TASK 2: Global CSS — enhanced with animations
# ─────────────────────────────────────────────

GLOBAL_CSS = """
<style>
/* ═══ worldmonitor-inspired palette ═══
   Base bg:      #0b0f1a
   Panel:        #111827
   Border:       #1e293b
   Cyan (data):  #38bdf8
   Amber (alert):#e8b84b
   Green (safe): #34d399
   Text:         #e2e8f0
   Muted:        #64748b
   ══════════════════════════════════════ */

/* ── Kill Gradio orange tokens ── */
*, *::before, *::after {
    --color-accent:        #38bdf8 !important;
    --color-accent-soft:   rgba(56,189,248,0.12) !important;
    --button-primary-background-fill:       linear-gradient(135deg,#0c2240,#0d3060) !important;
    --button-primary-background-fill-hover: linear-gradient(135deg,#0d3060,#1050a0) !important;
    --button-primary-border-color:  #38bdf8 !important;
    --button-primary-text-color:    #38bdf8 !important;
    --button-secondary-background-fill: transparent !important;
    --button-secondary-border-color: #1e293b !important;
    --button-secondary-text-color:   #64748b !important;
    --color-orange-500: #38bdf8 !important;
    --color-orange-400: #38bdf8 !important;
    --color-orange-300: #7dd3fc !important;
    --slider-color: #38bdf8 !important;
}
html, body, .gradio-container {
    background: #0b0f1a !important;
    font-family: 'Inter','Segoe UI',system-ui,sans-serif !important;
    color: #e2e8f0 !important;
}

/* ── Tab strip ── */
.tab-nav { background: #0d1120 !important; border-bottom: 1px solid #1e293b !important; }
.tab-nav button {
    background: transparent !important; color: #64748b !important;
    border: none !important; border-bottom: 2px solid transparent !important;
    font-size: 12px !important; font-weight: 600 !important;
    letter-spacing: 0.1em !important; text-transform: uppercase !important;
    padding: 12px 20px !important; transition: all 0.22s ease !important;
}
.tab-nav button:hover {
    color: #cbd5e1 !important;
    border-bottom-color: rgba(56,189,248,.3) !important;
    background: rgba(56,189,248,.03) !important;
}
.tab-nav button.selected {
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8 !important;
    background: rgba(56,189,248,.04) !important;
}

/* ── Textboxes ── */
textarea, input[type=text] {
    background: #0d1424 !important; border: 1px solid #1e293b !important;
    color: #cbd5e1 !important; font-family: 'JetBrains Mono','Fira Code',monospace !important;
    font-size: 13px !important; border-radius: 6px !important; line-height: 1.65 !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
textarea:focus, input[type=text]:focus {
    border-color: rgba(56,189,248,.45) !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,.08) !important; outline: none !important;
}

/* ── Labels ── */
label > span, label span {
    color: #475569 !important; font-size: 11px !important; font-weight: 600 !important;
    letter-spacing: 0.12em !important; text-transform: uppercase !important;
}

/* ── Primary buttons — enhanced hover ── */
button, .gr-button, button.primary, button[data-testid="primary"],
button[variant="primary"], div[data-testid="primary"] button {
    background: linear-gradient(135deg,#0c2240,#0d3060) !important;
    border: 1px solid #38bdf8 !important; color: #38bdf8 !important;
    font-size: 13px !important; font-weight: 600 !important;
    letter-spacing: 0.1em !important; border-radius: 6px !important;
    transition: all 0.22s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    box-shadow: 0 0 0 rgba(56,189,248,0) !important;
}
button:hover, button[data-testid="primary"]:hover {
    background: linear-gradient(135deg,#0d3060,#1050a0) !important;
    box-shadow: 0 0 22px rgba(56,189,248,.28), 0 4px 16px rgba(0,0,0,.4) !important;
    color: #7dd3fc !important; border-color: #7dd3fc !important;
    transform: translateY(-1px) !important;
}
button:active {
    transform: translateY(0px) !important;
    box-shadow: 0 0 10px rgba(56,189,248,.15) !important;
}

/* ── Secondary buttons ── */
button[data-testid="secondary"], button.secondary {
    background: #0d1424 !important; border: 1px solid #1e293b !important;
    color: #64748b !important; font-size: 12px !important; box-shadow: none !important;
    transition: all 0.2s ease !important;
}
button[data-testid="secondary"]:hover, button.secondary:hover {
    border-color: rgba(56,189,248,.4) !important; color: #38bdf8 !important;
    background: rgba(56,189,248,.05) !important;
    box-shadow: 0 0 12px rgba(56,189,248,.1) !important;
    transform: translateY(-1px) !important;
}

/* ── Upload zone ── */
.upload-container, [data-testid="upload-btn"], .gr-file-upload, .file-preview {
    background: #0d1424 !important; border: 1px dashed #1e293b !important;
    border-radius: 8px !important; color: #475569 !important;
    transition: border-color 0.2s !important;
}
.upload-container:hover, [data-testid="upload-btn"]:hover {
    border-color: rgba(56,189,248,.35) !important;
}

/* ── Image frames ── */
.netra-img img, .netra-img video {
    object-fit: contain !important; width: 100% !important; height: auto !important;
    max-height: 460px !important; border-radius: 6px !important; border: 1px solid #1e293b !important;
    transition: border-color 0.2s !important;
}
.netra-img img:hover { border-color: rgba(56,189,248,.3) !important; }
.netra-img .image-container, .netra-img > div { background: #080c14 !important; border-radius: 6px !important; }

/* ── Gradio Rows — fade-in-up animation ── */
@keyframes netraFadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
.gradio-container .gr-block,
.gradio-container .gr-row,
.gradio-container .gr-column {
    animation: netraFadeUp 0.4s ease-out both;
}

/* ── Slider, checkbox ── */
input[type=range] { accent-color: #38bdf8 !important; }
input[type=range]::-webkit-slider-thumb { background: #38bdf8 !important; box-shadow: 0 0 6px rgba(56,189,248,.5) !important; }
input[type=range]::-webkit-slider-runnable-track { background: #1e293b !important; }
input[type=checkbox] { accent-color: #38bdf8 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0b0f1a; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #38bdf8; }

/* ── Frame counter ── */
#netra-frame-counter textarea {
    text-align: center !important; font-size: 13px !important; color: #38bdf8 !important;
    background: transparent !important; border: none !important;
    letter-spacing: 0.2em !important; padding: 0 !important; resize: none !important;
}

/* ═══ Drone loading overlay ═══ */
@keyframes droneFloat {
    0%,100%{transform:translateY(0) rotate(0deg);}
    30%{transform:translateY(-9px) rotate(1.5deg);}
    70%{transform:translateY(6px) rotate(-1.5deg);}
}
@keyframes propSpin { to{transform:rotate(360deg);} }
@keyframes radarRing {
    0%{transform:translate(-50%,-50%) scale(.4);opacity:.7;}
    100%{transform:translate(-50%,-50%) scale(3);opacity:0;}
}
@keyframes scanBar { 0%{left:-30%} 100%{left:110%} }
@keyframes loaderBlink { 0%,100%{opacity:1} 50%{opacity:.2} }

.netra-loading-overlay {
    display:none; position:fixed; inset:0;
    background:rgba(7,11,20,0.95); backdrop-filter:blur(10px) saturate(1.3);
    z-index:99999; flex-direction:column; align-items:center; justify-content:center; gap:26px;
}
.netra-loading-overlay.active { display:flex; }
.loader-drone-wrap { position:relative; width:96px; height:96px; animation:droneFloat 2s ease-in-out infinite; }
.loader-drone-body {
    position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
    width:22px; height:22px; background:#38bdf8; border-radius:5px;
    box-shadow:0 0 20px rgba(56,189,248,1),0 0 40px rgba(56,189,248,.4);
}
.loader-drone-arm {
    position:absolute; top:50%; left:50%; width:84px; height:2px;
    background:linear-gradient(90deg,transparent,rgba(56,189,248,.8),transparent);
    transform-origin:center; opacity:.9;
}
.loader-drone-arm-h{transform:translate(-50%,-50%) rotate(0deg);}
.loader-drone-arm-v{transform:translate(-50%,-50%) rotate(90deg);}
.loader-prop {
    position:absolute; width:18px; height:18px;
    border:2px solid #38bdf8; border-radius:50%; box-shadow:0 0 10px rgba(56,189,248,.6);
}
.loader-prop::before {
    content:''; position:absolute; top:50%; left:-6px; width:28px; height:2px;
    background:#38bdf8; transform:translateY(-50%);
    animation:propSpin .2s linear infinite; transform-origin:center;
}
.loader-prop-tl{top:0;left:0} .loader-prop-tr{top:0;right:0}
.loader-prop-bl{bottom:0;left:0} .loader-prop-br{bottom:0;right:0}
.loader-radar {
    position:absolute; top:50%; left:50%; width:78px; height:78px;
    border:1px solid rgba(56,189,248,.3); border-radius:50%;
    animation:radarRing 2s ease-out infinite;
}
.loader-bar {
    width:200px; height:2px; background:#1e293b; border-radius:2px;
    position:relative; overflow:hidden;
}
.loader-bar::after {
    content:''; position:absolute; top:0; height:100%; width:28%;
    background:linear-gradient(90deg,transparent,#38bdf8,transparent);
    animation:scanBar 1.3s linear infinite;
}
.loader-text {
    font-size:13px; font-weight:700; letter-spacing:.4em; color:#38bdf8;
    animation:loaderBlink 1.4s ease-in-out infinite;
}
.loader-subtext { font-size:11px; letter-spacing:.12em; color:#334155; }
</style>

<div class="netra-loading-overlay" id="netraLoader">
    <div class="loader-drone-wrap">
        <div class="loader-radar"></div>
        <div class="loader-drone-arm loader-drone-arm-h"></div>
        <div class="loader-drone-arm loader-drone-arm-v"></div>
        <div class="loader-drone-body"></div>
        <div class="loader-prop loader-prop-tl"></div>
        <div class="loader-prop loader-prop-tr"></div>
        <div class="loader-prop loader-prop-bl"></div>
        <div class="loader-prop loader-prop-br"></div>
    </div>
    <div class="loader-text">ANALYSING</div>
    <div class="loader-bar"></div>
    <div class="loader-subtext">DUAL MODEL INFERENCE  ·  SENTINEL REASONING</div>
</div>
<script>
(function(){
    var L=document.getElementById('netraLoader');
    function show(){L.classList.add('active');}
    function hide(){L.classList.remove('active');}
    function hook(){
        document.querySelectorAll('button').forEach(function(b){
            var t=b.textContent.trim();
            if(t==='ANALYSE'||t==='RUN ANALYSIS') b.addEventListener('click',show);
        });
    }
    setTimeout(hook,1800);
    new MutationObserver(function(ms){ms.forEach(function(m){if(m.addedNodes.length)hide();});})
        .observe(document.body,{childList:true,subtree:true});
})();
</script>
"""

# ─────────────────────────────────────────────
# TASK 2: Header — enhanced with radar ring, orbit drone, live clock
# ─────────────────────────────────────────────

HEADER = """
<style>
@keyframes hScan    { 0%{transform:translateX(-100%)} 100%{transform:translateX(500%)} }
@keyframes hBlink   { 0%,100%{opacity:1} 50%{opacity:.25} }
@keyframes hSlide   { from{opacity:0;transform:translateY(-10px)} to{opacity:1;transform:translateY(0)} }
@keyframes hProp    { to{transform:rotate(360deg)} }
@keyframes hPulse   { 0%,100%{box-shadow:0 0 6px rgba(56,189,248,.25)} 50%{box-shadow:0 0 18px rgba(56,189,248,.7)} }
@keyframes hShim    { 0%{background-position:0% center} 100%{background-position:200% center} }
@keyframes hGrid    { 0%{background-position:0 0} 100%{background-position:32px 32px} }
@keyframes hRadar   { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }
@keyframes hRadarRing {
    0%  { transform:translate(-50%,-50%) scale(0.5); opacity:0.6; }
    100%{ transform:translate(-50%,-50%) scale(2.2); opacity:0; }
}
@keyframes hOrbit {
    from{transform:rotate(0deg) translateX(46px) rotate(0deg);}
    to  {transform:rotate(360deg) translateX(46px) rotate(-360deg);}
}
@keyframes hFloat {
    0%,100%{transform:translateY(0) rotate(0deg);}
    50%{transform:translateY(-4px) rotate(1deg);}
}

.nh {
    background: linear-gradient(135deg,#07091a 0%,#0d1535 50%,#080d1f 100%);
    border: 1px solid #1e293b; border-radius: 12px;
    padding: 18px 28px; display: flex; align-items: center; justify-content: space-between;
    position: relative; overflow: hidden; margin-bottom: 12px;
    animation: hSlide .55s ease-out;
    box-shadow: 0 1px 0 rgba(56,189,248,.06) inset, 0 0 60px rgba(56,189,248,.05), 0 0 120px rgba(56,189,248,.02);
}
.nh::before {
    content:''; position:absolute; inset:0; pointer-events:none;
    background-image: linear-gradient(rgba(56,189,248,.018) 1px,transparent 1px),
                      linear-gradient(90deg,rgba(56,189,248,.018) 1px,transparent 1px);
    background-size: 32px 32px; animation: hGrid 10s linear infinite;
}
.nh-scan {
    position:absolute; top:0; bottom:0; width:70px; pointer-events:none;
    background:linear-gradient(90deg,transparent,rgba(56,189,248,.06),transparent);
    animation: hScan 5s linear infinite;
}

/* ── Main drone (left) ── */
.nh-left { display:flex; align-items:center; gap:22px; position:relative; }
.nh-drone-wrap {
    position:relative; width:72px; height:72px; flex-shrink:0;
    animation: hFloat 3.5s ease-in-out infinite;
}
/* Radar rings emanating */
.nh-radar-ring {
    position:absolute; top:50%; left:50%;
    width:60px; height:60px;
    border:1px solid rgba(56,189,248,.35); border-radius:50%;
    animation: hRadarRing 2.4s ease-out infinite;
    pointer-events:none;
}
.nh-radar-ring:nth-child(2) { animation-delay:0.8s; }
.nh-radar-ring:nth-child(3) { animation-delay:1.6s; }
/* Sweep arm */
.nh-sweep {
    position:absolute; top:50%; left:50%; width:28px; height:1px;
    transform-origin:left center;
    background:linear-gradient(90deg,rgba(56,189,248,.7),transparent);
    animation:hRadar 3s linear infinite;
    pointer-events:none;
}
/* Main drone body */
.nh-body {
    position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
    width:14px; height:14px; background:#38bdf8; border-radius:3px;
    box-shadow:0 0 14px rgba(56,189,248,.9),0 0 28px rgba(56,189,248,.35),0 0 48px rgba(56,189,248,.15);
    z-index:2;
}
.nh-arm {
    position:absolute; top:50%; left:50%; width:52px; height:2px;
    background:linear-gradient(90deg,transparent,rgba(56,189,248,.75),transparent);
    transform-origin:center; opacity:.8; z-index:1;
}
.nh-arm-h{transform:translate(-50%,-50%) rotate(0deg);}
.nh-arm-v{transform:translate(-50%,-50%) rotate(90deg);}
.nh-prop {
    position:absolute; width:14px; height:14px;
    border:2px solid #38bdf8; border-radius:50%; box-shadow:0 0 7px rgba(56,189,248,.5);
    z-index:2;
}
.nh-prop::before {
    content:''; position:absolute; top:50%; left:-5px; width:22px; height:2px;
    background:#38bdf8; transform:translateY(-50%);
    animation:hProp .22s linear infinite; transform-origin:center;
}
.nh-prop-tl{top:0;left:0} .nh-prop-tr{top:0;right:0}
.nh-prop-bl{bottom:0;left:0} .nh-prop-br{bottom:0;right:0}

/* ── Orbiting mini-drone ── */
.nh-orbit-container {
    position:absolute; top:50%; left:50%;
    width:0; height:0; pointer-events:none; z-index:3;
}
.nh-orbit-drone {
    position:absolute;
    animation: hOrbit 7s linear infinite;
    transform-origin:0 0;
}
.nh-orbit-body {
    width:6px; height:6px; background:#7dd3fc; border-radius:2px;
    box-shadow:0 0 6px rgba(125,211,252,.8);
    margin:-3px 0 0 -3px;
}

/* ── Titles ── */
.nh-titles { animation:hSlide .7s ease-out; }
.nh-name {
    font-size:32px; font-weight:800; letter-spacing:6px;
    background:linear-gradient(120deg,#38bdf8 0%,#7dd3fc 35%,#e2e8f0 65%,#38bdf8 100%);
    background-size:200% auto; -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    font-family:'Inter',system-ui,sans-serif; line-height:1.05;
    animation:hShim 4s linear infinite;
}
.nh-sub { font-size:10px; color:#334155; letter-spacing:.25em; margin-top:3px; }
.nh-desc { font-size:12px; color:#475569; margin-top:2px; letter-spacing:.05em; }

/* ── Right panel ── */
.nh-right { display:flex; flex-direction:column; align-items:flex-end; gap:8px; position:relative; }
.nh-status {
    display:flex; align-items:center; gap:7px; font-size:11px; font-weight:600;
    color:#34d399; letter-spacing:.1em;
    animation:hBlink 2.2s ease-in-out infinite;
}
.nh-dot {
    width:8px; height:8px; background:#34d399; border-radius:50%;
    box-shadow:0 0 8px #34d399,0 0 16px rgba(52,211,153,.4);
}
.nh-badges { display:flex; gap:5px; flex-wrap:wrap; justify-content:flex-end; }
.nh-badge {
    padding:3px 10px; border-radius:20px; font-size:10px; font-weight:600;
    letter-spacing:.06em; animation:hPulse 3s ease-in-out infinite;
    cursor:default; transition:transform 0.18s;
}
.nh-badge:hover { transform:translateY(-1px); }
.nb-cyan   { background:rgba(56,189,248,.08); border:1px solid rgba(56,189,248,.28); color:#38bdf8; }
.nb-slate  { background:rgba(100,116,139,.08); border:1px solid rgba(100,116,139,.28); color:#94a3b8; }
.nb-amber  { background:rgba(232,184,75,.07);  border:1px solid rgba(232,184,75,.25);  color:#e8b84b; }
.nb-green  { background:rgba(52,211,153,.07);  border:1px solid rgba(52,211,153,.25);  color:#34d399; }
.nh-clock  { font-size:11px; color:#334155; letter-spacing:.12em; font-family:'JetBrains Mono',monospace; }
.nh-univ   { font-size:10px; color:#1e293b; letter-spacing:.06em; }
</style>

<div class="nh">
    <div class="nh-scan"></div>
    <div class="nh-left">
        <div class="nh-drone-wrap">
            <!-- Radar rings -->
            <div class="nh-radar-ring"></div>
            <div class="nh-radar-ring"></div>
            <div class="nh-radar-ring"></div>
            <!-- Sweep arm -->
            <div class="nh-sweep"></div>
            <!-- Main drone -->
            <div class="nh-arm nh-arm-h"></div>
            <div class="nh-arm nh-arm-v"></div>
            <div class="nh-body"></div>
            <div class="nh-prop nh-prop-tl"></div>
            <div class="nh-prop nh-prop-tr"></div>
            <div class="nh-prop nh-prop-bl"></div>
            <div class="nh-prop nh-prop-br"></div>
            <!-- Orbiting mini drone -->
            <div class="nh-orbit-container">
                <div class="nh-orbit-drone"><div class="nh-orbit-body"></div></div>
            </div>
        </div>
        <div class="nh-titles">
            <div class="nh-name">NETRA</div>
            <div class="nh-sub">NEURAL ENHANCED THREAT RECOGNITION &amp; ANALYSIS</div>
            <div class="nh-desc">Drone Detection System &nbsp;·&nbsp; v1.3 &nbsp;·&nbsp; SENTINEL Agent Active</div>
        </div>
    </div>
    <div class="nh-right">
        <div class="nh-status"><div class="nh-dot"></div>SYSTEM ONLINE</div>
        <div class="nh-badges">
            <span class="nh-badge nb-cyan">RT-DETR-L</span>
            <span class="nh-badge nb-cyan">Dual Model</span>
            <span class="nh-badge nb-slate">CBAM P3/P4/P5</span>
            <span class="nh-badge nb-amber">SENTINEL · ReAct</span>
            <span class="nh-badge nb-green">Ollama · Offline</span>
        </div>
        <div class="nh-clock" id="netra-clock">──:──:──</div>
        <div class="nh-univ">GITAM University, Hyderabad &nbsp;·&nbsp; Final Year Project 2026</div>
    </div>
</div>
<script>
(function(){
    var clk = document.getElementById('netra-clock');
    if(!clk) return;
    function tick(){
        var d=new Date();
        var h=String(d.getHours()).padStart(2,'0');
        var m=String(d.getMinutes()).padStart(2,'0');
        var s=String(d.getSeconds()).padStart(2,'0');
        clk.textContent = h+':'+m+':'+s + ' LOCAL';
    }
    tick();
    setInterval(tick, 1000);
})();
</script>
"""

FOOTER = """
<div style="margin-top:12px;padding:10px 0;border-top:1px solid #1e293b;
            text-align:center;font-size:10px;letter-spacing:.1em;color:#334155;">
    NETRA v1.3 &nbsp;·&nbsp; RT-DETR-L + Dual Model + CBAM P3/P4/P5 &nbsp;·&nbsp;
    SENTINEL Agent (ReAct 5-step) &nbsp;·&nbsp; Frame Tracker &nbsp;·&nbsp;
    Fully Offline &nbsp;·&nbsp; Zero Cloud Dependency
</div>
"""


# ─────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────

with gr.Blocks(title="NETRA — Drone Detection System") as demo:

    gr.HTML(GLOBAL_CSS)
    gr.HTML(HEADER)

    with gr.Tabs():

        # ── Tab 1: Single Detection ──────────────────────
        with gr.TabItem("Single Detection"):
            with gr.Row():
                with gr.Column(scale=1):
                    # TASK 1: sources=["upload"] removes the webcam option
                    input_image = gr.Image(
                        label="Upload Image", type="numpy",
                        sources=["upload"],
                        elem_classes=["netra-img"]
                    )
                    detect_btn = gr.Button("ANALYSE", variant="primary", size="lg")

                with gr.Column(scale=2):
                    annotated_out = gr.Image(
                        label="Detection Result",
                        elem_classes=["netra-img"]
                    )

            with gr.Row():
                with gr.Column(scale=1):
                    status_box = gr.Textbox(label="Detection Status", lines=9,  interactive=False)
                    threat_box = gr.Textbox(label="Threat Level",     lines=1,  interactive=False)

                with gr.Column(scale=2):
                    reasoning_box = gr.Textbox(
                        label="SENTINEL Insight  (5-step reasoning · dual-model analysis)",
                        lines=6, interactive=False
                    )
                    alert_box   = gr.Textbox(label="Operator Alert",      lines=2, interactive=False)
                    actions_box = gr.Textbox(label="Recommended Actions", lines=4, interactive=False)

            detect_btn.click(
                fn=run_detection,
                inputs=[input_image],
                outputs=[annotated_out, status_box, reasoning_box,
                         alert_box, threat_box, actions_box]
            )

        # ── Tab 2: Frame-by-Frame Drone Tracker ─────────
        with gr.TabItem("Frame-by-Frame Drone Tracker"):
            gr.HTML("""
            <div style="font-family:'Inter','Segoe UI',system-ui,sans-serif;font-size:13px;
                        color:#64748b;padding:10px 4px 8px;letter-spacing:0.04em;">
                Upload video frames in order — both models run on each frame.
                Drones are tracked with persistent colour-coded IDs and fading motion trails.
                Use <strong style="color:#38bdf8;">◀ / ▶</strong> to step through frames.
            </div>""")
            with gr.Row():
                with gr.Column(scale=1):
                    batch_files = gr.File(
                        label="Upload Frames (JPG / PNG)",
                        file_count="multiple", file_types=["image"]
                    )
                    batch_btn = gr.Button("RUN ANALYSIS", variant="primary")

                with gr.Column(scale=2):
                    frame_viewer = gr.Image(
                        label="Frame Viewer",
                        elem_classes=["netra-img"],
                        interactive=False
                    )
                    with gr.Row():
                        prev_btn     = gr.Button("◀  PREV", variant="secondary", size="sm")
                        frame_label  = gr.Textbox(
                            value="—", interactive=False, show_label=False,
                            container=False,
                            elem_id="netra-frame-counter"
                        )
                        next_btn     = gr.Button("NEXT  ▶", variant="secondary", size="sm")

            batch_btn.click(
                fn=run_frame_batch,
                inputs=[batch_files],
                outputs=[frame_viewer, frame_label]
            )
            prev_btn.click(
                fn=lambda lbl: _nav_frame(lbl, -1),
                inputs=[frame_label],
                outputs=[frame_viewer, frame_label]
            )
            next_btn.click(
                fn=lambda lbl: _nav_frame(lbl, +1),
                inputs=[frame_label],
                outputs=[frame_viewer, frame_label]
            )

        # ── Tab 3: View Logs ─────────────────────────────
        with gr.TabItem("View Logs"):
            with gr.Row():
                with gr.Column(scale=1):
                    stats_box = gr.Textbox(
                        label="Session Statistics",
                        lines=10, interactive=False,
                        value="No data yet. Run detections first."
                    )
                    with gr.Row():
                        refresh_btn = gr.Button("Refresh",       variant="secondary")
                        clear_btn   = gr.Button("Clear History", variant="secondary")
                    export_btn = gr.Button("Export CSV", variant="secondary")
                    export_out = gr.File(label="Download Session Log (.csv)")

                with gr.Column(scale=2):
                    history_box = gr.Textbox(
                        label="Detection History Log",
                        lines=20, interactive=False,
                        value="No detections yet."
                    )

            refresh_btn.click(fn=refresh_dashboard,  inputs=[], outputs=[history_box, stats_box])
            clear_btn.click(fn=clear_session,         inputs=[], outputs=[history_box, stats_box])
            export_btn.click(fn=export_session_csv,   inputs=[], outputs=[export_out])

        # ── Tab 4: Project Details ───────────────────────
        with gr.TabItem("Project Details"):
            gr.HTML(PROJECT_DETAILS_HTML)

    gr.HTML(FOOTER)


if __name__ == "__main__":
    _device   = get_device()
    _log_path = os.path.join(tempfile.gettempdir(), "netra_alerts.txt")
    _host     = os.getenv("OLLAMA_HOST",     "http://localhost:11434")
    _model    = os.getenv("NETRA_LLM_MODEL", "llama3.2:3b")
    _port     = int(os.getenv("PORT",        "7860"))
    _share    = os.getenv("GRADIO_SHARE",    "false").lower() == "true"

    print("\n" + "="*58)
    print("  NETRA v1.3 — Drone Detection System")
    print("  Neural Enhanced Threat Recognition & Analysis")
    print("  GITAM University, Hyderabad | Final Year Project")
    print("="*58)
    print(f"  Device          : {_device.upper()}")
    print("  Inference       : Dual-model (DUT + VisioDECT)")
    print("  Agent           : SENTINEL — ReAct 5-step reasoning")
    print(f"  Escalation log  : {_log_path}")
    print(f"  LLM             : {_model} via Ollama ({_host})")
    print("  Webcam          : Disabled (upload-only)")
    print("  View Logs       : Single [S] + batch [B] source-tagged")
    print("  Ensure ollama is running: ollama serve")
    print("="*58 + "\n")
    demo.launch(share=_share, server_name="0.0.0.0", server_port=_port)
