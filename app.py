"""
SENTINEL v3.0 — Drone Detection System
RT-DETR-L + Multi-Scale CBAM + Multi-Agent LLM Layer
Niketh | GITAM Deemed University | B.Tech CSE (AI/ML) | 2026

Two AI Agents:
  1. SENTINEL Analyst  — per-detection, ReAct pattern, threat trajectory
  2. Mission Summariser — end-of-session, full session intelligence report

Run:
    1. ollama serve       (separate terminal)
    2. python3 app.py     (from ~/droneui/)
"""

import cv2
import json
import time
import torch
import numpy as np
import gradio as gr
import ollama
from PIL import Image
from datetime import datetime
from ultralytics import RTDETR
import torch.nn as nn

# ─────────────────────────────────────────────
# CBAM — must be defined before loading best.pt
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


class ConvWithCBAM(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.conv = conv
        self.cbam = CBAM(conv.conv.out_channels)

    def forward(self, x):
        return self.cbam(self.conv(x))


# ─────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────

MODEL_PATHS = {
    "DUT Anti-UAV (Occlusion)":    "best_dut.pt",
    "VisioDECT (Multi-Scenario)":  "best_visiodect.pt"
}
loaded_models = {}

def get_model(model_name):
    if model_name not in loaded_models:
        path = MODEL_PATHS[model_name]
        try:
            model = RTDETR(path)
            model.to("mps" if torch.backends.mps.is_available() else "cpu")
            loaded_models[model_name] = model
        except FileNotFoundError:
            return None, f"Model file not found: {path}"
        except Exception as e:
            return None, str(e)
    return loaded_models[model_name], None


# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────

session_history = []   # max 20 entries

def add_to_history(detected, confidence, threat, trajectory, quadrant, model_name, agent_confidence):
    entry = {
        "time":             datetime.now().strftime("%H:%M:%S"),
        "detected":         detected,
        "confidence":       round(confidence, 3),
        "threat":           threat,
        "trajectory":       trajectory,
        "quadrant":         quadrant,
        "model":            model_name,
        "agent_confidence": agent_confidence
    }
    session_history.append(entry)
    if len(session_history) > 20:
        session_history.pop(0)

def get_history_text():
    if not session_history:
        return "No detections yet."
    lines = []
    for e in reversed(session_history):
        icon = "🔴" if e["detected"] else "🟢"
        lines.append(
            f"{icon} [{e['time']}]  {'DRONE' if e['detected'] else 'CLEAR'}"
            f"  |  Conf: {e['confidence']:.2f}"
            f"  |  Threat: {e['threat']}"
            f"  |  Trajectory: {e['trajectory']}"
            f"  |  Agent certainty: {e['agent_confidence']}%"
        )
    return "\n".join(lines)

def get_stats_text():
    if not session_history:
        return "No data yet. Run detections first."
    total = len(session_history)
    dets  = sum(1 for e in session_history if e["detected"])
    avg_conf = (sum(e["confidence"] for e in session_history if e["detected"]) / dets
                if dets > 0 else 0.0)
    threat_counts = {}
    for e in session_history:
        threat_counts[e["threat"]] = threat_counts.get(e["threat"], 0) + 1
    dominant = max(threat_counts, key=threat_counts.get)
    traj_counts = {}
    for e in session_history:
        traj_counts[e["trajectory"]] = traj_counts.get(e["trajectory"], 0) + 1
    dominant_traj = max(traj_counts, key=traj_counts.get)
    return (
        f"Session Statistics\n"
        f"──────────────────────\n"
        f"Total scans       : {total}\n"
        f"Detections        : {dets}\n"
        f"Detection rate    : {(dets/total*100):.1f}%\n"
        f"Avg confidence    : {avg_conf:.1%}\n"
        f"Dominant threat   : {dominant}\n"
        f"Dominant trajectory: {dominant_traj}"
    )

def export_session_json():
    if not session_history:
        return "No session data to export."
    export = {
        "session_exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_scans": len(session_history),
        "detections": sum(1 for e in session_history if e["detected"]),
        "log": session_history
    }
    path = f"/tmp/sentinel_session_{datetime.now().strftime('%H%M%S')}.json"
    with open(path, "w") as f:
        json.dump(export, f, indent=2)
    return path

def clear_session():
    session_history.clear()
    return "No detections yet.", "No data yet. Run detections first."

def refresh_dashboard():
    return get_history_text(), get_stats_text()


# ─────────────────────────────────────────────
# Rule-based threat level
# ─────────────────────────────────────────────

def compute_threat_level(detected, confidence, history):
    if not detected:
        return "NONE"
    recent = sum(1 for e in history[-5:] if e["detected"])
    if confidence >= 0.75 or recent >= 4:
        return "CRITICAL"
    elif confidence >= 0.55 or recent >= 3:
        return "HIGH"
    elif confidence >= 0.35 or recent >= 2:
        return "MEDIUM"
    return "LOW"

def compute_trajectory(history):
    """Assess threat trajectory from recent confidence trend."""
    recent = [e for e in history[-5:] if e["detected"]]
    if len(recent) < 2:
        return "INSUFFICIENT DATA"
    confs = [e["confidence"] for e in recent]
    delta = confs[-1] - confs[0]
    if delta > 0.08:
        return "ESCALATING"
    elif delta < -0.08:
        return "DE-ESCALATING"
    return "STABLE"


# ─────────────────────────────────────────────
# AGENT 1 — SENTINEL Analyst (per-detection)
# ─────────────────────────────────────────────

SENTINEL_SYSTEM = """You are SENTINEL, an autonomous AI surveillance analyst embedded in a real-time drone detection system.
Your role is to analyse detection data, reason about threat context, and provide structured operational intelligence.

You reason in steps before concluding:
STEP 1 - ASSESS: What do the current detection metrics indicate?
STEP 2 - CONTEXTUALISE: What does session history and threat trajectory suggest?
STEP 3 - DECIDE: What is the actual threat level and trajectory direction?
STEP 4 - ACT: What specific actions must the operator take right now?
STEP 5 - SELF-EVALUATE: How confident are you in this assessment (0-100)?

Respond ONLY in valid JSON. No text outside the JSON.
{
  "threat_level": "NONE or LOW or MEDIUM or HIGH or CRITICAL",
  "trajectory": "ESCALATING or STABLE or DE-ESCALATING or INSUFFICIENT DATA",
  "reasoning": "2-3 sentence chain-of-thought across all 5 steps",
  "alert": "1-2 sentence direct operational alert",
  "actions": ["specific action 1", "specific action 2", "specific action 3"],
  "escalate": true or false,
  "agent_confidence": 0-100
}"""

def run_sentinel_agent(detected, confidence, threat_level, trajectory, quadrant, model_name, ms):
    if session_history:
        recent = session_history[-5:]
        history_str = "\n".join([
            f"  [{e['time']}] {'DETECTED' if e['detected'] else 'CLEAR'} "
            f"conf={e['confidence']:.2f} threat={e['threat']} traj={e['trajectory']}"
            for e in recent
        ])
        det_count = sum(1 for e in session_history if e["detected"])
        history_ctx = (
            f"SESSION HISTORY (last {len(recent)} of {len(session_history)} scans):\n"
            f"{history_str}\n"
            f"Total detections this session: {det_count}/{len(session_history)}"
        )
    else:
        history_ctx = "SESSION HISTORY: No prior scans this session."

    user_msg = f"""CURRENT SCAN:
- Status: {'DRONE DETECTED' if detected else 'NO DRONE DETECTED'}
- Confidence: {confidence:.1%}
- Rule-based threat: {threat_level}
- Confidence trajectory: {trajectory}
- Image quadrant: {quadrant}
- Model: {model_name}
- Inference time: {ms:.0f}ms

{history_ctx}

Apply your 5-step reasoning. Respond in JSON only."""

    try:
        resp = ollama.chat(
            model="llama3.2:3b",
            messages=[
                {"role": "system", "content": SENTINEL_SYSTEM},
                {"role": "user",   "content": user_msg}
            ],
            options={"temperature": 0.3}
        )
        raw = resp["message"]["content"].strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
        return (
            result.get("threat_level",     threat_level),
            result.get("trajectory",       trajectory),
            result.get("reasoning",        ""),
            result.get("alert",            "Detection processed."),
            result.get("actions",          ["Monitor area", "Log event", "Continue scanning"]),
            result.get("escalate",         False),
            result.get("agent_confidence", 50),
            json.dumps(result, indent=2)
        )
    except json.JSONDecodeError:
        fallback = {
            "threat_level": threat_level, "trajectory": trajectory,
            "reasoning": "JSON parse failed — raw response returned.",
            "alert": f"{'DRONE DETECTED' if detected else 'CLEAR'} — Conf: {confidence:.1%}",
            "actions": ["Monitor area", "Log detection", "Continue scanning"],
            "escalate": confidence > 0.6, "agent_confidence": 40
        }
        return (threat_level, trajectory, fallback["reasoning"], fallback["alert"],
                fallback["actions"], fallback["escalate"], 40, json.dumps(fallback, indent=2))
    except Exception as e:
        return (threat_level, trajectory, "SENTINEL offline.", f"[Offline] {str(e)}",
                ["Check ollama serve"], False, 0, "{}")


# ─────────────────────────────────────────────
# AGENT 2 — Mission Summariser (end-of-session)
# ─────────────────────────────────────────────

SUMMARISER_SYSTEM = """You are the SENTINEL Mission Summariser, an AI analyst that reviews completed surveillance sessions.
Your role is to analyse the full session log and produce a structured mission intelligence report.

You assess:
- Overall threat level for the session
- Threat pattern and trajectory over time
- Spatial patterns (which quadrants had most activity)
- Confidence trend (improving, degrading, or stable detections)
- Recommended follow-up actions for command

Respond ONLY in valid JSON. No text outside the JSON.
{
  "session_threat_assessment": "CLEAR or LOW or MODERATE or HIGH or CRITICAL",
  "pattern_analysis": "2-3 sentences describing the threat pattern observed",
  "spatial_summary": "1-2 sentences on quadrant distribution",
  "confidence_trend": "IMPROVING or STABLE or DEGRADING",
  "key_findings": ["finding 1", "finding 2", "finding 3"],
  "recommended_followup": ["action 1", "action 2", "action 3"],
  "mission_summary": "3-4 sentence executive summary for command"
}"""

def run_mission_summariser():
    if not session_history:
        return "No session data to summarise. Run some detections first."

    total   = len(session_history)
    dets    = sum(1 for e in session_history if e["detected"])
    avg_conf = (sum(e["confidence"] for e in session_history if e["detected"]) / dets
                if dets > 0 else 0.0)

    log_str = "\n".join([
        f"  [{e['time']}] {'DETECTED' if e['detected'] else 'CLEAR'} "
        f"conf={e['confidence']:.2f} threat={e['threat']} "
        f"trajectory={e['trajectory']} quadrant={e['quadrant']}"
        for e in session_history
    ])

    user_msg = f"""FULL SESSION LOG:
Total scans: {total}
Total detections: {dets}
Detection rate: {(dets/total*100):.1f}%
Average confidence (detections only): {avg_conf:.1%}

Chronological log:
{log_str}

Produce the mission intelligence report in JSON only."""

    try:
        resp = ollama.chat(
            model="llama3.2:3b",
            messages=[
                {"role": "system", "content": SUMMARISER_SYSTEM},
                {"role": "user",   "content": user_msg}
            ],
            options={"temperature": 0.2}
        )
        raw = resp["message"]["content"].strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())

        report = (
            f"MISSION INTELLIGENCE REPORT\n"
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'='*50}\n\n"
            f"SESSION THREAT ASSESSMENT: {result.get('session_threat_assessment','N/A')}\n"
            f"CONFIDENCE TREND: {result.get('confidence_trend','N/A')}\n\n"
            f"PATTERN ANALYSIS:\n{result.get('pattern_analysis','')}\n\n"
            f"SPATIAL SUMMARY:\n{result.get('spatial_summary','')}\n\n"
            f"KEY FINDINGS:\n" +
            "\n".join([f"  • {f}" for f in result.get("key_findings", [])]) +
            f"\n\nRECOMMENDED FOLLOW-UP:\n" +
            "\n".join([f"  >> {a}" for a in result.get("recommended_followup", [])]) +
            f"\n\nEXECUTIVE SUMMARY:\n{result.get('mission_summary','')}\n"
        )
        return report

    except Exception as e:
        return f"Mission Summariser error: {str(e)}\n\nRaw session: {total} scans, {dets} detections."


# ─────────────────────────────────────────────
# Quadrant helper
# ─────────────────────────────────────────────

def get_quadrant(cx, cy, w, h):
    return f"{'upper' if cy < h/2 else 'lower'}-{'left' if cx < w/2 else 'right'}"


# ─────────────────────────────────────────────
# Core inference
# ─────────────────────────────────────────────

def run_detection(image, conf_threshold, model_name):
    if image is None:
        return (None, None, "No image uploaded.", "", "", "NONE",
                "INSUFFICIENT DATA", "", "", "{}", "0%", "")

    model, err = get_model(model_name)
    if err:
        return (None, None, err, "", "", "NONE",
                "INSUFFICIENT DATA", "", "", "{}", "0%", "")

    img_h, img_w  = image.shape[:2]
    original_pil  = Image.fromarray(cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB))
    annotated     = image.copy()

    start   = time.time()
    results = model.predict(source=image, conf=conf_threshold, imgsz=640, verbose=False)
    ms      = (time.time() - start) * 1000

    boxes    = results[0].boxes
    detected = boxes is not None and len(boxes) > 0
    best_conf, best_quad = 0.0, "centre"

    if detected:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 80), 2)
            label = f"Drone {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (x1, y1-th-8), (x1+tw+4, y1), (0, 255, 80), -1)
            cv2.putText(annotated, label, (x1+2, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
            if conf > best_conf:
                best_conf = conf
                best_quad = get_quadrant((x1+x2)//2, (y1+y2)//2, img_w, img_h)
        cv2.rectangle(annotated, (0,0), (img_w, 38), (0, 60, 0), -1)
        cv2.putText(annotated, f"DRONE DETECTED  |  Conf: {best_conf:.2f}  |  {ms:.0f}ms",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 80), 2)
    else:
        cv2.rectangle(annotated, (0,0), (img_w, 38), (10, 10, 40), -1)
        cv2.putText(annotated, f"NO DRONE DETECTED  |  {ms:.0f}ms",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 180, 255), 2)

    threat_level = compute_threat_level(detected, best_conf, session_history)
    trajectory   = compute_trajectory(session_history)

    (agent_threat, agent_traj, reasoning, alert,
     actions, escalate, agent_conf, raw_json) = run_sentinel_agent(
        detected, best_conf, threat_level, trajectory, best_quad, model_name, ms
    )

    add_to_history(detected, best_conf, agent_threat, agent_traj, best_quad, model_name, agent_conf)

    annotated_pil = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

    status = (
        f"{'DRONE DETECTED' if detected else 'AREA CLEAR'}\n"
        f"Confidence  : {best_conf:.1%}\n"
        f"Threat      : {agent_threat}\n"
        f"Trajectory  : {agent_traj}\n"
        f"Quadrant    : {best_quad}\n"
        f"Detections  : {len(boxes) if detected else 0}\n"
        f"Inference   : {ms:.0f}ms\n"
        f"Escalate    : {'YES ⚠️' if escalate else 'No'}"
    )
    actions_text    = "\n".join([f">> {a}" for a in actions])
    agent_conf_text = f"{agent_conf}% certainty"

    return (
        original_pil, annotated_pil, status, reasoning, alert,
        agent_threat, agent_traj, actions_text, raw_json,
        agent_conf_text, get_history_text()
    )


def run_batch(files, conf_threshold, model_name):
    if not files:
        return "No files uploaded."
    results = []
    for f in files:
        img     = np.array(Image.open(f.name).convert("RGB"))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out     = run_detection(img_bgr, conf_threshold, model_name)
        status, alert, threat, traj, agent_conf = out[2], out[4], out[5], out[6], out[9]
        fname   = f.name.split("/")[-1]
        results.append(
            f"FILE: {fname}\n{status}\n"
            f"Trajectory : {traj}\n"
            f"Agent cert : {agent_conf}\n"
            f"ALERT: {alert}\n{'─'*44}"
        )
    return "\n\n".join(results)


# ─────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────

with gr.Blocks(title="SENTINEL v3.0 — Drone Detection") as demo:

    gr.Markdown("""
# SENTINEL v3.0 — Drone Detection System
**RT-DETR-L + Multi-Scale CBAM (P3/P4/P5) | Multi-Agent LLM Layer | Fully Offline**
*Agent 1: SENTINEL Analyst (per-detection ReAct) · Agent 2: Mission Summariser (end-of-session)*
Niketh · GITAM Deemed University · B.Tech CSE (AI/ML) 2026
""")

    with gr.Tabs():

        # ── Tab 1: Single Detection ─────────────────────
        with gr.TabItem("Single Detection"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_image    = gr.Image(label="Upload Image", type="numpy", height=280)
                    model_selector = gr.Dropdown(
                        choices=list(MODEL_PATHS.keys()),
                        value="DUT Anti-UAV (Occlusion)", label="Model"
                    )
                    conf_slider = gr.Slider(
                        minimum=0.1, maximum=0.7, step=0.05,
                        value=0.25, label="Confidence Threshold",
                        info="Lower = catches occluded drones · Higher = fewer false positives"
                    )
                    detect_btn = gr.Button("ANALYSE", variant="primary", size="lg")

                with gr.Column(scale=2):
                    with gr.Row():
                        original_out  = gr.Image(label="Original",        height=280)
                        annotated_out = gr.Image(label="Detection Result", height=280)

            with gr.Row():
                with gr.Column(scale=1):
                    status_box   = gr.Textbox(label="Detection Status",    lines=9,  interactive=False)
                    threat_box   = gr.Textbox(label="Threat Level",        lines=1,  interactive=False)
                    traj_box     = gr.Textbox(label="Threat Trajectory",   lines=1,  interactive=False)
                    certaint_box = gr.Textbox(label="Agent Certainty",     lines=1,  interactive=False)

                with gr.Column(scale=2):
                    reasoning_box = gr.Textbox(
                        label="SENTINEL Chain-of-Thought Reasoning",
                        lines=4, interactive=False
                    )
                    alert_box   = gr.Textbox(label="Operator Alert",       lines=2,  interactive=False)
                    actions_box = gr.Textbox(label="Recommended Actions",  lines=4,  interactive=False)
                    raw_json_box = gr.Textbox(
                        label="SENTINEL Raw JSON Output (Agent Intelligence)",
                        lines=6, interactive=False
                    )

            # Hidden output for history
            hidden_history = gr.Textbox(visible=False)

            detect_btn.click(
                fn=run_detection,
                inputs=[input_image, conf_slider, model_selector],
                outputs=[
                    original_out, annotated_out, status_box,
                    reasoning_box, alert_box, threat_box, traj_box,
                    actions_box, raw_json_box, certaint_box, hidden_history
                ]
            )

        # ── Tab 2: Batch Processing ─────────────────────
        with gr.TabItem("Batch Processing"):
            gr.Markdown("### Upload multiple images — each processed through full SENTINEL pipeline")
            with gr.Row():
                with gr.Column(scale=1):
                    batch_files = gr.File(
                        label="Upload Images (JPG/PNG)",
                        file_count="multiple", file_types=["image"]
                    )
                    batch_model = gr.Dropdown(
                        choices=list(MODEL_PATHS.keys()),
                        value="DUT Anti-UAV (Occlusion)", label="Model"
                    )
                    batch_conf = gr.Slider(
                        minimum=0.1, maximum=0.7, step=0.05,
                        value=0.25, label="Confidence Threshold"
                    )
                    batch_btn = gr.Button("RUN BATCH", variant="primary")

                with gr.Column(scale=2):
                    batch_output = gr.Textbox(label="Batch Results", lines=28, interactive=False)

            batch_btn.click(
                fn=run_batch,
                inputs=[batch_files, batch_conf, batch_model],
                outputs=[batch_output]
            )

        # ── Tab 3: Session Dashboard ────────────────────
        with gr.TabItem("Session Dashboard"):
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
                    export_btn    = gr.Button("Export Session JSON", variant="secondary")
                    export_out    = gr.File(label="Download Session Log")

                with gr.Column(scale=2):
                    history_box = gr.Textbox(
                        label="Detection History Log (Most Recent First)",
                        lines=20, interactive=False,
                        value="No detections yet."
                    )

            refresh_btn.click(fn=refresh_dashboard,  inputs=[], outputs=[history_box, stats_box])
            clear_btn.click(fn=clear_session,         inputs=[], outputs=[history_box, stats_box])
            export_btn.click(fn=export_session_json,  inputs=[], outputs=[export_out])

        # ── Tab 4: Mission Summariser (Agent 2) ─────────
        with gr.TabItem("Mission Report (Agent 2)"):
            gr.Markdown("""
### Mission Summariser — SENTINEL Agent 2
Analyses the **complete session log** and produces a structured mission intelligence report.
Run this at the end of a scanning session. Requires at least 3 detections for meaningful output.
""")
            summarise_btn  = gr.Button("GENERATE MISSION REPORT", variant="primary", size="lg")
            mission_output = gr.Textbox(
                label="Mission Intelligence Report",
                lines=30, interactive=False,
                value="Click 'Generate Mission Report' after running detections."
            )
            summarise_btn.click(fn=run_mission_summariser, inputs=[], outputs=[mission_output])

    gr.Markdown("""
---
**SENTINEL v3.0** · RT-DETR-L + Multi-Scale CBAM (P3/P4/P5) · Two AI Agents: SENTINEL Analyst + Mission Summariser
llama3.2:3b via Ollama · ReAct pattern · Session memory · Fully offline · Zero cloud dependency
""")


if __name__ == "__main__":
    print("[SENTINEL v3.0] Starting...")
    print("[SENTINEL v3.0] Agent 1: SENTINEL Analyst — per-detection ReAct")
    print("[SENTINEL v3.0] Agent 2: Mission Summariser — end-of-session report")
    print("[SENTINEL v3.0] Ensure ollama is running (ollama serve)")
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
