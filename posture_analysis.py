#!/usr/bin/env python3
"""
TITAN/GAIA — Posture & Proximity Metrics Extractor
=====================================================
Reads all JSON outputs from the pipeline and extracts body language metrics
for NT and TEA groups. Outputs separate reports per group.

Usage:
    Place this script in the TCC root folder (next to 'output/') and run:
    
    python posture_analysis.py

    That's it! It auto-detects the output folder structure.

Output structure:
    output/
    ├── Neurotipico/
    │   └── relatorio_body/
    │       ├── posture_report_NT.txt
    │       └── posture_summary_NT.csv
    └── TEA/
        └── relatorio_body/
            ├── posture_report_TEA.txt
            └── posture_summary_TEA.csv
"""

import json
import os
import sys
import math
import csv
from pathlib import Path
import statistics


def find_base_dir():
    """Auto-detect the 'output' folder relative to this script's location."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Script lives in TCC/, output/ is a sibling
    base_dir = os.path.join(script_dir, "output")
    if os.path.isdir(base_dir):
        print(f"[INFO] Auto-detected output folder: {base_dir}")
        return base_dir
    # Fallback: maybe script is inside output/
    if os.path.isdir(os.path.join(script_dir, "Neurotipico")):
        print(f"[INFO] Auto-detected output folder: {script_dir}")
        return script_dir
    print(f"[ERROR] Could not find 'output/' folder near {script_dir}")
    print(f"        Place this script in the TCC root folder (next to 'output/').")
    sys.exit(1)


def find_json_files(base_dir):
    """Find all JSON files organized by group."""
    groups = {}
    for group_name in ["Neurotipico", "TEA"]:
        json_dir = os.path.join(base_dir, group_name, "json")
        if not os.path.isdir(json_dir):
            print(f"[WARN] Directory not found: {json_dir}")
            continue
        files = sorted([
            os.path.join(json_dir, f) for f in os.listdir(json_dir)
            if f.endswith(".json") and f.startswith("json_")
        ])
        groups[group_name] = files
        print(f"[INFO] Found {len(files)} JSON files in {group_name}/json/")
    return groups


def get_person(frame, role):
    """Extract person dict by role from a frame."""
    for p in frame.get("people", []):
        if p.get("role") == role:
            return p
    return None


def get_keypoint(person, name):
    """Get a specific keypoint by name."""
    for kp in person.get("keypoints", []):
        if kp["name"] == name:
            return kp
    return None


def compute_head_center(person):
    """Compute head center from nose or eyes."""
    nose = get_keypoint(person, "nose")
    if nose and nose["conf"] > 0.3:
        return (nose["x"], nose["y"])
    # fallback: midpoint of eyes
    le = get_keypoint(person, "left_eye")
    re = get_keypoint(person, "right_eye")
    if le and re and le["conf"] > 0.2 and re["conf"] > 0.2:
        return ((le["x"] + re["x"]) / 2, (le["y"] + re["y"]) / 2)
    return None


def compute_hip_center(person):
    """Compute hip center from left and right hip keypoints."""
    lh = get_keypoint(person, "left_hip")
    rh = get_keypoint(person, "right_hip")
    if lh and rh and lh["conf"] > 0.5 and rh["conf"] > 0.5:
        return ((lh["x"] + rh["x"]) / 2, (lh["y"] + rh["y"]) / 2)
    return None


def euclidean(p1, p2):
    """2D Euclidean distance."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def process_json(filepath):
    """Process a single JSON file and extract all posture/proximity metrics."""
    print(f"  Processing: {os.path.basename(filepath)}...", end=" ", flush=True)

    with open(filepath, "r", encoding="utf-8") as f:
        frames = json.load(f)

    # Metrics accumulators
    metrics = {
        "total_frames": len(frames),
        "valid_frames": 0,
        # Per-role posture
        "guardian_shoulder_tilt": [],
        "guardian_torso_height": [],
        "guardian_hand_distance": [],
        "child_shoulder_tilt": [],
        "child_torso_height": [],
        "child_hand_distance": [],
        # Proximity
        "interperson_distance_head": [],
        "interperson_distance_hip": [],
        # Movement (frame-to-frame head displacement)
        "guardian_head_displacement": [],
        "child_head_displacement": [],
    }

    prev_guardian_head = None
    prev_child_head = None

    for frame in frames:
        if not frame.get("id_locked", False):
            continue

        guardian = get_person(frame, "GUARDIAN")
        child = get_person(frame, "CHILD")

        if not guardian or not child:
            continue

        metrics["valid_frames"] += 1

        # --- Posture metrics (from pipeline's posture dict) ---
        g_posture = guardian.get("posture", {})
        c_posture = child.get("posture", {})

        if "shoulder_tilt_degrees" in g_posture:
            metrics["guardian_shoulder_tilt"].append(abs(g_posture["shoulder_tilt_degrees"]))
        if "torso_height_px" in g_posture:
            metrics["guardian_torso_height"].append(g_posture["torso_height_px"])
        if "hand_distance_px" in g_posture:
            metrics["guardian_hand_distance"].append(g_posture["hand_distance_px"])

        if "shoulder_tilt_degrees" in c_posture:
            metrics["child_shoulder_tilt"].append(abs(c_posture["shoulder_tilt_degrees"]))
        if "torso_height_px" in c_posture:
            metrics["child_torso_height"].append(c_posture["torso_height_px"])
        if "hand_distance_px" in c_posture:
            metrics["child_hand_distance"].append(c_posture["hand_distance_px"])

        # --- Inter-person distance ---
        g_head = compute_head_center(guardian)
        c_head = compute_head_center(child)
        if g_head and c_head:
            metrics["interperson_distance_head"].append(euclidean(g_head, c_head))

        g_hip = compute_hip_center(guardian)
        c_hip = compute_hip_center(child)
        if g_hip and c_hip:
            metrics["interperson_distance_hip"].append(euclidean(g_hip, c_hip))

        # --- Frame-to-frame movement (head displacement as proxy for activity) ---
        if g_head:
            if prev_guardian_head is not None:
                metrics["guardian_head_displacement"].append(euclidean(g_head, prev_guardian_head))
            prev_guardian_head = g_head

        if c_head:
            if prev_child_head is not None:
                metrics["child_head_displacement"].append(euclidean(c_head, prev_child_head))
            prev_child_head = c_head

    print(f"{metrics['valid_frames']}/{metrics['total_frames']} valid frames")
    return metrics


def summarize(values):
    """Return mean, std, median, min, max for a list of values."""
    if not values:
        return {"mean": None, "std": None, "median": None, "min": None, "max": None, "n": 0}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def generate_session_summary(metrics):
    """Generate a summary dict from raw metrics."""
    summary = {}
    for key in [
        "guardian_shoulder_tilt", "guardian_torso_height", "guardian_hand_distance",
        "child_shoulder_tilt", "child_torso_height", "child_hand_distance",
        "interperson_distance_head", "interperson_distance_hip",
        "guardian_head_displacement", "child_head_displacement",
    ]:
        summary[key] = summarize(metrics[key])
    summary["valid_frames"] = metrics["valid_frames"]
    summary["total_frames"] = metrics["total_frames"]
    return summary


def format_val(v, decimals=1):
    """Format a value for display."""
    if v is None:
        return "—"
    return f"{v:.{decimals}f}"


def write_group_report(out_path, group_name, sessions):
    """Write a text report for a single group."""
    with open(out_path, "w", encoding="utf-8") as rpt:
        rpt.write("=" * 80 + "\n")
        rpt.write(f"  TITAN/GAIA — POSTURE & PROXIMITY REPORT — {group_name.upper()}\n")
        rpt.write("=" * 80 + "\n\n")

        for session_name, s in sessions.items():
            rpt.write(f"  Session: {session_name}\n")
            rpt.write(f"    Valid frames: {s['valid_frames']}/{s['total_frames']}\n\n")

            rpt.write(f"    GUARDIAN posture:\n")
            rpt.write(f"      Shoulder tilt (abs°):  mean={format_val(s['guardian_shoulder_tilt']['mean'])}  std={format_val(s['guardian_shoulder_tilt']['std'])}\n")
            rpt.write(f"      Torso height (px):     mean={format_val(s['guardian_torso_height']['mean'])}  std={format_val(s['guardian_torso_height']['std'])}\n")
            rpt.write(f"      Hand distance (px):    mean={format_val(s['guardian_hand_distance']['mean'])}  std={format_val(s['guardian_hand_distance']['std'])}\n\n")

            rpt.write(f"    CHILD posture:\n")
            rpt.write(f"      Shoulder tilt (abs°):  mean={format_val(s['child_shoulder_tilt']['mean'])}  std={format_val(s['child_shoulder_tilt']['std'])}\n")
            rpt.write(f"      Torso height (px):     mean={format_val(s['child_torso_height']['mean'])}  std={format_val(s['child_torso_height']['std'])}\n")
            rpt.write(f"      Hand distance (px):    mean={format_val(s['child_hand_distance']['mean'])}  std={format_val(s['child_hand_distance']['std'])}\n\n")

            rpt.write(f"    PROXIMITY:\n")
            rpt.write(f"      Head-to-head (px):     mean={format_val(s['interperson_distance_head']['mean'])}  std={format_val(s['interperson_distance_head']['std'])}\n")
            rpt.write(f"      Hip-to-hip (px):       mean={format_val(s['interperson_distance_hip']['mean'])}  std={format_val(s['interperson_distance_hip']['std'])}\n\n")

            rpt.write(f"    MOVEMENT (frame-to-frame head displacement):\n")
            rpt.write(f"      Guardian (px/frame):   mean={format_val(s['guardian_head_displacement']['mean'])}  std={format_val(s['guardian_head_displacement']['std'])}\n")
            rpt.write(f"      Child (px/frame):      mean={format_val(s['child_head_displacement']['mean'])}  std={format_val(s['child_head_displacement']['std'])}\n")
            rpt.write(f"\n{'·'*60}\n\n")

        # Group averages
        metric_keys = [
            ("guardian_shoulder_tilt", "Guardian Shoulder Tilt (abs°)"),
            ("child_shoulder_tilt", "Child Shoulder Tilt (abs°)"),
            ("guardian_hand_distance", "Guardian Hand Distance (px)"),
            ("child_hand_distance", "Child Hand Distance (px)"),
            ("interperson_distance_head", "Head-to-Head Distance (px)"),
            ("interperson_distance_hip", "Hip-to-Hip Distance (px)"),
            ("guardian_head_displacement", "Guardian Movement (px/frame)"),
            ("child_head_displacement", "Child Movement (px/frame)"),
        ]

        rpt.write(f"\n  GROUP AVERAGES ({group_name.upper()}):\n")
        rpt.write(f"  {'─'*60}\n")
        for key, label in metric_keys:
            session_means = [s[key]["mean"] for s in sessions.values() if s[key]["mean"] is not None]
            if session_means:
                group_mean = statistics.mean(session_means)
                group_std = statistics.stdev(session_means) if len(session_means) > 1 else 0.0
                rpt.write(f"    {label:40s}  mean={group_mean:.1f}  std={group_std:.1f}  (n={len(session_means)})\n")
        rpt.write("\n")


def write_group_csv(out_path, group_name, sessions):
    """Write a CSV summary for a single group."""
    with open(out_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow([
            "Group", "Session", "ValidFrames",
            "G_ShoulderTilt_mean", "G_ShoulderTilt_std",
            "C_ShoulderTilt_mean", "C_ShoulderTilt_std",
            "G_HandDist_mean", "G_HandDist_std",
            "C_HandDist_mean", "C_HandDist_std",
            "HeadDist_mean", "HeadDist_std",
            "HipDist_mean", "HipDist_std",
            "G_Movement_mean", "G_Movement_std",
            "C_Movement_mean", "C_Movement_std",
        ])
        for session_name, s in sessions.items():
            writer.writerow([
                group_name, session_name, s["valid_frames"],
                format_val(s["guardian_shoulder_tilt"]["mean"]),
                format_val(s["guardian_shoulder_tilt"]["std"]),
                format_val(s["child_shoulder_tilt"]["mean"]),
                format_val(s["child_shoulder_tilt"]["std"]),
                format_val(s["guardian_hand_distance"]["mean"]),
                format_val(s["guardian_hand_distance"]["std"]),
                format_val(s["child_hand_distance"]["mean"]),
                format_val(s["child_hand_distance"]["std"]),
                format_val(s["interperson_distance_head"]["mean"]),
                format_val(s["interperson_distance_head"]["std"]),
                format_val(s["interperson_distance_hip"]["mean"]),
                format_val(s["interperson_distance_hip"]["std"]),
                format_val(s["guardian_head_displacement"]["mean"]),
                format_val(s["guardian_head_displacement"]["std"]),
                format_val(s["child_head_displacement"]["mean"]),
                format_val(s["child_head_displacement"]["std"]),
            ])


def main():
    base_dir = find_base_dir()

    print("=" * 70)
    print("  TITAN/GAIA — Posture & Proximity Metrics Extractor")
    print("=" * 70)

    groups = find_json_files(base_dir)
    if not groups:
        print("[ERROR] No JSON files found. Check --base_dir path.")
        sys.exit(1)

    all_results = {}

    for group_name, files in groups.items():
        print(f"\n{'='*70}")
        print(f"  Processing group: {group_name} ({len(files)} files)")
        print(f"{'='*70}")

        group_results = {}
        for filepath in files:
            session_name = os.path.basename(filepath).replace("json_", "").replace(".json", "")
            raw = process_json(filepath)
            group_results[session_name] = generate_session_summary(raw)

        all_results[group_name] = group_results

        # ── Output into relatorio_body inside each group folder ──
        body_dir = os.path.join(base_dir, group_name, "relatorio_body")
        os.makedirs(body_dir, exist_ok=True)

        suffix = "NT" if "neuro" in group_name.lower() else "TEA"
        report_path = os.path.join(body_dir, f"posture_report_{suffix}.txt")
        csv_path = os.path.join(body_dir, f"posture_summary_{suffix}.csv")

        write_group_report(report_path, group_name, group_results)
        write_group_csv(csv_path, group_name, group_results)

        print(f"\n  [OK] Report: {report_path}")
        print(f"  [OK] CSV:    {csv_path}")

    print(f"\n{'='*70}")
    print(f"  DONE! Output structure:")
    for group_name in all_results:
        suffix = "NT" if "neuro" in group_name.lower() else "TEA"
        print(f"    {group_name}/relatorio_body/posture_report_{suffix}.txt")
        print(f"    {group_name}/relatorio_body/posture_summary_{suffix}.csv")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()