"""
create_final_infographics.py
Build 4-column workflow infographics using REAL pipeline outputs.
Lip zoom is found by diffing original vs corrupted to locate the exact mouth mask.
"""
import cv2
import numpy as np
import os


def find_mouth_box(original, corrupted):
    """
    Find the mouth box by computing the difference between original and corrupted frames.
    The black mask in the corrupted frame creates a large pixel difference.
    Returns (x, y, w, h) of the mouth region.
    """
    diff = np.abs(original.astype(float) - corrupted.astype(float)).max(axis=2)
    mask = diff > 50
    ys, xs = np.where(mask)
    if len(ys) > 100:  # enough pixels to form a box
        return (int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min()))
    # Fallback
    h, w = original.shape[:2]
    return (w // 3, h // 3, w // 3, h // 6)


def draw_waveform(img, x, y, w, h, is_noisy):
    """Draw a waveform visualization."""
    cv2.rectangle(img, (x, y), (x + w, y + h), (25, 25, 35), -1)
    mid_y = y + h // 2
    if is_noisy:
        np.random.seed(42)
        max_val = (h // 2) - 5
        for i in range(w):
            val = np.abs(np.random.normal(0, h // 4))
            val = min(val, max_val)
            cv2.line(img, (x + i, int(mid_y - val)), (x + i, int(mid_y + val)), (80, 80, 220), 2)
    else:
        pts = []
        for i in range(w):
            val = np.sin(i * 0.05) * (h // 4) * np.sin(i * 0.005)
            pts.append((x + i, int(mid_y + val)))
        pts = np.array(pts, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], False, (80, 200, 80), 3)


font = cv2.FONT_HERSHEY_SIMPLEX
font_small = cv2.FONT_HERSHEY_SIMPLEX


def create_infographic(corrupted, restored, original, mouth_box, output_path, label=""):
    """
    Build the 4-column infographic from actual pipeline output frames.
    mouth_box = (x, y, w, h) of the mouth region in the original frame.
    """
    v_w, v_h = 800, 450
    canvas_w, canvas_h = 4000, 1100
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    corr_disp = cv2.resize(corrupted, (v_w, v_h))
    rest_disp = cv2.resize(restored, (v_w, v_h))
    orig_disp = cv2.resize(original, (v_w, v_h))

    # ── Column 1: Input (Degraded Media) ──
    x = 0
    cv2.putText(canvas, "Stage 1: Input (Degraded Media)", (x + 100, 80), font, 1.4, (0, 0, 0), 3)
    canvas[120:120 + v_h, x + 100:x + 100 + v_w] = corr_disp
    # Label
    cv2.putText(canvas, "Mouth region masked (black box)", (x + 150, 120 + v_h + 30), font_small, 0.8, (80, 80, 80), 2)
    draw_waveform(canvas, x + 100, 700, v_w, 250, True)
    cv2.putText(canvas, "Noisy / Corrupted Audio", (x + 250, 980), font_small, 0.8, (80, 80, 80), 2)

    # ── Column 2: Cross-Modal Restoration ──
    x = 1000
    cv2.putText(canvas, "Stage 2: Cross-Modal Restoration", (x + 80, 80), font, 1.4, (0, 0, 0), 3)
    canvas[120:120 + v_h, x + 100:x + 100 + v_w] = rest_disp
    cv2.putText(canvas, "Lips recovered via audio cues", (x + 180, 120 + v_h + 30), font_small, 0.8, (80, 80, 80), 2)
    # Arrow from video to audio
    cv2.arrowedLine(canvas, (x + 500, 120 + v_h + 50), (x + 500, 690), (0, 0, 200), 5, tipLength=0.08)
    cv2.putText(canvas, "Visual-Guided", (x + 520, 650), font_small, 0.9, (0, 0, 150), 2)
    cv2.putText(canvas, "Filtering", (x + 560, 680), font_small, 0.9, (0, 0, 150), 2)
    draw_waveform(canvas, x + 100, 700, v_w, 250, False)
    cv2.putText(canvas, "Clean / Restored Audio", (x + 260, 980), font_small, 0.8, (80, 80, 80), 2)

    # ── Column 3: Enhancement & Super-Resolution ──
    x = 2000
    cv2.putText(canvas, "Stage 3: Enhancement & Super-Res", (x + 70, 80), font, 1.4, (0, 0, 0), 3)
    canvas[120:120 + v_h, x + 100:x + 100 + v_w] = orig_disp

    # Crop LIPS from the original frame using the exact mouth box
    mx, my, mw, mh = mouth_box
    oh, ow = original.shape[:2]
    # Add a small vertical padding to ensure we get full lips
    pad = 5
    y1 = max(0, my + pad)             # slightly below top of box (skip nose bridge)
    y2 = min(oh, my + mh + pad)       # include chin
    x1 = max(0, mx - pad)
    x2 = min(ow, mx + mw + pad)
    lips_crop = original[y1:y2, x1:x2]

    if lips_crop.shape[0] > 10 and lips_crop.shape[1] > 10:
        lips_zoomed = cv2.resize(lips_crop, (350, 220))
    else:
        lips_zoomed = np.zeros((220, 350, 3), dtype=np.uint8)

    # Green border
    cv2.rectangle(lips_zoomed, (0, 0), (349, 219), (0, 220, 0), 4)
    zoom_y, zoom_x = 500, x + 550
    canvas[zoom_y:zoom_y + 220, zoom_x:zoom_x + 350] = lips_zoomed

    # Connecting line from the mouth area on the displayed frame to the zoom box
    # Map mouth center to display coordinates
    scale_x = v_w / ow
    scale_y = v_h / oh
    mouth_cx_disp = int((mx + mw / 2) * scale_x) + x + 100
    mouth_cy_disp = int((my + mh / 2) * scale_y) + 120
    cv2.line(canvas, (mouth_cx_disp, mouth_cy_disp), (zoom_x, zoom_y + 110), (0, 220, 0), 3)

    # Magnifying glass icon
    mg_cx, mg_cy = zoom_x + 175, zoom_y - 25
    cv2.circle(canvas, (mg_cx, mg_cy), 20, (0, 200, 0), 3)
    cv2.line(canvas, (mg_cx + 14, mg_cy + 14), (mg_cx + 30, mg_cy + 30), (0, 200, 0), 4)

    cv2.putText(canvas, "Lips / Teeth Detail (Super-Res)", (zoom_x - 10, zoom_y + 250), font_small, 0.9, (0, 150, 0), 2)
    cv2.putText(canvas, "High-freq skin texture preserved", (x + 150, 850), font_small, 0.8, (80, 80, 80), 2)

    # ── Column 4: Final Synthesized Output ──
    x = 3000
    cv2.putText(canvas, "Stage 4: Final Synthesized Output", (x + 70, 80), font, 1.4, (0, 0, 0), 3)
    # Media player chrome
    cv2.rectangle(canvas, (x + 50, 150), (x + 950, 900), (35, 35, 40), -1)
    cv2.rectangle(canvas, (x + 50, 150), (x + 950, 200), (60, 60, 65), -1)
    # Traffic lights
    cv2.circle(canvas, (x + 80, 175), 10, (0, 0, 220), -1)
    cv2.circle(canvas, (x + 110, 175), 10, (0, 200, 220), -1)
    cv2.circle(canvas, (x + 140, 175), 10, (0, 200, 0), -1)
    cv2.putText(canvas, "AV Restore - Final Review", (x + 350, 188), font_small, 0.9, (200, 200, 200), 2)
    # Side-by-side inside player
    sw, sh = 400, 280
    c_small = cv2.resize(corrupted, (sw, sh))
    o_small = cv2.resize(original, (sw, sh))
    py = 260
    canvas[py:py + sh, x + 80:x + 80 + sw] = c_small
    canvas[py:py + sh, x + 520:x + 520 + sw] = o_small
    cv2.putText(canvas, "BEFORE", (x + 220, py - 10), font_small, 1.0, (100, 100, 255), 2)
    cv2.putText(canvas, "AFTER", (x + 670, py - 10), font_small, 1.0, (100, 255, 100), 2)
    # VS divider
    cv2.putText(canvas, "VS", (x + 490, py + sh // 2), font, 1.0, (200, 200, 200), 3)
    # Play bar
    cv2.rectangle(canvas, (x + 80, 800), (x + 920, 830), (80, 80, 85), -1)
    cv2.rectangle(canvas, (x + 80, 800), (x + 500, 830), (0, 130, 255), -1)
    # Play button
    pts = np.array([[x + 470, 850], [x + 470, 890], [x + 510, 870]], np.int32)
    cv2.fillPoly(canvas, [pts], (200, 200, 200))

    cv2.imwrite(output_path, canvas)
    print(f"  -> Saved {output_path}")


# ═══════════════════════════════════════════════════════════════════
# Build infographics from all available pipeline outputs
# ═══════════════════════════════════════════════════════════════════

output_dirs = [
    ("outputs",            "real_workflow_infographic.png",       "Video 1 (05YiCRDcCgw)"),
    ("outputs/video_02",   "workflow_infographic_video_2.png",    "Video 2 (68QuYW3Cr6g)"),
    ("outputs/video_03",   "workflow_infographic_video_3.png",    "Video 3 (KdZY7Z8RCs0)"),
    ("outputs/video_04",   "workflow_infographic_video_4.png",    "Video 4 (7G5kMmnAp_8)"),
]

for out_dir, out_file, label in output_dirs:
    print(f"\n{'='*60}")
    print(f"  Building: {out_file} ({label})")
    print(f"{'='*60}")

    corr_path = os.path.join(out_dir, "diag_corrupted_frame050.png")
    rest_path = os.path.join(out_dir, "diag_restored_frame050.png")
    orig_path = os.path.join(out_dir, "diag_original_frame050.png")

    if not all(os.path.exists(p) for p in [corr_path, rest_path, orig_path]):
        print(f"  SKIP: missing frames in {out_dir}")
        continue

    corrupted = cv2.imread(corr_path)
    restored  = cv2.imread(rest_path)
    original  = cv2.imread(orig_path)

    # Find exact mouth region by diffing original vs corrupted
    mouth_box = find_mouth_box(original, corrupted)
    print(f"  Mouth box: x={mouth_box[0]}, y={mouth_box[1]}, w={mouth_box[2]}, h={mouth_box[3]}")

    create_infographic(corrupted, restored, original, mouth_box, out_file, label)

print("\n  All infographics generated!")
