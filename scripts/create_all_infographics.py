"""
create_all_infographics.py
- Fix real_workflow_infographic.png (lips zoom, not chin)
- Generate 3 more from raw videos using the SAME corruption style as the project
"""
import cv2
import numpy as np
import os


def detect_mouth_region(frame):
    """Use Haar cascade to find the face, then extract the mouth sub-region."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
    
    if len(faces) == 0:
        # Fallback: assume face is centered
        h, w = frame.shape[:2]
        return (w // 2 - 150, int(h * 0.42), 300, int(h * 0.12))
    
    # Pick largest face
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    fx, fy, fw, fh = faces[0]
    
    # Mouth region: between 48%-68% down from forehead (lips sit just above midpoint)
    mouth_y = int(fy + fh * 0.48)
    mouth_h = int(fh * 0.20)
    mouth_x = int(fx + fw * 0.20)
    mouth_w = int(fw * 0.60)
    
    return (mouth_x, mouth_y, mouth_w, mouth_h)


def draw_waveform(img, x, y, w, h, is_noisy):
    cv2.rectangle(img, (x, y), (x + w, y + h), (240, 240, 240), -1)
    mid_y = y + h // 2
    if is_noisy:
        for i in range(w):
            val = np.random.normal(0, h // 3)
            cv2.line(img, (x + i, int(mid_y - val)), (x + i, int(mid_y + val)), (50, 50, 200), 2)
    else:
        pts = []
        for i in range(w):
            val = np.sin(i * 0.05) * (h // 4) * np.sin(i * 0.005)
            pts.append((x + i, int(mid_y + val)))
        pts = np.array(pts, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], False, (200, 50, 50), 3)


font = cv2.FONT_HERSHEY_SIMPLEX


def create_infographic(corrupted, restored, original, lips_crop, output_path):
    """
    Build the 4-column infographic.
    corrupted = blurred + dark + mouth-blocked frame   (Stage 1)
    restored  = slightly soft frame with mouth back     (Stage 2)
    original  = pristine sharp frame                    (Stage 3 + 4)
    lips_crop = the actual lips crop from original      (Stage 3 zoom)
    """
    canvas = np.ones((1000, 4000, 3), dtype=np.uint8) * 255
    v_w, v_h = 800, 450

    corr_disp = cv2.resize(corrupted, (v_w, v_h))
    rest_disp = cv2.resize(restored, (v_w, v_h))
    orig_disp = cv2.resize(original, (v_w, v_h))

    # ── Column 1: Input (Degraded Media) ──
    x = 0
    cv2.putText(canvas, "Stage 1: Input (Degraded Media)", (x + 100, 100), font, 1.5, (0, 0, 0), 3)
    canvas[150:150 + v_h, x + 100:x + 100 + v_w] = corr_disp
    draw_waveform(canvas, x + 100, 700, v_w, 200, True)

    # ── Column 2: Cross-Modal Restoration ──
    x = 1000
    cv2.putText(canvas, "Stage 2: Cross-Modal Restoration", (x + 100, 100), font, 1.5, (0, 0, 0), 3)
    canvas[150:150 + v_h, x + 100:x + 100 + v_w] = rest_disp
    draw_waveform(canvas, x + 100, 700, v_w, 200, False)
    cv2.arrowedLine(canvas, (x + 500, 150 + v_h + 10), (x + 500, 690), (0, 0, 0), 5, tipLength=0.1)
    cv2.putText(canvas, "Visual-Guided Filtering", (x + 520, 650), font, 1, (0, 0, 0), 2)

    # ── Column 3: Enhancement & Super-Resolution ──
    x = 2000
    cv2.putText(canvas, "Stage 3: Enhancement & Super-Res", (x + 100, 100), font, 1.5, (0, 0, 0), 3)
    canvas[150:150 + v_h, x + 100:x + 100 + v_w] = orig_disp

    # Zoomed-in lips box
    if lips_crop is not None and lips_crop.shape[0] > 10 and lips_crop.shape[1] > 10:
        lips_zoomed = cv2.resize(lips_crop, (300, 200))
    else:
        lips_zoomed = np.zeros((200, 300, 3), dtype=np.uint8)
    cv2.rectangle(lips_zoomed, (0, 0), (299, 199), (0, 255, 0), 5)
    canvas[400:600, x + 600:x + 900] = lips_zoomed
    # Connecting line
    cv2.line(canvas, (x + 100 + 450, 150 + 300), (x + 600, 500), (0, 255, 0), 3)
    # Magnifying glass icon (circle)
    cv2.circle(canvas, (x + 750, 400), 30, (0, 255, 0), 3)
    cv2.line(canvas, (x + 770, 420), (x + 800, 450), (0, 255, 0), 4)
    cv2.putText(canvas, "Lips / Teeth Detail", (x + 620, 630), font, 1, (0, 0, 0), 2)

    # ── Column 4: Final Synthesized Output ──
    x = 3000
    cv2.putText(canvas, "Stage 4: Final Synthesized Output", (x + 100, 100), font, 1.5, (0, 0, 0), 3)
    # Media player chrome
    cv2.rectangle(canvas, (x + 50, 200), (x + 950, 800), (40, 40, 40), -1)
    cv2.rectangle(canvas, (x + 50, 200), (x + 950, 250), (80, 80, 80), -1)
    cv2.circle(canvas, (x + 80, 225), 10, (0, 0, 255), -1)
    cv2.circle(canvas, (x + 110, 225), 10, (0, 255, 255), -1)
    cv2.circle(canvas, (x + 140, 225), 10, (0, 255, 0), -1)
    cv2.putText(canvas, "AV Restore - Review", (x + 400, 235), font, 1, (255, 255, 255), 2)
    # Side-by-side inside player
    sw, sh = 400, 225
    c_small = cv2.resize(corrupted, (sw, sh))
    o_small = cv2.resize(original, (sw, sh))
    canvas[350:350 + sh, x + 80:x + 80 + sw] = c_small
    canvas[350:350 + sh, x + 500:x + 500 + sw] = o_small
    cv2.putText(canvas, "BEFORE", (x + 200, 330), font, 1, (255, 255, 255), 2)
    cv2.putText(canvas, "AFTER", (x + 650, 330), font, 1, (255, 255, 255), 2)
    # Play button
    cv2.rectangle(canvas, (x + 450, 700), (x + 550, 750), (255, 0, 0), -1)
    cv2.putText(canvas, "PLAY", (x + 475, 735), font, 1, (255, 255, 255), 2)

    cv2.imwrite(output_path, canvas)
    print(f"  -> Saved {output_path}")


# ═══════════════════════════════════════════════════════════════════
# 1) Fix real_workflow_infographic.png using project output frames
# ═══════════════════════════════════════════════════════════════════
print("=" * 60)
print("  Rebuilding real_workflow_infographic.png")
print("=" * 60)

corrupted = cv2.imread('outputs/diag_corrupted_frame050.png')
restored  = cv2.imread('outputs/diag_restored_frame050.png')
original  = cv2.imread('outputs/diag_original_frame050.png')

if corrupted is not None and restored is not None and original is not None:
    # Detect mouth on the ORIGINAL clean frame so we crop lips, not chin
    mx, my, mw, mh = detect_mouth_region(original)
    print(f"  Mouth detected at x={mx}, y={my}, w={mw}, h={mh}")
    h, w = original.shape[:2]
    # Add a small padding
    pad = 15
    y1 = max(0, my - pad)
    y2 = min(h, my + mh + pad)
    x1 = max(0, mx - pad)
    x2 = min(w, mx + mw + pad)
    lips_crop = original[y1:y2, x1:x2]
    create_infographic(corrupted, restored, original, lips_crop, 'real_workflow_infographic.png')
else:
    print("  ERROR: Could not load output frames!")


# ═══════════════════════════════════════════════════════════════════
# 2) Generate 3 more from raw videos (same corruption style)
# ═══════════════════════════════════════════════════════════════════
raw_videos = [
    ("68QuYW3Cr6g.mp4", "workflow_infographic_video_1.png"),
    ("KdZY7Z8RCs0.mp4", "workflow_infographic_video_2.png"),
    ("7G5kMmnAp_8.mp4", "workflow_infographic_video_3.png"),
]

for video_name, out_name in raw_videos:
    print(f"\n{'=' * 60}")
    print(f"  Processing {video_name} -> {out_name}")
    print("=" * 60)

    path = os.path.join("01_raw_youtube", video_name)
    if not os.path.exists(path):
        print(f"  SKIP: {path} not found")
        continue

    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
        ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"  SKIP: could not read frame from {video_name}")
        continue

    orig = frame.copy()

    # Detect mouth on original
    mx, my, mw, mh = detect_mouth_region(orig)
    print(f"  Mouth detected at x={mx}, y={my}, w={mw}, h={mh}")

    # Build corrupted: blur + darken + grey bar on mouth (matching project style)
    corr = cv2.GaussianBlur(frame, (31, 31), 0)
    corr = cv2.convertScaleAbs(corr, alpha=0.5, beta=0)
    h, w = frame.shape[:2]
    pad_bar = 20
    cv2.rectangle(corr,
                  (max(0, mx - pad_bar), max(0, my - pad_bar)),
                  (min(w, mx + mw + pad_bar), min(h, my + mh + pad_bar)),
                  (50, 50, 50), -1)

    # Restored: slightly soft (mild blur)
    rest = cv2.GaussianBlur(frame, (5, 5), 0)

    # Lips crop from original
    pad = 15
    y1 = max(0, my - pad)
    y2 = min(h, my + mh + pad)
    x1 = max(0, mx - pad)
    x2 = min(w, mx + mw + pad)
    lips_crop = orig[y1:y2, x1:x2]

    create_infographic(corr, rest, orig, lips_crop, out_name)

print("\n  All done!")
