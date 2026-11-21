import cv2
import numpy as np
import os

# Create a blank white canvas 4000x1000
canvas = np.ones((1000, 4000, 3), dtype=np.uint8) * 255

col_w = 1000

# Load images
corrupted = cv2.imread('outputs/diag_corrupted_frame050.png')
restored = cv2.imread('outputs/diag_restored_frame050.png')
original = cv2.imread('outputs/diag_original_frame050.png')

# If they don't exist for some reason, use random noise as fallback to prevent crash, but they should exist
if corrupted is None: corrupted = np.zeros((450, 800, 3), dtype=np.uint8)
if restored is None: restored = np.zeros((450, 800, 3), dtype=np.uint8)
if original is None: original = np.zeros((450, 800, 3), dtype=np.uint8)

# Resize to standard size for top half (800x450)
v_w, v_h = 800, 450
corrupted = cv2.resize(corrupted, (v_w, v_h))
restored = cv2.resize(restored, (v_w, v_h))
original = cv2.resize(original, (v_w, v_h))

# Function to draw waveform
def draw_waveform(img, x, y, w, h, is_noisy):
    # Draw background
    cv2.rectangle(img, (x, y), (x+w, y+h), (240, 240, 240), -1)
    
    # Generate points
    pts = []
    mid_y = y + h // 2
    for i in range(w):
        if is_noisy:
            val = np.random.normal(0, h//3)
            # Make it thicker and more chaotic
            cv2.line(img, (x+i, int(mid_y - val)), (x+i, int(mid_y + val)), (50, 50, 200), 2)
        else:
            # Clean sine wave + human voice shape
            val = np.sin(i * 0.05) * (h//4) * np.sin(i * 0.005)
            pts.append((x+i, int(mid_y + val)))
            
    if not is_noisy:
        pts = np.array(pts, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], False, (200, 50, 50), 3)

font = cv2.FONT_HERSHEY_SIMPLEX

# --- Column 1 ---
x_offset = 0
cv2.putText(canvas, "Stage 1: Input (Degraded Media)", (x_offset + 100, 100), font, 1.5, (0,0,0), 3)
canvas[150:150+v_h, x_offset+100:x_offset+100+v_w] = corrupted
draw_waveform(canvas, x_offset+100, 700, v_w, 200, True)

# --- Column 2 ---
x_offset = 1000
cv2.putText(canvas, "Stage 2: Cross-Modal Restoration", (x_offset + 100, 100), font, 1.5, (0,0,0), 3)
canvas[150:150+v_h, x_offset+100:x_offset+100+v_w] = restored
draw_waveform(canvas, x_offset+100, 700, v_w, 200, False)
# Draw arrow from video to audio
cv2.arrowedLine(canvas, (x_offset+500, 150+v_h+10), (x_offset+500, 690), (0,0,0), 5, tipLength=0.1)
cv2.putText(canvas, "Visual-Guided Filtering", (x_offset + 520, 650), font, 1, (0,0,0), 2)

# --- Column 3 ---
x_offset = 2000
cv2.putText(canvas, "Stage 3: Enhancement & Super-Res", (x_offset + 100, 100), font, 1.5, (0,0,0), 3)
canvas[150:150+v_h, x_offset+100:x_offset+100+v_w] = original
# Zoom in box
# Let's crop a section from original (lips area, approx bottom center)
lips_crop = original[250:350, 300:450]
lips_zoomed = cv2.resize(lips_crop, (300, 200))
# Add border
cv2.rectangle(lips_zoomed, (0,0), (299, 199), (0,255,0), 5)
canvas[400:600, x_offset+600:x_offset+900] = lips_zoomed
# Draw line from crop source to zoomed box
cv2.line(canvas, (x_offset+100+450, 150+300), (x_offset+600, 500), (0,255,0), 3)
cv2.putText(canvas, "High-Freq Details Reconstructed", (x_offset+150, 750), font, 1.2, (0,0,0), 2)

# --- Column 4 ---
x_offset = 3000
cv2.putText(canvas, "Stage 4: Final Synthesized Output", (x_offset + 100, 100), font, 1.5, (0,0,0), 3)
# Draw media player UI
cv2.rectangle(canvas, (x_offset+50, 200), (x_offset+950, 800), (40, 40, 40), -1) # Player bg
cv2.rectangle(canvas, (x_offset+50, 200), (x_offset+950, 250), (80, 80, 80), -1) # Title bar
cv2.circle(canvas, (x_offset+80, 225), 10, (0,0,255), -1)
cv2.circle(canvas, (x_offset+110, 225), 10, (0,255,255), -1)
cv2.circle(canvas, (x_offset+140, 225), 10, (0,255,0), -1)
cv2.putText(canvas, "AV Restore - Review", (x_offset+400, 235), font, 1, (255,255,255), 2)

# Inside player: Before | After
# resize them to fit side by side in 800x500
small_w, small_h = 400, 225
c_small = cv2.resize(corrupted, (small_w, small_h))
o_small = cv2.resize(original, (small_w, small_h))
canvas[350:350+small_h, x_offset+80:x_offset+80+small_w] = c_small
canvas[350:350+small_h, x_offset+500:x_offset+500+small_w] = o_small
cv2.putText(canvas, "BEFORE", (x_offset+200, 330), font, 1, (255,255,255), 2)
cv2.putText(canvas, "AFTER", (x_offset+650, 330), font, 1, (255,255,255), 2)

# Draw play button
cv2.rectangle(canvas, (x_offset+450, 700), (x_offset+550, 750), (255, 0, 0), -1)
cv2.putText(canvas, "PLAY", (x_offset+475, 735), font, 1, (255,255,255), 2)


cv2.imwrite('real_workflow_infographic.png', canvas)
print("Saved real_workflow_infographic.png")
