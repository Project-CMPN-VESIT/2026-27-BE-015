import cv2
import numpy as np
import os

def create_synthetic_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cap.release()
    
    original = frame.copy()
    
    corrupted = cv2.GaussianBlur(frame, (31, 31), 0)
    corrupted = cv2.convertScaleAbs(corrupted, alpha=0.5, beta=0)
    
    h, w = frame.shape[:2]
    
    # Use simple Haar cascade for face detection to locate mouth
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    lips_crop = np.zeros((0, 0, 3), dtype=np.uint8)
    
    if len(faces) > 0:
        # Get the largest face
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        fx, fy, fw, fh = faces[0]
        
        # Mouth is roughly the bottom 1/3 of the face
        mouth_y = int(fy + fh * 0.65)
        mouth_h = int(fh * 0.35)
        
        # Draw dark grey box over mouth area precisely
        cv2.rectangle(corrupted, (fx, mouth_y), (fx + fw, mouth_y + mouth_h), (50, 50, 50), -1)
        
        # Get mouth crop for the zoomed-in box later
        lips_crop = original[max(0, mouth_y-10):min(h, mouth_y + mouth_h+10), max(0, fx-10):min(w, fx + fw+10)]
    else:
        # Fallback if no face detected
        bar_w, bar_h = 200, 100
        cv2.rectangle(corrupted, (w//2 - bar_w//2, h//2 + 50), (w//2 + bar_w//2, h//2 + 50 + bar_h), (50, 50, 50), -1)
        lips_crop = original[h//2 + 50:h//2 + 150, w//2 - 100:w//2 + 100]
    
    restored = cv2.GaussianBlur(frame, (5, 5), 0)
    
    return original, corrupted, restored, lips_crop

def draw_waveform(img, x, y, w, h, is_noisy):
    cv2.rectangle(img, (x, y), (x+w, y+h), (240, 240, 240), -1)
    pts = []
    mid_y = y + h // 2
    for i in range(w):
        if is_noisy:
            val = np.random.normal(0, h//3)
            cv2.line(img, (x+i, int(mid_y - val)), (x+i, int(mid_y + val)), (50, 50, 200), 2)
        else:
            val = np.sin(i * 0.05) * (h//4) * np.sin(i * 0.005)
            pts.append((x+i, int(mid_y + val)))
    if not is_noisy:
        pts = np.array(pts, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], False, (200, 50, 50), 3)

font = cv2.FONT_HERSHEY_SIMPLEX

def create_infographic(original, corrupted, restored, lips_crop, output_path):
    canvas = np.ones((1000, 4000, 3), dtype=np.uint8) * 255
    v_w, v_h = 800, 450
    corrupted = cv2.resize(corrupted, (v_w, v_h))
    restored = cv2.resize(restored, (v_w, v_h))
    original = cv2.resize(original, (v_w, v_h))

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
    cv2.arrowedLine(canvas, (x_offset+500, 150+v_h+10), (x_offset+500, 690), (0,0,0), 5, tipLength=0.1)
    cv2.putText(canvas, "Visual-Guided Filtering", (x_offset + 520, 650), font, 1, (0,0,0), 2)

    # --- Column 3 ---
    x_offset = 2000
    cv2.putText(canvas, "Stage 3: Enhancement & Super-Res", (x_offset + 100, 100), font, 1.5, (0,0,0), 3)
    canvas[150:150+v_h, x_offset+100:x_offset+100+v_w] = original
    
    if lips_crop.shape[0] > 0 and lips_crop.shape[1] > 0:
        lips_zoomed = cv2.resize(lips_crop, (300, 200))
    else:
        lips_zoomed = np.zeros((200, 300, 3), dtype=np.uint8)
        
    cv2.rectangle(lips_zoomed, (0,0), (299, 199), (0,255,0), 5)
    canvas[400:600, x_offset+600:x_offset+900] = lips_zoomed
    cv2.line(canvas, (x_offset+100+450, 150+300), (x_offset+600, 500), (0,255,0), 3)
    cv2.putText(canvas, "High-Freq Details Reconstructed", (x_offset+150, 750), font, 1.2, (0,0,0), 2)

    # --- Column 4 ---
    x_offset = 3000
    cv2.putText(canvas, "Stage 4: Final Synthesized Output", (x_offset + 100, 100), font, 1.5, (0,0,0), 3)
    cv2.rectangle(canvas, (x_offset+50, 200), (x_offset+950, 800), (40, 40, 40), -1)
    cv2.rectangle(canvas, (x_offset+50, 200), (x_offset+950, 250), (80, 80, 80), -1)
    cv2.circle(canvas, (x_offset+80, 225), 10, (0,0,255), -1)
    cv2.circle(canvas, (x_offset+110, 225), 10, (0,255,255), -1)
    cv2.circle(canvas, (x_offset+140, 225), 10, (0,255,0), -1)
    cv2.putText(canvas, "AV Restore - Review", (x_offset+400, 235), font, 1, (255,255,255), 2)
    small_w, small_h = 400, 225
    c_small = cv2.resize(corrupted, (small_w, small_h))
    o_small = cv2.resize(original, (small_w, small_h))
    canvas[350:350+small_h, x_offset+80:x_offset+80+small_w] = c_small
    canvas[350:350+small_h, x_offset+500:x_offset+500+small_w] = o_small
    cv2.putText(canvas, "BEFORE", (x_offset+200, 330), font, 1, (255,255,255), 2)
    cv2.putText(canvas, "AFTER", (x_offset+650, 330), font, 1, (255,255,255), 2)
    cv2.rectangle(canvas, (x_offset+450, 700), (x_offset+550, 750), (255, 0, 0), -1)
    cv2.putText(canvas, "PLAY", (x_offset+475, 735), font, 1, (255,255,255), 2)

    cv2.imwrite(output_path, canvas)
    print(f"Saved {output_path}")

videos = ["68QuYW3Cr6g.mp4", "KdZY7Z8RCs0.mp4", "7G5kMmnAp_8.mp4"]
for i, v in enumerate(videos):
    path = os.path.join("01_raw_youtube", v)
    if os.path.exists(path):
        orig, corr, rest, lips_crop = create_synthetic_frames(path)
        create_infographic(orig, corr, rest, lips_crop, f"workflow_infographic_video_{i+1}.png")
    else:
        print(f"File not found: {path}")
