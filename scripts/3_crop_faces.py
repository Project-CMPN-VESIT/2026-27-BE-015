import cv2
import dlib
import os
import glob
from moviepy.editor import ImageSequenceClip
import numpy as np # dlib uses numpy

print("\n--- Running Face Cropping with 'dlib' ---")

# Initialize dlib's built-in face detector
try:
    face_detector = dlib.get_frontal_face_detector()
    print("SUCCESS: dlib face detector loaded.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load dlib. Ensure you installed it correctly. Error: {e}")
    exit()

# Define folders
input_dir = "Obama_Dataset/01_raw_youtube"
output_dir = "Obama_Dataset/03_cropped_faces"
os.makedirs(output_dir, exist_ok=True)

# Define target output size
TARGET_SIZE = (512, 512)

# Get all the raw mp4 files
video_files = glob.glob(os.path.join(input_dir, "*.mp4"))

if not video_files:
    print("\nCRITICAL ERROR: No .mp4 files were found. Check your '01_raw_youtube' folder.")
    exit()

print(f"\nFound {len(video_files)} videos. Starting Face Cropping process...")

for vid_path in video_files:
    basename = os.path.basename(vid_path)
    out_path = os.path.join(output_dir, basename)
    
    if os.path.exists(out_path):
        print(f"'{basename}' already processed. Skipping.")
        continue
        
    print(f"\n--- Processing: {basename} ---")
    
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print(f"  ERROR: Could not open '{vid_path}'. Skipping.")
        continue
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video Info: FPS = {fps}, Total Frames = {total_frames}")
    
    cropped_frames = []
    
    frame_count = 0
    faces_detected_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        h, w, _ = frame.shape
        
        # dlib works on grayscale images, which is faster
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        detections = face_detector(img_gray, 1) # The '1' upsamples the image, finding more faces
        
        if len(detections) > 0:
            faces_detected_count += 1
            d = detections[0] # Grab the first face
            
            # Get coordinates from dlib's rectangle object
            xmin, ymin, xmax, ymax = d.left(), d.top(), d.right(), d.bottom()
            box_w, box_h = xmax - xmin, ymax - ymin
            
            # ADD PADDING
            pad_w, pad_h = int(box_w * 0.3), int(box_h * 0.4)
            x1, y1 = max(0, xmin - pad_w), max(0, ymin - pad_h)
            x2, y2 = min(w, xmax + pad_w), min(h, ymax + pad_h)
            
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size > 0:
                face_resized = cv2.resize(face_crop, TARGET_SIZE)
                cropped_frames.append(cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB))
    
    cap.release()
    
    print(f"  Finished processing. Total faces detected: {faces_detected_count} / {frame_count} frames.")
    
    if cropped_frames:
        print(f"  About to save video with {len(cropped_frames)} frames...")
        clip = ImageSequenceClip(cropped_frames, fps=fps)
        clip.write_videofile(out_path, codec='libx264', audio=False, logger='bar')
        print(f"  SUCCESS: Video saved to '{out_path}'")
    else:
        print(f"  WARNING: No faces were detected in '{basename}'. No output video was created.")

print("\n--- dlib Face Cropping Script Complete! ---")