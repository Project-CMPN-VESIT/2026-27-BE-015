import os
import subprocess
import glob

# 1. Define your folders
input_dir = "Obama_Dataset/01_raw_youtube"
output_dir = "Obama_Dataset/02_processed_audio"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# 2. Find all the downloaded MP4 videos
video_files = glob.glob(os.path.join(input_dir, "*.mp4"))

print(f"Found {len(video_files)} videos. Starting 16kHz Audio Extraction...\n")

# 3. Loop through each video and extract the audio
for vid_path in video_files:
    # Get the filename without the .mp4 extension
    basename = os.path.basename(vid_path).split('.')[0]
    out_wav_path = os.path.join(output_dir, f"{basename}.wav")
    
    # Check if we already processed this file (saves time if you restart)
    if not os.path.exists(out_wav_path):
        print(f"Extracting audio from: {basename}.mp4")
        
        # The FFmpeg command: 
        # -vn (ignore video), -ac 1 (mono channel), -ar 16000 (16kHz sample rate)
        cmd =[
            'ffmpeg', 
            '-i', vid_path, 
            '-vn', 
            '-ac', '1', 
            '-ar', '16000', 
            out_wav_path, 
            '-y' # Overwrite if exists
        ]
        
        # Run FFmpeg silently so it doesn't flood your terminal
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    else:
        print(f"Already extracted: {basename}.wav (Skipping)")

print("\n--- Audio Extraction Complete! Check your 02_processed_audio folder. ---")