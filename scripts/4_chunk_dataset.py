import os
import subprocess
import glob

print("\n--- Starting Local Dataset Chunking ---")

# 1. Define Local Paths
input_dir = "Obama_Dataset/03_cropped_faces"
out_dir = "Obama_Dataset/04_training_chunks"
os.makedirs(out_dir, exist_ok=True)

CHUNK_LENGTH = 4 # 4-second chunks (ideal for Diffusion)
videos = glob.glob(os.path.join(input_dir, "*.mp4"))

if not videos:
    print(f"CRITICAL ERROR: No .mp4 files found in {input_dir}. Please download them from Google Drive first!")
    exit()

print(f"Found {len(videos)} perfect videos. Slicing into 4-second pairs...\n")

chunk_counter = 0

for vid_path in videos:
    basename = os.path.basename(vid_path)
   
    # Get exact duration of video using FFprobe
    duration_cmd =[
        'ffprobe', '-v', 'error', '-show_entries',
        'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', vid_path
    ]
    try:
        duration = float(subprocess.check_output(duration_cmd).decode('utf-8').strip())
    except Exception as e:
        print(f"  Warning: Could not read duration for {basename}. Skipping. Error: {e}")
        continue

    start_time = 0
    print(f"  -> Chunking {basename}...")
   
    # Slice into 4 second chunks
    while start_time + CHUNK_LENGTH <= duration:
       
        out_vid = os.path.join(out_dir, f"chunk_{chunk_counter:05d}.mp4")
        out_aud = os.path.join(out_dir, f"chunk_{chunk_counter:05d}.wav")
       
        # 1. Extract 4-sec Video Chunk (Silent)
        cmd_vid =[
            'ffmpeg', '-y', '-ss', str(start_time), '-t', str(CHUNK_LENGTH),
            '-i', vid_path, '-c:v', 'libx264', '-preset', 'fast', '-an', out_vid
        ]
       
        # 2. Extract 4-sec Audio Chunk (Forced to 16kHz Mono for Wav2Vec)
        cmd_aud =[
            'ffmpeg', '-y', '-ss', str(start_time), '-t', str(CHUNK_LENGTH),
            '-i', vid_path, '-c:a', 'pcm_s16le', '-ar', '16000', '-ac', '1', out_aud
        ]
       
        # Run FFmpeg silently
        subprocess.run(cmd_vid, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        subprocess.run(cmd_aud, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
       
        start_time += CHUNK_LENGTH
        chunk_counter += 1

print(f"\n--- SUCCESS! Created {chunk_counter} Video/Audio chunk pairs. ---")
print(f"Check your '{out_dir}' folder. PHASE 1 IS 100% COMPLETE!")