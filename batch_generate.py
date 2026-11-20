"""
batch_generate.py -- Run the pipeline on multiple videos to generate test outputs.
Each video gets its own output directory under outputs/.
"""

import subprocess
import sys
import os
import time

# Videos to process with their output names
JOBS = [
    {
        "video": "05YiCRDcCgw.mp4",
        "output_dir": "outputs/video_01_obama_speech",
        "max_frames": 300,
        "start_frame": 500,
        "label": "Video 1 - Obama Address (10s)",
    },
    {
        "video": "68QuYW3Cr6g.mp4",
        "output_dir": "outputs/video_02_obama_weekly",
        "max_frames": 300,
        "start_frame": 500,
        "label": "Video 2 - Obama Weekly (10s)",
    },
    {
        "video": "KdZY7Z8RCs0.mp4",
        "output_dir": "outputs/video_03_obama_press",
        "max_frames": 300,
        "start_frame": 500,
        "label": "Video 3 - Obama Press Conference (10s)",
    },
    {
        "video": "7G5kMmnAp_8.mp4",
        "output_dir": "outputs/video_04_obama_remarks",
        "max_frames": 300,
        "start_frame": 500,
        "label": "Video 4 - Obama Remarks (10s)",
    },
]


def run_job(job, job_num, total):
    video_path = os.path.join("01_raw_youtube", job["video"])
    if not os.path.isfile(video_path):
        print(f"  [SKIP] Video not found: {video_path}")
        return False

    print(f"\n{'='*70}")
    print(f"  JOB {job_num}/{total}: {job['label']}")
    print(f"  Input:  {job['video']}")
    print(f"  Frames: {job['max_frames']} (from frame {job['start_frame']})")
    print(f"  Output: {job['output_dir']}")
    print(f"{'='*70}\n")

    cmd = [
        sys.executable, "main.py",
        "--input", video_path,
        "--output-dir", job["output_dir"],
        "--max-frames", str(job["max_frames"]),
        "--start-frame", str(job["start_frame"]),
        "--skip-vae",
    ]

    start = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n  [OK] {job['label']} completed in {elapsed:.1f}s")
        return True
    else:
        print(f"\n  [FAIL] {job['label']} failed (exit code {result.returncode})")
        return False


def main():
    print("\n" + "=" * 70)
    print("  BATCH VIDEO GENERATION")
    print(f"  Processing {len(JOBS)} videos...")
    print("=" * 70)

    total_start = time.time()
    results = []

    for i, job in enumerate(JOBS, 1):
        success = run_job(job, i, len(JOBS))
        results.append((job["label"], success))

    total_time = time.time() - total_start

    print("\n\n" + "=" * 70)
    print("  BATCH COMPLETE")
    print("=" * 70)
    for label, success in results:
        status = "OK" if success else "FAIL"
        print(f"  [{status}] {label}")
    print(f"\n  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
