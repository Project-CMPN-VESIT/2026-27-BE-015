"""
Diagnostic script: compare corrupted vs restored audio, and extract multiple video frames.
"""
import numpy as np
import os

# ============ AUDIO COMPARISON ============
print("=" * 60)
print("AUDIO DIAGNOSIS")
print("=" * 60)

import soundfile as sf

orig_audio, sr1 = sf.read("comparison_original_audio.wav")
corr_audio, sr2 = sf.read("comparison_corrupted_audio.wav")
rest_audio, sr3 = sf.read("comparison_audio_restored_audio.wav")

print(f"Original audio:  {len(orig_audio)} samples, sr={sr1}")
print(f"Corrupted audio: {len(corr_audio)} samples, sr={sr2}")
print(f"Restored audio:  {len(rest_audio)} samples, sr={sr3}")

# Check if corrupted and restored are identical
min_len = min(len(corr_audio), len(rest_audio))
corr_trim = corr_audio[:min_len]
rest_trim = rest_audio[:min_len]

diff = np.abs(corr_trim - rest_trim)
print(f"\nCorrupted vs Restored difference:")
print(f"  Max diff:  {diff.max():.6f}")
print(f"  Mean diff: {diff.mean():.6f}")
print(f"  Samples that differ (>1e-6): {(diff > 1e-6).sum()} / {min_len}")

if diff.max() < 1e-6:
    print("  >>> PROBLEM: Corrupted and Restored audio are IDENTICAL! Restoration did nothing.")
else:
    pct = (diff > 1e-6).sum() / min_len * 100
    print(f"  >>> {pct:.1f}% of samples were changed by restoration.")

# Check where corruption is (zeros in corrupted)
zero_mask = np.abs(corr_audio) < 1e-10
print(f"\nCorrupted audio zero regions: {zero_mask.sum()} samples ({zero_mask.sum()/len(corr_audio)*100:.1f}%)")

# Check if those zeros got filled in restored
if len(rest_audio) >= len(corr_audio):
    rest_at_zeros = rest_audio[:len(corr_audio)][zero_mask]
    filled = np.abs(rest_at_zeros) > 1e-10
    print(f"Restored audio fills {filled.sum()}/{zero_mask.sum()} zero-gaps ({filled.sum()/max(1,zero_mask.sum())*100:.1f}%)")

# ============ VIDEO FRAME EXTRACTION ============
print("\n" + "=" * 60)
print("VIDEO FRAME EXTRACTION (every 10th frame)")
print("=" * 60)

import cv2

for label, path in [
    ("original", "comparison_original_video_only.mp4"),
    ("corrupted", "comparison_corrupted_video_only.mp4"),
    ("restored", "comparison_restored_video_only.mp4"),
]:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n{label}: {total} frames")
    
    for idx in [0, 10, 20, 30, 50, 70, 90]:
        if idx >= total:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            fname = f"diag_{label}_frame{idx:03d}.png"
            cv2.imwrite(fname, frame)
            print(f"  Saved {fname}")
    cap.release()

print("\nDone!")
