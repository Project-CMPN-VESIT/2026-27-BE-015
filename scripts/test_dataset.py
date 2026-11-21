import sys, os
import torch
from torch.utils.data import DataLoader

# Bulletproof path routing (Finds the root Capstone folder automatically)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from utils.dataset import CachedLatentDataset

# UPDATE THIS string if your folder has extra text like "05_cached_latents-2..."
FOLDER_NAME = "05_cached_latents" 
CACHE_DIR = os.path.join(ROOT_DIR, "Obama_Dataset", FOLDER_NAME)

print(f"Looking for data in: {CACHE_DIR}")

try:
    dataset = CachedLatentDataset(cache_dir=CACHE_DIR)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    clean, corrupted, audio = next(iter(dataloader))
    print(f"✅ Clean Video Shape: {clean.shape}")       # Expects: [2, 4, 64, 64]
    print(f"✅ Corrupted Video Shape: {corrupted.shape}") # Expects:[2, 4, 64, 64]
    print(f"✅ Audio Embedding Shape: {audio.shape}")     # Expects: [2, Seq_Len, 768]
    print("\n🚀 DATASET TEST PASSED! You are ready for the U-Net test.")
except Exception as e:
    print(f"\n❌ FAILED: {e}")