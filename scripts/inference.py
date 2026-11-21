import os
import torch
from diffusers import DDPMScheduler, AutoencoderKL
from tqdm import tqdm
import numpy as np
import moviepy.editor as mpy
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from models.unet import AudioGuidedVideoUNet

# --- CONFIGURATION ---
CHECKPOINT_PATH = "checkpoints/unet_epoch_10.pth"
OUTPUT_DIR = "inference_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_FILE_NAME = "cache_00001.pt"
TEST_DATA_PATH = os.path.join(ROOT_DIR, "Obama_Dataset", "05_cached_latents", TEST_FILE_NAME)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
0
def main():
    print(f"🚀 Running Inference on: {DEVICE.upper()}")
    
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"❌ Checkpoint not found at: {CHECKPOINT_PATH}")
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"❌ Test data not found at: {TEST_DATA_PATH}")

    print("Loading models...")
    model = AudioGuidedVideoUNet().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(DEVICE)
    vae.eval()
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    print(f"Loading test data from: {TEST_DATA_PATH}")
    data = torch.load(TEST_DATA_PATH, map_location='cpu')
    corrupted_vid_latents = data['latent_corrupt'].unsqueeze(0).to(DEVICE)
    audio_embeddings = data['audio_emb_clean'].unsqueeze(0).to(DEVICE)
    
    num_frames = corrupted_vid_latents.shape[1]
    latents = torch.randn_like(corrupted_vid_latents)

    print("Denoising video frame by frame...")
    scheduler.set_timesteps(1000)
    output_frames = []

    for f_idx in tqdm(range(num_frames), desc="Generating Frames"):
        corrupted_frame_latent = corrupted_vid_latents[:, f_idx, :, :, :]
        frame_latents = latents[:, f_idx, :, :, :]

        for t in scheduler.timesteps:
            with torch.no_grad():
                noise_pred = model(frame_latents, corrupted_frame_latent, t.unsqueeze(0).to(DEVICE), audio_embeddings)
                frame_latents = scheduler.step(noise_pred, t, frame_latents).prev_sample
        
        with torch.no_grad():
            frame_latents_unscaled = frame_latents / 0.18215
            image_tensor = vae.decode(frame_latents_unscaled).sample
        
        image = (image_tensor / 2 + 0.5).clamp(0, 1)
        image = image.squeeze(0).cpu().permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)
        output_frames.append(image)

    print("Stitching video...")
    audio_filename = TEST_FILE_NAME.replace("cache_", "chunk_").replace(".pt", ".wav")
    original_audio_path = os.path.join(ROOT_DIR, "Obama_Dataset", "04_training_chunks", audio_filename)
    
    if not os.path.exists(original_audio_path):
        raise FileNotFoundError(f"❌ Matching audio file not found: {original_audio_path}")

    clip = mpy.ImageSequenceClip(output_frames, fps=30) # Assuming 30 FPS
    audio = mpy.AudioFileClip(original_audio_path)
    final_clip = clip.set_audio(audio)

    output_filename = f"epoch_{CHECKPOINT_PATH.split('_')[-1].split('.')[0]}_sample.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    final_clip.write_videofile(output_path, codec='libx264')

    print(f"🎉 Inference Complete! Video saved to: {output_path}")

if __name__ == "__main__":
    main()