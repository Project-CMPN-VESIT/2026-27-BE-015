import sys, os
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from models.unet import AudioGuidedVideoUNet

# CPU is fine for this tiny test
device = "cpu" 
print(f"Testing U-Net Math on: {device.upper()}...")

model = AudioGuidedVideoUNet().to(device)

# DUMMY TENSORS (Batch Size 1, Single Frame)
dummy_noisy = torch.randn(1, 4, 64, 64).to(device)
dummy_context = torch.randn(1, 4, 64, 64).to(device)
dummy_audio = torch.randn(1, 199, 768).to(device) # The whole 4-second audio context
dummy_timesteps = torch.tensor([500]).to(device)

print("Pushing frame through Cross-Attention layers...")
output = model(dummy_noisy, dummy_context, dummy_timesteps, dummy_audio)

print(f"✅ Output Shape: {output.shape}") 
assert output.shape == (1, 4, 64, 64), "Math Error!"
print("\n🚀 U-NET MATH TEST PASSED! Memory optimized.")