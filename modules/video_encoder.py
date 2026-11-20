"""
video_encoder.py — Encode video frames to latent space using the pretrained
VAE from Stable Diffusion (HuggingFace diffusers).
"""

import torch
import torch.nn.functional as F
import numpy as np
from diffusers import AutoencoderKL
import logging

logger = logging.getLogger(__name__)


class VideoEncoder:
    """
    Encode video frames into latent representations using the pretrained
    Stable Diffusion VAE. Also provides decode functionality for reconstruction.
    """

    MODEL_NAME = "stabilityai/sd-vae-ft-mse"
    LATENT_CHANNELS = 4
    DOWNSCALE_FACTOR = 8  # VAE downscales by 8x in each spatial dim

    def __init__(self, device: str = None, target_size: int = 256):
        """
        Args:
            device: 'cuda', 'cpu', or None (auto-detect)
            target_size: Resize frames to this square size before encoding
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.target_size = target_size

        logger.info(f"Loading VAE: {self.MODEL_NAME}")
        self.vae = AutoencoderKL.from_pretrained(self.MODEL_NAME).to(self.device)
        self.vae.eval()

        self.latent_size = target_size // self.DOWNSCALE_FACTOR
        logger.info(
            f"VAE loaded on {self.device}, "
            f"target: {target_size}x{target_size} -> "
            f"latent: {self.latent_size}x{self.latent_size}x{self.LATENT_CHANNELS}"
        )

    def _preprocess(self, frames: np.ndarray) -> torch.Tensor:
        """
        Preprocess numpy frames (BGR, uint8) to VAE input format.

        Args:
            frames: (N, H, W, 3) uint8 BGR

        Returns:
            Tensor (N, 3, target_size, target_size), float32, range [-1, 1]
        """
        import cv2

        processed = []
        for frame in frames:
            # BGR -> RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize
            resized = cv2.resize(rgb, (self.target_size, self.target_size))
            processed.append(resized)

        # (N, H, W, 3) -> (N, 3, H, W)
        tensor = torch.from_numpy(np.array(processed)).float()
        tensor = tensor.permute(0, 3, 1, 2) / 255.0
        # Normalize to [-1, 1]
        tensor = tensor * 2.0 - 1.0
        return tensor

    def _postprocess(self, tensor: torch.Tensor, original_size: tuple = None) -> np.ndarray:
        """
        Convert VAE output tensor back to numpy frames.

        Args:
            tensor: (N, 3, H, W) float32 in [-1, 1]
            original_size: (H, W) to resize back to

        Returns:
            (N, H, W, 3) uint8 BGR numpy array
        """
        import cv2

        # Denormalize
        tensor = (tensor + 1.0) / 2.0
        tensor = tensor.clamp(0, 1)

        # (N, 3, H, W) -> (N, H, W, 3)
        frames = tensor.permute(0, 2, 3, 1).cpu().numpy()
        frames = (frames * 255).astype(np.uint8)

        if original_size is not None:
            resized = []
            for frame in frames:
                # RGB -> BGR
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                bgr = cv2.resize(bgr, (original_size[1], original_size[0]))
                resized.append(bgr)
            return np.array(resized)
        else:
            result = []
            for frame in frames:
                result.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            return np.array(result)

    @torch.no_grad()
    def encode(self, frames: np.ndarray, batch_size: int = 4) -> torch.Tensor:
        """
        Encode frames to VAE latent space.

        Args:
            frames: (N, H, W, 3) uint8 BGR numpy array
            batch_size: Process this many frames at once

        Returns:
            Tensor (N, 4, latent_h, latent_w) — latent representations
        """
        tensor = self._preprocess(frames)
        all_latents = []

        for i in range(0, len(tensor), batch_size):
            batch = tensor[i : i + batch_size].to(self.device)
            posterior = self.vae.encode(batch).latent_dist
            latents = posterior.mean  # Use mean for deterministic encoding
            all_latents.append(latents.cpu())

        combined = torch.cat(all_latents, dim=0)
        logger.info(f"Encoded {len(frames)} frames -> latents {combined.shape}")
        return combined

    @torch.no_grad()
    def decode(
        self, latents: torch.Tensor, original_size: tuple = None, batch_size: int = 4
    ) -> np.ndarray:
        """
        Decode latent representations back to frames.

        Args:
            latents: (N, 4, latent_h, latent_w) tensor
            original_size: (H, W) to resize decoded frames
            batch_size: Process this many at once

        Returns:
            (N, H, W, 3) uint8 BGR numpy array
        """
        all_frames = []

        for i in range(0, len(latents), batch_size):
            batch = latents[i : i + batch_size].to(self.device)
            decoded = self.vae.decode(batch).sample
            all_frames.append(decoded.cpu())

        combined = torch.cat(all_frames, dim=0)
        frames = self._postprocess(combined, original_size)
        logger.info(f"Decoded {len(latents)} latents -> frames {frames.shape}")
        return frames

    @torch.no_grad()
    def encode_region(
        self, frames: np.ndarray, bboxes: list, pad_size: int = 64
    ) -> list:
        """
        Encode specific regions (e.g., mouth) from frames.

        Args:
            frames: (N, H, W, 3) uint8 BGR
            bboxes: List of (x1, y1, x2, y2) per frame (or None)
            pad_size: Resize cropped region to this size for encoding

        Returns:
            List of latent tensors (one per valid bbox)
        """
        import cv2

        latents_list = []
        for i, bbox in enumerate(bboxes):
            if bbox is None:
                latents_list.append(None)
                continue

            x1, y1, x2, y2 = bbox
            region = frames[i, y1:y2, x1:x2].copy()

            # Resize region to square
            region_resized = cv2.resize(region, (pad_size, pad_size))
            region_rgb = cv2.cvtColor(region_resized, cv2.COLOR_BGR2RGB)

            # Normalize
            tensor = torch.from_numpy(region_rgb).float().permute(2, 0, 1) / 255.0
            tensor = tensor * 2.0 - 1.0
            tensor = tensor.unsqueeze(0)

            # Pad to minimum VAE size (must be multiple of 8)
            if pad_size < 64:
                p = (64 - pad_size) // 2
                tensor = F.pad(tensor, (p, p, p, p), mode="reflect")

            tensor = tensor.to(self.device)
            posterior = self.vae.encode(tensor).latent_dist
            latent = posterior.mean.cpu()
            latents_list.append(latent)

        valid = sum(1 for l in latents_list if l is not None)
        logger.info(f"Encoded {valid}/{len(bboxes)} mouth regions")
        return latents_list
