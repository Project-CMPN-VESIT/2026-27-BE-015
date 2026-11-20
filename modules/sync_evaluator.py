"""
sync_evaluator.py — Evaluate audio-visual synchronization quality.

Computes cross-modal similarity between audio embeddings (Wav2Vec2)
and video features (mouth region intensity/motion) as a proxy for
SyncNet-style lip sync scoring (LSE-D metric).

Also computes standard quality metrics:
  - SSIM (Structural Similarity Index) for video frames
  - PSNR (Peak Signal-to-Noise Ratio) for video frames
  - STOI (Short-Time Objective Intelligibility) for audio
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import logging

logger = logging.getLogger(__name__)


def _safe_import_stoi():
    """Try to import pystoi, return None if unavailable."""
    try:
        from pystoi import stoi
        return stoi
    except ImportError:
        logger.warning("pystoi not installed. STOI metric will be skipped.")
        return None


class SyncEvaluator:
    """
    Evaluates the quality of audio-visual restoration using multiple metrics.
    """

    def __init__(self, audio_encoder=None, device: str = None):
        """
        Args:
            audio_encoder: AudioEncoder instance (for LSE-D computation)
            device: torch device
        """
        self.audio_encoder = audio_encoder
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def compute_ssim(
        self, original_frames: np.ndarray, restored_frames: np.ndarray
    ) -> dict:
        """
        Compute SSIM between original and restored video frames.

        Returns:
            dict with 'mean_ssim', 'per_frame_ssim', 'min_ssim', 'max_ssim'
        """
        n_frames = min(len(original_frames), len(restored_frames))
        ssim_scores = []

        for i in range(n_frames):
            orig_gray = cv2.cvtColor(original_frames[i], cv2.COLOR_BGR2GRAY)
            rest_gray = cv2.cvtColor(restored_frames[i], cv2.COLOR_BGR2GRAY)

            score = ssim(orig_gray, rest_gray)
            ssim_scores.append(score)

        scores = np.array(ssim_scores)
        result = {
            "mean_ssim": float(scores.mean()),
            "min_ssim": float(scores.min()),
            "max_ssim": float(scores.max()),
            "std_ssim": float(scores.std()),
            "per_frame_ssim": scores.tolist(),
        }
        logger.info(f"SSIM: mean={result['mean_ssim']:.4f}, "
                     f"min={result['min_ssim']:.4f}, max={result['max_ssim']:.4f}")
        return result

    def compute_psnr(
        self, original_frames: np.ndarray, restored_frames: np.ndarray
    ) -> dict:
        """
        Compute PSNR between original and restored video frames.

        Returns:
            dict with 'mean_psnr', 'per_frame_psnr', 'min_psnr', 'max_psnr'
        """
        n_frames = min(len(original_frames), len(restored_frames))
        psnr_scores = []

        for i in range(n_frames):
            score = psnr(original_frames[i], restored_frames[i])
            psnr_scores.append(score)

        scores = np.array(psnr_scores)
        result = {
            "mean_psnr": float(scores.mean()),
            "min_psnr": float(scores.min()),
            "max_psnr": float(scores.max()),
            "std_psnr": float(scores.std()),
            "per_frame_psnr": scores.tolist(),
        }
        logger.info(f"PSNR: mean={result['mean_psnr']:.2f} dB, "
                     f"min={result['min_psnr']:.2f}, max={result['max_psnr']:.2f}")
        return result

    def compute_stoi(
        self,
        original_audio: np.ndarray,
        restored_audio: np.ndarray,
        sr: int,
    ) -> dict:
        """
        Compute STOI between original and restored audio.

        Returns:
            dict with 'stoi_score' or None if pystoi unavailable
        """
        stoi_fn = _safe_import_stoi()
        if stoi_fn is None:
            return {"stoi_score": None, "note": "pystoi not installed"}

        # Align lengths
        min_len = min(len(original_audio), len(restored_audio))
        orig = original_audio[:min_len]
        rest = restored_audio[:min_len]

        try:
            score = stoi_fn(orig, rest, sr, extended=False)
            result = {"stoi_score": float(score)}
            logger.info(f"STOI: {score:.4f}")
            return result
        except Exception as e:
            logger.warning(f"STOI computation failed: {e}")
            return {"stoi_score": None, "error": str(e)}

    def _extract_mouth_features(
        self, frames: np.ndarray, mouth_bboxes: list
    ) -> np.ndarray:
        """
        Extract simple features from mouth regions:
        mean intensity + horizontal/vertical gradients.

        Returns:
            (n_frames, feature_dim) numpy array
        """
        features = []
        feature_dim = 64  # Resize mouth to 8x8 and flatten

        for i, bbox in enumerate(mouth_bboxes):
            if bbox is None:
                features.append(np.zeros(feature_dim))
                continue

            x1, y1, x2, y2 = bbox
            mouth = cv2.cvtColor(frames[i, y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

            if mouth.size == 0:
                features.append(np.zeros(feature_dim))
                continue

            # Resize to fixed size and flatten
            resized = cv2.resize(mouth, (8, 8)).astype(np.float32) / 255.0
            features.append(resized.flatten())

        return np.array(features)

    def compute_sync_score(
        self,
        frames: np.ndarray,
        audio: np.ndarray,
        sr: int,
        fps: float,
        mouth_bboxes: list,
    ) -> dict:
        """
        Compute lip sync score (LSE-D proxy) using cross-modal
        correlation between audio embeddings and mouth visual features.

        A higher score indicates better audio-visual synchronization.

        Returns:
            dict with 'lse_d', 'mean_correlation', 'per_frame_correlation'
        """
        if self.audio_encoder is None:
            logger.warning("No audio encoder provided, skipping sync score")
            return {"lse_d": None, "note": "No audio encoder"}

        n_frames = len(frames)

        # Get audio embeddings aligned to video frames
        audio_emb = self.audio_encoder.get_frame_aligned_embeddings(
            audio, sr, fps, n_frames
        )  # (n_frames, 768)

        # Get mouth visual features
        mouth_features = self._extract_mouth_features(frames, mouth_bboxes)
        mouth_features = torch.from_numpy(mouth_features).float()  # (n_frames, 64)

        # Project audio embeddings to same dim as mouth features
        # Simple linear projection (not learned, uses PCA-like approach)
        audio_emb_np = audio_emb.cpu().numpy()  # (n_frames, 768)

        # Reduce audio dim to match mouth features via averaging blocks
        block_size = audio_emb_np.shape[1] // mouth_features.shape[1]
        audio_reduced = audio_emb_np.reshape(n_frames, -1, block_size).mean(axis=2)
        audio_reduced = torch.from_numpy(audio_reduced).float()

        # Normalize both
        audio_norm = F.normalize(audio_reduced, dim=1)
        mouth_norm = F.normalize(mouth_features, dim=1)

        # Compute per-frame cosine similarity
        per_frame_sim = (audio_norm * mouth_norm).sum(dim=1).numpy()

        # LSE-D: mean of (1 - similarity) — lower is better sync
        lse_d = float(np.mean(1.0 - per_frame_sim))
        mean_corr = float(np.mean(per_frame_sim))

        result = {
            "lse_d": lse_d,
            "mean_correlation": mean_corr,
            "std_correlation": float(np.std(per_frame_sim)),
            "per_frame_correlation": per_frame_sim.tolist(),
        }
        logger.info(f"Sync Score — LSE-D: {lse_d:.4f}, Correlation: {mean_corr:.4f}")
        return result

    def evaluate_all(
        self,
        original_frames: np.ndarray,
        restored_frames: np.ndarray,
        original_audio: np.ndarray,
        restored_audio: np.ndarray,
        sr: int,
        fps: float,
        mouth_bboxes: list,
    ) -> dict:
        """
        Run all evaluation metrics.

        Returns:
            dict with all metrics
        """
        logger.info("=" * 60)
        logger.info("EVALUATION METRICS")
        logger.info("=" * 60)

        results = {}

        # Video quality metrics
        logger.info("--- Video Quality ---")
        results["ssim"] = self.compute_ssim(original_frames, restored_frames)
        results["psnr"] = self.compute_psnr(original_frames, restored_frames)

        # Audio quality metric
        logger.info("--- Audio Quality ---")
        results["stoi"] = self.compute_stoi(original_audio, restored_audio, sr)

        # Sync metric
        logger.info("--- Audio-Visual Sync ---")
        results["sync"] = self.compute_sync_score(
            restored_frames, restored_audio, sr, fps, mouth_bboxes
        )

        logger.info("=" * 60)
        logger.info("SUMMARY:")
        logger.info(f"  SSIM:  {results['ssim']['mean_ssim']:.4f}")
        logger.info(f"  PSNR:  {results['psnr']['mean_psnr']:.2f} dB")
        if results["stoi"]["stoi_score"] is not None:
            logger.info(f"  STOI:  {results['stoi']['stoi_score']:.4f}")
        if results["sync"]["lse_d"] is not None:
            logger.info(f"  LSE-D: {results['sync']['lse_d']:.4f}")
        logger.info("=" * 60)

        return results
