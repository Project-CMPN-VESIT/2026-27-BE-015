"""
audio_encoder.py — Encode audio waveforms to embeddings using pretrained Wav2Vec2.
Uses HuggingFace Transformers for the facebook/wav2vec2-base model.
"""

import torch
import numpy as np
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import logging

logger = logging.getLogger(__name__)


class AudioEncoder:
    """
    Encode audio waveform into dense embeddings using pretrained Wav2Vec2.
    Outputs per-frame embeddings aligned with video frame rate.
    """

    MODEL_NAME = "facebook/wav2vec2-base"

    def __init__(self, device: str = None):
        """
        Args:
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Loading Wav2Vec2 model: {self.MODEL_NAME}")
        self.processor = Wav2Vec2Processor.from_pretrained(self.MODEL_NAME)
        self.model = Wav2Vec2Model.from_pretrained(self.MODEL_NAME).to(self.device)
        self.model.eval()

        self.embedding_dim = self.model.config.hidden_size  # 768
        logger.info(
            f"Wav2Vec2 loaded on {self.device}, "
            f"embedding dim: {self.embedding_dim}"
        )

    @torch.no_grad()
    def encode(self, audio: np.ndarray, sr: int = 16000) -> torch.Tensor:
        """
        Encode full audio waveform to Wav2Vec2 hidden states.

        Args:
            audio: 1-D float32 numpy array
            sr: sample rate (must be 16000 for Wav2Vec2)

        Returns:
            Tensor of shape (1, T, 768) where T is the number of
            output time steps from Wav2Vec2
        """
        inputs = self.processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self.device)

        outputs = self.model(input_values)
        # last_hidden_state: (1, T, 768)
        embeddings = outputs.last_hidden_state

        logger.info(
            f"Audio encoded: {len(audio)} samples -> "
            f"{embeddings.shape[1]} time steps x {embeddings.shape[2]}d"
        )
        return embeddings

    @torch.no_grad()
    def encode_chunked(
        self, audio: np.ndarray, sr: int = 16000, chunk_sec: float = 5.0
    ) -> torch.Tensor:
        """
        Encode audio in chunks to reduce memory usage on long clips.

        Args:
            audio: 1-D float32 numpy array
            sr: sample rate
            chunk_sec: duration of each chunk in seconds

        Returns:
            Tensor of shape (1, T_total, 768)
        """
        chunk_samples = int(chunk_sec * sr)
        all_embeddings = []

        for start in range(0, len(audio), chunk_samples):
            end = min(start + chunk_samples, len(audio))
            chunk = audio[start:end]

            if len(chunk) < 400:  # Minimum length for Wav2Vec2
                chunk = np.pad(chunk, (0, 400 - len(chunk)))

            emb = self.encode(chunk, sr)
            all_embeddings.append(emb)

        combined = torch.cat(all_embeddings, dim=1)
        logger.info(f"Chunked encoding: {combined.shape[1]} total time steps")
        return combined

    @torch.no_grad()
    def get_frame_aligned_embeddings(
        self, audio: np.ndarray, sr: int, fps: float, n_frames: int
    ) -> torch.Tensor:
        """
        Get audio embeddings aligned to video frames.
        Wav2Vec2 outputs ~50 features/sec. This resamples to match fps.

        Args:
            audio: 1-D float32 numpy array
            sr: sample rate
            fps: video frame rate
            n_frames: number of video frames

        Returns:
            Tensor of shape (n_frames, 768)
        """
        # Encode full audio
        embeddings = self.encode_chunked(audio, sr)  # (1, T, 768)
        embeddings = embeddings.squeeze(0)  # (T, 768)

        T = embeddings.shape[0]

        # Wav2Vec2 outputs at ~49.9 Hz (320 sample stride at 16kHz)
        wav2vec_fps = sr / 320.0

        # Compute frame indices into the embedding sequence
        frame_indices = torch.arange(n_frames, dtype=torch.float32)
        # Map video frame to Wav2Vec2 time step
        emb_indices = (frame_indices / fps * wav2vec_fps).long()
        emb_indices = emb_indices.clamp(0, T - 1)

        frame_embeddings = embeddings[emb_indices]  # (n_frames, 768)
        logger.info(f"Frame-aligned embeddings: {frame_embeddings.shape}")
        return frame_embeddings
