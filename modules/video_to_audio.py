"""
video_to_audio.py — Restore corrupted audio segments using video (lip) information.

Improved Pipeline (per-segment reconstruction):
1. Identify corrupted audio segments from the corruption mask
2. For short gaps (<30ms): waveform crossfade interpolation (seamless)
3. For longer gaps: per-segment STFT reconstruction with:
     - Cubic spline magnitude interpolation from surrounding context
     - Phase interpolation (unwrapped) for smooth phase continuity
     - iSTFT reconstruction preserving phase coherence
     - Optional Griffin-Lim refinement for very long gaps
4. Apply video-derived energy envelope from mouth motion for natural prosody
5. Smooth boundary transitions with overlap-add crossfading

The mouth motion (optical flow of lip regions) provides an energy envelope
that modulates the reconstructed audio to match expected speech patterns.
"""

import cv2
import numpy as np
import librosa
import logging
from scipy import interpolate as scipy_interpolate

logger = logging.getLogger(__name__)


class VideoToAudioRestorer:
    """
    Restores corrupted audio segments using per-segment spectral
    reconstruction with phase-aware interpolation, modulated by
    video-derived mouth motion energy.
    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_iter_griffin_lim: int = 60,
        interpolation_context: int = 5,
    ):
        """
        Args:
            n_fft: FFT window size
            hop_length: STFT hop length
            win_length: STFT window length
            n_iter_griffin_lim: Number of Griffin-Lim iterations
            interpolation_context: Number of context frames for spectral interpolation
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_iter_griffin_lim = n_iter_griffin_lim
        self.interpolation_context = interpolation_context

    def _compute_mouth_motion_energy(
        self, frames: np.ndarray, mouth_bboxes: list, fps: float, sr: int
    ) -> np.ndarray:
        """
        Compute mouth motion energy from consecutive frames using optical flow.
        Returns an energy envelope aligned to audio sample rate.

        Args:
            frames: (N, H, W, 3) uint8 BGR
            mouth_bboxes: list of (x1, y1, x2, y2) or None
            fps: video frame rate
            sr: audio sample rate

        Returns:
            1-D energy envelope array of length (N_frames * sr / fps)
        """
        n_frames = len(frames)
        frame_energies = np.zeros(n_frames)

        for i in range(1, n_frames):
            bbox = mouth_bboxes[i] if mouth_bboxes[i] is not None else mouth_bboxes[i - 1]
            prev_bbox = mouth_bboxes[i - 1] if mouth_bboxes[i - 1] is not None else bbox

            if bbox is None or prev_bbox is None:
                frame_energies[i] = frame_energies[i - 1]
                continue

            # Extract mouth regions
            x1, y1, x2, y2 = bbox
            px1, py1, px2, py2 = prev_bbox

            curr_mouth = cv2.cvtColor(frames[i, y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            prev_mouth = cv2.cvtColor(frames[i - 1, py1:py2, px1:px2], cv2.COLOR_BGR2GRAY)

            if curr_mouth.size == 0 or prev_mouth.size == 0:
                continue

            # Resize to common size
            target_size = (64, 32)
            curr_resized = cv2.resize(curr_mouth, target_size)
            prev_resized = cv2.resize(prev_mouth, target_size)

            # Compute absolute difference as motion energy
            diff = np.abs(curr_resized.astype(float) - prev_resized.astype(float))
            frame_energies[i] = np.mean(diff)

        # Normalize energy to [0, 1]
        max_energy = frame_energies.max()
        if max_energy > 0:
            frame_energies /= max_energy

        # Smooth the energy curve
        kernel_size = max(3, int(fps * 0.1))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones(kernel_size) / kernel_size
        frame_energies = np.convolve(frame_energies, kernel, mode="same")

        # Upsample to audio sample rate
        total_audio_samples = int(n_frames / fps * sr)
        audio_energy = np.interp(
            np.linspace(0, n_frames - 1, total_audio_samples),
            np.arange(n_frames),
            frame_energies,
        )

        logger.info(
            f"Mouth motion energy: mean={audio_energy.mean():.3f}, "
            f"max={audio_energy.max():.3f}"
        )

        return audio_energy

    # ================================================================
    # NEW: Per-segment reconstruction helpers
    # ================================================================

    def _find_corrupted_segments(self, mask: np.ndarray) -> list:
        """
        Find contiguous corrupted segments as (start, end) sample pairs.

        Args:
            mask: boolean array — True where corrupted

        Returns:
            List of (start_sample, end_sample) tuples
        """
        segments = []
        in_seg = False
        seg_start = 0

        for i in range(len(mask)):
            if mask[i] and not in_seg:
                seg_start = i
                in_seg = True
            elif not mask[i] and in_seg:
                segments.append((seg_start, i))
                in_seg = False

        if in_seg:
            segments.append((seg_start, len(mask)))

        return segments

    def _waveform_interpolate(
        self, before: np.ndarray, after: np.ndarray, gap_len: int
    ) -> np.ndarray:
        """
        For short gaps (<30ms): smooth waveform crossfade between the
        tail of 'before' and the head of 'after'.

        Uses sinusoidal (equal-power) crossfade for perceptually smooth
        transitions, which is standard in professional audio codecs.

        Args:
            before: audio samples immediately preceding the gap
            after: audio samples immediately following the gap
            gap_len: number of samples to fill

        Returns:
            1-D float32 array of length gap_len
        """
        if len(before) == 0 and len(after) == 0:
            return np.zeros(gap_len, dtype=np.float32)

        if len(before) == 0:
            # Fade in from silence to after
            fade = np.linspace(0, 1, gap_len).astype(np.float32)
            return after[0] * fade if len(after) > 0 else np.zeros(gap_len, dtype=np.float32)

        if len(after) == 0:
            # Fade out from before to silence
            fade = np.linspace(1, 0, gap_len).astype(np.float32)
            return before[-1] * fade

        # Equal-power (sinusoidal) crossfade
        t = np.linspace(0, np.pi / 2, gap_len).astype(np.float32)
        fade_out = np.cos(t)   # 1 -> 0
        fade_in = np.sin(t)    # 0 -> 1

        # Use a small tail/head from context for smoother interpolation
        n_tail = min(len(before), gap_len)
        n_head = min(len(after), gap_len)

        # Create the tail waveform (repeat last few samples pattern)
        if n_tail >= 2:
            tail = np.interp(
                np.linspace(0, n_tail - 1, gap_len),
                np.arange(n_tail),
                before[-n_tail:],
            )
        else:
            tail = np.full(gap_len, before[-1])

        # Create the head waveform
        if n_head >= 2:
            head = np.interp(
                np.linspace(0, n_head - 1, gap_len),
                np.arange(n_head),
                after[:n_head],
            )
        else:
            head = np.full(gap_len, after[0])

        result = tail * fade_out + head * fade_in
        return result.astype(np.float32)

    def _reconstruct_segment(
        self,
        before_ctx: np.ndarray,
        after_ctx: np.ndarray,
        gap_len: int,
        sr: int,
    ) -> np.ndarray:
        """
        Reconstruct a single corrupted audio segment using local STFT
        with phase-aware interpolation (NO global Griffin-Lim).

        Strategy:
        1. Build composite signal: [before_context | zeros | after_context]
        2. Compute STFT of the composite
        3. Identify which STFT frames fall inside the gap
        4. Interpolate MAGNITUDE using cubic spline from surrounding context
        5. Interpolate PHASE (unwrapped) linearly for phase continuity
        6. Reconstruct complex STFT and invert with iSTFT
        7. Extract the gap portion from the reconstructed signal

        This preserves phase coherence and avoids the metallic artifacts
        of running Griffin-Lim on the full signal.

        Args:
            before_ctx: audio samples before the gap (clean)
            after_ctx: audio samples after the gap (clean)
            gap_len: number of missing samples
            sr: sample rate

        Returns:
            1-D float32 array of length gap_len
        """
        # Ensure minimum context length for stable STFT
        min_ctx = self.n_fft * 2
        if len(before_ctx) < min_ctx and len(after_ctx) < min_ctx:
            # Not enough context for spectral method, fall back to waveform
            return self._waveform_interpolate(before_ctx, after_ctx, gap_len)

        # Build composite: [before | gap(zeros) | after]
        composite = np.concatenate([
            before_ctx,
            np.zeros(gap_len, dtype=np.float32),
            after_ctx,
        ])

        # STFT of the composite
        stft = librosa.stft(
            composite,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        n_bins, n_frames = stft.shape

        # Determine which STFT frames are in the gap region
        gap_sample_start = len(before_ctx)
        gap_sample_end = gap_sample_start + gap_len

        gap_frame_mask = np.zeros(n_frames, dtype=bool)
        for t in range(n_frames):
            # Frame center position in samples
            frame_center = t * self.hop_length + self.win_length // 2
            if gap_sample_start <= frame_center < gap_sample_end:
                gap_frame_mask[t] = True

        gap_indices = np.where(gap_frame_mask)[0]

        if len(gap_indices) == 0:
            # Gap smaller than one STFT frame, use waveform interpolation
            return self._waveform_interpolate(before_ctx, after_ctx, gap_len)

        # Find clean context frames (before and after the gap)
        all_before = np.where(~gap_frame_mask & (np.arange(n_frames) < gap_indices[0]))[0]
        all_after = np.where(~gap_frame_mask & (np.arange(n_frames) > gap_indices[-1]))[0]

        # Use up to N context frames from each side
        ctx_n = max(self.interpolation_context, 5)
        before_frames = all_before[-ctx_n:] if len(all_before) > 0 else np.array([], dtype=int)
        after_frames = all_after[:ctx_n] if len(all_after) > 0 else np.array([], dtype=int)

        known_frames = np.concatenate([before_frames, after_frames]).astype(int)

        if len(known_frames) < 2:
            # Not enough context frames for interpolation
            return self._waveform_interpolate(before_ctx, after_ctx, gap_len)

        known_x = known_frames.astype(float)

        # Interpolation per frequency bin
        for b in range(n_bins):
            known_mag = magnitude[b, known_frames]
            known_ph = phase[b, known_frames]

            # --- Magnitude: cubic spline (clamped to non-negative) ---
            if len(known_x) >= 4:
                try:
                    mag_fn = scipy_interpolate.interp1d(
                        known_x, known_mag, kind="cubic",
                        fill_value="extrapolate", bounds_error=False,
                    )
                except ValueError:
                    mag_fn = scipy_interpolate.interp1d(
                        known_x, known_mag, kind="linear",
                        fill_value="extrapolate", bounds_error=False,
                    )
            else:
                mag_fn = scipy_interpolate.interp1d(
                    known_x, known_mag, kind="linear",
                    fill_value="extrapolate", bounds_error=False,
                )

            # --- Phase: linear interpolation on unwrapped phase ---
            unwrapped_ph = np.unwrap(known_ph)
            phase_fn = scipy_interpolate.interp1d(
                known_x, unwrapped_ph, kind="linear",
                fill_value="extrapolate", bounds_error=False,
            )

            # Fill gap frames
            gap_x = gap_indices.astype(float)
            magnitude[b, gap_indices] = np.maximum(0.0, mag_fn(gap_x))
            phase[b, gap_indices] = phase_fn(gap_x)

        # Reconstruct complex STFT with interpolated magnitude + phase
        reconstructed_stft = magnitude * np.exp(1j * phase)

        # Optional: run a few Griffin-Lim refinement iterations for very long gaps
        # This refines the phase estimate while keeping the magnitude fixed
        if gap_len > int(0.1 * sr):  # > 100ms gaps
            n_refine = min(self.n_iter_griffin_lim, 30)
            for _ in range(n_refine):
                audio_est = librosa.istft(
                    reconstructed_stft,
                    hop_length=self.hop_length,
                    win_length=self.win_length,
                    n_fft=self.n_fft,
                    length=len(composite),
                )
                stft_est = librosa.stft(
                    audio_est,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    win_length=self.win_length,
                )
                # Keep interpolated magnitude, update phase only for gap frames
                new_phase = np.angle(stft_est)
                for t in gap_indices:
                    reconstructed_stft[:, t] = magnitude[:, t] * np.exp(1j * new_phase[:, t])

        # iSTFT -> time domain
        reconstructed = librosa.istft(
            reconstructed_stft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_fft=self.n_fft,
            length=len(composite),
        )

        # Extract the gap portion
        gap_audio = reconstructed[gap_sample_start:gap_sample_end]

        if len(gap_audio) < gap_len:
            gap_audio = np.pad(gap_audio, (0, gap_len - len(gap_audio)))

        return gap_audio[:gap_len].astype(np.float32)

    def _energy_match(
        self,
        gap_fill: np.ndarray,
        before_ctx: np.ndarray,
        after_ctx: np.ndarray,
    ) -> np.ndarray:
        """
        Scale the gap fill so its RMS energy matches the average RMS
        of the surrounding context. This is the single most important
        step for making fills sound natural — without it, the fills
        are much quieter than the surrounding speech.

        Args:
            gap_fill: reconstructed gap audio
            before_ctx: clean audio before the gap
            after_ctx: clean audio after the gap

        Returns:
            Energy-matched gap fill
        """
        # Compute RMS of context (use whichever side is available)
        ctx_parts = []
        if len(before_ctx) > 0:
            ctx_parts.append(before_ctx)
        if len(after_ctx) > 0:
            ctx_parts.append(after_ctx)

        if len(ctx_parts) == 0:
            return gap_fill

        context = np.concatenate(ctx_parts)
        ctx_rms = np.sqrt(np.mean(context ** 2)) + 1e-10
        fill_rms = np.sqrt(np.mean(gap_fill ** 2)) + 1e-10

        # Scale fill to match context energy
        scale = ctx_rms / fill_rms
        # Clamp scale to avoid extreme amplification of noise
        scale = min(scale, 5.0)

        return (gap_fill * scale).astype(np.float32)

    def restore(self, corrupted_data: dict) -> np.ndarray:
        """
        Main audio restoration pipeline (per-segment reconstruction).

        For each corrupted segment:
        1. Extract clean context audio from both sides of the gap
        2. Short gaps (<30ms): smooth waveform crossfade interpolation
        3. Longer gaps: per-segment spectral reconstruction with
           cubic magnitude interpolation + phase-aware iSTFT
        4. Energy-match the fill to surrounding context RMS
        5. Crossfade at boundaries for seamless transitions
        6. Apply video-derived mouth motion energy as a BOOST
        7. Final safety pass: ensure no zeros remain in gap regions

        Args:
            corrupted_data: dict from StreamCorruptor.corrupt()

        Returns:
            restored_audio: 1-D float32 numpy array
        """
        corrupted_audio = corrupted_data["corrupted_audio"]
        audio_mask = corrupted_data["audio_corruption_mask"]
        original_audio = corrupted_data["original_audio"]
        frames = corrupted_data["original_frames"]
        mouth_bboxes = corrupted_data["mouth_bboxes"]
        fps = corrupted_data["fps"]
        sr = corrupted_data["sr"]

        if not audio_mask.any():
            logger.info("No audio corruption detected, returning original")
            return corrupted_audio.copy()

        logger.info("Starting audio restoration (per-segment reconstruction)...")

        # Start with the corrupted audio — uncorrupted regions are already clean
        restored = corrupted_audio.copy()

        # Find all corrupted segments
        segments = self._find_corrupted_segments(audio_mask)
        logger.info(f"Found {len(segments)} corrupted segments to restore")

        # Context length: 200ms on each side of every gap (more = better)
        context_len = int(0.20 * sr)
        short_gap_threshold = int(0.03 * sr)  # 30ms

        for seg_idx, (seg_start, seg_end) in enumerate(segments):
            gap_len = seg_end - seg_start

            # Get clean context from each side (from corrupted_audio,
            # which is clean in non-corrupted regions)
            ctx_start = max(0, seg_start - context_len)
            ctx_end = min(len(corrupted_audio), seg_end + context_len)

            before_ctx = corrupted_audio[ctx_start:seg_start]
            after_ctx = corrupted_audio[seg_end:ctx_end]

            # Choose reconstruction method based on gap length
            if gap_len < short_gap_threshold:
                # Short gap: waveform crossfade (perceptually seamless)
                gap_fill = self._waveform_interpolate(before_ctx, after_ctx, gap_len)
            else:
                # Longer gap: spectral reconstruction with phase awareness
                gap_fill = self._reconstruct_segment(before_ctx, after_ctx, gap_len, sr)

            # CRITICAL: Match the energy of the fill to surrounding context
            # Without this, the fills are much quieter than the speech around them
            gap_fill = self._energy_match(gap_fill, before_ctx, after_ctx)

            # Place reconstructed audio into the gap
            restored[seg_start:seg_end] = gap_fill

            # Smooth crossfade at the leading boundary (10ms)
            cf_samples = min(int(0.010 * sr), len(before_ctx) // 2, gap_len // 4)
            if cf_samples > 1:
                fade = np.sin(np.linspace(0, np.pi / 2, cf_samples)).astype(np.float32)
                restored[seg_start:seg_start + cf_samples] = (
                    before_ctx[-cf_samples:] * (1 - fade) + gap_fill[:cf_samples] * fade
                )

            # Smooth crossfade at the trailing boundary (10ms)
            cf_samples = min(int(0.010 * sr), len(after_ctx) // 2, gap_len // 4)
            if cf_samples > 1:
                fade = np.cos(np.linspace(0, np.pi / 2, cf_samples)).astype(np.float32)
                restored[seg_end - cf_samples:seg_end] = (
                    gap_fill[-cf_samples:] * fade + after_ctx[:cf_samples] * (1 - fade)
                )

        logger.info(f"Segment reconstruction complete for {len(segments)} gaps")

        # Apply mouth motion energy as a BOOST to restored segments
        # (lip movement = speech activity = fill should be louder)
        motion_energy = self._compute_mouth_motion_energy(
            frames, mouth_bboxes, fps, sr
        )

        target_len = len(restored)
        if len(motion_energy) > target_len:
            motion_energy = motion_energy[:target_len]
        elif len(motion_energy) < target_len:
            motion_energy = np.pad(
                motion_energy, (0, target_len - len(motion_energy))
            )

        # Use mouth motion as a BOOST (1.0 to 1.3x), NOT a reduction
        energy_boost = 1.0 + motion_energy * 0.3  # Scale to [1.0, 1.3]
        for seg_start, seg_end in segments:
            e = min(seg_end, target_len)
            restored[seg_start:e] *= energy_boost[seg_start:e]

        # SAFETY: Ensure no zeros remain in corrupted regions
        # Fill any remaining zeros with very low-level shaped noise
        remaining_zeros = audio_mask & (np.abs(restored) < 1e-10)
        if remaining_zeros.any():
            n_remaining = remaining_zeros.sum()
            # Generate comfort noise matching the overall signal level
            signal_rms = np.sqrt(np.mean(restored[~audio_mask] ** 2)) + 1e-10
            comfort_noise = np.random.randn(len(restored)).astype(np.float32) * signal_rms * 0.05
            restored[remaining_zeros] = comfort_noise[remaining_zeros]
            logger.info(f"Filled {n_remaining} remaining zero samples with comfort noise")

        # Normalize to prevent clipping
        max_val = np.abs(restored).max()
        if max_val > 1.0:
            restored = restored / max_val * 0.95

        logger.info(f"Audio restoration complete: {len(restored)} samples")
        return restored
