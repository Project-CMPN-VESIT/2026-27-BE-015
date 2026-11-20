"""
output_writer.py — Merge restored video frames and audio into final output video.
Uses OpenCV for frame writing and ffmpeg for audio muxing.
"""

import cv2
import numpy as np
import subprocess
import os
import tempfile
import logging

logger = logging.getLogger(__name__)


class OutputWriter:
    """
    Writes restored frames + audio to a final output video file
    using OpenCV and ffmpeg.
    """

    def __init__(self, output_dir: str = "outputs"):
        """
        Args:
            output_dir: Directory to save output files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _write_frames_to_video(
        self, frames: np.ndarray, fps: float, output_path: str
    ) -> str:
        """
        Write numpy frames to a video file using OpenCV.

        Args:
            frames: (N, H, W, 3) uint8 BGR
            fps: frame rate
            output_path: output file path

        Returns:
            Path to the written video
        """
        h, w = frames.shape[1], frames.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        if not writer.isOpened():
            raise RuntimeError(f"Cannot create video writer: {output_path}")

        for frame in frames:
            writer.write(frame)

        writer.release()
        logger.info(f"Wrote {len(frames)} frames to {output_path}")
        return output_path

    def _write_audio(
        self, audio: np.ndarray, sr: int, output_path: str
    ) -> str:
        """
        Write audio waveform to a WAV file.

        Args:
            audio: 1-D float32 array
            sr: sample rate
            output_path: output file path

        Returns:
            Path to the written audio
        """
        import soundfile as sf

        # Normalize to prevent clipping
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val * 0.95

        sf.write(output_path, audio, sr)
        logger.info(f"Wrote audio ({len(audio)} samples) to {output_path}")
        return output_path

    def _mux_audio_video(
        self, video_path: str, audio_path: str, output_path: str
    ) -> str:
        """
        Merge video and audio tracks using ffmpeg.

        Args:
            video_path: path to the video file (no audio)
            audio_path: path to the audio file
            output_path: final output path

        Returns:
            Path to muxed output
        """
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            output_path,
            "-y",
            "-loglevel", "error",
        ]

        logger.info(f"Muxing video + audio -> {output_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg muxing failed: {result.stderr}")

        file_size = os.path.getsize(output_path)
        logger.info(f"Output video: {output_path} ({file_size / 1024 / 1024:.1f} MB)")
        return output_path

    def write(
        self,
        frames: np.ndarray,
        audio: np.ndarray,
        fps: float,
        sr: int,
        output_name: str = "restored_output",
    ) -> dict:
        """
        Full output pipeline: write frames + audio + mux together.

        Args:
            frames: (N, H, W, 3) uint8 BGR restored frames
            audio: 1-D float32 restored audio
            fps: frame rate
            sr: audio sample rate
            output_name: base name for output files

        Returns:
            dict with paths to all output files
        """
        temp_video = os.path.join(self.output_dir, f"{output_name}_video_only.mp4")
        temp_audio = os.path.join(self.output_dir, f"{output_name}_audio.wav")
        final_output = os.path.join(self.output_dir, f"{output_name}.mp4")

        # Write video frames
        self._write_frames_to_video(frames, fps, temp_video)

        # Write audio
        self._write_audio(audio, sr, temp_audio)

        # Mux together
        self._mux_audio_video(temp_video, temp_audio, final_output)

        # Write corrupted versions for comparison
        return {
            "final_video": final_output,
            "video_only": temp_video,
            "audio_only": temp_audio,
        }

    def write_comparison(
        self,
        original_frames: np.ndarray,
        corrupted_frames: np.ndarray,
        restored_frames: np.ndarray,
        original_audio: np.ndarray,
        corrupted_audio: np.ndarray,
        restored_audio: np.ndarray,
        fps: float,
        sr: int,
        base_name: str = "comparison",
    ) -> dict:
        """
        Write original, corrupted, and restored videos for side-by-side comparison.

        Audio logic:
        - Original video  -> original audio
        - Corrupted video  -> corrupted audio (both streams degraded)
        - Restored video   -> ORIGINAL audio (audio was the clean input that
                              drove the lip restoration, so it stays intact)
        - Audio restoration is saved as a separate file for demo purposes

        Returns:
            dict with paths to all comparison files
        """
        paths = {}

        # Original
        logger.info("Writing original video...")
        paths["original"] = self.write(
            original_frames, original_audio, fps, sr, f"{base_name}_original"
        )

        # Corrupted (both video + audio degraded)
        logger.info("Writing corrupted video...")
        paths["corrupted"] = self.write(
            corrupted_frames, corrupted_audio, fps, sr, f"{base_name}_corrupted"
        )

        # Restored video + ORIGINAL audio
        # (Audio is the clean signal that DRIVES the video restoration,
        #  so there's no reason to degrade it in the output)
        logger.info("Writing restored video (with original audio)...")
        paths["restored"] = self.write(
            restored_frames, original_audio, fps, sr, f"{base_name}_restored"
        )

        # Also save the video-to-audio restoration as a separate demo
        # (original video frames + Griffin-Lim restored audio)
        logger.info("Writing audio restoration demo...")
        paths["audio_restored"] = self.write(
            original_frames, restored_audio, fps, sr, f"{base_name}_audio_restored"
        )

        # Side-by-side comparison frame export
        logger.info("Creating side-by-side comparison...")
        comparison_frames = self._create_side_by_side(
            original_frames, corrupted_frames, restored_frames
        )
        comparison_video = os.path.join(
            self.output_dir, f"{base_name}_side_by_side.mp4"
        )
        self._write_frames_to_video(comparison_frames, fps, comparison_video)
        paths["side_by_side"] = comparison_video

        return paths

    def _create_side_by_side(
        self,
        original: np.ndarray,
        corrupted: np.ndarray,
        restored: np.ndarray,
        target_height: int = 360,
    ) -> np.ndarray:
        """
        Create side-by-side comparison frames: Original | Corrupted | Restored
        """
        n_frames = min(len(original), len(corrupted), len(restored))
        comparison_frames = []

        for i in range(n_frames):
            # Resize all to same height
            aspect = original[i].shape[1] / original[i].shape[0]
            target_w = int(target_height * aspect)

            o = cv2.resize(original[i], (target_w, target_height))
            c = cv2.resize(corrupted[i], (target_w, target_height))
            r = cv2.resize(restored[i], (target_w, target_height))

            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(o, "Original", (10, 30), font, 0.7, (0, 255, 0), 2)
            cv2.putText(c, "Corrupted", (10, 30), font, 0.7, (0, 0, 255), 2)
            cv2.putText(r, "Restored", (10, 30), font, 0.7, (255, 255, 0), 2)

            # Separator lines
            sep = np.ones((target_height, 3, 3), dtype=np.uint8) * 128

            combined = np.hstack([o, sep, c, sep, r])
            comparison_frames.append(combined)

        logger.info(f"Created {len(comparison_frames)} side-by-side frames")
        return np.array(comparison_frames)
