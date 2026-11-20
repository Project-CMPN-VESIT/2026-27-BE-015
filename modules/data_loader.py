"""
data_loader.py — Load video files, extract frames and audio waveform.
Uses OpenCV for video frames and ffmpeg + librosa for audio extraction.
"""

import cv2
import librosa
import numpy as np
import subprocess
import tempfile
import os
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads a video file, extracts individual frames as numpy arrays
    and the audio waveform as a 1-D numpy array.
    """

    def __init__(self, video_path: str, target_sr: int = 16000, max_frames: int = 0, start_frame: int = 0):
        """
        Args:
            video_path: Path to the input video file (.mp4, .avi, etc.)
            target_sr: Target sample rate for audio extraction (default 16 kHz for speech models)
            max_frames: Maximum number of frames to extract (0 = all frames)
            start_frame: Frame index to start extraction from (for skipping intros)
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        self.video_path = video_path
        self.target_sr = target_sr
        self.max_frames = max_frames
        self.start_frame = start_frame

    def extract_frames(self) -> tuple:
        """
        Extract all frames from the video using OpenCV.

        Returns:
            frames: numpy array of shape (N, H, W, 3) in BGR format
            fps: frames per second of the source video
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Video: {width}x{height} @ {fps:.1f} fps, {total} total frames")

        # Skip to start_frame
        if self.start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            logger.info(f"Skipping to frame {self.start_frame}")

        frames = []
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            count += 1
            if self.max_frames > 0 and count >= self.max_frames:
                break

        cap.release()
        logger.info(f"Extracted {len(frames)} frames (from frame {self.start_frame})")

        if len(frames) == 0:
            raise RuntimeError("No frames extracted from video")

        self._fps = fps  # Store for audio sync
        return np.array(frames), fps

    def extract_audio(self) -> tuple:
        """
        Extract audio waveform from the video using ffmpeg subprocess and librosa.
        If start_frame is set, extracts only the corresponding audio segment.

        Returns:
            audio: 1-D numpy float32 array of the waveform
            sr: sample rate
        """
        temp_dir = tempfile.mkdtemp()
        temp_audio = os.path.join(temp_dir, "audio.wav")

        # Compute time offset if start_frame is set
        fps = getattr(self, '_fps', 30.0)
        start_time = self.start_frame / fps if self.start_frame > 0 else 0

        cmd = [
            "ffmpeg",
        ]
        if start_time > 0:
            cmd += ["-ss", f"{start_time:.3f}"]
        cmd += [
            "-i", self.video_path,
            "-ac", "1",                    # mono
            "-ar", str(self.target_sr),    # target sample rate
            "-vn",                         # no video
            "-acodec", "pcm_s16le",        # 16-bit PCM
            temp_audio,
            "-y",                          # overwrite
            "-loglevel", "error",
        ]

        logger.info(f"Extracting audio with ffmpeg (offset {start_time:.2f}s)...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr}")

        audio, sr = librosa.load(temp_audio, sr=self.target_sr)
        logger.info(f"Audio: {len(audio)} samples @ {sr} Hz ({len(audio)/sr:.2f} sec)")

        # Cleanup
        os.remove(temp_audio)
        os.rmdir(temp_dir)

        return audio, sr

    def load(self) -> dict:
        """
        Full extraction pipeline: frames + audio.

        Returns:
            dict with keys: frames, fps, audio, sr, video_path
        """
        logger.info(f"Loading video: {self.video_path}")

        frames, fps = self.extract_frames()
        audio, sr = self.extract_audio()

        # Compute expected audio length and trim/pad to match
        expected_audio_len = int(len(frames) / fps * sr)
        if len(audio) > expected_audio_len:
            audio = audio[:expected_audio_len]
        elif len(audio) < expected_audio_len:
            audio = np.pad(audio, (0, expected_audio_len - len(audio)))

        return {
            "frames": frames,
            "fps": fps,
            "audio": audio,
            "sr": sr,
            "video_path": self.video_path,
        }
