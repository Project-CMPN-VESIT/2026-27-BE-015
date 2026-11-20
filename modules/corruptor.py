"""
corruptor.py -- Simulate packet loss by corrupting video and audio streams.
Video: Mask the mouth region using MediaPipe Face Landmarker (Tasks API).
Audio: Zero out random contiguous segments to simulate dropouts.
"""

import cv2
import numpy as np
import os
import urllib.request
import logging

logger = logging.getLogger(__name__)

# MediaPipe lip landmark indices (inner + outer lip contour)
# These are the indices from the 478-point Face Landmarker model
OUTER_LIP_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
]
INNER_LIP_INDICES = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
]
ALL_LIP_INDICES = list(set(OUTER_LIP_INDICES + INNER_LIP_INDICES))

# Model URL for MediaPipe Face Landmarker
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
FACE_LANDMARKER_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "face_landmarker.task"
)


def _download_face_landmarker():
    """Download the MediaPipe Face Landmarker model if not present."""
    model_path = os.path.abspath(FACE_LANDMARKER_MODEL_PATH)
    if os.path.isfile(model_path):
        logger.info(f"Face Landmarker model found: {model_path}")
        return model_path

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    logger.info(f"Downloading Face Landmarker model to {model_path}...")
    urllib.request.urlretrieve(FACE_LANDMARKER_MODEL_URL, model_path)
    logger.info("Download complete.")
    return model_path


class StreamCorruptor:
    """
    Corrupts video frames (mouth region masking) and audio
    (random segment zeroing) to simulate packet loss.
    """

    def __init__(
        self,
        mouth_mask_expand: float = 1.4,
        audio_corrupt_ratio: float = 0.3,
        audio_min_gap_ms: int = 50,
        audio_max_gap_ms: int = 200,
        seed: int = 42,
    ):
        """
        Args:
            mouth_mask_expand: Multiplier to expand the detected mouth bounding box
            audio_corrupt_ratio: Fraction of audio to corrupt (0.0 to 1.0)
            audio_min_gap_ms: Minimum gap duration in milliseconds
            audio_max_gap_ms: Maximum gap duration in milliseconds
            seed: Random seed for reproducibility
        """
        self.mouth_mask_expand = mouth_mask_expand
        self.audio_corrupt_ratio = audio_corrupt_ratio
        self.audio_min_gap_ms = audio_min_gap_ms
        self.audio_max_gap_ms = audio_max_gap_ms
        self.rng = np.random.RandomState(seed)

        # Download and initialize Face Landmarker
        model_path = _download_face_landmarker()

        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceLandmarker,
            FaceLandmarkerOptions,
            RunningMode,
        )

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)
        self._mp = mp  # Keep reference for Image creation

    def _get_mouth_bbox(self, frame: np.ndarray) -> tuple:
        """
        Detect mouth region bounding box using MediaPipe Face Landmarker.

        Returns:
            (x1, y1, x2, y2) or None if no face detected
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=rgb,
        )

        result = self.landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]

        # Collect lip landmark coordinates
        lip_xs = []
        lip_ys = []
        for idx in ALL_LIP_INDICES:
            if idx < len(landmarks):
                lm = landmarks[idx]
                lip_xs.append(int(lm.x * w))
                lip_ys.append(int(lm.y * h))

        if not lip_xs:
            return None

        # Compute bounding box with expansion
        cx = (min(lip_xs) + max(lip_xs)) // 2
        cy = (min(lip_ys) + max(lip_ys)) // 2
        bw = int((max(lip_xs) - min(lip_xs)) * self.mouth_mask_expand)
        bh = int((max(lip_ys) - min(lip_ys)) * self.mouth_mask_expand)

        x1 = max(0, cx - bw // 2)
        y1 = max(0, cy - bh // 2)
        x2 = min(w, cx + bw // 2)
        y2 = min(h, cy + bh // 2)

        return (x1, y1, x2, y2)

    def corrupt_video(self, frames: np.ndarray) -> tuple:
        """
        Mask the mouth region in all frames with black rectangles.

        Args:
            frames: (N, H, W, 3) uint8 array

        Returns:
            corrupted_frames: Same shape as input, mouth region blacked out
            mouth_bboxes: List of (x1,y1,x2,y2) or None per frame
            corruption_mask: Boolean array (N, H, W) -- True where corrupted
        """
        corrupted = frames.copy()
        bboxes = []
        h, w = frames.shape[1], frames.shape[2]
        mask = np.zeros((len(frames), h, w), dtype=bool)

        detected = 0
        last_bbox = None

        for i in range(len(frames)):
            bbox = self._get_mouth_bbox(frames[i])

            if bbox is not None:
                last_bbox = bbox
                detected += 1
            elif last_bbox is not None:
                # Re-use last detected bbox for continuity
                bbox = last_bbox

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                corrupted[i, y1:y2, x1:x2] = 0  # Black out mouth region
                mask[i, y1:y2, x1:x2] = True

            bboxes.append(bbox)

        logger.info(
            f"Corrupted {len(frames)} frames, "
            f"face detected in {detected}/{len(frames)} frames"
        )

        return corrupted, bboxes, mask

    def corrupt_audio(self, audio: np.ndarray, sr: int) -> tuple:
        """
        Zero out random segments of the audio waveform.

        Args:
            audio: 1-D float32 array
            sr: sample rate

        Returns:
            corrupted_audio: Same shape, with zeroed segments
            corruption_mask: Boolean array -- True where corrupted
        """
        corrupted = audio.copy()
        mask = np.zeros(len(audio), dtype=bool)

        total_samples = len(audio)
        target_corrupt = int(total_samples * self.audio_corrupt_ratio)
        corrupted_count = 0

        min_samples = int(self.audio_min_gap_ms / 1000 * sr)
        max_samples = int(self.audio_max_gap_ms / 1000 * sr)

        attempts = 0
        max_attempts = 1000

        while corrupted_count < target_corrupt and attempts < max_attempts:
            gap_len = self.rng.randint(min_samples, max_samples + 1)
            start = self.rng.randint(0, max(1, total_samples - gap_len))
            end = min(start + gap_len, total_samples)

            # Only corrupt if not already corrupted
            if not mask[start:end].any():
                corrupted[start:end] = 0.0
                mask[start:end] = True
                corrupted_count += end - start

            attempts += 1

        actual_ratio = mask.sum() / total_samples
        logger.info(
            f"Audio corruption: {mask.sum()} / {total_samples} samples "
            f"({actual_ratio:.1%}) zeroed"
        )

        return corrupted, mask

    def corrupt(self, data: dict) -> dict:
        """
        Full corruption pipeline for video + audio.

        Args:
            data: dict from DataLoader.load() with frames, audio, etc.

        Returns:
            dict with original + corrupted data and masks
        """
        logger.info("Starting stream corruption...")

        corrupted_frames, mouth_bboxes, video_mask = self.corrupt_video(
            data["frames"]
        )
        corrupted_audio, audio_mask = self.corrupt_audio(
            data["audio"], data["sr"]
        )

        return {
            # Original
            "original_frames": data["frames"],
            "original_audio": data["audio"],
            # Corrupted
            "corrupted_frames": corrupted_frames,
            "corrupted_audio": corrupted_audio,
            # Masks
            "video_corruption_mask": video_mask,
            "audio_corruption_mask": audio_mask,
            "mouth_bboxes": mouth_bboxes,
            # Metadata
            "fps": data["fps"],
            "sr": data["sr"],
            "video_path": data["video_path"],
        }

    def __del__(self):
        if hasattr(self, "landmarker"):
            self.landmarker.close()
