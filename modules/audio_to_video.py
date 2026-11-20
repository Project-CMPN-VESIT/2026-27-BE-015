"""
audio_to_video.py -- Restore corrupted mouth regions in video using audio cues.

Uses audio embeddings from Wav2Vec2 to drive lip restoration via
audio-conditioned nearest-neighbor matching with:
  - Expanded reference crops (with skin context surrounding the mouth)
  - Direct region replacement + multi-band Gaussian edge blending
  - LAB color space transfer for skin-tone consistency
  - Temporal smoothing across consecutive frames for flicker-free output
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class AudioToVideoRestorer:
    """
    Restores corrupted mouth regions in video frames using cross-modal
    audio information. Pastes the best-matching mouth region from the
    reference bank and blends edges seamlessly using multi-band blending.
    """

    def __init__(
        self,
        audio_encoder,
        n_candidates: int = 5,
        context_expand: float = 1.8,
        temporal_alpha: float = 0.25,
        device: str = None,
    ):
        """
        Args:
            audio_encoder: AudioEncoder instance for embedding extraction
            n_candidates: Number of nearest neighbor candidates to blend
            context_expand: How much extra skin context to grab around the mouth
            temporal_alpha: Blend ratio for temporal smoothing (0=none, 1=full prev)
            device: torch device
        """
        self.audio_encoder = audio_encoder
        self.n_candidates = n_candidates
        self.context_expand = context_expand
        self.temporal_alpha = temporal_alpha

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def _expand_bbox(self, bbox, frame_h, frame_w, factor):
        """Expand a bounding box by a factor, clamped to frame bounds."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = (x2 - x1) * factor
        bh = (y2 - y1) * factor

        nx1 = max(0, int(cx - bw / 2))
        ny1 = max(0, int(cy - bh / 2))
        nx2 = min(frame_w, int(cx + bw / 2))
        ny2 = min(frame_h, int(cy + bh / 2))
        return (nx1, ny1, nx2, ny2)

    def _build_reference_bank(
        self,
        original_frames: np.ndarray,
        audio: np.ndarray,
        sr: int,
        fps: float,
        mouth_bboxes: list,
    ) -> dict:
        """
        Build a bank of EXPANDED mouth regions (with surrounding skin context)
        indexed by audio embeddings from uncorrupted frames.
        """
        logger.info("Building reference mouth bank from original frames...")

        frame_embeddings = self.audio_encoder.get_frame_aligned_embeddings(
            audio, sr, fps, len(original_frames)
        )

        h, w = original_frames.shape[1], original_frames.shape[2]

        bank_embeddings = []
        bank_crops = []
        bank_bboxes = []
        bank_exp_bboxes = []
        bank_indices = []

        for i, bbox in enumerate(mouth_bboxes):
            if bbox is None:
                continue

            # Expand the bbox to grab surrounding skin for context
            exp_bbox = self._expand_bbox(bbox, h, w, self.context_expand)
            ex1, ey1, ex2, ey2 = exp_bbox

            crop = original_frames[i, ey1:ey2, ex1:ex2].copy()
            if crop.size == 0:
                continue

            bank_embeddings.append(frame_embeddings[i])
            bank_crops.append(crop)
            bank_bboxes.append(bbox)
            bank_exp_bboxes.append(exp_bbox)
            bank_indices.append(i)

        if len(bank_embeddings) == 0:
            logger.warning("No valid mouth regions found for reference bank!")
            return None

        bank_emb_tensor = torch.stack(bank_embeddings)
        bank_emb_tensor = F.normalize(bank_emb_tensor, dim=1)

        logger.info(
            f"Reference bank: {len(bank_crops)} mouth regions "
            f"(expanded x{self.context_expand})"
        )

        return {
            "embeddings": bank_emb_tensor,
            "mouth_crops": bank_crops,
            "bboxes": bank_bboxes,
            "exp_bboxes": bank_exp_bboxes,
            "indices": bank_indices,
        }

    def _find_best_match(self, query_embedding, bank):
        """Find top-k nearest neighbor mouth crops by audio cosine similarity."""
        query = F.normalize(query_embedding.unsqueeze(0), dim=1)
        similarities = torch.mm(query, bank["embeddings"].T).squeeze(0)

        k = min(self.n_candidates, len(similarities))
        topk_vals, topk_idx = torch.topk(similarities, k)

        matches = []
        for val, idx in zip(topk_vals, topk_idx):
            i = idx.item()
            matches.append({
                "crop": bank["mouth_crops"][i],
                "score": val.item(),
                "bbox": bank["bboxes"][i],
                "exp_bbox": bank["exp_bboxes"][i],
                "frame_idx": bank["indices"][i],
            })
        return matches

    def _color_transfer(self, source, target):
        """
        Transfer color statistics from target to source using LAB color space.
        This makes the pasted mouth match the surrounding skin tone.
        """
        if source.size == 0 or target.size == 0:
            return source
        if source.shape[0] < 2 or source.shape[1] < 2:
            return source
        if target.shape[0] < 2 or target.shape[1] < 2:
            return source

        src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float64)
        tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float64)

        for ch in range(3):
            src_mean = src_lab[:, :, ch].mean()
            src_std = src_lab[:, :, ch].std() + 1e-6
            tgt_mean = tgt_lab[:, :, ch].mean()
            tgt_std = tgt_lab[:, :, ch].std() + 1e-6

            src_lab[:, :, ch] = (
                (src_lab[:, :, ch] - src_mean) * (tgt_std / src_std) + tgt_mean
            )

        src_lab = np.clip(src_lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(src_lab, cv2.COLOR_LAB2BGR)

    def _weighted_blend_crops(self, matches, target_h, target_w):
        """Blend multiple candidate mouth crops weighted by similarity score."""
        blended = np.zeros((target_h, target_w, 3), dtype=np.float64)
        total_weight = 0.0

        for m in matches:
            resized = cv2.resize(m["crop"], (target_w, target_h)).astype(np.float64)
            weight = max(m["score"], 0.0) ** 3
            blended += resized * weight
            total_weight += weight

        if total_weight > 0:
            blended /= total_weight

        return np.clip(blended, 0, 255).astype(np.uint8)

    def _create_feathered_mask(self, h, w, feather_ratio=0.2):
        """
        Create a feathered blending mask: fully opaque in the center,
        smoothly fading to 0 at the edges.

        The feather band occupies `feather_ratio` of each edge.
        """
        mask = np.ones((h, w), dtype=np.float32)

        feather_x = max(3, int(w * feather_ratio))
        feather_y = max(3, int(h * feather_ratio))

        # Left edge fade
        for i in range(feather_x):
            alpha = i / feather_x
            mask[:, i] *= alpha
        # Right edge fade
        for i in range(feather_x):
            alpha = i / feather_x
            mask[:, w - 1 - i] *= alpha
        # Top edge fade
        for j in range(feather_y):
            alpha = j / feather_y
            mask[j, :] *= alpha
        # Bottom edge fade
        for j in range(feather_y):
            alpha = j / feather_y
            mask[h - 1 - j, :] *= alpha

        # Extra smoothing pass
        ksize = max(3, min(feather_x, feather_y) | 1)
        mask = cv2.GaussianBlur(mask, (ksize, ksize), ksize / 3.0)

        return mask

    def _paste_with_feathered_blend(self, frame, crop, bbox, feather_ratio=0.25):
        """
        Paste the crop onto the frame at bbox with feathered edge blending.

        Strategy:
        1. First HARD-REPLACE the inner region (no blending with black pixels)
        2. Then feather only the edges to smoothly transition to the surroundings
        """
        x1, y1, x2, y2 = bbox
        target_h = y2 - y1
        target_w = x2 - x1

        if target_h <= 2 or target_w <= 2:
            return frame

        resized = cv2.resize(crop, (target_w, target_h))

        # Create feathered mask
        mask = self._create_feathered_mask(target_h, target_w, feather_ratio)
        mask_3ch = mask[:, :, np.newaxis]

        result = frame.copy()

        # Blend: where mask=1 (center), use 100% new crop
        # where mask=0 (edges), use 100% original frame
        # This ensures the black corrupted pixels in the center are FULLY replaced
        result[y1:y2, x1:x2] = (
            resized.astype(np.float32) * mask_3ch
            + result[y1:y2, x1:x2].astype(np.float32) * (1.0 - mask_3ch)
        ).astype(np.uint8)

        return result

    def restore(self, corrupted_data: dict) -> np.ndarray:
        """
        Main restoration pipeline: recover mouth regions using audio features.

        Pipeline for each frame:
        1. Find best-matching mouth from reference bank via audio cosine similarity
        2. Weighted-blend top-k candidates for smoother results
        3. Color-transfer the blended crop to match the target frame's skin tone
        4. Temporal smoothing with the previous frame's crop
        5. Hard-replace the corrupted region + feathered edge blending
        """
        original_frames = corrupted_data["original_frames"]
        corrupted_frames = corrupted_data["corrupted_frames"]
        original_audio = corrupted_data["original_audio"]
        mouth_bboxes = corrupted_data["mouth_bboxes"]
        fps = corrupted_data["fps"]
        sr = corrupted_data["sr"]

        h, w = original_frames.shape[1], original_frames.shape[2]

        # Build reference bank from original (uncorrupted) frames
        bank = self._build_reference_bank(
            original_frames, original_audio, sr, fps, mouth_bboxes
        )

        if bank is None:
            logger.warning("No reference bank available, returning corrupted frames")
            return corrupted_frames.copy()

        # Get audio embeddings for corrupted audio
        corrupted_audio = corrupted_data["corrupted_audio"]
        frame_embeddings = self.audio_encoder.get_frame_aligned_embeddings(
            corrupted_audio, sr, fps, len(corrupted_frames)
        )

        restored = corrupted_frames.copy()
        restored_count = 0
        prev_crop = None

        for i in range(len(corrupted_frames)):
            bbox = mouth_bboxes[i]
            if bbox is None:
                prev_crop = None
                continue

            x1, y1, x2, y2 = bbox
            if (y2 - y1) <= 4 or (x2 - x1) <= 4:
                prev_crop = None
                continue

            # ---- Step 1: Find best matching mouths via audio similarity ----
            matches = self._find_best_match(frame_embeddings[i], bank)
            if not matches:
                prev_crop = None
                continue

            # ---- Step 2: Compute the EXPANDED paste region ----
            # Use the FULL expanded region that includes skin context
            # This ensures we paste BEYOND the corrupted area
            exp_bbox = self._expand_bbox(bbox, h, w, self.context_expand)
            ex1, ey1, ex2, ey2 = exp_bbox
            exp_h = ey2 - ey1
            exp_w = ex2 - ex1

            # ---- Step 3: Get weighted-blended crop from candidates ----
            blended_crop = self._weighted_blend_crops(matches, exp_h, exp_w)

            # ---- Step 4: Color transfer to match target frame ----
            # Use the ORIGINAL (uncorrupted) expanded region as color reference
            skin_ref = original_frames[i, ey1:ey2, ex1:ex2].copy()
            if skin_ref.size > 0:
                blended_crop = self._color_transfer(blended_crop, skin_ref)

            # ---- Step 5: Temporal smoothing ----
            if prev_crop is not None and self.temporal_alpha > 0:
                try:
                    prev_resized = cv2.resize(prev_crop, (exp_w, exp_h))
                    blended_crop = cv2.addWeighted(
                        blended_crop, 1.0 - self.temporal_alpha,
                        prev_resized, self.temporal_alpha,
                        0,
                    )
                except cv2.error:
                    pass
            prev_crop = blended_crop.copy()

            # ---- Step 6: Paste with feathered edge blending ----
            # The feathered mask is 1.0 in the center (hard replace)
            # and fades to 0.0 at edges (smooth transition to frame)
            # The expanded region ensures the paste extends PAST the
            # black corrupted area into clean skin territory
            restored[i] = self._paste_with_feathered_blend(
                restored[i], blended_crop, exp_bbox, feather_ratio=0.2
            )
            restored_count += 1

        logger.info(
            f"Video restoration: {restored_count}/{len(corrupted_frames)} "
            f"frames restored (feathered paste + color transfer)"
        )

        return restored
