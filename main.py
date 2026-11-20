"""
main.py — Audio-Visual Packet Loss Concealment System

Complete pipeline that:
1. Loads a video file (Obama dataset)
2. Extracts frames + audio
3. Corrupts the stream (mouth region + audio segments)
4. Encodes audio/video with pretrained models
5. Restores video using audio cues (audio -> video)
6. Restores audio using video cues (video -> audio)
7. Evaluates quality (SSIM, PSNR, STOI, LSE-D)
8. Outputs the restored video

Usage:
    python main.py --input input.mp4
    python main.py --input 01_raw_youtube/video.mp4 --max-frames 150
    python main.py --input 01_raw_youtube/video.mp4 --output-dir results
"""

import argparse
import json
import logging
import os
import sys
import time
import glob
import numpy as np

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def print_banner():
    banner = (
        "\n"
        "+======================================================================+\n"
        "|                                                                      |\n"
        "|     Audio-Visual Packet Loss Concealment System                      |\n"
        "|                                                                      |\n"
        "|     Cross-modal restoration of corrupted video/audio streams         |\n"
        "|     Using Wav2Vec2 + Stable Diffusion VAE + Griffin-Lim              |\n"
        "|                                                                      |\n"
        "+======================================================================+\n"
    )
    print(banner)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audio-Visual Packet Loss Concealment System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input video file (.mp4)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for output files (default: outputs)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=150,
        help="Max frames to process (0=all, default=150 for quick demo)",
    )
    parser.add_argument(
        "--audio-corrupt-ratio",
        type=float,
        default=0.3,
        help="Fraction of audio to corrupt (default: 0.3)",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=500,
        help="Frame index to start from, skips video intros (default: 500)",
    )
    parser.add_argument(
        "--mouth-expand",
        type=float,
        default=1.4,
        help="Mouth mask expansion factor (default: 1.4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: 'cuda' or 'cpu' (default: auto-detect)",
    )
    parser.add_argument(
        "--skip-vae",
        action="store_true",
        help="Skip VAE encoding (saves GPU memory)",
    )
    parser.add_argument(
        "--no-comparison",
        action="store_true",
        help="Skip writing comparison videos",
    )
    return parser.parse_args()


def resolve_input(input_path: str) -> str:
    """Resolve the input video path, handling relative paths and dataset directory."""
    if os.path.isfile(input_path):
        return os.path.abspath(input_path)

    # Check in the Obama dataset directory
    dataset_dir = os.path.join(os.path.dirname(__file__), "01_raw_youtube")
    candidate = os.path.join(dataset_dir, input_path)
    if os.path.isfile(candidate):
        return os.path.abspath(candidate)

    # If input is "auto" or "first", pick the first video in dataset
    if input_path.lower() in ("auto", "first", "any"):
        videos = sorted(glob.glob(os.path.join(dataset_dir, "*.mp4")))
        if videos:
            logger.info(f"Auto-selected: {os.path.basename(videos[0])}")
            return os.path.abspath(videos[0])

    raise FileNotFoundError(
        f"Input video not found: {input_path}\n"
        f"Checked: {input_path}, {candidate if 'candidate' in dir() else 'N/A'}"
    )


def main():
    print_banner()
    args = parse_args()

    total_start = time.time()

    # --- Resolve Input ---
    input_path = resolve_input(args.input)
    logger.info(f"Input video: {input_path}")
    logger.info(f"Output dir:  {args.output_dir}")
    logger.info(f"Max frames:  {args.max_frames}")
    logger.info(f"Device:      {args.device or 'auto'}")

    os.makedirs(args.output_dir, exist_ok=True)

    # =================================================================
    # STAGE 1: DATA LOADING
    # =================================================================
    logger.info("=" * 60)
    logger.info("STAGE 1: Loading Video Data")
    logger.info("=" * 60)

    from modules.data_loader import DataLoader

    loader = DataLoader(
        video_path=input_path,
        target_sr=16000,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
    )
    data = loader.load()

    logger.info(
        f"Loaded: {len(data['frames'])} frames @ {data['fps']:.1f} fps, "
        f"Audio: {len(data['audio'])} samples @ {data['sr']} Hz"
    )

    # =================================================================
    # STAGE 2: STREAM CORRUPTION
    # =================================================================
    logger.info("=" * 60)
    logger.info("STAGE 2: Corrupting Stream (Simulating Packet Loss)")
    logger.info("=" * 60)

    from modules.corruptor import StreamCorruptor

    corruptor = StreamCorruptor(
        mouth_mask_expand=args.mouth_expand,
        audio_corrupt_ratio=args.audio_corrupt_ratio,
    )
    corrupted_data = corruptor.corrupt(data)

    logger.info(
        f"Video corruption: "
        f"{corrupted_data['video_corruption_mask'].sum()} pixels masked"
    )
    logger.info(
        f"Audio corruption: "
        f"{corrupted_data['audio_corruption_mask'].sum()} samples zeroed"
    )

    # =================================================================
    # STAGE 3: ENCODING (Audio + Video)
    # =================================================================
    logger.info("=" * 60)
    logger.info("STAGE 3: Encoding with Pretrained Models")
    logger.info("=" * 60)

    # Audio encoding with Wav2Vec2
    logger.info("--- Audio Encoder (Wav2Vec2) ---")
    from modules.audio_encoder import AudioEncoder

    audio_enc = AudioEncoder(device=args.device)
    audio_embeddings = audio_enc.encode(data["audio"], sr=data["sr"])
    logger.info(f"Audio embeddings: {audio_embeddings.shape}")

    # Video encoding with VAE (optional, for latent space analysis)
    video_latents = None
    if not args.skip_vae:
        logger.info("--- Video Encoder (Stable Diffusion VAE) ---")
        from modules.video_encoder import VideoEncoder

        video_enc = VideoEncoder(device=args.device, target_size=256)

        # Encode a subset of frames to demonstrate latent space
        sample_indices = np.linspace(
            0, len(data["frames"]) - 1, min(16, len(data["frames"]))
        ).astype(int)
        sample_frames = data["frames"][sample_indices]

        video_latents = video_enc.encode(sample_frames, batch_size=4)
        logger.info(f"Video latents: {video_latents.shape}")

        # Demonstrate encode-decode roundtrip
        decoded_frames = video_enc.decode(
            video_latents,
            original_size=(data["frames"].shape[1], data["frames"].shape[2]),
        )
        logger.info(f"Decode roundtrip: {decoded_frames.shape}")

        # Encode mouth regions
        mouth_latents = video_enc.encode_region(
            data["frames"],
            corrupted_data["mouth_bboxes"],
        )
        valid_latents = sum(1 for l in mouth_latents if l is not None)
        logger.info(f"Mouth region latents: {valid_latents} encoded")

        # Free GPU memory
        del video_enc
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        logger.info("Skipping VAE encoding (--skip-vae)")

    # =================================================================
    # STAGE 4: AUDIO -> VIDEO RESTORATION
    # =================================================================
    logger.info("=" * 60)
    logger.info("STAGE 4: Audio -> Video Restoration (Lip Recovery)")
    logger.info("=" * 60)

    from modules.audio_to_video import AudioToVideoRestorer

    a2v_restorer = AudioToVideoRestorer(
        audio_encoder=audio_enc,
        n_candidates=5,
        device=args.device,
    )
    restored_frames = a2v_restorer.restore(corrupted_data)
    logger.info(f"Video restoration complete: {restored_frames.shape}")

    # =================================================================
    # STAGE 5: VIDEO -> AUDIO RESTORATION
    # =================================================================
    logger.info("=" * 60)
    logger.info("STAGE 5: Video -> Audio Restoration (Speech Recovery)")
    logger.info("=" * 60)

    from modules.video_to_audio import VideoToAudioRestorer

    v2a_restorer = VideoToAudioRestorer(
        n_fft=1024,
        hop_length=256,
        n_iter_griffin_lim=60,
    )
    restored_audio = v2a_restorer.restore(corrupted_data)
    logger.info(f"Audio restoration complete: {len(restored_audio)} samples")

    # =================================================================
    # STAGE 6: EVALUATION
    # =================================================================
    logger.info("=" * 60)
    logger.info("STAGE 6: Quality Evaluation")
    logger.info("=" * 60)

    from modules.sync_evaluator import SyncEvaluator

    evaluator = SyncEvaluator(audio_encoder=audio_enc, device=args.device)

    metrics = evaluator.evaluate_all(
        original_frames=data["frames"],
        restored_frames=restored_frames,
        original_audio=data["audio"],
        restored_audio=restored_audio,
        sr=data["sr"],
        fps=data["fps"],
        mouth_bboxes=corrupted_data["mouth_bboxes"],
    )

    # Save metrics to JSON
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    # Convert numpy arrays and non-serializable types
    metrics_serializable = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            metrics_serializable[key] = {
                k: v for k, v in value.items()
                if not isinstance(v, (list,)) or len(v) < 50
            }
            # Truncate long lists
            for k, v in value.items():
                if isinstance(v, list) and len(v) >= 50:
                    metrics_serializable[key][k] = f"[{len(v)} values, mean={np.mean(v):.4f}]"
        else:
            metrics_serializable[key] = value

    with open(metrics_path, "w") as f:
        json.dump(metrics_serializable, f, indent=2, default=str)
    logger.info(f"Metrics saved to: {metrics_path}")

    # =================================================================
    # STAGE 7: OUTPUT WRITING
    # =================================================================
    logger.info("=" * 60)
    logger.info("STAGE 7: Writing Output Videos")
    logger.info("=" * 60)

    from modules.output_writer import OutputWriter

    writer = OutputWriter(output_dir=args.output_dir)

    if not args.no_comparison:
        # Write all comparison videos
        output_paths = writer.write_comparison(
            original_frames=data["frames"],
            corrupted_frames=corrupted_data["corrupted_frames"],
            restored_frames=restored_frames,
            original_audio=data["audio"],
            corrupted_audio=corrupted_data["corrupted_audio"],
            restored_audio=restored_audio,
            fps=data["fps"],
            sr=data["sr"],
        )
    else:
        # Write only the restored output (with original audio)
        output_paths = writer.write(
            frames=restored_frames,
            audio=data["audio"],  # Use ORIGINAL audio
            fps=data["fps"],
            sr=data["sr"],
        )

    # =================================================================
    # DONE
    # =================================================================
    total_time = time.time() - total_start

    print("\n")
    print("+============================================================+")
    print("|                    PIPELINE COMPLETE                        |")
    print("+============================================================+")
    print(f"|  Total time:  {total_time:.1f} seconds")
    print(f"|  Frames:      {len(data['frames'])}")
    print(f"|  SSIM:        {metrics['ssim']['mean_ssim']:.4f}")
    print(f"|  PSNR:        {metrics['psnr']['mean_psnr']:.2f} dB")

    stoi_val = metrics['stoi'].get('stoi_score')
    if stoi_val is not None:
        print(f"|  STOI:        {stoi_val:.4f}")

    lsed_val = metrics['sync'].get('lse_d')
    if lsed_val is not None:
        print(f"|  LSE-D:       {lsed_val:.4f}")

    print("+------------------------------------------------------------+")
    print("|  Output Files:")

    if isinstance(output_paths, dict):
        def print_paths(d, prefix=""):
            for key, val in d.items():
                if isinstance(val, dict):
                    print_paths(val, prefix=f"{key}/")
                elif isinstance(val, str):
                    name = f"{prefix}{key}"
                    print(f"|    {name}: {os.path.basename(val)}")
        print_paths(output_paths)

    print("+============================================================+")
    print()


if __name__ == "__main__":
    main()
