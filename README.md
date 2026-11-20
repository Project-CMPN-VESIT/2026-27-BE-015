# Audio-Visual Packet Loss Concealment System

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-F9AB00?logo=huggingface)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)

## Overview
The **Audio-Visual Packet Loss Concealment System** is an advanced machine learning pipeline that performs cross-modal restoration of corrupted video and audio streams. It uses **Wav2Vec2** for dense acoustic feature extraction, combined with signal processing techniques — per-segment spectral reconstruction, phase-aware interpolation, and energy-matched gap filling — to restore lost audio packets. On the video side, it recovers corrupted mouth regions by matching audio embeddings to a reference bank of clean mouth crops with LAB color space blending and feathered-edge compositing.

## Features
- **Cross-Modal Restoration**:
  - **Audio to Video (Lip Recovery)**: Retrieves missing visual lip movements using intact audio chunks.
  - **Video to Audio (Speech Recovery)**: Reconstructs lost audio packets leveraging visual phoneme/mouth movements.
- **Advanced Corruption Simulation**: Injects realistic packet loss scenarios, simulating network degradation by dropping audio sequences and masking visual facial/mouth data.
- **Neural Encoding**: `Wav2Vec2` for dense acoustic feature extraction and frame-aligned audio embeddings. Optionally, `Stable Diffusion VAE` for latent video frame analysis.
- **Robust Post-Processing**: Includes LAB color space skin-tone matching, feathered edge blending, and temporal smoothing to eliminate visual artifacts during frame reconstruction.
- **Comprehensive Evaluation Suite**: Automatically generates quantitative assessments comparing the restored streams against base truth using SSIM, PSNR, STOI, and LSE-D metrics.

## Core Restoration Methodologies

### 1. Video Restoration (Audio $\rightarrow$ Video)
* **Goal**: Reconstruct corrupted or missing mouth regions using audio signals.
* **Mechanism**: 
  - **Feature Extraction**: Extracts deep acoustic embeddings using **Wav2Vec2** from the unbroken audio track.
  - **Reference Matching**: Scans a dynamic bank of uncorrupted mouth crops to find the optimal nearest-neighbor via cosine similarity matching on the audio embeddings.
  - **Seamless Compositing**: Takes the best-matching mouth crops, temporally smoothes them, and surgically pastes them back into the corrupted hole using **LAB color space skin-tone matching** and **feathered-edge blending**. (Note: No generative diffusion model is used to synthesize the mouth).

### 2. Audio Restoration (Video $\rightarrow$ Audio)
* **Goal**: Synthesize missing audio packets using the speaker's lip movements.
* **Mechanism** (Per-Segment Reconstruction): 
  - **Gap Detection**: Identifies each contiguous corrupted segment from the corruption mask.
  - **Short Gaps (<30ms)**: Filled using equal-power sinusoidal waveform crossfade — perceptually seamless, identical to professional VoIP codecs.
  - **Longer Gaps (30ms–200ms)**: Reconstructed per-segment using local STFT with:
    - **Cubic spline magnitude interpolation** from surrounding clean context frames.
    - **Phase-aware interpolation** (unwrapped phase continuity) to avoid metallic artifacts.
    - **Griffin-Lim refinement** (only for gaps >100ms) to polish phase estimates.
  - **RMS Energy Matching**: Each fill is scaled so its energy matches the surrounding speech — prevents the fill from sounding quieter than the context.
  - **Visual Energy Boost**: Mouth motion optical flow provides a speech-activity envelope. This BOOSTS (not attenuates) the reconstructed audio during active lip movement.
  - **Comfort Noise Safety Net**: Any remaining zero-valued samples are filled with spectrally-shaped comfort noise to guarantee no audible silence drops.

> **Note**: Signal-processing-based reconstruction cannot recover the actual spoken words — the information is destroyed by packet loss. The goal is to make the gaps *inaudible* by filling them with plausible, energy-matched audio. True speech recovery would require a neural speech synthesis model (e.g., Lip2Wav, Voicebox) trained on the specific speaker.

## Pipeline Architecture
The execution pipeline is modularized into seven distinct stages:
1. **Data Loading**: Ingests video (`.mp4`), extracting sequential frames and sampling audio at `16kHz`.
2. **Stream Corruption**: Randomly zeroes audio segments and blacks out mouth regions via MediaPipe face landmarks to simulate packet loss.
3. **Encoding**: Extracts dense audio embeddings via `Wav2Vec2`. Optionally encodes video frames to latent space via `Stable Diffusion VAE` (skippable with `--skip-vae`).
4. **Audio → Video Restoration**: Matches audio embeddings to a reference bank of clean mouth crops via cosine similarity, then pastes the best match with color-corrected feathered blending.
5. **Video → Audio Restoration**: Per-segment spectral reconstruction with phase-aware interpolation, energy matching, and lip-motion boosting. Short gaps use waveform crossfade; longer gaps use local STFT reconstruction.
6. **Quality Evaluation**: Computes SSIM, PSNR (video), STOI (audio), and LSE-D (audio-visual sync) metrics.
7. **Output Writing**: Renders original, corrupted, restored, and side-by-side comparison `.mp4` files with muxed audio.

## Installation

### Prerequisites
- Python 3.8+
- [FFmpeg](https://ffmpeg.org/download.html) installed and accessible in your system `PATH`.
- A discrete GPU with CUDA capability is highly recommended for model inference.

### Setup
Clone the repository and install the project dependencies:

```bash
git clone https://github.com/sudarshan026/AV-Restoration.git
cd AV-Restoration

# It is recommended to use a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

You can launch the entire end-to-end pipeline with a single command passing your input video:

```bash
python main.py --input path/to/your/video.mp4
```

To run a quick demonstration using specific dataset configurations:
```bash
python main.py --input auto --max-frames 150 --output-dir results
```

### CLI Arguments
| Argument | Description | Default |
| :--- | :--- | :--- |
| `--input` | Path to the target input video (`.mp4`). Use `auto` to select from dataset. | **Required** |
| `--output-dir` | Directory to save restored `.mp4` outputs and `metrics.json`. | `outputs` |
| `--max-frames` | Limit the number of frames to process (useful for quick testing). | `150` |
| `--start-frame` | Index to commence processing (bypasses video introductions). | `500` |
| `--audio-corrupt-ratio` | Proportion of audio packets to synthetically drop. | `0.3` |
| `--mouth-expand` | Factor to expand the MediaPipe bounding box of the mouth. | `1.4` |
| `--device` | Device compute fallback: `cuda` or `cpu`. | Auto |
| `--skip-vae` | Bypasses the VAE encoding stage to save VRAM footprint. | `False` |
| `--no-comparison`| Suppresses the generation of side-by-side comparison output videos. | `False` |

## Project Structure
```text
cap/
├── 01_raw_youtube/        # Dataset repository for raw inputs.
├── models/                # Local weights and model checkpoint caching.
├── modules/
│   ├── audio_encoder.py   # Latent feature extraction via Wav2Vec2.
│   ├── audio_to_video.py  # Restores mouth frames using acoustic inputs.
│   ├── corruptor.py       # Simulates missing packets (audio & visual blanks).
│   ├── data_loader.py     # Parses input AV stream using OpenCV and Librosa.
│   ├── output_writer.py   # Assembles multiplexed comparison clips.
│   ├── sync_evaluator.py  # Generates SSIM, PSNR, STOI, and LSE-D values.
│   ├── video_encoder.py   # Optional latent frame encoding via Stable Diffusion VAE.
│   └── video_to_audio.py  # Per-segment audio restoration with phase-aware reconstruction.
├── outputs/               # Default render destination for pipeline results.
├── batch_generate.py      # Script for batch processing dataset videos.
├── main.py                # Pipeline entry point & orchestrator.
└── requirements.txt       # Python package dependencies.
```

## Performance Metrics Output
Post execution, the system saves a comprehensively structured `metrics.json` file analyzing:
- **SSIM (Structural Similarity Index)**: Contextual visual degradation check between original and restored visual.
- **PSNR (Peak Signal-to-Noise Ratio)**: Standard absolute error visual accuracy metric.
- **STOI (Short-Time Objective Intelligibility)**: Measures human-perceived audio clarity indexing against frequency variants.
- **LSE-D (Lip-Sync Error Distance)**: Determines mathematically how well constructed audio alignments match lip physical movements based on sync-net embeddings.

## Acknowledgements
Built for high-fidelity communication restoration, leveraging:
- [HuggingFace Transformers](https://huggingface.co/docs/transformers) — pre-trained `Wav2Vec2` for acoustic embeddings
- [MediaPipe](https://ai.google.dev/edge/mediapipe) — Face Landmarker for mouth region detection
- [Librosa](https://librosa.org/) — STFT, Griffin-Lim, and audio processing
- [OpenCV](https://opencv.org/) — Frame processing, color transfer, and seamless blending
- [Stable Diffusion VAE](https://huggingface.co/stabilityai/sd-vae-ft-mse) (optional) — Latent space frame analysis
