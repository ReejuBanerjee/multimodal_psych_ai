# Explainable Multimodal AI for Psychological Diagnostics
 
> A multi-task, late-fusion deep learning system that combines facial vision, voice audio, and behavioral telemetry to detect sustained clinical stress states — designed to catch what single-modality models miss, like "smiling depression."
 
**Authors:** Reeju Banerjee & Shourya Shrivastava
**Registration Number:** RA2511003010548 & RA2511003010559 respectively.
**Institution:** SRM Institute of Science and Technology, Kattankulathur
 
---
 
## Table of Contents
 
- [Project Overview](#project-overview)
- [Key Technical Features](#key-technical-features)
- [Performance and Ablation Study](#performance-and-ablation-study)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Running the Live Demo](#running-the-live-demo)
- [Training the Model](#training-the-model)
- [License](#license)
---
 
## Project Overview
 
Current mental health AI systems rely almost exclusively on facial recognition. But human psychology is complex — patients experiencing severe stress or depression often mask their feelings by forcing a smile (the **"Smiling Depression" blind spot**). A standard vision model will misclassify this as a positive emotional state, failing the patient entirely.
 
Our solution is a **Multi-Task Late-Fusion Architecture** that captures three distinct data streams simultaneously:
 
1. **Webcam Vision** — facial expression analysis
2. **Microphone Audio** — vocal tone and prosody
3. **Behavioral Telemetry** — keyboard and mouse interaction patterns
By fusing all three, the model covers the blind spots of any single modality, diagnosing four sustained clinical stress states: **Healthy, Mild Stress, Moderate Stress,** and **Severe Stress.**
 
---
 
## Key Technical Features
 
| Feature | Description |
| :--- | :--- |
| **Multi-Task Late Fusion** | Independent feature extractors for Vision (CNN), Audio (FNN / 1D-CNN on MFCCs), and Tabular data (FNN). Streams are fused late in the pipeline to jointly output discrete stress classifications *and* continuous severity regression scores. |
| **Focal Loss Optimization** | A custom Focal Loss dynamically scales gradients, heavily penalizing the network on ambiguous boundary cases (e.g., distinguishing Mild vs. Moderate stress). |
| **Behavioral Telemetry Integration** | Tracks keystroke frequency and mouse movement variance as a proxy for physiological arousal and cognitive load — a reliable diagnostic anchor even when the camera or microphone fails. |
| **Real-Time Live Inference** | A fully integrated streaming UI (`live_demo.py`) captures live webcam feeds, audio buffers, and `pynput` telemetry to generate active, on-the-fly predictions with dynamic temperature scaling. |
 
---
 
## Performance and Ablation Study
 
To validate the necessity of the multimodal approach, we conducted a strict ablation study by blinding individual modalities. The results show a **~30-point accuracy leap** when late-fusion is applied over any single modality.
 
| Modality | Sub-Model | Accuracy | Macro F1-Score |
| :--- | :--- | :---: | :---: |
| Face-Only | CNN (48×48 Grayscale) | 40.75% | 14.48% |
| Voice-Only | FNN (40-dim MFCCs) | 40.75% | 14.48% |
| Telemetry-Only | FNN (18-dim Behavior) | 65.88% | 39.31% |
| **Fused (All)** | **Late-Fusion Architecture** | **95.38%** | **94.44%** |
 
### Regression Metrics (Severity Tracking)
 
| Metric | Score |
| :--- | :---: |
| R² (Explained Variance) | 0.9871 |
| Mean Absolute Error (MAE) | 0.8966 |
 
---
 
## Repository Structure
 
```text
MULTIMODAL-PSYCH-AI/
│
├── data/                               # Master Dataset Directory
│   ├── audio/                          # RAVDESS raw audio files (WAV)
│   ├── tabular/                        # Processed telemetry (Train/Test CSVs)
│   └── vision/                         # FER image subsets (JPG/PNG)
│
├── src/                                # Core Source Code & Logic
│   ├── dataloader.py                   # PyTorch dataset, MFCC, & image transforms
│   ├── models.py                       # Late-Fusion Multi-Task Neural Network class
│   ├── train.py                        # Training loop with Focal Loss & LR scheduler
│   ├── evaluate.py                     # Unseen test set inference and metric calculation
│   ├── evaluate_isolated.py            # Ablation study testing script
│   ├── live_demo.py                    # Real-time webcam/audio presentation streamer
│   ├── utils.py                        # Custom Focal Loss implementation
│   ├── visualize_batch.py              # Generates the 2x4 visual inference grid
│   └── generate_charts.py              # Generates presentation metric graphs
│
├── psych_model_weights.pth             # Saved PyTorch model weights
├── inference_grid.png                  # Output: batch evaluation visual grid
└── classification_chart.png            # Output: final classification metrics graph
```
 
---
 
## Installation
 
**1. Clone the repository and navigate to the project directory:**
 
```bash
git clone https://github.com/ReejuBanerjee/multimodal_psych_ai.git
cd multimodal_psych_ai
```
 
**2. Install the required dependencies:**
 
```bash
pip install torch torchvision torchaudio opencv-python sounddevice pynput matplotlib seaborn scikit-learn pandas numpy
```
 
---
 
## Running the Live Demo
 
Ensure you have a working webcam and microphone connected, then run the live streamer:
 
```bash
cd src
python live_demo.py
```
 
**Demo Controls:**
 
| Key | Action |
| :---: | :--- |
| `0` – `3` | Manually override the predicted state (for presentation purposes) |
| `r` | Return to live AI auto-tracking |
| `q` | Quit the demo |
 
---
 
## Training the Model
 
If you wish to retrain the model weights from scratch:
 
**1.** Ensure your `data/` folder is populated with the required audio, vision, and tabular files.
 
**2.** Execute the training script:
 
```bash
cd src
python train.py
```
 
**3.** Evaluate the newly trained model:
 
```bash
python evaluate.py
```
 
**4.** Generate the ablation study and visualization grids:
 
```bash
python evaluate_isolated.py
python visualize_batch.py
```
 
---
