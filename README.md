# Face Recognition — DeiT vs DenseNet

Face recognition pipeline that compares two backbones — a DeiT transformer and a DenseNet CNN — while sharing the same MediaPipe-based preprocessing. Use the notebooks to benchmark both models on identical datasets and contrast latency/accuracy trade-offs.

## Repository Map

- **`deit_face_recognition.ipynb`** — end-to-end experimentation notebook (data scan, DeiT fine-tuning, evaluation, and inference helpers).
- **`densenet_face_recognition.ipynb`** — DenseNet-focused notebook with the same preprocessing pipeline for a CNN baseline.
- **`app.py`** — Gradio interface that reuses the MediaPipe preprocessing stack for real-time predictions.
- **`face_deit_best.pth`** — checkpoint produced by the DeiT notebook (weights, class list, and training cfg).
- **`face_densenet_best.h5`** — saved DenseNet weights for quick inference.
- **`dataset-train/`** — identity folders used for supervised training (`dataset-train/train/<person_name>/*.jpg|png|heic|webp`).

## Contributors

| Nama | NIM |
| ---- | --- |
| Lois Novel E Gurning | 122140098 |
| Fiqri Aldiansyah | 122140152 |

## Model Variants & Comparison

| Model | Backbone | Notes |
| ----- | -------- | ----- |
| DeiT | Vision Transformer (`deit_tiny_patch16_224`) | Stronger accuracy on larger datasets, higher compute cost. |
| DenseNet | DenseNet-121 (configurable inside the notebook) | Faster inference on CPU, good baseline for limited data. |

## Requirements

| Dependency | Purpose |
| ---------- | ------- |
| `torch`, `torchvision`, `timm` | DeiT backbone and training utilities |
| `albumentations` | Strong augmentations for DeiT |
| `mediapipe==0.10.14` | Face detection & alignment |
| `opencv-python`, `numpy`, `pillow` | Image processing helpers |
| `gradio` | Web UI for inference |
| `jupyter`/`ipykernel` (not in `requirements.txt`) | Needed if you plan to run the notebook locally |

Install everything inside a Python 3.10 environment (requirement for MediaPipe):

> On Hugging Face Spaces, keep `runtime.txt` set to `python-3.10` so the build image matches the dependency constraints.

## Training Workflow (`deit_face_recognition.ipynb`)

1. **Configure** — edit `BASE_CFG` in the notebook to point to your dataset root, tweak hyperparameters, and limit samples if needed.
2. **Preprocess** — the notebook scans `dataset-train/train`, detects faces via MediaPipe, applies CLAHE + bilateral filtering, and caches the enhanced crops in memory.
3. **Tuning / Training** — run the hyperparameter sweep section or jump to the manual training cell. Both paths rely on identical Albumentations transforms and training utilities.
4. **Evaluation & Inference** — use the built-in evaluation cells to inspect metrics, confusion matrices, and run sample predictions. Saving the best model produces `face_deit_best.pth`.

## Directory Structure

```
.
├── README.md
├── app.py
├── requirements.txt
├── deit_face_recognition.ipynb
├── densenet_face_recognition.ipynb
├── face_deit_best.pth
├── face_densenet_best.h5
└── dataset-train/
    └── train/
        └── <person_name>/
            └── *.jpg|png|heic|webp
```

