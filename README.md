# Face Recognition — DeiT + MediaPipe

Face recognition pipeline that combines DeiT image transformers with a MediaPipe-based face preprocessing stack. The repository contains:

- **`deit_face_recognition.ipynb`** — end-to-end experimentation notebook (data scan, DeiT fine-tuning, evaluation, and inference helpers).
- **`app.py`** — Gradio interface that reuses the MediaPipe preprocessing stack for real-time predictions.
- **`face_deit_best.pth`** — checkpoint produced by the notebook (weights, class list, and training cfg).
- **`dataset-train/`** — identity folders used for supervised training (`dataset-train/train/<person_name>/*.jpg|png|heic|webp`).

## Highlights
- **MediaPipe preprocessing**: MediaPipe FaceDetection crops the sharpest face, pads it to a square region, resizes to DeiT input size, then applies CLAHE + bilateral filtering to stabilize illumination.
- **Transformer backbone**: Uses the `timm` DeiT family (`deit_tiny_patch16_224` or `deit_small_patch16_224`) with Albumentations augmentation recipes defined in the notebook.
- **Consistent inference interface**: `app.py` loads everything from the saved checkpoint (architecture, class list, Albumentations config) so that Gradio and the notebook stay in sync.
- **Hugging Face Space ready**: `requirements.txt` + `runtime.txt` lock dependencies (Torch 2.x, MediaPipe 0.10.x, Python 3.10) for reproducible deployments.

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

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> On Hugging Face Spaces, keep `runtime.txt` set to `python-3.10` so the build image matches the dependency constraints.

## Dataset Layout

```
dataset-train/
└── train/
    ├── Person_A/
    │   ├── img1.jpg
    │   └── img2.png
    └── Person_B/
        └── ...
```

Every folder name becomes the target label. The notebook scans these directories, runs MediaPipe preprocessing once per image, and caches failures (missing faces, unreadable files, etc.).

## Training Workflow (`deit_face_recognition.ipynb`)

1. **Configure** — edit `BASE_CFG` in the notebook to point to your dataset root, tweak hyperparameters, and limit samples if needed.
2. **Preprocess** — the notebook scans `dataset-train/train`, detects faces via MediaPipe, applies CLAHE + bilateral filtering, and caches the enhanced crops in memory.
3. **Tuning / Training** — run the hyperparameter sweep section or jump to the manual training cell. Both paths rely on identical Albumentations transforms and training utilities.
4. **Evaluation & Inference** — use the built-in evaluation cells to inspect metrics, confusion matrices, and run sample predictions. Saving the best model produces `face_deit_best.pth`.

All helper functions (transforms, dataset wrappers, training loops, scheduler) mirror the approach used in the food-classification template but consume the MediaPipe-preprocessed tensors.

## Running the Gradio App (`app.py`)

```bash
python app.py
```

Key environment variables:

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `MODEL_PATH` | `face_deit_best.pth` | Checkpoint to load (contains cfg, weights, and class names). |
| `MEDIAPIPE_MODEL` | from checkpoint cfg or `0` | FaceDetection model selection (`0` = short-range, `1` = full-range). |
| `MEDIAPIPE_MIN_CONF` | from cfg or `0.5` | Minimum detection confidence. |
| `MEDIAPIPE_BBOX_MARGIN` | from cfg or `0.2` | Extra padding ratio when cropping faces. |

During inference the app:

1. Converts the uploaded image to RGB and passes it to MediaPipe.
2. Crops and resizes the most confident face, applies CLAHE + bilateral filtering.
3. Runs the DeiT model, shows the top-3 identity probabilities, and overlays the bounding box on the uploaded photo.

## Deploying to Hugging Face Spaces

1. Push the repository (including `app.py`, `requirements.txt`, `runtime.txt`, checkpoint, and any assets) to a new Space.
2. Set the Space SDK to **Gradio** and entry point to `app.py`.
3. Ensure hardware selections match your needs (`cpu-basic` is sufficient unless you require GPU inference).
4. On every build, Spaces runs `pip install -r requirements.txt` under Python 3.10 and launches `python app.py`.

## Tips & Troubleshooting

- **MediaPipe install errors**: use Python 3.10; other versions (e.g., 3.13) currently lack prebuilt wheels.
- **No face detected**: uploads must contain a clear single face. Increase `MEDIAPIPE_MIN_CONF` or adjust `mediapipe_bbox_margin` if detections are too tight.
- **Custom checkpoints**: after retraining, copy the new `.pth` file to the deployment artifact and update `MODEL_PATH` accordingly.
- **Dataset updates**: rerun the notebook preprocessing cell so new images enter the cache and metadata DataFrame before retraining.

Happy experimenting!
