import os
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import gradio as gr
import mediapipe as mp
import numpy as np
import timm
import torch
from PIL import Image, ImageDraw


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "face_deit_best.pth"))

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model checkpoint tidak ditemukan: {MODEL_PATH}")

def build_eval_transform(img_size: int):
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

artifact = torch.load(MODEL_PATH, map_location="cpu")
class_names = artifact["class_names"]
model = timm.create_model(artifact["arch"], pretrained=False, num_classes=len(class_names))
model.load_state_dict(artifact["model"], strict=True)
model.to(DEVICE).eval()
cfg = artifact.get("cfg", {})
img_size = cfg.get("img_size", 224)
transform = build_eval_transform(img_size)


def _parse_env_float(name: str, default: float):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _parse_env_int(name: str, default: int):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


mediapipe_model = _parse_env_int("MEDIAPIPE_MODEL", int(cfg.get("mediapipe_model", 0)))
mediapipe_min_conf = _parse_env_float("MEDIAPIPE_MIN_CONF", float(cfg.get("mediapipe_min_conf", 0.5)))
mediapipe_bbox_margin = _parse_env_float("MEDIAPIPE_BBOX_MARGIN", float(cfg.get("mediapipe_bbox_margin", 0.2)))

mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(
    model_selection=mediapipe_model,
    min_detection_confidence=mediapipe_min_conf,
)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def detect_and_align(rgb_img: np.ndarray, image_size: int):
    results = face_detector.process(rgb_img)
    detections = results.detections if results else None
    if not detections:
        return None
    best_det = max(detections, key=lambda det: det.score[0] if det.score else 0.0)
    bbox = best_det.location_data.relative_bounding_box
    h, w, _ = rgb_img.shape
    x_min = max(bbox.xmin, 0.0) * w
    y_min = max(bbox.ymin, 0.0) * h
    box_w = max(bbox.width, 0.0) * w
    box_h = max(bbox.height, 0.0) * h
    side = max(box_w, box_h) * (1.0 + mediapipe_bbox_margin)
    cx = x_min + box_w / 2.0
    cy = y_min + box_h / 2.0
    x1 = int(max(cx - side / 2.0, 0))
    y1 = int(max(cy - side / 2.0, 0))
    x2 = int(min(cx + side / 2.0, w))
    y2 = int(min(cy + side / 2.0, h))
    if x2 <= x1 or y2 <= y1:
        return None
    face_rgb = rgb_img[y1:y2, x1:x2]
    if face_rgb.size == 0:
        return None
    aligned = cv2.resize(face_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    det_score = float(best_det.score[0]) if best_det.score else 0.0
    abs_bbox = np.array([x1, y1, x2, y2])
    return {
        "aligned_rgb": aligned,
        "bbox": abs_bbox,
        "det_score": det_score,
    }

def enhance_face(rgb_img: np.ndarray):
    ycrcb = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    eq_rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    smooth = cv2.bilateralFilter(eq_rgb, d=9, sigmaColor=75, sigmaSpace=75)
    return smooth


def preprocess_image(pil_image: Image.Image):
    rgb = np.ascontiguousarray(np.array(pil_image.convert("RGB"), dtype=np.uint8))
    meta = detect_and_align(rgb, img_size)
    if meta is None:
        raise gr.Error("Tidak ada wajah yang terdeteksi. Gunakan foto dengan wajah jelas.")
    enhanced = enhance_face(meta["aligned_rgb"])
    tensor = transform(image=enhanced)["image"].unsqueeze(0).to(DEVICE)
    return tensor, meta, rgb

def annotate(rgb_image: np.ndarray, bbox: np.ndarray, label: str, score: float):
    img = Image.fromarray(rgb_image.copy())
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = bbox.astype(int)
    draw.rectangle([(x1, y1), (x2, y2)], outline="lime", width=4)
    text = f"{label} ({score:.1%})"
    text_bg_w, text_bg_h = draw.textsize(text)
    padding = 4
    draw.rectangle(
        [(x1, max(0, y1 - text_bg_h - padding * 2)), (x1 + text_bg_w + padding * 2, y1)],
        fill="lime",
    )
    draw.text((x1 + padding, max(0, y1 - text_bg_h - padding)), text, fill="black")
    return img

def predict(image: Image.Image):
    if image is None:
        raise gr.Error("Silakan unggah foto terlebih dahulu.")
    tensor, meta, rgb = preprocess_image(image)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    top_idx = probs.argsort()[::-1][:3]
    result_dict = {class_names[i]: float(probs[i]) for i in top_idx}
    best_label = class_names[top_idx[0]]
    annotated = annotate(rgb, meta["bbox"], best_label, probs[top_idx[0]])
    return result_dict, annotated

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Foto Wajah"),
    outputs=[
        gr.Label(num_top_classes=3, label="Prediksi Identitas"),
        gr.Image(label="Deteksi & Identitas"),
    ],
    title="Face Recognition",
    description=(
        "Unggah foto wajah, aplikasi akan mendeteksi wajah dan menampilkan identitas"
    ),
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
