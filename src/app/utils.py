# src/app/utils.py

import hashlib
from pathlib import Path
from PIL import Image, ImageOps
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap


def default_color_for_name(name: str) -> str:
    h = int(hashlib.md5(name.encode()).hexdigest()[:6], 16)
    return f"#{h:06x}"


def detect_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def safe_close_image(image):
    if image:
        try:
            image.close()
        except Exception:
            pass


def load_image(image_path):
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def pil_to_pixmap(image):
    if image is None:
        return QPixmap()

    if image.mode != "RGB":
        image = image.convert("RGB")

    w, h = image.size
    data = image.tobytes("raw", "RGB")
    qimg = QImage(data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def scale_to_fit(pixmap, target_size):
    if pixmap.isNull():
        return pixmap, 1.0

    tw = max(1, int(target_size.width()))
    th = max(1, int(target_size.height()))
    pw = max(1, pixmap.width())
    ph = max(1, pixmap.height())

    scale = min(tw / pw, th / ph, 1.0)
    new_w = max(1, int(pw * scale))
    new_h = max(1, int(ph * scale))

    scaled = pixmap.scaled(
        new_w,
        new_h,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation
    )
    return scaled, scale


def validate_bbox(bbox, img_width, img_height):
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(img_width, x1))
    y1 = max(0, min(img_height, y1))
    x2 = max(0, min(img_width, x2))
    y2 = max(0, min(img_height, y2))
    return [x1, y1, x2, y2]


def format_class_display(classes, max_display=3):
    if not classes:
        return "None selected"
    
    display = ", ".join(classes[:max_display])
    if len(classes) > max_display:
        display += f" +{len(classes) - max_display} more"
    return display
