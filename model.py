"""Modelos y preprocesamiento compartidos por la app y el ciclo de entrenamiento."""
import json
import os
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
CFG = json.load(open(ROOT / "config.json", encoding="utf-8"))
CLASSES = CFG["class_names"]
IMG_SIZE = 224

# clave visible en la app -> (archivo de pesos, tipo)
MODELOS = {
    "EfficientNet-B0 (transfer learning)": ("effnet_b0_tl.pt", "eff"),
    "CNN desde cero": ("cnn_scratch.pt", "cnn"),
    "ViT-B/16 (Vision Transformer)": ("vit_b16.pt", "vit"),
}
DEFECTO = "EfficientNet-B0 (transfer learning)"


# ----------------------------- Arquitecturas -----------------------------
def _block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.MaxPool2d(2))


class CNN(nn.Module):
    def __init__(self, n=8):
        super().__init__()
        self.features = nn.Sequential(_block(3, 32), _block(32, 64), _block(64, 128), _block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.4),
                                  nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
                                  nn.Linear(128, n))

    def forward(self, x):
        return self.head(self.features(x))


class TLNet(nn.Module):
    def __init__(self, backbone="efficientnet_b0", n=8):
        super().__init__()
        self.base = timm.create_model(backbone, pretrained=False, num_classes=0)  # los pesos vienen del .pt
        f = self.base.num_features
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(f, 256), nn.ReLU(inplace=True),
                                  nn.Dropout(0.2), nn.Linear(256, n))

    def forward(self, x):
        return self.head(self.base(x))


def construir(tipo, n=None):
    """Devuelve (modelo, (mean, std)) con la normalización usada en el entrenamiento."""
    n = n or len(CLASSES)
    if tipo == "cnn":
        return CNN(n), (CFG["mean_train"], CFG["std_train"])
    if tipo == "eff":
        m = TLNet("efficientnet_b0", n)
        dc = timm.data.resolve_data_config({}, model=m.base)
        return m, (dc["mean"], dc["std"])
    if tipo == "vit":
        m = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=n, drop_rate=0.1)
        dc = timm.data.resolve_data_config({}, model=m)
        return m, (dc["mean"], dc["std"])
    raise ValueError(tipo)


# ----------------------------- Preprocesamiento -----------------------------
def letterbox(im, size=IMG_SIZE):
    """Corrige EXIF, escala sin deformar y rellena con el color del borde."""
    im = ImageOps.exif_transpose(im).convert("RGB")
    arr = np.asarray(im)
    borde = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    color = tuple(int(v) for v in np.median(borde, axis=0))
    w, h = im.size
    s = size / max(w, h)
    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    lienzo = Image.new("RGB", (size, size), color)
    lienzo.paste(im, ((size - im.size[0]) // 2, (size - im.size[1]) // 2))
    return lienzo


class Clasificador:
    def __init__(self, clave):
        archivo, tipo = MODELOS[clave]
        ruta = _ruta_pesos(clave, archivo)
        if not ruta.exists():
            raise FileNotFoundError(f"No se encontraron los pesos del modelo: {ruta}")
        self.clave = clave
        self.modelo, (mean, std) = construir(tipo)
        self.modelo.load_state_dict(torch.load(ruta, map_location="cpu"))
        self.modelo.eval()
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    @torch.inference_mode()
    def probs(self, img):
        x = torch.from_numpy(np.asarray(letterbox(img), dtype=np.float32) / 255.0).permute(2, 0, 1)
        x = ((x - self.mean) / self.std).unsqueeze(0)
        return F.softmax(self.modelo(x).float(), dim=1)[0].numpy()


def _ruta_pesos(clave, archivo):
    if clave == DEFECTO:
        volume = os.getenv("MODEL_VOLUME")
        if volume:
            return Path(volume) / "actual.pt"
        actual = MODELS_DIR / "actual.pt"
        if actual.exists():
            return actual
    return MODELS_DIR / archivo


_CACHE = {}


def obtener(clave):
    if clave not in _CACHE:
        _CACHE[clave] = Clasificador(clave)
    return _CACHE[clave]


def disponibles():
    return [
        clave for clave, (archivo, _) in MODELOS.items()
        if _ruta_pesos(clave, archivo).exists()
    ]


# ----------------------------- Presentación de resultados (ES / EN) -----------------------------
FRUTA_ES = {"Apple": "manzana", "Banana": "banana", "Orange": "naranja", "Strawberry": "fresa"}
FRUTA_EN = {"Apple": "apple", "Banana": "banana", "Orange": "orange", "Strawberry": "strawberry"}
EMOJI = {"Apple": "🍎", "Banana": "🍌", "Orange": "🍊", "Strawberry": "🍓"}


def _partir(c):
    estado = "Fresh" if c.startswith("Fresh") else "Rotten"
    return estado, c[len(estado):]


def nombre(c):
    estado, fruta = _partir(c)
    return f"{EMOJI.get(fruta, '')} {FRUTA_ES.get(fruta, fruta).capitalize()} {'fresca' if estado == 'Fresh' else 'podrida'}"


def etiquetas(p):
    """Diccionario {nombre visible: probabilidad} para gr.Label."""
    return {nombre(c): float(p[i]) for i, c in enumerate(CLASSES)}


def resumen(p, umbral, ms, clave):
    i = int(np.argmax(p)); conf = float(p[i]); c = CLASSES[i]
    estado, fruta = _partir(c)
    p_fresca = float(sum(p[k] for k, cc in enumerate(CLASSES) if cc.startswith("Fresh")))
    p_fruta = {}
    for k, cc in enumerate(CLASSES):
        p_fruta[_partir(cc)[1]] = p_fruta.get(_partir(cc)[1], 0.0) + float(p[k])
    fruta_top = max(p_fruta, key=p_fruta.get)
    pie = f"\n\n⏱ {ms:.0f} ms · {clave}"
    if conf < umbral:
        return (f"### ❓ No estoy seguro\nMayor probabilidad: **{nombre(c)}** ({conf:.0%}), por debajo del umbral "
                f"({umbral:.0%}).\n\nPrueba con más luz, la fruta centrada y un fondo liso." + pie)
    es_fresca = estado == "Fresh"
    return (f"### {nombre(c)}\n*{estado.lower()} {FRUTA_EN[fruta]}* · confianza **{conf:.0%}**\n\n"
            f"| | Predicción | Prob. |\n|---|---|---|\n"
            f"| Fruta | {FRUTA_ES[fruta_top].capitalize()} / {FRUTA_EN[fruta_top]} | {p_fruta[fruta_top]:.0%} |\n"
            f"| Estado | {'Fresca / fresh' if es_fresca else 'Podrida / rotten'} | "
            f"{(p_fresca if es_fresca else 1 - p_fresca):.0%} |" + pie)
