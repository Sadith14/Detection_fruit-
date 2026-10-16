"""Evalúa un candidato contra el modelo actual o prueba una imagen individual."""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import DataLoader

from model import CLASSES, MODELS_DIR, construir, disponibles, nombre, obtener
from training_utils import ImageDataset, normalize_batch


def _default_model_path():
    actual = MODELS_DIR / "actual.pt"
    return actual if actual.exists() else MODELS_DIR / "effnet_b0_tl.pt"


def _load_classifier(path):
    model, (mean, std) = construir("eff")
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model, mean, std


@torch.inference_mode()
def evaluate(model_path, loader):
    model, mean, std = _load_classifier(model_path)
    predictions = []
    labels = []
    for images, targets in loader:
        images = normalize_batch(images, mean, std)
        predictions.extend(model(images).argmax(dim=1).cpu().tolist())
        labels.extend(targets.tolist())
    return {
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
    }


def evaluate_candidate(args):
    baseline_path = args.baseline or _default_model_path()
    candidate_path = Path(args.candidate)
    if not baseline_path.is_file():
        raise FileNotFoundError(f"No existe el modelo actual: {baseline_path}")
    if not candidate_path.is_file():
        raise FileNotFoundError(f"No existe el candidato: {candidate_path}")

    dataset = ImageDataset(args.validation_dir, require_all_classes=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    baseline = evaluate(baseline_path, loader)
    candidate = evaluate(candidate_path, loader)
    improved = candidate["f1_macro"] > baseline["f1_macro"]
    metrics = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "validation_dir": str(args.validation_dir),
        "baseline_model": str(baseline_path),
        "candidate_model": str(candidate_path),
        "selection_metric": "f1_macro",
        "baseline": baseline,
        "candidate": candidate,
        "improved": improved,
    }
    output = Path(args.metrics_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"actual.pt  f1_macro={baseline['f1_macro']:.6f} accuracy={baseline['accuracy']:.6f}")
    print(f"candidato  f1_macro={candidate['f1_macro']:.6f} accuracy={candidate['accuracy']:.6f}")
    print(f"Métricas guardadas en {output}")
    if not improved:
        print("El candidato no mejora el f1_macro del modelo actual; no debe promoverse.", file=sys.stderr)
        return 1
    return 0


def smoke_test(image_path):
    from PIL import Image

    img = Image.open(image_path)
    for clave in disponibles():
        classifier = obtener(clave)
        classifier.probs(img)
        start = time.perf_counter()
        probabilities = classifier.probs(img)
        elapsed = (time.perf_counter() - start) * 1000
        top = probabilities.argsort()[::-1][:3]
        print(f"\n{clave}  ({elapsed:.0f} ms)")
        for index in top:
            print(f"  {nombre(CLASSES[index]):<28} {probabilities[index]:6.1%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", help="Imagen individual para la prueba rápida")
    parser.add_argument("--validation-dir", help="Raíz fija de validación, con una carpeta por clase")
    parser.add_argument("--baseline", type=Path, help="Pesos del modelo actual (por defecto actual.pt)")
    parser.add_argument("--candidate", default=str(MODELS_DIR / "candidato.pt"))
    parser.add_argument("--metrics-output", default="metricas.json")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.image:
        smoke_test(args.image)
        return 0
    if not args.validation_dir:
        parser.error("indica --validation-dir para comparar modelos o una ruta de imagen para la prueba rápida")
    if args.batch_size < 1:
        parser.error("--batch-size debe ser mayor que cero")
    return evaluate_candidate(args)


if __name__ == "__main__":
    raise SystemExit(main())
