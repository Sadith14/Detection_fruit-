"""Reentrena solo la cabeza de EfficientNet-B0 usando embeddings congelados."""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import MODELS_DIR, construir
from training_utils import ImageDataset, normalize_batch


def _select_current_model(explicit_path):
    if explicit_path:
        path = Path(explicit_path)
    else:
        active = MODELS_DIR / "actual.pt"
        path = active if active.exists() else MODELS_DIR / "effnet_b0_tl.pt"
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el modelo actual/base: {path}")
    return path


def _extract_embeddings(model, dataset, mean, std, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_embeddings = []
    all_labels = []
    model.base.eval()
    with torch.inference_mode():
        for images, labels in loader:
            images = normalize_batch(images, mean, std)
            embeddings = model.base(images)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)
    return torch.cat(all_embeddings), torch.cat(all_labels)


def train(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(args.cpu_threads)

    current_path = _select_current_model(args.current_model)
    model, (mean, std) = construir("eff")
    model.load_state_dict(torch.load(current_path, map_location="cpu"))
    for parameter in model.base.parameters():
        parameter.requires_grad = False
    model.base.eval()

    roots = [args.train_dir]
    if args.new_data_dir:
        roots.append(args.new_data_dir)
    dataset = ImageDataset(roots, require_all_classes=True, allow_empty_roots=True)
    embeddings, labels = _extract_embeddings(model, dataset, mean, std, args.batch_size)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loss_function = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)

    for epoch in range(args.epochs):
        model.head.train()
        order = torch.randperm(len(labels), generator=generator)
        total_loss = 0.0
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model.head(embeddings[indices])
            loss = loss_function(logits, labels[indices])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indices)
        print(f"epoch {epoch + 1}/{args.epochs} - loss={total_loss / len(labels):.6f}")

    output = Path(args.candidate_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    print(f"Candidato guardado en {output} (base: {current_path}; imágenes: {len(dataset)})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True, help="Dataset original con subcarpetas por clase")
    parser.add_argument("--new-data-dir", help="Imágenes nuevas ya descargadas, con subcarpetas por clase")
    parser.add_argument("--current-model", help="Pesos actuales; por defecto actual.pt o effnet_b0_tl.pt")
    parser.add_argument("--candidate-output", default=str(MODELS_DIR / "candidato.pt"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.cpu_threads < 1:
        parser.error("epochs, batch-size y cpu-threads deben ser mayores que cero")
    train(args)


if __name__ == "__main__":
    main()
