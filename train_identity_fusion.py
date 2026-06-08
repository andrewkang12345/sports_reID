from __future__ import annotations

import argparse

from soccer_identity.training.train_fusion import TrainConfig, train_fusion_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight roster-conditioned fusion head.")
    parser.add_argument("--train_jsonl", required=True, help="JSONL rows with candidate features and binary labels.")
    parser.add_argument("--output_path", required=True, help="Where to write the trained torch checkpoint.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        train_jsonl=args.train_jsonl,
        output_path=args.output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
    )
    if args.device:
        cfg.device = args.device
    metrics = train_fusion_model(cfg)
    print(metrics)


if __name__ == "__main__":
    main()
