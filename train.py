import argparse
from pathlib import Path

import torch

from src.config import load_config
from src.trainer import PINNTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PINN for direct cooling battery module.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/refactor_default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        #default="cuda" if torch.cuda.is_available() else "cpu",
        default="cpu",
        help="Training device: cpu or cuda.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="pinn_checkpoint.pt",
        help="Path to save the trained model state.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    trainer = PINNTrainer(cfg, device=args.device, output_dir=Path(args.save).parent)
    trainer.train()
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    trainer.save(args.save)
    print(f"Saved checkpoint to {args.save}")
    # 基于独立测试集评估误差
    trainer.evaluate_on_test()
    trainer.evaluate()


if __name__ == "__main__":
    main()
