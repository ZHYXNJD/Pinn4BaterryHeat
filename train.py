import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from src.config import config_to_dict, load_config
from src.trainer import PINNTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PINN for direct cooling battery module.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/refactor_weights_scheme_d.yaml",
        # default="configs/data_only.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device: cpu or cuda. Default: cuda if available.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Base output directory for checkpoints, test_results, plots, and TensorBoard logs.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="pinn_checkpoint.pt",
        help="Checkpoint filename. Default derives from config name, e.g. pinn_refactor_weights_scheme_a.pt",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    output_base = Path(args.output_dir)
    output_dir = output_base / f"{cfg.conditions[0].name}_{timestamp}"
    checkpoint_dir = output_dir / "checkpoints"
    test_results_dir = output_dir / "test_results"
    checkpoint_filename = args.save if args.save != "pinn_checkpoint.pt" else f"pinn_{cfg.conditions[0].name}.pt"
    checkpoint_path = checkpoint_dir / checkpoint_filename

    output_dir.mkdir(parents=True, exist_ok=True)
    training_config = {
        "config_path": args.config,
        "config": config_to_dict(cfg),
        "runtime": {
            "device": args.device,
            "output_dir": str(output_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "test_results_dir": str(test_results_dir),
            "checkpoint_filename": checkpoint_filename,
        },
    }
    config_json_path = output_dir / "training_config.json"
    with config_json_path.open("w", encoding="utf-8") as f:
        json.dump(training_config, f, indent=2, ensure_ascii=False)
    print(f"Saved training config to {config_json_path}")

    trainer = PINNTrainer(
        cfg,
        device=args.device,
        output_dir=str(output_dir),
        checkpoint_dir=str(checkpoint_dir),
        test_results_dir=str(test_results_dir),
        config_name=cfg.conditions[0].name,
    )
    trainer.train()

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer.save(str(checkpoint_path))
    print(f"Saved checkpoint to {checkpoint_path}")

    trainer.evaluate_on_test()
    trainer.evaluate()


if __name__ == "__main__":
    main()
