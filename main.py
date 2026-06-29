import argparse
import json
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SplineDataset
from model import CovarianceMLP
from train import fit
from eval import evaluate


def set_seed(seed):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_run_dir(run_name):
    """
    Create a new result folder for one experiment.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if run_name is None:
        run_name = "spline"

    run_dir = Path("results") / f"{run_name}_{timestamp}"

    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)

    return run_dir


def save_config(config, run_dir):
    """
    Save experiment configuration.
    """
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)


def build_dataloaders(config):
    """
    Create train, validation and test dataloaders.
    """
    train_dataset = SplineDataset(
        num_samples=config["train_samples"],
        num_points=config["num_points"],
        num_knots=config["num_knots"],
        length_scale=config["length_scale"],
        seed=config["seed"],
    )

    val_dataset = SplineDataset(
        num_samples=config["val_samples"],
        num_points=config["num_points"],
        num_knots=config["num_knots"],
        length_scale=config["length_scale"],
        seed=config["seed"] + 1,
    )

    test_dataset = SplineDataset(
        num_samples=config["test_samples"],
        num_points=config["num_points"],
        num_knots=config["num_knots"],
        length_scale=config["length_scale"],
        seed=config["seed"] + 2,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


def build_model_and_optimizer(config, device):
    """
    Create the model and optimizer.
    """
    model = CovarianceMLP(
        input_dim=config["num_points"],
        hidden_dim=config["hidden_dim"],
        use_batch_norm=config["use_batch_norm"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
    )

    return model, optimizer


def load_best_checkpoint(model, run_dir, device):
    """
    Load the best checkpoint from the current run.
    """
    checkpoint_path = run_dir / "checkpoints" / "best_model.pth"

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Loaded best checkpoint from epoch {checkpoint['epoch']}")

    return model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        default="train_eval",
        choices=["train", "eval", "train_eval"],
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default="spline",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path for eval mode only.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = {
        "seed": 0,
        "num_points": 50,
        "num_knots": 5,
        "length_scale": 4.0,

        "train_samples": 35000,
        "val_samples": 1000,
        "test_samples": 1000,

        "batch_size": 64,
        "num_epochs": 200,
        "learning_rate": 1e-4,
        "grad_clip": 5.0,

        "hidden_dim": 100,
        "use_batch_norm": True,
    }

    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    run_dir = create_run_dir(args.run_name)
    save_config(config, run_dir)

    print("Run directory:", run_dir)

    train_loader, val_loader, test_loader = build_dataloaders(config)

    model, optimizer = build_model_and_optimizer(config, device)

    if args.mode in ["train", "train_eval"]:
        fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            config=config,
            device=device,
            run_dir=run_dir,
        )

    if args.mode == "train_eval":
        model = load_best_checkpoint(
            model=model,
            run_dir=run_dir,
            device=device,
        )

        evaluate(
            model=model,
            test_loader=test_loader,
            device=device,
            run_dir=run_dir,
        )

    if args.mode == "eval":
        if args.checkpoint is None:
            raise ValueError("You must provide --checkpoint when using --mode eval")

        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        evaluate(
            model=model,
            test_loader=test_loader,
            device=device,
            run_dir=run_dir,
        )


if __name__ == "__main__":
    main()