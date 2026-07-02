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
    print("=" * 80)
    print("[START] Launching spline structured uncertainty experiment")
    print("=" * 80)

    args = parse_args()
    print(f"[ARGS] mode       = {args.mode}")
    print(f"[ARGS] run_name   = {args.run_name}")
    print(f"[ARGS] checkpoint = {args.checkpoint}")

    config = {
        "seed": 0,
        "num_points": 50,
        "num_knots": 5,
        "length_scale": 4.0,

        "train_samples": 10000,
        "val_samples": 500,
        "test_samples": 500,

        "batch_size": 64,
        "num_epochs": 50,
        "learning_rate": 1e-4,
        "grad_clip": 5.0,

        "hidden_dim": 50,
        "use_batch_norm": False,
    }

    print("\n[CONFIG] Experiment configuration:")
    for key, value in config.items():
        print(f"  - {key}: {value}")

    print("\n[SEED] Setting random seed...")
    set_seed(config["seed"])
    print("[SEED] Done.")

    print("\n[DEVICE] Checking device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Using device: {device}")

    if torch.cuda.is_available():
        print(f"[DEVICE] GPU name: {torch.cuda.get_device_name(0)}")
        print(f"[DEVICE] CUDA version: {torch.version.cuda}")
    else:
        print("[DEVICE] CUDA not available, using CPU.")

    print("\n[RUN DIR] Creating result directory...")
    run_dir = create_run_dir(args.run_name)
    print(f"[RUN DIR] Created: {run_dir}")

    print("\n[CONFIG] Saving config...")
    save_config(config, run_dir)
    print("[CONFIG] Saved.")

    print("\n[DATALOADERS] Building train / val / test datasets...")
    train_loader, val_loader, test_loader = build_dataloaders(config)
    print("[DATALOADERS] Done.")

    print(f"[DATALOADERS] Train batches: {len(train_loader)}")
    print(f"[DATALOADERS] Val batches:   {len(val_loader)}")
    print(f"[DATALOADERS] Test batches:  {len(test_loader)}")

    print("\n[DATALOADERS] Checking one batch...")
    batch = next(iter(train_loader))
    print(f"[BATCH] mu shape:          {batch['mu'].shape}")
    print(f"[BATCH] x shape:           {batch['x'].shape}")
    print(f"[BATCH] Sigma_true shape:  {batch['Sigma_true'].shape}")
    print(f"[BATCH] mu dtype:          {batch['mu'].dtype}")
    print(f"[BATCH] x dtype:           {batch['x'].dtype}")
    print(f"[BATCH] Sigma_true dtype:  {batch['Sigma_true'].dtype}")

    print("\n[MODEL] Building model and optimizer...")
    model, optimizer = build_model_and_optimizer(config, device)
    print("[MODEL] Done.")

    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"[MODEL] Total parameters:     {num_params}")
    print(f"[MODEL] Trainable parameters: {trainable_params}")

    print("\n[MODEL] Testing one forward pass...")
    mu_test = batch["mu"].to(device)
    with torch.no_grad():
        raw_cholesky = model(mu_test)

    print(f"[MODEL] Input shape:  {mu_test.shape}")
    print(f"[MODEL] Output shape: {raw_cholesky.shape}")
    print(f"[MODEL] Expected output dim: {config['num_points'] * (config['num_points'] + 1) // 2}")
    print("[MODEL] Forward pass OK.")

    if args.mode in ["train", "train_eval"]:
        print("\n" + "=" * 80)
        print("[TRAIN] Starting training...")
        print("=" * 80)

        fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            config=config,
            device=device,
            run_dir=run_dir,
        )

        print("\n[TRAIN] Training finished.")

    if args.mode == "train_eval":
        print("\n" + "=" * 80)
        print("[CHECKPOINT] Loading best checkpoint...")
        print("=" * 80)

        model = load_best_checkpoint(
            model=model,
            run_dir=run_dir,
            device=device,
        )

        print("[CHECKPOINT] Best checkpoint loaded.")

        print("\n" + "=" * 80)
        print("[EVAL] Starting evaluation...")
        print("=" * 80)

        evaluate(
            model=model,
            test_loader=test_loader,
            device=device,
            run_dir=run_dir,
        )

        print("[EVAL] Evaluation finished.")

    if args.mode == "eval":
        print("\n" + "=" * 80)
        print("[EVAL MODE] Loading checkpoint...")
        print("=" * 80)

        if args.checkpoint is None:
            raise ValueError("You must provide --checkpoint when using --mode eval")

        print(f"[CHECKPOINT] Loading from: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("[CHECKPOINT] Loaded.")

        print("\n[EVAL] Starting evaluation...")

        evaluate(
            model=model,
            test_loader=test_loader,
            device=device,
            run_dir=run_dir,
        )

        print("[EVAL] Evaluation finished.")

    print("\n" + "=" * 80)
    print("[END] Experiment completed.")
    print(f"[END] Results saved in: {run_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()