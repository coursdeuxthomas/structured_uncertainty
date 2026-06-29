import json
import torch

from loss import structured_gaussian_nll


def train_one_epoch(model, dataloader, optimizer, device, grad_clip=5.0):
    """
    Train the model for one epoch.
    """
    model.train()

    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        mu = batch["mu"].to(device)
        x = batch["x"].to(device)

        optimizer.zero_grad()

        raw_cholesky = model(mu)

        loss = structured_gaussian_nll(
            raw_cholesky=raw_cholesky,
            x=x,
            mu=mu,
            reduction="mean",
        )

        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def validate(model, dataloader, device):
    """
    Evaluate the model on the validation set.
    """
    model.eval()

    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        mu = batch["mu"].to(device)
        x = batch["x"].to(device)

        raw_cholesky = model(mu)

        loss = structured_gaussian_nll(
            raw_cholesky=raw_cholesky,
            x=x,
            mu=mu,
            reduction="mean",
        )

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, checkpoint_path):
    """
    Save a model checkpoint.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }

    torch.save(checkpoint, checkpoint_path)


def fit(model, train_loader, val_loader, optimizer, config, device, run_dir):
    """
    Full training loop.

    This function does not create datasets or folders.
    It only trains the model and saves checkpoints/history inside run_dir.
    """
    num_epochs = config["num_epochs"]
    grad_clip = config["grad_clip"]

    best_val_loss = float("inf")
    history = {
        "train_loss": [],
        "val_loss": [],
    }

    checkpoint_dir = run_dir / "checkpoints"

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=grad_clip,
        )

        val_loss = validate(
            model=model,
            dataloader=val_loader,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch [{epoch:03d}/{num_epochs}] "
            f"Train NLL: {train_loss:.4f} "
            f"Val NLL: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                checkpoint_path=checkpoint_dir / "best_model.pth",
            )

            print(f"Saved best model with Val NLL: {best_val_loss:.4f}")

        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=4)

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=num_epochs,
        train_loss=train_loss,
        val_loss=val_loss,
        checkpoint_path=checkpoint_dir / "last_model.pth",
    )

    return history