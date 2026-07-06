import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from loss import structured_gaussian_nll, vector_to_lower_triangular


@torch.no_grad()
def compute_predicted_covariance(model, mu):
    """
    Compute L, precision and covariance from model output.

    Lambda = Sigma^{-1} = L L^T
    Sigma = Lambda^{-1}
    """
    raw_cholesky = model(mu)

    batch_size, n = mu.shape

    L, _ = vector_to_lower_triangular(
        raw_cholesky=raw_cholesky,
        n=n,
        diag_transform="exp",
    )

    precision = torch.bmm(L, L.transpose(1, 2))
    sigma_pred = torch.linalg.inv(precision)

    return raw_cholesky, L, precision, sigma_pred


@torch.no_grad()
def sample_from_precision_cholesky(L):
    """
    Sample epsilon ~ N(0, Sigma) from precision Cholesky L.

    We have:
        Lambda = Sigma^{-1} = L L^T

    Draw:
        u ~ N(0, I)

    Solve:
        L^T epsilon = u
    """
    batch_size, n, _ = L.shape
    device = L.device
    dtype = L.dtype

    u = torch.randn(batch_size, n, 1, device=device, dtype=dtype)

    # On resout A x = b avec A = L^T, x = epsilon et b = u
    epsilon = torch.linalg.solve_triangular(
        L.transpose(1, 2),
        u,
        upper=True,
    )

    return epsilon.squeeze(-1) #[batch_size, n, 1] avant


def plot_signal(mu, x_true, x_sample, figure_path):
    """
    Plot mu, true signal and sampled signal.
    """
    plt.figure(figsize=(10, 5))

    plt.plot(mu, label="mu")
    plt.plot(x_true, label="x true")
    plt.plot(x_sample, label="mu + sampled residual")

    plt.title("Spline sample from predicted covariance")
    plt.xlabel("Point index")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()


def plot_covariance(sigma_true, sigma_pred, figure_path):
    """
    Plot true covariance, predicted covariance and absolute error.
    """
    diff = np.abs(sigma_true - sigma_pred)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axes[0].imshow(sigma_true)
    axes[0].set_title("Sigma true")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(sigma_pred)
    axes[1].set_title("Sigma predicted")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(diff)
    axes[2].set_title("Absolute error")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()


@torch.no_grad()
def evaluate(model, test_loader, device, run_dir):
    """
    Evaluate the model and save figures/metrics inside run_dir.
    """
    model.eval()

    total_nll = 0.0
    total_cov_mse = 0.0
    num_batches = 0

    for batch in test_loader:
        mu = batch["mu"].to(device)
        x = batch["x"].to(device)
        sigma_true = batch["Sigma_true"].to(device)

        raw_cholesky, L, precision, sigma_pred = compute_predicted_covariance(
            model=model,
            mu=mu,
        )

        nll = structured_gaussian_nll(
            raw_cholesky=raw_cholesky,
            x=x,
            mu=mu,
            reduction="mean",
        )

        cov_mse = torch.mean((sigma_pred - sigma_true) ** 2)

        total_nll += nll.item()
        total_cov_mse += cov_mse.item()
        num_batches += 1

    metrics = {
        "test_nll": total_nll / num_batches,
        "covariance_mse": total_cov_mse / num_batches,
    }

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Evaluation metrics:")
    print(metrics)

    # Save one visual example
    batch = next(iter(test_loader))

    mu = batch["mu"].to(device)
    x = batch["x"].to(device)
    sigma_true = batch["Sigma_true"].to(device)

    raw_cholesky, L, precision, sigma_pred = compute_predicted_covariance(
        model=model,
        mu=mu,
    )

    eps_sample = sample_from_precision_cholesky(L)
    x_sample = mu + eps_sample

    idx = 0

    mu_np = mu[idx].detach().cpu().numpy()
    x_np = x[idx].detach().cpu().numpy()
    x_sample_np = x_sample[idx].detach().cpu().numpy()

    sigma_true_np = sigma_true[idx].detach().cpu().numpy()
    sigma_pred_np = sigma_pred[idx].detach().cpu().numpy()

    figure_dir = run_dir / "figures"

    plot_signal(
        mu=mu_np,
        x_true=x_np,
        x_sample=x_sample_np,
        figure_path=figure_dir / "signal_comparison.png",
    )

    plot_covariance(
        sigma_true=sigma_true_np,
        sigma_pred=sigma_pred_np,
        figure_path=figure_dir / "covariance_comparison.png",
    )

    return metrics