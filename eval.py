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

    # Inversion via le facteur de Cholesky, en float64 :
    # cholesky_inverse(L) = (L L^T)^{-1} = Sigma, plus stable et moins cher
    # que torch.linalg.inv(precision) quand la précision est mal conditionnée.
    sigma_pred = torch.cholesky_inverse(L.double()).to(L.dtype)

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

    # même échelle de couleur pour les 3 images
    vmin = min(sigma_true.min(), sigma_pred.min(), diff.min())
    vmax = max(sigma_true.max(), sigma_pred.max(), diff.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axes[0].imshow(sigma_true, vmin=vmin, vmax=vmax)
    axes[0].set_title("Sigma true")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(sigma_pred, vmin=vmin, vmax=vmax)
    axes[1].set_title("Sigma predicted")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(diff, vmin=vmin, vmax=vmax)
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
    num_batches = 0

    # métriques par échantillon (et non par batch) pour pouvoir
    # calculer médiane / quantiles : la MSE de covariance est à queue
    # lourde, la moyenne seule est dominée par quelques échantillons
    # mal conditionnés
    per_sample_cov_mse = []
    per_sample_min_diag = []
    per_sample_cond = []

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

        cov_mse = torch.mean((sigma_pred - sigma_true) ** 2, dim=(1, 2))

        # diagnostics de conditionnement :
        # - min des l_ii : un l_ii trop petit = précision quasi singulière
        #   dans une direction, donc covariance qui explose après inversion
        # - conditionnement de la précision (float64)
        min_diag = torch.diagonal(L, dim1=1, dim2=2).min(dim=1).values
        cond = torch.linalg.cond(precision.double())

        per_sample_cov_mse.append(cov_mse.cpu())
        per_sample_min_diag.append(min_diag.cpu())
        per_sample_cond.append(cond.cpu())

        total_nll += nll.item()
        num_batches += 1

    cov_mse_all = torch.cat(per_sample_cov_mse)
    min_diag_all = torch.cat(per_sample_min_diag)
    cond_all = torch.cat(per_sample_cond)

    metrics = {
        "test_nll": total_nll / num_batches,
        "covariance_mse": cov_mse_all.mean().item(),
        "covariance_mse_median": cov_mse_all.median().item(),
        "covariance_mse_p90": torch.quantile(cov_mse_all, 0.9).item(),
        "covariance_mse_max": cov_mse_all.max().item(),
        "min_cholesky_diag": min_diag_all.min().item(),
        "precision_cond_median": cond_all.median().item(),
        "precision_cond_max": cond_all.max().item(),
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