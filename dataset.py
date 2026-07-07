import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.interpolate import CubicSpline


def generate_spline(num_points=50, num_knots=5):
    """
    Generate smooth spline mu of size num_points
    """
    x_knots = np.linspace(0, 1, num_knots)
    y_knots = np.random.randn(num_knots)

    x = np.linspace(0, 1, num_points)
    spline = CubicSpline(x_knots, y_knots)

    mu = spline(x)
    return mu.astype(np.float32)


def generate_prototype_covariance(num_points=50, length_scale=4.0):
    """
    Create a proto cov 
    if points are close cov is high
    """
    idx = np.arange(num_points)
    dist = np.abs(idx[:, None] - idx[None, :])

    proto_cov = np.exp(-dist / length_scale)

    return proto_cov.astype(np.float32)


def generate_covariance_from_mu(mu, proto_cov, min_scale=0.05):
    """
    sigma(mu)

    Idea
    - the covariance depends on the local amplitude of mu
    - we scale the prototype covariance by |mu|
    """
    scale = np.abs(mu) + min_scale

    Sigma = proto_cov * np.outer(scale, scale) # scale_i * scale_j, si scale_i et scale_j sont grands, cov grande

    # petit jitter pour stabilité numérique
    Sigma += 1e-4 * np.eye(len(mu), dtype=np.float32)

    return Sigma.astype(np.float32)


def sample_correlated_noise(Sigma):
    """
    Select epsilon ~ N(0, Sigma).
    """
    z = np.random.randn(Sigma.shape[0]).astype(np.float32)
    L = np.linalg.cholesky(Sigma)
    eps = L @ z
    return eps.astype(np.float32)


class SplineDataset(Dataset):
    """
    Dataset synthetic  :
        mu : spline smooth
        x  : mu + correlated noise eps
        Sigma_true : cov to generate eps

    During training, onlu use of mu and x
    sigma_true used for eval

    resample_noise :
        False -> x is fixed once at construction (val / test)
        True  -> a fresh eps is drawn at every __getitem__ (train),
                 using the precomputed Cholesky factor of Sigma_true,
                 so the model never sees the same noise twice.
    """

    def __init__(
        self,
        num_samples=35000,
        num_points=50,
        num_knots=5,
        length_scale=4.0,
        seed=None,
        resample_noise=False,
    ):
        self.num_samples = num_samples
        self.num_points = num_points
        self.num_knots = num_knots
        self.length_scale = length_scale
        self.resample_noise = resample_noise

        if seed is not None:
            np.random.seed(seed)

        self.proto_cov = generate_prototype_covariance(
            num_points=num_points,
            length_scale=length_scale,
        )

        mus = []
        xs = []
        sigmas = []
        chols = []

        for _ in range(num_samples):
            mu = generate_spline(
                num_points=self.num_points,
                num_knots=self.num_knots,
            )

            Sigma_true = generate_covariance_from_mu(
                mu=mu,
                proto_cov=self.proto_cov,
            )

            # Cholesky computed once here, reused at every __getitem__
            # when resample_noise=True (only a matvec L @ z per draw)
            L = np.linalg.cholesky(Sigma_true).astype(np.float32)

            z = np.random.randn(self.num_points).astype(np.float32)
            eps = L @ z
            x = mu + eps

            mus.append(mu)
            xs.append(x.astype(np.float32))
            sigmas.append(Sigma_true)
            chols.append(L)

        self.mus = torch.from_numpy(np.stack(mus).astype(np.float32))
        self.xs = torch.from_numpy(np.stack(xs).astype(np.float32))
        self.sigmas = torch.from_numpy(np.stack(sigmas).astype(np.float32))
        self.chols = torch.from_numpy(np.stack(chols).astype(np.float32))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.resample_noise:
            z = torch.randn(self.num_points)
            eps = self.chols[idx] @ z
            x = self.mus[idx] + eps
        else:
            x = self.xs[idx]

        return {
            "mu": self.mus[idx],
            "x": x,
            "Sigma_true": self.sigmas[idx],
        }


if __name__ == "__main__":
    dataset = SplineDataset(num_samples=10, seed=0)

    sample = dataset[0]

    print("mu shape:", sample["mu"].shape)
    print("x shape:", sample["x"].shape)
    print("Sigma_true shape:", sample["Sigma_true"].shape)