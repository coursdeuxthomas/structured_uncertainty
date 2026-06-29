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

    Sigma = proto_cov * np.outer(scale, scale)

    # petit jitter pour stabilité numérique
    Sigma += 1e-4 * np.eye(len(mu), dtype=np.float32)

    return Sigma.astype(np.float32)


def sample_correlated_noise(Sigma):
    """
    Select epsilon ~ N(0, Sigma).
    """
    mean = np.zeros(Sigma.shape[0], dtype=np.float32)
    eps = np.random.multivariate_normal(mean, Sigma)

    return eps.astype(np.float32)


class SplineDataset(Dataset):
    """
    Dataset synthetic  :
        mu : spline smooth
        x  : mu + correlated noise eps
        Sigma_true : cov to generate eps

    During training, onlu use of mu and x
    sigma_true used for eval
    """

    def __init__(
        self,
        num_samples=35000,
        num_points=50,
        num_knots=5,
        length_scale=4.0,
        seed=None,
    ):
        self.num_samples = num_samples
        self.num_points = num_points
        self.num_knots = num_knots
        self.length_scale = length_scale

        if seed is not None:
            np.random.seed(seed)

        self.proto_cov = generate_prototype_covariance(
            num_points=num_points,
            length_scale=length_scale,
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        mu = generate_spline(
            num_points=self.num_points,
            num_knots=self.num_knots,
        )

        Sigma_true = generate_covariance_from_mu(
            mu=mu,
            proto_cov=self.proto_cov,
        )

        eps = sample_correlated_noise(Sigma_true)

        x = mu + eps

        return {
            "mu": torch.from_numpy(mu),
            "x": torch.from_numpy(x.astype(np.float32)),
            "Sigma_true": torch.from_numpy(Sigma_true),
        }


if __name__ == "__main__":
    dataset = SplineDataset(num_samples=10, seed=0)

    sample = dataset[0]

    print("mu shape:", sample["mu"].shape)
    print("x shape:", sample["x"].shape)
    print("Sigma_true shape:", sample["Sigma_true"].shape)