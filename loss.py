import torch


def vector_to_lower_triangular(raw_cholesky, n, diag_transform="exp", clamp_log_diag=10.0):
    """
    Convertit la sortie du réseau en matrice triangulaire inférieure L.

    raw_cholesky : Tensor de shape [B, n(n+1)//2]
        Sortie brute du réseau.
        Elle contient les coefficients de la partie triangulaire inférieure de L.

    n : int
        Dimension du signal. Pour les splines de l'article, n = 50.

    diag_transform : str
        Transformation appliquée à la diagonale pour garantir l_ii > 0.
        Dans l'article, le réseau estime log(l_ii), donc on utilise exp.

    clamp_log_diag : float
        Limite les valeurs de la diagonale brute avant exp pour éviter les explosions numériques.

    Retourne :
        L : Tensor de shape [B, n, n]
        log_diag : Tensor de shape [B, n]
            log des éléments diagonaux de L.
    """

    batch_size = raw_cholesky.shape[0]
    expected_size = n * (n + 1) // 2

    if raw_cholesky.shape[1] != expected_size:
        raise ValueError(
            f"raw_cholesky should have shape [B, {expected_size}], "
            f"but got {raw_cholesky.shape}"
        )

    device = raw_cholesky.device
    dtype = raw_cholesky.dtype

    L = torch.zeros(batch_size, n, n, device=device, dtype=dtype)

    row_idx, col_idx = torch.tril_indices(row=n, col=n, offset=0, device=device)

    L[:, row_idx, col_idx] = raw_cholesky

    raw_diag = torch.diagonal(L, dim1=1, dim2=2) #le réseau estime log(l_ii)

    if diag_transform == "exp":
        log_diag = torch.clamp(raw_diag, min=-clamp_log_diag, max=clamp_log_diag)
        diag = torch.exp(log_diag)
    else:
        raise ValueError(f"Unknown diag_transform: {diag_transform}")

    L = L - torch.diag_embed(raw_diag) + torch.diag_embed(diag)

    return L, log_diag


def structured_gaussian_nll(raw_cholesky, x, mu, reduction="mean"):
    """
    Calcule la negative log-likelihood utilisée dans l'article :

        loss = log|Sigma| + (x - mu)^T Sigma^{-1} (x - mu)

    avec :

        Sigma^{-1} = Lambda = L L^T

    Donc :

        loss = -2 sum_i log(l_ii) + || L^T (x - mu) ||^2

    raw_cholesky : Tensor [B, n(n+1)//2]
        Sortie du réseau.

    x : Tensor [B, n]
        Signal observé : x = mu + epsilon.

    mu : Tensor [B, n]
        Spline lisse, entrée du réseau.

    reduction : str
        "mean" : moyenne sur le batch
        "sum"  : somme sur le batch
        "none" : une loss par exemple

    Retourne :
        loss : Tensor scalaire si reduction="mean" ou "sum",
               Tensor [B] si reduction="none".
    """

    if x.shape != mu.shape:
        raise ValueError(f"x and mu must have same shape, got {x.shape} and {mu.shape}")

    if x.dim() != 2:
        raise ValueError(f"x and mu must be [B, n], got {x.shape}")

    batch_size, n = x.shape

    L, log_diag = vector_to_lower_triangular(
        raw_cholesky=raw_cholesky,
        n=n,
        diag_transform="exp",
    )

    residual = x - mu

    residual = residual.unsqueeze(-1)

    y = torch.bmm(L.transpose(1, 2), residual)

    y = y.squeeze(-1)

    mahalanobis = torch.sum(y ** 2, dim=1)

    log_det_sigma = -2.0 * torch.sum(log_diag, dim=1)

    loss_per_sample = log_det_sigma + mahalanobis

    if reduction == "mean":
        return loss_per_sample.mean()

    if reduction == "sum":
        return loss_per_sample.sum()

    if reduction == "none":
        return loss_per_sample

    raise ValueError(f"Unknown reduction: {reduction}")


if __name__ == "__main__":
    batch_size = 4
    n = 50
    output_size = n * (n + 1) // 2

    raw_cholesky = torch.zeros(batch_size, output_size, requires_grad=True)

    mu = torch.zeros(batch_size, n)
    x = torch.randn(batch_size, n)

    loss = structured_gaussian_nll(raw_cholesky, x, mu)

    print("loss:", loss.item())

    loss.backward()

    print("raw_cholesky grad shape:", raw_cholesky.grad.shape)