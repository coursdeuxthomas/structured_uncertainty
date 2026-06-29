import torch
import torch.nn as nn


class CovarianceMLP(nn.Module):
    """
    Réseau de prédiction de covariance pour les splines.

    Entrée :
        mu : Tensor [B, n]

    Sortie :
        raw_cholesky : Tensor [B, n(n+1)//2]

    La sortie correspond aux coefficients de la partie triangulaire
    inférieure de L, où :

        Lambda = Sigma^{-1} = L L^T

    Attention :
        - les coefficients hors diagonale sont prédits directement ;
        - les coefficients diagonaux seront interprétés comme log(l_ii)
          dans loss.py.
    """

    def __init__(
        self,
        input_dim=50,
        hidden_dim=100,
        use_batch_norm=True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_batch_norm = use_batch_norm

        self.output_dim = input_dim * (input_dim + 1) // 2

        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))

        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))

        layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, hidden_dim))

        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))

        layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, self.output_dim))

        self.net = nn.Sequential(*layers)

        self._initialize_last_layer()

    def _initialize_last_layer(self):
        """
        Initialise doucement la dernière couche.

        Idée :
        - au début, on veut éviter que le réseau prédise des valeurs énormes ;
        - si les valeurs diagonales brutes sont proches de 0,
          alors exp(0) = 1 dans loss.py ;
        - donc L commence proche d'une matrice raisonnable.
        """

        last_layer = self.net[-1]

        if isinstance(last_layer, nn.Linear):
            nn.init.normal_(last_layer.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(last_layer.bias)

    def forward(self, mu):
        """
        Passe avant du réseau.

        mu : Tensor [B, input_dim]

        Retour :
            raw_cholesky : Tensor [B, output_dim]
        """

        if mu.dim() != 2:
            raise ValueError(f"mu must have shape [B, n], got {mu.shape}")

        if mu.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected mu with {self.input_dim} points, got {mu.shape[1]}"
            )

        raw_cholesky = self.net(mu)

        return raw_cholesky


if __name__ == "__main__":
    batch_size = 4
    n = 50

    model = CovarianceMLP(input_dim=n)

    mu = torch.randn(batch_size, n)

    raw_cholesky = model(mu)

    print("mu shape:", mu.shape)
    print("raw_cholesky shape:", raw_cholesky.shape)
    print("expected output dim:", n * (n + 1) // 2)