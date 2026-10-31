import torch
from torch import nn


class UnconditionalMLPVAE(nn.Module):
    """Fully connected VAE with four hidden layers per encoder/decoder."""

    def __init__(self, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = 256
        self.decoder_type = "mlp"
        self.hidden_dims = [512, 512, 512, 512]

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 256, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
        )
        self.fc_means = nn.Linear(512, latent_dim)
        self.fc_logvars = nn.Linear(512, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 256 * 256),
        )

    def encode(self, x):
        """
        Input:  x, shape (B, 1, 256, 256), with values in [0, 1].
        Output: means and logvar, each with shape (B, latent_dim).
        """
        features = self.encoder(x)
        return self.fc_means(features), self.fc_logvars(features)

    def decode(self, z):
        """
        Input:  z, shape (B, latent_dim).
        Output: image logits, shape (B, 1, 256, 256).
        """
        return self.decoder(z).reshape(-1, 1, 256, 256)

    def forward(self, x):
        """
        Input:  x, shape (B, 1, 256, 256).
        Output: reconstruction logits, means, and logvar.
        """
        means, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = means + std * torch.randn_like(std)
        return self.decode(z), means, logvar

    @torch.no_grad()
    def sample(self, count):
        """
        Input:  count, the number of images to generate.
        Output: images, shape (count, 1, 256, 256), with values in [0, 1].
        """
        z = torch.randn(count, self.latent_dim, device=next(self.parameters()).device)
        return self.decode(z).sigmoid()
