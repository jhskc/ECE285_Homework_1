import torch
from torch import nn


class UnconditionalConvVAE(nn.Module):
    """Unconditional Convolutional VAE for single grayscale channel 256 x 256 images."""

    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = 256
        self.decoder_type = "transposed_conv"

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Last encoder conv layer output is of shape (B, 256, 8, 8). Is 16,384 features when nn.Flatten()ed.
        self.fc_means     = nn.Linear(in_features=256 * 8 * 8, out_features=latent_dim)
        self.fc_logvars = nn.Linear(in_features=256 * 8 * 8, out_features=latent_dim) 
        self.fc_decoder = nn.Linear(in_features=latent_dim, out_features=256 * 8 * 8)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1),
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
        features = self.fc_decoder(z).reshape(-1, 256, 8, 8)
        return self.decoder(features)

    def forward(self, x):
        """
        Input:  x, shape (B, 1, 256, 256), with values in [0, 1].
        Output: reconstruction logits, shape (B, 1, 256, 256).
                means and logvar, each with shape (B, latent_dim).
        """
        means, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar) # To ensure positive variance , exponentiate in forward pass. 
        epsilon = torch.randn_like(std)
        z = means + std * epsilon
        return self.decode(z), means, logvar

    @torch.no_grad()
    def sample(self, count):
        """
        Input:  count, the number of images to generate.
        Output: images, shape (count, 1, 256, 256), with values in [0, 1].
        """
        device = next(self.parameters()).device
        z = torch.randn(count, self.latent_dim, device=device)
        return self.decode(z).sigmoid()
