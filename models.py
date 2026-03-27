"""
Deep Packet — PyTorch models for encrypted traffic classification.

Implementation of the approach described in:
    Lotfollahi, M., Jafari Siavoshani, M., Shirali Hossein Zade, R., & Saberian, M. (2017).
    "Deep Packet: A Novel Approach For Encrypted Traffic Classification Using Deep Learning."
    https://arxiv.org/abs/1709.02656

Models:
    - DeepPacketCNN: 1D CNN for raw packet classification
    - StackedAutoEncoder: Pre-trained stacked autoencoder for feature extraction + classification
"""

import torch
from torch import nn


class DeepPacketCNN(nn.Module):
    """1D Convolutional Neural Network for encrypted traffic classification.

    Architecture (from the paper):
        - Conv1D (1 → 200 filters, kernel=5, stride=2) + BN + ReLU
        - Conv1D (200 → 100 filters, kernel=4, stride=1) + BN + ReLU
        - AvgPool1d(2)
        - 7 fully connected layers (100*372 → 600 → 500 → 400 → 300 → 200 → 100 → 50)
          each with Dropout(0.25) + ReLU
        - Output layer + LogSoftmax

    Input: (batch, 1, 1500) — raw packet bytes (padded/truncated to 1500)
    Output: (n_classes,) — log-probabilities

    Args:
        n_classes: Number of traffic classes to predict.
    """

    def __init__(self, n_classes: int):
        super().__init__()

        # Convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv1d(1, 200, kernel_size=5, stride=2),
            nn.BatchNorm1d(200),
            nn.ReLU(inplace=True),
            nn.Conv1d(200, 100, kernel_size=4, stride=1),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(2),
        )

        # Fully connected classifier
        # Input size: 100 * 372 = 37200 (determined by conv output for 1500-length input)
        self.classifier = nn.Sequential(
            nn.Linear(100 * 372, 600),
            nn.Dropout(0.25),
            nn.ReLU(True),
            nn.Linear(600, 500),
            nn.Dropout(0.25),
            nn.ReLU(True),
            nn.Linear(500, 400),
            nn.Dropout(0.25),
            nn.ReLU(True),
            nn.Linear(400, 300),
            nn.Dropout(0.25),
            nn.ReLU(True),
            nn.Linear(300, 200),
            nn.Dropout(0.25),
            nn.ReLU(True),
            nn.Linear(200, 100),
            nn.Dropout(0.25),
            nn.ReLU(True),
            nn.Linear(100, 50),
            nn.Dropout(0.25),
            nn.ReLU(True),
        )

        self.fc_out = nn.Linear(50, n_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, 1, 1500) — raw packet bytes.

        Returns:
            Log-probabilities of shape (batch, n_classes).
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        x = self.fc_out(x)
        return self.log_softmax(x)


class AutoEncoder(nn.Module):
    """Single-layer autoencoder for greedy pre-training.

    Combines encoding and decoding in one module. During training,
    the forward pass computes the reconstruction loss and updates
    weights internally (layer-wise pre-training strategy from the paper).

    Args:
        input_size: Dimensionality of input features.
        output_size: Dimensionality of the encoded representation (bottleneck).
    """

    def __init__(self, input_size: int, output_size: int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.Dropout(0.05),
            nn.ReLU(True),
        )
        self.decoder = nn.Linear(output_size, input_size)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input. During training, also updates weights via reconstruction loss.

        Args:
            x: Input tensor.

        Returns:
            Encoded representation (detached).
        """
        x = x.detach()

        y = self.encoder(x)

        if self.training:
            x_reconstructed = self.decoder(y)
            loss = self.criterion(x_reconstructed, x.data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return y.detach()


class StackedAutoEncoder(nn.Module):
    """Stacked autoencoder with classification head.

    Architecture (from the paper):
        - 5 pre-trained autoencoders (1500 → 400 → 300 → 200 → 100 → 50)
        - Classification layer (50 → n_classes) + LogSoftmax

    Pre-training strategy:
        Each autoencoder is trained greedily layer-by-layer:
        1. Train AE1 on raw input (1500 → 400)
        2. Encode data through AE1, train AE2 on encoded data (400 → 300)
        3. Repeat for remaining layers
        4. Fine-tune the full network end-to-end

    Args:
        n_classes: Number of traffic classes to predict.
    """

    def __init__(self, n_classes: int):
        super().__init__()

        self.ae1 = AutoEncoder(1500, 400)
        self.ae2 = AutoEncoder(400, 300)
        self.ae3 = AutoEncoder(300, 200)
        self.ae4 = AutoEncoder(200, 100)
        self.ae5 = AutoEncoder(100, 50)

        self.fc_out = nn.Linear(50, n_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through stacked autoencoders + classifier.

        Args:
            x: Input tensor of shape (batch, 1500) — flattened packet bytes.

        Returns:
            Log-probabilities of shape (batch, n_classes).
        """
        x = self.ae1(x)
        x = self.ae2(x)
        x = self.ae3(x)
        x = self.ae4(x)
        x = self.ae5(x)

        x = self.fc_out(x)
        return self.log_softmax(x)
