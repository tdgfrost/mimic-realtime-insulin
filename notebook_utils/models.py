import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm  # <- helps with temporal difference stability


class GroupedEmbeddings(nn.Module):
    """
    Class for doing one-to-many embeddings for multiple different floats i.e., value, time, etc.
    """

    def __init__(self, n_features: int, n_groups: int, hidden_dim: int = 128, out_dim: int = 128):
        super().__init__()
        self.n_groups = n_groups
        self.n_features = n_features
        self.hidden_dim = hidden_dim

        self.feature_embed = nn.Embedding(n_features, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.conv1 = nn.Conv1d(
                in_channels=n_groups,
                out_channels=n_groups * hidden_dim * 2,
                kernel_size=1,
                groups=n_groups
        )
        self.conv2 = nn.Conv1d(
                in_channels=n_groups * hidden_dim,
                out_channels=n_groups * out_dim,
                kernel_size=1,
                groups=n_groups
        )

    def forward(self, features: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        N, L, _ = x.shape
        f_emb = self.feature_embed(features.long()).transpose(1, 2).view(N, 1, self.hidden_dim, L)

        x = x.transpose(1, 2)
        x = F.glu(self.conv1(x), dim=1)
        x = x.view(N, self.n_groups, self.hidden_dim, L)
        x = x + f_emb

        # Second projection
        x = x.view(N, -1, L)
        x = self.conv2(x)

        x = x.transpose(1, 2).view(N, L, self.n_groups, -1)
        x = x.sum(dim=2)
        out = self.norm(x)
        return out


class CNNLSTMModel(nn.Module):
    """
    PyTorch model using CNN + LSTM
    """

    def __init__(
            self,
            n_features: int,
            hidden_dim: int = 64,
            n_cnn_layers: int = 2,
            n_lstm_layers: int = 1,
            dropout: float = 0.2,
            out_dim: int = 1
    ):
        super().__init__()
        self.n_cnn_layers = n_cnn_layers
        self.n_lstm_layers = n_lstm_layers

        # Embeddings
        self.embedding_net = GroupedEmbeddings(n_features=n_features, n_groups=2, hidden_dim=hidden_dim, out_dim=hidden_dim)
        self.embedding_dropout = nn.Dropout1d(dropout)

        # CNN Layers
        cnn_blocks = []
        for _ in range(n_cnn_layers):
            cnn_blocks.extend([
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(num_groups=8, num_channels=hidden_dim),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout1d(dropout),
            ])
        self.cnn = nn.Sequential(*cnn_blocks)

        # LSTM Layer
        self.h0 = nn.Parameter(torch.zeros(n_lstm_layers, 1, hidden_dim))
        self.c0 = nn.Parameter(torch.zeros(n_lstm_layers, 1, hidden_dim))
        nn.init.normal_(self.h0, mean=0.0, std=0.01)
        nn.init.normal_(self.c0, mean=0.0, std=0.01)

        self.lstm = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True
        )

        # Dense Decoding Layers
        self.dense = nn.Sequential(
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def soft_update(self, target_model: nn.Module, polyak_tau: float = 0.005):
        """
        Applies an exponential moving average update to the target model parameters.
        """
        with torch.no_grad():
            for param, target_param in zip(self.parameters(), target_model.parameters()):
                target_param.data.lerp_(param.data, polyak_tau)

    def get_lengths_after_conv(self, nan_mask: torch.Tensor) -> torch.Tensor:
        """
        Get the new sequence lengths after CNN convolutions
        """
        real_mask = (~nan_mask).float()
        for _ in range(self.n_cnn_layers):
            real_mask = F.avg_pool1d(real_mask, kernel_size=2, stride=2)

        real_mask = real_mask.squeeze(1)
        lengths = (real_mask == 1).sum(dim=-1).view(-1, 1, 1) - 1

        return lengths.clamp(min=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expects input of shape (N, L, 3): feature, time, value
        """
        N, L, _ = x.shape
        features, float_inputs = x[..., 0], x[..., 1:]

        # Find the NaNs and mask them
        src_nan_mask = float_inputs[:, :, 0].isnan().view(N, 1, L)
        float_inputs = float_inputs.nan_to_num(nan=0.0)
        features = features.nan_to_num(nan=0.0)

        # Perform embedding
        hidden_state = self.embedding_net(features, float_inputs)

        # Prepare for CNN: (N, L, C) -> (N, C, L)
        hidden_state = hidden_state.transpose(1, 2).contiguous()
        hidden_state = self.embedding_dropout(hidden_state)

        # Perform convolutions
        hidden_state = self.cnn(hidden_state)

        # Prepare for LSTM: (N, C, L') -> (N, L', C)
        hidden_state = hidden_state.transpose(1, 2).contiguous()

        # Update post-cnn NaN mask
        lengths = self.get_lengths_after_conv(src_nan_mask).view(N, 1, 1)

        # Apply LSTM
        h0_expanded = self.h0.expand(self.n_lstm_layers, N, -1).contiguous()
        c0_expanded = self.c0.expand(self.n_lstm_layers, N, -1).contiguous()
        lstm_out, _ = self.lstm(hidden_state, (h0_expanded, c0_expanded))

        # Gather the relevant hidden state based on sequence length
        hidden_state = torch.take_along_dim(lstm_out, lengths, dim=1).squeeze(1)

        # Decode
        return self.dense(hidden_state)
