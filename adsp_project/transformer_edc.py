# adsp_project/transformer_edc.py

import torch
import torch.nn as nn


class TransformerEDC(nn.Module):
    """
    Transformer-based model that maps a static room feature vector x
    to a downsampled EDC sequence y_hat of length seq_len.

    x: (batch, feature_dim)
    y_hat: (batch, seq_len)
    """

    def __init__(
        self,
        feature_dim: int,
        seq_len: int = 400,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.d_model = d_model

        # Project input features to transformer dimension
        self.cond_proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )

        # Learnable query tokens, one per time step
        self.query_tokens = nn.Parameter(
            torch.randn(seq_len, d_model) * 0.02
        )

        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(seq_len, d_model) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (batch, seq, dim)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Output projection to scalar per time step
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, feature_dim)
        returns y_hat: (batch, seq_len)
        """
        batch_size = x.size(0)

        # Conditioner token from features
        cond = self.cond_proj(x)           # (batch, d_model)
        cond_token = cond.unsqueeze(1)     # (batch, 1, d_model)

        # Query tokens for each time step
        q = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        q = q + self.pos_encoding.unsqueeze(0)  # add positional encodings

        # Concatenate conditioner + queries
        tokens = torch.cat([cond_token, q], dim=1)  # (batch, 1+seq_len, d_model)

        # Transformer encoder
        h = self.transformer(tokens)  # (batch, 1+seq_len, d_model)

        # Drop conditioner token, keep sequence
        h_seq = h[:, 1:, :]  # (batch, seq_len, d_model)

        # Project to 1 value per time step
        y_hat = self.out_proj(h_seq).squeeze(-1)  # (batch, seq_len)

        return y_hat
