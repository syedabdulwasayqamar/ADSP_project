# models/transformer_model.py

import torch
import torch.nn as nn
import pytorch_lightning as pl
from models.positional_encoding import PositionalEncoding

class EDCTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model,
        nhead,
        num_layers,
        output_dim,
        dim_feedforward=512,
        dropout=0.1,
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # 🔑 THIS IS THE FIX
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, x):
        """
        x: [batch, seq_len, input_dim]
        """

        # HARD ASSERT — fail early
        assert x.dim() == 3, f"x must be 3D, got {x.shape}"

        # Project
        x = self.input_projection(x)   # [B, T, d_model]

        # Transformer (NO PERMUTE!)
        x = self.transformer_encoder(x)  # [B, T, d_model]

        # Take first time step
        x = x[:, 0, :]  # [B, d_model]

        # Output
        out = self.output_layer(x)  # [B, output_dim]

        return out
