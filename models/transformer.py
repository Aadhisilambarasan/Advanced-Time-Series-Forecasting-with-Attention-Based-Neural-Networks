import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TemporalTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=32,
        n_heads=2,
        num_layers=1,
        dim_feedforward=128,
        dropout=0.3,
        forecast_horizon=1,
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True
            )
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(d_model, forecast_horizon)

    def forward(self, x, return_attention=False):

        device = x.device
        x = self.input_projection(x)
        x = self.positional_encoding(x)

        attentions = []

        for layer in self.layers:

            if return_attention:
                # Self-attention block
                attn_output, attn_weights = layer.self_attn(
                    x,
                    x,
                    x,
                    need_weights=True,
                    average_attn_weights=False  # IMPORTANT
                )

                # attn_weights shape:
                # (batch, heads, seq_len, seq_len)
                
                x = layer.norm1(x + layer.dropout1(attn_output))

                # Feed-forward block
                x2 = layer.linear2(
                    layer.dropout(
                        layer.activation(layer.linear1(x))
                    )
                )
                x = layer.norm2(x + layer.dropout2(x2))

                attentions.append(attn_weights)

            else:
                x = layer(x)

        # Global average pooling
        x = torch.mean(x, dim=1)
        x = self.dropout(x)
        output = self.output_layer(x)

        if return_attention:
            return output, attentions

        return output
