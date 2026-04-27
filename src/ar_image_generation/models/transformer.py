import math

import torch
import torch.nn.functional as F
from torch import nn


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D]
            attention_mask: optional [N, N] boolean mask.
                True means query position may attend to key position.
                False means attention is blocked.

        Returns:
            [B, N, D]
        """

        batch_size, sequence_length, dim = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)

        queries, keys, values = qkv.unbind(dim=0)

        scores = queries @ keys.transpose(-2, -1)
        scores = scores * self.scale

        if attention_mask is not None:
            if attention_mask.shape != (sequence_length, sequence_length):
                raise ValueError(
                    "attention_mask must have shape "
                    f"({sequence_length}, {sequence_length}), got {tuple(attention_mask.shape)}"
                )

            mask = attention_mask.to(device=x.device)
            scores = scores.masked_fill(~mask[None, None, :, :], float("-inf"))

        attention = F.softmax(scores, dim=-1)
        attention = self.attn_dropout(attention)

        out = attention @ values
        out = out.transpose(1, 2).contiguous()
        out = out.reshape(batch_size, sequence_length, dim)

        return self.proj_dropout(self.proj(out))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()

        hidden_dim = int(dim * mlp_ratio)

        self.attn_norm = nn.LayerNorm(dim)
        self.attn = SelfAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(
            dim=dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), attention_mask=attention_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TransformerBackbone(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)

        return self.norm(x)
