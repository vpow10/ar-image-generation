from dataclasses import dataclass

import torch
from torch import nn

from ar_image_generation.models.transformer import TransformerBackbone


@dataclass(slots=True)
class GaussianPrimitiveARConfig:
    num_primitives: int = 256
    primitive_dim: int = 70
    dim: int = 512
    depth: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.10
    class_conditional: bool = True
    num_classes: int = 9
    log_std_min: float = -5.0
    log_std_max: float = 1.0
    mean_loss_weight: float = 0.05


@dataclass(slots=True)
class GaussianPrimitiveAROutput:
    mean: torch.Tensor
    log_std: torch.Tensor
    loss: torch.Tensor
    nll_loss: torch.Tensor
    mean_loss: torch.Tensor


class GaussianPrimitiveAR(nn.Module):
    def __init__(self, cfg: GaussianPrimitiveARConfig) -> None:
        super().__init__()

        self.cfg = cfg

        self.input_proj = nn.Linear(cfg.primitive_dim, cfg.dim)
        self.position_embedding = nn.Embedding(cfg.num_primitives, cfg.dim)
        self.bos_embedding = nn.Parameter(torch.zeros(1, 1, cfg.dim))

        if cfg.class_conditional:
            self.class_embedding = nn.Embedding(cfg.num_classes, cfg.dim)
        else:
            self.class_embedding = None

        self.transformer = TransformerBackbone(
            dim=cfg.dim,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
        )

        self.mean_head = nn.Linear(cfg.dim, cfg.primitive_dim)
        self.log_std_head = nn.Linear(cfg.dim, cfg.primitive_dim)

        dim_weights = self._build_dim_weights(cfg.primitive_dim)
        self.register_buffer("dim_weights", dim_weights, persistent=False)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.bos_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        if self.class_embedding is not None:
            nn.init.normal_(self.class_embedding.weight, mean=0.0, std=0.02)

        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, -1.5)

    def _build_dim_weights(self, primitive_dim: int) -> torch.Tensor:
        weights = torch.ones(primitive_dim, dtype=torch.float32)

        # position x,y
        weights[0:2] = 4.0

        # scale x,y
        weights[2:4] = 3.0

        # rotation
        weights[4:5] = 1.0

        # opacity
        weights[5:6] = 2.0

        # features keep weight 1.0

        return weights

    def _prepare_labels(
        self,
        labels: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.LongTensor | None:
        if not self.cfg.class_conditional:
            return None

        if labels is None:
            return torch.randint(
                low=0,
                high=self.cfg.num_classes,
                size=(batch_size,),
                device=device,
                dtype=torch.long,
            )

        labels = labels.to(device=device, dtype=torch.long)

        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels.squeeze(1)

        if labels.ndim != 1:
            raise ValueError(f"Expected labels [B], got shape {tuple(labels.shape)}")

        if labels.shape[0] != batch_size:
            raise ValueError(f"Expected {batch_size} labels, got {labels.shape[0]}")

        return labels

    def _causal_attention_mask(
        self,
        sequence_length: int,
        device: torch.device,
    ) -> torch.BoolTensor:
        return torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=device,
        ).tril()

    def _build_input_embeddings(
        self,
        *,
        prefix_primitives: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size = prefix_primitives.shape[0]
        prefix_length = prefix_primitives.shape[1]
        sequence_length = prefix_length + 1
        device = prefix_primitives.device

        if sequence_length > self.cfg.num_primitives:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"num_primitives={self.cfg.num_primitives}"
            )

        bos = self.bos_embedding.expand(batch_size, 1, -1)

        if prefix_length == 0:
            x = bos
        else:
            prefix_embeddings = self.input_proj(prefix_primitives)
            x = torch.cat([bos, prefix_embeddings], dim=1)

        position_ids = torch.arange(sequence_length, device=device, dtype=torch.long)
        x = x + self.position_embedding(position_ids)[None, :, :]

        if self.class_embedding is not None:
            prepared_labels = self._prepare_labels(
                labels,
                batch_size=batch_size,
                device=device,
            )
            assert prepared_labels is not None
            x = x + self.class_embedding(prepared_labels)[:, None, :]

        return x

    def forward(
        self,
        primitives: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> GaussianPrimitiveAROutput:
        if primitives.ndim != 3:
            raise ValueError(
                f"Expected primitives [B, N, D], got {tuple(primitives.shape)}"
            )

        batch_size, num_primitives, primitive_dim = primitives.shape

        if num_primitives != self.cfg.num_primitives:
            raise ValueError(
                f"Expected {self.cfg.num_primitives} primitives, got {num_primitives}"
            )

        if primitive_dim != self.cfg.primitive_dim:
            raise ValueError(
                f"Expected primitive dim {self.cfg.primitive_dim}, got {primitive_dim}"
            )

        prefix = primitives[:, :-1, :]
        targets = primitives

        x = self._build_input_embeddings(
            prefix_primitives=prefix,
            labels=labels,
        )

        attention_mask = self._causal_attention_mask(
            sequence_length=x.shape[1],
            device=x.device,
        )

        hidden = self.transformer(x, attention_mask=attention_mask)

        mean = self.mean_head(hidden)
        log_std = self.log_std_head(hidden).clamp(
            min=self.cfg.log_std_min,
            max=self.cfg.log_std_max,
        )

        std = torch.exp(log_std)

        dim_weights = self.dim_weights.to(
            device=primitives.device, dtype=primitives.dtype
        )

        squared_error = ((targets - mean) / std).square()
        nll = 0.5 * squared_error + log_std
        nll_loss = (nll * dim_weights[None, None, :]).sum() / (
            batch_size * num_primitives * dim_weights.sum()
        )

        mean_loss = ((targets - mean).square() * dim_weights[None, None, :]).sum() / (
            batch_size * num_primitives * dim_weights.sum()
        )

        loss = nll_loss + self.cfg.mean_loss_weight * mean_loss

        return GaussianPrimitiveAROutput(
            mean=mean,
            log_std=log_std,
            loss=loss,
            nll_loss=nll_loss,
            mean_loss=mean_loss,
        )

    @torch.no_grad()
    def generate(
        self,
        *,
        batch_size: int,
        labels: torch.Tensor | None,
        device: torch.device,
        temperature: float = 0.25,
        geometry_noise_scale: float = 0.7,
        feature_noise_scale: float = 0.05,
    ) -> torch.Tensor:
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        self.eval()

        prepared_labels = self._prepare_labels(
            labels,
            batch_size=batch_size,
            device=device,
        )

        generated = torch.empty(
            batch_size,
            0,
            self.cfg.primitive_dim,
            device=device,
        )

        for _ in range(self.cfg.num_primitives):
            x = self._build_input_embeddings(
                prefix_primitives=generated,
                labels=prepared_labels,
            )

            attention_mask = self._causal_attention_mask(
                sequence_length=x.shape[1],
                device=device,
            )

            hidden = self.transformer(x, attention_mask=attention_mask)
            last_hidden = hidden[:, -1:, :]

            mean = self.mean_head(last_hidden)
            log_std = self.log_std_head(last_hidden).clamp(
                min=self.cfg.log_std_min,
                max=self.cfg.log_std_max,
            )

            std = torch.exp(log_std)

            noise = torch.randn_like(mean)
            noise[..., 0:6] = noise[..., 0:6] * geometry_noise_scale
            noise[..., 6:] = noise[..., 6:] * feature_noise_scale

            next_primitive = mean + temperature * std * noise
            next_primitive = next_primitive.clamp(-3.0, 3.0)

            generated = torch.cat([generated, next_primitive], dim=1)

        return generated
