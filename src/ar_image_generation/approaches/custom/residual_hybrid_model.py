from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ar_image_generation.models.transformer import TransformerBackbone


@dataclass(slots=True)
class GaussianPrimitiveResidualHybridARConfig:
    num_primitives: int = 256
    geometry_dim: int = 6
    num_quantizers: int = 2
    num_feature_codes: int = 1024
    dim: int = 512
    depth: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.10
    class_conditional: bool = True
    num_classes: int = 9
    log_std_min: float = -5.0
    log_std_max: float = 0.0
    mean_loss_weight: float = 0.05
    code_loss_weight: float = 1.0


@dataclass(slots=True)
class GaussianPrimitiveResidualHybridAROutput:
    geometry_mean: torch.Tensor
    geometry_log_std: torch.Tensor
    feature_logits: list[torch.Tensor]
    loss: torch.Tensor
    geometry_nll_loss: torch.Tensor
    geometry_mean_loss: torch.Tensor
    feature_code_loss: torch.Tensor


class GaussianPrimitiveResidualHybridAR(nn.Module):
    def __init__(self, cfg: GaussianPrimitiveResidualHybridARConfig) -> None:
        super().__init__()

        self.cfg = cfg

        self.geometry_proj = nn.Linear(cfg.geometry_dim, cfg.dim)

        self.feature_code_embeddings = nn.ModuleList(
            [
                nn.Embedding(cfg.num_feature_codes, cfg.dim)
                for _ in range(cfg.num_quantizers)
            ]
        )

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

        self.geometry_mean_head = nn.Linear(cfg.dim, cfg.geometry_dim)
        self.geometry_log_std_head = nn.Linear(cfg.dim, cfg.geometry_dim)

        self.feature_logits_heads = nn.ModuleList(
            [
                nn.Linear(cfg.dim, cfg.num_feature_codes)
                for _ in range(cfg.num_quantizers)
            ]
        )

        self.code_condition_projs = nn.ModuleList(
            [nn.Linear(cfg.dim, cfg.dim) for _ in range(max(cfg.num_quantizers - 1, 0))]
        )

        geometry_weights = torch.tensor(
            [4.0, 4.0, 3.0, 3.0, 1.0, 2.0], dtype=torch.float32
        )
        self.register_buffer("geometry_weights", geometry_weights, persistent=False)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.bos_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        for embedding in self.feature_code_embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)

        if self.class_embedding is not None:
            nn.init.normal_(self.class_embedding.weight, mean=0.0, std=0.02)

        nn.init.zeros_(self.geometry_log_std_head.weight)
        nn.init.constant_(self.geometry_log_std_head.bias, -1.5)

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
        self, sequence_length: int, device: torch.device
    ) -> torch.BoolTensor:
        return torch.ones(
            sequence_length, sequence_length, dtype=torch.bool, device=device
        ).tril()

    def _embed_feature_codes(self, codes: torch.LongTensor) -> torch.Tensor:
        if codes.shape[-1] != self.cfg.num_quantizers:
            raise ValueError(
                f"Expected {self.cfg.num_quantizers} code streams, got {codes.shape[-1]}"
            )

        embeddings = [
            embedding(codes[..., index])
            for index, embedding in enumerate(self.feature_code_embeddings)
        ]

        return torch.stack(embeddings, dim=0).sum(dim=0)

    def _build_input_embeddings(
        self,
        *,
        prefix_geometry: torch.Tensor,
        prefix_codes: torch.LongTensor,
        labels: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size = prefix_geometry.shape[0]
        prefix_length = prefix_geometry.shape[1]
        sequence_length = prefix_length + 1
        device = prefix_geometry.device

        if sequence_length > self.cfg.num_primitives:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds num_primitives={self.cfg.num_primitives}"
            )

        bos = self.bos_embedding.expand(batch_size, 1, -1)

        if prefix_length == 0:
            x = bos
        else:
            prefix_embeddings = self.geometry_proj(
                prefix_geometry
            ) + self._embed_feature_codes(prefix_codes)
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

    def _feature_logits(
        self,
        hidden: torch.Tensor,
        feature_codes: torch.LongTensor | None,
    ) -> list[torch.Tensor]:
        logits: list[torch.Tensor] = []
        conditioned_hidden = hidden

        for index, head in enumerate(self.feature_logits_heads):
            logits.append(head(conditioned_hidden))

            if index < self.cfg.num_quantizers - 1:
                if feature_codes is None:
                    raise ValueError(
                        "feature_codes must be provided for conditioned logits."
                    )

                code_embedding = self.feature_code_embeddings[index](
                    feature_codes[..., index]
                )
                conditioned_hidden = hidden + self.code_condition_projs[index](
                    code_embedding
                )

        return logits

    def forward(
        self,
        geometry: torch.Tensor,
        feature_codes: torch.LongTensor,
        labels: torch.Tensor | None = None,
    ) -> GaussianPrimitiveResidualHybridAROutput:
        batch_size, num_primitives, geometry_dim = geometry.shape

        if num_primitives != self.cfg.num_primitives:
            raise ValueError(
                f"Expected {self.cfg.num_primitives} primitives, got {num_primitives}"
            )

        if geometry_dim != self.cfg.geometry_dim:
            raise ValueError(
                f"Expected geometry dim {self.cfg.geometry_dim}, got {geometry_dim}"
            )

        if feature_codes.shape != (batch_size, num_primitives, self.cfg.num_quantizers):
            raise ValueError(
                f"Expected feature codes {(batch_size, num_primitives, self.cfg.num_quantizers)}, "
                f"got {tuple(feature_codes.shape)}"
            )

        prefix_geometry = geometry[:, :-1, :]
        prefix_codes = feature_codes[:, :-1, :]

        x = self._build_input_embeddings(
            prefix_geometry=prefix_geometry,
            prefix_codes=prefix_codes,
            labels=labels,
        )

        attention_mask = self._causal_attention_mask(x.shape[1], x.device)
        hidden = self.transformer(x, attention_mask=attention_mask)

        geometry_mean = self.geometry_mean_head(hidden)
        geometry_log_std = self.geometry_log_std_head(hidden).clamp(
            min=self.cfg.log_std_min,
            max=self.cfg.log_std_max,
        )

        feature_logits = self._feature_logits(hidden, feature_codes)

        std = torch.exp(geometry_log_std)
        weights = self.geometry_weights.to(device=geometry.device, dtype=geometry.dtype)

        squared_error = ((geometry - geometry_mean) / std).square()
        geometry_nll = 0.5 * squared_error + geometry_log_std
        geometry_nll_loss = (geometry_nll * weights[None, None, :]).sum() / (
            batch_size * num_primitives * weights.sum()
        )

        geometry_mean_loss = (
            (geometry - geometry_mean).square() * weights[None, None, :]
        ).sum() / (batch_size * num_primitives * weights.sum())

        code_losses = []
        for index, logits in enumerate(feature_logits):
            code_loss = F.cross_entropy(
                logits.reshape(-1, self.cfg.num_feature_codes),
                feature_codes[..., index].reshape(-1),
            )
            code_losses.append(code_loss)

        feature_code_loss = torch.stack(code_losses).mean()

        loss = (
            geometry_nll_loss
            + self.cfg.mean_loss_weight * geometry_mean_loss
            + self.cfg.code_loss_weight * feature_code_loss
        )

        return GaussianPrimitiveResidualHybridAROutput(
            geometry_mean=geometry_mean,
            geometry_log_std=geometry_log_std,
            feature_logits=feature_logits,
            loss=loss,
            geometry_nll_loss=geometry_nll_loss,
            geometry_mean_loss=geometry_mean_loss,
            feature_code_loss=feature_code_loss,
        )

    def _sample_feature_code(
        self,
        logits: torch.Tensor,
        *,
        temperature: float,
        top_k: int | None,
    ) -> torch.LongTensor:
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        logits = logits / temperature

        if top_k is not None and top_k > 0 and top_k < logits.shape[-1]:
            values, indices = torch.topk(logits, k=top_k, dim=-1)
            filtered_logits = torch.full_like(logits, float("-inf"))
            filtered_logits.scatter_(dim=-1, index=indices, src=values)
            logits = filtered_logits

        probabilities = torch.softmax(logits, dim=-1)
        sampled = torch.multinomial(probabilities, num_samples=1)

        return sampled.squeeze(-1)

    @torch.no_grad()
    def generate(
        self,
        *,
        batch_size: int,
        labels: torch.Tensor | None,
        device: torch.device,
        geometry_temperature: float = 0.30,
        code_temperature: float = 1.0,
        top_k: int | None = 128,
    ) -> tuple[torch.Tensor, torch.LongTensor]:
        geometry = torch.empty(
            batch_size,
            0,
            self.cfg.geometry_dim,
            device=device,
        )
        feature_codes = torch.empty(
            batch_size,
            0,
            self.cfg.num_quantizers,
            dtype=torch.long,
            device=device,
        )

        prepared_labels = self._prepare_labels(
            labels,
            batch_size=batch_size,
            device=device,
        )

        for _ in range(self.cfg.num_primitives):
            x = self._build_input_embeddings(
                prefix_geometry=geometry,
                prefix_codes=feature_codes,
                labels=prepared_labels,
            )

            attention_mask = self._causal_attention_mask(x.shape[1], x.device)
            hidden = self.transformer(x, attention_mask=attention_mask)
            last_hidden = hidden[:, -1:, :]

            geometry_mean = self.geometry_mean_head(last_hidden)
            geometry_log_std = self.geometry_log_std_head(last_hidden).clamp(
                min=self.cfg.log_std_min,
                max=self.cfg.log_std_max,
            )

            geometry_std = torch.exp(geometry_log_std)
            next_geometry = (
                geometry_mean
                + geometry_temperature * geometry_std * torch.randn_like(geometry_mean)
            )
            next_geometry = next_geometry.clamp(-3.0, 3.0)

            sampled_codes: list[torch.Tensor] = []
            conditioned_hidden = last_hidden

            for index, head in enumerate(self.feature_logits_heads):
                logits = head(conditioned_hidden)[:, 0, :]

                next_code = self._sample_feature_code(
                    logits,
                    temperature=code_temperature,
                    top_k=top_k,
                )
                sampled_codes.append(next_code)

                if index < self.cfg.num_quantizers - 1:
                    code_embedding = self.feature_code_embeddings[index](next_code)[
                        :, None, :
                    ]
                    conditioned_hidden = last_hidden + self.code_condition_projs[index](
                        code_embedding
                    )

            next_codes = torch.stack(sampled_codes, dim=-1)

            geometry = torch.cat([geometry, next_geometry], dim=1)
            feature_codes = torch.cat([feature_codes, next_codes[:, None, :]], dim=1)

        return geometry, feature_codes
