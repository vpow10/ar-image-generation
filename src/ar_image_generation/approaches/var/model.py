from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ar_image_generation.approaches.base import AutoregressiveApproach, SamplingConfig
from ar_image_generation.approaches.registry import register_approach
from ar_image_generation.approaches.var.multiscale import (
    build_multiscale_token_grids,
    flatten_multiscale_token_grids,
)
from ar_image_generation.approaches.var.sampler import sample_tokens_from_logits
from ar_image_generation.approaches.var.schedule import VARSchedule
from ar_image_generation.data.batch import ImageBatch
from ar_image_generation.models.transformer import TransformerBackbone
from ar_image_generation.tokenizers.base import ImageTokenizer


@dataclass(slots=True)
class VAROutput:
    logits: torch.Tensor
    targets: torch.LongTensor
    loss: torch.Tensor


@register_approach("var")
class VARApproach(AutoregressiveApproach):
    name = "var"

    def __init__(
        self,
        *,
        vocab_size: int,
        latent_shape: tuple[int, int],
        dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        scales: tuple[int, ...] | list[int] = (1, 2, 4, 8),
        class_conditional: bool = False,
    ) -> None:
        super().__init__()

        if class_conditional:
            raise NotImplementedError("This project currently uses unconditional VAR generation.")

        if latent_shape[0] != latent_shape[1]:
            raise ValueError(f"VAR expects a square latent shape, got {latent_shape}")

        self.vocab_size = vocab_size
        self.latent_shape = latent_shape
        self.schedule = VARSchedule(scales=tuple(scales))

        if self.schedule.final_scale != latent_shape[0]:
            raise ValueError(
                f"Final VAR scale {self.schedule.final_scale} must match latent shape {latent_shape}"
            )

        self.dim = dim
        self.sequence_length = 1 + self.schedule.num_tokens

        self.token_embedding = nn.Embedding(vocab_size, dim)

        self.query_embedding = nn.Embedding(self.schedule.num_scales, dim)

        # +1 because index 0 is reserved for BOS; scale k uses k + 1.
        self.scale_embedding = nn.Embedding(self.schedule.num_scales + 1, dim)
        self.position_embedding = nn.Embedding(self.sequence_length, dim)

        self.bos_embedding = nn.Parameter(torch.zeros(1, 1, dim))

        self.transformer = TransformerBackbone(
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.lm_head = nn.Linear(dim, vocab_size)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.bos_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.scale_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.query_embedding.weight, mean=0.0, std=0.02)

    def _scale_position_ids(self, scale_index: int, device: torch.device) -> torch.LongTensor:
        scale_slice = self.schedule.scale_slices(include_bos=True)[scale_index]
        return torch.arange(scale_slice.start, scale_slice.stop, device=device, dtype=torch.long)

    def _previous_scale_position_ids(
        self,
        scale_index: int,
        device: torch.device,
    ) -> torch.LongTensor | None:
        if scale_index == 0:
            return None

        position_ids = [
            self._scale_position_ids(previous_scale_index, device)
            for previous_scale_index in range(scale_index)
        ]
        return torch.cat(position_ids, dim=0)

    def _previous_scale_ids(
        self,
        scale_index: int,
        device: torch.device,
    ) -> torch.LongTensor | None:
        if scale_index == 0:
            return None

        ids = []
        for previous_scale_index, scale in enumerate(self.schedule.scales[:scale_index]):
            ids.append(
                torch.full(
                    (scale * scale,),
                    previous_scale_index + 1,
                    device=device,
                    dtype=torch.long,
                )
            )

        return torch.cat(ids, dim=0)

    def _build_bos_embeddings(self, batch_size: int, device: torch.device) -> torch.Tensor:
        bos_position_ids = torch.zeros(1, device=device, dtype=torch.long)
        bos_scale_ids = torch.zeros(1, device=device, dtype=torch.long)

        bos = self.bos_embedding.expand(batch_size, -1, -1)
        bos = bos + self.position_embedding(bos_position_ids)[None, :, :]
        bos = bos + self.scale_embedding(bos_scale_ids)[None, :, :]

        return bos

    def _build_context_embeddings(
        self,
        *,
        context_tokens: torch.LongTensor | None,
        scale_index: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if context_tokens is None:
            return None

        position_ids = self._previous_scale_position_ids(scale_index, device)
        scale_ids = self._previous_scale_ids(scale_index, device)

        if position_ids is None or scale_ids is None:
            return None

        if context_tokens.shape != (batch_size, position_ids.numel()):
            raise ValueError(
                "Invalid context token shape. "
                f"Expected {(batch_size, position_ids.numel())}, got {tuple(context_tokens.shape)}"
            )

        context = self.token_embedding(context_tokens)
        context = context + self.position_embedding(position_ids)[None, :, :]
        context = context + self.scale_embedding(scale_ids)[None, :, :]

        return context

    def _build_query_embeddings(
        self,
        *,
        scale_index: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        scale = self.schedule.scales[scale_index]
        target_length = scale * scale

        position_ids = self._scale_position_ids(scale_index, device)
        query_ids = torch.full(
            (target_length,),
            scale_index,
            device=device,
            dtype=torch.long,
        )
        scale_ids = torch.full(
            (target_length,),
            scale_index + 1,
            device=device,
            dtype=torch.long,
        )

        query = self.query_embedding(query_ids)
        query = query + self.position_embedding(position_ids)
        query = query + self.scale_embedding(scale_ids)

        return query[None, :, :].expand(batch_size, -1, -1)

    def predict_scale_logits(
        self,
        *,
        context_tokens: torch.LongTensor | None,
        scale_index: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Predict all tokens of one target scale in parallel.

        Context consists of BOS plus all lower-resolution token grids.
        Target positions are represented by learned query embeddings.
        """

        bos = self._build_bos_embeddings(batch_size, device)
        context = self._build_context_embeddings(
            context_tokens=context_tokens,
            scale_index=scale_index,
            batch_size=batch_size,
            device=device,
        )
        query = self._build_query_embeddings(
            scale_index=scale_index,
            batch_size=batch_size,
            device=device,
        )

        if context is None:
            sequence = torch.cat([bos, query], dim=1)
        else:
            sequence = torch.cat([bos, context, query], dim=1)

        hidden = self.transformer(sequence)
        query_hidden = hidden[:, -query.shape[1] :, :]

        return self.lm_head(query_hidden)

    def forward(self, tokens: torch.LongTensor) -> VAROutput:
        """
        Args:
            tokens: full-resolution tokenizer tokens [B, H, W]

        Returns:
            logits over multiscale targets [B, 85, vocab_size] for scales [1, 2, 4, 8]
        """

        batch_size = tokens.shape[0]
        device = tokens.device

        multiscale_grids = build_multiscale_token_grids(tokens, self.schedule)
        targets = flatten_multiscale_token_grids(multiscale_grids)

        logits_by_scale: list[torch.Tensor] = []
        generated_contexts: list[torch.LongTensor] = []

        for scale_index, target_grid in enumerate(multiscale_grids):
            if generated_contexts:
                context_tokens = torch.cat(generated_contexts, dim=1)
            else:
                context_tokens = None

            logits = self.predict_scale_logits(
                context_tokens=context_tokens,
                scale_index=scale_index,
                batch_size=batch_size,
                device=device,
            )
            logits_by_scale.append(logits)

            # During training we condition higher scales on ground-truth lower scales.
            generated_contexts.append(target_grid.reshape(batch_size, -1))

        logits = torch.cat(logits_by_scale, dim=1)

        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
        )

        return VAROutput(
            logits=logits,
            targets=targets,
            loss=loss,
        )

    def training_step(
        self,
        batch: ImageBatch,
        tokenizer: ImageTokenizer,
    ) -> dict[str, torch.Tensor]:
        tokenizer.eval()

        with torch.no_grad():
            tokens = tokenizer.encode(batch.images)

        output = self(tokens)

        return {
            "loss": output.loss,
        }

    @torch.no_grad()
    def generate(
        self,
        tokenizer: ImageTokenizer,
        batch_size: int,
        labels: torch.Tensor | None,
        device: torch.device,
        sampling_cfg: SamplingConfig,
    ) -> torch.Tensor:
        del labels

        self.eval()
        tokenizer.eval()

        generated_grids: list[torch.LongTensor] = []

        for scale_index, scale in enumerate(self.schedule.scales):
            if generated_grids:
                context_tokens = torch.cat(
                    [grid.reshape(batch_size, -1) for grid in generated_grids],
                    dim=1,
                )
            else:
                context_tokens = None

            logits = self.predict_scale_logits(
                context_tokens=context_tokens,
                scale_index=scale_index,
                batch_size=batch_size,
                device=device,
            )

            sampled_tokens = sample_tokens_from_logits(
                logits,
                temperature=sampling_cfg.temperature,
                top_k=sampling_cfg.top_k,
                top_p=sampling_cfg.top_p,
            )

            generated_grids.append(sampled_tokens.reshape(batch_size, scale, scale))

        final_tokens = generated_grids[-1]
        return tokenizer.decode(final_tokens)
