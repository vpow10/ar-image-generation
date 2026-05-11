from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ar_image_generation.approaches.base import AutoregressiveApproach, SamplingConfig
from ar_image_generation.approaches.registry import register_approach
from ar_image_generation.approaches.var.multiscale import (
    build_multiscale_sequence_from_final_tokens,
    build_multiscale_sequence_from_images,
    build_scale_ids,
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
        scales: tuple[int, ...] | list[int] = (1, 2, 4, 8, 16),
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
        self.num_tokens = self.schedule.num_tokens

        self.token_embedding = nn.Embedding(vocab_size, dim)

        self.scale_embedding = nn.Embedding(self.schedule.num_scales + 1, dim)
        self.position_embedding = nn.Embedding(self.num_tokens, dim)

        self.bos_embedding = nn.Parameter(torch.zeros(1, 1, dim))

        target_scale_ids = build_scale_ids(
            self.schedule,
            include_bos=False,
            device=None,
        )
        self.register_buffer(
            "target_scale_ids",
            target_scale_ids,
            persistent=False,
        )

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
        prefix_tokens: torch.LongTensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Shifted autoregressive input.

        targets:
            [x0, x1, ..., xN]

        input:
            [BOS, x0, ..., xN-1]
        """

        prefix_length = prefix_tokens.shape[1]
        sequence_length = prefix_length + 1

        if sequence_length > self.num_tokens:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds max VAR length {self.num_tokens}"
            )

        bos = self.bos_embedding.expand(batch_size, 1, -1)

        if prefix_length == 0:
            x = bos
        else:
            previous_token_embeddings = self.token_embedding(prefix_tokens)
            x = torch.cat([bos, previous_token_embeddings], dim=1)

        position_ids = torch.arange(
            sequence_length,
            device=device,
            dtype=torch.long,
        )

        scale_ids = self.target_scale_ids[:sequence_length].to(device=device) + 1

        x = x + self.position_embedding(position_ids)[None, :, :]
        x = x + self.scale_embedding(scale_ids)[None, :, :]

        return x

    def forward_sequence(self, targets: torch.LongTensor) -> VAROutput:
        """
        Args:
            targets: multiscale target sequence [B, num_tokens]
        """

        batch_size = targets.shape[0]
        device = targets.device

        if targets.shape[1] != self.num_tokens:
            raise ValueError(
                f"Expected target sequence length {self.num_tokens}, got {targets.shape[1]}"
            )

        prefix_tokens = targets[:, :-1]

        x = self._build_input_embeddings(
            prefix_tokens=prefix_tokens,
            batch_size=batch_size,
            device=device,
        )

        attention_mask = self._causal_attention_mask(
            sequence_length=x.shape[1],
            device=device,
        )

        hidden = self.transformer(x, attention_mask=attention_mask)
        logits = self.lm_head(hidden)

        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
        )

        return VAROutput(
            logits=logits,
            targets=targets,
            loss=loss,
        )

    def forward(self, tokens: torch.LongTensor) -> VAROutput:
        """
        Test/debug forward from final-resolution tokens only.

        Real training uses `training_step`, which builds image-pyramid targets.
        """

        targets = build_multiscale_sequence_from_final_tokens(tokens, self.schedule)

        if targets.shape[1] != self.num_tokens:
            raise ValueError(
                "forward(tokens) is only a debug path. "
                "Use training_step(...) for real image-pyramid VAR training."
            )

        return self.forward_sequence(targets)

    def training_step(
        self,
        batch: ImageBatch,
        tokenizer: ImageTokenizer,
    ) -> dict[str, torch.Tensor]:
        tokenizer.eval()

        with torch.no_grad():
            targets = build_multiscale_sequence_from_images(
                batch.images,
                tokenizer,
                self.schedule,
            )

        output = self.forward_sequence(targets)

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

        generated_tokens = torch.empty(
            batch_size,
            0,
            dtype=torch.long,
            device=device,
        )

        for _ in range(self.num_tokens):
            x = self._build_input_embeddings(
                prefix_tokens=generated_tokens,
                batch_size=batch_size,
                device=device,
            )

            attention_mask = self._causal_attention_mask(
                sequence_length=x.shape[1],
                device=device,
            )

            hidden = self.transformer(x, attention_mask=attention_mask)
            logits = self.lm_head(hidden[:, -1:, :])

            next_token = sample_tokens_from_logits(
                logits,
                temperature=sampling_cfg.temperature,
                top_k=sampling_cfg.top_k,
                top_p=sampling_cfg.top_p,
            )

            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

        final_scale_slice = self.schedule.scale_slices(include_bos=False)[-1]
        final_scale = self.schedule.final_scale

        final_tokens = generated_tokens[:, final_scale_slice]
        final_tokens = final_tokens.reshape(batch_size, final_scale, final_scale)

        return tokenizer.decode(final_tokens)
