from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ar_image_generation.approaches.base import AutoregressiveApproach, SamplingConfig
from ar_image_generation.approaches.registry import register_approach
from ar_image_generation.approaches.var.sampler import sample_tokens_from_logits
from ar_image_generation.data.batch import ImageBatch
from ar_image_generation.models.transformer import TransformerBackbone
from ar_image_generation.tokenizers.base import ImageTokenizer


@dataclass(slots=True)
class RasterOutput:
    logits: torch.Tensor
    targets: torch.LongTensor
    loss: torch.Tensor


@register_approach("raster")
class RasterApproach(AutoregressiveApproach):
    name = "raster"

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
        class_conditional: bool = False,
    ) -> None:
        super().__init__()

        if class_conditional:
            raise NotImplementedError("This project currently uses unconditional generation.")

        self.vocab_size = vocab_size
        self.latent_shape = latent_shape
        self.num_tokens = latent_shape[0] * latent_shape[1]

        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(self.num_tokens, dim)
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

    def _causal_mask(self, length: int, device: torch.device) -> torch.BoolTensor:
        return torch.ones(length, length, device=device, dtype=torch.bool).tril()

    def _embed(
        self,
        tokens: torch.LongTensor,  # T can be 0 during generation
        device: torch.device,
    ) -> torch.Tensor:
        batch_size = tokens.shape[0]

        bos = self.bos_embedding.expand(batch_size, -1, -1)

        if tokens.shape[1] > 0:
            seq = torch.cat([bos, self.token_embedding(tokens)], dim=1)
        else:
            seq = bos

        position_ids = torch.arange(seq.shape[1], device=device, dtype=torch.long)
        return seq + self.position_embedding(position_ids)[None]

    def forward(self, tokens: torch.LongTensor) -> RasterOutput:
        """
        Args:
            tokens: [B, H, W]

        Returns:
            logits [B, N, vocab_size] and scalar loss over N = H*W raster positions
        """
        batch_size = tokens.shape[0]
        device = tokens.device

        flat = tokens.reshape(batch_size, -1)

        # Input:  [BOS, t0, ..., t_{N-2}]  length N
        # Target: [t0,  t1, ..., t_{N-1}]  length N
        seq = self._embed(flat[:, :-1], device)
        hidden = self.transformer(seq, attention_mask=self._causal_mask(seq.shape[1], device))
        logits = self.lm_head(hidden)

        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            flat.reshape(-1),
        )

        return RasterOutput(logits=logits, targets=flat, loss=loss)

    def training_step(
        self,
        batch: ImageBatch,
        tokenizer: ImageTokenizer,
    ) -> dict[str, torch.Tensor]:
        tokenizer.eval()

        with torch.no_grad():
            tokens = tokenizer.encode(batch.images)

        output = self(tokens)
        return {"loss": output.loss}

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

        generated = torch.zeros(batch_size, 0, dtype=torch.long, device=device)

        for _ in range(self.num_tokens):
            seq = self._embed(generated, device)
            hidden = self.transformer(seq, attention_mask=self._causal_mask(seq.shape[1], device))
            logits = self.lm_head(hidden[:, -1:, :])

            next_token = sample_tokens_from_logits(
                logits,
                temperature=sampling_cfg.temperature,
                top_k=sampling_cfg.top_k,
                top_p=sampling_cfg.top_p,
            )

            generated = torch.cat([generated, next_token], dim=1)

        final_tokens = generated.reshape(batch_size, *self.latent_shape)
        return tokenizer.decode(final_tokens)
