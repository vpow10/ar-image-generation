import math

import torch
import torch.nn.functional as F
from torch import nn

from ar_image_generation.approaches.base import AutoregressiveApproach, SamplingConfig
from ar_image_generation.approaches.registry import register_approach
from ar_image_generation.approaches.var.sampler import sample_tokens_from_logits
from ar_image_generation.data.batch import ImageBatch
from ar_image_generation.models.transformer import TransformerBackbone
from ar_image_generation.tokenizers.base import ImageTokenizer


def mask_ratio(t: torch.Tensor | float, kind: str) -> torch.Tensor | float:
    if kind == "cosine":
        if isinstance(t, torch.Tensor):
            return torch.cos(t * (math.pi / 2))
        return math.cos(t * (math.pi / 2))
    if kind == "linear":
        return 1.0 - t
    if kind == "square":
        return 1.0 - t * t
    raise ValueError(f"Unknown mask schedule: {kind}")


@register_approach("maskgit")
class MaskGITApproach(AutoregressiveApproach):
    name = "maskgit"

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
        num_classes: int = 9,
        num_iterations: int = 8,
        mask_schedule: str = "cosine",
        noise_scale: float = 4.5,
        cfg_dropout: float = 0.1,
        cfg_scale: float = 3.0,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.latent_shape = latent_shape
        self.num_tokens = latent_shape[0] * latent_shape[1]
        self.mask_token_id = vocab_size
        self.class_conditional = class_conditional
        self.num_classes = num_classes
        self.null_class_id = num_classes
        self.num_iterations = num_iterations
        self.mask_schedule = mask_schedule
        self.noise_scale = noise_scale
        self.cfg_dropout = cfg_dropout
        self.cfg_scale = cfg_scale

        self.token_embedding = nn.Embedding(vocab_size + 1, dim)
        self.position_embedding = nn.Embedding(self.num_tokens, dim)

        if class_conditional:
            self.class_embedding = nn.Embedding(num_classes + 1, dim)

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
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        if self.class_conditional:
            nn.init.normal_(self.class_embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.LongTensor,
        labels: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        position_ids = torch.arange(self.num_tokens, device=tokens.device, dtype=torch.long)
        seq = self.token_embedding(tokens) + self.position_embedding(position_ids)[None]
        if self.class_conditional:
            if labels is None:
                raise ValueError("class_conditional model requires labels in forward()")
            seq = seq + self.class_embedding(labels)[:, None, :]
        hidden = self.transformer(seq)
        return self.lm_head(hidden)

    def training_step(
        self,
        batch: ImageBatch,
        tokenizer: ImageTokenizer,
    ) -> dict[str, torch.Tensor]:
        tokenizer.eval()

        with torch.no_grad():
            tokens = tokenizer.encode(batch.images)

        batch_size = tokens.shape[0]
        device = tokens.device
        flat = tokens.reshape(batch_size, -1)

        r = torch.rand(batch_size, 1, device=device)
        ratio = mask_ratio(r, self.mask_schedule)

        noise = torch.rand(batch_size, self.num_tokens, device=device)
        mask = noise < ratio

        forced = noise.argmin(dim=1)
        mask[torch.arange(batch_size, device=device), forced] = True

        masked_input = torch.where(mask, torch.full_like(flat, self.mask_token_id), flat)

        labels = batch.labels if self.class_conditional else None
        if self.class_conditional and labels is None:
            raise ValueError("class_conditional training requires batch.labels")

        if self.class_conditional and self.cfg_dropout > 0.0:
            drop = torch.rand(batch_size, device=device) < self.cfg_dropout
            labels = torch.where(drop, torch.full_like(labels, self.null_class_id), labels)

        logits = self(masked_input, labels)

        loss = F.cross_entropy(
            logits[mask].reshape(-1, self.vocab_size),
            flat[mask].reshape(-1),
        )

        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        tokenizer: ImageTokenizer,
        batch_size: int,
        labels: torch.Tensor | None,
        device: torch.device,
        sampling_cfg: SamplingConfig,
    ) -> torch.Tensor:
        self.eval()
        tokenizer.eval()

        if self.class_conditional:
            if labels is None:
                labels = torch.randint(0, self.num_classes, (batch_size,), device=device)
        else:
            labels = None

        state = torch.full(
            (batch_size, self.num_tokens),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        unmasked = torch.zeros(batch_size, self.num_tokens, dtype=torch.bool, device=device)

        use_cfg = self.class_conditional and self.cfg_scale != 1.0
        null_labels = (
            torch.full((batch_size,), self.null_class_id, dtype=torch.long, device=device)
            if use_cfg
            else None
        )

        for step in range(self.num_iterations):
            if use_cfg:
                logits_cond = self(state, labels)
                logits_uncond = self(state, null_labels)
                logits = logits_uncond + self.cfg_scale * (logits_cond - logits_uncond)
            else:
                logits = self(state, labels)

            sampled = sample_tokens_from_logits(
                logits,
                temperature=sampling_cfg.temperature,
                top_k=sampling_cfg.top_k,
                top_p=sampling_cfg.top_p,
            )

            probs = F.softmax(logits / sampling_cfg.temperature, dim=-1)
            confidence = torch.log(probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1) + 1e-10)

            t_next = (step + 1) / self.num_iterations
            noise_temp = self.noise_scale * (1.0 - t_next)
            if noise_temp > 0:
                gumbel = -torch.log(-torch.log(torch.rand_like(confidence) + 1e-10) + 1e-10)
                confidence = confidence + noise_temp * gumbel

            confidence = torch.where(unmasked, torch.full_like(confidence, float("inf")), confidence)

            ratio_next = mask_ratio(t_next, self.mask_schedule)
            n_masked = int(math.floor(self.num_tokens * float(ratio_next)))
            n_keep = max(1, min(self.num_tokens, self.num_tokens - n_masked))

            top_idx = confidence.topk(k=n_keep, dim=-1).indices
            new_unmasked = torch.zeros_like(unmasked)
            new_unmasked.scatter_(dim=1, index=top_idx, value=True)

            kept = torch.where(unmasked, state, sampled)
            state = torch.where(new_unmasked, kept, torch.full_like(state, self.mask_token_id))
            unmasked = new_unmasked

        final_tokens = state.reshape(batch_size, *self.latent_shape)
        return tokenizer.decode(final_tokens)
