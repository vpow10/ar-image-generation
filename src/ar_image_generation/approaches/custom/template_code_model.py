from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ar_image_generation.models.transformer import TransformerBackbone


@dataclass(slots=True)
class ResidualTemplateCodeARConfig:
    num_primitives: int = 256
    num_quantizers: int = 2
    num_feature_codes: int = 1024
    num_templates_per_class: int = 8
    dim: int = 512
    depth: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.10
    class_conditional: bool = True
    num_classes: int = 9


@dataclass(slots=True)
class ResidualTemplateCodeAROutput:
    logits: list[torch.Tensor]
    loss: torch.Tensor
    code_loss: torch.Tensor


class ResidualTemplateCodeAR(nn.Module):
    def __init__(self, cfg: ResidualTemplateCodeARConfig) -> None:
        super().__init__()

        self.cfg = cfg

        self.code_embeddings = nn.ModuleList(
            [
                nn.Embedding(cfg.num_feature_codes, cfg.dim)
                for _ in range(cfg.num_quantizers)
            ]
        )

        self.position_embedding = nn.Embedding(cfg.num_primitives, cfg.dim)
        self.template_embedding = nn.Embedding(cfg.num_templates_per_class, cfg.dim)
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

        self.code_heads = nn.ModuleList(
            [
                nn.Linear(cfg.dim, cfg.num_feature_codes)
                for _ in range(cfg.num_quantizers)
            ]
        )

        self.code_condition_projs = nn.ModuleList(
            [nn.Linear(cfg.dim, cfg.dim) for _ in range(max(cfg.num_quantizers - 1, 0))]
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.bos_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.template_embedding.weight, mean=0.0, std=0.02)

        for embedding in self.code_embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)

        if self.class_embedding is not None:
            nn.init.normal_(self.class_embedding.weight, mean=0.0, std=0.02)

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

    def _prepare_template_ids(
        self,
        template_ids: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.LongTensor:
        if template_ids is None:
            return torch.randint(
                low=0,
                high=self.cfg.num_templates_per_class,
                size=(batch_size,),
                device=device,
                dtype=torch.long,
            )

        template_ids = template_ids.to(device=device, dtype=torch.long)

        if template_ids.ndim == 2 and template_ids.shape[1] == 1:
            template_ids = template_ids.squeeze(1)

        if template_ids.ndim != 1:
            raise ValueError(
                f"Expected template ids [B], got shape {tuple(template_ids.shape)}"
            )

        if template_ids.shape[0] != batch_size:
            raise ValueError(
                f"Expected {batch_size} template ids, got {template_ids.shape[0]}"
            )

        if template_ids.min().item() < 0:
            raise ValueError("template_ids must be non-negative.")

        if template_ids.max().item() >= self.cfg.num_templates_per_class:
            raise ValueError(
                f"template_ids must be smaller than {self.cfg.num_templates_per_class}"
            )

        return template_ids

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

    def _embed_codes(self, codes: torch.LongTensor) -> torch.Tensor:
        if codes.shape[-1] != self.cfg.num_quantizers:
            raise ValueError(
                f"Expected {self.cfg.num_quantizers} code streams, got {codes.shape[-1]}"
            )

        embeddings = [
            embedding(codes[..., index])
            for index, embedding in enumerate(self.code_embeddings)
        ]

        return torch.stack(embeddings, dim=0).sum(dim=0)

    def _build_input_embeddings(
        self,
        *,
        prefix_codes: torch.LongTensor,
        labels: torch.Tensor | None,
        template_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size = prefix_codes.shape[0]
        prefix_length = prefix_codes.shape[1]
        sequence_length = prefix_length + 1
        device = prefix_codes.device

        if sequence_length > self.cfg.num_primitives:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds num_primitives={self.cfg.num_primitives}"
            )

        bos = self.bos_embedding.expand(batch_size, 1, -1)

        if prefix_length == 0:
            x = bos
        else:
            x = torch.cat([bos, self._embed_codes(prefix_codes)], dim=1)

        position_ids = torch.arange(sequence_length, device=device, dtype=torch.long)
        x = x + self.position_embedding(position_ids)[None, :, :]

        prepared_template_ids = self._prepare_template_ids(
            template_ids,
            batch_size=batch_size,
            device=device,
        )
        x = x + self.template_embedding(prepared_template_ids)[:, None, :]

        if self.class_embedding is not None:
            prepared_labels = self._prepare_labels(
                labels,
                batch_size=batch_size,
                device=device,
            )
            assert prepared_labels is not None
            x = x + self.class_embedding(prepared_labels)[:, None, :]

        return x

    def _code_logits(
        self,
        hidden: torch.Tensor,
        codes: torch.LongTensor | None,
    ) -> list[torch.Tensor]:
        logits: list[torch.Tensor] = []
        conditioned_hidden = hidden

        for index, head in enumerate(self.code_heads):
            logits.append(head(conditioned_hidden))

            if index < self.cfg.num_quantizers - 1:
                if codes is None:
                    raise ValueError("codes must be provided for conditioned logits.")
                code_embedding = self.code_embeddings[index](codes[..., index])
                conditioned_hidden = hidden + self.code_condition_projs[index](
                    code_embedding
                )

        return logits

    def forward(
        self,
        codes: torch.LongTensor,
        labels: torch.Tensor | None = None,
        template_ids: torch.Tensor | None = None,
    ) -> ResidualTemplateCodeAROutput:
        if codes.ndim != 3:
            raise ValueError(
                f"Expected codes [B, N, Q], got shape {tuple(codes.shape)}"
            )

        batch_size, num_primitives, num_quantizers = codes.shape

        if num_primitives != self.cfg.num_primitives:
            raise ValueError(
                f"Expected {self.cfg.num_primitives} primitives, got {num_primitives}"
            )

        if num_quantizers != self.cfg.num_quantizers:
            raise ValueError(
                f"Expected {self.cfg.num_quantizers} quantizers, got {num_quantizers}"
            )

        prefix_codes = codes[:, :-1, :]

        x = self._build_input_embeddings(
            prefix_codes=prefix_codes,
            labels=labels,
            template_ids=template_ids,
        )

        attention_mask = self._causal_attention_mask(x.shape[1], x.device)
        hidden = self.transformer(x, attention_mask=attention_mask)

        logits = self._code_logits(hidden, codes)

        losses = []
        for index, stream_logits in enumerate(logits):
            loss = F.cross_entropy(
                stream_logits.reshape(-1, self.cfg.num_feature_codes),
                codes[..., index].reshape(-1),
            )
            losses.append(loss)

        code_loss = torch.stack(losses).mean()

        return ResidualTemplateCodeAROutput(
            logits=logits,
            loss=code_loss,
            code_loss=code_loss,
        )

    def _sample_code(
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
        template_ids: torch.Tensor | None,
        device: torch.device,
        code_temperature: float = 0.85,
        top_k: int | None = 64,
    ) -> tuple[torch.LongTensor, torch.LongTensor | None, torch.LongTensor]:
        prepared_labels = self._prepare_labels(
            labels,
            batch_size=batch_size,
            device=device,
        )

        prepared_template_ids = self._prepare_template_ids(
            template_ids,
            batch_size=batch_size,
            device=device,
        )

        codes = torch.empty(
            batch_size,
            0,
            self.cfg.num_quantizers,
            dtype=torch.long,
            device=device,
        )

        for _ in range(self.cfg.num_primitives):
            x = self._build_input_embeddings(
                prefix_codes=codes,
                labels=prepared_labels,
                template_ids=prepared_template_ids,
            )

            attention_mask = self._causal_attention_mask(x.shape[1], x.device)
            hidden = self.transformer(x, attention_mask=attention_mask)
            last_hidden = hidden[:, -1:, :]

            sampled_codes: list[torch.Tensor] = []
            conditioned_hidden = last_hidden

            for index, head in enumerate(self.code_heads):
                logits = head(conditioned_hidden)[:, 0, :]

                sampled_code = self._sample_code(
                    logits,
                    temperature=code_temperature,
                    top_k=top_k,
                )
                sampled_codes.append(sampled_code)

                if index < self.cfg.num_quantizers - 1:
                    code_embedding = self.code_embeddings[index](sampled_code)[
                        :, None, :
                    ]
                    conditioned_hidden = last_hidden + self.code_condition_projs[index](
                        code_embedding
                    )

            next_codes = torch.stack(sampled_codes, dim=-1)
            codes = torch.cat([codes, next_codes[:, None, :]], dim=1)

        return codes, prepared_labels, prepared_template_ids
