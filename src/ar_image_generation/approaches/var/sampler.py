import torch
import torch.nn.functional as F


def apply_top_k_top_p_filtering(
    logits: torch.Tensor,
    *,
    top_k: int | None,
    top_p: float | None,
) -> torch.Tensor:
    """
    Args:
        logits: [..., vocab_size]

    Returns:
        Filtered logits with impossible tokens set to -inf.
    """

    filtered_logits = logits.clone()
    vocab_size = filtered_logits.shape[-1]

    if top_k is not None and top_k > 0:
        top_k = min(top_k, vocab_size)
        values, _ = torch.topk(filtered_logits, k=top_k, dim=-1)
        min_values = values[..., -1, None]
        filtered_logits = filtered_logits.masked_fill(filtered_logits < min_values, float("-inf"))

    if top_p is not None:
        if not 0.0 < top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p

            # Keep at least the most likely token.
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            sorted_logits = sorted_logits.masked_fill(sorted_indices_to_remove, float("-inf"))

            restored_logits = torch.full_like(filtered_logits, float("-inf"))
            restored_logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
            filtered_logits = restored_logits

    return filtered_logits


def sample_tokens_from_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> torch.LongTensor:
    """
    Args:
        logits: [B, N, vocab_size]

    Returns:
        sampled tokens [B, N]
    """

    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    logits = logits / temperature
    logits = apply_top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

    probs = F.softmax(logits, dim=-1)
    sampled = torch.multinomial(probs.reshape(-1, probs.shape[-1]), num_samples=1)

    return sampled.reshape(logits.shape[0], logits.shape[1])
