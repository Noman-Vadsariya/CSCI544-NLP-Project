"""Simplified LoRA merging for single-chunk contexts (adds bias only)."""

import torch
from jaxtyping import Float, Integer
from torch import Tensor


def combine_lora(
    generated_loras: dict[str, dict[str, Tensor]],
    n_chunks: Integer[Tensor, "n_ctx"],
    lora_bias: dict[str, dict[str, Tensor]] | None = None,
) -> dict[str, dict[str, Tensor]]:
    """Combine generated LoRA with bias. Single-chunk only (n_chunks should be all 1s)."""
    first_module = next(iter(generated_loras))
    base_rank = generated_loras[first_module]["A"].shape[-2]

    combined_loras: dict[str, dict[str, Tensor]] = {}

    for module_name, module_loras in generated_loras.items():
        combined_loras[module_name] = {}
        for matrix_key in ("A", "B"):
            loras = module_loras[matrix_key]  # [n_ctx, n_layers, r, dim]

            if lora_bias is not None:
                bias = lora_bias[module_name][matrix_key]  # [n_layers, r, dim]
                # Concatenate bias along rank dimension for each context
                bias_expanded = bias.unsqueeze(0).expand(loras.shape[0], -1, -1, -1)
                loras = torch.cat([loras, bias_expanded], dim=-2)

            combined_loras[module_name][matrix_key] = loras

    return combined_loras
