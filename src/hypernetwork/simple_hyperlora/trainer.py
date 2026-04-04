"""
Training utilities: DistillationTrainer (KL loss) and CrossEntropyTrainer (CE loss).

Both support per-context average loss and L1 regularization on generated LoRA weights.
"""

import logging

import torch
from torch import nn
from transformers import Trainer
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import IntervalStrategy

logger = logging.getLogger()


# ---------------------------------------------------------------------------
# Per-context loss computation
# ---------------------------------------------------------------------------

def per_ctx_loss_ce(inputs, labels, loss):
    """Compute per-context averaged CE loss (loss has 0 at masked positions)."""
    n_queries_per_ctx = inputs["n_queries"].tolist()
    position_ids = inputs["position_ids"].squeeze(0)
    label_mask = labels.squeeze(0) != -100
    label_pos_ids = label_mask * position_ids
    label_pos_ids_diff = label_pos_ids.diff(
        append=torch.tensor([0], device=position_ids.device)
    )
    start_label_pos = torch.where((label_pos_ids_diff > 0) * ~label_mask)[0]
    end_label_pos = torch.where((label_pos_ids_diff < 0) * label_mask)[0]
    label_seq_lens = end_label_pos - start_label_pos

    qa_losses = torch.stack([
        loss[s : s + l].mean() for s, l in zip(start_label_pos, label_seq_lens)
    ])
    per_ctx_losses = [ql.mean() for ql in torch.split(qa_losses, n_queries_per_ctx)]
    return torch.stack(per_ctx_losses)


def per_ctx_loss_kl(inputs, labels, loss):
    """Compute per-context averaged KL loss (loss is compact, label indices only)."""
    n_queries_per_ctx = inputs["n_queries"].tolist()
    position_ids = inputs["position_ids"].squeeze(0)
    label_mask = labels.squeeze(0) != -100
    label_pos_ids = label_mask * position_ids
    label_pos_ids_diff = label_pos_ids.diff(
        append=torch.tensor([0], device=position_ids.device)
    )
    start_label_pos = torch.where((label_pos_ids_diff > 0) * ~label_mask)[0]
    end_label_pos = torch.where((label_pos_ids_diff < 0) * label_mask)[0]
    label_seq_lens = end_label_pos - start_label_pos

    cu_lens = torch.cumsum(label_seq_lens, dim=0)
    start_indices = torch.cat((torch.tensor([0], device=cu_lens.device), cu_lens[:-1]))

    qa_losses = torch.stack([
        loss[s:e].mean() for s, e in zip(start_indices, cu_lens)
    ])
    per_ctx_losses = [ql.mean() for ql in torch.split(qa_losses, n_queries_per_ctx)]
    return torch.stack(per_ctx_losses)


# ---------------------------------------------------------------------------
# Custom batch sampling (for per-context loss normalization)
# ---------------------------------------------------------------------------

class ModulatedModelTrainer(Trainer):
    def get_batch_samples(self, epoch_iterator, num_batches, device):
        batch_samples = []
        num_items_in_batch = None

        for _ in range(num_batches):
            try:
                batch_samples.append(next(epoch_iterator))
            except StopIteration:
                break

        if (
            len(batch_samples) > 0
            and "labels" in batch_samples[0]
            and "n_ctx_chunks" in batch_samples[0]
        ):
            num_items_in_batch = {
                "ctx": torch.tensor(
                    sum(b["n_ctx_chunks"].numel() for b in batch_samples)
                ).to(device),
                "labels": sum(
                    (b["labels"].ne(-100)).sum() for b in batch_samples
                ).to(device),
            }

        return batch_samples, num_items_in_batch


# ---------------------------------------------------------------------------
# DistillationTrainer (KL loss)
# ---------------------------------------------------------------------------

class DistillationTrainer(ModulatedModelTrainer):
    def __init__(self, *args, **kwargs):
        self.gen_lora_l1_reg_coef = kwargs.pop("gen_lora_l1_reg_coef", 0.0)
        self.use_per_ctx_average_loss = kwargs.pop("use_per_ctx_average_loss", False)
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        is_train = num_items_in_batch is not None
        labels = inputs.pop("labels", None)
        label_pos = torch.where(labels != -100)
        outputs, (gen_loras, _) = model(**inputs, return_generated_lora=True)

        if "logprobs_vals" not in inputs:
            zero = torch.tensor(0.0)
            return (zero, outputs) if return_outputs else zero

        target_logp = inputs.pop("logprobs_vals").squeeze(0)
        indices = inputs.pop("logprobs_indices").squeeze(0)

        # KL divergence loss
        logits = outputs.logits[label_pos[0], label_pos[1] - 1]  # shift back 1
        logq_denom = torch.logsumexp(logits, dim=-1, keepdim=True)
        logq_selected = logits.gather(1, indices) - logq_denom
        p = target_logp.exp()
        loss = -(p * logq_selected).sum(dim=-1)

        if self.use_per_ctx_average_loss:
            loss = per_ctx_loss_kl(inputs, labels, loss)

        if is_train:
            denom = num_items_in_batch["ctx"] if self.use_per_ctx_average_loss else num_items_in_batch["labels"]
            loss = loss.sum() / denom
        else:
            loss = loss.mean()

        # L1 regularization on generated LoRA
        l1_norm = _compute_l1_norm(gen_loras)
        if is_train:
            l1_norm /= num_items_in_batch["ctx"]

        total_loss = loss + self.gen_lora_l1_reg_coef * l1_norm

        scaler = self.args.gradient_accumulation_steps if is_train else 1
        if self.args.average_tokens_across_devices and is_train:
            total_loss *= self.accelerator.num_processes
            scaler *= self.accelerator.num_processes

        if _should_log(self):
            self.log({"kl_loss": loss.item() * scaler, "gen_lora_l1_norm": l1_norm.item() * scaler})

        return (total_loss, outputs) if return_outputs else total_loss


# ---------------------------------------------------------------------------
# CrossEntropyTrainer (CE loss)
# ---------------------------------------------------------------------------

def causal_lm_ce_loss(logits, labels, vocab_size):
    """Compute per-token cross-entropy loss (no reduction)."""
    logits = logits.float()
    labels = nn.functional.pad(labels, (0, 1), value=-100)
    shift_labels = labels[..., 1:].contiguous().view(-1).to(logits.device)
    logits = logits.view(-1, vocab_size)
    return nn.functional.cross_entropy(logits, shift_labels, reduction="none")


class CrossEntropyTrainer(ModulatedModelTrainer):
    def __init__(self, *args, **kwargs):
        self.gen_lora_l1_reg_coef = kwargs.pop("gen_lora_l1_reg_coef", 0.0)
        self.use_per_ctx_average_loss = kwargs.pop("use_per_ctx_average_loss", False)
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        is_train = num_items_in_batch is not None
        labels = inputs.pop("labels", None)
        outputs, (gen_loras, _) = model(**inputs, return_generated_lora=True)
        logits = outputs.logits

        loss = causal_lm_ce_loss(logits, labels, model.config.vocab_size)

        if self.use_per_ctx_average_loss:
            loss = per_ctx_loss_ce(inputs, labels, loss)

        if is_train:
            denom = num_items_in_batch["ctx"] if self.use_per_ctx_average_loss else num_items_in_batch["labels"]
            loss = loss.sum() / denom
        else:
            loss = loss.mean()

        # L1 regularization
        l1_norm = _compute_l1_norm(gen_loras)
        if is_train:
            l1_norm /= num_items_in_batch["ctx"]

        total_loss = loss + self.gen_lora_l1_reg_coef * l1_norm

        scaler = self.args.gradient_accumulation_steps if is_train else 1
        if self.args.average_tokens_across_devices and is_train:
            total_loss *= self.accelerator.num_processes
            scaler *= self.accelerator.num_processes

        if _should_log(self):
            self.log({"ce_loss": loss.item() * scaler, "gen_lora_l1_norm": l1_norm.item() * scaler})

        return (total_loss, outputs) if return_outputs else total_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_l1_norm(gen_loras):
    l1 = 0
    n = len(gen_loras)
    for lora in gen_loras.values():
        l1 += lora["A"].abs().sum(0).mean() + lora["B"].abs().sum(0).mean()
    return l1 / n


def _should_log(trainer):
    return (
        (trainer.state.global_step == 1 and trainer.args.logging_first_step)
        or (
            trainer.args.logging_strategy == IntervalStrategy.STEPS
            and trainer.state.global_step % trainer.state.logging_steps == 0
        )
    )


def get_decay_parameter_names(model) -> list[str]:
    """Get parameter names for weight decay (exclude embeddings, biases, norms, scalers)."""
    return get_parameter_names(
        model,
        [nn.Embedding, nn.LayerNorm],
        ["scaler", "bias", "layernorm", "rmsnorm", "latents_q"],
    )
