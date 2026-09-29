from __future__ import annotations
import torch
import torch.nn.functional as F


def kd_topk_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    top_idx: torch.Tensor,
    top_logprob: torch.Tensor,
    kl_mask: torch.Tensor,
    alpha_ce: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vocab = logits.shape[-1]
    logsumexp = torch.logsumexp(logits.float(), dim=-1)
    gathered = torch.gather(logits, dim=-1, index=top_idx)
    student_logq = gathered.float() - logsumexp.unsqueeze(-1)
    teacher_logp = top_logprob.float()
    teacher_p = teacher_logp.exp()
    kl_per_pos = (teacher_p * (teacher_logp - student_logq)).sum(-1)
    mask = kl_mask.to(kl_per_pos.dtype)
    kl = (kl_per_pos * mask).sum() / mask.sum().clamp(min=1.0)
    ce = F.cross_entropy(
        logits[:, :-1].reshape(-1, vocab), labels[:, 1:].reshape(-1), ignore_index=-100
    )
    loss = (1.0 - alpha_ce) * kl
    if alpha_ce > 0.0:
        loss = loss + alpha_ce * ce
    return (loss, ce.detach(), kl.detach())
