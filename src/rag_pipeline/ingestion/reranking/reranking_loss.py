"""
reranking_loss.py
=================
AdaptiveListwiseLoss — listwise cross-entropy with continuous hard-negative
weighting for cross-encoder reranker fine-tuning.

Loss derivation
---------------
Given model scores s = [s_pos, s_neg_0, ..., s_neg_{N-1}] for a query group:

    Base loss   : L_base    = -log_softmax(s)[0]
    Hardness    : h_i       = s_neg_i - s_pos   (closer → harder)
    Weights     : w_i       = softmax(h)[i]      (differentiable, sum = 1)
    Penalty     : L_penalty = Σ_i w_i * -log_softmax(s)[i+1]
    Total       : L         = L_base + alpha * L_penalty

Padding positions (mask == False) are excluded from both the softmax
distribution and the penalty via -inf masking and 0.0 fill respectively,
preventing 0 * inf = NaN.

Public API
----------
    AdaptiveListwiseLoss(alpha)
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class AdaptiveListwiseLoss(nn.Module):
    """
    Listwise cross-entropy with adaptive hard-negative weighting.

    Parameters
    ----------
    alpha : float
        Weight of the adaptive negative penalty term relative to the base
        cross-entropy loss. 0.0 reduces to standard listwise CE; 1.0 gives
        equal weight to both terms. Sourced from ray_training.alpha in
        configs/rerankers.json.
    """

    def __init__(self, alpha: float) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                f"alpha must be in [0, 1], got {alpha}. "
                "Check ray_training.alpha in configs/rerankers.json."
            )
        self.alpha = alpha
        logger.debug("AdaptiveListwiseLoss initialised with alpha=%.3f", alpha)

    def forward(
        self,
        scores: torch.Tensor,
        mask:   torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the adaptive listwise loss for a batch of query groups.

        Parameters
        ----------
        scores : (B, G)  raw logits from the cross-encoder
                         G = 1 positive + N negatives
        mask   : (B, G)  bool; True = real candidate, False = padding

        Returns
        -------
        Scalar loss tensor.
        """
        # Exclude padding from the softmax distribution
        masked_scores = scores.masked_fill(~mask, float("-inf"))
        log_probs     = F.log_softmax(masked_scores, dim=-1)   # (B, G)

        # --- Base loss: push positive to rank 0 ---
        base_loss = -log_probs[:, 0]                           # (B,)

        # --- Adaptive penalty ---
        neg_scores = scores[:, 1:]                             # (B, N)  raw
        pos_scores = scores[:, 0:1]                            # (B, 1)
        neg_mask   = mask[:, 1:]                               # (B, N)

        # Difference-based hardness; padding excluded via -inf before softmax
        hardness = (neg_scores - pos_scores).masked_fill(~neg_mask, float("-inf"))
        weights  = torch.softmax(hardness, dim=-1) * neg_mask.float()

        # Fill masked log-probs with 0.0 to avoid 0 * inf = NaN
        neg_log_probs = log_probs[:, 1:].masked_fill(~neg_mask, 0.0)
        penalty       = (weights * (-neg_log_probs)).sum(dim=-1)   # (B,)

        loss = (base_loss + self.alpha * penalty).mean()

        if torch.isnan(loss):
            logger.error(
                "NaN loss detected — scores min=%.4f max=%.4f  "
                "check for degenerate batches or extreme logits",
                scores.min().item(), scores.max().item(),
            )

        return loss
