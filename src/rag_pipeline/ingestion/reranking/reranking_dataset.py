"""
reranking_dataset.py
====================
Dataset and collation for listwise reranker fine-tuning.

Each item in the triples list becomes a QueryGroup: one query paired with
its positive answer and N hard negatives.  Groups with fewer than
max_negatives real negatives are padded with the positive text; padding
positions are masked out in the loss so they contribute no gradient.

Design notes
------------
* The raw triples list is accepted as an in-memory argument (loaded once
  on the driver, shared via Ray object store) — no per-worker file I/O.
* __getitem__ returns a plain dict instead of a dataclass to eliminate
  per-sample Python object allocation on the hot path.
* make_collate_fn uses a fast HuggingFace tokenizer and batch_encode_plus
  with padding="longest" so sequence length is tight to the actual batch,
  not padded to max_length globally.
* The validity mask is pre-allocated as a (B, G) bool tensor and filled
  with scatter ops — no list-of-lists → tensor conversion per batch.

Public API
----------
    QueryGroup          dataclass  (used for type annotations / external callers)
    RerankerDataset     torch.utils.data.Dataset
    make_collate_fn()   -> Callable
"""
from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# QueryGroup — kept as a dataclass for external callers / type annotations,
# but the dataset itself stores plain dicts for speed.
# ---------------------------------------------------------------------------

@dataclass
class QueryGroup:
    """
    One training example: a query with its positive and hard negatives.

    Attributes
    ----------
    query            : the question text
    positive         : the correct answer
    negatives        : hard negatives padded to max_negatives with the positive
    n_real_negatives : count of real (non-padding) negatives
    """
    query:            str
    positive:         str
    negatives:        list[str]
    n_real_negatives: int


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RerankerDataset(Dataset):
    """
    Wraps an in-memory triples list as a torch Dataset.

    Parameters
    ----------
    triples       : list of triple dicts, already loaded by the driver.
                    Each dict must contain "query", "positive", and either
                    "hard_negatives" or "negatives".
    max_negatives : maximum hard negatives per group; shorter lists are
                    padded with the positive text (masked out in loss).

    Notes
    -----
    Accepts a pre-loaded list rather than a file path so the data is read
    once on the Ray driver and shared through the object store — workers
    never touch disk for training data.
    """

    def __init__(self, triples: list[dict[str, Any]], max_negatives: int) -> None:
        self._groups: list[dict[str, Any]] = []
        skipped = 0

        for item in triples:
            try:
                negs   = item.get("hard_negatives") or item.get("negatives") or []
                negs   = list(negs[:max_negatives])
                n_real = len(negs)

                # Pad to fixed width so batches are rectangular
                while len(negs) < max_negatives:
                    negs.append(item["positive"])

                # Plain dict — avoids dataclass __init__ overhead on hot path
                self._groups.append({
                    "query":            item["query"],
                    "positive":         item["positive"],
                    "negatives":        negs,
                    "n_real_negatives": n_real,
                })
            except KeyError as exc:
                logger.warning(
                    "Skipping malformed triple (missing key %s): %s", exc, item
                )
                skipped += 1

        logger.info(
            "RerankerDataset ready — %d groups loaded, %d skipped, "
            "max_negatives=%d",
            len(self._groups), skipped, max_negatives,
        )

    def __len__(self) -> int:
        return len(self._groups)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._groups[idx]


# ---------------------------------------------------------------------------
# Collate function factory
# ---------------------------------------------------------------------------

def make_collate_fn(tokenizer: PreTrainedTokenizerFast, max_length: int) -> Callable:
    """
    Build a collate function for DataLoader.

    Each group is tokenised as (1 + max_negatives) query-passage pairs:
        [query, positive], [query, neg_0], ..., [query, neg_{N-1}]

    The tokenizer must be a fast tokenizer (use_fast=True).  Padding is
    set to "longest" so the sequence dimension is tight to the batch —
    not inflated to max_length on every call.

    Returns a batch dict with shapes:
        input_ids / attention_mask / token_type_ids : (B, G, seq_len)
        mask                                        : (B, G)  bool
            True  = real candidate (positive or real negative)
            False = padding position (zeroed out in loss)
    """

    def _collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        B = len(batch)
        G = 1 + len(batch[0]["negatives"])   # 1 positive + max_negatives

        # Build flat pair lists in one pass
        queries:  list[str] = []
        passages: list[str] = []
        for g in batch:
            queries.append(g["query"])
            passages.append(g["positive"])
            for neg in g["negatives"]:
                queries.append(g["query"])
                passages.append(neg)

        try:
            encoded = tokenizer.batch_encode_plus(
                list(zip(queries, passages)),
                padding="longest",          # tight to the batch, not max_length
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
        except Exception:
            logger.error("Tokenisation failed\n%s", traceback.format_exc())
            raise

        L = encoded["input_ids"].shape[1]

        # Pre-allocate bool mask — all True, then zero out padding cols
        mask = torch.ones(B, G, dtype=torch.bool)
        for i, g in enumerate(batch):
            n_real = g["n_real_negatives"]
            # Positions [n_real+1 .. G-1] are padding (0-indexed; col 0 = positive)
            if n_real < G - 1:
                mask[i, n_real + 1:] = False

        result: dict[str, torch.Tensor] = {
            "input_ids":      encoded["input_ids"].view(B, G, L),
            "attention_mask": encoded["attention_mask"].view(B, G, L),
            "mask":           mask,
        }
        if "token_type_ids" in encoded:
            result["token_type_ids"] = encoded["token_type_ids"].view(B, G, L)

        return result

    return _collate