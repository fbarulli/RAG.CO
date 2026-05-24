"""
reranking_training_ray.py
=========================
Entry point for Ray Train reranker fine-tuning.

Orchestrates:
    RayTrainingConfig   — loads all hyperparameters from configs/rerankers.json
    RerankerDataset     — query groups from cached triples
    AdaptiveListwiseLoss— listwise CE with adaptive hard-negative weighting
    TorchTrainer        — Ray distributed training loop

Usage
-----
    # Default: all settings from configs/rerankers.json
    python -m rag_pipeline.ingestion.reranking.reranking_training_ray

    # CLI overrides (any subset):
    python -m rag_pipeline.ingestion.reranking.reranking_training_ray \\
        --epochs 5 --alpha 0.3 --num-workers 2
"""
from __future__ import annotations

import json
import logging
import math
import traceback
from pathlib import Path

import ray
import ray.train
import torch
import torch.nn as nn
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from configs.benchmark_cli import create_base_parser
from rag_pipeline.ingestion.reranking.reranking_config_ray import RayTrainingConfig
from rag_pipeline.ingestion.reranking.reranking_dataset import RerankerDataset, make_collate_fn
from rag_pipeline.ingestion.reranking.reranking_loss import AdaptiveListwiseLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ray worker loop
# ---------------------------------------------------------------------------

def train_loop_per_worker(config: dict) -> None:
    """
    Runs inside each Ray worker process.

    Receives a flat config dict (from RayTrainingConfig.to_dict()) since Ray
    serialises train_loop_config across process boundaries.
    """
    import ray.train.torch as ray_torch

    try:
        worker_rank = ray.train.get_context().get_world_rank()
        device      = ray_torch.get_device()
    except Exception:
        worker_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Worker %d starting on device %s", worker_rank, device)

    # --- Model ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)
        model     = AutoModelForSequenceClassification.from_pretrained(
            config["model_name"], num_labels=config["num_labels"]
        )
    except Exception:
        logger.error("Failed to load model '%s'\n%s", config["model_name"], traceback.format_exc())
        raise

    model = ray_torch.prepare_model(model)

    # --- Data ---
    try:
        dataset    = RerankerDataset(config["triples"], config["max_negatives"])
        collate    = make_collate_fn(tokenizer, config["max_length"])
        dataloader = DataLoader(
            dataset,
            batch_size  = config["batch_size"],
            shuffle     = False,       # DistributedSampler handles shuffling
            collate_fn  = collate,
            num_workers = config["dataloader_num_workers"],
            pin_memory  = True,
        )
        dataloader = ray_torch.prepare_data_loader(dataloader)
    except Exception:
        logger.error("Failed to build dataloader\n%s", traceback.format_exc())
        raise

    # --- Optimiser + scheduler ---
    optimizer    = torch.optim.AdamW(
        model.parameters(),
        lr           = config["lr"],
        weight_decay = config["weight_decay"],
    )
    total_steps  = math.ceil(len(dataloader) * config["epochs"])
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    loss_fn      = AdaptiveListwiseLoss(alpha=config["alpha"]).to(device)
    scaler       = torch.amp.GradScaler("cuda", enabled=config["fp16"])

    logger.info(
        "Training — epochs=%d  steps=%d  warmup=%d  lr=%.2e  alpha=%.2f",
        config["epochs"], total_steps, warmup_steps, config["lr"], config["alpha"],
    )

    global_step    = 0
    _pending_metrics: list[dict] = []

    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0.0

        for batch in dataloader:
            try:
                has_tti = "token_type_ids" in batch
                batch   = {k: v.to(device) for k, v in batch.items()}
                B, G, L = batch["input_ids"].shape

                flat_kwargs: dict = {
                    "input_ids":      batch["input_ids"].view(B * G, L),
                    "attention_mask": batch["attention_mask"].view(B * G, L),
                }
                if has_tti:
                    flat_kwargs["token_type_ids"] = batch["token_type_ids"].view(B * G, L)

                with torch.amp.autocast("cuda", enabled=config["fp16"]):
                    scores = model(**flat_kwargs).logits.squeeze(-1).view(B, G)
                    loss   = loss_fn(scores, batch["mask"])

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() > 0:
                    scheduler.step()

            except Exception:
                logger.error(
                    "Step %d failed\n%s", global_step, traceback.format_exc()
                )
                raise

            epoch_loss  += loss.item()
            global_step += 1

            _pending_metrics.append({"step": global_step, "loss": loss.item(), "epoch": epoch})
            if global_step % config["log_every_n_steps"] == 0:
                logger.info(
                    "epoch=%d  step=%d  loss=%.4f",
                    epoch, global_step, loss.item(),
                )
                ray.train.report(_pending_metrics[-1])
                _pending_metrics.clear()

        avg_loss = epoch_loss / len(dataloader)
        logger.info("Epoch %d/%d complete — avg_loss=%.4f", epoch + 1, config["epochs"], avg_loss)
        ray.train.report({"epoch_loss": avg_loss, "epoch": epoch})

    # --- Save (rank 0 only) ---
    if worker_rank == 0:
        out = Path(config["output_dir"])
        try:
            out.mkdir(parents=True, exist_ok=True)
            raw_model = model.module if hasattr(model, "module") else model
            raw_model.save_pretrained(out)
            tokenizer.save_pretrained(out)
            logger.info("Model saved → %s", out)
        except Exception:
            logger.error("Failed to save model\n%s", traceback.format_exc())
            raise


# ---------------------------------------------------------------------------
# Direct single-GPU training (no Ray) — use this for Colab
# ---------------------------------------------------------------------------

def train_direct(config: dict) -> None:
    """
    Pure PyTorch training loop — no Ray dependencies.
    Use this for single-GPU environments (Colab, local GPU).
    Identical hyperparameters and loss as train_loop_per_worker.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("train_direct starting on device %s", device)

    # --- Model ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)
        model     = AutoModelForSequenceClassification.from_pretrained(
            config["model_name"], num_labels=config["num_labels"]
        ).to(device)
    except Exception:
        logger.error("Failed to load model '%s'\n%s", config["model_name"], traceback.format_exc())
        raise

    # --- Data ---
    try:
        dataset    = RerankerDataset(config["triples"], config["max_negatives"])
        collate    = make_collate_fn(tokenizer, config["max_length"])
        dataloader = DataLoader(
            dataset,
            batch_size  = config["batch_size"],
            shuffle     = True,
            collate_fn  = collate,
            num_workers = config["dataloader_num_workers"],
            pin_memory  = device.type == "cuda",
        )
    except Exception:
        logger.error("Failed to build dataloader\n%s", traceback.format_exc())
        raise

    # --- Optimiser + scheduler ---
    optimizer    = torch.optim.AdamW(
        model.parameters(),
        lr           = config["lr"],
        weight_decay = config["weight_decay"],
    )
    total_steps  = math.ceil(len(dataloader) * config["epochs"])
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    loss_fn      = AdaptiveListwiseLoss(alpha=config["alpha"]).to(device)
    fp16         = config["fp16"] and device.type == "cuda"
    scaler       = torch.amp.GradScaler("cuda", enabled=fp16)

    logger.info(
        "Training — epochs=%d  steps=%d  warmup=%d  lr=%.2e  alpha=%.2f  fp16=%s",
        config["epochs"], total_steps, warmup_steps, config["lr"], config["alpha"], fp16,
    )

    global_step = 0

    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0.0

        for batch in dataloader:
            try:
                has_tti = "token_type_ids" in batch
                batch   = {k: v.to(device) for k, v in batch.items()}
                B, G, L = batch["input_ids"].shape

                flat_kwargs: dict = {
                    "input_ids":      batch["input_ids"].view(B * G, L),
                    "attention_mask": batch["attention_mask"].view(B * G, L),
                }
                if has_tti:
                    flat_kwargs["token_type_ids"] = batch["token_type_ids"].view(B * G, L)

                with torch.amp.autocast("cuda", enabled=fp16):
                    scores = model(**flat_kwargs).logits.squeeze(-1).view(B, G)
                    loss   = loss_fn(scores, batch["mask"])

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() > 0:
                    scheduler.step()

            except Exception:
                logger.error("Step %d failed\n%s", global_step, traceback.format_exc())
                raise

            epoch_loss  += loss.item()
            global_step += 1

            if global_step % config["log_every_n_steps"] == 0:
                logger.info("epoch=%d  step=%d  loss=%.4f", epoch, global_step, loss.item())

        avg_loss = epoch_loss / len(dataloader)
        logger.info("Epoch %d/%d complete — avg_loss=%.4f", epoch + 1, config["epochs"], avg_loss)

    # --- Save ---
    out = Path(config["output_dir"])
    try:
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out)
        tokenizer.save_pretrained(out)
        logger.info("Model saved → %s", out)
    except Exception:
        logger.error("Failed to save model\n%s", traceback.format_exc())
        raise


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args, _ = create_base_parser("Reranker Ray training").parse_known_args()

    try:
        cfg = RayTrainingConfig.from_rerankers_json()
    except Exception:
        logger.error("Config load failed\n%s", traceback.format_exc())
        raise

    cfg.apply_cli_overrides(args)

    logger.info("Effective config:\n%s", json.dumps(cfg.to_dict(), indent=2))

    _triples_path = Path(cfg.triples_path)
    if not _triples_path.exists():
        raise FileNotFoundError(f"Triples not found: {_triples_path}. Run create_training_triples.py first.")
    triples = json.loads(_triples_path.read_text(encoding="utf-8"))
    logger.info("Loaded %d triples from %s", len(triples), _triples_path)
    cfg_dict = cfg.to_dict()
    cfg_dict["triples"] = triples

    ray.init(ignore_reinit_error=True)

    trainer = TorchTrainer(
        train_loop_per_worker = train_loop_per_worker,
        train_loop_config     = cfg_dict,
        scaling_config        = ScalingConfig(
            num_workers = cfg.num_workers,
            use_gpu     = cfg.use_gpu,
        ),
    )

    try:
        result = trainer.fit()
        logger.info("Training complete: %s", result)
    except Exception:
        logger.error("Training failed\n%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
