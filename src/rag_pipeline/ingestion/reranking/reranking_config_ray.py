"""
reranking_config_ray.py
=======================
RayTrainingConfig dataclass — single source of truth for all Ray training
hyperparameters. All values are read from configs/rerankers.json via the
Paths class. No hardcoded defaults exist here; the JSON file owns every value.

Public API
----------
    RayTrainingConfig.from_rerankers_json() -> RayTrainingConfig
    cfg.apply_cli_overrides(args)
    cfg.to_dict() -> dict
"""
from __future__ import annotations

import json
import logging
import traceback
from argparse import Namespace
from dataclasses import dataclass

from typing import Any

from rag_pipeline.core.paths import Paths

logger = logging.getLogger(__name__)


@dataclass
class RayTrainingConfig:
    """
    Mirrors configs/rerankers.json > ray_training block.

    Every field is populated exclusively from the JSON — no inline defaults.
    CLI overrides are applied after loading via apply_cli_overrides().
    """
    # model
    model_name:             str
    max_length:             int
    num_labels:             int

    # data
    triples_path:           str
    max_negatives:          int
    dataloader_num_workers: int

    # training
    output_dir:             str
    epochs:                 int
    batch_size:             int
    lr:                     float
    weight_decay:           float
    warmup_ratio:           float
    grad_clip:              float
    fp16:                   bool
    log_every_n_steps:      int

    # loss
    alpha:                  float

    # ray
    num_workers:            int
    use_gpu:                bool

    # ------------------------------------------------------------------ #
    # Loaders                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_rerankers_json(cls) -> "RayTrainingConfig":
        """
        Load and validate from configs/rerankers.json.

        Resolution order for every value:
            ray_training block  >  training block  (no hardcoded fallback)

        Raises
        ------
        RuntimeError
            If a required key is missing from the JSON or paths cannot be
            resolved.
        """
        rerankers_path = Paths.base() / "configs" / "rerankers.json"
        logger.info("Loading RayTrainingConfig from %s", rerankers_path)

        try:
            raw: dict[str, Any] = json.loads(rerankers_path.read_text(encoding="utf-8"))
        except Exception:
            logger.error("Failed to read %s\n%s", rerankers_path, traceback.format_exc())
            raise RuntimeError(f"Could not load rerankers config: {rerankers_path}")

        rt:  dict[str, Any] = raw.get("ray_training", {})
        tr:  dict[str, Any] = raw.get("training", {})

        if not rt:
            raise RuntimeError(
                "configs/rerankers.json is missing the 'ray_training' block. "
                "Add it before running training."
            )

        # ---- model -------------------------------------------------------
        model_key  = cls._require(rt, "model_key", "ray_training.model_key")
        model_name = cls._resolve_model_name(raw, model_key)

        # ---- paths -------------------------------------------------------
        sample_size  = cls._require(tr, "sample_size", "training.sample_size")
        triples_file = f"triples_sample_{sample_size}.json"
        triples_path = str(
            Paths.experiments_dir()
            / "reranker_training"
            / triples_file
        )

        output_subdir = cls._require(rt, "output_subdir", "ray_training.output_subdir")
        output_dir    = str(Paths.experiments_dir() / "reranker_models" / output_subdir)

        # ---- build -------------------------------------------------------
        cfg = cls(
            model_name             = model_name,
            max_length             = cls._require(rt, "max_length",             "ray_training.max_length"),
            num_labels             = cls._require(rt, "num_labels",             "ray_training.num_labels"),
            triples_path           = triples_path,
            max_negatives          = cls._require(rt, "max_negatives",          "ray_training.max_negatives"),
            dataloader_num_workers = cls._require(rt, "dataloader_num_workers", "ray_training.dataloader_num_workers"),
            output_dir             = output_dir,
            epochs                 = cls._require(rt, "epochs",                 "ray_training.epochs"),
            batch_size             = cls._require(rt, "batch_size",             "ray_training.batch_size"),
            lr                     = cls._require(rt, "lr",                     "ray_training.lr"),
            weight_decay           = cls._require(rt, "weight_decay",           "ray_training.weight_decay"),
            warmup_ratio           = cls._require(rt, "warmup_ratio",           "ray_training.warmup_ratio"),
            grad_clip              = cls._require(rt, "grad_clip",              "ray_training.grad_clip"),
            fp16                   = cls._require(rt, "fp16",                   "ray_training.fp16"),
            log_every_n_steps      = cls._require(rt, "log_every_n_steps",      "ray_training.log_every_n_steps"),
            alpha                  = cls._require(rt, "alpha",                  "ray_training.alpha"),
            num_workers            = cls._require(rt, "num_workers",            "ray_training.num_workers"),
            use_gpu                = cls._require(rt, "use_gpu",                "ray_training.use_gpu"),
        )

        logger.info("RayTrainingConfig loaded — model=%s  epochs=%d  batch=%d",
                    cfg.model_name, cfg.epochs, cfg.batch_size)
        return cfg

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _require(block: dict[str, Any], key: str, full_path: str) -> Any:
        """Raise a clear error if a required JSON key is absent."""
        if key not in block:
            raise RuntimeError(
                f"Required key '{full_path}' is missing from configs/rerankers.json."
            )
        return block[key]

    @staticmethod
    def _resolve_model_name(raw: dict[str, Any], model_key: str) -> str:
        """Look up the HuggingFace model string from the models list by name."""
        for m in raw.get("models", []):
            if m.get("name") == model_key:
                logger.debug("Resolved model_key '%s' → '%s'", model_key, m["model"])
                return m["model"]
        raise RuntimeError(
            f"model_key '{model_key}' not found in configs/rerankers.json models list. "
            f"Available: {[m.get('name') for m in raw.get('models', [])]}"
        )

    def apply_cli_overrides(self, args: Namespace) -> None:
        """
        Overwrite config fields with non-None CLI values.

        argparse uses underscores for dest names even when the flag uses
        hyphens, so we map hyphenated CLI names → dataclass field names.
        """
        mapping: dict[str, str] = {
            "epochs":        "epochs",
            "batch_size":    "batch_size",
            "lr":            "lr",
            "alpha":         "alpha",
            "warmup_ratio":  "warmup_ratio",
            "num_workers":   "num_workers",
            "max_negatives": "max_negatives",
            "max_length":    "max_length",
            "output":        "output_dir",
            "triples":       "triples_path",
            "model_name":    "model_name",
        }
        for cli_key, field_name in mapping.items():
            val = getattr(args, cli_key, None)
            if val is not None:
                logger.debug("CLI override: %s = %s", field_name, val)
                setattr(self, field_name, val)

    def to_dict(self) -> dict[str, Any]:
        """Flat serialisable dict for Ray train_loop_config."""
        return {k: v for k, v in self.__dict__.items()}
