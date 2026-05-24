'''
src/rag_pipeline/core/paths.py
'''
import json
from pathlib import Path
from typing import Optional, Dict, Any


class Paths:
    """Single source of truth — strictly reads from configs/paths.json"""
    _base: Optional[Path] = None
    _config: Optional[Dict[str, Any]] = None

    @classmethod
    def base(cls) -> Path:
        if cls._base is None:
            current = Path(__file__).resolve()
            for parent in [current] + list(current.parents):
                if (parent / "pyproject.toml").exists():
                    cls._base = parent
                    break
            else:
                raise RuntimeError("Could not find project root (pyproject.toml)")
        return cls._base

    @classmethod
    def _load_config(cls) -> Dict[str, Any]:
        if cls._config is None:
            config_path = cls.base() / "configs" / "paths.json"
            try:
                with open(config_path, encoding="utf-8") as f:
                    cls._config = json.load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to load configs/paths.json: {e}")
            if not cls._config:
                raise RuntimeError("configs/paths.json is empty")
        return cls._config

    @classmethod
    def _resolve(cls, key: str) -> Path:
        return cls.base() / cls._load_config()[key]

    @classmethod
    def raw_dir(cls) -> Path:
        return cls._resolve("raw_dir")

    @classmethod
    def processed_dir(cls) -> Path:
        return cls._resolve("processed_dir")

    @classmethod
    def experiments_dir(cls) -> Path:
        return cls._resolve("experiments_dir")

    @classmethod
    def clean_jsonl(cls) -> Path:
        return cls._resolve("clean_jsonl")

    @classmethod
    def test_jsonl(cls) -> Path:
        return cls._resolve("test_jsonl")

    @classmethod
    def topic_assignments(cls) -> Path:
        return cls._resolve("topic_assignments")

    @classmethod
    def reranker_results_dir(cls) -> Path:
        return cls._resolve("reranker_results_dir")

    @classmethod
    def input_file(cls, stage: str) -> Path:
        mapping = cls._load_config().get("input_mapping", {})
        if stage not in mapping:
            raise ValueError(f"Unknown stage: {stage}. Available: {list(mapping)}")
        return cls.base() / mapping[stage]

    @classmethod
    def output_file(cls, stage: str) -> Path:
        mapping = cls._load_config().get("output_mapping", {})
        if stage not in mapping:
            raise ValueError(f"Unknown stage: {stage}. Available: {list(mapping)}")
        return cls.base() / mapping[stage]
