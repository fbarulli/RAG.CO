"""
configs/benchmark_cli.py
================
Argparse parser factories for all benchmark and ingestion scripts.

Each script calls the appropriate factory, parses its args, then merges
them into a ``BenchmarkConfig``::

    from .benchmark_cli import create_ingestion_parser
    from .benchmark_config import BenchmarkConfig

    args   = create_ingestion_parser().parse_args()
    config = BenchmarkConfig.from_defaults().merge_args(args)

Keeping parsers here (rather than in benchmark_config.py) means adding a
new script never touches the config dataclass.

Public API
----------
    create_base_parser()            -> argparse.ArgumentParser
    create_ingestion_parser()       -> argparse.ArgumentParser
    create_benchmark_parser()       -> argparse.ArgumentParser
    create_multi_benchmark_parser() -> argparse.ArgumentParser
"""
from __future__ import annotations
import argparse
from pathlib import Path

def _add_path_args(parser: argparse.ArgumentParser) -> None:
    """Add path-related arguments shared by all scripts."""
    g = parser.add_argument_group('paths')
    g.add_argument('--test-set', type=Path, default=None, help='Path to test.jsonl')
    g.add_argument('--clean-path', type=Path, default=None, help='Path to clean.jsonl')
    g.add_argument('--topic-path', type=Path, default=None, help='Path to topic assignments JSON')
    g.add_argument('--configs-path', type=Path, default=None, help='Path to retrieval_configs.json')
    g.add_argument('--output-dir', type=Path, default=None, help='Output directory for results')
    g.add_argument('--cache-dir', type=Path, default=None, help='Cache directory for embeddings')

def _add_qdrant_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group('qdrant')
    g.add_argument('--qdrant-host', type=str, default=None, help='Qdrant host (default: localhost)')
    g.add_argument('--qdrant-port', type=int, default=None, help='Qdrant port (default: 6333)')

def _add_es_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group('elasticsearch')
    g.add_argument('--es-host', type=str, default=None, help='Elasticsearch host URL')
    g.add_argument('--es-index', type=str, default=None, help='Elasticsearch index name')

def _add_model_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group('models')
    g.add_argument('--models', type=str, nargs='+', default=None, help='Models to use (default: all enabled models in models.json)')
    g.add_argument('--config', type=str, default=None, help='Single retrieval config to run (default: all configs)')

def _add_tuning_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group('settings')
    g.add_argument('--top-k', type=int, default=None, help='Results to retrieve per query')
    g.add_argument('--encode-batch-size', type=int, default=None, help='Encoding batch size')
    g.add_argument('--batch-size', type=int, default=None, help='Qdrant upsert batch size')

def _add_flag_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group('behaviour')
    g.add_argument('--force-encode', action='store_true', default=False, help='Re-encode even if a cached .npy file exists')
    g.add_argument('--skip-existing', action='store_true', default=None, help='Skip Qdrant collections that already have the right doc count')
    g.add_argument('--no-skip-existing', dest='skip_existing', action='store_false', help='Always recreate Qdrant collections')
    g.add_argument('--no-detail', action='store_true', default=False, help='Skip per-config detail report')
    g.add_argument('--auto-prepare', action='store_true', default=False, help='Auto-generate missing topic assignments before benchmarking')
    g.add_argument('--resume', action='store_true', default=None, help='Resume from a previous benchmark checkpoint')
    g.add_argument('--no-resume', dest='resume', action='store_false', help='Start the benchmark from scratch')
    g.add_argument('--reset', action='store_true', default=False, help='Wipe benchmark state before running')
    g.add_argument('--fail-fast', action='store_true', default=False, help='Abort on the first model/config error')
    g.add_argument('--timeout', type=int, default=None, help='Per-model timeout in seconds (default: 3600)')
    parser.add_argument('--sample-size', type=int, default=0, help='Number of queries to sample. 0 = full dataset (default).')

def create_base_parser(description: str='', **kwargs) -> argparse.ArgumentParser:
    """
    Standalone parser with all shared arguments.

    Scripts that need only a subset of arguments should use the
    script-specific factories below, which call this internally.
    """
    parser = argparse.ArgumentParser(description=description, **kwargs)
    _add_path_args(parser)
    _add_qdrant_args(parser)
    _add_es_args(parser)
    _add_model_args(parser)
    _add_tuning_args(parser)
    _add_flag_args(parser)
    return parser

def create_ingestion_parser() -> argparse.ArgumentParser:
    """
    Parser for ``p02_ingest_models.py``.

    Adds ``--model`` (singular) to target a single model by name,
    separate from ``--models`` (plural) which accepts a list.
    ``merge_args`` in ``BenchmarkConfig`` normalises both to ``config.models``.
    """
    parser = create_base_parser(description='Ingest FAQ documents into Qdrant for one or all embedding models.')
    parser.add_argument('--model', type=str, default=None, help="Ingest only this model (must match 'name' in models.json). Overrides --models.")
    return parser

def create_benchmark_parser() -> argparse.ArgumentParser:
    """
    Parser for ``p03_benchmark.py`` (benchmarks models).

    If ``--model`` is omitted, the script runs all models found in the config.
    """
    parser = create_base_parser(description='Run the retrieval benchmark for embedding/reranker models.')
    parser.add_argument('--model', type=str, required=False, default=None, help="Model to benchmark (matches 'name' in models.json). Omitting this runs all models.")
    return parser

def create_multi_benchmark_parser() -> argparse.ArgumentParser:
    """
    Parser for ``p04_multi_model_benchmark.py``.

    No additional arguments beyond the shared base — ``--models`` from the
    base parser is sufficient for selecting a subset of models.
    """
    return create_base_parser(description='Run the retrieval benchmark across all (or selected) embedding models.')

def create_generation_parser() -> argparse.ArgumentParser:
    """
    Parser for ``p06_answer_generation.runner.py``.

    Adds generation-specific arguments for prompt styles and top-k values.
    """
    parser = create_base_parser(description='Run answer generation evaluation across prompt styles and top-k values.')
    g = parser.add_argument_group('generation')
    g.add_argument('--model', type=str, default=None, help='Retrieval model to use for context (default: from config)')
    g.add_argument('--retrieval-config', type=str, default=None, help='Retrieval config name (for reference, not actually used in retrieval)')
    g.add_argument('--llm-model', type=str, default=None, help='LLM model for generation (default: nvidia_nim/meta/llama-3.1-70b-instruct)')
    g.add_argument('--prompt-style', type=str, default=None, help='Single prompt style to use (default: strict)')
    g.add_argument('--styles', type=str, nargs='+', default=None, help='List of prompt styles to test, e.g. --styles strict relaxed minimal')
    g.add_argument('--top-k-list', type=int, nargs='+', default=None, help='List of top_k values to test, e.g. --top-k-list 1 3 5')
    g.add_argument('--limit', type=int, default=None, help='Limit number of test queries for quick testing')
    g.add_argument('--temperature', type=float, default=None, help='Override temperature (otherwise per-prompt-style)')
    g.add_argument('--max-tokens', type=int, default=None, help='Override max tokens (otherwise per-prompt-style)')
    return parser