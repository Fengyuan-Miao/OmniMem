"""Unified OmniMem command-line interface."""

from __future__ import annotations

import argparse
import sys
from importlib import import_module
from typing import List, Optional


RUNNERS = {
    "gme": "memgallery_gme_pipeline",
    "dual": "memgallery_dual_encoder_pipeline",
    "topic": "topic_memory.memgallery_topic_pipeline",
    "svi": "memgallery_svi_pipeline",
    "opd": "opd_mm_baseline.memgallery_pipeline",
    "opd-interactive": "opd_mm_baseline.memgallery_interactive_pipeline",
    "opd-online": "opd_mm_baseline.memgallery_online_pipeline",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnimem",
        description="OmniMem multimodal memory experiments and benchmarks.",
    )
    parser.add_argument("--version", action="version", version="OmniMem 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)
    benchmark = commands.add_parser("benchmark", help="Run a benchmark pipeline.")
    benchmarks = benchmark.add_subparsers(dest="benchmark", required=True)
    memgallery = benchmarks.add_parser(
        "memgallery",
        help="Run a Mem-Gallery evaluation.",
    )
    memgallery.add_argument("method", choices=sorted(RUNNERS))
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if (
        len(raw_args) >= 3
        and raw_args[0] == "benchmark"
        and raw_args[1] == "memgallery"
        and raw_args[2] in RUNNERS
    ):
        runner = import_module(RUNNERS[raw_args[2]])
        runner.main(raw_args[3:])
        return

    parser = build_parser()
    args = parser.parse_args(raw_args)
    if args.command != "benchmark" or args.benchmark != "memgallery":
        parser.error("unsupported command")
    runner = import_module(RUNNERS[args.method])
    runner.main([])
