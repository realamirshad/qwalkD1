#!/usr/bin/env python3
"""CLI entry point: evaluate all four feature spaces for one (T, permutation).

Example:
    python run_experiment.py --steps 30 --permutation 0,1,2,3
"""

from __future__ import annotations

import argparse
import sys

from qw.config import SEED, ExperimentConfig
from qw.data import load_breast_cancer_split
from qw.evaluate import MODELS, evaluate_all, format_results_table
from qw.registry import WALKS
from qw.selection import SELECTORS


def _permutation(text: str) -> tuple[int, ...]:
    """Parse ``"0,1,2,3"`` into a tuple of ints."""
    try:
        return tuple(int(part) for part in text.replace(" ", "").split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers, got {text!r}"
        ) from None


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface. Defaults mirror :class:`ExperimentConfig`."""
    defaults = ExperimentConfig()
    p = argparse.ArgumentParser(
        description="Quantum walk feature expansion vs classical baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--steps", type=int, default=defaults.n_steps, help="walk steps T")
    p.add_argument(
        "--permutation",
        type=_permutation,
        default=defaults.permutation,
        help="feature-to-parameter assignment, e.g. 2,0,3,1",
    )
    p.add_argument("--walk", choices=sorted(WALKS), default=defaults.walk)
    p.add_argument("--selector", choices=SELECTORS, default=defaults.selector)
    p.add_argument("--model", choices=MODELS, default=defaults.model)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--test-size", type=float, default=defaults.test_size)
    p.add_argument("--poly-degree", type=int, default=defaults.poly_degree)
    return p


def main(argv: list[str] | None = None) -> int:
    """Run every feature space once and print the results table."""
    args = build_parser().parse_args(argv)
    cfg = ExperimentConfig(
        seed=args.seed,
        n_steps=args.steps,
        walk=args.walk,
        selector=args.selector,
        model=args.model,
        permutation=args.permutation,
        test_size=args.test_size,
        poly_degree=args.poly_degree,
    )

    data = load_breast_cancer_split(cfg)
    print(f"config: {cfg.summary()}")
    print(f"data:   {data.summary()}")

    results = evaluate_all(data, cfg)

    qw_step = next(
        (r for r in results if r.space == "qw"), None
    )
    if qw_step is not None:
        print(f"walk:   {qw_step.pipeline.named_steps['space'].describe()}")
    print()
    print(format_results_table(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
