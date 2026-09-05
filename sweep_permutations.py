#!/usr/bin/env python3
"""Search all 4! permutations by cross-validation on the training set only.

The winning permutation is chosen without ever looking at the test set; the
held-out score is reported once, afterwards, for the winner alone.
"""

from __future__ import annotations

import argparse
import sys
from itertools import permutations

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from qw.config import ExperimentConfig
from qw.data import load_breast_cancer_split
from qw.evaluate import evaluate_space, format_results_table, make_model
from qw.spaces import build_space


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=ExperimentConfig().n_steps)
    p.add_argument("--space", default="original+qw", choices=["qw", "original+qw"])
    p.add_argument("--folds", type=int, default=5)
    args = p.parse_args(argv)

    cfg = ExperimentConfig(n_steps=args.steps)
    data = load_breast_cancer_split(cfg)

    pipeline = Pipeline(
        [("space", build_space(args.space, cfg)), ("model", make_model(cfg.model, cfg.seed))]
    )
    key = "space__permutation" if args.space == "qw" else "space__qw__permutation"

    search = GridSearchCV(
        pipeline,
        param_grid={key: list(permutations(range(cfg.spec.n_params)))},
        scoring="accuracy",
        cv=StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=cfg.seed),
        n_jobs=-1,
    )
    search.fit(data.X_train, data.y_train)

    best = tuple(search.best_params_[key])
    print(f"space={args.space} T={args.steps} folds={args.folds}")
    print(f"best permutation: {list(best)}  [{cfg.spec.describe_permutation(best)}]")
    print(f"cv accuracy:      {search.best_score_:.4f}  (train only)")
    print()

    winner = ExperimentConfig(n_steps=args.steps, permutation=best)
    print(format_results_table([evaluate_space(s, data, winner)
                                for s in ("original", "polynomial", "qw", "original+qw")]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
