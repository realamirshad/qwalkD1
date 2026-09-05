"""One evaluation function for every feature space.

Same model, same split, same seed, same metrics throughout; the only thing that
varies between rows of the results table is the space transformer.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qw.config import ExperimentConfig
from qw.data import Dataset
from qw.spaces import SPACES, build_space

#: Classifier keys accepted by :func:`make_model`.
MODELS: tuple[str, ...] = ("logreg",)


@dataclass(frozen=True)
class Metrics:
    """Test-set scores for one feature space."""

    accuracy: float
    f1: float
    precision: float
    recall: float

    def as_row(self) -> tuple[float, ...]:
        return (self.accuracy, self.f1, self.precision, self.recall)


@dataclass(frozen=True)
class SpaceResult:
    """Everything one space produced, including the fitted pipeline."""

    space: str
    metrics: Metrics
    n_features: int
    pipeline: Pipeline


def make_model(name: str, seed: int) -> Pipeline:
    """Build the classifier: a scaler and an estimator in a ``Pipeline``.

    Args:
        name: One of :data:`MODELS`.
        seed: Classifier seed.

    Returns:
        An unfitted ``Pipeline``.
    """
    if name == "logreg":
        clf = LogisticRegression(max_iter=5000, random_state=seed)
    else:
        raise KeyError(f"unknown model {name!r}; available: {list(MODELS)}")
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def evaluate_space(space: str, data: Dataset, cfg: ExperimentConfig) -> SpaceResult:
    """Fit on train, score on test, for exactly one feature space.

    The space transformer and the classifier are fitted together inside one
    ``Pipeline``, so ``data.X_test`` only ever reaches ``transform``/``predict``.

    Args:
        space: One of :data:`~qw.spaces.SPACES`.
        data: The shared stratified split.
        cfg: Experiment configuration.

    Returns:
        A :class:`SpaceResult` with test metrics and the fitted pipeline.
    """
    pipeline = Pipeline(
        [
            ("space", build_space(space, cfg)),
            ("model", make_model(cfg.model, cfg.seed)),
        ]
    )
    pipeline.fit(data.X_train, data.y_train)
    y_pred = pipeline.predict(data.X_test)

    metrics = Metrics(
        accuracy=float(accuracy_score(data.y_test, y_pred)),
        f1=float(f1_score(data.y_test, y_pred)),
        precision=float(precision_score(data.y_test, y_pred, zero_division=0)),
        recall=float(recall_score(data.y_test, y_pred, zero_division=0)),
    )
    n_features = int(
        pipeline.named_steps["space"].transform(data.X_train[:1]).shape[1]
    )
    return SpaceResult(
        space=space, metrics=metrics, n_features=n_features, pipeline=pipeline
    )


def evaluate_all(data: Dataset, cfg: ExperimentConfig) -> list[SpaceResult]:
    """Run :func:`evaluate_space` for every space in :data:`~qw.spaces.SPACES`."""
    return [evaluate_space(space, data, cfg) for space in SPACES]


def format_results_table(results: list[SpaceResult]) -> str:
    """Render results as a fixed-width table, best accuracy marked with ``*``."""
    header = f"{'space':<14}{'dims':>6}{'accuracy':>11}{'f1':>9}{'precision':>11}{'recall':>9}"
    lines = [header, "-" * len(header)]
    best = max(r.metrics.accuracy for r in results) if results else 0.0
    for r in results:
        mark = " *" if r.metrics.accuracy == best else ""
        lines.append(
            f"{r.space:<14}{r.n_features:>6}"
            f"{r.metrics.accuracy:>11.4f}{r.metrics.f1:>9.4f}"
            f"{r.metrics.precision:>11.4f}{r.metrics.recall:>9.4f}{mark}"
        )
    return "\n".join(lines)
