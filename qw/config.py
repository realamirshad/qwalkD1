"""One place for every experiment knob.

No module below this one hard-codes a seed, a step count, a parameter count or
a model choice; they all read :class:`ExperimentConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass

from qw.registry import get_walk
from qw.walkspec import WalkSpec

#: The single seed. Everything stochastic derives from it.
SEED = 42


@dataclass(frozen=True)
class ExperimentConfig:
    """Fully determines a run. Frozen so nothing can mutate it mid-experiment.

    Attributes:
        seed: Seed for the split, the selector and the classifier.
        n_steps: Walk steps ``T``.
        walk: Registry key of the walk variant.
        selector: Feature-selection method key (see :mod:`qw.selection`).
        model: Classifier key (see :mod:`qw.evaluate`).
        permutation: ``permutation[i]`` is the index, among the selected
            features in selector order, of the feature feeding walk parameter
            ``i``.  Parameters are ordered as ``spec.param_names``.
        test_size: Fraction held out by the stratified split.
        poly_degree: Degree for the polynomial baseline space.
    """

    seed: int = SEED
    n_steps: int = 30
    walk: str = "1d"
    selector: str = "mutual_info"
    model: str = "logreg"
    permutation: tuple[int, ...] = (0, 1, 2, 3)
    test_size: float = 0.25
    poly_degree: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(self, "permutation", tuple(int(i) for i in self.permutation))
        self.spec.validate_permutation(self.permutation)
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {self.n_steps}")
        if not 0.0 < self.test_size < 1.0:
            raise ValueError(f"test_size must be in (0, 1), got {self.test_size}")

    @property
    def spec(self) -> WalkSpec:
        """The :class:`~qw.walkspec.WalkSpec` this config runs against."""
        return get_walk(self.walk)

    @property
    def n_selected(self) -> int:
        """Number of features to select -- always the walk's parameter count."""
        return self.spec.n_params

    def summary(self) -> str:
        """One-line, log-friendly description of the run."""
        return (
            f"walk={self.walk} T={self.n_steps} k={self.n_selected} "
            f"selector={self.selector} model={self.model} seed={self.seed} "
            f"perm={list(self.permutation)} [{self.spec.describe_permutation(self.permutation)}]"
        )
