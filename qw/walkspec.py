"""The seam between walk variants.

Everything that differs between the 1D walk (4 parameters, one coin) and the
2D walk (10 parameters, two coins, entangled initial state) is captured by a
:class:`WalkSpec`.  The rest of the pipeline -- selection, mapping, spaces,
evaluation, CLI -- is written against this object and never against a literal
parameter count, so adding 2D means adding a module plus one registry entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

ParamRange = tuple[float, float]

RunFn = Callable[[np.ndarray, int], np.ndarray]
ExtractFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class WalkSpec:
    """Static description of one quantum-walk variant.

    Attributes:
        name: Registry key, e.g. ``"1d"``.
        param_names: Walk parameters in canonical order, e.g.
            ``("theta", "phi", "alpha", "beta")``.  Its length is both the
            number of walk parameters and the number of features to select.
        param_ranges: Closed interval each parameter is scaled into, aligned
            with ``param_names``.
        n_features_out: Width of the matrix returned by ``extract``.
        run: ``(params, n_steps) -> psi``.  ``params`` has shape
            ``(n_samples, n_params)`` with columns ordered as ``param_names``.
        extract: ``psi -> (n_samples, n_features_out)``.
    """

    name: str
    param_names: tuple[str, ...]
    param_ranges: tuple[ParamRange, ...]
    n_features_out: int
    run: RunFn
    extract: ExtractFn

    def __post_init__(self) -> None:
        if len(self.param_names) != len(self.param_ranges):
            raise ValueError(
                f"walk {self.name!r}: {len(self.param_names)} names but "
                f"{len(self.param_ranges)} ranges"
            )

    @property
    def n_params(self) -> int:
        """Number of walk parameters, i.e. the number of features to select."""
        return len(self.param_names)

    def scale_to_ranges(self, unit: np.ndarray) -> np.ndarray:
        """Map columns in ``[0, 1]`` onto ``param_ranges``.

        Args:
            unit: Shape ``(n_samples, n_params)``, each column in ``[0, 1]``.

        Returns:
            Shape ``(n_samples, n_params)`` in physical parameter units.
        """
        if unit.shape[1] != self.n_params:
            raise ValueError(
                f"walk {self.name!r} expects {self.n_params} columns, got {unit.shape[1]}"
            )
        lo = np.array([r[0] for r in self.param_ranges], dtype=float)
        hi = np.array([r[1] for r in self.param_ranges], dtype=float)
        return lo + unit * (hi - lo)

    def describe_permutation(self, permutation: Sequence[int]) -> str:
        """Render a permutation as ``theta<-f2 phi<-f0 ...`` for logging."""
        return " ".join(
            f"{name}<-f{src}" for name, src in zip(self.param_names, permutation)
        )

    def validate_permutation(self, permutation: Sequence[int]) -> tuple[int, ...]:
        """Check ``permutation`` is a bijection on ``range(n_params)``."""
        perm = tuple(int(i) for i in permutation)
        if sorted(perm) != list(range(self.n_params)):
            raise ValueError(
                f"walk {self.name!r}: permutation must be a rearrangement of "
                f"{list(range(self.n_params))}, got {list(perm)}"
            )
        return perm
