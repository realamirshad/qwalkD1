"""Walk registry -- the single place a new walk variant is wired in.

Adding the 2D walk means writing ``qw/walk2d.py`` (10 parameters, two coins,
mu/chi entangled initial state) and adding one line here.  No other module
needs to change.
"""

from __future__ import annotations

from qw.walk1d import WALK_1D
from qw.walkspec import WalkSpec

WALKS: dict[str, WalkSpec] = {
    WALK_1D.name: WALK_1D,
    # WALK_2D.name: WALK_2D,
}


def get_walk(name: str) -> WalkSpec:
    """Look up a walk variant by name, e.g. ``"1d"``."""
    try:
        return WALKS[name]
    except KeyError:
        raise KeyError(
            f"unknown walk {name!r}; available: {sorted(WALKS)}"
        ) from None
