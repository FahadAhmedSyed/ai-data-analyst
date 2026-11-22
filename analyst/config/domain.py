"""Region rules: date order and seasons (ADR 0009).

Only New Zealand is defined for the MVP. Adding a region means adding one
``RegionRules`` entry; nothing else reads region-specific constants.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True)
class RegionRules:
    """Conventions that change how questions and dates are read."""

    code: str
    name: str
    day_first: bool
    seasons: Mapping[str, tuple[int, ...]] = field(default_factory=dict)


REGIONS: Mapping[str, RegionRules] = MappingProxyType(
    {
        "NZ": RegionRules(
            code="NZ",
            name="New Zealand",
            day_first=True,
            seasons=MappingProxyType(
                {
                    "summer": (12, 1, 2),
                    "autumn": (3, 4, 5),
                    "winter": (6, 7, 8),
                    "spring": (9, 10, 11),
                }
            ),
        ),
    }
)


def get_region(code: str) -> RegionRules:
    """Return the rules for a region code such as ``"NZ"``."""
    try:
        return REGIONS[code]
    except KeyError:
        msg = f"Unknown region {code!r}; known regions: {', '.join(sorted(REGIONS))}"
        raise ValueError(msg) from None
