from __future__ import annotations

import pytest

from analyst.config.domain import REGIONS, get_region


def test_nz_rules():
    nz = get_region("NZ")
    assert nz.name == "New Zealand"
    assert nz.day_first is True
    assert nz.seasons["summer"] == (12, 1, 2)
    assert nz.seasons["winter"] == (6, 7, 8)


def test_each_region_assigns_every_month_to_exactly_one_season():
    for region in REGIONS.values():
        months = sorted(m for season in region.seasons.values() for m in season)
        assert months == list(range(1, 13)), region.code


def test_unknown_region_is_rejected_with_the_known_list():
    with pytest.raises(ValueError, match="known regions: NZ"):
        get_region("XX")


def test_regions_cannot_be_modified():
    with pytest.raises(TypeError):
        REGIONS["XX"] = REGIONS["NZ"]  # type: ignore[index]
    with pytest.raises(TypeError):
        REGIONS["NZ"].seasons["summer"] = (1,)  # type: ignore[index]
