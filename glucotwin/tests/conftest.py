import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import add_targets, build_features, static_features  # noqa: E402
from src.preprocessing import preprocess  # noqa: E402
from src.synthetic_data import generate_cohort  # noqa: E402


@pytest.fixture(scope="session")
def small_tables():
    """A small, fast cohort (8 patients x 4 days) shared by all tests."""
    return generate_cohort(n_patients=8, n_days=4, seed=7)


@pytest.fixture(scope="session")
def small_dataset(small_tables):
    grid, quality, issues = preprocess(small_tables["timeseries"], small_tables["patients"])
    static = static_features(small_tables)
    features = add_targets(build_features(grid, static))
    return dict(grid=grid, quality=quality, issues=issues, static=static, features=features)
