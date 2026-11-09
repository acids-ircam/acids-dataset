import gin
import pytest

# generates tests/scripted/conv_embedding.ts (used by several tests) if it does not exist
from tests import module_tests  # noqa: F401

# long parametrized grids, excluded with -m "not slow" (e.g. in continuous integration)
SLOW_TESTS = {"test_raw_parser", "test_pitch_features", "test_update_dataset_features"}


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.originalname in SLOW_TESTS:
            item.add_marker(pytest.mark.slow)


@pytest.fixture(autouse=True)
def clear_gin_bindings():
    """some tests parse gin configs globally : bindings (e.g. of datasets, through presets) are cleared after each test, 
    such that they do not leak into other tests. Constants are kept, as tests set them in different ways."""
    yield
    with gin.unlock_config():
        gin.clear_config(clear_constants=False)
