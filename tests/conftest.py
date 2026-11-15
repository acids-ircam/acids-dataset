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


BEAT_TRACKING_TESTS = {"test_beat_tracking"}


@pytest.fixture(autouse=True)
def skip_if_beat_checkpoint_unavailable(request):
    """beat tracking needs a pretrained checkpoint, downloaded on first use : skipped if the download fails (e.g. offline)."""
    if request.node.originalname not in BEAT_TRACKING_TESTS:
        return
    from acids_dataset.features.beat_this.inference import load_checkpoint
    try:
        load_checkpoint("final0")
    except ValueError as e:
        pytest.skip("beat tracking checkpoint unavailable : %s" % (e.__cause__ or e))


@pytest.fixture(autouse=True)
def clear_gin_bindings():
    """some tests parse gin configs globally : bindings (e.g. of datasets, through presets) are cleared after each test, 
    such that they do not leak into other tests. Constants are kept, as tests set them in different ways."""
    yield
    with gin.unlock_config():
        gin.clear_config(clear_constants=False)
