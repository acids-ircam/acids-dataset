# Development

Tests are run with [pytest](https://docs.pytest.org) :

```bash
pip install -e ".[ui]" pytest pytest-timeout gradio_client
pytest tests -m "not slow"     # fast suite (~10 minutes)
pytest tests -m slow           # long parametrized grids
```

Long parametrized grids are marked as `slow` in `tests/conftest.py`. `tests/test_transforms.py` has known failures (see `TODO.md`), and is not run in CI.

The continuous integration (`.github/workflows/tests.yml`) runs on each push and pull request :

| Job | Content |
| --- | ------- |
| `python-version` | checks with [vermin](https://github.com/netromdk/vermin) that the code does not use features of Python versions above 3.10 (the minimum declared in `pyproject.toml`) |
| `tests` | fast suite on Python 3.10, 3.11, 3.12 and 3.13, with the latest versions of dependencies |
| `minimum-versions` | fast suite on Python 3.10, with the minimum versions of `numpy`, `scipy`, `scikit-learn` and `librosa` declared in `pyproject.toml` (read from it, and used as pip constraints) |

When raising a minimum version in `pyproject.toml`, the `minimum-versions` job tests the new version without other changes.

The documentation is built with [MkDocs](https://www.mkdocs.org), and published on GitHub Pages by `.github/workflows/docs.yml` :

```bash
pip install -e ".[docs]"    # or pip install -r docs/requirements.txt
mkdocs serve                # local preview at http://127.0.0.1:8000
mkdocs build --strict       # static site in site/
```

The pages of the [documentation](index.md) mirror the sections of the README : keep both in sync when documenting a feature.
