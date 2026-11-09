"""python -m acids_dataset.ui --path ... launches the dataset browser (see acids_dataset.ui.cli)."""
from absl import app
from acids_dataset.ui import cli

if __name__ == "__main__":
    cli.import_flags()
    app.run(cli.main)
