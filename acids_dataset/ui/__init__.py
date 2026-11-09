"""Dataset browser : a simple web interface to explore pre-processed datasets and raw audio folders (statistics, 
items and files, augmentations, partitions). Requires gradio (pip install acids-dataset[ui]).

    from acids_dataset.ui import launch
    launch("path/to/dataset")
"""
from .backend import DatasetBrowser


def launch(path, port: int | None = None, share: bool = False, inbrowser: bool = True, **dataset_kwargs):
    """launches the dataset browser for a pre-processed dataset or a raw audio folder. dataset_kwargs are given 
    to the dataset (e.g. sr, chunk_length, features, meta_regexp, preset)."""
    from .app import launch as launch_app
    return launch_app(path, port=port, share=share, inbrowser=inbrowser, **dataset_kwargs)


def build_app(browser: DatasetBrowser):
    """builds the gradio interface of a DatasetBrowser, without launching it."""
    from .app import build_app as build
    return build(browser)


__all__ = ['DatasetBrowser', 'launch', 'build_app']
