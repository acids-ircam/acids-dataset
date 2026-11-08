"""Gradio interface of the dataset browser."""
import json
import random
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import gradio as gr
except ImportError as e:
    raise ImportError("the dataset browser requires gradio : pip install acids-dataset[ui]") from e

from .backend import DatasetBrowser

HEALTH_HEADERS = ["index", "audio path", "flags", "peak (dB)", "rms (dB)", "clipping ratio", "dc offset", "silence ratio"]
SAMPLE_SIZES = {"100 items": 100, "1000 items": 1000, "whole dataset": None}


# conversions & plots

def to_gradio_audio(sr: int, audio: np.ndarray):
    """(channels, samples) float audio to gradio format (sr, int16 array of shape (samples, channels))."""
    if audio is None:
        return None
    audio = np.asarray(audio, dtype=np.float32)
    audio = audio.reshape(-1, audio.shape[-1])
    peak = np.abs(audio).max()
    if peak > 1: audio = audio / peak
    return sr, (np.clip(audio, -1, 1) * 32767).astype(np.int16).T


def _plot_waveform(ax, audio, sr, title=None):
    t = np.arange(audio.shape[-1]) / sr
    for c in range(audio.shape[0]):
        ax.plot(t, audio[c], linewidth=0.5, alpha=0.8, label="channel %d"%c)
    ax.set_xlim(0, t[-1] if len(t) else 1)
    ax.set_xlabel("time (s)")
    if title: ax.set_title(title)


def _plot_spectrogram(ax, audio, sr, title=None):
    mono = audio.mean(0)
    n_fft = 1024 if mono.shape[-1] >= 1024 else max(16, 2 ** int(np.log2(max(mono.shape[-1], 2))))
    ax.specgram(mono + 1e-9, NFFT=n_fft, Fs=sr, noverlap=n_fft // 2, cmap="magma")
    ax.set_ylabel("frequency (Hz)")
    ax.set_xlabel("time (s)")
    if title: ax.set_title(title)


def audio_figure(audio, sr, features=None):
    """waveform, spectrogram, and 1d / 2d features of an item."""
    features = {k: v for k, v in (features or {}).items() if isinstance(v, np.ndarray) and v.size > 1}
    n_rows = 2 + len(features)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 2.2 * n_rows), squeeze=False)
    axes = axes[:, 0]
    if audio is not None:
        _plot_waveform(axes[0], audio, sr, "waveform")
        _plot_spectrogram(axes[1], audio, sr, "spectrogram")
    for ax, (name, value) in zip(axes[2:], features.items()):
        value = np.squeeze(value)
        if value.ndim == 1:
            ax.plot(value)
        else:
            value = value.reshape(-1, value.shape[-1]).astype(np.float64)
            if np.nanmin(value) >= 0:
                # spectrogram-like features are displayed in decibels
                value = 10 * np.log10(np.maximum(value, 1e-10))
                value = np.maximum(value, np.nanmax(value) - 80)
            ax.imshow(value, aspect="auto", origin="lower", interpolation="nearest", cmap="magma")
        ax.set_title(name)
    fig.tight_layout()
    return fig


def comparison_figure(original, augmented, sr):
    fig, axes = plt.subplots(2, 2, figsize=(12, 5))
    _plot_waveform(axes[0, 0], original, sr, "original")
    _plot_waveform(axes[0, 1], augmented, sr, "augmented")
    _plot_spectrogram(axes[1, 0], original, sr)
    _plot_spectrogram(axes[1, 1], augmented, sr)
    fig.tight_layout()
    return fig


def histogram_figure(values, title):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    if len(values) > 0:
        ax.hist(values, bins=100)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def counts_figure(counts_by_group: dict, title: str):
    """bar plot of value counts, for one or several groups (e.g. partitions)."""
    values = sorted({v for counts in counts_by_group.values() for v in counts})
    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(values) * max(1, len(counts_by_group))), 3.5))
    width = 0.8 / max(1, len(counts_by_group))
    for i, (group, counts) in enumerate(counts_by_group.items()):
        ax.bar(np.arange(len(values)) + i * width, [counts.get(v, 0) for v in values], width=width, label=group)
    ax.set_xticks(np.arange(len(values)) + 0.4 - width / 2)
    ax.set_xticklabels(values, rotation=45, ha="right")
    ax.set_ylabel("items")
    ax.set_title(title)
    if len(counts_by_group) > 1: ax.legend()
    fig.tight_layout()
    return fig


def _feature_summary(features):
    summary = {}
    for name, value in features.items():
        if isinstance(value, np.ndarray):
            finite = value[np.isfinite(value)] if np.issubdtype(value.dtype, np.number) else value
            summary[name] = {'shape': list(value.shape), 'dtype': str(value.dtype)}
            if np.issubdtype(value.dtype, np.number) and finite.size:
                summary[name].update({'mean': float(finite.mean()), 'min': float(finite.min()), 'max': float(finite.max())})
            if value.size <= 8:
                summary[name]['value'] = value.tolist()
        else:
            summary[name] = value
    return summary


# application

def build_app(browser: DatasetBrowser) -> "gr.Blocks":
    summary = browser.summary()
    n_items, n_files = summary['n_items'], summary['n_files']

    with gr.Blocks(title="acids-dataset browser") as app:
        gr.Markdown("## acids-dataset browser\n`%s` (%s, %d items, %d files)"%(summary['path'], summary['mode'], n_items, n_files))

        # overview
        with gr.Tab("Overview"):
            gr.JSON(value=summary, label="dataset")
            gr.Dataframe(value=browser.file_table(), headers=["index", "audio path", "duration (s)", "sr", "channels", "chunks" if not browser.is_online else "metadata"],
                         label="files", interactive=False, wrap=True)
            with gr.Accordion("Health report", open=False):
                gr.Markdown("Clipping, DC offset, silence and level of files (audio of pre-processed datasets is rebuilt from stored chunks).")
                with gr.Row():
                    health_max_files = gr.Number(value=None, precision=0, label="maximum number of files (empty for all)")
                    health_only_flagged = gr.Checkbox(value=True, label="only flagged files")
                    health_button = gr.Button("compute")
                health_summary = gr.JSON(label="summary")
                health_table = gr.Dataframe(headers=HEALTH_HEADERS, label="files", interactive=False, wrap=True)

        def compute_health(max_files, only_flagged):
            report = browser.health(max_files=None if not max_files else int(max_files))
            level = lambda v: None if v is None else round(v, 2)
            rows = [[r['index'], r['audio_path'], ", ".join(r['flags']), level(r['peak_db']), level(r['rms_db']), 
                     r['clipping_ratio'], round(r['dc_offset'], 4), round(r['silence_ratio'], 3)] 
                    for r in report['files'] if r['flags'] or not only_flagged]
            return report['summary'], rows
        health_button.click(compute_health, [health_max_files, health_only_flagged], [health_summary, health_table], api_name="health")

        # browse items and files
        with gr.Tab("Browse"):
            with gr.Tab("Items"):
                with gr.Row():
                    item_index = gr.Number(value=0, precision=0, label="item index (0 - %d)"%(n_items - 1), minimum=0, maximum=n_items - 1)
                    previous_button, next_button, random_button = gr.Button("previous"), gr.Button("next"), gr.Button("random")
                item_audio = gr.Audio(label="audio", type="numpy")
                item_plot = gr.Plot(label="audio and features")
                item_info = gr.JSON(label="metadata and features")
            with gr.Tab("Files"):
                with gr.Row():
                    file_index = gr.Number(value=0, precision=0, label="file index (0 - %d)"%(n_files - 1), minimum=0, maximum=n_files - 1)
                    file_start = gr.Number(value=0., label="start (s)")
                    file_length = gr.Number(value=10., label="length (s, 0 for whole file)")
                    file_button = gr.Button("load")
                file_audio = gr.Audio(label="audio", type="numpy")
                file_plot = gr.Plot(label="audio")

        def show_item(index):
            item = browser.item(int(index or 0))
            info = {'metadata': item['metadata'], 'features': _feature_summary(item['features'])}
            return item['index'], to_gradio_audio(item['sr'], item['audio']), audio_figure(item['audio'], item['sr'], item['features']), info

        item_outputs = [item_index, item_audio, item_plot, item_info]
        item_index.submit(show_item, item_index, item_outputs)
        previous_button.click(lambda i: show_item((int(i or 0) - 1) % n_items), item_index, item_outputs)
        next_button.click(lambda i: show_item((int(i or 0) + 1) % n_items), item_index, item_outputs)
        random_button.click(lambda: show_item(random.randrange(n_items)), None, item_outputs)

        def show_file(index, start, length):
            sr, audio = browser.file_audio(int(index or 0), start or 0., length or None)
            return to_gradio_audio(sr, audio), audio_figure(audio, sr)
        file_button.click(show_file, [file_index, file_start, file_length], [file_audio, file_plot])

        # statistics
        with gr.Tab("Statistics"):
            with gr.Tab("Numeric features"):
                gr.Markdown("Statistics of features over random items, or over the whole dataset." +
                            (" Features of online datasets are computed on loaded audio : statistics over the whole dataset may take a while." if browser.is_online else ""))
                with gr.Row():
                    numeric_feature = gr.Dropdown(choices=browser.array_feature_names, value=(browser.array_feature_names or [None])[0], label="feature")
                    sample_size = gr.Radio(choices=list(SAMPLE_SIZES.keys()), value="100 items", label="computed on")
                    stats_button = gr.Button("compute")
                stats_json = gr.JSON(label="statistics")
                stats_plot = gr.Plot(label="distribution")
            with gr.Tab("Categorical features"):
                with gr.Row():
                    categorical_feature = gr.Dropdown(choices=browser.categorical_feature_names, value=(browser.categorical_feature_names or [None])[0], label="feature")
                    counts_button = gr.Button("show")
                counts_plot = gr.Plot(label="items per value")
                counts_json = gr.JSON(label="items per value")

        def compute_stats(feature, size, progress=gr.Progress(track_tqdm=False)):
            if not feature:
                return {"error": "no numeric feature in dataset"}, None
            result = browser.feature_stats(feature, n_items=SAMPLE_SIZES[size])
            return result['stats'], histogram_figure(result['values'], feature)
        stats_button.click(compute_stats, [numeric_feature, sample_size], [stats_json, stats_plot])

        def show_counts(feature):
            if not feature:
                return None, {"error": "no categorical feature in dataset"}
            counts = browser.categorical_counts(feature)
            return counts_figure({"dataset": counts}, feature), counts
        counts_button.click(show_counts, categorical_feature, [counts_plot, counts_json])

        # augmentations
        with gr.Tab("Augmentations"):
            gr.Markdown("Applies transforms (configured in `acids_dataset/configs/transforms`) to an item, in the given order.")
            with gr.Row():
                augment_index = gr.Number(value=0, precision=0, label="item index", minimum=0, maximum=n_items - 1)
                augment_transforms = gr.Dropdown(choices=browser.available_transforms(), multiselect=True, label="transforms")
                augment_force = gr.Checkbox(value=True, label="always apply (otherwise, with transform probabilities)")
                augment_button = gr.Button("apply")
            with gr.Row():
                original_audio = gr.Audio(label="original", type="numpy")
                augmented_audio = gr.Audio(label="augmented", type="numpy")
            augment_errors = gr.Markdown()
            augment_plot = gr.Plot(label="original / augmented")

        def apply_augmentations(index, transforms, force):
            item = browser.item(int(index or 0))
            if item['audio'] is None:
                return None, None, "dataset has no waveform", None
            augmented, errors = browser.augment(item['audio'], transforms or [], force=force)
            message = "\n".join("- **could not apply** %s"%e for e in errors)
            return (to_gradio_audio(item['sr'], item['audio']), to_gradio_audio(item['sr'], augmented), message,
                    comparison_figure(item['audio'], augmented, item['sr']))
        augment_button.click(apply_augmentations, [augment_index, augment_transforms, augment_force],
                             [original_audio, augmented_audio, augment_errors, augment_plot])

        # partitions
        with gr.Tab("Partitions"):
            partition_names = browser.list_partitions()
            with gr.Tab("View"):
                with gr.Row():
                    partition_name = gr.Dropdown(choices=partition_names, value=(partition_names or [None])[0], label="partition")
                    refresh_button = gr.Button("refresh list")
                    partition_button = gr.Button("show")
            with gr.Tab("Create"):
                gr.Markdown("Splits the dataset with `dataset.split`. Random partitions split items ; partitions by features keep "
                            "each feature value in a single partition (`original_path` keeps chunks of a file together).")
                with gr.Row():
                    create_ratios = gr.Textbox(value="train: 0.8, valid: 0.2", label="partitions (name: ratio, ...)")
                    create_features = gr.Dropdown(choices=browser.split_features, multiselect=True, value=["original_path"], 
                                                  label="group by features (empty for random items)")
                    create_seed = gr.Number(value=0, precision=0, label="seed")
                with gr.Row():
                    create_balance = gr.Checkbox(value=False, label="balance partition sizes (with features)")
                    create_tolerance = gr.Slider(minimum=0., maximum=1., value=0.1, step=0.05, label="tolerance of balanced partitions")
                    create_use_feature = gr.Checkbox(value=False, label="use 'partition' feature if available")
                with gr.Row():
                    create_name = gr.Textbox(value="", label="name (to write the partition)")
                    create_overwrite = gr.Checkbox(value=False, label="overwrite")
                    preview_button = gr.Button("preview")
                    create_button = gr.Button("create", variant="primary")
                create_status = gr.Markdown()
            partition_sizes = gr.JSON(label="sizes")
            partition_warning = gr.Markdown()
            partition_plot = gr.Plot(label="feature distributions")

        refresh_button.click(lambda: gr.Dropdown(choices=browser.list_partitions()), None, partition_name)

        def show_report(report):
            if report['shared_files']:
                warning = "**Warning** : files are shared between partitions (items of a same file may leak between them)\n\n" + \
                          "\n".join("- %s : %d files"%(k, len(v)) for k, v in report['shared_files'].items())
            else:
                warning = "No file is shared between partitions."
            figure = None
            if report['distributions']:
                feature = list(report['distributions'].keys())[0]
                figure = counts_figure(report['distributions'][feature], "%s per partition"%feature)
            return report['sizes'], warning, figure

        def show_partition(name):
            if not name:
                return {"error": "no partition found ; partitions can be created in the Create tab"}, "", None
            return show_report(browser.partition_report(name))
        partition_button.click(show_partition, partition_name, [partition_sizes, partition_warning, partition_plot])

        def create_partition(ratios, features, seed, balance, tolerance, use_feature, name, overwrite, write):
            try:
                report = browser.create_partition(ratios, features=features, balance_cardinality=balance, tolerance=tolerance, 
                                                  seed=None if seed is None else int(seed), use_partition_feature=use_feature, 
                                                  name=name if write else None, overwrite=overwrite)
            except (ValueError, FileExistsError, AssertionError) as e:
                return "**Error** : %s"%e, gr.Dropdown(), {}, "", None
            sizes, warning, figure = show_report(report)
            if write:
                status = "Partition **%s** written."%name.strip()
                dropdown = gr.Dropdown(choices=browser.list_partitions(), value=name.strip())
            else:
                status = "Preview (not written)."
                dropdown = gr.Dropdown()
            return status, dropdown, sizes, warning, figure

        create_inputs = [create_ratios, create_features, create_seed, create_balance, create_tolerance, create_use_feature, create_name, create_overwrite]
        create_outputs = [create_status, partition_name, partition_sizes, partition_warning, partition_plot]
        preview_button.click(lambda *args: create_partition(*args, write=False), create_inputs, create_outputs, api_name="preview_partition")
        create_button.click(lambda *args: create_partition(*args, write=True), create_inputs, create_outputs, api_name="create_partition")

        app.load(lambda: show_item(0), None, item_outputs)
    return app


def launch(path, port: int | None = None, share: bool = False, inbrowser: bool = True, **dataset_kwargs):
    """launches the dataset browser for a pre-processed dataset or a raw audio folder (dataset_kwargs are given to
    the dataset, e.g. sr, chunk_length, features, meta_regexp, preset)."""
    browser = DatasetBrowser.from_path(path, **dataset_kwargs)
    app = build_app(browser)
    app.launch(server_port=port, share=share, inbrowser=inbrowser)
    return app
