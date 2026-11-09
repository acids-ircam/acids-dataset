import sys
import re
import yaml
import os
import statistics
from collections import Counter
from absl import flags, app
from pprint import pprint
from pathlib import Path

import sys
import sys; sys.path.append(str(Path(__file__).parent.parent))
from acids_dataset import get_loader_from_path, get_metadata_from_path, get_fragment_class_from_path

def import_flags():
    flags.DEFINE_string('path', None, 'dataset path (pre-processed dataset, or raw audio folder)', required=True)
    flags.DEFINE_boolean('files', False, 'list files within dataset')
    flags.DEFINE_boolean('check_metadata', False, 'check metadata discrepencies in dataset')
    flags.DEFINE_multi_integer('idx', [], 'parses fragment with idx %s (file index for raw audio folders)')
    flags.DEFINE_multi_string('key', [], 'parses fragment with key %s (audio path for raw audio folders)')
    flags.DEFINE_boolean('health', False, 'health report of files (clipping, DC offset, silence, level)')
    flags.DEFINE_integer('health_max_files', None, 'maximum number of files of the health report')
    # raw audio folders
    flags.DEFINE_boolean('write_index', None, 'raw audio folders : write the .ad index file (asked if not given)')
    flags.DEFINE_string('index_dir', None, 'raw audio folders : folder of index files (default : in the dataset folder)')
    flags.DEFINE_multi_string('meta_regexp', [], 'raw audio folders : regexp for metadata parsing')
    flags.DEFINE_multi_string('filter', [], 'raw audio folders : wildcard to filter target files')
    flags.DEFINE_multi_string('exclude', [], 'raw audio folders : wildcard to exclude target files')
    flags.DEFINE_boolean('rescan', False, 'raw audio folders : ignore existing index and re-scan every file')


separator = '-'*10


def print_health(dataset, max_files=None):
    """prints the health report of a dataset (see acids_dataset.health) : summary and flagged files."""
    from acids_dataset.health import dataset_health
    report = dataset_health(dataset, max_files=max_files, progress=True)
    print(separator)
    print('health')
    pprint(report['summary'], sort_dicts=False)
    flagged = [r for r in report['files'] if r['flags']]
    if flagged:
        print('flagged files :')
        for r in flagged:
            print(f"{r['audio_path']}: {', '.join(r['flags'])} (peak {r['peak_db']:.1f} dB, clipping {r['clipping_ratio']:.2e}, "
                  f"dc {r['dc_offset']:.3f}, silence {r['silence_ratio']:.2f})")
    else:
        print('no flagged file.')
    return report


def ask_yes_no(question: str) -> bool:
    out = None
    while out is None:
        answer = input(question + ' (y/n): ').strip().lower()
        if answer in ['y', 'yes']:
            out = True
        elif answer in ['n', 'no']:
            out = False
    return out


def resolve_write_index(dataset_path: Path, index_path: Path, write_index: bool | None) -> bool:
    """decides whether the index file of a raw audio folder is written. Existing indexes are updated ; otherwise, the
    user is asked (if possible), else the folder is only scanned in memory."""
    if write_index is not None:
        return write_index
    if index_path.exists():
        return True
    if not sys.stdin.isatty():
        print(f'no index found for {dataset_path} ; scanning without writing index (use --write_index to write it)')
        return False
    print(f'no index found for {dataset_path}. The index caches file information, such that next scans are faster.')
    return ask_yes_no(f'write index in {index_path}? (otherwise, the folder is only scanned in memory)')


def _format_duration(seconds: float) -> str:
    hours, rem = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}h{minutes:02d}m{secs:02d}s" if hours else f"{minutes}m{secs:02d}s"


def online_summary(dataset) -> dict:
    """summary of a scanned raw audio folder."""
    files = dataset.files
    durations = [f['duration'] for f in files]
    summary = {
        'paths': dataset.path,
        'n_files': len(files),
        'n_seconds': sum(durations),
        'duration': _format_duration(sum(durations)),
        'file_durations': {'min': min(durations), 'median': statistics.median(durations), 'max': max(durations)},
        'sample_rates': dict(Counter(f['sr'] for f in files)),
        'channels': dict(Counter(f['channels'] for f in files)),
        'codecs': dict(Counter(f['codec'] for f in files)),
        'invalid_files': dataset.invalid_files,
    }
    if dataset._meta_regexp:
        summary['meta_regexp'] = dataset._meta_regexp
        summary['metadata'] = {k: len(v) for k, v in dataset.feature_hash.items() if k != 'original_path'}
    return summary


def file_description(file: dict) -> str:
    desc = f"{file['audio_path']}\n\tpath: {file['path']}\n\tduration: {file['duration']:.3f}s ({file['n_frames']} samples)"
    desc += f"\n\tsr: {file['sr']}\n\tchannels: {file['channels']}\n\tcodec: {file['codec']}"
    for k, v in file['metadata'].items():
        desc += f"\n\t{k}: {v}"
    return desc


def info_online(path, write_index=None, index_dir=None, meta_regexp=[], filters=[], exclude=[], rescan=False, files=False, check_metadata=False, idx=[], key=[]):
    """prints information about a raw (not pre-processed) audio folder, scanned as an online dataset."""
    from acids_dataset.datasets import OnlineAudioDataset
    paths = [Path(p) for p in (path if isinstance(path, (list, tuple)) else [path])]
    index_paths = [OnlineAudioDataset.index_path_for(p, index_dir) for p in paths]
    write_index = all(resolve_write_index(p, i, write_index) for p, i in zip(paths, index_paths))
    dataset = OnlineAudioDataset(paths, index_dir=index_dir, meta_regexp=meta_regexp, filters=filters, exclude=exclude,
                                 rescan=rescan, write_index=write_index)
    print('raw audio folder (not pre-processed) : scanned as an online dataset')
    print('index : ' + (', '.join(map(str, index_paths)) if write_index else 'not written (scanned in memory)'))
    print(separator)
    print('metadata')
    pprint(online_summary(dataset), sort_dicts=False)
    if files:
        print(separator)
        print('files : ')
        for f in dataset.files:
            print(f"{f['audio_path']}: {f['duration']:.3f}s, sr={f['sr']}, channels={f['channels']}" +
                  "".join(f", {k}={v}" for k, v in f['metadata'].items()))
    if check_metadata and dataset._meta_regexp:
        missing_metadata = {f['audio_path']: [k for k, v in f['metadata'].items() if v is None] for f in dataset.files}
        missing_metadata = {k: v for k, v in missing_metadata.items() if len(v) > 0}
        if len(missing_metadata) == 0:
            print('-------\nNo metadata discrepencies.')
        else:
            print('-------\n[WARNING] Found metadata discrepencies :')
            for k, v in missing_metadata.items():
                print(f"{k}: {v} missing")
    file_map = {f['audio_path']: i for i, f in enumerate(dataset.files)}
    for i in idx:
        print('\n--INDEX %d :'%i)
        if 0 <= i < len(dataset.files):
            print(file_description(dataset.files[i]))
        else:
            print('index %d not in dataset'%i)
    for k in key:
        print('\n--KEY %s : '%k)
        if k in file_map:
            print(file_description(dataset.files[file_map[k]]))
        else:
            print('key %s not in dataset'%k)
    return dataset


def info_preprocessed(path, files=False, check_metadata=False, idx=[], key=[]):
    """prints information about a pre-processed dataset."""
    import lmdb
    dataset_path = Path(path)
    loader = get_loader_from_path(path)
    env = lmdb.open(str(dataset_path), readonly=True, lock=False)
    print('environment parameters : ')
    pprint(env.stat())
    env.close()
    metadata = get_metadata_from_path(path)
    print(separator)
    print('metadata')
    pprint(metadata)
    if files:
        feature_hash = loader.feature_hash
        print(separator)
        print('files : ')
        for f, f_idx in feature_hash['original_path'].items():
            print(f'{re.sub(f"^{dataset_path}", "", f)}: {len(f_idx)} chunks')
        if check_metadata:
            missing_metadata = {}
            metadata_keys = list(metadata['features'].keys())
            for f_key, ae in loader.iter_fragments():
                missing = [k for k in metadata_keys if not (ae.has_buffer(k) or ae.has_metadata(k))]
                if len(missing) > 0:
                    missing_metadata[ae.get_metadata().get('audio_path', f_key)] = missing
            if len(missing_metadata) == 0:
                print('-------\nNo metadata discrepencies.')
            else:
                print('-------\n[WARNING] Found metadata discrepencies :')
                for k, v in missing_metadata.items():
                    print(f"{k}: {v} missing")
    for i in idx : 
        print('\n--INDEX %d :'%i)
        try: 
            print(loader[i].description)
        except IndexError:
            print('index %d not in dataset'%i)
    for k in key:
        print('\n--KEY %s : '%k)
        try:
            print(loader[k].description)
        except Exception:
            print('key %s not in dataset'%k)


def main(argv):
    FLAGS = flags.FLAGS
    from acids_dataset.datasets import is_preprocessed_path
    if is_preprocessed_path(FLAGS.path):
        info_preprocessed(FLAGS.path, files=FLAGS.files, check_metadata=FLAGS.check_metadata, idx=FLAGS.idx, key=FLAGS.key)
        if FLAGS.health:
            from acids_dataset.datasets import AudioDataset
            print_health(AudioDataset(FLAGS.path), max_files=FLAGS.health_max_files)
    else:
        dataset = info_online(FLAGS.path, write_index=FLAGS.write_index, index_dir=FLAGS.index_dir, meta_regexp=FLAGS.meta_regexp,
                              filters=FLAGS.filter, exclude=FLAGS.exclude, rescan=FLAGS.rescan, files=FLAGS.files,
                              check_metadata=FLAGS.check_metadata, idx=FLAGS.idx, key=FLAGS.key)
        if FLAGS.health:
            print_health(dataset, max_files=FLAGS.health_max_files)


if __name__ == "__main__":
    import_flags()
    app.run(main)
