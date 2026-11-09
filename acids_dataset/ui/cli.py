from absl import flags


def import_flags():
    flags.DEFINE_string('path', None, 'dataset path (pre-processed dataset, or raw audio folder)', required=True)
    flags.DEFINE_integer('port', None, 'server port')
    flags.DEFINE_boolean('share', False, 'create a public link')
    flags.DEFINE_boolean('inbrowser', True, 'open the interface in the browser')
    # raw audio folders
    flags.DEFINE_string('preset', None, 'raw audio folders : dataset preset')
    flags.DEFINE_integer('sample_rate', None, 'raw audio folders : sampling rate')
    flags.DEFINE_integer('channels', None, 'raw audio folders : number of channels')
    flags.DEFINE_float('chunk_length', None, 'raw audio folders : chunk length in seconds (default : 1)')
    flags.DEFINE_enum('batch_by', 'auto', ['files', 'dataset', 'auto'], 'raw audio folders : chunking strategy')
    flags.DEFINE_multi_string('feature', [], 'raw audio folders : features computed on loaded audio (config names)')
    flags.DEFINE_multi_string('meta_regexp', [], 'raw audio folders : regexp for metadata parsing')
    flags.DEFINE_multi_string('filter', [], 'raw audio folders : wildcard to filter target files')
    flags.DEFINE_multi_string('exclude', [], 'raw audio folders : wildcard to exclude target files')
    flags.DEFINE_boolean('write_index', None, 'raw audio folders : write the .ad index file (asked if not given)')
    flags.DEFINE_string('index_dir', None, 'raw audio folders : folder of index files')


def main(argv):
    from pathlib import Path
    from .. import info
    from ..datasets import is_preprocessed_path, OnlineAudioDataset
    from . import launch
    FLAGS = flags.FLAGS
    kwargs = {}
    if not is_preprocessed_path(FLAGS.path):
        index_path = OnlineAudioDataset.index_path_for(FLAGS.path, FLAGS.index_dir)
        kwargs = dict(write_index=info.resolve_write_index(Path(FLAGS.path), index_path, FLAGS.write_index), 
                      index_dir=FLAGS.index_dir, batch_by=FLAGS.batch_by, features=FLAGS.feature, 
                      meta_regexp=FLAGS.meta_regexp, filters=FLAGS.filter, exclude=FLAGS.exclude)
        if FLAGS.preset is not None: kwargs['preset'] = FLAGS.preset
        if FLAGS.sample_rate is not None: kwargs['sr'] = FLAGS.sample_rate
        if FLAGS.channels is not None: kwargs['channels'] = FLAGS.channels
        if FLAGS.chunk_length is not None: kwargs['chunk_length'] = float(FLAGS.chunk_length)
    launch(FLAGS.path, port=FLAGS.port, share=FLAGS.share, inbrowser=FLAGS.inbrowser, **kwargs)


if __name__ == "__main__":
    from absl import app
    import_flags()
    app.run(main)
