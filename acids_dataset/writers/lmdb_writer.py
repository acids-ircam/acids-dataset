import os
import subprocess
import pdb
import tempfile
import pdb
import logging
import collections
import dill 
import shutil
import lmdb
import tqdm
from pathlib import Path
from .utils import audio_paths_from_dir, FeatureHash, KeyIterator, StatusBytes, status_bytes, checklist
from .. import get_fragment_class, get_parser_class_from_path, get_metadata_from_path, get_fragment_class_from_path
from ..utils import get_folder_size
from ..parsers import RawParser, FileNotReadException
from ..fragments import AudioFragment, AcidsFragment
from ..features import AcidsDatasetFeature
from typing import List, Callable, Literal, Tuple
import gin
import yaml


non_buffer_keys = ['feature_hash', 'feature_status', 'features', 'keygen']
VERBOSE_PARSING = False

def print_files(label): 
    print(label, 'open files : ', len(os.listdir('/proc/self/fd')))

@gin.configurable(module="writer")
class LMDBWriter(object):
    KeyGenerator = KeyIterator 
    _max_percent_for_db_size = 0.34
    def __init__(
        self, 
        dataset_path: str | Path, 
        output_path: str | Path,
        fragment_class: str | AudioFragment = "AcidsFragment", 
        features: List[AcidsDatasetFeature] | None = None, 
        dir_parser: Callable = audio_paths_from_dir,
        parser: str | object = RawParser, 
        max_db_size: int | float | None = None, 
        valid_exts: List[str] | None = None,
        filters: List[str] = [], 
        exclude: List[str] = [],
        check: bool = False, 
        force: bool = False,
        waveform: bool = True,
        log: str | None = None
    ):
        """Object responsible for writing LMDb databases.
        The design here is to instantiate to write environments, and access static functions for reading preprocessed data.

        Args:
            dataset_path (str | Path): input dataset path  
            output_path (str | Path): preprocessed output path 
            dir_parser (Callable, optional): Callback used to locate parsed audio files from input datasets.
            parser (str | object, optional): Parser used to extract fragments from a given audio file. 
            fragment_class (str | AudioFragment, optional): AudioFragment subclass used for each fragment. Defaults to "AcidsFragment".
            features (List[AcidsDatasetFeature] | None, optional): Optional list of AcidsDatasetFeature objects for attached features. Defaults to None.
            max_db_size (int | float | None, optional): Maximum dataset size in Gb. Defaults to 30% of target drive.
            dyndb (bool, optional): Allowing dynamic re-growing of database. Defaults to False.
            valid_exts (List[str] | None, optional): Valid audio extensions, None for every readable extension. Defaults to None.
            filters (List[str], optional): List of glob-like patterns to filter audio paths. Defaults to [].
            exclude (List[str], optional): List of glob-like patterns to exclude audio paths. Defaults to [].
            check (bool, optional): Waits for user prompt before proceeding to parsing. Defaults to False.
            force (bool, optional): Do not raise exception if folder already exists. Defaults to False.
            waveform (bool, optional): Parse waveform, or only features. Defaults to True.

        Raises:
            ValueError: target path already exist. 
        """

        # parse dataset
        self.dataset_path = list(map(lambda x: Path(x).resolve(), checklist(dataset_path)))
        self._parse_dataset(dir_parser, valid_exts, filters, exclude)
        # create output
        self.output_path = Path(output_path).resolve()
        if self.output_path.exists():
            if force:
                shutil.rmtree(self.output_path)
            else:
                raise ValueError('%s seems to exist. Please provide a free path'%output_path)
        os.makedirs(self.output_path)
        # record parsers
        self.parser = parser
        self.max_db_size = max_db_size or self.default_db_size(self.output_path)
        if isinstance(fragment_class, str): fragment_class = get_fragment_class(fragment_class)
        self.fragment_class = fragment_class
        self.features = features or []
        self.metadata = {'filters': filters, 'exclude': exclude, 'max_db_size': self.max_db_size}
        self.waveform = waveform
        self.log = log
        if check:
            print(f'Dataset path : {dataset_path}')
            print(f'Output path : {output_path}')
            print(f'filters: {filters};\texclude: {exclude}')
            print(f'features : {features}')
            print(f'Found {len(self._files)} files')
            if not waveform:
                print('No waveform import')
            out = None
            while out is None:
                out = input('proceed? (y/n): ')
                if out.lower() == "y":
                    out = True
                elif out.lower() == "n":
                    out = False
                else:
                    out = None
            if not out: 
                exit()

    @classmethod
    def default_db_size(self, path):
        return self._max_percent_for_db_size * (shutil.disk_usage(path)[2] // (2**30))

    def _init_feature_hash(self):
        """inits internal feature hash"""
        feature_hash = FeatureHash(original_path={})
        return feature_hash

    def _init_feature_status(self, features): 
        feature_status = {f.feature_name: StatusBytes() for f in features}
        feature_status['waveform'] = StatusBytes()
        return feature_status

    def open(self): 
        return lmdb.open(str(self.output_path.resolve()), 
                        map_size = int(self.max_db_size * 1024 ** 3))
    
    @staticmethod
    def get_feature_name(f):
        """returns feature name for a given AcidsFeature object."""
        return getattr(f, "feature_name", type(f).__name__.lower())

    def _parse_dataset(self, dir_parser, valid_exts = None, filters=None, exclude=None, dataset_path=None):
        """Extract target audio files from the directories"""
        dataset_path = dataset_path or self.dataset_path
        self._files = []
        for path in checklist(self.dataset_path):
            if not path.exists(): 
                raise FileNotFoundError("%s folder does not exist."%path)
            self._files.extend(dir_parser(path, valid_exts = valid_exts, flt=filters, exclude=exclude))
        if len(self._files) == 0:
            raise RuntimeError(f'no valid files were found in {dataset_path} with flt={filters} and exclude={exclude}.')
        self.valid_exts = valid_exts
        self.filters = filters
        self.exclude = exclude

    def _update_metadata(self, metadata, metadata_dict):
        """updates internal metadata"""
        for k, v in metadata.items():
            if k not in metadata_dict: metadata_dict[k] = v
        return metadata_dict

    @staticmethod
    def _extract_features(fragment, features, current_key, feature_hash, feature_status, overwrite=False):
        """extract features from a given fragment"""
        for feature in features:
            feature_name = feature.feature_name
            if fragment.has_buffer(feature_name) or fragment.has_metadata(feature_name):
                if not overwrite:
                    if VERBOSE_PARSING: logging.info(f"metadata {feature_name} already present in fragment {current_key} ; skipping")
                    return False 
            feature.extract(fragment=fragment, 
                            current_key=current_key, 
                            feature_hash=feature_hash)
            feature_status[feature.feature_name].push(feature.read(fragment)) is not None
        return True

    @classmethod
    def _close_features(cls, features):
        """terminate features objects before quitting"""
        for feature in features:
            feature.close()
        
    @classmethod
    def _add_feature_hash_to_lmdb(cls, txn, feature_hash):
        """append feature hash to lmdb"""
        txn.put(
            cls.KeyGenerator.from_str("feature_hash"),
            dill.dumps(dict(feature_hash))
        )

    @classmethod
    def _add_features_to_lmdb(cls, txn, features):
        """binarize and append features to database"""
        binarized_features = collections.OrderedDict()
        for f in features:
            binarized_features[f.feature_name] = f
        txn.put(
            cls.KeyGenerator.from_str("features"),
            dill.dumps(binarized_features)
        )

    @classmethod
    def _add_feature_status_to_lmdb(cls, txn, feature_status):
        for k, v in feature_status.items(): 
            feature_status[k].close()
        txn.put(
            cls.KeyGenerator.from_str("feature_status"),
            dill.dumps(feature_status)
        )

    @classmethod
    def _add_file_to_lmdb(cls, 
                          txn, 
                          parser, 
                          features, 
                          feature_hash, 
                          feature_status,
                          key_generator, 
                          fragment_class, 
                          dataset_path):
        n_seconds = 0
        current_file = parser.audio_path
        for load_fn in parser:
            try:
                current_data = load_fn(fragment_class)
                current_file = str(Path(current_file).relative_to(dataset_path.resolve().parent))
                feature_hash['original_path'][current_file] = feature_hash['original_path'].get(current_file, [])
                for fragment in current_data:
                    current_key = next(key_generator)
                    n_seconds += float(fragment.get_metadata().get('length', '0'))
                    feature_hash['original_path'][str(current_file)].append(current_key)
                    cls._extract_features(fragment, features, current_key, feature_hash, feature_status)
                    txn.put(
                        current_key.encode(), 
                        fragment.serialize()
                    )
                    feature_status['waveform'].push(fragment.has_buffer('waveform'))
            except FileNotReadException: 
                pass
        return n_seconds

    @classmethod
    def _add_keygen_to_lmdb(cls, txn, keygen):
        """append hash key generator to database. Useful when updating a database."""
        txn.put(
            cls.KeyGenerator.from_str("keygen"),
            dill.dumps(keygen)
        )

    def _retrieve_basedir_from_path(self, audio_path):
        for s in checklist(self.dataset_path):
            if Path(audio_path).is_relative_to(s):
                return s
        raise FileNotFoundError('Could not extract dataset path from %s'%audio_path)

    def build(self, compact: bool = False):
        """Builds the pre-processed database."""
        env = self.open()
        n_seconds = 0
        metadata = {}
        feature_hash = self._init_feature_hash()
        feature_status = self._init_feature_status(self.features)
        key_generator = iter(KeyIterator())
        with env.begin(write=True) as txn:
            for i, current_file in tqdm.tqdm(enumerate(self._files), total=len(self._files)):
                dataset_path = self._retrieve_basedir_from_path(current_file)
                parser = self.parser(current_file, features=self.features, dataset_path = dataset_path, waveform=self.waveform) 
                if len(metadata) == 0:
                    metadata = self._update_metadata(parser.get_metadata(), metadata)
                parsed_time = self._add_file_to_lmdb(txn, parser, self.features, feature_hash, feature_status, key_generator, self.fragment_class, dataset_path)
                n_seconds += parsed_time
                if self.log: 
                    pid = os.getpid()
                    try: 
                        n_objs = len(os.listdir(f'/proc/{pid}/fd'))
                        with open(self.log, "a+") as file: 
                            file.write(f'{i}\t{current_file}->{key_generator.current_idx} ({n_objs} files opened)\n')
                    except: 
                        pass

            logging.info('adding feature hash to database...')
            type(self)._add_feature_hash_to_lmdb(txn, feature_hash)
            self._close_features(self.features)
            logging.info('adding features to database...')
            type(self)._add_features_to_lmdb(txn, self.features)
            type(self)._add_keygen_to_lmdb(txn, key_generator)
            type(self)._add_feature_status_to_lmdb(txn, feature_status)

        # write metadata
        metadata_path = self.output_path / "metadata.yaml"
        n_chunks = key_generator.current_idx
        with open(metadata_path, "w+") as f:
            yaml.safe_dump({
                "n_seconds": n_seconds,
                "n_chunks": n_chunks,
                "writer_class": type(self).__name__,
                "fragment_class": self.fragment_class.__name__, 
                "features": {self.get_feature_name(f): str(f) for f in self.features},
                "parser_class": self.parser.__name__,
                **self.metadata, 
                **metadata,
            }, f)
        extras_path = self.output_path / "code"
        os.makedirs(extras_path, exist_ok=True)
        fragment_class_name = self.fragment_class.__module__.split('.')[-1]

        # write additional files
        proto_path = Path(__file__).parent / ".." / "fragments" / "interfaces" / f"{fragment_class_name}.proto"
        if os.path.exists(proto_path):
            shutil.copyfile(proto_path, extras_path / proto_path.name)
        compiled_path = Path(__file__).parent / ".." / "fragments" / "compiled" / f"{fragment_class_name}_pb2.py"
        if os.path.exists(compiled_path):
            shutil.copyfile(compiled_path, extras_path / compiled_path.name)
        with open(self.output_path / "config.gin", "w+") as f:
            f.write(gin.config_str())

        if compact: 
            with tempfile.TemporaryDirectory() as tmpdir: 
                for f in ['data.mdb', 'lock.mdb']:
                    shutil.move(os.path.join(self.output_path, f), os.path.join(tmpdir, f))
                with lmdb.open(tmpdir, meminit=False, map_async=True) as env:
                    env.copy(path=str(self.output_path), compact=True)

    @classmethod
    def get_feature_hash(cls, txn):
        feature_hash = dill.loads(txn.get('feature_hash'.encode('utf-8'))) 
        return feature_hash

    @classmethod
    def get_features(cls, txn=None):
        features = dill.loads(txn.get('features'.encode('utf-8'))) 
        return features 

    @classmethod
    def iter_fragment_keys(cls, txn):
        non_keys = list(map(cls.KeyGenerator.from_str, non_buffer_keys))
        for key in txn.cursor().iternext(values=False):
            if key not in non_keys:
                yield key
        
    @classmethod
    def status_from_buffer_name(cls, buffer_name, txn, fragment_class):
        status = bytearray(1024)
        bs = StatusBytes()
        for i, key in enumerate(cls.iter_fragment_keys(txn)):
            bs.push(fragment_class(txn.get(key)).has_buffer(buffer_name))
        return bs 
    
    @classmethod 
    def status_from_feature(cls, f: AcidsDatasetFeature, txn: lmdb.Transaction, fragment_class):
        status = bytearray(1024)
        bs = StatusBytes()
        for i, key in enumerate(cls.iter_fragment_keys(txn)):
            bs.push(f.read(fragment_class(txn.get(key))) is not None)
        return bs 
    
    @classmethod
    def update(cls, 
        path: str, 
        features: List[AcidsDatasetFeature] | None = None,
        data: List[str | Path] | None = None, 
        check: bool = True, 
        overwrite: bool = False, 
        filters: List[str] | None = None,
        exclude: List[str] | None = None,
        dir_parser: Callable = audio_paths_from_dir,
        valid_exts: List[str] | None = None, 
        max_db_size: int = 100,
        show_status: bool = True,
    ):
        path = Path(path).resolve()
        assert path.exists(), f"preprocessed dataset {path} does not seem to exist."
        features = features or []
        data = data or []
        assert len(features) + len(data) > 0, "update needs at least one operation to perform; here both features and data are empty"
        for i, d in enumerate(data):
            data[i] = Path(d).resolve()
            if not data[i].exists(): raise FileNotFoundError(f'folder {data[i]} does not seem to exist')
        max_db_size = max(max_db_size or cls.default_db_size(path), 1.1 * get_folder_size(path))

        # create env
        env = lmdb.open(str(path), map_size=int(max_db_size * 1024 ** 3))
        metadata = get_metadata_from_path(path)
        n_chunks = metadata['n_chunks']
        n_seconds = metadata['n_seconds']

        # check
        if check:
            ('updating target dataset %s :'%path)
            if len(features) > 0:
                print('adding features : ')
                for i, f in enumerate(features):
                    print('%d: %s'%(i, f))
            if len(data) > 0:
                print('parsing additional audio files : ')
                for i, d in enumerate(data):
                    print('%d: %s'%(i, d))

            out = None
            while out is None:
                out = input('proceed? (y/n): ')
                if out.lower() == "y":
                    out = True
                elif out.lower() == "n":
                    out = False
                else:
                    out = None
            if not out: 
                exit()
            
        # parse features
        with env.begin(write=True) as txn:
            existing_features = list(dill.loads(txn.get(cls.KeyGenerator.from_str('features'))).values())
            existing_features_names = [f.feature_name for f in existing_features]

            # parse features
            fragment_class = get_fragment_class_from_path(path)
            feature_hash = cls.get_feature_hash(txn)

            # check feature status bytearray
            feature_status = txn.get(cls.KeyGenerator.from_str('feature_status'))
            if not feature_status:
                feature_status = {k.feature_name: None for k in existing_features}
            else:
                feature_status = dill.loads(feature_status)
            if not "waveform" in feature_status:
                feature_status["waveform"] = cls.status_from_buffer_name("waveform", txn, fragment_class)
            for f in existing_features: 
                if not f.feature_name in feature_status:
                    feature_status[f.feature_name] = cls.status_from_feature(f, txn, fragment_class)
                    txn.put(cls.KeyGenerator.from_str('feature_status'), dill.dumps(feature_status))

            if len(features) > 0:
                for f in features: 
                    if f.feature_name in existing_features_names and not overwrite:
                        continue
                    if (f.feature_name not in feature_hash):
                        feature_hash[f.feature_name] = {}
                    feature_status[f.feature_name] = StatusBytes()
                    iterator = cls.iter_fragment_keys(txn) if not show_status else tqdm.tqdm(cls.iter_fragment_keys(txn), desc="computing feature %s"%f.feature_name, total=metadata['n_chunks'])
                    for key in iterator:
                        fragment = fragment_class(txn.get(key))
                        needs_write = cls._extract_features(fragment, [f], key, feature_hash, feature_status, overwrite=overwrite)
                        if needs_write:
                            txn.put(key, fragment.serialize())

            # check feature hash for every present feature

            # then, add additional data if needed
            parser_class = get_parser_class_from_path(path)
            files = []
            filters = filters or []
            exclude = exclude or []
            features = existing_features + features
            if len(data) > 0:
                for data_path in data:
                    files.append((data_path, 
                                dir_parser(data_path, valid_exts = valid_exts, flt=filters, exclude=exclude)))
                
                new_file_list = set(sum([d[1] for d in files], []))
                original_file_list = set(feature_hash['original_path'].keys())
                common_files = new_file_list.intersection(original_file_list)

                if not overwrite: 
                    new_file_list.difference_update(common_files)
                
                try:
                    key_generator = dill.loads(txn.get(b'keygen'))
                except Exception as e:
                    key_generator = iter(KeyIterator(n_chunks))
                status_bar = tqdm.tqdm(total=len(new_file_list), desc="parsing additional files...")
                status_bar.display()
                
                for data_path, current_files in files:
                    data_path = data_path.resolve().absolute()
                    for current_file in current_files:
                        status_bar.update(1)
                        parser = parser_class(current_file, 
                                              features=features, 
                                              dataset_path = data_path)
                        parsed_time = cls._add_file_to_lmdb(txn, parser, features, feature_hash, feature_status, key_generator, fragment_class, data_path)
                        n_seconds += parsed_time
                n_chunks = key_generator.current_idx
            metadata.update(n_seconds=n_seconds, n_chunks=n_chunks, features={cls.get_feature_name(f): str(f) for f in features})

            cls._add_feature_hash_to_lmdb(txn, feature_hash)
            cls._add_features_to_lmdb(txn, features)
            cls._add_feature_status_to_lmdb(txn, feature_status)

            # write metadata
            metadata_path = path / "metadata.yaml"
            with open(metadata_path, "w+") as f:
                yaml.safe_dump(metadata, f)

        env.close()
        cls._close_features(features)
        
    @classmethod
    def add_feature_hash(cls, path, feature, feature_hash, max_db_size = None, keygen=None):
        path = Path(path).resolve()
        if max_db_size is None and (path/"metadata.yaml").exists():
            with open(str(path/"metadata.yaml")) as f:
                max_db_size = yaml.safe_load(f).get('max_db_size')
        if max_db_size is None: max_db_size = cls._default_max_db_size
        assert path.exists(), f"preprocessed dataset {path} does not seem to exist."
        keygen = keygen or cls.KeyGenerator

        env = lmdb.open(str(path), map_size=int(max_db_size * 1024 ** 3))
        fh_key = keygen.from_str('feature_hash')
        with env.begin(write=True) as txn:
            original_feature_hash = dill.loads(txn.get(fh_key))
            original_feature_hash[feature] = feature_hash
            txn.put(fh_key, dill.dumps(original_feature_hash))

    @classmethod
    def append_to_feature_metadata(cls, path, feature, metadata, max_db_size=None, keygen=None):
        path = Path(path).resolve()
        assert path.exists(), f"preprocessed dataset {path} does not seem to exist."
        if max_db_size is None and (path/"metadata.yaml").exists():
            with open(str(path/"metadata.yaml")) as f:
                max_db_size = yaml.safe_load(f).get('max_db_size')
        if max_db_size is None: max_db_size = cls._default_max_db_size
        keygen = keygen or cls.KeyGenerator

        # create env
        env = lmdb.open(str(path), map_size=int(max_db_size * 1024 ** 3))
        f_key = keygen.from_str("features")
        with env.begin(write=True) as txn:
            original_feature = dill.loads(txn.get(f_key))[feature]
            original_feature.metadata.update(metadata)
            # write feature hash
            txn.put(f_key, dill.dumps(original_feature))


class LMDBLoader(object):

    def __init__(self, 
                 db_path: str | Path, 
                 output_type: Literal["numpy", "torch"] = "torch",
                 lazy_import: bool = False, 
                 lazy_paths: str | List[str] | None = None
                 ):
        """LMDBLoader is the object called when loading a database.

        Args:
            db_path (str | Path): pre-processed database path
            output_type (str, optional): output type. Defaults to "numpy".
            lazy_import (bool, optional): _description_. Defaults to False.
            lazy_paths (str | List[str] | None, optional): _description_. Defaults to None.
        """
        self._db_path = db_path
        self._metadata = get_metadata_from_path(db_path)
        self._fragment_class = get_fragment_class(self._metadata.get('fragment_class', "AcidsFragment"))
        self._output_type = output_type
        self._length = self._metadata.get('n_chunks')
        keygen_class = getattr((locals().get(self._metadata.get('writer_class')) or LMDBWriter), "KeyGenerator", KeyIterator)
        filter_keys = list(map(lambda x: keygen_class.from_str(x), non_buffer_keys))
        
        with self.database.begin() as txn:
            self._keys = list(filter(lambda x: x not in filter_keys, txn.cursor().iternext(values=False)))
            self._length = len(self._keys)
            keygen = txn.get('keygen'.encode('utf-8'))
            self._keygen = type(self).writer.KeyGenerator() if keygen is None else dill.loads(keygen)
            features = txn.get(self._keygen.from_str('features'))
            self._features = [] if features is None else dill.loads(features)
            feature_status = txn.get(self._keygen.from_str('feature_status')) 
            self._feature_status = None if feature_status is None else dill.loads(feature_status)

    def __len__(self):
        return self._length

    def __getitem__(self, idx: int | str| bytes):
        if isinstance(idx, int):
            idx_key = self._keys[idx]
        elif isinstance(idx, str):
            idx_key = self._keygen.from_str(idx)
        else:
            assert isinstance(idx, bytes), "__getitem__ must be either int or bytes"
            idx_key = idx
        with self.database.begin(write=False) as txn:
            fg = self._fragment_class(txn.get(idx_key), output_type=self._output_type)
        return fg

    def __contains__(self, idx: bytes):
        return idx in self._keys

    def get_key_from_idx(self, idx: int):
        return self._keys[idx]

    def open(self, path=None, readonly=True, lock=True):
        if path is None: path = self._db_path
        return lmdb.open(str(path), lock=lock, readonly=readonly)

    @property
    def database(self):
        return self.open(self._db_path, readonly=True) 

    @property
    def feature_status(self):
        return self._feature_status

    @property
    def feature_hash(self):
        return self.get_feature_hash()

    @property
    def keygen(self): 
        return self._keygen

    @property
    def features(self): 
        return self._features

    def get_feature_hash(self, txn=None):
        transaction = txn or self.database.begin()
        feature_hash = dill.loads(transaction.get('feature_hash'.encode('utf-8'))) 
        if txn is None: transaction.__exit__()
        return feature_hash

    def get_features(self, txn=None):
        transaction = txn or self.database.begin()
        features = dill.loads(transaction.get('features'.encode('utf-8'))) 
        if txn is None: transaction.__exit__()
        return features 

    def iter_fragment_keys(self, as_bytes=True, txn=None):
        transaction = txn or self.database.begin()
        non_keys = list(map(self._keygen.from_str, non_buffer_keys))
        for key in transaction.cursor().iternext(values=False):
            if key not in non_keys:
                if as_bytes:
                    yield key
                else:
                    yield self._keygen.to_str(key)
        if txn is None: transaction.__exit__()
            
    def iter_fragments(self, txn: lmdb.Environment | None = None) -> Tuple[str, AudioFragment]:
        """iter_fragments can be used to iterate contained fragments.

        Args:
            txn (lmdb.Environment | None, optional): optional LMDB environement. Defaults to None.

        Yields:
            : key, fragment (Tuple[str, AcidsFragment]): key and fragments
        """
        transaction = txn or self.database.begin()
        non_keys = list(map(self._keygen.from_str, non_buffer_keys))
        with self.database.begin(readonly=True) as txn:
            for key in txn.cursor().iternext(values=False):
                if key not in non_keys:
                    yield key, self._fragment_class(txn.get(key))
        if txn is None: transaction.__exit__()

    def parse_from_path(self, path):
        """parse_from_path direcly returns the iterator, and the feature hash, from a given database.

        Args:
            path (_type_): _description_

        Returns:
            _type_: _description_
        """
        env = self.open(path)
        with env.begin() as txn:
            feature_hash = self.get_feature_hash(txn)
            iterator = self.iter_fragments(txn)
        return iterator, feature_hash


LMDBWriter.loader = LMDBLoader
LMDBLoader.writer = LMDBWriter

