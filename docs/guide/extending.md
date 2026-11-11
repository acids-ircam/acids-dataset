# Extending acids-dataset

## Customizing features

If you want to customize a feature, you can copy a feature configuration file (see `acids_dataset/configs/features/`) and place it a `ad_configs` subfolder of your current working directory. It is recommended to give it a custom name to avoid naming problems. For examples, if you want a custom CQT with 257 bins (instead of the default 84), copy `acids_dataset/configs/features/cqt.gin` to `$(pwd)/ad_configs/cqt_rene.gin` (if your name is René), and modify the amount of bins :

```yaml
CQT:
  name=cqt_rene
  n_bins=257
  device=%DEVICE
  sr = %SAMPLE_RATE

ad.features.parse_features:
  features = @ad.features.CQT
```

and update your database using `acids-dataset update --path ... --feature cqt_rene`. The feature will be added as `cqt_rene`, as the name in the configuration as been specified. 

## Implementing features
You can add your own features by subclassing the `AcidsDatasetFeature` object. 

```python
class AcidsDatasetFeature(object):
    denylist = []
    has_hash = False # (1) <---- see below to see how datasets are hashed
    def __init__(
            self, 
            name: Optional[str] = None,
            hash_from_feature: Optional[Callable] = None, 
        ):
        super(CustomFeature, self).__init__(name, hash_from_feature)

    @property
    def default_feature_name(self):
        """defines the default feature name, is not provided"""
        return type(self).__name__.lower()

    def pre_chunk_hook(self, path, audio, sr) -> None:
      """this is a hook to perform some optional operations before audio chunking."""
      pass

    def close(self):
        """if some features has side effects like buffering, empty buffers and delete files"""
        pass

    def from_file(self, path, start_pos, chunk_length):
      # extraction code here....
      return feature

    def from_audio(self, audio):
      # extraction code here...
      return feature

    def from_fragment(self, fragment, write: bool = None):
        """extract the data from fragment, and writes into it if write is True"""
        audio_data = fragment.get_audio("waveform")
        meta = self.from_audio(audio_data)
        # you can also access the original file
        metadata = fragment.get_metadata()
        meta = self.from_file(metadata['audio_path'], metadata['start_pos'], metadata['chunk_length'])
        if write: 
          fragment.put_array(self.feature_name, meta)
        return meta
```

The `AcidsDatasetFeature` is the base object for all audio features. One instance is created for one feature, and the object is called when : 
- An audio file is gonna be chunked, through a call to `pre_chunk_hook` (for some buffering, or some analysis that would require the whole file)
- During writing, when the audio chunks are written in the lmdb database. 
- In online mode, when an audio chunk is loaded.

`AcidsDatasetFeature` also automatically fills up a hash during writing, allowing to get all the audio indexes belonging to a given hash. However, this is not automatic: this hash process is performed if :  
- the `has_hash` attribute is `True` (the feature has then to be hashable ; arrays, tensors, or lists, are typically non hashable.)
- a `hash_from_feature(self, ft)` callback is defined in the class, or provided at initialization (allowing to be gin configurable). If a callback is provided at initialization, it erases in any case the one defined (or not) in the class.

## Customizing augmentations

Similarly to features, you can copy the `compress.gin` configuration as `thierry_compress.gin` (if your name is Thierry), and customize the inner parameters :

```yaml
Compress:
  name = "thierry_compress"
  threshold = -30
  amp_range = [-20, 10]
  attack = 0.1
  release = 0.3
  prob = 1.0
  limit = False
  sr = %SAMPLE_RATE

ad.transforms.parse_transform:
    transform = @ad.transforms.Compress
```

## Implementing transforms

A new transform can be done as follows :

```python
@gin.configurable(module="transforms")
class NewTransform(Transform):
    takes_as_input = Transform.input_types.torch
    allow_random = True

    def __init__(self, custom_param: int | None = None, **kwargs):
      super().__init__()

    def apply(self, x):
      # operations to x....
```

and... that's it! `apply` provides the main operating function for data processing, and is directly called by `Transform` if `p=None` or randomly called if `p>0`.
The `allow_random` attributes if transform can be randomized. You can also overload the method `apply_random(self, x)` to override how the transform behaves when `p>0`.
