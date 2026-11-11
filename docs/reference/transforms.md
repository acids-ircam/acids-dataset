# Transforms

Spatial and loudness transforms, and their TorchScript modules. For the list of augmentations, see the [Augmentations guide](../guide/augmentations.md).

## Base classes

::: acids_dataset.transforms.base
    options:
      show_root_heading: false
      members: [Transform, Compose]

## Spatial transforms

::: acids_dataset.transforms.spatial
    options:
      show_root_heading: false
      members: [MidSide]

## Ambisonics

::: acids_dataset.transforms.ambisonics
    options:
      show_root_heading: false
      members: [AmbisonicEncoder, AmbisonicDecoder, spherical_harmonics, get_layout, LAYOUTS]

## Loudness range

::: acids_dataset.transforms.misc
    options:
      show_root_heading: false
      members: [CompressRange, NormRange]

## TorchScript modules

Modules given by `transform.nn_module()` (see [TorchScript modules](../guide/augmentations.md#torchscript-modules)).

::: acids_dataset.transforms.nn_modules
    options:
      show_root_heading: false
      members: [TransformModule, IdentityModule, ChannelMatrixModule, MidSideModule, PreEmphasisModule, Log1pModule, Exp1mModule, NormalizeModule, LFilterModule, NormRangeModule, ComposeModule]
