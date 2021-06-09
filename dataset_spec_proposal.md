## Preface

The goal of this issue is to discuss ideas for making it easier to specify
custom dataset, or to build up documentation that explains why the existing
process is already minimal.

I find myself frustrated that I can't just point VISSL to a custom dataset on
the command line, and I think I have ideas on how to make this process easier,
but at the same time, I only have shallow familiarity with this project, and if
I'm missing something (perhaps related to distributed processing, with which
I'm less familiar), I'd like to help build docs to make the rational for the
existing process clearer.


## 🚀 Feature

Currently, using VISSL on a custom dataset requires 3 steps, and includes
manually editing a registry file [1, 2, 3].

The proposed feature is to allow specification of custom datasets in a single
step (not counting the creation of the data manifest file) via pointing to a
path to a data manifest file.

A data manifest is a single path on disk that contains all information about a
specific dataset split. In its simplest form, a manifest may simply contain a
list of image file paths (or uris) --- it contains pointers to the data for
which it is concerned. However, it can be more complex, and also contain
information about annotations within each image, and category labels relevant
to the dataset.

A data manifest file has several advantages over predefined folder structures. 
It is self-contained, 

Given some data manifest file, it should be possible to run SSL without any
extra preprocessing, registration, or code modification. For example:

```bash

python \
    tools/run_distributed_engines.py \
    config=test/integration_test/quick_simclr.yaml \
    config.DATA.TRAIN.DATA_PATHS=[/path/to/train_data_manifest_file]
    config.DATA.TEST.DATA_PATHS=[/path/to/test_data_manifest_file]
```



### Choice of Manifest Format

The specific format of a data manifest file is an important design choice.

The MS-COCO [4] format is a well-known and natural choice for a basic dataset
manifest because it provides a single file that points to a list of images, and
an optional set of annotations and categories associated with those images. 

However, there are shortcomings in the MS-COCO format, including: lack of
support support for polygons with holes and keypoints that to not conform to a
class skeleton. Furthermore, the official `pycocotools` API is difficult to
install and lacks support for dynamic modification of the dataset, among other
issues.  I've been working on a backwards-compatible extension called KW-COCO
(KW stands for Kitware, which is my employer) [5]. This addresses
the aforementioned issues and also provides an extension to multispectral
imagery, grouping of sequences of images into videos, and can be combined with
a secondary library `ndsampler` [6], to easily create data loaders that operate
on subregions within images and videos.
 
It is important to note that VISSL does support the coco2014 dataset, but the
way it is specified ignores the fact that the annotation file specifies paths
to the images in the annotation file [7]. Instead of simply pointing to an
annotation/manifest file and assuming that the images/assets are relative to
the manifest file itself (unless otherwise specified) VISSL assumes that there
is a folder structure with very specific and inflexible path names.
 
Albeit this is also a problem with the way that COCO-2014 dataset is
distributed.  The `"file_name"` in the image dictionary does not specify a
relative path to the expected file. In contrast, the kwcoco-spec does specify
that the `"file_name"` attribute should be a relative or absolute path to an
image file.  It also provides a method and a command-line tool `kwcoco reroot`
[8] to help ensure relative or absolute paths are correct before you pass the
data to an algorithm. This works similarly to `REMOVE_IMG_PATH_PREFIX` and
`NEW_IMG_PATH_PREFIX` in the vissl config [8], but instead of coupling this
step with the SSL training, using kwcoco allows this process to be decoupled
from training and prediction.


## Proposal

I would like to propose adding a either an mscoco or kwcoco data source to
vissl. I would recommend kwcoco over mscoco because it is easier to install,
has a command line tool and Python API such that developers can do
preprocessing in a way that is decoupled from the training code that it will be
fed to, and it comes with its own synthetic toydata, which I think  


What I would like to do is use kwcoco for generating toydata: 

```bash

# Autogenerates a toy train and test dataset
pip install kwcoco
kwcoco toydata --key=shapes32 --bundle_dpath=my_train_bundle
kwcoco toydata --key=shapes16 --bundle_dpath=my_test_bundle
```

This results in the following data structure:

```
my_train_bundle
├── _assets
│   └── images
│       ├── img_00001.png
│       ├── img_00002.png
│       ...
│       ├── img_00029.png
│       ├── img_00030.png
│       ├── img_00031.png
│       └── img_00032.png
└── data.kwcoco.json
```

And the data manifest file (`my_train_bundle/data.kwcoco.json`) looks like this (truncated for clarity):

```
{
"categories": [
    {"id": 0, "name": "background"},
    {"name": "star", "id": 3, "supercategory": "vector", "keypoints": []},
    {"name": "superstar", "id": 6, "supercategory": "raster", "keypoints": ["left_eye", "right_eye"]},
    {"name": "eff", "id": 7, "supercategory": "raster", "keypoints": ["top_tip", "mid_tip", "bot_tip"]}
],
"keypoint_categories": [
    {"name": "bot_tip", "id": 5, "reflection_id": null},
    {"name": "left_eye", "id": 1, "reflection_id": 2},
    {"name": "mid_tip", "id": 4, "reflection_id": null},
    {"name": "right_eye", "id": 2, "reflection_id": 1},
    {"name": "top_tip", "id": 3, "reflection_id": null}
],
"images": [
    {"width": 600, "height": 600, "channels": "rgb", "id": 1, "file_name": "_assets/images/img_00001.png"},
    {"width": 600, "height": 600, "channels": "rgb", "id": 2, "file_name": "_assets/images/img_00002.png"},
    ...
],
"annotations": [

    {"keypoints": [], "bbox": [234, 283, 162, 63], "area": 10206.0, "id": 1, "image_id": 1, "category_id": 3"segmentation": [...]},

    {"segmentation": [...], "keypoints": [{"xy": [344.175, 318.3], "keypoint_category_id": 3}, {"xy": [333.9, 363.7], "keypoint_category_id": 4}, {"xy": [308.475, 411.4], "keypoint_category_id": 5}], "bbox": [297, 307, 51, 109], "area": 5559.0, "id": 3, "image_id": 2, "category_id": 7},

    ...
]
}
```

Because all the information about my dataset is contained in the manifest file,
I there should be a way to just point whatever learning algorithm I want to use
at the manifest (assuming the information required by the learning algorithm
exists in the dataset).

I've forked VISSL to work on a proof of concept and added a kwcoco data source.

I would think I could do something as simple as:

```bash

python \
    tools/run_distributed_engines.py \
    config=test/integration_test/quick_simclr.yaml \
    config.DATA.TRAIN.DATA_SOURCES=[kwcoco] \
    config.DATA.TRAIN.DATA_PATHS=[my_train_bundle/data.kwcoco.json] \
    config.DATA.TEST.DATA_SOURCES=[kwcoco] \
    config.DATA.TEST.DATA_PATHS=[my_test_bundle/data.kwcoco.json]
```


But the current structure seems like it is forcing me to add these datasets to
a registry file, and I don't see the rational for why that would be necessary.


```
KeyError: "Dataset 'kwcoco' is not registered! Available datasets are: airstore_imagenet, ... google-imagenet1k-per10"
```


I have to add a line to `dataset_catelog.json` and then give extra information
like a dataset name and source:

```bash

python \
    tools/run_distributed_engines.py \
    config=test/integration_test/quick_simclr.yaml \
    config.CHECKPOINT.DIR="./my_toy_training" \
    config.DATA.TRAIN.DATA_SOURCES=[kwcoco] \
    config.DATA.TRAIN.DATA_PATHS=[my_train_bundle/data.kwcoco.json] \
    config.DATA.TRAIN.DATASET_NAMES=[kwcoco] \
    config.DATA.TEST.DATA_SOURCES=[kwcoco] \
    config.DATA.TEST.DATA_PATHS=[my_test_bundle/data.kwcoco.json] \
    config.DATA.TEST.DATASET_NAMES=[kwcoco]
```

I would think it would be possible to make something like `DATASET_NAMES` and
`DATA_SOURCES` coercible for certain types of data sources. For instance,
because the path specifies a kwcoco file, it is possible for the program to
realize it's pointing to a kwcoco source, and there does exist the concept of a
kwcoco dataset name, so it could pull that if it wasn't specified.
 

Note, if the dataset paths were passed to `kwcoco.CocoDataset.coerce` instead
of `kwcoco.CocoDataset` itself, it would also be possible to specify a path
like `special:shapes8`, which kwcoco coerce interprets as a command to generate
or use previously cached generated toy data. This would be an extension of the
currently existing "synthetic" argument in use by vissl.

## Summary

In short, I'm looking to improve the data loading for VISSL, ideally using the
kwcoco library and manifest format I've been developing. Given something in
this format it should be possible to "just run" self-supervised learning by
pointing a command line tool at the dataset. Thus the onus is no longer to
"conform to a structure specified by VISSL to use your data", instead it becomes
"VISSL supports datasets that conform to this standardized structure".

I'm wondering what thoughts on this proposal are, and what challenges I might
be overlooking.



References:

 * [1] https://github.com/facebookresearch/vissl/issues/197
 * [2] https://analyticsindiamag.com/guide-to-vissl-vision-library-for-self-supervised-learning/
 * [3] https://vissl.readthedocs.io/en/v0.1.5/extend_modules/custom_datasets.html
 * [4] https://cocodataset.org/#format-data
 * [5] https://pypi.org/project/kwcoco/
 * [6] https://pypi.org/project/ndsampler/
 * [7] https://github.com/facebookresearch/vissl/tree/395435a3460e9ef1328a76c1ebaa8011fb98e390/vissl/data
 * [8] https://kwcoco.readthedocs.io/en/latest/autoapi/kwcoco/coco_dataset/index.html#kwcoco.coco_dataset.MixinCocoExtras.reroot
 * [9] https://github.com/facebookresearch/vissl/blob/master/vissl/config/defaults.yaml#L219
