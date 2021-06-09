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


The specific format of a data manifest file is an important design choice. 

The MS-COCO [4] format is a well-known and natural choice for a basic dataset
manifest. 

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
 
It is important to note that VISSL does support the coco2014 dataset, but 
the way it is specified ignores  






There is an important design choice 
The choice of the manifest 

## Motivation & Examples

Tell us why the feature is useful.

Describe what the feature would look like, if it is implemented.
Best demonstrated using **code examples** in addition to words.



The advantage of using data manifest files instead of pointing to folders on disk.


References:

 * [1] https://github.com/facebookresearch/vissl/issues/197
 * [2] https://analyticsindiamag.com/guide-to-vissl-vision-library-for-self-supervised-learning/
 * [3] https://vissl.readthedocs.io/en/v0.1.5/extend_modules/custom_datasets.html
 * [4] https://cocodataset.org/#format-data
 * [5] https://pypi.org/project/kwcoco/
 * [6] https://pypi.org/project/ndsampler/
