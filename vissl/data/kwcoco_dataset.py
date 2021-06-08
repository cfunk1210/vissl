from typing import Callable, Dict, Set

from vissl.data.vissl_dataset_base import VisslDatasetBase
from vissl.config import AttrDict


class KWCocoDataset(VisslDatasetBase):
    """
    Example:
        import pkg_resources
        import yaml
        default_path = pkg_resources.resource_filename('vissl', 'config/defaults.yaml')
        cfg = AttrDict(yaml.load(open(default_path, 'r')))['config']

        split = "TRAIN"
        cfg["DATA"][split].DATASET_NAMES = ['do-names-matter?']
        cfg["DATA"][split].DATASET_PATHS = ['special:shapes8']
        cfg["DATA"][split].DATASET_SOURCES = ['kwcoco']
        from vissl.data import DATASET_SOURCE_MAP
        from vissl.data import DATA_SOURCES_WITH_SUBSET_SUPPORT

    """
    def __init__(
        self,
        cfg: AttrDict,
        split: str,
        dataset_source_map: Dict[str, Callable],
        data_sources_with_subset: Set[str],
        **kwargs,
    ):
        pass

    def __len__(self):
        """
        Size of the dataset. Assumption made there is only one data source
        """
        return self.num_samples(0)

    def __getitem__(self, idx: int):
        """
        Get the input sample for the minibatch for a specified data index.
        For each data object (if we are loading several datasets in a minibatch),
        we get the sample: consisting of {
            - image data,
            - label (if applicable) otherwise idx
            - data_valid: 0 or 1 indicating if the data is valid image
            - data_idx : index of the data in the dataset for book-keeping and debugging
        }

        Once the sample data is available, we apply the data transform on the sample.

        The final transformed sample is returned to be added into the minibatch.
        """

        if not self._labels_init and len(self.label_sources) > 0:
            self._load_labels()
            self._labels_init = True

        subset_idx = idx
        if self.data_limit >= 0 and self._can_random_subset_data_sources():
            if not self._subset_initialized:
                self._init_image_and_label_subset()
            subset_idx = self.image_and_label_subset[idx]

        # TODO: this doesn't yet handle the case where the length of datasets
        # could be different.
        item = {"data": [], "data_valid": [], "data_idx": []}
        for data_source in self.data_objs:
            data, valid = data_source[subset_idx]
            item["data"].append(data)
            item["data_idx"].append(idx)
            item["data_valid"].append(1 if valid else -1)

        # There are three types of label_type (data labels): "standard",
        # "sample_index", and "zero". "standard" uses the labels associated
        # with a data set (e.g. directory names). "sample_index" assigns each
        # sample a label that corresponds to that sample's index in the
        # dataset (first sample will have label == 0, etc.), and is used for
        # SSL tasks in which the label is arbitrary. "zero" assigns
        # each sample the label == 0, which is necessary when using the
        # CutMixUp collator because of the label smoothing that is built in
        # to its functionality.
        if (len(self.label_objs) > 0) or self.label_type == "standard":
            item["label"] = []
            for label_source in self.label_objs:
                if isinstance(label_source, list):
                    lbl = [entry[subset_idx] for entry in label_source]
                else:
                    lbl = _convert_lbl_to_long(label_source[subset_idx])
                item["label"].append(lbl)
        elif self.label_type == "sample_index":
            item["label"] = []
            for _ in range(len(self.data_objs)):
                item["label"].append(idx)
        elif self.label_type == "zero":
            item["label"] = []
            for _ in range(len(self.data_objs)):
                item["label"].append(0)
        else:
            raise ValueError(f"Unknown label type: {self.label_type}")

        # apply the transforms on the image
        if self.transform:
            item = self.transform(item)
        return item

    def num_samples(self, source_idx=0):
        """
        Size of the dataset. Assumption made there is only one data source
        """
        if self.data_limit >= 0:
            return self.data_limit
        return len(self.data_objs[source_idx])

    def get_global_batchsize(self):
        """
        The global batch size across all the trainers
        """
        from classy_vision.generic.distributed_util import get_world_size
        return self.get_batchsize_per_replica() * get_world_size()

    def post_process_batch(self, batch_data):
        """
        Identity function. Other base datasets may need to #post_process_batch.
        """
        return batch_data
