# from typing import Callable, Dict, Set
import logging
# from vissl.data.vissl_dataset_base import VisslDatasetBase
from fvcore.common.file_io import PathManager  # TODO iopath
from PIL import Image
from vissl.data.data_helper import QueueDataset, get_mean_image


def demo_cfg():
    import pkg_resources
    from vissl.config import AttrDict
    import yaml
    default_path = pkg_resources.resource_filename('vissl', 'config/defaults.yaml')
    with open(default_path, 'r') as file:
        data = yaml.load(file, Loader=yaml.SafeLoader)
    cfg = AttrDict(data)['config']
    return cfg


class KWCocoDataset(QueueDataset):
    """
    Example:
        cfg = demo_cfg()
        split = "TRAIN"
        cfg["DATA"][split].DATASET_NAMES = ['do-names-matter?']
        cfg["DATA"][split].DATASET_PATHS = ['special:shapes8']
        cfg["DATA"][split].DATA_SOURCES = ['kwcoco']
        from vissl.data import DATASET_SOURCE_MAP
        from vissl.data import DATA_SOURCES_WITH_SUBSET_SUPPORT
    """
    def __init__(self, cfg, data_source, path, split, dataset_name):
        import kwcoco
        self.cfg = cfg
        self.split = split
        self.dataset_name = dataset_name
        self.data_source = data_source
        self._path = path
        self.is_initialized = False

        self.dset = kwcoco.CocoDataset.coerce(path)

        # Simple kwcoco dataset just loads entire images.
        # In the future the ndsampler version should be able to handle
        # windowed regions.

        # Load a list of all images
        self.image_ids = list(self.dset.index.imgs.keys())

        # whether to use QueueDataset class to handle invalid images or not
        self.enable_queue_dataset = cfg["DATA"][self.split]["ENABLE_QUEUE_DATASET"]

    def __len__(self):
        """
        Size of the dataset. Assumption made there is only one data source
        """
        return len(self.image_ids)

    def __getitem__(self, index: int):
        """
        Ignore:
            from vissl.data.kwcoco_dataset import *  # NOQA
            cfg = demo_cfg()
            self = KWCocoDataset(cfg, None, 'special:shapes8', 'TRAIN', None)
            index = 0
            img, is_success = self[index]
        """
        gid = self.image_ids[index]
        image_path = self.dset.get_image_fpath(gid)

        is_success = True

        try:
            with PathManager.open(image_path, "rb") as fopen:
                img = Image.open(fopen).convert("RGB")

            if is_success and self.enable_queue_dataset:
                self.on_sucess(img)
        except Exception as e:
            logging.warning(
                f"Couldn't load: {image_path}. Exception: \n{e}"
            )
            is_success = False
            # if we have queue dataset class enabled, we try to use it to get
            # the seen valid images
            if self.enable_queue_dataset:
                img, is_success = self.on_failure()
                if img is None:
                    img = get_mean_image(
                        self.cfg["DATA"][self.split].DEFAULT_GRAY_IMG_SIZE
                    )
            else:
                img = get_mean_image(self.cfg["DATA"][self.split].DEFAULT_GRAY_IMG_SIZE)

        return img, is_success
