# from typing import Callable, Dict, Set
import logging
# from vissl.data.vissl_dataset_base import VisslDatasetBase
from fvcore.common.file_io import PathManager  # TODO iopath
from PIL import Image
from vissl.data.data_helper import get_mean_image
from vissl.data.data_helper import QueueDataset
from torch.utils.data import Dataset


def demo_cfg():
    import pkg_resources
    from vissl.config import AttrDict
    import yaml
    default_path = pkg_resources.resource_filename('vissl', 'config/defaults.yaml')
    with open(default_path, 'r') as file:
        data = yaml.load(file, Loader=yaml.SafeLoader)
    cfg = AttrDict(data)['config']
    return cfg


class KWCocoDataset(Dataset):
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
        print('CREATE KWCOCO DATASET')
        print('path = {!r}'.format(path))
        print('dataset_name = {!r}'.format(dataset_name))
        print('split = {!r}'.format(split))
        print('data_source = {!r}'.format(data_source))
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
        print('self.image_ids = {!r}'.format(self.image_ids))

        # whether to use QueueDataset class to handle invalid images or not
        # self.enable_queue_dataset = cfg["DATA"][self.split]["ENABLE_QUEUE_DATASET"]

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
            pil_img, is_success = self[index]
        """
        # print(len(self))
        # print('GETITEM: index = {!r}'.format(index))
        gid = self.image_ids[index]
        image_path = self.dset.get_image_fpath(gid)

        is_success = True

        with PathManager.open(image_path, "rb") as fopen:
            pil_img = Image.open(fopen).convert("RGB")

        if 0:
            try:
                with PathManager.open(image_path, "rb") as fopen:
                    pil_img = Image.open(fopen).convert("RGB")

                if is_success and self.enable_queue_dataset:
                    self.on_sucess(pil_img)
            except Exception as e:
                logging.warning(
                    f"Couldn't load: {image_path}. Exception: \n{e}"
                )
                is_success = False
                # if we have queue dataset class enabled, we try to use it to get
                # the seen valid images
                if self.enable_queue_dataset:
                    pil_img, is_success = self.on_failure()
                    if pil_img is None:
                        pil_img = get_mean_image(
                            self.cfg["DATA"][self.split].DEFAULT_GRAY_IMG_SIZE
                        )
                else:
                    pil_img = get_mean_image(self.cfg["DATA"][self.split].DEFAULT_GRAY_IMG_SIZE)

        if 0:
            import numpy as np
            import torch
            hwc = np.asarray(pil_img.convert("RGB"))
            chw = torch.from_numpy(hwc.transpose(2, 0, 1)).float()
            print('chw.shape = {!r}'.format(chw.shape))
            print('is_success = {!r}'.format(is_success))
            # is_success = torch.from_numpy(is_success)
            # is_success = torch.from_numpy(np.array([is_success]).astype(np.int64)).long()
            return chw, is_success
        return pil_img, is_success

    def get_labels(self):
        """  Be able to return labels when necessary.  


        Ignore:
            from vissl.data.kwcoco_dataset import *  # NOQA
            cfg = demo_cfg()
            self = KWCocoDataset(cfg, None, 'special:shapes8', 'TRAIN', None)
            self.get_labels()
            pil_img, is_success = self[index]
        """
        self.labels = list()
        for gid in sorted(self.image_ids):
            aids = list(self.dset.cid_to_aids[gid])
            if len(aids) == 0:
                raise ValueError(f'Every image needs an id if get_labels is being called.  Image Id {gid} has no labels')
                # logging.warning(f'Image Id {gid} has no label.  Setting it to 0.')
                self.labels.append(0)
            elif len(aids) > 1: 
                logging.warning(f'Image Id {gid} has more than one label.  Only using the first label.')
            self.labels.append(self.dset.anns[aids[0]]['category_id'])
        return self.labels
