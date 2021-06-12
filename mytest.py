"""



INSTALL DEPS
============

pip install git+https://github.com/facebookresearch/ClassyVision.git
pip install apex


pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html


git clone https://github.com/NVIDIA/apex $HOME/code/apex
cd $HOME/code/apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .



Relevant files
==============
~/code/vissl/vissl/config/defaults.yaml
~/code/vissl/configs/config/test/integration_test/quick_simclr.yaml
~/code/vissl/configs/config/dataset_catalog.json
~/code/vissl/vissl/data/kwcoco_dataset.py

~/code/kwcoco/kwcoco/data/grab_cifar.py


Demo CIFAR
==========

python -m kwcoco.data.grab_voc
python -m kwcoco.data.grab_cifar

kwcoco grab --cifar10=True
kwcoco stats $HOME/.cache/kwcoco/data/cifar10/cifar10.kwcoco.json
kwcoco split \
    --src $HOME/.cache/kwcoco/data/cifar10/cifar10.kwcoco.json \
    --dst1 $HOME/.cache/kwcoco/data/cifar10/cifar10-split1.kwcoco.json \
    --dst2 $HOME/.cache/kwcoco/data/cifar10/cifar10-split2.kwcoco.json \
    --rng 11719619003825326796890461364078 \
    --factor 3

kwcoco stats $HOME/.cache/kwcoco/data/cifar10/cifar10-split1.kwcoco.json
kwcoco stats $HOME/.cache/kwcoco/data/cifar10/cifar10-split2.kwcoco.json


# TODO: this should be available via python -m vissl.__main__
python \
    $HOME/code/vissl/tools/run_distributed_engines.py \
    config=test/integration_test/quick_simclr.yaml \
    config.CHECKPOINT.DIR="./my_toy_training_v2" \
    config.DATA.TRAIN.DATA_SOURCES=[kwcoco] \
    config.DATA.TRAIN.DATASET_NAMES=[kwcoco] \
    config.DATA.TEST.DATA_SOURCES=[kwcoco] \
    config.DATA.TEST.DATASET_NAMES=[kwcoco] \
    config.DATA.TRAIN.DATA_PATHS=[$HOME/.cache/kwcoco/data/cifar10/cifar10-split1.kwcoco.json] \
    config.DATA.TEST.DATA_PATHS=[$HOME/.cache/kwcoco/data/cifar10/cifar10-split1.kwcoco.json] \



Debug Stuff
===========

cd $HOME/code/vissl

python tools/run_distributed_engines.py --help



kwcoco toydata shapes512

$HOME/.cache/kwcoco/data/cifar10/cifar10.kwcoco.json
$HOME/.cache/kwcoco/data/cifar100/cifar100.kwcoco.json

kwcoco stats /home/joncrall/.cache/kwcoco/data/cifar100/cifar100.kwcoco.json
kwcoco stats /home/joncrall/.cache/kwcoco/data/cifar10/cifar10.kwcoco.json

COCO_FPATH=/home/joncrall/.cache/kwcoco/demodata_bundles/shapes_512_jsskkpoafycsnj/data.kwcoco.json


# I wish we could pass a path to config=

ls $HOME/code/vissl/configs/config
tree --filelimit 10 $HOME/code/vissl/configs/config
find $HOME/code/vissl/configs/config -iname "*cifar*"

python \
    $HOME/code/vissl/tools/run_distributed_engines.py \
    config=benchmark/low_shot_transfer/cifar100/eval_resnet_8gpu_transfer_cifar100_low_tune.yaml \
    config.CHECKPOINT.DIR="$HOME/work/vissl/cifar10_train" \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=1 \
    config.MACHINE.DEVICE=gpu \
    confie.DATA.NUM_DATALOADER_WORKERS=0 \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=3 \
    config.DATA.TEST.BATCHSIZE_PER_REPLICA=3 \
    config.DATA.TRAIN.DATA_PATHS=[$HOME/.cache/kwcoco/data/cifar10/cifar10-split1.kwcoco.json] \
    config.DATA.TEST.DATA_PATHS=[$HOME/.cache/kwcoco/data/cifar10/cifar10-split1.kwcoco.json] \
    config.DATA.TRAIN.DATA_SOURCES=[kwcoco] \
    config.DATA.TRAIN.DATASET_NAMES=[kwcoco] \
    config.DATA.TEST.DATA_SOURCES=[kwcoco] \
    config.DATA.TEST.DATASET_NAMES=[kwcoco]



python -c "import vissl.data.dataset_catalog"

python run_distributed_engines.py \
    hydra.verbose=true \
    config=quick_1gpu_resnet50_simclr \
    config.DATA.TRAIN.DATA_SOURCES=[synthetic] \
    config.CHECKPOINT.DIR="./synth_checkpoints" \
    config.TENSORBOARD_SETUP.USE_TENSORBOARD=true

"""
