"""
My testing commands

pip install git+https://github.com/facebookresearch/ClassyVision.git
pip install apex


pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html


git clone https://github.com/NVIDIA/apex $HOME/code/apex
cd $HOME/code/apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .


cd $HOME/code/vissl

python tools/run_distributed_engines.py --help


kwcoco toydata shapes8

COCO_FPATH=/home/joncrall/.cache/kwcoco/demodata_bundles/shapes_8_tphzxqtzakcghy/data.kwcoco.json

python \
    tools/run_distributed_engines.py \
    config=test/integration_test/quick_simclr.yaml \
    config.DATA.TRAIN.DATASET_NAMES=[kwcoco]
    config.DATA.TRAIN.DATA_SOURCES=[kwcoco]
    config.DATA.TRAIN.DATA_PATHS=[$COCO_FPATH]

    config.DATA.TRAIN.DATA_SOURCES=[synthetic] \
    config.CHECKPOINT.DIR="./checkpoints"


python run_distributed_engines.py \
    hydra.verbose=true \
    config=quick_1gpu_resnet50_simclr \
    config.DATA.TRAIN.DATA_SOURCES=[synthetic] \
    config.CHECKPOINT.DIR="./checkpoints" \
    config.TENSORBOARD_SETUP.USE_TENSORBOARD=true

"""
