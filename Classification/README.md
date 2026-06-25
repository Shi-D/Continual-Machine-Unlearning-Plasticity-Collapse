# Continual Unlearning Plasticity Collapse for Classification


This is the official repository for Continual Unlearning Plasticity Collapse for Classification.


## Setup

Installation requirements are described in `requirements.txt`.

- Use pip:

  ```
  pip install -r requirements.txt
  ```

- Use anaconda:

  ```
  conda install --file requirements.txt
  ```

## Getting Started

- Get an original model:

    ```
    python3 train.py --dir=./ckpt/ --dataset=[dataset name] --unlearn_type=random --data_path=data --transform=ResNet --model=PreResNet110 --epochs=200
    ```

- To complete continual unlearning and record PCU analysis, use the following command:

    ```
    python3 train.py --dir=[save dir] --dataset=[dataset name] --unlearn_method=[unlearn method] --unlearn_type=[class/random] --data_path=[data path] --transform=VGG --model=VGG16BN --lr=0.01 --forget_ratio=1 --epochs=3 --batch_size=256 --wd=5e-4 --save_freq=10 --original_pth=[original model's ckpt] --forget_times=180 --seed=42
    ```












