## A Study of Zero-cost proxies for Remote Sensing Image Segmentation

This repository contains code for paper [A Study of Zero-cost proxies for Remote Sensing Image Segmentation].

### Prerequisites
* Python 3.8
* Pytorch 1.8.1
* torch-scatter `pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.8.0+${CUDA}.html`
* torch-sparse `pip install torch-sparse==0.6.10 -f https://data.pyg.org/whl/torch-1.8.0+${CUDA}.html`
* torch-cluster `pip install torch-cluster==1.6.0 -f https://data.pyg.org/whl/torch-1.8.0+${CUDA}.html`
* torch-spline-conv `pip install torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-1.8.0+${CUDA}.html`
* torch-geometric `pip install torch-geometric==2.0.4`

```${CUDA} refers to cpu, cu101, cu102, cu111```

### Environments
* Ubuntu 18.04
* cuda 10.2
* cudnn 8.1.1

### Installation

<!-- Installing pytorch_geometric. -->

1. See [detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html) to install required packages.
2. `git clone https://github.com/auroua/nas_seg_detectron2.git`
3. `python -m pip install -e nas_seg_detectron2`


### Getting Started
#### 1. Datasets
This project uses the WHU Building and Massachusetts road datasets.

#### 2. Running Segmentation Algorithms DeepLab V3+
```bash
   # Train model  
   # As using the sync bn layer, the parameter --num-gpus must be 2
   python train_net_seg.py --config-file /home/albert_wei/WorkSpaces/nas_seg_detectron2/configs/Segmentation/PSPNet/Remote-SemanticSegmentation/Base-PSPNet-OS16-Semantic.yaml --num-gpus 2 --num-machines 1
```

```bash
   # Evaluate model
   python train_net_seg.py --config-file /home/albert_wei/WorkSpaces/nas_seg_detectron2/configs/Segmentation/PSPNet/Remote-SemanticSegmentation/Base-PSPNet-OS16-Semantic.yaml --num-gpus 2 --num-machines 1 --eval-only --resume
```

#### 3. Running Segmentation Algorithms PSPNet
```bash
   # Train model 
   # As using the sync bn layer, the parameter --num-gpus must be 2
   python train_net_seg.py --config-file /home/albert_wei/WorkSpaces/nas_seg_detectron2/configs/Segmentation/DeepLab/Remote-SemanticSegmentation/Base-DeepLabV3-plus-OS16-Semantic.yaml --num-gpus 2 --num-machines 1
```

```bash
   # Evaluate model
   python train_net_seg.py --config-file /home/albert_wei/WorkSpaces/nas_seg_detectron2/configs/Segmentation/DeepLab/Remote-SemanticSegmentation/Base-DeepLabV3-plus-OS16-Semantic.yaml --num-gpus 2 --num-machines 1 --eval-only --resume
```

#### 4. Using `NPENAS-NP` to search architecture from `SEG101`
```bash
   export DETECTRON2_DATASETS='/home/albert_wei/fdisk_a/datasets_train_seg/'
   cd nas_seg_detectron2/tools_nas
   # --config_file: config file path for neural architecture search
   # --gpus: using how many to finish search
   # --save_dir: save search results
   python train_nas_open_domain.py --config_file '/home/albert_wei/WorkSpaces/nas_seg_detectron2/configs_nas/seg/seg_101_npenas.yaml' --gpus 2 --save_dir '/home/albert_wei/fdisk_b/'
```

#### 5. Train the searched architecture
```bash
   cd nas_seg_detectron2/tools_nas
   # Train the searched architecture
   # You have to modify the following parameters in the config-file
   # SEARCHED_ARCHITECTURE: point to the searched architecture's folder, e.g., '/home/albert_wei/fdisk_a/train_output_remote/npenas_seg101_mass_road_20_epochs_2/1e4d24eb52f67424eabfe070ffbaee7ac2f31ca4f2e19a3c87680fbb4ed8167a'
   # OUTPUT_DIR: point to the folder where to save the training results
   python train_searched_architecture.py --config-file /home/albert_wei/WorkSpaces/nas_seg_detectron2/configs/Segmentation/Seg101/Remote-SemanticSegmentation/Base-Seg101-OS16-Semantic.yaml --num-gpus 1 --num-machines 1
   
   # Evaluate searched architecture
   python train_net_seg.py --config-file /home/albert_wei/WorkSpaces/nas_seg_detectron2/configs/Segmentation/PSPNet/Remote-SemanticSegmentation/Base-PSPNet-OS16-Semantic.yaml --num-gpus 2 --num-machines 1 --eval-only --resume
```

### Modifications to detectron2
1. Adding python file detectron2/data/datasets/builtin_remote to include the mass road and whu building datasets.

### Acknowledge
1. [detectron2](https://github.com/facebookresearch/detectron2)
2. [NPENAS](https://github.com/auroua/NPENASv1)
3. [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric)
4. [Zero-Cost-NAS](https://github.com/SamsungLabs/zero-cost-nas)


### Citation
If you find this project useful for your research, please cite our paper:
```bibtex
    @inproceedings{weistudy,
      title={A Study of Zero-Cost Proxies for Remote Sensing Image Segmentation},
      author={Wei, Chen and Guo, Tai Kai and Tang, Ping Yi and Ge, Yao Jun and Liang, Jimin},
      booktitle={First Conference on Automated Machine Learning (Late-Breaking Workshop)}
    }
```

## Contact
Chen Wei

email: weichen_3@stu.xidian.edu.cn
