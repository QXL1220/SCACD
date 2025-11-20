## 🗝️Let's Get Started!
### `A. Installation`
The repo is based on the [VMama repo](https://github.com/MzeroMiko/VMamba), thus you need to install it first. The following installation sequence is taken from the VMamba repo. Also, note that the code in this repo runs under Linux system. We have not tested whether it works under other OS.

**Step 1: Clone the repository:**

Clone this repository and navigate to the project directory:
```bash
git clone https://github.com/ChenHongruixuan/MambaCD.git
cd MambaCD
```


**Step 2: Environment Setup:**

It is recommended to set up a conda environment and installing dependencies via pip. Use the following commands to set up your environment:

***Create and activate a new conda environment***

```bash
conda create -n SCACD
conda activate SCACD
```

***Install dependencies***

```bash
pip install -r requirements.txt
cd kernels/selective_scan && pip install .
```


***Dependencies for "Detection" and "Segmentation" (optional in VMamba)***

```bash
pip install mmengine==0.10.1 mmcv==2.1.0 opencv-python-headless ftfy regex
pip install mmdet==3.3.0 mmsegmentation==1.2.2 mmpretrain==1.2.0
```

### `C. Data Preparation`
***Binary change detection***

The three datasets [SYSU](https://github.com/liumency/SYSU-CD), [LEVIR-CD](https://chenhao.in/LEVIR/) and [WHU-CD](https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html) are used for binary change detection experiments. Please download them and make them have the following folder/file structure:
```
${DATASET_ROOT}   # Dataset root directory, for example: /home/username/data/SYSU
├── train
│   ├── T1
│   │   ├──00001.png
│   │   ├──00002.png
│   │   ├──00003.png
│   │   ...
│   │
│   ├── T2
│   │   ├──00001.png
│   │   ... 
│   │
│   └── GT
│       ├──00001.png 
│       ...   
│   
├── test
│   ├── ...
│   ...
│  
├── train.txt   # Data name list, recording all the names of training data
└── test.txt    # Data name list, recording all the names of testing data
```

```


### `D. Model Training`


```bash
cd <project_path>/SCACD

python /mnt/workspace/SCACD/script/train_MambaBCD.py

```
### `E. Inference Using Our Weights`

```bash
cd <project_path>/MambaCD/changedetection\

python /mnt/workspace/SCACD/script/infer_MambaBCD.py
```



## ⚗️Main Results


* *The encoders for all the above SCACD models are the the VMamba architecture initialized with ImageNet pre-trained weight.*

* *Some of comparison methods are not open-sourced. Their accuracy and number of parameters are obtained based on our own implementation.*


### `A. Binary Change Detection on SYSU`

| Method |  Overall Accuracy | F1 Score | IoU | Kappa Coefficient | Param | GFLOPs | ckpts
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [FC-EF](https://arxiv.org/abs/1810.08462) | 87.49 | 73.14 | 57.66 | 64.99 | 17.13 | 45.74 | -- |
| [SNUNet](https://github.com/likyoo/Siam-NestedUNet) | 97.83 | 74.70  | 59.62  |  73.57  | 10.21  | 176.36 | -- |
| [DSIFN](https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images) |  89.59 | 78.80 | 65.02 | 71.92  |  35.73 | 329.03 | -- |
| [SiamCRNN-101](https://github.com/ChenHongruixuan/SiamCRNN/tree/master/FCN_version) | 90.77 | 80.44 | 67.28 | 74.40 | 63.44 | 224.30  | -- |
| [HANet](https://github.com/ChengxiHAN) |    89.52 | 77.41 | 63.14 | 70.59 | 2.61  | 70.68 | -- |
| [CGNet](https://github.com/ChengxiHAN/CGNet-CD) |  91.19 | 79.92 | 66.55 | 74.31 | 33.68 | 329.58 | -- |
| [TransUNetCD](https://ieeexplore.ieee.org/document/9761892) |  90.88 | 80.09 | 66.79  | 74.18 | 66.79  | 74.18| -- |
| [SwinSUNet](https://ieeexplore.ieee.org/document/9736956) |  91.51 | 81.58 | 68.89  | 76.06| 39.28 | 43.50 | -- |
| [ChangeFormer V4](https://github.com/wgcban/ChangeFormer) | 90.12 | 78.81 | 65.03 | 72.37  | 33.61 | 852.53 | -- |
| [BIT-101](https://github.com/justchenhao/BIT_CD) |  90.76 | 79.41 | 65.84 | 73.47 | 17.13 | 45.74 | -- |
| MambaBCD-Tiny | 91.36 | 81.29 | 68.48	| 75.68 | 17.13 | 45.74 | -- |
| MambaBCD-Small | 92.39  | 83.36 | 71.46 | 78.43 | 49.94 | 114.82 | [[GDrive](https://drive.google.com/file/d/1ZEPF6CvvFynL-yu_wpEYdpHMHl7tahpH/view?usp=drive_link)][[BaiduYun](https://pan.baidu.com/s/1f8iwuKCkElU9rc24_ZzXBw?pwd=46p5)] |
| MambaBCD-Base |  92.30 | 83.11 | 71.10 | 78.13 | 84.70 | 179.32 | -- |

### `B. Binary Change Detection on LEVIR-CD`
| Method |  Overall Accuracy | F1 Score | IoU | Kappa Coefficient | Param | GFLOPs | ckpts
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [FC-EF](https://arxiv.org/abs/1810.08462) | 97.54   | 70.42 |  54.34  | 69.14  | 17.13 | 45.74 | -- |
| [SNUNet](https://github.com/likyoo/Siam-NestedUNet) | 97.83 | 74.70  | 59.62  |  73.57  | 10.21  | 176.36 | -- |
| [DSIFN](https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images) | 98.70 | 84.07 | 72.52 | 83.39  |  35.73 | 329.03 | -- |
| [SiamCRNN-101](https://github.com/ChenHongruixuan/SiamCRNN/tree/master/FCN_version) | 98.67 | 83.20 |  71.23  | 82.50 | 63.44 | 224.30  | -- |
| [HANet](https://github.com/ChengxiHAN) |   98.22 | 77.56  |  63.34 |  76.63 | 2.61  | 70.68 | -- |
| [CGNet](https://github.com/ChengxiHAN/CGNet-CD) |  98.63 |  83.68 |  71.94  |  82.97 | 33.68 | 329.58 | -- |
| [TransUNetCD](https://ieeexplore.ieee.org/document/9761892) |  98.66 | 83.63 | 71.86 | 82.93 | 28.37 | 244.54 | -- |
| [SwinSUNet](https://ieeexplore.ieee.org/document/9736956) |  98.92 | 85.60 | 74.82 | 84.98| 39.28 | 43.50 | -- |
| [ChangeFormer V4](https://github.com/wgcban/ChangeFormer) |  98.01 | 75.87 | 61.12 | 74.83  | 33.61 | 852.53 | -- |
| [BIT-101](https://github.com/justchenhao/BIT_CD) |  98.60 | 82.53 | 70.26 | 81.80 | 17.13 | 45.74 | -- |
| MambaBCD-Tiny | 99.03 | 88.04 | 78.63 | 87.53 | 17.13 | 45.74 | [[GDrive](https://drive.google.com/file/d/1AtiXBBCoofi1e5g4STYUzBgJ1fYN4VhN/view?usp=drive_link)][[BaiduYun](https://pan.baidu.com/s/13dGC_J-wyIfoPwoPJ5Uc6Q?pwd=8ali)] |
| MambaBCD-Small | 99.02 | 87.81 | 78.27 | 87.30 | 49.94 | 114.82 | -- |
| MambaBCD-Base |  99.06 | 88.39 | 79.20 | 87.91 | 84.70 | 179.32 | -- |

### `C. Binary Change Detection on WHU-CD`
| Method |  Overall Accuracy | F1 Score | IoU | Kappa Coefficient | Param | GFLOPs | ckpts
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [FC-EF](https://arxiv.org/abs/1810.08462) | 98.87  | 84.89  | 73.74  | 84.30 | 17.13 | 45.74 | -- |
| [SNUNet](https://github.com/likyoo/Siam-NestedUNet) |  99.10  | 87.70 | 78.09 |  87.23 | 10.21  | 176.36 | -- |
| [DSIFN](https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images) | 99.31  |  89.91| 81.67| 89.56 |  35.73 | 329.03 | -- |
| [SiamCRNN-101](https://github.com/ChenHongruixuan/SiamCRNN/tree/master/FCN_version) | 99.19 | 89.10 | 80.34 | 88.68 | 63.44 | 224.30  | -- |
| [HANet](https://github.com/ChengxiHAN) |  99.16 | 88.16 | 78.82 | 87.72 | 2.61  | 70.68 | -- |
| [CGNet](https://github.com/ChengxiHAN/CGNet-CD) |  99.48 | 92.59 | 86.21 | 92.33 | 33.68 | 329.58 | -- |
| [TransUNetCD](https://ieeexplore.ieee.org/document/9761892) |  99.09 | 87.79 | 78.44 | 87.44 | 28.37 | 244.54 | -- |
| [SwinSUNet](https://ieeexplore.ieee.org/document/9736956) |  99.50 | 93.04 | 87.00 | 92.78 | 39.28 | 43.50 | -- |
| [ChangeFormer V4](https://github.com/wgcban/ChangeFormer) |  99.10 | 87.39 | 77.61 | 86.93 | 33.61 | 852.53 | -- |
| [BIT-101](https://github.com/justchenhao/BIT_CD) |  99.27 | 90.04 | 81.88 | 89.66 | 17.13 | 45.74 | -- |
| MambaBCD-Tiny |  99.57 | 94.09 | 88.84 | 93.87 | 17.13 | 45.74 | [[GDrive](https://drive.google.com/file/d/1ZLKXhGKgnWoyS0X8g3HS45a3X1MP_QE6/view?usp=drive_link)][[BaiduYun](https://pan.baidu.com/s/1DhTedGZdIC80y06tog1xbg?pwd=raf0)] |
| MambaBCD-Small |  99.57 | 94.06 | 88.79 | 93.84 | 49.94 | 114.82 | -- |
| MambaBCD-Base |  99.58 | 94.19 | 89.02 | 93.98 | 84.70 | 179.32 | [[GDrive]](https://drive.google.com/file/d/1K7aSuT3os7LR9rUvoyVNP-x0hWKZocrn/view?usp=drive_link)][[BaiduYun](https://pan.baidu.com/s/1o6Z6ecIJ59K9eB2KqNMD9w?pwd=4mqd)] |


## 🤝Acknowledgments
This project is based on VMamba ([paper](https://arxiv.org/abs/2401.10166), [code](https://github.com/MzeroMiko/VMamba)), ScanNet ([paper](https://arxiv.org/abs/2212.05245), [code](https://github.com/ggsDing/SCanNet)), xView2 Challenge ([paper](https://openaccess.thecvf.com/content_CVPRW_2019/papers/cv4gc/Gupta_Creating_xBD_A_Dataset_for_Assessing_Building_Damage_from_Satellite_CVPRW_2019_paper.pdf), [code](https://github.com/DIUx-xView/xView2_baseline)). Thanks for their excellent works!!

## 🙋Q & A
***For any questions, please feel free to [contact us.](mailto:Qschrx@gmail.com)***
