## 🗝️Let's Get Started!
### `A. Explanation of Code and Paper Relationship`

This code repository corresponds to the paper titled "Enhanced Remote Sensing Change Detection via Spatial-Reducing Cross-Attention and VMamba", which we submitted to The Visual Computer (manuscript ID: f397dd31-6bdb-421d-abaf-e1e1b9abe862). If you use this code or base your research on it, please cite the paper to support our work.

### `B. Installation`
The repo is based on the [VMama repo](https://github.com/MzeroMiko/VMamba), thus you need to install it first. The following installation sequence is taken from the VMamba repo. Also, note that the code in this repo runs under Linux system. We have not tested whether it works under other OS.

**Step 1: Clone the repository:**

Clone this repository and navigate to the project directory:
```bash
git clone https://github.com/QXL1220/SCACD.git
cd SCACD
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

python <project_path>/SCACD/script/train_MambaBCD.py

```
### `E. Inference Using Our Weights`

```bash
cd <project_path>/MambaCD/changedetection\

python <project_path>/SCACD/script/infer_MambaBCD.py
```

## ⚗️Main Results

* *The encoders for all the above SCACD models are the the VMamba architecture initialized with ImageNet pre-trained weight.*

* *Some of comparison methods are not open-sourced. Their accuracy and number of parameters are obtained based on our own implementation.*

## 🤝Acknowledgments
This project is based on VMamba ([paper](https://arxiv.org/abs/2401.10166), [code](https://github.com/MzeroMiko/VMamba)), ScanNet ([paper](https://arxiv.org/abs/2212.05245), [code](https://github.com/ggsDing/SCanNet)), xView2 Challenge ([paper](https://openaccess.thecvf.com/content_CVPRW_2019/papers/cv4gc/Gupta_Creating_xBD_A_Dataset_for_Assessing_Building_Damage_from_Satellite_CVPRW_2019_paper.pdf), [code](https://github.com/DIUx-xView/xView2_baseline)). Thanks for their excellent works!!

## 🙋Q & A
***For any questions, please feel free to [contact us.](yanweidong@nwpu.edu.cn)***
