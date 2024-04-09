## Dataset Overview

![channels_drive](./repo_pics/channels_drive.png)

![mask_label_drive](https://github.com/jzsherlock4869/PyTorch_Unet_for_RetinaVessel_Segmentation/blob/master/repo_pics/mask_label_drive.png "a. Original Image; b. Mask; c. Mannual Label")

```
tree -d DRIVE/
DRIVE/
├── test
│   ├── 1st_manual
│   ├── 2nd_manual
│   ├── images
│   └── mask
└── training
    ├── 1st_manual
    ├── images
    └── mask
```

## Preparation

Installing pip and nvidia-cuda-toolkit

```
sudo apt install python3-pip
sudo apt install nvidia-cuda-toolkit
```

Installing necessary libraries
```
pip3 install opencv-python
pip3 install tqdm
pip3 install tensorboardx
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install matplotlib
pip3 install scikit-image
pip3 install medpy
pip3 install h5py
```

## Training and Testing on DRIVE dataset

```
python train_model_mynet.py
python test_model_mynet.py
```

## Todo List

* Split evaluation set from train set
* Debug image augmentaion
* Add dice and cldice metrics for training and test 
* Undo resize image (currently give error)
* Sort and rearrange the codes for better usage.







