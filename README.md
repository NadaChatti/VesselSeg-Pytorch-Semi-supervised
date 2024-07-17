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
<!-- see code/parameters.py or run --help for arguments. -->
python train_model_mynet.py
python test_model_mynet.py
```







