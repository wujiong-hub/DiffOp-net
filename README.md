# DiffOp-net
DiffOp-net: A Differential Operator-based Fully Convolutional Network for Unsupervised Deformable Image Registration


## Environment

```bash
# Clone the repository
git clone https://github.com/wujiong-hub/DiffOp-net.git

# Create and activate a new conda environment
conda create -n diffopnet python=3.8
conda activate diffopnet

# Install PyTorch (please modify according to your server's CUDA version)
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Install MONAI and additional dependencies
pip install 'monai[all]'
pip install -r requirements.txt
```


## Training
```python
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=1238 train.py --data_root_path DATA_DIR --dist True --uniform_sample
```
## Testing
1. Do the inference process directly adopt our trained model:
```python
cd pretrained_weights/
wget https://huggingface.co/jwu2009/CDPDNet/resolve/main/cdpdnet.pth
cd ../
CUDA_VISIBLE_DEVICES=0 python test.py --data_root_path DATA_DIR --resume pretrained_weights/cdpdnet.pth --store_result 
```

2. Do the inference process using your own trained model:
```python
CUDA_VISIBLE_DEVICES=0 python test.py --data_root_path DATA_DIR --resume CHECKPOINT_PATH --store_result 
```






