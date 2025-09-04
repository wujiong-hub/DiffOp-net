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

# build lddmm related functions
cd ./Modules/PytorchLDDMM
python setup.py install 
```


## Training
```python
mkdir logs
python networkTraining.py Config_adni.ini 48 0
```
## Testing
python networkRegistration.py FCN_registration_adni.ini 48 ./logs/path_to_checkpoint





