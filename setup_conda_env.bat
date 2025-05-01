@echo off
echo Creating conda environment for MicDysphagiaFramework...

REM Create and activate conda environment
call conda create -n micdys python=3.9 -y
call conda activate micdys

REM Install PyTorch with CUDA support
call conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

REM Install other dependencies via pip
pip install numpy>=1.19.5,<2.1.0
pip install scipy>=1.7.1,<1.14.0
pip install pyyaml>=6.0,<7.0
pip install tqdm>=4.62.3,<5.0.0
pip install scikit-learn>=1.0.2,<1.7.0
pip install pandas>=1.3.5,<2.3.0
pip install matplotlib>=3.5.1,<4.0.0
pip install seaborn>=0.11.2,<0.14.0
pip install librosa>=0.9.1,<0.12.0
pip install soundfile>=0.10.3,<0.14.0
pip install timm>=0.9.2,<1.1.0
pip install einops>=0.6.0,<0.9.0
pip install torchmetrics>=0.11.0,<1.8.0
pip install pytorch-lightning>=2.0.0,<2.6.0
pip install tensorboard>=2.11.0,<2.20.0
pip install python-dotenv>=0.21.0,<1.2.0
pip install joblib>=1.1.0,<1.5.0
pip install pillow>=9.3.0,<12.0.0
pip install wandb>=0.15.0,<0.20.0

echo Environment setup completed!
echo To activate the environment, use: conda activate micdys 