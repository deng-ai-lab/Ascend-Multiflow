pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
pip install torch-geometric

pip install -r requirements.txt

# * for ESMFold
pip install "fair-esm[esmfold]"
conda install -c nvidia cuda-nvcc=12.1
conda install -c nvidia cuda-toolkit=12.1
conda install -c nvidia cuda-cccl=12.1
conda install -c conda-forge gcc=11 gxx=11
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
cd openfold
pip install -e . --no-build-isolation
cd ..
