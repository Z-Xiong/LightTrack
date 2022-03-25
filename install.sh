#!/bin/bash

echo "****************** Installing lmdb ******************"
pip install lmdb -i https://mirror.baidu.com/pypi/simple

echo ""
echo ""
echo "****************** Installing pytorch with cuda10 ******************"
conda install -y pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch

echo ""
echo ""
echo "****************** Installing pandas ******************"
pip install pandas -i https://mirror.baidu.com/pypi/simple

echo ""
echo ""
echo "****************** Installing opencv ******************"
pip install opencv-python -i https://mirror.baidu.com/pypi/simple

echo ""
echo ""
echo "****************** Installing cython ******************"
pip install cython==0.29.21 -i https://mirror.baidu.com/pypi/simple

echo ""
echo ""
echo "****************** Installing skimage ******************"
pip install scikit-image -i https://mirror.baidu.com/pypi/simple

echo ""
echo ""
echo "****************** Installing pillow ******************"
pip install 'pillow<7.0.0' -i https://mirror.baidu.com/pypi/simple

echo ""
echo ""
echo "****************** Installing scipy ******************"
pip install scipy -i https://mirror.baidu.com/pypi/simple

echo ""
echo ""
echo "****************** Installing shapely ******************"
pip install shapely -i https://mirror.baidu.com/pypi/simple

echo ""
echo ""
echo "****************** Installing easydict ******************"
pip install easydict -i https://mirror.baidu.com/pypi/simple

echo ""
echo ""
echo "****************** Installing jpeg4py python wrapper ******************"
pip install jpeg4py -i https://mirror.baidu.com/pypi/simple

echo ""
echo ""
echo '****************** Intalling mpi4py ******************'
conda install -y mpi4py -i https://mirror.baidu.com/pypi/simple

echo ""
echo ""
echo '****************** Intalling ray and hyperopt ******************'
pip uninstall --yes grpcio -i https://mirror.baidu.com/pypi/simple
pip install --upgrade setuptools -i https://mirror.baidu.com/pypi/simple
pip install --no-cache-dir grpcio>=1.28.1 -i https://mirror.baidu.com/pypi/simple
conda install -c conda-forge -y grpcio
pip install ray==0.8.7 -i https://mirror.baidu.com/pypi/simple
pip install ray[tune] -i https://mirror.baidu.com/pypi/simple
pip install hyperopt -i https://mirror.baidu.com/pypi/simple

echo ""
echo ""
echo "****************** Installing numba ******************"
pip install numba -i https://mirror.baidu.com/pypi/simple

echo ""
echo ""
echo "****************** Installing EfficientNet ******************"
cd lib/models/EfficientNet-PyTorch
pip install -e .
cd ../../..

echo ""
echo ""
echo "****************** Installing thop tool for FLOPs and Params computing ******************"
pip install --upgrade git+https://github.com/Lyken17/pytorch-OpCounter.git

echo ""
echo ""
echo "****************** Installing GOT-10K related packages ******************"
cd toolkit
pip install -r requirements.txt
conda install -y matplotlib==3.0.2
cd ..

echo ""
echo ""
echo "****************** Installing Cream ******************"
pip install yacs -i https://mirror.baidu.com/pypi/simple
conda install -y tensorboard
pip install timm==0.1.20 -i https://mirror.baidu.com/pypi/simple
pip install git+https://github.com/sovrasov/flops-counter.pytorch.git
pip install git+https://github.com/Tramac/torchscope.git

echo ""
echo ""
echo "****************** Installing tensorboardX, colorama ******************"
pip install tensorboardX -i https://mirror.baidu.com/pypi/simple
pip install colorama -i https://mirror.baidu.com/pypi/simple

echo ""
echo ""
echo "****************** Installation complete! ******************"
