#!/bin/bash

env_name=$1

conda create -n $env_name
eval "$(conda shell.bash hook)"
conda activate $env_name

echo "In ${env_name}"

which python
python --version

conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
pip install einops

echo "--------- Installing TRIQS CTQMC---------"
conda install -y -c conda-forge triqs=3.1.1
echo "--------- Installing CTQMC ---------"
conda install -y -c conda-forge triqs_cthyb
conda install -y pandas

pip install scikit-learn
pip install h5py
pip install seaborn
pip install pytest
pip install PyYAML

echo "--------- Install Transformer for SIAM --------------"

pip install .