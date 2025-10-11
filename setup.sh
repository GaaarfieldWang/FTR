#!/bin/bash

conda create -n ftr python=3.10 -y
conda activate ftr

pip install mujoco==2.3.0
pip install dm_control==1.0.8
pip uninstall dm_control -y
cd src/env/dm_control; python setup.py install; cd ../../..

pip install -r requirements.txt

pip install src/env/dmc2gym

conda install -c conda-forge evdev -y
pip install src/env/robosuite

# pip install src/mobile_sam

export MUJOCO_GL=glfw

conda install -c conda-forge gcc=11 gxx=11 -y
# conda install -c conda-forge gcc=12 gxx=12 -y

cd src/segment-anything-2-real-time
pip install -e .
cd ../..
