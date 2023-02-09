#!/bin/bash

source /etc/profile.d/lmod.sh

module purge
module load anaconda/3.9
module load vscode-server/20220909

nvidia-smi

LOCAL=/tmp
# Install skyline
python3 -m venv $LOCAL/centml_tools
cd $LOCAL/centml_tools
module unload  anaconda/3.9
source $LOCAL/centml_tools/bin/activate
echo Python: `which python`: `python --version`

pip install torch torchvision  --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu116
pip install ipywidgets jupyterlab notebook

git clone https://github.com/NVIDIA/apex
cd apex
export CUDA_HOME=/pkgs/cuda-11.6/
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cuda_ext"  --global-option="--deprecated_fused_adam" ./

# Skyline
git clone https://github.com/centml/skyline.git
pip install -e skyline --extra-index-url https://download.pytorch.org/whl/cu116

# Habitat
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pkgs/cuda-11.6/extras/CUPTI/lib64/
pip install http://centml-releases.s3-website.us-east-2.amazonaws.com/habitat/wheels-cu116/habitat_predict-1.0.0-0+cu116-py39-none-any.whl

# Visual Studio Extention
curl http://centml-releases.s3-website.us-east-2.amazonaws.com/skyline-vscode/skyline-vscode-0.0.1.vsix --output skyline-vscode-0.0.1.vsix

# Start code server
#port=$(python -c "import socket;s = socket.socket(socket.AF_INET, socket.SOCK_STREAM);s.bind(('', 0));addr = s.getsockname();print(addr[1]);s.close()")

skyline interactive  & #--port $port &
code-server --disable-telemetry --accept-server-license-terms --server-data-dir ./vscode_data 


