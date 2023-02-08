source /etc/profile.d/lmod.sh

module purge
module load anaconda/3.9
module load vscode-server/20220909

nvidia-smi

# Install skyline
python3 -m venv $HOME/centml_tools
module unload  anaconda/3.9
source $HOME/centml_tools/bin/activate
echo CUDA  : `which nvcc`: `nvcc --version`
echo Python: `which python`: `python --version`

cd $HOME/centml_tools/

# Skyline
git clone https://github.com/centml/skyline.git
pip install -e skyline

# Habitat
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pkgs/cuda-11.6/extras/CUPTI/lib64/
pip install http://centml-releases.s3-website.us-east-2.amazonaws.com/habitat/wheels-cu116/habitat_predict-1.0.0-0+cu116-py39-none-any.whl

# Visual Studio Extention
cd ./project
curl http://centml-releases.s3-website.us-east-2.amazonaws.com/skyline-vscode/skyline-vscode-0.0.1.vsix --output skyline-vscode-0.0.1.vsix
#code-server --install-extension skyline-vscode-0.0.1.vsix

# Start code server
skyline interactive &
code-server 


