# CentML Workshop at Vector Institute

This repository contains the artifacts for the workshop that CentML team ran at the Vector Institute on Feb 10th, 2023.

You can find workshop slides at [here](slides/CentML-Vector.pdf).


## Setting up skyline on Vector's GPU cluster

First, acquire a GPU interactive session
```bash
srun --gres=gpu:1 -c 8 --mem 16G --pty bash
```

Clone this repository
```
git clone https://github.com/CentML/VectorWorkshop.git
```

Install all the dependencies and launch code server
```
cd VectorWorkshop
bash setup.sh
```

## Using the profiler

If you never used VSCode tunning before, follow the last output from the terminal to authenticate via Github

Open your local vscode instance and install Extenstion: Remote - Tunnels (Again, you might need to authenticate)

Open Command Pallet and type "Remote-Tunnels: Connect to Tunnel, you should see `gpuxxx Online`"

Click the option, this will transfer you to the actual VSCode instance running on the GPU instance.



