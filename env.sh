# GAPartNet environment setup
export PATH="/home/rai-laptop/.local/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="12.0"
export FORCE_CUDA=1
export MAX_JOBS=4
export PATH="/home/rai-laptop/actance/GAPartNet/.venv/bin:$PATH"
