<!-- <div align="center"> -->



## 1. Hybrid, Ulysses, Ring (`sequence_parallel` folder)

Use ZeRO-3 + different sequence parallelism strategies.

### a. Setup






=================================================================================
## 2. Megatron-LM (`megatron` folder)

Use ZeRO-1 + Context-Parallelism.


### a. Setup

> [!NOTE]
>
> 1.Can share the same conda environment with above `hybrid`.
>
> 2.Please compile and install `apex`, `transformer-engine` first.


1. NVIDIA [APEX](https://github.com/NVIDIA/apex).
   
   `pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./`

   + May need to skip version checking in `def check_cuda_torch_binary_vs_bare_metal(cuda_dir):` in `setup.py`

2. NVIDIA [Transformer Engine](https://github.com/NVIDIA/TransformerEngine)

```bash
source ~/.bashrc
conda activate torch23
which python

cd ~/workdir/TransformerEngine

export CUDA_HOME=/mnt/petrelfs/share/cuda-12.2
export CUDA_PATH=/mnt/petrelfs/share/cuda-12.2
export CUDACXX=/mnt/petrelfs/share/cuda-12.2/bin/nvcc
export CXX=/mnt/petrelfs/share/gcc/gcc-9.4.0/bin/g++
export CC=/mnt/petrelfs/share/gcc/gcc-9.4.0/bin/gcc
export LD=/mnt/petrelfs/share/gcc/gcc-9.4.0/bin/g++
export LD_LIBRARY_PATH=/mnt/petrelfs/share/gcc/mpc-0.8.1/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/mnt/petrelfs/share/gcc/mpfr-2.4.2/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/mnt/petrelfs/share/gcc/gmp-4.3.2/lib:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=/mnt/petrelfs/share/gcc/mpc-0.8.1/include:$C_INCLUDE_PATH
export C_INCLUDE_PATH=/mnt/petrelfs/share/gcc/mpfr-2.4.2/include:$C_INCLUDE_PATH
export PATH=/mnt/petrelfs/share/cmake-3.13.4/bin:$PATH
export PATH=/mnt/petrelfs/share/cuda-12.2/bin/nvcc:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2-cudnn8.9/lib64:$LD_LIBRARY_PATH
export CUDNN_INCLUDE_DIR=/mnt/petrelfs/share/cudnn-8.9.6-cuda12/include
export CUDNN_PATH=/mnt/petrelfs/share/cudnn-8.9.6-cuda12/
# export TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"

pip install .
```

3. Megatron-LM


```bash
cd Megatron-LM

pip install -e .
```