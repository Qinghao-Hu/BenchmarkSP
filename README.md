<!-- <div align="center"> -->

### Run on Container

```shell
docker pull tonyhao96/benchmark:v0.1.0
docker run --name test --shm-size=16g --gpus all --net=host --pid=host -it tonyhao96/benchmark:v0.1.0
```

```shell
cd /opt/BenchmarkSP

```


## 1. Hybrid, Ulysses, Ring (`sequence_parallel` folder)

Use ZeRO-3 + different sequence parallelism strategies.

### a. Setup

```shell
conda create -n bench python=3.10 
conda activate bench
# this is optional if you prefer to system built-in nvcc.
conda install -c nvidia cuda-toolkit -y

wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

pip install -r requirements.txt
```

### b. Run various parallelism strategies

> [!NOTE]
>  💡Support parallelism strategies
> `hybrid`, `ring`, `zigzag`, `striped`, `ulysses`, `lightseq`
>
> Default: `sequence_length_per_gpu=4000`,`batch_size=1`, `sequence_parallel_degree=WORLD_SIZE`
>
> Example for `hybrid` (--ulysses_degree only works for `hybrid`):
>
> --parallel_mode hybrid \
>
> --ulysses_degree 8 \


1. ***Single Node***

> srun -p llm_s --job-name=benchmark -n 1 --gres=gpu:8 --ntasks-per-node=1 bash srun_single.sh

2. ***Multi Nodes***

Please modify the `num_machines` in `configs/multi_node.yaml`. Default is 2 nodes (16 GPUs).

> srun -p xxx --job-name=benchmark -n 2 --gres=gpu:8 --ntasks-per-node=1 bash srun.sh

=====================================================================
## 2. Megatron-LM (`megatron` folder)

Use ZeRO-1 + Context-Parallelism (i.e. Zigzag Ring).


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

3. NVIDIA [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)


```bash
cd Megatron-LM

pip install -e .
```

### b. Run Megatron-LM

```shell
cp ../BenchmarkSP/megatron/* ./

srun -p xxx --job-name=megatron -n 1 --gres=gpu:8 --ntasks-per-node=1 bash srun.sh
```

You can customize `SEQ_LENGTH_PER_GPU`, `context-parallel-size` and other parameters in `srun.sh`.