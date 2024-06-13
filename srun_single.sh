export TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"
export CXX=/mnt/petrelfs/share/gcc/gcc-9.4.0/bin/g++ 
export CC=/mnt/petrelfs/share/gcc/gcc-9.4.0/bin/gcc 
export LD=/mnt/petrelfs/share/gcc/gcc-9.4.0/bin/g++  
export LD_LIBRARY_PATH=/mnt/petrelfs/share/gcc/mpc-0.8.1/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/mnt/petrelfs/share/gcc/mpfr-2.4.2/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/mnt/petrelfs/share/gcc/gmp-4.3.2/lib:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=/mnt/petrelfs/share/gcc/mpc-0.8.1/include:$C_INCLUDE_PATH
export C_INCLUDE_PATH=/mnt/petrelfs/share/gcc/mpfr-2.4.2/include:$C_INCLUDE_PATH
export CUDA_HOME=/mnt/petrelfs/share/cuda-12.1
export CUDA_PATH=/mnt/petrelfs/share/cuda-12.1
export CUDACXX=/mnt/petrelfs/share/cuda-12.1/bin/nvcc

source ~/.bashrc
conda activate torch23
which python

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

n_node=$SLURM_JOB_NUM_NODES
echo "number of nodes:" $n_node
echo "node rank:" $SLURM_PROCID

ip_addr=$(echo "$master_addr" | sed 's/HOST-\([0-9]*\)-\([0-9]*\)-\([0-9]*\)-\([0-9]*\)/\1.\2.\3.\4/')
echo $ip_addr

SEQ_LENGTH_PER_GPU=4000
GPUS_PER_NODE=8
SEQ_LENGTH=$(($n_node*$SEQ_LENGTH_PER_GPU*$GPUS_PER_NODE))

accelerate launch \
--config_file configs/single_node.yaml \
--main_process_ip $MASTER_ADDR \
--main_process_port 12345 \
--machine_rank $SLURM_PROCID \
train.py \
--batch-size 1 \
--gradient-accumulate-every 1  \
--seed 123 \
--max-train-steps 30  \
--learning-rate 2e-5  \
--model NousResearch/Llama-2-7b-hf  \
--seq-length $SEQ_LENGTH \
--parallel_mode zigzag
# --parallel_mode hybrid \
# --ulysses_degree 8
# --parallel_mode ulysses

# --parallel_mode hybrid \
# --ulysses_degree 8

# --parallel_mode hybrid \
# --ulysses_degree 4 \
# --use_ulysses_lowdim

# srun -p llm_s --job-name=vila -n 1 --ntasks-per-node=1 --gres=gpu:8 bash srun_single.sh


# --error=log/error_log.txt  --ntasks-per-node=1 -w HOST-10-140-60-188