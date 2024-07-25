SEQ_LENGTH_PER_GPU=2048
GPUS_PER_NODE=8
SEQ_LENGTH=$(($n_node*$SEQ_LENGTH_PER_GPU*$GPUS_PER_NODE))

NODE_RANK=$SLURM_PROCID

LOG_DIR=.

accelerate launch \
--config_file configs/multi_node.yaml \
--main_process_ip $master_addr \
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
--parallel_mode ulysses \
--ulysses_degree 32 \
# --parallel_mode zigzag
# --parallel_mode ulysses

# --parallel_mode hybrid \
# --ulysses_degree 8

# --parallel_mode hybrid \
# --ulysses_degree 4 \
# --use_ulysses_lowdim

# srun -p llm_s --job-name=vila -n 2 --gres=gpu:8 --ntasks-per-node=1 bash srun.sh

# --error=log/error_log.txt
