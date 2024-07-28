SEQ_LENGTH_PER_GPU=2048
GPUS_PER_NODE=8
SEQ_LENGTH=$(($SEQ_LENGTH_PER_GPU*$GPUS_PER_NODE))

LOG_DIR=.
NSYS_ITER=10 # -1: off, >0 to enable recommend: 10
NSYS_ITER_RANGE=2

# /workspace/target-linux-x64/nsys profile --force-overwrite true -y 15 -d 3 -o ./nsys_reports/ours-8GPU_ring_512 accelerate launch \
accelerate launch \
--config_file configs/single_node_fsdp.yaml \
train.py \
--batch-size 1 \
--gradient-accumulate-every 1  \
--seed 123 \
--max-train-steps 15  \
--learning-rate 2e-5  \
--model NousResearch/Llama-2-7b-hf  \
--seq-length $SEQ_LENGTH \
--parallel_mode hybrid \
--ulysses_degree 1
# --parallel_mode zigzag
# --parallel_mode ulysses

# --parallel_mode hybrid \
# --ulysses_degree 8

# --parallel_mode hybrid \
# --ulysses_degree 4 \
# --use_ulysses_lowdim

# srun -p llm_s --job-name=vila -n 2 --gres=gpu:8 --ntasks-per-node=1 bash srun.sh

# --error=log/error_log.txt
