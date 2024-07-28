which python
source /root/miniconda3/bin/activate
cd /workspace/BenchmarkSP/

n_node=$SLURM_JOB_NUM_NODES

SEQ_LENGTH_PER_GPU=SEQ_LENGTH_PER_GPU
GPUS_PER_NODE=8
SEQ_LENGTH=$(($n_node*$SEQ_LENGTH_PER_GPU*$GPUS_PER_NODE))

NODE_RANK=$SLURM_PROCID

LOG_DIR=.
NSYS_ITER=10 # -1: off, >0 to enable recommend: 10
NSYS_ITER_RANGE=2

if (( $NSYS_ITER >= 0 )); then
    mkdir -p ${LOG_DIR}/nsys_reports
    NSYS_CMD="/workspace/target-linux-x64/nsys profile --force-overwrite true -o ${LOG_DIR}/nsys_reports/ours-32GPU-hybrid -y 45 -d 5 --capture-range=cudaProfilerApi"
else
    NSYS_CMD=""
fi

#/workspace/target-linux-x64/nsys profile --force-overwrite true -o ${LOG_DIR}/nsys_reports/ours-32GPU_hybrid_$SLURM_PROCID -y 45 -d 5 --capture-range=cudaProfilerApi 
accelerate launch \
--config_file configs/multi_node.yaml \
--main_process_ip $master_addr \
--main_process_port 12345 \
--machine_rank $SLURM_PROCID \
train.py \
--batch-size 1 \
--gradient-accumulate-every 1  \
--seed 123 \
--max-train-steps 3  \
--learning-rate 2e-5  \
--model NousResearch/Llama-2-7b-hf  \
--seq-length $SEQ_LENGTH \
--parallel_mode hybrid \
--ulysses_degree $ULYSESS_DEGREE \
# --parallel_mode zigzag
# --parallel_mode ulysses

# --parallel_mode hybrid \
# --ulysses_degree 8

# --parallel_mode hybrid \
# --ulysses_degree 4 \
# --use_ulysses_lowdim

# srun -p llm_s --job-name=vila -n 2 --gres=gpu:8 --ntasks-per-node=1 bash srun.sh

# --error=log/error_log.txt
