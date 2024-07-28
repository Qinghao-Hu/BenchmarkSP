cd /lustre/fsw/portfolios/nvr/users/dachengl/BenchmarkSP
source /lustre/fsw/portfolios/nvr/users/dachengl/anaconda3/bin/activate
conda activate bench

GPUS_PER_NODE=8
n_node=$SLURM_JOB_NUM_NODES
SEQ_LENGTH=$(($n_node*$SEQ_LENGTH_PER_GPU*$GPUS_PER_NODE))

NODE_RANK=$SLURM_PROCID
echo "parallel mode: $parallel_mode"

accelerate launch \
--config_file configs/multi_node_fsdp.yaml \
--main_process_ip $master_addr \
--main_process_port 12345 \
--num_processes $(($n_node*$GPUS_PER_NODE)) \
--num_machines $n_node \
--machine_rank $SLURM_PROCID \
train.py \
--batch-size 1 \
--gradient-accumulate-every 1  \
--seed 123 \
--max-train-steps 3  \
--learning-rate 2e-5  \
--model NousResearch/Llama-2-7b-hf  \
--seq-length $SEQ_LENGTH \
--parallel_mode $parallel_mode \
--ulysses_degree $ULYSSES_DEGREE
