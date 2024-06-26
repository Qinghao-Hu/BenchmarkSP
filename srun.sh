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
--config_file configs/multi_node.yaml \
--main_process_ip $MASTER_ADDR \
--main_process_port 12345 \
--num_machines $n_node \
--num_processes $(($n_node*$GPUS_PER_NODE)) \
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

# srun -p llm_s --job-name=vila -n 2 --gres=gpu:8 --ntasks-per-node=1 bash srun.sh

# --error=log/error_log.txt