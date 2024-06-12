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


torchrun --nproc-per-node 8 run_train.py \
--batch_size 1 \
--max_train_steps 30 \
--seq_length 4_000 \
--tensor_parallel_size 1 \
--pipeline_parallel_size 1 \
--context_parallel_size 8


# # srun -p llm_s --job-name=megatron -n 1 --gres=gpu:8 --ntasks-per-node=1 bash srun.sh
