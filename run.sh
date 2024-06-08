accelerate launch \
--config_file  configs/A6000.yaml \
train.py \
--batch-size 1 \
--gradient-accumulate-every 1  \
--seed 123 \
--max-train-steps 30  \
--learning-rate 2e-5  \
--model meta-llama/Llama-2-7b-hf  \
--seq-length 4_000 \
--parallel_mode zigzag

# --parallel_mode hybrid \
# --ulysses_degree 4 \
# --use_ulysses_lowdim

# srun -p llm_s --job-name=vila -n 2 --gres=gpu:8 --ntasks-per-node=1 bash run.sh