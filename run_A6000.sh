# NVTE_TORCH_COMPILE=0 
# NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 
# NVTE_FLASH_ATTN=1 
# NVTE_FWD_LAYERNORM_SM_MARGIN=0 
# NVTE_BWD_LAYERNORM_SM_MARGIN=0
# NVTE_BIAS_GELU_NVFUSION=0 
# NVTE_BIAS_DROPOUT_FUSION=0
# CUDA_DEVICE_MAX_CONNECTIONS=1 

accelerate launch \
--config_file configs/A6000.yaml \
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