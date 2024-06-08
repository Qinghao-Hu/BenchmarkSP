if [ -z "$1" ]; then
  echo "Usage: $0 <machine_rank>"
  exit 1
fi

MACHINE_RANK=$1
echo "MACHINE_RANK: $MACHINE_RANK"

accelerate launch \
--config_file  configs/mit.yaml \
--machine_rank $MACHINE_RANK \
train.py \
--batch-size 1 \
--gradient-accumulate-every 1  \
--seed 123 \
--max-train-steps 30  \
--learning-rate 2e-5  \
--model meta-llama/Llama-2-7b-hf  \
--seq-length 64_000 \
--parallel_mode hybrid

# srun -p llm_s --job-name=vila -n 2 --gres=gpu:8 --ntasks-per-node=1 bash run.sh