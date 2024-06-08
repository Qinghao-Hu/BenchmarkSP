from typing import Any
from enum import Enum


class HardwareType(Enum):
    A100 = "a100"
    H100 = "h100"
    RTX3090 = "rtx3090"

    @property
    def max_tflops(self):
        """
        Mappings for Maximum Throughput numbers of each GPU.
        Only includes FP16 for now.
        """
        max_tflop_mapping = {"a100": 312e12, "h100": 989.4e12, "rtx3090": 35.58e12}

        return max_tflop_mapping[self.value]


def get_megatron_flops(
    elapsed_time_per_iter,
    checkpoint=False,
    seq_len=4096 * 16,
    hidden_size=4096,
    num_layers=32,
    vocab_size=32000,
    global_batch_size=1,
    global_world_size=16,
    mlp_ratio=4,
    use_swiglu=True,
):
    """
    Calc flops based on the paper of Megatron https://deepakn94.github.io/assets/papers/megatron-sc21.pdf
    """

    checkpoint_activations_factor = 4 if checkpoint else 3

    if use_swiglu: #LLaMA using SwiGLU
        mlp_ratio = mlp_ratio * 3 / 2

    flops_per_iteration = (
        checkpoint_activations_factor
        * (
            (8 + mlp_ratio * 4) * global_batch_size * seq_len * hidden_size**2
            + 4 * global_batch_size * seq_len**2 * hidden_size
        )
    ) * num_layers + 6 * global_batch_size * seq_len * hidden_size * vocab_size

    tflops = flops_per_iteration / (elapsed_time_per_iter * global_world_size * (10**12))
    return tflops

# # LLaMA-2 7B with sequence parallel
# step_time = 18.51
# tflops = get_megatron_flops(step_time)
# print(f"tflops, {tflops}")
# print(f"mfu, {tflops/HardwareType.A100.max_tflops*1e12}")