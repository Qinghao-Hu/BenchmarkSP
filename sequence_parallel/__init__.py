from .globals import set_pg_manager, get_pg_manager
from .input_utils import extract_local_from_list
from .input_utils import extract_local_input_ids
from .input_utils import extract_local_position_ids
from .input_utils import prepare_inputs
from .ring.monkey_patch import apply_zigzag_ring_attn_monkey_patch_llama
from .ring.monkey_patch import apply_ring_attn_monkey_patch_llama
from .ring.monkey_patch import apply_striped_attn_monkey_patch_llama
from .lightseq.monkey_patch import apply_dist_flash_attn_monkey_patch_llama
from .ulysses.monkey_patch import apply_ulysses_attn_monkey_patch_llama
from .hybrid.monkey_patch import apply_hybrid_attn_monkey_patch_llama
from .hybrid.utils import set_seq_parallel_pg


def prepare_seq_parallel_inputs(seq_algo, input_ids, position_ids, target_ids, rank, world_size, device):
    if seq_algo == "disable":
        return {
            "local_input_ids": input_ids.to(device),
            "local_position_ids": position_ids.to(device),
            "local_target_ids": target_ids.to(device),
        }
    elif seq_algo in ["hybrid", "ring", "zigzag", "striped", "ulysses", "lightseq"]:
        return prepare_inputs(input_ids, position_ids, target_ids, rank, world_size, device, seq_algo)
    else:
        raise ValueError(f"Invalid seq_algo: {seq_algo}")


def apply_seq_parallel_monkey_patch(seq_algo):
    assert seq_algo in [
        "hybrid",
        "ring",
        "zigzag",
        "striped",
        "ulysses",
        "lightseq",
        "disable",
    ], f"Invalid seq_algo: {seq_algo}"
    if seq_algo == "disable":
        return
    elif seq_algo == "ring":
        apply_ring_attn_monkey_patch_llama()
    elif seq_algo == "zigzag":
        apply_zigzag_ring_attn_monkey_patch_llama()
    elif seq_algo == "striped":
        apply_striped_attn_monkey_patch_llama()
    elif seq_algo == "hybrid":
        apply_hybrid_attn_monkey_patch_llama()
    elif seq_algo == "ulysses":
        apply_ulysses_attn_monkey_patch_llama()
    elif seq_algo == "lightseq":
        apply_dist_flash_attn_monkey_patch_llama()
    else:
        raise ValueError(f"Invalid seq_algo: {seq_algo} ")


def prepare_dataloader(seq_algo, dataloader, acclerator):
    if seq_algo == "data_parallel":
        return acclerator.prepare(dataloader)
    else:
        return dataloader
