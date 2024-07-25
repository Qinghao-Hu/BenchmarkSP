import argparse
import torch
import math
from torch.utils.data import DataLoader
import torch.distributed as dist
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, set_seed, default_data_collator
from flash_attn.losses.cross_entropy import CrossEntropyLoss
from accelerate import Accelerator
from accelerate.utils import (
    InitProcessGroupKwargs,
    set_seed,
    DummyOptim,
    DummyScheduler,
)
from tqdm import tqdm
from datasets import load_dataset, load_from_disk, DatasetDict
from datetime import timedelta

from torch.profiler import profile, record_function, ProfilerActivity
from sequence_parallel import (
    set_seq_parallel_pg,
    prepare_seq_parallel_inputs,
    apply_seq_parallel_monkey_patch,
    prepare_dataloader,
)


def main(args):
    if args.wandb:
        import wandb

        wandb.login()
    set_seed(args.seed)

    timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=1_000_000))

    accelerator = Accelerator(
        mixed_precision="bf16",
        log_with="wandb" if args.wandb else None,
        kwargs_handlers=[timeout],
        # fsdp_plugin=fsdp_plugin,
    )
    accelerator.init_trackers(project_name=args.wandb, init_kwargs={"wandb": {"name": "Bench_SP"}})
    accelerator.print(f"Total GPUs: {accelerator.num_processes}")

    # config = AutoConfig.from_pretrained(args.model)
    # model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=accelerator.device,
        torch_dtype=torch.bfloat16,
        rope_theta=100_000,
        _attn_implementation="flash_attention_2",
    )

    if args.parallel_mode == "hybrid":
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        sp_ulysses_degree = min(args.ulysses_degree, world_size)

        sp_ring_degree = world_size // sp_ulysses_degree

        # TODO: Current sp_degree == world_size
        set_seq_parallel_pg(sp_ulysses_degree, sp_ring_degree, rank, world_size, args.use_ulysses_lowdim, args.ablate_no_comm, args.ablate_comm)
        # set_pg_manager(world_size, sp_ring_degree, args.use_ulysses_lowdim)

    apply_seq_parallel_monkey_patch(args.parallel_mode)

    try:
        train_dataset = load_dataset(args.dataset)
    except:
        train_dataset = load_from_disk(args.dataset)
    if isinstance(train_dataset, DatasetDict):
        train_dataset = train_dataset["train"]

    if "input_ids" not in train_dataset.column_names:
        raise RuntimeError("Dataset must include an `input_ids` feature")
    # remove everything that is not input_ids
    to_remove = [col for col in train_dataset.column_names if col != "input_ids"]
    train_dataset = train_dataset.remove_columns(to_remove)
    # train_dataset = train_dataset.shuffle(seed=args.seed)
    print("Dataset Size:", len(train_dataset))
    train_loader = DataLoader(
        train_dataset,
        collate_fn=default_data_collator,
        shuffle=True,
        batch_size=args.batch_size,
    )

    optim = DummyOptim(model.parameters(), lr=args.learning_rate)
    scheduler = DummyScheduler(
        optim,
        num_training_steps=args.max_train_steps,
        total_num_steps=args.max_train_steps,
        num_warmup_steps=args.warmup_steps,
    )
    model, optim, scheduler = accelerator.prepare(model, optim, scheduler)
    train_loader = prepare_dataloader(args.parallel_mode, train_loader, accelerator)
    if args.enable_grad_ckpt:
        model.gradient_checkpointing_enable()

    accelerator.register_for_checkpointing(scheduler)

    accelerator.print(f"Max train steps: {args.max_train_steps}")
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    model.train()
    loss_func = CrossEntropyLoss(inplace_backward=True)

    # with torch.profiler.profile(
    # activities=[
    #     torch.profiler.ProfilerActivity.CPU,
    #     torch.profiler.ProfilerActivity.CUDA,
    # ],
    # schedule=torch.profiler.schedule(
    #     wait=5, # During this phase profiler is not active.
    #     warmup=2, # During this phase profiler starts tracing, but the results are discarded.
    #     active=6, # During this phase profiler traces and records data.
    #     repeat=1), # Specifies an upper bound on the number of cycles.
    # on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./{args.parallel_mode}'),
    # # on_trace_ready=trace_handler,
    # # record_shapes=True,
    # # profile_memory=True,
    # # with_stack=True # Enable stack tracing, adds extra profiling overhead.
    # ) as profiler:

    print(f"Using {args.parallel_mode}")
    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"][..., : args.seq_length + 1][..., :-1]
        target_ids = batch["input_ids"][..., : args.seq_length + 1][..., 1:]
        position_ids = torch.arange(args.seq_length).unsqueeze(0).expand(input_ids.shape[0], -1)
        # shard the input_ids according to the world size and rank according to zig zag attention

        prepared = prepare_seq_parallel_inputs(
            args.parallel_mode,
            input_ids,
            position_ids,
            target_ids,
            accelerator.process_index,
            accelerator.num_processes,
            accelerator.device,
        )
        local_input_ids = prepared["local_input_ids"]
        local_position_ids = prepared["local_position_ids"]
        local_target_ids = prepared["local_target_ids"]

        accelerator.print(f"Step: {step}")
        loss_log = None
        with accelerator.accumulate(model):
            logits = model(
                local_input_ids,
                position_ids=local_position_ids,
            ).logits
            loss = loss_func(logits.reshape(-1, logits.shape[-1]), local_target_ids.reshape(-1))
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                gathered_loss = accelerator.reduce(loss.clone().detach(), "mean")
                loss_log = {
                    "loss": gathered_loss.item(),
                    "ppl": math.exp(gathered_loss.item()),
                }
                accelerator.log(loss_log, step=completed_steps)

            optim.step()
            scheduler.step()
            optim.zero_grad()
            # profiler.step()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                if loss_log is not None:
                    progress_bar.set_postfix(loss_log)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break
    if args.ablate_no_comm or args.ablate_comm:
        from sequence_parallel.ring.zigzag_ring_flash_attn import get_time_dict
        time_dict = get_time_dict()
        if torch.distributed.get_rank() == (torch.distributed.get_world_size() - 1):
            # print(time_dict)
            import numpy as np
            forward_time = np.asarray(time_dict["forward"]) * 1000000
            backward_time = np.asarray(time_dict["backward"]) * 1000000
            length = int(len(forward_time) * (1/3))
            forward_time = forward_time[length:]
            backward_time = backward_time[length:]
            print(f"(forward) mean: {forward_time.mean()} var: {forward_time.std()}")
            print(f"(backward) mean: {backward_time.mean()} var: {backward_time.std()}")
    accelerator.print(f"Training Finished")
    accelerator.end_training()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--batch-size", type=int, default=1)
    args.add_argument("--gradient-accumulate-every", type=int, default=1)
    args.add_argument("--wandb", type=str)
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--max-train-steps", type=int, default=400)
    args.add_argument("--warmup-steps", type=int, default=20)
    args.add_argument("--learning-rate", type=float, default=2e-5)
    args.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    args.add_argument("--dataset", type=str, default="PY007/slimpajama_llama_tokenized_upsample_4096_chunk_256K")
    args.add_argument("--lr-schedule", type=str, choices=["linear", "constant"], default="linear")
    args.add_argument("--log-loss", type=str)
    args.add_argument("--enable_grad_ckpt", action="store_true")
    args.add_argument("--seq-length", type=int, default=2048)
    args.add_argument("--ablate_no_comm", action="store_true")
    args.add_argument("--ablate_comm", action="store_true")
    args.add_argument(
        "--parallel_mode",
        type=str,
        choices=["hybrid", "ring", "zigzag", "striped", "ulysses", "lightseq", "disable"],
    )

    args.add_argument("--ulysses_degree", type=int, default=1)
    args.add_argument("--use_ulysses_lowdim", action="store_true")

    main(args.parse_args())
