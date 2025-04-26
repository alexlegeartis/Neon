"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from torch.utils.tensorboard import SummaryWriter

import json

# Added imports for gradient analysis and plotting
import matplotlib.pyplot as plt
from scipy.linalg import svd

writer = SummaryWriter()

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = "out"
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False  # disabled by default
wandb_project = "owt"
wandb_run_name = "gpt2"  # 'run' + str(time.time())
# data
dataset = "openwebtext"
gradient_accumulation_steps = 5 * 8  # used to simulate larger batch sizes
batch_size = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 600000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = "nccl"  # 'nccl', 'gloo', etc.
# system
device = (
    "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
)
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True  # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

# Add gradient analysis parameters
svd_interval = 5  # Extract gradients and plot SVD every 5 iterations
svd_dir = os.path.join(out_dir, "svd_plots")  # Directory to save SVD plots

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(svd_dir, exist_ok=True)  # Create directory for SVD plots
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)


data_dir = os.path.join("data", dataset)


def save_combined_svd_plot(svd_history):
    
    plt.figure(figsize=(12, 8))

    iterations = sorted(list(svd_history.keys()))

    
    if len(iterations) > 5:
        
        step = len(iterations) // 5
        selected_iterations = [iterations[i] for i in range(0, len(iterations), step)]
        
        if iterations[-1] not in selected_iterations:
            selected_iterations[-1] = iterations[-1]
    else:
        selected_iterations = iterations

   
    for iter_num in selected_iterations:
        values = svd_history[iter_num]
        plt.semilogy(values, "o-", label=f"Iteration {iter_num}")

    plt.grid(True)
    plt.title("Evolution of Singular Value Spectrum Across Training")
    plt.xlabel("Index")
    plt.ylabel("Singular Value (log scale)")
    plt.legend()

    
    plt.savefig(os.path.join(svd_dir, "svd_evolution.png"))
    plt.close()

    # Also save a version with all iterations (even if it's cluttered)
    plt.figure(figsize=(12, 8))
    for iter_num in iterations:
        values = svd_history[iter_num]
        plt.semilogy(values, "-", label=f"Iter {iter_num}", alpha=0.7)

    plt.grid(True)
    plt.title("Complete Singular Value Spectrum Evolution (All Iterations)")
    plt.xlabel("Index")
    plt.ylabel("Singular Value (log scale)")
    # Use a compact legend
    plt.legend(loc="upper right", fontsize="small", ncol=2)

    # Save the all-inclusive plot
    plt.savefig(os.path.join(svd_dir, "svd_evolution_all.png"))
    plt.close()

    # Save the data to a JSON file for external analysis
    with open(os.path.join(svd_dir, "svd_history.json"), "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        json_data = {str(k): v.tolist() for k, v in svd_history.items()}
        json.dump(json_data, f)


def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == "train":
        data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    else:
        data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )
    if device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# Function to extract and process gradients
def extract_gradients(model):
    # Get raw model if DDP is used
    raw_model = model.module if ddp else model

    # Collect all gradients
    grads = []
    total_params = 0
    for param in raw_model.parameters():
        if param.grad is not None:
            # Count parameters for logging purposes
            total_params += param.numel()
            # Move to CPU and convert to numpy
            grad_flat = param.grad.detach().cpu().float().flatten().numpy()
            grads.append(grad_flat)

    print(f"Total parameters with gradients: {total_params:,}")

    # For very large models, we'll sample the gradients instead of using all of them
    # to avoid memory issues with SVD computation
    max_samples = 10000  # Maximum number of gradient elements to use

    if total_params > max_samples:
        print(
            f"Model has {total_params:,} parameters, sampling {max_samples:,} for SVD analysis"
        )
        # Concatenate all gradients into a single vector
        all_grads = np.concatenate(grads)

        # Randomly sample from the gradient vector
        np.random.seed(42 + iter_num)  # For reproducibility but changing each iteration
        indices = np.random.choice(all_grads.shape[0], size=max_samples, replace=False)
        sampled_grads = all_grads[indices]

        # Use a rectangular matrix of reasonable size for SVD
        rows = 100
        cols = 100
        if len(sampled_grads) < rows * cols:
            rows = int(np.sqrt(len(sampled_grads)))
            cols = rows

        # Pad if necessary
        padded_size = rows * cols
        padded_grads = np.zeros(padded_size)
        padded_grads[: min(len(sampled_grads), padded_size)] = sampled_grads[
            : min(len(sampled_grads), padded_size)
        ]
        grad_matrix = padded_grads.reshape(rows, cols)
    else:
        # For smaller models, use all gradients
        all_grads = np.concatenate(grads)

        # Use a rectangular matrix of reasonable size
        rows = min(100, int(np.sqrt(all_grads.shape[0])))
        cols = rows

        # Pad if necessary
        padded_size = rows * cols
        padded_grads = np.zeros(padded_size)
        padded_grads[: min(len(all_grads), padded_size)] = all_grads[
            : min(len(all_grads), padded_size)
        ]
        grad_matrix = padded_grads.reshape(rows, cols)

    print(f"Gradient matrix shape for SVD: {grad_matrix.shape}")
    return grad_matrix


# Function to compute and plot singular values
def compute_and_plot_svd(grad_matrix, iter_num, svd_history):
    try:
        # Use torch SVD which can handle larger matrices more efficiently
        # Convert numpy array to torch tensor
        grad_tensor = torch.tensor(grad_matrix, dtype=torch.float32)

        # Use torch.svd instead of scipy.linalg.svd
        u, s, v = torch.svd(grad_tensor)

        # Convert back to numpy for plotting
        s_np = s.cpu().numpy()

        print(f"Successfully computed SVD with {len(s_np)} singular values")

        # Plot singular values for this iteration
        plt.figure(figsize=(10, 6))
        plt.semilogy(s_np, "o-")
        plt.grid(True)
        plt.title(f"Singular Value Spectrum at Iteration {iter_num}")
        plt.xlabel("Index")
        plt.ylabel("Singular Value (log scale)")

        # Save the plot
        plt.savefig(os.path.join(svd_dir, f"svd_iter_{iter_num}.png"))
        plt.close()

        # Store this iteration's singular values in our history
        svd_history[iter_num] = s_np

        # Update the combined plot every svd_combined_plot_interval iterations

        # Log top singular values to tensorboard
        num_to_log = min(10, len(s_np))
        for i in range(num_to_log):
            writer.add_scalar(f"SVD/singular_value_{i}", s_np[i], iter_num)

        # Also log the condition number (ratio of largest to smallest singular value)
        if len(s_np) > 1 and s_np[-1] > 1e-10:  # Avoid division by very small values
            condition_number = s_np[0] / s_np[-1]
            writer.add_scalar("SVD/condition_number", condition_number, iter_num)

        # Return singular values for further analysis if needed
        return s_np, svd_history

    except Exception as e:
        print(f"Error in SVD computation: {e}")
        print("Trying with a more robust fallback approach...")

        try:
            # Fallback to scipy's svd with lapack_driver='gesvd' which is more robust but slower
            u, s, vh = svd(grad_matrix, full_matrices=False, lapack_driver="gesvd")

            print(f"Fallback SVD succeeded with {len(s)} singular values")

            # Plot singular values
            plt.figure(figsize=(10, 6))
            plt.semilogy(s, "o-")
            plt.grid(True)
            plt.title(f"Singular Value Spectrum at Iteration {iter_num}")
            plt.xlabel("Index")
            plt.ylabel("Singular Value (log scale)")

            # Save the plot
            plt.savefig(os.path.join(svd_dir, f"svd_iter_{iter_num}.png"))
            plt.close()

            # Store this iteration's singular values in our history
            svd_history[iter_num] = s

            # Update the combined plot every svd_combined_plot_interval iterations

            # Log top singular values to tensorboard
            num_to_log = min(10, len(s))
            for i in range(num_to_log):
                writer.add_scalar(f"SVD/singular_value_{i}", s[i], iter_num)

            return s

        except Exception as e2:
            print(f"Both SVD methods failed: {e2}")
            print("Using eigendecomposition of Gram matrix as final fallback")

            # As a final fallback, compute eigenvalues of the Gram matrix
            # which is related to singular values: singular_values = sqrt(eigenvalues)
            gram = grad_matrix @ grad_matrix.T
            eigvals = np.linalg.eigvalsh(gram)
            # Sort in descending order and take square root
            eigvals = np.sqrt(np.sort(eigvals)[::-1])

            # Filter out negative or very small values that might result from numerical issues
            eigvals = eigvals[eigvals > 1e-10]

            if len(eigvals) > 0:
                # Plot eigenvalues
                plt.figure(figsize=(10, 6))
                plt.semilogy(eigvals, "o-")
                plt.grid(True)
                plt.title(
                    f"Approximated Singular Values at Iteration {iter_num} (via eigendecomposition)"
                )
                plt.xlabel("Index")
                plt.ylabel("Approximated Singular Value (log scale)")

                # Save the plot
                plt.savefig(os.path.join(svd_dir, f"svd_iter_{iter_num}_approx.png"))
                plt.close()

                # Store this iteration's singular values in our history
                svd_history[iter_num] = eigvals

                # Update the combined plot every svd_combined_plot_interval iterations

                # Log a few eigenvalues to tensorboard
                num_to_log = min(10, len(eigvals))
                for i in range(num_to_log):
                    writer.add_scalar(
                        f"SVD/approx_singular_value_{i}", eigvals[i], iter_num
                    )

                return eigvals
            else:
                print("Could not compute valid eigenvalues for SVD approximation")
                return np.array([0.0])  # Return a dummy value


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
svd_combined_plot_interval = max_iters

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    dropout=dropout,
)  # start with model_args from command line
if init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print(
            "defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)"
        )
    model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]

    # Load SVD history if it exists in the checkpoint
    if "svd_history" in checkpoint:
        svd_history = checkpoint["svd_history"]
        print(f"Loaded SVD history with {len(svd_history)} data points")

        # Create the combined plot from the loaded history
        if len(svd_history) > 0:
            print("Regenerating combined SVD plot from loaded history")
            save_combined_svd_plot()
elif init_from.startswith("gpt2"):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args["block_size"] = (
        block_size  # so that the checkpoint will have the right value
    )
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

# optimizer
optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type
)
if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# logging
if wandb_log and master_process:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch("train")  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0
svd_history = {}
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
        writer.add_scalar("Loss/train", losses["train"], iter_num)
        writer.add_scalar("Loss/val", losses["val"], iter_num)
        if wandb_log:
            wandb.log(
                {
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "lr": lr,
                    "mfu": running_mfu * 100,  # convert to percentage
                }
            )
        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                    "svd_history": svd_history,  # Save SVD history with checkpoint
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (
                micro_step == gradient_accumulation_steps - 1
            )
        with ctx:
            logits, loss = model(X, Y)
            loss = (
                loss / gradient_accumulation_steps
            )  # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch("train")
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Process gradients every svd_interval iterations
    if master_process and iter_num % svd_interval == 0 and iter_num > 0:
        # We need to unscale before extracting gradients if using mixed precision
        if dtype == "float16":
            scaler.unscale_(optimizer)

        # Extract gradients, compute SVD, and plot
        print(f"Extracting gradients and computing SVD at iteration {iter_num}")
        grad_matrix = extract_gradients(model)
        singular_values, svd_history = compute_and_plot_svd(
            grad_matrix, iter_num, svd_history
        )

        # Log some statistics about the singular values
        if len(singular_values) > 1:
            sv_mean = np.mean(singular_values)
            sv_std = np.std(singular_values)
            writer.add_scalar("SVD/mean", sv_mean, iter_num)
            writer.add_scalar("SVD/std", sv_std, iter_num)
            print(f"  Mean singular value: {sv_mean:.6f}, Std: {sv_std:.6f}")

            # Log rank approximation quality
            total_energy = np.sum(singular_values**2)
            for r in [1, 5, 10, 50, 100]:
                if r < len(singular_values):
                    energy_r = np.sum(singular_values[:r] ** 2) / total_energy
                    writer.add_scalar(f"SVD/energy_ratio_rank_{r}", energy_r, iter_num)

        # On the last iteration, make sure to generate the final combined plot

    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(
            f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
        )

    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

# Before finishing, make sure to generate the final combined plot
if master_process and len(svd_history) > 0:
    print("Generating final combined SVD plot")
    save_combined_svd_plot(svd_history)

writer.flush()
writer.close()
if ddp:
    destroy_process_group()
