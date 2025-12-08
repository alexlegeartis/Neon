"""
Here we experiment with differnt LMO-based optimizers on CIFAR airbench
"""

#############################################
#                  Setup                    #
#############################################
import os
import sys
with open(sys.argv[0]) as f:
    code = f.read()
import uuid
import random
from math import ceil
from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import pandas as pd
from optimizers import Dion, Muon, Neon, NormalizedMuon, SGDMuon, SignSGDMuon, zeropower_via_newtonschulz5, RandomNormalizedMuon



#############################################
#               Muon optimizer              #
#############################################


# it only makes Muon much less accurate - only about 90%, do not use it
class ErrorFeedbackMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False, sgd_coeff=0, error_feedback_decay=0.9):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)
        self.sgd_coeff = sgd_coeff
        self.error_feedback_decay = error_feedback_decay

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                # Initialize error feedback buffer if not exists
                if "error_feedback_buffer" not in state.keys():
                    state["error_feedback_buffer"] = torch.zeros_like(g)
                
                # Initialize momentum buffer if not exists
                if "momentum_buffer" not in state.keys():
                    state["momentum_buffer"] = torch.zeros_like(g)
                
                # Add accumulated error feedback to current gradient
                g = g + state["error_feedback_buffer"]
                
                # Apply momentum
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf

                # Normalize the weight
                p.data.mul_(len(p.data)**0.5 / p.data.norm())
                
                # Compute the update using zeropower approximation
                eps = 1e-12
                g_normalized = g / (g.norm() + eps)           
                update_part = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape)
                update = (1-self.sgd_coeff) * update_part + self.sgd_coeff * g_normalized
                
                # Apply the update
                p.data.add_(update, alpha=-lr)
                
                # Compute quantization error (difference between intended and actual update)
                # The error is the difference between the original gradient and the computed update
                quantization_error = g - update
                
                # Update error feedback buffer with decay
                state["error_feedback_buffer"].mul_(self.error_feedback_decay).add_(
                    quantization_error, alpha=(1 - self.error_feedback_decay)
                )

#############################################
#                DataLoader                 #
#############################################

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616))

def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)

def batch_crop(images, crop_size):
    r = (images.size(-1) - crop_size)//2
    shifts = torch.randint(-r, r+1, size=(len(images), 2), device=images.device)
    images_out = torch.empty((len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype)
    # The two cropping methods in this if-else produce equivalent results, but the second is faster for r > 2.
    if r <= 2:
        for sy in range(-r, r+1):
            for sx in range(-r, r+1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                images_out[mask] = images[mask, :, r+sy:r+sy+crop_size, r+sx:r+sx+crop_size]
    else:
        images_tmp = torch.empty((len(images), 3, crop_size, crop_size+2*r), device=images.device, dtype=images.dtype)
        for s in range(-r, r+1):
            mask = (shifts[:, 0] == s)
            images_tmp[mask] = images[mask, :, r+s:r+s+crop_size, :]
        for s in range(-r, r+1):
            mask = (shifts[:, 1] == s)
            images_out[mask] = images_tmp[mask, :, :, r+s:r+s+crop_size]
    return images_out

class CifarLoader:

    def __init__(self, path, train=True, batch_size=500, aug=None):
        data_path = os.path.join(path, 'train.pt' if train else 'test.pt')
        if not os.path.exists(data_path):
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            torch.save({'images': images, 'labels': labels, 'classes': dset.classes}, data_path)

        data = torch.load(data_path, map_location=torch.device('cuda'))
        self.images, self.labels, self.classes = data['images'], data['labels'], data['classes']
        # It's faster to load+process uint8 data than to load preprocessed fp16 data
        self.images = (self.images.half() / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
        self.proc_images = {} # Saved results of image processing to be done on the first epoch
        self.epoch = 0

        self.aug = aug or {}
        for k in self.aug.keys():
            assert k in ['flip', 'translate'], 'Unrecognized key: %s' % k

        self.batch_size = batch_size
        self.drop_last = train
        self.shuffle = train

    def __len__(self):
        return len(self.images)//self.batch_size if self.drop_last else ceil(len(self.images)/self.batch_size)

    def __iter__(self):

        if self.epoch == 0:
            images = self.proc_images['norm'] = self.normalize(self.images)
            # Pre-flip images in order to do every-other epoch flipping scheme
            if self.aug.get('flip', False):
                images = self.proc_images['flip'] = batch_flip_lr(images)
            # Pre-pad images to save time when doing random translation
            pad = self.aug.get('translate', 0)
            if pad > 0:
                self.proc_images['pad'] = F.pad(images, (pad,)*4, 'reflect')

        if self.aug.get('translate', 0) > 0:
            images = batch_crop(self.proc_images['pad'], self.images.shape[-2])
        elif self.aug.get('flip', False):
            images = self.proc_images['flip']
        else:
            images = self.proc_images['norm']
        # Flip all images together every other epoch. This increases diversity relative to random flipping
        if self.aug.get('flip', False):
            if self.epoch % 2 == 1:
                images = images.flip(-1)

        self.epoch += 1

        indices = (torch.randperm if self.shuffle else torch.arange)(len(images), device=images.device)
        for i in range(len(self)):
            idxs = indices[i*self.batch_size:(i+1)*self.batch_size]
            yield (images[idxs], self.labels[idxs])

#############################################
#            Network Definition             #
#############################################

# note the use of low BatchNorm stats momentum
class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.6, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1-momentum)
        self.weight.requires_grad = False
        # Note that PyTorch already initializes the weights to one and bias to zero

class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=3, padding='same', bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])

class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = Conv(channels_in,  channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x

class CifarNet(nn.Module):
    def __init__(self):
        super().__init__()
        widths = dict(block1=64, block2=256, block3=256)
        whiten_kernel_size = 2
        whiten_width = 2 * 3 * whiten_kernel_size**2
        self.whiten = nn.Conv2d(3, whiten_width, whiten_kernel_size, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(
            nn.GELU(),
            ConvGroup(whiten_width,     widths['block1']),
            ConvGroup(widths['block1'], widths['block2']),
            ConvGroup(widths['block2'], widths['block3']),
            nn.MaxPool2d(3),
        )
        self.head = nn.Linear(widths['block3'], 10, bias=False)
        for mod in self.modules():
            if isinstance(mod, BatchNorm):
                mod.float()
            else:
                mod.half()

    def reset(self):
        for m in self.modules():
            if type(m) in (nn.Conv2d, Conv, BatchNorm, nn.Linear):
                m.reset_parameters()
        w = self.head.weight.data
        w *= 1 / w.std()

    def init_whiten(self, train_images, eps=5e-4):
        c, (h, w) = train_images.shape[1], self.whiten.weight.shape[2:]
        patches = train_images.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1,c,h,w).float()
        patches_flat = patches.view(len(patches), -1)
        est_patch_covariance = (patches_flat.T @ patches_flat) / len(patches_flat)
        eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO='U')
        eigenvectors_scaled = eigenvectors.T.reshape(-1,c,h,w) / torch.sqrt(eigenvalues.view(-1,1,1,1) + eps)
        self.whiten.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))

    def forward(self, x, whiten_bias_grad=True):
        b = self.whiten.bias
        x = F.conv2d(x, self.whiten.weight, b if whiten_bias_grad else b.detach())
        x = self.layers(x)
        x = x.view(len(x), -1)
        return self.head(x) / x.size(-1)

############################################
#                 Logging                  #
############################################

def print_columns(columns_list, is_head=False, is_final_entry=False):
    print_string = ''
    for col in columns_list:
        print_string += '|  %s  ' % col
    print_string += '|'
    if is_head:
        print('-'*len(print_string))
    print(print_string)
    if is_head or is_final_entry:
        print('-'*len(print_string))

logging_columns_list = ['run   ', 'epoch', 'train_acc', 'val_acc', 'tta_val_acc', 'time_seconds']
def print_training_details(variables, is_final_entry):
    formatted = []
    for col in logging_columns_list:
        var = variables.get(col.strip(), None)
        if type(var) in (int, str):
            res = str(var)
        elif type(var) is float:
            res = '{:0.4f}'.format(var)
        else:
            assert var is None
            res = ''
        formatted.append(res.rjust(len(col)))
    print_columns(formatted, is_final_entry=is_final_entry)

############################################
#               Evaluation                 #
############################################

def infer(model, loader, tta_level=0):

    # Test-time augmentation strategy (for tta_level=2):
    # 1. Flip/mirror the image left-to-right (50% of the time).
    # 2. Translate the image by one pixel either up-and-left or down-and-right (50% of the time,
    #    i.e. both happen 25% of the time).
    #
    # This creates 6 views per image (left/right times the two translations and no-translation),
    # which we evaluate and then weight according to the given probabilities.

    def infer_basic(inputs, net):
        return net(inputs).clone()

    def infer_mirror(inputs, net):
        return 0.5 * net(inputs) + 0.5 * net(inputs.flip(-1))

    def infer_mirror_translate(inputs, net):
        logits = infer_mirror(inputs, net)
        pad = 1
        padded_inputs = F.pad(inputs, (pad,)*4, 'reflect')
        inputs_translate_list = [
            padded_inputs[:, :, 0:32, 0:32],
            padded_inputs[:, :, 2:34, 2:34],
        ]
        logits_translate_list = [infer_mirror(inputs_translate, net)
                                 for inputs_translate in inputs_translate_list]
        logits_translate = torch.stack(logits_translate_list).mean(0)
        return 0.5 * logits + 0.5 * logits_translate

    model.eval()
    test_images = loader.normalize(loader.images)
    infer_fn = [infer_basic, infer_mirror, infer_mirror_translate][tta_level]
    with torch.no_grad():
        return torch.cat([infer_fn(inputs, model) for inputs in test_images.split(2000)])

def evaluate(model, loader, tta_level=0):
    logits = infer(model, loader, tta_level)
    return (logits.argmax(1) == loader.labels).float().mean().item()

############################################
#            Seed Setting                  #
############################################

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Set deterministic behavior for CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set seed for numpy if available
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

############################################
#                Training                  #
############################################

def log_grad_frobenius_norms(model, step=None, epoch=None, loss=None, outputs=None, labels=None,
                             logger=None, silent=False, norm_data_list=None, val_acc=None):
    """
    Compute and log Frobenius, spectral, and nuclear norms of gradients.
    
    Args:
        model: The model to compute gradients for
        step: Current training step
        epoch: Current epoch
        loss: Current loss value (optional)
        outputs: Model outputs for computing train accuracy (optional)
        labels: Ground truth labels for computing train accuracy (optional)
        logger: Optional logger (e.g., tensorboard)
        silent: If True, don't print
        norm_data_list: Optional list to append norm data for DataFrame creation
        val_acc: Optional validation accuracy measured alongside train accuracy
    
    Returns:
        Tuple of (norms_dicts, total_norms)
    """
    frobenius_norms = {}
    spectral_norms = {}
    nuclear_norms = {}
    
    total_frobenius_sq = 0.0
    total_spectral_max = 0.0
    total_nuclear_sum = 0.0
    
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        g = param.grad.data
        
        # Reshape to 2D matrix for matrix norms (for conv layers: out_channels x (in_channels * kernel_h * kernel_w))
        # For 1D params, this just flattens them
        g_2d = g.reshape(len(g), -1).float()
        
        # Compute all three norms
        fro_norm = torch.linalg.norm(g_2d, ord='fro').item()
        spec_norm = torch.linalg.norm(g_2d, ord=2).item()
        nuc_norm = torch.linalg.norm(g_2d, ord='nuc').item()
        
        frobenius_norms[name] = fro_norm
        spectral_norms[name] = spec_norm
        nuclear_norms[name] = nuc_norm
        
        # Accumulate for total norms
        total_frobenius_sq += fro_norm ** 2
        total_spectral_max = max(total_spectral_max, spec_norm)
        total_nuclear_sum += nuc_norm
    
    total_frobenius = total_frobenius_sq ** 0.5
    total_spectral = total_spectral_max
    total_nuclear = total_nuclear_sum

    # Compute train accuracy if outputs and labels are provided
    train_acc = None
    if outputs is not None and labels is not None:
        with torch.no_grad():
            train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()

    # Store data for DataFrame if norm_data_list is provided
    if norm_data_list is not None:
        row_data = {
            'step': step,
            'epoch': epoch if epoch is not None else -1,
        }
        # Add loss if provided
        if loss is not None:
            row_data['loss'] = loss.item() if hasattr(loss, 'item') else float(loss)
        # Add train accuracy if computed
        if train_acc is not None:
            row_data['train_acc'] = train_acc
        # Add val accuracy if computed
        if val_acc is not None:
            row_data['val_acc'] = val_acc
        # Add per-layer norms
        for name in frobenius_norms.keys():
            row_data[f'{name}_frobenius'] = frobenius_norms[name]
            row_data[f'{name}_spectral'] = spectral_norms[name]
            row_data[f'{name}_nuclear'] = nuclear_norms[name]
        # Add total norms
        row_data['total_frobenius'] = total_frobenius
        row_data['total_spectral'] = total_spectral
        row_data['total_nuclear'] = total_nuclear
        norm_data_list.append(row_data)

    # Logging options
    if logger is not None:
        for k, v in frobenius_norms.items():
            logger.add_scalar(f"grad_frobenius/{k}", v, step)
        logger.add_scalar("grad_frobenius/total", total_frobenius, step)
        for k, v in spectral_norms.items():
            logger.add_scalar(f"grad_spectral/{k}", v, step)
        logger.add_scalar("grad_spectral/total", total_spectral, step)
        for k, v in nuclear_norms.items():
            logger.add_scalar(f"grad_nuclear/{k}", v, step)
        logger.add_scalar("grad_nuclear/total", total_nuclear, step)
    elif not silent:
        loss_str = f", Loss: {loss.item():.4f}" if loss is not None else ""
        train_acc_str = f", Train Acc: {train_acc:.4f}" if train_acc is not None else ""
        val_acc_str = f", Val Acc: {val_acc:.4f}" if val_acc is not None else ""
        print(f"[Step {step}] Total grad norms - Frobenius: {total_frobenius:.4f}, Spectral: {total_spectral:.4f}, Nuclear: {total_nuclear:.4f}{loss_str}{train_acc_str}{val_acc_str}")
        for k in frobenius_norms.keys():
            print(f"  {k}: Fro={frobenius_norms[k]:.4f}, Spec={spectral_norms[k]:.4f}, Nuc={nuclear_norms[k]:.4f}")
    
    return (frobenius_norms, spectral_norms, nuclear_norms), (total_frobenius, total_spectral, total_nuclear)

def save_norm_dataframe(norm_data_list, run=None, optimizer_name=None, combo_name=None, epoch_val_acc=None):
    """
    Convert norm data list to DataFrame and save to cifar_norms/ folder.
    
    Args:
        norm_data_list: List of dictionaries containing norm data
        run: Run identifier (optional)
        optimizer_name: Optimizer name for filename (optional)
        combo_name: Combo name for filename (optional, takes precedence over optimizer_name)
        epoch_val_acc: Dictionary mapping epoch -> val_acc to backfill (optional)
    
    Returns:
        Path to saved CSV file
    """
    if not norm_data_list:
        print("Warning: No norm data to save.")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(norm_data_list)
    
    # Backfill missing val_acc entries from epoch-level values if provided
    if epoch_val_acc is not None and 'epoch' in df.columns:
        epoch_vals = df['epoch'].map(epoch_val_acc)
        if 'val_acc' in df.columns:
            df['val_acc'] = df['val_acc'].fillna(epoch_vals)
        else:
            df['val_acc'] = epoch_vals
    
    # Create output directory
    output_dir = 'cifar_norms'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create meaningful filename - use combo_name if provided, otherwise optimizer_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_parts = ['grad_norms']
    if combo_name:
        filename_parts.append(combo_name)
    elif optimizer_name:
        filename_parts.append(optimizer_name)
    if run is not None:
        filename_parts.append(f'run{run}')
    filename_parts.append(timestamp)
    filename = '_'.join(filename_parts) + '.csv'
    
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    
    print(f"Saved norm data to: {os.path.abspath(filepath)}")
    print(f"DataFrame shape: {df.shape}, Columns: {len(df.columns)}")
    
    return filepath

def main(run, model, optimizer_config=None, combo_name=None):
    """
    Main training function.
    
    Args:
        run: Run identifier
        model: The model to train
        optimizer_config: Callable that takes filter_params and returns optimizer2, or None for default
        combo_name: Name for this optimizer combo (used in filename)
    
    Returns:
        TTA validation accuracy
    """
    batch_size = 2000
    bias_lr = 0.053
    head_lr = 0.67
    wd = 2e-6 * batch_size

    # Save run ID before it might be modified
    run_id = run

    # Initialize list to collect norm data for DataFrame
    norm_data_list = []
    
    # Dictionary to store val_acc for each epoch (will be backfilled into DataFrame)
    epoch_val_acc = {}

    test_loader = CifarLoader('cifar10', train=False, batch_size=2000)
    train_loader = CifarLoader('cifar10', train=True, batch_size=batch_size, aug=dict(flip=True, translate=2))
    is_warmup = run == 'warmup'
    if is_warmup:
        # The only purpose of the first run is to warmup the compiled model, so we can use dummy data
        train_loader.labels = torch.randint(0, 10, size=(len(train_loader.labels),), device=train_loader.labels.device)
    total_train_steps = ceil(50 * len(train_loader))
    whiten_bias_train_steps = ceil(3 * len(train_loader))

    # Create optimizers and learning rate schedulers
    filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
    norm_biases = [p for n, p in model.named_parameters() if 'norm' in n and p.requires_grad]
    param_configs = [dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd/bias_lr),
                     dict(params=norm_biases, lr=bias_lr, weight_decay=wd/bias_lr),
                     dict(params=[model.head.weight], lr=head_lr, weight_decay=wd/head_lr)]
    
    optimizer1 = torch.optim.SGD(param_configs, momentum=0.85, nesterov=True)#, fused=True)
    
    # Create optimizer2 using the provided config function, or use default
    if optimizer_config is not None:
        optimizer2 = optimizer_config(filter_params)
    else:
        # Default optimizer
        optimizer2 = Neon(filter_params, neon_mode='kyfan', lr=0.04, momentum=0.65, nesterov=True, sgd_coeff=0)
    
    optimizer_name = optimizer2.__class__.__name__
    
    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
            group["target_momentum"] = group.get("momentum", 0)  # default to 0 if not set

    # For accurately timing GPU code
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    time_seconds = 0.0
    def start_timer():
        starter.record()
    def stop_timer():
        ender.record()
        torch.cuda.synchronize()
        nonlocal time_seconds
        time_seconds += 1e-3 * starter.elapsed_time(ender)

    model.reset()
    step = 0

    # Initialize the whitening layer using training images
    start_timer()
    train_images = train_loader.normalize(train_loader.images[:5000])
    model.init_whiten(train_images)
    stop_timer()

    ############################
    # Pre-training evaluation  #
    ############################
    train_acc = None
    val_acc = evaluate(model, test_loader, tta_level=0)
    epoch = -1
    epoch_val_acc[epoch] = val_acc
    print_training_details(locals(), is_final_entry=False)
    run = None  # Only print the run number once

    for epoch in range(ceil(total_train_steps / len(train_loader))):

        ####################
        #     Training     #
        ####################

        # Restart optimizers at each epoch
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
            # Reset optimizer state to clear momentum and other accumulated state
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"]
            # Clear optimizer state dict to reset momentum buffers
            opt.state.clear()
        
        start_timer()
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs, whiten_bias_grad=(step < whiten_bias_train_steps))
            loss = F.cross_entropy(outputs, labels, label_smoothing=0.2, reduction='sum')
            loss.backward()
            for group in optimizer1.param_groups[:1]:
                group["lr"] = group["initial_lr"] * (1 - step / whiten_bias_train_steps)
            for group in optimizer1.param_groups[1:] + optimizer2.param_groups:
                group["lr"] = group["initial_lr"] * (1 - step / total_train_steps)

            '''
            Technique to change momentum, which, however, we didn't find useful
            for group in optimizer2.param_groups:
                eta = (step / total_train_steps) # percentage of training
                group["momentum"] = (group["target_momentum"] - 0.1) * eta + group["target_momentum"] * (1 - eta)
            '''
            if step % 5 == 0 and not is_warmup:
                val_acc_step = evaluate(model, test_loader, tta_level=0)
                model.train()
                _, (total_fro, total_spec, total_nuc) = log_grad_frobenius_norms(
                    model,
                    step=step,
                    epoch=epoch,
                    loss=loss,
                    outputs=outputs,
                    labels=labels,
                    logger=None,
                    silent=False,
                    norm_data_list=norm_data_list,
                    val_acc=val_acc_step
                )
                loss_val = loss.item() if hasattr(loss, 'item') else float(loss)
                train_acc_val = (outputs.detach().argmax(1) == labels).float().mean().item()
                print(f"Epoch {epoch}, Step {step}: Total grad norms - Frobenius: {total_fro:.4f}, Spectral: {total_spec:.4f}, Nuclear: {total_nuc:.4f}, Loss: {loss_val:.4f}, Train Acc: {train_acc_val:.4f}, Val Acc: {val_acc_step:.4f}")

            for opt in optimizers:
                opt.step()
            step += 1
            model.zero_grad(set_to_none=True)
            if step >= total_train_steps:
                break
        stop_timer()

        ####################
        #    Evaluation    #
        ####################

        # Save the accuracy and loss from the last training batch of the epoch
        train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()
        val_acc = evaluate(model, test_loader, tta_level=0)
        # Store val_acc for this epoch to backfill into DataFrame
        epoch_val_acc[epoch] = val_acc
        print_training_details(locals(), is_final_entry=False)
        run = None # Only print the run number once

    ####################
    #  TTA Evaluation  #
    ####################

    start_timer()
    tta_val_acc = evaluate(model, test_loader, tta_level=2)
    stop_timer()
    epoch = 'eval'
    print_training_details(locals(), is_final_entry=True)

    # Save norm data to DataFrame if not warmup
    if not is_warmup and norm_data_list:
        save_norm_dataframe(
            norm_data_list,
            run=run_id,
            optimizer_name=optimizer_name,
            combo_name=combo_name,
            epoch_val_acc=epoch_val_acc
        )

    return tta_val_acc

def run_optimizer_experiments(num_runs=1, warmup=True):
    """
    Manager function to run experiments with different optimizer2 configurations.
    
    Args:
        num_runs: Number of repetitions for each optimizer config
        warmup: Whether to run a warmup iteration first
    
    Returns:
        Dictionary mapping combo_name -> list of accuracies
    """
    # Set seed for reproducibility

    # We re-use the compiled model between runs to save the non-data-dependent compilation time
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    # model.compile(mode='max-autotune')

    # Define optimizer configurations with names
    optimizer_configs = {
        'Muon': lambda filter_params: Muon(
            filter_params, lr=0.24, momentum=0.6, nesterov=True, norm_weight=False
        ),
        'Neon': lambda filter_params: Neon(
            filter_params, neon_mode='kyfan', lr=0.24, momentum=0.6, nesterov=True, norm_weight=False
        ),
        'F-Neon': lambda filter_params: Neon(
            filter_params, neon_mode='kyfan', lr=0.4, momentum=0.65, nesterov=True, sgd_coeff=0.5
        ),
        # 'Neon_kyfan_lr04': lambda filter_params: Neon(
        #     filter_params, neon_mode='kyfan', lr=0.4, momentum=0.65, nesterov=True, sgd_coeff=0
        # ),
        # 'FNeon_kyfan_lr04': lambda filter_params: Neon(
        #     filter_params, neon_mode='kyfan', lr=0.4, momentum=0.65, nesterov=True, sgd_coeff=0.5
        # ),
        'Fanion-5': lambda filter_params: Neon(
            filter_params, neon_mode='kyfan', lr=0.24, momentum=0.6, nesterov=True, sgd_coeff=0, k=5,
        ),
        'F-Fanion-5': lambda filter_params: Neon(
            filter_params, neon_mode='kyfan', lr=0.4, momentum=0.65, nesterov=True, sgd_coeff=0.5, k=5,
        ),
        # 'Neon_kyfan_k10_lr045': lambda filter_params: Neon(
        #     filter_params, neon_mode='kyfan', lr=0.45, k=10, momentum=0.65, nesterov=True, sgd_coeff=0
        # ),
        # 'FNeon_kyfan_k10_lr045': lambda filter_params: Neon(
        #     filter_params, neon_mode='kyfan', lr=0.45, k=10, momentum=0.65, nesterov=True, sgd_coeff=0.5
        # ),
        
        'F-Muon': lambda filter_params: NormalizedMuon(
            filter_params, lr=0.4, momentum=0.65, sgd_coeff=0.5, nesterov=True, norm_weight=False
        ),
        'S-Muon': lambda filter_params: SignSGDMuon(
            filter_params, lr=0.42, momentum=0.65, nesterov=True, sgd_coeff=0.5, sign_lr_mult=0.003, norm_weight=False 
        ),
        'SignSGD': lambda filter_params: SignSGDMuon(
            filter_params, lr=1, momentum=0.95, nesterov=True, sgd_coeff=1, sign_lr_mult=0.003, norm_weight=False 
        ),
        'NSGD': lambda filter_params: NormalizedMuon(filter_params, lr=0.5, momentum=0.95, sgd_coeff=1, nesterov=True, norm_weight=False)
    }
    
    # You can add more configurations here or modify existing ones
    
    results = {}
    
    print_columns(logging_columns_list, is_head=True)
    
    # Run warmup if requested
    if warmup:
        print("\n=== Running warmup ===")
        main('warmup', model, optimizer_config=None, combo_name=None)
    
    # Run each optimizer configuration
    for combo_name, optimizer_config in optimizer_configs.items():
        print(f"\n=== Running experiment: {combo_name} ===")
        set_seed(42)
        accs = torch.tensor([
            main(run, model, optimizer_config=optimizer_config, combo_name=combo_name) 
            for run in range(num_runs)
        ])
        results[combo_name] = accs
        print(f'{combo_name} - Mean: %.4f    Std: %.4f' % (accs.mean(), accs.std()))
    
    # Save summary log
    log_dir = os.path.join('logs', str(uuid.uuid4()))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'log.pt')
    torch.save(dict(code=code, results=results), log_path)
    print(f"\nSummary log saved to: {os.path.abspath(log_path)}")
    
    return results

if __name__ == "__main__":
    # Run experiments with different optimizer configurations
    results = run_optimizer_experiments(num_runs=1, warmup=True)
    
    # Print final summary
    print("\n=== Final Summary ===")
    for combo_name, accs in results.items():
        print(f'{combo_name}: Mean=%.4f, Std=%.4f' % (accs.mean(), accs.std()))