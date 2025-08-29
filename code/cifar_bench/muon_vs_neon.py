# Testing Neon on the benchmark, the code works for Muon around 3s indeed on a powerful GPU
# Neon is far worse. Somehow its accuracy more depends on the batchsize.
from math import ceil
import torch
torch.backends.cudnn.benchmark = True
from torch import nn
import torch.nn.functional as F
import airbench_muon as airbench
import matplotlib.pyplot as plt
import time
import os
import numpy as np

from optimizers import Neon, NormalizedMuon, zeropower_via_newtonschulz5

code_version = '13may_muon_neon'

# Neon learning rate schedule by epoch (0-indexed)
# Modify these values to change the learning rate at each epoch
NEON_LR_SCHEDULE = {
    0: 0.24,    # Initial learning rate
    1: 0.24,    # Epoch 1
    2: 0.24,    # Epoch 2
    3: 0.24,    # Epoch 3
    4: 0.24,    # Epoch 4
    5: 0.24,   # Epoch 5
    6: 0.24,   # Epoch 6
    7: 0.24,  # Epoch 7
    8: 0.24,
    9: 0.24,
    10: 0.24,
}


# note the use of low BatchNorm stats momentum
class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.6, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1-momentum)
        self.weight.requires_grad = False
        # Note that PyTorch already initializes the weights to one and bias to zero

class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=3, padding="same", bias=False)

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
            ConvGroup(whiten_width,     widths["block1"]),
            ConvGroup(widths["block1"], widths["block2"]),
            ConvGroup(widths["block2"], widths["block3"]),
            nn.MaxPool2d(3),
        )
        self.head = nn.Linear(widths["block3"], 10, bias=False)
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
        eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO="U")
        eigenvectors_scaled = eigenvectors.T.reshape(-1,c,h,w) / torch.sqrt(eigenvalues.view(-1,1,1,1) + eps)
        self.whiten.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))

    def forward(self, x, whiten_bias_grad=True):
        b = self.whiten.bias
        x = F.conv2d(x, self.whiten.weight, b if whiten_bias_grad else b.detach())
        x = self.layers(x)
        x = x.view(len(x), -1)
        return self.head(x) / x.size(-1)

def main(optimizer_type='neon', sgd_coeff=0):
    """Run training with specified optimizer type ('muon' or 'neon')"""
    num_epochs = 10
    model = CifarNet().cuda().to(memory_format=torch.channels_last)

    batch_size = 2000
    bias_lr = 0.053
    head_lr = 0.67
    wd = 2e-6 * batch_size

    # Print training parameters
    print("\n" + "="*50)
    print(f"Training Parameters for {optimizer_type.upper()}:")
    print(f"Batch Size: {batch_size}")
    print(f"Bias Learning Rate: {bias_lr}")
    print(f"Head Learning Rate: {head_lr}")
    print(f"Weight Decay: {wd}")
    
    test_loader = airbench.CifarLoader("cifar10", train=False, batch_size=batch_size)
    train_loader = airbench.CifarLoader("cifar10", train=True, batch_size=batch_size,
                                        aug=dict(flip=True, translate=2))
    total_train_steps = ceil(num_epochs * len(train_loader))
    whiten_bias_train_steps = ceil(3 * len(train_loader))

    # Create optimizers and learning rate schedulers
    filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad] # for convolution
    norm_biases = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
    param_configs = [dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd/bias_lr), # whitening or normalization layer
                     dict(params=norm_biases,         lr=bias_lr, weight_decay=wd/bias_lr), # batch normalization
                     dict(params=[model.head.weight], lr=head_lr, weight_decay=wd/head_lr)] # output
    optimizer1 = torch.optim.SGD(param_configs, momentum=0.85, nesterov=True) #, fused=True)
    
    # Select optimizer based on parameter
    if optimizer_type.lower() == 'muon':
        optimizer2 = NormalizedMuon(filter_params, lr=0.24, momentum=0.6, nesterov=True, sgd_coeff=sgd_coeff)
        print(f"Muon Learning Rate: 0.24")
        print(f"Muon Momentum: 0.6")
        print(f"Muon Nesterov: True")
    elif optimizer_type.lower() == 'adamw':
        adamw_lr = 1e-2
        # optimizer2 = torch.optim.AdamW(filter_params, lr=adamw_lr, weight_decay=wd)
        optimizer2 = torch.optim.SGD(filter_params, momentum=0.85, nesterov=True)
        print(f"AdamW Learning Rate: {adamw_lr}")
        print(f"AdamW Weight Decay: {wd}")
    else:  # default to neon
        # neon_mode = 'accurate'
        neon_mode = 'fast'
        # Use the initial learning rate from the schedule
        neon_lr = 0.4 # 0.2 # NEON_LR_SCHEDULE[0]
        neon_momentum = 0.65
        optimizer2 = Neon(filter_params, lr=neon_lr, momentum=neon_momentum, 
        nesterov=True, neon_mode=neon_mode,
                          iter_num = 50, sgd_coeff=0)
                      #iter_num=1 * batch_size / 500)
        print(f"Neon Learning Rate: {neon_lr}")
        print(f"Neon Momentum: {neon_momentum}")
        print(f"Neon Mode: {neon_mode}")
        print(f"Neon Nesterov: True")
        # Manual learning rate schedule for Neon
        '''
        if optimizer_type.lower() == 'neon':
            print("Neon Learning Rate Schedule:")
            for epoch, lr in NEON_LR_SCHEDULE.items():
                print(f"  Epoch {epoch}: {lr}")'''
    print("="*50 + "\n")
    
    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    model.reset()
    step = 0

    # Initialize the whitening layer using training images
    train_images = train_loader.normalize(train_loader.images[:5000])
    model.init_whiten(train_images)

    # Metrics tracking
    epochs = []
    train_losses = []
    test_accs = []
    test_losses = []
    times = []
    start_time = time.time()

    # Track metrics for epoch 0 (initialization)
    model.eval()
    test_acc, test_loss = evaluate_model(model, test_loader)
    epochs.append(0)
    test_accs.append(test_acc)
    test_losses.append(test_loss)
    times.append(0)
    
    for epoch in range(ceil(total_train_steps / len(train_loader))):
        epoch_loss = 0
        batch_count = 0
        '''
        # Apply manual learning rate schedule for Neon at the beginning of each epoch
        if optimizer_type.lower() == 'neon' and epoch in NEON_LR_SCHEDULE:
            new_lr = NEON_LR_SCHEDULE[epoch]
            for group in optimizer2.param_groups:
                group["lr"] = new_lr
                group["initial_lr"] = new_lr
            print(f"Epoch {epoch}: Setting Neon learning rate to {new_lr}")
        '''
            
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs, whiten_bias_grad=(step < whiten_bias_train_steps))
            loss = F.cross_entropy(outputs, labels, label_smoothing=0.2, reduction="sum")
            epoch_loss += loss.item()
            batch_count += inputs.size(0)
            
            loss.backward()
            for group in optimizer1.param_groups[:1]:
                group["lr"] = group["initial_lr"] * (1 - step / whiten_bias_train_steps)
            
            # Only apply automatic learning rate decay to non-Neon optimizer or Muon optimizer
            #if optimizer_type.lower() != 'neon':
            for group in optimizer1.param_groups[1:]+optimizer2.param_groups:
                group["lr"] = group["initial_lr"] * (1 - step / total_train_steps)
            #else:
                # For Neon, only apply to optimizer1 groups
            #    for group in optimizer1.param_groups[1:]:
            #        group["lr"] = group["initial_lr"] * (1 - step / total_train_steps)
                    
            for opt in optimizers:
                opt.step()
            model.zero_grad(set_to_none=True)
            step += 1
            if step >= total_train_steps:
                break
        
        # Track metrics after each epoch
        avg_train_loss = epoch_loss / batch_count if batch_count > 0 else float('inf')
        train_losses.append(avg_train_loss)
        
        # Evaluate on test set
        model.eval()
        test_acc, test_loss = evaluate_model(model, test_loader)
        
        current_time = time.time() - start_time
        epochs.append(epoch + 1)
        test_accs.append(test_acc)
        test_losses.append(test_loss)
        times.append(current_time)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Test Acc={test_acc:.4f}, Test Loss={test_loss:.4f}, Time={current_time:.2f}s")

    final_acc = test_accs[-1]
    print(f"Final Accuracy: {final_acc:.4f}")
    
    return {
        'epochs': epochs,
        'train_losses': train_losses,
        'test_accs': test_accs,
        'test_losses': test_losses,
        'times': times,
        'final_acc': final_acc
    }

def evaluate_model(model, test_loader, tta_level=2):
    """Evaluate model on test set, returning both accuracy and loss"""
    total_loss = 0
    with torch.no_grad():
        acc = airbench.evaluate(model, test_loader, tta_level=tta_level)
        # Calculate loss
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels, reduction="sum")
            total_loss += loss.item()
    
    avg_loss = total_loss # / len(test_loader.dataset) -- this we do not know
    return acc, avg_loss

def plot_comparison():
    """Run both optimizers and plot comparison metrics"""
    # Create the output directory for figures
    figures_dir = f"figures/{code_version}"
    os.makedirs(figures_dir, exist_ok=True)
    print(f"Saving figures to: {figures_dir}")
    # Run with Muon optimizer
     # Run with Neon optimizer
    
    print("Training with Neon optimizer...")
    neon_results = main(optimizer_type='neon', sgd_coeff=0.5)
    print("Training with Muon optimizer...")
    muon_results = main(optimizer_type='muon', sgd_coeff=0)
    print("Training with AdamW optimizer...") #it's not Adam, it's SGD!
    adamw_results = main(optimizer_type='adamw', sgd_coeff=0)
    

    
    # Define more pleasing color palette
    muon_color = '#4363d8'  # Vibrant blue
    neon_color = '#e6194B'  # Crimson red
    adamw_color = '#3cb44b' # Green
    background_color = '#f8f9fa'  # Light gray background
    grid_color = '#dddddd'  # Subtle grid lines
    
    # 1. Test Accuracy vs Epoch - in a separate window
    plt.figure(figsize=(10, 8), facecolor=background_color)
    ax = plt.gca()
    ax.set_facecolor(background_color)
    plt.plot(muon_results['epochs'], muon_results['test_accs'], color=muon_color, linestyle='-', linewidth=2.5, marker='o', markersize=6, label='Muon')
    plt.plot(neon_results['epochs'], neon_results['test_accs'], color=neon_color, linestyle='-', linewidth=2.5, marker='s', markersize=6, label='Neon')
    plt.plot(adamw_results['epochs'], adamw_results['test_accs'], color=adamw_color, linestyle='-', linewidth=2.5, marker='^', markersize=6, label='AdamW')
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Test Accuracy', fontsize=14, fontweight='bold')
    plt.title('Test Accuracy vs Epoch', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, framealpha=0.9)
    plt.grid(True, alpha=0.3, color=grid_color)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/test_acc_vs_epoch.pdf", bbox_inches='tight')
    plt.close()  # Close the figure to create a new one
    
    # 2. Test Loss vs Epoch - in a separate window
    plt.figure(figsize=(10, 8), facecolor=background_color)
    ax = plt.gca()
    ax.set_facecolor(background_color)
    plt.plot(muon_results['epochs'], muon_results['test_losses'], color=muon_color, linestyle='-', linewidth=2.5, marker='o', markersize=6, label='Muon')
    plt.plot(neon_results['epochs'], neon_results['test_losses'], color=neon_color, linestyle='-', linewidth=2.5, marker='s', markersize=6, label='Neon')
    plt.plot(adamw_results['epochs'], adamw_results['test_losses'], color=adamw_color, linestyle='-', linewidth=2.5, marker='^', markersize=6, label='AdamW')
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Test Loss', fontsize=14, fontweight='bold')
    plt.title('Test Loss vs Epoch', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, framealpha=0.9)
    plt.grid(True, alpha=0.3, color=grid_color)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/test_loss_vs_epoch.pdf", bbox_inches='tight')
    plt.close()  # Close the figure to create a new one
    
    # 3. Test Accuracy vs Time - in a separate window
    plt.figure(figsize=(10, 8), facecolor=background_color)
    ax = plt.gca()
    ax.set_facecolor(background_color)
    plt.plot(muon_results['times'], muon_results['test_accs'], color=muon_color, linestyle='-', linewidth=2.5, marker='o', markersize=6, label='Muon')
    plt.plot(neon_results['times'], neon_results['test_accs'], color=neon_color, linestyle='-', linewidth=2.5, marker='s', markersize=6, label='Neon')
    plt.plot(adamw_results['times'], adamw_results['test_accs'], color=adamw_color, linestyle='-', linewidth=2.5, marker='^', markersize=6, label='AdamW')
    plt.xlabel('Time (seconds)', fontsize=14, fontweight='bold')
    plt.ylabel('Test Accuracy', fontsize=14, fontweight='bold')
    plt.title('Test Accuracy vs Time', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, framealpha=0.9)
    plt.grid(True, alpha=0.3, color=grid_color)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/test_acc_vs_time.pdf", bbox_inches='tight')
    plt.close()  # Close the figure to create a new one
    
    # 4. Test Loss vs Time - in a separate window
    plt.figure(figsize=(10, 8), facecolor=background_color)
    ax = plt.gca()
    ax.set_facecolor(background_color)
    plt.plot(muon_results['times'], muon_results['test_losses'], color=muon_color, linestyle='-', linewidth=2.5, marker='o', markersize=6, label='Muon')
    plt.plot(neon_results['times'], neon_results['test_losses'], color=neon_color, linestyle='-', linewidth=2.5, marker='s', markersize=6, label='Neon')
    plt.plot(adamw_results['times'], adamw_results['test_losses'], color=adamw_color, linestyle='-', linewidth=2.5, marker='^', markersize=6, label='AdamW')
    plt.xlabel('Time (seconds)', fontsize=14, fontweight='bold')
    plt.ylabel('Test Loss', fontsize=14, fontweight='bold')
    plt.title('Test Loss vs Time', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, framealpha=0.9)
    plt.grid(True, alpha=0.3, color=grid_color)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/test_loss_vs_time.pdf", bbox_inches='tight')
    plt.close()  # Close the figure to create a new one
    
    print(f"Muon Final Accuracy: {muon_results['final_acc']:.4f}")
    print(f"Neon Final Accuracy: {neon_results['final_acc']:.4f}")
    print(f"AdamW Final Accuracy: {adamw_results['final_acc']:.4f}")
    print(f"All plots saved to {figures_dir}/")

if __name__ == "__main__":
    plot_comparison()
    
    # For original benchmark with multiple runs
    # accs = torch.tensor([main(optimizer_type='neon')['final_acc'] for run in range(25)])
    # print("Mean: %.4f    Std: %.4f" % (accs.mean(), accs.std()))
