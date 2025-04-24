"""
neon_muon_sgd.py
Combines SGD and Muon optimizers for training a simple perceptron model
Based on the approach from neon_light.py
"""
from os import putenv
putenv("HSA_OVERRIDE_GFX_VERSION", "9.0.0")

import os
import sys
import time
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from math import ceil

# Enable ROCm backend and compilation
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#############################################
#               Muon optimizer              #
#############################################

#@torch.compile(mode='max-autotune')
def zeropower_via_newtonschulz5(G, steps=3, eps=1e-7):
    """Simplified Newton-Schulz iteration for whitening"""
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    # Add numerical stability
    norm = X.norm() + eps
    if norm < eps:
        return torch.zeros_like(X)
    X /= norm
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                if 'momentum_buffer' not in state.keys():
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group['nesterov'] else buf

                # Add numerical stability
                norm = p.data.norm()
                if norm < 1e-8:
                    continue
                p.data.mul_(len(p.data)**0.5 / norm)
                
                update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape)
                p.data.add_(update, alpha=-lr)

#############################################
#               Neon optimizer              #
#############################################

#@torch.compile(mode='max-autotune')
def u1s1v1t_torch(W, num_iter=20, eps=1e-8):
    """Power iteration using PyTorch operations"""
    # Ensure W is 2D
    if len(W.shape) > 2:
        W = W.reshape(W.size(0), -1)
    
    # Initialize v with correct shape
    v = torch.randn(W.size(1), device=W.device, dtype=W.dtype)
    v_norm = v.norm()
    if v_norm < eps:
        return torch.zeros_like(W)
    v /= v_norm
    
    for _ in range(num_iter):
        # Matrix-vector multiplication
        u = F.linear(v, W)
        u_norm = u.norm()
        if u_norm < eps:
            return torch.zeros_like(W)
        u /= u_norm
        
        # Transpose matrix-vector multiplication
        v = F.linear(u, W.T)
        v_norm = v.norm()
        if v_norm < eps:
            return torch.zeros_like(W)
        v /= v_norm
    
    # Compute first singular value - fix transpose warning
    sigma1 = (u.reshape(1, -1) @ F.linear(v, W)).squeeze()
    
    # Reshape u and v for outer product
    u = u.reshape(-1, 1)  # shape: (m, 1)
    v = v.reshape(-1, 1)  # shape: (n, 1)
    
    # Return scaled outer product
    return sigma1 * (u @ v.T)


import numpy as np
def u1s1v1t(W, num_iter=30):
    v = np.random.randn(W.shape[1])
    v /= np.linalg.norm(v)
    
    for _ in range(num_iter):
        u = W @ v
        u /= np.linalg.norm(u)
        v = W.T @ u
        v /= np.linalg.norm(v)
    sigma1 = (u.T @ W @ v)
    u = u.reshape(-1, 1)         # shape: (5, 1)
    v = v.reshape(-1, 1)         # shape: (3, 1)
    return sigma1 * u @ v.T


class Neon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                if 'momentum_buffer' not in state.keys():
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group['nesterov'] else buf

                # Add numerical stability
                norm = p.data.norm()
                if norm < 1e-8:
                    continue
                p.data.mul_(len(p.data)**0.5 / norm)
                
                # Reshape gradient to 2D if needed
                g_reshaped = g.reshape(g.size(0), -1)
                update = u1s1v1t_torch(g_reshaped).view(g.shape)
                p.data.add_(update, alpha=-lr)

#############################################
#                DataLoader                 #
#############################################

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465), device=device)
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616), device=device)

#@torch.compile(mode='max-autotune')
def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)

#@torch.compile(mode='max-autotune')
def batch_crop(images, crop_size):
    r = (images.size(-1) - crop_size)//2
    shifts = torch.randint(-r, r+1, size=(len(images), 2), device=images.device)
    images_out = torch.empty((len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype)
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

        data = torch.load(data_path, map_location='cpu')
        self.images, self.labels, self.classes = data['images'], data['labels'], data['classes']
        self.images = (self.images.float() / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

        self.normalize = T.Normalize(CIFAR_MEAN.cpu(), CIFAR_STD.cpu())
        self.proc_images = {}
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
            if self.aug.get('flip', False):
                images = self.proc_images['flip'] = batch_flip_lr(images)
            pad = self.aug.get('translate', 0)
            if pad > 0:
                self.proc_images['pad'] = F.pad(images, (pad,)*4, 'reflect')

        if self.aug.get('translate', 0) > 0:
            images = batch_crop(self.proc_images['pad'], self.images.shape[-2])
        elif self.aug.get('flip', False):
            images = self.proc_images['flip']
        else:
            images = self.proc_images['norm']
        if self.aug.get('flip', False):
            if self.epoch % 2 == 1:
                images = images.flip(-1)

        self.epoch += 1

        indices = (torch.randperm if self.shuffle else torch.arange)(len(images), device='cpu')
        for i in range(len(self)):
            idxs = indices[i*self.batch_size:(i+1)*self.batch_size]
            yield (images[idxs].to(device), self.labels[idxs].to(device))

#############################################
#            Network Definition             #
#############################################

class SimplePerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(32*32*3, 512)
        self.linear2 = nn.Linear(512, 10)
        self.activ = nn.GELU()
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.activ(x)
        x = self.linear2(x)
        return x

############################################
#                Training                  #
############################################

def run_training(model, optimizers, train_loader, test_loader, total_epochs, optimizer_name):
    """Run training for a given model and optimizers"""
    print(f"\nTraining with {optimizer_name}:")
    start_time = time.time()
    epochs = []
    accs = []
    times = []
    
    for epoch in range(total_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            for opt in optimizers:
                opt.zero_grad()
            
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            
            for opt in optimizers:
                for group in opt.param_groups:
                    group["lr"] = group["initial_lr"] * (1 - epoch / total_epochs)
                opt.step()
            
            if not torch.isnan(loss):
                total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{total_epochs}")
        print(f"Train Loss: {total_loss/len(train_loader):.3f} | Train Acc: {train_acc:.3f}%")
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * correct / total
        print(f"Test Acc: {test_acc:.3f}%")
        
        epochs.append(epoch + 1)
        accs.append(test_acc)
        times.append(time.time() - start_time)  # Time since start of this optimizer's training

    total_time = time.time() - start_time
    print(f"\n{optimizer_name} Results:")
    print(f"Best Accuracy: {max(accs):.2f}%")
    print(f"Training Time: {total_time:.2f} seconds")
    
    return epochs, accs, times

def create_optimizers(model, optimizer_type, head_lr=0.1, bias_lr=0.053, wd=2e-6*128):
    """Create optimizers for a given model and optimizer type"""
    linear_params = [p for n, p in model.named_parameters() if 'weight' in n and 'linear1' in n]
    head_params = [p for n, p in model.named_parameters() if 'weight' in n and 'linear2' in n]
    bias_params = [p for n, p in model.named_parameters() if 'bias' in n]
    
    param_configs = [
        dict(params=head_params, lr=head_lr, weight_decay=wd/head_lr),
        dict(params=bias_params, lr=bias_lr, weight_decay=wd/bias_lr)
    ]
    
    optimizer1 = torch.optim.SGD(param_configs, momentum=0.85, nesterov=True, fused=True)
    
    if optimizer_type == 'neon':
        optimizer2 = Neon(linear_params, lr=0.24, momentum=0.6, nesterov=True)
    elif optimizer_type == 'muon':
        optimizer2 = Muon(linear_params, lr=0.24, momentum=0.6, nesterov=True)
    elif optimizer_type == 'sgd':
        optimizer2 = torch.optim.SGD([dict(params=linear_params, lr=0.24, weight_decay=wd/0.24)], 
                                   momentum=0.85, nesterov=True, fused=True)
    elif optimizer_type == 'adamw':
        optimizer2 = torch.optim.AdamW([dict(params=linear_params, lr=0.24, weight_decay=wd/0.24)], 
                                     betas=(0.9, 0.999), eps=1e-8)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    optimizers = [optimizer1, optimizer2]
    
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    
    return optimizers

def main():
    batch_size = 128
    total_epochs = 5
    wd = 2e-6 * batch_size  # weight decay
    bias_lr = 0.053
    head_lr = 0.1

    train_loader = CifarLoader('cifar10', train=True, batch_size=batch_size, aug=dict(flip=True, translate=2))
    test_loader = CifarLoader('cifar10', train=False, batch_size=batch_size)

    # Create directory for plots if it doesn't exist
    os.makedirs('../figs/mlp24', exist_ok=True)

    # Lists to store all results
    all_epochs = []
    all_accs = []
    all_times = []
    optimizer_names = ['Neon', 'Muon', 'SGD', 'AdamW']

    # Train with each optimizer
    for opt_name in optimizer_names:
        model = SimplePerceptron().to(device)
        optimizers = create_optimizers(model, opt_name.lower(), head_lr, bias_lr, wd)
        epochs, accs, times = run_training(model, optimizers, train_loader, test_loader, total_epochs, opt_name)
        all_epochs.append(epochs)
        all_accs.append(accs)
        all_times.append(times)

    # Plot and save the results
    import matplotlib.pyplot as plt
    
    # Plot accuracy vs epochs
    plt.figure(figsize=(10, 5))
    colors = ['b-', 'r-', 'g-', 'y-']
    for i, (epochs, accs, name) in enumerate(zip(all_epochs, all_accs, optimizer_names)):
        plt.plot(epochs, accs, colors[i], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy vs Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('../figs/mlp24/accuracy_vs_epochs.png')
    plt.close()

    # Plot accuracy vs time
    plt.figure(figsize=(10, 5))
    for i, (times, accs, name) in enumerate(zip(all_times, all_accs, optimizer_names)):
        plt.plot(times, accs, colors[i], label=name)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy vs Training Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('../figs/mlp24/accuracy_vs_time.png')
    plt.close()

if __name__ == "__main__":
    main() 