"""
neon_muon_sgd.py
Combines SGD and Muon optimizers for training a simple perceptron model
Based on the approach from neon_light.py
"""
# from os import putenv
# putenv("HSA_OVERRIDE_GFX_VERSION", "9.0.0")

import os
import sys
import time
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from math import ceil
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

from config import (
    device, CIFAR_MEAN, CIFAR_STD, TRAIN_CONFIG, AUG_CONFIG,
    PLOT_CONFIG, OPTIMIZER_CONFIGS
)
from models import SimplePerceptron, ModerateCIFARModel, AdvancedCIFARModel
from optimizers import Muon, Neon

# Enable ROCm backend and compilation
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled = True



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
        self.epoch += 1
        indices = torch.randperm(len(self.images)) if self.shuffle else torch.arange(len(self.images))
        for i in range(0, len(self.images) - self.batch_size + 1 if self.drop_last else len(self.images), self.batch_size):
            idx = indices[i:i+self.batch_size]
            images = self.proc_images['norm'][idx]
            if self.aug.get('flip', False):
                images = batch_flip_lr(images)
            if self.aug.get('translate', 0) > 0:
                images = batch_crop(self.proc_images['pad'][idx], 32)
            yield images.to(device), self.labels[idx].to(device)


############################################
#                Training                  #
############################################

def create_optimizer(model, optimizer_type, **kwargs):
    """Create optimizers for each layer of the model.
    For Neon optimizer, each layer gets a different tau parameter.
    For other optimizers, all layers use the same configuration."""
    optimizers = []
    
    # Get all layers that have parameters
    layers = []
    for name, module in model.named_modules():
        if list(module.parameters()):
            layers.append((name, module))
    
    for name, layer in layers:
        if optimizer_type == 'neon':
            # For Neon, create a new optimizer with random tau for each layer
            import random
            tau = random.uniform(0.01, 1)  # Random tau between 0.1 and 1.0
            layer_kwargs = kwargs.copy()
            layer_kwargs['tau'] = tau
            optimizers.append(Neon(layer.parameters(), **layer_kwargs))
        elif optimizer_type == 'muon':
            optimizers.append(Muon(layer.parameters(), **kwargs))
        elif optimizer_type == 'sgd':
            optimizers.append(torch.optim.SGD(layer.parameters(), **kwargs))
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizers

def run_training(model, optimizers, train_loader, test_loader, total_epochs, optimizer_label):
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Initialize lists to store metrics
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    times = []
    start_time = time.time()
    
    for epoch in range(total_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            # Zero gradients for all optimizers
            for optimizer in optimizers:
                optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Step all optimizers
            for optimizer in optimizers:
                optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        train_acc = 100. * train_correct / train_total
        train_losses.append(train_loss/len(train_loader))
        train_accs.append(train_acc)
        
        # Test
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        test_acc = 100. * test_correct / test_total
        test_losses.append(test_loss/len(test_loader))
        test_accs.append(test_acc)
        times.append(time.time() - start_time)
        
        print(f'Epoch: {epoch+1}/{total_epochs} | '
              f'Train Loss: {train_losses[-1]:.3f} | '
              f'Train Acc: {train_acc:.2f}% | '
              f'Test Loss: {test_losses[-1]:.3f} | '
              f'Test Acc: {test_acc:.2f}% | '
              f'Optimizer: {optimizer_label}')
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'times': times
    }

def plot_results(results, model_name):
    # Create directory for plots
    save_dir = Path(PLOT_CONFIG['save_dir'])
    save_dir.mkdir(exist_ok=True)
    
    # Default colors for different optimizer types
    default_colors = {
        'neon': 'b-',
        'muon': 'r-',
        'sgd': 'g-'
    }
    
    # Plot accuracy vs epochs
    plt.figure(figsize=PLOT_CONFIG['figsize'], dpi=PLOT_CONFIG['dpi'])
    for opt_key, result in results.items():
        opt_config = OPTIMIZER_CONFIGS[opt_key]
        color = PLOT_CONFIG['colors'].get(opt_key, default_colors.get(opt_config['type'], 'k-'))
        plt.plot(range(1, len(result['test_accs']) + 1), result['test_accs'],
                color, label=opt_config['label'])
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title(f'Test Accuracy vs Epochs ({model_name})')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / f'accuracy_vs_epochs_{model_name.lower()}.png')
    plt.close()
    
    # Plot accuracy vs time
    plt.figure(figsize=PLOT_CONFIG['figsize'], dpi=PLOT_CONFIG['dpi'])
    for opt_key, result in results.items():
        opt_config = OPTIMIZER_CONFIGS[opt_key]
        color = PLOT_CONFIG['colors'].get(opt_key, default_colors.get(opt_config['type'], 'k-'))
        plt.plot(result['times'], result['test_accs'],
                color, label=opt_config['label'])
    plt.xlabel('Time (seconds)')
    plt.ylabel('Test Accuracy (%)')
    plt.title(f'Test Accuracy vs Training Time ({model_name})')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / f'accuracy_vs_time_{model_name.lower()}.png')
    plt.close()
    
    # Plot training loss vs epochs
    plt.figure(figsize=PLOT_CONFIG['figsize'], dpi=PLOT_CONFIG['dpi'])
    for opt_key, result in results.items():
        opt_config = OPTIMIZER_CONFIGS[opt_key]
        color = PLOT_CONFIG['colors'].get(opt_key, default_colors.get(opt_config['type'], 'k-'))
        plt.plot(range(1, len(result['train_losses']) + 1), result['train_losses'],
                color, label=opt_config['label'])
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title(f'Training Loss vs Epochs ({model_name})')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / f'loss_vs_epochs_{model_name.lower()}.png')
    plt.close()
    
    # Plot training loss vs time
    plt.figure(figsize=PLOT_CONFIG['figsize'], dpi=PLOT_CONFIG['dpi'])
    for opt_key, result in results.items():
        opt_config = OPTIMIZER_CONFIGS[opt_key]
        color = PLOT_CONFIG['colors'].get(opt_key, default_colors.get(opt_config['type'], 'k-'))
        plt.plot(result['times'], result['train_losses'],
                color, label=opt_config['label'])
    plt.xlabel('Time (seconds)')
    plt.ylabel('Training Loss')
    plt.title(f'Training Loss vs Training Time ({model_name})')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / f'loss_vs_time_{model_name.lower()}.png')
    plt.close()

def main(model_type='simple'):
    # Create data loaders
    train_loader = CifarLoader(
        'data',
        train=True,
        batch_size=TRAIN_CONFIG['batch_size'],
        aug=AUG_CONFIG
    )
    test_loader = CifarLoader(
        'data',
        train=False,
        batch_size=TRAIN_CONFIG['batch_size']
    )
    
    # Create initial model
    if model_type == 'simple':
        model = SimplePerceptron()
    elif model_type == 'moderate':
        model = ModerateCIFARModel()
    elif model_type == 'advanced':
        model = AdvancedCIFARModel()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train with each optimizer
    results = {}
    for opt_key in TRAIN_CONFIG['optimizers']:
        opt_config = OPTIMIZER_CONFIGS[opt_key]
        print(f"\nTraining with {opt_config['label']}:")
        
        # Create a new model instance and copy the initial state
        if model_type == 'simple':
            current_model = SimplePerceptron()
        elif model_type == 'moderate':
            current_model = ModerateCIFARModel()
        else:
            current_model = AdvancedCIFARModel()
        
        # Copy the initial state from the original model
        current_model.load_state_dict(model.state_dict())
        
        # Create optimizers for each layer
        optimizers = create_optimizer(current_model, opt_config['type'], **{
            k: v for k, v in opt_config.items() 
            if k not in ['type', 'label', 'num_epochs']
        })
        
        # Train model
        results[opt_key] = run_training(
            current_model,
            optimizers,
            train_loader,
            test_loader,
            opt_config['num_epochs'],
            opt_config['label']
        )
    
    # Plot results
    plot_results(results, model.__class__.__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='moderate', 
                      choices=['simple', 'moderate', 'advanced'],
                      help='Model type to train (simple, moderate, or advanced)')
    args = parser.parse_args()
    main(model_type=args.model) 