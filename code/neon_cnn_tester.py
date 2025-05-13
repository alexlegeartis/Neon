import time
import torch
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path

from config import (
    device, CIFAR_MEAN, CIFAR_STD, TRAIN_CONFIG, AUG_CONFIG,
    PLOT_CONFIG, OPTIMIZER_CONFIGS
)
from models import SimplePerceptron, ModerateCIFARModel, AdvancedCIFARModel, CifarNet, ComplexMLP
from optimizers import Muon, Neon
from cifar_loader import CifarLoader

# Enable ROCm backend and compilation
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled = True




############################################
#                Training                  #
############################################

def lr_lambda(current_step, total_steps):
    # total_steps = num_epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)
    cooldown_steps = int(0.1 * total_steps)
    if current_step < warmup_steps:
        return current_step / warmup_steps  # Linear warmup
    elif current_step > total_steps - cooldown_steps:
        return max(0.0, (total_steps - current_step) / cooldown_steps)  # Linear cooldown
    else:
        return 1.0  # Constant base_lr

def create_optimizer(model, total_steps, optimizer_type, **kwargs):
    """Create optimizers and schedulers for each layer of the model.
    For Neon optimizer, each layer gets a different tau parameter.
    For other optimizers, all layers use the same configuration."""
    optimizers = []
    schedulers = []
    
    # Get all layers that have parameters
    layers = []
    for name, module in model.named_modules():
        if list(module.parameters()):
            layers.append((name, module))
    
    for name, layer in layers:
        # Separate parameters based on their shapes
        matrix_params = []
        non_matrix_params = []
        
        for p in layer.parameters():
            if p.requires_grad:
                if len(p.shape) == 2:  # Matrix parameters
                    matrix_params.append(p)
                else:  # Non-matrix parameters (bias, conv weights, etc.)
                    non_matrix_params.append(p)
        
        # Create optimizer for non-matrix parameters using SGD
        if non_matrix_params:
            sgd_optimizer = torch.optim.SGD(
                non_matrix_params,
                momentum=0.85,
                nesterov=True,
                lr=kwargs.get('lr', 0.1)
            )
            optimizers.append(sgd_optimizer)
        
        # Create optimizer for matrix parameters based on optimizer_type
        if matrix_params:
            if optimizer_type == 'neon':
                # For Neon, create a new optimizer with random tau for each layer
                import random
                tau = random.uniform(0.01, 1)  # Random tau between 0.1 and 1.0
                layer_kwargs = kwargs.copy()
                layer_kwargs['tau'] = tau
                optimizer = Neon(matrix_params, **layer_kwargs)
            elif optimizer_type == 'muon':
                optimizer = Muon(matrix_params, **kwargs)
            elif optimizer_type == 'sgd':
                optimizer = torch.optim.SGD(matrix_params, **kwargs)
            else:
                raise ValueError(f"Unknown optimizer type: {optimizer_type}")
            
            optimizers.append(optimizer)
    
    return optimizers, schedulers

def run_training(model, optimizers, schedulers, train_loader, test_loader, total_epochs, 
                optimizer_label, optimizer_type, opt_key):
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Initialize lists to store metrics
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    times = []
    start_time = time.time()
    
    for epoch in range(total_epochs + 1):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            # Zero gradients for all optimizers
            for optimizer in optimizers:
                optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = crite