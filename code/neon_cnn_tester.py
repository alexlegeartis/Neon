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
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Step all optimizers
            if epoch > 0:  # we want to measure the situation before everything
                for optimizer in optimizers:
                    optimizer.step()
        
            # Step all schedulers after each epoch
            if epoch > 0:
                for scheduler in schedulers:
                    scheduler.step()
        
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
        
        # Print current learning rates
        if epoch > 0:
            lrs = [param_group['lr'] for optimizer in optimizers for param_group in optimizer.param_groups]
            print(f'Epoch: {epoch}/{total_epochs} | '
                  f'Train Loss: {train_losses[-1]:.3f} | '
                  f'Train Acc: {train_acc:.2f}% | '
                  f'Test Loss: {test_losses[-1]:.3f} | '
                  f'Test Acc: {test_acc:.2f}% | '
                  f'LR: {min(lrs):.6f} | '
                  f'Optimizer: {optimizer_label}')
        else:
            print(f'Epoch: {epoch}/{total_epochs} | '
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
    save_dir.parent.mkdir(exist_ok=True)
    save_dir.mkdir(exist_ok=True)
    
    # Plot accuracy vs epochs
    plt.figure(figsize=PLOT_CONFIG['figsize'], dpi=PLOT_CONFIG['dpi'])
    for opt_key, result in results.items():
        opt_config = OPTIMIZER_CONFIGS[opt_key]
        plt.plot(range(0, len(result['test_accs'])), result['test_accs'],
                label=opt_config['label'])
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title(f'Test Accuracy vs Epochs ({model_name})')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / f'accuracy_vs_epochs_{model_name.lower()}.svg', format='svg', bbox_inches='tight')
    plt.close()
    
    # Plot accuracy vs time
    plt.figure(figsize=PLOT_CONFIG['figsize'], dpi=PLOT_CONFIG['dpi'])
    for opt_key, result in results.items():
        opt_config = OPTIMIZER_CONFIGS[opt_key]
        plt.plot(result['times'], result['test_accs'],
                label=opt_config['label'])
    plt.xlabel('Time (seconds)')
    plt.ylabel('Test Accuracy (%)')
    plt.title(f'Test Accuracy vs Training Time ({model_name})')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / f'accuracy_vs_time_{model_name.lower()}.svg', format='svg', bbox_inches='tight')
    plt.close()
    
    # Plot training loss vs epochs
    plt.figure(figsize=PLOT_CONFIG['figsize'], dpi=PLOT_CONFIG['dpi'])
    for opt_key, result in results.items():
        opt_config = OPTIMIZER_CONFIGS[opt_key]
        plt.plot(range(0, len(result['train_losses'])), result['train_losses'],
                label=opt_config['label'])
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title(f'Training Loss vs Epochs ({model_name})')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / f'loss_vs_epochs_{model_name.lower()}.svg', format='svg', bbox_inches='tight')
    plt.close()
    
    # Plot training loss vs time
    plt.figure(figsize=PLOT_CONFIG['figsize'], dpi=PLOT_CONFIG['dpi'])
    for opt_key, result in results.items():
        opt_config = OPTIMIZER_CONFIGS[opt_key]
        plt.plot(result['times'], result['train_losses'],
                label=opt_config['label'])
    plt.xlabel('Time (seconds)')
    plt.ylabel('Training Loss')
    plt.title(f'Training Loss vs Training Time ({model_name})')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / f'loss_vs_time_{model_name.lower()}.svg', format='svg', bbox_inches='tight')
    plt.close()

def main():
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
    
    # Create initial model based on config
    model_type = TRAIN_CONFIG['model_type']
    if model_type == 'simple':
        model = SimplePerceptron()
    elif model_type == 'big_mlp':
        model = ComplexMLP()
    elif model_type == 'moderate':
        model = ModerateCIFARModel()
    elif model_type == 'advanced':
        model = AdvancedCIFARModel()
    elif model_type == 'CifarNet':
        model = CifarNet()
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
        elif model_type == 'big_mlp':
            current_model = ComplexMLP()    
        elif model_type == 'moderate':
            current_model = ModerateCIFARModel()
        elif model_type == 'advanced':
            current_model = AdvancedCIFARModel()
        elif model_type == 'CifarNet':
            current_model = CifarNet()
        
        # Copy the initial state from the original model
        current_model.load_state_dict(model.state_dict())
        
        # Create optimizers and schedulers for each layer
        optimizers, schedulers = create_optimizer(current_model, len(train_loader) * opt_config['num_epochs'], opt_config['type'], **{
            k: v for k, v in opt_config.items() 
            if k not in ['type', 'label', 'num_epochs']
        })
        
        # Train model
        results[opt_key] = run_training(
            current_model,
            optimizers,
            schedulers,
            train_loader,
            test_loader,
            opt_config['num_epochs'],
            opt_config['label'],
            opt_config['type'],
            opt_key
        )
    
    # Plot results
    plot_results(results, model.__class__.__name__)

if __name__ == "__main__":
    main() 