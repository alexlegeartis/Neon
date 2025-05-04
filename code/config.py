import math
import torch

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset configuration
CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465), device=device)
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616), device=device)

batch_size = 2000

# Model configurations
MODEL_CONFIGS = {
    'simple': {
        'input_size': 32 * 32 * 3,
        'hidden_size': 1000,
        'output_size': 10
    },
    'big_mlp': {},
    'moderate': {
        'channels': [32, 64, 128],
        'output_size': 10
    },
    'advanced': {
        'channels': [64, 128, 256, 512],
        'output_size': 10
    }
}
'''
Old optimizers:

    'neon_fastest': {
        'type': 'neon',
        'label': 'Neon (Fast, small iter_num)',
        'num_epochs': 30,
        'lr': 0.1,
        'momentum': 0.6,
        'nesterov': True,
        'neon_mode': 'fast',
        'iter_num': 10,
    },
    'neon_fast': {
        'type': 'neon',
        'label': 'Neon (Fast)',
        'num_epochs': 30,
        'lr': 0.1,
        'momentum': 0.6,
        'nesterov': True,
        'neon_mode': 'fast',
        'iter_num': 100,
    },
    'neon_acc': {
        'type': 'neon',
        'label': 'Neon (Accurate)',
        'num_epochs': 30,
        'lr': 0.1,
        'momentum': 0.6,
        'nesterov': True,
        'neon_mode': 'accurate',
        'iter_num': 100,
        'k': 2,
    },




'''
# Optimizer labels and configurations
OPTIMIZER_CONFIGS = {
     'neon_acc': {
        'type': 'neon',
        'label': 'Neon (Accurate)',
        'num_epochs': 30,
        'lr': 0.1,
        'momentum': 0.3,
        'nesterov': True,
        'neon_mode': 'accurate',
        'iter_num': 40,
        'k': 3,
    },
    'neon_acc_no_mom': {
        'type': 'neon',
        'label': 'Neon (Accurate No momentum)',
        'num_epochs': 30,
        'lr': 0.1,
        'momentum': 0,
        'nesterov': False,
        'neon_mode': 'accurate',
        'iter_num': 100,
        'k': 2,
    },
    'muon_fast': {
        'type': 'muon',
        'label': 'Muon (Fast)',
        'num_epochs': 30,
        'lr': 0.1,
        'momentum': 0.6,
        'nesterov': True,
    },
    'sgd_accurate': {
        'type': 'sgd',
        'label': 'SGD (Accurate)',
        'num_epochs': 30,
        'lr': 0.001 * math.sqrt(batch_size / 2000),
        'momentum': 0.85,
        'nesterov': True,
        'weight_decay': 2e-6 * batch_size
    }
}

# Training configuration
TRAIN_CONFIG = {
    'batch_size': batch_size,
    'model_type': 'simple',
    'optimizers': list(OPTIMIZER_CONFIGS.keys())  # List of optimizer keys to use
}

# Data augmentation configuration
AUG_CONFIG = {
    'flip': True,
    'translate': 4
}

# Plotting configuration
PLOT_CONFIG = {
    'save_dir': 'figures',
    'figsize': (10, 5),
    'dpi': 100
} 