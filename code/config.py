import torch

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset configuration
CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465), device=device)
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616), device=device)

# Model configurations
MODEL_CONFIGS = {
    'simple': {
        'input_size': 32 * 32 * 3,
        'hidden_size': 1000,
        'output_size': 10
    },
    'moderate': {
        'channels': [32, 64, 128],
        'output_size': 10
    },
    'advanced': {
        'channels': [64, 128, 256, 512],
        'output_size': 10
    }
}

# Optimizer labels and configurations
OPTIMIZER_CONFIGS = {
    'neon_fast': {
        'type': 'neon',
        'label': 'Neon (Fast)',
        'num_epochs': 20,
        'lr': 0.1,
        'momentum': 0.6,
        'nesterov': True,
    },
    'neon_acc': {
        'type': 'neon',
        'label': 'Neon (Accurate)',
        'num_epochs': 20,
        'lr': 0.05,
        'momentum': 0.6,
        'nesterov': True,
    },
    'muon_fast': {
        'type': 'muon',
        'label': 'Muon (Fast)',
        'num_epochs': 20,
        'lr': 0.1,
        'momentum': 0.6,
        'nesterov': True,
    },
    'sgd_accurate': {
        'type': 'sgd',
        'label': 'SGD (Accurate)',
        'num_epochs': 20,
        'lr': 0.05,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': 2e-6 * 128
    }
}

# Training configuration
TRAIN_CONFIG = {
    'batch_size': 2000,
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
    'colors': {
        'neon_fast': 'b-',
        'neon_accurate': 'b--',
        'muon_fast': 'r-',
        'muon_accurate': 'r--',
        'sgd_fast': 'g-',
        'sgd_accurate': 'g--'
    },
    'figsize': (10, 5),
    'dpi': 100
} 