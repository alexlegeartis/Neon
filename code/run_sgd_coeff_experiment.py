"""
Script to run airbench_muon.py with different sgd_coeff values and plot the results.
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from math import ceil
from airbench_muon import CifarNet, CifarLoader, evaluate, BatchNorm, Conv, ConvGroup

def create_modified_main_function(sgd_coeff):
    """
    Create a modified version of the main function with the specified sgd_coeff.
    """
    def modified_main(run, model):
        batch_size = 2000
        bias_lr = 0.053
        head_lr = 0.67
        wd = 2e-6 * batch_size

        test_loader = CifarLoader("cifar10", train=False, batch_size=2000)
        train_loader = CifarLoader("cifar10", train=True, batch_size=batch_size, aug=dict(flip=True, translate=2))
        if run == "warmup":
            # The only purpose of the first run is to warmup the compiled model, so we can use dummy data
            train_loader.labels = torch.randint(0, 10, size=(len(train_loader.labels),), device=train_loader.labels.device)
        total_train_steps = torch.ceil(torch.tensor(8 * len(train_loader)))
        whiten_bias_train_steps = torch.ceil(torch.tensor(3 * len(train_loader)))

        # Create optimizers and learning rate schedulers
        filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
        norm_biases = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
        param_configs = [dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd/bias_lr),
                         dict(params=norm_biases,         lr=bias_lr, weight_decay=wd/bias_lr),
                         dict(params=[model.head.weight], lr=head_lr, weight_decay=wd/head_lr)]
        optimizer1 = torch.optim.SGD(param_configs, momentum=0.85, nesterov=True, fused=True)
        
        # Import NormalizedMuon here to avoid circular imports
        from airbench_muon import NormalizedMuon
        optimizer2 = NormalizedMuon(filter_params, lr=0.24, momentum=0.6, nesterov=True, sgd_coeff=sgd_coeff)
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

        for epoch in range(ceil(total_train_steps / len(train_loader))):

            ####################
            #     Training     #
            ####################

            start_timer()
            model.train()
            for inputs, labels in train_loader:
                outputs = model(inputs, whiten_bias_grad=(step < whiten_bias_train_steps))
                torch.nn.functional.cross_entropy(outputs, labels, label_smoothing=0.2, reduction="sum").backward()
                for group in optimizer1.param_groups[:1]:
                    group["lr"] = group["initial_lr"] * (1 - step / whiten_bias_train_steps)
                for group in optimizer1.param_groups[1:]+optimizer2.param_groups:
                    group["lr"] = group["initial_lr"] * (1 - step / total_train_steps)
                for group in optimizer2.param_groups:
                    eta = (step / total_train_steps) # percentage of training
                    # group["momentum"] = (group["target_momentum"] - 0.1) * eta + group["target_momentum"] * (1 - eta)
            
                for opt in optimizers:
                    opt.step()
                model.zero_grad(set_to_none=True)
                step += 1
                if step >= total_train_steps:
                    break
            stop_timer()

            ####################
            #    Evaluation    #
            ####################

            # Save the accuracy and loss from the last training batch of the epoch
            train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()
            val_acc = evaluate(model, test_loader, tta_level=0)
            # print_training_details(locals(), is_final_entry=False)  # Commented out to reduce output
            run = None # Only print the run number once

        ####################
        #  TTA Evaluation  #
        ####################

        start_timer()
        tta_val_acc = evaluate(model, test_loader, tta_level=2)
        stop_timer()
        epoch = "eval"
        # print_training_details(locals(), is_final_entry=True)  # Commented out to reduce output

        return tta_val_acc
    
    return modified_main

def run_experiment_with_sgd_coeff(sgd_coeff, num_runs=20):
    """
    Run the training experiment with a specific sgd_coeff value.
    
    Args:
        sgd_coeff (float): The sgd_coeff value to use
        num_runs (int): Number of runs to perform
    
    Returns:
        tuple: (mean_accuracy, std_accuracy)
    """
    print(f"Running experiment with sgd_coeff = {sgd_coeff}")
    
    # Create and compile the model
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    model.compile(mode="max-autotune")
    
    # Get the modified main function
    modified_main = create_modified_main_function(sgd_coeff)
    
    # Warmup run
    print("  Running warmup...")
    modified_main("warmup", model)
    
    # Actual runs
    print(f"  Running {num_runs} experiments...")
    accs = torch.tensor([modified_main(run, model) for run in range(num_runs)])
    
    mean_acc = accs.mean().item()
    std_acc = accs.std().item()
    
    print(f"  Results: Mean = {mean_acc:.4f}, Std = {std_acc:.4f}")
    print()
    
    return mean_acc, std_acc

def create_standalone_plotter():
    """Create a standalone script for plotting results from saved files."""
    plotter_script = '''"""
Standalone script to plot SGD coefficient experiment results.
Run this script after the experiment to visualize results from saved .pt files.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_results(sgd_coeffs, mean_accs, std_accs):
    """Create a scatter plot with error bars showing the results."""
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with error bars
    plt.errorbar(sgd_coeffs, mean_accs, yerr=std_accs, 
                fmt='o', markersize=8, capsize=5, capthick=2,
                color='blue', ecolor='red', alpha=0.8, linewidth=2)
    
    # Customize the plot
    plt.xlabel('SGD Coefficient', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Model Accuracy vs SGD Coefficient', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Set y-axis limits
    plt.ylim(0.90, 0.96)
    
    # Add horizontal line at 94.01 (the original reported accuracy)
    plt.axhline(y=0.9401, color='red', linestyle='--', alpha=0.7, 
                label='Original: 94.01%')
    plt.legend(fontsize=12)
    
    # Add value labels above points
    for coeff, mean, std in zip(sgd_coeffs, mean_accs, std_accs):
        plt.text(coeff, mean + std + 0.001, f'{mean:.4f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('sgd_coeff_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_results_from_file(filename='sgd_coeff_experiment_results.pt'):
    """Load and plot results from a saved .pt file."""
    try:
        # Load the results
        results = torch.load(filename)
        
        sgd_coeffs = results['sgd_coeffs']
        mean_accs = results['mean_accs']
        std_accs = results['std_accs']
        
        print(f"Loaded results from {filename}")
        print(f"Number of experiments: {len(sgd_coeffs)}")
        print(f"Best accuracy: {results['best_acc']:.4f} ± {results['best_std']:.4f} at sgd_coeff = {results['best_coeff']}")
        
        # Create the plot
        plot_results(sgd_coeffs, mean_accs, std_accs)
        
    except FileNotFoundError:
        print(f"File {filename} not found. Run the experiment first.")
    except Exception as e:
        print(f"Error loading file: {e}")

if __name__ == "__main__":
    # Try to plot the main results file
    plot_results_from_file('sgd_coeff_experiment_results.pt')
    
    # You can also specify a different file:
    # plot_results_from_file('quick_test_results.pt')
'''
    
    with open('plot_results.py', 'w') as f:
        f.write(plotter_script)
    
    print("Standalone plotting script created: plot_results.py")

def main_experiment():
    """Main function to run all experiments."""
    
    # Define the range of sgd_coeff values
    sgd_coeffs = np.arange(-0.1, 1.2, 0.1)
    sgd_coeffs = [round(coeff, 1) for coeff in sgd_coeffs]
    
    print("Starting SGD Coefficient Experiment")
    print("=" * 50)
    print(f"Testing {len(sgd_coeffs)} different sgd_coeff values: {sgd_coeffs}")
    print()
    
    # Store results
    mean_accs = []
    std_accs = []
    
    # Run experiments for each sgd_coeff value
    for sgd_coeff in sgd_coeffs:
        try:
            # Run the experiment
            mean_acc, std_acc = run_experiment_with_sgd_coeff(sgd_coeff, num_runs=20)
            mean_accs.append(mean_acc)
            std_accs.append(std_acc)
        except Exception as e:
            print(f"  Error with sgd_coeff {sgd_coeff}: {e}")
            mean_accs.append(0.0)
            std_accs.append(0.0)
    
    # Plot the results
    print("Creating visualization...")
    plot_results(sgd_coeffs, mean_accs, std_accs)
    
    # Also save a standalone plotting script
    create_standalone_plotter()
    
    # Print summary
    print("\n" + "=" * 50)
    print("EXPERIMENT SUMMARY")
    print("=" * 50)
    for coeff, mean, std in zip(sgd_coeffs, mean_accs, std_accs):
        print(f"sgd_coeff = {coeff:5.1f}: {mean:.4f} ± {std:.4f}")
    
    # Find best performing coefficient
    best_idx = np.argmax(mean_accs)
    best_coeff = sgd_coeffs[best_idx]
    best_acc = mean_accs[best_idx]
    best_std = std_accs[best_idx]
    
    print(f"\nBest performing sgd_coeff: {best_coeff} with accuracy {best_acc:.4f} ± {best_std:.4f}")
    
    # Save results to file
    results = {
        'sgd_coeffs': sgd_coeffs,
        'mean_accs': mean_accs,
        'std_accs': std_accs,
        'best_coeff': best_coeff,
        'best_acc': best_acc,
        'best_std': best_std
    }
    
    # Save in a more compatible format
    torch.save(results, 'full_sgd_coeff_experiment_results.pt', _use_new_zipfile_serialization=False)
    print(f"\nResults saved to 'full_sgd_coeff_experiment_results.pt'")
    
    # Also save as a pickle file for better compatibility
    import pickle
    with open('full_sgd_coeff_experiment_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"Results also saved to 'full_sgd_coeff_experiment_results.pkl'")

if __name__ == "__main__":
    main_experiment()
