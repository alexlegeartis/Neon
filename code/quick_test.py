"""
Quick test script to run a small experiment with fewer runs.
Useful for testing the setup before running the full experiment.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from run_sgd_coeff_experiment import create_modified_main_function
from plot_results import plot_results, plot_results_from_file

def quick_test():
    """Run a quick test with fewer coefficient values and fewer runs."""
    
    print("Running Quick Test - SGD Coefficient Experiment")
    print("=" * 50)
    
    # Test fewer coefficient values for quick testing
    # sgd_coeffs = [-0.1, 0.0, 0.5, 1.0, 1.1]
    sgd_coeffs = np.arange(-0.5, 1.3, 0.1)
    print(f"Testing {len(sgd_coeffs)} different sgd_coeff values: {sgd_coeffs}")
    print()
    
    # Store results
    mean_accs = []
    std_accs = []
    
    # Run experiments for each sgd_coeff value
    for sgd_coeff in sgd_coeffs:
        print(f"Running experiment with sgd_coeff = {sgd_coeff}")
        
        # Create and compile the model
        from airbench_muon import CifarNet
        model = CifarNet().cuda().to(memory_format=torch.channels_last)
        model.compile(mode="max-autotune")
        
        # Get the modified main function
        modified_main = create_modified_main_function(sgd_coeff)
        
        # Warmup run
        print("  Running warmup...")
        modified_main("warmup", model)
        
        # Actual runs (fewer for quick testing)
        num_runs = 10
        print(f"  Running {num_runs} experiments...")
        accs = torch.tensor([modified_main(run, model) for run in range(num_runs)])
        
        mean_acc = accs.mean().item()
        std_acc = accs.std().item()
        
        print(f"  Results: Mean = {mean_acc:.4f}, Std = {std_acc:.4f}")
        print()
        
        mean_accs.append(mean_acc)
        std_accs.append(std_acc)
    
    # Plot the results
    print("Creating visualization...")
    plot_results(sgd_coeffs, mean_accs, std_accs)
    
    # Print summary
    print("\n" + "=" * 50)
    print("QUICK TEST SUMMARY")
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
        'best_std': best_std,
        'note': 'Quick test with only 5 runs per coefficient'
    }
    
    # Save in a more compatible format
    torch.save(results, 'quick_test_results.pt', _use_new_zipfile_serialization=False)
    print(f"\nQuick test results saved to 'quick_test_results.pt'")
    
    # Also save as a pickle file for better compatibility
    import pickle
    with open('quick_test_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"Results also saved to 'quick_test_results.pkl'")

if __name__ == "__main__":
    quick_test()
