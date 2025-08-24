"""
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
    ymin = 0.85
    sgd_coeffs = np.array(sgd_coeffs)
    mean_accs = np.array(mean_accs)
    std_accs = np.array(std_accs)

    # Create a mask for points above ymin
    mask = mean_accs > ymin

    # Plot only the filtered points
    plt.errorbar(
        1 - sgd_coeffs[mask],       # x values
        mean_accs[mask],            # y values
        yerr=std_accs[mask],        # errors
        fmt='o', markersize=8, capsize=5, capthick=2,
        color='#440154', ecolor='#440154', alpha=0.8, linewidth=2 # deep purple
    )
    # Customize the plot
    plt.xlabel('α', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # Set y-axis limits
    plt.ylim(ymin, 0.96)
    
    # Add horizontal line at 94.01 (the original reported accuracy)
    plt.axhline(y=0.9401, color='gray', linestyle='--', alpha=0.7, 
                label='Top Muon: 94.01%')
    plt.legend(fontsize=12)
    
    # Add value labels above points
    for coeff, mean, std in zip(sgd_coeffs, mean_accs, std_accs):
        plt.text(1 - coeff, max(ymin, mean + std), f'{mean:.4f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    plt.xticks(np.arange(-0.2, 1.3, 0.2))
    plt.tight_layout()
    plt.savefig('sgd_coeff_results.pdf', dpi=300, bbox_inches='tight')
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

def list_available_files():
    """List all available .pt result files."""
    import os
    import glob
    
    pt_files = glob.glob("*.pt")
    if pt_files:
        print("Available result files:")
        for file in pt_files:
            print(f"  - {file}")
    else:
        print("No .pt result files found in current directory.")
    
    return pt_files

if __name__ == "__main__":
    # List available files
    available_files = list_available_files()
    
    if available_files:
        # Try to plot the main results file first
        if 'sgd_coeff_experiment_results.pt' in available_files:
            print("\nPlotting main experiment results...")
            plot_results_from_file('sgd_coeff_experiment_results.pt')
        elif 'quick_test_results.pt' in available_files:
            print("\nPlotting quick test results...")
            plot_results_from_file('quick_test_results.pt')
        else:
            # Plot the first available file
            print(f"\nPlotting results from {available_files[0]}...")
            plot_results_from_file(available_files[0])
    
    print("\nTo plot a specific file, run:")
    print("  plot_results_from_file('your_filename.pt')")
