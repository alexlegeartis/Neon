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
        fmt='o', markersize=6, capsize=5, capthick=2,
        color='#440154', ecolor='#440154', alpha=0.8, linewidth=2 # deep purple
    )
    # Customize the plot
    plt.xlabel('α', fontsize=18)
    plt.ylabel('Accuracy, %', fontsize=18)
    plt.grid(True, alpha=0.3)
    
    # Set y-axis limits
    plt.ylim(ymin, 0.96)
    
    # Add horizontal line at 94.01 (the original reported accuracy)
    plt.axhline(y=0.9401, color='gray', linestyle='--', alpha=0.7, 
                label='Top Muon: 94.01%')
    plt.legend(fontsize=16)
    
    # Add value labels above points
    
    #for coeff, mean, std in zip(sgd_coeffs, mean_accs, std_accs):
    #    plt.text(1 - coeff, max(ymin, mean + std), f'{100 * mean:.2f}', 
    #            ha='center', va='bottom', fontweight='bold', fontsize=15)
    plt.xticks(np.arange(0, 1.7, 0.2))
    plt.tight_layout()
    plt.savefig('sgd_coeff_results.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def plot_results_from_file(filename='sgd_coeff_experiment_results.pt'):
    """Load and plot results from a saved .pt file."""
    try:
        # Load the results with weights_only=False to handle numpy arrays
        results = torch.load(filename, weights_only=False)
        
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
        print("\nTrying alternative loading method...")
        try:
            # Alternative loading method for newer PyTorch versions
            results = torch.load(filename, map_location='cpu')
            
            sgd_coeffs = results['sgd_coeffs']
            mean_accs = results['mean_accs']
            std_accs = results['std_accs']
            
            print(f"Successfully loaded results using alternative method!")
            print(f"Number of experiments: {len(sgd_coeffs)}")
            print(f"Best accuracy: {results['best_acc']:.4f} ± {results['best_std']:.4f} at sgd_coeff = {results['best_coeff']}")
            
            # Create the plot
            plot_results(sgd_coeffs, mean_accs, std_accs)
            
        except Exception as e2:
            print(f"Alternative loading also failed: {e2}")
            print("\nTrying pickle file as fallback...")
            try:
                # Try to load the corresponding pickle file
                pickle_filename = filename.replace('.pt', '.pkl')
                import pickle
                with open(pickle_filename, 'rb') as f:
                    results = pickle.load(f)
                
                sgd_coeffs = results['sgd_coeffs']
                mean_accs = results['mean_accs']
                std_accs = results['std_accs']
                
                print(f"Successfully loaded results from pickle file {pickle_filename}!")
                print(f"Number of experiments: {len(sgd_coeffs)}")
                print(f"Best accuracy: {results['best_acc']:.4f} ± {results['best_std']:.4f} at sgd_coeff = {results['best_coeff']}")
                
                # Create the plot
                plot_results(sgd_coeffs, mean_accs, std_accs)
                
            except Exception as e3:
                print(f"Pickle loading also failed: {e3}")
                print("\nThis might be due to PyTorch version compatibility issues.")
                print("Try updating your PyTorch version or check the file format.")
                print("\nAvailable files in current directory:")
                import glob
                for f in glob.glob("*.*"):
                    print(f"  - {f}")

def list_available_files():
    """List all available result files."""
    import os
    import glob
    
    pt_files = glob.glob("*.pt")
    pkl_files = glob.glob("*.pkl")
    
    if pt_files or pkl_files:
        print("Available result files:")
        for file in sorted(pt_files + pkl_files):
            print(f"  - {file}")
    else:
        print("No result files found in current directory.")
    
    return pt_files + pkl_files

if __name__ == "__main__":
    # List available files
    available_files = list_available_files()
    
    if available_files:
        # Try to plot the main results file first
        if 'muon_alphas.pt' in available_files:
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
