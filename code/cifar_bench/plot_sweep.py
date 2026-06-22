import torch
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1. Load the data
    filename = 'lr_robustness_results.pt'
    try:
        data = torch.load(filename)
        # Handle case where file contains metadata wrapper dict(code=..., results=...)
        if 'results' in data and isinstance(data['results'], dict):
            data = data['results']
    except FileNotFoundError:
        print(f"Error: {filename} not found. Ensure the training sweep has completed.")
        return

    # 2. Setup the Plot (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = {
        'Muon': '#1f77b4',           # Blue
        
        # Orange Family
        'NormalizedMuon': '#ff7f0e', # Dark Orange
        'NSGD': '#ffbb78',           # Light Orange
        
        # Green Family
        'SignSGDMuon': '#2ca02c',    # Dark Green
        'SignSGD': '#98df8a',        # Light Green
        'SignSGDNSGD': '#aabbcc'
    }

    # 3. Process and Plot each optimizer
    for opt_name, lr_dict in data.items():
        # Sort learning rates so lines are drawn correctly left-to-right
        lrs = sorted(list(lr_dict.keys()))
        
        acc_means, acc_stds = [], []
        loss_means, loss_stds = [], []
        
        for lr in lrs:
            # Extract metrics based on the new dictionary structure
            accs = np.array(lr_dict[lr]['acc'])
            losses = np.array(lr_dict[lr]['loss'])
            
            acc_means.append(accs.mean())
            acc_stds.append(accs.std())
            
            loss_means.append(losses.mean())
            loss_stds.append(losses.std())
            
        acc_means = np.array(acc_means)
        acc_stds = np.array(acc_stds)
        loss_means = np.array(loss_means)
        loss_stds = np.array(loss_stds)
        
        color = colors.get(opt_name, None)
        
        # --- Plot Accuracy on Left Axis (ax1) ---
        ax1.plot(lrs, acc_means, marker='o', linewidth=2, label=opt_name, color=color)
        ax1.fill_between(lrs, acc_means - acc_stds, acc_means + acc_stds, alpha=0.2, color=color)
        
        # --- Plot Loss on Right Axis (ax2) ---
        ax2.plot(lrs, loss_means, marker='s', linewidth=2, linestyle='--', label=opt_name, color=color)
        ax2.fill_between(lrs, loss_means - loss_stds, loss_means + loss_stds, alpha=0.2, color=color)

    # 4. Format the Accuracy Chart (ax1)
    ax1.set_xscale('log')
    ax1.set_xlabel('Learning Rate (Optimizer 2)', fontsize=12)
    ax1.set_ylabel('TTA Validation Accuracy', fontsize=12)
    ax1.set_title('Test Accuracy vs. Learning Rate', fontsize=14, fontweight='bold')
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend(fontsize=11)
    
    # 5. Format the Loss Chart (ax2)
    ax2.set_xscale('log')
    ax2.set_xlabel('Learning Rate (Optimizer 2)', fontsize=12)
    ax2.set_ylabel('Final Training Loss', fontsize=12)
    ax2.set_title('Training Loss vs. Learning Rate', fontsize=14, fontweight='bold')
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    
    # Ensure the Y-axis for loss has 0 as a baseline (avoids misleading zoomed axes)
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=11)
    
    # 6. Save and Show
    fig.tight_layout()
    plot_filename = 'lr_robustness_plot.png'
    plt.savefig(plot_filename, dpi=300)
    print(f"Plot successfully saved to {plot_filename}")
    
    plt.show()

if __name__ == "__main__":
    main()