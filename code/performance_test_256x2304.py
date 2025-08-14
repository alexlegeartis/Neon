import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import statistics

# Import the functions to test
from optimizers import zeropower_via_newtonschulz5
from matrix_functions import one_sv_svds_approximation

def generate_test_matrix(shape=(256, 2304), device='cuda', dtype=torch.float32, seed=42):
    """Generate a test matrix with specified shape and properties."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    matrix = torch.randn(shape, device=device, dtype=dtype)
    return matrix

def time_function(func, matrix, num_runs=10, warmup_runs=3, **kwargs):
    """Time a function with warmup runs and multiple measurements."""
    device = matrix.device
    
    # Warmup runs
    for _ in range(warmup_runs):
        torch.cuda.synchronize() if device.type == 'cuda' else None
        _ = func(matrix.clone(), **kwargs)
        torch.cuda.synchronize() if device.type == 'cuda' else None
    
    # Actual timing runs
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.perf_counter()
        result = func(matrix.clone(), **kwargs)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return times, result

def calculate_accuracy_metrics(original, approx1, approx2):
    """Calculate various accuracy metrics between the original and approximations."""
    metrics = {}
    
    # Frobenius norm errors
    frobenius_norm = torch.norm(original, 'fro')
    error1 = torch.norm(original - approx1, 'fro') / frobenius_norm
    error2 = torch.norm(original - approx2, 'fro') / frobenius_norm
    
    metrics['frobenius_error_newton'] = error1.item()
    metrics['frobenius_error_svd'] = error2.item()
    
    # Spectral norm errors
    spectral_norm = torch.norm(original, 2)
    spectral_error1 = torch.norm(original - approx1, 2) / spectral_norm
    spectral_error2 = torch.norm(original - approx2, 2) / spectral_norm
    
    metrics['spectral_error_newton'] = spectral_error1.item()
    metrics['spectral_error_svd'] = spectral_error2.item()
    
    # Matrix properties comparison
    metrics['original_frobenius_norm'] = frobenius_norm.item()
    metrics['original_spectral_norm'] = spectral_norm.item()
    metrics['newton_frobenius_norm'] = torch.norm(approx1, 'fro').item()
    metrics['svd_frobenius_norm'] = torch.norm(approx2, 'fro').item()
    
    return metrics

def run_comprehensive_test():
    """Run comprehensive performance and accuracy tests."""
    print("=" * 80)
    print("Performance Test: zeropower_via_newtonschulz5 vs one_sv_svds_approximation")
    print("Matrix shape: 256 x 2304")
    print("=" * 80)
    
    # Test parameters
    shape = (10600, 10040)
    num_runs = 20
    warmup_runs = 5
    
    # Test on different devices if available
    # devices = ['cpu']
    devices = []
    if torch.cuda.is_available():
        devices.append('cuda')
    
    results = defaultdict(dict)
    
    for device in devices:
        print(f"\n--- Testing on {device.upper()} ---")
        
        # Generate test matrix
        matrix = generate_test_matrix(shape, device=device)
        print(f"Generated matrix shape: {matrix.shape}")
        print(f"Matrix dtype: {matrix.dtype}")
        print(f"Matrix device: {matrix.device}")
        
        # Test Newton-Schulz method
        print("\nTesting zeropower_via_newtonschulz5...")
        newton_times, newton_result = time_function(
            zeropower_via_newtonschulz5, 
            matrix, 
            num_runs=num_runs, 
            warmup_runs=warmup_runs
        )
        
        # Test SVD approximation method
        print("Testing one_sv_svds_approximation...")
        svd_times, svd_result = time_function(
            one_sv_svds_approximation, 
            matrix, 
            num_runs=num_runs, 
            warmup_runs=warmup_runs,
            num_iter=30
        )
        
        # Calculate accuracy metrics
        accuracy_metrics = calculate_accuracy_metrics(matrix, newton_result, svd_result)
        
        # Store results
        results[device] = {
            'newton_times': newton_times,
            'svd_times': svd_times,
            'newton_result': newton_result,
            'svd_result': svd_result,
            'accuracy_metrics': accuracy_metrics
        }
        
        # Print timing statistics
        newton_mean = statistics.mean(newton_times)
        newton_std = statistics.stdev(newton_times) if len(newton_times) > 1 else 0
        svd_mean = statistics.mean(svd_times)
        svd_std = statistics.stdev(svd_times) if len(svd_times) > 1 else 0
        
        print(f"\nTiming Results ({device.upper()}):")
        print(f"Newton-Schulz: {newton_mean:.6f} ± {newton_std:.6f} seconds")
        print(f"SVD Approximation: {svd_mean:.6f} ± {svd_std:.6f} seconds")
        print(f"Speedup factor: {svd_mean/newton_mean:.2f}x")
        
        print(f"\nAccuracy Metrics ({device.upper()}):")
        print(f"Frobenius error (Newton): {accuracy_metrics['frobenius_error_newton']:.6f}")
        print(f"Frobenius error (SVD): {accuracy_metrics['frobenius_error_svd']:.6f}")
        print(f"Spectral error (Newton): {accuracy_metrics['spectral_error_newton']:.6f}")
        print(f"Spectral error (SVD): {accuracy_metrics['spectral_error_svd']:.6f}")
    
    return results

def create_performance_plots(results):
    """Create performance visualization plots."""
    if not results:
        print("No results to plot")
        return
    
    devices = list(results.keys())
    fig, axes = plt.subplots(2, len(devices), figsize=(6*len(devices), 10))
    if len(devices) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, device in enumerate(devices):
        device_results = results[device]
        newton_times = device_results['newton_times']
        svd_times = device_results['svd_times']
        
        # Timing comparison boxplot
        axes[0, i].boxplot([newton_times, svd_times], 
                          labels=['Newton-Schulz', 'SVD Approx'])
        axes[0, i].set_title(f'Execution Time Comparison\n({device.upper()})')
        axes[0, i].set_ylabel('Time (seconds)')
        axes[0, i].grid(True, alpha=0.3)
        
        # Accuracy comparison bar plot
        accuracy = device_results['accuracy_metrics']
        methods = ['Newton-Schulz', 'SVD Approx']
        frobenius_errors = [accuracy['frobenius_error_newton'], accuracy['frobenius_error_svd']]
        spectral_errors = [accuracy['spectral_error_newton'], accuracy['spectral_error_svd']]
        
        x = np.arange(len(methods))
        width = 0.35
        
        axes[1, i].bar(x - width/2, frobenius_errors, width, label='Frobenius Error', alpha=0.8)
        axes[1, i].bar(x + width/2, spectral_errors, width, label='Spectral Error', alpha=0.8)
        axes[1, i].set_title(f'Approximation Error Comparison\n({device.upper()})')
        axes[1, i].set_ylabel('Relative Error')
        axes[1, i].set_xticks(x)
        axes[1, i].set_xticklabels(methods)
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('/legeartis/Neon/code/performance_comparison_256x2304.png', dpi=300, bbox_inches='tight')
    print(f"\nPerformance plots saved as: performance_comparison_256x2304.png")
    plt.show()

def detailed_analysis(results):
    """Perform detailed analysis of the results."""
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)
    
    for device, device_results in results.items():
        print(f"\n--- {device.upper()} Analysis ---")
        
        newton_times = device_results['newton_times']
        svd_times = device_results['svd_times']
        accuracy = device_results['accuracy_metrics']
        
        # Statistical analysis
        newton_stats = {
            'mean': statistics.mean(newton_times),
            'median': statistics.median(newton_times),
            'std': statistics.stdev(newton_times) if len(newton_times) > 1 else 0,
            'min': min(newton_times),
            'max': max(newton_times)
        }
        
        svd_stats = {
            'mean': statistics.mean(svd_times),
            'median': statistics.median(svd_times),
            'std': statistics.stdev(svd_times) if len(svd_times) > 1 else 0,
            'min': min(svd_times),
            'max': max(svd_times)
        }
        
        print(f"\nNewton-Schulz Statistics:")
        print(f"  Mean: {newton_stats['mean']:.6f}s")
        print(f"  Median: {newton_stats['median']:.6f}s")
        print(f"  Std Dev: {newton_stats['std']:.6f}s")
        print(f"  Min: {newton_stats['min']:.6f}s")
        print(f"  Max: {newton_stats['max']:.6f}s")
        
        print(f"\nSVD Approximation Statistics:")
        print(f"  Mean: {svd_stats['mean']:.6f}s")
        print(f"  Median: {svd_stats['median']:.6f}s")
        print(f"  Std Dev: {svd_stats['std']:.6f}s")
        print(f"  Min: {svd_stats['min']:.6f}s")
        print(f"  Max: {svd_stats['max']:.6f}s")
        
        print(f"\nPerformance Comparison:")
        print(f"  Speedup (mean): {svd_stats['mean']/newton_stats['mean']:.2f}x")
        print(f"  Speedup (median): {svd_stats['median']/newton_stats['median']:.2f}x")
        
        print(f"\nAccuracy Analysis:")
        print(f"  Newton-Schulz Frobenius error: {accuracy['frobenius_error_newton']:.6e}")
        print(f"  SVD Approximation Frobenius error: {accuracy['frobenius_error_svd']:.6e}")
        print(f"  Accuracy ratio (Frobenius): {accuracy['frobenius_error_svd']/accuracy['frobenius_error_newton']:.2f}")
        print(f"  Newton-Schulz Spectral error: {accuracy['spectral_error_newton']:.6e}")
        print(f"  SVD Approximation Spectral error: {accuracy['spectral_error_svd']:.6e}")
        print(f"  Accuracy ratio (Spectral): {accuracy['spectral_error_svd']/accuracy['spectral_error_newton']:.2f}")

def main():
    """Main function to run all tests."""
    print("Starting comprehensive performance test...")
    
    # Run the main test
    results = run_comprehensive_test()
    
    # Create visualizations
    try:
        create_performance_plots(results)
    except Exception as e:
        print(f"Warning: Could not create plots: {e}")
    
    # Detailed analysis
    detailed_analysis(results)
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main()
