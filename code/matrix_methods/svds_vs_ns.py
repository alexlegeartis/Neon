import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import statistics

import cupy as cp
from cupyx.scipy.sparse.linalg import svds as cupyx_svds
import torch.utils.dlpack as thd

def several_sv_svds_approximation(W_torch, k, num_iter=50):
    """SVD approximation using the top k singular values and corresponding vectors."""
    # Store original device and dtype
    original_device = W_torch.device
    original_dtype = W_torch.dtype
    
    W = cp.from_dlpack(thd.to_dlpack(W_torch)).astype(cp.float32)
    U, S, Vt = cupyx_svds(W, k=min([k, W.shape[0] - 1, W.shape[1] - 1]), maxiter=num_iter, which='LM')

    # Convert back to torch tensors and ensure they're on the correct device
    approx_torch_U = thd.from_dlpack(U.toDlpack()).to(device=original_device, dtype=original_dtype)
    approx_torch_S = thd.from_dlpack(S.toDlpack()).to(device=original_device, dtype=original_dtype)
    approx_torch_Vt = thd.from_dlpack(Vt.toDlpack()).to(device=original_device, dtype=original_dtype)
    
    return approx_torch_U, approx_torch_S, approx_torch_Vt

# Import the functions to test
def zeropower_via_newtonschulz5(G, steps=3, eps=1e-7):
    """Simplified Newton-Schulz iteration for whitening"""
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    # Add numerical stability
    norm = X.norm() + eps
    if norm < eps:
        return torch.zeros_like(X)
    X /= norm
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

def one_sv_svds_approximation(W_torch, num_iter=30):
    """SVD approximation using the top k singular values and corresponding vectors."""
    k = 1
    # Store original device and dtype
    original_device = W_torch.device
    original_dtype = W_torch.dtype
    
    W = cp.from_dlpack(thd.to_dlpack(W_torch)).astype(cp.float32)
    U, S, Vt = cupyx_svds(W, k=min([k, W.shape[0] - 1, W.shape[1] - 1]), maxiter=num_iter, which='LM')

    approx = U @ Vt #cp.diag(S) - we don't need this!
    approx_torch = thd.from_dlpack(approx.toDlpack()).to(device=original_device, dtype=original_dtype)
    return approx_torch, float(S[0]) # sigma1

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


def run_comprehensive_test():
    """Run comprehensive performance and accuracy tests."""
    print("=" * 80)
    print("Performance Test: zeropower_via_newtonschulz5 vs one_sv_svds_approximation")
    print("Matrix shape: 256 x 2304")
    print("=" * 80)
    
    # Test parameters
    shape = (1060, 1000)
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
        # several_svd_svds
        svd_times, svd_result = time_function(
            several_sv_svds_approximation, 
            matrix, 
            num_runs=num_runs, 
            warmup_runs=warmup_runs,
            num_iter=30,
            k=10
        )
        '''
        svd_times, svd_result = time_function(
            one_sv_svds_approximation, 
            matrix, 
            num_runs=num_runs, 
            warmup_runs=warmup_runs,
            num_iter=30
        )'''
        
        
        # Store results
        results[device] = {
            'newton_times': newton_times,
            'svd_times': svd_times,
            'newton_result': newton_result,
            'svd_result': svd_result,
        }
        
        # Print timing statistics
        newton_mean = statistics.mean(newton_times)
        newton_std = statistics.stdev(newton_times) if len(newton_times) > 1 else 0
        svd_mean = statistics.mean(svd_times)
        svd_std = statistics.stdev(svd_times) if len(svd_times) > 1 else 0
        
        print(f"\nTiming Results ({device.upper()}):")
        print(f"Newton-Schulz: {newton_mean:.6f} ± {newton_std:.6f} seconds")
        print(f"SVD Approximation: {svd_mean:.6f} ± {svd_std:.6f} seconds")
        print(f"T_SVDS / T_NS factor: {svd_mean/newton_mean:.2f}x")
        
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
        print(f"  T_SVDS / T_NS (mean): {svd_stats['mean']/newton_stats['mean']:.2f}x")
        print(f"  T_SVDS / T_NS (median): {svd_stats['median']/newton_stats['median']:.2f}x")
        
        
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
