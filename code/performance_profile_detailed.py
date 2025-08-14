import torch
import cupy as cp
import cupyx.scipy.sparse.linalg as cupyx_svds
import time
import numpy as np

def profile_svd_operations_detailed(W_torch, k, num_iter=30, num_runs=5):
    """Profile the time taken by each operation with multiple runs for stability."""
    
    print(f"Matrix shape: {W_torch.shape}")
    print(f"k: {k}, num_iter: {num_iter}, num_runs: {num_runs}")
    print("-" * 60)
    
    times = {
        'torch_to_cupy': [],
        'svd': [],
        'approx': [],
        'cupy_to_torch': []
    }
    
    for run in range(num_runs):
        # Operation 1: Torch to CuPy conversion
        start = time.time()
        W = cp.from_dlpack(torch.utils.dlpack.to_dlpack(W_torch))
        torch_to_cupy_time = time.time() - start
        times['torch_to_cupy'].append(torch_to_cupy_time)
        
        # Operation 2: SVD computation
        start = time.time()
        U, S, Vt = cupyx_svds.svds(W, k=k, maxiter=num_iter, which='LM')
        svd_time = time.time() - start
        times['svd'].append(svd_time)
        
        # Operation 3: Matrix multiplication for approximation
        start = time.time()
        approx = U @ cp.diag(S) @ Vt
        approx_time = time.time() - start
        times['approx'].append(approx_time)
        
        # Operation 4: CuPy to Torch conversion
        start = time.time()
        approx_torch = torch.utils.dlpack.from_dlpack(approx.toDlpack())
        cupy_to_torch_time = time.time() - start
        times['cupy_to_torch'].append(cupy_to_torch_time)
        
        print(f"Run {run+1}: SVD={svd_time:.6f}s, Conv={torch_to_cupy_time+cupy_to_torch_time:.6f}s, Approx={approx_time:.6f}s")
    
    # Calculate averages
    avg_times = {k: np.mean(v) for k, v in times.items()}
    total_time = sum(avg_times.values())
    
    print("-" * 60)
    print("AVERAGE TIMES:")
    print(f"1. Torch to CuPy conversion: {avg_times['torch_to_cupy']:.6f} seconds")
    print(f"2. SVD computation: {avg_times['svd']:.6f} seconds")
    print(f"3. Approximation calculation: {avg_times['approx']:.6f} seconds")
    print(f"4. CuPy to Torch conversion: {avg_times['cupy_to_torch']:.6f} seconds")
    print("-" * 60)
    print(f"Total time: {total_time:.6f} seconds")
    print(f"SVD computation percentage: {(avg_times['svd']/total_time)*100:.1f}%")
    print(f"Data conversion percentage: {((avg_times['torch_to_cupy'] + avg_times['cupy_to_torch'])/total_time)*100:.1f}%")
    print(f"Approximation calculation percentage: {(avg_times['approx']/total_time)*100:.1f}%")
    
    return avg_times

def main():
    # Test with a larger matrix for more significant differences
    N = 5000
    k = 10
    
    print(f"Detailed profiling with N = {N}, k = {k}")
    print("=" * 80)
    
    # Generate random matrix
    np.random.seed(452)
    W_numpy = np.random.randn(N, N).astype(np.float32)
    W_torch = torch.from_numpy(W_numpy).cuda()
    
    profile_svd_operations_detailed(W_torch, k, num_iter=30, num_runs=5)

if __name__ == '__main__':
    main()
