# this file shows that 99.5% of all time if spent on SVDS (and only 0.1% on conversion)
# this means that the problem is with Lanczos or its implementation!
import torch
import cupy as cp
import cupyx.scipy.sparse.linalg as cupyx_svds
import time
import numpy as np

def profile_svd_operations(W_torch, k, num_iter=30):
    """Profile the time taken by each operation in the SVD approximation."""
    
    print(f"Matrix shape: {W_torch.shape}")
    print(f"k: {k}, num_iter: {num_iter}")
    print("-" * 50)
    
    # Operation 1: Torch to CuPy conversion
    start = time.time()
    W = cp.from_dlpack(torch.utils.dlpack.to_dlpack(W_torch))
    torch_to_cupy_time = time.time() - start
    print(f"1. Torch to CuPy conversion: {torch_to_cupy_time:.6f} seconds")
    
    # Operation 2: SVD computation
    start = time.time()
    U, S, Vt = cupyx_svds.svds(W, k=k, maxiter=num_iter, which='LM')
    svd_time = time.time() - start
    print(f"2. SVD computation: {svd_time:.6f} seconds")
    
    # Operation 3: Matrix multiplication for approximation
    start = time.time()
    approx = U @ cp.diag(S) @ Vt
    approx_time = time.time() - start
    print(f"3. Approximation calculation: {approx_time:.6f} seconds")
    
    # Operation 4: CuPy to Torch conversion
    start = time.time()
    approx_torch = torch.utils.dlpack.from_dlpack(approx.toDlpack())
    cupy_to_torch_time = time.time() - start
    print(f"4. CuPy to Torch conversion: {cupy_to_torch_time:.6f} seconds")
    
    # Summary
    total_time = torch_to_cupy_time + svd_time + approx_time + cupy_to_torch_time
    print("-" * 50)
    print(f"Total time: {total_time:.6f} seconds")
    print(f"SVD computation percentage: {(svd_time/total_time)*100:.1f}%")
    print(f"Data conversion percentage: {((torch_to_cupy_time + cupy_to_torch_time)/total_time)*100:.1f}%")
    print(f"Approximation calculation percentage: {(approx_time/total_time)*100:.1f}%")
    
    return {
        'torch_to_cupy': torch_to_cupy_time,
        'svd': svd_time,
        'approx': approx_time,
        'cupy_to_torch': cupy_to_torch_time,
        'total': total_time
    }

def main():
    # Test with different matrix sizes
    sizes = [1000, 2000, 3000, 5000]
    k_values = [5, 10, 20]
    
    for N in sizes:
        print(f"\n{'='*60}")
        print(f"Testing with N = {N}")
        print(f"{'='*60}")
        
        # Generate random matrix
        np.random.seed(452)
        W_numpy = np.random.randn(N, N).astype(np.float32)
        W_torch = torch.from_numpy(W_numpy).cuda()
        
        for k in k_values:
            if k >= min(N, N):  # Skip if k is too large
                continue
            print(f"\n--- k = {k} ---")
            profile_svd_operations(W_torch, k, num_iter=30)

if __name__ == '__main__':
    main()
