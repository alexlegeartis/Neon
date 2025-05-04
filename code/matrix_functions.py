import cupy as cp
import time
import numpy as np
import torch
import torch.utils.dlpack as thd
from cupyx.scipy.sparse.linalg import svds as cupyx_svds

def find_tau(S, tau_start, eps=1e-2):
    smax = cp.max(S)
    f_mid = cp.sum(cp.maximum(S - tau_start, 0))
    if abs(f_mid - tau_start) < eps * smax:
        # print("Already ok")
        return tau_start
    # print("Not ok")
    if f_mid > tau_start:
        low = tau_start
    else:
        high = high = tau_start
 
    low, high = 0, smax
    while high - low > eps * smax:
        mid = (low + high) / 2
        f_mid = cp.sum(cp.maximum(S - mid, 0))
        if f_mid > mid:
            low = mid
        else:
            high = mid
    return (low + high) / 2

def k_sv_svds_approximation_dlpack(W_torch, k, tau, num_iter=30):
    """SVD approximation using the top k singular values and corresponding vectors."""
    W = cp.from_dlpack(thd.to_dlpack(W_torch)).astype(cp.float32)
    kmax = max(W.shape[0] - 2, W.shape[1] - 2) # k + 1 < min(m, n) due to the method
    U, S, Vt = cupyx_svds(W, k=min([k, kmax]), maxiter=num_iter, which='LM') 
    opt_tau = find_tau(S, tau)
    if k == kmax:
        print(f"Warning: Lanczos algorithm reached {kmax} singular values")
    if S[0] > opt_tau and k < kmax:
        #print(S)
        # print(opt_tau)
        return k_sv_svds_approximation_dlpack(W_torch, k + 1, tau, num_iter)
        # print(S[-1])
    S_thresholded = cp.maximum(S - opt_tau, 0)
    # print(sum(S_thresholded), opt_tau)
    approx = U @ cp.diag(S_thresholded) @ Vt
    approx_torch = thd.from_dlpack(approx.toDlpack()) 
    return approx_torch, opt_tau, k

def one_sv_svds_approximation(W_torch, num_iter=30):
    """SVD approximation using the top k singular values and corresponding vectors."""
    k = 1
    # assert (W.shape[0] > 3 and W.shape[1] > 3), "Dimensions of W must be greater than 3"
    W = cp.from_dlpack(thd.to_dlpack(W_torch)).astype(cp.float32)
    # if W.shape[0] <= 3 or W.shape[1] <= 3:
        #print(f"strange {W}")
        #return W_torch
    U, S, Vt = cupyx_svds(W, k=min([k, W.shape[0] - 1, W.shape[1] - 1]), maxiter=num_iter, which='LM')
    approx = U @ cp.diag(S) @ Vt
    approx_torch = thd.from_dlpack(approx.toDlpack()) 
    return approx_torch

def lanczos_svdt(W_torch, k, num_iter=30):
    """SVD approximation using the top k singular values and corresponding vectors."""
    
    # assert (W.shape[0] > 3 and W.shape[1] > 3), "Dimensions of W must be greater than 3"
    W = cp.from_dlpack(thd.to_dlpack(W_torch))
    U, S, Vt = cupyx_svds(W, k=k, maxiter=num_iter, which='LM')
    approx = U @ cp.diag(S) @ Vt
    approx_torch = thd.from_dlpack(approx.toDlpack()) 
    return approx_torch

def main():
    # Parameters
    N = 5000
    k = 5
    tau = 0.01

    # Generate random matrix
    np.random.seed(452)
    W_numpy = np.random.randn(N, N)
    W_torch = torch.from_numpy(W_numpy).cuda()

    # Standard SVD
    start = time.time()
    U, S, Vt = np.linalg.svd(W_numpy)
    result_precise = U[:,:k] @ np.diag(S[:k]) @ Vt[:k,:]
    time_precise = time.time() - start

    # Approximate SVD with thresholding
    start = time.time()
    result_approx = lanczos_svdt(W_torch, k, 40)
    time_approx = time.time() - start

    # Convert results to numpy for comparison
    result_approx_np = result_approx.cpu().numpy()

    # Calculate relative Frobenius error
    frobenius_norm = np.linalg.norm(W_numpy, 'fro')
    error = np.linalg.norm(result_precise - result_approx_np, 'fro') / frobenius_norm

    print(f"Standard SVD time: {time_precise:.4f} seconds")
    print(f"Approximate SVD time: {time_approx:.4f} seconds")
    print(f"Relative Frobenius error: {error:.6f}")

if __name__ == '__main__':
    main()