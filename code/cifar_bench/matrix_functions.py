import cupy as cp
import time
import numpy as np
import torch
import torch.utils.dlpack as thd
from cupyx.scipy.sparse.linalg import svds as cupyx_svds


# we use to compute tau for the Oseledets' update
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


def find_tau_torch(S, tau_start, eps=1e-2):
    smax = torch.max(S)
    f_mid = torch.sum(torch.clamp(S - tau_start, min=0))
    if abs(f_mid - tau_start) < eps * smax:
        # print("Already ok")
        return tau_start
    # print("Not ok")
    if f_mid > tau_start:
        low = tau_start
    else:
        high = tau_start
 
    low, high = torch.tensor(0.0, device=S.device), smax
    while high - low > eps * smax:
        mid = (low + high) / 2
        f_mid = torch.sum(torch.clamp(S - mid, min=0))
        if f_mid > mid:
            low = mid
        else:
            high = mid
    return (low + high) / 2


# we use to run the old F*-Neon
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


# slow approach that uses full SVD and torch
def svd_full_approximation(A, tau=0):
    k = 0
    if (k == 0):
        k = min(A.shape[0],A.shape[1])
    # A = A.double() # uncomment this if you need higher precision
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    opt_tau = find_tau_torch(S, tau)
    S_thresholded = torch.clamp(S - opt_tau, min=0)
    # print(sum(S_thresholded), opt_tau)
    approx = U @ torch.diag(S_thresholded) @ Vh
    return approx, opt_tau, k

    # S_thresholded = torch.clamp(S - tau, min=0) 
    # A_reconstructed = U[:,:k] @ torch.diag(S[:k]) @ Vh[:k,:]
    # print(f"condition {S[0]/S[-1]:.4e}")
    return A_reconstructed


# code produced by Cursor, so it needs some verification
def randomized_svd_torch_full(G, n_components=1, n_oversamples=5, n_iter=2):
    """
    Full randomized SVD in PyTorch to compute top k singular triplets.
    
    This function implements the randomized SVD algorithm as described in:
    Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure 
    with randomness: Probabilistic algorithms for constructing approximate 
    matrix decompositions. SIAM review, 53(2), 217-288.
    
    Args:
        G (torch.Tensor): Input matrix of shape (m, n)
        n_components (int): Number of singular values/vectors to compute
        n_oversamples (int): Extra dimensions to improve accuracy (default: 5)
        n_iter (int): Number of power iterations for accuracy (default: 2)
    
    Returns:
        tuple: (U, S, Vt) where:
            - U (torch.Tensor): Left singular vectors of shape (m, n_components)
            - S (torch.Tensor): Singular values of shape (n_components,)
            - Vt (torch.Tensor): Right singular vectors of shape (n_components, n)
    
    Note:
        This function is more efficient than full SVD for large matrices when
        only a few singular values are needed. The accuracy improves with
        n_oversamples and n_iter, but at the cost of more computation.
    """
    # Ensure G is a torch tensor
    if not isinstance(G, torch.Tensor):
        G = torch.tensor(G, device=G.device if hasattr(G, 'device') else 'cpu')
    
    m, n = G.shape
    
    # Handle edge cases
    if m == 0 or n == 0:
        raise ValueError("Matrix dimensions must be positive")
    
    if n_components > min(m, n):
        n_components = min(m, n)
        print(f"Warning: n_components reduced to {n_components}")
    
    l = n_components + n_oversamples
    l = min(l, min(m, n))  # Ensure l doesn't exceed matrix dimensions
    
    # Ensure we use torch dtypes - avoid half precision for QR decomposition
    dtype = G.dtype if isinstance(G.dtype, torch.dtype) else torch.float32
    if dtype == torch.float16:
        dtype = torch.float32  # QR decomposition not supported for half precision on CUDA
    device = G.device
    
    # Convert to appropriate dtype if needed
    G_compute = G.to(dtype) if G.dtype != dtype else G
    
    # Step 1: Random test matrix
    Omega = torch.randn(n, l, device=device, dtype=dtype)
    
    # Step 2: Sample Y = G * Omega
    Y = G_compute @ Omega
    
    # Step 3: Power iterations (optional, improves accuracy if spectrum decays slowly)
    if n_iter > 0:
        for _ in range(n_iter):
            Y = G_compute @ (G_compute.T @ Y)
    
    # Step 4: Orthonormalize Y (QR)
    Q, _ = torch.linalg.qr(Y, mode='reduced')
    
    # Step 5: B = Q^T * G  (small matrix)
    B = Q.T @ G_compute
    
    # Step 6: SVD of small matrix B
    Ub, S, Vt = torch.linalg.svd(B, full_matrices=False)
    
    # Step 7: Recover U
    U = Q @ Ub
    
    # Return only the requested number of components
    return U[:, :n_components], S[:n_components], Vt[:n_components, :]


# the same idea, but we use only u1, sigma1, v1t, which is preposterous
def randomized_svd_torch(G, n_components=1, n_oversamples=5, n_iter=2):
    """
    Randomized SVD in PyTorch to compute the top singular triplet.
    
    This is a simplified version that returns only the first (largest) 
    singular value and corresponding singular vectors.
    
    Args:
        G (torch.Tensor): Input matrix of shape (m, n)
        n_components (int): Must be 1 for this function (kept for compatibility)
        n_oversamples (int): Extra dimensions to improve accuracy (default: 5)
        n_iter (int): Number of power iterations for accuracy (default: 2)
    
    Returns:
        tuple: (u1, sigma1, v1) where:
            - u1 (torch.Tensor): First left singular vector of shape (m,)
            - sigma1 (torch.Tensor): First singular value (scalar)
            - v1 (torch.Tensor): First right singular vector of shape (n,)
    
    Note:
        This function is optimized for computing only the largest singular
        value and vectors. For multiple singular values, use 
        randomized_svd_torch_full instead.
    """
    # Ensure G is a torch tensor
    if not isinstance(G, torch.Tensor):
        G = torch.tensor(G, device=G.device if hasattr(G, 'device') else 'cpu')
    
    m, n = G.shape
    
    # Handle edge cases
    if m == 0 or n == 0:
        raise ValueError("Matrix dimensions must be positive")
    
    if n_components > min(m, n):
        n_components = min(m, n)
        print(f"Warning: n_components reduced to {n_components}")
    
    l = n_components + n_oversamples
    l = min(l, min(m, n))  # Ensure l doesn't exceed matrix dimensions
    
    # Ensure we use torch dtypes - avoid half precision for QR decomposition
    dtype = G.dtype if isinstance(G.dtype, torch.dtype) else torch.float32
    if dtype == torch.float16:
        dtype = torch.float32  # QR decomposition not supported for half precision on CUDA
    device = G.device
    
    # Convert to appropriate dtype if needed
    G_compute = G.to(dtype) if G.dtype != dtype else G
    
    # Step 1: Random test matrix
    Omega = torch.randn(n, l, device=device, dtype=dtype)
    
    # Step 2: Sample Y = G * Omega
    Y = G_compute @ Omega
    
    # Step 3: Power iterations (optional, improves accuracy if spectrum decays slowly)
    if n_iter > 0:
        for _ in range(n_iter):
            Y = G_compute @ (G_compute.T @ Y)
    
    # Step 4: Orthonormalize Y (QR)
    Q, _ = torch.linalg.qr(Y, mode='reduced')
    
    # Step 5: B = Q^T * G  (small matrix)
    B = Q.T @ G_compute
    
    # Step 6: SVD of small matrix B
    Ub, S, Vt = torch.linalg.svd(B, full_matrices=False)
    
    # Step 7: Recover U
    U = Q @ Ub
    
    # Take the first singular triplet
    u1 = U[:, 0]
    sigma1 = S[0]
    v1 = Vt[0, :]
    
    return u1, sigma1, v1


# used in k-Neon with possibility of Error Feedback
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

# used in Neon
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


# similar to several_sv_svds, but returns only a product
def lanczos_svdt(W_torch, k, num_iter=50):
    """SVD approximation using the top k singular values and corresponding vectors."""
    
    # Store original device and dtype
    original_device = W_torch.device
    original_dtype = W_torch.dtype
    
    # assert (W.shape[0] > 3 and W.shape[1] > 3), "Dimensions of W must be greater than 3"
    W = cp.from_dlpack(thd.to_dlpack(W_torch))
    U, S, Vt = cupyx_svds(W, k=k, maxiter=num_iter, which='LM')
    approx = U @ cp.diag(S) @ Vt
    approx_torch = thd.from_dlpack(approx.toDlpack()).to(device=original_device, dtype=original_dtype)
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