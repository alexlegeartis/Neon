import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from optimizers import Muon, Neon
import time


def random_psd_with_nuclear_norm(size, nuclear_norm_value, device='cuda'):
    # Step 1: random orthogonal Q
    Q, _ = torch.linalg.qr(torch.randn(size, size, device=device))

    # Step 2: random nonnegative eigenvalues summing to nuclear_norm_value
    eigvals = torch.rand(size, device=device)
    eigvals = eigvals / eigvals.sum() * nuclear_norm_value  # scale sum exactly

    # Step 3: construct PSD matrix
    M = Q @ torch.diag(eigvals) @ Q.T
    return M

class CompositeFunction:
    """
    A composite function f(X) = (1/n) * sum(fi(X)) where:
    - X is an m x n matrix
    - Each fi = 1/2 <X-Si, Mi (X-Si) Ni> where Mi, Ni are symmetric positive semidefinite and Si are random matrices
    - Each fi has Lipschitz continuous gradient with respect to nuclear/spectral norm
    - Stochastic gradients have bounded variance sigma^2
    """
    
    def __init__(self, m, n, num_functions=100, norm_type='nuclear', sigma=0.1, device='cuda'):
        self.m = m
        self.n = n
        self.num_functions = num_functions
        self.norm_type = norm_type
        self.sigma = sigma
        self.device = device
        
        # Generate random matrices for each function component
        self.M_matrices = []  # Left multipliers (symmetric positive semidefinite)
        self.N_matrices = []  # Right multipliers (symmetric positive semidefinite)
        self.S_matrices = []  # Shift matrices
        self.lipschitz_constants = []  # Lipschitz constants for each component
        target_nuclear_norm = 10
        print(f"Generating {num_functions} matrices on {device}...")
        
        for i in range(num_functions):
            # Generate symmetric positive semidefinite matrix Mi
            Mi = random_psd_with_nuclear_norm(m, target_nuclear_norm, device)
            
            # Generate symmetric positive semidefinite matrix Ni
            Ni = random_psd_with_nuclear_norm(n, target_nuclear_norm, device)
            
            # Generate random shift matrix Si
            Si = torch.zeros(m, n, device=device)
            
            # Compute Lipschitz constant bounds for this component
            # Bound 1: ||M||_nuclear * ||N||_op
            M_nuclear = torch.linalg.matrix_norm(Mi, ord='nuc')
            N_operator = torch.linalg.matrix_norm(Ni, ord=2)
            bound1 = M_nuclear * N_operator
            
            # Bound 2: ||M||_op * ||N||_nuclear
            M_operator = torch.linalg.matrix_norm(Mi, ord=2)
            N_nuclear = torch.linalg.matrix_norm(Ni, ord='nuc')
            bound2 = M_operator * N_nuclear
            
            # Lipschitz constant for this component is the minimum of the two bounds
            lipschitz_const = min(bound1.item(), bound2.item())
            
            self.M_matrices.append(Mi)
            self.N_matrices.append(Ni)
            self.S_matrices.append(Si)
            self.lipschitz_constants.append(lipschitz_const)
        
        # Overall Lipschitz constant for the composite function
        self.overall_lipschitz = sum(self.lipschitz_constants) / self.num_functions
        print(f"Generated {num_functions} components with overall Lipschitz constant: {self.overall_lipschitz:.6f}")
        print(f"Component Lipschitz constants range: [{min(self.lipschitz_constants):.6f}, {max(self.lipschitz_constants):.6f}]")
    
    def objective(self, X):
        """Compute the full objective function f(X) = (1/n) * sum(fi(X))"""
        # Vectorized computation for better CUDA performance
        losses = torch.stack([self._component_function(X, i) for i in range(self.num_functions)])
        return torch.mean(losses)
    
    def _component_function(self, X, i):
        """Compute individual component function fi(X) = 1/2 <X-Si, Mi (X-Si) Ni>"""
        Mi = self.M_matrices[i]
        Ni = self.N_matrices[i]
        Si = self.S_matrices[i]
        
        # Compute X - Si
        X_minus_Si = X - Si
        
        # Compute Mi (X-Si) Ni
        Mi_X_minus_Si = Mi @ X_minus_Si
        Mi_X_minus_Si_Ni = Mi_X_minus_Si @ Ni
        
        # Compute the inner product <X-Si, Mi (X-Si) Ni>
        # This is trace((X-Si)^T @ Mi (X-Si) Ni) = trace((X-Si) @ Mi (X-Si) Ni)
        inner_product = torch.sum(X_minus_Si * Mi_X_minus_Si_Ni)
        
        return 0.5 * inner_product
    
    def gradient(self, X):
        """Compute the full gradient ∇f(X)"""
        # Vectorized computation for better CUDA performance
        grads = torch.stack([self._component_gradient(X, i) for i in range(self.num_functions)])
        return torch.mean(grads, dim=0)
    
    def _component_gradient(self, X, i):
        """Compute gradient of individual component ∇fi(X) for fi = 1/2 <X-Si, Mi (X-Si) Ni>"""
        Mi = self.M_matrices[i]
        Ni = self.N_matrices[i]
        Si = self.S_matrices[i]
        
        # Compute X - Si
        X_minus_Si = X - Si
        
        # For fi = 1/2 <X-Si, Mi (X-Si) Ni>, the gradient is:
        # ∇fi(X) = 1/2 * (Mi (X-Si) Ni + Mi^T (X-Si) Ni^T)
        
        # First term: Mi (X-Si) Ni
        term1 = Mi @ X_minus_Si @ Ni
        
        # Second term: Mi^T (X-Si) Ni^T
        term2 = Mi.T @ X_minus_Si @ Ni.T
        
        # Gradient is the sum of both terms
        return 0.5 * (term1 + term2)
    
    def stochastic_gradient(self, X, batch_size=1):
        """
        Compute stochastic gradient with bounded variance sigma^2
        Returns: (gradient, variance_bound)
        """
        if batch_size >= self.num_functions:
            return self.gradient(X), 0.0
        
        # Randomly sample batch_size functions
        indices = torch.randperm(self.num_functions, device=self.device)[:batch_size]
        
        # Vectorized computation for better CUDA performance
        batch_grads = torch.stack([self._component_gradient(X, idx) for idx in indices])
        batch_grad = torch.mean(batch_grads, dim=0)
        
        # Compute actual variance bound based on theoretical analysis
        variance_bound = self._compute_gradient_variance_bound(X, batch_size)
        
        return batch_grad, variance_bound
    
    def _compute_gradient_variance_bound(self, X, batch_size):
        """
        Compute theoretical variance bound for stochastic gradient
        
        For batch size b and total functions n, the variance is:
        E(||g(X) - ∇f(X)||²) ≤ (n-b)/(n*b) * (1/n) * Σ||∇fi(X)||²
        
        This follows from:
        1. Unbiasedness: E[g(X)] = ∇f(X)
        2. Variance decomposition: Var(g(X)) = E[Var(g(X)|B)] + Var(E[g(X)|B])
        3. Random sampling properties
        """
        n = self.num_functions
        
        if batch_size >= n:
            return 0.0
        
        # Compute variance coefficient: (n-b)/(n*b)
        variance_coeff = (n - batch_size) / (n * batch_size)
        
        # Compute sum of squared gradient norms: (1/n) * Σ||∇fi(X)||²
        grad_norms_squared = []
        for i in range(n):
            grad_i = self._component_gradient(X, i)
            grad_norm_sq = torch.linalg.matrix_norm(grad_i, ord='fro') ** 2
            grad_norms_squared.append(grad_norm_sq)
        
        # Average squared gradient norm
        avg_grad_norm_sq = torch.mean(torch.stack(grad_norms_squared))
        
        # Total variance bound
        variance_bound = variance_coeff * avg_grad_norm_sq
        
        return variance_bound.item()
    
    def stochastic_objective(self, X, batch_size=1):
        """
        Compute stochastic objective function using a subset of components
        Returns: (objective_value, variance_bound)
        """
        if batch_size >= self.num_functions:
            return self.objective(X), 0.0
        
        # Randomly sample batch_size functions
        indices = torch.randperm(self.num_functions, device=self.device)[:batch_size]
        
        # Vectorized computation for better CUDA performance
        batch_losses = torch.stack([self._component_function(X, idx) for idx in indices])
        batch_objective = torch.mean(batch_losses)
        
        # Compute actual variance bound based on theoretical analysis
        variance_bound = self._compute_objective_variance_bound(X, batch_size)
        
        return batch_objective, variance_bound
    
    def _compute_objective_variance_bound(self, X, batch_size):
        """
        Compute theoretical variance bound for stochastic objective
        
        For batch size b and total functions n, the variance is:
        E(||f_b(X) - f(X)||²) ≤ (n-b)/(n*b) * (1/n) * Σ||fi(X)||²
        
        This follows from similar principles as gradient variance
        """
        n = self.num_functions
        
        if batch_size >= n:
            return 0.0
        
        # Compute variance coefficient: (n-b)/(n*b)
        variance_coeff = (n - batch_size) / (n * batch_size)
        
        # Compute sum of squared function values: (1/n) * Σ||fi(X)||²
        func_values_squared = []
        for i in range(n):
            func_i = self._component_function(X, i)
            func_values_squared.append(func_i ** 2)
        
        # Average squared function value
        avg_func_value_sq = torch.mean(torch.stack(func_values_squared))
        
        # Total variance bound
        variance_bound = variance_coeff * avg_func_value_sq
        
        return variance_bound.item()
    
    def analyze_stochastic_variance(self, X, batch_sizes=[1, 5, 10, 25, 50, 100]):
        """
        Comprehensive analysis of stochastic gradient variance
        
        This function analyzes how the variance of stochastic gradients
        changes with different batch sizes and provides theoretical insights.
        """
        print("\n" + "="*60)
        print("STOCHASTIC GRADIENT VARIANCE ANALYSIS")
        print("="*60)
        
        n = self.num_functions
        print(f"Total number of functions: n = {n}")
        print(f"Matrix dimensions: {self.m} × {self.n}")
        print(f"Overall Lipschitz constant: {self.overall_lipschitz:.6f}")
        
        print("\nTheoretical Variance Analysis:")
        print("-" * 40)
        print("For batch size b, the variance bound is:")
        print("E(||g(X) - ∇f(X)||²) ≤ (n-b)/(n*b) * (1/n) * Σ||∇fi(X)||²")
        print("\nKey insights:")
        print("1. Variance decreases as batch size increases")
        print("2. Variance coefficient: (n-b)/(n*b)")
        print("3. When b=n, variance = 0 (full gradient)")
        print("4. When b=1, variance = (n-1)/n * (1/n) * Σ||∇fi(X)||²")
        
        print("\nBatch Size Analysis:")
        print("-" * 40)
        print(f"{'Batch Size':<12} {'Variance Coeff':<15} {'Grad Variance':<15} {'Obj Variance':<15}")
        print("-" * 70)
        
        for batch_size in batch_sizes:
            if batch_size > n:
                continue
                
            # Compute variance coefficient
            variance_coeff = (n - batch_size) / (n * batch_size)
            
            # Compute actual variance bounds
            grad_variance = self._compute_gradient_variance_bound(X, batch_size)
            obj_variance = self._compute_objective_variance_bound(X, batch_size)
            
            print(f"{batch_size:<12} {variance_coeff:<15.6f} {grad_variance:<15.6f} {obj_variance:<15.6f}")
        
        print("\nConvergence Implications:")
        print("-" * 40)
        print("1. Smaller batch sizes → Higher variance → Slower convergence")
        print("2. Larger batch sizes → Lower variance → Faster convergence")
        print("3. Optimal batch size balances computation cost vs. variance")
        print("4. Variance scales as O(1/b) for large batch sizes")
        
        # Compute theoretical convergence rate
        print(f"\nTheoretical Convergence Rate (for batch size 1):")
        print(f"Expected gradient norm: O(1/√T) where T is number of iterations")
        print(f"Variance bound: {self._compute_gradient_variance_bound(X, 1):.6f}")
        
        return {
            'batch_sizes': batch_sizes,
            'variance_coeffs': [(n - b) / (n * b) for b in batch_sizes if b <= n],
            'grad_variances': [self._compute_gradient_variance_bound(X, b) for b in batch_sizes if b <= n],
            'obj_variances': [self._compute_objective_variance_bound(X, b) for b in batch_sizes if b <= n]
        }

def run_optimization_comparison(m=64, n=128, num_iterations=1000, plot_interval=50, num_nodes=100):
    """
    Compare Muon, Neon, and Nesterov SGD optimizers on the composite function minimization
    
    Args:
        m: Matrix rows
        n: Matrix columns
        num_iterations: Total number of optimization iterations
        plot_interval: Interval for measuring and plotting data (default: 50)
    """
    print(f"Running optimization comparison with m={m}, n={n}")
    print(f"Number of iterations: {num_iterations}")
    print(f"Plotting data every {plot_interval} iterations")
    print("-" * 50)
    
    # Check GPU availability first
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cpu':
        print("WARNING: Running on CPU - Neon optimizer requires GPU/CUDA!")
        return None
    
    # Initialize the composite function (matrices already on GPU)
    composite_func = CompositeFunction(m, n, num_functions=num_nodes, norm_type='nuclear', sigma=0.1, device=device)
    
    # Initialize starting point
    # X should be m x n to match the function structure fi = 1/2 <X-Si, Mi (X-Si) Ni>
    X_init = torch.randn(m, n, dtype=torch.float32, device=device)
    X_init.requires_grad_(False)
    
    # Perform comprehensive variance analysis
    print("\nPerforming variance analysis...")
    variance_analysis = composite_func.analyze_stochastic_variance(X_init)
    
    # Test Muon optimizer
    print("Testing Muon optimizer...")
    X_muon = X_init.clone().to(device)
    muon_optimizer = Muon([X_muon], lr=0.01, nesterov=True, momentum=0.005)
    
    muon_losses = []
    muon_grad_norms = []
    muon_nuclear_norms = []
    muon_spectral_norms = []
    muon_times = []
    muon_grad_variances = []
    muon_loss_variances = []
    
    start_time = time.time()
    bs = 5
    for iteration in range(num_iterations):
        iter_start = time.time()
        
        # Compute stochastic gradient
        grad, grad_variance = composite_func.stochastic_gradient(X_muon, batch_size=bs)
        
        # Set gradient manually (since we're not using autograd)
        X_muon.grad = grad
        # Print progress every 100 iterations
        if iteration % 100 == 0:
            full_grad = composite_func.gradient(X_muon)
            # Compute current loss for progress display
            current_loss, _ = composite_func.stochastic_objective(X_muon, batch_size=num_nodes)
            grad_reshaped = full_grad.reshape(full_grad.size(0), -1)
            current_grad_norm = torch.linalg.matrix_norm(grad_reshaped, ord='nuc').item()
            print(f"Muon - Iter {iteration}: Loss = {current_loss.item():.6f}, Grad Norm = {current_grad_norm:.6f}")
    
        
        # Record metrics only every plot_interval iterations
        if iteration % plot_interval == 0:
            loss, loss_variance = composite_func.stochastic_objective(X_muon, batch_size=num_nodes)
            full_grad = composite_func.gradient(X_muon)
            grad_norm = full_grad.norm().item()
            
            # Compute nuclear and spectral norms of gradient
            grad_reshaped = full_grad.reshape(full_grad.size(0), -1)
            nuclear_norm = torch.linalg.matrix_norm(grad_reshaped, ord='nuc').item()
            spectral_norm = torch.linalg.matrix_norm(grad_reshaped, ord=2).item()
            
            muon_losses.append(loss.item())
            muon_grad_norms.append(grad_norm)
            muon_nuclear_norms.append(nuclear_norm)
            muon_spectral_norms.append(spectral_norm)
            muon_times.append(time.time() - iter_start)
            muon_grad_variances.append(grad_variance)
            muon_loss_variances.append(loss_variance)
        
        # Optimizer step
        muon_optimizer.step()
    
    # Ensure we record the final iteration if not already recorded
    if (num_iterations - 1) % plot_interval != 0:
        loss, loss_variance = composite_func.stochastic_objective(X_muon, batch_size=num_nodes)
        full_grad = composite_func.gradient(X_muon)
        grad_norm = full_grad.norm().item()
        
        # Compute nuclear and spectral norms of gradient
        grad_reshaped = full_grad.reshape(full_grad.size(0), -1)
        nuclear_norm = torch.linalg.matrix_norm(grad_reshaped, ord='nuc').item()
        spectral_norm = torch.linalg.matrix_norm(grad_reshaped, ord=2).item()
        
        muon_losses.append(loss.item())
        muon_grad_norms.append(grad_norm)
        muon_nuclear_norms.append(nuclear_norm)
        muon_spectral_norms.append(spectral_norm)
        muon_times.append(time.time() - iter_start)
        muon_grad_variances.append(grad_variance)
        muon_loss_variances.append(loss_variance)
        
        
    muon_total_time = time.time() - start_time
    print(f"Muon completed in {muon_total_time:.2f} seconds")
    
    # Test Neon optimizer
    print("\nTesting Neon optimizer...")
    X_neon = X_init.clone().to(device)
    neon_optimizer = Neon([X_neon], lr=0.01, neon_mode='fast', iter_num=50,
                            nesterov=True, momentum=0.005)
    
    neon_losses = []
    neon_grad_norms = []
    neon_nuclear_norms = []
    neon_spectral_norms = []
    neon_times = []
    neon_grad_variances = []
    neon_loss_variances = []
    
    start_time = time.time()
    
    for iteration in range(num_iterations):
        iter_start = time.time()
        
        # Compute stochastic gradient
        grad, grad_variance = composite_func.stochastic_gradient(X_neon, batch_size=bs)
        
        # Set gradient manually
        X_neon.grad = grad
        # Print progress every 100 iterations
        if iteration % 100 == 0:
            # Compute current loss for progress display
            current_loss = composite_func.objective(X_neon)
            full_grad = composite_func.gradient(X_neon)
            current_grad_norm = full_grad.norm().item()
            print(f"Neon - Iter {iteration}: Loss = {current_loss.item():.6f}, Grad Norm = {current_grad_norm:.6f}")
    

        
        # Record metrics only every plot_interval iterations
        if iteration % plot_interval == 0:
            loss = composite_func.objective(X_neon)
            full_grad = composite_func.gradient(X_neon)
            grad_norm = full_grad.norm().item()
            
            # Compute nuclear and spectral norms of gradient
            grad_reshaped = full_grad.reshape(full_grad.size(0), -1)
            nuclear_norm = torch.linalg.matrix_norm(grad_reshaped, ord='nuc').item()
            spectral_norm = torch.linalg.matrix_norm(grad_reshaped, ord=2).item()
            
            neon_losses.append(loss.item())
            neon_grad_norms.append(grad_norm)
            neon_nuclear_norms.append(nuclear_norm)
            neon_spectral_norms.append(spectral_norm)
            neon_times.append(time.time() - iter_start)
            neon_grad_variances.append(grad_variance)
            neon_loss_variances.append(0.0)  # No variance for full objective
                # Optimizer step
        neon_optimizer.step()
        
    # Ensure we record the final iteration if not already recorded
    if (num_iterations - 1) % plot_interval != 0:
        loss = composite_func.objective(X_neon)
        full_grad = composite_func.gradient(X_neon)
        grad_norm = full_grad.norm().item()
        
        # Compute nuclear and spectral norms of gradient
        grad_reshaped = full_grad.reshape(full_grad.size(0), -1)
        nuclear_norm = torch.linalg.matrix_norm(grad_reshaped, ord='nuc').item()
        spectral_norm = torch.linalg.matrix_norm(grad_reshaped, ord=2).item()
        
        neon_losses.append(loss.item())
        neon_grad_norms.append(grad_norm)
        neon_nuclear_norms.append(nuclear_norm)
        neon_spectral_norms.append(spectral_norm)
        neon_times.append(time.time() - iter_start)
        neon_grad_variances.append(grad_variance)
        neon_loss_variances.append(0.0)  # No variance for full objective
        
        
    neon_total_time = time.time() - start_time
    print(f"Neon completed in {neon_total_time:.2f} seconds")
    
    # Test Nesterov SGD optimizer
    print("\nTesting Nesterov SGD optimizer...")
    X_nesterov = X_init.clone().to(device)
    nesterov_optimizer = torch.optim.SGD([X_nesterov], lr=0.4, momentum=0.9, nesterov=True)
    
    nesterov_losses = []
    nesterov_grad_norms = []
    nesterov_nuclear_norms = []
    nesterov_spectral_norms = []
    nesterov_times = []
    nesterov_grad_variances = []
    nesterov_loss_variances = []
    
    start_time = time.time()
    
    for iteration in range(num_iterations):
        iter_start = time.time()
        
        # Compute stochastic gradient
        grad, grad_variance = composite_func.stochastic_gradient(X_nesterov, batch_size=bs)
        
        # Set gradient manually
        X_nesterov.grad = grad
        # Print progress every 100 iterations
        if iteration % 100 == 0:
            # Compute current loss for progress display
            current_loss = composite_func.objective(X_nesterov)
            full_grad = composite_func.gradient(X_nesterov)
            current_grad_norm = full_grad.norm().item()
            print(f"Nesterov - Iter {iteration}: Loss = {current_loss.item():.6f}, Grad Norm = {current_grad_norm:.6f}")
    

        
        # Record metrics only every plot_interval iterations
        if iteration % plot_interval == 0:
            loss = composite_func.objective(X_nesterov)
            full_grad = composite_func.gradient(X_nesterov)
            grad_norm = full_grad.norm().item()
            
            # Compute nuclear and spectral norms of gradient
            grad_reshaped = full_grad.reshape(full_grad.size(0), -1)
            nuclear_norm = torch.linalg.matrix_norm(grad_reshaped, ord='nuc').item()
            spectral_norm = torch.linalg.matrix_norm(grad_reshaped, ord=2).item()
            
            nesterov_losses.append(loss.item())
            nesterov_grad_norms.append(grad_norm)
            nesterov_nuclear_norms.append(nuclear_norm)
            nesterov_spectral_norms.append(spectral_norm)
            nesterov_times.append(time.time() - iter_start)
            nesterov_grad_variances.append(grad_variance)
            nesterov_loss_variances.append(loss_variance)
                # Optimizer step
        if num_iterations < 3000: # TODO
            nesterov_optimizer.step()
        
    # Ensure we record the final iteration if not already recorded
    if (num_iterations - 1) % plot_interval != 0:
        loss = composite_func.objective(X_nesterov)
        full_grad = composite_func.gradient(X_nesterov)
        grad_norm = full_grad.norm().item()
        
        # Compute nuclear and spectral norms of gradient
        grad_reshaped = full_grad.reshape(full_grad.size(0), -1)
        nuclear_norm = torch.linalg.matrix_norm(grad_reshaped, ord='nuc').item()
        spectral_norm = torch.linalg.matrix_norm(grad_reshaped, ord=2).item()
        
        nesterov_losses.append(loss.item())
        nesterov_grad_norms.append(grad_norm)
        nesterov_nuclear_norms.append(nuclear_norm)
        nesterov_spectral_norms.append(spectral_norm)
        nesterov_times.append(time.time() - iter_start)
        nesterov_grad_variances.append(grad_variance)
        nesterov_loss_variances.append(loss_variance)
        
        
    nesterov_total_time = time.time() - start_time
    print(f"Nesterov SGD completed in {nesterov_total_time:.2f} seconds")
    
    # Results summary
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(f"Final Muon loss: {muon_losses[-1]:.6f}")
    print(f"Final Neon loss: {neon_losses[-1]:.6f}")
    print(f"Final Nesterov SGD loss: {nesterov_losses[-1]:.6f}")
    print(f"Muon total time: {muon_total_time:.2f}s")
    print(f"Neon total time: {neon_total_time:.2f}s")
    print(f"Nesterov SGD total time: {nesterov_total_time:.2f}s")
    print(f"Muon avg time per iteration: {np.mean(muon_times):.6f}s")
    print(f"Neon avg time per iteration: {np.mean(neon_times):.6f}s")
    print(f"Nesterov SGD avg time per iteration: {np.mean(nesterov_times):.6f}s")
    
    # Plot results
    plot_comparison_results(muon_losses, neon_losses, nesterov_losses,
                           muon_grad_norms, neon_grad_norms, nesterov_grad_norms,
                           muon_nuclear_norms, neon_nuclear_norms, nesterov_nuclear_norms,
                           muon_spectral_norms, neon_spectral_norms, nesterov_spectral_norms,
                           muon_times, neon_times, nesterov_times,
                           muon_grad_variances, neon_grad_variances, nesterov_grad_variances,
                           muon_loss_variances, neon_loss_variances, nesterov_loss_variances,
                           num_iterations, plot_interval)
    
    return {
        'muon_losses': muon_losses,
        'neon_losses': neon_losses,
        'nesterov_losses': nesterov_losses,
        'muon_grad_norms': muon_grad_norms,
        'neon_grad_norms': neon_grad_norms,
        'nesterov_grad_norms': nesterov_grad_norms,
        'muon_nuclear_norms': muon_nuclear_norms,
        'neon_nuclear_norms': neon_nuclear_norms,
        'nesterov_nuclear_norms': nesterov_nuclear_norms,
        'muon_spectral_norms': muon_spectral_norms,
        'neon_spectral_norms': neon_spectral_norms,
        'nesterov_spectral_norms': nesterov_spectral_norms,
        'muon_times': muon_times,
        'neon_times': neon_times,
        'nesterov_times': nesterov_times,
        'muon_grad_variances': muon_grad_variances,
        'neon_grad_variances': neon_grad_variances,
        'nesterov_grad_variances': nesterov_grad_variances,
        'muon_loss_variances': muon_loss_variances,
        'neon_loss_variances': neon_loss_variances,
        'nesterov_loss_variances': nesterov_loss_variances
    }

def plot_comparison_results(muon_losses, neon_losses, nesterov_losses,
                           muon_grad_norms, neon_grad_norms, nesterov_grad_norms,
                           muon_nuclear_norms, neon_nuclear_norms, nesterov_nuclear_norms,
                           muon_spectral_norms, neon_spectral_norms, nesterov_spectral_norms,
                           muon_times, neon_times, nesterov_times,
                           muon_grad_variances, neon_grad_variances, nesterov_grad_variances,
                           muon_loss_variances, neon_loss_variances, nesterov_loss_variances,
                           num_iterations, plot_interval):
    """Plot the comparison results for three optimizers: Muon, Neon, and Nesterov SGD"""
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    
    # Create x-axis for plotting (only the iterations where we recorded data)
    plot_iterations = list(range(0, num_iterations, plot_interval))
    if num_iterations - 1 not in plot_iterations:
        plot_iterations.append(num_iterations - 1)  # Include the final iteration
    
    # Loss comparison
    axes[0, 0].plot(plot_iterations, muon_losses, 'b-', label='Muon', linewidth=2)
    axes[0, 0].plot(plot_iterations, neon_losses, 'r-', label='Neon', linewidth=2)
    axes[0, 0].plot(plot_iterations, nesterov_losses, 'g-', label='Nesterov SGD', linewidth=2)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss vs Iteration')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # Gradient norm comparison
    axes[0, 1].plot(plot_iterations, muon_grad_norms, 'b-', label='Muon', linewidth=2)
    axes[0, 1].plot(plot_iterations, neon_grad_norms, 'r-', label='Neon', linewidth=2)
    axes[0, 1].plot(plot_iterations, nesterov_grad_norms, 'g-', label='Nesterov SGD', linewidth=2)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Gradient Norm')
    axes[0, 1].set_title('Gradient Norm vs Iteration')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # Time per iteration comparison
    axes[1, 0].plot(plot_iterations, muon_times, 'b-', label='Muon', linewidth=2)
    axes[1, 0].plot(plot_iterations, neon_times, 'r-', label='Neon', linewidth=2)
    axes[1, 0].plot(plot_iterations, nesterov_times, 'g-', label='Nesterov SGD', linewidth=2)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Time per Iteration (s)')
    axes[1, 0].set_title('Time per Iteration vs Iteration')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Cumulative time comparison
    muon_cumulative = np.cumsum(muon_times)
    neon_cumulative = np.cumsum(neon_times)
    nesterov_cumulative = np.cumsum(nesterov_times)
    axes[1, 1].plot(plot_iterations, muon_cumulative, 'b-', label='Muon', linewidth=2)
    axes[1, 1].plot(plot_iterations, neon_cumulative, 'r-', label='Neon', linewidth=2)
    axes[1, 1].plot(plot_iterations, nesterov_cumulative, 'g-', label='Nesterov SGD', linewidth=2)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Cumulative Time (s)')
    axes[1, 1].set_title('Cumulative Time vs Iteration')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Nuclear norm comparison
    axes[2, 0].plot(plot_iterations, muon_nuclear_norms, 'b-', label='Muon', linewidth=2)
    axes[2, 0].plot(plot_iterations, neon_nuclear_norms, 'r-', label='Neon', linewidth=2)
    axes[2, 0].plot(plot_iterations, nesterov_nuclear_norms, 'g-', label='Nesterov SGD', linewidth=2)
    axes[2, 0].set_xlabel('Iteration')
    axes[2, 0].set_ylabel('Nuclear Norm of Gradient')
    axes[2, 0].set_title('Nuclear Norm of Gradient vs Iteration')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_yscale('log')
    
    # Spectral norm comparison
    axes[2, 1].plot(plot_iterations, muon_spectral_norms, 'b-', label='Muon', linewidth=2)
    axes[2, 1].plot(plot_iterations, neon_spectral_norms, 'r-', label='Neon', linewidth=2)
    axes[2, 1].plot(plot_iterations, nesterov_spectral_norms, 'g-', label='Nesterov SGD', linewidth=2)
    axes[2, 1].set_xlabel('Iteration')
    axes[2, 1].set_ylabel('Spectral Norm of Gradient')
    axes[2, 1].set_title('Spectral Norm of Gradient vs Iteration')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].set_yscale('log')
    
    # Gradient variance comparison
    axes[3, 0].plot(plot_iterations, muon_grad_variances, 'b-', label='Muon', linewidth=2)
    axes[3, 0].plot(plot_iterations, neon_grad_variances, 'r-', label='Neon', linewidth=2)
    axes[3, 0].plot(plot_iterations, nesterov_grad_variances, 'g-', label='Nesterov SGD', linewidth=2)
    axes[3, 0].set_xlabel('Iteration')
    axes[3, 0].set_ylabel('Gradient Variance Bound')
    axes[3, 0].set_title('Gradient Variance Bound vs Iteration')
    axes[3, 0].legend()
    axes[3, 0].grid(True, alpha=0.3)
    axes[3, 0].set_yscale('log')
    
    # Loss variance comparison
    axes[3, 1].plot(plot_iterations, muon_loss_variances, 'b-', label='Muon', linewidth=2)
    axes[3, 1].plot(plot_iterations, neon_loss_variances, 'r-', label='Neon', linewidth=2)
    axes[3, 1].plot(plot_iterations, nesterov_loss_variances, 'g-', label='Nesterov SGD', linewidth=2)
    axes[3, 1].set_xlabel('Iteration')
    axes[3, 1].set_ylabel('Loss Variance Bound')
    axes[3, 1].set_title('Loss Variance Bound vs Iteration')
    axes[3, 1].legend()
    axes[3, 1].grid(True, alpha=0.3)
    axes[3, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('optimizer_comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_stochastic_gradients():
    """Test the stochastic gradient implementation"""
    print("Testing stochastic gradient implementation...")
    
    m, n = 32, 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    composite_func = CompositeFunction(m, n, num_functions=50, norm_type='nuclear', sigma=0.1, device=device)
    # X should be m x n to match the function structure fi = 1/2 <X-Si, Mi (X-Si) Ni>
    X = torch.randn(m, n, dtype=torch.float32, device=device) * 0.1
    
    # Test different batch sizes
    batch_sizes = [1, 5, 10, 25, 50]
    
    print(f"{'Batch Size':<12} {'Variance Bound':<15} {'Grad Norm':<15}")
    print("-" * 50)
    
    for batch_size in batch_sizes:
        grad, var_bound = composite_func.stochastic_gradient(X, batch_size)
        grad_norm = grad.norm().item()
        print(f"{batch_size:<12} {var_bound:<15.6f} {grad_norm:<15.6f}")
    
    print("\nStochastic gradient test completed!")

if __name__ == "__main__":
    # Test stochastic gradients
    # test_stochastic_gradients()
    
    print("\n" + "=" * 60)
    print("RUNNING MAIN OPTIMIZATION COMPARISON")
    print("=" * 60)
    
    # Run the main comparison
    results = run_optimization_comparison(
        m=1000,           # Matrix rows
        n=100,          # Matrix columns
        num_nodes=20,
        num_iterations=50000,  # Number of optimization iterations
        plot_interval=100,     # Plot data every 50 iterations
    )
    
    print("\nOptimization comparison completed!")
    print("Results saved to 'optimizer_comparison_results.png'")
