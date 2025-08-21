from typing import Optional, Tuple

import torch

from benchmark_runner import MatrixProblem


def _rand_orthogonal(n: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    Q, _ = torch.linalg.qr(torch.randn(n, n, device=device, dtype=dtype))
    return Q


def _random_psd_matrix(
    n: int,
    eig_range: Tuple[float, float] = (0.1, 1.0),
    rank: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    device = device or torch.device("cpu")
    r = n if rank is None else max(1, min(rank, n))
    Q = _rand_orthogonal(n, device=device, dtype=dtype)
    eig_min, eig_max = eig_range
    vals = eig_min + (eig_max - eig_min) * torch.rand(r, device=device, dtype=dtype)
    if r < n:
        diag = torch.cat([vals, torch.zeros(n - r, device=device, dtype=dtype)])
    else:
        diag = vals
    D = torch.diag(diag)
    return Q @ D @ Q.T


class RandomQuadraticPSDProblem(MatrixProblem):
    """
    f(X) = 0.5 * <X - S, M (X - S) N>, where M ∈ R^{m×m}, N ∈ R^{n×n} are symmetric PSD.
    Gradient: ∇f(X) = M (X - S) N.
    """

    def __init__(
        self,
        m: int,
        n: int,
        eig_range_M: Tuple[float, float] = (0.1, 1.0),
        eig_range_N: Tuple[float, float] = (0.1, 1.0),
        rank_M: Optional[int] = None,
        rank_N: Optional[int] = None,
        shift_scale: float = 1.0,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.m = m
        self.n = n
        self.device = device or torch.device("cpu")
        if seed is not None:
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
        self.M = _random_psd_matrix(m, eig_range=eig_range_M, rank=rank_M, device=self.device)
        self.N = _random_psd_matrix(n, eig_range=eig_range_N, rank=rank_N, device=self.device)
        self.S = shift_scale * torch.randn(m, n, device=self.device)

    def objective(self, X: torch.Tensor) -> torch.Tensor:
        # Ensure parameters are on the same device as X
        target = X.device
        if self.M.device != target:
            self.M = self.M.to(target)
        if self.N.device != target:
            self.N = self.N.to(target)
        if self.S.device != target:
            self.S = self.S.to(target)
        Xs = X - self.S
        return 0.5 * torch.sum(Xs * (self.M @ Xs @ self.N))

    def gradient(self, X: torch.Tensor) -> torch.Tensor:
        # Ensure parameters are on the same device as X
        target = X.device
        if self.M.device != target:
            self.M = self.M.to(target)
        if self.N.device != target:
            self.N = self.N.to(target)
        if self.S.device != target:
            self.S = self.S.to(target)
        Xs = X - self.S
        return self.M @ Xs @ self.N

    def lipschitz_constant(self) -> float:
        """
        Lipschitz constant of the gradient w.r.t. Frobenius norm.
        For ∇f(X) = M (X - S) N, L = ||M||_2 · ||N||_2.
        """
        L_M = torch.linalg.matrix_norm(self.M, ord=2)
        L_N = torch.linalg.matrix_norm(self.N, ord=2)
        return (L_M * L_N).item()


class AXBLeastSquaresProblem(MatrixProblem):
    """
    f(X) = 0.5 * ||A X B - C||_F^2, with A ∈ R^{m×m}, B ∈ R^{n×n}, C ∈ R^{m×n}.
    Gradient: ∇f(X) = A^T (A X B - C) B^T.
    """

    def __init__(
        self,
        m: int,
        n: int,
        a_eig_range: Tuple[float, float] = (0.5, 1.5),
        b_eig_range: Tuple[float, float] = (0.5, 1.5),
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.m = m
        self.n = n
        self.device = device or torch.device("cpu")
        if seed is not None:
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
        # Build A and B as well-conditioned symmetric PD to control curvature
        self.A = _random_psd_matrix(m, eig_range=a_eig_range, device=self.device)
        self.B = _random_psd_matrix(n, eig_range=b_eig_range, device=self.device)
        self.C = torch.randn(m, n, device=self.device)

    def objective(self, X: torch.Tensor) -> torch.Tensor:
        target = X.device
        if self.A.device != target:
            self.A = self.A.to(target)
        if self.B.device != target:
            self.B = self.B.to(target)
        if self.C.device != target:
            self.C = self.C.to(target)
        R = self.A @ X @ self.B - self.C
        return 0.5 * torch.sum(R * R)

    def gradient(self, X: torch.Tensor) -> torch.Tensor:
        target = X.device
        if self.A.device != target:
            self.A = self.A.to(target)
        if self.B.device != target:
            self.B = self.B.to(target)
        if self.C.device != target:
            self.C = self.C.to(target)
        R = self.A @ X @ self.B - self.C
        return self.A.T @ R @ self.B.T

    def lipschitz_constant(self) -> float:
        """
        Lipschitz constant of the gradient w.r.t. Frobenius norm.
        For ∇f(X) = A^T A X B B^T - A^T C B^T, L = ||A||_2^2 · ||B||_2^2.
        """
        L_A = torch.linalg.matrix_norm(self.A, ord=2)
        L_B = torch.linalg.matrix_norm(self.B, ord=2)
        return (L_A * L_A * L_B * L_B).item()


class RidgeShiftedQuadraticProblem(MatrixProblem):
    """
    f(X) = 0.5 * ||X - S||_F^2 + (lambda/2) * ||X||_F^2
    Gradient: ∇f(X) = (1 + lambda) X - S
    """

    def __init__(
        self,
        m: int,
        n: int,
        lam: float = 0.1,
        shift_scale: float = 1.0,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.m = m
        self.n = n
        self.lam = lam
        self.device = device or torch.device("cpu")
        if seed is not None:
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
        self.S = shift_scale * torch.randn(m, n, device=self.device)

    def objective(self, X: torch.Tensor) -> torch.Tensor:
        if self.S.device != X.device:
            self.S = self.S.to(X.device)
        R = X - self.S
        return 0.5 * torch.sum(R * R) + 0.5 * self.lam * torch.sum(X * X)

    def gradient(self, X: torch.Tensor) -> torch.Tensor:
        if self.S.device != X.device:
            self.S = self.S.to(X.device)
        return (1.0 + self.lam) * X - self.S

    def lipschitz_constant(self) -> float:
        """
        Lipschitz constant of the gradient w.r.t. Frobenius norm.
        For ∇f(X) = (1 + λ) X - S, L = 1 + λ.
        """
        return 1.0 + float(self.lam)


