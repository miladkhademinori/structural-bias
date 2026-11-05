"""pipeline.backend -- the minibatch loops, in numpy (reference) or torch (GPU acceleration).

Every stochastic loop in the pipeline draws its minibatch indices, permutations and
initial weights from a numpy Generator seeded by ``registry.derive_seed`` BEFORE the loop
starts, and both backends consume exactly the same index streams.  The two backends
therefore run the same algorithm on the same batches and differ only in floating-point
rounding (numpy float64 vs torch float32 on the device).  ``resolve_device`` picks the
backend::

    REPLICATION_DEVICE=auto    cuda if torch sees a GPU, else numpy      (default)
    REPLICATION_DEVICE=numpy   the reference implementation
    REPLICATION_DEVICE=cpu     torch on the CPU (useful to test the torch path without a GPU)
    REPLICATION_DEVICE=cuda    torch on the GPU

Each result dict records ``backend`` so the provenance of every number is explicit.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import numpy as np

try:  # torch is optional; the numpy path needs nothing but numpy
    import torch
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    HAVE_TORCH = False


def resolve_device(device: Optional[str] = None) -> str:
    d = device or os.environ.get("REPLICATION_DEVICE", "auto")
    if d == "auto":
        return "cuda" if (HAVE_TORCH and torch.cuda.is_available()) else "numpy"
    if d == "numpy":
        return d
    if d in ("cpu", "cuda"):
        if not HAVE_TORCH:
            raise RuntimeError(f"REPLICATION_DEVICE={d} needs torch, which is not installed")
        if d == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("REPLICATION_DEVICE=cuda but torch.cuda.is_available() is False")
        return d
    raise ValueError(f"unknown device {d!r}; use auto|numpy|cpu|cuda")


def describe(device: Optional[str] = None) -> dict:
    d = resolve_device(device)
    out = dict(backend=d, dtype="float64" if d == "numpy" else "float32")
    if d != "numpy":
        out["torch"] = torch.__version__
        if d == "cuda":
            out["gpu"] = torch.cuda.get_device_name(0)
    return out


# ---------------------------------------------------------------------------
# Softmax-CE minibatch SGD over a class subset ("seen" classes), optional clipping
# ---------------------------------------------------------------------------

def sgd_subset(W: np.ndarray, b: np.ndarray, X: np.ndarray, y_pos: np.ndarray, seen: np.ndarray,
               idx: np.ndarray, *, eta: float, tau: float, lam: float, clip: bool,
               record_norms: bool = False, device: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """K steps of (clipped) minibatch SGD on the mean CE over the classes ``seen`` (sorted class ids),
    with ridge lam/2 (||W_seen||^2 + ||b_seen||^2).  ``y_pos`` holds label positions within ``seen``
    and ``idx`` is the [K, batch] matrix of sample indices into X.  Rows of W outside ``seen`` are
    untouched.  Returns the updated (W, b) as numpy float64 arrays and, if requested, the K gradient
    norms (before clipping)."""
    d = resolve_device(device)
    if d == "numpy":
        return _sgd_subset_numpy(W, b, X, y_pos, seen, idx, eta, tau, lam, clip, record_norms)
    return _sgd_subset_torch(W, b, X, y_pos, seen, idx, eta, tau, lam, clip, record_norms, d)


def _sgd_subset_numpy(W, b, X, y_pos, seen, idx, eta, tau, lam, clip, record_norms):
    W = np.array(W, dtype=np.float64); b = np.array(b, dtype=np.float64)
    Ws = W[seen].copy(); bs = b[seen].copy()
    X = np.asarray(X, np.float64); K, n = idx.shape
    ar = np.arange(n); norms = np.empty(K) if record_norms else None
    for k in range(K):
        bi = idx[k]; Xb = X[bi]; yb = y_pos[bi]
        S = Xb @ Ws.T + bs
        S -= S.max(axis=1, keepdims=True)
        P = np.exp(S); P /= P.sum(axis=1, keepdims=True)
        P[ar, yb] -= 1.0; P /= n
        gW = P.T @ Xb; gW += lam * Ws
        gb = P.sum(axis=0); gb += lam * bs
        gn = np.sqrt(np.sum(gW * gW) + np.sum(gb * gb))
        if record_norms:
            norms[k] = gn
        if clip and gn > tau:
            sc = tau / gn; gW *= sc; gb *= sc
        Ws -= eta * gW; bs -= eta * gb
    W[seen] = Ws; b[seen] = bs
    return W, b, norms


def _sgd_subset_torch(W, b, X, y_pos, seen, idx, eta, tau, lam, clip, record_norms, dev):
    W = np.array(W, dtype=np.float64); b = np.array(b, dtype=np.float64)
    Ws = torch.as_tensor(W[seen], dtype=torch.float32, device=dev)
    bs = torch.as_tensor(b[seen], dtype=torch.float32, device=dev)
    Xt = torch.as_tensor(np.asarray(X, np.float32), device=dev)
    yp = torch.as_tensor(np.asarray(y_pos, np.int64), device=dev)
    I = torch.as_tensor(np.asarray(idx, np.int64), device=dev)
    K, n = idx.shape
    ar = torch.arange(n, device=dev)
    norms = torch.empty(K, device=dev) if record_norms else None
    tau_t = float(tau)
    with torch.no_grad():
        for k in range(K):
            bi = I[k]; Xb = Xt[bi]; yb = yp[bi]
            S = Xb @ Ws.T + bs
            P = torch.softmax(S, dim=1)
            P[ar, yb] -= 1.0; P /= n
            gW = P.T @ Xb; gW += lam * Ws
            gb = P.sum(0); gb += lam * bs
            gn = torch.sqrt((gW * gW).sum() + (gb * gb).sum())
            if record_norms:
                norms[k] = gn
            if clip:
                # scale = min(1, tau/||g||); written so that ||g|| = 0 or tau = inf cannot produce NaN
                sc = torch.where(gn > tau_t, tau_t / torch.clamp(gn, min=1e-300), torch.ones_like(gn))
                gW *= sc; gb *= sc
            Ws -= eta * gW; bs -= eta * gb
    W[seen] = Ws.double().cpu().numpy(); b[seen] = bs.double().cpu().numpy()
    return W, b, (norms.double().cpu().numpy() if record_norms else None)


# ---------------------------------------------------------------------------
# AdamW on the full C-way CE (the papers' head-training protocol)
# ---------------------------------------------------------------------------

def adamw_ce(theta: np.ndarray, X: np.ndarray, y: np.ndarray, C: int, lam: float, *, lr: float, eta_min: float,
             perms: Sequence[np.ndarray], batch: int, decay_bias: bool, betas=(0.9, 0.999), eps=1e-8,
             device: Optional[str] = None) -> Tuple[np.ndarray, int]:
    """AdamW (decoupled decay lam, cosine lr from lr to eta_min over all steps) on mean CE over C classes.
    ``perms`` is one permutation of the sample indices per epoch; batches are consecutive slices of it.
    Returns (theta, steps).  theta = [W.ravel(), b] with W [C, d]."""
    d = resolve_device(device)
    if d == "numpy":
        return _adamw_numpy(theta, X, y, C, lam, lr, eta_min, perms, batch, decay_bias, betas, eps)
    return _adamw_torch(theta, X, y, C, lam, lr, eta_min, perms, batch, decay_bias, betas, eps, d)


def _adamw_numpy(theta, X, y, C, lam, lr, eta_min, perms, batch, decay_bias, betas, eps):
    X = np.asarray(X, np.float64); n, dim = X.shape
    W = np.array(theta[:C * dim], np.float64).reshape(C, dim); b = np.array(theta[C * dim:], np.float64)
    mW = np.zeros_like(W); vW = np.zeros_like(W); mb = np.zeros_like(b); vb = np.zeros_like(b)
    steps_per_epoch = int(np.ceil(n / batch)); total = len(perms) * steps_per_epoch; t = 0
    b1, b2 = betas
    for perm in perms:
        for s in range(steps_per_epoch):
            bi = perm[s * batch:(s + 1) * batch]; Xb = X[bi]; yb = y[bi]; m = len(bi)
            S = Xb @ W.T + b; S -= S.max(axis=1, keepdims=True)
            P = np.exp(S); P /= P.sum(axis=1, keepdims=True)
            P[np.arange(m), yb] -= 1.0; P /= m
            gW = P.T @ Xb; gb = P.sum(axis=0)
            lr_t = eta_min + 0.5 * (lr - eta_min) * (1 + np.cos(np.pi * t / max(total, 1)))
            t += 1
            mW *= b1; mW += (1 - b1) * gW; vW *= b2; vW += (1 - b2) * gW * gW
            mb *= b1; mb += (1 - b1) * gb; vb *= b2; vb += (1 - b2) * gb * gb
            c1 = 1 - b1 ** t; c2 = 1 - b2 ** t
            W -= lr_t * ((mW / c1) / (np.sqrt(vW / c2) + eps) + lam * W)
            b -= lr_t * ((mb / c1) / (np.sqrt(vb / c2) + eps) + (lam * b if decay_bias else 0.0))
    return np.concatenate([W.ravel(), b]), t


def _adamw_torch(theta, X, y, C, lam, lr, eta_min, perms, batch, decay_bias, betas, eps, dev):
    n, dim = X.shape
    Xt = torch.as_tensor(np.asarray(X, np.float32), device=dev); yt = torch.as_tensor(np.asarray(y, np.int64), device=dev)
    W = torch.as_tensor(np.asarray(theta[:C * dim], np.float32).reshape(C, dim), device=dev).clone()
    b = torch.as_tensor(np.asarray(theta[C * dim:], np.float32), device=dev).clone()
    mW = torch.zeros_like(W); vW = torch.zeros_like(W); mb = torch.zeros_like(b); vb = torch.zeros_like(b)
    steps_per_epoch = int(np.ceil(n / batch)); total = len(perms) * steps_per_epoch; t = 0
    b1, b2 = betas
    with torch.no_grad():
        for perm in perms:
            P_all = torch.as_tensor(np.asarray(perm, np.int64), device=dev)
            for s in range(steps_per_epoch):
                bi = P_all[s * batch:(s + 1) * batch]; Xb = Xt[bi]; yb = yt[bi]; m = bi.numel()
                P = torch.softmax(Xb @ W.T + b, dim=1)
                P[torch.arange(m, device=dev), yb] -= 1.0; P /= m
                gW = P.T @ Xb; gb = P.sum(0)
                lr_t = eta_min + 0.5 * (lr - eta_min) * (1 + np.cos(np.pi * t / max(total, 1)))
                t += 1
                mW.mul_(b1).add_(gW, alpha=1 - b1); vW.mul_(b2).addcmul_(gW, gW, value=1 - b2)
                mb.mul_(b1).add_(gb, alpha=1 - b1); vb.mul_(b2).addcmul_(gb, gb, value=1 - b2)
                c1 = 1 - b1 ** t; c2 = 1 - b2 ** t
                W -= lr_t * ((mW / c1) / (torch.sqrt(vW / c2) + eps) + lam * W)
                b -= lr_t * ((mb / c1) / (torch.sqrt(vb / c2) + eps) + (lam * b if decay_bias else 0.0))
    return np.concatenate([W.double().cpu().numpy().ravel(), b.double().cpu().numpy()]), t


# ---------------------------------------------------------------------------
# Full-batch L-BFGS on the regularized CE (the reference optimizer; exact minimizers)
# ---------------------------------------------------------------------------

def lbfgs_ce(theta0: np.ndarray, X: np.ndarray, y: np.ndarray, C: int, lam: float, *, decay_bias: bool = True,
             maxiter: int = 2000, gtol: float = 1e-9, device: Optional[str] = None) -> Tuple[np.ndarray, dict]:
    """Minimize mean CE over C classes + (lam/2)(||W||^2 + ||b||^2 if decay_bias) by L-BFGS from theta0.
    numpy backend: scipy L-BFGS-B; torch backend: torch.optim.LBFGS with strong-Wolfe line search, in float64.
    Returns (theta, info) with info = dict(nit, max_grad, fun, success)."""
    d = resolve_device(device)
    if d == "numpy":
        from scipy import optimize
        from pipeline.heads import _ce_value_grad
        X = np.asarray(X, np.float64)   # one copy here, not one per objective evaluation
        res = optimize.minimize(_ce_value_grad, np.asarray(theta0, np.float64), args=(X, y, C, lam, decay_bias), jac=True,
                                method="L-BFGS-B", options=dict(maxiter=maxiter, ftol=1e-15, gtol=gtol))
        return res.x, dict(nit=int(res.nit), max_grad=float(np.abs(res.jac).max()), fun=float(res.fun), success=bool(res.success), backend="numpy")
    dim = X.shape[1]
    Xt = torch.as_tensor(np.asarray(X, np.float64), device=d); yt = torch.as_tensor(np.asarray(y, np.int64), device=d)
    W = torch.as_tensor(np.asarray(theta0[:C * dim], np.float64).reshape(C, dim), device=d).clone().requires_grad_(True)
    b = torch.as_tensor(np.asarray(theta0[C * dim:], np.float64), device=d).clone().requires_grad_(True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=maxiter, tolerance_grad=gtol, tolerance_change=1e-15,
                            history_size=20, line_search_fn="strong_wolfe")

    def objective():
        loss = torch.nn.functional.cross_entropy(Xt @ W.T + b, yt)
        loss = loss + 0.5 * lam * ((W * W).sum() + ((b * b).sum() if decay_bias else 0.0))
        return loss

    def closure():
        opt.zero_grad(); loss = objective(); loss.backward(); return loss

    opt.step(closure)
    with torch.no_grad():
        opt.zero_grad()
    loss = objective(); loss.backward()
    with torch.no_grad():
        max_grad = float(max(W.grad.abs().max(), b.grad.abs().max()))
        theta = np.concatenate([W.detach().cpu().numpy().ravel(), b.detach().cpu().numpy()])
    nit = opt.state[W].get("n_iter", -1)
    return theta, dict(nit=int(nit), max_grad=max_grad, fun=float(loss.detach()), success=bool(max_grad <= 10 * gtol), backend=d)


# ---------------------------------------------------------------------------
# Random-projection ridge head (RanPAC-style): Gram accumulation and solve
# ---------------------------------------------------------------------------

class GramRidge:
    """Accumulates G = sum relu(XP)^T relu(XP) and Q = sum relu(XP)^T onehot(y) over incremental data and solves
    the ridge system once at the end (RanPAC's class-prototype-free variant)."""

    def __init__(self, P: np.ndarray, N: int, device: Optional[str] = None):
        self.dev = resolve_device(device); self.N = N; self.M = P.shape[1]
        if self.dev == "numpy":
            self.P = np.asarray(P, np.float64); self.G = np.zeros((self.M, self.M)); self.Q = np.zeros((self.M, N))
        else:
            self.P = torch.as_tensor(np.asarray(P, np.float32), device=self.dev)
            self.G = torch.zeros((self.M, self.M), device=self.dev); self.Q = torch.zeros((self.M, N), device=self.dev)

    def _phi(self, X):
        if self.dev == "numpy":
            return np.maximum(np.asarray(X, np.float64) @ self.P, 0.0)
        return torch.relu(torch.as_tensor(np.asarray(X, np.float32), device=self.dev) @ self.P)

    def update(self, X, y, chunk=4096):
        for s in range(0, len(y), chunk):
            H = self._phi(X[s:s + chunk]); yy = y[s:s + chunk]
            if self.dev == "numpy":
                Y = np.zeros((len(yy), self.N)); Y[np.arange(len(yy)), yy] = 1.0
                self.G += H.T @ H; self.Q += H.T @ Y
            else:
                Y = torch.zeros((len(yy), self.N), device=self.dev); Y[torch.arange(len(yy), device=self.dev), torch.as_tensor(np.asarray(yy, np.int64), device=self.dev)] = 1.0
                self.G += H.T @ H; self.Q += H.T @ Y
        return self

    def solve(self, ridge: Optional[float] = None):
        if self.dev == "numpy":
            lam = ridge if ridge is not None else 1e-3 * np.trace(self.G) / self.M
            self.W = np.linalg.solve(self.G + lam * np.eye(self.M), self.Q)
        else:
            G = self.G.double(); Q = self.Q.double()
            lam = ridge if ridge is not None else 1e-3 * float(torch.trace(G)) / self.M
            self.W = torch.linalg.solve(G + lam * torch.eye(self.M, device=self.dev, dtype=torch.float64), Q).float()
        self.ridge = float(lam)
        return self

    def predict(self, X):
        if self.dev == "numpy":
            return np.argmax(self._phi(X) @ self.W, axis=1)
        return torch.argmax(self._phi(X) @ self.W, dim=1).cpu().numpy()
