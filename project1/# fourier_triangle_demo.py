# fourier_triangle_demo.py
# Goal: Approximate f(t) = 1 - |t| on [-pi, pi] with its Fourier cosine series.
#       Closed-form coefficients (no numerical integration).
#       f is even → only cos terms; only odd n contribute; a_n ~ 1/n^2.
# Tools: numpy, matplotlib
# Time: ~10 minutes

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1) Define the target function and grid
# -----------------------------
npts = 2001
t = np.linspace(-np.pi, np.pi, npts, endpoint=False)

def f_true(tt):
    return 1.0 - np.abs(tt)

f = f_true(t)

# -----------------------------
# 2) Closed-form Fourier coefficients for f(t)=1-|t|
#    Even function ⇒ cosine series: f(t) ~ a0/2 + sum_{n=1}^\infty a_n cos(n t)
#    a0 = (1/π) ∫_{-π}^{π} f(t) dt = 2 - π
#    a_n = (1/π) ∫_{-π}^{π} f(t) cos(n t) dt = (2/(π n^2)) [1 - (-1)^n]
#         ⇒ a_n = 0 for even n, and a_n = 4/(π n^2) for odd n.
# -----------------------------
a0 = 2.0 - np.pi  # scalar

def a_n(n):
    # vectorized: return coefficients for array n
    n = np.asarray(n, dtype=int)
    coeff = np.zeros_like(n, dtype=float)
    odd_mask = (n % 2 == 1)
    coeff[odd_mask] = 4.0 / (np.pi * (n[odd_mask]**2))
    return coeff

# -----------------------------
# 3) Partial sums using only odd cosine terms
#    S_N(t) = a0/2 + sum_{k=0}^{N-1} a_{n_k} cos(n_k t), with n_k = 1,3,5,...,2N-1.
# -----------------------------
def partial_sum(tt, N):
    n_odd = 2*np.arange(N) + 1             # [1,3,5,...,2N-1]
    an = a_n(n_odd)                         # shape (N,)
    # Broadcast cos(n t): get an (N, len(tt)) matrix via outer product
    C = np.cos(np.outer(n_odd, tt))         # cos(n_k * t_j)
    return a0/2.0 + (an[:, None] * C).sum(axis=0)

# -----------------------------
# 4) Try a few truncations and compare visually
# -----------------------------
Ns = [1, 3, 7, 25]  # students can tweak live
approxs = [partial_sum(t, N) for N in Ns]

plt.figure(figsize=(7,4))
plt.plot(t, f, 'k-', lw=1.8, label="true f(t)=1-|t|")
for N, y in zip(Ns, approxs):
    plt.plot(t, y, lw=1, label=f"N={N} odd cos terms")
plt.xlim(-np.pi, np.pi)
plt.ylim(1 - np.pi - 0.2, 1.2)  # show the negative region near ±π
plt.xlabel("t")
plt.ylabel("value")
plt.title("Fourier cosine series of f(t)=1-|t|  (even; only odd n)")
plt.legend(ncol=2, fontsize=9)
plt.tight_layout()
plt.savefig("fig_fourier_triangle.png", dpi=150)
print("Saved fig_fourier_triangle.png")

# -----------------------------
# 5) Simple error metric away from the corner
#    (Corner at t=0 slows convergence; ignore a small window near 0)
# -----------------------------
mask = np.abs(t) > 0.15
def rmse(a, b): return np.sqrt(np.mean((a - b)**2))

errs = [rmse(y[mask], f[mask]) for y in approxs]
for N, e in zip(Ns, errs):
    print(f"N={N:<3d}  RMSE (away from corner) = {e:.5f}")
