# newton_cooling_demo.py
# Goal: Compare analytic vs. forward-Euler solutions of T' = -k (T - Tref)
# Tools: numpy, matplotlib, sympy
# Time: ~10 minutes

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# -----------------------------
# 1) Problem parameters
# -----------------------------
T0   = 90.0      # initial temperature (e.g., °F)
Tref = 60.0      # ambient/reference temperature (°F)
k    = 0.015     # heat-loss coefficient [1/s]
t_end = 1800.0   # simulate for 1800 s (30 min)

# -----------------------------
# 2) Analytic (symbolic) solution with SymPy
#     T(t) = Tref + (T0 - Tref) * exp(-k t)
# -----------------------------
t = sp.symbols('t', real=True)
T_sym = Tref + (T0 - Tref) * sp.exp(-k * t)   # closed-form
# Optionally confirm by d/dt (not required for speed):
# sp.simplify(sp.diff(T_sym, t) + k*(T_sym - Tref))  # -> 0
T_analytic = sp.lambdify(t, T_sym, 'numpy')   # fast numeric function

# -----------------------------
# 3) Forward-Euler integrator
#     T_{n+1} = T_n + dt * (-k) * (T_n - Tref)
# -----------------------------
def euler_cooling(T0, Tref, k, dt, t_end):
    n = int(np.ceil(t_end / dt))
    tt = np.linspace(0.0, n*dt, n+1)
    T  = np.empty(n+1, dtype=float)
    T[0] = T0
    for i in range(n):
        T[i+1] = T[i] + dt * (-k) * (T[i] - Tref)
    return tt, T

# -----------------------------
# 4) Compare multiple time steps
# -----------------------------
dts = [10.0, 2.0, 0.5]     # try coarser → finer
colors = ['tab:orange', 'tab:green', 'tab:red']

# high-res grid for analytic curve
t_plot = np.linspace(0.0, t_end, 1000)
T_plot = T_analytic(t_plot)

plt.figure(figsize=(7,4))
plt.plot(t_plot/60.0, T_plot, 'k-', lw=2.0, label="analytic")

errors_end = []  # store end-time absolute error for each dt

for dt, c in zip(dts, colors):
    tt, T_num = euler_cooling(T0, Tref, k, dt, t_end)
    plt.plot(tt/60.0, T_num, '.', ms=3, color=c, label=f"Euler dt={dt:g}s")
    # record error at final time (quick scalar indicator)
    err = abs(T_num[-1] - T_analytic(tt[-1]))
    errors_end.append((dt, err))

plt.xlabel("time [min]")
plt.ylabel("temperature [°F]")
plt.title("Newton cooling: analytic vs. Euler")
plt.legend()
plt.tight_layout()
plt.savefig("cooling_comparison.png", dpi=150)
print("Saved cooling_comparison.png")

# -----------------------------
# 5) Convergence (error vs. dt) on a log-log plot
# -----------------------------
errors_end = np.array(errors_end, dtype=float)  # shape (m,2): [dt, err]
dts_arr = errors_end[:,0]
errs_arr = errors_end[:,1]

plt.figure(figsize=(5,4))
plt.loglog(dts_arr, errs_arr, 'o-', label="|T_num(t_end) - T_exact(t_end)|")
# Reference ~ O(dt) (Euler is first-order)
ref = errs_arr[-1] * (dts_arr / dts_arr[-1])**1
plt.loglog(dts_arr, ref, '--', label="~ dt^1 reference")
plt.gca().invert_xaxis()
plt.xlabel("time step dt [s]")
plt.ylabel("end-time abs error")
plt.title("Convergence of forward-Euler (first order)")
plt.legend()
plt.tight_layout()
plt.savefig("cooling_convergence.png", dpi=150)
print("Saved cooling_convergence.png")
