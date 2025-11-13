# trajectory_tangent_normal_demo.py
# Goal: Plot a parametric curve r(t) and draw unit tangent T-hat and normal N-hat vectors.
# Tools: numpy, matplotlib
# Time: ~10 minutes

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1) Define a parametric curve r(t)
#    We'll use a simple parabola: r(t) = [ t , 0.5 t^2 ]
#    (smooth, non-self-intersecting; good for clean vectors)
# -----------------------------
t = np.linspace(-2.0, 2.0, 401)
x = t
y = 0.5 * t**2

# -----------------------------
# 2) First derivatives (velocity-like): r'(t) = [x', y']
#    We use np.gradient to keep it fully numpy & vectorized.
#    NOTE: gradient uses uniform spacing dt = t[1]-t[0]
# -----------------------------
dt = t[1] - t[0]
x_t = np.gradient(x, dt)
y_t = np.gradient(y, dt)

# Speed (magnitude of r'(t))
speed = np.sqrt(x_t**2 + y_t**2)

# Unit tangent: T-hat = r'(t) / ||r'(t)||
# Avoid divide-by-zero: add a tiny epsilon
eps = 1e-12
Tx = x_t / (speed + eps)
Ty = y_t / (speed + eps)

# -----------------------------
# 3) Unit normal N-hat
#    In 2D, a simple perpendicular to T-hat is obtained by rotating T by +90°:
#      N = [-Ty, Tx], then normalize.
#    (This choice gives one of the two normals; either orientation is fine.)
# -----------------------------
Nx = -Ty
Ny =  Tx
N_norm = np.sqrt(Nx**2 + Ny**2) + eps
Nx /= N_norm
Ny /= N_norm

# -----------------------------
# 4) Choose a few sample points to plot vectors (not too many arrows)
# -----------------------------
idx = np.linspace(20, len(t)-20, 10, dtype=int)  # 10 interior points
x_s, y_s = x[idx], y[idx]
Tx_s, Ty_s = Tx[idx], Ty[idx]
Nx_s, Ny_s = Nx[idx], Ny[idx]
spd_s     = speed[idx]

# Scale arrows by a small factor for visibility
scale_T = 0.4
scale_N = 0.4

# -----------------------------
# 5) (Optional) curvature diagnostic (nice talking point)
#    κ = |x' y'' - y' x''| / ( (x'^2 + y'^2)^(3/2) )
# -----------------------------
x_tt = np.gradient(x_t, dt)
y_tt = np.gradient(y_t, dt)
curv = np.abs(x_t * y_tt - y_t * x_tt) / (speed**3 + eps)
curv_s = curv[idx]

# -----------------------------
# 6) Plot the curve, color by speed, and add T/N arrows
# -----------------------------
plt.figure(figsize=(7,5))
scatter = plt.scatter(x, y, c=speed, s=8, cmap="viridis", label="trajectory (color = speed)")
plt.quiver(x_s, y_s, Tx_s, Ty_s, angles='xy', scale_units='xy', scale=1/scale_T,
           color='tab:orange', width=0.005, label=r"$\hat{\mathbf T}$")
plt.quiver(x_s, y_s, Nx_s, Ny_s, angles='xy', scale_units='xy', scale=1/scale_N,
           color='tab:red', width=0.005, label=r"$\hat{\mathbf N}$")

cb = plt.colorbar(scatter)
cb.set_label("speed = ||r'(t)||")

plt.axis('equal')
plt.xlabel("x(t)")
plt.ylabel("y(t)")
plt.title("Parametric curve with unit tangent and normal")
plt.legend()
plt.tight_layout()
plt.savefig("fig_tangent_normal.png", dpi=150)
print("Saved fig_tangent_normal.png")

# -----------------------------
# 7) Print a tiny table for discussion at the sampled points
# -----------------------------
print("\nSample points diagnostics (t, speed, curvature):")
for i, j in enumerate(idx):
    print(f"t={t[j]: .2f}  speed={spd_s[i]: .3f}  kappa={curv_s[i]: .3f}")
