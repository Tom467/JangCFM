import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# ============================================================
# Parameters
# ============================================================

Nr = 800
r_h = 2.0
Rmax = 600.0

dt_flow = 0.05
T_final = 1.0
Nt = int(T_final / dt_flow)

eps_Q = 1e-6

# Perturbation parameters
epsilon = 0.02     # amplitude
sigma = 3.0        # width

# ============================================================
# Radial grid
# ============================================================

r = np.linspace(r_h, Rmax, Nr)
dr = r[1] - r[0]

# ============================================================
# 1. Harmonic v
# ============================================================

A = lil_matrix((Nr, Nr))
b = np.zeros(Nr)

for i in range(1, Nr - 1):
    rp = r[i] + 0.5 * dr
    rm = r[i] - 0.5 * dr

    A[i, i - 1] = rm**2 / dr**2
    A[i, i]     = -(rp**2 + rm**2) / dr**2
    A[i, i + 1] = rp**2 / dr**2

A[0, 0] = 1.0
b[0] = 0.0
A[-1, -1] = 1.0
b[-1] = -1.0

A = A.tocsr()
v = spsolve(A, b)

# ============================================================
# 2. Conformal flow
# ============================================================

u = np.ones_like(r)
for _ in range(Nt):
    u *= np.exp(v * dt_flow)
    u[-1] = 1.0

# ============================================================
# 3. Zero scalar curvature (z ≡ 1)
# ============================================================

z = np.ones_like(r)

# ============================================================
# 4. Warping factor φ and perturbation
# ============================================================

phi = u * z

perturbation = 1.0 + epsilon * np.exp(-(r - r_h)**2 / sigma**2)
phi_pert = phi * perturbation

# Derivative
phi_p = np.zeros_like(phi_pert)
phi_p[1:-1] = (phi_pert[2:] - phi_pert[:-2]) / (2 * dr)
phi_p[0] = phi_p[1]
phi_p[-1] = phi_p[-2]

# ============================================================
# 5. Q-ODE
# ============================================================

Q = np.zeros_like(r)
break_index = None

for i in range(Nr - 1):
    if abs(Q[i]) >= 1.0 - eps_Q:
        break_index = i
        break

    rhs = phi_p[i] * np.sqrt(max(0.0, 1.0 - Q[i]**2))
    Q[i + 1] = Q[i] + dr * rhs

if break_index is None:
    print("Q diagnostics (perturbed φ):")
    print("  No breakdown detected up to Rmax")
else:
    print("Q diagnostics (perturbed φ):")
    print(f"  Breakdown detected at r ≈ {r[break_index]:.4f}")
    print(f"  |Q| ≈ {abs(Q[break_index]):.6f}")

# ============================================================
# 6. Recover f'
# ============================================================

fp = np.zeros_like(Q)
valid = np.abs(Q) < 1.0
fp[valid] = Q[valid] / (phi_pert[valid] * np.sqrt(1.0 - Q[valid]**2))
fp[~valid] = np.nan

# ============================================================
# Plots
# ============================================================

plt.figure(figsize=(7,4))
plt.plot(r, Q)
plt.axhline(1.0, linestyle='--', color='k')
plt.axhline(-1.0, linestyle='--', color='k')
plt.xlabel("r")
plt.ylabel("Q(r)")
plt.title("Q(r) with perturbed φ")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,4))
plt.plot(r, fp)
plt.xlabel("r")
plt.ylabel("f'(r)")
plt.title("Jang slope f'(r) (perturbed φ)")
plt.tight_layout()
plt.show()
