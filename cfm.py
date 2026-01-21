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

eps_Q = 1e-6   # threshold for detecting breakdown |Q| -> 1

# ============================================================
# Radial grid
# ============================================================

r = np.linspace(r_h, Rmax, Nr)
dr = r[1] - r[0]

# ============================================================
# 1. Solve Laplacian v = 0
# ============================================================

A = lil_matrix((Nr, Nr))
b = np.zeros(Nr)

for i in range(1, Nr - 1):
    rp = r[i] + 0.5 * dr
    rm = r[i] - 0.5 * dr

    A[i, i - 1] = rm**2 / dr**2
    A[i, i]     = -(rp**2 + rm**2) / dr**2
    A[i, i + 1] = rp**2 / dr**2

# Boundary conditions
A[0, 0] = 1.0
b[0] = 0.0

A[-1, -1] = 1.0
b[-1] = -1.0

A = A.tocsr()
v = spsolve(A, b)

# ============================================================
# 2. Conformal flow: evolve u
# ============================================================

u = np.ones_like(r)

for _ in range(Nt):
    u *= np.exp(v * dt_flow)
    u[-1] = 1.0

# ============================================================
# 3. Zero–scalar–curvature equation for z (R = 0)
# ============================================================

Az = lil_matrix((Nr, Nr))
bz = np.zeros(Nr)

for i in range(1, Nr - 1):
    rp = r[i] + 0.5 * dr
    rm = r[i] - 0.5 * dr

    Az[i, i - 1] = rm**2 / dr**2
    Az[i, i]     = -(rp**2 + rm**2) / dr**2
    Az[i, i + 1] = rp**2 / dr**2

# Regularity at horizon: z'(r_h) = 0
Az[0, 0] = 1.0
Az[0, 1] = -1.0
bz[0] = 0.0

# Asymptotics
Az[-1, -1] = 1.0
bz[-1] = 1.0

Az = Az.tocsr()
z = spsolve(Az, bz)

# ============================================================
# 4. Warping factor
# ============================================================

phi = u * z

# Compute phi'
phi_prime = np.zeros_like(phi)
phi_prime[1:-1] = (phi[2:] - phi[:-2]) / (2 * dr)
phi_prime[0] = phi_prime[1]
phi_prime[-1] = phi_prime[-2]

# ============================================================
# 5. Generalized Jang equation: Q-ODE
#     Q' = phi'(r) * sqrt(1 - Q^2)
# ============================================================

Q = np.zeros_like(r)
break_index = None

for i in range(Nr - 1):
    if abs(Q[i]) >= 1.0 - eps_Q:
        break_index = i
        break

    rhs = phi_prime[i] * np.sqrt(max(0.0, 1.0 - Q[i]**2))
    Q[i + 1] = Q[i] + dr * rhs

if break_index is None:
    print("Q diagnostics:")
    print("  No breakdown detected up to Rmax")
else:
    print("Q diagnostics:")
    print(f"  Breakdown detected at r ≈ {r[break_index]:.4f}")
    print(f"  |Q| ≈ {abs(Q[break_index]):.6f}")

# ============================================================
# 6. Recover f'(r) where valid
# ============================================================

fp = np.zeros_like(Q)
valid = np.abs(Q) < 1.0

fp[valid] = Q[valid] / (phi[valid] * np.sqrt(1.0 - Q[valid]**2))
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
plt.title("Normalized Jang gradient Q(r)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,4))
plt.plot(r, fp)
plt.xlabel("r")
plt.ylabel("f'(r)")
plt.title("Jang slope f'(r)")
plt.tight_layout()
plt.show()
