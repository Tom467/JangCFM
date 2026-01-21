import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
r_h   = 2.0
Rmax  = 300.0
N     = 4000
eps_Q = 1e-6        # closeness to |Q|=1
eps_d = 1e-3        # threshold for large derivative

r = np.linspace(r_h, Rmax, N)
dr = r[1] - r[0]

# ------------------------------------------------------------
# Step 1: Harmonic equation for v
# ------------------------------------------------------------
diag  = np.zeros(N)
upper = np.zeros(N-1)
lower = np.zeros(N-1)

for i in range(1, N-1):
    rp = r[i] + dr/2
    rm = r[i] - dr/2
    upper[i]   = rp**2 / dr**2
    lower[i-1] = rm**2 / dr**2
    diag[i]    = -(rp**2 + rm**2) / dr**2

diag[0]  = 1.0
diag[-1] = 1.0

A = diags([lower, diag, upper], offsets=[-1,0,1], format='csr')

b = np.zeros(N)
b[0]  = 0.0
b[-1] = -1.0

v = spsolve(A, b)

# ------------------------------------------------------------
# Step 2: Conformal quantities
# ------------------------------------------------------------
u   = np.exp(v)
phi = u.copy()

# ------------------------------------------------------------
# Step 3: Jang ODE for Q
# ------------------------------------------------------------
Q      = np.zeros(N)
Qprime = np.zeros(N)

breakdown_radius = None
saturation_radius = None

for i in range(N-1):
    dphi = (phi[i+1] - phi[i]) / dr
    rhs  = (1 - Q[i]**2) * (2/r[i] + dphi/phi[i])

    Qprime[i] = rhs
    Q[i+1]    = Q[i] + dr * rhs

    # Detect approach to |Q|=1
    if saturation_radius is None and abs(Q[i+1]) > 1 - eps_Q:
        saturation_radius = r[i+1]

    # Detect Jaracz-type breakdown (large slope near |Q|=1)
    if abs(Q[i+1]) > 1 - eps_Q and abs(rhs) > eps_d:
        breakdown_radius = r[i+1]
        break

Qprime[-1] = Qprime[-2]

# ------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------
print("\nResult A diagnostics:")

if breakdown_radius is not None:
    print(f"  Finite-radius breakdown detected at r = {breakdown_radius:.3f}")
else:
    print("  No finite-radius Jaracz-type breakdown detected")

if saturation_radius is not None:
    print(f"  Q approaches |Q|=1 asymptotically near r ≈ {saturation_radius:.3f}")
else:
    print("  Q remains bounded away from |Q|=1")

print(f"  max |Q|     = {np.max(np.abs(Q)):.6f}")
print(f"  max |Q'|    = {np.max(np.abs(Qprime)):.6e}")

# ------------------------------------------------------------
# Plots
# ------------------------------------------------------------
plt.figure(figsize=(10,5))
plt.plot(r, Q, label=r"$Q(r)$")
plt.axhline(1,  color='k', linestyle='--', linewidth=0.8)
plt.axhline(-1, color='k', linestyle='--', linewidth=0.8)
plt.xlabel(r"$r$")
plt.ylabel(r"$Q$")

plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(r, Qprime, label=r"$Q'(r)$")
plt.xlabel(r"$r$")
plt.ylabel(r"$Q'$")

plt.legend()
plt.tight_layout()
plt.show()
