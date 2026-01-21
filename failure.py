import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
r_h   = 2.0          # horizon radius (M=1)
Rmax  = 600.0        # outer boundary
N     = 4000         # grid resolution
eps   = 1e-10        # breakdown tolerance

r = np.linspace(r_h, Rmax, N)
dr = r[1] - r[0]

# ------------------------------------------------------------
# Step 1: Solve harmonic equation for v(r)
#   (r^2 v')' = 0
#   v(r_h)=0, v(Rmax)=-1
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
# Step 2: Conformal factor and warping factor
# ------------------------------------------------------------
u   = np.exp(v)
phi = u.copy()   # spherical symmetry choice
z   = np.ones_like(r)

# ------------------------------------------------------------
# Step 3: Generalized Jang ODE for Q(r)
#
#   Q' = (1-Q^2) * ( 2/r + (phi'/phi) )
#
# with Q(r_h)=0
# ------------------------------------------------------------
Q = np.zeros(N)

for i in range(N-1):
    dphi = (phi[i+1] - phi[i]) / dr
    rhs  = (1 - Q[i]**2) * (2/r[i] + dphi/phi[i])
    Q[i+1] = Q[i] + dr * rhs

    # Detect Jaracz-type breakdown
    if abs(Q[i+1]) > 1 - eps:
        print(f"Breakdown detected at r = {r[i+1]:.3f}")
        break
else:
    print("Result A diagnostics:")
    print("  No Jaracz-type breakdown detected")
    print(f"  max |Q| = {np.max(np.abs(Q)):.6f}")

# ------------------------------------------------------------
# Step 4: Plot diagnostics
# ------------------------------------------------------------
plt.figure(figsize=(10,6))
plt.plot(r, Q, label=r"$Q(r)$")
plt.axhline(1,  color='k', linestyle='--', linewidth=0.8)
plt.axhline(-1, color='k', linestyle='--', linewidth=0.8)
plt.xlabel(r"$r$")
plt.ylabel(r"$Q$")
plt.title("Result A: Jang slope for Jang/CFM system")
plt.legend()
plt.tight_layout()
plt.show()
