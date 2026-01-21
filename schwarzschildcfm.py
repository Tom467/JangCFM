import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------
r_h = 2.0          # horizon radius
Rmax = 400.0       # outer boundary
N = 1600            # number of grid points

# Grid
r = np.linspace(r_h, Rmax, N)
dr = r[1] - r[0]

# -----------------------------
# Build radial Laplacian for v
# (spherical symmetry)
# -----------------------------
main_diag = np.zeros(N)
upper_diag = np.zeros(N-1)
lower_diag = np.zeros(N-1)

for i in range(1, N-1):
    main_diag[i] = -2.0 / dr**2
    upper_diag[i] = 1.0 / dr**2 + 1.0 / (r[i] * dr)
    lower_diag[i-1] = 1.0 / dr**2 - 1.0 / (r[i] * dr)

A = sp.diags(
    [lower_diag, main_diag, upper_diag],
    offsets=[-1, 0, 1],
    format="csr"
)

# RHS
b = np.zeros(N)

# -----------------------------
# Boundary conditions
# -----------------------------
# v(r_h) = 0
A[0, :] = 0.0
A[0, 0] = 1.0
b[0] = 0.0

# v(Rmax) = -1
A[-1, :] = 0.0
A[-1, -1] = 1.0
b[-1] = -1.0

# -----------------------------
# Solve
# -----------------------------
v_num = spla.spsolve(A, b)

# -----------------------------
# Analytic solution
# -----------------------------
v_exact = -1.0 + r_h / r

# -----------------------------
# Errors
# -----------------------------
linf_error = np.max(np.abs(v_num - v_exact))
l2_error = np.sqrt(np.sum((v_num - v_exact)**2) * dr)

print("Schwarzschild benchmark diagnostics:")
print(f"  L_inf error = {linf_error:.6e}")
print(f"  L2 error    = {l2_error:.6e}")

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(7, 5))
plt.plot(r, v_exact, 'k--', label='Exact Schwarzschild')
plt.plot(r, v_num, 'r', label='Numerical')
plt.xlabel("r")
plt.ylabel("v(r)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
