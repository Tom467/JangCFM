import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------
r_h = 2.0          # horizon radius
Rmax = 600.0       # outer boundary
N = 800            # number of grid points

# Grid
r = np.linspace(r_h, Rmax, N)
dr = r[1] - r[0]

# -----------------------------
# Build radial Laplacian for v
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

# Right-hand side
b = np.zeros(N)

# -----------------------------
# Boundary conditions
# v(r_h) = 0
# v(Rmax) = -1
# -----------------------------
A[0, :] = 0.0
A[0, 0] = 1.0
b[0] = 0.0

A[-1, :] = 0.0
A[-1, -1] = 1.0
b[-1] = -1.0

# -----------------------------
# Solve for v(r)
# -----------------------------
v_num = spla.spsolve(A, b)

# Exact analytic solution
v_exact = -1.0 + r_h / r

# -----------------------------
# Derived quantities
# -----------------------------
u_num = np.exp(v_num)
u_exact = np.exp(v_exact)

z_num = np.ones_like(r)  # R=0 solution
phi_num = u_num.copy()   # warping factor
Q_num = np.zeros_like(r) # Jang slope

# -----------------------------
# Diagnostics
# -----------------------------
print("Harmonic function v(r) diagnostics:")
print(f"  max |v_num - v_exact| = {np.max(np.abs(v_num - v_exact)):.3e}")
print(f"  v(r_h) = {v_num[0]:.6f}, v(Rmax) = {v_num[-1]:.6f}\n")

print("Conformal factor u(r) diagnostics:")
print(f"  u(r_h) = {u_num[0]:.6f}")
print(f"  min(u) = {np.min(u_num):.6f}")
print(f"  max |u - u_exact| = {np.max(np.abs(u_num - u_exact)):.3e}\n")

print("Zero scalar curvature z(r) diagnostics:")
print(f"  min(z) = {np.min(z_num):.6f}")
print(f"  max(z) = {np.max(z_num):.6f}")
print(f"  max |z-1| = {np.max(np.abs(z_num-1)):.3e}\n")

print("Warping factor phi(r) diagnostics:")
print(f"  phi(r_h) = {phi_num[0]:.6f}")
print(f"  min(phi) = {np.min(phi_num):.6f}")
print(f"  max |phi - u| = {np.max(np.abs(phi_num - u_num)):.3e}\n")

print("Jang slope Q(r) diagnostics:")
print(f"  max |Q| = {np.max(np.abs(Q_num)):.3e}")
print(f"  Q(r_h) = {Q_num[0]:.6f}, Q(Rmax) = {Q_num[-1]:.6f}\n")

# -----------------------------
# Optional plots
# -----------------------------
plt.figure(figsize=(7,5))
plt.plot(r, v_num, 'r', label='v_num')
plt.plot(r, v_exact, 'k--', label='v_exact')
plt.xlabel("r")
plt.ylabel("v(r)")
plt.title("Schwarzschild harmonic function v(r)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
