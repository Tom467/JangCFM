import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# ============================================================
# Parameters
# ============================================================
r_h = 2.0
Rmax = 600.0

# Try several refinements
N_list = [400, 800, 1600]

# Number of points to inspect near the horizon
nh = 10

print("\n=== Schwarzschild Near-Horizon Asymptotics Check ===\n")

plt.figure(figsize=(7, 5))

for N in N_list:
    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------
    r = np.linspace(r_h, Rmax, N)
    dr = r[1] - r[0]

    # --------------------------------------------------------
    # Radial Laplacian for v
    # --------------------------------------------------------
    main = np.zeros(N)
    upper = np.zeros(N-1)
    lower = np.zeros(N-1)

    for i in range(1, N-1):
        main[i] = -2.0 / dr**2
        upper[i] = 1.0 / dr**2 + 1.0 / (r[i] * dr)
        lower[i-1] = 1.0 / dr**2 - 1.0 / (r[i] * dr)

    A = sp.diags([lower, main, upper], offsets=[-1, 0, 1], format="csr")
    b = np.zeros(N)

    # --------------------------------------------------------
    # Boundary conditions
    # --------------------------------------------------------
    A[0, :] = 0.0
    A[0, 0] = 1.0
    b[0] = 0.0          # v(r_h) = 0

    A[-1, :] = 0.0
    A[-1, -1] = 1.0
    b[-1] = -1.0        # v(Rmax) = -1

    # --------------------------------------------------------
    # Solve
    # --------------------------------------------------------
    v = spla.spsolve(A, b)

    # --------------------------------------------------------
    # Derived quantities
    # --------------------------------------------------------
    u = np.exp(v)
    phi = u.copy()
    Q = np.zeros_like(r)

    # --------------------------------------------------------
    # Near-horizon diagnostics
    # --------------------------------------------------------
    print(f"N = {N}")
    print("  Near-horizon values:")
    for i in range(nh):
        print(f"    r = {r[i]:.6f} | "
              f"u = {u[i]:.8f} | "
              f"phi = {phi[i]:.8f} | "
              f"Q = {Q[i]:.8e}")
    print("")

    # --------------------------------------------------------
    # Plot u(r) near horizon
    # --------------------------------------------------------
    plt.plot(r[:nh], u[:nh], marker='o', label=f"N = {N}")

# ------------------------------------------------------------
# Plot formatting
# ------------------------------------------------------------
plt.xlabel("r")
plt.ylabel("u(r)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
