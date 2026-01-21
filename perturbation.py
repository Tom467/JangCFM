import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------
r_h = 2.0
Rmax = 600.0
N = 4000
eps_list = [-0.5, -0.25, 0.0, 0.25, 0.5]   # perturbation strengths
L = 20.0                                  # decay length for perturbation

r = np.linspace(r_h, Rmax, N)
dr = r[1] - r[0]

# -----------------------------
# Baseline Schwarzschild data
# -----------------------------
v = -1.0 + r_h / r          # harmonic function
u = np.exp(v)               # conformal factor

# -----------------------------
# Perturbation profile p(r)
# -----------------------------
def p_profile(r):
    return np.exp(-(r - r_h) / L)

p = p_profile(r)

# -----------------------------
# Solve Q ODE for each epsilon
# -----------------------------
results = {}

for eps in eps_list:
    phi = u * (1.0 + eps * p)
    phi_prime = np.gradient(phi, dr)

    Q = np.zeros_like(r)
    Qp = np.zeros_like(r)

    breakdown = False
    breakdown_r = None

    for i in range(N - 1):
        rhs = (1.0 - Q[i]**2) * (2.0 / r[i] + phi_prime[i] / phi[i])
        Qp[i] = rhs
        Q[i+1] = Q[i] + dr * rhs

        if abs(Q[i+1]) > 1.0 + 1e-6:
            breakdown = True
            breakdown_r = r[i+1]
            break

    results[eps] = {
        "Q": Q,
        "Qp": Qp,
        "phi": phi,
        "breakdown": breakdown,
        "breakdown_r": breakdown_r,
        "max_Q": np.max(np.abs(Q)),
        "max_Qp": np.max(np.abs(Qp))
    }

# -----------------------------
# Diagnostics
# -----------------------------
print("\nWarping factor robustness diagnostics:\n")
for eps in eps_list:
    res = results[eps]
    if res["breakdown"]:
        print(f"  eps = {eps:+.2f}: breakdown at r ≈ {res['breakdown_r']:.2f}")
    else:
        print(f"  eps = {eps:+.2f}: no finite-radius breakdown | "
              f"max|Q| = {res['max_Q']:.6f}, max|Q'| = {res['max_Qp']:.6e}")

# -----------------------------
# Plot Q(r)
# -----------------------------
plt.figure(figsize=(7,5))
for eps in eps_list:
    plt.plot(r, results[eps]["Q"], label=f"$\\varepsilon={eps}$")
plt.axhline(1.0, color='k', linestyle='--', linewidth=0.8)
plt.axhline(-1.0, color='k', linestyle='--', linewidth=0.8)
plt.xlabel(r"$r$")
plt.ylabel(r"$Q(r)$")

plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# Plot Q'(r)
# -----------------------------
plt.figure(figsize=(7,5))
for eps in eps_list:
    plt.plot(r, results[eps]["Qp"], label=f"$\\varepsilon={eps}$")
plt.xlabel(r"$r$")
plt.ylabel(r"$Q'(r)$")

plt.legend()
plt.tight_layout()
plt.show()
