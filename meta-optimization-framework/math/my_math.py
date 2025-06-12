try:
    import numpy as np
    USE_NUMPY = True
except ImportError:
    import importlib
    math = importlib.import_module('math')
    USE_NUMPY = False

import matplotlib.pyplot as plt

# --- Fixed constants for demonstration ---
S = 0.8                    # Symbolic output S(x)
N = 0.6                    # Neural output N(x)
lambda1 = 1.0              # Cognitive penalty weight
lambda2 = 1.0              # Efficiency penalty weight
R_cognitive = 0.3          # Cognitive penalty R_cognitive
R_efficiency = 0.2         # Efficiency penalty R_efficiency
P_biased = 0.75            # Bias-adjusted probability P(H|E,β)

# Precompute the regularization factor
if USE_NUMPY:
    regularization = np.exp(-(lambda1 * R_cognitive + lambda2 * R_efficiency))
else:
    regularization = math.exp(-(lambda1 * R_cognitive + lambda2 * R_efficiency))

# Range of α values from 0 to 1
if USE_NUMPY:
    alphas = np.linspace(0, 1, 200)
else:
    alphas = [i / 199 for i in range(200)]

# Compute Ψ for each α:
# Ψ = [α·S + (1−α)·N] × regularization × P_biased
if USE_NUMPY:
    Psi = (alphas * S + (1 - alphas) * N) * regularization * P_biased
else:
    Psi = [(alpha * S + (1 - alpha) * N) * regularization * P_biased for alpha in alphas]

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(alphas, Psi, color='blue', linewidth=2)
plt.title("Ψ(x) vs Integration Weight α", fontsize=14)
plt.xlabel("α (symbolic weight)", fontsize=12)
plt.ylabel("Ψ(x)", fontsize=12)
plt.grid(alpha=0.3)

# Annotate a few points
for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
    psi_val = (a * S + (1 - a) * N) * regularization * P_biased
    plt.scatter([a], [psi_val], color='red')
    plt.text(a, psi_val + 0.005, f"{psi_val:.3f}",
             ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()
