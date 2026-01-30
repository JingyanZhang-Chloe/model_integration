from seir_model import *
import numpy as np
from scipy.optimize import lsq_linear

t = np.linspace(0, 1000, 100)
S, E, I, R = simulate_seir(t)

#I_data = I + np.random.normal(0, 0.01 * I, size=I.shape)
I_data = I
B1, B2, B3, B4, B5, B6 = get_blocks(I_data, t)

X = np.column_stack([t, B2, B3, B4, B5, B6])
y = I_data - I_data[0]

cond_number = np.linalg.cond(X)
print(f"Condition Number of Matrix X: {cond_number:.2e}")

print(f"{"=" * 20}")

K, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
check_val = - 2 * K[1] * K[2]
consistency_error = abs(K[4] - check_val) / abs(K[4]) * 100

print(f"K5 (Actual): {K[4]:.2e}")
print(f"K5 (Predicted from K2, K3): {check_val:.2e}")
print(f"Consistency Error: {consistency_error:.2f}%")

lb = [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
ub = [ np.inf,  0.0,    0.0,     np.inf,  0.0,    0.0]

res = lsq_linear(X, y, bounds=(lb, ub))
K_constrained = res.x

check_val = - 2 * K_constrained[1] * K_constrained[2]
consistency_error = abs(K_constrained[4] - check_val) / abs(K_constrained[4]) * 100

print(f"K5 (Actual): {K_constrained[4]:.2e}")
print(f"K5 (Predicted from K2, K3): {check_val:.2e}")
print(f"Consistency Error: {consistency_error:.2f}%")


from sklearn.linear_model import Ridge

# alpha_ridge is the penalty. Start with a small value.
ridge = Ridge(alpha=1e-5, fit_intercept=False)
ridge.fit(X, y)
K_ridge = ridge.coef_

# Now check the consistency
check_val = -2 * K_ridge[1] * K_ridge[2]
consistency_error = abs(K_ridge[4] - check_val) / abs(K_ridge[4]) * 100
print(f"Ridge Consistency Error: {consistency_error:.2f}%")