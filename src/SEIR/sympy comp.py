import sympy as sp

t = sp.symbols('t')
S = sp.Function('S')(t)
E = sp.Function('E')(t)
I = sp.Function('I')(t)
R = sp.Function('R')(t)

E_prime = sp.diff(E, t)
E_double_prime = sp.diff(E, t, 2)

S_prime = sp.diff(S, t)
I_prime = sp.diff(I, t)

alpha = sp.symbols('alpha')
sigma = sp.symbols('sigma')
gamma = sp.symbols('gamma')

expr = alpha * S_prime * I + alpha * S * I_prime - gamma * E_prime
expr_no_S = expr.subs(S, (E_prime + sigma * E) / alpha * I)
expr_no_S_S_prime = expr_no_S.subs(S_prime, - E_prime - sigma * E)

print(expr_no_S_S_prime)

