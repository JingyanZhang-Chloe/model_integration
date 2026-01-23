"""
Rejected
"""

from SIR_basic_initial_condition_analysis import *

# initial condition
S0 = 0.9999
I0 = 0.0001
R0 = 0
y = S0, I0, R0

# true parameters
beta = 0.3
gamma = 0.05

# time
t = np.linspace(0, 100, 50)

# simulate SIR
solution = scipy.integrate.odeint(SIR, [S0, I0, R0], t, args=(beta, gamma))
S, I, R = solution.T

I_data = I + np.random.normal(0, 0.01*I, size=I.shape)

def run_print(beta0, gamma0, S00, plot=False):
    results = run_experiments_alpha(I_data, t, beta, gamma, S0, beta0, gamma0, S00)
    print_experiments(results, plot=plot)