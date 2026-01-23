"""
Rejected

Since for standard SIR model, beta is in range around [0.1, 0.9]. If we use count with no division by N
We need to manually change the true parameter beta to match the assumption.
Once we fix the beta to be identical to beta * / N, our model is identical to one that is with division by N and with beta *
Hence we could eliminate this case
"""

from SIR_basic_initial_condition_analysis import *

# initial condition
S0 = 9999
I0 = 1
R0 = 0
y = S0, I0, R0

# true parameters
beta = 0.00003
gamma = 0.05

t = np.linspace(0, 100, 50)

# simulate SIR
solution = scipy.integrate.odeint(SIR_no_divide_N, [S0, I0, R0], t, args=(beta, gamma))
S, I, R = solution.T

def run_print(I_data, beta0, gamma0, S00, plot=False):
    results = run_experiments_alpha(I_data, t, beta, gamma, S0, beta0, gamma0, S00)
    print_experiments(results, plot=plot)