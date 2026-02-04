"""
Rejected

Since ratio models are identical to count with division N. But in real life
One ofter observe the data from state I, but we may have difficulty knowing N when collecting the data
Hence using count with division N seems to be more convenient
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

t = np.linspace(0, 100, 50)

# simulate SIR
solution = scipy.integrate.odeint(SIR, [S0, I0, R0], t, args=(beta, gamma))
S, I, R = solution.T

I_data = I + np.random.normal(0, 0.01*I, size=I.shape)

def run_print(beta0, gamma0, S00, plot=False):
    results = run_experiments(I_data, t, beta, gamma, S0, beta0, gamma0, S00)
    print_experiments(results, plot=plot)