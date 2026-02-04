"""
In this script, we treat S, I and R as actual count values of the corresponding states.
We normalize the SIR model by dividing N in the equations as defined in SIR_basic.
"""

from SIR_basic_initial_condition_analysis import *

# initial condition
S0 = 9999
I0 = 1
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

def run_print(I_data, beta0, gamma0, S00, plot=False):
    results = run_experiments(I_data, t, beta, gamma, S0, beta0, gamma0, S00)
    print_experiments(results, plot=plot)