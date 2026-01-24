import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import simpson
from scipy.optimize import least_squares


# True value
S0 = 9999
I0 = 1
R0 = 0

beta = 0.3
gamma = 0.05


def SIR(y, t, beta, gamma):
    """
    :param y: initial condition
    :param t: time
    :param beta: para
    :param gamma: para
    :param S0: para
    :return: the SIR model ODEs
    """
    S, I, R = y
    N = S + I + R
    dS = - beta * S * I / N
    dI = beta * S * I / N - gamma * I
    dR = gamma * I
    return [dS, dI, dR]


def simulate_sir(t):
    solution = scipy.integrate.odeint(SIR, [S0, I0, R0], t, args=(beta, gamma))
    return solution.T


def plot_sir(S, I, R, t):
    plt.plot(t, S, label="True S")
    plt.plot(t, I, label="True I")
    plt.plot(t, R, label="True R")
    plt.legend()
    plt.show()


def residual(paras, I_data, t):
    """
    alpha = beta / N
    :param paras:
    :param I_data:
    :param t:
    :return:
    """
    beta, gamma, alpha = paras

    I0 = I_data[0]
    I_int = np.array([simpson(I_data[:i + 1], t[:i + 1]) for i in range(len(t))])
    I_int2 = np.array([simpson(I_data[:i + 1] ** 2, t[:i + 1]) for i in range(len(t))])
    int_double = 1 / 2 * (I_int ** 2)

    I_hat = I0 + (beta - gamma) * I_int - alpha * I_int2 - alpha * gamma * int_double
    return I_hat - I_data


def run_experiments(beta0, gamma0, S00, I_data, t):
    beta_list = []
    gamma_list = []
    S0_list = []
    def callback(x):
        beta_list.append(x[0])
        gamma_list.append(x[1])
        S0_list.append(x[0] / x[2] - I_data[0])

    alpha0 = beta0 / (S00 + I_data[0])
    x0 = [beta0, gamma0, alpha0]
    res = least_squares(residual, x0, args=(I_data, t), callback=callback)
    estimate = np.array([res.x[0], res.x[1], (res.x[0] / res.x[2] - I_data[0])])

    return {
        "beta_list": np.array(beta_list),
        "gamma_list": np.array(gamma_list),
        "S0_list": np.array(S0_list),
        "estimated": estimate,
        "true": (beta, gamma, S0),
        "t": t,
        "initial_guess": [beta0, gamma0, S00],
        "I_data": I_data,
    }


def plot_results(result):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot Beta
    axes[0].plot(result['beta_list'], label="Est Beta", marker='o', color='orange', markersize=4)
    axes[0].axhline(y=result['true'][0], label="True", color='black', linestyle='--')
    axes[0].set_title("Beta Convergence")
    axes[0].legend()

    # Plot Gamma
    axes[1].plot(result['gamma_list'], label="Est Gamma", marker='o', color='green', markersize=4)
    axes[1].axhline(y=result['true'][1], label="True", color='black', linestyle='--')
    axes[1].set_title("Gamma Convergence")
    axes[1].legend()

    # Plot S0
    axes[2].plot(result['S0_list'], label="Est S0", marker='o', color='blue', markersize=4)
    axes[2].axhline(y=result['true'][2], label="True", color='black', linestyle='--')
    axes[2].set_title("S0 Convergence")
    axes[2].legend()

    plt.tight_layout()
    plt.show()


def print_results(result):
    beta_true, gamma_true, S0_true = result["true"]
    beta_est, gamma_est, S0_est = result["estimated"]
    beta_init, gamma_init, S0_init = result["initial_guess"]

    beta_err = abs((beta_est - beta_true) / beta_true) * 100
    gamma_err = abs((gamma_est - gamma_true) / gamma_true) * 100
    S0_err = abs((S0_est - S0_true) / S0_true) * 100

    est_arr = np.array(result['estimated'])
    true_arr = np.array(result['true'])
    cost = np.sum((est_arr - true_arr) ** 2)

    iterations = len(result["beta_list"])

    print("=" * 82)
    print(f"{'OPTIMIZATION RESULTS':^82}")
    print("=" * 82)

    print("--- CONVERGENCE ---")
    print(f"Iterations: {iterations}")
    print(f"Final Cost: {cost}")

    print("--- BETA ---")
    print(f"True:      {beta_true}")
    print(f"Initial:   {beta_init}")
    print(f"Estimated: {beta_est}")
    print(f"Error (%): {beta_err}")

    print("--- GAMMA ---")
    print(f"True:      {gamma_true}")
    print(f"Initial:   {gamma_init}")
    print(f"Estimated: {gamma_est}")
    print(f"Error (%): {gamma_err}")

    print("--- S0 ---")
    print(f"True:      {S0_true}")
    print(f"Initial:   {S0_init}")
    print(f"Estimated: {S0_est}")
    print(f"Error (%): {S0_err}")
