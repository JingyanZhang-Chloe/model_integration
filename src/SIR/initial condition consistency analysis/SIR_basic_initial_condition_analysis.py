import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import simpson
from scipy.optimize import least_squares


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


def SIR_no_divide_N(y, t, beta, gamma):
    """
    :param y: initial condition
    :param t: time
    :param beta: para
    :param gamma: para
    :param S0: para
    :return: the SIR model ODEs
    """
    S, I, R = y
    dS = - beta * S * I
    dI = beta * S * I - gamma * I
    dR = gamma * I
    return [dS, dI, dR]


# Function to run experiment with no alpha applied
def run_experiments(I_data, t, beta, gamma, S0, beta0, gamma0, S00):

    def residual(paras, I_data):
        I0 = I_data[0]
        I_int = np.array([simpson(I_data[:i + 1], t[:i + 1]) for i in range(len(t))])
        I_int2 = np.array([simpson(I_data[:i + 1] ** 2, t[:i + 1]) for i in range(len(t))])
        int_double = 1 / 2 * (I_int ** 2)
        beta, gamma, S0 = paras
        I_hat = I0 + ((beta * S0 + beta * I0) - gamma) * I_int - beta * I_int2 - beta * gamma * int_double
        return I_data - I_hat

    beta_list = []
    gamma_list = []
    S0_list = []

    def callback(x):
        beta_list.append(x[0])
        gamma_list.append(x[1])
        S0_list.append(x[2])

    x0 = [beta0, gamma0, S00]
    res = least_squares(residual, x0, args=(I_data,), callback=callback)

    return {
        "beta_list": np.array(beta_list),
        "gamma_list": np.array(gamma_list),
        "S0_list": np.array(S0_list),
        "estimated": res.x,
        "true": (beta, gamma, S0),
        "t": t,
        "initial_guess": x0,
        "I_data": I_data,
        "alpha": False
    }


# Function to run experiment with alpha
def run_experiments_alpha(I_data, t, beta, gamma, S0, beta0, gamma0, S00):

    def residual(paras, I_data):
        I0 = I_data[0]
        I_int = np.array([simpson(I_data[:i + 1], t[:i + 1]) for i in range(len(t))])
        I_int2 = np.array([simpson(I_data[:i + 1] ** 2, t[:i + 1]) for i in range(len(t))])
        int_double = 1 / 2 * (I_int ** 2)
        beta, gamma, alpha = paras
        I_hat = I0 + ((alpha + beta * I0) - gamma) * I_int - beta * I_int2 - beta * gamma * int_double
        return I_data - I_hat

    beta_list = []
    gamma_list = []
    S0_list = []

    def callback(x):
        beta_list.append(x[0])
        gamma_list.append(x[1])
        S0_list.append(x[2]/x[0])

    x0 = [beta0, gamma0, beta0*S00]
    res = least_squares(residual, x0, args=(I_data,), callback=callback)
    estimated = np.array([res.x[0], res.x[1], res.x[2]/res.x[0]])

    return {
        "beta_list": np.array(beta_list),
        "gamma_list": np.array(gamma_list),
        "S0_list": np.array(S0_list),
        "estimated": estimated,
        "true": (beta, gamma, S0),
        "t": t,
        "initial_guess": [beta0, gamma0, S00],
        "I_data": I_data,
        "alpha": True
    }


def print_experiments(result, plot=False):
    # Unpack values
    beta_true, gamma_true, S0_true = result["true"]
    beta_est, gamma_est, S0_est = result["estimated"]
    beta_init, gamma_init, S0_init = result["initial_guess"]

    # Calculate relative errors (No rounding)
    beta_err = abs((beta_est - beta_true) / beta_true) * 100 if beta_true != 0 else 0.0
    gamma_err = abs((gamma_est - gamma_true) / gamma_true) * 100 if gamma_true != 0 else 0.0
    S0_err = abs((S0_est - S0_true) / S0_true) * 100 if S0_true != 0 else 0.0

    # Convergence cost (MSE)
    est_arr = np.array(result['estimated'])
    true_arr = np.array(result['true'])
    cost = np.sum((est_arr - true_arr) ** 2)

    iterations = len(result["beta_list"])
    mode = "WITH ALPHA TRICK" if result.get('alpha') else "STANDARD (NO ALPHA)"

    # --- RAW DATA PRINTING ---
    print("\n" + "=" * 40)
    print(f"EXPERIMENT REPORT: {mode}")
    print("=" * 40)

    print("--- CONVERGENCE ---")
    print(f"Iterations: {iterations}")
    print(f"Final Cost: {cost}")
    print("-" * 20)

    # Group by Parameter for easy comparison
    print("--- BETA ---")
    print(f"True:      {beta_true}")
    print(f"Initial:   {beta_init}")
    print(f"Estimated: {beta_est}")
    print(f"Error (%): {beta_err}")

    print("-" * 20)

    print("--- GAMMA ---")
    print(f"True:      {gamma_true}")
    print(f"Initial:   {gamma_init}")
    print(f"Estimated: {gamma_est}")
    print(f"Error (%): {gamma_err}")

    print("-" * 20)

    print("--- S0 ---")
    print(f"True:      {S0_true}")
    print(f"Initial:   {S0_init}")
    print(f"Estimated: {S0_est}")
    print(f"Error (%): {S0_err}")

    print("=" * 40 + "\n")

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Plot Beta
        axes[0].plot(result['beta_list'], label="Est Beta", marker='o', color='orange', markersize=4)
        axes[0].axhline(y=beta_true, label="True", color='black', linestyle='--')
        axes[0].set_title("Beta Convergence")
        axes[0].legend()

        # Plot Gamma
        axes[1].plot(result['gamma_list'], label="Est Gamma", marker='o', color='green', markersize=4)
        axes[1].axhline(y=gamma_true, label="True", color='black', linestyle='--')
        axes[1].set_title("Gamma Convergence")
        axes[1].legend()

        # Plot S0
        axes[2].plot(result['S0_list'], label="Est S0", marker='o', color='blue', markersize=4)
        axes[2].axhline(y=S0_true, label="True", color='black', linestyle='--')
        axes[2].set_title("S0 Convergence")
        axes[2].legend()

        plt.tight_layout()
        plt.show()


def print_experiments_(result, plot=False):
    # Unpack values
    beta_true, gamma_true, S0_true = result["true"]
    beta_est, gamma_est, S0_est = result["estimated"]
    beta_init, gamma_init, S0_init = result["initial_guess"]

    # Calculate relative errors (%)
    beta_err = abs((beta_est - beta_true) / beta_true) * 100
    gamma_err = abs((gamma_est - gamma_true) / gamma_true) * 100
    S0_err = abs((S0_est - S0_true) / S0_true) * 100

    iterations = len(result["beta_list"])

    # --- PRINTING ---
    print("=" * 82)
    print(f"{'OPTIMIZATION RESULTS':^82}")
    if result['alpha']:
        print("WITH ALPHA")
    print("=" * 82)

    # Table Header
    # Cols: Param (10) | True (12) | Initial (12) | Estimated (12) | Error (10)
    print(f"| {'Parameter':<10} | {'True Value':>12} | {'Initial':>12} | {'Estimated':>12} | {'Error (%)':>10} |")
    print("-" * 82)

    # Rows
    print(f"| {'Beta':<10} | {beta_true} | {beta_init} | {beta_est} | {beta_err}% |")
    print(f"| {'Gamma':<10} | {gamma_true} | {gamma_init} | {gamma_est} | {gamma_err}% |")
    print(f"| {'S0':<10} | {S0_true} | {S0_init} | {S0_est} | {S0_err}% |")

    print("-" * 82)

    # Convergence Info
    print(f"{'Convergence Statistics':^82}")
    print("-" * 82)
    print(f"Total Iterations:    {iterations}")
    print(f"Final Cost (MSE):    {np.sum((result['estimated'] - result['true']) ** 2)}")
    print("=" * 82)
    print("\n")

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Plot Beta
        axes[0].plot(result['beta_list'], label="Est Beta", marker='o', color='orange', markersize=4)
        axes[0].axhline(y=beta_true, label="True", color='black', linestyle='--')
        axes[0].set_title("Beta Convergence")
        axes[0].legend()

        # Plot Gamma
        axes[1].plot(result['gamma_list'], label="Est Gamma", marker='o', color='green', markersize=4)
        axes[1].axhline(y=gamma_true, label="True", color='black', linestyle='--')
        axes[1].set_title("Gamma Convergence")
        axes[1].legend()

        # Plot S0
        axes[2].plot(result['S0_list'], label="Est S0", marker='o', color='blue', markersize=4)
        axes[2].axhline(y=S0_true, label="True", color='black', linestyle='--')
        axes[2].set_title("S0 Convergence")
        axes[2].legend()

        plt.tight_layout()
        plt.show()
