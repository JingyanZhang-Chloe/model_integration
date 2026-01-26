import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import simpson
from scipy.optimize import least_squares


# True value
S0 = 9999
I0 = 1
R0 = 0

alpha = 0.00003
gamma = 0.05

scales = [0.00001, 0.01, 10000]

def SIR(y, t, alpha, gamma):
    """
    :param y: initial condition
    :param t: time
    :param beta: para
    :param gamma: para
    :param S0: para
    :return: the SIR model ODEs
    """
    S, I, R = y
    dS = - alpha * S * I
    dI = alpha * S * I - gamma * I
    dR = gamma * I
    return [dS, dI, dR]


def simulate_sir(t):
    solution = scipy.integrate.odeint(SIR, [S0, I0, R0], t, args=(alpha, gamma))
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
    gamma = gamma
    S0 = S0
    :param paras:
    :param I_data:
    :param t:
    :return:
    """
    alpha, gamma, S0 = paras

    I0 = I_data[0]
    I_int = np.array([simpson(I_data[:i + 1], t[:i + 1]) for i in range(len(t))])
    I_int2 = np.array([simpson(I_data[:i + 1] ** 2, t[:i + 1]) for i in range(len(t))])
    int_double = 1 / 2 * (I_int ** 2)

    I_hat = I0 + (alpha * (S0 + I0) - gamma) * I_int - alpha * I_int2 - alpha * gamma * int_double
    return I_hat - I_data


def residual_mul(paras, I_data, t):
    """
    alpha = beta / N
    gamma = gamma
    mul = alpha * S0
    :param paras:
    :param I_data:
    :param t:
    :return:
    """
    alpha, gamma, mul = paras

    I0 = I_data[0]
    I_int = np.array([simpson(I_data[:i + 1], t[:i + 1]) for i in range(len(t))])
    I_int2 = np.array([simpson(I_data[:i + 1] ** 2, t[:i + 1]) for i in range(len(t))])
    int_double = 1 / 2 * (I_int ** 2)

    I_hat = I0 + ((mul + alpha * I0) - gamma) * I_int - alpha * I_int2 - alpha * gamma * int_double
    return I_hat - I_data


def run_experiments(alpha0, gamma0, S00, I_data, t, rescale=True):
    alpha_list = []
    gamma_list = []
    S0_list = []
    def callback(x):
        alpha_list.append(x[0])
        gamma_list.append(x[1])
        S0_list.append(x[2])

    x0 = [alpha0, gamma0, S00]
    if rescale:
        res = least_squares(residual, x0, args=(I_data, t), bounds=(0, np.inf), x_scale=scales, callback=callback)
    else:
        res = least_squares(residual, x0, args=(I_data, t), bounds=(0, np.inf), callback=callback)

    return {
        "alpha_list": np.array(alpha_list),
        "gamma_list": np.array(gamma_list),
        "S0_list": np.array(S0_list),
        "estimated": res.x,
        "true": (alpha, gamma, S0),
        "t": t,
        "initial_guess": x0,
        "I_data": I_data,
    }


def run_experiments_mul(alpha0, gamma0, S00, I_data, t):
    alpha_list = []
    gamma_list = []
    S0_list = []

    def callback(x):
        alpha_list.append(x[0])
        gamma_list.append(x[1])
        S0_list.append(x[2] / x[0])

    x0 = [alpha0, gamma0, S00 * alpha0]
    res = least_squares(residual_mul, x0, args=(I_data, t), bounds=(0, np.inf), callback=callback)
    estimated = np.array([res.x[0], res.x[1], res.x[2] / res.x[0]])

    return {
        "alpha_list": np.array(alpha_list),
        "gamma_list": np.array(gamma_list),
        "S0_list": np.array(S0_list),
        "estimated": estimated,
        "true": (alpha, gamma, S0),
        "t": t,
        "initial_guess": [alpha0, gamma0, S00],
        "I_data": I_data,
    }


def plot_results(result):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot Beta
    axes[0].plot(result['alpha_list'], label="Est Alpha", marker='o', color='orange', markersize=4)
    axes[0].axhline(y=result['true'][0], label="True", color='black', linestyle='--')
    axes[0].set_title("Alpha Convergence")
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


def plot_three_results(result, result_, result__):
    """
    result : from standard non-rescaled least squares
    result_ : from parameter rescaling
    result__ : from the multiplicative re-parameterization method
    :param result:
    :param result_:
    :param result__:
    :return:
    """
    # Create a 3x3 subplot grid
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    # Organize inputs for easy looping
    # Format: (Data_Dictionary, Method_Name_Label)
    methods = [
        (result, "No Rescale"),
        (result_, "Rescale"),
        (result__, "Mul Method")
    ]

    for i, (data, name) in enumerate(methods):
        # --- Alpha Plot (Column 0) ---
        axes[i, 0].plot(data['alpha_list'], label="Est Alpha", marker='o', color='orange', markersize=4)
        axes[i, 0].axhline(y=data['true'][0], label="True", color='black', linestyle='--')
        axes[i, 0].set_title(f"{name}: Alpha Convergence")
        axes[i, 0].set_xlabel("Iterations")
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)

        # --- Gamma Plot (Column 1) ---
        axes[i, 1].plot(data['gamma_list'], label="Est Gamma", marker='o', color='green', markersize=4)
        axes[i, 1].axhline(y=data['true'][1], label="True", color='black', linestyle='--')
        axes[i, 1].set_title(f"{name}: Gamma Convergence")
        axes[i, 1].set_xlabel("Iterations")
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)

        # --- S0 Plot (Column 2) ---
        axes[i, 2].plot(data['S0_list'], label="Est S0", marker='o', color='blue', markersize=4)
        axes[i, 2].axhline(y=data['true'][2], label="True", color='black', linestyle='--')
        axes[i, 2].set_title(f"{name}: S0 Convergence")
        axes[i, 2].set_xlabel("Iterations")
        axes[i, 2].legend()
        axes[i, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def print_results(result):
    alpha_true, gamma_true, S0_true = result["true"]
    alpha_est, gamma_est, S0_est = result["estimated"]
    alpha_init, gamma_init, S0_init = result["initial_guess"]

    alpha_err = abs((alpha_est - alpha_true) / alpha_true) * 100
    gamma_err = abs((gamma_est - gamma_true) / gamma_true) * 100
    S0_err = abs((S0_est - S0_true) / S0_true) * 100

    est_arr = np.array(result['estimated'])
    true_arr = np.array(result['true'])
    cost = np.sum((est_arr - true_arr) ** 2)

    iterations = len(result["alpha_list"])

    print("=" * 82)
    print(f"{'OPTIMIZATION RESULTS':^82}")
    print("=" * 82)

    print("--- CONVERGENCE ---")
    print(f"Iterations: {iterations}")
    print(f"Final Cost: {cost}")

    print("--- BETA ---")
    print(f"True:      {alpha_true}")
    print(f"Initial:   {alpha_init}")
    print(f"Estimated: {alpha_est}")
    print(f"Error (%): {alpha_err}")

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
