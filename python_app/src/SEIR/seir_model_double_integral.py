import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import cumulative_simpson
from scipy.optimize import least_squares

# True values (based on the paper by Mizuka Komatsu) (slightly changed)
S0 = 9999
E0 = 0
I0 = 1
R0 = 0

alpha = 0.00002
sigma = 0.01
gamma = 0.005

# alpha, sigma, gamma, S0, E0
scales = [0.00001, 0.01, 0.001, 10000, 0.1]


def SEIR(y, t, alpha, sigma, gamma):
    """
    :param y: initial condition
    :param t: time
    :param beta: para
    :param gamma: para
    :param S0: para
    :return: the SIR model ODEs
    """
    S, E, I, R = y
    dS = - alpha * S * I
    dE = alpha * S * I - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * I
    return [dS, dE, dI, dR]


def simulate_seir(t):
    solution = scipy.integrate.odeint(SEIR, [S0, E0, I0, R0], t, args=(alpha, sigma, gamma))
    return solution.T


def plot_seir(S, E, I, R, t):
    plt.plot(t, S, label="True S")
    plt.plot(t, E, label="True E")
    plt.plot(t, I, label="True I")
    plt.plot(t, R, label="True R")
    plt.legend()
    plt.show()


def get_blocks(I_data, t):
    """
    return the 6 blocks
    :param I_data:
    :param t:
    :return: B1, B2, B3, B4, B5, B6
    """
    I0 = I_data[0]
    I_int = cumulative_simpson(I_data, x=t, initial=0)
    I_int_2 = cumulative_simpson(I_data ** 2, x=t, initial=0)

    B1 = 1
    B2 = I_int
    B3 = cumulative_simpson(I_data**2 - I0 ** 2, x=t, initial=0)
    B4 = cumulative_simpson(I_int, x=t, initial=0)
    B5 = cumulative_simpson(I_int_2, x=t, initial=0)
    B6 = cumulative_simpson(I_int ** 2, x=t, initial=0)

    return B1, B2, B3, B4, B5, B6


# standard least square with rescale, non-linear
def residual(paras, I_data, B1, B2, B3, B4, B5, B6, t):
    """
    alpha = beta / N
    sigma = sigma
    gamma = gamma
    S0 = S0
    E0 = E0
    :param paras:
    :param I_data:
    :param t:
    :return:
    """
    alpha, sigma, gamma, S0, E0 = paras
    I0 = I_data[0]

    C1 = sigma * (E0 - I0) * t * B1
    C2 = - (sigma + gamma) * B2
    C3 = - 1/2 * alpha * B3
    C4 = (alpha * sigma * (S0 + E0 + I0) - sigma * gamma) * B4
    C5 = - alpha * (gamma + sigma) * B5
    C6 = - 1/2 * alpha * sigma * gamma * B6

    I_hat = I0 + C1 + C2 + C3 + C4 + C5 + C6
    return I_hat - I_data


def run_experiments(alpha0, sigma0, gamma0, S00, E00, I_data, t, rescale=True):
    alpha_list = []
    sigma_list = []
    gamma_list = []
    S0_list = []
    E0_list = []

    def callback(x):
        alpha_list.append(x[0])
        sigma_list.append(x[1])
        gamma_list.append(x[2])
        S0_list.append(x[3])
        E0_list.append(x[4])

    x0 = [alpha0, sigma0, gamma0, S00, E00]
    B1, B2, B3, B4, B5, B6 = get_blocks(I_data, t)
    if rescale:
        res = least_squares(residual, x0, args=(I_data, B1, B2, B3, B4, B5, B6, t), bounds=(0, np.inf), x_scale=scales,
                            callback=callback)
    else:
        res = least_squares(residual, x0, args=(I_data, B1, B2, B3, B4, B5, B6, t), bounds=(0, np.inf),
                            callback=callback)

    return {
        "alpha_list": np.array(alpha_list),
        "sigma_list": np.array(sigma_list),
        "gamma_list": np.array(gamma_list),
        "S0_list": np.array(S0_list),
        "E0_list": np.array(E0_list),
        "estimated": res.x,
        "true": (alpha, sigma, gamma, S0, E0),
        "t": t,
        "initial_guess": x0,
        "I_data": I_data,
    }


def plot_results(result):
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    # Parameters to plot (list_key, true_index, title, color)
    params = [
        ('alpha_list', 0, "Alpha", 'orange'),
        ('sigma_list', 1, "Sigma", 'purple'),
        ('gamma_list', 2, "Gamma", 'green'),
        ('S0_list', 3, "S0", 'blue'),
        ('E0_list', 4, "E0", 'red')
    ]

    for i, (key, idx, title, col) in enumerate(params):
        axes[i].plot(result[key], label=f"Est {title}", marker='o', color=col, markersize=4)
        axes[i].axhline(y=result['true'][idx], label="True", color='black', linestyle='--')
        axes[i].set_title(f"{title} Convergence")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_three_results(result, result_, result__):
    # 3 Methods (Rows) x 5 Parameters (Columns)
    fig, axes = plt.subplots(3, 5, figsize=(22, 12))

    methods = [
        (result, "No Rescale"),
        (result_, "Rescale"),
        (result__, "Mul Method")
    ]

    param_configs = [
        ('alpha_list', 0, "Alpha", 'orange'),
        ('sigma_list', 1, "Sigma", 'purple'),
        ('gamma_list', 2, "Gamma", 'green'),
        ('S0_list', 3, "S0", 'blue'),
        ('E0_list', 4, "E0", 'red')
    ]

    for i, (data, name) in enumerate(methods):
        for j, (key, idx, title, col) in enumerate(param_configs):
            axes[i, j].plot(data[key], label=f"Est {title}", marker='o', color=col, markersize=4)
            axes[i, j].axhline(y=data['true'][idx], label="True", color='black', linestyle='--')
            axes[i, j].set_title(f"{name}: {title}")
            axes[i, j].set_xlabel("Iterations")
            axes[i, j].legend()
            axes[i, j].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def print_results(result):
    alpha_t, sigma_t, gamma_t, S0_t, E0_t = result["true"]
    alpha_e, sigma_e, gamma_e, S0_e, E0_e = result["estimated"]
    alpha_i, sigma_i, gamma_i, S0_i, E0_i = result["initial_guess"]

    def get_err(est, true):
        return abs((est - true) / true) * 100 if true != 0 else np.nan

    est_arr = np.array(result['estimated'])
    true_arr = np.array(result['true'])
    cost = np.sum((est_arr - true_arr) ** 2)
    iterations = len(result["alpha_list"])

    print("=" * 82)
    print(f"{'SEIR OPTIMIZATION RESULTS':^82}")
    print("=" * 82)
    print(f"Iterations: {iterations} | Final L2 Parameter Cost: {cost:.6f}")

    param_labels = ["ALPHA", "SIGMA", "GAMMA", "S0", "E0"]
    trues = [alpha_t, sigma_t, gamma_t, S0_t, E0_t]
    inits = [alpha_i, sigma_i, gamma_i, S0_i, E0_i]
    ests = [alpha_e, sigma_e, gamma_e, S0_e, E0_e]

    for label, t, i, e in zip(param_labels, trues, inits, ests):
        print(f"\n--- {label} ---")
        print(f"True:      {t}")
        print(f"Initial:   {i}")
        print(f"Estimated: {e}")
        print(f"Error (%): {get_err(e, t):.4f}%")
