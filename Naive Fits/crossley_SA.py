import os
import time
from scipy.integrate import solve_ivp
from parameters import gamma, today
from SALib.sample import sobol as sobol_sam
from SALib.analyze import sobol as sobol_ana
import numpy as np
import matplotlib.pyplot as plt


# parameters as defined in Crossley et al.
spacing = 0.1  # grid spacing, see \Delta in Appendix B of Crossley et al.
L = 30  # 100  # length of the domain
I = int(L / spacing)  # number of grid points
alpha = 5  # length of the domain that is initially occupied by cells
D_cells = 0.5  # diffusivity of tumour cells in the absence of ECM
# M = 1  # ECM density that inhibits all tumour cell movement
K = 1  # carrying capacity

problem = {
    'num_vars': 3,
    'names': ['ECM density', 'proliferation probability', 'ECM degeneration'],
    'bounds': [[0.2, 0.8],
               [0.00005, 0.4],  # [0.0001, 0.6],
               [0.2, 100]]  # [0.5, 100]]
}


def crossleys_model(T, m_0, r, k):
    """
    Runs Crossley's model for a choice of parameters.
    :param T: time for which the PDE solution should be returned
    :param m_0: ECM density
    :param r: proliferation probability
    :param k: ECM degradation rate
    :return:
    """

    def crossley_parameterized_ode(t, vals):
        """
        This is the space-discretized version of Crossley's parameterized model, derived from equations (7) and (10) in
        her paper, using the discretization scheme as in equation (40).
        :param t: a placeholder, necessary for the calling signature of scipy.integrate.solve_ivp.
        :param vals: the discretized values of u and m of the previous time step. This has to be an array of length 2*I
        (= number of equations).
        :return: [u_0, u_1, ..., u_{I-1}, m_0, m_1, ..., m_{I-1}]
        """
        u = vals[0:I]
        m = vals[I:2 * I]
        u_out = np.zeros(I)
        # ORIGINAL DISCRETIZATION
        # u_out[0] = 2 * (u[1] - u[0]) + u[0] * (1 - u[0] - m[0])
        # u_out[I - 1] = 2 * (u[I - 2] - u[I - 1]) + u[I - 1] * (1 - u[I - 1] - m[I - 1])
        # u_i is build from [u_{i-1}, u_i, u_{i+1}], and m_i from [m_{i-1}, m_i, m_{i+1}]
        # example: d u_1 / d t is build with u_0, u_1 and u_2, i.e. u[0], u[1] and u[2]
        # u_out[1:I - 1] = D_cells / (2 * spacing * spacing) * (u[0:I - 2] * (1 - m[1:I - 1]/K) + u[1:I - 1] * (m[2:I]/K + m[0:I - 2]/K - 2) + u[2:I] * (1 - m[1:I - 1]/K)) + r * u[1:I - 1] * (1 - (u[1:I - 1] + m[1:I - 1])/K)
        # CORRECT DISCRETIZATION
        u_out[0] = D_cells / (spacing * spacing) * (u[1] - u[0]) + u[0] * (1 - u[0] - m[0])
        u_out[I - 1] = D_cells / (spacing * spacing) * (u[I - 2] - u[I - 1] - u[I - 2] * m[I - 1] + u[I - 1] * m[I - 2]) + u[I - 1] * (1 - u[I - 1] - m[I - 1])
        u_out[1:I - 1] = D_cells / (spacing * spacing) * (
                    u[0:I - 2] * (1 - m[1:I - 1] / K) + u[1:I - 1] * (m[2:I] / K + m[0:I - 2] / K - 2) + u[2:I] * (
                        1 - m[1:I - 1] / K)) + r * u[1:I - 1] * (1 - (u[1:I - 1] + m[1:I - 1]) / K)
        m_out = -k * m * u
        return np.ravel([u_out, m_out])

    # define the initial values
    u0 = np.zeros(I)
    m0 = np.zeros(I)
    for x in range(I):
        if x * spacing < alpha:  # u_0 (x) = 1 for x < alpha, 0 else
            u0[x] = 1
        if x * spacing >= alpha:  # m_0 (x) = m_0 for x >= alpha, 0 else
            m0[x] = m_0

    init_vals = np.ravel([u0, m0])

    # solve the space-discretized system
    sol = solve_ivp(crossley_parameterized_ode, [0, T], init_vals)

    # solution at t = T
    timepoints = sol.t
    end_t = len(timepoints) - 1

    return sol.y[:, end_t]


def generate_param_values():
    param_values = sobol_sam.sample(problem, N)  # N*(2*num_vars + 2) by num_vars matrix of parameter combinations

    # save param_values
    filename = os.path.join("SA_RESULTS_PDEs/CROSSLEY", f"param_values_{today}")
    np.savez(filename, N=N, param_values=param_values)


def run_model_SA():
    """
    Runs the model for the sensitivity analysis and saves the results in the folder SA_RESULTS.
    """
    print("PARAMETERIZED CROSSLEY",
          f"Starting to compute {N * (2 * 3 + 2)} model runs. Results are stored in SA_RESULTS_PDEs/CROSSLEY/model_results_{today}_t={T}.npz")

    # load parameter values
    param_file = np.load(os.path.join("SA_RESULTS_PDEs/CROSSLEY", f"param_values_{today}.npz"))
    param_values = param_file['param_values']

    model_results = np.zeros([param_values.shape[0], 2 * I])  # model outputs are saved here

    # CODE WITHOUT MULTIPROCESSING
    mean_time_per_iteration = None
    for i, X in enumerate(param_values):  # each row of model_results is the result of one run of the model with the given parameter combination
        start = time.time()
        model_results[i, :] = crossleys_model(T, *X)
        end = time.time()
        mean_time_per_iteration = (end - start) * (
                    1 - gamma) + mean_time_per_iteration * gamma if mean_time_per_iteration else end - start
        print('model run nr. {} took {}mins -> end approx. {}\n'.format(i + 1, np.round((end - start) / 60, 2),
                                                                        time.ctime(
                                                                            time.time() + mean_time_per_iteration * (
                                                                                        N * (2 * 3 + 2) - (
                                                                                            i + 1) - 1))))

    # save model_results
    filename = os.path.join("SA_RESULTS_PDEs/CROSSLEY", f"model_results_{today}_t={T}")
    np.savez(filename, N=N, t=T, model_results=model_results)


def run_analysis_SA(model_results):
    model_results_cells = model_results[:, :I]
    model_results_ecm = model_results[:, I:]
    sobol_indices_cells = [sobol_ana.analyze(problem, y) for y in model_results_cells.T]
    sobol_indices_ecm = [sobol_ana.analyze(problem, y) for y in model_results_ecm.T]

    # save the analysis
    filename = os.path.join("SA_RESULTS_PDEs/CROSSLEY", f"sobol_indices_{today}_t={T}")
    np.savez(filename, N=N, t=T, sobol_indices_cells=sobol_indices_cells, sobol_indices_ecm=sobol_indices_ecm)


def plot_S1(S1s_cells, S1s_ecm):
    x = np.linspace(0, L, I)
    fig = plt.figure()
    gs = fig.add_gridspec(3, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])

    for i, ax in enumerate([ax1, ax2, ax3]):
        # making Nan values 0
        # for j, s1 in enumerate(S1s_cells[:, i]):
        #     if np.isnan(s1):
        #         S1s_cells[:, i][j] = 0
        ax.plot(x, S1s_cells[:, i], label=r'S1$_\mathregular{{{}}}$'.format(problem["names"][i]))
        ax.set_xlabel("x")
        ax.set_ylabel("First-order Sobol index (cells)")

        ax.set_ylim(0, 1.04)

        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()

        ax.legend(loc='upper right')

    ax4 = fig.add_subplot(gs[0, 1])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[2, 1])

    for i, ax in enumerate([ax4, ax5, ax6]):
        # making Nan values 0
        # for j, s1 in enumerate(S1s_ecm[:, i]):
        #     if np.isnan(s1):
        #         S1s_ecm[:, i][j] = 0
        ax.plot(x, S1s_ecm[:, i], label=r'S1$_\mathregular{{{}}}$'.format(problem["names"][i]))
        ax.set_xlabel("x")
        ax.set_ylabel("First-order Sobol index (ECM)")

        ax.set_ylim(0, 1.04)

        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()

        ax.legend(loc='upper right')
    fig.suptitle(f"Parameterized Crossley: First-order Sobol indices at t={T} (left: cells, right: ECM)\n Parameter ranges: {problem['names'][0]}: {problem['bounds'][0]}, {problem['names'][1]}: {problem['bounds'][1]}, {problem['names'][2]}: {problem['bounds'][2]}")
    plt.show()


def plot_indices(indices_cells, indices_ecm, type_of_index, index_abbrev, S1_conf_cells=None, S1_conf_ecm=None, make_NaN_zero=False):
    """
    Possible values for index_abbrev: "S1", "S2", "ST". If S1 is supposed to be plotted, the 95% confidence intervals
    can be handed to the plot function as well.
    """
    x = np.linspace(0, L, I)
    fig = plt.figure()
    gs = fig.add_gridspec(3, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])

    s2_indexing = [[0, 0, 1], [1, 2, 2],
                   ['ECM density - proliferation probability', 'ECM density - ECM degeneration',
                    'proliferation probability - ECM degeneration']]

    if index_abbrev == "S1":  # prepare data for the confidence intervals
        if S1_conf_cells is not None:
            cellconfs1 = []
            cellconfs2 = []
            cellconfs3 = []
            for param, conflist in enumerate([cellconfs1, cellconfs2, cellconfs3]):
                for i in range(I):
                    if not np.isnan(indices_cells[i, param]):
                        temp = [indices_cells[i, param] - S1_conf_cells[i, param],
                                indices_cells[i, param] + S1_conf_cells[i, param]]
                    elif make_NaN_zero:
                        temp = [0, 0]
                    else:
                        temp = [indices_cells[i, param], indices_cells[i, param]]
                    conflist.append(temp)
            cellconfs = [cellconfs1, cellconfs2, cellconfs3]
        if S1_conf_ecm is not None:
            ecmconfs1 = []
            ecmconfs2 = []
            ecmconfs3 = []
            for param, conflist in enumerate([ecmconfs1, ecmconfs2, ecmconfs3]):
                for i in range(I):
                    if not np.isnan(indices_ecm[i, param]):
                        temp = [indices_ecm[i, param] - S1_conf_ecm[i, param],
                                indices_ecm[i, param] + S1_conf_ecm[i, param]]
                    elif make_NaN_zero:
                        temp = [0, 0]
                    else:
                        temp = [indices_ecm[i, param], indices_ecm[i, param]]
                    conflist.append(temp)
            ecmconfs = [ecmconfs1, ecmconfs2, ecmconfs3]

    for i, ax in enumerate([ax1, ax2, ax3]):
        # making Nan values 0
        if make_NaN_zero:
            for j, s in enumerate(indices_cells[:, i]):
                if np.isnan(s):
                    indices_cells[:, i][j] = 0
        if index_abbrev == "S2":
            # in case of the 2nd order Sobol indices, indices_cells is a 300x3x3 array, where the interactions are saved as
            # follows (with x1 = 'ECM density', x2 = 'proliferation probability', x3 = 'ECM degeneration'):
            # indices_cells[t, 0, 1]: interaction of x1 and x2 at time t
            # indices_cells[t, 0, 2]: interaction of x1 and x3 at time t
            # indices_cells[t, 1, 2]: interaction of x2 and x3 at time t
            ax.plot(x, indices_cells[:, s2_indexing[0][i], s2_indexing[1][i]], label=r'{}$_\mathregular{{{}}}$'.format(index_abbrev, s2_indexing[2][i]))
        else:
            ax.plot(x, indices_cells[:, i], label=r'{}$_\mathregular{{{}}}$'.format(index_abbrev, problem["names"][i]))

        if index_abbrev == "S1":
            if S1_conf_cells is not None:
                all_lower, all_upper = map(list, zip(*cellconfs[i]))
                # moving average for smoothing
                w = 1  # window length
                all_lower = np.convolve(all_lower, np.ones(w), 'same') / w
                all_upper = np.convolve(all_upper, np.ones(w), 'same') / w
                ax.fill_between(x, all_lower, all_upper, alpha=0.5)

        ax.set_xlabel("x")
        ax.set_ylabel(f"{type_of_index} Sobol index (cells)")

        ax.set_ylim(0, 1.04)

        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()

        ax.legend(loc='upper right')

    ax4 = fig.add_subplot(gs[0, 1])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[2, 1])

    for i, ax in enumerate([ax4, ax5, ax6]):
        # making Nan values 0
        if make_NaN_zero:
            for j, s in enumerate(indices_ecm[:, i]):
                if np.isnan(s):
                    indices_ecm[:, i][j] = 0
        if index_abbrev == "S2":
            ax.plot(x, indices_ecm[:, s2_indexing[0][i], s2_indexing[1][i]], label=r'{}$_\mathregular{{{}}}$'.format(index_abbrev, s2_indexing[2][i]))
        else:
            ax.plot(x, indices_ecm[:, i], label=r'{}$_\mathregular{{{}}}$'.format(index_abbrev, problem["names"][i]))

        if index_abbrev == "S1":
            if S1_conf_ecm is not None:
                all_lower, all_upper = map(list, zip(*ecmconfs[i]))
                # moving average for smoothing
                w = 1  # window length
                all_lower = np.convolve(all_lower, np.ones(w), 'same') / w
                all_upper = np.convolve(all_upper, np.ones(w), 'same') / w
                ax.fill_between(x, all_lower, all_upper, alpha=0.5)

        ax.set_xlabel("x")
        ax.set_ylabel(f"{type_of_index} Sobol index (ECM)")

        ax.set_ylim(0, 1.04)

        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()

        ax.legend(loc='upper right')
    fig.suptitle(f"Parameterized Crossley: {type_of_index} Sobol indices at t={T} (left: cells, right: ECM)\n Parameter ranges: {problem['names'][0]}: {problem['bounds'][0]}, {problem['names'][1]}: {problem['bounds'][1]}, {problem['names'][2]}: {problem['bounds'][2]}")
    plt.show()


if __name__ == "__main__":
    T = 10
    N = 2048  # N*(2*num_vars + 2) is the number of samples; N needs to be a power of 2
    # generate parameter values
    # nur auskommentieren wenn man wirklich neue Parameter möchte!!! generate_param_values()
    # run model
    run_model_SA()
    # load results for the SA
    r = np.load(os.path.join("SA_RESULTS_PDEs/CROSSLEY", f"model_results_{today}_t={T}.npz"))
    mr = r['model_results']
    run_analysis_SA(mr)

    # create plots
    s_load = np.load(os.path.join("SA_RESULTS_PDEs/CROSSLEY", f"sobol_indices_{today}_t={T}.npz"), allow_pickle=True)
    s_cells = s_load['sobol_indices_cells']
    s_ecm = s_load['sobol_indices_ecm']
    S1s_cells = np.array([s['S1'] for s in s_cells])
    S1s_ecm = np.array([s['S1'] for s in s_ecm])
    S1_confs_cells = np.array([s['S1_conf'] for s in s_cells])
    S1_confs_ecm = np.array([s['S1_conf'] for s in s_ecm])

    # plot_S1(S1s_cells, S1s_ecm)
    plot_indices(S1s_cells, S1s_ecm, "first-order", "S1", S1_confs_cells, S1_confs_ecm, make_NaN_zero=False)