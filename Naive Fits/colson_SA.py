import os
import time
from scipy.integrate import solve_ivp
from parameters import gamma, today
from SALib.sample import sobol as sobol_sam
from SALib.analyze import sobol as sobol_ana
import numpy as np
import matplotlib.pyplot as plt


# parameters as defined in Colson et al. (with discretization parameters from Crossley et al.)
spacing = 0.1  # grid spacing, see \Delta in Appendix B of Crossley et al.
L = 30  # 100  # length of the domain
I = int(L / spacing)  # number of grid points
sigma = 5  # length of the domain that is initially occupied by cells
omega = 1  # sharpness of the initial boundary between the tumour and healthy tissue
D_cells = 0.5  # diffusivity of tumour cells in the absence of ECM
M = 1  # ECM density that inhibits all tumour cell movement
K = 1  # carrying capacity

# today = datetime.date.today()

problem = {
    'num_vars': 3,
    'names': ['ECM density', 'proliferation probability', 'ECM degeneration'],
    'bounds': [[0.2, 0.8],
               [0.00005, 0.1],  # [0.0001, 0.1],
               [0.2, 100]]  # [0.5, 100]]
}


def colsons_model(T, m_0, r, k):
    """
    Runs Colson's model for a choice of parameters.
    :param T: time for which the PDE solution should be returned
    :param m_0: ECM density
    :param r: growth rate
    :param k: ECM degradation rate
    :return:
    """

    def colson_parameterized_ode(t, vals):
        """
        This is the space-discretized version of Colson's model, derived from equation (1.2) in her paper. The same
        discretization scheme as in Crossley et al., equation (40), is used.
        :param t: a placeholder, necessary for the calling signature of scipy.integrate.solve_ivp.
        :param vals: the discretized values of u and m of the previous time step. This has to be an array of length 2*I
        (= number of equations).
        :return: [u_0, u_1, ..., u_{I-1}, m_0, m_1, ..., m_{I-1}]
        """
        u = vals[0:I]
        m = vals[I:2 * I]
        u_out = np.zeros(I)
        # DISCRETIZATION FROM THE PAPER (WRONG)
        # u_out[0] = 2 * (u[1] - u[0]) + u[0] * (1 - u[0] - m[0])
        # u_out[I - 1] = 2 * (u[I - 2] - u[I - 1]) + u[I - 1] * (1 - u[I - 1] - m[I - 1])
        # u_i is build from [u_{i-1}, u_i, u_{i+1}], and m_i from [m_{i-1}, m_i, m_{i+1}]
        # example: d u_1 / d t is build with u_0, u_1 and u_2, i.e. u[0], u[1] and u[2]
        # u_out[1:I - 1] = D_cells / (2 * spacing * spacing) * ((2 - m[0:I - 2] / M - m[1:I - 1] / M) * u[0:I - 2] - (4 - m[0:I - 2] / M - 2 * m[1:I - 1] / M - m[2:I] / M) * u[1:I - 1] + (2 - m[1:I - 1] / M - m[2:I] / M) * u[2:I]) + u[1:I - 1] * r * (1 - u[1:I - 1] / K)
        # DISCRETIZATION CORRECTED
        u_out[0] = D_cells / (spacing * spacing) * (u[1] - u[0]) + u[0] * (1 - u[0] - m[0])
        u_out[I - 1] = D_cells / (spacing * spacing) * (u[I - 2] - u[I - 1] - u[I - 2] * m[I - 1] + u[I - 1] * m[I - 2]) + u[
            I - 1] * (1 - u[I - 1] - m[I - 1])
        u_out[1:I - 1] = D_cells / (spacing * spacing) * ((2 - m[0:I - 2] / M - m[1:I - 1] / M) * u[0:I - 2] - (
                    4 - m[0:I - 2] / M - 2 * m[1:I - 1] / M - m[2:I] / M) * u[1:I - 1] + (
                                                                      2 - m[1:I - 1] / M - m[2:I] / M) * u[2:I]) + u[
                                                                                                                   1:I - 1] * r * (
                                     1 - u[1:I - 1] / K)
        m_out = -k * m * u
        return np.ravel([u_out, m_out])

    # define the initial values (see equation (4.1) in Colson et al.)
    u0 = np.zeros(I)
    m0 = np.zeros(I)
    for x in range(I):
        if x * spacing < sigma - omega:
            u0[x] = 1
        elif x * spacing < sigma:
            u0[x] = np.exp(1 - 1 / (1 - ((x * spacing - sigma + omega) / omega) ** 2))
            m0[x] = m_0 * (1 - u0[x])
        if x * spacing >= sigma:
            m0[x] = m_0

    init_vals = np.ravel([u0, m0])

    # solve the space-discretized system
    sol = solve_ivp(colson_parameterized_ode, [0, T], init_vals)

    # solution at t = T
    timepoints = sol.t
    end_t = len(timepoints) - 1

    return sol.y[:, end_t]


def generate_param_values():
    param_values = sobol_sam.sample(problem, N)  # N*(2*num_vars + 2) by num_vars matrix of parameter combinations

    # save param_values
    filename = os.path.join("SA_RESULTS_PDEs/COLSON", f"param_values_{today}")
    np.savez(filename, N=N, param_values=param_values)

def run_model_SA():
    """
    Runs the model for the sensitivity analysis and saves the results in the folder SA_RESULTS.
    """
    print("PARAMETERIZED COLSON",
        f"Starting to compute {N * (2 * 3 + 2)} model runs. Results are stored in SA_RESULTS_PDEs/COLSON/model_results_{today}_t={T}.npz")

    # load parameter values
    param_file = np.load(os.path.join("SA_RESULTS_PDEs/COLSON", f"param_values_{today}.npz"))
    param_values = param_file['param_values']
    model_results = np.zeros([param_values.shape[0], 2 * I])  # model outputs are saved here

    # CODE WITHOUT MULTIPROCESSING
    mean_time_per_iteration = None
    for i, X in enumerate(param_values):  # each row of model_results is the result of one run of the model with the given parameter combination
        start = time.time()
        model_results[i, :] = colsons_model(T, *X)
        end = time.time()
        mean_time_per_iteration = (end - start) * (
                    1 - gamma) + mean_time_per_iteration * gamma if mean_time_per_iteration else end - start
        print('model run nr. {} took {}mins -> end approx. {}\n'.format(i + 1, np.round((end - start) / 60, 2),
                                                                        time.ctime(
                                                                            time.time() + mean_time_per_iteration * (
                                                                                        N * (2 * 3 + 2) - (
                                                                                            i + 1) - 1))))

    # save model_results
    filename = os.path.join("SA_RESULTS_PDEs/COLSON", f"model_results_{today}_t={T}")
    np.savez(filename, N=N, t=T, model_results=model_results)
    # can be loaded with r = np.load(os.path.join("SA_RESULTS", f"model_results_{today}.npz")), where
    # r['model_results'], r['N'] and r['t'] yield the variables


def run_analysis_SA(model_results):
    model_results_cells = model_results[:, :I]
    model_results_ecm = model_results[:, I:]
    sobol_indices_cells = [sobol_ana.analyze(problem, y) for y in model_results_cells.T]
    sobol_indices_ecm = [sobol_ana.analyze(problem, y) for y in model_results_ecm.T]

    # save the analysis
    filename = os.path.join("SA_RESULTS_PDEs/COLSON", f"sobol_indices_{today}_t={T}")
    np.savez(filename, N=N, t=T, sobol_indices_cells=sobol_indices_cells, sobol_indices_ecm=sobol_indices_ecm)


def plot_indices(indices_cells, indices_ecm, type_of_index, index_abbrev):
    """
    Possible values for index_abbrev: "S1", "S2", "ST".
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

    for i, ax in enumerate([ax1, ax2, ax3]):
        # making Nan values 0
        # for j, s in enumerate(indices_cells[:, i]):
        #     if np.isnan(s):
        #         indices_cells[:, i][j] = 0
        if index_abbrev == "S2":
            # in case of the 2nd order Sobol indices, indices_cells is a 300x3x3 array, where the interactions are saved as
            # follows (with x1 = 'ECM density', x2 = 'proliferation probability', x3 = 'ECM degeneration'):
            # indices_cells[t, 0, 1]: interaction of x1 and x2 at time t
            # indices_cells[t, 0, 2]: interaction of x1 and x3 at time t
            # indices_cells[t, 1, 2]: interaction of x2 and x3 at time t
            ax.plot(x, indices_cells[:, s2_indexing[0][i], s2_indexing[1][i]], label=r'{}$_\mathregular{{{}}}$'.format(index_abbrev, s2_indexing[2][i]))
        else:
            ax.plot(x, indices_cells[:, i], label=r'{}$_\mathregular{{{}}}$'.format(index_abbrev, problem["names"][i]))
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
        # for j, s in enumerate(indices_ecm[:, i]):
        #     if np.isnan(s):
        #         indices_ecm[:, i][j] = 0
        if index_abbrev == "S2":
            ax.plot(x, indices_ecm[:, s2_indexing[0][i], s2_indexing[1][i]], label=r'{}$_\mathregular{{{}}}$'.format(index_abbrev, s2_indexing[2][i]))
        else:
            ax.plot(x, indices_ecm[:, i], label=r'{}$_\mathregular{{{}}}$'.format(index_abbrev, problem["names"][i]))
        ax.set_xlabel("x")
        ax.set_ylabel(f"{type_of_index} Sobol index (ECM)")

        ax.set_ylim(0, 1.04)

        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()

        ax.legend(loc='upper right')
    fig.suptitle(f"Parameterized Colson: {type_of_index} Sobol indices at t={T} (left: cells, right: ECM)\n Parameter ranges: {problem['names'][0]}: {problem['bounds'][0]}, {problem['names'][1]}: {problem['bounds'][1]}, {problem['names'][2]}: {problem['bounds'][2]}")
    plt.show()


if __name__ == "__main__":
    N = 2048  # N*(2*num_vars + 2) is the number of samples; N needs to be a power of 2
    T = 10
    # generate parameter values
    ####### only unucomment if you really need new parameters # generate_param_values()
    # run model
    run_model_SA()
    # # load results for the SA
    r = np.load(os.path.join("SA_RESULTS_PDEs/COLSON", f"model_results_{today}_t={T}.npz"))
    mr = r['model_results']
    run_analysis_SA(mr)

    # create plots
    s_load = np.load(os.path.join("SA_RESULTS_PDEs/COLSON", f"sobol_indices_{today}_t={T}.npz"), allow_pickle=True)
    s_cells = s_load['sobol_indices_cells']
    s_ecm = s_load['sobol_indices_ecm']
    S1s_cells = np.array([s['S1'] for s in s_cells])
    S1s_ecm = np.array([s['S1'] for s in s_ecm])
    S2s_cells = np.array([s['S2'] for s in s_cells])
    S2s_ecm = np.array([s['S2'] for s in s_ecm])
    STs_cells = np.array([s['ST'] for s in s_cells])
    STs_ecm = np.array([s['ST'] for s in s_ecm])
    # check weird S1-values
    # S1confs_ecm = np.array([s["S1_conf"] for s in s_ecm])
    # cellconfs1 = []
    # cellconfs2 = []
    # cellconfs3 = []
    # for param, conflist in enumerate([cellconfs1, cellconfs2, cellconfs3]):
    #     for i in range(I):
    #         if not np.isnan(S1s_ecm[i, param]):
    #             temp = [S1s_ecm[i, param] - S1confs_ecm[i, param], S1s_ecm[i, param] + S1confs_ecm[i, param]]
    #             conflist.append(temp)
    # print(cellconfs1)
    plot_indices(S1s_cells, S1s_ecm, "First-order", "S1")
    plot_indices(STs_cells, STs_ecm, "Total", "ST")
    plot_indices(S2s_cells, S2s_ecm, "Second-order", "S2")
