import os
import numpy as np
from plots import oneD_output
import matplotlib.pyplot as plt
import matplotlib


problem = {'names': ['ECM density', 'ECM degeneration', 'proliferation probability']}


def get_confidence_intervals(indices, confs, index_abbrev, make_NaN_zero):
    confs1 = []
    confs2 = []
    confs3 = []
    domain_length = indices.shape[0]
    s2_indexing = [[0, 0, 1], [1, 2, 2]]
    for param, conflist in enumerate([confs1, confs2, confs3]):
        for i in range(domain_length):
            # in the case of S2 the arrays are built differently
            # confs[t, 0, 1]: interaction of x1 and x2 at time t
            # confs[t, 0, 2]: interaction of x1 and x3 at time t
            # confs[t, 1, 2]: interaction of x2 and x3 at time t
            if index_abbrev == "S2":
                if not np.isnan(indices[i, s2_indexing[0][param], s2_indexing[1][param]]):
                    temp = [indices[i, s2_indexing[0][param], s2_indexing[1][param]] - confs[i, s2_indexing[0][param], s2_indexing[1][param]],
                            indices[i, s2_indexing[0][param], s2_indexing[1][param]] + confs[i, s2_indexing[0][param], s2_indexing[1][param]]]
                elif make_NaN_zero:
                    temp = [0, 0]
                else:
                    temp = [-confs[i, s2_indexing[0][param], s2_indexing[1][param]], confs[i, s2_indexing[0][param], s2_indexing[1][param]]]
            else:
                if not np.isnan(indices[i, param]):
                    temp = [indices[i, param] - confs[i, param],
                            indices[i, param] + confs[i, param]]
                elif make_NaN_zero:
                    temp = [0, 0]
                else:
                    # temp = [indices[i, param], indices[i, param]]
                    temp = [-confs[i, param], confs[i, param]]
            conflist.append(temp)
    return [confs1, confs2, confs3]


def plot_all_models_indices(cpm_fem_indexfile, colson_indexfile, crossley_indexfile, index_abbrev, type_of_index, t, make_NaN_zero=False):
    """
    The index files should be saved in the same folder.
    """
    all_models_cell_indices = []
    all_models_ecm_indices = []
    all_models_cell_confs = []
    all_models_ecm_confs = []
    for file in [cpm_fem_indexfile, colson_indexfile, crossley_indexfile]:
        s_load = np.load(os.path.join("DATA_FOR_PLOTS", f"{file}.npz"), allow_pickle=True)
        s_cells = s_load['sobol_indices_cells']
        s_ecm = s_load['sobol_indices_ecm']
        indices_cells = np.array([s[index_abbrev] for s in s_cells])
        indices_ecm = np.array([s[index_abbrev] for s in s_ecm])
        key = index_abbrev + "_conf"
        conf_cells = np.array([s[key] for s in s_cells])
        conf_ecm = np.array([s[key] for s in s_ecm])
        conf_ints_cells = get_confidence_intervals(indices_cells, conf_cells, index_abbrev, make_NaN_zero)
        conf_ints_ecm = get_confidence_intervals(indices_ecm, conf_ecm, index_abbrev, make_NaN_zero)
        all_models_cell_indices.append(indices_cells)
        all_models_ecm_indices.append(indices_ecm)
        all_models_cell_confs.append(conf_ints_cells)
        all_models_ecm_confs.append(conf_ints_ecm)

    plt.rcParams.update({'font.size': 12})

    fig = plt.figure()
    gs = fig.add_gridspec(3, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])

    s2_indexing = [[0, 0, 1], [1, 2, 2],
                   ['ECM density - proliferation probability', 'ECM density - ECM degeneration',
                    'proliferation probability - ECM degeneration']]

    model_names = ["CPM-FEM", "Colson", "Crossley"]
    colors = ['#d6cbd3', '#bdcebe', '#ada397']

    for i, ax in enumerate([ax1, ax2, ax3]):
        for model in [0, 1, 2]:
            indices_cells = all_models_cell_indices[model]
            conf_cells = all_models_cell_confs[model]
            if model == 0:
                domain_length = indices_cells.shape[0]
                x = range(domain_length)
            else:
                spacing = 0.1  # grid spacing, see \Delta in Appendix B of Crossley et al.
                L = 29  # 100  # length of the domain
                I = int(L / spacing)  # number of grid points
                x = np.linspace(0, L, I)
                indices_cells = indices_cells[:-10, :]
            # making Nan values 0
            if make_NaN_zero:
                for j, s in enumerate(indices_cells[:, i]):
                    if np.isnan(s[j]):
                        indices_cells[:, i][j] = 0
            if index_abbrev == "S2":
                # in case of the 2nd order Sobol indices, indices_cells is a 300x3x3 array, where the interactions are saved as
                # follows (with x1 = 'ECM density', x2 = 'proliferation probability', x3 = 'ECM degeneration'):
                # indices_cells[t, 0, 1]: interaction of x1 and x2 at time t
                # indices_cells[t, 0, 2]: interaction of x1 and x3 at time t
                # indices_cells[t, 1, 2]: interaction of x2 and x3 at time t
                ax.plot(x, indices_cells[:, s2_indexing[0][i], s2_indexing[1][i]], label=f"{model_names[model]}", color=colors[model])
            else:
                ax.plot(x, indices_cells[:, i], label=f"{model_names[model]}", color=colors[model])

            all_lower, all_upper = map(list, zip(*conf_cells[i]))
            # moving average for smoothing
            w = 1  # window length
            all_lower = np.convolve(all_lower, np.ones(w), 'same') / w
            all_upper = np.convolve(all_upper, np.ones(w), 'same') / w
            if model > 0:
                ax.fill_between(x, all_lower[:-10], all_upper[:-10], alpha=0.5, color=colors[model])
            else:
                ax.fill_between(x, all_lower, all_upper, alpha=0.5, color=colors[model])
        if i == 2:
            ax.set_xlabel("x")

        if index_abbrev == "S2":
            ax.set_ylabel(r'{}$_\mathregular{{{}}}$ (cells)'.format(index_abbrev, s2_indexing[2][i]))
        else:
            ax.set_ylabel(r'{}$_\mathregular{{{}}}$ (cells)'.format(index_abbrev, problem["names"][i]))

        ax.set_ylim(-0.04, 1.04)

        if i == 1 and index_abbrev == "S2":
            ax.yaxis.set_label_position("right")

        ax.yaxis.tick_right()
        # ax.legend(loc='upper right')
        if i == 0:
            ax.legend()
        elif i == 1:
            ax.legend()
        else:
            ax.legend()


    ax4 = fig.add_subplot(gs[0, 1])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[2, 1])

    for i, ax in enumerate([ax4, ax5, ax6]):
        for model in [0, 1, 2]:
            indices_ecm = all_models_ecm_indices[model]
            conf_ecm = all_models_ecm_confs[model]
            if model == 0:
                domain_length = indices_ecm.shape[0]
                x = range(domain_length)
            else:
                spacing = 0.1  # grid spacing, see \Delta in Appendix B of Crossley et al.
                L = 29  # 100  # length of the domain
                I = int(L / spacing)  # number of grid points
                x = np.linspace(0, L, I)
                indices_ecm = indices_ecm[:-10, :]
            # making Nan values 0
            if make_NaN_zero:
                for j, s in enumerate(indices_ecm[:, i]):
                    if np.isnan(s[j]):
                        indices_ecm[:, i][j] = 0
            if index_abbrev == "S2":
                # in case of the 2nd order Sobol indices, indices_cells is a 300x3x3 array, where the interactions are saved as
                # follows (with x1 = 'ECM density', x2 = 'proliferation probability', x3 = 'ECM degeneration'):
                # indices_cells[t, 0, 1]: interaction of x1 and x2 at time t
                # indices_cells[t, 0, 2]: interaction of x1 and x3 at time t
                # indices_cells[t, 1, 2]: interaction of x2 and x3 at time t
                ax.plot(x, indices_ecm[:, s2_indexing[0][i], s2_indexing[1][i]], label=f"{model_names[model]}",
                        color=colors[model])
            else:
                ax.plot(x, indices_ecm[:, i], label=f"{model_names[model]}", color=colors[model])

            all_lower, all_upper = map(list, zip(*conf_ecm[i]))
            # moving average for smoothing
            w = 1  # window length
            all_lower = np.convolve(all_lower, np.ones(w), 'same') / w
            all_upper = np.convolve(all_upper, np.ones(w), 'same') / w
            if model > 0:
                ax.fill_between(x, all_lower[:-10], all_upper[:-10], alpha=0.5, color=colors[model])
            else:
                ax.fill_between(x, all_lower, all_upper, alpha=0.5, color=colors[model])

        if i == 2:
            ax.set_xlabel("x")
        if index_abbrev == "S2":
            ax.set_ylabel(r'{}$_\mathregular{{{}}}$ (ECM)'.format(index_abbrev, s2_indexing[2][i]))
        else:
            ax.set_ylabel(r'{}$_\mathregular{{{}}}$ (ECM)'.format(index_abbrev, problem["names"][i]))

        ax.set_ylim(-0.04, 1.04)

        if i == 1 and index_abbrev == "S2":
            ax.yaxis.set_label_position("right")

        ax.yaxis.tick_right()
        if i == 0:
            ax.legend()
        else:
            ax.legend()
    fig.suptitle(
            f"{type_of_index} Sobol indices at t={t} for all models with 95% confidence intervals\n(left: cells, right: ECM)")
    plt.show()
    # savepath = "C:/Users/vroni/Documents/Uni/SS 23/Masterarbeit/Bilder/neue Diskretisierung - SA plots"
    # savename = f"{index_abbrev}_t={t}.pdf"
    # plt.savefig(os.path.join(savepath, savename))


def get_all_indices_and_return_S_123(all_indices):
    S_1 = all_indices[0][:, 0]
    S_2 = all_indices[0][:, 1]
    S_3 = all_indices[0][:, 2]
    indices = all_indices[1]
    S_12 = []
    S_13 = []
    S_23 = []
    domain_length = indices.shape[0]
    s2_indexing = [[0, 0, 1], [1, 2, 2]]
    for param, conflist in enumerate([S_12, S_13, S_23]):
        for i in range(domain_length):
            # in the case of S2 the arrays are built differently
            # confs[t, 0, 1]: interaction of x1 and x2 at time t
            # confs[t, 0, 2]: interaction of x1 and x3 at time t
            # confs[t, 1, 2]: interaction of x2 and x3 at time t
            temp = indices[i, s2_indexing[0][param], s2_indexing[1][param]]
            conflist.append(temp)
    ST_1 = all_indices[2][:, 0]
    ST_2 = all_indices[2][:, 1]
    ST_3 = all_indices[2][:, 2]
    S_123 = []
    all_lower = []
    all_upper = []
    for i in range(domain_length):
        # there are three possibilities to compute S_123, we will use the mean
        c1 = ST_1[i] - S_1[i] - S_12[i] - S_13[i]
        c2 = ST_2[i] - S_2[i] - S_12[i] - S_23[i]
        c3 = ST_3[i] - S_3[i] - S_13[i] - S_23[i]
        mean = np.mean([c1, c2, c3])
        S_123.append(mean)
        std = np.std([c1, c2, c3])
        all_lower.append(mean - std)
        all_upper.append(mean + std)
    return S_123, all_upper, all_lower


def compute_third_order(file, t):
    all_cell_indices = []
    all_ecm_indices = []
    s_load = np.load(os.path.join("DATA_FOR_PLOTS", f"{file}.npz"), allow_pickle=True)
    s_cells = s_load['sobol_indices_cells']
    s_ecm = s_load['sobol_indices_ecm']
    # read indices
    for index_abbrev in ['S1', 'S2', 'ST']:
        indices_cells = np.array([s[index_abbrev] for s in s_cells])
        indices_ecm = np.array([s[index_abbrev] for s in s_ecm])
        all_cell_indices.append(indices_cells)
        all_ecm_indices.append(indices_ecm)
    # compute S_123 for the cells
    S_123c, all_upperc, all_lowerc = get_all_indices_and_return_S_123(all_cell_indices)
    # compute S_123 for the ECM
    S_123e, all_uppere, all_lowere = get_all_indices_and_return_S_123(all_ecm_indices)
    # S_123 = ST_1 - S_1 - S_12 - S_13
    # S_1 = all_cell_indices[0][:, 0]
    # S_2 = all_cell_indices[0][:, 1]
    # S_3 = all_cell_indices[0][:, 2]
    # indices = all_cell_indices[1]
    # S_12 = []
    # S_13 = []
    # S_23 = []
    # domain_length = indices.shape[0]
    # s2_indexing = [[0, 0, 1], [1, 2, 2]]
    # for param, conflist in enumerate([S_12, S_13, S_23]):
    #     for i in range(domain_length):
    #         # in the case of S2 the arrays are built differently
    #         # confs[t, 0, 1]: interaction of x1 and x2 at time t
    #         # confs[t, 0, 2]: interaction of x1 and x3 at time t
    #         # confs[t, 1, 2]: interaction of x2 and x3 at time t
    #         temp = indices[i, s2_indexing[0][param], s2_indexing[1][param]]
    #         conflist.append(temp)
    # ST_1 = all_cell_indices[2][:, 0]
    # ST_2 = all_cell_indices[2][:, 1]
    # ST_3 = all_cell_indices[2][:, 2]
    # S_123 = []
    # all_lower = []
    # all_upper = []
    # for i in range(domain_length):
    #     # there are three possibilities to compute S_123, we will use the mean
    #     c1 = ST_1[i] - S_1[i] - S_12[i] - S_13[i]
    #     c2 = ST_2[i] - S_2[i] - S_12[i] - S_23[i]
    #     c3 = ST_3[i] - S_3[i] - S_13[i] - S_23[i]
    #     mean = np.mean([c1, c2, c3])
    #     S_123.append(mean)
    #     std = np.std([c1, c2, c3])
    #     all_lower.append(mean - std)
    #     all_upper.append(mean + std)

    domain_length = all_cell_indices[1].shape[0]
    x = range(domain_length)
    fig = plt.figure()
    gs = fig.add_gridspec(1, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    lables = ["cells", "ECM"]
    S_123 =[S_123c, S_123e]
    all_lower = [all_lowerc, all_lowere]
    all_upper = [all_upperc, all_uppere]

    for i, ax in enumerate([ax1, ax2]):
        ax.plot(x, S_123[i], color='#eca1a6')
        ax.fill_between(x, all_lower[i], all_upper[i], alpha=0.5, color='#eca1a6')
        ax.set_xlabel("x")

        ax.set_ylabel(r'$S_\mathregular{{123}}$ ({})'.format(lables[i]))
        ax.set_ylim(-0.04, 1.04)
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
    fig.suptitle(
        f"Third-order Sobol indices at t={t} of the CPM-FEM model including one standard deviation\n(left: cells, right: ECM)")
    plt.show()



if __name__ == "__main__":
    matplotlib.rcParams['pdf.fonttype'] = 42

    cpmfem_t10 = "CPM-FEM/sobol_indices_2023-12-20_t=10"
    cpmfem_t20 = "CPM-FEM/sobol_indices_2023-12-20_t=20"
    cpmfem_t30 = "CPM-FEM/sobol_indices_2023-12-20_t=30"

    colson_t10 = "COLSON/N=2048_new_discretization/sobol_indices_2024-01-08_t=10"
    colson_t20 = "COLSON/N=2048_new_discretization/sobol_indices_2024-01-08_t=20"
    colson_t30 = "COLSON/N=2048_new_discretization/sobol_indices_2024-01-08_t=30"

    crossleyN2048_t10 = "CROSSLEY/N=2048_new_discretization/sobol_indices_2024-01-08_t=10"
    crossleyN2048_t20 = "CROSSLEY/N=2048_new_discretization/sobol_indices_2024-01-08_t=20"
    crossleyN2048_t30 = "CROSSLEY/N=2048_new_discretization/sobol_indices_2024-01-08_t=30"

    # crossleyN1024_t10 = "CROSSLEY/N=1024/sobol_indices_2023-12-31_t=10"
    # crossleyN1024_t20 = "CROSSLEY/N=1024/sobol_indices_2023-12-31_t=20"
    # crossleyN1024_t30 = "CROSSLEY/N=1024/sobol_indices_2023-12-31_t=30"

    # compute_third_order(cpmfem_t30, 30)

    plot_all_models_indices(cpmfem_t10, colson_t10, crossleyN2048_t10, 'ST', "Total-order", 10, make_NaN_zero=False)
