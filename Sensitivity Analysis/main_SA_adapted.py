import os
import time
from SALib.sample import sobol as sobol_sam
from SALib.analyze import sobol as sobol_ana
import numpy as np
from parameters import today, celldiam, totalL, gamma
from main import main_model
from plots import oneD_output
import matplotlib.pyplot as plt
import multiprocessing as mp

# SPECIFY THE NUMBER OF PARALLEL PROCESSES (if None, the system's number of CPUs will be used)
parallel_processes = 10

t = 30

problem = {
    'num_vars': 3,
    'names': ['ECMconcentr', 'distanceinfl', 'prolifprob'],
    'bounds': [[0.2, 0.8],
               [0.25 * celldiam, 2 * celldiam],
               [0.0001, 0.01]]  # [0, 0.01]]
}

N = 1024  # N*(2*num_vars + 2) is the number of samples; N needs to be a power of 2


# def generate_samples(number_of_files=1):
#     """
#     Computes the Saltelli samples (Saltelli-modified Sobol sequence) and splits them into the given number of arrays
#     which are then saved as npz-files in the folder SA_RESULTS.
#     :param number_of_files: the number of npz-files in which one wants the parameters to be split (makes sense if one
#     runs the simulation using multiple computers, then each computer can handle one file).
#     """
#     param_values = sobol_sam.sample(problem, N)  # N*(2*num_vars + 2) by num_vars matrix of parameter combinations
#     # add t as a parameter
#     param_values = np.insert(param_values, 0, t, axis=1)
#     # add the runIDs as a parameter
#     total_nr_of_runs = int(param_values.shape[0])
#     runIDs = range(total_nr_of_runs)
#     param_values = np.insert(param_values, 0, runIDs, axis=1)
#
#     # number of parameter combinations that are saved in each file (the files will probably have different lengths)
#     quotient, remainder = divmod(total_nr_of_runs, number_of_files)
#     filesizes = [quotient] * number_of_files
#     for i in range(remainder):
#         filesizes[i] += 1
#
#     assigned_runs = 0
#     for file_nr in range(number_of_files):
#         param_values_current_file = param_values[assigned_runs:(assigned_runs+filesizes[file_nr])]
#         filename = os.path.join("SA_RESULTS", f"param_values_incl_runIDs_{today}_file_nr_{file_nr}")
#         np.savez(filename, N=N, t=t, param_values=param_values_current_file)
#         assigned_runs += filesizes[file_nr]


def run_model_SA(param_file_name, load_ongoing=False, day=None, step=None, L=None):
    """
    Runs the model for the sensitivity analysis and saves the results in the folder SA_RESULTS.
    :param param_file_name: name of the npz-file in which the parameters are saved.
    :param load_ongoing: if set True, an ongoing ensemble of model runs will be continued. The parameters used for this
    ensemble is taken from the file param_values_incl_runIDs_{day}.npz. The old and new results will be saved in
    model_results_{day}.npz. The other parameters need to be specified if load_ongoing is set to True.
    :param day: the day of the start of the model runs. It is used for param_values_incl_runIDs_{day}.npz and
    model_results_{day}.npz.
    :param step: make sure that there is a file concentration{time}.out for each of the runIDs!
    :param L: usually equal to totalL except if there has been a parameter change.
    """
    if load_ongoing:
        if not [x for x in (day, step, L) if x is None]:  # check if all parameters are specified
            # load the parameter values
            param_file = np.load(os.path.join("SA_RESULTS", f"{param_file_name}.npz"))
            param_values = param_file['param_values']
            total_runs_list = param_values[:, 0].tolist()
            total_runs_list = [int(element) for element in total_runs_list]
            lowest_runID = min(total_runs_list)
            total_runs = len(total_runs_list)
            # check which runs are done
            # total_runs = N*(2*3 + 2)
            completed_runs = param_values[:, 0].tolist()
            completed_runs = [int(element) for element in completed_runs]
            for id in total_runs_list:
                if os.path.exists(os.path.join('RESULTS', f'{day}-{id}')):  # folder exists, but run is not done (the
                    # ctags1-condition comes from the issue that sometimes, multiprocessing creates a folder where only
                    # ctags0 is present and no increment_number)
                    if os.path.exists(os.path.join(f'RESULTS/{day}-{id}', 'increment_number.out')) or not os.path.exists(os.path.join(f'RESULTS/{day}-{id}', 'ctags1.out')):
                        completed_runs.remove(id)
                else:  # folder does not exist, i.e. run has not started yet
                    completed_runs.remove(id)
            # check if at least one run was completed
            if len(completed_runs) > 0:
                # load the results for all of the completed runs performed at day
                previous_model_results = load_model_results(day, results_folder="RESULTS", npz_file_ready=False, step=step, range_runIDs=completed_runs, L=L, use_only_specific_IDs=True, lowest_runID=lowest_runID)
                # set up array where all results can be saved
                remainder = np.zeros([total_runs - previous_model_results.shape[0], 2 * L])
                model_results = np.concatenate((previous_model_results, remainder), axis=0)
                # turn param_values into list
                param_list = param_values.tolist()
                # remove the parameter sets whose runs already have been completed
                completed_runs_normed = [item - lowest_runID for item in completed_runs]
                remaining_params = [item for index, item in enumerate(param_list) if index not in completed_runs_normed]
                remaining_runIDs = total_runs_list
                remaining_runIDs = [runID for runID in remaining_runIDs if runID not in completed_runs]
            else:  # no run has been completed so far
                remaining_params = param_values.tolist()
                model_results = np.zeros([param_values.shape[0], 2 * totalL])
                remaining_runIDs = total_runs_list
            # continue the simulations
            print(f"{len(completed_runs)} runs completed. Continuing with the remaining {len(remaining_runIDs)} runs.")
            pool = mp.Pool(parallel_processes)  # the parameter of mp.Pool() defines the number of parallel processes
            results_raw = pool.starmap(main_model, remaining_params)
            for i, result in enumerate(results_raw):
                row = remaining_runIDs[i]
                model_results[row, :] = result
        else:
            print("In the function run_model_SA(), day, next_runID, step, and L need to be specified.")
            return
    else:
        # load the parameter values
        param_file = np.load(os.path.join("SA_RESULTS", f"{param_file_name}.npz"))
        param_values = param_file['param_values']
        nr_runs = param_values.shape[0]
        # model outputs are saved here:
        model_results = np.zeros([nr_runs, 2 * totalL])

        print(f"Starting to compute {nr_runs} model runs. Results are stored in the folders RESULTS/{today}-...")

        # CODE WITHOUT MULTIPROCESSING
        # mean_time_per_iteration = None
        # for i, X in enumerate(param_values):  # each row of model_results is the result of one run of the model with the given parameter combination
        #     start = time.time()
        #     model_results[i, :] = main_model(t, *X)
        #     end = time.time()
        #     mean_time_per_iteration = (end - start) * (
        #                 1 - gamma) + mean_time_per_iteration * gamma if mean_time_per_iteration else end - start
        #     print('model run nr. {} took {}mins -> end approx. {}\n'.format(i + 1, np.round((end - start) / 60, 2),
        #                                                                     time.ctime(
        #                                                                         time.time() + mean_time_per_iteration * (
        #                                                                                     N * (2 * 3 + 2) - (
        #                                                                                         i + 1) - 1))))

        # CODE WITH MULTIPROCESSING
        # convert from np-array to list (necessary for the mapping)
        # each list in param_values looks like [runID_i, t, ECMconcentr_i, distanceinfl_i, prolifprob_i]
        param_list = param_values.tolist()
        # start the parallel processes
        pool = mp.Pool(parallel_processes)  # the parameter of mp.Pool() defines the number of parallel processes
        results_raw = pool.starmap(main_model, param_list)
        for i, result in enumerate(results_raw):
            model_results[i, :] = result

    # save model_results
    filename = os.path.join("SA_RESULTS", f"model_results_{today}")
    np.savez(filename, N=N, t=t, model_results=model_results)
    # can be loaded with r = np.load(os.path.join("SA_RESULTS", f"model_results_{today}.npz")), where
    # r['model_results'], r['N'] and r['t'] yield the variables


def load_model_results(day, step, npz_file_ready=True, results_folder=None, range_runIDs=None, L=None, use_only_specific_IDs=False, lowest_runID=None, movavg=False):
    """
    Loads the model results and returns them in a format that can be processed by run_analysis_SA.
    :param day: the date in the name of the run-folders.
    :param results_folder: the folder in which the npz-file or all results-folders are saved.
    :param npz_file_ready: if the results are already saved as an npz-file in the folder SA_RESULTS, only day needs to
    be specified. If this is not the case, then npz_file_ready needs to be set to False and all parameters need to be
    given.
    :param step: make sure that there is a file concentration{time}.out for each of the runIDs!
    :param range_runIDs: list of the first and the last runID one wishes to load. First value inclusive, last value
    exclusive.
    :param L: usually equal to totalL except if there has been a parameter change.
    :param use_only_specific_IDs: if set True, range_runIDs is not treated as a range but as a list and only the results
    for the IDs in that list are being loaded. The array model_results is of the size (largest run ID) x L, where the
    rows whose index is not among the specified runIDs are filled with zeros.
    :param lowest_runID: lowest run ID in the parameter file.
    :param movavg: if set True, the data will be smoothed using a moving average with window length 3.
    :return:
    """
    if npz_file_ready:
        file = np.load(os.path.join("SA_RESULTS", f"total_model_results_{day}_t={step}.npz"))
        raw_model_results = file['model_results']
        if movavg:
            nrows = raw_model_results.shape[0]
            L = int(raw_model_results.shape[1] / 2)
            w = 3  # window length for the moving average
            # newL = max(L, w) - min(L, w) + 1  # size of cells and ecm after convolving
            model_results = np.zeros([nrows, 2 * L])
            for run in range(nrows):
                cells = raw_model_results[run, :L]
                ecm = raw_model_results[run, L:]
                # cells = np.convolve(cells, np.ones(w), 'valid') / w
                # ecm = np.convolve(ecm, np.ones(w), 'valid') / w
                cells = np.convolve(cells, np.ones(w), 'same') / w
                ecm = np.convolve(ecm, np.ones(w), 'same') / w
                model_results[run, :] = np.ravel(np.array([cells, ecm]))
            return model_results
        else:
            return raw_model_results
    elif not [x for x in (step, results_folder, range_runIDs, L) if x is None]:  # check if all parameters are specified
        if use_only_specific_IDs and lowest_runID:
            nrows = max(range_runIDs) - lowest_runID + 1
            model_results = np.zeros([nrows, 2 * L])
            for run in range_runIDs:
                cells, ecm = oneD_output(f"{results_folder}/{day}-{run}", L, step)
                if movavg:
                    w = 3  # window length
                    cells = np.convolve(cells, np.ones(w), 'valid') / w
                    ecm = np.convolve(ecm, np.ones(w), 'valid') / w
                output = np.ravel(np.array([cells, ecm]))
                model_results[int(run - lowest_runID), :] = output
        else:
            model_results = np.zeros([len(range(range_runIDs[0], range_runIDs[1])), 2 * L])
            for i, run in enumerate(range(range_runIDs[0], range_runIDs[1])):
                cells, ecm = oneD_output(f"{results_folder}/{day}-{run}", L, step)
                if movavg:
                    w = 3  # window length
                    cells = np.convolve(cells, np.ones(w), 'valid') / w
                    ecm = np.convolve(ecm, np.ones(w), 'valid') / w
                output = np.ravel(np.array([cells, ecm]))
                model_results[i, :] = output
        return model_results
    else:
        print("In the function load_model_results(), all parameters need to be specified if npz_file_ready is set to False.")


# FUNCTIONS FOR THE SENSITIVITY ANALYSIS

def run_analysis_SA(model_results, step=t):
    """
    Performs the Sensitivity Analysis for the given model results and the globally defined problem.
    :param model_results: a numpy-array with the results, loaded for instance by load_model_results().
    :param step: the time step within the model at which the sensitivity should be analyzed. It has to be equal to step
    in load_model_results(). If not indicated, the global value t will be used.
    :return: nothing, the result is saved in an npz-file and can be loaded using
    np.load(os.path.join("SA_RESULTS", f"sobol_indices_{today}_t={step}.npz"), allow_pickle=True). Its keys
    "sobol_indices_cells" and "sobol_indices_ecm are a Python dicts themselves with the keys "S1", "S2", "ST",
    "S1_conf", "S2_conf", and "ST_conf".
    """
    domain_length = int(model_results.shape[1] / 2)
    # model_results_cells = model_results[:, :totalL]
    # model_results_ecm = model_results[:, totalL:]
    model_results_cells = model_results[:, :domain_length]
    model_results_ecm = model_results[:, domain_length:]
    sobol_indices_cells = [sobol_ana.analyze(problem, y) for y in model_results_cells.T]
    sobol_indices_ecm = [sobol_ana.analyze(problem, y) for y in model_results_ecm.T]

    # save the analysis
    filename = os.path.join("SA_RESULTS", f"sobol_indices_{today}_t={step}.npz")
    np.savez(filename, N=N, t=step, sobol_indices_cells=sobol_indices_cells, sobol_indices_ecm=sobol_indices_ecm)


def plot_indices(indices_cells, indices_ecm, type_of_index, index_abbrev, step=t):
    """
    Possible values for index_abbrev: "S1", "S2", "ST".
    """
    domain_length = indices_cells.shape[0]
    # x = range(totalL)
    x = range(domain_length)
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

    fig.suptitle(f"CPM-FEM model: {type_of_index} Sobol indices at t={step} (left: cells, right: ECM)\n Parameter ranges: {problem['names'][0]}: {problem['bounds'][0]}, {problem['names'][1]}: {problem['bounds'][1]}, {problem['names'][2]}: {problem['bounds'][2]}")
    plt.show()


def plot_indices2(indices_cells, indices_ecm, type_of_index, index_abbrev, S1_conf_cells=None, S1_conf_ecm=None, make_NaN_zero=False, step=t):
    """
    Possible values for index_abbrev: "S1", "S2", "ST". If S1 is supposed to be plotted, the 95% confidence intervals
    can be handed to the plot function as well.
    """
    domain_length = indices_cells.shape[0]
    # x = range(totalL)
    x = range(domain_length)
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
                for i in range(domain_length):
                    if not np.isnan(indices_cells[i, param]):
                        temp = [indices_cells[i, param] - S1_conf_cells[i, param],
                                indices_cells[i, param] + S1_conf_cells[i, param]]
                    elif make_NaN_zero:
                        temp = [0, 0]
                    else:
                        # temp = [indices_cells[i, param], indices_cells[i, param]]
                        temp = [-S1_conf_cells[i, param], S1_conf_cells[i, param]]
                    conflist.append(temp)
            cellconfs = [cellconfs1, cellconfs2, cellconfs3]
        if S1_conf_ecm is not None:
            ecmconfs1 = []
            ecmconfs2 = []
            ecmconfs3 = []
            for param, conflist in enumerate([ecmconfs1, ecmconfs2, ecmconfs3]):
                for i in range(domain_length):
                    if not np.isnan(indices_ecm[i, param]):
                        temp = [indices_ecm[i, param] - S1_conf_ecm[i, param],
                                indices_ecm[i, param] + S1_conf_ecm[i, param]]
                    elif make_NaN_zero:
                        temp = [0, 0]
                    else:
                        # temp = [indices_ecm[i, param], indices_ecm[i, param]]
                        temp = [-S1_conf_ecm[i, param], S1_conf_ecm[i, param]]
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
    fig.suptitle(f"CPM-FEM model: {type_of_index} Sobol indices at t={step} (left: cells, right: ECM)\n Parameter ranges: {problem['names'][0]}: {problem['bounds'][0]}, {problem['names'][1]}: {problem['bounds'][1]}, {problem['names'][2]}: {problem['bounds'][2]}")
    plt.show()


if __name__ == "__main__":
    # generate_samples(2)

    step = t
    # start new ensemble of model runs - if you need to interrupt the simulations and want to continue at a later time,
    # comment the following line and un-comment the line below that one
    # run_model_SA("param_values_incl_runIDs_2023-12-20_file_nr_0")
    # un-comment this line if you interrupted and now want to continue an ongoing run
    # run_model_SA("param_values_incl_runIDs_2023-12-20_file_nr_1", load_ongoing=True, day=today, step=step, L=totalL)

    # mr_pirmin = load_model_results("2023-12-20", results_folder="RESULTS_PIRMIN", npz_file_ready=False, step=step, range_runIDs=[4096, 8192], L=totalL)
    # mr_me = load_model_results("2023-12-20", results_folder="RESULTS_ICH", npz_file_ready=False, step=step, range_runIDs=[0, 4096], L=totalL)
    #
    # mr_total = np.concatenate((mr_me, mr_pirmin), axis=0)
    #
    # filename = os.path.join("SA_RESULTS", f"total_model_results_{today}_t={step}")
    # np.savez(filename, N=N, t=step, model_results=mr_total)

    # mr = load_model_results("2023-12-20")
    # mr = load_model_results("2023-12-19", npz_file_ready=False, step=step, range_runIDs=[0, 16], L=totalL)
    # mr = load_model_results("2023-12-20", step)
    # run_analysis_SA(mr, step=step)
    #
    s_load = np.load(os.path.join("SA_RESULTS", f"sobol_indices_{today}_t={step}.npz"), allow_pickle=True)
    s_cells = s_load['sobol_indices_cells']
    s_ecm = s_load['sobol_indices_ecm']
    S1s_cells = np.array([s['S1'] for s in s_cells])
    S1confs_cells = np.array([s["S1_conf"] for s in s_cells])
    S1s_ecm = np.array([s['S1'] for s in s_ecm])
    S1confs_ecm = np.array([s["S1_conf"] for s in s_ecm])
    # ## check some odd values
    # ## when the confidence intervals include zero one can treat them as zero according to https://github.com/SALib/SALib/issues/102
    # # cellconfs1 = []
    # # cellconfs2 = []
    # # cellconfs3 = []
    # # for param, conflist in enumerate([cellconfs1, cellconfs2, cellconfs3]):
    # #     for i in range(totalL):
    # #         if not np.isnan(S1s_cells[i, param]):
    # #             temp = [S1s_cells[i, param] - S1confs_cells[i, param], S1s_cells[i, param] + S1confs_cells[i, param]]
    # #             conflist.append(temp)
    # # print(cellconfs1)
    # # print(cellconfs2)
    # # print(cellconfs3)
    # # STs_cells = np.array([s['ST'] for s in s_cells])
    # # STs_ecm = np.array([s['ST'] for s in s_ecm])
    #
    # plot_indices(S1s_cells, S1s_ecm, type_of_index="First-order", index_abbrev="S1", step=step)
    # plot_indices2(S1s_cells, S1s_ecm, type_of_index="First-order", index_abbrev="S1", step=step, S1_conf_cells=S1confs_cells, S1_conf_ecm=S1confs_ecm)
    S2s_cells = np.array([s['S2'] for s in s_cells])
    S2s_ecm = np.array([s['S2'] for s in s_ecm])
    plot_indices(S2s_cells, S2s_ecm, type_of_index="Second-order", index_abbrev="S2", step=step)
