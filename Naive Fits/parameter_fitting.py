import os
from parameters import today, totalL, average_runs, celldiam, voxsize
from read import generate_runID
from colson_SA import colsons_model
from crossley_SA import crossleys_model
from pdes_all_parameters import crossleys_model_all_params, colsons_model_all_params
from main import main_model
from plots import oneD_output
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
from shapely.geometry import LineString
from scipy.stats import ttest_ind, mannwhitneyu, normaltest
from matplotlib.legend_handler import HandlerBase

T = 60  # 600  # the number of iterations and the time point at which the PDE should be evaluated


def solve_crossley(num_data_points):
    """
    Generates data for the non-linear least square fit.
    :param num_data_points: the number of data points which should be returned; for instance totalL (+ 1 ?).
    :return: xdata, an array of the spatial points, and ydata, the corresponding points of the solution.
    """
    # parameters as defined in Crossley et al.
    spacing = 0.1  # grid spacing, see \Delta in Appendix B of Crossley et al.
    L = totalL  # 30  # 200  # length of the domain
    I = int(L / spacing)  # number of grid points
    alpha = 5  # length of the domain that is initially occupied by cells
    m_0 = 0.8  # ECM density, 0.2 or 0.8
    l = 50  # degradation rate
    # additional parameters for the parameterized version
    D_cells = 0.5  # diffusivity of tumour cells in the absence of ECM
    K = 1  # carrying capacity
    r = 0.1  # proliferation probability

    # define the initial values
    u0 = np.zeros(I)
    m0 = np.zeros(I)
    for x in range(I):
        if x * spacing < alpha:  # u_0 (x) = 1 for x < alpha, 0 else
            u0[x] = 1
        if x * spacing >= alpha:  # m_0 (x) = m_0 for x >= alpha, 0 else
            m0[x] = m_0

    init_vals = np.ravel([u0, m0])

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
        # CORRECT DISCRETIZATION
        u_out[0] = D_cells / (spacing * spacing) * (u[1] - u[0]) + u[0] * (1 - u[0] - m[0])
        u_out[I - 1] = D_cells / (spacing * spacing) * (
                    u[I - 2] - u[I - 1] - u[I - 2] * m[I - 1] + u[I - 1] * m[I - 2]) + u[I - 1] * (
                                   1 - u[I - 1] - m[I - 1])
        u_out[1:I - 1] = D_cells / (spacing * spacing) * (
                u[0:I - 2] * (1 - m[1:I - 1] / K) + u[1:I - 1] * (m[2:I] / K + m[0:I - 2] / K - 2) + u[2:I] * (
                1 - m[1:I - 1] / K)) + r * u[1:I - 1] * (1 - (u[1:I - 1] + m[1:I - 1]) / K)
        m_out = -l * m * u
        return np.ravel([u_out, m_out])

    sol = solve_ivp(crossley_parameterized_ode, [0, T], init_vals)
    timepoints = sol.t
    end_t = len(timepoints) - 1
    sol_end_t = sol.y[:, end_t]
    sol_end_t_cells = sol_end_t[0:I]
    sol_end_t_ecm = sol_end_t[I:2 * I]

    # get only as many data points as desired
    x = np.linspace(0, 2*L, I)
    indices = np.linspace(0, I - 1, 2*num_data_points).astype(int)
    xdata = x[indices]
    cells = sol_end_t_cells[indices[:num_data_points]]
    ecm = sol_end_t_ecm[indices[:num_data_points]]
    ydata = np.ravel([cells, ecm])

    return xdata, ydata


def solve_crossley_curvefit(x, l, r, alpha):  #(x, l, r, D_cells, K, alpha):
    """
    Generates data for the non-linear least square fit.
    :param num_data_points: the number of data points which should be returned; for instance totalL (+ 1 ?).
    :return: xdata, an array of the spatial points, and ydata, the corresponding points of the solution.
    """
    # parameters as defined in Crossley et al.
    spacing = 0.1  # grid spacing, see \Delta in Appendix B of Crossley et al.
    L = 2000  # totalL  # 30  # 200  # length of the domain
    I = int(L / spacing)  # number of grid points
    # alpha = 5  # length of the domain that is initially occupied by cells
    m_0 = 1  # ECM density, 0.2 or 0.8
    # l = 50  # degradation rate
    # additional parameters for the parameterized version
    D_cells = 0.5  # diffusivity of tumour cells in the absence of ECM
    K = 1  # carrying capacity
    # r = 0.1  # proliferation probability

    # define the initial values
    u0 = np.zeros(I)
    m0 = np.zeros(I)
    for x in range(I):
        if x * spacing < alpha:  # u_0 (x) = 1 for x < alpha, 0 else
            u0[x] = 0.25
        if x * spacing >= alpha:  # m_0 (x) = m_0 for x >= alpha, 0 else
            m0[x] = m_0

    init_vals = np.ravel([u0, m0])

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
        # CORRECT DISCRETIZATION
        u_out[0] = D_cells / (spacing * spacing) * (u[1] - u[0]) + u[0] * (1 - u[0] - m[0])
        u_out[I - 1] = D_cells / (spacing * spacing) * (
                    u[I - 2] - u[I - 1] - u[I - 2] * m[I - 1] + u[I - 1] * m[I - 2]) + u[I - 1] * (
                                   1 - u[I - 1] - m[I - 1])
        u_out[1:I - 1] = D_cells / (spacing * spacing) * (
                u[0:I - 2] * (1 - m[1:I - 1] / K) + u[1:I - 1] * (m[2:I] / K + m[0:I - 2] / K - 2) + u[2:I] * (
                1 - m[1:I - 1] / K)) + r * u[1:I - 1] * (1 - (u[1:I - 1] + m[1:I - 1]) / K)
        m_out = -l * m * u
        return np.ravel([u_out, m_out])

    sol = solve_ivp(crossley_parameterized_ode, [0, T], init_vals)
    timepoints = sol.t
    end_t = len(timepoints) - 1
    sol_end_t = sol.y[:, end_t]
    sol_end_t_cells = sol_end_t[0:I]
    sol_end_t_ecm = sol_end_t[I:2 * I]

    # get only as many data points as desired
    num_data_points = totalL
    x = np.linspace(0, 2*L, I)
    indices = np.linspace(0, I - 1, num_data_points).astype(int)
    xdata = x[indices]
    cells = sol_end_t_cells[indices]
    ecm = sol_end_t_ecm[indices]
    ydata = np.ravel([cells, ecm])

    # ydata = []
    # for pos in x:
    #     ydata.append(sol_end_t_cells[find_closest_value(pos)])

    return ydata


def solve_colson(num_data_points):
    # parameters as defined in Colson et al. (with discretization parameters from Crossley et al.)
    spacing = 0.1  # grid spacing, see \Delta in Appendix B of Crossley et al.
    L = totalL  # 30  # 100  # length of the domain
    I = int(L / spacing)  # number of grid points
    sigma = 5  # length of the domain that is initially occupied by cells
    omega = 1  # sharpness of the initial boundary between the tumour and healthy tissue
    D_cells = 0.5  # diffusivity of tumour cells in the absence of ECM
    M = 1  # ECM density that inhibits all tumour cell movement
    m_0 = 0.8  # ECM density
    K = 1  # carrying capacity
    r = 0.1  # growth rate
    k = 1  # degradation rate

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

    # plot at different points in time
    timepoints = sol.t
    middle_t = round(len(timepoints) / 2)
    end_t = len(timepoints) - 1

    sol_time1 = sol.y[:, 0]  # u and m at this point in time are given by sol_time1[0:I] and sol_time1[I:2*I], resp.
    sol_end_t = sol.y[:, end_t]
    sol_end_t_cells = sol_end_t[0:I]
    sol_end_t_ecm = sol_end_t[I:2 * I]

    # get only as many data points as desired
    x = np.linspace(0, 2 * L, I)
    indices = np.linspace(0, I - 1, 2 * num_data_points).astype(int)
    xdata = x[indices]
    cells = sol_end_t_cells[indices[:num_data_points]]
    ecm = sol_end_t_ecm[indices[:num_data_points]]
    ydata = np.ravel([cells, ecm])

    return xdata, ydata


def solve_colson_curvefit(x, k, r, D_cells, M, K, sigma, conversion):
    """
    :param x: the x-data
    :param k: degradation rate
    :param r: growth rate
    """
    # parameters as defined in Colson et al. (with discretization parameters from Crossley et al.)
    spacing = 0.1  # grid spacing, see \Delta in Appendix B of Crossley et al.
    L = totalL  # 30  # 100  # length of the domain
    I = int(L / spacing)  # number of grid points
    # sigma = 5  # length of the domain that is initially occupied by cells
    # omega = 1  # sharpness of the initial boundary between the tumour and healthy tissue
    # D_cells = 0.5  # diffusivity of tumour cells in the absence of ECM
    # M = 1  # ECM density that inhibits all tumour cell movement
    m_0 = 0.8  # ECM density
    # K = 1  # carrying capacity
    # r = 0.1  # growth rate
    # k = 1  # degradation rate

    omega = sigma * conversion

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

    # plot at different points in time
    timepoints = sol.t
    middle_t = round(len(timepoints) / 2)
    end_t = len(timepoints) - 1

    sol_time1 = sol.y[:, 0]  # u and m at this point in time are given by sol_time1[0:I] and sol_time1[I:2*I], resp.
    sol_end_t = sol.y[:, end_t]
    sol_end_t_cells = sol_end_t[0:I]
    sol_end_t_ecm = sol_end_t[I:2 * I]

    # get only as many data points as desired
    num_data_points = totalL
    x = np.linspace(0, 2 * L, I)
    indices = np.linspace(0, I - 1, num_data_points).astype(int)
    xdata = x[indices]
    cells = sol_end_t_cells[indices]
    ecm = sol_end_t_ecm[indices]
    ydata = np.ravel([cells, ecm])
    # x = np.linspace(0, 2 * L, I)
    # indices = np.linspace(0, I - 1, 2 * num_data_points).astype(int)
    # xdata = x[indices]
    # cells = sol_end_t_cells[indices[:num_data_points]]
    # ecm = sol_end_t_ecm[indices[:num_data_points]]
    # ydata = np.ravel([cells, ecm])

    return ydata


def dummy_model(t, a, b, c):
    x = np.linspace(0, 29, 30)
    ecm = a / (1 + np.exp(-b * (x - c)))
    cells = a / (1 + np.exp(-b * (-x - c)))
    return np.ravel(np.array([cells, ecm]))


def write_parameters(ECMconcentr, distanceinfl, prolifprob):
    with open(os.path.join("PARAMETER_ESTIMATION", f"used_parameters_{today}.out"), "w") as ofp:
        for param in [ECMconcentr, distanceinfl, prolifprob]:
            ofp.write(f"{param}\n")


def find_closest_value(val):
    array = np.arange(2*totalL)
    closest_value = min(array, key=lambda element: abs(element - val))
    return int(closest_value)


def cpm_fem_results(x_array, prolifprob):
    ECMconcentr = 0.8
    distanceinfl = 0.3*celldiam
    # does average_runs runs and returns the y-value for the corresponding closest x-value
    print(f"Starting {average_runs} runs with ECMconcentr={ECMconcentr}, distanceinfl={distanceinfl}, prolifprob={prolifprob}.\nThey will be saved for the date {today}.")
    # array in which the results should be saved
    next_runID = generate_runID(today)
    next_runIDs = []
    for i in range(average_runs):
        next_runIDs.append(next_runID + i)
    model_results = np.zeros([average_runs, 2*totalL])
    # generate a parameter list s.t. multiprocessing can be used
    param_values = []
    for i in range(average_runs):
        param_values.append([next_runIDs[i], T, ECMconcentr, distanceinfl, prolifprob])
    # multiprocessing
    pool = mp.Pool(10)  # the parameter of mp.Pool() defines the number of parallel processes
    results_raw = pool.starmap(main_model, param_values)
    # results_raw = pool.starmap(dummy_model, param_values)
    for i, result in enumerate(results_raw):
        model_results[i, :] = result
    # create the average over all runs at each spatial point
    averaged_runs = np.zeros(2*totalL)
    for i in range(2*totalL):
        averaged_runs[i] = np.mean(model_results[:, i])

    # find the closest value to x in the indexing of averaged_runs
    results = []
    for x in x_array:
        results.append(averaged_runs[find_closest_value(x)])
    return results


def fit_one_parameter():
    xdata, ydata = solve_crossley(totalL)
    popt, pcov, infodict, mesg, ier = curve_fit(cpm_fem_results, xdata, ydata, maxfev=50, p0=0.0050000526026461635, bounds=(0.0001, 0.03), full_output=True)
    return popt, pcov, infodict, mesg, ier


def fit_two_parameters(crossley=False, colson=False, list_of_runIDs=None):
    """
    The reverse way - fit Crossley (proliferation and dagradation rate) to an average of CPM-FEM simulations.
    """
    # xdata, _ = solve_crossley(totalL)
    xdata = range(2*totalL)
    # run the model ...
    # y_cpm_fem = cpm_fem_results(xdata, 0.005)
    # ... or load previous runs
    if list_of_runIDs:
        # list_of_runIDs = list(range(201, 301))  # no durotaxis: list(range(51, 101)) + list(range(151, 201))  # all effects: list(range(1, 51)) + list(range(101, 151))
        model_results = np.zeros([average_runs, 2 * totalL])
        for row, runID in enumerate(list_of_runIDs):
            cells, ecm = oneD_output(f"RESULTS/{today}-{runID}", totalL, T)
            model_results[row, :] = np.ravel(np.array([cells, ecm]))
        averaged_runs = np.zeros(2 * totalL)
        for i in range(2 * totalL):
            averaged_runs[i] = np.mean(model_results[:, i])
        # find the closest value to x in the indexing of averaged_runs
        y_cpm_fem = []
        for pos in xdata:
            y_cpm_fem.append(averaged_runs[find_closest_value(pos)])
    # or load one specific file
    # cells, ecm = oneD_output(f"RESULTS/2024-01-06-1", totalL, T)
    # model_results = np.ravel(np.array([cells, ecm]))
    # y_cpm_fem = []
    # for pos in xdata:
    #     y_cpm_fem.append(model_results[find_closest_value(pos)])
    if crossley:
        # popt, pcov, infodict, mesg, ier = curve_fit(solve_crossley_curvefit, xdata, y_cpm_fem, p0=[0.2, 0.005, 0.5, 1, 5], full_output=True)  # lambda, r, D_cells, K, alpha
        # popt, pcov, infodict, mesg, ier = curve_fit(solve_crossley_curvefit, xdata, y_cpm_fem, p0=[0.2, 0.005], full_output=True)
        popt, pcov, infodict, mesg, ier = curve_fit(solve_crossley_curvefit, xdata, y_cpm_fem,
                                                    p0=[0.2, 0.005, 0.5, 1, 5],
                                                    bounds=([0, 0, 0, 0, 0], [100, 1, 5, 5, 30]),
                                                    full_output=True)  # lambda, r, D_cells, K, alpha
    elif colson:
        # popt, pcov, infodict, mesg, ier = curve_fit(solve_colson_curvefit, xdata, y_cpm_fem, p0=[0.2, 0.005, 0.5, 1, 1, 5, 0.2], bounds=([0, 0, 0, 0, 0, 0, 0], [100, 1, 5, 5, 5, 30, 1]), full_output=True)  # lambda, r, D_cells, M, K, sigma, conversion
        popt, pcov, infodict, mesg, ier = curve_fit(solve_colson_curvefit, xdata, y_cpm_fem, p0=[0.2, 0.005], full_output=True)
    return popt, pcov, infodict, mesg, ier


def plot_crossley_and_fitted_CPM_FEM(l, r, prol=None, list_of_runIDs=None, crossley=False, colson=False):
    """
    Plots the PDE model together with the fitted CPM-FEM. One of the parameters needs to be indicated.
    :param prol: prolifprob; if indicated, average_runs runs with this parameter will be performed and then plotted.
    :param list_of_runIDs: if indicated, will average and plot the data from these runs. Format: [1, 2, 3, ...]
    """
    # x, _ = solve_crossley(totalL)
    x = range(2*totalL)
    if crossley:
        y_crossley = solve_crossley_curvefit(x, l, r)
    elif colson:
        y_crossley = solve_colson_curvefit(x, l, r)
    else:
        print("In the function plot_crossley_and_fitted_CPM_FEM() you need to choose the PDE model!")
        return
    if prol:
        y_cpm_fem = cpm_fem_results(x, prol)
    else:  # if list_of_runIDs is indicated
        model_results = np.zeros([average_runs, 2*totalL])
        for row, runID in enumerate(list_of_runIDs):
            cells, ecm = oneD_output(f"RESULTS/{today}-{runID}", totalL, T)
            model_results[row, :] = np.ravel(np.array([cells, ecm]))
        averaged_runs = np.zeros(2 * totalL)
        for i in range(2 * totalL):
            averaged_runs[i] = np.mean(model_results[:, i])
        # find the closest value to x in the indexing of averaged_runs
        y_cpm_fem = []
        for pos in x:
            y_cpm_fem.append(averaged_runs[find_closest_value(pos)])
    # plot
    plt.figure()
    plt.plot(x[:totalL], y_crossley[:totalL])
    plt.plot(x[:totalL], y_crossley[totalL:])
    plt.plot(x[:totalL], y_cpm_fem[:totalL])
    plt.plot(x[:totalL], y_cpm_fem[totalL:])
    plt.show()


def nice_plot_of_fitted_models(t, p_crossley, p_colson, list_of_runIDs):
    """
    Makes a nice, high resultion plot of both fitted PDEs with the averaged CPM-FEM results.
    :param p_crossley: array of the fitted degradation rate and the proliferation probability for Crossley
    :param p_colson: same as p_crossley, but for Colson
    :param list_of_runIDs: averages and plots the data from these runs. Format: [1, 2, 3, ...]
    """
    # get the x-axis for the CPM-FEM plots
    # x_cpm_fem, _ = solve_crossley(totalL)
    x_cpm_fem = range(2*totalL)
    # get the CPM-FEM data
    model_results = np.zeros([average_runs, 2 * totalL])
    for row, runID in enumerate(list_of_runIDs):
        cells, ecm = oneD_output(f"RESULTS/{today}-{runID}", totalL, t)
        model_results[row, :] = np.ravel(np.array([cells, ecm]))
    averaged_runs = np.zeros(2 * totalL)
    std_lower = np.zeros(2 * totalL)
    std_upper = np.zeros(2 * totalL)
    for i in range(2 * totalL):
        mean_i = np.mean(model_results[:, i])
        std_i = np.std(model_results[:, i])
        averaged_runs[i] = mean_i
        std_lower[i] = mean_i - std_i
        std_upper[i] = mean_i + std_i
    y_cpm_fem = averaged_runs
    # find the closest value to x in the indexing of averaged_runs
    #y_cpm_fem = []
    #for pos in x_cpm_fem:
    #    y_cpm_fem.append(averaged_runs[find_closest_value(pos)])

    # the x-axis for the PDEs
    I = int(totalL / 0.1)
    x_pde = np.linspace(0, totalL, I)
    # the PDE solutions
    y_crossley = crossleys_model(t, 0.8, p_crossley[1], p_crossley[0])  # solve_crossley_curvefit(x_pde, p_crossley[0], p_crossley[1])
    y_colson = colsons_model(t, 0.8, p_colson[1], p_colson[0])  # solve_colson_curvefit(x_pde, p_colson[0], p_colson[1])

    # plot
    plt.figure()
    plt.plot(x_pde, y_crossley[:I], color='#ada397', label="Crossley")
    plt.plot(x_pde, y_crossley[I:], color='#ada397', linestyle="dashed")
    plt.plot(x_pde, y_colson[:I], color='#bdcebe', label="Colson")
    plt.plot(x_pde, y_colson[I:], color='#bdcebe', linestyle="dashed")
    plt.plot(x_cpm_fem[:totalL], y_cpm_fem[:totalL], color='#d6cbd3', label="CPM-FEM")
    plt.plot(x_cpm_fem[:totalL], y_cpm_fem[totalL:], color='#d6cbd3', linestyle="dashed")
    plt.fill_between(x_cpm_fem[:totalL], std_lower[:totalL], std_upper[:totalL], alpha=0.5, color='#d6cbd3')
    plt.fill_between(x_cpm_fem[:totalL], std_lower[totalL:], std_upper[totalL:], alpha=0.5, color='#d6cbd3')
    plt.xlabel("x")
    plt.ylabel("cell/ECM density")
    plt.title(f"Average of {len(list_of_runIDs)} CPM-FEM model evaluations with fitted PDE models at t = {t}")
    plt.legend()
    plt.show()


def nice_plot_of_fitted_models_various_t(t_list, p_crossley, p_colson, list_of_runIDs):
    """
    Makes a nice, high resultion plot of both fitted PDEs with the averaged CPM-FEM results.
    :param t_list: list of 4 timepoints.
    :param p_crossley: array of the fitted degradation rate and the proliferation probability for Crossley
    :param p_colson: same as p_crossley, but for Colson
    :param list_of_runIDs: averages and plots the data from these runs. Format: [1, 2, 3, ...]
    """
    # get the x-axis for the CPM-FEM plots
    # x_cpm_fem, _ = solve_crossley(totalL)
    x_cpm_fem = range(2*totalL)
    # the x-axis for the PDEs
    I = int(totalL / 0.1)
    x_pde = np.linspace(0, totalL, I)
    x_pde = x_pde[:(I-10)]  # only up to 29 instead of 30

    all_y_cpm_fem = []
    all_upper = []
    all_lower = []
    all_y_crossley = []
    all_y_colson = []

    for t in t_list:
        # get the CPM-FEM data
        model_results = np.zeros([average_runs, 2 * totalL])
        for row, runID in enumerate(list_of_runIDs):
            cells, ecm = oneD_output(f"RESULTS/{today}-{runID}", totalL, t)
            model_results[row, :] = np.ravel(np.array([cells, ecm]))
        averaged_runs = np.zeros(2 * totalL)
        std_lower = np.zeros(2 * totalL)
        std_upper = np.zeros(2 * totalL)
        for i in range(2 * totalL):
            mean_i = np.mean(model_results[:, i])
            std_i = np.std(model_results[:, i])
            averaged_runs[i] = mean_i
            std_lower[i] = mean_i - std_i
            std_upper[i] = mean_i + std_i
        y_cpm_fem = averaged_runs
        # find the closest value to x in the indexing of averaged_runs
        #y_cpm_fem = []
        #for pos in x_cpm_fem:
        #    y_cpm_fem.append(averaged_runs[find_closest_value(pos)])

        # the PDE solutions
        if len(p_colson) < 3:
            y_crossley = crossleys_model(t, 0.8, p_crossley[1], p_crossley[0])  # solve_crossley_curvefit(x_pde, p_crossley[0], p_crossley[1])
            y_colson = colsons_model(t, 0.8, p_colson[1], p_colson[0])  # solve_colson_curvefit(x_pde, p_colson[0], p_colson[1])
        else:
            y_crossley = crossleys_model_all_params(t, *p_crossley)
            y_colson = colsons_model_all_params(t, *p_colson)

        all_y_cpm_fem.append(y_cpm_fem)
        all_upper.append(std_upper)
        all_lower.append(std_lower)
        all_y_crossley.append(y_crossley)
        all_y_colson.append(y_colson)

    plt.rcParams.update({'font.size': 12})
    fig = plt.figure()
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    axes = [ax1, ax2, ax3, ax4]

    class AnyObjectHandler(HandlerBase):
        def create_artists(self, legend, orig_handle,
                           x0, y0, width, height, fontsize, trans):
            l1 = plt.Line2D([x0, y0 + width], [0.7 * height, 0.7 * height],
                            linestyle=orig_handle[1], color=orig_handle[0])
            l2 = plt.Line2D([x0, y0 + width], [0.3 * height, 0.3 * height],
                            color=orig_handle[0])
            return [l1, l2]

    for i, t in enumerate(t_list):
        axes[i].plot(x_pde, list(all_y_crossley[i][:(I-10)]), color='#ada397', label="Crossley")
        axes[i].plot(x_pde, list(all_y_crossley[i][I:(2*I-10)]), color='#ada397', linestyle="dashed")
        axes[i].plot(x_pde, list(all_y_colson[i][:(I-10)]), color='#bdcebe', label="Colson")
        axes[i].plot(x_pde, list(all_y_colson[i][I:(2*I-10)]), color='#bdcebe', linestyle="dashed")
        axes[i].plot(x_cpm_fem[:totalL], list(all_y_cpm_fem[i][:totalL]), color='#d6cbd3', label="CPM-FEM")
        axes[i].plot(x_cpm_fem[:totalL], list(all_y_cpm_fem[i][totalL:]), color='#d6cbd3', linestyle="dashed")
        axes[i].fill_between(x_cpm_fem[:totalL], all_lower[i][:totalL], all_upper[i][:totalL], alpha=0.5, color='#d6cbd3')
        axes[i].fill_between(x_cpm_fem[:totalL], all_lower[i][totalL:], all_upper[i][totalL:], alpha=0.5, color='#d6cbd3')
        axes[i].set_ylim(-0.08, 1.1)
        if i == 2 or i == 3:
            axes[i].set_xlabel("x")
        axes[i].set_ylabel("cell/ECM density")
        axes[i].set_title(f"t = {t}")
        axes[i].legend([("#ada397", "--"), ("#bdcebe", "--"), ("#d6cbd3", "--")], ['Crossley', "Colson", "CPM-FEM"], handler_map={tuple: AnyObjectHandler()})
        # axes[i].legend()
    # plt.title(f"Average of {len(list_of_runIDs)} CPM-FEM model evaluations with fitted PDE models at t = {t}")
    # plt.legend()
    plt.show()


def nice_plot_of_fitted_models_specific_run_one_t(t, p_crossley, p_colson):
    """
    Makes a nice, high resultion plot of both fitted PDEs with specific run of the CPM-FEM results.
    :param t_list: list of 4 timepoints.
    :param p_crossley: array of the fitted degradation rate and the proliferation probability for Crossley
    :param p_colson: same as p_crossley, but for Colson
    """
    # get the x-axis for the CPM-FEM plots
    # x_cpm_fem, _ = solve_crossley(totalL)
    x_cpm_fem = range(2*totalL)
    # the x-axis for the PDEs
    I = int(totalL / 0.1)
    x_pde = np.linspace(0, totalL, I)
    # x_pde = x_pde[:(I-10)]  # only up to 29 instead of 30

    all_y_cpm_fem = []
    all_y_crossley = []
    all_y_colson = []

    # get the CPM-FEM data
    cells, ecm = oneD_output(f"RESULTS/2024-01-06-1", totalL, t)
    y_cpm_fem = np.ravel(np.array([cells, ecm]))

    # the PDE solutions
    if len(p_colson) < 3:
        y_crossley = crossleys_model(t, 0.8, p_crossley[1], p_crossley[0])  # solve_crossley_curvefit(x_pde, p_crossley[0], p_crossley[1])
        y_colson = colsons_model(t, 0.8, p_colson[1], p_colson[0])  # solve_colson_curvefit(x_pde, p_colson[0], p_colson[1])
    else:
        y_crossley = crossleys_model_all_params(t, *p_crossley)
        y_colson = colsons_model_all_params(t, *p_colson)

    all_y_cpm_fem.append(y_cpm_fem)
    all_y_crossley.append(y_crossley)
    all_y_colson.append(y_colson)

    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots()

    class AnyObjectHandler(HandlerBase):
        def create_artists(self, legend, orig_handle,
                           x0, y0, width, height, fontsize, trans):
            l1 = plt.Line2D([x0, y0 + width], [0.7 * height, 0.7 * height],
                            linestyle=orig_handle[1], color=orig_handle[0])
            l2 = plt.Line2D([x0, y0 + width], [0.3 * height, 0.3 * height],
                            color=orig_handle[0])
            return [l1, l2]

    i = 0
    ax.plot(x_pde, list(all_y_crossley[i][:I]), color='#ada397', label="Crossley")
    ax.plot(x_pde, list(all_y_crossley[i][I:]), color='#ada397', linestyle="dashed")
    ax.plot(x_pde, list(all_y_colson[i][:I]), color='#bdcebe', label="Colson")
    ax.plot(x_pde, list(all_y_colson[i][I:]), color='#bdcebe', linestyle="dashed")
    ax.plot(x_cpm_fem[:totalL], list(all_y_cpm_fem[i][:totalL]), color='#d6cbd3', label="CPM-FEM")
    ax.plot(x_cpm_fem[:totalL], list(all_y_cpm_fem[i][totalL:]), color='#d6cbd3', linestyle="dashed")
    ax.set_ylim(-0.08, 1.1)
    ax.set_xlabel("x")
    ax.set_ylabel("cell/ECM density")
    ax.set_title(f"t = {t}")
    ax.legend([("#ada397", "--"), ("#bdcebe", "--"), ("#d6cbd3", "--")], ['Crossley', "Colson", "CPM-FEM"], handler_map={tuple: AnyObjectHandler()})
    # plt.title(f"Average of {len(list_of_runIDs)} CPM-FEM model evaluations with fitted PDE models at t = {t}")
    # plt.legend()
    plt.show()


def find_intersections(t, p_crossley, p_colson, list_of_runIDs, with_std=False, return_only_CPM_intersections=False):
    """
    Finds the intersection point (x-value) of the respective cell and ecm curves.
    :return:
    """
    # PDE models
    I = int(totalL / 0.1)
    x_pde = np.linspace(0, totalL, I)
    y_05_pde = [0.5] * I
    if len(p_crossley) < 3:
        y_crossley = crossleys_model(t, 0.8, p_crossley[1], p_crossley[0])
    else:
        y_crossley = y_crossley = crossleys_model_all_params(t, *p_crossley)
    y_crossley_cells = y_crossley[:I]
    y_crossley_ecm = y_crossley[I:]
    first_line_crossley = LineString(np.column_stack((x_pde, y_crossley_cells)))
    # second_line_crossley = LineString(np.column_stack((x_pde, y_crossley_ecm)))
    second_line_crossley = LineString(np.column_stack((x_pde, y_05_pde)))
    intersection_crossley = first_line_crossley.intersection(second_line_crossley)
    xcrossley, _ = intersection_crossley.xy

    if len(p_colson) < 3:
        y_colson = colsons_model(t, 0.8, p_colson[1], p_colson[0])
    else:
        y_colson = colsons_model_all_params(t, *p_colson)
    y_colson_cells = y_colson[:I]
    y_colson_ecm = y_colson[I:]
    first_line_colson = LineString(np.column_stack((x_pde, y_colson_cells)))
    # second_line_colson = LineString(np.column_stack((x_pde, y_colson_ecm)))
    second_line_colson = LineString(np.column_stack((x_pde, y_05_pde)))
    intersection_colson = first_line_colson.intersection(second_line_colson)
    if intersection_colson.geom_type == 'Point':
        xcolson, _ = intersection_colson.xy
    else:
        xcolson = [0]

    # CPM-FEM
    x_cpm_fem = range(totalL)
    y_05_cpm_fem = [0.5] * totalL
    model_results = np.zeros([average_runs, 2 * totalL])
    for row, runID in enumerate(list_of_runIDs):
        cells, ecm = oneD_output(f"RESULTS/{today}-{runID}", totalL, t)
        model_results[row, :] = np.ravel(np.array([cells, ecm]))

    if with_std or return_only_CPM_intersections:
        # with error estimate (std)
        cpm_fem_intersections = []
        for run in range(len(list_of_runIDs)):
            y_cpm_fem_cells = model_results[run, :totalL]
            y_cpm_fem_ecm = model_results[run, totalL:]
            first_line_cpm_fem = LineString(np.column_stack((x_cpm_fem, y_cpm_fem_cells)))
            # second_line_cpm_fem = LineString(np.column_stack((x_cpm_fem, y_cpm_fem_ecm)))
            second_line_cpm_fem = LineString(np.column_stack((x_cpm_fem, y_05_cpm_fem)))
            intersection_cpm_fem = first_line_cpm_fem.intersection(second_line_cpm_fem)
            if intersection_cpm_fem.geom_type == 'Point':
                xcpm_fem, _ = intersection_cpm_fem.xy
                cpm_fem_intersections.append(xcpm_fem[0])
        if return_only_CPM_intersections:
            return cpm_fem_intersections
        xcpm_fem_mean = np.mean(cpm_fem_intersections)
        xcpm_fem_std = np.std(cpm_fem_intersections)
        return xcrossley[0], xcolson[0], xcpm_fem_mean, xcpm_fem_std
    else:
        # without error estimate
        averaged_runs = np.zeros(2 * totalL)
        for i in range(2 * totalL):
            mean_i = np.mean(model_results[:, i])
            averaged_runs[i] = mean_i
        y_cpm_fem = averaged_runs
        y_cpm_fem_cells = y_cpm_fem[:totalL]
        y_cpm_fem_ecm = y_cpm_fem[totalL:]
        first_line_cpm_fem = LineString(np.column_stack((x_cpm_fem, y_cpm_fem_cells)))
        # second_line_cpm_fem = LineString(np.column_stack((x_cpm_fem, y_cpm_fem_ecm)))
        second_line_cpm_fem = LineString(np.column_stack((x_cpm_fem, y_05_cpm_fem)))
        intersection_cpm_fem = first_line_cpm_fem.intersection(second_line_cpm_fem)
        xcpm_fem, _ = intersection_cpm_fem.xy


        return xcrossley[0], xcolson[0], xcpm_fem[0]



def wave_speed(t_list, p_crossley, p_colson, list_of_runIDs, with_std=False):
    """
    The time points in t_list need to be equal steps, otherwise the formula for the wave speed is incorrect
    """
    if with_std:
        int_crossley = []
        int_colson = []
        int_cpm_fem = []
        std_cpm_fem = []
        for t in t_list:
            xcrossley, xcolson, xcpm_fem_mean, xcpm_fem_std = find_intersections(t, p_crossley, p_colson, list_of_runIDs, with_std=True)
            int_crossley.append(xcrossley)
            int_colson.append(xcolson)
            int_cpm_fem.append(xcpm_fem_mean)
            std_cpm_fem.append(xcpm_fem_std)
    else:
        int_crossley = []
        int_colson = []
        int_cpm_fem = []
        for t in t_list:
            xcrossley, xcolson, xcpm_fem = find_intersections(t, p_crossley, p_colson, list_of_runIDs)
            int_crossley.append(xcrossley)
            int_colson.append(xcolson)
            int_cpm_fem.append(xcpm_fem)

    # the evolution of wave speed over time
    t_list_difference = t_list[1] - t_list[0]

    travelled_distances_crossley = [j - i for i, j in zip(int_crossley[:-1], int_crossley[1:])]
    wave_speeds_crossley = []
    for d in travelled_distances_crossley:
        wave_speeds_crossley.append(d/t_list_difference)

    travelled_distances_colson = [j - i for i, j in zip(int_colson[:-1], int_colson[1:])]
    wave_speeds_colson = []
    for d in travelled_distances_colson:
        wave_speeds_colson.append(d / t_list_difference)

    travelled_distances_cpm_fem = [j - i for i, j in zip(int_cpm_fem[:-1], int_cpm_fem[1:])]
    if with_std:
        travelled_distances_cpm_fem_std = [i + j for i, j in zip(std_cpm_fem[:-1], std_cpm_fem[1:])]
        wave_speeds_cpm_fem = []
        wave_speeds_cpm_fem_std = []
        for index in range(len(travelled_distances_cpm_fem)):
            wave_speeds_cpm_fem.append(travelled_distances_cpm_fem[index] / t_list_difference)
            wave_speeds_cpm_fem_std.append(travelled_distances_cpm_fem_std[index] / t_list_difference)
    else:
        wave_speeds_cpm_fem = []
        for d in travelled_distances_cpm_fem:
            wave_speeds_cpm_fem.append(d / t_list_difference)

    # the mean wave speed
    # mean_travelled_distance_crossley = np.mean(travelled_distances_crossley)
    # the mean wave speed for the last 4 time steps
    mean_travelled_distance_crossley = np.mean(travelled_distances_crossley[-4:])
    mean_wave_speed_crossley = mean_travelled_distance_crossley/ t_list_difference

    # mean_travelled_distance_colson = np.mean(travelled_distances_colson)
    # the mean wave speed for the last 4 time steps
    mean_travelled_distance_colson = np.mean(travelled_distances_colson[-4:])
    mean_wave_speed_colson = mean_travelled_distance_colson / t_list_difference

    # mean_travelled_distance_cpm_fem = np.mean(travelled_distances_cpm_fem)
    # the mean wave speed for the last 4 time steps
    mean_travelled_distance_cpm_fem = np.mean(travelled_distances_cpm_fem[-4:])
    mean_wave_speed_cpm_fem = mean_travelled_distance_cpm_fem / t_list_difference

    if with_std:
        return [wave_speeds_crossley, wave_speeds_colson, wave_speeds_cpm_fem, wave_speeds_cpm_fem_std], \
               [mean_wave_speed_crossley, mean_wave_speed_colson, mean_wave_speed_cpm_fem]
    else:
        return [wave_speeds_crossley, wave_speeds_colson, wave_speeds_cpm_fem], \
               [mean_wave_speed_crossley, mean_wave_speed_colson, mean_wave_speed_cpm_fem]


def plot_wave_speeds(t_list, p_crossley, p_colson, list_of_runIDs, type_cpm_fem, with_std=False):
    """
    The time points in t_list need to be equal steps, otherwise the formula for the wave speed is incorrect
    """
    wave_speed_evolution, mean_wave_speed = wave_speed(t_list, p_crossley, p_colson, list_of_runIDs, with_std=with_std)
    # umrechnung in micrometer per time step
    resized_wave_speed_evolution = []
    for model in wave_speed_evolution:
        temp = []
        for ws in model:
            temp.append(ws*voxsize*1e6)
        resized_wave_speed_evolution.append(temp)

    if with_std:
        all_upper = []
        all_lower = []
        for i in range(len(resized_wave_speed_evolution[3])):
            all_upper.append(resized_wave_speed_evolution[2][i] + resized_wave_speed_evolution[3][i])
            all_lower.append(resized_wave_speed_evolution[2][i] - resized_wave_speed_evolution[3][i])

    plt.rcParams.update({'font.size': 12})
    x_vals = t_list[1:]
    fig, ax = plt.subplots()
    ax.plot(x_vals, resized_wave_speed_evolution[1], color='#bdcebe', label="Colson")
    ax.axhline(mean_wave_speed[1] * voxsize * 1e6, color='#bdcebe', linestyle='dashed', label=f"Colson mean after t = 40: {round(mean_wave_speed[1]*voxsize*1e6, 4)}")
    ax.plot(x_vals, resized_wave_speed_evolution[2], color='#d6cbd3', label="CPM-FEM")
    ax.axhline(mean_wave_speed[2] * voxsize * 1e6, color='#d6cbd3', linestyle='dashed', label=f"CPM-FEM mean after t = 40: {round(mean_wave_speed[2] * voxsize * 1e6, 4)}")
    ax.plot(x_vals, resized_wave_speed_evolution[0], color='#ada397', label="Crossley")
    ax.axhline(mean_wave_speed[0] * voxsize * 1e6, color='#ada397', linestyle='dashed', label=f"Crossley mean after t = 40: {round(mean_wave_speed[0] * voxsize * 1e6, 4)}")
    if with_std:
        ax.fill_between(x_vals, all_lower, all_upper, alpha=0.5, color='#d6cbd3')
    # ax.plot(x_vals, wave_speed_evolution[0], color='#ada397', label="Crossley")
    # ax.plot(x_vals, wave_speed_evolution[1], color='#bdcebe', label="Colson")
    # ax.plot(x_vals, wave_speed_evolution[2], color='#d6cbd3', label="CPM-FEM")
    if with_std:
        ax.set_ylim(-1.5, 2.5)
    # else:
    #     ax.set_ylim(0.05, 1.04)
    ax.legend()
    ax.set_title(f"Mean wave speeds of the fitted PDEs and the {type_cpm_fem}")
    ax.set_xlabel("time step t")
    ax.set_ylabel("wave speed in micrometer/time step")
    plt.show()


def statistical_analysis_wavespeed(t_list, list_of_runIDs_all_effects, list_of_runIDs_noDurotaxis):
    """
    The first list of runIDs is from the model which is suspected to have the faster waves.
    """
    cpm_fem_intersections_all_effects = []
    cpm_fem_intersections_noDurotaxis = []
    for t in t_list:
        intersections_at_t_all_effects = find_intersections(t, [1, 1], [1, 1], list_of_runIDs_all_effects, return_only_CPM_intersections=True)
        cpm_fem_intersections_all_effects.append(intersections_at_t_all_effects)
        intersections_at_t_noDurotaxis = find_intersections(t, [1, 1], [1, 1], list_of_runIDs_noDurotaxis, return_only_CPM_intersections=True)
        cpm_fem_intersections_noDurotaxis.append(intersections_at_t_noDurotaxis)
    # T-test for the means of two independent samples at each t
    pvalues = []
    for i in range(len(t_list)):
        # test for normal distribution
        all_effects_normaltest = normaltest(cpm_fem_intersections_all_effects[i])
        noDurotaxis_normaltest = normaltest(cpm_fem_intersections_noDurotaxis[i])
        if all_effects_normaltest.pvalue < 0.05 and noDurotaxis_normaltest.pvalue < 0.05:
            # alternative (both not normally distributed) is accepted
            # t-test for two samples from any distribution
            # 'greater' means that H1 is: "the mean wave speed for all_effects is greater than the mean wave speed for noDurotaxis"
            testtype = "Mann-Whitney-U-Test"
            ttest = mannwhitneyu(cpm_fem_intersections_all_effects[i], cpm_fem_intersections_noDurotaxis[i],
                                 alternative='greater')
        else:
            # H0 (both normally distributed) is accepted
            # t-test for two normally distributed samples
            testtype = "T-Test"
            ttest = ttest_ind(cpm_fem_intersections_all_effects[i], cpm_fem_intersections_noDurotaxis[i], equal_var=True, alternative='greater')
        print(f"At t={t_list[i]}, the {testtype} yields {ttest}")
        pvalues.append(ttest.pvalue)
    # plot p-values
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots()
    ax.plot(t_list, pvalues, color='#eca1a6')
    ax.set_yscale('log')
    ax.set_ylim(1e-10, 1.5)
    ax.set_xlabel("time step t")
    ax.set_ylabel("p-value (log-scale)")
    plt.show()


def aic(ssr, n_param, n_obs):
    return n_obs * np.log(ssr / n_obs) + 2 * (n_param + 1)  # https://www.sciencedirect.com/science/article/pii/S0893965917301623#sec2


if __name__ == "__main__":
    runIDs_all_effects = list(range(1, 51)) + list(range(101, 151))
    runIDs_no_durotaxis = list(range(51, 101)) + list(range(151, 201))
    runIDs_no_attachment = list(range(201, 301))

    # colson = False
    # crossley = True
    # a, b, c, d, e = fit_two_parameters(colson=colson, crossley=crossley, list_of_runIDs=runIDs_all_effects)
    # a, b, c, d, e = fit_two_parameters(colson=colson, crossley=crossley)
    # print(a)  # CROSSLEY: a bei average_runs=10: [0.38350009 0.0585738 ]; bei average_runs=20: [0.4446647 0.0464012]; bei average_runs=20 und korrigierter celldivision: [0.39701261 0.03031938] bzw [12.05531912  0.20668651]
    # # COLSON: a bei average_runs=20: [0.1       0.0168384]; mit kleinerem lower bound für die degradation: [0.09714533 0.01741091] bzw [4.23630093 0.1       ] bzw [2.58918287 0.14933267]
    # residuals = c['fvec']
    # SSR = 0
    # for r in residuals:
    #     SSR += r*r
    # print(SSR)
    # print(len(residuals))
    # AIC
    # n_obsv = len(residuals)
    # n_params = len(a)
    # aic = aic(SSR, n_params, n_obsv)
    # if colson:
    #     name = "COLSON"
    # elif crossley:
    #     name = "CROSSLEY"
    # with open(os.path.join("PARAMETER_ESTIMATION", f"{name}_ALLPARAMETERS_conversion_noDurotaxis_{today}.out"), "w") as ofp:
    #     if colson:
    #         ofp.write(f"estimated lambda, r, D_cells, M, K, = {a}\n")
    #     elif crossley:
    #         ofp.write(f"estimated lambda, r, D_cells, K, alpha = {a}\n")
    #     ofp.write(f"residuals = {c['fvec']}\n")
    #     ofp.write(f"SSR = {SSR}\n")
    #     ofp.write(f"AIC = {aic}")  # preferred model is the one with the minimum AIC value
    # 0.00010009
    # plot_crossley_and_fitted_CPM_FEM(a[0], a[1], list_of_runIDs=[1, 2, 3, 4, 5])
    #plot_crossley_and_fitted_CPM_FEM(a[0], a[1], list_of_runIDs=[51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70], colson=colson, crossley=crossley)
    # nice_plot_of_fitted_models([12.05531912, 0.20668651], [2.58918287, 0.14933267], [51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70])
    # nice_plot_of_fitted_models([12.05531912, 0.20668651], [2.58918287, 0.14933267],
   #                             [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    # nice_plot_of_fitted_models([5.07418479, 0.14152198], [0.78997654, 0.11438085],
    #                            [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40])

    # ALL EFFECTS
    colson_alleffects = [0.22685912, 0.11102502]
    crossley_alleffects = [3.79894459, 0.15610441]
    ### colson_allparameters_all_effects = [0.12283884,  0.07007302,  0.49779157,  1.15249192,  1.06527212,  0.73440198, 5.89587047, -1.13272539]
    ### crossley_allparameters_alleffects = [0.12566585, 0.89574371, 0.50515977, 0.96263961, 0.6770551,  5.00000062]
    ### crossley_allparametersXinit_alleffects = [2.96673831, 0.13430309, 0.5909787,  1.02802739]
    # ALL PARAMETERS
    crossley_alleffects_noOmega = [3.37692885, 0.13516617, 0.56209632, 1.02117005, 5.00000071]
    crossley_alleffects_constrained_opt = [2.09565483, 0.12681223, 0.68466201, 1.04212047, 5.00000303]  # SSR in this case: 0.7128865746953651
    ### colson_alleffects_noOmega = [0.19242947, 0.10659566, 0.50754267, 1.64046981, 1.06215741, 0.94978169]
    colson_alleffects_conversion = [0.18469128, 0.10370328, 0.44498306, 1.84415783, 1.053272, 1.51377626, 0.51384825]
    # runIDs_all_effects = list(range(1, 51)) + list(range(101, 151))
    # nice_plot_of_fitted_models(20, [9.14542447, 0.21333374], [0.81316856, 0.14146783], list(range(1, 51)))
    nice_plot_of_fitted_models_various_t([0, 20, 40, 60], crossley_alleffects, colson_alleffects, runIDs_all_effects)
    # find_intersections(60, [9.14542447, 0.21333374], [0.81316856, 0.14146783], list(range(1, 51)))
    # wave_speed_evolution, mean_wave_speed = wave_speed(list(range(0, 61, 10)), [9.14542447, 0.21333374], [0.81316856, 0.14146783], list(range(1, 51)))
    # plot_wave_speeds(list(range(0, 61, 5)), crossley_alleffects, colson_alleffects, runIDs_all_effects, "full CPM-FEM model", with_std=False)
    # nice_plot_of_fitted_models_various_t([0, 20, 40, 60], crossley_alleffects_noOmega, colson_alleffects_conversion, runIDs_all_effects)
    # nice_plot_of_fitted_models_various_t([0, 20, 40, 60], crossley_alleffects_constrained_opt, colson_alleffects_conversion, runIDs_all_effects)
    # plot_wave_speeds(list(range(0, 61, 5)), crossley_alleffects_constrained_opt, colson_alleffects_conversion, runIDs_all_effects, "full CPM-FEM model", with_std=False)

    # NO DUROTAXIS
    crossley_nodurotaxis = [3.26470732, 0.13456643]
    colson_nodurotaxis = [0.18804447, 0.09804925]
    # colson_nodurotaxis_noOmega = [0.17084346, 0.09102945, 0.5161239,  1.61262684, 1.09629376, 1.25813569]
    crossley_nodurotaxis_noOmega = [2.55357492, 0.12064769, 0.54480928, 1.0327499,  5.00000069]
    colson_nodurotaxis_conversion = [0.16520003, 0.08789593, 0.49688464, 1.64552291, 1.0937287,  1.55612395, 0.34248735]
    # runIDs_no_durotaxis = list(range(51, 101)) + list(range(151, 201))
    # nice_plot_of_fitted_models_various_t([0, 20, 40, 60], crossley_nodurotaxis, colson_nodurotaxis, runIDs_no_durotaxis)
    # plot_wave_speeds(list(range(0, 61, 5)), crossley_nodurotaxis, colson_nodurotaxis, runIDs_no_durotaxis, "CPM-FEM model without durotaxis")
    # nice_plot_of_fitted_models_various_t([0, 20, 40, 60], crossley_nodurotaxis_noOmega, colson_nodurotaxis_conversion, runIDs_no_durotaxis)
    # plot_wave_speeds(list(range(0, 61, 5)), crossley_nodurotaxis_noOmega, colson_nodurotaxis_conversion, runIDs_no_durotaxis,
    #                  "CPM-FEM model without durotaxis")

    # statistical_analysis_wavespeed(list(range(0, 61, 5)), runIDs_all_effects, runIDs_no_durotaxis)

    # NO ATTACHMENT COST
    colson_noattachment = [0.27812403, 0.12191855]
    crossley_noattachment = [5.61658284, 0.16367495]
    crossley_noattachment_noOmega = [4.48195667, 0.1647455,  0.58525991, 0.9762381,  5.00000067]
    colson_noattachment_conversion = [0.22386147, 0.13049048, 0.48744022, 1.5910783, 1.00205085, 0.309451, 0.49634715]
    # runIDs_no_attachment = list(range(201, 301))
    # nice_plot_of_fitted_models_various_t([0, 20, 40, 60], crossley_noattachment, colson_noattachment, runIDs_no_attachment)
    # plot_wave_speeds(list(range(0, 61, 5)), crossley_noattachment, colson_noattachment, runIDs_no_attachment, "CPM-FEM model without attachment cost")
    # nice_plot_of_fitted_models_various_t([0, 20, 40, 60], crossley_noattachment_noOmega, colson_noattachment_conversion, runIDs_no_attachment)
    # plot_wave_speeds(list(range(0, 61, 5)), crossley_noattachment_noOmega, colson_noattachment_conversion,
    #                  runIDs_no_attachment, "CPM-FEM model without attachment cost")

    # statistical_analysis_wavespeed(list(range(0, 61, 5)), runIDs_no_attachment, runIDs_all_effects)

    # large domain experiment
    colson_LDE = [0.01248872, 0.00555775]
    crossley_lde = [0.08705253, 0.01643115]
    # nice_plot_of_fitted_models_specific_run_one_t(600, crossley_lde, colson_LDE)
