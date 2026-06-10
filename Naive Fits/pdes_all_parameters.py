from scipy.integrate import solve_ivp
import numpy as np


# parameters as defined in Crossley et al.
spacing = 0.1  # grid spacing, see \Delta in Appendix B of Crossley et al.
L = 30  # 100  # length of the domain
I = int(L / spacing)  # number of grid points
m_0 = 0.8
# alpha = 5
# sigma = 5
# omega = 1


def crossleys_model_all_params(T, k, r, D_cells, K, alpha):
    """
    Runs Crossley's model for a choice of parameters.
    :param T: time for which the PDE solution should be returned
    :param m_0: ECM density
    :param r: proliferation probability
    :param k: ECM degradation rate
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


def colsons_model_all_params(T, k, r, D_cells, M, K, sigma, conversion):
    """
    Runs Colson's model for a choice of parameters.
    :param T: time for which the PDE solution should be returned
    :param m_0: ECM density
    :param r: growth rate
    :param k: ECM degradation rate
    :return:
    """

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