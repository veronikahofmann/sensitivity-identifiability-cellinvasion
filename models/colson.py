"""
This module is not part of the CPM-FEM model. It is an implementation of Colson's parameterized PDE model. Executing the
script solves the equations and plots them for three time steps.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# parameters as defined in Colson et al. (with discretization parameters from Crossley et al.)
spacing = 0.1  # grid spacing, see \Delta in Appendix B of Crossley et al.
L = 50  # length of the domain
I = int(L/spacing)  # number of grid points
sigma = 0.2  # length of the domain that is initially occupied by cells
omega = 0.1  # sharpness of the initial boundary between the tumour and healthy tissue
D_cells = 0.5  # diffusivity of tumour cells in the absence of ECM
M = 1  # ECM density that inhibits all tumour cell movement
m_0 = 0.8  # ECM density
K = 1  # carrying capacity
r = 0.1  # growth rate
k = 2  # degradation rate
T_max = 50  # largest point in time for which the solution is supposed to be calculated


def colson_parameterized_ode(t, vals):
    """
    This is the spatially discretized, parameterized version of Colson's model.
    :param t: a placeholder, necessary for the calling signature of scipy.integrate.solve_ivp.
    :param vals: the discretized values of u and m of the previous time step. This has to be an array of length 2*I
    (= number of equations).
    :return: [u_0, u_1, ..., u_{I-1}, m_0, m_1, ..., m_{I-1}]
    """
    u = vals[0:I]
    m = vals[I:2*I]
    u_out = np.zeros(I)
    u_out[0] = D_cells / (spacing * spacing) * (u[1] - u[0]) + u[0] * (1 - u[0] - m[0])
    u_out[I - 1] = D_cells / (spacing * spacing) * (u[I - 2] - u[I - 1] - u[I - 2] * m[I - 1] + u[I - 1] * m[I - 2]) + u[I - 1] * (1 - u[I - 1] - m[I - 1])
    u_out[1:I - 1] = D_cells / (spacing * spacing) * ((2 - m[0:I - 2] / M - m[1:I - 1] / M) * u[0:I - 2] - (4 - m[0:I - 2] / M - 2 * m[1:I - 1] / M - m[2:I] / M) * u[1:I - 1] + (2 - m[1:I - 1] / M - m[2:I] / M) * u[2:I]) + u[1:I - 1] * r * (1 - u[1:I - 1] / K)
    m_out = -k * m * u
    return np.ravel([u_out, m_out])


# define the initial values (see equation (4.1) in Colson et al.)
u0 = np.zeros(I)
m0 = np.zeros(I)
for x in range(I):
    if x*spacing < sigma - omega:
        u0[x] = 1
    elif x*spacing < sigma:
        u0[x] = np.exp(1 - 1/(1 - ((x * spacing - sigma + omega)/omega)**2))
        m0[x] = m_0 * (1 - u0[x])
    if x*spacing >= sigma:
        m0[x] = m_0

init_vals = np.ravel([u0, m0])

# solve the space-discretized system
sol = solve_ivp(colson_parameterized_ode, [0, T_max], init_vals)

# plot at different points in time
timepoints = sol.t
middle_t = round(len(timepoints)/2)
end_t = len(timepoints) - 1

sol_time1 = sol.y[:, 0]  # u and m at this point in time are given by sol_time1[0:I] and sol_time1[I:2*I], resp.
sol_time2 = sol.y[:, middle_t]
sol_time3 = sol.y[:, end_t]

x = np.linspace(0, L, I)

plt.figure()
plt.subplot(131)
plt.plot(x, sol_time1[0:I], label="cell density", color='#eca1a6')
plt.plot(x, sol_time1[I:2*I], label="ECM density", color='#ada397', linestyle='dashed')
plt.legend()
plt.title('Solution of Colson et al.s model at t = ' + str(round(timepoints[0])))
plt.subplot(132)
plt.plot(x, sol_time2[0:I], color='#eca1a6')
plt.plot(x, sol_time2[I:2*I], color='#ada397', linestyle='dashed')
plt.title('Solution of Colson et al.s model at t = ' + str(round(timepoints[middle_t])))
plt.subplot(133)
plt.plot(x, sol_time3[0:I], color='#eca1a6')
plt.plot(x, sol_time3[I:2*I], color='#ada397', linestyle='dashed')
plt.title('Solution of Colson et al.s model at t = ' + str(round(timepoints[end_t])))
plt.show()