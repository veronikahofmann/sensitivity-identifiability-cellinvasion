import os
import matplotlib.pyplot as plt
import numpy as np


def oneD_output(path, L, t):
    """
        Returns the cell and ECM concentrations in 1D at time t.
        :param L: L is the radius of the largest circle that can be placed in the domain, given in pixels.
        """
    cell_spread_timesteps = []
    ECM_spread_timesteps = []
    str_save = np.loadtxt(os.path.join(path, 'concentration' + str(t) + '.out'))
    for i in range(L):
        cell_spread_timesteps.append(str_save[i])
    for i in range(L, 2 * L):
        ECM_spread_timesteps.append(str_save[i])
    return cell_spread_timesteps, ECM_spread_timesteps
