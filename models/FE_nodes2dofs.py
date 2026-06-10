import numpy as np
from parameters import NN


def place_node_forces_in_f(pn, nrrdof):
    """
    Set up the force vector.
    """
    f = np.zeros(nrrdof)
    cnt = 0
    for n in range(NN):
        if not pn[n].restrictx:
            f[cnt] = pn[n].fx
            cnt += 1
        if not pn[n].restricty:
            f[cnt] = pn[n].fy
            cnt += 1
    return f


def set_disp_of_prev_incr(pn, nrrdof):
    """
    Prepare and estimate of the next displacement vector u using the last u.
    """
    u = np.zeros(nrrdof)
    cnt = 0
    for n in range(NN):
        if not pn[n].restrictx:
            u[cnt] = pn[n].ux
            cnt += 1
        if not pn[n].restricty:
            u[cnt] = pn[n].uy
            cnt += 1
    return u


def disp_to_nodes(pn, u):
    """
    Save the calculated displacements in the list of nodes.
    """
    n = 0
    cnt = 0
    for n in range(NN):
        pn[n].ux = 0.0
        pn[n].uy = 0.0
        if not pn[n].restrictx:
            pn[n].ux = u[cnt]
            cnt += 1
        if not pn[n].restricty:
            pn[n].uy = u[cnt]
            cnt += 1
