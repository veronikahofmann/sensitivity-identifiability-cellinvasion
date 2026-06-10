import numpy as np
import math
from parameters import poisson, youngs, voxsize, NVX, NNX


def set_matrix_B(pB, r, s):
    """
    calculates the strain displacement matrix (same for all elements if they have the same shape and size)
    :param pB: copy of the strain displacement matrix B: np.zeros((3, 8))
    :param r: r,s are the local coordinates in the isoparametric element (Gauss integration points)
    :param s: r,s are the local coordinates in the isoparametric element (Gauss integration points)
    """
    kr = [-1, 1, 1, -1]
    ks = [-1, -1, 1, 1]
    # shape function and derivatives in point r, s
    dNdx = np.zeros(4)
    dNdy = np.zeros(4)

    for i in range(4):
        # value of the shape function belonging to node i
        # N[i] = 0.25 * (1 + kx[i] * localx) * (1 + ky[i] * localy);
        dNdx[i] = (2 / voxsize) * 0.25 * kr[i] * (1 + ks[i] * s)
        dNdy[i] = (2 / voxsize) * 0.25 * ks[i] * (1 + kr[i] * r)

        # calculate strain displacement matrix B
        pB[0][2 * i] = dNdx[i]
        pB[1][(2 * i) + 1] = dNdy[i]
        pB[2][2 * i] = dNdy[i]
        pB[2][(2 * i) + 1] = dNdx[i]

    return pB


def set_klocal():
    """
    calculates the local element stiffness matrix K_(e)
    :return: K_(e) (8x8 matrix)
    """
    klocal = np.zeros((8, 8))

    # node positions in local coordinate system
    # nx = [-1, 1, 1, -1];
    # ny = [-1, -1, 1, 1];

    # two-point Gaussian integration
    # local coordinates of the integration points(1 / sqrt(3) = 0.57735027
    intgrx = [-0.57735, 0.57735, 0.57735, -0.57735]
    intgry = [-0.57735, -0.57735, 0.57735, 0.57735]

    def material_matrix(pD):
        """
        calculates the stiffness matrix of the material (linear elastic isotropic)
        :param pD: copy of the material matrix D: np.zeros((3, 3))
        """
        planestress = True
        if planestress:
            es = youngs / (1 - poisson * poisson)
            pD[0][0] = es * 1
            pD[1][1] = es * 1
            pD[0][1] = es * poisson
            pD[1][0] = es * poisson
            pD[2][2] = es * 0.5 * (1 - poisson)
        else:  # plane strain
            es = youngs / ((1 + poisson) * (1 - 2 * poisson));
            pD[0][0] = es * (1 - poisson)
            pD[1][1] = es * (1 - poisson)
            pD[0][1] = es * poisson
            pD[1][0] = es * poisson
            pD[2][2] = es * 0.5 * (1 - 2 * poisson)

        return pD

    D = material_matrix(np.zeros((3, 3)))  # material matrix

    for i in range(4):
        B = set_matrix_B(np.zeros((3, 8)), intgrx[i], intgry[i]) # // calculate matrix B in integration point i
        Bt = B.transpose()
        BD = Bt @ D  # Bt is 8x3, D is 3x3
        BDB = BD @ B  # BD is 8x3, B is 3x8

        # Integration over the volume: this leads to adding to local stiffness matrix for each integration point
        # dV = dx * dy * dz = det(J) * dr * ds * dt
        # For cubic voxel elements the volume represented by one integration point is equal to dV = (0.5 * voxsize) ^ 3
        dV = 0.25 * voxsize * voxsize

        # adding BDB*dV to the local stiffness matrix (klocal = sum_{i=1}^4 (BDBdV)_i, where i denotes the integration
        # points)
        klocal += BDB * dV

    return klocal


def get_estrains(pn, e):
    """
    Calculates the elastic strain within one element.
    :param pn: list of NOD() objects
    :param e: number of the element of interest.
    :returns: the strain vector \varepsilon = = (\varepsilon_{11}, \varepsilon_{22}, 2 \varepsilon_{12}).
    """
    # Define B matrix
    B = np.zeros((3, 8))
    set_matrix_B(B, 0, 0)

    vy = e // NVX
    vx = e % NVX

    # Determine corner node numbers of this voxel
    n00 = vx + vy * NNX
    n10 = (vx + 1) + vy * NNX
    n11 = (vx + 1) + (vy + 1) * NNX
    n01 = vx + (vy + 1) * NNX

    u = np.zeros(8)
    u[0] = pn[n00].ux
    u[1] = pn[n00].uy
    u[2] = pn[n10].ux
    u[3] = pn[n10].uy
    u[4] = pn[n11].ux
    u[5] = pn[n11].uy
    u[6] = pn[n01].ux
    u[7] = pn[n01].uy

    estrains = [0, 0, 0]

    # Strain-displacement relation
    for i in range(3):
        for j in range(8):
            estrains[i] += B[i][j] * u[j]

    return estrains


def get_princs(str, pL1, pL2, v1, v2, strain):
    """
    Calculates the eigenvalues and -vectors of the strain tensor.
    :param str: the strain vector, as returned by get_estrains().
    :param pL1: placeholder for the first eigenvalue.
    :param pL2: placeholder for the second eigenvalue.
    :param v1: placeholder for the first eigenvector.
    :param v2: placeholder for the first eigenvector.
    :param strain: some leftover parameter from an earlier version.
    :return: the eigenvalues and -vectors of the strain tensor.
    """
    xx, yy = str[0], str[1]
    if strain:
        xy = 0.5 * str[2]
    else:
        xy = str[2]

    if xy == 0:
        pL1 = xx
        v1[0], v1[1] = 1, 0
        pL2 = yy
        v2[0], v2[1] = 0, 1
    else:
        T = xx + yy  # Trace
        D = xx * yy - xy * xy  # Determinant
        T2D = T * T / 4 - D

        if T2D <= 0:
            pL1 = xx
            v1[0], v1[1] = 1, 0
            pL2 = yy
            v2[0], v2[1] = 0, 1
        else:
            sqT2D = math.sqrt(T2D)
            pL1 = T / 2 + sqT2D
            pL2 = T / 2 - sqT2D

            Q1 = (pL1 - xx) / xy
            R1 = math.sqrt(1 + Q1 * Q1)
            v1[0], v1[1] = 1 / R1, Q1 / R1

            Q2 = (pL2 - xx) / xy
            R2 = math.sqrt(1 + Q2 * Q2)
            v2[0], v2[1] = 1 / R2, Q2 / R2
    return pL1, pL2, v1, v2
