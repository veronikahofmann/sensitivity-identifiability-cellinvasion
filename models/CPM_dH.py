"""
calculate costs (dH) used in cellular Potts movement
"""
from parameters import NVX, youngs, JCM, JCC, targetvolume, inelasticity, MAXDHSTR, thresholdstiff, stiffsensitivity, \
    stiffeningstiff
from FE_local import get_estrains, get_princs
import math
import numpy as np


def calcdH(pv, pn, csize, xt, xs, pick, ttag, stag):
    """
    Calculates the Hamiltonian which is used for the decision rule.
    :param pv: the list of ELE() objects.
    :param pn: the list of NOD() objects.
    :param csize: the list of cellsizes.
    :param xt: target element.
    :param xs: starting element.
    :param pick: the spatial relation of xs to xt.
    :param ttag: ctag of the retracting cell (target element)
    :param stag: ctag of the expanding cell (starting element)
    :return: the change in system energy provided the occupant of xs expands to xt.
    """
    dHcontact = calcdHcontact(pv, xt, ttag, stag)
    dHvol = calcdHvol(csize, ttag, stag)
    dHstr = calcdHstrain(pn, xt, xs, pick, ttag, stag)
    dH = dHcontact + dHvol + dHstr  # to ignore one of the effects, just delete the corresponding summand
    return dH


def calcdHcontact(pv, xt, ttag, stag):
    """
    Check the contact cost for a move from stag to ttag (element no. xt).
    """
    dHcontact, Hcontact, Hcontactn = 0.0, 0.0, 0.0
    nbs = [0] * 8

    nbs[0] = xt - 1 + NVX
    nbs[1] = xt + NVX
    nbs[2] = xt + 1 + NVX
    nbs[7] = xt - 1
    nbs[3] = xt + 1
    nbs[6] = xt - 1 - NVX
    nbs[5] = xt - NVX
    nbs[4] = xt + 1 - NVX

    for n in range(8):
        nbtag = pv[nbs[n]].ctag
        Hcontact += contactenergy(ttag, nbtag)  # contact energy if cell ttag remains at element xt
        Hcontactn += contactenergy(stag, nbtag)  # contact energy if cell stag expands to element xt

    dHcontact = Hcontactn - Hcontact  # dHcontact < 0 if a move costs less energy, > 0 if the current config. costs less
    return dHcontact


def contactenergy(tag1, tag2):
    """
    Calculates the necessary contact energy between the occupants of tag1 and tag2.
    """
    J = 0.0

    if tag1 != tag2:
        if tag1 <= 0 or tag2 <= 0:
            J = JCM
        else:
            J = JCC

    return J


def calcdHvol(csize, ttag, stag):
    """
    Check the cost in size change for an expansion from stag to ttag.
    """

    dHvolA = 0
    dHvolB = 0
    V0 = targetvolume

    if ttag > 0:  # Cell ttag retracts
        V = csize[ttag - 1]
        eV = (V - V0) / V0
        eVn = (V - 1 - V0) / V0
        dHvolA = inelasticity * (eVn * eVn - eV * eV)

    if stag > 0:  # Cell stag expands
        V = csize[stag - 1]
        eV = (V - V0) / V0
        eVn = (V + 1 - V0) / V0
        dHvolB = inelasticity * (eVn * eVn - eV * eV)

    dHvol = dHvolA + dHvolB
    return dHvol


def calcdHstrain(pn, xt, xs, pick, ttag, stag):
    """
    Calculates the durotactic cost of an expansion from xs to xt.
    """
    dHstrain = 0.0
    q = np.sqrt(0.5)
    vmx = [-q, 0, q, 1, q, 0, -q, -1]  # vmx = [-q, 0, q, q, 0, -q, -1, 1]
    vmy = [q, 1, q, 0, -q, -1, -q, 0]  # vmy = [q, 1, q, 0, 0, -1, -q, -q]
    vm = [vmx[pick], vmy[pick]]  # pick is the neighbour-ID in a range from 0 to 7 (counter-clockwise

    if stag > 0:  # Expansion
        estrains = get_estrains(pn, xt)
        L1, L2 = 0.0, 0.0
        v1, v2 = [0.0, 0.0], [0.0, 0.0]
        L1, L2, v1, v2 = get_princs(estrains, L1, L2, v1, v2, 1)
        vmv1 = vm[0] * v1[0] + vm[1] * v1[1]
        vmv2 = vm[0] * v2[0] + vm[1] * v2[1]
        E1 = youngs
        if L1 > 0:  # only non-negative stretch, i.e. substrate extension, leads to stiffening
            E1 *= (1 + L1 / stiffeningstiff)
        E2 = youngs
        if L2 > 0:
            E2 *= (1 + L2 / stiffeningstiff)
        dHstrain -= sige(E1) * vmv1 * vmv1 + sige(E2) * vmv2 * vmv2

    if ttag > 0:  # Retraction
        estrains = get_estrains(pn, xt)
        L1, L2 = 0.0, 0.0
        v1, v2 = [0.0, 0.0], [0.0, 0.0]
        L1, L2, v1, v2 = get_princs(estrains, L1, L2, v1, v2, 1)
        vmv1 = vm[0] * v1[0] + vm[1] * v1[1]
        vmv2 = vm[0] * v2[0] + vm[1] * v2[1]
        E1 = youngs
        if L1 > 0:  # only non-negative stretch, i.e. substrate extension, leads to stiffening
            E1 *= (1 + L1 / stiffeningstiff)
        E2 = youngs
        if L2 > 0:
            E2 *= (1 + L2 / stiffeningstiff)
        dHstrain += sige(E1) * vmv1 * vmv1 + sige(E2) * vmv2 * vmv2

    return dHstrain


def sige(L):
    """
    The durotaxis function h from the thesis.
    """
    x = stiffsensitivity * (L - thresholdstiff)
    sigL = MAXDHSTR / (1 + math.exp(-x))  # Sigmoid function
    # x = (L - beta)*(L - beta)
    # gamma_sq = gamma*gamma
    # sigL = alpha * math.exp(-x/(2*gamma_sq))  # Gaussian function
    return sigL