"""
cellular Potts movement and connectivity constraint
"""
from parameters import NVX, NVY, numele, immotility
from CPM_dH import calcdH
import random
import math


def splitcheckCCR(pv, csize, xt, ttag):
    """
    Checks if a cell movement violates the connectivity constraint.
    :param pv: our list of elements, as defined in main
    :param csize: list of cell sizes, i.e. the number of elements that is occupied by each cell
    :param xt: target pixel (= target element)
    :param ttag: target label
    :return: Boolean split. If False, a move to the target element xt is possible.
    """
    split = False
    #
    #   xt-1+NVX   xt+NVX   xt+1+NVX
    #     xt-1       xt       xt+1
    #   xt-1-NVX   xt-NVX   xt+1-NVX
    #
    nbs = [xt-1+NVX, xt+NVX, xt+1+NVX, xt-1, xt+1, xt-1-NVX, xt-NVX, xt+1-NVX]  # element IDs of the neighbours
    prev = pv[nbs[7]].ctag  # neighbour of xt at the lower right
    in_count = 0
    greys = []
    CCAlabels = [1] * numele
    CCAlabels[xt] = 3
    nrblue = -1
    startnb = -1

    for n in range(8):
        curr = pv[nbs[n]].ctag
        if prev != ttag and curr == ttag:  # we enter the cell from outside
            in_count += 1
        prev = curr

    if in_count > 1:  # there may be a split (we entered the current cell more than once; we don't count how often we leave)
        for v in range(numele):
            CCAlabels[v] = 1

        for n in range(8):
            nb = nbs[n]
            if pv[nb].ctag == ttag:
                CCAlabels[nb] = 0
                nrblue += 1
                startnb = nb

        CCAlabels[startnb] = 2
        nrgrey = 1
        greys.append(startnb)

        while nrgrey > 0 and nrblue > 0:
            nrgrey0 = nrgrey
            for i in range(nrgrey0):
                g = greys[i]
                nbsg = [g-1+NVX, g+NVX, g+1+NVX, g-1, g+1, g-1-NVX, g-NVX, g+1-NVX]
                for n in range(8):
                    nb = nbsg[n]
                    if pv[nb].ctag == ttag and CCAlabels[nb] < 2:
                        if CCAlabels[nb] == 0:
                            nrblue -= 1
                        CCAlabels[nb] = 2
                        nrgrey += 1
                        greys.append(nb)

            for i in range(nrgrey0):
                g = greys[i]
                CCAlabels[g] = 3
                greys[i] = greys[nrgrey - 1]
                nrgrey -= 1

        if nrblue:
            split = True

    return split


def CPM_moves(pv, pn, csize):
    """
    One call of this function is equivalent to "one Monte-Carlo step" as it is defined in the thesis. numele elements
    are randomly selected, then a neighboring element is randomly selected, and it is decided whether the occupant of
    the neighboring element expands to the element that was selected first, following the decision rule in the theses.
    """
    NRsteps = numele
    nbs = [0] * 8  # Initialize neighbors of the target pixel
    mt_random = random.SystemRandom()  # Initialize a secure random number generator, ignores seed()

    for _ in range(NRsteps):
        xt = mt_random.randint(0, numele - 1)  # Pick a random element
        xty, xtx = divmod(xt, NVX)

        if 0 < xtx < NVX - 1 and 0 < xty < NVY - 1:  # Exclude the outer rim
            nbs[0] = xt - 1 + NVX
            nbs[1] = xt + NVX
            nbs[2] = xt + 1 + NVX
            nbs[7] = xt - 1
            nbs[3] = xt + 1
            nbs[6] = xt - 1 - NVX
            nbs[5] = xt - NVX
            nbs[4] = xt + 1 - NVX

            pick = mt_random.randint(0, 7)  # Pick a random neighbor
            xs = nbs[pick]

            ttag = pv[xt].ctag  # ctag of the retracting cell (target element)
            stag = pv[xs].ctag  # ctag of the expanding cell (starting element)

            go_on = 0
            if ttag != stag:  # Don't bother if no difference
                go_on = 1
                if ttag > 0:  # if a cell in xt (retracting)
                    if splitcheckCCR(pv, csize, xt, ttag):  # check if moving from xs to xt splits the cell in xt
                        go_on = 0
                    if csize[ttag - 1] == 1:  # Cell cannot disappear (constraint may be removed)
                        go_on = 0
                if ttag == -1 or stag == -1:  # ECM elements cannot move, and cells cannot go where ECM elements are
                    go_on = 0

            if go_on:
                dH = calcdH(pv, pn, csize, xt, xs, pick, ttag, stag)
                try:
                    prob = math.exp(-immotility * dH)
                except OverflowError:
                    raise OverflowError(f"math range error, with xt = {xt}, ttag = {ttag}, xs = {xs}, stag = {stag} got dH = {dH}")
                if prob > (mt_random.random()):  # A move is made
                    pv[xt].ctag = stag
                    if ttag > 0:
                        csize[ttag - 1] -= 1
                    if stag > 0:
                        csize[stag - 1] += 1
