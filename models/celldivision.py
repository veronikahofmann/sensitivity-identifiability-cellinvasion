from parameters import NVX, NVY, numele, targetvolume, prolifprob
import math
import copy
import random


def space_check(pv, stag, NRc):
    """
    Checks if there is enough space around the mother cell to place a daughter into.
    :param pv:
    :param stag: the ctag of the potential mother cell.
    :return: list of potential daughter cell element configurations
    """
    # identify all elements that are part of cell stag
    motherelm = []
    for v in range(numele):
        if pv[v].ctag == stag:
            motherelm.append(v)
    # identify the free elements directly next to the cell
    motherrim = []
    for v in motherelm:
        #
        #   v-1+NVX   v+NVX   v+1+NVX
        #     v-1       v       v+1
        #   v-1-NVX   v-NVX   v+1-NVX
        #
        nbs = [0] * 8
        nbs[0] = v - 1 + NVX
        nbs[1] = v + NVX
        nbs[2] = v + 1 + NVX
        nbs[7] = v - 1
        nbs[3] = v + 1
        nbs[6] = v - 1 - NVX
        nbs[5] = v - NVX
        nbs[4] = v + 1 - NVX
        for i in range(8):
            if pv[nbs[i]].ctag == 0:  # motherrim only contains free elements (no other cell, no ECM)
                motherrim.append(nbs[i])
    random.shuffle(motherrim)  # permutate motherrim s.t. the neighbours are in mean equally likely to be selected
    # the next lines are very similar to the code which initializes the cells. Starting from each of the free elements
    # next to the mother cell, we are checking if there is enough space for a daughter cell.

    def find_free_neighbours(v, pvc):
        """
        Finds neighbouring elements of element v that are neither occupied by other cells/ECM nor located at
        the outer rim.
        """
        nbs = [0] * 8  # Initialize neighbors of the target pixel
        #
        #   v-1+NVX   v+NVX   v+1+NVX
        #     v-1       v       v+1
        #   v-1-NVX   v-NVX   v+1-NVX
        #
        viy = v // NVX
        vix = v - viy * NVX
        if 0 < vix < NVX - 1 and 0 < viy < NVY - 1:  # excludes outer rim; should not lead to exceptions
            # as v is not at the outer rim
            nbs[0] = v - 1 + NVX
            nbs[1] = v + NVX
            nbs[2] = v + 1 + NVX
            nbs[7] = v - 1
            nbs[3] = v + 1
            nbs[6] = v - 1 - NVX
            nbs[5] = v - NVX
            nbs[4] = v + 1 - NVX
            invalidchoice = True
            while invalidchoice:
                # pick one of the neighbours at random
                i = random.randint(0, len(nbs) - 1)  # if len(nbs) = 0, this raises a ValueError
                # check if the picked neighbour element is occupied by another cell or ECM
                if pvc[nbs[i]].ctag == 0:
                    # check if the picked neighbour element is at the outer rim
                    nbsy = nbs[i] // NVX
                    nbsx = nbs[i] - nbsy * NVX
                    if 0 < nbsx < NVX - 1 and 0 < nbsy < NVY - 1:
                        invalidchoice = False
                        return nbs[i]
                nbs.remove(nbs[i])
        else:  # if v is already in the outer rim:
            return -1

    daughter_attempts = []
    reqspace = math.ceil(targetvolume)  # the required number of elements for a new cell
    for v in motherrim:
        # first, we create a copy of pv so that nothing is changed in pv
        pv_copy = copy.deepcopy(pv)
        daughter = [v]
        settled = 1
        # configpossible = True  # tells us whether there is still hope for a possible placement for a daughtercell
        tries_per_v = 100
        while (settled < reqspace) and (tries_per_v > 0):
            tries = 100  # we randomly try 100 initial configurations for each cell before giving up
            new_element = -1  # to avoid "UnboundLocalError: local variable 'new_element' referenced before assignment"
            while tries > 0:
                j = random.randint(0, len(daughter) - 1)
                try:
                    new_element = find_free_neighbours(daughter[j], pv_copy)
                    break
                # sometimes, find_free_neighbours fails to find a free neighbour of
                # occupied_by_current_cell[j] => then we pick a different j to start from
                except ValueError:
                    tries -= 1
            if new_element > -1:
                pv_copy[new_element].ctag = NRc + 1
                daughter.append(new_element)
                settled += 1
            else:
                tries_per_v -= 1
        if settled == reqspace:
            return daughter
    return []


def celldivision(pv, NRc):
    """
    Check which cells are elegible for division in terms of proliferation probability and available space. If both
    conditions are fulfilled, the function places a daughter cell next to the cell and sets the original cell's age back
    to 0.
    """
    current_NRc = NRc
    for vy in range(NVY):
        for vx in range(NVX):
            v = vx + vy * NVX
            if pv[v].ctag > 0 and pv[v].prolif == 0:  # only continue if the cell has not reproduced in the current iteration
                r = random.random()
                if r < prolifprob:
                    stag = pv[v].ctag
                    daughter = space_check(pv, stag, current_NRc)
                    if daughter:
                        # reset the proliferation watch of the cell stag in every element which it occupies
                        for el in range(numele):
                            if pv[el].ctag == stag:
                                pv[el].prolif = 1
                        current_NRc += 1
                        for dv in daughter:
                            pv[dv].ctag = current_NRc
                            pv[dv].prolif = 1  # the daughter is not supposed to proliferate right after birth
                    else:  # if no daughter cell could be placed, this cell will not have another chance in this cycle
                        for el in range(numele):
                            if pv[el].ctag == stag:
                                pv[el].prolif = 1
        # reset the prolif-property for all cells
        for e in range(numele):
            pv[e].prolif = 0
    return current_NRc
