"""
initialize cells, impose external forces and fixations
"""
from parameters import NNY, NNX, NVX, NVY, targetvolume, sphere_rad_param, force, maxcells, ECMconcentr
import random
import math


def set_restrictions(pn):
    """
    Fixes the boundary nodes.
    :param pn: list filled with NOD-elements, i.e. the list of nodes
    """
    for ny in range(NNY):
        for nx in range(NNX):
            n = nx + ny * NNX
            if nx == 0 or nx == NNX - 1 or ny == 0 or ny == NNY - 1:
                pn[n].restrictx = True
                pn[n].restricty = True


def init_cells(pv):
    """
    Places the cells on the grid and returns their number. Note that no cells are being placed on the boundary elements.
    :param pv: list filled with ELE-elements, i.e. the list of elements.
    :return: NRc, the number of cells placed on the grid.
    """
    NRc = 0
    for vy in range(NVY):
        for vx in range(NVX):
            if maxcells:
                if NRc == maxcells:
                    return NRc
            v = vx + vy * NVX
            if (vx > 0) and (vx < NVX - 1) and (vy > 0) and (vy < NVY - 1):  # Exclude the outer rim
                r01 = random.random()
                d = 0.25 / targetvolume
                if r01 < d:
                    # if (vx == NVX // 2) and (vy == NVY // 2):
                    # if ((vx == NVX // 2 - 7) or (vx == NVX // 2 + 7)) and (vy == NVY // 2):
                    # dx = vx - NVX // 2
                    # dy = vy - NVY // 2
                    # d = math.sqrt(dx * dx + dy * dy)
                    # if (d < NVX / 8.0) and (r01 < 1.5 / TARGETVOLUME):
                    NRc += 1
                    pv[v].ctag = NRc

                    def find_free_neighbours(v):
                        """
                        Finds neighbouring elements of element v that are neither occupied by other cells nor located at
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
                                # check if the picked neighbour element is occupied by another cell
                                if pv[nbs[i]].ctag == 0:
                                    # check if the picked neighbour element is at the outer rim
                                    nbsy = nbs[i] // NVX
                                    nbsx = nbs[i] - nbsy * NVX
                                    if 0 < nbsx < NVX - 1 and 0 < nbsy < NVY - 1:
                                        invalidchoice = False
                                        return nbs[i]
                                nbs.remove(nbs[i])
                        else:  # if v is already in the outer rim:
                            return -1

                    settled = 1
                    reqspace = math.ceil(targetvolume)  # the required number of elements for this cell
                    occupied_by_current_cell = [v]
                    while settled < reqspace:
                        not_found = True
                        while not_found:
                            tries = 5000  # we randomly try 5000 initial configurations for each cell before giving up
                            while tries > 0:
                                j = random.randint(0, len(occupied_by_current_cell) - 1)
                                try:
                                    new_element = find_free_neighbours(occupied_by_current_cell[j])
                                    break
                                # sometimes, find_free_neighbours fails to find a free neighbour of
                                # occupied_by_current_cell[j] => then we pick a different j to start from
                                except ValueError:
                                    tries -= 1
                            if new_element > -1:
                                pv[new_element].ctag = NRc
                                occupied_by_current_cell.append(new_element)
                                settled += 1
                                not_found = False
    return NRc


def init_spheroid(pv):
    """
    Places the cells in a spheroid on the grid and returns their number. Note that no cells are being placed on the
    boundary elements.
    :param pv: list filled with ELE-elements, i.e. the list of elements.
    :return: NRc, the number of cells placed on the grid.
    """
    NRc = 0
    middleY = round(NVY / 2)
    middleX = round(NVX / 2)
    # middle = middleX + middleY * NVX
    if maxcells:
        sphere_rad = math.floor(math.sqrt((maxcells*targetvolume)/math.pi))
        if sphere_rad > middleX:
            raise ValueError(f"parameter maxcells is too large: the calculated sphere radius is {sphere_rad}, "
                             f"expected values below {middleX}")
    else:
        sphere_rad = sphere_rad_param
    for vy in range(NVY):
        for vx in range(NVX):
            v = vx + vy * NVX
            # check if v ist still free and lies inside spheroid
            if (pv[v].ctag == 0) and \
                    ((vx - middleX) * (vx - middleX) + (vy - middleY) * (vy - middleY) < sphere_rad * sphere_rad):
                NRc += 1
                pv[v].ctag = NRc

                # below here, the code is the same as in init_cells()
                def find_free_neighbours(v):
                    """
                    Finds neighbouring elements of element v that are neither occupied by other cells nor located at
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
                            # check if the picked neighbour element is occupied by another cell
                            if pv[nbs[i]].ctag == 0:
                                # check if the picked neighbour element is at the outer rim
                                nbsy = nbs[i] // NVX
                                nbsx = nbs[i] - nbsy * NVX
                                if 0 < nbsx < NVX - 1 and 0 < nbsy < NVY - 1:
                                    invalidchoice = False
                                    return nbs[i]
                            nbs.remove(nbs[i])
                    else:  # if v is already in the outer rim:
                        return -1

                settled = 1
                reqspace = math.ceil(targetvolume)  # the required number of elements for this cell
                occupied_by_current_cell = [v]
                while settled < reqspace:
                    not_found = True
                    while not_found:
                        tries = 5000  # we randomly try 5000 initial configurations for each cell before giving up
                        while tries > 0:
                            j = random.randint(0, len(occupied_by_current_cell) - 1)
                            try:
                                new_element = find_free_neighbours(occupied_by_current_cell[j])
                                break
                            # sometimes, find_free_neighbours fails to find a free neighbour of
                            # occupied_by_current_cell[j] => then we pick a different j to start from
                            except ValueError:
                                tries -= 1
                        if new_element > -1:
                            pv[new_element].ctag = NRc
                            occupied_by_current_cell.append(new_element)
                            settled += 1
                            not_found = False
                # only continue if maxcells has not been reached (provided maxcells is not None)
                if maxcells:
                    if NRc == maxcells:
                        return NRc
    return NRc


def init_ECM(pv):
    """
    Puts an ECM element on each element that is not occupied by a cell.
    All ECM elements have the same ctag: -1.
    :param pv: list filled with ELE-elements, i.e. the list of elements.
    :param ECM_concentration: the ECM concentration, default is 1.
    """
    for vy in range(NVY):
        for vx in range(NVX):
            v = vx + vy * NVX
            if pv[v].ctag == 0:
                r = random.random()
                if r < ECMconcentr:
                    pv[v].ctag = -1


def set_forces(pn):
    """
    If the parameter 'load' in parameters.py is set to something other than 0, the domain boundaries can be pre-loaded.
    """
    a = (0.0 / 6.0) * 3.1416

    for n in range(NNX * NNY):
        pn[n].fx = 0.0
        pn[n].fy = 0.0

    for ny in range(NNY):
        for nx in range(NNX):
            n = nx + ny * NNX

            # lower plate (iy==0) loading
            if ny == 0:
                pn[n].fx += math.sin(a) * math.cos(a) * force
                pn[n].fy += -math.cos(a) * math.cos(a) * force

            # upper plate (iy==NNY-1) loading
            if ny == NNY - 1:
                pn[n].fx += -math.sin(a) * math.cos(a) * force
                pn[n].fy += math.cos(a) * math.cos(a) * force

            # left plate (ix==0) loading
            if nx == 0:
                pn[n].fx += -math.sin(a) * math.sin(a) * force
                pn[n].fy += math.sin(a) * math.cos(a) * force

            # right plate (ix==NNX-1) loading
            if nx == NNX - 1:
                pn[n].fx += math.sin(a) * math.sin(a) * force
                pn[n].fy += -math.sin(a) * math.cos(a) * force

    for ny in range(NNY):
        for nx in range(NNX):
            n = nx + ny * NNX

            # for loading on the side of a plate, forces are lower
            if (nx == 0 or nx == NNX - 1) and (ny == 0 or ny == NNY - 1):
                pn[n].fx *= 0.5
                pn[n].fy *= 0.5
