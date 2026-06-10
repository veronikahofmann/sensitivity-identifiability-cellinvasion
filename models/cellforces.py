"""
calculate cell traction forces
"""
from parameters import NNX, NNY, NVX, voxsize, cellforce
from write import write_total_cellforces
from numpy import sqrt


def cell_forces(pv, pn, csize, NRc, path):
    """
    Computes the traction forces of the cells and saves them as properties of the respective nodes.
    :param pv: the list of ELE() objects.
    :param pn: the list of NOD() objects.
    :param csize: the list of cellsizes.
    :param NRc: the total number of cells in the current configuration.
    :param path: the folder where the results are saved.
    """
    for ny in range(1, NNY - 1):
        for nx in range(1, NNX - 1):
            n = nx + ny * NNX
            pn[n].fx = 0
            pn[n].fy = 0

    totalF = 0
    for c in range(NRc):
        # Determine which nodes belong to cell c
        NRcelln = 0  # number of nodes in cell c
        cellnodes = []  # list of the nodes in cell c

        for ny in range(1, NNY - 1):
            for nx in range(1, NNX - 1):
                n = nx + ny * NNX
                cnttag = 0

                for vy in range(ny - 1, ny + 1):
                    for vx in range(nx - 1, nx + 1):
                        v = vx + vy * NVX

                        if pv[v].ctag == c + 1:
                            cnttag += 1

                if cnttag > 0:  # All cell nodes
                    cellnodes.append(n)
                    NRcelln += 1

        # Calculate forces between cell nodes
        for i in range(NRcelln):
            n = cellnodes[i]
            ny = n // NNX
            nx = n % NNX

            for j in range(NRcelln):
                n2 = cellnodes[j]
                dny = (n2 // NNX - ny) * voxsize  # y distance between n and n2, unit: [m]
                dnx = (n2 % NNX - nx) * voxsize  # x distance between n and n2

                forcex = cellforce * dnx
                forcey = cellforce * dny

                pn[n].fx += forcex
                pn[n].fy += forcey
        f_sum = 0
        for i in range(NRcelln):
            n = cellnodes[i]
            f_sum += abs(sqrt(pn[n].fx*pn[n].fx + pn[n].fy*pn[n].fy))
        totalF += f_sum
    write_total_cellforces(totalF/NRc, path)
