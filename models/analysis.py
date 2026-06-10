import os
from parameters import NVX, NVY, numele
from ECM_degradation import elements_per_layer
from read import read_cells
from structures import ELE


def cell_spread(pv):
    """
    Counts the cells and calculates their density per ring around the grid center.
    :param pv: the list with the ELE() objects.
    :return: two lists, a list with the structure [(# of cells in ring 1)/(# of elements in ring 1), ...] and one that
    does the same for the ECM elements.
    """
    # find the center element
    middleY = round(NVY / 2)
    middleX = round(NVX / 2)
    middle = middleX + middleY * NVX
    L, layercontents = elements_per_layer(middle, middleX)
    # count the elements per layer that are occupied by cells
    cells_per_layer = []
    ECM_per_layer = []
    for layer in layercontents:
        cellcnt = 0
        ECMcnt = 0
        for e in layer:
            if pv[e].ctag > 0:
                cellcnt += 1
            elif pv[e].ctag == -1:
                ECMcnt += 1
        cells_per_layer.append(cellcnt)
        ECM_per_layer.append(ECMcnt)
    return L, [cells_per_layer[i] / len(layercontents[i]) for i in range(0, L)], \
           [ECM_per_layer[i] / len(layercontents[i]) for i in range(0, L)]


def cell_spread_from_file(path, t):
    """
    Computes the concentrations of cells and ECM just like cell_spread(), but for a completed run.
    :param path: folder in the format RESULTS/2023-12-22-runID
    :param t: the increment
    :return: the number of levels
    """
    pv_file = [ELE() for n in range(numele)]
    read_cells(pv_file, t, path)
    L, cellspread, ecmspread = cell_spread(pv_file)
    with open(os.path.join(path, f"concentration{t}.out"), "w") as ofp:
        for i in range(L):
            ofp.write(f"{cellspread[i]}\n")
        for i in range(L):
            ofp.write(f"{ecmspread[i]}\n")
    return L


def total_occupancy(pv):
    """
    Returns the percentage of elements that are occupied by cells (excluding the outer rim which is never touched by
    the cells).
    :param pv: the list with the ELE() objects.
    """
    elecnt = 0
    cellcnt = 0
    for vy in range(NVY):
        for vx in range(NVX):
            v = vx + vy * NVX
            if (vx > 0) and (vx < NVX - 1) and (vy > 0) and (vy < NVY - 1):  # Exclude the outer rim
                elecnt += 1
                if pv[v].ctag > 0:
                    cellcnt += 1
    return cellcnt / elecnt
