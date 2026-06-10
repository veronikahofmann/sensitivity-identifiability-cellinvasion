import os
from parameters import NVX, NVY, NN, numele
from FE_local import get_princs, get_estrains
from analysis import cell_spread


def write_increment(increment, path):
    """
    Saves the current increment.
    """
    # if the file "increment_number.out" does not exist, it will be created
    with open(os.path.join(path, "increment_number.out"), "w") as ofp:
        ofp.write(str(increment))


def write_cells(pv, increment, path):
    """
    Saves the current configuration.
    """
    # Convert the increment value to a string
    astring = str(increment)

    # Create the filename based on the increment value
    with open(os.path.join(path, "ctags" + astring + ".out"), "w") as ofp:
        for vy in range(NVY):
            for vx in range(NVX):
                v = vx + vy * NVX
                ofp.write(str(pv[v].ctag) + " ")
            ofp.write("\n")


def write_pstrain(pv, pn, increment, path):
    """
    Saves the current strains.
    """
    estrains = [0.0, 0.0, 0.0]
    v1 = [0.0, 0.0]
    v2 = [0.0, 0.0]
    L1, L2 = 0.0, 0.0
    astring = str(increment)

    with open(os.path.join(path, "pstrain" + astring + ".out"), "w") as ofp:
        for v in range(numele):
            estrains = get_estrains(pn, v)
            L1, L2, v1, v2 = get_princs(estrains, L1, L2, v1, v2, 1)
            if L1 > L2:
                ofp.write(f"{int(1000000 * L1)} ")
                ofp.write(f"{int(1000 * v1[0])} ")
                ofp.write(f"{int(1000 * v1[1])} ")
                ofp.write(f"{int(1000000 * L2)}\n")
            else:
                ofp.write(f"{int(1000000 * L2)} ")
                ofp.write(f"{int(1000 * v2[0])} ")
                ofp.write(f"{int(1000 * v2[1])} ")
                ofp.write(f"{int(1000000 * L1)}\n")


def write_forces(pn, increment, path):
    """
    Saves the current traction forces.
    """
    with open(os.path.join(path, f"forces{increment}.out"), "w") as ofp:
        for n in range(NN):
            ofp.write(f"{pn[n].fx}\n")
        for n in range(NN):
            ofp.write(f"{pn[n].fy}\n")


def write_concentrations(pv, increment, path):
    """
    Saves the current cell- and ECM-densities.
    """
    L, cellspread, ecmspread = cell_spread(pv)
    with open(os.path.join(path, f"concentration{increment}.out"), "w") as ofp:
        for i in range(L):
            ofp.write(f"{cellspread[i]}\n")
        for i in range(L):
            ofp.write(f"{ecmspread[i]}\n")
    return L


def write_total_cellforces(average_cellforce, path):
    """
    Saves the total cellforce.
    """
    with open(os.path.join(path, f"average_cellforce.txt"), "a") as ofp:
        ofp.write(f"{average_cellforce}\n")
