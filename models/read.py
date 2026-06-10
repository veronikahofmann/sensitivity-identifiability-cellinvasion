import os
from parameters import numele


def read_increment(path):
    """
    Reads the file where the current increment is saved, provided the run is still ongoing.
    """
    incr = 0

    try:
        with open(os.path.join(path, "increment_number.out"), "r") as ifp:
            incr = int(ifp.read())
            incr += 1  # Increment because this iteration was already finished when written
    except FileNotFoundError:
        pass  # If the file does not exist, incr remains 0

    return incr


def read_cells(pv, increment, path):
    """
    Reads the current configuration.
    """
    NRc = 0

    try:
        with open(os.path.join(path, f"ctags{increment}.out"), "r") as ifp:
            v = 0
            for line in ifp:
                for value in line.split():
                    anint = int(value)
                    pv[v].ctag = anint
                    v += 1
                    NRc = max(anint, NRc)

            if v != numele:
                print("\nERROR while loading densities")
    except FileNotFoundError:
        pass  # File not found, NRc will be zero

    return NRc

