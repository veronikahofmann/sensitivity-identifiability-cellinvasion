import numpy as np
import os
import time
from parameters import today, numele, NNX, NNY, NDOF, NRINC, gammaMovAvg
from structures import ELE, NOD
from init_conditions import set_restrictions, init_cells, set_forces, init_spheroid, init_ECM
from read import read_increment, read_cells
from FE_local import set_klocal
from FE_assembly2 import assembly, arrange_dofpos, reduce_K
from FE_nodes2dofs import place_node_forces_in_f, set_disp_of_prev_incr, disp_to_nodes
from FE_solver import solvePCG
from write import write_cells, write_pstrain, write_increment, write_forces, write_concentrations
from cellforces import cell_forces
from cellmoves import CPM_moves
from ECM_degradation import degradeECM
from celldivision import celldivision
from plots import force_movie, strain_movie, plot_spread_timeseries

# define an ID for the run
runID = 1

# create folder for the results
newpath = r'RESULTS/{}-{}'.format(today, runID)
if not os.path.exists(newpath):
    os.makedirs(newpath)

### THE GRID IS INITIALIZED ###
# definition of elements and nodes
pv = [ELE() for n in range(numele)]
pn = [NOD() for m in range(NNX*NNY)]

### THE CELLS ARE PLACED ON THE GRID ###
# check if there is a simulation to continue
startincr = read_increment(newpath)
if startincr == 0:
    # start new simulation by placing cells on the grid - alternatively, you can use init_cells
    NRc = init_spheroid(pv)  # init_cells(pv)  # the number of cells on the grid (calculated through their sizes)
    # place the ECM elements on the grid
    init_ECM(pv)
else:
    # load ongoing simulation
    print(f"LOAD SIMULATION {today}-{runID} at iteration {startincr}")
    NRc = read_cells(pv, startincr, newpath)

csize = [0] * NRc  # csize stores the number of elements that is occupied by each cell
for v in range(numele):
    if pv[v].ctag > 0:
        csize[pv[v].ctag - 1] += 1

# adding forces (loads) to the elements and fixing the nodes on the domain boundary
set_forces(pn)
set_restrictions(pn)
dofpos, nrrdof = arrange_dofpos(pn)

# local K-matrix
klocal = set_klocal()
# global K-matrix
kcol = np.zeros(10*NDOF, dtype=int)
kval = np.zeros(10*NDOF, dtype=float)
# assembly
kcol, kval, klocal = assembly(kcol, kval, klocal, pv)
# reduction according to the DOFs
kcol, kval = reduce_K(kcol, kval, dofpos, nrrdof)

# simulation
mean_time_per_iteration = None
iteration = startincr

while iteration <= NRINC:
    start = time.time()
    # save the current configuration
    write_cells(pv, iteration, newpath)

    ### THE FORCES EXERTED BY THE CELLS ON THE GRID ARE COMPUTED ###
    cell_forces(pv, pn, csize, NRc, newpath)

    # force vector
    f = place_node_forces_in_f(pn, nrrdof)
    # estimate of the displacement vector using the displacements of the previous iteration
    u = set_disp_of_prev_incr(pn, nrrdof)

    ### THE GRID DEFORMATIONS ARE COMPUTED ###
    u = solvePCG(kcol, kval, u, f, nrrdof)
    # assign the calculated displacements to the nodes
    disp_to_nodes(pn, u)

    # save strains and forces every 5th iteration
    if iteration % 5 == 0:
        write_pstrain(pv, pn, iteration, newpath)
        write_forces(pn, iteration, newpath)
    # save cell- and ECM-distributions every 10th iteration
    if iteration % 10 == 0:
        L = write_concentrations(pv, iteration, newpath)

    ### THE NEW CONFIGURATION IS COMPUTED VIA ONE MONTE CARLO STEP ###
    CPM_moves(pv, pn, csize)

    ### THE ECM DEGRADATION IS COMPUTED ###
    degradeECM(pv)

    ### THE CELL PROLIFERATION IS COMPUTED ###
    NRc_new = celldivision(pv, NRc)
    if NRc_new > NRc:
        csize = [0] * NRc_new  # csize stores the number of elements that is occupied by each cell
        for v in range(numele):
            if pv[v].ctag > 0:
                csize[pv[v].ctag - 1] += 1
        NRc = NRc_new

    write_increment(iteration, newpath)  # document the finished iteration in case of interruption

    end = time.time()

    mean_time_per_iteration = (end - start) * (1 - gammaMovAvg) + mean_time_per_iteration * gammaMovAvg if mean_time_per_iteration else end - start
    if iteration % 1 == 0:
        print('\n step nr. {} took {}mins -> end approx. {}'.format(iteration, np.round((end - start) / 60, 2),
                                                                    time.ctime(time.time() + mean_time_per_iteration * (NRINC - iteration - 1))))

    iteration += 1

# delete the file that saves the current increment once the simulation is done
os.remove(os.path.join(newpath, 'increment_number.out'))

force_movie(newpath, runID)
strain_movie(newpath, runID)
plot_spread_timeseries(newpath, runID, L)
