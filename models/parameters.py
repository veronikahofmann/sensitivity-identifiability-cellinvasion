"""
Here all parameter values are saved.
"""
import math
import datetime

today = str(datetime.date.today())  # either str(datetime.date.today()) or a date in the format "2023-12-16"

NVX = 60  # number of elements in x-direction
NVY = NVX  # number of elements in y-direction
numele = NVX * NVY  # total number of elements

NNX = NVX + 1  # number of nodes in x-direction
NNY = NVY + 1  # number of nodes in y-direction
NN = NNX * NNY  # total number of nodes
NDOF = 2 * NN  # degrees of freedom (each node has one DOF in x- and one in y-direction)
voxsize = 0.0000025  # side length of quadratic element in [m]
NRINC = 30  # number of iterations
gammaMovAvg = 0.95  # moving average param (for runtime estimation)

# material properties
poisson = 0.3  # 0.45  # for isotropic materials between 0 and 0.5 (0: maximal volume change, 0.5: incompressible)
youngs = 10  # realistic values for cell culture substrates: 0.5-32 kPa

# mechanical loading of the elements
load = 0
force = load*voxsize

# parameters for the FEM solver
accuracy = 0.00001
maxnriter = 1000

# cell properties
maxcells = 2  # maximal number of cells
sphere_rad_param = 4  # round(NVX / 10)  # given in elements; here: spheroid diameter = 1/5 of the domain length
celldiam = 0.000010  # 0.000020  # cell diameter [m]
cellrvox = (celldiam/2)/voxsize  # cell radius [pixels]
targetvolume = math.pi*cellrvox*cellrvox  # targetvolume, i.e. the area occupied by the cell [pixels]
inelasticity = 500.0  # strength of volume constraint [/m4]
immotility = 1.0

# adhesive properties
nostickJ = 500000.0  # [/m] contact penalty for none-adhesive surface
JCM = nostickJ*voxsize  # cell-medium (1.25)
JCC = 2.5  # 2.5*JCM  # cell-cell

# mechanic properties of cell movements
MAXDHSTR = 10.0  # maximum guidance term, unscaled at the moment
thresholdstiff = 14.5  # 15e3  # threshold stiffness for durotaxis [kPa]
stiffsensitivity = 0.5  # 0.0005  # steepness of durotaxis sigmoid [/kPa]
stiffeningstiff = 0.1  # steepness of strain-stiffening
thickness = 10e-6  # [m] effective thickness of substrate (= 1 micrometer)
cellforce = 0.01  # 1.0e-5/thickness  # [N/m] the same as mu in van Oers paper

# parameters for the ECM degeneration
distanceinfl = 1.5*celldiam  # distance between ECM element and farthest influential site [m]
pixelinfl = distanceinfl/voxsize  # radius of area of protease influence [pixels]
ECMconcentr = 1.0  # ECM concentration at the moment of initialization

# parameters for the cell division
prolifprob = 0.0001  # the probability for each cell to proliferate after one Monte Carlo step

