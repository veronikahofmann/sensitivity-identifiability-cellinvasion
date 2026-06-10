from parameters import NVX, NVY, numele, voxsize, pixelinfl
import matplotlib.pyplot as plt
import math
import random


def elements_per_layer(xt, pixels):
    """
    Finds all elements that lie inside xt's area of influence and groups them in their respective layers.
    :param xt: starting element (the element where the ECM element of interest is present).
    :param pixels: the radius of the area of influence in elements. Can be a decimal (will be rounded down).
    :return: number of layers, [[elements in the closest layer], ..., [elements in the farthest layer]]
    """
    L = math.floor(pixels)  # it is assumed that one layer consists of a one-pixel-ring
    layerbounds = [voxsize*i for i in range(1, L+1)]
    # initialize the list containing the layer elements
    layercontents = [[] for _ in range(L)]
    # x and y coordinates of xt
    xty = xt // NVX
    xtx = xt - xty * NVX
    for vy in range(NVY):
        for vx in range(NVX):
            v = vx + vy * NVX
            dvy = (xty - vy) * voxsize  # y distance between v and xt
            dvx = (xtx - vx) * voxsize  # x distance between v and xt
            d = math.sqrt(dvy*dvy + dvx*dvx)  # total distance
            # only continue if element v is at least within the farthest layer
            if d < layerbounds[-1]:
                for i in range(L):
                    if layerbounds[i] >= d:
                        layercontents[i].append(v)
                        break
    return L, layercontents


def generate_weights(L, kind='exponential'):
    """
    Generates the weights w_i to calculate the influence of each layer.
    :param L: number of layers
    :param kind: the kind of weights. So far, only 'exponential' is available.
    :return: a list of L weights in descending order (to match the sorting from elements_per_layer)
    """
    weights = []
    if kind == 'exponential':
        # parameter of the decay steepness: a > 1 leads to a decrease for farther away layers and the larger a becomes,
        # the steeper the decay. a < 1 leads to an increase for farther away layers.
        a = 1.5
        for i in range(1, L+1):
            weights.append((1-a)/(1-pow(a, L)) * pow(a, i-1))
    weights.reverse()
    return weights


def degradeECM(pv):
    """
    Iterates over all elements. If the element is occupied by ECM, it's degradation probability is computed using the
    method from the thesis.
    """
    mt_random = random.SystemRandom()  # ignore seed() (not reproducible)
    for vy in range(NVY):
        for vx in range(NVX):
            v = vx + vy * NVX
            if pv[v].ctag > -1:  # don't bother if there is no ECM element, go to the next v
                continue
            L, areainfl = elements_per_layer(v, pixelinfl)
            w = generate_weights(L)
            # count the elements per layer that are occupied by cells
            cells_per_layer = []
            for layer in areainfl:
                cnt = 0
                for e in layer:
                    if pv[e].ctag > 0:
                        cnt += 1
                cells_per_layer.append(cnt)
            rel_occupancy = [cells_per_layer[i]/len(areainfl[i]) for i in range(0, L)]
            f = sum([w[i]*rel_occupancy[i] for i in range(0, L)])
            if f > 1:
                raise ValueError(f"value below or equal to 1 expected, got {f} at element {v}")
            r = mt_random.random()  # random number in [0, 1)
            if r < f:  # ECM element gets degraded
                pv[v].ctag = 0
