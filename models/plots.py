import os
from parameters import NVX, NVY, NNX, NNY, NN, voxsize, NRINC, youngs, today
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


def strain_movie(path, runID):
    """
    Visualizes the cells' movements and the strains.
    """
    NEX = NVX
    NEY = NEX
    NEX2 = 2 * NEX
    NEY2 = 2 * NEY

    # initialize first image
    M = []

    fig, ax = plt.subplots(figsize=(7, 7))

    # every fifth time step is plotted
    for incr in range(0, NRINC+1, 5):  # NRINC
        ctags = np.loadtxt(os.path.join(path, 'ctags' + str(incr) + '.out'))
        ctags = ctags.ravel()
        bigctags = np.zeros(
            (NEY2, NEX2))  # careful, bigctags augments the grid by factor two (for one element, four pixels are used=

        for ey in range(0, NEY):
            for ex in range(0, NEX):
                e = ex + ey * NEX
                tagval = ctags[e]
                bigctags[ey * 2:ey * 2 + 2, ex * 2:ex * 2 + 2] = tagval

        bigfield = np.full((NEY2, NEX2), 3)

        for ey in range(1, NEY2):
            for ex in range(1, NEX2):
                if bigctags[ey, ex] > 0:
                    bigfield[ey, ex] = 2
                if (bigctags[ey, ex] != bigctags[ey - 1, ex]) or (bigctags[ey, ex] != bigctags[ey, ex - 1]):
                    if (bigctags[ey, ex] > 0) or (bigctags[ey-1, ex] > 0) or (bigctags[ey, ex-1] > 0):
                        bigfield[ey, ex] = 1
                if bigctags[ey, ex] == -1:
                    bigfield[ey, ex] = 4

        # ax.clear()
        colors = ['firebrick', 'red', 'white', 'wheat']
        cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)
        im = ax.imshow(bigfield, cmap=cmap1, aspect='equal')
        ax.invert_yaxis()
        # plt.axis('off')

        str_save = np.loadtxt(os.path.join(path, 'pstrain' + str(incr) + '.out'))
        stri = str_save[:, 0] * 1e-10
        REF = 0.1
        strsc = stri / REF
        AMPL = 1e2  # scaling factor for the strain-lines
        strsc = strsc * AMPL

        SPACING = 1
        SHOWTR = 0.0  # element size 2.5e-06 * 2.5e-06

        xpos = []
        ypos = []
        strx = []
        stry = []

        for ey in range(0, NEY):
            for ex in range(0, NEX):
                e = ex + ey * NEX
                if stri[e] > SHOWTR:
                    xpos.append(ex * 2)
                    ypos.append(ey * 2)
                    strx.append(strsc[e] * str_save[e, 1] / 100)
                    stry.append(strsc[e] * str_save[e, 2] / 100)

        qv = plt.quiver(xpos, ypos, strx, stry, scale=1, color='black', headaxislength=0, headwidth=0, headlength=0)

        M.append([im, qv])

    ani = animation.ArtistAnimation(plt.gcf(), M, interval=40, blit=True)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=1, metadata=dict(artist='Me'), bitrate=1800)

    # create folder where the plot is supposed to be saved in
    savepath = r'PLOTS/{}'.format(today)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    ani.save(os.path.join(savepath, 'strains-{}_iterations{}_NVX{}_youngs{}.mp4'.format(runID, NRINC, NVX, youngs)),
             writer=writer)


def force_movie(path, runID):
    """
    Visualizes the cells' movements and the forces.
    """
    NEX = NVX
    NEY = NEX
    NEX2 = 2 * NEX
    NEY2 = 2 * NEY

    # initialize first image
    M = []

    fig, ax = plt.subplots(figsize=(7, 7))

    # every fifth time step is plotted
    for incr in range(0, NRINC+1, 5):  # NRINC
        ctags = np.loadtxt(os.path.join(path, 'ctags' + str(incr) + '.out'))
        ctags = ctags.ravel()
        bigctags = np.zeros(
            (NEY2, NEX2))  # careful, bigctags augments the grid by factor two (for one element, four pixels are used=

        for ey in range(0, NEY):
            for ex in range(0, NEX):
                e = ex + ey * NEX
                tagval = ctags[e]
                bigctags[ey * 2:ey * 2 + 2, ex * 2:ex * 2 + 2] = tagval

        bigfield = np.full((NEY2, NEX2), 3)

        for ey in range(1, NEY2):
            for ex in range(1, NEX2):
                if bigctags[ey, ex] > 0:
                    bigfield[ey, ex] = 2
                if (bigctags[ey, ex] != bigctags[ey - 1, ex]) or (bigctags[ey, ex] != bigctags[ey, ex - 1]):
                    if (bigctags[ey, ex] > 0) or (bigctags[ey - 1, ex] > 0) or (bigctags[ey, ex - 1] > 0):
                        bigfield[ey, ex] = 1
                if bigctags[ey, ex] == -1:
                    bigfield[ey, ex] = 4

        # ax.clear()
        colors = ['firebrick', 'red', 'white', 'wheat']
        cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)
        im = ax.imshow(bigfield, cmap=cmap1, aspect='equal')
        ax.invert_yaxis()
        plt.axis('off')

        str_save = np.loadtxt(os.path.join(path, 'forces' + str(incr) + '.out'))
        SHOWTR = 1e-7  # element size 2.5e-06 * 2.5e-06
        AMPL = 1e4  # scaling factor for the force arrows

        xpos = []
        ypos = []
        strx = []
        stry = []

        for ny in range(1, NNY+1):
            for nx in range(1, NNX+1):
                n = nx + ny * NNX
                if abs(str_save[n]) > SHOWTR:
                    xpos.append(nx * 2 - 0.5)
                    ypos.append(ny * 2 - 0.5)
                    strx.append(str_save[n] * AMPL)
                    stry.append(str_save[n + NN] * AMPL)

        qv = plt.quiver(xpos, ypos, strx, stry, scale=1, color='blue')

        M.append([im, qv])

    ani = animation.ArtistAnimation(plt.gcf(), M, interval=40, blit=True)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=1, metadata=dict(artist='Me'), bitrate=1800)

    # create folder where the plot is supposed to be saved in
    savepath = r'PLOTS/{}'.format(today)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    ani.save(os.path.join(savepath, 'forces-{}_iterations{}_NVX{}_youngs{}.mp4'.format(runID, NRINC, NVX, youngs)),
             writer=writer)


def plot_spread_timeseries(path, runID, L):
    """
    Plots the cell concentration in 1D for various times.
    :param L: L is the number of layers determined in the function analysis.cell_spread().
    """
    cell_spread_timesteps = []
    ECM_spread_timesteps = []
    timesteps = range(0, NRINC+1, 10)
    for incr in timesteps:
        str_save = np.loadtxt(os.path.join(path, 'concentration' + str(incr) + '.out'))
        temp1 = []
        temp2 = []
        for i in range(L):
            temp1.append(str_save[i])
        for i in range(L, 2*L):
            temp2.append(str_save[i])
        cell_spread_timesteps.append(temp1)
        ECM_spread_timesteps.append(temp2)
    fig, ax = plt.subplots()
    x_max = len(cell_spread_timesteps[0])
    x = range(x_max)
    for c in range(len(cell_spread_timesteps)):
        alp = (c+1)/len(cell_spread_timesteps)
        lab = 't = ' + str(timesteps[c])
        ax.plot(x, cell_spread_timesteps[c], color='blue', alpha=alp, label=lab)
        ax.plot(x, ECM_spread_timesteps[c], color='wheat', alpha=alp)
    ax.legend()
    ax.set_xlabel('x (distance from center)')
    ax.set_ylabel('cell and ECM density')
    savepath = r'PLOTS/{}'.format(today)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    plt.savefig(os.path.join(savepath, 'cell_spread_run_{}'.format(runID)))

