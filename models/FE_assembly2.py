import numpy as np
from parameters import NDOF, NVX, NVY, NNX, NNY, NN


def assembly(kcol, kval, klocal, pv):
    """
    Assembles element stiffness matrices into global stiffness matrix.
    """
    def nonzero(kcol, kval, ig, jg, value):
        if jg == ig:  # diagonal of K
            kval[10*ig] += value  # adding Ef*klocal[il][jl] to the diagonal of K
        elif jg > ig:  # upper triangle of K (jg is columns, ig is rows)
            lim = 10*ig + kcol[10*ig]  # last entry of kcol's line ig that is already filled
            alreadynonzero = False
            for a in range(10*ig+1, lim):  # going through line ig of K
                if kcol[a] == jg:  # in line ig of K, there is a nonzero entry in column jg
                    alreadynonzero = True
                    b = a
            if not alreadynonzero:  # it is still K[ig][jg] = 0
                kcol[lim] = jg  # adding column jg to the nonzero columns of line ig
                kval[lim] = value
                kcol[10*ig] += 1
            else:
                kval[b] += value

    for d in range(NDOF):
        kcol[10*d] = 1  # entries 0, 10, 20, ... of kcol say how many nonzero entries are in line 0, 1, 2, ... of K
        kval[10*d] = 0.0  # filling the diagonal of K with 0s

    for vy in range(NVY):  # going through the elements in y-direction
        for vx in range(NVX):  # elements in x-direction
            Ef = 1  # multiply klocal with Ef depending on local Young's modulus E

            # determine corner node numbers of this element; element nodes are numbered like this:
            #     n01  *  --  *  n11
            #          |      |
            #     n00  *  --  *  n10
            n00 = vx + vy*NNX
            n10 = (vx+1) + vy*NNX
            n11 = (vx+1) + (vy+1)*NNX
            n01 = vx + (vy+1)*NNX
            # giving each node its global 2 degrees of freedom
            # topv = [n00->, n00Î, n10->, n10Î, n11->, n11Î, n01->, n01Î]  (Î is an upwards arrow)
            topv = [2*n00, 2*n00+1, 2*n10, 2*n10+1, 2*n11, 2*n11+1, 2*n01, 2*n01+1]

            for il in range(8):
                for jl in range(8):
                    value = Ef*klocal[il][jl]
                    ig = topv[il]  # global row
                    jg = topv[jl]  # global column
                    nonzero(kcol, kval, ig, jg, value)

    print("\nASSEMBLY COMPLETED")
    # die nächste Zeile ist eventuell falsch, hab ich mir zum testen dazu überlegt
    return kcol, kval, klocal


def arrange_dofpos(pn):
    """
    Creates a list dofpos where each array element stands for a DOF. DOFs that are restricted get a -1, while the
    remaining DOFs get a number from 0 upwards.
    :param pn: the list of NOD-elements
    :return: dofpos and cnt
    """
    dofpos = []
    cnt = 0
    for n in range(NN):
        if pn[n].restrictx:
            dofpos.append(-1)
        else:
            cnt += 1
            dofpos.append(cnt)
        if pn[n].restricty:
            dofpos.append(-1)
        else:
            cnt += 1
            dofpos.append(cnt)
    return dofpos, cnt


def reduce_K(kcol, kval, dofpos, nrrdof):
    """
    For all DOFs with a -1 (= restricted), we need to remove the row and column from K
    """
    def clustercheck(dofpos):
        """
        Returns the values by which the column index needs to be reduced in the unrestricted nodes. Example:
        :param dofpos: if given by [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 2, 3, 4, -1, -1, -1, -1, 5, 6, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        :return: then reducer will be [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 10, 0, 0, 0, 0, 14, 14, 14, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        """
        reduce_by = [0] * len(dofpos)
        restriction_cnt = 0
        for i in range(len(dofpos)):
            if dofpos[i] == -1:
                restriction_cnt += 1
            else:
                reduce_by[i] = restriction_cnt
        return reduce_by
    reducer = clustercheck(dofpos)
    kcolnew = np.zeros(10 * nrrdof, dtype=int)
    kvalnew = np.zeros(10 * nrrdof, dtype=float)
    rn = 0
    for ro in range(NDOF):
        if dofpos[ro] == -1:
            lim = 10 * ro + kcol[10 * ro]  # lim = Anzahl der nicht-null-Einträge in Zeile ro
            # change everything to zero in this row
            for a in range(10 * ro, lim):  # läuft durch alle nicht-null-Einträge von kval in Zeile ro
                kcol[a] = 0
                kval[a] = 0.0
        else:  # Zeile in der wichtige Infos stehen
            # Spalten durchgehen
            for co in range(1, 10):
                if dofpos[kcol[10*ro + co]] == -1:  # if this column corresponds to a restricted DOF
                    kcol[10*ro + co] = 0
                    kval[10*ro + co] = 0.0
                    # reduce the counter of non-zero entries of the current row by one in kcol
                    kcol[10*ro] -= 1
                # Die nächsten drei Zeilen könnten auch falsch sein, bisher nur für 4 Elemente getestet
                else:  # if the column corresponds to an unrestricted DOF
                    if kcol[10*ro + co] > 0:  # only does something if the kval-entry is non-zero
                        kcol[10*ro + co] = kcol[10*ro + co] - reducer[kcol[10*ro + co]]  # - colshift
        # bei den Zeilen, bei denen nur noch der Diagonaleintrag ungleich null ist, steht in kcol 0 als erster Eintrag.
        # Das ist so nicht ganz richtig in unserem System und muss durch eine 1 ersetzt werden.
            if all(i == 0 for i in kcol[10*ro:10*(ro+1)]):  # if all entries in row ro are 0
                kcol[10*ro] = 1
            # befüllen der neuen, kleineren K-Matrix
            kcolnew[10*rn:10*(rn+1)] = kcol[10*ro:10*(ro+1)]
            kvalnew[10 * rn:10 * (rn + 1)] = kval[10 * ro:10 * (ro + 1)]
            rn += 1
    return kcolnew, kvalnew

