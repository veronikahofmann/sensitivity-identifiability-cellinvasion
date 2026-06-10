from parameters import accuracy


def solvePCG(kcol, kval, u, f, nrrdof):
    """
    Solves the system K * u = f using the PCG algorithm.
    :return: the new displacement vector u
    """
    ui, ri, diag, invC, zi, pi, qi = [0] * nrrdof, [0] * nrrdof, [0] * nrrdof, [0] * nrrdof, [0] * nrrdof, [0] * nrrdof, [0] * nrrdof
    for i in range(nrrdof):  # for each row in K
        ui[i] = u[i]  # ui ist eventuell genau das gleiche wie u??
        diag[i] = kval[10*i]
        if diag[i] != 0.0:
            invC[i] = 1.0/diag[i]  # invC = inv(diag(K))
        else:
            invC[i] = 0.0

    def calc_Kdotx(kcol, kval, diag, x, b, nrrdof):
        """
        Dot product K*x = b
        """
        for r in range(nrrdof):
            b[r] = diag[r] * x[r]
        for r in range(nrrdof):
            lim = 10 * r + kcol[10 * r]
            for a in range(10 * r + 1, lim):
                b[r] += kval[a] * x[kcol[a]]
                b[kcol[a]] += kval[a] * x[r]

    calc_Kdotx(kcol, kval, diag, ui, qi, nrrdof)
    rhoinew, initrho = 0.0, 0.0
    for i in range(nrrdof):
        ri[i] = f[i] - qi[i]  # r0 = f-K*u0
        zi[i] = invC[i] * ri[i]  # z0 = inv(C)*r0
        pi[i] = zi[i]  # p0 = z0
        rhoinew += ri[i] * zi[i]  # rhoi = zi*ri
        initrho += invC[i] * f[i] * f[i]  # for accuracy
    # start iterative solve
    itera = 0
    while rhoinew > accuracy * initrho:
    # for itera in range(0, (rhoinew > accuracy * initrho)):
        rhoi = rhoinew
        calc_Kdotx(kcol, kval, diag, pi, qi, nrrdof)  #  qi = K*pi
        pq = sum([pi[i] * qi[i] for i in range(nrrdof)])
        alfi = rhoi / pq  # alfi = rhoi/(pi*qi)
        for i in range(nrrdof):
            ui[i] += alfi * pi[i]  # u_{i+1} = ui+alfi*pi
            ri[i] -= alfi * qi[i]  # r_{i+1} = ri-alfi*qi
            zi[i] = invC[i] * ri[i]  # z_{i+1} = inv(C)*r_{i+1}
        rhoinew = sum([ri[i] * zi[i] for i in range(nrrdof)])  # rho_{i+1} = r_{i+1} * z_{i+1}
        beti = rhoinew / rhoi  # beti = rhoinew/rhoi
        for i in range(nrrdof):
            pi[i] = zi[i] + pi[i] * beti  # p_{i+1} = z_{i+1} + betai * pi
        # if itera % 10 == 0:
        #     print("iteration {}, rhoinew/initrho = {}".format(str(itera), str(rhoinew / initrho)))
        itera += 1
    # print("Iteration stopped at iteration {}".format(str(itera)))
    for i in range(nrrdof):
        u[i] = ui[i]
    return u
