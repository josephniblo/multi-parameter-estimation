import numpy as np
import scipy.linalg as linalg
import itertools as it
import scipy.sparse as sparse
import cvxpy as cp

def genMat(dim):
    ''' genMat(dim) returns a dim x dim Hermitian PSD
    matrix of unit trace'''

    # seed rng
    np.random.seed()

    # generate random matrix
    B = np.array(((np.random.randn(dim, dim)) +
        1j*(np.random.randn(dim, dim))))

    # impose condition for positive semi-definite matrix
    B = (B.T.conj().copy() @ B.copy()) + 0.05*(np.identity(dim)/dim)

    # impose unit trace and return
    return (B/np.trace(B)).copy()

def allProjectors(num):
    ''' allProjectors(num) returns matrix of all Pauli basis projectors
    (in quadratic form) for a num-photon state space. Operators
    are grouped by basis in order:
        'Z..Z'->'Z..X'->'Z..Y'->... '''

    if num > 4:
        newl = '\n'
        print(f'Warning! Heavily memory intensive. Consider\
               {newl} using precalculated projectors!')
        # disallow for now
        # return

    # basis labels
    bases = ['Z', 'X', 'Y']

    # list of basis combinations for num photons
    lb = list(it.product(bases, repeat=num))

    # init output array
    out = retProj(''.join(lb[0]))

    for n in range(1, len(lb)):
        out = sparse.vstack([out, retProj(''.join(lb[n]))])

        # progress reporting
        print(f'{n}/{len(lb)} Complete!')

    return out

def pLab(s):
    ''' returns elements in a projector basis as list of strings'''

    lab = list()

    # iter var
    i = 0;

    # iterate through str to get corresponding projectors
    for elem in s:
        if elem == 'Z':
            lab.append(['H', 'V'])
        elif elem == 'X':
            lab.append(['D', 'A'])
        elif elem == 'Y':
            lab.append(['R', 'L'])
        else:
            print(f'Error: {elem} is not a valid basis label! (position {i})')

        i += 1

    # cartesian product of labels
    return list(it.product(*lab))


def retProj(str):
    ''' retProj(str) returns the projector set associated with
    the string str (in quadratic form).

    e.g.
        retProj('ZX') returns cartesian product of projectors
        'HVDA'  '''

    # init projector list and labels
    proj, lstr = list(), list()

    lstr = pLab(str)

    sz = len(lstr)

    # get projectors
    pr = quadfromLabel(''.join(lstr[0]))
    pr

    for n in range(1, sz):
        pr = sparse.vstack([pr, quadfromLabel(''.join(lstr[n]))])

    return pr

def quadfromLabel(lab):
    ''' quadfromLabel(lab) returns the quadratic form of the projector
    specified by string lab. Now with sparse matrices!!

    e.g.
        quadfromLabel('HH') = numpy.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    '''

    # define Z basis projectors
    H = sparse.lil_matrix(np.array([[1], [0]]))
    V = H[ ::-1, :].copy()

    # iteration var
    i = 0

    # initialise projector ket
    ket = sparse.lil_matrix(np.array([1]))

    # get projector from each photon label
    for elem in lab:
        # get current projector
        if elem == 'H':
            cp = H
        elif elem == 'V':
            cp = V
        elif elem == 'D':
            cp = (1/np.sqrt(2)) * (H + V)
        elif elem == 'A':
            cp = (1/np.sqrt(2)) * (H - V)
        elif elem == 'R':
            cp = (1/np.sqrt(2)) * (H + 1j*V)
        elif elem == 'L':
            cp = (1/np.sqrt(2)) * (H - 1j*V)
        else:
            print(f'Error: {elem} is not a valid projector label (position {i})')
            break

        ket = sparse.kron(ket, cp)

        i += 1

    # return operator in quadratic form
    return sparse.kron(ket, ket.transpose().conj()).reshape(1, -1, order='F').copy().tocoo()

def concurr(rho):
    ''' concurr(rho) returns concurrence (degree of entanglement) between
    states represented by a density matrix rho.'''

    # get eigenvalues of R operator
    lam = linalg.eigvals(R_op(rho))

    # rank eigenvalues
    lam.sort()

    # get no. eigenvals
    ne = lam.shape[0]

    # calculate concurrence
    C = np.sqrt(lam[-1] - np.sum(np.sqrt(lam[:-1])))

    # set to zero if negative
    if C < 0:
        C = 0

    # round concurrence
    C = np.real(np.round(C, decimals = 5))
    # print
    # print(f'States are {C*100:{3}.{1}f}% Entangled!')

    return C

def R_op(rho):
    ''' Applies operator: R = sqrt(rho) * sqrt(rho_au) where
    rho is a density matrix representing a mixed ensemble of qubits and
    rho_au is the anti-unitary transform of rho.
    '''

    # get no. qubits
    d = int(np.log2(rho.shape[0]))

    # generate anti-unitary operator for d-qubit space
    au_op = pauli(2)
    for n in range(d-1):
        au_op = np.kron(au_op, pauli(2))

    # anti-unitary transform of rho
    rho_au = au_op @ rho.conj() @ au_op

    # return R operator
    return linalg.sqrtm(rho) @ linalg.sqrtm(rho_au)

def pauli(n):
    ''' pauli(n) (n = 0:3) returns nth Pauli spin matrix'''

    if n == 0:
        sigma = np.array([[1, 0], [0, 1]])
    elif n == 1:
        sigma = np.array([[0, 1], [1, 0]])
    elif n == 2:
        sigma = np.array([[0, -1j], [1j, 0]])
    elif n == 3:
        sigma = np.array([[1, 0], [0, -1]])
    else:
        print(f'Error! {n} is not an integer in range [0, 3]!')

    return sigma

def linen(rho):
    ''' linen(rho) returns linear entropy of a given density matrix rho '''

    Sl = np.real(1 - np.trace(np.linalg.matrix_power(rho, 2)))

    # print(f'State is {(1-Sl)*100:{3}.{1}f}% Pure!')

    return Sl

def fid(target, recon):
    ''' fid(target, recon) returns fidelity of a reconstructed density
    operator 'recon' to a target operator 'target' '''

    print("target", target)
    print("recon", recon)

    # matrix sqrt of target state
    sqtar = linalg.sqrtm(target)

    print("sqtar", sqtar)

    # intermediate
    inter = sqtar @ recon @ sqtar

    # return fidelity
    F = (np.trace(linalg.sqrtm(inter)))**2

    return np.real(np.round(F, decimals=5))

def tomoSolve(m, P):
    ''' tomosolve(m, P) returns positive semi-definite matrix X
    which best fits measurements m corresponding to quadratic form
    projectors P.

    Arguments:
        > P: (n x d^2) matrix (where d is 2 ^ number of qubits in state)
             where each row is a projector in quadratic form.
        > m: (n x 1) vector of measurements corresponding to the projectors
             in P.

    Outputs:
        > X: (d x d) matrix corresponding to solution.
    '''

    # get d from projector length
    d = int(np.sqrt(P.shape[1]))

    # n.b. due to cvxpy's implementation of variable special properties,
    # we must define 2 variables with the positive semi-definite and
    # complex properties respectively and constrain them to be equal

    # initialise target variable with Hermitian constraint
    X = cp.Variable((d, d), hermitian=True)

    # create target var with complex value constraint
    x = cp.Variable((d, d), complex=True)

    # define objective
    obj = cp.Minimize(cp.norm((P @ cp.reshape(X, (d**2, 1), order='F'))-m))

    # herm&complex, PSD,  unit trace constraint
    const = [X == x, X >> 0, cp.trace(X)-1 == 0]

    # construct problem in cvxpy
    prob = cp.Problem(obj, const)

    # solve problem and update variable fields
    #prob.solve(verbose=True) # for statistics output
    prob.solve(solver = cp.SCS, eps=1e-8) # for no print
    # print exit status
    #print(f'Status: {prob.status}')

    # return solution
    return X.value.T

def exDat(DataFrame):
    '''Selects appropriate data extraction function for data in question.

    Arguments:
        > DataFrame: pandas dataframe of measurement data

    Outputs:
        > tomodic: dictionary of measurements indexed by projector label
    '''

    if len(DataFrame) == 9:
        tomodic = qb2FlipSolve(DataFrame)
    elif len(DataFrame) == 36:
        tomodic = qb2Solve(DataFrame)
    elif len(DataFrame) == 324:
        tomodic = qb4Solve(DataFrame)

    return tomodic

def qb2FlipSolve(qb2Flip):
    ''' Translates data from 2-qubit detector-flip scheme measurements
    to dict with projector labels as keys and (non-normalised)
    data as values. This form is used by dataSolve() to return
    a density matrix.

    Arguments:
        > qb2Flip: pandas dataframe containing 2-qubit flip data

    Outputs:
        > tomodic: dictionary of data indexed by associated projector

    '''

    # extract labels
    lab = list(qb2Flip['basis']) #change labels to basis in all 3 data extractions

    # remove trailing zero
    for i, elem in enumerate(lab):
        lab[i] = elem[:-1]

    # return individual projector labels
    pr = list()

    for el in lab:
        L = pLab(el)

        for i, p in enumerate(L):
            L[i] = ''.join(p)

        pr.extend(L)

    # get proj. measurement data from dataframe
    dat = np.array(qb2Flip.loc[:, 'ccTT':'ccRR']).astype(float)

    dat = np.reshape(dat, (-1, 1))

    # construct dictionary
    tomodic = { key : dat[i].item() for i, key in enumerate(pr)}

    return tomodic

def qb2Solve(qb2):
    ''' Translates data from 2-qubit scheme measurements
    to dict with projector labels as keys and (non-normalised)
    data as values. This form is used by dataSolve() to return
    a density matrix.

    Arguments:
        > qb2Flip: pandas dataframe containing 2-qubit data

    Outputs:
        > tomodic: dictionary of data indexed by associated projector

    '''

    # extract labels
    lab = list(qb2['basis'])

    # remove trailing zero
    for i, elem in enumerate(lab):
    #    lab[i] = elem[:-1] not needed
        lab[i] = elem

    # get proj. measurement data from dataframe
    dat = np.array(qb2.loc[:, 'cc12']).astype(float)

    dat = np.reshape(dat, (-1, 1))
    # construct dictionary
    tomodic = { key : dat[i].item() for i, key in enumerate(lab)}

    # get desired order for easy normalisation

    # basis labels
    lb = list(it.product(['Z', 'X', 'Y'], repeat=2))

    ord = list()

    # get individual projectors
    for k in lb:
        L = pLab(''.join(k))

        for i, p in enumerate(L):
            L[i] = ''.join(p)

        ord.extend(L)

    # reorder dict
    for key in ord:
        tomodic[key] = tomodic.pop(key)

    return tomodic

def qb4Solve(qb4):
    ''' Translates data from 4-qubit detector-flip scheme measurements
    to dict with projector labels as keys and (non-normalised)
    data as values. This form is used by dataSolve() to return
    a density matrix.

    Arguments:
        > qb4: pandas dataframe containing 2-qubit flip data

    Outputs:
        > tomodic: dictionary of data indexed by associated projector

    '''

    # extract labels from dataframe
    lab = list(qb4['basis'])

    # new list for expanded basis labels
    exlab = list()

    # loop through all entries
    for i, k in enumerate(lab):
        # isolate last 2 elements in basis label
        L = pLab(k[2:5])

        # loop through each projector in basis
        for j, v in enumerate(L):
            # build new projector string
            sP = ''.join([k[0:2], ''.join(v)])

            # append to list
            exlab.append(sP)

    # organise data
    udat = np.array(qb4.loc[:, 'ccTT':'ccRR']).astype(float) #unflipped
    fdat = np.array(qb4.loc[:, 'flipccTT':'flipccRR']) #flipped

    # flip flipped data
    fdat = np.flip(fdat, axis=1)

    # sum data
    dat = udat + fdat

    # reorder data to linear
    dat = np.reshape(dat, (-1, 1))

    # create dict
    tomodic = { key : dat[i].item() for i, key in enumerate(exlab)}

    # get proper order of basis labels for normalisation

    # basis labels
    lb = list(it.product(['Z', 'X', 'Y'], repeat=4))

    ord = list()

    # get individual projectors
    for k in lb:
        L = pLab(''.join(k))

        for i, p in enumerate(L):
            L[i] = ''.join(p)

        ord.extend(L)

    # reorder dict
    for key in ord:
        tomodic[key] = tomodic.pop(key)

    return tomodic

def dataSolve(datadic, *args):
    '''Normalises measurement data and calls convex solver.

    Arguments:

        > datadic: dictionary of data indexed by projector label

    Output:

        > rho: solved density matrix

        > tomodic: dictionary of normalised data indexed by prj. lab

    '''

    dat = np.array(list(datadic.values())).reshape(-1, 1)

    # get number of data points
    n = dat.shape[0]

    # get number of qubits from n
    nq = np.log(n)/np.log(6)

    # get number of measurements in a basis of n qubits
    bl = int(2**nq)

    # normalise
    tl = np.zeros([bl, 1])

    for i, v in enumerate(dat):
        tl[i%bl] = v

        if i%bl == bl-1:
            s = np.sum(tl)
            dat[i-(bl-1):i+1] = tl/s


    tomodic = { key : dat[i].item() for i, key in enumerate(list(datadic.keys()))}

    # get projectors from labels
    lb = list(tomodic.keys())

    sz = len(lb)
    if not args:

        # init output array
        proj = quadfromLabel(lb[0])

        # append projectors
        for n in range(1, sz):
            proj = sparse.vstack([proj, quadfromLabel(lb[n])])
    else:
        proj = args[0]

    # print(proj)

    # solve
    rho = tomoSolve(np.array(list(tomodic.values())).reshape((sz, 1)), proj.copy())

    return rho, tomodic

def labFlip(datadic, photon, basis):
    '''
    Flips specified photon mislabeled bases.

    Arguments:

        > datadic: dict of data indexed by projector label

        > photon: index of photon in question (1:n == Left:Right)

        > basis: mismatched basis label ['Z', 'X', 'Y']

    Output:

        > datadic

    '''

    # get list of projector labels
    lab = list(datadic.keys())

    # get number of data points
    n = len(lab)

    # get number of qubits from n
    nq = int(np.log(n)/np.log(6))

    # loop through labels
    for i, k in enumerate(lab):

        # convert string to char list for replacement
        tl = list(k)

        if basis == 'Z':

            if k[photon-1] == 'H':
                tl[photon-1] = 'V'
            elif k[photon-1] == 'V':
                tl[photon-1] = 'H'

        elif basis == 'X':

            if k[photon-1] == 'D':
                tl[photon-1] = 'A'
            elif k[photon-1] == 'A':
                tl[photon-1] = 'D'

        elif basis == 'Y':

            if k[photon-1] == 'R':
                tl[photon-1] = 'L'
            elif k[photon-1] == 'L':
                tl[photon-1] = 'R'

        lab[i] = ''.join(tl)


    vals = list(datadic.values())

    datadic = { key : vals[i] for i, key in enumerate(lab)}

    # get desired order for easy normalisation

    # basis labels
    lb = list(it.product(['Z', 'X', 'Y'], repeat=nq))

    ord = list()

    # get individual projectors
    for k in lb:
        L = pLab(''.join(k))

        for i, p in enumerate(L):
            L[i] = ''.join(p)

        ord.extend(L)

    # reorder dict
    for key in ord:
        datadic[key] = datadic.pop(key)

    return datadic

def poissGen(datadic, N):
    '''Samples from Poisson distribution with means as the values
    of datadic.

    Arguments:
        > datadic: dict of measurement counts indexed by projector label

        > N: number of times to repeat

    Outputs:
        > plist: list of dicts with new measurement values
    '''

    # extract data
    dat = list(datadic.values())
    # instantiate list for output
    plist = list()

    # repeat N times
    for i in range(0, N):

        # instantiate new dict
        tdic = dict()

        # generate new value sampled from poisson distribution
        for i, v in enumerate(datadic.keys()):
            # sample and add to temporary dict
            tdic[v] = float(np.random.poisson(int(dat[i])))

        # insert compiled dict into list
        plist.append(tdic)

    return plist

def monteCarlo(plist, target, stats):
    '''Calculates errors on fidelity, concurrence, and purity
    based on poissonian distributed projection simulations.

    Arguments:
        > plist: list of dicts with poissonian simulated data indexed
                 by projector label

        > target: target state (for fidelity calculation)

        > stats: dict of fidelity (F) and linear
                 entropy (S)

    Output:
        > dF: array of newly calculated fidelities

        > dS: array of newly calculated linear entropies
    '''

    # instantiate error for values in question
    Fe = Se = np.zeros(int(len(plist)))

    # generate projectors here (as remain invariant)
    lb = list(plist[0].keys())

    sz = len(lb)

    # init output array
    projj = quadfromLabel(lb[0])

    # append projectors
    for n in range(1, sz):
        projj = sparse.vstack([projj, quadfromLabel(lb[n])])

    # iterate through plist
    for i, v in enumerate(plist):
        print(f'{i}/{sz}')

        # convert ith data set to density matrix
        rho = dataSolve(v, projj.copy())[0]

        # generate key statistics
        Ft = fid(target, rho)
        St = linen(rho)

        Fe[i] = Ft
        Se[i] = St

        # replace error if necessary
        # if Fe < np.abs(stats['F'] - Ft):
        #     Fe = np.abs(stats['F'] - Ft)
        #
        # if Ce < np.abs(stats['C'] - Ct):
        #     Ce = np.abs(stats['C'] - Ct)
        #
        # if Se < np.abs(stats['S'] - St):
        #     Se = np.abs(stats['S'] - St)

    return Fe, Se

def GHZ(n):
    ''' Returns n-qubit GHZ state (vector form) in computational basis.

    Arguments:
        > n: integer specifying number of qubits

    Outputs:
        > state: n-qubit GHZ state in vector form
    '''

    # define H, V in comp. basis
    H = np.array([[1], [0]])
    V = np.array([[0], [1]])

    # generate n states in these bases
    c = H.copy()
    p = V.copy()

    for i in range(0, n-1):
        c = np.kron(c, H)
        p = np.kron(p, V)

    return (1/np.sqrt(2)) * (c + p)

def cumKron(l):
    ''' Returns cumulative Kronecker product of all list items
    e.g. cumKron((X, Y, Z)) = np.kron(np.kron(X, Y), Z)

    Arguments:
        > l: list of items for product

    Output:
        > K: cumulative Kronecker product of all items
    '''

    # initialise product var to first item in list
    K = l[0]

    # loop through remaining items
    for i in range(1, len(l)):
        K = np.kron(K, l[i])

    return K


def stabGHZ(n):
    ''' Returns array of stabilizers of n-qubit GHZ state.

    Arguments:
        > n: number of qubits in ensemble

    Outputs:
        > S: array of the 2**n stabilizers of n-qubit GHZ state such that
             S[n] is a (2**n) by (2**n) matrix stabilizing the state
    '''
    Z = pauli(3)
    X = pauli(1)
    Y = pauli(2)
    Id = pauli(0)

    # size
    sz = 2**n

    # initialise S
    # S = np.zeros((sz, sz, sz))
    S = list()

    # start with generators
    # all X
    S.append(cumKron([X] * n))

    # instantiate list of all Identity
    lid = [Id] * n

    # remaining generators
    for i in range(0, n-1):
        # modify list with 2 Z entries
        mlid = lid.copy()
        mlid[i] = Z
        mlid[i+1] = Z

        S.append(cumKron(mlid))


    # take products of generators
    ts = list() # temporary list
    # start from 2-item products and move upward
    for i in range(2, n+1):
        # get list of lists of generators for product
        prd = list(it.combinations(S, i))

        for j, v in enumerate(prd):
            tl = list(v)

            tpr = tl[0]
            for k in range(1, len(tl)):
                tpr = tpr @ tl[k]

            ts.append(tpr)

    for i, v in enumerate(ts):
        S.append(v)

    # last stabilizer is identity
    S.append(np.identity(2**n))

    return np.array(S)

def witMeas(rho, M, Nt, ps):
    ''' generates confidences that density matrix is non-separable

    Arguments:
        > rho: density matrix of target state
        > M: (2**n) array of M matrices (size 2**n x 2**n) with n the number of
             qubits
        > Nt: maximum number of copies of state available
        > ps: separable bound

    Outputs:
        > Cv: vector of confidences that state is not separable
    '''

    # get qubit number
    n = int(np.log2(rho.shape[0]))

    # vector of minimum confidences
    Cv = np.zeros(Nt)

    for N in range(1, Nt+1):

        while True:
            # number of successes
            S = 0

            for i in range(0, N):
                # sample random number in range of stabilizer group size
                r = np.random.randint(0, 2**n)

                # probability of success
                p = np.trace(M[r]@rho)

                # simulate clicks
                pclick = np.random.random_sample()
                if (pclick < p):
                    S += 1

            # deviation from separable bound
            delta = (S/N) - ps

            if (np.round(delta, decimals=5) > 0):

                x = ps + delta
                y = ps

                g = ((x/y)**(-N*x))*(((1-x)/(1-y))**(-N*(1-x)))
                #
                # D = x*np.log(x/y) + (1-x)*np.log((1-x)/(1-y))
                #
                # np.exp(-N*D) # verifies that rearranging is sound

                C = 1 - g

                break
            else:
                continue

        Cv[N-1] = C

    return [Cv, S/N]
