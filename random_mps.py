"""
A variety of functions to produce different random MPSs
"""
import numpy as np

def random_mps_uniform(L, chi=50, d=2, seed=12345):
    """ Generates a random MPS with a fixed bond dimension """
    mps = []
    np.random.seed(seed)
    for i in range(L):
        chiL = chiR = chi
        if i == 0:
            chiL = 1
        if i == L-1:
            chiR = 1
        t = 0.5 - np.random.rand(d, chiL, chiR)
        mps.append(t / np.linalg.norm(t))
    norm = np.sqrt(mps_overlap(mps, mps))
    mps[0] /= norm
    return(mps)

def random_mps_nonuniform(L, chi_min=50, chi_max=200, d=2, seed=12345):
    """ Generates a random MPS with a random bond dimension """
    mps = []
    np.random.seed(seed)
    for i in range(L):
        chiR = np.random.randint(chi_min, chi_max)
        if i == 0:
            chiL = 1
        if i == L-1:
            chiR = 1
        t = 0.5 - np.random.rand(d, chiL, chiR)
        mps.append(t / np.linalg.norm(t))
        chiL = chiR
    norm = np.sqrt(mps_overlap(mps, mps))
    mps[0] /= norm
    return(mps)

def random_mps(L, num_layers=10, d=2, seed=None):
    """ Returns a random MPS by starting with a product state and applying a 
    series of two-site unitary gates. This isn't trotterizes, just one after
    the other.
    Parameters
    ----------
    L : int
        Length of MPS
    num_layers : int
        Number of layers of unitary gates to apply. One layer is a single sweep
        back and forth.
    d : int
        Physical dimension
    """
    if seed:
        np.random.seed(seed)
    Psi = []
    for i in range(L):
        b = np.zeros((d, 1, 1))
        b[np.random.choice(range(d)), 0, 0] = 1.0
        Psi.append(b)

    num_sweeps = 0
    while num_sweeps < 2 * num_layers:
        psi = Psi[0]
        for i in range(L-1):
            U = unitary_group.rvs(d*d).reshape([d,d,d,d])
            theta = np.tensordot(Psi[i], Psi[i+1], [2,1])
            theta = np.tensordot(theta, U, [[0,2],[2,3]]).transpose([0,2,3,1])
            chiL, d1, d2, chiR = theta.shape
            theta = theta.reshape((chiL * d1, d2 * chiR))
            q, r = np.linalg.qr(theta)
            Psi[i] = q.reshape((chiL, d1, -1)).transpose([1,0,2])
            Psi[i+1] = r.reshape((-1,d2,chiR,)).transpose([1,0,2])
        Psi = [psi.transpose([0,2,1]) for psi in Psi[::-1]]
        num_sweeps += 1
    return(mps_2form(Psi, 'B'))

def random_mps_N_unitaries(L, num_unitaries):
    """
    Starting from a product state, applies N two site unitaries
    """

    Psi = []
    d = 2
    for i in range(L):
        b = np.zeros((d, 1, 1))
        b[np.random.choice(range(d)), 0, 0] = 1.0
        Psi.append(b)
    for i in range(num_unitaries):
        j = i % (L - 1)
        #print(f"applying unitary to site {0}, {1}".format(j, j+1))
        U = unitary_group.rvs(4).reshape([2,2,2,2])
        theta = np.tensordot(Psi[j], Psi[j+1], [2,1])
        theta = np.tensordot(U, theta, [[2,3],[0,2]]).transpose([2,0,1,3])

        chiL, d1, d2, chiR = theta.shape
        theta = theta.reshape((chiL*d1, chiR*d2))
        q, r = np.linalg.qr(theta)
        Psi[j] = q.reshape((chiL,d1,-1)).transpose([1,0,2])
        Psi[j+1] = r.reshape((-1,d2,chiR)).transpose([1,0,2])
    return(mps_2form(Psi, 'B'))


