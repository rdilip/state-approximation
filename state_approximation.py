from renyin_splitter import split_psi 
from misc import group_legs, ungroup_legs, mps_2form, mps_overlap
import numpy as np
import scipy
from math import floor
from scipy.stats import unitary_group


def random_mps_2(L, chi=50, d=2, seed=12345):
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
        b = 0.5 - np.random.rand(d, 1, 1)
        b /= np.linalg.norm(b)
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
            Psi[i] = q.reshape((d1, chiL, -1))
            Psi[i+1] = r.reshape((-1,d2,chiR,)).transpose([1,0,2])
        Psi = [psi.transpose([0,2,1]) for psi in Psi[::-1]]
        num_sweeps += 1
    return(mps_2form(Psi, 'B'))

def mpo_on_mps(mpo, mps):
    """ Applies an MPO to an MPS. In this case, as we've structured it A is
    actually the MPS and Lambda is the MPO. 

    A convention:

    2
    |
    A - 0(p)
    |
    1

    Lambda convention:
         3
         |
    2 -- B -- 0(p)
         |
         1
    """
    assert len(mpo) == len(mps)
    mps_output = []
    for i in range(len(mpo)):
        a = mps[i]
        B = mpo[i]
        out = np.tensordot(B, a, [0,0]).transpose([0,1,3,2,4])
        d0, d1, d2, d3, d4 = out.shape
        mps_output.append(out.reshape([d0, d1*d2, d3*d4]))
    return(mps_output)

def moses_move(Psi, truncation_par):
    """ Performs a moses move on a wavefunction Psi. These are three index
    tensors for a 1D MPS. Differs from isotns moses move because the one
    column wavefunction is shifted to the right and is disentangled. 

    Psi must be in B-form. Index convention should be 
        1
        |
        T-0
        |
        2
    Parameters
    ----------
    Psi : list of np.Array
        Original 1D wavefucntion

    Returns
    -------
    A : list of np.Array
        List of 3 index tensors
    Lambda : list of np.Array
        List of 4 index tensors
    """
    # Initialize from bottom
    eta_max = truncation_par['bond_dimensions']['eta_max']
    chi_max = truncation_par['bond_dimensions']['chi_max']
    eta = 1
    if len(Psi[0].shape) == 3:
        d0, chiR, chiL = Psi[0].shape # mps
    else:
        d0, eta, chiR, chiL = Psi[0].shape # mpo
    Tri = Psi[0].reshape(d0 * eta, chiR, chiL)

    L = len(Psi)
    Lambda = []
    A = []

    for j in range(L):
        Tri = Tri.reshape((d0 * eta, chiL, chiR))
        #Tri = Tri.transpose([1,2,0])
        #Tri = Tri.transpose([0, 2, 1])
        dL, dR = 2,2

        # Check what happens at boundary ...
        if j == L - 1:
            dL = 1
            dR = np.min([eta, Tri.shape[0]])
        a, S, B, info = split_psi(Tri,
                                  dL,
                                  dR,
                                  truncation_par={
                                    'chi_max': eta_max,
                                    'p_trunc': truncation_par['p_trunc']
                                  },
                                  n=2,
                                  eps=1.e-6,
                                  verbose=0)
        dL, dR = a.shape[1], a.shape[2]

        B = B.reshape((dR, B.shape[1], d0, eta))

        B = B.transpose([2,3,0,1])
        # a can stay the same
        Lambda.append(B)
        A.append(a)
        if j < L - 1:
            eta = S.shape[2]
            chiL = Psi[j+1].shape[2]
            Tri = np.tensordot(S, Psi[j+1], [1,1]).transpose([2,1,0,3])
            Tri = Tri.reshape([d0 * eta, dL, chiL])
        else:
            Lambda[j] = Lambda[j] * S
    return(A, Lambda)


def mps2mpo(mps):
    """ Converts an MPS in a reasonable form to an MPO with Mike and Frank's
    index conventions
    """
    # TODO insert diagrams into docstring
    mpo = []
    for i, A in enumerate(mps):
        d0, chiL, chiR = A.shape
        mpo.append(A.reshape((d0, 1, chiL, chiR)))
    return(mpo)

def mpo2mps(mpo):
    """ Converts an MPO to an MPS where the second index is assumed to be 
    trivial """
    mps = []
    for i, A in enumerate(mpo):
        d0, _, chiL, chiR = A.shape
        mps.append(A.reshape((d0, chiL, chiR)))
    return(mps)

def contract_ABS(A, B, S):
    if len(A.shape) == 4:
        C = np.tensordot(A, B, [1,0])
        T = np.tensordot(C, S, [[2,5],[0,2]])
    elif len(A.shape) == 3:
        C = np.tensordot(A, B, [2,0])
        T = np.tensordot(C, S, [[1,2],[0,2]]).transpose([0,2,1])
    else:
        raise ValueError("Invalid dimensions")
    return(T)

def entanglement_entropy(Psi_inp):
    """ Brute calculates the von Neumann entropy. Can do more easily by
    taking SVDs, but this is a good sanity check.
    Parameters
    Psi : np.Array
        Tensor representing wavefunction. Can be a matrix product state but
        does not have to be.
    """
    T = Psi_inp[0]
    if len(T.shape) == 4:
        Psi = mpo2mps(Psi_inp)
    else:
        Psi = Psi_inp.copy()
    L = len(Psi)
    d0, _, _ = Psi[0].shape
    for psi in Psi[1:]:
        T = np.tensordot(T, psi, [-1,1])
    T = T.reshape([d0] * len(Psi)).reshape([d0**(L//2), d0**(L - L//2)])
    s = np.linalg.svd(T, compute_uv=False)
    return -np.sum(s**2 * np.log(s**2))

if __name__ == '__main__':
    Psi = random_mps(10, 10, d=2, seed=12345)
    Psi_mpo = mps2mpo(Psi)
    A0, Lambda = moses_move_simple(Psi_mpo, truncation_par)
    Lambda = mps_2form(Lambda, 'B')
    out = mpo2mps(mpo_on_mpo(A0, Lambda))
    print(mps_overlap(out, Psi))
