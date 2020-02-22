from renyin_splitter import split_psi 
from misc import group_legs, ungroup_legs, mps_2form, mps_overlap, mpo_on_mpo
import numpy as np
import scipy
from scipy.linalg import expm
import matplotlib.pyplot as plt
from math import floor
from scipy.stats import unitary_group
from moses_simple import moses_move as moses_move_simple

def evolve_state(L, T, dt, H=None):
    """ Gives a physical state by performing TEBD-style time evolution on 
    a product state.
    
    Only set up for local hilbert space 2 and first order trotterization.
    """
    d = 2
    if H is None:
        H = H_TFI(L, g=0.5, J = 1.0)
    if H[0].ndim == 4:
        H = [h.reshape(d*d,d*d) for h in H]

    U = [expm(-dt * h).reshape([d]*4) for h in H]

    b = np.zeros((2,1,1))
    b[0,0,0] = 1.0
    Psi = [b.copy() for i in range(L)]

    num_sweeps = int(T // dt)

    num_sweeps = num_sweeps if num_sweeps % 2 == 0 else num_sweeps + 1
    Ss = []

    for i in range(num_sweeps):
        print(i)
        for bond in range(L - 1):
            try:
                theta = np.tensordot(Psi[bond], Psi[bond+1], [2,1]).transpose([1,0,2,3])
            except: 
                breakpoint()
            theta = np.tensordot(U[bond], theta, [[2,3],[1,2]]).transpose([2,0,1,3])
            chiL, d1, d2, chiR = theta.shape

            theta = theta.reshape(chiL*d1, chiR*d2)
            u,s,v = np.linalg.svd(theta)
            u = u.reshape(chiL, d1, u.shape[-1]).transpose([1,0,2])
            v = (np.diag(s) @ v).reshape(s.shape[0], d2, chiR).transpose([1,0,2])
            
            if i == num_sweeps - 1:
                Ss.append(s)
            Psi[bond] = u
            Psi[bond+1] = v
        breakpoint()
        Psi = [psi.transpose([0,2,1]) for psi in Psi[::-1]]
        U = [u.transpose([1,0,3,2]) for u in U[::-1]]
    return(Psi, Ss)
            

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
    #for i in range(L):
    #    b = 0.5 - np.random.rand(d, 1, 1)
    #    b /= np.linalg.norm(b)
    #    Psi.append(b)

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
    if len(Psi_inp[0].shape) == 4:
        Psi = mpo2mps(Psi_inp)
    else:
        Psi = Psi_inp.copy()
    ds = [Psi[0].shape[0]]
    L = len(Psi)
    T = Psi[0]

    for psi in Psi[1:]:
        T = np.tensordot(T, psi, [-1,1])
        ds.append(psi.shape[0])
    T = T.reshape(np.product(ds[:L//2]), np.product(ds[L//2:]))
    s = np.linalg.svd(T, compute_uv=False)
    s = s[s > 1.e-8]
    return -np.sum(s**2 * np.log(s**2))

def invert_mpo(mpo):
    return([T.transpose([0,1,3,2]) for T in mpo[::-1]])

def contract_all_mpos(mpos):
    """ Given a list of MPOS, contracts them all from left to right """
    out = mpos[0]
    for mpo in mpos[1:]:
        out = mpo_on_mpo(out, mpo)
    return(out)

def shift_leg(X, Y, leg, bond_dim):
    """ Given a leg on tensor X, shifts the leg down to tensor Y. Assumes that 
    X is contracted with Y via leg 2 on X to leg 3 on Y. bond_dim is the 
    desired bond dimension of the final contraction (may be some truncation)  """
    ndim = X.ndim
    X = X.transpose([*range(leg), *range(leg+1,ndim), *leg])

def H_TFI(L, g, J=1):
    """
    1-d Hamiltonian of TFI model.
    List of gates for TFI = -g X - J ZZ. Deals with edges to make gX uniform everywhere

    """
    sx = np.array([[0, 1], [1, 0]])
    sz = np.array([[1, 0], [0, -1]])
    id = np.eye(2)
    d = 2

    def h(gl, gr, J):

        return (-np.kron(sz, sz) * J - gr * np.kron(id, sx) -
                gl * np.kron(sx, id)).reshape([d] * 4)

    H = []
    for j in range(L - 1):
        gl = 0.5 * g
        gr = 0.5 * g
        if j == 0:
            gl = g
        if j == L - 2:
            gr = g
        H.append((-J * np.kron(sz, sz) - gr* np.kron(id, sx) -\
                    gl * np.kron(sx, id)).reshape([d]*4))
    return H

if __name__ == '__main__':

    Psi = random_mps_2(8, chi=200, d=2, seed=12345)
    As = []
    Ss = []
    Lambda = mps2mpo(Psi)
    eps = 1.e-10
    m = 0

    while m < 100:
        if m % 2 == 1:
            Lambda = invert_mpo(Lambda)
        A0, Lambda = moses_move_simple(Lambda)
        if m % 2 == 1:
            Lambda = invert_mpo(Lambda)
            A0 = invert_mpo(A0)
        As.append(A0)
        Ss.append(entanglement_entropy(Lambda))
        m += 1

    plt.semilogy(Ss) 
    plt.show()

