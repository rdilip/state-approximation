from renyin_splitter import split_psi 
from rfunc import pad
from misc import group_legs, ungroup_legs, mps_2form, mps_overlap, mpo_on_mpo,\
     mps_entanglement_spectrum
import numpy as np
import scipy
from scipy.linalg import expm
import matplotlib.pyplot as plt
from scipy.stats import unitary_group
from moses_simple import moses_move as moses_move_simple
import gc

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
        print(f"applying unitary to site {j}, {j+1}")
        U = unitary_group.rvs(4).reshape([2,2,2,2])
        theta = np.tensordot(Psi[j], Psi[j+1], [2,1])
        theta = np.tensordot(U, theta, [[2,3],[0,2]]).transpose([2,0,1,3])

        chiL, d1, d2, chiR = theta.shape
        theta = theta.reshape((chiL*d1, chiR*d2))
        q, r = np.linalg.qr(theta)
        Psi[j] = q.reshape((chiL,d1,-1)).transpose([1,0,2])
        Psi[j+1] = r.reshape((-1,d2,chiR)).transpose([1,0,2])
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
        gc.collect()
    return(out)

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

def diagonal_expansion(Psi, eta, debug_mode=False):
    """ This performs an expansion where the tensor network shifts diagonally.
      |  |  |  |  | 
    --A--A--A--A--A--


          |  |  |  |  | 
       .--A--A--A--A--A--
       |  |  |  |  |  |
       |  |  |  |  | 
     --A--A--A--A--A--

    etc. 

    At some point I should adjust these diagrams so they can be mapped onto
    inputs...
    """
    # Check MPO
    if Psi[0].ndim == 3:
        Psi = mps2mpo(Psi)
    d = Psi[0].shape[0]
    # Grouping first and second tensor
    pW1, pE1, chiS1, chiN1 = Psi[0].shape
    pW2, pE2, chiS2, chiN2 = Psi[1].shape
    assert chiS2 == chiN1
    assert pE1 == pE2 == chiS1 == 1

    psi = np.tensordot(Psi[0], Psi[1], [3,2]).transpose([0,3,1,4,2,5])
    psi = psi.reshape([pW1*pW2, pE1*pE2, chiS1, chiN2])
    Psi[1] = psi
    Psi.pop(0)

    truncation_par = {"bond_dimensions": dict(eta_max=eta, chi_max=100), "p_trunc": 0}
    if eta == 1:
        A0, Lambda = moses_move_simple(Psi, truncation_par, debug_mode=True)
    else:
        A0, Lambda = moses_move_simple(Psi, truncation_par, debug_mode=False)
    # Splitting physical leg out of first tensor
    psi = A0[0]
    pL, pR, chiS, chiN = psi.shape
    assert pL == d*d
    psi = psi.reshape(d, d, pR, chiS, chiN).transpose([0,3,1,2,4])
    psi = psi.reshape(d*chiS, d*pR*chiN)
    psi0, psi1 = np.linalg.qr(psi)
    psi0 = psi0.reshape(d, chiS, -1, 1).transpose([0,3,1,2])
    psi1 = psi1.reshape(-1,d,pR,chiN).transpose([1,2,0,3])
    A0[0] = psi1
    A0.insert(0, psi0)
    
    # Splitting last tensor of A0
    psi = A0[-1]
    pL, pR, chiS, chiN = psi.shape
    # assert pR == 4
    assert chiN == 1
    if pR == 4:
        psi = psi.reshape(pL, d, d, chiS, chiN).transpose([0,1,3,4,2]).reshape(pL,d,chiS,d)
    A0[-1] = psi

    # send second physical leg to the top

    # Splitting last tensor of Lambda
    psi = Lambda[-1]
    pL, pR, chiS, chiN = Lambda[-1].shape
    assert(pR == chiN == 1)
    #assert pL == 4
    if pL != 4:
        psi = pad(psi, 0, 4)
    psi = psi.reshape(d,d,pR,chiS,chiN).transpose([0,2,3,1,4])
    psi = psi.reshape(d*pR*chiS,d)
    q, r = np.linalg.qr(psi)
    q = q.reshape(d,pR,chiS,-1)
    r = r.reshape(-1,d,1,1).transpose([1,2,0,3])
    Lambda[-1] = q
    Lambda.append(r)

    return(A0, Lambda)

def contract_diagonal_expansion(A0, Lambda):
    """ Contraction is not just mpo on mpo, because of the overhand on each
    side."""
    out = A0.copy()
    for i in range(1, len(A0)):
        prod = np.tensordot(A0[i], Lambda[i-1], [1,0])
        prod = group_legs(prod, [[0],[3],[1,4],[2,5]])[0]
        out[i] = prod
    last_tensor = np.tensordot(A0[-1], Lambda[-2], [1,0])
    last_tensor = np.tensordot(last_tensor, Lambda[-1], [[2,5],[0,2]])
    last_tensor = group_legs(last_tensor, [[0],[2,4],[1,3],[5]])[0]
    out[-1] = last_tensor
    return(out)

def contract_series_diagonal_expansions(As, Lambda, n=None):
    """ Contracts a list of As and a final Lambda wavefunction. n is the number
    of layers to contract."""

    if n is None:
        n = len(As)
    contracted = Lambda.copy()
    for i in range(-1, -(n + 1), -1):
        contracted = contract_diagonal_expansion(As[i], contracted)
    return contracted

def multiple_diagonal_expansions(Psi, n):
    """ Perform n diagonal expansions. Returns all the Ai and Lambda such that
    \prod A_0 A_1...A_{n-1} Lambda = Psi 
    """
    fidelity = [mps_overlap(Psi, Psi)]
    As, Lambdas = [], []
    Ss = [entanglement_entropy(Psi)]
    Ss_2 = [mps_entanglement_spectrum(Psi)[len(Psi)//2]]
    Lambda = Psi.copy()
    change_points = []

    # Unpack all shapes of tensors in Psi and select largest bond dimension 
    eta_max = max(list(sum([i.shape for i in Lambda], ())))

    for i in range(n):
        if i == 48:
            A0, Lambda = diagonal_expansion(Lambda, eta_max, debug_mode=True)
        else:
            A0, Lambda = diagonal_expansion(Lambda, eta_max, debug_mode=False)

        As.append(A0)
        Lambdas.append(Lambda)
        Lambda = mps_2form(Lambda, 'B')
        Ss.append(entanglement_entropy(Lambda))

        fidelity.append(np.linalg.norm(mps_overlap(Psi, Lambda)))
        if eta_max == 1:
            return As, Lambda, Ss, Ss_2, fidelity, change_points, Lambdas

        if Ss[-2] - Ss[-1] < 1.e-8:
            #eta_max = int(eta_max / 2)
            #change_points.append(i)
            #print(f"Changing eta to {eta_max}")
            return As, Lambda, Ss, Ss_2, fidelity, change_points, Lambdas

    return As, Lambda, Ss, Ss_2, fidelity, change_points, Lambdas


def check_isometry(Psi, i):
    """
    Checks that the wavefunction Psi is in canonical form with 
    orthogonality center at index i
    """
    if Psi[i].ndim != 4:
        Psi = mps2mpo(Psi)
    for j in range(i):
        psi = Psi[j]
        pL, pR, chiS, chiN = psi.shape
        iso = np.tensordot(psi, psi.conj(), [[0,2],[0,2]])
        iso = iso.reshape([pR*chiN, pR*chiN])

        if not np.allclose(iso, np.eye(pR*chiN), rtol=1.e-8):
            return False

    for j in range(i+1, len(Psi)):
        psi = Psi[j]
        pL, pR, chiS, chiN = psi.shape
        iso = np.tensordot(psi, psi.conj(), [[0,1,3],[0,1,3]])
        if not np.allclose(iso, np.eye(chiS), rtol=1.e-8):
            return False
    return True

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

