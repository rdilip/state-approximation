"""
Module that variationally decomposes an MPS into a series of quantum gates
and a low-entanglement MPS
"""

from renyin_splitter import split_psi 
from rfunc import pad, pad_mps
from misc import group_legs, ungroup_legs, mps_2form, mps_overlap, mpo_on_mpo,\
     mps_entanglement_spectrum
import numpy as np
import scipy
from scipy.linalg import expm
import matplotlib.pyplot as plt
from scipy.stats import unitary_group
from moses_simple import moses_move as moses_move_simple
from moses_variational import moses_move as moses_move_variational
from moses_variational_shifted import moses_move as moses_move_shifted
from disentanglers import disentangle_S2, disentangle_brute
from tebd import tebd


def mps2mpo(mps):
    """ Converts an MPS an MPO with Mike and Frank's index conventions
    Parameters
    ----------
    mps : list of np.Array
        Should have index format phys, chiL, chiR
    Returns
    -------
    mpo : list of np.Array
    """
    mpo = []
    for i, A in enumerate(mps):
        d0, chiL, chiR = A.shape
        mpo.append(A.reshape((d0, 1, chiL, chiR)))
    return(mpo)

def mpo2mps(mpo):
    """ Converts an MPO to an MPS.
    Parameters
    ----------
    mpo : list of np.Array
        Should have index format p_left, p_right (trivial), chiL, chiR
    Returns
    -------
    mps : list of np.Array
    """
    mps = []
    for i, A in enumerate(mpo):
        d0, _, chiL, chiR = A.shape
        mps.append(A.reshape((d0, chiL, chiR)))
    return(mps)

def _contract_ABS(A, B, S):
    """ Contracts a tri-split tensor. 
    Parameters
    ----------
    A, B, S : np.Array
        Follows conventions defined in renyin_splitter.py
    Returns
    -------
    T : np.Array
        Contracted tensor.
    """
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
    ----------
    Psi : np.Array
        Tensor representing wavefunction. Can be a matrix product state but
        does not have to be.
    Returns
    -------
    S : np.float
        entanglement entropy of the state
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
    """ Inverts an MPO along non physical degrees of freedom """
    return([T.transpose([0,1,3,2]) for T in mpo[::-1]])

def contract_all_mpos(mpos):
    """ Given a list of MPOS, contracts them all from left to right """
    out = mpos[0]
    for mpo in mpos[1:]:
        out = mpo_on_mpo(out, mpo)
    return(out)

def H_TFI(L, g, J=1.):
    """
    1-d Hamiltonian of TFI model.
    List of gates for TFI = -g X - J ZZ. Deals with edges to make gX uniform everywhere

    Parameters
    ----------
    L : int
        Length of chain
    g : float
        Transverse coupling
    J : float
        NN coupling 
    Returns
    -------
    H : list of np.Array
        List of two site tensors.
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

def diagonal_expansion(Psi, eta=None, disentangler=disentangle_S2):
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

    Parameters
    ----------
    Psi : list of np.Array
        Can be either in mpo format with one trivial leg or in mps format.
    eta : int
        Maximum bond dimension along Lambda
    disentangler : fn
        Disentangler function. Should accept theta (index format chiL, pL, pR,
        chiR and return U.theta, U.
    Returns
    -------
    A : list of np.Array
        Left column tensor
    Lambda : list of np.Array
        Right column tensor
    """
    # Check MPO
    if Psi[0].ndim == 3:
        Psi = mps2mpo(Psi)
    if eta is None:
        eta = max(sum([i.shape for i in Psi], ()))
    d = Psi[0].shape[0]
    # Grouping first and second tensor
    Psi = Psi.copy()
    Psi_copy = Psi.copy() # we do horrible things to Psi
    pW1, pE1, chiS1, chiN1 = Psi[0].shape
    pW2, pE2, chiS2, chiN2 = Psi[1].shape
    assert chiS2 == chiN1
    assert pE1 == pE2 == chiS1 == 1

    psi = np.tensordot(Psi[0], Psi[1], [3,2]).transpose([0,3,1,4,2,5])
    psi = psi.reshape([pW1*pW2, pE1*pE2, chiS1, chiN2])
    Psi[1] = psi
    Psi.pop(0)

    #if eta < 4:
    #    #Psi = pad_mps(Psi, 4)
    #    #eta_max = 4
    #else:
    #    eta_max = eta

    truncation_par = {"bond_dimensions": dict(eta_max=eta, chi_max=100), "p_trunc": 0}
    A0, Lambda = moses_move_simple(Psi, truncation_par, disentangler)

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

    d1 = d2 = d
    psi = psi.reshape(d1,d2,pR,chiS,chiN).transpose([0,2,3,1,4])
    psi = psi.reshape(d*pR*chiS,d)
    theta = psi.reshape(pR*chiS,d,d,1)

     
    Utheta, U = disentangler(theta)
    if A0[-1].shape[3] != 2:
        A0[-1] = pad(A0[-1], 3, 2)
    A0[-1] = np.tensordot(A0[-1], U, [[1,3],[2,3]]).transpose([0,2,1,3])

    # NOTE: I think there may be a bug somewhere here, that we're only
    # getting away with because everything is bond dimension 2... no evidence
    # to suggest this but just have a feeling.

    psi = psi.reshape(d,pR,chiS,d,1,1)
    psi = np.tensordot(psi, U.conj(), [[0,3],[2,3]]).transpose([4,0,1,5,2,3])
    pL1, pR1, chiS, pL2, pR2, chiN = psi.shape
    psi = psi.reshape(pL1*pR1*chiS, pL2*pR2*chiN)
   
    q, r = np.linalg.qr(psi)
    q = q.reshape(pL1, pR1, chiS, -1)
    r = r.reshape(-1, pL2, pR2, chiN).transpose([1,2,0,3])
    Lambda[-1] = q

    Lambda.append(r)

    # Variational moses move

    #A0, Lambda = moses_move_shifted(Psi_copy, A=A0, Lambda=Lambda)
    return A0, Lambda

def contract_diagonal_expansion(A0, Lambda):
    """ Contracts A0 and Lambda with Lambda shifted one tensor upwards.
    Parameters
    ----------
    A0 : list of np.Array
        Left column mpo
    Lambda : list of np.Array
        Right column wavefunction
    Returns
    -------
    contracted : list of np.Array
        A0.Lambda
    """
    out = A0.copy()
    Lambda = Lambda.copy()
    for i in range(1, len(A0)-1):
        prod = np.tensordot(A0[i], Lambda[i-1], [1,0])
        prod = group_legs(prod, [[0],[3],[1,4],[2,5]])[0]
        out[i] = prod

    last_tensor = np.tensordot(A0[-1], Lambda[-2], [1,0])
    last_tensor = np.tensordot(last_tensor, Lambda[-1], [[2,5],[0,2]])

    last_tensor = group_legs(last_tensor, [[0],[2,4],[1,3],[5]])[0]
    out[-1] = last_tensor
    return(out)

def contract_series_diagonal_expansions(As, Lambda, n=None):
    """ Contracts a list of As and a final Lambda wavefunction.
    Parameters
    ----------
    As : list of lists of np.Arrays
        List of single column wavefunctions shifted relative to each other,
        contracted from left to right 
    Lambda : list of np.Array
        Final physical wavefunction
    n : int
        Number of layers to contract (obviously < len(As))
    Returns
    -------
    contracted : list of np.Array
        Full contracted mps.
    """

    if n is None:
        n = len(As)
    contracted = Lambda.copy()
    for i in range(-1, -(n + 1), -1):
        contracted = contract_diagonal_expansion(As[i], contracted)
    return contracted

def multiple_diagonal_expansions(Psi, n):
    """ Perform n diagonal expansions. Returns all the Ai and Lambda such that
    \prod A_0 A_1...A_{n-1} Lambda ~= Psi

    By default will halve max bond dimension of Lambda at each step

    Parameters
    ----------
    Psi : list of np.Array
        Wavefunction to expand
    n : int
        Number of contractions. No reason to do more than log_2 chi_max
    """
    As, Lambdas = [], []
    #Ss = [entanglement_entropy(Psi)]
    Lambda = Psi.copy()
    info = dict(Ss=[], Lambdas=[])
    eta_max = max(sum([i.shape for i in Psi], ()))

    count_no_change = 0

    for i in range(n):
        if eta_max == 0:
            return As, Lambda, info
        A0, Lambda = diagonal_expansion(Lambda.copy(), eta=eta_max)
                        
        As.append(A0)
        Lambda = mps_2form(Lambda, 'B')
        #info['Ss'].append(entanglement_entropy(Lambda))
        print(len(Lambda))
        info['Lambdas'].append(Lambda)

        eta_max = int(eta_max / 2)
    return As, Lambda, info

if __name__ == '__main__':
    tebd_state, _, _ = tebd(10, 1.5, 0.1)
    Psi = mps2mpo(tebd_state.copy())
    Lambda = Psi.copy()
    As, Lambda, info = multiple_diagonal_expansions(Psi,10)
    out = contract_series_diagonal_expansions(As, Lambda)
    

    # No halving, no variational
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,4))
    ax.semilogy(info['Ss'], 'o-')
    ax.set_ylabel("Entanglement entropy", fontsize=14)
    ax.set_xlabel("Number of iterations", fontsize=14)

    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    ax.set_title("Variational " + rf"$\langle\Psi|\tilde\Psi\rangle = {round(mps_overlap(out, Psi),5)}$", fontsize=16)
    plt.tight_layout()

    plt.savefig("img/nv_vs_v_mm.png", bbox_inches='tight', dpi=150)


