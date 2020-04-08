# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import scipy
from contraction_shifted import contract_series_diagonal_expansion,\
                                contract_diagonal_expansion_bottom,\
                                contract_diagonal_expansion_top
from misc import mpo_on_mpo, mps_overlap
from glob import glob
import time

def var_1site_Lambda(Psi, A, Lambda):
    """ Variationally sweeps through Lambda and optimizes tensors to maximize overlap between 
    Psi and A.Lambda. Assumes the shifted protocol. 
    Parameters
    ----------
    Psi : list of np.Array
    A : list of np.Array
    Lambda : list of np.Array
    
    Returns
    -------
    Lambda, Lp_list :
        Lp_list of a list of environments starting from the bottom. You can actually reuse the
        lp_list, but you do need to be a bit careful about removing the last value and adding
        one to the start
    """
    # Initializing environments. The convention is always to group all the indices that will
    # form the final environment first, then the indices that will be contracted over.
    # The bottom most left part has shape [pL, pR, NL, NR]

    Lp_list = []
    L = len(A)
    assert L == len(Psi) == len(Lambda)

    Lp = np.tensordot(Psi[0], A[0].conj(), [[0,2],[0,2]]).transpose([0,1,3,2])
    Lp_list.append(Lp)
    for i in range(1, L):
        Lp = np.tensordot(Lp, Lambda[i-1].conj(), [[0,3],[1,2]])
        Lp = np.tensordot(Lp, A[i].conj(), [[1,2],[2,1]])
        Lp = np.tensordot(Lp, Psi[i], [[0,2],[2,0]]).transpose([2,3,1,0])

        Lp_list.append(Lp)
    
    
    Lambdap = [[] for i in range(L)]
    Rp = Lp_list[L-1].copy()
    pE, chiN, pW, chiS = Rp.shape
    theta = Rp.reshape(pE*chiN*pW, chiS)
    X, S, Z = np.linalg.svd(theta, full_matrices=False)
    Lambdap[L-1] = X.reshape(pE, chiN, pW, -1).transpose([2,0,3,1])
    
    Rp = Lambdap[L-1].transpose([1,3,0,2])
    Rp_list = [Rp]
    # Now go down
    for i in range(L-2, -1, -1):

        # NOTE: What I"m doing right now is super inefficient, but might help?
        Lp = Lp_list[i]
        Rp = np.tensordot(Rp, Psi[i+1], [[1,0],[3,1]])
        Rp = np.tensordot(Rp, A[i+1].conj(), [[0,2],[3,0]])
        env = np.tensordot(Rp, Lp, [[1,3],[1,2]])

        chiN, pW, pE, chiS = env.shape
        X, S, Z = np.linalg.svd(env.reshape(chiN*pW*pE, chiS), full_matrices=False)
        Lambdap[i] = X.reshape(chiN, pW, pE, -1).transpose([1,2,3,0])

        Rp = np.tensordot(Rp, Lambdap[i].conj(), [[0,2],[3,0]]).transpose([2,0,1,3])
        Rp_list.append(Rp)

    Lp = Lp_list[0]
    for i in range(0, L-1):
        Rp = Rp_list[L-2-i]
        Lp = np.tensordot(Lp, Psi[i+1], [1,2])
        Lp = np.tensordot(Lp, A[i+1].conj(), [[1,3],[2,0]])
        env = np.tensordot(Lp, Rp, [[2,3,5],[0,1,2]])
        pE, chiS, pW, chiN = env.shape
        X, S, Z = np.linalg.svd(env.reshape(chiS*pE*pW, chiN), full_matrices=False)
        Lambdap[i] = X.reshape(pE, chiS, pW, -1).transpose([2,0,1,3])
        Lp = np.tensordot(Lp, Lambdap[i].conj(), [[4,0,1],[0,1,2]])
        Lp_list[i+1] = Lp

    Lp = Lp.transpose([2,0,3,1])
    pW, pE, chiS, chiN = Lp.shape

    X, S, Z = np.linalg.svd(Lp.reshape(pE*chiS*pW, chiN), full_matrices=False)
    Lambdap[L-1] = X.reshape(pW, pE, chiS, -1)

    R = np.diag(S) @ Z
    assert R.size == 1
    R /= np.linalg.norm(R)
    Lambdap[L-1] = Lambdap[L-1] * R[0,0]

    return Lambdap, Lp_list

def var_A(Psi, A, Lambda, Lp_list=None):   
    """ Variationally sweeps through A and optimizes tensors to maximize overlap between 
    Psi and A.Lambda. Assumes the shifted protocol. 
    Parameters
    ----------
    Psi : list of np.Array
    A : list of np.Array
    Lambda : list of np.Array
    
    Returns
    -------
    Lambda, Lp_list :
        Lp_list of a list of environments starting from the bottom. The shifting means you can't
        reuse this with var_A, since the environments actually need to be different. So it's
        probably only useful for debugging.
    """
    L = len(Psi)

    if Lp_list is None:
        # Here to catch errors
        Lp_list = [None]
        Lp = np.tensordot(Psi[0], A[0].conj(), [[0,2],[0,2]]).transpose([0,1,3,2])
        Lp_list.append(Lp)
        # Don't need to go all the way for the A column
        for i in range(1, L-1):
            Lp = np.tensordot(Lp, Psi[i], [1,2])
            Lp = np.tensordot(Lp, A[i].conj(), [[1,3],[2,0]])
            Lp = np.tensordot(Lp, Lambda[i-1].conj(), [[4,0,1],[0,1,2]])
            Lp_list.append(Lp)

    Ap = [[] for i in range(L)]
    Ap = A.copy()

    Rp = Lambda[-1].conj().transpose([1,3,0,2])
    for i in range(L - 1, 0, -1):
        Rp = np.tensordot(Rp, Psi[i], [[0,1],[1,3]])
        Rp = np.tensordot(Rp, Lambda[i-1].conj(), [1,3])
        env = np.tensordot(Rp, Lp_list[i], [[2,4,5],[1,0,3]]).transpose([1,3,2,0])
        pW, chiS, pE, chiN = env.shape
        
        X, S, Z = np.linalg.svd(env.reshape(pW*chiS, pE*chiN), full_matrices=False)
        Ap[i] = np.dot(X, Z).reshape(pW, chiS, pE, chiN).transpose(0,2,1,3)
        Rp = np.tensordot(Rp, Ap[i].conj(), [[1,3,0],[0,1,3]]).transpose([1,0,3,2])

    env = np.tensordot(Rp, Psi[0], [[0,1],[1,3]]).transpose(2,3,0,1)
    pW, chiS, chiN, pE = env.shape

    X, S, Z = np.linalg.svd(env.reshape(pW*chiS, pE*chiN), full_matrices=False)
    Ap[0] = np.dot(X, Z).reshape(pW, chiS, pE, chiN).transpose(0,2,1,3)

    return Ap

def moses_move(Psi, A, Lambda, N=10, get_fidelity=False):
    fidelities = []
    for i in range(N):
        Lambda, Lp_list = var_1site_Lambda(Psi, A, Lambda)
        A = var_A(Psi, A, Lambda)
        if get_fidelity:
            out = contract_diagonal_expansion_top(A, Lambda)
            fidelities.append(np.linalg.norm(mps_overlap(out, Psi)))
    if get_fidelity:
        return A, Lambda, fidelities
    else:
        return A, Lambda

def optimize_single_site(Psi, As, num_sweeps):
    """
    Given a wavefunction Psi s.t. Psi = (\Prod A_i).Lambda, this function 
    iteratively optimizes single site unitaries by replacing Lambda with a 
    product state. This will give us an approximation to the original state,
    but it's unclear how good of an approximation it will be. 

    Parameters
    ----------
    Psi : list of np.Array
        Wavefunction (in MPS or MPO form) to be decomposed. 
    As : list of list of np.Array
        List of column wavefunctions, evaluated left to right using shifted
        protocol.
    num_sweeps : int
        Number of sweeps from top to bottom
    Returns
    -------
    Us : list of np.Array
        List of unitaries 
    As : list of list of np.Array
        List of column wavefunctions. This is the same as the original list, 
        but As[-1] has been contracted with Us.
    """
    for i in range(num_sweeps):
        if i % 10 == 0:
            print(i)
        Us, As, Lambda = _optimize_single_site_sweep(Psi, As)
    return Us, As, Lambda

def apply_Us_to_A(A, Us):
    """ 
    Applies unitaries to A using shifted protocol, left to right.
    """
    A = A.copy()
    n = len(A)
    
    assert len(Us) == n
    for i in range(1, n):
        A[i] = np.tensordot(A[i], Us[i-1].conj(), [1,0]).transpose(0,3,1,2)
    A[n-1] = np.tensordot(A[n-1], Us[n-1].conj(), [3,0])
    return A

def optimize_single_site_sweep_fast(Psi, As, Lp_list=None):
    m = 0
    F = []
    go = True
    while m < 100 and go:
        As, Us, Lambda, Lp_list = _optimize_single_site_sweep_fast(Psi, As, Lp_list = None)
        out = contract_series_diagonal_expansion(As, Lambda, mode='top')
        F.append(np.linalg.norm(mps_overlap(out, Psi)))
        if m > 2:
            go = (F[-2] - F[-1]) > 1.e-8
        m += 1
    return As, Lambda, F

def _optimize_single_site_sweep_fast(Psi, As, Lp_list=None):
    """ 
    Performs a single sweep from top to bottom of the single site optimization.
    optimize_single_site() just calls this function multiple times. Takes 
    advantage of trivial legs to run faster than the normal contraction method.
    Parameters
    ----------
    Psi : list of np.Array
        Wavefunction (in MPS or MPO form) to be decomposed. 
    As : list of list of np.Array
        List of column wavefunctions, evaluated left to right using shifted
        protocol.
    Returns
    -------
    Us : list of np.Array
        List of unitaries 
    As : list of list of np.Array
        List of column wavefunctions. This is the same as the original list, 
        but As[-1] has been contracted with Us.
    Lambda : list of np.Array
        Product state
    """
    L = len(Psi)
    Psi = Psi.copy()
    As = As.copy()
    Psi = [psi.transpose([1,0,2,3]) for psi in Psi]
    Us = [np.eye(2) for i in range(L)]
    assert np.allclose(L, [len(A) for A in As])

    b = np.zeros([2,1,1,1])
    b[0,0,0,0] = 1.0
    Lambda = [b.copy() for i in range(L)]

    A0 = [a.conj() for a in As[0]]
    A = mpo_on_mpo(Psi, A0)
    for i in range(1, len(As)):
        Ai = [a.conj() for a in As[i]]
        A = contract_diagonal_expansion_bottom(A, Ai)

    assert A[0].shape[2] == 1
    first_shape, last_shape = A[0].shape, A[-1].shape

    # Lp_list[i] is the environment below tensor i in Lambda
    if Lp_list is None:
        Lp = A[0].reshape(first_shape[0], first_shape[1], first_shape[3]).transpose(0,2,1)
        Lp_list = [Lp]
        for i in range(1, L):
            Lp = np.tensordot(Lp, A[i].conj(), [1,2])
            Lp = np.tensordot(Lp, Lambda[i-1].conj(), [[3,0,1],[0,1,2]])
            Lp_list.append(Lp)

    # Rp_list[i] is the environment above tensor i on A
    shape = Lambda[-1].shape
    Rp = Lambda[-1].reshape(shape[0], shape[1], shape[2]).transpose(0,2,1)
    env = np.tensordot(Lp_list[-1], Rp, [[0,2],[2,1]])
    assert env.shape == (2,2)
    X, S, Z = np.linalg.svd(env, full_matrices=False)
    U = X @ Z
    Us[-1] = U

    A[-1] = np.tensordot(A[-1], U, [3,0])

    for i in range(L-1, 0, -1):
        Lp = Lp_list[i-1]
        Lp_i = np.tensordot(Lp, A[i].conj(), [1,2])
        Lp_i = np.tensordot(Lp_i, Lambda[i-1].conj(), [[0,1],[1,2]])
        env = np.tensordot(Lp_i, Rp, [[2,4,0],[0,1,2]])
        X, S, Z = np.linalg.svd(env, full_matrices=False)
        U = X @ Z
        Us[i-1] = U
        A[i] = np.tensordot(A[i], U, [1,0]).transpose(0,3,1,2)
        Lp_i = np.tensordot(Lp_i, U, [1,0])
        Lp_i = np.einsum('ijklk->ijl', Lp_i)
        Lp_list[i] = Lp_i

        Rp = np.tensordot(Rp, A[i].conj(), [[0,2],[3,0]])
        Rp = np.tensordot(Rp, Lambda[i-1].conj(), [[0,1],[3,0]]).transpose(0,2,1)
    
    A = apply_Us_to_A(As[-1], Us)
    As[-1] = A

    return As, Us, Lambda, Lp_list

def _optimize_single_site_sweep(Psi, As):
    """ 
    Performs a single sweep from top to bottom of the single site optimization.
    optimize_single_site() just calls this function multiple times.
    Parameters
    ----------
    Psi : list of np.Array
        Wavefunction (in MPS or MPO form) to be decomposed. 
    As : list of list of np.Array
        List of column wavefunctions, evaluated left to right using shifted
        protocol.
    Returns
    -------
    Us : list of np.Array
        List of unitaries 
    As : list of list of np.Array
        List of column wavefunctions. This is the same as the original list, 
        but As[-1] has been contracted with Us.
    Lambda : list of np.Array
        Product state
    """
    start = time.time()

    L = len(Psi)
    assert np.allclose(L, [len(A) for A in As])
    b = np.zeros([2,1,1,1])
    b[0,0,0,0] = 1.0
    Lambda = [b.copy() for i in range(L)]
    A = contract_series_diagonal_expansion(As[:-1], As[-1]) 
    print(time.time() - start)
    # Starting left parts
    Lp = np.tensordot(Psi[0], A[0].conj(), [[0,2],[0,2]]).transpose(0,1,3,2)
    Lp_list = [Lp]
    for i in range(1, L):
        Lp = np.tensordot(Lp, Psi[i], [1,2])
        Lp = np.tensordot(Lp, A[i].conj(), [[3,1],[0,2]])
        Lp = np.tensordot(Lp, Lambda[i-1].conj(), [[0,1],[1,2]]).transpose(0,1,3,5,4,2)
        # Last two indices are the region to trace over
        Lp_list.append(Lp)
        Lp = np.trace(Lp, axis1=4, axis2=5)

    Rp = np.einsum('ijklmm->ijkl', Lp_list[-1])
    U = np.tensordot(Rp, Lambda[-1].conj(), [[0,1,3],[1,3,2]])
    X, S, Z = np.linalg.svd(U, full_matrices=False)
    U = np.dot(X, Z)
    A[-1] = np.tensordot(A[-1], U, [3,0])
    Us = [U]

    Rp = Lambda[-1].transpose(1,3,0,2)
    for i in range(L-1, 0, -1):
        U = np.tensordot(Rp, Lp_list[i], [[0,1,2,3],[0,1,2,3]])
        X, S, Z = np.linalg.svd(U, full_matrices=False)
        U = np.dot(X, Z)
        A[i] = np.tensordot(A[i], U, [1,0]).transpose(0,3,1,2)
        Us.append(U)

        Rp = np.tensordot(Rp, Psi[i], [[0,1],[1,3]])
        Rp = np.tensordot(Rp, A[i].conj(), [[0,2],[3,0]])
        Rp = np.tensordot(Rp, Lambda[i-1].conj(), [[0,2],[3,0]]).transpose(2,0,1,3)

    Us = Us[::-1]

    # You need to do this!
    A = As[-1]
    for i in range(1,L):
        A[i] = np.tensordot(A[i], Us[i-1], [1,0]).transpose(0,3,1,2)
    A[L-1] = np.tensordot(A[L-1], Us[L-1], [3,0])
    As = As[:-1].copy()
    As.append(A)
    return(Us, As, Lambda)


