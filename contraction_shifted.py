"""
Module for contractions in the shifted protocol.
"""

import numpy as np
import warnings
from misc import group_legs, mps_2form
from copy import deepcopy
from rfunc import mps2mpo, mpo2mps

def contract_diagonal_expansion_top(A0, Lambda):
    """ 
    Contracts A0 and Lambda with Lambda shifted one tensor upwards. The bottom
    most tensor of A0 becomes the bottom most tensor of the output mpo, and the
    top most tensor of the mpo is the contraction of A0[-1], Lambda[-1], and
    Lambda[-2]

    is a bit awkward -- we can't 
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
    out = deepcopy(A0)
    Lambda = deepcopy(Lambda)
    for i in range(1, len(A0)-1):
        #print(f"\t{i}")
        prod = np.tensordot(A0[i], Lambda[i-1], [1,0])
        prod = group_legs(prod, [[0],[3],[1,4],[2,5]])[0]
        out[i] = prod

    last_tensor = np.tensordot(A0[-1], Lambda[-2], [1,0])
    last_tensor = np.tensordot(last_tensor, Lambda[-1], [[2,5],[0,2]])

    last_tensor = group_legs(last_tensor, [[0],[2,4],[1,3],[5]])[0]
    out[-1] = last_tensor
    return(out)

def contract_diagonal_expansion_bottom(A0, A1):
    """ 
    Contracts A0 and A1 with Lambda shifted one tensor upwards. The top 
    most tensor of A0 becomes the bottom most tensor of the output mpo, and the
    bottom tensor of the mpo is the contraction of A0[0], A1[0], and
    A1[1]

    is a bit awkward -- we can't 
    Parameters
    ----------
    A0 : list of np.Array
        Left column mpo
    A1 : list of np.Array
        Right column wavefunction
    Returns
    -------
    contracted : list of np.Array
        A0.Lambda
    """
    L = len(A1)
    assert L == len(A0)
    contracted = [[] for i in range(L)]
    first_tensor = np.tensordot(A0[0], A0[1], [3,2])
    first_tensor = np.tensordot(first_tensor, A1[0], [[1,4],[2,0]])
    first_tensor = group_legs(first_tensor, [[0,2],[4],[1],[3,5]])[0]
    contracted[0] = first_tensor
    for i in range(2, L):
        contracted[i-1] = np.tensordot(A0[i], A1[i-1], [1,0])
        contracted[i-1] = group_legs(contracted[i-1], [[0],[3],[1,4],[2,5]])[0]
    last_tensor = group_legs(A1[-1], [[0,2],[1],[3]])[0]
    last_tensor = last_tensor.reshape(1,*last_tensor.shape).transpose(0,2,1,3)
    contracted[L-1] = last_tensor
    return contracted

def contract_mpo(mpo):
    """ Contracts an mpo """
    out = mpo[0]
    for i in range(1, len(mpo)):
        out = np.tensordot(out, mpo[i], [-1,-2])
    return out


def contract_diagonal_expansion_full(A0, A1):
    """
    Contracts two columns shifted relative to each other. Outputs an MPO. 
    This destroys the isometric form, but we're contracting so who cares.
    Parameters
    ----------
    A0 : list of np.Array
        Left hand column
    A1 : list of np.Array
        Right hand column
    Returns
    -------
    mpo : list of np.Array
        Output mpo
    """
    L = len(A0)
    assert L == len(A1)
    out = [[] for i in range(L)]

    R = deepcopy(A0[0])
    for i in range(L-1):
        contracted = np.tensordot(R, A0[i+1], [3,2])
        contracted = np.tensordot(contracted, A1[i], [[1,4],[2,0]]).transpose(0,1,4,2,3,5)
        pL, chiS, pR, pL_next, chiNW, chiNE = contracted.shape
        contracted = contracted.reshape(pL*chiS*pR, pL_next*chiNW*chiNE)
        Q, R = np.linalg.qr(contracted)
        out[i] = Q.reshape(pL, chiS, pR, -1).transpose(0,2,1,3)
        R = R.reshape(-1, pL_next, chiNW, chiNE).transpose(1,3,0,2)
    R = np.tensordot(R, A1[L-1], [[1,3],[2,0]]).transpose(0,2,1,3)
    out[L-1] = R
    return(out)

def contract_series_diagonal_expansion(As, Lambda, n=None, mode='exact'):
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
    mode : str
        How to contract. We can take advantage of trivial legs sometimes and
        speed up contraction 
    Returns
    -------
    contracted : list of np.Array
        Full contracted mps.
    """

    if n is None:
        n = len(As)
    if mode == 'exact':
        warnings.warn("The exact mode can take a very long time to run.")
        contracted = As[0]
        for i in range(1, n):
            contracted = contract_diagonal_expansion_full(contracted, As[i])
        contracted = contract_diagonal_expansion_full(contracted, Lambda)
    if mode == 'top':
        contracted = deepcopy(Lambda)
        for i in range(n-1, -1, -1):
            contracted = contract_diagonal_expansion_top(As[i], contracted)
            contracted = mps_2form(contracted, form='A', svd_min=1.e-10)
    return contracted

