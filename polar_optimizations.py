""" This module contains functions for performing a polar optimization over
a series of MPOs. """

import numpy as np
from misc import group_legs, ungroup_legs

def contract_without_grouping(X, Y):
    """
    Contracts two rank 4 tensors in order WESN without grouping the legs.
    Output format is W E (S left) (S right) (N left) (N right)
    """
    return np.tensordot(X, Y, [[1,0]]).transpose([0,3,1,4,2,5])

def _contract_around_i(lenv, renv, mpo, i):
    """ 
    lenv and renv are left and right environments, which are mpos.
    mpo is the center mpo on which are are removing the ith index.

    Return env -- a list of mpos for tensors above i and below i. Tensors
    at the same level as i need to be dealt with separately.
    """
    L = len(mpo)
    assert L == len(lenv) == len(renv)
    env = lenv.copy()
    env[i] = None

    # contract left:
    for j in range(i):
        env[j+1] = contract_without_grouping(lenv[j+1], mpo[j])
    for j in range(i+1, L-1):
        env[j+1] = contract_without_grouping(lenv[j+1], mpo[j])
    env[L-1] = np.tensordot(env[L-1], mpo[L-1], [[4,5],[0,2]])
    
    # contract right
    for j in range(1, i):
        env[j+1] = np.tensordot(env[j+1], renv[j-1], [1,0]).transpose([0,6,2,3,7,4,5,8])
    for j in range(i+1, L-2):
        env[j+1] = np.tensordot(env[j+1], renv[j-1], [1,0]).transpose([0,6,2,3,7,4,5,8])
    last_three_tensors = np.tensordot(renv[L-1], renv[L-2], [2,3])
    last_three_tensors = np.tensordot(last_three_tensors, renv[L-3], [5,3])
    env[L-1] = np.tensordot(env[L-1], last_three_tensors, [[1,4,5],[5,3,0]]) 
    return env

def _check_dim(tn):
    # mpo
    if tn.ndim == 4:
        if tn[0].shape[2] != 1 or tn[-1].shape[3] != 1:
            raise ValueError("MPO must have trivial exterior legs")
    elif tn.ndim == 3:
        if tn[0].shape[1] != 1 or tn[-1].shape[2] != 1:
            raise ValueError("MPS must have trivial exterior legs")

def _mpo_on_mps(mpo, mps, side='L'):
    """ 
    Contracts an MPO with an MPS. 

    Parameters
    ----------
    mpo
        MPO with index format WESN
    mps
        mps with index format pLR (physical left right)
    side
        Either L or R. If L, the MPS is taken to be to the left and is contracted
        with the W index of the MPo. If R, the MPS is contracted with index E
    Returns
    -------
    contracted
        A single large contracted tensor with indices 0 through L-1 corresponding
        to physical legs.
    """
    assert side in ['L', 'R']
    L = len(mpo)
    d = mps[0].shape[0]
    _check_dim(mpo)
    _check_dim(mps)

    if side == 'R':
        mpo = [T.transpose([1,0,2,3]) for T in mpo]
    contracted_list = []
    for i, t in enumerate(mps):
        out = np.tensordot(t, mpo[t], [0,0])
        contracted_list.append(out.transpose([2,0,3,1,4]))

    contracted = contracted_list[0]
    for i in range(1, L):
        contracted = np.tensordot(contracted, contracted_list[i], [[3,4],[1,2]])
    contracted = contracted.reshape([d] * L)
    return contracted

def _contract_env(env, prev, i):
    """ Contracting the environment can be a small pain, because the bulk
    tensors look different than the boundary tensors (because of this offsetting
    that reflects a quantum circuit). This function handles that and contracts
    env[i] with prev. The index is provided because the behavior of the function
    does depend on the index called.

    It is the caller's responsibility to ensure that env[i] is NOT the removed
    tensor. Otherwise you will suffer an error.

    Parameters
    ----------
    env : list
        Object returned by _contract_around_i()
    prev : np.Array
        Contracted tensor of all tensors below i
    i : int
        Contracts env[i] w prev
    """
    if i == 1:
        out = np.tensordot(prev, env[i], [3,2])
    elif i == 2:
        out = np.tensordot(prev, env[i], [[6,7],[2,3]])
    elif i == len(env) - 1:
        out = np.tensordot(prev, env[-1], [[-1,-2,-3], [7,2,1]])
    return out
        
def construct_env(lenv, renv, mpo, inp, out, i):
    """ Input acts from the left, so is the highly entangled state """
    env = _contract_around_i(lenv, renv, mpo, i)
    everything_below_i = env[0]
    # TODO handle edge case where i is the topmost tensor
    # shit that is a truly awful edge case...
    everything_above_i = env[i+1]
    for j in range(i):
        everything_below_ = _contract_env(env, everything_below_i, i)
    for j in range(i+1)


