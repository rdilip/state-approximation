"""
Module that variationally decomposes an MPS into a series of quantum gates
and a low-entanglement MPS
"""

from rfunc import pad, pad_mps
from misc import group_legs, ungroup_legs, mps_2form, mps_overlap, mpo_on_mpo,\
     mps_entanglement_spectrum, svd_theta_UsV
import numpy as np
from scipy.linalg import expm
from scipy.stats import unitary_group
from moses_simple import moses_move as moses_move_simple
from moses_variational_shifted import moses_move as moses_move_shifted,\
                                      optimize_single_site_sweep_fast,\
                                      optimize_single_site_sweep_faster
from disentanglers import disentangle_S2, disentangle_brute, disentangle_ls
from tebd import tebd
from glob import glob
import pickle
from contraction_shifted import contract_series_diagonal_expansion
import warnings
from copy import deepcopy
from rfunc import mps2mpo, mpo2mps, entanglement_entropy

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

def invert_mpo(mpo):
    """ Inverts an MPO along non physical degrees of freedom """
    return([T.transpose([0,1,3,2]) for T in mpo[::-1]])

def diagonal_expansion(Psi, eta=None, disentangler=disentangle_S2, num_sweeps=None,
                        final_run=False):
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
    # Padding to make sure we get nice 2 site gates....
    L = len(Psi)
    Psi = deepcopy(Psi)
    max_bond_dim = max(sum([i.shape for i in Psi], ()))
    if max_bond_dim < 4:
        for i in range(L):
            shape = Psi[i].shape
            if i != L-1:
                Psi[i] = pad(Psi[i], -1, 4)
            if i != 0:
                Psi[i] = pad(Psi[i], -2, 4)
            
    # Check MPO

    if Psi[0].ndim == 3:
        Psi = mps2mpo(Psi)
    if eta is None:
        eta = 2*max(sum([i.shape for i in Psi], ()))
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
    if pR != 4:
        psi = pad(psi, 1, 4)
    psi = psi.reshape(pL, d, d, chiS, chiN).transpose([0,1,3,4,2]).reshape(pL,d,chiS,d)
    A0[-1] = psi.copy()

    # send second physical leg to the top

    # Splitting last tensor of Lambda
    psi = Lambda[-1]
    pL, pR, chiS, chiN = Lambda[-1].shape
    assert(pR == chiN == 1)
    if pL != 4:
        psi = pad(psi, 0, 4)

    d1 = d2 = d
    psi = psi.reshape(d1,d2,pR,chiS,chiN).transpose([0,2,3,1,4])
    psi = psi.reshape(d*pR*chiS,d)
    theta = psi.reshape(pR*chiS,d,d,1)

     
    Utheta, U = disentangler(theta)
    debug = A0.copy()
    A0[-1] = np.tensordot(A0[-1], U, [[1,3],[2,3]]).transpose([0,2,1,3])

    # NOTE: I think there may be a bug somewhere here, that we're only
    # getting away with because everything is bond dimension 2... no evidence
    # to suggest this but just have a feeling.

    psi = psi.reshape(d,pR,chiS,d,1,1)
    psi = np.tensordot(psi, U.conj(), [[0,3],[2,3]]).transpose([4,0,1,5,2,3])
    pL1, pR1, chiS, pL2, pR2, chiN = psi.shape
    psi = psi.reshape(pL1*pR1*chiS, pL2*pR2*chiN)

    # TODO change this
    X, s, Z, chi_c, _ = svd_theta_UsV(psi, eta, p_trunc=0.0)
    q = X
    r = np.diag(s) @ Z
    #q, r = np.linalg.qr(psi)

    q = q.reshape(pL1, pR1, chiS, -1)
    r = r.reshape(-1, pL2, pR2, chiN).transpose([1,2,0,3])
    Lambda[-1] = q

    Lambda.append(r)

    # Variational moses move
    if num_sweeps is not None:
        A0, Lambda, F = moses_move_shifted(Psi_copy, 
                                           A=A0,
                                           Lambda=Lambda,
                                           N=num_sweeps,
                                           get_fidelity=True,
                                           final_run=final_run)
    return A0, Lambda, F

def multiple_diagonal_expansions(Psi, 
                                 depth, 
                                 num_sweeps=10,
                                 schedule_eta=None,
                                 schedule_mode='trunc',
                                 verbose=True,
                                 disentangler=disentangle_S2):
    """
    Perform n diagonal expansions. Returns all the Ai and Lambda such that
    \prod A_0 A_1...A_{n-1} Lambda ~= Psi

    By default will halve max bond dimension of Lambda at each step

    Parameters
    ----------
    Psi : list of np.Array
        Wavefunction to expand
    depth : int
        Number of contractions. No reason to do more than log_2 chi_max
    num_sweeps : int 
        Number of variational sweeps.
    schedule_eta : list
        List of bond dimensions. If not supplied, defaults to halving. 
    schedule_mode : str
        String that determiens how to interpret the schedule. The following 
        are valid options.
            * trunc : Throws away schedule_eta[i] on the ith step, but does not
                      account for any disentangling from the MM.
            * trunc_exact : Throws away schedule_eta[i] on the ith step. Performs
                            MM twice to determine how much to throw away.
            * entropy : Keeps schedule_eta[i] on the ith step, not accounting
                        for MM disentangling.
            * entropy_exact : Keeps schedule_eta[i] on the ith step, accounts
                             for MM disentangling.
            * bond_dim : Uses max bond dim of schedule_eta[i] on the ith step.
        The difference between "mode" and "mode_exact" is as follows. When we
        do the MM on Lambda s.t. Lambda_n = A.Lambda{n+1}, we need to know the
        bond dimension to truncate at each step. If we just do mode, we guess
        this bond dimension from Lambda_n, but this will probably be an over-
        estimate because the MM does some disentangling. If we do mode_exact,
        it takes 2x the time, but we do the MM, figure out how much entanglement
        is peeled away, then do the MM again for real. 
    """

    if Psi[0].ndim == 3:
        Psi = mps2mpo(Psi)
    max_bond_dim = max(sum([i.shape[-2:] for i in Psi], ()))
    if schedule_eta is None:
        schedule_eta = [max_bond_dim for i in range(depth)]

    As, Lambdas = [], []
    L = len(Psi)
    Lambda = Psi.copy()
    info = dict(Ss=[], Lambdas=[], fidelities=[])
    eta_max = max(sum([i.shape for i in Psi], ()))
    prev_eta = max_bond_dim
    
    schedule = []
    for i in range(depth):
        if i != depth - 1:
            if schedule_mode == 'bond_dim':
                eta_max = schedule_eta[i]
            elif schedule_mode == 'trunc':
                s = mps_entanglement_spectrum(Lambda)[int(L//2)]
                eta_max = len(s) - np.where(np.cumsum((s**2)[::-1]) > schedule_eta[i])[0][0]
            elif schedule_mode == 'trunc_exact':
                A0_trial, Lambda_trial, F = diagonal_expansion(Lambda.copy(),
                                                               eta=10000,
                                                               num_sweeps=num_sweeps,
                                                               final_run=False,
                                                               disentangler=disentangler)
                s = mps_entanglement_spectrum(Lambda_trial)[int(L//2)]
                eta_max = len(s) - np.where(np.cumsum((s**2)[::-1]) > schedule_eta[i])[0][0]
            elif schedule_mode == 'entropy':
                s = mps_entanglement_spectrum(Lambda)[int(L//2)]
                all_possible_truncations = []

                decreasing = True

                diff = np.float('inf')
                prev_diff = diff
                for chi_max in range(len(s)+1, 0, -1):
                    s_trunc = s[:chi_max]
                    s_trunc /= np.linalg.norm(s_trunc)
                    ee = -np.sum((s**2) * np.log(s**2))
                    diff = np.abs(ee - schedule_eta[i])
                    if diff > prev_diff:
                        eta_max = chi_max + 1
                        break
                    else:
                        prev_diff = diff
            elif schedule_mode == 'entropy_exact':
                if verbose:
                    print(f"Starting depth {i+1}")
                A0_trial, Lambda_trial, F = diagonal_expansion(Lambda.copy(),
                                                               eta=10000,
                                                               num_sweeps=num_sweeps,
                                                               final_run=False,
                                                               disentangler=disentangler)
                s = mps_entanglement_spectrum(Lambda_trial)[int(L//2)]
                all_possible_truncations = []

                decreasing = True

                diff = np.float('inf')
                prev_diff = diff
                for chi_max in range(len(s)+1, 0, -1):
                    s_trunc = s[:chi_max]
                    s_trunc /= np.linalg.norm(s_trunc)
                    ee = -np.sum((s**2) * np.log(s**2))
                    diff = np.abs(ee - schedule_eta[i] - 0.5)
                    if diff > prev_diff:
                        eta_max = chi_max + 1
                        break
                    else:
                        prev_diff = diff 
            else:
                raise ValueError("Not a valid scheduling mode.")


        # There is no situation in which we shouldn't be doing this...
        if i == depth - 1:
            eta_max = 1
        schedule.append(eta_max)

        if prev_eta == 1:
            if verbose:
                print(f"Reached product state after {i} iterations.")
            break

        prev_eta = eta_max
        if eta_max == 1:
            num_sweeps *= 2
        A0, Lambda, F = diagonal_expansion(Lambda.copy(), eta=eta_max, num_sweeps=num_sweeps,\
                                           final_run=(eta_max==1), disentangler=disentangler)
        As.append(A0)

        Lambda = mps_2form(Lambda, 'B')
        info['Ss'].append(entanglement_entropy(Lambda))
        info['Lambdas'].append(Lambda)
        info['fidelities'].append(F)
        
    info['schedule'] = schedule

    if Psi[0].ndim == 3:
        Psi = mps2mpo(Psi)
    if verbose:
        print(schedule)

    return As, Lambda, info

def expansion_from_left(Psi, depth):
    """
    Expands a wavefunction Psi in the following manner. Performs a MM so that
    Psi = A.Lambda. Contracts Psi with A.conj() (looking at overlap) and throws
    away Lambda, then performs another moses move. 
    """
    Psi = deepcopy(Psi)
    if Psi[0].ndim == 3:
        Psi = mps2mpo(Psi)
    As = []
    num_sweeps=10

    for i in range(depth):
        print(i)
        if i == depth-1:
            eta_max = 1
        else:
            eta_max = max(sum([i.shape[-2:] for i in Psi],()))
        A0, Lambda, F = diagonal_expansion(Psi,
                                           eta=eta_max,
                                           num_sweeps=num_sweeps,
                                           final_run=(i==depth-1))
        As.append(deepcopy(A0))
        A0 = [a.conj() for a in A0]
        Psi = [psi.transpose(1,0,2,3) for psi in Psi]
        Psi = mpo_on_mpo(Psi, A0)
        Psi = [psi.transpose(1,0,2,3) for psi in Psi]
        return Psi
    return As, Lambda
        


if __name__ == '__main__':
    fnames = glob("sh_comparison/*pkl") 
    
    for fname in fnames:
        print(fname)
        with open(fname, "rb") as f:
            data = pickle.load(f)
        fname = "sh_data/" + fname.split("/")[1]
        with open(fname, "rb") as f:
            Psi = pickle.load(f)

        As = data['As']
        Lambda = data['Lambda']

        Psi = mps2mpo(Psi)
        As, Lambda, Fs = optimize_single_site_sweep_fast(Psi, As)

        fname = "adam_comparison/" + fname.split("/")[1]
        with open(fname, "wb+") as f:
            pickle.dump(dict(As=As, Lambda=Lambda, Fs=Fs), f)

