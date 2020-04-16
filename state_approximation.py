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
from disentanglers import disentangle_S2, disentangle_brute
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
    
    #if num_sweeps is not None:
    #    A0, Lambda = moses_move_shifted(Psi_copy, A=A0, Lambda=Lambda, N=num_sweeps)
    if num_sweeps is not None:
        if final_run:
            print("Final run starting")

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
                                 schedule_eta=None):
    """ Perform n diagonal expansions. Returns all the Ai and Lambda such that
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
    """

    if schedule_eta[0] is None:
        MODE = 'auto'
    else:
        MODE = 'p_trunc'

    if Psi[0].ndim == 3:
        Psi = mps2mpo(Psi)
    max_bond_dim = max(sum([i.shape for i in Psi], ()))
    if schedule_eta is None:
        schedule_eta = [max_bond_dim for i in range(depth)]

    As, Lambdas = [], []
    Lambda = Psi.copy()
    info = dict(Ss=[], Lambdas=[], fidelities=[])
    eta_max = max(sum([i.shape for i in Psi], ()))

    count_no_change = 0
    product_state = False
    prev_eta = max_bond_dim
    
    schedule = []

    for i in range(depth):
        if type(schedule_eta[i]) == float or type(schedule_eta[i]) == np.float64:
            if i == depth - 1:
                eta_max = 1
            else:
                s = mps_entanglement_spectrum(Lambda)[int(len(Lambda)//2)]
                eta_max = len(s) - np.where(np.cumsum((s**2)[::-1]) > schedule_eta[i])[0][0]
            schedule.append(eta_max)
        else:
            eta_max = schedule_eta[i]

        if MODE == 'auto':
            A0_trial, Lambda_trial, F = diagonal_expansion(Lambda.copy(), eta=10000, num_sweeps=num_sweeps,\
                                           final_run=False)
            s = mps_entanglement_spectrum(Lambda_trial)[int(len(Lambda_trial)//2)]
            ee = -np.sum((s**2) * np.log(s**2))
            desired_ee = ee * ((depth - (i+1)) / float(depth))
            all_possible_truncations = []
            for chi_max in range(len(s)):
                s_trunc = s[:chi_max+1]
                all_possible_truncations.append(s_trunc / np.linalg.norm(s_trunc))
            all_possible_ee = [-np.sum((s**2) * np.log(s**2)) for s in all_possible_truncations]
            eta_max = np.argmin(np.abs(all_possible_ee - desired_ee)) + 1


        print("\t" + f"eta_max={eta_max}")
        if prev_eta == 1:
            print(f"Reached product state after {i} iterations.")
            break
        if i == depth-1 and eta_max != 1:
            breakpoint()
            print("On the final run, but final state won't be trivial.")

        prev_eta = eta_max
        if eta_max == 1:
            num_sweeps *= 2

        A0, Lambda, F = diagonal_expansion(Lambda.copy(), eta=eta_max, num_sweeps=num_sweeps,\
                                           final_run=(eta_max==1))
        As.append(A0)
        

        Lambda = mps_2form(Lambda, 'B')
        info['Ss'].append(entanglement_entropy(Lambda))
        info['Lambdas'].append(Lambda)
        info['fidelities'].append(F)
        
    info['schedule'] = schedule

    if Psi[0].ndim == 3:
        Psi = mps2mpo(Psi)

    return As, Lambda, info

        
def largest_power_of_two(num):
    """ Returns the largest power of two less than this number """
    assert num > 1
    power_of_two = 1.
    while power_of_two < num:
        power_of_two *= 2
    return int(power_of_two / 2.)

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

    #Ts = np.linspace(0.0, 9.9, 100)
    #fnames = [f"T{round(i,1)}.pkl" for i in Ts]
    #for i, fname in enumerate(fnames):
    #    with open(f"/space/ge38huj/state_approximation/sh_data/{fname}", "rb") as f:
    #        sh_state = pickle.load(f)
    #    Psi = mps2mpo(sh_state.copy())
    #    Lambda = Psi.copy()
    #    print("Starting expansion")
    #    As, Lambda, info = multiple_diagonal_expansions(Psi,100)
    #    out = contract_series_diagonal_expansions(As, Lambda)
    #    overlap = mps_overlap(out, Psi)

    #    output_file = dict(As=As, Lambda=Lambda, info=info, fidelity=overlap, T=Ts[i])
    #    
    #    with open(f"sh_comparison_500_sweeps/{fname}", "wb+") as f:
    #        pickle.dump(output_file, f)
