from matplotlib import pyplot as plt
import numpy as np
from rfunc import pad
import scipy as sp
from disentanglers import disentangle_S2, disentangle_brute
from misc import svd, svd_theta_UsV, group_legs, ungroup_legs
from warnings import warn
import time

def find_closest_factors(num):
    factor = int(np.sqrt(num))
    while factor >= 0:
        if num % factor == 0:
            return factor, int(num / factor)
        else:
            factor -= 1

def split_psi(Psi,
              dL,
              dR,
              truncation_par={
                  'chi_max': 1000,
                  'p_trunc': 1e-10
              },
              verbose=0,
              n=2,
              eps=1e-6,
              max_iter=120,
              init_from_polar=True,
              pref='dL',
              disentangler=disentangle_S2):
    """ Given a tripartite state psi.shape = d x mL x mR, find an
    approximation
	
        psi = A.Lambda

    where A.shape = dL x dR x d  is an isometry that "splits" d --> dL x
    dR and Lambda.shape = mL dL dR mR is a 2-site TEBD-style wavefunction
    of unit norm and maximum Schmidt rank 'chi_max.'

    The solution for Lambda is given by the MPS-type decomposition

    Lambda_{mL dL dR mR} = \sum_eta  S_{dL mL eta} B_{dR eta mR}

    where 1 < eta <= chi_max, and Lambda has unit-norm

    Arguments:
  
        psi:  shape= d, mL, mR
       
        dL, dR: ints specifying splitting dimensions (dL,dR maybe
        reduced to smaller values)
       
        truncation_par  = {'chi_max':, 'p_trunc':}  truncation
        parameters for the SVD of Lambda; p_trunc is acceptable
        discarded weight
       
        eps: precision of iterative solver routine
       
        max_iter: max iterations of routine (through warning if this is
        reached!)

        pref: Whether to push the high bond dimension to dL or dR

        disentangler: Which disentangler to use. We should probably come up
        with some standard format for disentanglers...
  
    Returns:
  
        A: d x dL x dR S: dL x mL x eta B: dR x eta x mR
  
        info = {} , a dictionary of (optional) errors
                'error': 'trunc_leg','trunc_bond' """

    d, mL, mR = Psi.shape

    dL1, dR1 = find_closest_factors(min(d,mL*mR))
    if pref == 'dL':
        dL, dR = max(dL1, dR1), min(dL1, dR1)
    elif pref == 'dR':
        dL, dR = min(dL1, dR1), max(dL1, dR1)
    else:
        raise ValueError
 

    X, y, Z = svd(Psi.reshape(-1, mL*mR), full_matrices=False)
    D2 = len(y)

    
    A = X
    theta = (Z.T * y).T
    if init_from_polar:
        psi = theta.reshape((D2, mL, mR))
        #First truncate psi to (D2, dL, dR) based on Schmidt values
        #This fix has sharp edges
        if mL > dL:
            rho = np.tensordot(psi, psi.conj(), axes=[[0, 2], [0, 2]])
            p, u = np.linalg.eigh(rho)
            if mR < dR:
                u = u[:, -(dL*dR):]
            else:
                u = u[:, -dL:]
            psi = np.tensordot(psi, u.conj(), axes=[[1],
                                                    [0]]).transpose([0, 2, 1])
        if mR > dR:
            rho = np.tensordot(psi, psi.conj(), axes=[[0, 1], [0, 1]])
            p, u = np.linalg.eigh(rho)
            if mL < dL:
                u = u[:,-(dL*dR):]
            else:
                u = u[:, -dR:]
            psi = np.tensordot(psi, u.conj(), axes=[[2], [0]])
        psi /= np.linalg.norm(psi)
        u, s, v = svd(psi.reshape(D2, D2), full_matrices=False)
        Zp = np.dot(u, v)
        A = np.dot(A, Zp)

        theta = np.dot(Zp.T.conj(), theta)


    # Disentangle the two-site wavefunction

    theta = np.reshape(theta, (dL, dR, mL, mR))  #view TEBD style
    theta = np.transpose(theta, (2, 0, 1, 3))

    s = np.linalg.svd(theta, compute_uv=False)
    sb = -np.log(np.sum(s[np.abs(s) > 1.e-10]**2))

    chiL, d1, d2, chiR = theta.shape
    U = np.eye(d1*d2).reshape(d1,d2,d1,d2)
    if disentangler == disentangle_S2:
        theta, U = disentangle_S2(theta,
                                      eps=10 * eps,
                                      max_iter=max_iter)
    else:
        theta, U  = disentangler(theta)

    #theta, U = disentangle_brute(theta)
    s = np.linalg.svd(theta, compute_uv=False)
    s = s[np.abs(s) > 1.e-10]
    A = np.tensordot(A,
                 np.reshape(U, (dL, dR, dL * dR)),
                 axes=([1], [2]))

    theta = np.transpose(theta, [1, 0, 2, 3])
    theta = np.reshape(theta, (dL * mL, dR * mR))

    #X, s, Z = svd(theta, full_matrices=False)
    X, s, Z, chi_c, trunc_bond = svd_theta_UsV(theta, truncation_par['chi_max'],p_trunc=0.0)
    chi_c = len(s)

    S = np.reshape(X, (dL, mL, chi_c))
    S = S * s

    
    B = np.reshape(Z, (chi_c, dR, mR))
    B = np.transpose(B, (1, 0, 2))

    if dR == 1:
        A = pad(A, 2, 2)
        B = pad(B, 0, 2)

    return A, S, B

def cleanup(a, S, factor):
    """ Given a, S, B, assuming that part of the high bond dimension leg was
    split off and is contained in the first leg of a, redistributes the legs.

    Parameters
    ----------
    factor: The factor that the bond dimension was reduced by.
    """
    # TODO write a docstring that isn't stupidly incomprehensible

    mL, dL, dR = a.shape
    _, chiV, eta = S.shape
    breakpoint()

    pL = int(mL / factor)

    a = a.reshape(pL, factor, dL, dR)

    breakpoint()
    theta = np.tensordot(a, S, [2,0]).transpose([0,2,1,3,4])
    breakpoint()
    theta = theta.reshape(pL, dR, factor*chiV, eta)
    breakpoint()
    theta = theta.reshape(pL * dR, factor*chiV*eta)
    breakpoint()
    a, s = np.linalg.qr(theta)
    breakpoint()
    a = a.reshape(pL, a.shape[1], dR)
    breakpoint()
    s = s.reshape(a.shape[1], factor*chiV, eta)
    return(a, s)

def smallest_factor(num):
    """ Returns the smallest factor of num """
    m = 2
    while m <= np.sqrt(num):
        if num % m == 0:
            return m
        m += 1
    raise ValueError("Bond dimension is prime. Can't enforce constraint")
