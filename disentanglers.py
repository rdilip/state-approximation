# coding: utf-8
"""
Conventions:
U is a two site unitary. Indices 2 and 3 act on indices 1 and 2 of the TEBD
style wavefunction theta. 
"""
import numpy as onp
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import unitary_group
import warnings

def disentangle_S2(theta,eps = 1e-9, max_iter = 1000):
    ml,d1,d2,mr=theta.shape
    U = onp.eye(d1*d2, dtype=theta.dtype)    
    m = 0
    go = True
    Ss = []
    while m < max_iter and go:
        s, u = U2(theta) 
        U = onp.dot(u, U)
        u = u.reshape((d1,d2,d1,d2))
        theta = onp.tensordot(u.conj(), theta, axes = [[2, 3], [1, 2]]).transpose([2, 0, 1, 3])
        Ss.append(s)
        if m > 1:
            go = Ss[-2] - Ss[-1] > eps
        m+=1
    return theta, U.reshape([d1, d2, d1, d2]), Ss

def disentangle_brute(theta):
    warnings.warn("The brute disentangler doesn't work with complex values")
    mL, d1, d2, mR = theta.shape

    # Initial guess
    v0 = onp.random.random(int(((d1*d2)**2 - d1*d2) / 2))
    v0 = onp.array(v0)
    #v0_imag = onp.random.random(int(((d1*d2)**2 - d1*d2) / 2))
    #v0 += 1.j * v0_imag

    res = minimize(renyi_entropy_v, x0=v0, args=(theta))
    U = cayley(res.x, d1*d2).reshape([d1,d2,d1,d2])
    Utheta = onp.tensordot(U.conj(), theta, [[2,3],[1,2]]).transpose([2,0,1,3])
    return(Utheta, U)

def initial_guess(mode, theta):
    """ Returns an initial guess for a Cayley parametrization of a D by D unitary.
    mode : str
        One of random or polar
    theta : onp.Array
        Wavefunction to be disentangled. This is technically more information than
        we always need -- for random we only need the dimensions of theta -- 
        but it doesn't matter too much.
    """
    mL, d1, d2, mR = theta.shape
    if mode == 'random':
        return(onp.random.random(int(((d1*d2)**2 - d1*d2) / 2)))
    if mode == 'polar':
        X = theta.transpose([1,2,0,3]).reshape([d1*d2, mL*mR])
        p, u = onp.linalg.polar(X)
        u = u.reshape([d1*d2, mL*mR]).transpose([2,0,1,3])
        # TODO finish this...

def U2(theta):
    chi = theta.shape
    rhoL = onp.tensordot(theta, onp.conj(theta), axes = [[2, 3], [2, 3]])

    dS = onp.tensordot(rhoL, theta, axes = [[2, 3], [0, 1] ])
    dS = onp.tensordot( onp.conj(theta), dS, axes = [[0, 3], [0, 3]])

    dS = dS.reshape((chi[1]*chi[2], -1))
    s2 = onp.trace( dS )
    
    X, Y, Z = onp.linalg.svd(dS)
    return -onp.log(s2), (onp.dot(X, Z).T).conj()

def cayley(v,D):
    A = onp.zeros([D,D], dtype='complex128')
    cnt = 0
    for i in range(D):
        for j in range(i):
            A[i,j] =  v[cnt]
            A[j,i] = -v[cnt].conj()
            cnt = cnt + 1
    return onp.dot(onp.eye(D)+A,onp.linalg.inv(onp.eye(D)-A))

def renyi_entropy_v(v, Psi):
    chi1,d1,d2,chi2 = Psi.shape[0],Psi.shape[1],Psi.shape[2],Psi.shape[3]
    D = d1*d2
    U = cayley(v,D).reshape([d1,d2,d1,d2])
    Psi_p = onp.tensordot(U,Psi,axes=([2,3],[1,2])).transpose(2,0,1,3).reshape([d1*chi1,d2*chi2])
    
    s = onp.linalg.svd(Psi_p,compute_uv=False)
    if onp.any(onp.imag(s) > 1.e-10):
        raise ValueError

    s = onp.array(s[s>10**(-10)])**2
    return -onp.log(onp.sum(s**2))    
    
def renyi_entropy(Psi, alpha):
    if len(Psi.shape) == 4:
        chiL, d1, d2, chiR  = Psi.shape
        Psi = Psi.reshape((chiL*d1, chiR*d2))
    s = onp.linalg.svd(Psi,compute_uv=False)
    if onp.any(onp.imag(s) > 1.e-10):
        raise ValueError
    s = onp.array(s[s>10**(-10)])**2
    return (1/(1-alpha))*onp.log(onp.sum(s**alpha))
