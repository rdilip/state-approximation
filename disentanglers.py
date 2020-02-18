# coding: utf-8
"""
Conventions:
U is a two site unitary. Indices 2 and 3 act on indices 1 and 2 of the TEBD
style wavefunction theta. 
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def disentangle_S2(theta,eps = 1e-9, max_iter = 200):
    ml,d1,d2,mr=theta.shape
    U = np.eye(d1*d2, dtype = theta.dtype)
    
    m = 0
    go = True
    Ss = []
    while m < max_iter and go:
        s, u = U2(theta) 
        U = np.dot(u, U)
        u = u.reshape((d1,d2,d1,d2))
        theta = np.tensordot(u.conj(), theta, axes = [[2, 3], [1, 2]]).transpose([2, 0, 1, 3])
        Ss.append(s)
        if m > 1:
            go = Ss[-2] - Ss[-1] > eps
        m+=1
    return theta, U.reshape([d1, d2, d1, d2])

def U2(theta):
    chi = theta.shape
    rhoL = np.tensordot(theta, np.conj(theta), axes = [[2, 3], [2, 3]])

    dS = np.tensordot(rhoL, theta, axes = [[2, 3], [0, 1] ])
    dS = np.tensordot( np.conj(theta), dS, axes = [[0, 3], [0, 3]])

    dS = dS.reshape((chi[1]*chi[2], -1))
    s2 = np.trace( dS )
    
    X, Y, Z = np.linalg.svd(dS)
    return -np.log(s2), (np.dot(X, Z).T).conj()

def orthogonal(v,D):
    A = np.zeros([D,D])
    cnt = 0
    for i in range(D):
        for j in range(i):
            A[i,j] =  v[cnt]
            A[j,i] = -v[cnt]
            cnt = cnt + 1
    return np.dot(np.eye(D)+A,np.linalg.inv(np.eye(D)-A))

def renyi_entropy_v(Psi,v):
    chi1,d1,d2,chi2 = Psi.shape[0],Psi.shape[1],Psi.shape[2],Psi.shape[3]
    D = d1*d2
    U = orthogonal(v,D).reshape([d1,d2,d1,d2])
    Psi_p = np.tensordot(U,Psi,axes=([2,3],[1,2])).transpose(2,0,1,3).reshape([d1*chi1,d2*chi2])
    
    s = np.linalg.svd(Psi_p,compute_uv=False)
    s = np.array(s[s>10**(-10)])**2
    return -np.log(np.sum(s**2))    
    
def renyi_entropy(Psi, alpha):
    if len(Psi.shape) == 4:
        chiL, d1, d2, chiR  = Psi.shape
        Psi = Psi.reshape((chiL*d1, chiR*d2))
    s = np.linalg.svd(Psi,compute_uv=False)
    s = np.array(s[s>10**(-10)])**2
    return (1/(1-alpha))*np.log(np.sum(s**alpha))
