from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
from disentangler import renyi_hessian
from misc import svd, svd_theta_UsV, group_legs, ungroup_legs
import time



def U2(psi):
    """Entanglement minimization via 2nd Renyi entropy
    
        psi.shape = mL, dL, dR, mR
        
        Returns  S2 and polar disentangler U
    """
    chi = psi.shape

    rhoL = np.tensordot(psi, psi.conj(), axes=[[2, 3], [2, 3]])
    dS = np.tensordot(rhoL, psi, axes=[[2, 3], [0, 1]])
    dS = np.tensordot(psi.conj(), dS, axes=[[0, 3], [0, 3]])
    dS = dS.reshape((chi[1] * chi[2], -1))
    s2 = np.trace(dS)

    X, Y, Z = svd(dS)
    return -np.log(s2), (np.dot(X, Z).T).conj()


def disentangle_2(psi, eps=1e-6, max_iter=30, verbose=0):
    """Disentangles a wavefunction via 2-renyi polar iteration
        
        Returns psiD, U, {Ss}
        
        psiD = U psi
        
        Ss are Renyi2 at each step
    """

    Ss = []
    chi = psi.shape
    U = np.eye(chi[1] * chi[2], dtype=psi.dtype)
    m = 0
    go = True
    while m < max_iter and go:
        s, u = U2(psi)
        U = np.dot(u, U)
        u = u.reshape((chi[1], chi[2], chi[1], chi[2]))
        psi = np.tensordot(u, psi, axes=[[2, 3], [1, 2]]).transpose([2, 0, 1, 3])
        Ss.append(s)
        if m > 1:
            go = Ss[-2] - Ss[-1] > eps
        m += 1

    if verbose:
        print("disentangle2 evaluations:", m, "dS", -np.diff(Ss))

    return psi, U, {'Ss': Ss}
