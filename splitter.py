""" Rohit's module for splitting -- essentially, a cleaned up version of Mike's
code, since I'm having some trouble feeling comfortable with everything there """

import numpy as np
#from misc import svd_theta_UsV
from disentanglers import disentangle_S2
from renyin_disentangler import disentangle_2
from rfunc import svd_trunc, svd

def svd_split(T, 
              dL,
              dR,
              truncation_par,
              disentangler=None,
              disentangler_params=None,
              split_mode='KF'):
    """ Splits a tensor T into three tensors A, B, S using a series of SVDs.
    Trying to follow Mike and Frank's codes as much as possible.
    
        mL
        |
    d --T-- mR

    Parameters
    ----------
    T : np.Array
        3 index tensor 
    dL : int
        Left splitting bond dimension. May be reduced further.
    dR : int
        Right splitting bond dimension. May be reduced further
    truncation_par : dict
        Truncation parameters
    disentangler : fn
        Disentangler function. The function should accept a TEBD style wavefunction
        theta with indices numbered left to right, and a dictionary of parameters
        disentangler_params. For more info, see disentanglers.py
    disentangler_params : dict
        Parameters for the disentangler function.
    split_mode : str
        Determines how to do the middle dim splitting. If 'MF', uses Mike and
        Frank's convention (this does not conserve particle number). If 'KF', 
        keeps full dimensions
        """
    assert split_mode in ['MF', 'KF']
    if disentangler is None:
        disentangler = disentangle_S2
    if disentangler_params is None:
        disentangler_params = dict(max_iter=200, eps=1.e-9)

    d, mL, mR = T.shape

    T = T.reshape(d, mL*mR)

    if split_mode == 'KF':
        A, s, R, err = svd_trunc(T, eta='full', p_trunc=0)
        dL, dR = closest_factors(len(s))
    if split_mode == 'MF':
        # No idea why Mike and Frank did this, but if it works it works...
        dL = np.min([dL, mL])
        dR = np.min([dR, mR])
        if dL * dR > d:
            dR = min([int(np.rint(np.sqrt(d))), dR])
            dL = min([d // dR, dL])
        A, s, R, err = svd_trunc(T, eta=dL*dR, p_trunc=0)

    #theta = np.diag(s) @ R
    theta = (R.T * s).T
    D2 = len(s)

    # init from polar
    init_from_polar=True
    if init_from_polar:
        psi = theta.reshape((D2, mL, mR))
        #First truncate psi to (D2, dL, dR) based on Schmidt values
        if mL > dL:
            rho = np.tensordot(psi, psi.conj(), axes=[[0, 2], [0, 2]])
            p, u = np.linalg.eigh(rho)
            u = u[:, -dL:]
            psi = np.tensordot(psi, u.conj(), axes=[[1],
                                                    [0]]).transpose([0, 2, 1])
        if mR > dR:
            rho = np.tensordot(psi, psi.conj(), axes=[[0, 1], [0, 1]])
            p, u = np.linalg.eigh(rho)
            u = u[:, -dR:]
            psi = np.tensordot(psi, u.conj(), axes=[[2], [0]])

        psi /= np.linalg.norm(psi)
        u, s, v = svd(psi.reshape(D2, D2), full_matrices=False)
        Zp = np.dot(u, v)
        A = np.dot(A, Zp)

        theta = np.dot(Zp.T.conj(), theta)
    
    theta = theta.reshape(dL, dR, mL, mR)
    theta = theta.transpose([2,0,1,3])

    # Disentangling
    Utheta, U = disentangler(theta, **disentangler_params)
    theta, U, info = disentangle_2(theta, eps=10*eps, max_iter=200,verbose=0)
    R = Utheta.transpose([2,3,0,1]).reshape([dR*mR, dL*mL])

    B, S = np.linalg.qr(R)
    B, s, Z, backerr = svd_trunc(R, eta='full', p_trunc=0.0)

    S = np.diag(s) @ Z

    B = B.reshape(dR, mR, B.shape[1]).transpose((0,2,1))
    S = S.reshape(S.shape[0], dL, mL).transpose((1,2,0))

    A = A.reshape(d, dL, dR)
    A = np.tensordot(A, U.conj(), [[1,2],[2,3]])

    info = {'s_Lambda': s,
            'error': err + backerr,
            'd_error': err
            }

    return(A, S, B, info)

def closest_factors(num):
    """ Given input number num, finds A, B s.t. |A-B| is minimized and 
    A*B = num """
    factor = int(np.sqrt(num))
    while factor >= 0:
        if num % factor == 0:
            return factor, int(num / factor)
        else:
            factor -= 1


