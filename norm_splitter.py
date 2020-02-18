from __future__ import absolute_import
# from __future__ import division
from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from misc import svd

### psi is the 3-leg wavefunction
#     1
#     |
# 0--- ---2
#    psi
#
# 0 is the "special" leg to be split

###  Limits on split bond dimensions:
#          mL
#          |
#        S |
#         / \
#    dL  /   \  chi_max
#       /     \
# d ---/----------- mR
#      A  dR   B
#

### Leg ordering and arrows of returned A, S, B
#            1
#       A  _/
#          /`
#  0 ->---/--->- 2
#
#          1
#          |
#          v
#        S |
#        _/ \_
#        /` '\
#       0     2
#
#
#           1
#            _
#           '\  B
#             \
#         0->----<- 2
#


def split_psi(Psi,
              dL,
              dR,
              truncation_par={
                  'chi_max': 32,
                  'p_trunc': 1e-6
              },
              verbose=0,
              max_iter=50,
              A=None):
    d, mL, mR = Psi.shape

    ###WARNING: not sure the errors reported here are right
    if np.any(A):
        chi = np.min([truncation_par['chi_max'], dR * mR, dL * mL])
        A, Psi_p = split_Psi_helper(Psi,
                                    dL,
                                    dR,
                                    chi,
                                    max_iter,
                                    verbose,
                                    A_hint=A,
                                    S2_weight=1.0)
        trunc_leg = np.linalg.norm(
            Psi - np.tensordot(A, Psi_p, axes=[[1, 2], [1, 2]]))**2
    else:
        chi = 1
        trunc_leg = 1
        while trunc_leg > 1e-16 and chi <= truncation_par[
                'chi_max'] and chi <= np.min([dR * mR, dL * mL]):
            A, Psi_p = split_Psi_helper(Psi,
                                        dL,
                                        dR,
                                        chi,
                                        max_iter,
                                        verbose=0,
                                        A_hint=A,
                                        S2_weight=1.0)
            trunc_leg = np.linalg.norm(
                Psi - np.tensordot(A, Psi_p, axes=[[1, 2], [1, 2]]))**2
            chi += 1

    Psi_p = np.transpose(Psi_p, [1, 0, 2, 3])  # dL mL dR mR
    Psi_p = np.reshape(Psi_p, (dL * mL, dR * mR))

    X, s2, Z, chi_c, trunc_bond = svd_theta_UsV(
        Psi_p, truncation_par['chi_max'], p_trunc=truncation_par['p_trunc'])
    S = np.reshape(X, (dL, mL, chi_c))
    S = np.tensordot(S, np.diag(s2), axes=(2, 0))

    B = np.reshape(Z, (chi_c, dR, mR))
    B = np.transpose(B, (1, 0, 2))

    info = {'error': trunc_leg + trunc_bond, 'd_error': trunc_leg}
    return A, S, B, info


def split_Psi_helper(Psi,
                     dL,
                     dR,
                     m,
                     N_split,
                     verbose=0,
                     S2_weight=0.,
                     A_hint=None):
    """ Solve Psi = A.Lambda where A is an isometry that "splits" a leg and Lambda is a 2-site 
	    wavefunction of maximum Schmidt rank 'm'
	
		Psi has a physical leg (d) and L/R auxilliaries (mL, mR)
		
		A:d ---> dL x dR is the isometric splitting; |dL x dR| < d (isometry)
		
	
		Psi:  d, mL, mR
		A: d, dL, dR
		Lambda: mL, dL, dR, mR
		
		return A, Lambda
	"""
    def find_A(Psi, A, Lambda, AdPsi=None):
        d, dL, dR = A.shape
        if AdPsi is None:
            dS = np.tensordot(Psi, Lambda.conj(), axes=[[1, 2], [0, 3]])
            X, y, Z = svd(dS.reshape([d, dL * dR]), full_matrices=False)
            A = np.dot(X, Z).reshape([d, dL, dR])
        else:
            dS = np.tensordot(AdPsi, Lambda.conj(), axes=[[2, 3], [0, 3]])

            if S2_weight != 0.:
                psi = AdPsi.transpose([2, 0, 1, 3])
                rhoL = np.tensordot(psi, psi.conj(), axes=[[2, 3], [2, 3]])
                dS2 = np.tensordot(rhoL, psi, axes=[[2, 3], [0, 1]])
                dS2 = np.tensordot(psi.conj(), dS2, axes=[[0, 3], [0, 3]])

                dS += dS2 * S2_weight

            X, y, Z = svd(dS.reshape([dL * dR, dL * dR]),
                          full_matrices=False)

            U = np.dot(X, Z).reshape([dL, dR, dL, dR])
            A = np.tensordot(A, U, axes=[[1, 2], [0, 1]])
        return A, y

    def find_Lambda(Psi, A, truncation_par):
        AdPsi = Lambda = np.tensordot(A.conj(), Psi, axes=[[0], [0]])
        dL, dR, mL, mR = Lambda.shape
        Lambda = Lambda.transpose([2, 0, 1, 3]).reshape([mL * dL, dR * mR])

        X, y, Z = svd(Lambda)

        chi = truncation_par['chi']
        norm = np.linalg.norm(y[:chi])

        Lambda = np.dot(X[:, :chi] * (y[:chi] / norm),
                        Z[:chi, :]).reshape([mL, dL, dR, mR])
        return Lambda, y, 1 - norm, AdPsi

    if A_hint is None:
        """Initial guess for A"""
        d, mL, mR = Psi.shape

        X, y, Z = svd(Psi.reshape([d, mL * mR]))

        if verbose == 2:
            norm = np.linalg.norm(y[:dL * dR])
            print("Best case error:", 1 - norm)
        A = X[:, :dL * dR].reshape((-1, dL, dR))
    else:
        A = A_hint

    truncation_par = {}
    truncation_par['chi'] = m
    error = []
    Lambda = 0

    ne_old = 1.
    delta_ne = 1.
    cnt = 0
    while delta_ne > 10e-12 and cnt < N_split:
        cnt = cnt + 1
        Lambdaold = Lambda
        Lambda, y, p_trunc, AdPsi = find_Lambda(Psi, A, truncation_par)
        Aold = A
        A, y = find_A(Psi, A, Lambda, AdPsi=AdPsi)
        ne = 0.5 * np.linalg.norm(
            np.tensordot(AdPsi - Lambda.transpose([1, 2, 0, 3]),
                         Lambda.conj(),
                         axes=[[2, 3], [0, 3]]))
        delta_ne = np.abs(ne - ne_old) / ne

        ne_old = ne
        if verbose == 2:
            print(i, end=' ')
            print(
                p_trunc, 1 - np.sum(y), 0.5 * np.linalg.norm(
                    np.tensordot(AdPsi - Lambda.transpose([1, 2, 0, 3]),
                                 Lambda.conj(),
                                 axes=[[2, 3], [0, 3]])),
                np.linalg.norm(Aold - A))
            error.append(p_trunc)

    Lambda, y, p_trunc, AdPsi = find_Lambda(Psi, A, truncation_par)
    if verbose == 2:
        plt.plot(error)
        plt.yscale('log')
        plt.show()
    chi = y.shape[0]
    return A, Lambda
