from __future__ import absolute_import
#from __future__ import division
from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy import linalg
from misc import svd, svd_theta_UsV
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
              eps=1e-12):
    """ Given a tripartite state psi.shape = d x mL x mR   , find an approximation

			psi = A.Lambda

		where A.shape = dL x dR x d  is an isometry that "splits" d --> dL x dR and Lambda.shape = mL dL dR mR
		is a 2-site TEBD-style wavefunction of unit norm and maximum Schmidt rank 'chi_max.'

		The solution for Lambda is given by the MPS-type decomposition

			Lambda_{mL dL dR mR} = \sum_eta  S_{dL mL eta} B_{dR eta mR}

		where 1 < eta <= chi_max, and Lambda has unit-norm

		Arguments:

			psi:  shape= d, mL, mR

			dL, dR: ints specifying splitting dimensions (dL,dR maybe reduced to smaller values)

			truncation_par  = {'chi_max':, 'p_trunc':}  truncation parameters for the SVD of Lambda; p_trunc is acceptable discarded weight

			eps: precision of iterative solver routine

			max_iter: max iterations of routine (through warning if this is reached!)

			verbose:


		Returns:

			A: d x dL x dR
			S: dL x mL x eta
			B: dR x eta x mR

			info = {} , a dictionary of (optional) errors

				'error': 'trunc_leg','trunc_bond'
	"""

    d, mL, mR = Psi.shape

    A, Psi_p, dL, dR, trunc_leg = disentangle_split_Psi_helper(Psi,
                                                               dL,
                                                               dR,
                                                               max_iter,
                                                               verbose=verbose,
                                                               eps=eps)
    Psin = np.tensordot(A, Psi_p, axes=[[1, 2], [1, 2]])
    #print("a |dPsi|^2",  np.linalg.norm(Psin - Psi)**2, trunc_leg)

    Psi_p = np.transpose(Psi_p, [1, 0, 2, 3])
    Psi_p = np.reshape(Psi_p, (dL * mL, dR * mR))

    X, s2, Z, chi_c, trunc_bond = svd_theta_UsV(Psi_p,
                                                truncation_par['chi_max'],
                                                p_trunc=3e-16)

    S = np.reshape(X, (dL, mL, chi_c))
    #S = np.tensordot(S,np.diag(s2),axes=(2,0))
    S = S * s2

    B = np.reshape(Z, (chi_c, dR, mR))
    B = np.transpose(B, (1, 0, 2))

    #Psin = np.tensordot(A,
    #                    np.tensordot(S, B, axes=[[2], [1]]),
    #                    axes=[[1, 2], [0, 2]])
    #print("|dPsi|^2",  np.linalg.norm(Psin - Psi)**2, trunc_leg + trunc_bond)

    #TODO: technically theses errors are only good to lowest order I think
    info = {
        'error': trunc_leg + trunc_bond,
        'd_error': trunc_leg,
        's_Lambda': s2
    }
    return A, S, B, info


def disentangle_split_Psi_helper(Psi, dL, dR, max_iter, verbose=0, eps=1e-12):
    """ Solve Psi = A.Lambda where A is an isometry that "splits" a leg using disentangle_Psi

		Psi has a physical leg (d) and L/R auxilliaries (mL, mR)

		A:d ---> dL x dR is the isometric splitting; |dL x dR| < d (isometry)

		Psi:  d, mL, mR
		A: d, dL, dR
		Lambda: mL, dL, dR, mR

		return A, Lambda,trunc_leg
	"""

    # Get the isometry
    d, mL, mR = Psi.shape

    theta = np.reshape(Psi, (d, mL * mR))

    dL = np.min([dL, mL])
    dR = np.min([dR, mR])

    if dL * dR > d:
        dR = min([int(np.rint(np.sqrt(d))), dR])
        dL = min([d // dR, dL])

    X, y, Z, D2, trunc_leg = svd_theta_UsV(theta, dL * dR, p_trunc=1e-14)

    if D2 < dL * dR:
        dL = min([int(np.sqrt(D2)),
                  dL])  #NOTE this may completely ignore scheduling hints
        dR = min([D2 // dL, dR])

    #print "Leg-SVD:", D2, trunc_leg, np.linalg.svd(theta, compute_uv = False)
    D2 = dR * dL
    X = X[:, :D2]
    nrm = np.linalg.norm(y)
    nrm_t = np.linalg.norm(y[:D2])
    y = y[:D2] / nrm_t * nrm
    Z = Z[:D2, :]
    # We assum theta is normed from the beginning, such that the norm**2 of y
    # before normalization is (1 - trunc_leg)
    # [TODO] remove nrm, since nrm should be 1 from svd_theta_UsV.
    trunc_leg2 = (nrm**2 - nrm_t**2) * (1 - trunc_leg) / nrm**2

    A = X
    theta = (Z.T * y).T

    # Disentangle the two-site wavefunction

    theta = np.reshape(theta, (dL, dR, mL, mR))

    theta = np.transpose(theta, (2, 0, 1, 3))
    thetap, U, S = disentangle_Psi(theta,
                                   eps=eps,
                                   max_iter=max_iter,
                                   verbose=verbose)
    #print "Dis. Ss:", S
    #print "dL, dR:", dL, dR
    A = np.tensordot(A,
                     np.reshape(np.conj(U), (dL, dR, dL * dR)),
                     axes=([1], [2]))

    return A, thetap, dL, dR, trunc_leg + trunc_leg2


def U2(psi):
    """Entanglement minimization via 2nd Renyi entropy

		Returns S2 and 2-site U
	"""
    chi = psi.shape
    rhoL = np.tensordot(psi, np.conj(psi), axes=[[2, 3], [2, 3]])
    dS = np.tensordot(rhoL, psi, axes=[[2, 3], [0, 1]])
    dS = np.tensordot(np.conj(psi), dS, axes=[[0, 3], [0, 3]])
    dS = dS.reshape((chi[1] * chi[2], -1))
    s2 = np.trace(dS)

    X, Y, Z = svd(dS)

    return -np.log(s2), (np.dot(X, Z).T).conj()


def disentangle_Psi(psi, eps=1e-12, max_iter=300, verbose=0,
                    init_from='polar'):
    """ Disentangles a 2-site TEBD-style wavefunction.

		psi = mL, dL, dR, mR		; mL, mR are like "auxilliary" and dL, dR the "physical"

		Find the 2-site unitary U,

			psi --> U.psi = psi', U = dL, dR, dL, dR

		which minimizes the nth L/R Renyi entropy.

		Returns psi', U, Ss, where Ss are entropies during iterations

	"""

    Ss = []
    chi = psi.shape
    U = np.eye(chi[1] * chi[2], dtype=psi.dtype)

    #print "Entering Dis, shape = ", psi.shape
    if init_from == 'polar':
        u = np.eye(chi[1] * chi[2], dtype=psi.dtype) + (
            0.5 - np.random.rand(chi[1] * chi[2], chi[1] * chi[2]))
        x = np.min([chi[1] * chi[2], chi[0] * chi[3]])
        u[:x, :] = psi.copy().transpose([0, 3, 1,
                                         2]).reshape(chi[0] * chi[3],
                                                     chi[1] * chi[2])[:x, :]
        #u = u + 0.*(np.random.random(u.shape) - 0.5)
        X, l, Y = svd(u)
        u = np.dot(X, Y)

        U = np.dot(u, U)
        u = u.reshape((chi[1], chi[2], chi[1], chi[2]))
        psi = np.tensordot(u, psi, axes=[[2, 3], [1,
                                                  2]]).transpose([2, 0, 1, 3])

    m = 0
    go = True
    while m < max_iter and go:
        s, u = U2(psi)
        U = np.dot(u, U)
        u = u.reshape((chi[1], chi[2], chi[1], chi[2]))
        psi = np.tensordot(u, psi, axes=[[2, 3], [1,
                                                  2]]).transpose([2, 0, 1, 3])
        Ss.append(s)
        if m > 10:
            go = Ss[-2] - Ss[-1] > eps
        m += 1
    return psi, U.reshape([chi[1], chi[2], chi[1], chi[2]]), Ss
