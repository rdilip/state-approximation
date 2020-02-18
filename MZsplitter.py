from __future__ import absolute_import
# from __future__ import division
from __future__ import print_function
import numpy as np
import scipy as sp
import scipy.linalg

try:
    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm
except:
    print("no plotting")

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
def split_psi(psi,
              dL,
              dR,
              truncation_par={
                  'chi_max': 256,
                  'p_trunc': 1e-6
              },
              eps=1e-8,
              max_iter=300,
              X=None,
              verbose=0,
              final_A_iter=True,
              constrain=False):
    """ Given a tripartite state psi.shape = d x mL x mR   , find an approximation


			psi ~= A.Lambda = A.Ad.psi

		where A.shape = dL x dR x d  is an isometry that "splits" d --> dL x dR and Lambda.shape = mL dL dR mR is a 2-site TEBD-style wavefunction of unit norm and maximum Schmidt rank 'chi_max.'

		The solution for Lambda is given by the MPS-type decomposition

			Lambda_{mL dL dR mR} = \sum_eta  S_{dL mL eta} B_{dR eta mR}   (*note* ordering convention in S, B: chosen to agree with TenPy-MPS)

		where 1 < eta <= chi_max, and Lambda has unit-norm

		Arguments:

			psi:  shape= d, mL, mR

			dL, dR: ints specifying splitting dimensions

			truncation_par  = {'chi_max':, 'p_trunc':}  truncation parameters for the SVD of Lambda; p_trunc is acceptable discarded weight

			eps: precision of iterative solver routine

			max_iter: max iterations of routine (through warning if this is reached!)

			constrain: if True, then the span of the isometry "A" will agree with the dominant Schmidt vectors of the index to be split.

			X: initial guess for A, provided as X[d, dl*dr]

			verbose:


		Returns:

			A: d x dL x dR
			S: dL x mL x eta
			B: dR x eta x mR

			info = {} , a dictionary of (optional) errors and entanglement info

				'error': |Psi - A.Lambda |^2,   where Lambda = S.B
				'num_iter': num iterations of solver

				's_AdPsi': Schmidt spectrum of (Ad.psi)_{dL mL , dR mR} BEFORE truncation to chi_max and WITHOUT normalizing
				's_Lambda': Schmidt spectrum of normalized Lambda = S.B


	"""

    ###
    #
    #   This disentangler works on steepest descent of 2nd-moment of reduced density matrix:
    #
    #           1: Lam = Ad Psi  (NOT normalized)
    #           2: Form rho_R from Lam (NOT normalized)
    #           3: F = Tr( rho_R^2)   < Tr( rho_R)

    d, mL, mR = psi.shape

    if dL * dR > d:
        raise ValueError

    #Initialize guess for A via reduced density matrix of psi + some noise
    if X is None:
        psinoise = np.random.random((d, mL * mR)) - 0.5
        psinoise *= (0.1 / np.linalg.norm(psinoise))
        psinoise += psi.reshape((d, mL * mR))
        if d < mL * mR:
            rho = np.dot(psinoise, psinoise.T.conj())  #TODO symm
            p, X = np.linalg.eigh(rho)
            perm = np.argsort(-p)
            X = X[:, perm]
            p = p[perm]
        else:
            X, Y, Z = svd(psinoise, full_matrices=True)
            p = Y**2

        opt = np.sqrt(np.sum(p[:dL * dR]))
    else:
        opt = np.nan

    if constrain:
        A = X[:, :dL * dR]
        psi = np.tensordot(A.conj(), psi, axes=[[0], [0]])
        A = np.eye(dL * dR, dtype=X.dtype)
    else:
        A = X[:, :dL * dR]

#     symm: Computes a matrix-matrix product where one input matrix is symmetric
#         C := alpha*A*B + beta*C
#         or
#         C := alpha*B*A + beta*C
#
#     syrk: performs one of the symmetric rank k operations
#         C := alpha*A*A**T + beta*C,
#         or
#         C := alpha*A**T*A + beta*C
#
#     gemm: general matrix-matrix product

    syrk = sp.linalg.get_blas_funcs('syrk', [A])
    symm = sp.linalg.get_blas_funcs('symm', [A])
    gemm = sp.linalg.get_blas_funcs('gemm', [A])

    if A.dtype == np.complex:
        syrk = sp.linalg.get_blas_funcs('herk', [A])
        symm = sp.linalg.get_blas_funcs('hemm', [A])

    # Operations are O( L^2 R), with L = dL mL, R = dR mR, so transpose to mimick favorable cse
    if dL * mL > dR * mR and False:
        trans = True
        dL, dR = dR, dL
        mL, mR = mR, mL
        psi = psi.transpose([0, 2, 1])
    else:
        trans = False
    #Ansatz for lambda: lam = Ad.psi
    #psi = np.asfortranarray(psi.reshape((d, -1)))
    psi = np.ascontiguousarray(psi.reshape((-1, mL * mR)))

    # Order is lam_{left, right} = lam_{dL mL, dR mR}
    #lam = gemm(1., A, psi, trans_a = 2)
    lam = np.dot(A.T.conjugate(), psi)
    #lamF = np.asfortranarray(lam.reshape((dL, dR, mL, mR)).transpose([0, 2, 1, 3]).reshape((dL*mL, dR*mR)))
    lam = np.ascontiguousarray(
        lam.reshape((dL, dR, mL, mR)).transpose([0, 2, 1, 3]).reshape(
            (dL * mL, dR * mR)))

    S1s = []  # Tr( rhoR )
    S2s = []  # Tr( rhoR^2 )
    dA = []  # |A_n - A_n+1|
    m = 0
    go = True

    # print "IN1", lam.reshape(-1)
    # Iterate via polar-decomposition of steepest descent on S2
    while (m < max_iter and go) or m < 20:
        # rhoL is the reduced density matrix with dR*mR traces out.
        # rhoL = lam * lam.H

        # rhoL = syrk(1., lam.T, trans=1)  #seems fastee for some reason but...
        # rhoL = syrk(1., lamF) # = dot(lam, lam.T), will be herk for complex
        rhoL = np.dot(lam, lam.T.conjugate())

        dS = rhoL

        # We want this: dS = np.dot(rhoL.T, lam.conjugate())
        # symm side=1, carry out alpha*B*A
        # (lam.T * dS).T = dS.T * lam
        # dS = symm(1., dS, lam.T, side=1).T
        # dS = symm(1., rhoL, lamF, side = 0) #will be hemm in complex
        dS = (lam.T.conjugate().dot(dS)).T

        dS = np.ascontiguousarray(
            dS.reshape((dL, mL, dR, mR)).transpose([0, 2, 1, 3]).reshape(
                (dL * dR, mL * mR)))  # dS = rhoL.lam --> dL x dR, mL x mR

        dS = np.dot(psi, dS.T)
        #dS = gemm(1., psi, dS, trans_b = 2) #psi.lam^H rhoL

        U, Y, V = svd(dS, full_matrices=False)

        A = np.dot(U, V)
        # A = np.dot(Z.T, X.T).T #do it in fortran form

        ### Statistics
        S1s.append(np.trace(rhoL))
        S2s.append(np.sum(Y))
        #This is to  close approximation Tr(rho_L^2) as A_n \sim A_{n+1}

        ### New Lambda from new A.T ###
        lam = np.dot(A.T.conjugate(), psi)
        #lam = gemm(1., A, psi, trans_a = 2)

        lam = np.ascontiguousarray(
            lam.reshape((dL, dR, mL, mR)).transpose([0, 2, 1, 3]).reshape(
                (dL * mL, dR * mR)))
        #lamF = np.asfortranarray(lam.reshape((dL, dR, mL, mR)).transpose([0, 2, 1, 3]).reshape((dL*mL, dR*mR)))

        m += 1

        if m > 1:
            go = S2s[-1] - S2s[-2] > eps

    if m == max_iter:
        print("Reached iter_max=", m, "S2s = ...", S2s[-5:])

    if final_A_iter:  #diagonalize BL and BR
        l = lam.reshape((dL, mL, dR, mR))
        rho = np.tensordot(l, l.conj(), axes=[[1, 2, 3], [1, 2, 3]])
        p, u = np.linalg.eigh(-rho)  #just to sort eigenvalues otherway
        l = np.tensordot(u.conj(), l, axes=[[0], [0]])
        rho = np.tensordot(l, l.conj(), axes=[[0, 1, 3], [0, 1, 3]])
        p, u = np.linalg.eigh(-rho)
        l = np.tensordot(u.conj(), l, axes=[[0], [2]]).transpose([1, 2, 0, 3])
        lam = l.reshape((dL * mL, dR * mR))

    #Now truncate ansatz for Lambda via SVD
    A = A.reshape((-1, dL, dR))

    if trans:
        dL, dR = dR, dL
        mL, mR = mR, mL
        A = A.transpose([0, 2, 1])
        lam = lam.T
        psi = psi.reshape((-1, mR, mL)).transpose([0, 2, 1]).reshape(
            (-1, mL * mR))

    U, s, V = svd(lam, compute_uv=True, full_matrices=False)

    cum = np.cumsum(s**2)

    # cum[-1] = sum(s**2) is the best we could hope for at this chiV, chiH, when eta = \infinity.
    # So there is no reason to capture the state to an error more accurate than  1 - cum[-1]
    #
    #

    #This is the error at eta = infinity, induced by finite dL*dR < d
    d_error = 2 - 2 * np.sqrt(cum[-1])
    # We select eta to induce an error of the same order
    target_error = np.max(
        [truncation_par['p_trunc'],
         0. * d_error])
    eta = np.min([
        np.count_nonzero((1 - cum) > 2 * target_error) + 1,
        truncation_par['chi_max']
    ])

    nrm = np.linalg.norm(s[:eta])
    S = U[:, :eta] * (s[:eta] / nrm)
    B = V[:eta, :]

    # Having truncated Lambda, we can optionally optimized 'A'
    if final_A_iter:
        lam_trunc = np.dot(S, B).reshape(
            (dL, mL, dR, mR)).transpose([0, 2, 1, 3]).reshape(
                (dL * dR, mL * mR))
        rho = np.tensordot(psi, lam_trunc.conj(), axes=[[1], [1]])
        U, l, V = svd(rho, full_matrices=False)
        A = np.dot(U, V).reshape((-1, dL, dR))
        nrm = np.sum(l)

    S = S.reshape((dL, mL, -1))
    B = B.reshape((-1, dR, mR)).transpose([1, 0, 2])

    if constrain:
        A = np.tensordot(X[:, :dL * dR], A, axes=[[1], [0]])

    if verbose:
        print("			Disentangle Psi Evaluations:", m, ". Error at eta-", eta,
              " :", 2 - 2 * nrm)
        if verbose > 1:
            print("			eps(eta)", (1 - np.cumsum(s**2))[:4])
        #print "		S2s:", np.array(S2s)[:3]
        #print "		S1s:", np.array(S1s)[:3]
        #print "		2nd-Renyis: ", (np.array(S2s)/np.array(S1s)**2)[:3]

    if verbose > 1:
        plt.subplot(1, 3, 1)
        plt.plot(np.array(S2s) / np.array(S1s)**2, '--')
        plt.plot(np.array(S2s), '-')
        plt.title('S2s')
        plt.legend(['S2', 'S2/S1^2'])
        plt.subplot(1, 3, 2)
        plt.plot(S1s, '-')
        plt.title('S1s')
        plt.subplot(1, 3, 3)
        plt.plot(1 - np.cumsum(s**2), '.-')
        plt.yscale('log')
        plt.ylim([1e-10, 1])
        plt.title(r'$\epsilon = 1 - \sum^\eta_i p_i$')
        plt.xlabel(r'$\eta$')
        plt.ylabel(r'$\epsilon$')
        plt.tight_layout()
        plt.show()

    info = {
        'num_iter': m,
        'opt_error': 2 - 2 * opt,
        'd_error': d_error,
        'error': 2 - 2 * nrm,
        's_AdPsi': s,
        's_Lambda': s[:eta] / nrm
    }

    return A, S, B, info


def autod_split_psi(psi, truncation_par, verbose=0):

    d, mL, mR = psi.shape

    chi_max = truncation_par['chi_max']
    chiH_max = chi_max['chiH_max']
    chiV_max = chi_max['chiV_max']
    eta_max = chi_max['etaV_max']
    chiT_max = chiH_max * chiV_max

    #Initialize guess for A via reduced density matrix of psi + some noise

    #psinoise = np.random.random((d, mL*mR))-0.5
    #psinoise *= (0.001/np.linalg.norm(psinoise))
    psinoise = psi.reshape((d, mL * mR))
    if d < mL * mR:
        rho = np.dot(psinoise, psinoise.T.conj())  #TODO symm
        p, X = np.linalg.eigh(rho)
        perm = np.argsort(-p)
        X = X[:, perm]
    else:
        X, Y, Z = svd(psinoise, full_matrices=True)

    S2s = {}
    S1s = {}
    etaFOMs = {}
    p_truncs = {}

    best_F = 0.

    max_ = np.min([chiT_max, d])
    for dR in range(1, 1 + max_):

        dL = max_ // dR

        a, S, B, info = split_psi(psi,
                                  dL,
                                  dR,
                                  truncation_par={
                                      'chi_max': eta_max,
                                      'p_trunc': truncation_par['p_trunc']
                                  },
                                  verbose=0,
                                  X=X)
        s = info['s_AdPsi']

        S1, S2, etaFOM = np.sum(s**2), np.sum(s**4), 1 - np.sum(s[:eta_max]**2)

        S1s[(dL, dR)] = S1
        S2s[(dL, dR)] = S2
        etaFOMs[(dL, dR)] = etaFOM

        #print dL, dR, S2, "|",
        if S2 > best_F * (1 - 0.00000001):
            aSB = a, S, B, info
            dLdR = dL, dR
            best_F = S2
    #print
    return dLdR[0], dLdR[1], aSB[0], aSB[1], aSB[2], aSB[3]


def split_landscape(psi, chiT_max, eta_max, verbose=0):
    """ Given the tri-partite wavefunction psi (as in split_psi), report statistics on the error landscape
		in the dL, dR plane
	"""

    dMax = np.min([psi.shape[0], chiT_max])
    p_truncs = {}
    S2s = np.zeros((dMax, dMax)) * np.nan
    S1s = np.zeros((dMax, dMax)) * np.nan
    etaFOM = np.zeros((dMax, dMax)) * np.nan
    Ss = np.zeros((dMax, dMax)) * np.nan
    for dL in range(1, dMax + 1):
        for dR in range(1, dMax + 1):
            if dL * dR > dMax:
                continue

            a, S, B, info = split_psi(psi,
                                      dL,
                                      dR, {
                                          'chi_max': eta_max,
                                          'p_trunc': 0.
                                      },
                                      verbose=0)
            if verbose:
                print(dL, dR, ' | ', end=' ')
            s = info['s_AdPsi']
            p_truncs[(dL, dR)] = 1 - np.cumsum(s**2)
            S1s[dL - 1, dR - 1] = np.sum(s**2)
            S2s[dL - 1, dR - 1] = np.sum(s**4)
            p = s**2 / np.sum(s**2)
            Ss[dL - 1, dR - 1] = -np.sum(p * np.log(p))
            #etaFOM[dL-1, dR-1] = 1 - np.sum(s[:eta_max]**2)
            etaFOM[dL - 1, dR - 1] = info['error']

    best = np.nanargmax(S2s)
    dL, dR = best / dMax + 1, best % dMax + 1

    if verbose:
        print("Best dL, dR = , S2 = ", dL, dR, -np.log(S2s[dL - 1, dR - 1]),
              -np.log(S2s[1, 1]))

    if verbose > 1:
        plt.subplot(1, 3, 1)
        vmin = 1e-9
        vmax = 1
        for i in range(dMax):
            for j in range(dMax):
                S1s[i, j] = np.min([1. - 1.1 * vmin, S1s[i, j]])
        plt.imshow(1 - S1s,
                   interpolation='nearest',
                   cmap=plt.cm.jet,
                   vmin=vmin,
                   vmax=vmax,
                   norm=LogNorm(vmin=vmin, vmax=vmax),
                   extent=[0.5, dMax + 0.5, dMax + 0.5, 0.5])
        plt.ylabel(r'$\chi_V$', fontsize=16)
        plt.xlabel(r'$\chi_H$', fontsize=16)
        plt.colorbar(orientation='horizontal')
        plt.title(r'$1 - S_1$')
        plt.subplot(1, 3, 2)
        vmin = np.nanmin(-np.log(S2s))
        vmax = np.nanmax(-np.log(S2s))
        plt.imshow(-np.log(S2s),
                   interpolation='nearest',
                   cmap=plt.cm.jet,
                   vmin=vmin,
                   vmax=vmax,
                   extent=[0.5, dMax + 0.5, dMax + 0.5, 0.5])
        plt.ylabel(r'$\chi_V$', fontsize=16)
        plt.xlabel(r'$\chi_H$', fontsize=16)
        plt.colorbar(orientation='horizontal')
        plt.title(r'$\log(S_2)$')
        plt.subplot(1, 3, 3)
        vmin = np.nanmin(etaFOM)
        vmax = np.nanmax(etaFOM)
        plt.imshow(etaFOM,
                   interpolation='nearest',
                   cmap=plt.cm.jet,
                   vmin=vmin,
                   vmax=vmax,
                   norm=LogNorm(vmin=vmin, vmax=vmax),
                   extent=[0.5, dMax + 0.5, dMax + 0.5, 0.5])
        plt.ylabel(r'$\chi_V$', fontsize=16)
        plt.xlabel(r'$\chi_H$', fontsize=16)
        try:
            plt.colorbar(orientation='horizontal')
        except:
            pass
        plt.title(r'$\epsilon(\eta), \eta = $' + str(eta_max))
        plt.show()
        """
		leg = []
		for k, e in sorted(p_truncs.iteritems()):
			leg.append(str(k) + " {:10.4f}".format(-np.log(S2s[k[0]-1, k[1]-1])) )
			plt.plot(e, '-')

		plt.legend(leg)
		plt.xscale('log')
		plt.yscale('log')
		plt.xlabel(r'$\eta$', fontsize = 16)
		plt.ylabel(r'$\epsilon$', fontsize = 16)
		plt.show()
		"""
    return dL, dR


def split_quad(psi,
               dL,
               dR,
               truncation_par={
                   'chi_max': 256,
                   'p_trunc': 1e-6
               },
               eps=1e-9,
               max_iter=300,
               X=None,
               verbose=0,
               constrain=True,
               dense_form=False,
               mode='S2'):
    """ Given a four-partite state psi.shape = d x mL x mC x mR   , find an approximation

			psi = A.Lambda

		where A.shape = dL x dR x d  is an isometry that "splits" d --> dL x dR and Lambda.shape = dL mL mC dR mR  is a wavefunction of unit norm.

		The solution for Lambda is given by the MPS-type decomposition

			Lambda_{mL mC dL dR mR} = \sum_eta  S_{dL mL mC eta} B_{dR eta mR}   (*note* ordering convention in S, B: chosen to agree with TenPy-MPS)

		where 1 < eta <= chi_max, and Lambda has unit-norm

		Arguments:

			psi:  shape= d, mL, mC, mR

			dL, dR: ints specifying splitting dimensions

			truncation_par  = {'chi_max':, 'p_trunc':}  truncation parameters for the SVD of Lambda; p_trunc is acceptable discarded weight

			eps: precision of iterative solver routine

			max_iter: max iterations of routine (through warning if this is reached!)

			dense_form = False / True :If dense form, rather than making Lambda = S.B, returns B = None and

				S = dL x dR x mL x mC x mR


		Returns:

			A: d x dL x dR
			S: dL x mL x mC x eta			(unless dense_form)
			B: dR x eta x mR



			info = {} , a dictionary of (optional) errors and entanglement info

				'error': |Psi - A.Lambda |^2,   where Lambda = S.B
				'num_iter': num iterations of solver

				's_AdPsi': Schmidt spectrum of (Ad.psi)_{dL mL , dR mR} BEFORE truncation to chi_max and WITHOUT normalizing
				's_Lambda': Schmidt spectrum of normalized Lambda = S.B


	"""

    ###
    #
    #	This disentangler works on steepest descent of 2nd-moment of reduced density matrix:
    #
    #		1: Lam = Ad Psi  (NOT normalized)
    #		2: Form rho_R from Lam (NOT normalized)
    #		3: F = Tr( rho_R^2)   < Tr( rho_R)

    d, mL, mC, mR = psi.shape

    if dL * dR > d:
        raise ValueError

    #Initialize guess for A via reduced density matrix of psi + some noise
    if X is None:
        #psinoise = np.random.random((d, mL*mC*mR))-0.5
        #psinoise *= (0.5/np.linalg.norm(psinoise))
        psinoise = psi.reshape((d, mL * mC * mR))
        if d < mL * mC * mR:
            rho = np.dot(psinoise, psinoise.T)  #TODO symm
            p, X = np.linalg.eigh(rho)
            perm = np.argsort(-p)
            X = X[:, perm]
            p = p[perm]
        else:
            X, Y, Z = svd(psinoise, full_matrices=True)
            p = Y**2

    opt = np.nan  #np.sqrt(np.sum(p[:dL*dR]))

    if mode == 'vN':
        constrain = True
        raise NotImplemented

    if constrain:
        A = X[:, :dL * dR]
        psi = np.tensordot(A.conj(), psi, axes=[[0], [0]])
        #A = np.eye(dL*dR, dtype = X.dtype) + (np.random.random((dL*dR, dL*dR)) - 0.5)*0.001
        A = X[:dL * dR, :dL * dR].conj().T
        U, s, V = svd(A)
        A = np.dot(U, V)
        #A, r = np.linalg.qr(A)

        #print np.dot(X[:, :dL*dR], A)
    else:
        #A = np.asfortranarray(X[:, :dL*dR])
        A = X[:, :dL * dR]

    if mode == 'vN':
        U, Psi = split_quad_vN()

    syrk = sp.linalg.get_blas_funcs('syrk', [A])
    symm = sp.linalg.get_blas_funcs('symm', [A])
    gemm = sp.linalg.get_blas_funcs('gemm', [A])

    #Ansatz for lambda: lam = Ad.psi
    #psi = np.asfortranarray(psi.reshape((d, -1)))
    psi = np.ascontiguousarray(psi.reshape((-1, mL * mC * mR)))

    #Order is lam_{left, right} = lam_{dL dR  mL mC mR}
    lam = np.dot(A.T.conj(), psi)

    #Order is lam_{left, C, right} = lam_{dLmL,  mC,  dR mR}
    lam = np.ascontiguousarray(
        lam.reshape((dL, dR, mL, mC, mR)).transpose([0, 2, 3, 1, 4]).reshape(
            (dL * mL, mC, dR * mR)))

    S1s = []  # Tr( rhoR )
    S2s = []  # Tr( rhoR^2 )
    dA = []  #|A_n - A_n+1|
    m = 0
    go = True

    #Iterate via polar-decomposition of steepest descent on S2L*S2R

    while m < max_iter and go:

        l = lam.reshape((dL * mL, mC * dR * mR))
        rhoL = np.dot(l, l.T)
        S1s.append(np.trace(rhoL))

        #p, x = np.linalg.eigh(rhoL)
        #p[p<1e-14] = 1e-14
        #p = p/np.sum(p)
        #lp = -np.nan_to_num(np.log(p))
        #lp = -p
        #dSL = np.dot(x*lp, x.T)
        #S2L = np.dot(p, p)
        #S2L = np.exp(-np.dot(lp, p))
        dSL = rhoL
        S2L = np.vdot(rhoL, rhoL)
        dSL = np.dot(dSL, l).reshape(
            (dL, mL, mC, dR, mR)).transpose([0, 3, 1, 2, 4]).reshape(
                (dL * dR, mL * mC * mR))

        l = lam.reshape((dL * mL * mC, dR * mR))
        rhoR = np.dot(l.T, l)
        #p, x = np.linalg.eigh(rhoR)
        #p[p<1e-14] = 1e-14
        #p = p/np.sum(p)
        #lp =  -np.nan_to_num(np.log(p))
        #lp = -p
        #dSR = np.dot(x*lp, x.T)
        #S2R = np.exp(-np.dot(lp, p))
        #S2R = np.dot(p, p)
        dSR = rhoR
        S2R = np.vdot(rhoR, rhoR)
        dSR = np.dot(dSR, l.T).reshape(
            (dR, mR, dL, mL, mC)).transpose([2, 0, 3, 4, 1]).reshape(
                (dL * dR, mL * mC * mR))

        #print "Ss", S2L, S2R, np.trace(rhoL), np.trace(rhoR)

        #S2s.append(S2R*S2L/S2C)

        #dS = S2R*np.ascontiguousarray(dSL.reshape((dL, mL, mC, dR, mR)).transpose([0, 3, 1, 2, 4]).reshape((dL*dR, mL*mC*mR)))
        #dS+= S2L*np.ascontiguousarray(dSR.reshape((dR, mR, dL, mL, mC)).transpose([2, 0, 3, 4, 1]).reshape((dL*dR, mL*mC*mR)))  # dS = rhoL.lam --> dL x dR, mL x mR

        #dS+=0.5*S2R*S2L/err*np.ascontiguousarray(l.T.reshape((dR, mR, dL, mL, mC)).transpose([2, 0, 3, 4, 1]).reshape((dL*dR, mL*mC*mR)))

        dS = (S2R) * dSL + (S2L) * dSR
        S2s.append(S2R * S2L)

        #dS = np.ascontiguousarray(dSR.reshape((dR, mR, dL, mL, mC)).transpose([2, 0, 3, 4, 1]).reshape((dL*dR, mL*mC*mR)))  # dS = rhoL.lam --> dL x dR, mL x mR
        #S2s.append(S2R)

        #dS = dSL + dSR - dSC
        #dS1 = np.dot(psi, dS.T)
        #U, Y, V = svd(dS1, full_matrices=False)
        #A1 = np.dot(U, V)

        #dS = (S2R/S2C)*dSL + (S2L/S2C)*dSR -0.*(S2R*S2L/(S2C**2))*dSC
        dS = np.dot(psi, dS.T)
        U, Y, V = svd(dS, full_matrices=False)
        A = np.dot(U, V)

        #print np.linalg.norm(A - A1), np.linalg.norm(dS - dS1), ":",
        #A = np.dot(Z.T, X.T).T #do it in fortran form

        ### Statistics

        #S2s.append(np.sum(Y)/2.) #This is to  close approximation Tr(rho_L^2) as A_n \sim A_{n+1}

        lam = np.dot(A.T, psi)
        lam = np.ascontiguousarray(
            lam.reshape((dL, dR, mL, mC, mR)).transpose([0, 2, 3, 1, 4]))

        #print "Y", Y
        m += 1

        if m > 1:
            go = np.abs((S2s[-1] - S2s[-2])) / S2s[-1] > eps

    if m == max_iter:
        print("Reached iter_max=", m, "S2s = ...", S2s[-5:])

    info = {'num_iter': m, 'opt_error': 2 - 2 * opt}

    #Now truncate ansatz for Lambda via SVD

    #rho = np.dot(psi, psi.T)		#TODO symm
    #p, X = np.linalg.eigh(rho)
    #perm = np.argsort(-p)
    #X = X[:, perm]

    #A = X[:, :dL*dR]
    #perm = np.random.permutation(dL*dR)
    #A = A[:, perm]
    #X = np.random.random((d, dL*dR)) - 0.5
    #A, r = np.linalg.qr(X)

    #lam = np.dot(A.T, psi)
    #lam = np.ascontiguousarray(lam.reshape((dL, dR, mL, mC, mR)).transpose([0, 2, 3, 1, 4]).reshape((dL, mL, mC, dR, mR)))
    A = A.reshape((-1, dL, dR))

    if dense_form:

        S = lam.transpose([0, 3, 1, 2, 4])
        B = None
        nrm = np.linalg.norm(S)
        S = S / nrm
        lam = lam.reshape((dL * mL * mC, dR * mR))
        s = svd(lam, compute_uv=False)
        cum = np.cumsum(s**2)

        nrm = np.sqrt(cum[-1])
        d_error = 2 - 2 * nrm

    else:

        lam = lam.reshape((dL * mL * mC, dR * mR))
        U, s, V = svd(lam, compute_uv=True, full_matrices=False)

        cum = np.cumsum(s**2)

        # cum[-1] = sum(s**2) is the best we could hope for at this chiV, chiH, when eta = \infinity.
        # So there is no reason to capture the state to an error more accurate than  1 - cum[-1]
        #

        d_error = 2 - 2 * np.sqrt(
            cum[-1]
        )  #This is the error at eta = infinity, induced by finite dL*dR < d
        target_error = np.max([
            truncation_par['p_trunc'], 0.5 * d_error
        ])  # We select eta to induce an error of the same order
        eta = np.min([
            np.count_nonzero((1 - cum) > 2 * target_error) + 1,
            truncation_par['chi_max']
        ])

        nrm = np.linalg.norm(s[:eta])
        S = U[:, :eta] * (s[:eta] / nrm)
        B = V[:eta, :]

        #Having truncated Lambda, we can optionally optimized 'A'

        S = S.reshape((dL, mL * mC, -1))
        B = B.reshape((-1, dR, mR)).transpose([1, 0, 2])
        info['s_Lambda'] = s[:eta] / nrm

    info['s_AdPsi'] = s
    info['d_error'] = d_error
    info['error'] = 2 - 2 * nrm
    info['S2LR'] = S2s[-1]
    info['S1'] = S1s[-1]

    if constrain:
        A = np.tensordot(X[:, :dL * dR], A, axes=[[1], [0]])

    if verbose:
        print("			Disentangle Psi Evaluations:", m, ". Error at eta-", eta,
              " :", 2 - 2 * nrm)
        if verbose > 1:
            print("			eps(eta)", (1 - np.cumsum(s**2))[:4])
        #print "		S2s:", np.array(S2s)[:3]
        #print "		S1s:", np.array(S1s)[:3]
        #print "		2nd-Renyis: ", (np.array(S2s)/np.array(S1s)**2)[:3]

    if verbose > 1:
        plt.subplot(1, 3, 1)
        plt.plot(np.array(S2s) / np.array(S1s)**2, '--')
        plt.plot(np.array(S2s), '-')
        plt.title('S2s')
        plt.legend(['S2', 'S2/S1^2'])
        plt.subplot(1, 3, 2)
        plt.plot(S1s, '-')
        plt.title('S1s')
        plt.subplot(1, 3, 3)
        plt.plot(1 - np.cumsum(s**2), '.-')
        plt.yscale('log')
        plt.ylim([1e-10, 1])
        plt.title(r'$\epsilon = 1 - \sum^\eta_i p_i$')
        plt.xlabel(r'$\eta$')
        plt.ylabel(r'$\epsilon$')
        plt.tight_layout()
        plt.show()

    return A, S, B, info


def autod_split_quad(psi, truncation_par, verbose=0, dense_form=False):

    d, mL, mC, mR = psi.shape
    chi_max = truncation_par['chi_max']
    chiH_max = chi_max['chiH_max']
    chiV_max = chi_max['chiV_max']
    eta_max = chi_max['etaV_max']
    chiT_max = chiH_max * chiV_max

    if dense_form:
        eta0_max = -1
    S2s = {}
    S1s = {}
    etaFOMs = {}
    p_truncs = {}

    best_F = -0.01

    max = np.min([chiT_max, d])
    p_trunc = truncation_par['p_trunc']

    for dR in range(1, np.min([max, int(np.ceil(np.sqrt(max)))]) + 1):
        dL = max / dR
        #if dR > mC*mR or dL > mL*mC:
        #	continue

        #print "Trying:", dL, dR
        a, S, B, info = split_quad(psi,
                                   dL,
                                   dR,
                                   truncation_par={
                                       'chi_max': eta0_max,
                                       'p_trunc': truncation_par['p_trunc']
                                   },
                                   verbose=0,
                                   dense_form=dense_form)
        s = info['s_AdPsi']

        S1, S2, etaFOM = info['S1'], info['S2LR'], 1 - np.sum(s[:eta_max]**2)

        S1s[(dL, dR)] = S1
        S2s[(dL, dR)] = S2
        etaFOMs[(dL, dR)] = etaFOM

        #print dL, dR, S2, "|",
        if S2 / (1 - S1 + p_trunc) > best_F * (1 - 0.00000001):
            aSB = a, S, B, info
            dLdR = dL, dR
            best_F = S2 / (1 - S1 + p_trunc)

    for dL in range(1, np.min([max, int(np.floor(np.sqrt(max)))]) + 1):
        dR = max / dL
        #if dR > mC*mR or dL > mL*mC:
        #	continue
        if (dL, dR) in S2s:
            continue
        #print "Trying:", dL, dR
        a, S, B, info = split_quad(psi,
                                   dL,
                                   dR,
                                   truncation_par={
                                       'chi_max': eta_max,
                                       'p_trunc': truncation_par['p_trunc']
                                   },
                                   verbose=0,
                                   dense_form=dense_form)
        s = info['s_AdPsi']

        S1, S2, etaFOM = info['S1'], info['S2LR'], 1 - np.sum(s[:eta_max]**2)

        S1s[(dL, dR)] = S1
        S2s[(dL, dR)] = S2
        etaFOMs[(dL, dR)] = etaFOM

        #print dL, dR, S2, "|",
        if S2 / (1 - S1 + p_trunc) > best_F * (1 - 0.00000001):
            aSB = a, S, B, info
            dLdR = dL, dR
            best_F = S2 / (1 - S1 + p_trunc)

    #print "S2s", S2s
    #print "S1s", S1s
    return dLdR[0], dLdR[1], aSB[0], aSB[1], aSB[2], aSB[3]
