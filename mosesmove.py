from __future__ import absolute_import
# from __future__ import division
from __future__ import print_function
import numpy as np

from misc import svd, group_legs, ungroup_legs, mpo_on_mpo, mps_group_legs, mps_overlap

try:
    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm
except:
    print("no plotting")

#LOAD YOUR SPLITTER HERE  - see mine for conventions
from MZsplitter import split_psi, split_quad, split_landscape, autod_split_psi, autod_split_quad

### Convention for MPS to be split: Psi = [B0, B1, ...]
#      3
#      v
#      |
#  0->- -<-1
#      |
#      v
#      2
#
# 0, 1 are physical
# 2, 3 are virtual
# arrows denote the canonical form assumed
#
# Here B0 is the "bottom most" tensor (the first to be split), so is a wavefunction (all-in arrows)
# The test data is in this form.

### Pent is the 5-leg wavefunction
#      4
#	   |
#  2--- ---3
#     / \
#    0   1

### Tri is the 3-leg wavefunction formed by grouping [0 2], [4], [1 3] of Pent
#     1
#     |
# 0=== ===2
#
# 0 is the "special" leg to be split

###  Limits on split bond dimensions:
#
#          |
#        S |
#         / \
#  chi_V /   \  eta
#       /     \
#   ===/--->---====
#      a chi_H B

###  Leg ordering and arrows of returned A, S, B
#          1
#          |
#        S |
#         / \
#        /   \
#       /     \
# 0 ---/-----------2
#      A       B
#
#
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


def moses_move(
        Psi,
        truncation_par={
            'bond_dimensions': {
                'etaH_max': 4,
                'etaV_max': 4,
                'chiV_max': 4,
                'chiH_max': 4
            },
            'p_trunc': 1e-6
        },
        save_info=None,
        test_splitscape_at=[],
        transpose=False,
        scheduleV=None,
        scheduleH=None,
        verbose=0,
):
    """ Splits a 2-sided MPS Psi = [b0, b1, ...] according to
		
			Psi = A Lambda
			
		B0 is the "bottom" of Psi, and Psi MUST be in B-form (arrows pointing downward). Returns
		
			A = [a0, . . .
			Lambda = [l0, . . .
			
		which are in A-form (arrows pointing UPWARDS)
		
		Options:
		
		- truncation_par: etaH/V are for the Lambda, chiH/V for A, and p_trunc is tolerance
		
		- scheduleH, scheduleV: optional list of integers specifying horizontal / vertical bond dimensions. If shorter than length of state, cycles through periodically.
		
		- save_info != None, saves progress data to a .pdf (save_info in pre-pended to the name)
		
		- test_splitscape_at = [j1, j2, ...] is a list of iterations at which to do an analysis of errors in the chiV, chiH plane.
		
		- transpose = True, solves
			
				Psi = Lambda B
				
			instead (returned as B, Lambda).
		
		
		Returns:
		
			A, Lambda, info
		
	"""

    if transpose:
        Psi = transpose_mpo(Psi)

    if scheduleH is None and scheduleV is None:
        has_schedule = False
    else:
        has_schedule = True

    bond_dimensions = truncation_par['bond_dimensions']

    if 'eta_max' in bond_dimensions:
        etaV_max = etaH_max = bond_dimensions['eta_max']
    else:
        etaV_max = bond_dimensions['etaV_max']
        etaH_max = bond_dimensions['etaH_max']

    chiV_max = bond_dimensions['chiV_max']
    chiH_max = bond_dimensions['chiH_max']
    chiT_max = chiV_max * chiH_max

    L = len(Psi)

    FindchiV = 0

    Lambda = []
    A = []

    #Current dimensions of the Pent tensor about to be split
    eta = 1
    chiV = 1
    pL = Psi[0].shape[0]
    pR = Psi[0].shape[1]
    chi = Psi[0].shape[3]

    #Initialize Pent from bottom of MPS
    Pent = Psi[0].reshape((chiV, eta, pL, pR, chi))

    errors = []
    d_errors = []
    dS = []
    Ih = []
    for j in range(L):
        Tri = Pent.transpose([0, 2, 4, 1, 3])
        #Tri = special, vertical, horizontal
        Tri = Tri.reshape((chiV * pL, chi, eta * pR))

        d = Tri.shape[0]

        #Entanglement information on tripartite-wavefunction
        S, I = ent_info(Tri)
        Ih.append(np.exp(I[1] / 2))
        Sold = S[1]

        if verbose > 1:
            print()
            print("iter ", j, "| d0, dV, dH:", Tri.shape[0], Tri.shape[1],
                  Tri.shape[2])
            print("		exp[ S0, Iv/2, Ih/2 ]:", np.exp(S[0]), np.exp(I[2] / 2),
                  np.exp(I[1] / 2), np.exp(I[2] / 2 - I[1] / 2.))
            print("		old Sv:", S[1])

        r = np.exp(
            I[2] / 2 - I[1] / 2.
        )  #Crude estimate of ratio of vertical / hortizontal bond dimension: the mutual info

        if j in test_splitscape_at:
            split_landscape(Tri, chiT_max, etaV_max, verbose=2)

        if j < L - 1:

            if j < FindchiV and has_schedule is False:  #choose based on optimimum found in split_landscape
                dL, dR, a, S, B, info = autod_split_psi(Tri,
                                                        truncation_par,
                                                        verbose=0)

            else:

                if scheduleH is not None:
                    dR = np.min([d, scheduleH[j % len(scheduleH)]])
                    dL = np.min([d / dR, chiV_max])
                    if scheduleV is not None:
                        dL = np.min([dL, scheduleV[j % len(scheduleV)]])

                elif scheduleV is not None:
                    dL = np.min([d, scheduleV[j % len(scheduleV)]])
                    dR = np.min([d / dL, chiH_max])

                elif chiT_max < d:
                    if r > 1:
                        dR = int(np.round(np.sqrt(1. * chiT_max / r)))
                        dL = chiT_max / dR
                    if r <= 1.:
                        dL = int(np.round(np.sqrt(1. * r * chiT_max)))
                        dR = chiT_max / dL
                else:
                    reff = []
                    dRs = []
                    for dR in range(1, d + 1):
                        if np.mod(d, dR) == 0:
                            reff.append(1. * d / dR**2)
                            dRs.append(dR)
                    #print dRs, reff, np.abs(np.log(np.array(reff/r)))
                    best = np.argmin(np.abs(np.log(np.array(reff / r))))
                    dR = dRs[best]
                    dL = d / dRs[best]

                a, S, B, info = split_psi(Tri,
                                          dL,
                                          dR,
                                          truncation_par={
                                              'chi_max': etaV_max,
                                              'p_trunc':
                                              truncation_par['p_trunc']
                                          },
                                          verbose=0)

            B = B.reshape(
                (dR, B.shape[1], eta,
                 pR)).transpose([0, 3, 2,
                                 1])  #Put into 2-side wavefunction ordering
            a = a.reshape((chiV, pL, dL, dR)).transpose([1, 3, 0, 2])

        else:
            dR = np.min([etaH_max, Tri.shape[0]])
            dL = 1
            U, s, V = svd(Tri.reshape((Tri.shape[0], -1)), compute_uv=True)

            target_p_trunc = np.max(
                [np.mean(errors), truncation_par['p_trunc']])
            cum = np.cumsum(s**2)
            dR = np.min([
                np.count_nonzero((1 - cum / cum[-1]) > target_p_trunc) + 1, dR
            ])
            nrm = np.linalg.norm(s[:dR])
            a = U[:, :dR].reshape((chiV, pL, 1, dR)).transpose([1, 3, 0, 2])
            B = ((V[:dR, :].T * s[:dR] / nrm).T).reshape(
                (dR, 1, eta, pR)).transpose([0, 3, 2, 1])
            info['error'] = 2 - 2 * nrm
            info['d_error'] = 2 - 2 * nrm
            info['s_AdPsi'] = np.array([1.])

        ### The splitter may or may not return this stuff
        if verbose > 1:
            print("		chiV, chiH:", dL, dR, 1. * dL / dR, r)

        try:
            if verbose > 1:
                print("		Error:", info['error'])
            errors.append(info['error'])
            d_errors.append(info['d_error'])
        except KeyError:
            pass
        try:
            s = info['s_AdPsi']
            p = s**2
            p = p[p > 1e-16]
            p = p / np.sum(p)
            Sv = -np.sum(p * np.log(p))
            if verbose > 1:
                print("		new Sv:", Sv)
            dS.append(Sold - Sv)
        except KeyError:
            pass

        Lambda.append(B)
        A.append(a)

        if j < L - 1:
            pL = Psi[j + 1].shape[0]
            pR = Psi[j + 1].shape[1]
            chi = Psi[j + 1].shape[3]
            #Obtain Pent by merging S into the next B of MPS
            Pent = np.tensordot(S, Psi[j + 1], axes=[[1], [2]])

        chiV = dL
        eta = B.shape[-1]

    if save_info is not None:
        plt.subplot(1, 3, 1)
        plt.plot(errors, '.')
        plt.yscale('log')
        plt.ylim([1e-8, 0.1])
        plt.title(r'$\epsilon$')
        plt.subplot(1, 3, 2)
        plt.plot(dS, '.')
        plt.title(r'$\Delta S_\lambda$')
        plt.ylim([-0.1, .6])
        plt.subplot(1, 3, 3)
        plt.plot(Ih, '.')
        plt.ylim([1., 2.5])
        plt.title(r'$I_H$')

        plt.savefig(save_info + '_' + str(etaV_max) + '_' + str(chiV_max) +
                    '_' + str(chiH_max) + '.pdf')

    if verbose > 1:
        print("Total Error / L ", np.sum(errors) / L)

    info = {
        'total_error': np.sum(errors),
        'errors': errors,
        'total_d_error': np.sum(d_errors),
        'd_errors': d_errors,
        'dS': dS
    }

    if transpose:
        A = transpose_mpo(A)
        Lambda = transpose_mpo(Lambda)

    return A, Lambda, info


def ent_info(Psi):
    """Given tripartite state Psi = d, mL, mR, returns
	
		S = [S0, S1, S2]:  entanglement entropy of leg
		I = [I12, I20, I10] : mutual informations between legs
	"""

    #d, mL, mR = Psi.shape

    S = []
    for j in range(3):
        s = svd(Psi.reshape((Psi.shape[0], -1)), compute_uv=False)
        s = s[s > 1e-16]
        S.append(-2 * np.inner(s**2, np.log(s)))
        Psi = np.transpose(Psi, [1, 2, 0])

    return S, [S[j - 1] + S[(j + 1) % 3] - S[j] for j in range(3)]


#            EnvL / EnvR
#                     1
#                    /
#                   /
#          0 ------[
#                   \
#                    \
#                     2


def sweeped_moses_move(Psi,
                       truncation_par={
                           'chi_max': {
                               'eta_max': 32,
                               'chiV_max': 4,
                               'chiH_max': 4
                           },
                           'p_trunc': 1e-6
                       },
                       save_info='',
                       test_splitscape_at=[],
                       schedule=None,
                       verbose=0):

    chi_max = truncation_par['chi_max']
    if 'eta_max' in chi_max:
        etaV_max = etaV_max = chi_max['eta_max']
    else:
        etaV_max = chi_max['etaV_max']
        etaH_max = chi_max['etaH_max']
    chiV_max = chi_max['chiV_max']
    chiH_max = chi_max['chiH_max']
    chiT_max = chiV_max * chiH_max

    L = len(Psi)

    FindchiV = L

    #Cache the left and right canonical forms of Psi. Psi[] will hold the orthgonal-center wavefunctions

    PsiA = [None] * L  #left-canonical tensors
    PsiB = [None] + Psi[1:]  #right-canonical tensors
    PsiS = [Psi[0]] + [None] * (L - 1)
    for j in range(L - 1):
        psi, pipe = group_legs(PsiS[j], ([0, 1, 2], [3]))
        q, r = np.linalg.qr(psi)
        PsiA[j] = ungroup_legs(q, pipe)
        PsiS[j + 1] = np.tensordot(r, PsiB[j + 1],
                                   axes=[[1], [2]]).transpose([1, 2, 0, 3])

    EL = [None] * L
    ER = [None] * L
    A = [None] * L
    Lambda = [None] * L

    #Initial guess is A = Id, Lam = Psi
    #for j in range(L):
    #	Lambda[j] = Psi[j].copy()
    #	A[j] = np.ones(1).reshape((1, 1, 1, 1))

    EL[-1] = np.ones(1).reshape((1, 1, 1))
    for j in range(L):
        ER[j] = np.eye(Psi[j].shape[3]).reshape(
            (Psi[j].shape[3], 1, Psi[j].shape[3]))

    d_errors = []
    errors = []
    T_errors = []

    def step(j, Quad, dir='R', fixd=False):

        #Form the 1-site wavefunction in ansatz basis by attaching environments to Psi
        if dir == 'L':
            Quad = Quad.transpose([4, 5, 2, 3, 0, 1])

        chiV = Quad.shape[0]
        eta = Quad.shape[1]
        pL = Quad.shape[2]
        pR = Quad.shape[3]
        chiVT = Quad.shape[4]
        etaT = Quad.shape[5]
        #print "cV, eta", chiV, eta, chiVT, etaT

        Quad, pipe = group_legs(
            Quad, [[0, 2], [4], [5], [1, 3]
                   ])  #Eventually, we will view this as 4-partite wavefunction
        d = Quad.shape[0]
        T_errors.append(2 - 2 * np.linalg.norm(Quad))
        Quad = Quad / np.linalg.norm(Quad)

        Tri, pipe = group_legs(Quad, [[0], [1, 2], [3]])

        #Entanglement information on tripartite-wavefunction
        S, I = ent_info(Tri)
        r = np.exp(I[2] / 2 - I[1] / 2.)
        r = r
        if False:
            print()
            print("iter ", j, "| d0, dV, dH:", Tri.shape[0], Tri.shape[1],
                  Tri.shape[2])
            print("		exp[ S0, Iv/2, Ih/2 ], r:", np.exp(S[0]),
                  np.exp(I[2] / 2), np.exp(I[1] / 2), r)
            print("		old Sv:", S[1])

        #print "dL, dR", dL, dR

        if j < L - 1 or dir == 'L':

            if Quad.shape[1] == 1:  #STUPID KLDUGEEEEEEE
                quad = Quad.transpose([0, 2, 1, 3])
            else:
                quad = Quad

            do_tri = (quad.shape[2] == 1)  #D = 1 so tri_splitter is faster

            if j < FindchiV and schedule is None:  #choose based on optimimum found in split_landscape
                if do_tri:
                    dL, dR, a, S, B, info = autod_split_psi(Tri,
                                                            truncation_par,
                                                            verbose=0)
                    S = S.reshape((S.shape[0], S.shape[1], 1, S.shape[2]))
                else:
                    dL, dR, a, S, B, info = autod_split_quad(quad,
                                                             truncation_par,
                                                             verbose=0)
            else:
                if schedule is not None:
                    dL = np.min([d, schedule[j % len(schedule)]])
                    dR = d / dL
                elif chiT_max < d:
                    if r > 1:
                        dR = int(np.round(np.sqrt(1. * chiT_max / r)))
                        dL = chiT_max / dR
                    if r <= 1.:
                        dL = int(np.round(np.sqrt(1. * r * chiT_max)))
                        dR = chiT_max / dL
                else:
                    reff = []
                    dRs = []
                    for dR in range(1, d + 1):
                        if np.mod(d, dR) == 0:
                            reff.append(1. * d / dR**2)
                            dRs.append(dR)
                    #print dRs, reff, np.abs(np.log(np.array(reff/r)))
                    best = np.argmin(np.abs(np.log(np.array(reff / r))))
                    dR = dRs[best]
                    dL = d / dRs[best]

                if do_tri:
                    a, S, B, info = split_psi(Tri,
                                              dL,
                                              dR,
                                              truncation_par={
                                                  'chi_max':
                                                  etaV_max,
                                                  'p_trunc':
                                                  truncation_par['p_trunc']
                                              },
                                              verbose=0)
                    S = S.reshape((S.shape[0], S.shape[1], 1, S.shape[2]))
                else:
                    a, S, B, info = split_quad(quad,
                                               dL,
                                               dR,
                                               truncation_par={
                                                   'chi_max':
                                                   etaV_max,
                                                   'p_trunc':
                                                   truncation_par['p_trunc']
                                               },
                                               verbose=0)

            d_errors.append(info['d_error'])
            errors.append(info['error'])
            B = B.reshape(
                (dR, B.shape[1], eta,
                 pR)).transpose([0, 3, 2,
                                 1])  #Put into 2-side wavefunction ordering
            a = a.reshape((chiV, pL, dL, dR)).transpose([1, 3, 0, 2])

            l = np.tensordot(a, B, axes=[[1], [0]])

            if dir == 'L':
                a = a.transpose([0, 1, 3, 2])
                B = B.transpose([0, 1, 3, 2])
                l = np.tensordot(ER[j % L], l.conj(), axes=[[1, 2], [1, 4]])
                l = np.tensordot(PsiB[j], l, axes=[[0, 1, 3], [1, 3, 0]])
                ER[(j - 1) % L] = l
            else:
                l = np.tensordot(EL[(j - 1) % L],
                                 l.conj(),
                                 axes=[[1, 2], [1, 4]])
                l = np.tensordot(PsiA[j], l, axes=[[0, 1, 2], [1, 3, 0]])
                EL[j] = l

                #X = np.tensordot(l, l, axes = [[1, 2], [1, 2]])
                #print np.linalg.eigvalsh(X)
        else:
            dR = etaH_max
            U, s, V = svd(Tri.reshape((Tri.shape[0], -1)), compute_uv=True)
            target_p_trunc = np.max(
                [np.mean(errors), truncation_par['p_trunc']])
            cum = np.cumsum(s**2)
            dR = np.min([
                np.count_nonzero((1 - cum / cum[-1]) > target_p_trunc) + 1, dR
            ])
            nrm = np.linalg.norm(s[:dR])
            a = U[:, :dR].reshape((chiV, pL, 1, dR)).transpose([1, 3, 0, 2])
            B = ((V[:dR, :].T * s[:dR] / nrm).T).reshape(
                (dR, 1, eta, pR)).transpose([0, 3, 2, 1])
            info = {}
            info['error'] = 2 - 2 * nrm
            info['d_error'] = 2 - 2 * nrm
            info['s_AdPsi'] = np.array([1.])
            errors.append(info['error'])
            T_errors.append(T_errors[-1] + 2 - 2 * nrm)

        Lambda[j] = B
        A[j] = a

    for k in range(0):
        #print "Right"
        for j in range(L - 1):
            Quad = PsiS[j]
            Quad = np.tensordot(EL[(j - 1) % L], Quad, axes=[[0], [2]])
            Quad = np.tensordot(Quad, ER[j], axes=[[4], [0]])
            step(j, Quad, dir='R')
        #print [l.shape[3] for l in A[:-1]]
        #print "Left"
        for j in range(L - 1, 0, -1):
            Quad = PsiS[j]
            Quad = np.tensordot(EL[(j - 1) % L], Quad, axes=[[0], [2]])
            Quad = np.tensordot(Quad, ER[j], axes=[[4], [0]])
            step(j, Quad, dir='L', fixd=True)
        #print [l.shape[2] for l in A[1:]]

    #print "Right"
    for j in range(L - 1):
        Quad = PsiS[j]
        Quad = np.tensordot(EL[(j - 1) % L], Quad, axes=[[0], [2]])
        Quad = np.tensordot(Quad, ER[j], axes=[[4], [0]])
        step(j, Quad, dir='R', fixd=True)
    #print [l.shape[3] for l in A[:-1]]
    j = L - 1
    Quad = PsiS[j]
    Quad = np.tensordot(EL[(j - 1) % L], Quad, axes=[[0], [2]])
    Quad = np.tensordot(Quad, ER[j], axes=[[4], [0]])
    step(j, Quad, dir='R')

    #print "e", np.cumsum(errors)
    #print "Te", list(np.array(T_errors))
    return A, Lambda, {
        'errors': errors,
        'total_error': T_errors[-1],
        'total_d_error': np.sum(d_errors[-L:])
    }
