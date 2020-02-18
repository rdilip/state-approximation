from misc import *
import scipy as sp
import time
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


def var_Lambda_1site(Psi, A, Lambda):
    '''
    Input:
        Psi with shape [psi_phys_1, psi_phys_2, psi_down, psi_up]
        A tensor with shape (A_phys, A_right, A_down, A_up)
        L tensor with shape (L_left, L_phys, L_down, L_up)

        Lp tensor with shape (_psi_up, _l_up_dim, _a_up),
        Rp tensor with shape (_psi_down, _l_down, _a_down)

        Frank's right is my up
        Left move == down move

    '''
    L = len(Psi)
    Lp = np.zeros([1, 1, 1])
    Lp[0, 0, 0] = 1.
    Lp_list = [Lp]

    # Create Cache for "down" enviornment
    # by contracting from the bottom.
    for i in range(L):
        # ([_psi_up], _l_up, _a_up),
        # (psi_phys_1, psi_phys_2, [psi_down], psi_up)
        Lp = np.tensordot(Lp, Psi[i], axes=(0, 2))
        # (_l_up, [_a_up], [psi_phys_1], psi_phys_2, psi_up),
        # ([A_phys], A_right, [A_down], A_up)
        Lp = np.tensordot(Lp, A[i].conj(), axes=([1, 2], [2, 0]))
        # ([_l_up], [psi_phys_2], psi_up, [A_right], A_up)
        # ([L_left], [L_phys], [L_down], L_up)
        Lp = np.tensordot(Lp, Lambda[i].conj(), axes=([0, 3, 1], [2, 0, 1]))
        # (psi_up, A_up, L_up)
        Lp = Lp.transpose([0, 2, 1])
        # (psi_up, L_up, A_up)
        Lp_list.append(Lp)

    ## Left move
    # Our down move from top
    Rp = np.zeros([1, 1, 1])  # (_psi_down, _l_down, _a_down)
    Rp[0, 0, 0] = 1.
    Rp_list = [Rp]
    Lambdap = [[] for i in range(L)]  # The new lambda tensor, pointing down
    ####
    ####
    # Lambdap = Lambda[:]

    for i in range(L - 1, -1, -1):
        # ([_psi_down], _l_down, _a_down) (psi_phys_1, psi_phys_2, psi_down, [psi_up])
        Rp = np.tensordot(Rp, Psi[i], axes=(0, 3))
        # (_l_down, [_a_down], [psi_phys_1], psi_phys_2, psi_down)
        # ([A_phys], A_right, A_down, [A_up])
        Rp = np.tensordot(Rp, A[i].conj(), axes=([2, 1], [0, 3]))
        # (_l_down, psi_phys_2, [psi_down], A_right, [A_down])
        # ([_psi_up], _l_up, [_a_up])
        theta = np.tensordot(Rp, Lp_list[i], axes=([2, 4], [0, 2]))
        # theta (_l_down, psi_phys_2, A_right, _l_up)
        theta = theta.transpose(2, 1, 3, 0)
        # theta (L_left, L_phys, L_down, L_up)

        d1, d2, chi1, chi2 = theta.shape
        Q, R = np.linalg.qr(
            theta.transpose(0, 1, 3, 2).reshape(d1 * d2 * chi2,
                                                chi1))  #, mode='raw')
        Lambdap[i] = Q.reshape(d1, d2, chi2, -1).transpose(0, 1, 3, 2)

        if i == 0:
            assert R.size == 1
            R = R / np.linalg.norm(R)
            Lambdap[i] = Lambdap[i] * R[0,0]

        # ([_l_down], [psi_phys_2], psi_down, [A_right], A_down)
        # ([L_left], [L_phys], L_down, [L_up])
        Rp = np.tensordot(Rp, Lambdap[i].conj(), axes=([0, 1, 3], [3, 1, 0]))
        # (_psi_down, _a_down, _l_down)
        Rp = Rp.transpose(0, 2, 1)
        # (_psi_down, _l_down, _a_down)
        Rp_list.append(Rp)
        # print(1-np.abs(overlap(Psi, A, Lambdap)))
        # import pdb;pdb.set_trace()

    ## Right move
    # Our up move from bottom
    Lp = np.zeros([1, 1, 1])  # (_psi_up, _l_up_dim, _a_up)
    Lp[0, 0, 0] = 1.
    Lp_list = [Lp]
    Lambdap = [[] for i in range(L)]

    for i in range(L):
        Lp = np.tensordot(Lp, Psi[i], axes=(0, 2))
        Lp = np.tensordot(Lp, A[i].conj(), axes=([1, 2], [2, 0]))
        theta = np.tensordot(Lp, Rp_list[L - 1 - i], axes=([2, 4], [0, 2]))
        theta = theta.transpose(2, 1, 0, 3)

        d1, d2, chi1, chi2 = theta.shape
        Q, R = np.linalg.qr(theta.reshape(d1 * d2 * chi1, chi2))
        Lambdap[i] = Q.reshape(d1, d2, chi1, -1)

        if i == L-1:
            assert R.size == 1
            R = R / np.linalg.norm(R)
            Lambdap[i] = Lambdap[i] * R[0,0]

        Lp = np.tensordot(Lp, Lambdap[i].conj(), axes=([0, 1, 3], [2, 1, 0]))
        Lp = Lp.transpose(0, 2, 1)
        Lp_list.append(Lp)

    return Lambdap, Lp_list

# def var_Lambda_1site(Psi, A, Lambda, Lp_list=None):
#     L = len(Psi)
# 
#     if Lp_list is None:
#         Lp = np.zeros([1, 1, 1])
#         Lp[0, 0, 0] = 1.
#         Lp_list = [Lp]
# 
#         for i in range(L):
#             Lp = np.tensordot(Lp, Psi[i], axes=(0, 2))
#             Lp = np.tensordot(Lp, A[i].conj(), axes=([1, 2], [2, 0]))
#             Lp = np.tensordot(Lp, Lambda[i].conj(), axes=([0, 3, 1], [2, 0, 1]))
#             Lp = Lp.transpose([0, 2, 1])
#             Lp_list.append(Lp)
# 
#     for i in range(2):
#         Rp = np.zeros([1, 1, 1])
#         Rp[0, 0, 0] = 1.
#         Rp_list = [Rp]
#         Lambdap = [[] for i in range(L)]
# 
#         for i in range(L - 1, -1, -1):
#             Rp = np.tensordot(Rp, Psi[i], axes=(0, 3))
#             Rp = np.tensordot(Rp, A[i].conj(), axes=([1, 2], [3, 0]))
# 
#             theta = np.tensordot(Rp, Lp_list[i], axes=([2, 4], [0, 2]))
#             theta = theta.transpose(2, 1, 3, 0)
# 
#             d1, d2, chi1, chi2 = theta.shape
#             #Q,R = np.linalg.qr(theta.transpose(0,1,3,2).reshape(d1*d2*chi2,chi1))
#             Q, Y, Z = svd(theta.transpose(0, 1, 3, 2).reshape(d1 * d2 * chi2, chi1),full_matrices=False)
#             Lambdap[i] = Q.reshape(d1, d2, chi2, -1).transpose(0, 1, 3, 2)
#             if i == 0:
#                 assert Y.size == 1
#                 Y = Y / np.linalg.norm(Y)
#                 Lambdap[i] = Lambdap[i] * Y[0]
# 
#             Rp = np.tensordot(Rp, Lambdap[i].conj(), axes=([0, 1, 3], [3, 1, 0]))
#             Rp = Rp.transpose(0, 2, 1)
#             Rp_list.append(Rp)
# 
#         A = mps_invert(A)
#         Psi = mps_invert(Psi)
#         Lambda = mps_invert(Lambdap)
#         Lp_list = Rp_list
# 
#     return Lambda, Lp_list

def var_Lambda_2site(Psi, A, Lambda, eta, Lp_list):
    ##  [TODO]
    ##  making this function complex compatible
    ##
    L = len(Psi)

    if Lp_list is None:
        Lp = np.zeros([1, 1, 1])
        Lp[0, 0, 0] = 1.
        Lp_list = [Lp]

        for i in range(L):
            Lp = np.tensordot(Lp, Psi[i], axes=(0, 2))
            Lp = np.tensordot(Lp, A[i], axes=([1, 2], [2, 0]))
            Lp = np.tensordot(Lp, Lambda[i], axes=([0, 3, 1], [2, 0, 1]))
            Lp = Lp.transpose([0, 2, 1])
            Lp_list.append(Lp)

    #### Left move
    Rp = np.zeros([1, 1, 1])
    Rp[0, 0, 0] = 1.
    Rp_list = [Rp]
    Lambdap = [[] for i in range(L)]

    for i in range(L - 1, 0, -1):
        Rp = np.tensordot(Rp, Psi[i], axes=(0, 3))
        Rp = np.tensordot(Rp, A[i], axes=([2, 1], [0, 3]))

        theta = np.tensordot(Psi[i - 1], Rp, axes=(3, 2))
        theta = np.tensordot(theta, A[i - 1], axes=([0, 6], [0, 3]))

        theta = np.tensordot(theta, Lp_list[i - 1], axes=([1, 6], [0, 2]))
        theta = theta.transpose(0, 4, 5, 2, 3, 1)
        d1a, d2a, chia, d1b, d2b, chib = theta.shape
        theta = theta.reshape(d1a * d2a * chia, d1b * d2b * chib)

        U, s, V, eta_new, p_trunc = svd_theta_UsV(theta, eta)

        M = V.reshape(eta_new, d1b, d2b, chib).transpose(2, 1, 0, 3)
        Rp = np.tensordot(Rp, M, axes=([0, 1, 3], [3, 1, 0]))
        Rp = Rp.transpose(0, 2, 1)
        Rp_list.append(Rp)

    #### Right move
    Lp = np.zeros([1, 1, 1])
    Lp[0, 0, 0] = 1.
    Lp_list = [Lp]
    Lambdap = [[] for i in range(L)]

    for i in range(L - 1):
        Lp = np.tensordot(Lp, Psi[i], axes=(0, 2))
        Lp = np.tensordot(Lp, A[i], axes=([1, 2], [2, 0]))

        theta = np.tensordot(Lp, Psi[i + 1], axes=(2, 2))
        theta = np.tensordot(theta, A[i + 1], axes=([3, 4], [2, 0]))
        theta = np.tensordot(theta, Rp_list[L - 2 - i], axes=([4, 6], [0, 2]))
        theta = theta.transpose(1, 2, 0, 3, 4, 5)

        d1a, d2a, chia, d1b, d2b, chib = theta.shape
        theta = theta.reshape(d1a * d2a * chia, d1b * d2b * chib)

        U, s, V, eta_new, p_trunc = svd_theta_UsV(theta, eta)

        M = U.reshape(d1a, d2a, chia, eta_new).transpose(1, 0, 2, 3)
        Lambdap[i] = M

        Lp = np.tensordot(Lp, M, axes=([0, 1, 3], [2, 1, 0]))
        Lp = Lp.transpose(0, 2, 1)
        Lp_list.append(Lp)

    M = np.dot(np.diag(s), V).reshape(eta_new, d1b, d2b,
                                      chib).transpose(2, 1, 0, 3)
    Lambdap[L - 1] = M

    return Lambdap, Lp_list


def var_A(Psi, A, Lambda, Lp_list=None):
    L = len(Psi)
    if Lp_list is None:
        Lp = np.zeros([1, 1, 1])
        Lp[0, 0, 0] = 1.
        Lp_list = [Lp]

        for i in range(L):
            Lp = np.tensordot(Lp, Psi[i], axes=(0, 2))
            Lp = np.tensordot(Lp, A[i].conj(), axes=([1, 2], [2, 0]))
            Lp = np.tensordot(Lp, Lambda[i].conj(), axes=([0, 3, 1], [2, 0, 1]))
            Lp = Lp.transpose([0, 2, 1])
            Lp_list.append(Lp)

    #### Left move
    Rp = np.zeros([1, 1, 1])
    Rp[0, 0, 0] = 1.
    Rp_list = [Rp]
    Ap = [[] for i in range(L)]

    for i in range(L - 1, -1, -1):
        Rp = np.tensordot(Rp, Psi[i], axes=(0, 3))
        Rp = np.tensordot(Rp, Lambda[i].conj(), axes=([0, 3], [3, 1]))
        theta = np.tensordot(Rp, Lp_list[i], axes=([2, 4], [0, 1]))
        theta = theta.transpose(1, 3, 2, 0)

        chiL, chiD, chiR, chiU = theta.shape
        X, s, Y = svd(theta.reshape(chiL * chiD, chiR * chiU),
                      full_matrices=False)
        Ap[i] = np.dot(X, Y).reshape(chiL, chiD, chiR,
                                     chiU).transpose(0, 2, 1, 3)

        Rp = np.tensordot(Rp, Ap[i].conj(), axes=([0, 1, 3], [3, 0, 1]))
        Rp_list.append(Rp)

    return Ap, Lp_list, Rp_list


def moses_move(
        Psi,
        A,
        Lambda,
        truncation_par={'bond_dimensions': {
            'eta_max': 4,
            'chi_max': 4
        }},
        N=10,
        tol=1e-8,
        verbose=0):
    """ Splits a 2-sided MPS Psi = [b0, b1, ...] using a variational approach according to
		
			Psi = A Lambda
			
		B0 is the "bottom" of Psi, and Psi MUST be in B-form (arrows pointing downward). Returns
		
			A = [a0, . . .
			Lambda = [l0, . . .
			
		which are in A-form (arrows pointing UPWARDS)
		
		Options:
		
		- truncation_par: {'bond_dimensions': {'eta_max': 4,'chi_max':4}}
				
		Returns:
		
			A = [a0, a1, ...]
			
			Lambda = [l0, l1, ...]
			
			info = {'errors': [e0, e1, ...]  The local error incurred at each split; error is of form |psi - a l|^2
					's':[ s0[], s1[], ...] The Schmidt spectra of each bond as read from each splitting step. Note, this is not strictly 
					the schmidt spectra of Lambda because of errors occured at future steps, but is as errors --> 0.} [TODO]
	"""

    eta_max = truncation_par['bond_dimensions']['eta_max']

    if A == None or Lambda == None:
        chi_max = truncation_par['bond_dimensions']['chi_max']
        A = []
        L = len(Psi)
        Lambda = []

        for i in range(L):
            chi_m = chi_max

            Psi_chi_l, Psi_chi_r, Psi_chi_d, Psi_chi_u = Psi[i].shape

            if i == 0:
                chi_d = eta_d = 1
            else:
                chi_d = chi_u
                eta_d = eta_max

            if i == L - 1:
                chi_u = eta_u = 1
            else:
                chi_u = chi_max
                eta_u = eta_max

                d = Psi_chi_l * chi_d

                if d < chi_max * chi_max:
                    chi_u = np.min([int(np.sqrt(d)), chi_max])
                    chi_m = np.min([d / chi_u, chi_max])

            #      3                 3         3
            #      v                 ^         ^
            #      |                 |         |
            #  0->- -<-1         0->- ->1  0->- -<-1
            #      |                 |         |
            #      v                 ^         ^
            #      2                 2         2

            A.append((0.5 - np.random.rand(Psi_chi_l, chi_m, chi_d, chi_u)) / 1000.)
            chi_l_p, chi_r_p = Psi_chi_l, np.min([chi_m, Psi_chi_r])
            chi_d_p, chi_u_p = np.min([chi_d, Psi_chi_d]), np.min([chi_u, Psi_chi_u])
            A[-1][:chi_l_p, :chi_r_p, :chi_d_p, :chi_u_p] = Psi[i][:chi_l_p, :chi_r_p, :chi_d_p, :chi_u_p]

            Lambda.append((0.5 - np.random.rand(chi_m, Psi_chi_r, eta_d, eta_u)) / 1000.)
            X = np.min([chi_l_p, chi_r_p])
            Lambda[-1][:X, :X, 0, 0] = np.eye(X)

    Lp_list = None
    # print("before err", Psi_AL_overlap(Psi, A, Lambda))
    err_prev = 1.
    for i in range(N):
        A, Lp_list, Rp_list = var_A(Psi, A, Lambda, Lp_list)
        err_now = np.real(2*(1-Rp_list[-1]))[0,0,0]
        # print("i = ", i, "err", err_now)
        Lambda, Lp_list = var_Lambda_1site(Psi, A, Lambda)
        # Lambda, Lp_list = var_Lambda_2site(Psi,A,Lambda,eta_max,Lp_list=None)
        err_now = np.real(2*(1-Lp_list[-1]))[0,0,0]
        # print("i = ", i, "err", err_now)
        err_diff = err_prev - err_now
        if err_diff < 1e-8:
            if verbose > 0:
                print("at iter : ", i, "err_diff = ", err_diff, "err = ", err_now)

            break
        else:
            err_prev = err_now

    assert np.isclose(Psi_AL_overlap(Psi, A, Lambda), err_now)

    return A, Lambda, {'error': Psi_AL_overlap(Psi, A, Lambda)}
