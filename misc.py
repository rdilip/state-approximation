#from __future__ import print_function
import sys
if sys.version_info[0] == 2:
    from itertools import izip as zip
    import cPickle as pickle
else:
    import pickle

import numpy as np
import scipy as sp
import scipy.linalg
import warnings
import gc

###Tensor stuff


def group_legs(a, axes):
    """ 
    Given list of lists like axes = [ [l1, l2], [l3], [l4 . . . ]]
    does a transposition of np.array "a" according to l1 l2 l3...
    followed by a reshape according to parantheses.

    Return the reformed tensor along with a "pipe" which can be used to
    undo the move.

    pipe has the format of the old shape, and the permutation of the legs.
          
    """

    nums = [len(k) for k in axes]

    flat = []
    for ax in axes:
        flat.extend(ax)

    a = np.transpose(a, flat)
    perm = np.argsort(flat)

    oldshape = a.shape

    shape = []
    oldshape = []
    m = 0
    for n in nums:
        shape.append(np.prod(a.shape[m:m + n]))
        oldshape.append(a.shape[m:m + n])
        m += n

    a = np.reshape(a, shape)

    pipe = (oldshape, perm)

    return a, pipe


def ungroup_legs(a, pipe):
    """ Given the output of group_legs,  recovers the original tensor (inverse
    operation)
		
      For any singleton grouping [l],  allows the dimension to have changed
      (the new dim is inferred from 'a').  """
    if a.ndim != len(pipe[0]):
        raise ValueError
    shape = []
    for j in range(a.ndim):
        if len(pipe[0][j]) == 1:
            shape.append(a.shape[j])
        else:
            shape.extend(pipe[0][j])

    a = a.reshape(shape)
    a = a.transpose(pipe[1])
    return a

#### MPS / MPO STUFF
def transpose_mpo(Psi):
    """Transpose row / column (e.g. bra / ket) of an MPO"""
    return [b.transpose([1, 0, 2, 3]) for b in Psi]


def mps_group_legs(Psi, axes='all'):
    """ Given an 'MPS' with a higher number of physical legs (say, 2 or 3), with B tensors

			physical leg_1 x physical leg_2 x . . . x virtual_left x virtual_right
			
		groups the physical legs according to axes = [ [l1, l2], [l3], . .. ] etc, 
		
		Example:
		
		
			Psi-rank 2,	axes = [[0, 1]]  will take MPO--> MPS
			Psi-rank 2, axes = [[1], [0]] will transpose MPO
			Psi-rank 3, axes = [[0], [1, 2]] will take to MPO
		
		If axes = 'all', groups all of them together.
		
		Returns:
			Psi
			pipes: list which will undo operation
	"""

    if axes == 'all':
        axes = [list(range(Psi[0].ndim - 2))]

    psi = []
    pipes = []
    for j in range(len(Psi)):
        ndim = Psi[j].ndim
        b, pipe = group_legs(Psi[j], axes + [[ndim - 2], [ndim - 1]])

        psi.append(b)
        pipes.append(pipe)

    return psi, pipes


def mps_ungroup_legs(Psi, pipes):
    """Inverts mps_group_legs given its output"""
    psi = []
    for j in range(len(Psi)):
        psi.append(ungroup_legs(Psi[j], pipes[j]))

    return psi


def mps_invert(Psi):
    """ Applies spatial reflection along length of MPS"""
    np = Psi[0].ndim - 2
    return [b.transpose(list(range(np)) + [-1, -2]) for b in Psi[::-1]]

def mps_2form(Psi, form='A', normalize=False, chi_max=None, svd_min=None):
    """Puts an mps with an arbitrary # of legs into A or B-canonical form
    """

    #View as mps
    Psi, pipes = mps_group_legs(Psi, axes='all')

    if form == 'B':
        Psi = [b.transpose([0, 2, 1]) for b in Psi[::-1]]

    L = len(Psi)
    T = Psi[0]

    if chi_max is not None or svd_min is not None:
        normalize = True


    for j in range(L - 1):
        T, pipe = group_legs(T, [[0, 1], [2]])  #view as matrix

        if chi_max is None and svd_min is None:
            A, s = np.linalg.qr(T)  #T = A s can be given from QR
        elif chi_max is not None and svd_min is None:
            A, s, B = np.linalg.svd(T, full_matrices=False)
            A = A[:, :chi_max]
            B = B[:chi_max, :]
            s = s[:chi_max]

            s = s / np.linalg.norm(s)
            s = np.diag(s) @ B
        else:
            A, s, B = np.linalg.svd(T, full_matrices=False)
            s = s[s > svd_min]
            chi_max = len(s)

            A = A[:, :chi_max]
            B = B[:chi_max, :]

            s = s / np.linalg.norm(s)
            s = np.diag(s) @ B

        Psi[j] = ungroup_legs(A, pipe)
        T = np.tensordot(s, Psi[j + 1],
                         axes=[[1],
                               [1]]).transpose([1, 0,
                                                2])  #Absorb s into next tensor

    if normalize:
        Psi[L - 1] = T / np.linalg.norm(T)
    else:
        Psi[L - 1] = T

    if form == 'B':
        Psi = [b.transpose([0, 2, 1]) for b in Psi[::-1]]

    Psi = mps_ungroup_legs(Psi, pipes)

    return Psi


def peel(Psi, p, form='B'):
    """ Put Psi into canonical form, and reshape the physical legs to transfer p-dof from right to left
	"""

    D = [b.shape[:2] for b in Psi]

    if form is not None:
        psi = mps_2form(Psi, form)

    psi = [
        b.reshape((d[0] * p, d[1] / p, b.shape[2], b.shape[3]))
        for b, d in zip(psi, D)
    ]

    return psi


def mps_overlap(Psi0, Psi1):
    """
    #complex compatible
    <psi0|psi1>
    """

    #View as MPS
    if Psi0[0].ndim > 3:
        Psi0, pipes = mps_group_legs(Psi0)
    if Psi1[0].ndim > 3:
        Psi1, pipes = mps_group_legs(Psi1)

    O = np.tensordot(Psi0[0].conj(), Psi1[0], axes=[[0, 1], [0, 1]])
    for j in range(1, len(Psi0)):
        O = np.tensordot(O, Psi1[j], axes=[[1], [1]])
        O = np.tensordot(Psi0[j].conj(), O, axes=[[0, 1], [1, 0]])

    if O.shape != (1, 1):
        print(O.shape)
        raise ValueError

    return O[0, 0]


def mps_entanglement_spectrum(Psi, site_spectrum=None):
    """ Returns the entanglement spectrum on each bond. Spectrum is normalized
	
	"""
    Psi, pipes = mps_group_legs(Psi, axes='all')

    #First bring to A-form
    L = len(Psi)
    T = Psi[0]
    for j in range(L - 1):
        T, pipe = group_legs(T, [[0, 1], [2]])  #view as matrix
        A, s = np.linalg.qr(T)  #T = A s can be given from QR
        Psi[j] = ungroup_legs(A, pipe)
        T = np.tensordot(s, Psi[j + 1],
                         axes=[[1],
                               [1]]).transpose([1, 0,
                                                2])  #Absorb s into next tensor

    Psi[L - 1] = T

    #Flip the MPS around
    Psi = [b.transpose([0, 2, 1]) for b in Psi[::-1]]

    T = Psi[0]

    Ss = []
    for j in range(L - 1):

        if site_spectrum is not None:
            t, pipe = group_legs(T, [[0], [1, 2]])  #view as matrix
            U, s, V = np.linalg.svd(t, full_matrices=False)
            site_spectrum.append(s)

        T, pipe = group_legs(T, [[0, 1], [2]])  #view as matrix
        U, s, V = np.linalg.svd(T, full_matrices=False)
        Ss.append(s / np.linalg.norm(s))
        Psi[j] = ungroup_legs(U, pipe)
        s = ((V.T) * s).T
        T = np.tensordot(s, Psi[j + 1],
                         axes=[[1],
                               [1]]).transpose([1, 0, 2
                                                ])  #Absorb sV into next tensor

    return Ss[::-1]


def mpo_on_mpo(X, Y, form=None):
    """ Multiplies two two-sided MPS, XY = X*Y and optionally puts in a canonical form
	"""
    if X[0].ndim != 4 or Y[0].ndim != 4:
        raise ValueError

    XY = [
        group_legs(np.tensordot(x, y, axes=[[1], [0]]),
                   [[0], [3], [1, 4], [2, 5]])[0] for x, y in zip(X, Y)
    ]

    if form is not None:
        XY = (XY, form)

    return XY


######## PEPS stuff

### PEPs      4
#             |
#             |
#       1 --------- 2
#            /|
#     t =   / |
#          0  3
#
#
#	Stored in cartesian form PEPs[x][y], list-of-lists


def apply_col(v, Z, L):
    #   3       3
    # 0 T 1   0 T 1
    #   2       2
    v = np.tensordot(v, Z[0].transpose(0, 1, 2, 3), axes=(0, 0))
    for i in range(1, L):
        v = np.tensordot(v,
                         Z[i].transpose(0, 1, 2, 3),
                         axes=([0, L + 1], [0, 2]))
    v = np.trace(v, axis1=1, axis2=L + 1)
    return v


def peps_check_isometry(PEPs, l1, l2):

    Lx, Ly = len(PEPs), len(PEPs[0])

    I = np.zeros((Lx, Ly))

    for r in range(Lx):
        for c in range(Ly):
            t = PEPs[r][c]
            t, pipe = group_legs(
                np.tensordot(t.conj(), t, axes=[[0, l1, l2], [0, l1, l2]]),
                [[0, 1], [2, 3]])
            I[r, c] = np.linalg.norm(t - np.eye(t.shape[0])) / np.sqrt(
                t.shape[0])

    return I

def peps_overlap(Psi0, Psi1):
    L = len(Psi0) 
    v = np.ones(L * [1])
    for i in range(L):
        Z = []
        for j in range(L):
            Z_j = np.tensordot(Psi0[i][j], Psi1[i][j], axes=(0, 0))  # lrdulrdu
            Z_j = Z_j.transpose([0, 4, 1, 5, 2, 6, 3, 7])
            chi = Z_j.shape
            Z_j = Z_j.reshape([
                chi[0] * chi[1], chi[2] * chi[3], chi[4] * chi[5],
                chi[6] * chi[7]
            ])
            Z.append(Z_j)
        v = apply_col(v, Z, L)
    return v.item()


def invert_PEPs(PEPs):
    PEPs_I = []
    L = len(PEPs)
    for i in range(L - 1, -1, -1):
        PEPs_I_i = []
        for j in range(L):
            PEPs_I_i.append(PEPs[i][j].transpose([0, 2, 1, 3, 4]))
        PEPs_I.append(PEPs_I_i)
    return PEPs_I


def peps_print_chi(PEPs):
    """	o-- cH--o
		|       |
	    cV      |
		|       |
		o--xxx--o """

    Lx = len(PEPs)
    Ly = len(PEPs[0])
    for y in range(Ly - 1, -1, -1):

        print((" --{:^3d}--" * (Lx - 1)
               ).format(*[t.shape[2]
                          for t in [p[y]
                                    for p in PEPs[:-1]]]))  #+ "X"*(y==(Ly-1))
        if y > 0:
            print(("|       " * (Lx)))
            print(("{:<3d}     " *
                   (Lx)).format(*[t.shape[3] for t in [p[y] for p in PEPs]]))
            print(("|       " * (Lx)))


def peps_check_sanity(PEPs):
    """ Just checks compatible bond dimensions"""
    Lx = len(PEPs)
    Ly = len(PEPs[0])

    for x in range(Lx):
        for y in range(Ly):
            if x < Lx - 1:

                assert PEPs[x][y].shape[2] == PEPs[
                    x + 1][y].shape[1], "{} {} {}".format(
                        x, y, PEPs[x][y].shape)

            if y < Ly - 1:
                assert PEPs[x][y].shape[4] == PEPs[x][
                    y + 1].shape[3], "{} {}".format(x, y)

            assert PEPs[x][y].shape[0] == 2


def Psi_AL_overlap(Psi, A, Lambda):
    """
    #complex compatible
    Goal:
        Compute the global difference |Psi - A Lambda|^2 where Psi is two-sided MPS
        To first order in the error, should be the same as the total error reported by Moses.

        Error = || psi - A*Lambda ||^2
              = nPsi + nAL - 2 * RE [ <psi | A*Lambda > ]

        nPsi and nAL should be one.
    Return:
        Error : a real number
    """

    nPsi = np.real(mps_overlap(Psi, Psi))
    if np.abs(1 - nPsi) > 1e-10:
        print("Psi was not normalized properly: |Psi|^2 = ", nPsi)

    nL = np.real(mps_overlap(Lambda, Lambda))
    if np.abs(1 - nL) > 1e-10:
        print("Lambda was not normalized properly: |L|^2 = ", nL)

    ALambda = mpo_on_mpo(A, Lambda, form='A')
    ALambda, pipe = mps_group_legs(ALambda, 'all')

    nAL = np.real(mps_overlap(ALambda, ALambda))

    if np.abs(nL - nAL) > 1e-10:
        print("A.Lambda was not normalized properly: |L|^2, |AL|^2 = ", nL,
              nAL)

    OV = mps_overlap(ALambda, Psi)

    return nPsi + nAL - 2 * np.real(OV)


def svd(theta, compute_uv=True, full_matrices=True):
    """SVD with gesvd backup"""
    # RD resolve errors
    try:
        A,B,C = np.linalg.svd(theta,
                             compute_uv=compute_uv,
                             full_matrices=full_matrices)



        return((A,B,C))

    except np.linalg.linalg.LinAlgError:
        print("*gesvd*")
        return sp.linalg.svd(theta,
                             compute_uv=compute_uv,
                             full_matrices=full_matrices,
                             lapack_driver='gesvd')


def hosvd(Psi, mode='svd'):
    """Higher order SVD. Given rank-l wf Psi, computes Schmidt spectrum Si on each leg-i, as well as the unitaries Ui
	
		Psi = U1 U2 ... Ul X
	
		Returns X, U, S, the latter two as lists of arrays
	
	"""
    l = Psi.ndim

    S = []
    U = []

    #TODO - probably SVD is more accurate than eigh here??
    for j in range(l):  #For each leg

        if mode == 'eigh':
            rho = Psi.reshape((Psi.shape[0], -1))
            rho = np.dot(rho,
                         rho.T.conj())  #Compute density matrix of first leg
            p, u = np.linalg.eigh(rho)
            perm = np.argsort(-p)
            p = p[perm]
            p[p < 0] = 0.
            u = u[:, perm]
            Psi = np.tensordot(u.conj(), Psi, axes=[[0],
                                                    [0]])  #Strip off unitary
            s = np.sqrt(p)

        else:
            shp = Psi.shape
            Psi = Psi.reshape((shp[0], -1))
            u, s, v = svd(Psi, full_matrices=False)
            Psi = (v.T * s).T
            Psi = Psi.reshape((-1, ) + shp[1:])

        S.append(s)
        U.append(u)
        Psi = np.moveaxis(Psi, 0, -1)

    return Psi, U, S


def svd_theta_UsV(theta, eta, p_trunc=0.):
    """
    SVD of matrix, and resize + renormalize to dimension eta

    Returns: U, s, V, eta_new, p_trunc
        with s rescaled to unit norm
        p_trunc =  \sum_{i > cut}  s_i^2, where s is Schmidt spectrum of theta, REGARDLESS of whether theta is normalized
    """

    U, s, V = svd(theta, compute_uv=True, full_matrices=False)

    nrm = np.linalg.norm(s)
    if not np.isclose(nrm, 1., rtol=1.e-8):
        warnings.warn("svd_theta_UsV is throwing a warning here...")
    #assert(np.isclose(nrm, 1., rtol=1e-8))
    ## This assertion is made because if nrm is not equal to 1.,
    ## the report truncation error p_trunc should be normalized?

    if p_trunc > 0.:
        eta_new = np.min(
            [np.count_nonzero((nrm**2 - np.cumsum(s**2)) > p_trunc) + 1, eta])
    else:
        eta_new = eta

    nrm_t = np.linalg.norm(s[:eta_new])

    return U[:, :eta_new], nrm * s[:eta_new] / nrm_t, V[:eta_new, :], len(
        s[:eta_new]), nrm**2 - nrm_t**2


def svd_theta(theta, truncation_par):
    """ SVD and truncate a matrix based on truncation_par = {'chi_max': chi, 'p_trunc': p }
	
		Returns  normalized A, sB even if theta was not normalized
		
		info = {
		p_trunc =  \sum_{i > cut}  s_i^2, where s is Schmidt spectrum of theta, REGARDLESS of whether theta is normalized
		
	"""

    U, s, V = svd(theta, compute_uv=True, full_matrices=False)

    nrm = np.linalg.norm(s)
    if truncation_par.get('p_trunc', 0.) > 0.:
        eta_new = np.min([
            np.count_nonzero(
                (nrm**2 - np.cumsum(s**2)) > truncation_par.get('p_trunc', 0.))
            + 1,
            truncation_par.get('chi_max', len(s))
        ])
    else:
        eta_new = truncation_par.get('chi_max', len(s))
    nrm_t = np.linalg.norm(s[:eta_new])
    A = U[:, :eta_new]
    SB = ((V[:eta_new, :].T) * s[:eta_new] / nrm_t).T

    info = {
        'p_trunc': nrm**2 - nrm_t**2,
        's': s,
        'nrm': nrm,
        'eta': A.shape[1]
    }

    # RD: Seeing if this solves numerical errors
    A[np.where(np.abs(A) < 1.e-10)] = 0
    SB[np.where(np.abs(SB) < 1.e-10)] = 0
    return A, SB, info

def renyi(s, n):
    """n-th Renyi entropy from Schmidt spectrum s
    """
    s = s[s > 1e-16]
    if n == 1:
        return -2 * np.inner(s**2, np.log(s))
    elif n == 'inf':
        return -2 * np.log(s[0])
    else:
        return np.log(np.sum(s**(2 * n))) / (1 - n)


def Sn(psi, n = 1, group = [[0], [1]]):
    """ Given wf. psi, returns spectrum s & nth Renyi entropy via SVD
        group indicates how indices should be grouped into two parties
    
    """
    theta, pipe = group_legs(psi, group)
    s = svd(theta, compute_uv=False)
    S = renyi(s, n)
    return s, S
