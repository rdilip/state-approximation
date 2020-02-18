""" This module has functions for performing the basic TEBD2 algorithm on an 
isoTNS """

import sys
import os
from misc import *
from moses_variational import moses_move as moses_move_variational
from moses_simple import moses_move as moses_move_simple
import scipy.linalg
import pickle
from rfunc import single_site_exp, get_N

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

def peps_filling(PEPs, truncation_par=None):
    """ Calculates <N>. """
    if not truncation_par:
        truncation_par = {
            'bond_dimensions': {
                'eta_max': 24,
                'chi_max': 12
            },
            'p_trunc': 1e-10
        }
    d = PEPs[0][0].shape[0]
    L = len(PEPs[0])
    Ns = [[get_N(d-1) for i in range(L)]]
    PEPs, info = peps_ESWN_tebd(PEPs.copy(), [None, None, None, None],
                                truncation_par,
                                Os=[None, None, None, Ns],
                                N_variational=0,
                                scheduleH=None,
                                scheduleV=None,
                                moses=True)
    return(info['expectation_O'])

def tebd_on_mps(Psi, U, truncation_par, order='R', reduced_update=True,
                O=None):
    """
    Applies Trotter decomposed circuit U = U0 U1 . . . Un on an MPS. 
    Parameters
    ----------
    Psi : list
        Vertical wavefunction. List of np.Arrays, viewed as an MPS. Psi can
        have multiple physical legs, and the gate U acts on the first leg
        alone.
    truncation_par : dict
        Dictionary of truncation parameters. Mandatory keys are p_trunc and 
        chi_max. p_trunc is the leftover sum of singular values squared
        to throw away. chi_max is bond dimension (of QR along MPS)
    order : str
        Order can be either 'R' or 'L'. If order is 'R', sweeps left to right
        (Psi starts in B form and ends in A form). Note this is a little 
        ambiguous, since code starts in -+ form (OC at top left corner). Just
        tilt your head to the left.
    reduced_update : bool
        If True, follows Fig. 4 of https://arxiv.org/pdf/1503.05345v2.pdf 
        using a QR decomposition
    O : list
        List of local operators; each element should be either a rank 4 or 
        rank 2 tensor. 
    Returns
    -------
    Psi : list
        Wavefunction after TEBD sweep
    info : dict
        Truncation information
    """

    L = len(Psi)
    assert order in ['R', 'L']

    def sweep(Psi, U, O=None):
        """ Left to right sweep """
        psi = Psi[0]
        num_p = psi.ndim - 2  #number of physical legs
        p_trunc = 0.
        nrm = 1.

        expectation_O = []

        one_site_exp = False
        if O is not None:
            if len(O[0].shape) == 2:
                one_site_exp = True

        for j in range(L - 1):
            psi, pipe1L = group_legs(
                psi, [[0], list(range(1, num_p + 1)), [num_p + 1]
                      ])  # bring to 3-leg form

            B, pipe1R = group_legs(
                Psi[j + 1],
                [[0], [num_p],
                 list(range(1, num_p)) + [num_p + 1]])  # bring to 3-leg form

            if one_site_exp:
                expO = np.tensordot(O[j % len(O)], psi, [0, 0])
                expO = np.tensordot(expO, psi.conj(), [[0,1,2],[0,1,2]])
                expectation_O.append(expO.item())

            if reduced_update and psi.shape[0] * psi.shape[2] < psi.shape[1]:
                reduced_L = True
                psi, pipe2L = group_legs(psi, [[1], [0, 2]])
                QL, psi = np.linalg.qr(psi)
                psi = ungroup_legs(psi, pipe2L)
            else:
                reduced_L = False

            if reduced_update and B.shape[0] * B.shape[1] < B.shape[2]:
                reduced_R = True
                B, pipe2R = group_legs(B, [[0, 1], [2]])
                QR, B = np.linalg.qr(B.T)
                QR = QR.T
                B = B.T
                B = ungroup_legs(B, pipe2R)
            else:
                reduced_R = False

            theta = np.tensordot(psi, B, axes=[[-1], [-2]])  #Theta = s B
            if O is not None and not one_site_exp:
                Otheta = np.tensordot(O[j % len(O)],
                                      theta,
                                      axes=[[2, 3], [0, 2]])
                expectation_O.append(
                    np.tensordot(theta.conj(),
                                 Otheta,
                                 axes=[[0, 2, 1, 3], [0, 1, 2, 3]]))

            if U is not None:
                theta = np.tensordot(U[j % len(U)],
                                     theta,
                                     axes=[[2, 3], [0, 2]])  #Theta = U Theta
            else:
                theta = theta.transpose([0, 2, 1, 3])

            theta, pipeT = group_legs(theta,
                                      [[0, 2], [1, 3]])  #Turn into a matrix

            A, SB, info = svd_theta(theta, truncation_par)  #Theta = A s

            #Back to 3-leg form
            A = A.reshape(pipeT[0][0] + (-1, ))
            SB = SB.reshape((-1, ) + pipeT[0][1]).transpose([1, 0, 2])

            if reduced_L:
                A = np.tensordot(QL, A, axes=[[1], [1]]).transpose([1, 0, 2])
            if reduced_R:
                SB = np.dot(SB, QR)

            A = ungroup_legs(A, pipe1L)
            SB = ungroup_legs(SB, pipe1R)

            p_trunc += info['p_trunc']
            nrm *= info['nrm']

            Psi[j] = A
            psi = SB

        Psi[L - 1] = psi
        if one_site_exp:
            expO = np.tensordot(O[(L-1) % len(O)], psi, [0, 0])
            expO = np.tensordot(expO, psi.conj(), [[0,1,2,3,4],[0,1,2,3,4]])
            expectation_O.append(expO.item())
        return p_trunc, nrm, expectation_O

    if order == 'R':
        p_trunc, nrm, expectation_O = sweep(Psi, U, O)

    if order == 'L':
        Psi = mps_invert(Psi)
        if U is not None:
            U = [u.transpose([1, 0, 3, 2]) for u in U[::-1]]
        if O is not None and len(O[0].shape) == 4:
            O = [o.transpose([1, 0, 3, 2]) for o in O[::-1]]
        if O is not None and len(O[0].shape) == 2:
            O = O[::-1]# Exactly the same anyway...
        p_trunc, nrm, expectation_O = sweep(Psi, U, O)
        expectation_O = expectation_O[::-1]
        Psi = mps_invert(Psi)

    info = {'p_trunc': p_trunc, 'nrm': nrm, 'expectation_O': expectation_O}
    return Psi, info

def peps_sweep(PEPs,
               U,
               truncation_par,
               O=[None],
               moses=True,
               N_variational=0,
               scheduleH=None,
               scheduleV=None,
               verbose=True):
    """ Given a PEPs in (-+) form, with wavefunction in upper-left corner, 
    sweeps right to bring to (++) form, with wavefunction in upper-right corner.
    
    Parameters
    ----------
    PEPs : list of lists
        Isometric peps state. 
    U : list
        List of np.Arrays (4 index tensors). As the peps sweep occurs, applies 
        2-site TEBD gates U[j] on column j.
    truncation_par : dict
        Truncation params. See tebd_on_mps.
    O : list
        List of list of one or two site operators O. O[j] acts on the column j,
        and is a list of operators (so O should have length Lx, and O[j] should
        have length Ly). This will be tiled if len(O) < Lx. There are no checks
        for conmeasuraticity. 
    moses : bool
        If True, performs moses move.
    N_variational : 1
        Number of variational sweeps to perform.
    scheduleH, scheduleV : list
        Schedule. Never used. IDK what the format is. 
    verbose : bool
        If true, print info
    Returns
    -------
    PEPs : list of lists
        isoTNS state after TEBD2
    """
     # TODO fill out comment for schedule

    Psi = PEPs[0]  # One-column wavefunction
    Lx = len(PEPs)
    Ly = len(PEPs[0])
    min_p_trunc = truncation_par['p_trunc']
    target_p_trunc = truncation_par[
        'p_trunc']  #This will be adjusted according to moses-truncation errors
    tebd_2_mm_trunc = 0.1

    nrm = 1.
    tebd_error = []  #Total error from TEBD truncation (one per column)
    moses_error = []  #Total error from MM (one per column)
    moses_d_error = []  #Component of MM error coming from isometric projection
    eta0 = np.ones(
        (Lx - 1, Ly), dtype=np.int
    )  #These are the vertical bond dimensions of zero-column wf after Moses Move
    eta1 = np.ones(
        (Lx, Ly), dtype=np.int
    )  #These are the vertical bond dimensions of single-column wf after TEBD
    expectation_O = []

    if U is None:
        U = [None]
    if O is None:
        O = [None]

    for j in range(Lx):
        Psi, info = tebd_on_mps(
            Psi,
            U[j % len(U)],
            truncation_par={
                'p_trunc': target_p_trunc,
                'chi_max': truncation_par['bond_dimensions']['eta_max']
            },
            order='L',
            O=O[j % len(O)])
        tebd_error.append(info['p_trunc'])
        nrm *= info['nrm']
        expectation_O.append(info['expectation_O'])
        eta1[j][:] = [l.shape[4] for l in Psi]
        #Psi is now in B-form

        if j < Lx - 1:
            Psi, pipe = mps_group_legs(Psi, [[0, 1], [2]])  #View as MPO

            if moses:
                A, Lambda, info = moses_move_simple(Psi,
                                                    truncation_par,
                                                    verbose=verbose,
                                                    scheduleH=scheduleH,
                                                    scheduleV=scheduleV)

            if N_variational > 0:
                if moses:
                    A, Lambda, info = moses_move_variational(Psi,
                                                             A,
                                                             Lambda,
                                                             truncation_par,
                                                             N=N_variational)
                else:
                    A, Lambda, info = moses_move_variational(Psi,
                                                             None,
                                                             None,
                                                             truncation_par,
                                                             N=N_variational)

            moses_error.append(info.setdefault('error', np.nan))
            moses_d_error.append(info.setdefault('d_error', np.nan))

            if not np.isnan(moses_error[-1]):
                target_p_trunc = np.max([
                    tebd_2_mm_trunc * moses_error[-1] / len(Psi), min_p_trunc
                ])

            A = mps_ungroup_legs(A, pipe)
            PEPs[j] = A
            Psi, pipe = mps_group_legs(PEPs[j + 1], axes=[[1], [
                0, 2
            ]])  #Tack Lambda onto next column: Psi = Lambda B
            Psi = mpo_on_mpo(Lambda, Psi)
            Psi = mps_ungroup_legs(Psi, pipe)
            PEPs[j + 1] = Psi
            eta0[j][:] = [l.shape[3] for l in Lambda]
        else:
            Psi = mps_2form(
                Psi, 'A'
            ) 
            PEPs[j] = Psi

    if verbose:
        print(("{:>8.1e} " * Lx).format(*tebd_error))
        print("    ", end=' ')
        print(("{:>8.1e} " * (Lx - 1)).format(*moses_error))
        print("    ", end=' ')
        print(("{:>8.1e} " * (Lx - 1)).format(*moses_d_error))
        print()

    return PEPs, {
        'nrm': nrm,
        'expectation_O': expectation_O,
        'moses_error': moses_error,
        'moses_d_error': moses_d_error,
        'tebd_error': tebd_error,
        'eta0': eta0,
        'eta1': eta1
    }

def rotT(T):
    """ 90-degree counter clockwise rotation of tensor """
    return np.transpose(T, [0, 4, 3, 1, 2])

def peps_rotate(PEPs):
    """ 90-degree counter clockwise rotation of PEPs """
    Lx = len(PEPs)
    Ly = len(PEPs[0])

    rPEPs = [[None] * Lx for y in range(Ly)]
    for y in range(Ly):
        for x in range(Lx):
            #print  y, x, "<---", x, Ly - y - 1, rotT(PEPs[x][Ly - y - 1]).shape
            rPEPs[y][x] = rotT(PEPs[x][Ly - y - 1]).copy()

    return rPEPs

def peps_ESWN_tebd(PEPs,
                   Us,
                   truncation_par,
                   Os=None,
                   N_variational=1,
                   scheduleH=None,
                   scheduleV=None,
                   verbose=0,
                   moses=True):
    """ Applies four sweeps of TEBD to a PEPs as follows:
        1) Starting in B = (-+) form,  sweep right (east) applying a set of 
            1-column gates Us[0] , bringing to A = (++) form
        2) Rotate network 90-degrees counterclockwise, so that effectively 
            (++) ---> (-+)
        3) Repeat 4x

        Us = [UE, US, UW, UN] ; each Ui[x][y] is 2-site gate on column x, bond
            y when sweep is moving in direction 'i'.
        Assumes some sort of inversion symmetry in the gates UW, later will 
        have to fix a convention.

    Parameters
    ----------
    PEPs : list of lists
        isoTNS state 
    Us : list
        Unitary "sheets." Each element of this list is an input to peps_sweep,
        so this list should have at most four elements (tiled if fewer). Each
        element is a list of lists of unitary operators.
    truncation_par : dict
        Truncation parameters
    N_variational : int
        Number of variational sweeps
    schedule_H,V : list
        who even knows
    verbose : bool
        If True, prints output
    moses : bool
        Whether or not to do moses move. 

    Returns
    -------
    PEPs : list
        isoTNS state
    info : dict
        Truncation information
    """

    nrm = 1.
    moses_d_error = 0.
    moses_error = 0.
    tebd_error = 0.
    expectation_O = []

    if Us is None:
        Us = [[None]] * 4
    if Os is None:
        Os = [[None]] * 4

    for j in range(4):
        PEPs, info = peps_sweep(PEPs,
                                Us[j],
                                truncation_par,
                                Os[j],
                                verbose=verbose,
                                N_variational=N_variational,
                                scheduleH=scheduleH,
                                scheduleV=scheduleV,
                                moses=moses)
        #TODO . . . some possible inversion action on Us? Depends on convention
        expectation_O.append(info['expectation_O'])
        nrm *= info['nrm']
        moses_d_error += np.sum(
            info['moses_d_error']) / 4  #Total error / sweep
        moses_error += np.sum(info['moses_error']) / 4  #Total error / sweep
        tebd_error += np.sum(info['tebd_error']) / 4
        PEPs = peps_rotate(PEPs)

    return PEPs, {
        'expectation_O': expectation_O,
        'nrm': nrm,
        'moses_error': moses_error,
        'moses_d_error': moses_d_error,
        'tebd_error': tebd_error,
        'eta0_max': np.max(info['eta0'].flat),
        'eta1_max': np.max(info['eta1'].flat)
    }


def make_U(H, t):
    """ U = exp(-t H). make_U acts assuming H is in the correct input form
    for isoTNS (that is, a single list of lists) """
    d = H[0][0].shape[0]
    return [[
        sp.linalg.expm(-t * h.reshape((d**2, -1))).reshape([d] * 4) for h in Hc
    ] for Hc in H]

def flip_and_conj(H):
    """ Flips and conjugates... """
    # TODO put a real docstring
    H_flipped = H.copy()[::-1]
    return [[
        h.conj() for h in Hc
        ] for Hc in H_flipped]

# NOTE
# Properly, the second unitary list is linked to the fourth unitary list (and
# the first to the third) because they act on the same elements. This
# imposes constraints that p2_tebd doesn't manifestly satisfy. Be careful...

def p2_tebd(PEPs,
            H,
            dt,
            N,
            truncation_par,
            N_variational=1,
            scheduleH=None,
            scheduleV=None,
            verbose=1,
            moses=True,
            save=True):
    """
    Performs a series of sweeps across PEPs to perform TEBD2 algorithm using
    second order Trotterization. Currently set up to output the particle number
    at each step, and measure the energy at the end.

    Parameters
    ----------
    PEPs : list of list of np.Array
        isoTNS state. PEPs[i][j] is tensor on site (i,j)
    H : list
        Hamiltonian in the form [Hh_list, Hv_list]
    dt : float
        Time increment in TEBD
    N : int
        Number of sweeps to do. Each sweep is actually four peps_sweep s, 
        interlaced with rotations.
    truncation_par : dict
        Dictionary of truncation params.  Mandatory keys are p_trunc and 
        chi_max. p_trunc is the leftover sum of singular values squared
        to throw away. chi_max is bond dimension (of QR along MPS)
    N_variational : int
        Number of variational sweeps
    scheduleH, scheduleV : list
        who knows
    verbose : bool
        If true, prints outputs.
    moses : bool
        Whether or not to do moses move.
    save : bool
        Whether or not to save intermediate particle number
    Returns
    -------
    PEPs : list of lists of np.Array
        isoTNS state
    E_exp : list of lists of lists
        Expectation value of H. Each element of E_exp is a list of lists, 
        corresponding to the expectation_value over one sheet before rotation
    info : dict
        Dictionary of various truncation-ey things and such.
    """
    print(f"Truncation parameters: {truncation_par}")

    p_trunc = truncation_par['p_trunc']
    system_size = len(PEPs) * len(PEPs[0])
    # TODO: Maybe Hs should take a different format? Just a straight list.
    # Yes we should this is incredibly misleading...

    Hh1 = H[0][0]
    Hh2 = H[0][1]
    Hv1 = H[1][0]
    Hv2 = H[1][1]

    Uh1 = make_U(H[0][0], dt)
    Uh2 = make_U(H[0][1], dt)
    Uv1 = make_U(H[1][0], dt)
    Uv2 = make_U(H[1][1], dt)

    Uv1_half = make_U(Hv1, dt / 2.) # Half step for 2nd order trotter
    Hv1_half = [[h / 2. for h in Hc] for Hc in Hv2]

    L = len(PEPs)
    nmax = PEPs[0][0].shape[0] - 1
    Ns = [[np.diag(np.arange(nmax + 1)) for i in range(L)]]

    PEPs, info = peps_ESWN_tebd(PEPs, [Uv1_half, Uh1, Uv2, Uh2],
                                truncation_par,                                        
                                Os = [Ns, Ns, Ns, Ns],
                                N_variational=N_variational,
                                scheduleH=scheduleH,
                                scheduleV=scheduleV,
                                moses=moses)

    for j in range(N - 1):
        print(f"Iteration {j} with time dt={dt}")
        PEPs, info = peps_ESWN_tebd(PEPs, [Uv1, Uh1, Uv2, Uh2],
                                    truncation_par,
                                    Os = [Ns, Ns, Ns, Ns],
                                    N_variational=N_variational,
                                    scheduleH=scheduleH,
                                    scheduleV=scheduleV,
                                    moses=moses)
        if save: 
            particle_num_fname = f"particle_number_{j}_dt_{round(dt,3)}"
            if os.path.exists(particle_num_fname):
                particle_number_fname += "_eps"
            particle_num_fname += ".pkl"
            with open(particle_num_fname,"wb+") as f:
                pickle.dump(info["expectation_O"], f)

        print(np.mean(info["expectation_O"][-1]))

    particle_number = info["expectation_O"].copy()
    PEPs, info = peps_ESWN_tebd(PEPs, [Uv1_half, None, None, None],
                                truncation_par,
                                Os=[None, None, Hv1, Hh2],
                                N_variational=N_variational,
                                scheduleH=scheduleH,
                                scheduleV=scheduleV,
                                moses=moses)
    with open(f"particle_number_final.pkl","wb+") as f:
        pickle.dump(peps_filling(PEPs, truncation_par), f)

    info["particle_number"] = particle_number

    E_exp = np.sum([np.sum(o) for o in info['expectation_O']]) / system_size

    if verbose > 0:
        print(("{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}").format(
            '<H>', 'max(eta0)', 'max(eta1)', 'moses_d_err', 'moses_err',
            'tebd_err'))
        print(
            ("{:>12.7f}{:>12.2f}{:>12.2f}{:>12.2e}{:>12.2e}{:>12.2e}").format(
                E_exp, info['eta0_max'], info['eta1_max'],
                info['moses_d_error'] / system_size, info['moses_error'] / system_size,
                info['tebd_error'] / system_size))
        print()

    return PEPs, E_exp, info
