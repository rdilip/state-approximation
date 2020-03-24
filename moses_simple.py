import numpy as np
from renyin_splitter import split_psi as split_psi
import warnings
from disentanglers import disentangle_brute, disentangle_S2
from rfunc import pad

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
#      |
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
#          mL
#          |
#        S |
#         / \
#        /   \
#       /     \
# d ---/----------- mR
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
def moses_move(Psi, truncation_par=None, disentangler=disentangle_S2):
    """ Splits a 2-sided MPS Psi = [b0, b1, ...]  using a simple moses approach 
    according to
		
    Psi = A Lambda
			
    B0 is the "bottom" of Psi, and Psi MUST be in B-form (arrows pointing downward). 
    Parameters
    ----------
    Psi : list of np.Array
        One column wavefunction 
    truncation_par : dict

        Truncation params. The options are 
            truncation_par: {'bond_dimensions': 
                                {'etaH_max': 4,
                                 'etaV_max': 4,
                                 'chiV_max': 4,
                                 'chiH_max': 4},
                            'p_trunc':1e-6 }
            * etaH/V are for the Lambda (etaV is used only on the top row - 
                it over-rides chiH here)
            * if eta_max is provided instead, etaH = etaV = eta
            * chiH/V for A
            * if chi_max is provided instead, chiH = chiV = chi
            * p_trunc is tolerance: allows local truncations of this magnitudes
    Returns
    -------
    A : list of np.Array
        List of A tensors (one column wavefunction)
    Lambda : List of np.Array
        List of lambda tensors (zero column wavefunction)
    """
    if truncation_par is None:
        truncation_par = {
            'bond_dimensions': dict(eta_max=500, chi_max=500),
            'p_trunc': 1e-6
        }
    if 'eta_max' in truncation_par['bond_dimensions']:
        truncation_par['bond_dimensions']['etaH_max'] = truncation_par[
            'bond_dimensions']['etaV_max'] = truncation_par['bond_dimensions'][
                'eta_max']

    if 'chi_max' in truncation_par['bond_dimensions']:
        truncation_par['bond_dimensions']['chiH_max'] = truncation_par[
            'bond_dimensions']['chiV_max'] = truncation_par['bond_dimensions'][
                'chi_max']

    etaH_max = truncation_par['bond_dimensions']['etaH_max']
    etaV_max = truncation_par['bond_dimensions']['etaV_max']
    chiV_max = truncation_par['bond_dimensions']['chiV_max']
    chiH_max = truncation_par['bond_dimensions']['chiH_max']

    L = len(Psi)
    Lambda = []
    A = []
    
    # Current dimensions of the Pent tensor about to be split
    eta = 1
    chiV = 1
    pL = Psi[0].shape[0]
    pR = Psi[0].shape[1]
    chi = Psi[0].shape[3]

    # Initialize Pent from bottom of MPS
    Pent = Psi[0].reshape((chiV, eta, pL, pR, chi))
    Ss = []
    chiV_max = 2000
    chiH_max = 2000
    
    for j in range(L):
        Tri = Pent.transpose([0, 2, 4, 1, 3])
        Tri = Tri.reshape((chiV * pL, chi, eta * pR))

        dL = chiV_max
        dR = chiH_max
                   
        pref = 'dL'
        if j == L - 1:
            dR = 1
            dL = np.min([etaH_max, Tri.shape[0]])
            pref = 'dR'
        
        if j < L - 1:
            a, S, B = split_psi(Tri,
                                  dL,
                                  dR,
                                  truncation_par={
                                      'chi_max': etaV_max,
                                      'p_trunc': truncation_par['p_trunc']
                                  },
                                  verbose=0,
                                  pref=pref,
                                  disentangler=disentangler)

            dL, dR = a.shape[1], a.shape[2]

            B = B.reshape((dR, B.shape[1], eta, pR)).transpose([0, 3, 2, 1])
            # B now in shape [dR, pR, eta, -1]
            a = a.reshape((chiV, pL, dL, dR)).transpose([1, 3, 0, 2])

            Lambda.append(B)
            A.append(a)

            # TODO redundant
            if j < L - 1:
                pL = Psi[j + 1].shape[0]
                pR = Psi[j + 1].shape[1]
                chi = Psi[j + 1].shape[3]
                Pent = np.tensordot(S, Psi[j + 1], axes=[[1], [2]])
            else:
                breakpoint()
                Lambda[j] = Lambda[j] * S
            chiV = dL
            eta = B.shape[-1]
        else:
            d, mL, mR = Tri.shape
            a, B = np.linalg.qr(Tri.reshape(d, mL*mR))
            a = a.reshape(chiV, pL, 1, a.shape[-1]).transpose([1,3,0,2])
            B = B.reshape((B.shape[0], mL, pR, eta)).transpose([0,2,3,1])
            if a.shape[1] == 1:
                a = pad(a, 1, 2)
                B = pad(B, 0, 2)
            A.append(a)
            Lambda.append(B)
    return(A, Lambda)

def _evaluate(A, B):
    pL, _, chiV, dL = A.shape
    _, pR, eta, chi = B.shape
    C = np.tensordot(A, B, [1,0])
    C = C.reshape(pL, chiV, dL, pR, eta, chi)
    C = C.transpose([0,3,1,4,2,5])
    return(C)
