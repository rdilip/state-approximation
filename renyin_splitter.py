from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
#from disentangler.renyi2disentangler import disentangle_2
#from disentangler.cgdisentangler import disentangle_CG
from renyin_disentangler import disentangle_2
from misc import svd, svd_theta_UsV, group_legs, ungroup_legs
from warnings import warn
import time

def find_closest_factors(num):
    factor = int(np.sqrt(num))
    while factor >= 0:
        if num % factor == 0:
            return factor, int(num / factor)
        else:
            factor -= 1

def split_psi(Psi,
              dL,
              dR,
              truncation_par={
                  'chi_max': 32,
                  'p_trunc': 1e-6
              },
              verbose=0,
              n=2,
              eps=1e-6,
              max_iter=120,
              init_from_polar=True,
              pref='dL'):
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

    dL1, dR1 = find_closest_factors(min(d,mL*mR))
    if pref == 'dL':
        dL, dR = max(dL1, dR1), min(dL1, dR1)
    elif pref == 'dR':
        dL, dR = min(dL1, dR1), max(dL1, dR1)
    else:
        raise ValueError


    X, y, Z = svd(Psi.reshape(-1, mL*mR), full_matrices=False)
    D2 = len(y)
    
    A = X
    theta = (Z.T * y).T

    # Disentangle the two-site wavefunction
    theta = np.reshape(theta, (dL, dR, mL, mR))  #view TEBD style
    theta = np.transpose(theta, (2, 0, 1, 3))

    s = np.linalg.svd(theta, compute_uv=False)
    sb = -np.log(np.sum(s[np.abs(s) > 1.e-10]**2))
    theta, U, info = disentangle_2(theta,
                                  eps=10 * eps,
                                  max_iter=max_iter,
                                  verbose=verbose)
    s = np.linalg.svd(theta, compute_uv=False)
    s = s[np.abs(s) > 1.e-10]
    A = np.tensordot(A,
                 np.reshape(np.conj(U), (dL, dR, dL * dR)),
                 axes=([1], [2]))

    theta = np.transpose(theta, [1, 0, 2, 3])
    theta = np.reshape(theta, (dL * mL, dR * mR))

    X, s, Z = svd(theta, full_matrices=False)
    chi_c = len(s)

    S = np.reshape(X, (dL, mL, chi_c))
    S = S * s

    B = np.reshape(Z, (chi_c, dR, mR))
    B = np.transpose(B, (1, 0, 2))

   
    #TODO: technically theses errors are only good to lowest order I think
    return A, S, B
