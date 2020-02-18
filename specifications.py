###SPLIT PSI INDEX CONVENTIONS

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


###
#
#
#   Errors should be reported as   |X - Y|^2, NOT |X - Y|
#	This way they can be added in quadrature easily
#
#
#
def split_psi(psi,
              dL,
              dR,
              truncation_par={
                  'chi_max': 256,
                  'p_trunc': 1e-6
              },
              verbose=0,
              A=None,
              eps=1e-8,
              max_iter=100):
    """ Given a tripartite state psi.shape = d x mL x mR, find an approximation
		
			psi = A.Lambda
	
		where A.shape = d x dL x dR  is an isometry that "splits" d --> dL x dR and Lambda.shape = mL dL dR mR is a 2-site TEBD-style wavefunction of unit norm and maximum Schmidt rank 'chi_max.'
	
		The solution for Lambda is given by the MPS-type decomposition

			Lambda_{mL dL dR mR} = \sum_eta  S_{dL mL eta} B_{dR eta mR}   (*note* ordering convention in S, B: chosen to agree with TenPy-MPS)

		where 1 < eta <= chi_max, and Lambda has unit-norm


		Arguments:
		
			psi:  shape = d, mL, mR
			
			dL, dR: ints specifying the MAXIMUM for splitting dimensions. The actual dL, dR should be read off from the shape of the returned A.
			
			truncation_par  = {'chi_max':, 'p_trunc':}  truncation parameters for the SVD of Lambda; p_trunc is acceptable discarded weight
			
			verbose: integer, verbosity setting
		
		Hints: These may not be needed in your splitter
		
			A: initial guess for A, provided as A[d, dl, dr]
			
			eps: precision of iterative solver routine
			
			max_iter: max iterations of routine (through warning if this is reached!)

		Returns:
		
			A: d x dL x dR
			S: dL x mL x eta
			B: dR x eta x mR
		
			info = {} , a dictionary of (optional) errors and entanglement info
			
				'error': |Psi - A.Lambda |^2 where Lambda = S.B
				'd_error': |Psi - A A* Psi|^2 This is the error incurred purely by d > dL*dR, in limit eta ---> infty.
				
				'num_iter': num iterations of solver
				
				's_AdPsi': Schmidt spectrum of (Ad.psi)_{dL mL , dR mR} BEFORE truncation to chi_max and WITHOUT normalizing
				's_Lambda': Schmidt spectrum of normalized Lambda = S.B
		
	"""

    raise NotImplemented


def moses_move(
        Psi,
        truncation_par={
            'chi_max': {
                'eta_max': 4,
                'etaH_max': 4,
                'etaV_max': 4,
                'chi_max': 4,
                'chiV_max': 4,
                'chiH_max': 4
            },
            'p_trunc': 1e-6
        },
        transpose=False,
        scheduleV=None,
        scheduleH=None,
        verbose=0):
    """ Splits a 2-sided MPS Psi = [b0, b1, ...] according to
		
			Psi = A Lambda
			
		B0 is the "bottom" of Psi, and Psi MUST be in B-form (arrows pointing downward). Returns
		
			A = [a0, . . .
			Lambda = [l0, . . .
			
		which are in A-form (arrows pointing UPWARDS)
		
		Options:
		
		- truncation_par: {'chi_max': {'etaH_max': 4, 'etaV_max':4, 'chiV_max':4, 'chiH_max':4}, 'p_trunc':1e-6 }
		
				etaH/V are for the Lambda (etaV is used only on the top row - it over-rides chiH here)
					- if eta_max is provided instead, etaH = etaV = eta
					
				chiH/V for A
					- if chi_max is provided instead, chiH = chiV = chi
					
				p_trunc is tolerance: allows local truncations of this magnitudes
		
		- transpose = True, solves
			
				Psi = Lambda B
				
			instead (returned as B, Lambda).
			
		
		- scheduleH, scheduleV: optional list of integers specifying horizontal / vertical bond dimensions, starting from bottom. If shorter than length of state, cycles through periodically.
		
		
		
		Returns:
		
			A = [a0, a1, ...]
			
			Lambda = [l0, l1, ...]
			
			info = {'errors': [e0, e1, ...]    #Returns data from each split step, see splitter spec.
					'd_errors': [e0, e1, ...] 
					's_AdPsis':[s0, s1, ...]
					's_Lambdas':[s0, s1, ...]
					
					}
		
	"""

    raise NotImplemented

    if transpose:
        Psi = transpose_mpo(Psi)

    chi_max = truncation_par['chi_max']
    if 'eta_max' in chi_max:
        etaV_max = etaH_max = chi_max['eta_max']
    else:
        etaV_max = chi_max['etaV_max']
        etaH_max = chi_max['etaH_max']

    if 'chi_max' in chi_max:
        chiV_max = chiH_max = chi_max['eta_max']
    else:
        etaV_max = chi_max['etaV_max']
        etaH_max = chi_max['etaH_max']

    chiV_max = chi_max['chiV_max']
    chiH_max = chi_max['chiH_max']
    chiT_max = chiV_max * chiH_max

    info = {
        'errors': errors,
    }

    if transpose:
        A = transpose_mpo(A)
        Lambda = transpose_mpo(Lambda)

    return A, Lambda, info
