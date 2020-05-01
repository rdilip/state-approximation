from __future__ import print_function, absolute_import
import numpy as np
import scipy as sp
import scipy.linalg, scipy.sparse.linalg

if __name__ == "__main__":
	from helper_func import zero_if_close, zero_real_if_close, compute_entropy, random_unitary
else:
	from .helper_func import zero_if_close, zero_real_if_close, compute_entropy, random_unitary


########################################################################################################################
########################################################################################################################
class Renyi_Hessian(sp.sparse.linalg.LinearOperator):
	"""
	Attributes:
		psi:
		dim:  dimensions of each sbusystem. [len 4 tuple]
		dimB:  |BL||BR| [int]
		alpha:  Renyi index
		S:  Entropy
		Z:  Partition function
		s:  list of singular values between (A,BL) and (BR,C). [1-array]
		chi:  length of s [int]
		psiABL:  [3-array shaped (|A|,|BL|,chi)]
		psiBRC:  [3-array shaped (chi,|BR|,|C|)]
		num_matvec_calls:  counts the number of times matvec() is called. [int]

	Notations and conventions:
		The four subsystems are called A, BL, BR, and C, with
			L = A + BL, 
			B = BL + BR,
			R = BR + C.
		Z means Tr[rho^alpha]
		U are 2-tensors square matrices with dimension (|BL||BR|, |BL||BR|),
			U|psi> is applied to the B-legs of psi.
		Z[U] means Tr[rho[U]^alpha], where rho[U] is computed from U|psi>.
		X is an antiHermitian matrix which exponentiates to U:  U = 1 + X + X^2/2 + ...
		The four-index Hessian gives the second order correction:
			delta^2 Z/S = -(1/2) H_{abcd} X_{ba} X_{cd}
			            = (1/2) X*_{ab} H_{abcd} X_{cd}
		d2Z_matvec/d2S_matvec) computes H(X) and outputs a 2-tensor object H_{abcd} X_{cd}
"""

	def __init__(self, Renyi_index, psi_ABBC, func='S', vec_encoding='reshape', prep_Hessian=True):
		"""Initialize a Renyi_Hessian object:

	Parameters:
		Renyi_index:
		psi_ABBC:  [4-tensor]
		func:  Determines what to compute the Hessian for, either 'S' or 'Z".  (Or 'SZ', but use at your own risk!)
		vec_encoding:  Doesn't do anything yet.
		prep_Hessian [default True]:  Pre-compute stuff needed to run the Hessian.
"""
		assert isinstance(psi_ABBC, np.ndarray) and psi_ABBC.ndim == 4
		self.dim = psi_ABBC.shape
		self.dimB = self.dim[1] * self.dim[2]
		self.alpha = float(Renyi_index)
		assert self.alpha > 0
		psi_norm = np.linalg.norm(psi_ABBC.reshape(-1))
		if abs(1 - psi_norm) > 1e-8: print("Renyi_Hessian: Warning!  psi_ABBC not normalized: (1 - |psi|) = {}".format(1 - psi_norm))
		self.psi = psi_ABBC / psi_norm
		self.psi_dtype = psi_ABBC.dtype

		if vec_encoding == 'reshape':
			self.shape = (self.dimB**2, self.dimB**2)
			self.dtype = self.psi_dtype
		self.num_matvec_calls = 0

		if func == 'S': self.func = 'S'
		elif func == 'Z': self.func = 'Z'
		elif func == 'ZS' or func == 'SZ': self.func = 'ZS'
		else: raise ValueError
		self.svd_psi()
		self.compute_cache_objs(prep_Hessian=prep_Hessian)
		#print("Renyi_Hessian.__init__():  S[{}] = {}".format(self.alpha, self.entropy()))


	def _matvec(self, v):
		self.num_matvec_calls += 1
		dB = self.dimB
		X = self.X_vec_to_mat(v)
		if 'S' in self.func: HX = self.d2S_matvec(X)
		elif self.func == 'Z': HX = self.d2Z_matvec(X)
		return self.X_mat_to_vec(HX).reshape(v.shape)


	def check_consistency(self, tol=5e-14):
		dim = self.dim
		dA, dBL, dBR, dC = dim
		dB = self.dimB
		chi = self.chi
		alpha = self.alpha
		assert isinstance(dim, tuple) and len(dim) == 4 and np.all(np.array(dim) > 0)
		assert type(dA) == int and type(dBL) == int and type(dBR) == int and type(dC) == int
		assert type(dB) == int and dB == dBL * dBR and type(chi) == int and chi > 0
		assert type(alpha) == float and alpha > 0.
		psi = self.psi
		assert isinstance(psi, np.ndarray) and psi.shape == dim and psi.dtype == self.psi_dtype
		s = self.s
		assert isinstance(s, np.ndarray) and s.shape == (chi,) and s.dtype == np.float
	##	Internal consistency checks
		#print( np.tensordot(np.dot( self.psiABL, np.diag(self.s) ), self.psiBRC, axes=[2,0] ) )
		#print(self.psi)
		S = self.S
		Z = self.Z
		func = self.func
		assert func == 'Z' or func == 'S' or func == 'ZS'
		tab_sL, tab_sR = np.meshgrid(s, s)
		if 'Z' in func:		# Z stuff
			assert isinstance(self.cache_psi_2am1, np.ndarray) and self.cache_psi_2am1.shape == (dA,dB,dC) and self.cache_psi_2am1.dtype == self.psi_dtype
			assert self.cache_termO_factor.dtype == np.float
			assert self.cache_termM_factor.dtype == np.float
			MtoO_factor = (self.cache_termM_factor + tab_sR**(2*alpha-2)) * tab_sL / tab_sR
			MtoO_factor_diff = np.max(np.abs(MtoO_factor - self.cache_termO_factor))
			if MtoO_factor_diff > 1e-6:
				print("Renyi_Hessian.check_consistency():  MtoO_factor_diff too large: {}".format(MtoO_factor_diff))
				#raise ValueError(MtoO_factor_diff)
		if 'S' in func:		# S stuff
			assert isinstance(self.cache_p_am1, np.ndarray) and self.cache_p_am1.shape == (chi,) and self.cache_p_am1.dtype == np.float
			assert isinstance(self.cache_psi_rshp, np.ndarray) and self.cache_psi_rshp.shape == (dB,dA*dC) and self.cache_psi_rshp.dtype == self.psi_dtype
			assert isinstance(self.cache_psi_Sweight, np.ndarray) and self.cache_psi_Sweight.shape == (dB,dA*dC) and self.cache_psi_Sweight.dtype == self.psi_dtype
		if 'S' in func and 'Z' in func:
			Ofactor_diff = np.max(np.abs(self.cache_StermO_factor * (1-alpha) / alpha * Z - self.cache_termO_factor))
			if Ofactor_diff > tol: raise RuntimeError("Ofactor_diff", Ofactor_diff)
			Mfactor_diff = np.max(np.abs(self.cache_StermM_factor * (1-alpha) / alpha * Z - self.cache_termM_factor - 1))
			if Mfactor_diff > tol: raise RuntimeError("Mfactor_diff", Mfactor_diff)
			s2am1factor_diff = np.max(np.abs( self.cache_psi_2am1.transpose(1,0,2) - self.cache_psi_rshp.reshape(dB,dA,dC) - self.cache_psi_Sweight.reshape(dB,dA,dC) * Z*(1-alpha)/alpha ))
			if s2am1factor_diff > tol: raise RuntimeError("s2am1factor_diff", s2am1factor_diff)
		#print("Renyi_Hessian.check_consistency():")


	################################################################################
	def rho_AC(self, join_AC=True):
		"""Gives the density matrix on subsystem AC.
	If join_AC is True, returns a 2-tensor with legs (AC),(A'C'), otherwise returns a 4-tensor with legs ordered A,C,A',C'."""
		rho_AC = np.tensordot(self.psi, self.psi.conj(), axes=[[1,2],[1,2]])
		if join_AC:
			dAC = self.dim[0] * self.dim[3]
			return rho_AC.reshape(dAC, dAC)
		else: return rho_AC


	def entropy(self, some_other_Renyi_index=None):
		"""Return the Renyi entropy."""
		if some_other_Renyi_index is not None:
			return compute_entropy(self.s**2, Renyi=some_other_Renyi_index)
		else: return self.S


	def get_Zgradient(self):
		"""Compute the 1st derivative with respect to X.
	Return a 2-tensor gradZ, such that
	  delta Z[U]  =  gradZ . delta U  +  O(delta U^2).
	Here . means sum over elementwise multiplication, no conjugation, etc."""
		gradZ_1 = np.tensordot(self.cache_psi_2am1.conj(), self.cache_psi_rshp_ex, axes=[[0,2],[0,2]])
		return self.alpha * (gradZ_1 - gradZ_1.conj().transpose())


	def get_Sgradient(self):
		"""Compute the 1st derivative with respect to X.
	Return a 2-tensor gradS, such that
	  delta S[U]  =  gradS . delta U  +  O(delta U^2).
	Here . means sum over elementwise multiplication, no conjugation, etc."""
		return self.cache_gradS


	def get_gradient_conj_vec(self):
		"""Return the gradient G = df/dX*, in vector form.  As such, delta S/Z = G* . delta X."""
		if 'S' in self.func: return self.X_mat_to_vec(self.cache_gradS.conj())
		elif self.func == 'Z': return self.X_mat_to_vec(self.get_Zgradient().conj())


	def get_F_dF(self):
		"""Returns the value and its gradient* (in vectorized form)."""
		if 'S' in self.func: return self.S, self.X_mat_to_vec(self.cache_gradS.conj())
		elif self.func == 'Z': return self.Z, self.X_mat_to_vec(self.get_Zgradient().conj())


	################################################################################
	##	Internals
	def svd_psi(self, sv_trunc=1e-15):
		"""Perform SVD on psi.

	sv_trunc: SVD cutoff; values below this are thrown away.

	Makes attributes chi, s, Z, S, psiABL, psiBRC."""
		dA, dBL, dBR, dC = self.dim
		dL = dA * dBL
		dR = dBR * dC
		U, s, V = np.linalg.svd(self.psi.reshape(dL, dR), full_matrices=False, compute_uv=True)
		select = s > sv_trunc
		self.s = s[select]
		self.s /= np.linalg.norm(self.s)		# normalize s
		self.chi = chi = len(self.s)
		if chi < min(dL, dR):
			if self.alpha < 1: print("Renyi_Hessian.svd_psi():  psi appears to be not full rank (shape = {}, chi = {}).  Shit is really going to hit the fan now.".format(self.dim, chi))
		self.psiABL = U[:, select].reshape(dA, dBL, chi)
		self.psiBRC = V[select].reshape(chi, dBR, dC)
	##	Compute Z and S
		self.Z = np.sum(s ** (2*self.alpha))
		self.S = compute_entropy(s**2, Renyi=self.alpha)


	def compute_cache_objs(self, prep_Hessian=True):
		"""
	Cached objects:
		cache_gradS:
		cache_psi_2am1:  psi but with s^{2a-1} in the middle, shaped-(A,B,C)
		cache_psi_2slogs:  psi but with -2s log(s) in the middle, shaped-(A,B,C)
		cache_psi_Sweight:  psi but with (s^{2a-1} - s) * a / (1-a) / Z in the middle, shaped-(B,AC)
		cache_StermO_factor
		cache_StermM_factor
		cache_p_am1
"""
		alpha = self.alpha
		chi = self.chi
		s = self.s
		dA, dBL, dBR, dC = self.dim; dB = self.dimB
		psiABL = self.psiABL		# A,BL,s
		psiBRC = self.psiBRC		# s,BR,C
		Z = self.Z
	##	Compute!
#TODO, get rid of cache_psi_rshp_ex
		self.cache_psi_rshp = self.psi.transpose(1,2,0,3).reshape(dB,dA*dC)
		self.cache_psi_rshp_ex = self.psi.reshape(dA,dB,dC)
	##	Store chi*chi diagonal operators (as 2-array)
		assert np.all(s > 1e-14)
		if prep_Hessian:
			tab_sL, tab_sR = np.meshgrid(s, s)
			tab_pL = tab_sL**2; tab_pR = tab_sR**2
			tab_p_diff = tab_pL - tab_pR
			tab_p_avg = (tab_pL + tab_pR) / 2
			tab_problems = (np.abs(tab_p_diff) / tab_p_avg < 1e-8).astype(int)		# places where pL ~ pR
			#tab_problems = (np.abs(tab_p_diff) < 1e-8).astype(int)		# old code, not as good
		##	tab_diff_1 is like pL-pR, except with small numbers replaced by 1
			tab_diff_1 = np.choose(tab_problems, [tab_p_diff, tab_problems])

	##	Code which prepares to compute the Hessian for Z
		if 'Z' in self.func:		# Z stuff
			psiL = psiABL * s**(2*alpha-1)		# numerically unstable for a < 1/2 ?
			self.cache_psi_2am1 = np.tensordot(psiL, psiBRC, axes=[[2],[0]]).reshape(dA,dB,dC)
		
		if 'Z' in self.func and prep_Hessian:
		##	*	sL sR (pL^{a-1} - pR^{a-1}) / (pL - pR)
			tab_diff_2am2 = tab_sL**(2*alpha-1) * tab_sR - tab_sL * tab_sR**(2*alpha-1)
			tab_1st_order = (alpha-1) * tab_p_avg**(alpha-2) * tab_sL * tab_sR
			#tab_1st_order = (alpha-1) * tab_p_avg**(alpha-2) * (tab_pL*tab_pR + (alpha-2)*(alpha-3) * tab_p_diff**2 / 24) / (tab_sL*tab_sR)
			self.cache_termO_factor = np.choose(tab_problems, [tab_diff_2am2/tab_diff_1, tab_1st_order])
		##	Break up (pL^a - pR^a) / (pL - pR) into three pieces:
		##	*	pL pR (pL^{a-2} - pR^{a-2}) / (pL - pR)
			tab_diff_2am4 = tab_pL**(alpha-1) * tab_pR - tab_pL * tab_pR**(alpha-1)
			tab_1st_order = (alpha-2) * tab_p_avg**(alpha-3) * tab_pL * tab_pR
			#tab_1st_order = (alpha-2) * tab_p_avg**(alpha-3) * (tab_pL*tab_pR + (alpha-3)*(alpha-4) * tab_p_diff**2 / 24)
			self.cache_termM_factor = np.choose(tab_problems, [tab_diff_2am4/tab_diff_1, tab_1st_order])
		##	*	pL^{a-1} and pR^{a-1}
			self.cache_rhoL_am1 = np.tensordot(psiABL * s**(2*alpha-2), psiABL.conj(), axes=[[2],[2]])		# A,BL,A*,BL*
			psiR = psiBRC.transpose(1,2,0) * s**(2*alpha-2)
			self.cache_rhoR_am1 = np.tensordot(psiR, psiBRC.conj(), axes=[[2],[0]])		# BR,C,BR*,C*
			#TODO, don't build rho*_am1
			
	##	Code to compute dS/dX
		if 'S' in self.func:		# S stuff
			neg_ln_p = -2 * np.log(s)
		##	psi with the factor  (s^{2a-1} - s) a/(1-a)/Z  instead of s in the SVD
			if alpha == 1: dS_factor = s * neg_ln_p
			elif abs(alpha - 1) < 1e-8: dS_factor = alpha * ( neg_ln_p + (1-alpha)/2 * neg_ln_p**2 ) * s / Z		# 1st order in (alpha-1)
			else: dS_factor = alpha * (s**(2*alpha-1) - s)  / (1-alpha) / Z
			self.cache_psi_Sweight = np.tensordot(psiABL * dS_factor, psiBRC, axes=[[2],[0]]).reshape(dA,dB,dC).transpose(1,0,2).reshape(dB,dA*dC)		# B,AC
			half_gradS = np.dot(self.cache_psi_Sweight.conj(), self.cache_psi_rshp.transpose())		# B*,B
			self.cache_gradS = half_gradS - half_gradS.transpose().conj()		# dS/dX;  dS = np.sum(dX * dS/dX)
			self.cache_gradS.setflags(write=False)		# just in case...

	##	Code which prepares to compute the Hessian for S
	##	In the code below, the 'tab_**' variables are various ways to the write the same expressions in different limits (without the a/Z prefactor)
		if 'S' in self.func and prep_Hessian:
		##	*	sL sR (pL^{a-1} - pR^{a-1}) / (pL-pR) * a / (1-a) / Z
			if alpha == 1:
			##	when a = 1,  -sL sR (Log(pL) - Log(pR)) / (pL-pR) / Z
				tab_diff_2am2 = tab_sL * tab_sR * (np.log(tab_pR) - np.log(tab_pL)) / tab_diff_1
				tab_1st_order = -np.ones_like(tab_diff_2am2)
			else:
				tab_diff_2am2 = (tab_sL**(2*alpha-1) * tab_sR - tab_sL * tab_sR**(2*alpha-1)) / tab_diff_1 / (1-alpha)
				tab_1st_order = -tab_p_avg**(alpha-1)
			self.cache_StermO_factor = np.choose(tab_problems, [tab_diff_2am2, tab_1st_order]) * (alpha / Z)
		##	Break up [ (pL^a - pR^a) / (pL-pR) - 1 ] * a / (1-a) / Z into three pieces:
		##	*	[ pL pR (pL^{a-2} - pR^{a-2}) / (pL-pR) + 1 ] * a / (1-a) / Z
			if alpha == 1:
				tab_diff_2am4 = (tab_pL*np.log(tab_pR) - tab_pR*np.log(tab_pL)) / tab_diff_1
				tab_1st_order = np.log(tab_p_avg) - 1
			else:
				tab_diff_2am4 = ( tab_pL*tab_pR * (tab_pL**(alpha-2) - tab_pR**(alpha-2)) / tab_diff_1 + 1 ) / (1-alpha)
				tab_1st_order = ((alpha-2) * tab_p_avg**(alpha-1) + 1) / (1-alpha)
			self.cache_StermM_factor = np.choose(tab_problems, [tab_diff_2am4, tab_1st_order]) * (alpha / Z)
		##	*	[ pL^{a-1} - 1] * a/(1-a)/Z  and  [ pR^{a-1} - 1 ] * a/(1-a)/Z
			if alpha == 1: p_am1_factor = neg_ln_p
			else: p_am1_factor = (s**(2*alpha-2) - 1) / (1 - alpha)
			self.cache_p_am1 = p_am1_factor * (alpha / Z)

		#self.check_consistency() 		# disable for performance


	################################################################################
	##	Other useful functions
	def get_rhoL(self, combine_legs=False):
		"""Compute the density matrix on the left (A + BL).
	Returns a 4-tensor with legs (L,L*) or (A,BL,A*,BL), based on combine_legs.*
"""
		rhoL = np.tensordot(self.psi, self.psi.conj(), axes=[[2,3],[2,3]])
		if combine_legs: return rhoL.reshape(self.dim[0] * self.dim[1], self.dim[0] * self.dim[1])
		else: return rhoL


	def get_delta_rhoL(self, X, combine_legs=False):
		"""Return delta rhoL[U], given U ~ 1 + X.
	Returns a 4-tensor with legs (L,L*) or (A,BL,A*,BL), based on combine_legs.*
"""
		dA, dBL, dBR, dC = self.dim
		Xpsi = np.tensordot(X, self.cache_psi_rshp_ex, axes=[[1],[1]]).transpose(1,0,2).reshape(dA,dBL,dBR*dC)		# A,BL,R
		psi_Xpsi = np.tensordot(self.psi.conj().reshape(dA,dBL,dBR*dC), Xpsi, axes=[[2],[2]])	# A*,BL*,A,B
		delta_rhoL = psi_Xpsi.conj() + psi_Xpsi.transpose(2,3,0,1)
		if combine_legs: return delta_rhoL.reshape(dA*dBL, dA*dBL)
		else: return delta_rhoL


	def get_delta2_rhoL(self, X, combine_legs=False):
		"""Return delta^2 rhoL[U], given U ~ 1 + X + X^2/2."""
		dA, dBL, dBR, dC = self.dim
		Xpsi = np.tensordot(X, self.cache_psi_rshp_ex, axes=[[1],[1]])		# B,A,C
		Xpsi_LR = Xpsi.transpose(1,0,2).reshape(dA*dBL, dBR*dC)		# L,R
		XXpsi = np.tensordot(X, Xpsi, axes=[[1],[0]]).transpose(1,0,2).reshape(dA*dBL, dBR*dC)		# L,R
		psi_XXpsi = np.dot(XXpsi, self.psi.reshape(dA*dBL, dBR*dC).transpose().conj())	# L,L*
		delta2_rhoL = np.dot(Xpsi_LR, Xpsi_LR.transpose().conj()) + psi_XXpsi / 2 + psi_XXpsi.transpose().conj() / 2
		if combine_legs: return delta2_rhoL
		else: return delta2_rhoL.reshape(self.dim)


	def X_mat_to_vec(self, m):
		"""Given a (dB,dB)-shaped matrix, return its vectorized form (len shape[0]).  The inverse function is X_mat_to_vec()."""
		return m.reshape(-1)
	##	TODO: Look up np.triu_indices and np.triu


	def X_vec_to_mat(self, v):
		"""Given a vector, convert it back to a (dB,dB)-shaped matrix.  The inverse function is X_vec_to_mat()."""
		return v.reshape(self.dimB, self.dimB)


	def d2Z_matvec(self, X):
		"""Compute H.X, returns a 2-tensor.  See documentation for the class."""
		dA, dBL, dBR, dC = self.dim; dB = self.dimB
		psi_rshp = self.cache_psi_rshp_ex	# A,B,C
		psiABL = self.psiABL		# A,BL,s
		psiBRC = self.psiBRC		# s,BR,C
		psi_2am1 = self.cache_psi_2am1		# A,B,C
		Xpsi = np.tensordot(X, psi_rshp, axes=[[1],[1]]).reshape(dBL,dBR,dA,dC)		# BL,BR,A,C
		Xmn = np.tensordot(psiABL.conj(), np.tensordot(psiBRC.conj(), Xpsi, axes=[[1,2],[1,3]]), axes=[[0,1],[2,1]])		# (A,BL,sL)*(sR,BL,A) --> sL,sR
	##	The (delta rho)^2 terms with X's on the same side
		neck_O = (self.cache_termO_factor * Xmn).transpose().conj()
	##	The terms with X's on opposite sides 
		neck_M = self.cache_termM_factor * Xmn
		fork = np.tensordot(psiABL, np.tensordot(neck_M + neck_O, psiBRC, axes=[[1],[0]]), axes=[[2],[0]])		# (A,BL,sL)*(sL,BR,C) --> A,BL,BR,C
		fork += np.tensordot(self.cache_rhoL_am1, Xpsi, axes=[[2,3],[2,0]])
		fork += np.tensordot(Xpsi.transpose(2,0,1,3), self.cache_rhoR_am1, axes=[[2,3],[2,3]])
	##	The delta^2 rho terms with two X's on the same side
		fork -= np.tensordot(X, self.cache_psi_2am1, axes=[[1],[1]]).transpose(1,0,2).reshape(dA,dBL,dBR,dC) / 2		# T term
		ret = np.tensordot(fork.reshape(dA,dB,dC), psi_rshp.conj(), axes=[[0,2],[0,2]])
		ret -= np.tensordot(Xpsi.reshape(dB,dA,dC), self.cache_psi_2am1.conj(), axes=[[1,2],[0,2]]) / 2		# B term
	##	Antihermitianize
		ret -= ret.transpose().conj()
		return self.alpha * ret


	def d2S_from_d2Z_matvec(self, X):
		"""Compute H.X, returns a 2-tensor.  Not the most efficient, nor does it work near alpha = 1.  See documentation for the class."""
		alpha = self.alpha
		if abs(1 - alpha) < 1e-8: print("Renyi_Hessian.d2S_from_d2Z_matvec():  NONONO (alpha = {})".format(alpha))
		dZdX = self.get_Zgradient()
		dZ = np.sum(X * dZdX)
	##	The formula is 2 delta^2 S[X] = 2 delta^2 Z[X] / (1-a) / Z - (delta Z[X])^2 / (1-a) / Z^2
		return self.d2Z_matvec(X) / (self.Z * (1-alpha)) - dZdX.conj() * (dZ / (1-alpha) / self.Z**2)


	def d2S_matvec(self, X):
		"""Compute H.X, returns a 2-tensor.  See documentation for the class."""
	##	There are six terms to consider: T,B,L,M,R,O
		dA, dBL, dBR, dC = dim = self.dim; dB = self.dimB; chi = self.chi
		psi_rshp = self.cache_psi_rshp	# B,AC
		psiABL_rshp = self.psiABL.reshape(dA*dBL,chi)		# L,s
		psiBRC_rshp = self.psiBRC.reshape(chi,dBR*dC)		# s,R
		psi_Sweight = self.cache_psi_Sweight	# B,AC
		Xpsi_BAC = np.dot(X, psi_rshp)	# B,AC
#TODO reshape Xpsi
		Xpsi = Xpsi_BAC.reshape(dBL,dBR,dA,dC).transpose(2,0,1,3)	# A,BL,BR,C
		Xpsi_sR = np.tensordot(Xpsi.reshape(dA,dBL,dBR*dC), psiBRC_rshp.transpose().conj(), axes=[[2],[0]])	# A,BL,sR
		Xpsi_sL = np.dot(psiABL_rshp.transpose().conj(), Xpsi.reshape(dA*dBL,dBR*dC))	# sL,R
		Xmn = np.dot(psiABL_rshp.transpose().conj(), Xpsi_sR.reshape(dA*dBL,chi))		# sL,sR
		##	The (delta rho)^2 term with X's on the same side (O)
		neck_O = (self.cache_StermO_factor * Xmn).transpose().conj()
		##	The term with X's on opposite sides (M)
		neck_M = self.cache_StermM_factor * Xmn
		##	Combine the O and M terms
		fork_sR = np.dot(psiABL_rshp, neck_M + neck_O).reshape(dA,dBL,chi)	# A,BL,sR
		##	Add the R term
		fork_sR += Xpsi_sR * self.cache_p_am1
		fork = np.tensordot(fork_sR, psiBRC_rshp, axes=[[2],[0]]).reshape(dim)	# A,BL,BR,C
		##	Add the L term
		fork += np.dot(psiABL_rshp * self.cache_p_am1, Xpsi_sL).reshape(dim)
		##	Add the T term
		fork -= np.dot(X, psi_Sweight).reshape(dBL,dBR,dA,dC).transpose(2,0,1,3) / 2
		ret2 = np.dot(fork.transpose(1,2,0,3).reshape(dB,dA*dC), psi_rshp.transpose().conj())	# B,B*
		##	Add the B term
		ret2 -= np.dot(Xpsi_BAC, psi_Sweight.transpose().conj()) / 2
	##	Antihermitianize
		ret2 -= ret2.transpose().conj()
		dSdX = self.cache_gradS		# include the 1st order^2 term (a-1) (delta S)^2
		return ret2 + (self.alpha-1) * dSdX.conj() * np.dot(X.reshape(-1), dSdX.reshape(-1))


	def mock_d2Z_matvec(self, X):
		"""Temporary code to compute d2Z a / (1-a) / Z.  For testing purposes."""
		dA, dBL, dBR, dC = self.dim; dB = self.dimB
		psi_rshp = self.cache_psi_rshp_ex	# A,B,C
		psiABL = self.psiABL		# A,BL,s
		psiBRC = self.psiBRC		# s,BR,C
		psi_2am1 = self.cache_psi_2am1		# A,B,C
		Xpsi = np.tensordot(X, psi_rshp, axes=[[1],[1]]).reshape(dBL,dBR,dA,dC)		# BL,BR,A,C
		Xmn = np.tensordot(psiABL.conj(), np.tensordot(psiBRC.conj(), Xpsi, axes=[[1,2],[1,3]]), axes=[[0,1],[2,1]])		# (A,BL,sL)*(sR,BL,A) --> sL,sR
	##	The (delta rho)^2 terms with X's on the same side
		neck_O = (self.cache_termO_factor * Xmn).transpose().conj()
	##	The terms with X's on opposite sides 
		neck_M = (self.cache_termM_factor + 1) * Xmn
		fork = np.tensordot(psiABL, np.tensordot(neck_M + neck_O, psiBRC, axes=[[1],[0]]), axes=[[2],[0]])		# (A,BL,sL)*(sL,BR,C) --> A,BL,BR,C
		fork += np.tensordot(self.cache_rhoL_am1, Xpsi, axes=[[2,3],[2,0]]) - Xpsi.transpose(2,0,1,3)	#L
		fork += np.tensordot(Xpsi.transpose(2,0,1,3), self.cache_rhoR_am1, axes=[[2,3],[2,3]]) - Xpsi.transpose(2,0,1,3)		#R
	##	The delta^2 rho terms with two X's on the same side
		fork -= np.tensordot(X, self.cache_psi_2am1 - psi_rshp, axes=[[1],[1]]).transpose(1,0,2).reshape(dA,dBL,dBR,dC) / 2	#T
		ret = np.tensordot(fork.reshape(dA,dB,dC), psi_rshp.conj(), axes=[[0,2],[0,2]])
		ret -= np.tensordot(Xpsi.reshape(dB,dA,dC), (self.cache_psi_2am1 - self.cache_psi_rshp_ex).conj(), axes=[[1,2],[0,2]]) / 2	#B
	##	Antihermitianize
		ret -= ret.transpose().conj()
		return (self.alpha * ret) / (self.Z * (1-self.alpha))


	def d2S_Hessian(self):
		raise NotImplementedError


	def d2Z_Hessian(self):
		raise NotImplementedError




########################################################################################################################
########################################################################################################################
##	Checks and Helpers

def psi_ABBC_entropy(psi, alpha):
	dL = psi.shape[0] * psi.shape[1]
	dR = psi.shape[2] * psi.shape[3]
	s = np.linalg.svd(psi.reshape(dL, dR), compute_uv=False)
	return compute_entropy(s**2, Renyi=alpha)


def psi_ABBC_Zalpha(psi, alpha):
	dL = psi.shape[0] * psi.shape[1]
	dR = psi.shape[2] * psi.shape[3]
	s = np.linalg.svd(psi.reshape(dL, dR), compute_uv=False)
	return np.sum(s**(2*alpha))
	

def apply_U_to_psiABBC(U, psi):
	"""Returns U |psi>.
	U is shaped (|B|,|B|), while psi is shaped (|A|,|BL|,|BR|,|C|)."""
	dA, dBL, dBR, dC = psi.shape
	return np.tensordot(U, psi.reshape(dA, dBL*dBR, dC), axes=[[1],[1]]).transpose(1,0,2).reshape(dA, dBL, dBR, dC)


def check_Renyi_Hessian_dS(dim=(2,2,2,2), alpha=1., eps=1e-3, tol_2nd_order_eps=0.5, tol_3rd_order_eps=0.8, seed=None):
	"""Tests the check_Renyi_Hessian_dS class, by comparing differences from small X = log(U)."""
	assert isinstance(dim, tuple) and len(dim) == 4
	RS = get_me_my_RandomState(seed)
	dimB = dim[1] * dim[2]
	psi = RS.standard_normal(size = dim) + 1j * RS.standard_normal(size = dim)
	psi /= np.linalg.norm(psi)
	RH0 = Renyi_Hessian(alpha, psi, func='ZS')
	RH0.check_consistency()
	S0 = RH0.entropy()
	Z0 = RH0.Z

	dS = RH0.get_Sgradient()
	dZ = RH0.get_Zgradient()
	dZ_from_dS = (1-alpha) * Z0 * dS
	if np.any(np.abs(dZ - dZ_from_dS) > 1e-13): raise RuntimeError(dZ, dZ_from_dS, dZ - dZ_from_dS)
	#print(zero_if_close(dS), '= dS\n')
	dX = np.zeros((dimB,dimB), dtype=np.complex);
	Nchoose2 = dimB * (dimB-1) // 2		# number of OD elements
	tableS = np.zeros( (Nchoose2*(Nchoose2+1)//2, 5), dtype=np.float )		# store the data for S
	tableZ = np.zeros( (Nchoose2*(Nchoose2+1)//2, 5), dtype=np.float )		# store the data for Z
	row = 0
	iOD = [ (i1,i2) for i1 in range(dimB) for i2 in range(i1) ]		# 0 <= i2 < i1 < dimB
	for idx12 in range(Nchoose2):
		i1, i2 = iOD[idx12]
		for idx34 in range(idx12 + 1):
			i3, i4 = iOD[idx34]
			dX[i1, i2] = eps * (0.7j+0.1)
			dX[i2, i1] = eps * (0.7j-0.1)
			dX[i3, i4] += eps * (0.1j-0.7)
			dX[i4, i3] += eps * (0.1j+0.7)
			Upsi = apply_U_to_psiABBC(sp.linalg.expm(dX), psi)
			S = psi_ABBC_entropy(Upsi, alpha)
			Z = psi_ABBC_Zalpha(Upsi, alpha)
			S1 = S0 + np.sum(dS*dX)
			Z1 = Z0 + np.sum(dZ*dX)
#			print("bb1", np.max(np.abs(RH0.d2S_matvec(dX) - RH0.d2S_from_d2Z_matvec(dX))))
#			print("bb2", np.max(np.abs(RH0.d2S_matvec(dX) - RH0.mock_d2Z_matvec(dX))))
#			print("bb3", np.max(np.abs(RH0.d2S_from_d2Z_matvec(dX) - RH0.d2S_matvec(dX))))
#			print(RH0.d2S_matvec(dX), "\n")
#			print(RH0.mock_d2Z_matvec(dX), "\n")
			d2S = np.sum(dX.conj() * RH0.d2S_matvec(dX)) / 2
			d2Z = np.sum(dX.conj() * RH0.d2Z_matvec(dX)) / 2
			S2 = S1 + d2S
			Z2 = Z1 + d2Z
			tableS[row] = np.array([S1.real-S0, S1.imag, S-S0, S-S1.real, S-S2.real])
			tableZ[row] = np.array([Z2.real-Z0, Z2.imag, Z-Z0, Z-Z1.real, Z-Z2.real])
		##	Clean up
			dX[i1, i2] = dX[i2, i1] = dX[i3, i4] = dX[i4, i3] = 0.
			row += 1
	if 0:
		print("S-S0 series (real, imag),    S-S0 err,    S-S1 err,     S-S2 err")
		print(tableS, "= table of data\n")
		#print(tableZ, "= table of data\n")
	max_S1_err = np.max(np.abs(tableS[:,3]))
	max_S2_err = np.max(np.abs(tableS[:,4]))
	max_Z1_err = np.max(np.abs(tableZ[:,3]))
	max_Z2_err = np.max(np.abs(tableZ[:,4]))
	print("check_Renyi_Hessian_dS( alpha = {}, dim = {}, eps = {} ):".format(alpha, dim, eps))
	print("\tMax S1/Z1 error = {} / {}".format(max_S1_err, max_Z1_err))
	print("\tMax S2/Z2 error = {} / {}".format(max_S2_err, max_Z2_err))
	if np.any(np.abs(tableS[:,1]) > 1e-14): raise RuntimeError
	if np.any(np.isnan(tableS)): raise RuntimeError
	if max_S1_err > tol_2nd_order_eps * eps**2: raise RuntimeError
	if max_S2_err > tol_3rd_order_eps * eps**3: raise RuntimeError
	if np.any(np.abs(tableZ[:,1]) > 1e-14): raise RuntimeError
	if max_Z1_err > tol_2nd_order_eps * eps**2: raise RuntimeError
	if max_Z2_err > tol_3rd_order_eps * eps**3: raise RuntimeError


def check_Renyi2_Hessian_dS(dim=(2,2,2,2), seed=None):
	"""Tests the check_Renyi_Hessian_dS class at alpha=2."""
	print("check_Renyi2_Hessian_dS(dim = {}, seed = {})".format(dim, seed))
	assert isinstance(dim, tuple) and len(dim) == 4
	RS = get_me_my_RandomState(seed)
	dimB = dim[1] * dim[2]
	psi = RS.standard_normal(size = dim) + 1j * RS.standard_normal(size = dim)
	psi /= np.linalg.norm(psi)
	RH = Renyi_Hessian(2., psi, func='Z')
	RH.check_consistency()

	S = RH.entropy()
	rhoL = RH.get_rhoL(combine_legs=True)
	for i in range(20):
		dX = RS.standard_normal(size = (dimB, dimB)) * 1e-3
		dX -= dX.transpose().conj()
		drhoL = RH.get_delta_rhoL(dX, combine_legs=True)
		d2rhoL = RH.get_delta2_rhoL(dX, combine_legs=True)
		dZ = complex( 2 * np.tensordot(rhoL, drhoL, axes=[[0,1],[1,0]]) )
		d2Z = complex( 2 * np.tensordot(rhoL, d2rhoL, axes=[[0,1],[1,0]]) + np.tensordot(drhoL, drhoL, axes=[[0,1],[1,0]]) )

		Upsi = apply_U_to_psiABBC(sp.linalg.expm(dX), psi)
		Z = psi_ABBC_Zalpha(Upsi, 2)
		print("\t*  Z - Z0 = {},  dZ = {},  d2Z = {},  Z - (Z0 + dZ + d2Z) = {}".format(Z-RH.Z, dZ.real, d2Z.real, Z-RH.Z-dZ.real-d2Z.real))
	##	Compare to get_Zgradient() and d2Z_matvec()
		RH_dZ = np.sum(RH.get_Zgradient() * dX)
		RH_d2Z = np.sum(dX.conj() * RH.d2Z_matvec(dX)) / 2
		#print("\tdZ - 2 Tr[rho drho] = {}".format(dZ - RH_dZ))
		print("\t   d2Z - Tr[rho d^2rho] - Tr[drho drho]/2 = {}".format(d2Z - RH_d2Z))
		assert abs(dZ - RH_dZ) < 1e-14
		assert abs(d2Z - RH_d2Z) < 1e-15



########################################################################################################################
########################################################################################################################
##	Temporary stuff that should eventually be moved/deleted

def get_me_my_RandomState(seed):
	"""Given a seed, returns a numpy.random.RandomState object.

	If seed is an integer, use it to generate the RandomState.
	If seed is a RandomState, then simply return it.
	Otherwise, let numpy pick a 'random' seed.
"""
	if isinstance(seed, int): RS = np.random.RandomState(5)
	elif isinstance(seed, np.random.RandomState): RS = seed
	else: RS = np.random.RandomState()
	return RS


def ldistengle_Renyi2(psi4, verbose=1):
	pass


def Werner_state_ABBC(f):
	"""Constructs a Werner state.  Return a purification in the form of a (2,2,2,2)-array

Parameters:
	f: paramater between 0 and 1.
		f = 0 is the mixed triplet state, f = 1/4 is the maximally mixed state, and f = 1 is a pure singlet state.

Returns a four-partite pure state (A,BL,BR,C), where rho[AC] is the Werner state, and (BL,BR) canonically purifies (A,C).
"""
	assert 0. <= f and f <= 1.
	amp_triplt = np.sqrt((1. - f) / 3.)
	psi = np.zeros((2,2,2,2), dtype=np.float)
	psi[0,0,1,1] = psi[1,1,0,0] = amp_triplt/2 + np.sqrt(f)/2
	psi[0,1,0,1] = psi[1,0,1,0] = amp_triplt/2 - np.sqrt(f)/2
	psi[0,0,0,0] = psi[1,1,1,1] = amp_triplt
	return psi


def rand_antiHermMx(dim, seed=None):
	RS = get_me_my_RandomState(seed)
	A =  RS.standard_normal(size = (dim,dim)) + 1j * RS.standard_normal(size = (dim,dim))
	A -= A.transpose().conj()
	return A


def rand_state_with_sv(s, dimL, dimR, seed=None):
	s = np.array(s)
	l = len(s)
	assert dimL >= l and dimR >= l
	s = s / np.linalg.norm(s)
	RS = get_me_my_RandomState(seed)
	psi = np.zeros((dimL, dimR), dtype=np.complex)
	psi[:l, :l] = np.diag(s)
	psi = np.dot(np.dot(random_unitary(dimL, RS), psi), random_unitary(dimR, RS))
	return psi
	


########################################################################################################################
########################################################################################################################
def run_renyi_hessian_main():
	print("==================== Renyi Hessian ====================")
	np.set_printoptions(linewidth=2000, precision=4, threshold=10000, suppress=False)

	RS = np.random.RandomState(5)
	#check_Renyi_Hessian_dS(dim=(3,2,2,2), alpha=0.7, eps=1e-2, seed=4)
	if 1:		# Tests
		check_Renyi_Hessian_dS(dim=(4,2,2,3), alpha=1., eps=1e-2, seed=4)
		#check_Renyi_Hessian_dS(dim=(2,2,2,2), alpha=0.5, eps=1e-2, seed=6)
		check_Renyi_Hessian_dS(dim=(4,4,2,2), alpha=0.5, eps=1e-2, seed=6)
		check_Renyi_Hessian_dS(dim=(3,2,5,5), alpha=1.3, eps=1e-2, seed=3)
		#check_Renyi2_Hessian_dS(dim=(2,3,4,5), seed=4)
		print()
	if 1:
		psi = RS.standard_normal(size=(5,2,3,4))
		psi /= np.linalg.norm(psi)
	if 0:
		psi = Werner_state_ABBC(0.28)
	alpha = 0.5
	RH = Renyi_Hessian(alpha, psi, func='ZS')
	dX = rand_antiHermMx(RH.dimB, RS)
	dY = rand_antiHermMx(RH.dimB, RS)
	#RH.matvec(np.zeros(RH.shape[1]))
	HdX = RH.d2Z_matvec(dX)
	HdY = RH.d2Z_matvec(dY)
	#print(HdX)
	print(zero_if_close( np.array([[np.sum(HdX*dX.conj()), np.sum(HdX*dY.conj())], [np.sum(HdY*dX.conj()), np.sum(HdY*dY.conj())]]) , tol=1e-13))
	if 0:
		dZ = RH.get_Zgradient()
		dS = RH.get_Sgradient()
		#print(zero_if_close(dS), "= dS")
		print(zero_if_close(dZ), "= dZ")
		dX = np.zeros((4,4)); dX[1,2] = 1e-3
		dX -= dX.transpose().conj()
		U = sp.linalg.expm(dX)
		#print(dX)
		#RH2 = Renyi_Hessian(alpha, apply_U_to_psiABBC(U, psi))
		#print("{}  +  {}  =  {},   err ~ {}".format( RH.entropy(), np.sum(dS*dX), RH.entropy()+np.sum(dS*dX), RH.entropy()+np.sum(dS*dX)-RH2.entropy() ))

if __name__ == "__main__":
	run_renyi_hessian_main()
