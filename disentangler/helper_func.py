from __future__ import print_function
import numpy as np
import scipy as sp
import time


##################################################
##	General array processing
def zero_if_close(a, tol=1e-15):
	"""Take an np.ndarry and set small elements to zero."""
	if a.dtype == np.complex128 or a.dtype == np.complex64:
		cr = np.array(np.abs(a.real) < tol, int)
		ci = np.array(np.abs(a.imag) < tol, int)
		ar = np.choose(cr, [a.real, np.zeros(a.shape)])
		ai = np.choose(ci, [a.imag, np.zeros(a.shape)])
		return ar + 1j * ai
	else:
		c = np.array(np.abs(a) < tol, int)
		return np.choose(c, [a, np.zeros_like(a)])


def zero_real_if_close(a, tol=1e-15):
	"""Take an np.ndarry and set small elements to zero.  Make it real if all its imaginary components vanishes."""
	zic_a = zero_if_close(a, tol=tol)
	if np.count_nonzero(zic_a.imag) > 0: return zic_a
	return zic_a.real


def assert_int_array(ar, shape, textname):
	"""Check that an array is integer and has the correct shape."""
	if not isinstance(ar, np.ndarray): raise AssertionError("{} ({}) not np.ndarray.".format(textname, type(ar)))
	if ar.dtype != np.int: raise AssertionError("{} dtype ({}) not np.int.".format(textname, ar.dtype))
	if ar.shape != shape: raise AssertionError("{} shape {} not {}.".format(textname, ar.shape, shape))


def matvec_to_array(L):
	"""Given an object with a matvec (taking an np.array), return its corresponding dense matrix."""
	#assert isinstance(L, sp.linalg.LinearOperator)
	##TODO
	raise NotImplementedError


def basis_vector(dim, k, dtype=np.int):
	v = np.zeros(dim, dtype=dtype)
	v[k] = 1
	return v 



##################################################
##	Discrete mathematics
def gcd(a, b):
	"""Greatest common divisor.  Return 0 if both a,b are zero, otherwise always return a non-negative number."""
	a = abs(a)
	b = abs(b)
	while b > 0:
		a, b = b, a % b		# after this, a > b
	return a


def lcm(a, b):
	if a * b == 0: return 0
	return abs(a * b) // gcd(a, b)


#@profile
def array_gcd(a):
	"""Given an array of integers, return their greatest common divisor.
	Raise a ValueError if a is 0-length array;
	Return 0 if all the values are zero;
	Otherwise, always return a positive integer."""
	if len(a) == 0: raise ValueError
	a = np.unique(np.abs(a))
	if len(a) == 1: return a[0]
	if a[0] == 0: a = a[1:]
	while len(a) > 1:
	## a[0] is guaranteed to be the smallest
		a[1:] %= a[0]
		a = np.unique(a)
		if a[0] == 0: a = a[1:]
	return a[0]


def perm_sign(perm):
	"""Given a permutation (of 0...n-1), return +1/-1 for the sign of the permutation."""
	p = np.array(perm)	# makes a copy
	n = len(p)
	sign = 1
	pl = np.argsort(p)
	for i in range(n - 1):
		x = p[i]		# the thing sitting at i
		if x != i:
			j = pl[i]	# where i is at the moment
		##	Swap
			p[j] = x
			pl[x] = j
		#	p[i] = i; pl[i] = i		# this isn't necessary, because we will not read element [i] again.
			sign = -sign
	return sign


def Jacobi_symbol(a, b):
	"""The Jacobi symbol is a generalization of the Legendre symbol.  Returns +1, -1, or 0.
	a is an integer, b is an odd positive integer."""
	assert isinstance(a, int) and isinstance(b, int)
	assert b > 0 and b % 2
	if gcd(a, b) > 1: return 0
	even_table = [0,1,0,-1,0,-1,0,1]
	J = 1
	if a < 0:
		raise NotImplementedError
	while a > 1:
		if b == 1: break
		a = a % b
		while a % 4 == 0: a = a // 4
		if a % 2 == 0:
			a //= 2
			J *= even_table[b % 8]
		if b > a:
			if a % 4 == 3 and b % 4 == 3: J = -J		# Quadratic reciprocity
			a,b = b,a
		##	At this point, a > b
	return J



def Smith_normal_form(Mx, check_results=True, print_results=False):
	"""Put a matrix in Smith Normal form.
	D = S . Mx . T is a diagonal matrix
	Returns a dictionary with keys 'D', 'S', 'Sinv', 'T', 'Tinv'
"""
	if not isinstance(Mx, np.ndarray): raise TypeError 
	assert Mx.ndim == 2
	assert Mx.dtype == np.int
	m, n = Mx.shape
	##	Helpers
	def min_positive(X, shift_ij):		# return the indices (i,j) for the smallest positive element
		max_p1 = np.abs(np.max(X)) + 1
		X_p = np.where(X > 0, X, np.ones_like(X) * max_p1)
		i,j = np.unravel_index(np.argmin(X_p), X.shape)
		return i + shift_ij, j + shift_ij
	def swaprows(r1, r2):
		tmp = A[r1].copy(); A[r1] = A[r2]; A[r2] = tmp
		tmp = S[r1].copy(); S[r1] = S[r2]; S[r2] = tmp
		tmp = Sinv[:, r1].copy(); Sinv[:, r1] = Sinv[:, r2]; Sinv[:, r2] = tmp
	def swapcols(c1, c2):
		tmp = A[:, c1].copy(); A[:, c1] = A[:, c2]; A[:, c2] = tmp
		tmp = T[:, c1].copy(); T[:, c1] = T[:, c2]; T[:, c2] = tmp
		tmp = Tinv[c1].copy(); Tinv[c1] = Tinv[c2]; Tinv[c2] = tmp
	def negaterow(r1):
		A[r1] *= -1; S[r1] *= -1; Sinv[:, r1] *= -1
	def addrowto(r1, factor, r2):
		A[r2] += factor*A[r1]; S[r2] += factor*S[r1]; Sinv[:,r1] -= factor*Sinv[:,r2]
	def addcolto(c1, factor, c2):
		A[:, c2] += factor*A[:, c1]; T[:, c2] += factor*T[:, c1]; Tinv[c1] -= factor*Tinv[c2]
	##	The algorithm
	A = Mx.copy()
	S = np.eye(m, dtype=np.int); Sinv = np.eye(m, dtype=np.int)
	T = np.eye(n, dtype=np.int); Tinv = np.eye(n, dtype=np.int)
	for t in range(min(m, n)):
		#print(joinstr(['t = {},  A = '.format(t), A]))
		if check_results:
			assert np.all(A[:t, t:] == 0) and np.all(A[t:, :t] == 0)
			assert np.all(np.dot(np.dot(S, Mx), T) == A)
		##	Move the smallest element to the pivot
		if np.all(A[t:, t:] == 0): break
		min_i, min_j = min_positive(np.abs(A[t:, t:]), t)
		if min_i != t: swaprows(min_i, t)
		if min_j != t: swapcols(min_j, t)
		if A[t, t] < 0: negaterow(t)
		while True:
			#print joinstr(['        A = '.format(t), A])
			piv = A[t, t]
			if np.any(A[t+1:, t] % piv):
				pariah = min_positive(A[t:, t:t+1] % piv, t)
				addrowto(t, -(A[pariah] // piv), pariah[0])
				swaprows(t, pariah[0])		# move to the pivot point
				continue
			if np.any(A[t, t+1:] % piv):
				pariah = min_positive(A[t:t+1, t:] % piv, t)
				addcolto(t, -(A[pariah] // piv), pariah[1])
				swapcols(t, pariah[1])		# move to the pivot point
				continue
			if np.any(A[t:, t:] % piv):
				pariah = min_positive(A[t:, t:] % piv, t)
				addrowto(t, -(A[pariah[0], t] // piv), pariah[0])
				addrowto(pariah[0], 1, t)		# move the pariah to the t'th row.
				continue
			break		# at this point A[t:, t:] divides A[t, t]
		##	Eliminate elements in the t'th row and column
		for r in range(t+1, m): addrowto(t, -(A[r, t] // piv), r)
		for c in range(t+1, n): addcolto(t, -(A[t, c] // piv), c)
	##	Checks
	if check_results:
		assert np.all(np.dot(np.dot(S, Mx), T) == A)		## Check
		assert np.all(np.dot(S, Sinv) == np.eye(m))
		assert np.all(np.dot(T, Tinv) == np.eye(n))
	if print_results:
		from stringtools import joinstr
		print(joinstr([ 'D = ', A, ' =  S Mx T = ', S, Mx, T ]))
		for r in range(min(m, n)):
			if A[r, r] > 1: print('  {} {} = {} . Mx'.format(A[r, r], Tinv[r], S[r]))
		print()
	return {'D':A, 'S':S, 'Sinv':Sinv, 'T':T, 'Tinv':Tinv}



##################################################
##	Statistics
def compute_entropy(prob, Renyi=1):
	"""Return the Renyi entropy for a probability distribution.
	If Renyi is a list, then return a list.
	Parameters:
		prob:  a 1-array of non-negative numbers.  (Negative numbers will be set to zero.)
		Renyi:  the Renyi index.  For example, 0 -> rank, 1 -> Shannon, inf -> -log(largest p)."""
	if np.isscalar(Renyi): ri = np.array([Renyi])
	else: ri = np.array(Renyi)
	assert np.all(ri >= 0)
	if np.any(prob < -1e-15): print("Warning!  compute_entropy():  Probabilities have negative values.  Sum of negative values = {}".format(np.sum(np.choose(prob<0, [np.zeros_like(prob), prob]))))
	prob = np.choose(prob < 0, [prob, np.zeros_like(prob)])
	prob /= np.sum(prob)
	S = np.zeros(ri.shape, dtype=float)
	for i,a in enumerate(ri):
		if a == np.inf: S[i] = -np.log(np.max(prob))
		elif a == 0:
			S[i] = np.log(np.count_nonzero(prob))
		elif np.abs(a - 1.) < 1e-10:
			##	Use series S[1+x] = -l1 + (x/2)[l1^2 - l2] - (x^2/6)[2 l1^3 - 3 l1 l2 + l3] + ...
			log_p = np.log(np.choose(prob == 0, [prob, np.ones(prob.shape)]))
			S1 = -np.sum(prob * log_p)
			S[i] = S1 + (a-1)/2. * (S1**2 - np.sum(prob * log_p**2))
		else:
			S[i] = np.log(np.sum(np.power(prob, a))) / (1 - a)
	if np.isscalar(Renyi): return S[0]
	return S
	


#TODO, random_orthogonal
def random_orthogonal(size, RS):
	raise NotImplementedError

def random_unitary(size, RS):
	"""Return a size x size random unitary matrix.
	RS is a numpy.random.RandomState, or an integer seed for such a RandomState.
			
	The internet says it's evenly distributed under the Haar measure.
	   """
	if isinstance(RS, int):
		from numpy.random import RandomState
		RS = RandomState(RS)    # use RS as a seed
	Areal = RS.normal(loc=0, scale=1, size=size**2).reshape((size, size))
	Aimag = RS.normal(loc=0, scale=1, size=size**2).reshape((size, size))
	Q, R = sp.linalg.qr(Areal + 1j * Aimag, overwrite_a = True, check_finite = True)
	for i in range(size):
		if R[i,i].real < 0:
			Q[:, i] *= -1
	return Q



##################################################
##	Iterators
def integers_0pn():
	"""Yields a sequence of integers 0,1,-1,2,-2, etc."""
	yield 0
	n = 1
	while True:
		yield n; yield -n
		n += 1
def integers_0np():
	"""Yields a sequence of integers 0,-1,1,-2,2, etc."""
	yield 0
	n = 1
	while True:
		yield -n; yield n
		n += 1
def integers_np():
	"""Yields a sequence of integers -1,1,-2,2, etc."""
	n = 1
	while True:
		yield -n; yield n
		n += 1


####################################################################################################
####################################################################################################
##	q-deformed SU(2) stuff

r"""Properties of Wigner 6j-symbols:

Symmetries:
	\{a, b, c; d, e, f\} = \{b, a, c; e, d, f\}    (swap 1st & 2nd columns)
	                     = \{c, b, c; f, e, d\}    (swap 1st & 3rd columns)
	                     = \{d, e, c; a, b, d\}    (swap rows in 1st & 2nd columns)
	\{a, b, c; k/2-d, k/2-e, k/2-f\} = (-1)^{k+a+b+c} \{a, b, c; d, e, f\}
Corollaries:
	\{k/2-a, k/2-b, c; k/2-d, k/2-e, f\} = (-1)^{k+a+b+d+e} \{a, b, c; d, e, f\}  (this follows from above)

Limiting cases:
	\{a, b, c; b, a, 0\} = (-1)^{a+b+c} / \sqrt{d_a d_b}
	\{a, b, c; k/2-b, k/2-a, k/2\} = (-1)^k / \sqrt{d_a d_b}
"""

def qSU2_fusion_abc(k, a, b, c):
	"""Return True if (a,b,c) can fuse to vacuum.
	Note that (a,b,c) are integers from [0,k] inclusive."""
	if (a + b + c) % 2: return False
	if a + b < c: return False
	if b + c < a: return False
	if c + a < b: return False
	if a + b + c > 2 * k: return False
	return True


def qSU2_Wigner_6j_array(k, timer=False):
	"""Return the q-deformed Wigner-6j symbols, as a 6-array.
	
	k: the `level', where q = exp(2 pi i / (k+2)).
		The array returned has shape (k+1, k+1, k+1, k+1, k+1, k+1)."""
	return qSU2(k).Wigner6j_array(timer=timer)
#	t0 = time.time()
#	assert type(k) == int
#	assert k >= 0
#	kp1 = k + 1
#	twok = 2 * k
#
#	##	Compute the quantum dimensions, and the [n]!'s
#	QD = np.sin( np.arange(1, k+2) * np.pi / (k+2) )
#	QD /= QD[0]		# QD[n-1] = [n]_q
#	qf = np.zeros(k+3, np.float)
#	qf[0] = 1
#	for n in range(1, k+2): qf[n] = qf[n-1] * QD[n-1]
#	q_fact_2 = np.zeros(2 * len(qf) - 1, np.float)
#	q_fact_2[0::2] = qf		# q_fact_2[2n] = [n]!
#
#	qf_zp1_sign = q_fact_2[2 : 2*k+3] * (1 - np.mod(np.arange(2*k+1),4))		# qf_zp1_sign[2z] = qf[z+1] * (-1)**z
#	Delta = np.zeros((kp1, kp1, kp1), np.float)
#	Fusion_abc = np.zeros((kp1, kp1, kp1), np.int)
#	for a in range(kp1):
#		for b in range(kp1):
#			for c in range(max(a-b, b-a), min(a+b, 2*k-a-b) + 1, 2):
#				Delta[a,b,c] = q_fact_2[b+c-a] * q_fact_2[c+a-b] * q_fact_2[a+b-c] / q_fact_2[a+b+c+2]
#				Fusion_abc[a,b,c] = 1
#	Delta = np.sqrt(Delta)
#
#	W6j = np.zeros((kp1, kp1, kp1, kp1, kp1, kp1), np.float)
#	Fusion_abcd = np.tensordot(Fusion_abc, Fusion_abc, axes=[[2],[0]])
#	j1245list = np.array(np.nonzero(Fusion_abcd)).transpose().tolist()
#	for J1245 in j1245list:
#		j1 = J1245[0]; j2 = J1245[1]; j4 = J1245[2]; j5 = J1245[3]
#		j14 = j1+j4; j25 = j2+j5
#		if j14 > j25: continue
#		if j1 > j4 or j2 > j5: continue
#		##	Only compute terms with j1 <= j4, j2 <= j5, and j14 <= j25
#		j12 = j1+j2; j45 = j4+j5; j15 = j1+j5; j24 = j2+j4
#		j1245 = j15+j24
#		j3_min = max(j1 - j2, j2 - j1, j4 - j5, j5 - j4)
#		j3_max = min(j12, twok - j12, j45, twok - j45)
#		j6_min = max(j1 - j5, j5 - j1, j4 - j2, j2 - j4)
#		j6_max = min(j15, twok - j15, j24, twok - j24)
#		Deltas = np.outer(Delta[j1, j2] * Delta[j4, j5], Delta[j1, j5] * Delta[j4, j2])
#		for j3 in np.arange(j3_min, j3_max + 1, 2):
#			j123 = j12+j3; j453 = j45+j3
#			for j6 in range(j6_min, j6_max + 1, 2):
#				j156 = j15+j6; j426 = j24+j6;
#				j2356 = j25+j3+j6; j1346 = j14+j3+j6
#				min_z = max(j123, j156, j426, j453)
#				max_z = min(j2356, j1346, j1245, twok)
#				Sum = 0.
#				for z in range(min_z, max_z + 1, 2):
#					D1 = q_fact_2[z-j123] * q_fact_2[z-j156] * q_fact_2[z-j426] * q_fact_2[z-j453] \
#							* q_fact_2[j2356-z] * q_fact_2[j1346-z] * q_fact_2[j1245-z]
#					Sum += qf_zp1_sign[z] / D1
#				v = Deltas[j3,j6] * Sum
#				W6j[j1, j2, j3, j4, j5, j6] = v
#				W6j[j4, j5, j3, j1, j2, j6] = v
#				W6j[j1, j5, j6, j4, j2, j3] = v
#				W6j[j4, j2, j6, j1, j5, j3] = v
#				W6j[j2, j1, j3, j5, j4, j6] = v
#				W6j[j5, j4, j3, j2, j1, j6] = v
#				W6j[j5, j1, j6, j2, j4, j3] = v
#				W6j[j2, j4, j6, j5, j1, j3] = v
#
#	W6j.setflags(write=False)
#	if timer: print "qSU2_Wigner_6j_array({}): {} seconds".format(k, time.time() - t0)
#	return W6j


class qSU2_data(object):
	"""Data for q-deformed SU(2) symbols.
	Here q is a root of unity exp[2pi / (k+2)].

	Special cases are:
		[0] = [1] = [k+1] = 1, [k+2] = 0.
		[0]! = 1, [k+2]! = 0
"""
	def __init__(self, k):
		self.k = k
	##	Set up [n] and [n]!
		sin_list = np.sin( np.arange(1, k+2) * np.pi / (k+2) )
		QD = sin_list / sin_list[0]
		qf = np.zeros(k+3, np.float); qf[0] = 1
		for n in range(1, k+2): qf[n] = qf[n-1] * QD[n-1]
		qfact_2 = np.zeros(2 * len(qf) - 1, np.float)
		qfact_2[0::2] = qf		# qfact_2[2n] = [n]!
		self.qf_zp1_sign = qfact_2[2 : 2*k+3] * (1 - np.mod(np.arange(2*k+1),4))		# qf_zp1_sign[2z] = qf[z+1] * (-1)**z
		self.QD = QD
		self.qfact_2 = qfact_2
	##	Set up Fusion_abc, and Delta function
		Delta2 = np.zeros((k+1, k+1, k+1), np.float)
		Fusion_abc = np.zeros((k+1, k+1, k+1), np.int)
		for a in range(k+1):
			for b in range(k+1):
				for c in range(max(a-b, b-a), min(a+b, 2*k-a-b) + 1, 2):
					Delta2[a,b,c] = qfact_2[b+c-a] * qfact_2[c+a-b] * qfact_2[a+b-c] / qfact_2[a+b+c+2]
					Fusion_abc[a,b,c] = 1
		self.Delta = np.sqrt(Delta2)
		self.Fusion = Fusion_abc
		self.W6jval = {}
		self.W6j = self.W6j_from_dict
	
	def compute_W6j_val(self, j1, j2, j3, j4, j5, j6):
		j123 = j1+j2+j3; j453 = j4+j5+j3; j156 = j1+j5+j6; j426 = j2+j4+j6;
		j1245 = j1+j4+j2+j5; j2356 = j2+j5+j3+j6; j1346 = j1+j4+j3+j6
		min_z = max(j123, j156, j426, j453)
		max_z = min(j2356, j1346, j1245, 2 * self.k)
		Delta = self.Delta
		qfact_2 = self.qfact_2
		Deltas = Delta[j1,j2,j3] * Delta[j4,j5,j3] * Delta[j1,j5,j6] * Delta[j2,j4,j6]
		Sum = 0.
		for z in range(min_z, max_z + 1, 2):
			D1 = qfact_2[z-j123] * qfact_2[z-j156] * qfact_2[z-j426] * qfact_2[z-j453] \
					* qfact_2[j2356-z] * qfact_2[j1346-z] * qfact_2[j1245-z]
			Sum += self.qf_zp1_sign[z] / D1
		return Deltas * Sum

	def Wigner6j_array(self, timer=False):
		if hasattr(self, 'W6j_array'): return self.W6j_array
		t0 = time.time()
		k = self.k
		twok = 2 * k
		qfact_2 = self.qfact_2
		qf_zp1_sign = self.qf_zp1_sign
		W6j_array = np.zeros((k+1, k+1, k+1, k+1, k+1, k+1), np.float)
		Fusion_abcd = np.tensordot(self.Fusion, self.Fusion, axes=[[2],[0]])
		j1245list = np.array(np.nonzero(Fusion_abcd)).transpose().tolist()
		for J1245 in j1245list:
			j1 = J1245[0]; j2 = J1245[1]; j4 = J1245[2]; j5 = J1245[3]
			j14 = j1+j4; j25 = j2+j5
			if j14 > j25: continue
			if j1 > j4 or j2 > j5: continue
		##	Only compute terms with j1 <= j4, j2 <= j5, and j14 <= j25
			j12 = j1+j2; j45 = j4+j5; j15 = j1+j5; j24 = j2+j4; j1245 = j15+j24
			j3_min = max(j1 - j2, j2 - j1, j4 - j5, j5 - j4)
			j3_max = min(j12, twok - j12, j45, twok - j45)
			j6_min = max(j1 - j5, j5 - j1, j4 - j2, j2 - j4)
			j6_max = min(j15, twok - j15, j24, twok - j24)
			Deltas = np.outer(self.Delta[j1, j2] * self.Delta[j4, j5], self.Delta[j1, j5] * self.Delta[j4, j2])
			for j3 in np.arange(j3_min, j3_max + 1, 2):
				j123 = j12+j3; j453 = j45+j3
				for j6 in range(j6_min, j6_max + 1, 2):
					j156 = j15+j6; j426 = j24+j6;
					j2356 = j25+j3+j6; j1346 = j14+j3+j6
					min_z = max(j123, j156, j426, j453)
					max_z = min(j2356, j1346, j1245, twok)
					Sum = 0.
					for z in range(min_z, max_z + 1, 2):
						D1 = qfact_2[z-j123] * qfact_2[z-j156] * qfact_2[z-j426] * qfact_2[z-j453] \
								* qfact_2[j2356-z] * qfact_2[j1346-z] * qfact_2[j1245-z]
						Sum += qf_zp1_sign[z] / D1
					v = Deltas[j3,j6] * Sum
					W6j_array[j1, j2, j3, j4, j5, j6] = v
					W6j_array[j4, j5, j3, j1, j2, j6] = v
					W6j_array[j1, j5, j6, j4, j2, j3] = v
					W6j_array[j4, j2, j6, j1, j5, j3] = v
					W6j_array[j2, j1, j3, j5, j4, j6] = v
					W6j_array[j5, j4, j3, j2, j1, j6] = v
					W6j_array[j5, j1, j6, j2, j4, j3] = v
					W6j_array[j2, j4, j6, j5, j1, j3] = v
		self.W6j_array = zero_if_close(W6j_array)
		self.W6j_array.setflags(write=False)
		self.W6j = self.W6j_from_array
		if timer: print("qSU2_data({}).Wigner6j_array(): {} seconds".format(k, time.time() - t0))
		return self.W6j_array
	
	def W6j_from_dict(self, j1, j2, j3, j4, j5, j6):
		Fusion = self.Fusion
		if not Fusion[j1,j2,j3]: return 0.
		if not Fusion[j1,j5,j6]: return 0.
		if not Fusion[j4,j2,j6]: return 0.
		if not Fusion[j4,j5,j3]: return 0.
		j14 = j1 + j4; j25 = j2 + j5; j36 = j3 + j6
		if j14 < j25:
			if j36 < j14:   jL = [j3,j6,j1,j4,j2,j5]
			elif j36 < j25: jL = [j1,j4,j3,j6,j2,j5]
			else:           jL = [j1,j4,j2,j5,j3,j6]
		else:
			if j36 < j25:   jL = [j3,j6,j2,j5,j1,j4]
			elif j36 < j14: jL = [j2,j5,j3,j6,j1,j4]
			else:           jL = [j2,j5,j1,j4,j3,j6]
		if jL[0] > jL[1]: jL = [jL[1],jL[0],jL[2],jL[3],jL[5],jL[4]]
		if jL[2] > jL[3]: jL = [jL[0],jL[1],jL[3],jL[2],jL[5],jL[4]]
		j = (jL[0],jL[2],jL[4],jL[1],jL[3],jL[5])
		print("W6j_from_dict", (j1, j2, j3, j4, j5, j6), jL)
		if j not in self.W6jval: self.W6jval[j] = self.compute_W6j_val(*j)
		return self.W6jval[j]
	def W6j_from_array(self, j1, j2, j3, j4, j5, j6):
		return self.W6j_array[j1,j2,j3,j4,j5,j6]
	
	def check_Wigner_6j_symmetries(self, tol=1e-14):
		r"""
	Symmetries:
		\{b, a, c; e, d, f\} = \{a, b, c; d, e, f\}
		\{c, b, c; f, e, d\} = \{a, b, c; d, e, f\}
		\{d, e, c; a, b, d\} = \{a, b, c; d, e, f\}
		\{a, b, c; k/2-d, k/2-e, k/2-f\} = (-1)^{k+a+b+c} \{a, b, c; d, e, f\}
	"""
		raise NotImplementedError		# TODO
	
	def int_Wigner6j_array(self, timer=False):
		raise NotImplementedError		# TODO

qSU2_data_cache = {}

def qSU2(k):
	assert type(k) == int and k >= 0
	if k not in qSU2_data_cache:
		qSU2_data_cache[k] = qSU2_data(k)
	return qSU2_data_cache[k]
