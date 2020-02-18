""" Module for disentangling using jax. Implements autodifferentiation techniques
to extract a vector corresponding to the Jacobian of the cost function. 

There are three routines -- iterative disentangler (DNTGL{n}), vanilla
scipy.optimize (SCPY{n}), and scipy.optimize with a jax jacobian (JAX{n}), where
{n} is the Renyi entropy being optimized.

For psi with shape (2,5,5,2), time comparisons give

JAX0 < SCPY0 < DNTGL2
SCPY2 < DNTGL2 < JAX2

We ultimately care about the Schmidt rank (Renyi-0) case, but random wave
functions do not generally have small Schmidt rank, so it's unclear. Best is
probably to just plug it into TFI TEBD2 and check.
"""

import numpy as onp
import jax.numpy as np
import scipy
from scipy.optimize import minimize

# PARAMETRIZATIONS
# It might be that certain parametrizations explore the unitary space in a
# way more conducive towards finding minima. Worth checking at some point.
# When you do, make the minimization functions parametrization agnostic (i.e.,
# just pass a parametrization function)
def cayley_transform(v, n):
    """ Returns a unitary via a Cayley parametrization
    Parameters
    ----------
    v : list
        List of floats that determine a skew symmetric matrix. The unitary
        is parametrized as (I - A).(I + A)^-1
    n : int
        Dimension of the unitary matrix
    Returns
    -------
    """
    assert v == int(n * (n-1) / 2) # ensures compatibility. require both for debugging.
    A = np.zeros((n, n))
    # Does not work with jax
    A = jax.ops.index_update(A, onp.triu_indices_from(A, k=1), v)
    A -= A.T
    I = np.eye(n)
    return((I - A) @ np.linalg.inv(I + A))

# MINIMIZATION

def renyi_v(v, theta, alpha=2):
    """ Cost function to return the renyi entropy 
    v : list
        Unitary parametrization
    """
    chiL, d1, d2, chiR = theta.shape
    n = d1 * d2
    U = cayley_transform(v, n).reshape([d1,d2,d1,d2])
    Utheta = np.tensordot(theta, U, [[1,2],[2,3]]).transpose([0,2,3,1])
    return(renyi(Utheta, alpha=alpha))

def renyi(theta, alpha=2, output=False):
    """ Renyi entropy of a state theta
    Parameters
    ----------
    theta : np.Array 
        TEBD style wavefunction. 
    alpha : int
        Renyi index
    output : bool
        Whether or not to print the Schmidt rank
    Returns
    -------
    S: alpha-Renyi entropy
    """
    assert len(theta.shape) == 4
    chiL,d1,d2,chiR = theta.shape
    theta = theta.reshape([chiL*d1, chiR*d2])
    X, S, Z = np.linalg.svd(theta, full_matrices=False, compute_uv=True)
    S = S[np.abs(S) > 1.e-10]**2
    if output:
        print(len(S))
    if alpha== 0:
        return(np.log(len(S)))
    return((1/(1-alpha)) * np.log(np.sum(S**alpha)))

def disentangle_jax(theta):
    """ Given an input wavefunction, returns a disentangled wavefunction
    and the corresponding unitary

    Parameters
    ----------
    theta : np.Array
        Input wavefunction
    Returns
    -------
    theta : np.Array
        disentangled wavefunction
    U : np.Array
        Unitary disentangler
    """
    chiL, d1, d 
    v = 
    smin = minimize(renyi_v




