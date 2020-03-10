""" Recursive parameterization of a unitary matrix.

Conventions:
Uses the A vectors as defined in arxiv math-ph/0504049. I also have a writeup. 
The list of vectors should proceed in increasing size, e.g., As[0] is a vector
of length 1, and there are n-1 such vectors. 

The input to the parametrization should concatenate all the As, then all the 
thetas, then the alphas and the betas (no alphas and betas included atm because
complex isn't working super well). 

Two parameter modes: cart represents cartesian input vectors, sph represents
spherical input vectors. cart means you have to do a constrained optimization.

For an n by n matrix, you need n-1 vectors a^{(j)} with length 1, 2,..., n-1
and n-1 angles theta.

"""
# TODO Still some problems with complex numbers -- only use real for now.
# At some point need to add alpha and beta vectors, but no real purpose until
# unitary is working properly
import numpy as np

def construct_A(a, theta, n):
    """ Constructs A_{n,j} from the vector a (cartesian)
    a : np.Array
        An array of length j
    theta : float
        Just read the paper...math-ph/0504049
    n : int
        Dimension of unitary
    """
    j = len(a)
    Q = np.array(np.zeros((j+1, j+1)), dtype='complex128')
    Q[:-1, :-1] = np.eye(j) - (1-np.cos(theta)) * np.outer(a.conj(), a)
    Q[:-1, -1] = np.sin(theta) * a
    Q[-1, :-1] = -np.sin(theta) * np.conj(a)
    Q[-1,-1] = np.cos(theta)

    A_nj = np.array(np.eye(n), dtype='complex128')
    A_nj[:j+1, :j+1] = Q
    return(A_nj)

def construct_V(params, n, mode='sph'):
    assert mode in ['cart', 'sph']
    assert np.isclose(len(params), (n - 1) + (n - 2)*(n-1) / 2 + (mode == 'cart'))
    As, thetas = process_parameters(params, n, mode=mode)
    return(_construct_V(As, thetas))

def stack_params(vectors, thetas):
    parameterization = vectors[0]
    for a in vectors[1:]:
        parameterization = np.hstack((parameterization, a))
    for theta in thetas:
        parameterization = np.hstack((parameterization, theta))
    return(parameterization)

def process_parameters(params, n, mode='sph'):
    """ Given a single list of params, processes into a list of As and a list
    of thetas.
    params : list 
        List of parameters, using convention described in the docstring.
    n : int
        Size of corresponding unitary matrix (can technically get this from
        length of params, figure this out later)
    mode : str
        Either sph or cart. sph is spherical, cart is cartesian.
    """
    assert mode in ['sph', 'cart']

    if mode == 'sph':
        split_indices = np.cumsum(np.arange(n-2) + 1)
        split = np.split(params, split_indices)
        phis, thetas = split[:-1], split[-1]
        As = hyperspherical_to_cartesian(phis)
    else:
        split_indices = np.cumsum(np.arange(n-1) + 1)
        split = np.split(params, split_indices)
        As, thetas = split[:-1], split[-1]
    return(As, thetas)

def hyperspherical_to_cartesian_vector(phi):
    """ Converts a single hyperspherical vector of dim n-1
    to a normalized vector in R^n """
    cartesian_vector = []
    for i in range(len(phi)):
        cartesian_vector.append(np.prod(np.sin(phi[:i])) * np.cos(phi[i]))
    cartesian_vector.append(np.prod(np.sin(phi)))
    return np.array(cartesian_vector)

def hyperspherical_to_cartesian(phis):
    """ Converts from a list of n-1 phis to a normalized vector in R^n. """
    cartesian = [np.array([1.])]
    # First value should always be one.
    for phi in phis:
        cartesian.append(hyperspherical_to_cartesian_vector(phi))
    return cartesian
        
def _construct_V(As, thetas):
    """ Constructs a unitary matrix from a list of vectors and list of thetas.
    As : list of np.Array
        Each element with index i should be a normalized vector in R^{i+1} 
        (zero indexing). Should have n-1 vectors.
    thetas : list of floats
        The angles theta. Should have length l - 1
    """
    for i, a in enumerate(As):
        assert len(a) == i + 1
        assert np.isclose(np.linalg.norm(a), 1.0, rtol=1.e-8)
    n = len(As[-1]) + 1
    V = np.array(np.eye(n), dtype='complex128')
    for i, a in enumerate(As[::-1]):
        V = V @ construct_A(a, thetas[i], n)
    return(V)
