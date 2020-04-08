"""
Some functinos for quantum circuits
"""

import numpy as np
import pickle
import glob
from misc import mps_2form, mps_overlap
import warnings
from state_approximation import mps2mpo, mpo2mps, multiple_diagonal_expansions
import scipy

def hadamard():
    return (1.0 / np.sqrt(2)) * np.array([[1,1,],[1,-1]])

def cnot():
    matrix_form = np.zeros((4,4))
    matrix_form[:2,:2] = np.eye(2)
    matrix_form[2:, 2:] = np.array([[0,1],[1,0]])
    return matrix_form.reshape([2]*4)

def bell_unitary():
    """
    Returns a two site unitary for creating a bell pair
    """
    return np.tensordot(hadamard(), cnot(), [1,0])

def product_state(L):
    b = np.zeros((2,1,1))
    b[0,0,0] = 1.
    return [b.copy() for i in range(L)]

def random_product_state(L):
    Psi = []
    for i in range(L):
        b = np.zeros((2,1,1))
        b[:,0,0] = np.random.rand(2)
        Psi.append(b.copy())
    norm = mps_overlap(Psi, Psi)
    Psi[0] /= np.sqrt(norm)
    return Psi


def bell_pair_mps(L, num_unitaries):
    """
    Returns an MPS composed of repeated bell pairs
    """
    b = np.zeros((2,1,1))
    b[0,0,0] = 1.0
    Psi = [b.copy() for i in range(L)]
    U = bell_unitary()
    for i in range(num_unitaries):
        j = i % (L-1)
        if j == 0 and i != 0:
            Psi = mps_2form(Psi, 'B')
        theta = np.tensordot(Psi[j], Psi[j+1], [2,1]).transpose(1,0,2,3)
        theta = np.tensordot(theta, U, [[1,2],[0,1]]).transpose(0,2,3,1)
        chiL, pL, pR, chiR = theta.shape
        q, r = np.linalg.qr(theta.reshape(chiL*pL, chiR*pR))
        Psi[j] = q.reshape(chiL, pL, -1).transpose(1,0,2)
        Psi[j+1] = r.reshape(-1, pR, chiR).transpose(1,0,2)
    return Psi
            
def generate_state_from_unitary_list(Ulist, reverse=False, Lambda=None):
    """
    Generates a state from a list of unitary gate layer.
    Parameters
    ----------
    Ulist : list of lists of np.Array
        Each list is a single layer of unitary gates.
    Lambda : list of np.Array
        The starting state (usually a product state)
    reverse : bool
        If true, applies the sequence from right to left.
    """
    if Lambda is None:
        print("Starting from trivial state")
        b = np.zeros((2,1,1))
        b[0,0,0] = 1.
        L = len(Ulist[0]) + 1
        Lambda = [b.copy() for i in range(L)]
    if Lambda[0].ndim == 4:
        Lambda = mpo2mps(Lambda)

    if reverse:
        print("Sweeping from right to left.")
        Ulist = [[i.transpose(1,0,3,2) for i in Us[::-1]] for Us in Ulist]
        Lambda = [l.transpose(0,2,1) for l in Lambda[::-1]]

    Psi = Lambda.copy()
    for Us in Ulist:
        for i,U in enumerate(Us):
            theta = np.tensordot(Psi[i], Psi[i+1], [2,1])
            theta = np.tensordot(U, theta, [[0,1],[0,2]]).transpose(2,0,1,3)
            chiL, pL, pR, chiR = theta.shape
            q, r = np.linalg.qr(theta.reshape(chiL*pL, chiR*pR))

            Psi[i] = q.reshape(chiL, pL, -1).transpose(1,0,2)
            Psi[i+1] = r.reshape(-1, pR, chiR).transpose(1,0,2)
            Psi = mps_2form(Psi, 'B')
    if reverse:
        return [psi.transpose(0,2,1) for psi in Psi[::-1]]
    return Psi

def A_to_Ulist(A):
    """
    Converts a column vector A to a list of unitaries, sweeping from right to
    left.
    """
    Ulist = []
    for a in A:
        Ulist.append(a.transpose(1,3,2,0))
    Ulist[1] = np.tensordot(Ulist[1], Ulist[0].reshape(2,2), [2,0]).transpose(0,1,3,2)
    Ulist.pop(0)
    return Ulist

def unitaries_acting_on_trivial_state(A, Lambda):
    """ 
    Given a column vector A acting on a trivial vector Lambda, this function
    returns a list Ulist of unitaries that acts on the state |0>.
    """
    L = len(Lambda)
    Ulist = A_to_Ulist(A)
    single_sites = [unitary_to_trivial(l) for l in Lambda]
    for i in range(L-1):
        Ulist[i] = np.tensordot(single_sites[i].conj(), Ulist[i], [1, 0])
    Ulist[L-2] = np.tensordot(single_sites[L-1].conj(), Ulist[L-2],[1,1]).transpose(1,0,2,3)
    return Ulist

def perpendicular_vector(vect):
    angles = np.angle(vect)
    mags = np.abs(vect)
    return mags[::-1] * [1,-1] *np.exp(1.j*angles)

def unitary_to_trivial(T):
    """
    Given a tensor T in a trivial MPS, returns the unitary U s.t.
    U.v = trivial. So stick the first index of U onto the two site
    unitaries.
    """
    vect = T.reshape(2)
    perp = perpendicular_vector(vect)
    U = np.array([vect, perp]).conj()
    return U

def generate_state_from_unitary_right_to_left(Ulist, Lambda):
    """
    Generates a state with a series of unitaries sweeping from right to left.
    Should be incorporated with previous function (buggy when I tried earlier)
    """
    warnings.warn("This was just for debugging and is pretty deprecated")
    L = len(Lambda)
    # Some bookkeeping
    if Lambda[0].ndim == 3:
        Psi_from_Ulist = mps2mpo(Lambda)
    else:
        Psi_from_Ulist = Lambda.copy()

    # Only set up for a single sweep
    Ulist = Ulist[0].copy()

    assert L-1 == len(Ulist)

    for i in range(L-2,-1,-1):
        theta = np.tensordot(Psi_from_Ulist[i], Psi_from_Ulist[i+1], [3,2])
        theta = np.tensordot(Ulist[i], theta, [[0,1],[0,3]])
        theta = theta.transpose(0,2,3,1,4,5)
        pL, _, chiW, pR, _, chiE = theta.shape
        r, q = scipy.linalg.rq(theta.reshape(pL*chiW, pR*chiE), mode='economic')
        Psi_from_Ulist[i+1] = q.reshape(-1, pR, 1, chiE).transpose(1,2,0,3)
        Psi_from_Ulist[i] = r.reshape(pL, 1, chiW, -1)
    return Psi_from_Ulist

def get_initial_guess(Psi, num_layers):
    """
    This function uses the moses move to get an initial guess for the unitary
    series.
    """
    Psi = Psi.copy()
    Psi = [psi.transpose(0,2,1) for psi in Psi[::-1]]
    As, Lambda, info = multiple_diagonal_expansions(Psi, n=num_layers, mode='single_site_half')

    current_state = Lambda.copy()
    Ulist = [A_to_Ulist(a) for a in As[::-1]]
    Ulist[0] = unitaries_acting_on_trivial_state(As[-1], Lambda)
    Ulist = [[U.transpose(1,0,3,2) for U in Us[::-1]] for Us in Ulist]

    return Ulist

if __name__ == '__main__':
    for fname in glob.glob("sh_data/T*.pkl"):
        print(fname)
        with open(fname, "rb") as f:
            Psi = pickle.load(f)
        fname = fname.split("/")[1]
        for i in range(1,6):
            print("\t" + str(i))
            Ulist = get_initial_guess(Psi, i)
            with open(f"mm_initial_guesses/{i}_layers_{fname}", "wb+") as f:
                pickle.dump(Ulist, f)
