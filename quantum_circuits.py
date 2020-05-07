"""
Some functinos for quantum circuits
"""

import numpy as np
import pickle
from copy import deepcopy
import glob
from misc import mps_2form, mps_overlap
import warnings
from rfunc import mps2mpo, mpo2mps, entanglement_entropy
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

def check_unitary(T, axes=[0,1]):
    """
    Checks that the tensor is unitary with axes specifying all incoming
    legs.
    """
    all_legs = list(range(len(T.shape)))
    outgoing_legs = [leg for leg in all_legs if leg not in axes]
    outgoing_shape = [T.shape[leg] for leg in outgoing_legs]
    total_outgoing_size = np.prod(outgoing_shape)
    identity = np.tensordot(T, T.conj(), [axes, axes]).reshape(\
                            total_outgoing_size, total_outgoing_size)
    return np.allclose(np.eye(total_outgoing_size), identity)


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
            
def generate_state_from_unitary_list(Ulist,
                                     reverse=False,
                                     Lambda=None,
                                     check=False,
                                     entanglement=False):
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
    check : bool
        If true checks that everything is a unitary
    entanglement : bool
        if true, returns the entanglement at each layer.
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
        Lambda = [l.transpose(0,2,1) for l in Lambda[::-1]]
    Ss = [0.0]

    Psi = deepcopy(Lambda)
    for Us in Ulist:
        for i,U in enumerate(Us):
            if check:
                assert check_unitary(U)
                assert U.shape == (2,2,2,2)
            theta = np.tensordot(Psi[i], Psi[i+1], [2,1])
            theta = np.tensordot(U, theta, [[0,1],[0,2]]).transpose(2,0,1,3)
            chiL, pL, pR, chiR = theta.shape
            q, r = np.linalg.qr(theta.reshape(chiL*pL, chiR*pR))

            Psi[i] = q.reshape(chiL, pL, -1).transpose(1,0,2)
            Psi[i+1] = r.reshape(-1, pR, chiR).transpose(1,0,2)
        Psi = mps_2form(Psi, 'B')
        if entanglement:
            Ss.append(entanglement_entropy(Psi)[len(Psi)//2])
    if reverse:
        return [psi.transpose(0,2,1) for psi in Psi[::-1]]
    if entanglement:
        return Psi, Ss
    return Psi

def A_to_Ulist(A):
    """
    Converts a column vector A to a list of unitaries, sweeping from right to
    left.
    """
    # LEFT TO RIGHT
    Ulist = []
    for a in A[::-1]:
        Ulist.append(a.transpose(3,1,0,2))
    Ulist[-2] = np.tensordot(Ulist[-2], Ulist[-1].reshape(2,2), [3,0])
    Ulist.pop(-1)
    return Ulist
    #Ulist = []
    #for a in A:
    #    Ulist.append(a.transpose(1,3,2,0))
    #Ulist[1] = np.tensordot(Ulist[1], Ulist[0].reshape(2,2), [2,0]).transpose(0,1,3,2)
    #Ulist.pop(0)
    #return Ulist

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

def convert_column_to_unitary_layers(As):
    """
    Converts a left hand column in TN form to a series of unitary layers. The
    length of As corresponds to the number of layers. This assumes that the 
    right hand column is a spin up trivial state.
    """
    As = deepcopy(As)
    Ulist = [A_to_Ulist(a) for a in As[::-1]]
    #Ulist = [[U.transpose(1,0,3,2) for U in Us[::-1]] for Us in Ulist]
    return Ulist

if __name__ == '__main__':
    data_points = [round(i, 1) for i in np.linspace(0.5, 10.0, 20)]
    data_points = [4.0]
    for data_point in data_points:
        fname = "sh_data_long_chi32/T{0}.pkl".format(data_point)
        print(fname)
        with open(fname, "rb") as f:
            Psi = pickle.load(f)
        fname = fname.split("/")[1]
        for i in range(1,6):
            print("\t" + str(i))
            Ulist = get_initial_guess(Psi, i)
            with open("mm_initial_guesses_long_chi32/{0}_layers_{1}".format(i, fname), "wb+") as f:
                pickle.dump(Ulist, f)

