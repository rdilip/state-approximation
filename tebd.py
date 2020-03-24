import numpy as np
from misc import *
import scipy
from scipy import linalg
import matplotlib.pyplot as plt

def H_TFI(L, g, J=1.0):
    H = []
    s_z = np.array([[1,0],[0,-1]])
    id = np.eye(2)
    s_x = np.array([[0,1],[1,0]])
    for i in range(L-1):
        gL = gR = g / 2.
        if i == 0:
            gL = g
        if i == L - 2:
            gR = g
        h = -J * np.kron(s_z, s_z) - gL * np.kron(s_x, id) - gR * np.kron(id, s_x) 
        H.append(h.reshape([2]*4))
    return H

def H_dH(L, dJ):
    """
    Hamiltonian for the dimerized Heisenberg model, with h=0.
    """
    J = 1.0
    H = []
    Sp = [[0., 1.],[0., 0.]]
    Sm = [[0., 0.],[1., 0.]]

    for i in range(L-1):
        h = -(J + dJ * (-1.0)**i) * (np.kron(Sp, Sm) + np.kron(Sm, Sp))
        H.append(h.reshape([2]*4))
    return H 

def make_U(H_bonds, dt):
    d = 2
    H = [h.reshape(d*d, d*d) for h in H_bonds]
    U = [scipy.linalg.expm(-dt*h).reshape([d]*4) for h in H]
    return(U)

def mps_overlap(Psi0, Psi1):
    O = np.tensordot(Psi0[0].conj(), Psi1[0], axes=[[0, 1], [0, 1]])
    for j in range(1, len(Psi0)):
        O = np.tensordot(O, Psi1[j], axes=[[1], [1]])
        O = np.tensordot(Psi0[j].conj(), O, axes=[[0, 1], [1, 0]])

    if O.shape != (1, 1):
        print(O.shape)
        raise ValueError

    return O[0, 0]

def get_theta2(Psi, i):
    return(np.tensordot(Psi[i], Psi[i+1], [2,1]).transpose([1,0,2,3]))
    
def sweep(Psi, Ulist, ops=None):
    L = len(Psi)
    Ss = []
    
    two_site_exp = False
    one_site_exp = False
    exp_vals = []
    if ops is not None:
        two_site_exp = (ops[0].ndim == 4)
        one_site_exp = (ops[0].ndim == 2)
    if Ulist is None:
        Ulist = [np.eye(4).reshape([2]*4) for i in range(L)]
     
    for i in range(L - 1):
        # Expectation values
        theta = get_theta2(Psi, i)
        if one_site_exp:
            exp_val = np.tensordot(ops[i], Psi[i], [0,0])
            exp_val = np.tensordot(Psi[i].conj(), exp_val, [[0,1,2],[0,1,2]])
            exp_vals.append(exp_val)
        if two_site_exp:
            exp_val = np.tensordot(theta, ops[i], [[1,2],[2,3]])
            exp_val = np.tensordot(theta.conj(), exp_val, [[0,1,2,3],[0,2,3,1]])
            exp_vals.append(exp_val)

        Utheta = np.tensordot(Ulist[i], theta, [[2,3],[1,2]]).transpose([2,0,1,3])
        chiL, d1, d2, chiR = Utheta.shape
        Utheta = Utheta.transpose([1,0,2,3])
        u, s, v = np.linalg.svd(Utheta.reshape(d1*chiL, chiR*d2),full_matrices=False)
        s /= np.linalg.norm(s)
        Psi[i] = u.reshape(d1, chiL, u.shape[-1])
        Psi[i+1] = (np.diag(s) @ v).reshape(v.shape[0], d2, chiR).transpose([1,0,2])

        Ss.append(s)
        
    if one_site_exp:
        exp_val = np.tensordot(ops[L-1], Psi[L-1], [0,0])
        exp_val = np.tensordot(Psi[L-1].conj(), exp_val, [[0,1,2],[0,1,2]])
        exp_vals.append(exp_val)
    
    info = dict(Ss=Ss, exp_vals=exp_vals)
    return(Psi, info)

def tebd_single_pass(Psi, U, order='R', ops=None):
    if order == 'L':
        Psi = [psi.transpose([0,2,1]) for psi in Psi[::-1]]
        U = [u.transpose([1,0,3,2]) for u in U[::-1]]

    Psi, info = sweep(Psi, U, ops=ops)

    if order == 'L':
        Psi = [psi.transpose([0,2,1]) for psi in Psi[::-1]]
        U = [u.transpose([1,0,3,2]) for u in U[::-1]]
        info['exp_vals'] = info['exp_vals'][::-1]
    return(Psi, info['exp_vals'])

def tebd(L, g, dt):
    #H = H_TFI(L, g)
    H = H_dH(L, 0.1)

    U = make_U(H, dt)
    sz = np.array([[1,0],[0,-1]])

    mag = [sz.copy() for i in range(L)]
    id = [np.eye(4).reshape([2]*4) for i in range(L)]

    b = 0.5 - np.random.rand(2,1,1)
    b = np.zeros((2,1,1))
    b[0,0,0] = 1
    b /= np.linalg.norm(b)
    Psi = [b.copy() for i in range(L)]

    for i in range(100):
        Psi, info = tebd_single_pass(Psi, U, order='R', ops=H)
        Psi, info = tebd_single_pass(Psi, U, order='L', ops=H)

    Psi, info_2 = tebd_single_pass(Psi, id, order='R', ops=None)
    Psi, info_2 = tebd_single_pass(Psi, id, order='L', ops=mag)
    return(Psi, np.sum(info), info_2)

def test_tebd():
    L = 10
    J = 1
    sigma_z = np.array([[1,0],[0,-1]])
    Sz = []
    mag_z =[sigma_z.copy() for i in range(L)]
    H_exp = []
    gs = [0.0]
    gs = np.linspace(0.0, 1.5, 20)

    for g in gs:
        H_bonds = H_TFI(L, g)
        Psi, E, mag = tebd(L, g, 0.1)
        Sz.append(np.mean(mag))
        H_exp.append(E)

    return Psi, gs, H_exp, Sz

if __name__ == '__main__':
    L = 10
    J = 1
    sigma_z = np.array([[1,0],[0,-1]])
    Sz = []
    mag_z =[sigma_z.copy() for i in range(L)]
    for g in np.linspace(0, 1.5, 10):
        Psi, Ss = tebd(L, g, J, 1.0, 0.01)
        Psi, info = sweep(Psi, None, mag_z)
        Sz.append(info['exp_vals'])
    print(Sz)
    plt.scatter(np.linspace(0, 1.5, 10), Sz)
