from __future__ import absolute_import
# from __future__ import division
from __future__ import print_function
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

#from mosesmove import moses_move
from moses_simple import moses_move
from misc import *
from matplotlib import pyplot as plt
"""This code takes an n-leg TFI ladder, and decomposes it as

	Psi = A0 A1 A2 ... LambdaN
	

"""

np.set_printoptions(precision=4, suppress=True, linewidth=120)


def S(s):
    s = s[s > 1e-16]
    return -2 * np.vdot(s**2, np.log(s))


def split_columns(Psi, n, truncation_par):
    """ Given an MPO Psi for an n-leg ladder, with physical legs shaped as d, d^(n-1), succesively perform the tri-split Moses Move to bring into a canonical PEPs. Returns the isometries "As" and the renormalized set of Lambdas.

	"""

    print("Initial chi", np.max([b.shape[3] for b in Psi]))

    d = Psi[0].shape[0]
    As = []
    Ls = [Psi]
    Er = []

    for j in range(n - 1):
        #A, Lambda, info = sweeped_moses_move(Psi, truncation_par)

        A, Lambda, info = moses_move(Psi, truncation_par)
        overlap = Psi_AL_overlap(Psi, A, Lambda)

        #Optional: follow up with some variational sweeps. Not sure why this is so slow, should check implementation
        #print "|Psi - A Lam|^2 / L ", check_overlap(Psi, A, Lambda)/len(Psi)
        #A, Lambda = var_moses(Psi, A, Lambda, truncation_par['chi_max']['etaV_max'],N = 8)
        #overlap = check_overlap(Psi, A, Lambda)

        Ls.append(Lambda)
        Er.append(overlap)

        A = [
            a.reshape(-1, d, a.shape[1], a.shape[2],
                      a.shape[3]).transpose([1, 0, 2, 3, 4]) for a in A
        ]
        As.append(A)
        Psi = peel(Lambda, d)

        print("|Psi - A Lam|^2 / L :", Er[-1] / len(Psi))
        print("Errors", info['errors'])
        print("A.chiV", [a.shape[4] for a in A])
        print("A.chiH", [a.shape[2] for a in A])
        print("L.chiV", [l.shape[3] for l in Lambda])
        print()

    return As, Ls, Er


if __name__ == '__main__':
    filename = 'test_var_n2'
    with open('test_data/4TFI_J3.5.mps', 'r') as f:
        Psi = pickle.load(f)

    n = 4

    truncation_par = {
        'bond_dimensions': {
            'etaV_max': 40,
            'etaH_max': 16,
            'chiV_max': 3,
            'chiH_max': 3
        },
        'p_trunc': 1e-8
    }
    As, Ls, Er = split_columns(Psi, n, truncation_par)

    L = [
        l.reshape(l.shape[0], l.shape[1], 1, l.shape[2],
                  l.shape[3]).transpose([1, 0, 2, 3, 4]) for l in Ls[-1]
    ]
    PEPs = As + [L]
    peps_print_chi(PEPs)
    plt.gcf().clear()
    plt.subplot(211)

    data = {}
    data['Er'] = Er
    data['Ss'] = []

    Ss = []

    for psi in Ls:

        s = mps_entanglement_spectrum(psi)
        data['Ss'].append(s)
        plt.plot([S(p) for p in s])

    plt.ylabel(r'$S_V(x)$', fontsize=16)
    plt.xlabel(r'$x$', fontsize=16)
    plt.ylim([0, 0.22])
    plt.legend(list(range(n)))

    plt.subplot(212)

    for i in range(n):
        plt.plot(data['Ss'][i][10], '.-')

    plt.ylim([1e-9, 1.])
    plt.yscale('log')

    print(Er, ':', np.sum(Er))

    plt.savefig(filename + '.pdf')
    pickle.dump(data, open(filename + '.pkl', "wb"))
