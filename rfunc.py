""" Functions that I wrote, to keep Frank and Mike's code separate.

TEBD starts to the left. PEPs is a list of one-column wavefunctions (each column
indexes down y). 

[Psi0 Psi1 Psi2 ... PsiL-1]

Mike's thing says this:
   PEPs      4
             |
             |
       1 --------- 2
            /|
     t =   / |
          0  3

What I think it should be...
   PEPs      3
             |
         1-- t --2
            /|
           0 4 


	Stored in cartesian form PEPs[x][y], list-of-lists

4 on PEPs[x][i] connects to 3 on PEPs[x][i + 1], which makes absolutely
no sense. We say that we start in -+ form (top left) and then tebd down, then
moses move back up and end up in ++ form. But if 

"""
from misc import *
import os,warnings
import numpy as np
import glob
import scipy as sp

def get_N(Nmax = 1): return(np.diag(np.arange(Nmax + 1)))

def pad(T, axis, num):
    """ Given a tensor T with a shape (s1, s2,... sn), pads axis with 0s until
    the new shape is (s1, s2,... num,... sn)
    """
    shape = list(T.shape)
    npad = [(0,0) if i != axis else (0,num-shape[axis]) for i in range(T.ndim)]
    newT = np.pad(T, pad_width=npad, mode='constant', constant_values=0)
    return(newT)

def check_isometry(Psi, i):
    """
    Checks that the wavefunction Psi is in canonical form with 
    orthogonality center at index i
    """
    if Psi[i].ndim != 4:
        Psi = mps2mpo(Psi)
    for j in range(i):
        psi = Psi[j]
        pL, pR, chiS, chiN = psi.shape
        iso = np.tensordot(psi, psi.conj(), [[0,2],[0,2]])
        iso = iso.reshape([pR*chiN, pR*chiN])

        if not np.allclose(iso, np.eye(pR*chiN), rtol=1.e-8):
            return False

    for j in range(i+1, len(Psi)):
        psi = Psi[j]
        pL, pR, chiS, chiN = psi.shape
        iso = np.tensordot(psi, psi.conj(), [[0,1,3],[0,1,3]])
        if not np.allclose(iso, np.eye(chiS), rtol=1.e-8):
            return False
    return True



def strip(fname):
    if fname[-1] == '/':
        return(fname[:-1])
    return(fname)

def expectation_value(PEPs, ops, truncation_par):
    d = PEPs[0][0].shape[0]
    Nmax = d - 1
    if ops == "N":
        ops = get_N(Nmax = PEPs[0][0].shape[0] - 1)

def log(fname, param):
    """ Logs the parameters for run fname to file index.txt
    Parameters
    ----------
    fname: Name of file or directory of run.

    param: Dictionary of parameters (key is param name, value is param)
    """
    msg = fname + ": "
    for k, v in param.items():
        msg += f"{k}={np.round(v, 3)}, "

    # Check which folder we're in
    moveback = ""
    if fname[-1] == "/":
        fname = fname[:-1]
    if fname == os.getcwd().split("/")[-1]:
        moveback = "../"

    with open(moveback + "index.txt", "a+") as f:
        f.write("\n" + msg[:-2])

def make_rawdir(rawdir):
    """ Constructs a raw directory and cds inside,so that all files are saved
    inside the raw directory
    Parameters
    ----------
    rawdir: Name of the raw directory. If not given, then the program constructs
    a directory name based on the current folder name.
    """
    if not rawdir:
        rawdir = os.getcwd().split("/")[-1]
        rawdir += "_run"
        count = 0
        identifier = rawdir + str(count).zfill(3)
        while os.path.exists(identifier) or os.path.exists(identifier+".pkl"):
            count += 1 
            identifier = rawdir + str(count).zfill(3)
        rawdir += str(count).zfill(3)
    if not os.path.exists(rawdir):
        os.mkdir(rawdir)
    else:
        if os.listdir(rawdir):
            warnings.warn("You are saving into a nonempty directory.")
            save_into_dir = input("Are you sure you want to do this? (Y/N)")
            if save_into_dir is "N":
                raise ValueError("Directory already exists.")

    os.chdir(rawdir) # generated files should go into raw folder
    if rawdir[-1] != "/":
        rawdir += "/"
    return(rawdir)

def write_next_file(obj, identifier, params=None):
    """ Wrapper function that writes to a file and to the relevant index.txt """
    fname = get_next_filename(identifier)

    run_params = ": "
    if params:
        for k, v in params.items():
            run_params += f"{k}={v} "
   
    with open(fname, "wb+") as f:
        pickle.dump(obj, f)
    with open("/".join(fname.split('/')[:-1]) + '/index.txt', 'a+') as f:
        f.write(fname.split('/')[-1] + run_params + '\n')

def get_next_filename(identifier):
    """ All data files in a single directory generally have a common naming
    prescription -- make_rawdir uses the directory name. Sometimes I want to use
    a different name, so this will accept name pattern and find the correct
    index to generate a new file. 

    ~ example
    >> ls
       tenpy_bh_run00.pkl    tenpy_bh_run01.pkl    tenpy_bh_run02.pkl
    >> get_file_index("tenpy_bh")
       tenpy_bh_run03
    """
    ix = [-1]
    loc = "/".join(identifier.split("/")[:-1])
    if not os.path.exists(loc):
        os.makedirs(loc)
    for fname in glob.glob(identifier + "*.pkl"):
        ix.append(int(fname.split("_")[-1][3:5])) # runXX.pkl
    return(f"{identifier}_run{str(max(ix) + 1).zfill(2)}.pkl")

def local_state_from_fill(fill, nmax = 1):
    """ Returns a local state at a specified filling.
    Parameters
    ----------
    fill: float between 0.0 and 1.0

    n_max: Max number of bosons on site.
    """
    if fill > nmax:
        raise ValueError("Fill cannot be greater than max number of bosons per site")
    i = int(np.ceil(fill))
    p = np.sqrt(fill / i)
    state = [np.sqrt(1 - p*p)] + [0] * (i-1) + [p] + [0]*(nmax - i)
    return(state)

def svd(theta, compute_uv=True, full_matrices=True):
    """ SVD with gesvd backup """
    try:
        A,B,C = np.linalg.svd(theta,
                             compute_uv=compute_uv,
                             full_matrices=full_matrices)

        return((A,B,C))

    except np.linalg.linalg.LinAlgError:
        print("SVD failed. Trying *gesvd*")
        return sp.linalg.svd(theta,
                             compute_uv=compute_uv,
                             full_matrices=full_matrices,
                             lapack_driver='gesvd')

def svd_trunc(theta, eta, p_trunc=0.):
    """
    Returns the SVD of a matrix with renormalization to a dimension eta.
    Parameters
    ----------
    theta : np.Array
        Matrix in question. Usually a TEBD style wavefunction resized to
        (chiL*d1, chiR*d2)
    eta : int | str
        Max bond dimension. If eta is 'full', keep everything.
    p_trunc : float
        \sum_{i > cut} s_i^2, where s is the Schmidt spectrum of theta. Does
        not check for normalization of theta. 
    Returns
    -------
    U, S, V : np.Array
        Here, U @ np.diag(S) @ V ~= theta
    error : float
        Total error due to truncation
    """
    assert len(theta.shape) == 2

    U, s, V = svd(theta, compute_uv=True, full_matrices=False)
    nrm = np.linalg.norm(s)
    assert(np.isclose(nrm, 1., rtol=1e-8))

    if eta == 'full':
        return U, s, V, 0.0

    if p_trunc > 0.:
        eta_new = np.min(
            [np.count_nonzero((nrm**2 - np.cumsum(s**2)) > p_trunc) + 1, eta])
    else:
        eta_new = eta

    nrm_t = np.linalg.norm(s[:eta_new])
    return U[:, :eta_new], s[:eta_new] / nrm_t, V[:eta_new, :], nrm**2 - nrm_t**2

