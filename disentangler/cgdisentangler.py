i2mport numpy as np
import time
from misc import svd
import ls
import renyi_hessian
def dSn(psi, n=1):
    """ Returns H = dS / dX^D  and S for nth Renyi, with U = exp[X] and X^D = - X
        psi = El, l, r, Er
        
        Returns Sn, dSn
    """
    chi = psi.shape
    theta = psi.reshape((chi[0] * chi[1], chi[2] * chi[3]))
    X, s, Z = svd(theta, full_matrices=False)

    if n == 'inf':
        r = np.zeros_like(s)
        r[0] = 1 / s[0]
        S = -2 * np.log(s[0])
    elif n == 1:
        p = s**2
        lp = np.log(p)
        r = s * lp * (s > 1e-10)
        S = -np.inner(p, lp)
    else:
        tr_pn = np.sum(s**(2 * n))
        ss = s**(2 * n - 1.)
        r = ss / tr_pn * n / (n - 1)
        S = np.log(tr_pn) / (1 - n)
    
    Epsi = np.dot(X * r, Z).reshape(chi)
    dS = np.tensordot(psi, Epsi.conj(), axes=[[0, 3], [0, 3]])
    dS = dS.reshape((chi[1] * chi[2], -1))
    dS = (dS - dS.conj().T)
    return S, dS

def beta_PR(g1, g2):  #Polak-Ribiere
    return np.max([0, np.real(np.vdot(g2, g2 - g1) / np.vdot(g1, g1))])


def beta_FR(g1, g2):  #Fletcher-Reeves
    return np.real(np.vdot(g2, g2) / np.vdot(g1, g1))


def beta_SD(g1, g2):  #Steepest descent
    return 0.


def disentangle_CG(psi,
                  n=2,
                  eps=1e-6,
                  max_iter=30,
                  verbose=0,
                  beta=None,
                  pt=0.5):
    """Disentangles a wavefunction.
    Returns psi, U, Ss
    """
    if beta is None:
        beta = beta_PR

    mod = disentangler_model(psi.copy(), n=n, pt=pt)

    res = cg(mod,
             max_iter=max_iter,
             ls_args={'gtol': 0.05},
             eps=eps,
             verbose=verbose,
             beta=beta)
    Ss = res['F']

    if verbose > 1:
        print("Iter, Eval:", res['iter'], mod.total)

    if verbose > 2:
        dSs = res['Fp_nrm']
        plt.subplot(1, 3, 2)
        plt.plot(Ss, '.-')
        plt.subplot(1, 3, 3)
        plt.plot(dSs, '.-')
        plt.yscale('log')
        #plt.tight_layout()

    return mod.psi_k, mod.U, {'Ss': Ss, 'eval': mod.total, 'iter':res['iter'], 'times':res['times']}


def cg(model,
       eps=1e-6,
       max_iter=30,
       ls_args={},
       beta=beta_PR,
       verbose=0,
       fmin=None):
    """Conjugate gradient for n-th renyi entropy
    
        eps: target error in |df|
        beta: beta_PR, FR or SD
    """

    F = [model.F()]  #Objective
    Fp = [model.Fp()]  #Directional derivative at kth iterate
    Fp_nrm = [np.linalg.norm(Fp)]
    H = 0  #Search dir initiated at kth iterate
    Beta = []
    k = 0
    ts = [0.5]
    t = [time.time()]
    while Fp_nrm[-1] > eps and k < max_iter:
        if k > 0 and np.linalg.norm(F[-1] - F[-2]) < eps:
            break

        if k > 0:
            Beta = beta(model.ls_transport(Fp[-2]), Fp[-1])
        else:
            Beta = 0.

        Hnew = -Fp[-1] + Beta * H
        if k > 0:
            descent = np.vdot(Hnew, Fp[-1]).real  #Restart CG if not a descent direction
            if descent > 0.:
                Beta = 0.
                Hnew = -Fp[-1]
                descent = -Fp_nrm[-1]**2
        else:
            descent = -Fp_nrm[-1]**2
        H = Hnew
        #print alpha, descent, descent / alpha
        xk, Gk, pk, = model.ls_setup(H)
        #alpha = np.vdot(H, model.H()._matvec(H))
        if False and alpha > 0.:
            t0 = -descent / alpha
        else:
            t0 = np.mean(ts[-10:])
        #t0 = ts[-1]
        #print Gk[0], np.vdot(H[-1], Fp[-1]).real
        #Minimize along x = x0 + a d,   input: F0, x0, G0, d, F, G
        MT = ls.pymswolfe.StrongWolfeLineSearch(F[-1],
                                                xk,
                                                Gk,
                                                pk,
                                                model.ls_F_Fp,
                                                stp=t0,
                                                **ls_args)
        MT.search()
        p = np.argsort(np.array(MT.xs).reshape((-1,)))
        #with np.printoptions(precision=4):
        #    print(np.array(MT.xs).reshape((-1,))[p])
        #    print(np.array(MT.fs).reshape((-1,))[p])
        #print
        #plt.plot(np.array(MT.xs).reshape((-1,)))
        if verbose > 3:
            print("Starting |Fp|:", np.linalg.norm(Fp[-1]))
            print("----- LS --t0-beta-", t0, Beta[-1])
            print("X", MT.xs)
            print("F", MT.fs)
            print("dF/dt", MT.slopes)
            print()
            print("Final t, F, dF/dt", MT.stp, MT.fs[-1], MT.slopes[-1])
            print()

        model.ls_finalize(MT.xs[-1])
        F.append(MT.fs[-1])
        Fp.append(model.Fp())
        if k > 0:
            del Fp[0] #Pop last so this list doesn't grow
        Fp_nrm.append(np.linalg.norm(Fp[-1]))
        ts.append(MT.stp)
        k += 1
        t.append(time.time())
    if verbose > 2:
        plt.subplot(1, 2, 1)
        if fmin == None:
            fmin = F[-1]
        plt.plot([f - fmin + 1e-16 for f in F], '-o')
        plt.yscale('log')
        plt.title('F')

        plt.subplot(1, 2, 2)
        plt.plot(Fp_nrm, '-o')
        plt.yscale('log')
        plt.title('|dF|')
        plt.show()
    #print "CG:", len(Fp), [np.linalg.norm(fp) for fp in Fp]

    return {'F': F, 'Fp_nrm': Fp_nrm, 'ts': ts, 'iter': k, 'times': np.array(t) - t[0]}


"""    model: encode context and objective function for a sequence of conjugate gradient line searches
    
    At iteration k the model is at a "base point" I will denote by 'xk'. Line searches from the basepoint are of the form
    
        F(xk + t*Hk),  dF/dt = F'.Hk
    
    However, for convenience


"""


class disentangler_model(object):
    def __init__(self, psi, n, pt=0.5):

        self.n = n  #Renyi index
        self.psi_k = psi.copy()
        chi = self.chi = psi.shape
        self.U = np.eye(chi[1] * chi[2], dtype=psi.dtype)  #Unitary accumulated
        self.total = 0
        self.pt = pt
        self.F_k, self.Fp_k = dSn(self.psi_k, self.n)

    def ls_setup(self, H):
        """Get ready for a line search U(-i t H) S
            ls_F(t) : Fitness of U(-i t H) S
            ls_Fp(t) : partial_t F(t) (directional derivative)
        """

        if H.dtype == np.float:
            self.real = True
        else:
            self.real = False

        self.H = H
        self.Id = np.eye(H.shape[0], dtype=H.dtype)
        self.l, self.w = np.linalg.eigh(-1j * H)

        chi = self.chi
        self.psi_k = self.psi_k.transpose([1, 2, 0,3]).reshape([chi[1] * chi[2], -1])
        self.finalized = False
        self.t = 0
        return np.array([0.]), np.array([np.real(np.vdot(self.Fp_k, self.H))]), np.array([1.])

    def ls_F_Fp(self, t):
        """ ls_F(t) : Fitness of U(i t H) S
            
            ls_Fp(t) : partial_t F(t) (directional derivative)
        """
        t = t[0]
        chi = self.chi

        u = np.dot(self.w * np.exp(1j * self.l * t), self.w.T.conj())
        if self.real:
            u = u.real
        psi_t = np.dot(u, self.psi_k)

        psi_t = psi_t.reshape([chi[1], chi[2], chi[0], chi[3]])
        psi_t = psi_t.transpose([2, 0, 1, 3])

        #Calculate derivative
        S, dS = dSn(psi_t, self.n)
        self.total += 1
        self.t = t
        self.F_t, self.Fp_t, self.u_t, self.psi_t = S, dS, u, psi_t
        #print "C", t, np.vdot(dS, self.H)
        return S, np.array([np.real(np.vdot(dS, self.H))])

    def ls_finalize(self, t):
        """ S_{k+1} = U(-i t H) S_k"""
        t = t[0]

        if t != self.t:
            print(t, self.t)
            raise NotImplemented
        if t != 0:
            self.F_k, self.Fp_k, self.psi_k = self.F_t, self.Fp_t, self.psi_t
            self.psi_k /= np.linalg.norm(self.psi_k)
            self.U = np.dot(self.u_t, self.U)
        else:
            chi = self.chi
            psi_k = self.psi_k.reshape([chi[1], chi[2], chi[0], chi[3]])
            self.psi_k = psi_k.transpose([2, 0, 1, 3])

        u = np.dot(self.w * np.exp(1j * self.l * t * self.pt), self.w.T.conj())

        if self.real:
            u = u.real

        self.sqrtu = u
        self.finalized = True

    def ls_transport(self, v):
        """Transport a tangent vector 'v' in u(N) from S_k ---> S_{k+1}"""
        if not self.finalized:
            raise ValueError
        
        return np.dot(np.dot(self.sqrtu, v), self.sqrtu.T.conj())

    def F(self):
        if self.F_k is None:
            self.F_k, self.Fp_k = dSn(self.psi_k, self.n)
            self.total += 1
        return self.F_k

    def Fp(self):
        if self.Fp_k is None:
            self.F_k, self.Fp_k = dSn(self.psi_k, self.n)
            self.total += 1
        return self.Fp_k


class disentangler_model_new(object):

    def __init__(self, psi, n, pt=0.5, mode = 'S'):

        self.n = n  #Renyi index
        self.psi_k = psi.copy()
        chi = self.chi = psi.shape
        self.U = np.eye(chi[1] * chi[2], dtype=psi.dtype)  #Unitary accumulated
        self.total = 0
        self.pt = pt
        self.mode = mode
        
        self.Xshp = self.U.shape
        
        self.H_k = renyi_hessian.Renyi_Hessian(n, psi, mode)
        self.F_k, self.Fp_k = self.H_k.get_F_dF()
        #self.F_k, self.Fp_k = dSn(self.psi_k, self.n, mode)

    def ls_setup(self, X):
        """Get ready for a line search U t X) S
            ls_F(t) : Fitness of U(t X) S
            ls_Fp(t) : partial_t F(t) (directional derivative)
        """

        if X.ndim!=2:
            X = X.reshape(self.Xshp)
        
        if X.dtype == np.float:
            self.real = True
        else:
            self.real = False

        self.X = X
        self.l, self.w = np.linalg.eigh(-1j * X)

        chi = self.chi
        self.psi_k = self.psi_k.transpose([1, 2, 0, 3]).reshape([chi[1] * chi[2], -1])
        self.finalized = False
        self.t = 0
        return np.array([0.]), np.array([np.real(np.vdot(self.Fp_k, self.X))]), np.array([1.])

    def ls_F_Fp(self, t, ls = True):
        """ ls_F(t) : Fitness of U(t X) S
            
            ls_Fp(t) : partial_t F(t) (directional derivative)
        """
        t = t[0]
        chi = self.chi

        u = np.dot(self.w * np.exp(1j * self.l * t), self.w.T.conj())
        
        if self.real:
            u = u.real
        
        psi_t = np.dot(u, self.psi_k)

        psi_t = psi_t.reshape([chi[1], chi[2], chi[0], chi[3]])
        psi_t = psi_t.transpose([2, 0, 1, 3])

        #Calculate derivative
        #S, dS = dSn(psi_t, self.n, self.mode)
        self.H_t = renyi_hessian.Renyi_Hessian(self.n, psi_t, self.mode, prep_Hessian = True)
        S, dS = self.H_t.get_F_dF()
        
        self.total += 1
        self.t = t
        self.F_t, self.Fp_t, self.u_t, self.psi_t = S, dS, u, psi_t

        if ls:
            return S, np.array([np.real(np.vdot(dS, self.X))])
        else:
            return S, dS

    def ls_finalize(self, t):
        """ S_{k+1} = U(t X) S_k"""
        t = t[0]

        if t == 0.:
            chi = self.chi
            psi_k = self.psi_k.reshape([chi[1], chi[2], chi[0], chi[3]])
            self.psi_k = psi_k.transpose([2, 0, 1, 3])
            self.sqrtu = np.eye(len(self.w))
        elif t != self.t:
            print(t, self.t)
            raise NotImplemented
        if t != 0:
            self.F_k, self.Fp_k, self.psi_k = self.F_t, self.Fp_t, self.psi_t
            #self.H_k = renyi_hessian.Renyi_Hessian(self.n, self.psi_k, self.mode)
            self.H_k = self.H_t
            #self.H_k.compute_cache_objs()
            self.psi_k/= np.linalg.norm(self.psi_k)
            self.U = np.dot(self.u_t, self.U)
        
            u = np.dot(self.w * np.exp(1j * self.l * t * self.pt), self.w.T.conj())
            if self.real:
                u = u.real
            self.sqrtu = u

        self.X = self.u_t = self.psi_t = self.H_t = self.F_t = self.Fp_t = None
        self.finalized = True

    def ls_transport(self, v):
        """Transport a tangent vector 'v' in u(N) from S_k ---> S_{k+1}"""
        if not self.finalized:
            raise ValueError
        if v.ndim!=2:
            v = v.reshape(self.Xshp)
        return np.dot(np.dot(self.sqrtu, v), self.sqrtu.T.conj()).reshape((-1,))


    def H(self):
        if self.H_k is None:
            print "Why in H?"
            self.H_k = renyi_hessian.Renyi_Hessian(self.n, self.psi_k, self.mode)
        return self.H_k
    
    def F(self):
        if self.F_k is None:
            print "Why here in F?"
            self.F_k, self.Fp_k = dSn(self.psi_k, self.n, self.mode)
            self.total += 1
        return self.F_k

    def Fp(self):
        if self.Fp_k is None:
            print "Why here in Fp?"
            self.F_k, self.Fp_k = dSn(self.psi_k, self.n, self.mode)
            self.total += 1
        return self.Fp_k
