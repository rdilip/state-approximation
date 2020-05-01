import numpy as np
import scipy as sp
from misc import svd
from disentangler import renyi_hessian
import matplotlib.pyplot as plt
import ls
import time

def disentangle_newton(psi, n, eps = 1e-6, max_iter = 10, verbose = 0, TR_par = {},  cg_par = {}, do_NLCG = True, isNLCG = False, mode='S'):
    """Takes in
       
       psi_{a, bl, br, c}
       n = renyi index
       mode = 'S' or 'Z'
       
       max_iter = max # of newton (outer) iterations
       
       eps = stopping criteria,   |dS_fin| < eps*|dS_0|
       
       cg_par = {'max_iter': # of iterations for CG solution of newton equation (controls how many matvec)
               }
       
       TR_par = {} #trust region parameters
       Returns psi, U, info = {}
    """
    
    D = disentangler_model(psi, n = n, mode = mode, pt = 0.) #this thing stores context, e.g. current guess psi_k and U_k
    
    cg_max_iter = cg_par.setdefault('max_iter', None)

    TR = TR_par.setdefault('TR', 1.) #initial trust region
    TR_max = TR_par.setdefault('TR_max', 6.)
    gamma3 = TR_par.setdefault('gamma3', 4/3.) #how much to expand TR for good step
    eta1 = TR_par.setdefault('eta1', 0.25) #  Q < eta1 : bad step
    eta2 = TR_par.setdefault('eta2', 0.75) #  Q > eta2 : good step
    
    f = D.F() #Func value
    g = D.Fp().reshape((-1,)) #Derivative
    h = D.H() #Hessian
    
    fs = [f]
    gs = [np.linalg.norm(g)]

    p_last = None  #This will be for seeding Newton steps (do_NLCG)
    beta = None
    
    compute_T = False
    
    go = True
    k = 0
    cg_eval = 0
    t = [time.time()]
    while go and k < max_iter:
        if verbose:
            print
            print "Iteration:", k
        
        #Newton step  s = - H^{-1} g, subject to TR constraint
        s, cg_info = trust_region_cg(h, -g, trust_region = TR, max_iter = cg_max_iter,
                                     tol = np.min([0.05, np.linalg.norm(g)**0.5]), compute_T = compute_T, p0 = p_last)
        cg_eval+=cg_info['k']
        if verbose > 1 and p_last is not None:
            print "dS:", cg_info['dS'][0], np.sum(cg_info['dS'])
        S = np.linalg.norm(s) #Length of step
        sg = np.vdot(s, g) #Amount of descent
        shs = -np.vdot(s, cg_info['r']) - sg #Curvature <s, H s>
        dF_pred = (sg + 0.5*shs)
        if sg > 0.:
            print sg, shs
            raise ValueError
        if verbose:
            print "CG:", cg_info['code'], "@", cg_info['k'], " |r|:", cg_info['residuals'][-1]
        
        D.ls_setup(s) #Get ready to potentially search along direction s
        
        if verbose > 1: #Plot model vs objective
            ts = np.linspace(0, 1., 30)
            Ss = []; Gs = []
            for ti in ts:
                fl, gl = D.ls_F_Fp([ti])
                Ss.append(fl); Gs.append(gl)
            print "Exp:", fs[-1], sg, 0.5*shs
            print "Fit:", np.polyfit(ts, Ss, 4)[::-1] #Quartic model
            plt.plot(S*ts, Ss)
            plt.plot(S*ts, Ss[0] + sg*ts + 0.5*shs*ts*ts, '--')

        fn, gn = D.ls_F_Fp([1.], ls = False) #New f, g at x_{k+1} = x_k + 1*s
        gn = gn.reshape((-1,))
        gn_norm = np.linalg.norm(gn)
        dF = (fn - f)
        Q = dF/dF_pred
        if verbose:
            print "F", fs[-1], "-->", fn,   "|G|", gs[-1], "--->", gn_norm
            print "Q, dF_act, dF_pred, ", Q, dF, dF_pred
        
        #Bad step or negative curvature. Do a line search to help figure out new TR
        if dF_pred > -1e-15 and (Q < eta1 or shs < 0.):
            go = False
            if dF < 0.:
                D.ls_finalize([1.])
                hn = D.H()
            else:
                D.ls_finalize([0.])
                fn = f
                gn = g
                hn = h
        elif isNLCG or Q < eta1 or shs < 0.:
            sgn = np.vdot(gn, s) #Gradient at new iterate
            
            errF = dF - dF_pred #Error in F, G wrt quadratic model
            errG = sgn - (sg + shs)
        
            #Find coefficients of quartic model  M(s) = M_quad(s) + a3 s^3 + a4 s^4
            a3 =  (4*errF - errG)
            a4 = -(3*errF - errG)
            if  verbose:
                print "Line search!  a3, a4:", a3, a4
            #Find extrema of M(s)
            roots = np.roots([4*a4, 3*a3, shs, sg]) #TODO replace this with Cardano :)
            roots = np.sort(roots[np.abs(roots.imag) < 1e-15].real)
            if verbose:
                print "roots:", roots
            roots = roots[roots > 0.]
            if len(roots):
                stp = roots[0]
            else: #This case should only show up if  a4 < 0
                stp = 2.
                print "No roots!", sg, shs, a3, a4
            
            #Line search along "s"
            step_sign = np.sign(stp)
            MT = ls.pymswolfe.StrongWolfeLineSearch(fs[-1],
                                                np.array([0.]), #x
                                                np.array([sg]), #G
                                                np.array([1.]), #p
                                                D.ls_F_Fp,
                                                gtol = 0.1, #We want this criteria relative to before
                                                stp=stp,
                                                stpmax = 1.)

            MT.search()
            if  verbose:
                print "step pred, act:", stp, MT.stp
                print "t:", np.array(MT.xs).reshape((-1,))
                print "S(t):", MT.fs
                print "dS/dt:", MT.slopes
            
            #Take the step
            D.ls_finalize([MT.stp])
            fn = D.F()
            gn = D.Fp().reshape((-1,))
            gn_norm = np.linalg.norm(gn)
            hn = D.H()
            
            if verbose > 1:
                plt.plot([S*MT.stp], [fn], 'o')
    
        else:
            D.ls_finalize([1.])
            hn = D.H()
        
        if Q < eta1:
            #Adjust TR based on result of line search
            TR_old = TR
            TR = np.min( [TR, S*(0. + step_sign*MT.stp)])
            if verbose:
                print("TR:", TR_old, " --> ", TR)
        elif Q > eta2:
            #Good step; grow if S*gamma3 > TR
            TR_old = TR
            TR = np.min([np.max( [S*gamma3, TR]), TR_max])
            if verbose:
                print("TR:", TR_old, " --> ", TR)

        #print info['e_est'], info['residuals'][-1]

        f = fn
        g = gn
        h = hn
        
        fs.append(f)
        gs.append(gn_norm)
        
        if do_NLCG:
            if isNLCG or (cg_info['k'] < g.size and Q > eta1 and shs > 0):
                p_last = cg_info['p_last']
                if isNLCG or cg_info['code'] == 'converged':
                    beta = np.vdot(g, g + cg_info['r_prev']) / cg_info['residuals'][-2]**2
                    if verbose:
                        print ("beta:", beta)
                    if beta > 0.:
                        p_last = -g + beta*p_last
                        sp = np.vdot(g, p_last).real
                        if sp > 0.:
                            p_last = None
                    else:
                        p_last = None
            else:
                p_last = None
        else:
            p_last = None

        if gs[-1] < eps or (fs[-2] - fs[-1]) < 1e-15:
            go = False
        
        k+=1
        t.append(time.time())


    info = {'Ss': np.array(fs), '|Gs|': np.array(gs), 'h': h, 'g':g, 'times': np.array(t) - t[0], 'eval':D.total, 'k':k, 'cg_eval':cg_eval}
    #s, info = nd.trust_region_cg(h, -g/np.linalg.norm(g), max_iter = 50,  compute_T = True)
    #print "Estimate of spec(H):", info['e_est']
    if compute_T:
        info.update({'e_est':cg_info['e_est']})
    return D.psi_k, D.U, info
    
def trust_region_cg(A, b, x0 = None, tol = 1e-8, shift = 0., eig_bound = 0., trust_region=None, max_iter = None, compute_T = False, p0 = None):
    """
        Truncated CG  (Toint-Steihaug) solution of  A x = b, subject to |x| < trust_region
       
        Approximately minimizes   min_x 1/2 x A x - b  subject to |x| < trust_region
       
       A: a linear operator implementing matvec
       b: numpy array
       x0: initial guess; most properties are proved assuming x0 = 0.
       tol: iterate until  |A x  - b | < tol*|A x0  - b |
       eig_bound: if curvature is detected with pAp/pp <= eig_bound, it is treated as negative curvature and CG is truncated.
       max_iter: at most max_iter CG steps. If max_iter==None, truncates at k = size(x)
       shift: implicitly replace A --> A + shift
       compute_T: whether to compute and diagonalize the tridiagonal approximation of A,  V A V^D =  T, as per Lanczos
       
       Returns:
       
        x, info
        
        info{   'k': number of iterations,
                'eig_bounds': list of  <p|A|p>/<p|p> which upper & lower bound the spectrum of A,  l_min <= eig_bounds <= l_max
                'residuals': |A x - b| at each step
                
             If compute_T:
             
                'T': k x k tridiagonal Krylov approximation of A
                'e_est': Krylov spectrum of T
                'v_est': Krylov eignvectors, so e  = v^D A v
            }
       
    """
    
    if max_iter is None:
        max_iter = b.shape[0]
    
    if x0 is None:
        x = np.zeros((A.shape[0],))
        xx = 0.
        r = b
    else:
        x = x0
        xx = np.vdot(x, x).real #Length of x
        r = b - A._matvec(x).reshape((-1,)) - shift*x

    rr = np.vdot(r, r).real

    if p0 is not None:
        rp = np.vdot(r, p0)
        if rp < 0.:
            p0 = -p0
            rp = -rp
        p = p0
        pp = np.vdot(p, p).real
        pp0 = pp
        P0 = True
    else:
        p = r
        pp = rr
        P0 = False

    k = 0
    go = True
    
    eig_bounds = []
    residuals = [np.sqrt(rr)]
    alphas = []
    betas = []
    gammas = []
    #x_{j+1} = x_j + alpha_j p_j
    #r_j = b - A x_j
    #r_{j+1} = r_j - alpha_j A p_j
    #beta_j = |r_{j}|^2 / |r_{j-1}|^2
    #p_{j} = r_j + beta_{j-1} p_{j-1}
    #alpha_j = |r_j|^2  / <p_j A p_j>

    rjs = [r]
    while go:
        Ap = A._matvec(p).reshape((-1,)) + shift*p
        pAp = np.vdot(p, Ap).real
        gammas.append(pAp)
        eig_bounds.append(pAp/pp)
        
        if k==0 and P0:
            P0 = Ap/pAp
            alpha = rp/pAp
        else:
            alpha = rr/pAp
        alphas.append(alpha)
        
        if eig_bounds[-1] <= eig_bound:
            #print "Negative-curvature: pAp, pAp/pp", pAp, eig_bounds[-1], "at k = ", k
            go = False
            code = 'negative curvature'
            if trust_region:
                px = np.vdot(p, x).real
                d = px*px - pp*(xx - trust_region**2)
                alpha = (-px + np.sqrt(d))/pp
                #print "Hit trust region boundary: alpha'/alpha:", alpha/alphas[-1], "at k = ", k
            else:
                alpha = 0.

            #f(a) = 0.5*a^2 * pAp - a <p|r>
            #a  =  <p|r>/ <pAp>
            
        elif trust_region is not None:
            px = np.vdot(p, x).real
            xnxn = xx + 2*alpha*px + alpha**2*pp
            if xnxn > trust_region**2: #cutoff
                d = px**2 - pp*(xx - trust_region**2)
                alpha = (-px + np.sqrt(d))/pp
                #print "Hit trust region boundary: alpha'/alpha:", alpha/alphas[-1], "at k = ", k
                go = False
                code = 'trust boundary'
            else:
                xx = xnxn


        x = x + alpha*p
        r_prev = r
        r = r - alpha*Ap
        if compute_T:
            rjs.append(r)
        rr_ = np.vdot(r, r).real #new residual norm
        residuals.append(np.sqrt(rr_))

        #print rr +  beta*beta*pp,
        #pp = np.vdot(p, p) #| p_{j+1}|^2 = |r_{j+1}|^2 + beta^2_j | p_j |^2
        #print pp
        #print rr, pp, xx
        if k+1 >= max_iter:
            go = False
            code = 'max_iter'

        if residuals[-1] < tol*residuals[0]:
            go = False
            code = 'converged'

        if go:
            beta = rr_/rr
            betas.append(beta)
            rr = rr_

            if P0 is not False:
                beta0 = - np.vdot(P0, r)
                if k > 0:
                    p = r + beta*p + beta0*p0
                    pp = rr + pp*beta*beta + pp0*beta0*beta0
                else:
                    p = r + beta0*p0
                    pp = rr + pp0*beta0*beta0
            else:
                p = r + beta*p
                pp = rr + pp*beta*beta
        k+=1

    residuals = np.array(residuals)
    alphas = np.array(alphas)
    gammas = np.array(gammas)
    info = {'k': k, 'eig_bounds':eig_bounds, 'residuals':residuals, 'r': r, 'r_prev':r_prev, 'p_last':p, 'code':code, 'dS':0.5*alphas*alphas*gammas}
    
    if compute_T:
        #Form the triadiagonal Krylov rep for A. Notation is from Saad 201
        alphas = np.array(alphas)
        betas = np.array(betas)
        k_lanc = len(alphas)

        deltas = 1./alphas
        deltas[1:] +=betas/alphas[0:-1]
        etas = -np.sqrt(betas)/alphas[0:-1]
        
        T = np.diag(deltas[:k_lanc])
        for i in range(k_lanc-1):
            T[i, i+1] = etas[i]
            T[i+1, i] = etas[i]
        
        rjs_normed = (np.array(rjs[:k_lanc]).T/residuals[:k_lanc])
        e_est, v_est =  np.linalg.eigh(T)
        v_est = np.dot(v_est.T, rjs_normed[:, :k_lanc].T).T

        info.update({'e_est':e_est-shift, 'v_est':v_est, 'T':T, 'V': rjs_normed[:, :k_lanc] })
        
    return x, info

def dSn(psi, n=1, mode = 'S'):
    """ Returns H = dS / dU  and S for nth Renyi
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
    #print S, s, r, n
    Epsi = np.dot(X * r, Z).reshape(chi)
    dS = np.tensordot(psi, Epsi.conj(), axes=[[0, 3], [0, 3]])
    dS = dS.reshape((chi[1] * chi[2], -1))
    dS = (dS - dS.conj().T)
    
    if mode == 'Z':
        Z = np.exp((1-n)*S)
        return Z, dS*(1-n)*Z

    return S, dS

class disentangler_model(object):

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
            #self.H_k.compute_cache_objs() #TODO restore this once caching fixed
            self.psi_k/= np.linalg.norm(self.psi_k)
            self.U = np.dot(self.u_t, self.U)
            
            if self.pt:
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
        if self.pt:
            return np.dot(np.dot(self.sqrtu, v), self.sqrtu.T.conj())
        return v

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
