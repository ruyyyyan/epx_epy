## @package Lanczos This package is a very simple and straight-forward implementation of the
# Lanczos algorithm for sparse matrices.
#
# Usage: Create a LanczosSolver object with the desired properties (max. number of
# iterations, desired precision, invariant subspace detection threshold etc).
#
# Then either call the 'lanczos' method to compute ground state energy and ground
# state vector, or call the first lanczos pass individually and then explicitly
# diagonalize the resulting matrix to obtain spectral weights.'''

import numpy as np
import numpy.linalg as linalg
import scipy.sparse as sparse
import scipy.sparse.linalg as sp_la
import logging

## Return the adjunct of a matrix, which really is just
#  `conjugate(transpose(x))`.
def adj(x):
    return np.conjugate(np.transpose(x))

## This class represents an instance of a lanzcos solver with its own
#  set of parameters and intermediate states.
class LanczosSolver:

    ## Initialize the lanczos solver with its parameters.
    #
    #  @param kwargs The keywords to be used:
    #         - maxiter: Maximum number of iterations
    #         - precision: Desired tolerance for the lanczos
    #         - cond: If it's not "PRECISION", always run up to `maxiter`.
    #         - eps: Tolerance for detecting loss of orthogonality
    def __init__(self, **kwargs):
        self.maxiter = kwargs.get('maxiter')
        self.precision = kwargs.get('precision')
        self.cond = kwargs.get('cond')
        self.eps = kwargs.get('eps')
        
        '''
        Clears all computed quantities but keeps the parameters
        alpha and beta are a and b in Koch notes 
        But note that self.beta[j] = b_{j+1} of Koch's notes
        '''
        # previously use np.empty. See https://numpy.org/doc/stable/reference/generated/numpy.empty.html
        self.alpha = np.zeros(self.maxiter,dtype=complex)
        self.beta = np.zeros(self.maxiter-1,dtype=complex)
        self.m = None
        self.gse = None
        self.passed_first = False

    ## Clear all computed quantities but keep the parameters
    def reset(self):
        self.alpha = np.zeros(self.maxiter,dtype=complex)
        self.beta = np.zeros(self.maxiter - 1,dtype=complex)
        self.m = None
        self.gse = None
        self.passed_first = False
        
    ## The main function of this class, taking care of a full lanczos run.
    #
    #  @param kwargs Set of keyword arguments. Use
    #                - `x0`: Starting vector. Use something randomly initialized
    #                - `scratch`: Pre-allocated space for a scratch vector
    #                - `y`: Allocated space where the ground state vector will be stored
    #                - `H`: A sparse matrix representing the Hamiltonian
    #  @return ev The ground state eigenvalue of the Hamiltonian. Also returns the ground state
    #             vector in the numpy array `y`.
    def lanczos(self,**kwargs):
        start_vector = kwargs.get('x0')
        scratch_vector = kwargs.get('scratch')
        ground_state = kwargs.get('y')
        H = kwargs.get('H')

        if not abs(linalg.norm(start_vector) - 1) < self.eps:
            start_vector /= linalg.norm(start_vector)

        start_vector_copy = np.copy(start_vector)
        ev = self.lanczos_pass(mode = 'FIRST', x0 = start_vector_copy,
                          scratch = scratch_vector, y = ground_state, H = H)
        self.lanczos_pass(mode = 'SECOND', x0 = start_vector,
                     scratch = scratch_vector, y = ground_state, H = H)
        return ev

    ## Only perform the first pass of the Lanczos algorithm
    def first_pass(self,**kwargs):
        kwargs['mode'] = 'FIRST'
        self.lanczos_pass(**kwargs)
        
    ## Perform a pass (first or second) of the Lanczos algorithm
    def lanczos_pass(self,**kwargs):
        '''
        Following Koch's notes but important different here:
        self.beta[j] = b_{j+1} of Koch's notes
        '''
        mode = kwargs.get('mode',"FIRST")
        b = kwargs.get('x0')
        q = kwargs.get('scratch')
        r = kwargs.get('y')
        H = kwargs.get('H')
        y = None

        #Don't allow second pass to be called if first pass wasn't.
        assert mode == 'FIRST' or self.passed_first
        
        if mode == 'SECOND':
            tmp = np.diag(self.alpha[0:self.m]) + (np.diag(self.beta[0:self.m-1],k=1) +
                                                     np.diag(self.beta[0:self.m-1],k=-1) )
            V,D = linalg.eigh(tmp)
            indices = np.argsort(V)
            y = D[:,indices[0]]
            norm = np.conjugate(y).dot(y)
            y = y / np.sqrt(norm)
        q[:] = 0
        if mode == 'SECOND':
            r[:] = 0
        
        j = 0
        gse_old = 0
        gse_new = 0
        while (mode == 'FIRST' and j < self.maxiter) or (mode == 'SECOND' and j < self.m):
            if not j == 0:
                b *= -self.beta[j-1]
                q *= (1.0/self.beta[j-1])
                tmp = b
                b = q
                q = tmp

            if mode == 'SECOND':
                r += (y[j] * b)

            # compute H|v0>, H^2|v0> etc.
            q += H.dot(b)

            if mode == 'FIRST':
                self.alpha[j] = adj(b).dot(q)

            q += (-self.alpha[j] * b)
            

            if j < self.maxiter - 1 and mode == 'FIRST':
                self.beta[j] = linalg.norm(q)
                
                # for this case, see Koch's notes (discussion on Fig.3)
                if mode == 'FIRST' and self.beta[j] < self.eps:
                    print ("Beta very small: Invariant sub-space reached after ", j, " iterations")
                    break
            j += 1
            
            #Now diagonalize the tridiagonal symmetric matrix defined
            if (self.cond == 'PRECISION' or j == self.maxiter) and mode == 'FIRST':
                
            # Can try to monitor the relative error with No of lanczos iterations:
            # In this way, self.precision is useful. Otherwise, always maxiter of iterations
            # and then the tridiagonal matrix is diagonalized in lanczos_diag_T
            #if mode == 'FIRST':
                if j == self.maxiter and self.cond == 'PRECISION':
                    logging.warning("Warning: Max number of iterations reached")
                if j > 0:
                    gse_old = gse_new

                tmp = np.diag(self.alpha[0:j]) + (np.diag(self.beta[0:j-1],k=1) +
                                                        np.diag(self.beta[0:j-1],k=-1) )

                V,D = linalg.eigh(tmp)
                gse_new = V.min()
                ev = gse_new
                self.gse = gse_new

                if j > 1:
                    self.relative_error = abs((gse_old - gse_new) / gse_new)
                    #print('gse_old - gse_new error = ', self.relative_error)
                    logging.debug("error: %4.4g" % self.relative_error)
                    if self.relative_error < self.precision:
                        break
        
        self.m = j
        #print ('No of lanczos iterations m = ', self.m)
        
        #print('self.alpha =', self.alpha[0:j])
        #print('self.beta  =', self.beta[0:j])
        
        self.passed_first = True
        
        return self.gse

    
    ## Diagonalize the tridiagonal matrix `T` generated in the first pass of the Lanczos.
    def lanczos_diag_T(self):
        '''
        After completing the first pass, we have the tridiagonal matrix "T"
        the represents the Hamiltonian H in the Krylov subspace. This method
        diagonalizes T, returning the eigenvalues in the first and the
        eigenvectors in the second argument
        Following Koch's notes but important different here:
        self.beta[j] = b_{j+1} of Koch's notes
        '''
        m = self.m
        assert not m == 0
        tmp = np.diag(self.alpha[0:m-1]) + (np.diag(self.beta[0:m-2],k=1) +
                                            np.diag(self.beta[0:m-2],k=-1) )
        V, D = linalg.eigh(tmp)
        #Sort
        indices = np.argsort(V)
        V = V[indices]
        D = D[:,indices]
        return V,D

    def lanczos_invert_T(self,E,eta):
        m = self.m
        assert not m == 0
        tmp = (np.diag(E + 1j*eta - self.alpha[0:m-1]) 
               - np.diag(self.beta[0:m-2],k=1) 
               - np.diag(self.beta[0:m-2],k=-1) )
        
        #tmp_sparse = sparse.csr_matrix(tmp)
        b = np.zeros(m-1,dtype=complex)
        b[0] = 1
        x = np.linalg.solve(tmp, b)
#        x = sp_la.spsolve(tmp_sparse, b)
        return x[0]

    ####################################################################
    '''
    Below is implementation of Mona's notes on Oct.6, 2022 (see email)
    for computing off-diagonal <w0| G |v0>
    Directly compute continued fraction instead of generating
    tridiagonal matrix to get En and eigenvectors
    '''
    ####################################################################
    ## Perform a pass (first or second) of the Lanczos algorithm
    def compute_anbncn(self, **kwargs):
        '''
        For calculating off-diagonal <w0|G|v0>
        similar to and modified from lanczos_pass
        
        Important difference from lanczos_pass where
        self.beta[j] = b_{j+1} of Koch's notes
        
        here an and bn correspond to lanczos, and cn stores <w0|v0>
        But now redefine bn to have same len as an !!!
        because b0 is also useful for using Mona's formula <w0|G(z)|v0> = f_0(z)
        
        Also note b = start_vector is for pointer's =
        changing b will modify start_vector
        '''
        mode = kwargs.get('mode',"FIRST")
        b = kwargs.get('x0')
        e = kwargs.get('w0')
        q = kwargs.get('scratch')
        r = kwargs.get('y')
        H = kwargs.get('H')
        y = None
        
        an = np.zeros(self.maxiter,dtype=complex)
        bn = np.zeros(self.maxiter,dtype=complex)
        cn = np.zeros(self.maxiter,dtype=complex) 
        
        cn[0] = adj(e).dot(b)
        
        # b_n^2=<vn'|vn'> with |vn'> is unnormalized but b0 is special
        # because v0 is almost assumed to be already normalized
        bn[0] = 1.0
    
        #Don't allow second pass to be called if first pass wasn't.
        assert mode == 'FIRST' or self.passed_first
        
        q[:] = 0
        
        j = 0
        gse_old = 0
        gse_new = 0
        while (mode == 'FIRST' and j < self.maxiter) or (mode == 'SECOND' and j < self.m):
            if not j == 0:
                b *= -bn[j]
                q *= (1.0/bn[j])
                
                # swap b and q (see Koch's Table 1)
                tmp = b
                b = q
                q = tmp

            # compute H|v0>, H^2|v0> etc.
            q += H.dot(b)

            if mode == 'FIRST':
                an[j] = adj(b).dot(q)
                #print('j=',j, 'self.alpha =', self.alpha[j])

            q += (-an[j] * b)

            if j < self.maxiter - 1 and mode == 'FIRST':
                bn[j+1] = linalg.norm(q)
                #print('j=',j, 'self.beta =', self.beta[j])
                
                if j>0:
                    cn[j] = adj(e).dot(q)/bn[j+1]
                
                # for this case, see Koch's notes (discussion on Fig.3)
                if mode == 'FIRST' and bn[j+1] < self.eps:
                    print ("Beta very small: Invariant sub-space reached after ", j, " iterations")
                    break
            j += 1

            #Now diagonalize the tridiagonal symmetric matrix defined
            if (self.cond == 'PRECISION' or j == self.maxiter) and mode == 'FIRST':
                
            # Can try to monitor the relative error with No of lanczos iterations:
            # In this way, self.precision is useful. Otherwise, always maxiter of iterations
            # and then the tridiagonal matrix is diagonalized in lanczos_diag_T
            #if mode == 'FIRST':
                
                if j == self.maxiter and self.cond == 'PRECISION':
                    logging.warning("Warning: Max number of iterations reached")
                if j > 0:
                    gse_old = gse_new

                tmp = np.diag(an[0:j]) + (np.diag(bn[1:j],k=1) +
                                                        np.diag(bn[1:j],k=-1) )

                V,D = linalg.eigh(tmp)
                gse_new = V.min()
                ev = gse_new
                self.gse = gse_new

                if j > 1:
                    self.relative_error = abs((gse_old - gse_new) / gse_new)
                    #print('gse_old - gse_new error = ', self.relative_error)
                    logging.debug("error: %4.4g" % self.relative_error)
                    if self.relative_error < self.precision:
                        break
                        
        
        print ('No of lanczos iterations m = ', j)
        
#         print('an   =', an[0:j])
#         print('bn   =', bn[0:j])
#         print('w0vn =', cn[0:j])
        
        self.m = j
        self.passed_first = True
        
        # m iterations so that m values. see Koch's notes
        # note that when m reaches the Lanczos_maxiter, it will automatically cutoff j+1
        return an[0:j], bn[0:j], cn[0:j]
    
    
    def getAnfn(self, zvals, n, an, bn, cn):
        '''
        recursive function to obtain A_0 and f_0
        see Mona's notes and email on Oct.6, 2022
        m is L+1 of Koch's notes
        alpha and beta are a_n and b_n separately
        cn = <w0|vn> as an additional quantity for calculating <w0|G|v0>
        
        Note that here following above compute_anbncn
        b0 is also stored so that 
        NOT self.beta[j] = b_{j+1} of Koch's notes as in lanczos_pass
        '''
        m = self.m   
        An = np.zeros(len(zvals))
        fn = np.zeros(len(zvals))

        if n==m-1:
            factor = zvals-an[n]
            An = bn[n] / factor
            fn = cn[n] / factor
            
            return An, fn
        
        else:
            A, F = self.getAnfn(zvals, n+1, an, bn, cn)
            
            factor = zvals-an[n]-bn[n+1]*A
            An = bn[n] / factor
            fn = (cn[n]+bn[n+1]*F) / factor
            
            return An, fn
    