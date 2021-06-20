import numpy as np

class DSDR:
    """Z He, et al. Document Summarization based onData Reconstruction (2012)
    http://www.aaai.org/ocs/index.php/AAAI/AAAI12/paper/viewPaper/4991
    """
        
    @staticmethod
    def lin(V, m, lamb):
        '''DSDR with linear reconstruction

        Parameters
        ==========
        - V : 2d array_like, the candidate data set
        - m : int, the number of sentences to be selected
        - lamb : float, the trade off parameter

        Returns
        =======
        - L : list, the set of m summary sentences indices
        '''
        L = []
        B = np.dot(V, V.T) / lamb
        n = len(V)
        for t in range(m):
            scores = []
            for i in range(n):
                score = np.sum(B[:,i] ** 2) / (1. + B[i,i])
                scores += [(score, i)]
            max_score, max_i = max(scores)
            L += [max_i]
            B = B - np.outer(B[:,max_i], B[:,max_i]) / (1. + B[max_i,max_i])
        return L

    @staticmethod
    def non(V, gamma, eps=1.e-8):
        '''DSDR with nonnegative linear reconstruction
        
        Parameters
        ==========
        - V : 2d array_like, the candidate sentence set
        - gamma : float, > 0, the trade off parameter
        - eps : float, for converge

        Returns
        =======
        - beta : 1d array, the auxiliary variable to control candidate sentences
            selection
        '''
        V = np.array(V)
        n = len(V)
        A = np.ones((n,n))
        beta = np.zeros(n)
        VVT = np.dot(V, V.T) # V * V.T
        np.seterr(all='ignore')
        while True:
            _beta = np.copy(beta)
            beta = (np.sum(A ** 2, axis=0) / gamma) ** .5
            while True:
                _A = np.copy(A)
                A *= VVT / np.dot(A, VVT + np.diag(beta))
                A = np.nan_to_num(A) # nan (zero divide by zero) to zero
                if np.sum(A - _A) < eps: break
            if np.sum(beta - _beta) < eps: break
        return beta

if __name__ == '__main__':
    pass
