import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
# %matplotlib inline
import quantecon as qe
import scipy.optimize as optimize
import scipy.sparse as sparse
from quantecon import compute_fixed_point
from quantecon.markov import DiscreteDP
from Leapfrogging import leapfrogging
lf = leapfrogging()
import copy

class rls_class:
    # def __init__(self):
    
    def solve(self,G,ss,ESS0, stage_index):

        rlsp = {'maxEQ':np.NaN, 'print':np.NaN}
        rlsp['maxEQ']  =  10 # 50000 maximum number of iterations
        rlsp['print']  =  100  # 500 print every rlsp.print equilibria (0: no print, 1: print every, 2: print every second)

    #     % initialize matrices
        TAU = np.empty(rlsp['maxEQ'])
        TAU[:] = np.nan
        tau = np.size(stage_index) # start RLS at last stage
        iEQ = 0
        ESS = np.empty(rlsp['maxEQ'],dtype=object)
        ESS[:] = copy.deepcopy(ESS0)
        out = np.empty(rlsp['maxEQ'])
        out[:] = np.NaN
        while iEQ < rlsp['maxEQ']:
            TAU[iEQ] = tau
            ss, ESS[iEQ] = G(ss,ESS[iEQ], tau)

            if (np.mod(iEQ,rlsp['print'])==0):
                print(f'ESR[{iEQ}][\'esr\']  :')
                print(ESS[iEQ-1]['esr'])
                print(f'ESR[{iEQ}][\'bases\']:')
                print(ESS[iEQ-1]['bases'])

            # if nargout>2: # if there are less than 2 outputs xxx
            try:
                np.insert(out,iEQ,self.output(ss, ESS[iEQ]))
                print(f'we are in the try now at {iEQ} ')
            except:
                pass
            ESS[iEQ+1] = self.addOne(copy.deepcopy(ESS[iEQ]))
            changeindex_temp = np.nonzero((ESS[iEQ+1]['esr']-ESS[iEQ]['esr'])!=0)[0]
            changeindex = min(changeindex_temp)
            count_index = 0
            for i in stage_index:
                if changeindex<=i:
                    count_index += 1
            tau = count_index-1 #% tau0 is found
            if np.all(ESS[iEQ+1]['esr']==-1):
                break

            iEQ += 1
            # end % End of recursive lexicographical search
        TAU=TAU[1:iEQ]
    
        return ESS, TAU, out

    def output(self,ss, ESS):
        out = {}
        out['MPEesr'] = ESS['esr']
        out['V1'] = max([ss[1]['EQs'][1,1,1]['eq']['vN1'],ss[1]['EQs'][1,1,1]['eq']['vI1']]) #  xxx perhabs change ss[1] to 0
        out['V2'] = max([ss[1]['EQs'][1,1,1]['eq']['vN2'],ss[1]['EQs'][1,1,1]['eq']['vI2']]) #  xxx perhabs change ss[1] to 0
        return out 

    def addOne(self,addESS):
        # %if x[1,1,1] == -1
        # %    throw(error("This ESS has already overflown!"))
        # %end
        n = len(addESS['esr'])
        X = np.zeros(n)
        R = 1
        for i in np.flip(np.arange(n)):
            X[i] = int(np.mod(addESS['esr'][i] + R,addESS['bases'][i]+1)) # xxx added 1 as to not divide by 0
            # % mod(a,b) does division and returns the remainder given as a-div(a,b)*b
            R = lf.div(addESS['esr'][i] + R,addESS['bases'][i]+1)
            # % div(a,b) does division and truncates - rounding down - to nearest integer  .... floor(a/b)
        if R > 0:
        # % When exiting the loop R > 0 occurs when all ESS.number is max allowed
        # % which is 1 below the base.
        # % println("No more equilibria to check.")
            addESS['esr'] = -1*np.ones(n)
        else:
            addESS['esr'] = X
        return addESS