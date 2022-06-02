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
    # def __init__(self): # not really defining any parameters in rls so ommited it. might have to include it in the future.
    

    def solve(self,G,ss,ESS0, stage_index):

        rlsp = {'maxEQ':np.NaN, 'print':np.NaN}
        rlsp['maxEQ']  =  200000 # maximum number of iterations
        rlsp['print']  =  1000  # print every x equilibria

        # define matrices to store results
        TAU = np.empty(rlsp['maxEQ']+1) # number of TAU +1 for python indexing
        TAU[:] = np.nan # start as nan and fill out
        tau = np.size(stage_index)  # start RLS at last stage
        iEQ = 0 # first iteration
        ESS = np.empty(rlsp['maxEQ']+1,dtype=object) # dtype=object to support dictionaries
        ESS[:] = copy.deepcopy(ESS0)
        out = []

        while iEQ < rlsp['maxEQ']: # loop until max iterations or all equilibria are found
            TAU[iEQ] = copy.deepcopy(tau)
            ss, ESS[iEQ] = G(copy.deepcopy(ss),copy.deepcopy(ESS[iEQ]), tau)

            # print the progress every rlsp['print']
            if (np.mod(iEQ,rlsp['print'])==0):
                print(f'ESR[{iEQ}][\'esr\']  :')
                print(ESS[iEQ-1]['esr'])
                print(f'ESR[{iEQ}][\'bases\']:')
                print(ESS[iEQ-1]['bases'])


            # save the equilibria to out
            try:
                out.append(copy.deepcopy(self.output(ss, ESS[iEQ])))
            except:
                pass
            
            # continue onto next iteration
            ESS[iEQ+1] = copy.deepcopy(self.addOne(copy.deepcopy(ESS[iEQ])))


            # start of the find next tau
            changeindex_temp = np.nonzero((copy.deepcopy(ESS[iEQ+1]['esr'])-copy.deepcopy(ESS[iEQ]['esr']))!=0)[0]
            
            changeindex = min(changeindex_temp) +1
            count_index = 0
            for i in stage_index:
                if changeindex<=i:
                    count_index += 1
            tau = count_index-1 #% tau is found
            if np.all(ESS[iEQ+1]['esr']==-2): # if R > 0 for all ESS indexies
                print('goodbye')
                break
            
            """
            :fix this block
            """
            iEQ += 1
            # end % End of recursive lexicographical search


        TAU=TAU[:iEQ]
        return ESS, TAU, out

    def output(self,ss, ESS):
        out2 = {}
        out2['MPEesr'] = copy.deepcopy(ESS['esr'])
        out2['V1'] = np.maximum(ss[0]['EQs'][0,0,0]['eq']['vN1'],ss[0]['EQs'][0,0,0]['eq']['vI1']) 
        out2['V2'] = np.maximum(ss[0]['EQs'][0,0,0]['eq']['vN2'],ss[0]['EQs'][0,0,0]['eq']['vI2']) 
        return out2 


    def addOne(self,addESS):
        # %if x[1,1,1] == -1
        # %    throw(error("This ESS has already overflown!"))
        # %end
        n = len(addESS['esr'])
        X = np.zeros(n,dtype=numpy.int8)
        R = 1

        # xxx fix esr
        for i in np.flip(np.arange(n)):

            X[i] = np.mod(copy.deepcopy(addESS['esr'][i]) + R +1 ,copy.deepcopy(addESS['bases'][i])+1) 

            # if addESS['bases'][i] == 0:
            #     X[i] = copy.deepcopy(addESS['esr'][i]) + R 
            # else:
            #     X[i] = np.mod(copy.deepcopy(addESS['esr'][i]) + R,copy.deepcopy(addESS['bases'][i])) 
            

            # % mod(a,b) does division and returns the remainder given as a-div(a,b)*b
            

            R = lf.div(copy.deepcopy(addESS['esr'][i]) +R +1,copy.deepcopy(addESS['bases'][i] )+1) 
            # print(f"esr: {addESS['esr'][i]}")
            # print(f"bases: {addESS['bases'][i]}")
            # print(f'R: {R} ')

            # % div(a,b) does division and truncates - rounding down - to nearest integer  .... floor(a/b)
        if R > 0:
        # % When exiting the loop R > 0 occurs when all ESS.number is max allowed
        # % which is 1 below the base.
            print("No more equilibria to check.")
            addESS['esr'] = -2*np.ones(n) 
        else:
            addESS['esr'] = copy.deepcopy(X[:] -1)
        return addESS