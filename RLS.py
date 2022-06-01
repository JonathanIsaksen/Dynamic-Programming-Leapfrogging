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
        checki = 1 # display loop number for testing

        rlsp = {'maxEQ':np.NaN, 'print':np.NaN}
        rlsp['maxEQ']  =  50000 # 50000 maximum number of iterations
        rlsp['print']  =  1000  # 500 print every rlsp.print equilibria (0: no print, 1: print every, 2: print every second)

    #     % initialize matrices
        TAU = np.empty(rlsp['maxEQ']+1)
        TAU[:] = np.nan
        tau = np.size(stage_index)  # start RLS at last stage
        iEQ = 0
        ESS = np.empty(rlsp['maxEQ']+1,dtype=object)
        ESS[:] = copy.deepcopy(ESS0)
        out = []
        # out = [np.empty(rlsp['maxEQ']+1,dtype='object')]
        while iEQ < rlsp['maxEQ']:
            TAU[iEQ] = copy.deepcopy(tau)
            ss, ESS[iEQ] = G(copy.deepcopy(ss),copy.deepcopy(ESS[iEQ]), tau)

            if (np.mod(iEQ,rlsp['print'])==0):
                print(f'ESR[{iEQ}][\'esr\']  :')
                print(ESS[iEQ-1]['esr'])
                print(f'ESR[{iEQ}][\'bases\']:')
                print(ESS[iEQ-1]['bases'])


            """
            fix this block:
            """
            # if nargout>2: # if there are less than 2 outputs xxx
            try:
                out.append(copy.deepcopy(self.output(ss, ESS[iEQ])))
                # very slow:
                # out = copy.deepcopy(np.insert(out,iEQ,self.output(ss, ESS[iEQ])))
            except:
                pass

            # print(f'checki: {checki} ')
            # checki += 1
            ESS[iEQ+1] = copy.deepcopy(self.addOne(copy.deepcopy(ESS[iEQ])))


            changeindex_temp = np.nonzero((copy.deepcopy(ESS[iEQ+1]['esr'])-copy.deepcopy(ESS[iEQ]['esr']))!=0)[0]
            
            changeindex = min(changeindex_temp) +1
            count_index = 0
            for i in stage_index:
                if changeindex<=i:
                    count_index += 1
            tau = count_index-1 #% tau0 is found
            if np.all(ESS[iEQ+1]['esr']==-2): # changed to -2
                print('goodbye')
                break
            
            """
            :fix this block
            """
            iEQ += 1
            # end % End of recursive lexicographical search


        TAU=TAU[:iEQ-1]
        return ESS, TAU, out

    def output(self,ss, ESS):
        out2 = {}
        out2['MPEesr'] = copy.deepcopy(ESS['esr'])
        # out['V1'] = max([ss[1]['EQs'][1,1,1]['eq']['vN1'],ss[1]['EQs'][1,1,1]['eq']['vI1']]) #  xxx perhabs change ss[1] to 0
        # out['V2'] = max([ss[1]['EQs'][1,1,1]['eq']['vN2'],ss[1]['EQs'][1,1,1]['eq']['vI2']]) #  xxx perhabs change ss[1] to 0
        out2['V1'] = np.maximum(ss[0]['EQs'][0,0,0]['eq']['vN1'],ss[0]['EQs'][0,0,0]['eq']['vI1']) #  xxx perhabs change ss[0] to 0
        out2['V2'] = np.maximum(ss[0]['EQs'][0,0,0]['eq']['vN2'],ss[0]['EQs'][0,0,0]['eq']['vI2']) #  xxx perhabs change ss[0] to 0
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