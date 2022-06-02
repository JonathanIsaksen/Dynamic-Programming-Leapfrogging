# Define the leapfrogging model

"""
important stuff to look at:
    everything that uses indexing might have to be -1
"""
from tracemalloc import stop
import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
# %matplotlib inline
import quantecon as qe
import scipy.optimize as optimize
import scipy.sparse as sparse
from quantecon import compute_fixed_point
from quantecon.markov import DiscreteDP
import copy



class leapfrogging:

    def __init__(self, Cmax = 5, Cmin = 0, nC = 4, pf = 1, k0 = 0, k1 = 8.3, k2 = 1, R = 0.05, Dt = 1, r1 = lambda x1,x2: np.maximum(x2-x1,0)):
        """
        Descibe the variables here :(
        all the standard fixed parameters
        """

        self.Cmax, self.Cmin, self.nC, self.pf, self.k0, self.k1, self.k2, self.R, self.Dt  = Cmax, Cmin, nC, pf, k0, k1, k2, R, Dt
        """
        Define the Model parameters
        """
        
        self.p = np.ones(self.nC) # transition probability 
        self.populate_p() # ppupalate with the probabilities
        self.q = 1 - self.p
        self.C = np.linspace(self.Cmax,self.Cmin,self.nC)
        self.beta = np.exp(-self.R*self.Dt)
        self.T = (self.nC-1)*3 + 1
        self.nESS = self.nC * (self.nC + 1) * (2*self.nC+1)/6

        self.base_check = 0 # prints out the bases after every step in state_recursion (0 for off, 1 for on)
        self.in__state = 0 # print out where in state_recursion you are

        self.firm1 = np.empty((self.nC,self.nC,self.nC)) # initialize the two firms with empty values 
        self.firm1[:] = np.nan
        self.firm2 = self.firm1.copy()
        self.populate_firms()
        
        self.stage_index = self.rlsindex(self.nC)

        # Functional forms
        """
        Lambda functions: xxx (Describe these...)
        K
        Payoff
        r1
        r2
        Phi
        """
        self.K = lambda c: self.k0+self.k1/(1+self.k2*c) # investment cost of state of the art production cost c
        self.payoff = lambda x: (self.p-x)*self.Dt  # flow payoffs per unit of time
        self.r1 = r1
        self.r2 = lambda x1,x2: np.maximum(x1-x2,0)
        self.Phi = lambda vN,vI: np.maximum(vN, vI) # xxx this should maybe be made to accept lists.

    """
    xxx this is a very long if function :(
    to do:
    fix the functions dependent on mp (Think it's the same as self)
    fix all the for loops, might have to start on 0
    define all the self.solve functions
        
    """
    def state_recursion(self,ss,ESS,tau): 
        if tau == self.T:
            if self.in__state == 1:
                print('in T')
            ss,ESS = self.solve_last_corner(ss.copy(),ESS.copy())
            tau -= 1
            if self.base_check ==1:
                print('after:')
                print(f"{ESS['bases']}")
            
        if tau == self.T -1:
            if self.in__state == 1:
                print('in T-1')
            ss,ESS = self.solve_last_edge(ss.copy(),ESS.copy())
            tau -= 1
            if self.base_check ==1:
                print('after:')
                print(f"{ESS['bases']}")

        if tau == self.T -2:
            if self.in__state == 1:
                print('in T-2')
            ss,ESS = self.solve_last_interior(ss.copy(),ESS.copy())

            tau -= 1

            if self.base_check ==1:
                print('after:')
                print(f"{ESS['bases']}")
            
        
        dothis = 1
        while dothis == 1: # break when tau=0 
            if np.remainder(tau,3)==1: 
                if self.in__state == 1:
                    print('in mod==1')
                ic = int(np.ceil((tau+2)/3)) - 1 # -1 for python indexing xxx
                ss,ESS = self.solve_corner(ss.copy(),ic,ESS.copy())
                tau -= 1
                if tau == 0:
                    break

                if self.base_check ==1:
                    print('after:')
                    print(f"{ESS['bases']}")


            if np.remainder(tau,3)==0:
                if self.in__state == 1:
                    print('in mod==0')
                ic = int(np.ceil((tau+2)/3)) - 1 # python starts at 0
                ss, ESS = self.solve_edge(ss.copy(),ic,ESS.copy())
                tau -= 1

                if self.base_check ==1:
                    print('after:')
                    print(f"{ESS['bases']}")

            if np.remainder(tau,3) == 2:
                if self.in__state == 1:
                    print('in mod==2')
                ic = int(np.ceil((tau+2)/3)) - 1 # python starts at 0
                ss, ESS = self.solve_interior(ss.copy(),ic,ESS.copy())
                tau -= 1

                if self.base_check ==1:
                    print('after:')
                    print(f"{ESS['bases']}")

        # end of the scary while loop
        return ss, ESS
    
    """
    structure of ss: (this is wrong xxx, 'eq' has no empty now)
    array 4x1
        dict with 2 keys
            key:'EQs' 4x4x5 array (this might be wrong)
                key:'eq' at (n x n x k) (for n,k in {0,...,3})

    """
    def cSS(self,N):
			# % INPUT: N is natural positive number = number of stages
			# % OUTPUT: 1 x N struct tau representing stages of state space
			# % PURPOSE: Create state space structure to hold info on identified
			# % equilibria


			#     % P1 player 1 probability invest
			#     % vN1 value of not investing for player 1
			#     % vI1 value of investing for player 1
			#     % P2 player 2 probability of invest
			#     % vN2 value of not investing for player 2
			#     % vI2 value of investing for player 2 
			    
			#     % Initialize datastructure for the state space
        # disctionary
        eq = {'P1':0, 'vN1':0, 'vI1':0, 'P2':0, 'vN2':0, 'vI2':0} # 'P1':[], 'vN1':[], 'vI1':[], 'P2':[], 'vN2':[], 'vI2':[]
        # arrays which will consist of dictionaries
        EQs = np.empty(shape=(N,N,N+1),dtype='object')
        tau = np.empty(shape=(N),dtype='object')
        for i in range(N):
            tau[i] = {}
            tau[i]['EQs'] =  copy.deepcopy(EQs)  # container for identified equilibriums
            tau[i]['nEQ'] = np.zeros((i+1,i+1),dtype=int) # container for number of eqs in (x1,x2,c) point
            for h in range(N):
                for k in range(N):
                    for j in range(N+1):
                        tau[i]['EQs'][h,k,j] = {'eq':eq} 
			   

			# %  #  ##  ###  ####     State space with 4 stages.
			# %     ##  ###  ####     4x4 Hashtag field is reached with complete
			# %         ###  ####     technological development.
			# %              ####     for each hashtag (x1,x2,c) - point in state space - max 5 eq's
        return tau # should probably call this ss and not tau... xxx

    """
    Structure of ESS:
    Dict with 3 keys:
        index 4x4x4
        esr
        bases
    """
    def cESS(self,N):
        # % Create N x N x N array ess.index
        # % PURPOSE:
        # % ess.index(m,n,h) --> j
        # % where j is the index for ess.esr such that
        # % ess.esr(j)+1 is the equilibrium number played in state space
        # % point (m,n,h) this equilibrium is stored in the ss-object as
        # % ss(h).(m,n,j).eq = ss(h).(m,n,ess.esr(ess.index(m,n,h))+1)
        ess = {'index':[], 'esr':[], 'bases':[]}
        ess['index'] = np.empty(shape=(N,N,N),dtype='int')
        for ic in range(N):
            for ic1 in range(ic+1):
                for ic2 in range(ic+1):
                    ess['index'][ic1,ic2,ic]  =  int(self.essindex(N,ic1,ic2,ic) -1)
                    # % N*(N+1)*(2*N+1)/6 = sum(1^2 + 2^2 + 3^2 + ... + N^2)
                    ess['esr'] = np.full((1, int(N*(N+1)*(2*N+1)/6)), int(-1))[0]
                    ess['bases'] = np.zeros(int(N*(N+1)*(2*N+1)/6),dtype=int)
                    # %ess.n = 1:(N*(N+1)*(2*N+1)/6)

        return ess
    
    def essindex(self,x,ic1,ic2,ic):

        # % INPUT: x is count of technological levels
        # % OUTPUT: ess index number for point (m,n,h) i state space
        ici = ic +1
        ic1i = ic1 +1
        ic2i = ic2 +1

        if set([ic1i,ic2i]) == set([ici,ici]):
            index = 1 + self.div(x*(x+1)*(2*x+1),6) - self.div(ici*(ici+1)*(2*ici+1),6)
        elif ic2i == ici:
            index = 1 + self.div(x*(x+1)*(2*x+1),6) - self.div(ici*(ici+1)*(2*ici+1),6) + ic1i
        elif ic1i == ici:
            index = 1 + self.div(x*(x+1)*(2*x+1),6) - self.div(ici*(ici+1)*(2*ici+1),6) + ici - 1 + ic2i
        else:
            index = 1 + self.div(x*(x+1)*(2*x+1),6) - self.div(ici*(ici+1)*(2*ici+1),6) + 2*(ici - 1) + self.sub2ind([ici-1,ici-1],ic1i,ic2i) # xxx sub2ind might be a bit wonky
        return index

    def sub2ind(self,array_shape, rows, cols):
        ind = (cols-1)*array_shape[0] + rows
        return ind 

    def div(self,x,y):
        out = np.floor(x/y) # xxx dot notation before: out=floor(x./y)
        return out

    def quad(self,a, b, c):
        # % Solves:  ax^2  + bx + c = 0
        # % but also always return 0 and 1 as candidates for probability of
        # % investment
        d = b**2 - 4*a*c
        # xxx commented ou:
        if abs(a) < 1e-8:
            pstar = [0, 1, -c/b] # xxx no idea what happens here
        else: # probably doesn't need another else here
            if d < 0:
                pstar = [0, 1]
            elif d == 0.:
                pstar = [0, 1, -b/(2*a)]
            else:
                pstar = [0, 1, (-b - np.sqrt(d))/(2*a), (-b + np.sqrt(d))/(2*a)]
        return pstar
    
    def EQ(self,P1,vN1,vI1,P2,vN2,vI2,ic,ic1,ic2):
        list_eq = []
        for v in [P1,vN1,vI1,P2,vN2,vI2]:
            if isinstance(v,numpy.ndarray):
                if v[0]==True:
                    list_eq.append(1)
                elif not v[0]:
                    list_eq.append(0)
                else:
                    list_eq.append(v[0])
            else:
                if v==True:
                    list_eq.append(1)
                elif not v:
                    list_eq.append(0)
                else:
                    list_eq.append(v)
        eq = {'P1':list_eq[0], 'vN1':list_eq[1], 'vI1':list_eq[2], 'P2':list_eq[3], 'vN2':list_eq[4], 'vI2':list_eq[5]}
        return eq


    # define all the solve functions used by state_recursion
    def solve_last_corner(self,ss,ESS):
        h = self.nC-1 # Number of technological levels
        c = self.Cmin # State of the art marginal cost for last tech. level

        # Both players have state of the art technology implemented ic1=ic2=c

        # If K>0 the vN1 = r1/(1-beta) .... geometric sum
        vN1 = (self.r1(c,c)+self.beta * np.maximum(0,-self.K(c)))  /  (1-self.beta)
        vI1 = vN1 - self.K(c)
        P1 = vI1 > vN1 # Equivalent to 0>self.K(c)
        vN2 = (self.r2(c,c)+self.beta * np.maximum(0 , -self.K(c)))  /  (1-self.beta)
        vI2 = vN2 - self.K(c)
        P2 = vI2 > vN2 # Equivalent to 0>self.K(c) and hence equal to P1
        
        # OUTPUT is stored in ss
        # wtf is happening here xx
        ss[h]['EQs'][h,h,0]['eq'] = self.EQ(P1,vN1,vI1,P2,vN2,vI2,h,h,0) # changed to 0
        # Only one equilibrium is possible:
        ss[h]['nEQ'][h,h] = 0 # xxx changed to 0 from 1
        ESS['bases'][ESS['index'][h,h,h]] = 0 # xxx changed to 0 from 1
        return ss, ESS

    def solve_last_edge(self,ss,ESS):
        # % INPUT:
        # % cost and mp are global parameters
        # % ss state space structure (has info on eq's in corner of last layer)
        # % OUTPUT:
        # % Equilibria lf.EQ(P1, vN1, vI1, P2, vN2, vI2) for edge state space points
        # % of the final layer:
        # % Final layer <=> s=(x1,x2,c) with c = min(mp.C) 
        # % Edge <=> s=(x1,x2,c) with x2 = c = min(mp.C) and x1 > c or
        # % s=(x1,x2,c) with x1 = c = min(mp.C) and x2 > c

        ic = self.nC - 1 # Get the level of technology final layer # -1
        c = self.Cmin # Get state of the art marginal cost for tech. of final layer

        h = 0 # changed to 0 from 1
        # % h is used to select equilibria in the corner of the final layer but there
        # % is only ever 1 equilibria in the corner
        # % If we did not apply this apriori knowledge we would have to use ESS
        # % max(vN,vI | at the corner final layer)= mp.Phi(ss(ic).EQs(ic,ic,h).eq.vN1,ss(ic).EQs(ic,ic,h).eq.vI1)

        # Get the value of max choice in the corner of final layer s = (c,c,c)
        # xxx lots of nested functions here that are not defined
        g1_ccc = np.maximum(ss[ic]['EQs'][ic,ic,h]['eq']['vN1'],ss[ic]['EQs'][ic,ic,h]['eq']['vI1'])
        
        g2_ccc = np.maximum(ss[ic]['EQs'][ic,ic,h]['eq']['vN2'],ss[ic]['EQs'][ic,ic,h]['eq']['vI2'])
        # Player 2 is at the edge s=(x1,x2,c) with x2=c=min(mp.C) and x1>c
        # xxx start on 0?
        for ic1 in range(ic): # might have to change this to +1 xxx and ic1 to -1
            x1 = self.C[ic1] 
            vI1 = self.r1(x1,c) - self.K(c)  + self.beta * g1_ccc
            vN1search = lambda z: self.r1(x1,c) + self.beta * self.Phi(z,vI1) - z
            vN1 = optimize.fsolve(vN1search,0)[0] # xxx watchout for [0] it wasn't there before
            P1 = vI1 > vN1


            vN2 = ( self.r2(x1,c) + self.beta * (P1*g2_ccc+(1-P1)*self.Phi(0,-self.K(c))) )  /  ( 1-self.beta*(1-P1) )
            vI2 = vN2 - self.K(c)
            P2 = vI2 > vN2

            # xxx wtf is happening here

            ss[ic]['EQs'][ic1,ic,h]['eq'] = self.EQ(P1,vN1,vI1,P2,vN2,vI2,ic1,ic,h)

            ss[ic]['nEQ'][ic1,ic] = 0 # xxx changed to 0 from 1
            ESS['bases'][ESS['index'][ic1,ic,ic]] = 0 # xxx changed to 0 from 

        
        # xxx maybe start at 0?
        # Player 1 is at the edge s=(x1,x2,c) with x1=c=min(mp.C) and x2>c
        for ic2 in range(ic): # xxx might have to change to +1
            x2 = self.C[ic2]
            vI2 = self.r2(c,x2) - self.K(c) + self.beta * g2_ccc
            vN2search = lambda x: self.r2(c, x2) + self.beta*self.Phi(x,vI2)-x
            vN2 = optimize.fsolve(vN2search,0)
            P2 = vI2 > vN2


            vN1 = (self.r1(c, x2) + self.beta*(P2*g1_ccc+(1-P2)*self.Phi(0, -self.K(c))))  /  ( 1-self.beta*(1-P2) )
            vI1 = vN1-self.K(c)
            P1 = vI1 > vN1

            # wtf is happening here xx
            
            ss[ic]['nEQ'][ic1,ic] = 0 # xxx changed to 0 from 1
            ESS['bases'][ESS['index'][ic1,ic,ic]] = 0 # xxx changed to 0 from 1

            ss[ic]['EQs'][ic, ic2, 0]['eq'] = self.EQ(P1, vN1, vI1, P2, vN2, vI2,ic, ic2, 0) # changed to 0
            ss[ic]['nEQ'][ic, ic2] = 0 # maybe 0 here xxx
            ESS['bases'][ESS['index'][ic,ic2,ic]] = 0 # xxx changed to 0 from 1
        
        return ss,ESS
    """
    To do: xxx
    define quad
    Define the functions used :(
    """
    def solve_last_interior(self,ss,ESS):
        # xxx this is wrong, no output
        # outside loop xxx might have to change to -1 on these
        ic = self.nC -1
        c = self.C[ic]

        """
        I think they just find whether investing or not investing gives highest utility
        """
        # removed +1 from iC1, iC2, +1 ESS ... not sure why as that shouldn't fix it

        # g1 = lambda iC1, iC2, iC: np.maximum(ss[iC]['EQs'][iC1, iC2, 1+ ESS['esr'][ESS['index'][iC1,iC2,iC]]]['eq']['vN1'],ss[iC]['EQs'][iC1, iC2, 1+ ESS['esr'][ESS['index'][iC1,iC2,iC]]]['eq']['vI1'])
        # g2 = lambda iC1, iC2, iC: np.maximum(ss[iC]['EQs'][iC1, iC2, 1+ ESS['esr'][ESS['index'][iC1,iC2,iC]]]['eq']['vN2'],ss[iC]['EQs'][iC1, iC2, 1+ ESS['esr'][ESS['index'][iC1,iC2,iC]]]['eq']['vI2'])
        g1 = lambda iC1, iC2, iC: np.maximum(ss[iC]['EQs'][iC1, iC2, 1+ ESS['esr'][ESS['index'][iC1,iC2,iC]]]['eq']['vN1'],ss[iC]['EQs'][iC1, iC2, 1+ ESS['esr'][ESS['index'][iC1,iC2,iC]]]['eq']['vI1'])
        g2 = lambda iC1, iC2, iC: np.maximum(ss[iC]['EQs'][iC1, iC2, 1+ ESS['esr'][ESS['index'][iC1,iC2,iC]]]['eq']['vN2'],ss[iC]['EQs'][iC1, iC2, 1+ ESS['esr'][ESS['index'][iC1,iC2,iC]]]['eq']['vI2'])


        for ic1 in range(ic): #Player 1 loop begin # xxx maybe ad -1
            for ic2 in range(ic): #Player 2 loop begin                
                # Player 1 -> leads to P2 candidates
                # what is g1 xxx
                a = self.r1(self.C[ic1], self.C[ic2]) - self.K(c) + self.beta*g1(ic, ic2, ic) #check
                b = self.beta*(g1(ic, ic, ic)-g1(ic, ic2, ic)) # check
                d = self.r1(self.C[ic1],self.C[ic2])
                e = self.beta*g1(ic1, ic, ic)


                b_0 = - self.beta * b # check 
                b_1 = self.beta * g1(ic1, ic, ic) + (self.beta-1)*b - self.beta*a # check
                b_2 = self.r1(self.C[ic1],self.C[ic2]) + (self.beta-1) * a # check 

                pstar2 = self.quad(b_0, b_1, b_2)
                # always return 1 and 0 for the pure strategies


                # Player 2 -> leads to P1 candidates
                A = self.r2(self.C[ic1], self.C[ic2]) - self.K(c) + self.beta*g2(ic1, ic, ic)
                B = self.beta*(g2(ic, ic, ic)-g2(ic1, ic, ic))
                D = self.r2(self.C[ic1],self.C[ic2])
                E = self.beta*g2(ic, ic2, ic)

                d_0 = - self.beta * B
                d_1 = self.beta*g2(ic, ic2, ic) + (self.beta-1) * B - self.beta*A
                d_2 = self.r2(self.C[ic1],self.C[ic2]) + (self.beta-1) * A
                


                pstar1 = self.quad(d_0, d_1, d_2)

                    



                    # % Find equilibria based on candidates
                    # % Number of equilibria found are 0 to begin with
                count = 0 # changed to -1 from 0
                for i in range(len(pstar1)):
                        for j in range(len(pstar2)):

                            if i in [0,1] and j in [0,1]: # matlab code: all(ismember([i,j],[1,2])) # these are pure strategies
                                
                                # % If the polynomial is negative vI > vN
                                # % hence player invests set exPj=1 else 0
                                # % exP1 is best response to pstar2(j)
                                if b_2 + b_1 * pstar2[j] + b_0 * pstar2[j]**2 < 0:
                                    exP1 = 1
                                else:
                                    exP1 = 0

                                if d_2 + d_1 * pstar1[i] + d_0 * pstar1[i]**2 < 0:
                                    exP2 = 1
                                else:
                                    exP2 = 0

                                # % check if both are playing best response
                                # % in pure strategies. Players best response
                                # % should be equal to the candidate to which
                                # % the other player is best responding.
                                if abs(exP1 - pstar1[i]) < 1e-8 and abs(exP2-pstar2[j]) < 1e-8:
                                    
                                    # % if exP1=0 and pstar_i=0 true
                                    # % if exP1=1 and pstar_i=1 true
                                    # % Testing whether best response exP1 is
                                    # % equal to pstar1(i) to which Player 2
                                    # % is best responding ...
                                    count += 1
                                    vI1 = a + b*pstar2[j] 
                                    vN1 = (d + e*pstar2[j] + self.beta*(1-pstar2[j])*(a+b*pstar2[j]))*pstar1[i] + (1-pstar1[i])*(d+e*pstar2[j])/(1-self.beta*(1-pstar2[j]))

                                    vI2 = A + B*pstar1[i]; 
                                    vN2 = (D + E*pstar1[i] + self.beta*(1-pstar1[i])*(A+B*pstar1[i]))*pstar2[j] + (1-pstar2[j])*(D+E*pstar1[i])/(1-self.beta*(1-pstar1[i]))


                                    ss[ic]['EQs'][ic1, ic2, count-1]['eq'] = self.EQ(pstar1[i],vN1,vI1,pstar2[j],vN2,vI2,ic1, ic2, count-1)
                                    

                            elif i > 1 and j > 1 and pstar1[i] >= 0 and pstar2[j] >= 0 and pstar1[i] <= 1 and pstar2[j] <= 1: # change i and j > 1 from 2
                                count += 1
                                v1 = a + b * pstar2[j]
                                v2 = A + B * pstar1[i]
                                ss[ic]['EQs'][ic1, ic2, count-1]['eq'] = self.EQ(pstar1[i],v1,v1,pstar2[j],v2,v2,ic1, ic2, count-1)
                                


                                
                                

                            # end if i in [0,1]...
                        # no more j loop
                # no more i loop    
                ss[ic]['nEQ'][ic1, ic2] = count -1 
                ESS['bases'][ESS['index'][ic1,ic2,ic]] = count -1 # added minus 1
                
        return ss, ESS # end of solve_last_interior

    def  solve_corner(self,ss,ic,ESS):
        # % ss(i).EQs(m,n,h).eq
        # % i is layer index
        # % m is level of technology of P1
        # % n is level of technology of P2
        # % h is selection equilibrium number given by esr

        # % Corner: i given --> m=i and n=i
        # % We are solving the corner of not last stage ==> 
        # % (1) Technological development is possible ==> 
        # %       uncertainty over tech. development
        # % (2) Investment choice is inconsequetial ==>
        # %     No uncertainty about the other players investmentoutcome

        # % Get the marginal cost for the stage i under consideration
        c = self.C[ic]

        # % Get probability of technological development
        p = self.p[ic]

        # % Find 
        # % index for equilibrium selection h = 1 for simple selection rule
        # % Need ic+1 because ss(ic+1).EQs(ic,ic,h).eq is to be accessed
        
        h = int(ESS['esr'][ESS['index'][ic,ic,ic+1]] +1) # 

        vN1 = ( self.r1(c,c) + self.beta*p*max(ss[ic+1]['EQs'][ic,ic,h]['eq']['vN1'],ss[ic+1]['EQs'][ic,ic,h]['eq']['vI1']) + self.beta*(1-p)*max(0,-self.K(c)))/(1-(1-p)*self.beta )
        vI1 = vN1 - self.K(c)
        vN2 = ( self.r2(c,c) + self.beta*p*max(ss[ic+1]['EQs'][ic,ic,h]['eq']['vN2'],ss[ic+1]['EQs'][ic,ic,h]['eq']['vI2']) + self.beta*(1-p)*max(0,-self.K(c)) )/(1-(1-p)*self.beta )
        vI2 = vN2 - self.K(c)

        if vI1 > vN1: #% no investment uncertainty and no investments if K(c) > 0.
            P1 = 1
        else:
            P1 = 0
        if vI2 > vN2:
            P2 = 1
        else:
            P2 = 0

        # % Create output for return
        ss[ic]['EQs'][ic,ic,0]['eq'] = self.EQ(P1, vN1, vI1, P2, vN2 , vI2,ic,ic,0) # changed to 0
        ss[ic]['nEQ'][ic,ic] = 0 # xxx changed to 0 from 1
        ESS['bases'][ESS['index'][ic,ic,ic]] = 0 # xxx changed to 0 from 1
        # % No update of ESS.bases is necessary in principle: "there can BE ONLY ONE
        # % equilibrium"  https://www.youtube.com/watch?v=sqcLjcSloXs
        return ss,ESS

    def solve_edge(self,ss,ic,ESS):
        # % INPUTS:
        # % ss is state space
        # % cost are global parameters see documentation
        # % mp are global parameters see documentation
        # % OUTPUT: returning state space with containing calculated solutions

        # % PURPOSE: Function solves for quilibrium (P1, vN1, vI1, P2, vN2, vI2)
        # % on the edges of the state space - though not corner - 

        # % Get the marginal cost of layer ic
        c = self.C[ic]

        # % Get the probability of technological development occuring in layer ic
        p = self.p[ic]
        

        # % Creating some functions:
        # % DEPENDENCIES: p as global (not passed as argument) and h
        # % PURPOSE: Calculates expectations over technological development ...
        # % technological development occurs: ss(iC+1)
        # % technological development does not occur: ss(iC)

        # H1 = lambda iC1, iC2, iC: p*self.Phi(ss[iC+1]['EQs'][iC1, iC2, ESS['esr'][ESS['index'][iC1,iC2,iC+1]]+1]['eq']['vN1'],ss[iC+1]['EQs'][iC1, iC2, ESS['esr'][ESS['index'][iC1,iC2,iC+1]]+1]['eq']['vI1']) + (1-p)*self.Phi(ss[iC]['EQs'][iC1, iC2, ESS['esr'](ESS.index(iC1,iC2,iC))+1).eq.vN1,ss(iC).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC))+1).eq.vI1);
        # H2 = @(iC1, iC2, iC) p*mp.Phi(ss(iC+1).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC+1))+1).eq.vN2,ss(iC+1).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC+1))+1).eq.vI2) + (1-p)*mp.Phi(ss(iC).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC))+1).eq.vN2,ss(iC).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC))+1).eq.vI2);
        # removed +1 for some reason

        H1 = lambda iC1, iC2, iC: p*self.Phi(ss[iC+1]['EQs'][iC1, iC2, 1+ ESS['esr'][ESS['index'][iC1,iC2,iC+1]]]['eq']['vN1'],ss[iC+1]['EQs'][iC1, iC2, 1+ ESS['esr'][ESS['index'][iC1,iC2,iC+1]]]['eq']['vI1']) + (1-p)*self.Phi(ss[iC]['EQs'][iC1, iC2, 1+ ESS['esr'][ESS['index'][iC1,iC2,iC]]]['eq']['vN1'],ss[iC]['EQs'][iC1, iC2, 1+ ESS['esr'][ESS['index'][iC1,iC2,iC]]]['eq']['vI1'])
        H2 = lambda iC1, iC2, iC: p*self.Phi(ss[iC+1]['EQs'][iC1, iC2, 1+ ESS['esr'][ESS['index'][iC1,iC2,iC+1]]]['eq']['vN2'],ss[iC+1]['EQs'][iC1, iC2, 1+ ESS['esr'][ESS['index'][iC1,iC2,iC+1]]]['eq']['vI2']) + (1-p)*self.Phi(ss[iC]['EQs'][iC1, iC2, 1+ ESS['esr'][ESS['index'][iC1,iC2,iC]]]['eq']['vN2'],ss[iC]['EQs'][iC1, iC2, 1+ ESS['esr'][ESS['index'][iC1,iC2,iC]]]['eq']['vI2'])

        # H1 = lambda iC1, iC2, iC: p*self.Phi(ss[iC+1]['EQs'][iC1, iC2, ESS['esr'][ESS['index'][iC1,iC2,iC+1]]]['eq']['vN1'],ss[iC+1]['EQs'][iC1, iC2, ESS['esr'][ESS['index'][iC1,iC2,iC+1]]]['eq']['vI1']) + (1-p)*self.Phi(ss[iC]['EQs'][iC1, iC2, ESS['esr'][ESS['index'][iC1,iC2,iC]]]['eq']['vN1'],ss[iC]['EQs'][iC1, iC2, ESS['esr'][ESS['index'][iC1,iC2,iC]]]['eq']['vI1'])
        # H2 = lambda iC1, iC2, iC: p*self.Phi(ss[iC+1]['EQs'][iC1, iC2, ESS['esr'][ESS['index'][iC1,iC2,iC+1]]]['eq']['vN2'],ss[iC+1]['EQs'][iC1, iC2, ESS['esr'][ESS['index'][iC1,iC2,iC+1]]]['eq']['vI2']) + (1-p)*self.Phi(ss[iC]['EQs'][iC1, iC2, ESS['esr'][ESS['index'][iC1,iC2,iC]]]['eq']['vN2'],ss[iC]['EQs'][iC1, iC2, ESS['esr'][ESS['index'][iC1,iC2,iC]]]['eq']['vI2'])

        # % Efficiency ... why evaluate the call for each run of following loop? i is
        # % constant in domain outside loop!! What changes in the function is the
        # % ESS.

        # % if at (x1,c,c) edge, with x1<c - Player 2 is at the edge.
        for ic1 in range(ic): # index running over different technological levels for player1 not on edge xxx maybe start at 1
        # % Get the marginal cost for player 1
            c1 = self.C[ic1]
            # % First calculate vI1 depending only on known factors ... no uncertainty about Player 2 because he is at the edge
            vI1 = self.r1(c1,c) - self.K(c) + self.beta*H1(ic,ic,ic)

            vN1search = lambda z: self.r1(c1,c) + self.beta*(p*max(ss[ic+1]['EQs'][ic1,ic,ESS['esr'][ESS['index'][ic1,ic,ic+1]]+1]['eq']['vN1'], ss[ic+1]['EQs'][ic1, ic, ESS['esr'][ESS['index'][ic1,ic,ic+1]]+1]['eq']['vI1'])+(1-p)*max(z,vI1)) - z
            # removed +1
            # vN1search = lambda z: self.r1(c1,c) + self.beta*(p*max(ss[ic+1]['EQs'][ic1,ic,ESS['esr'][ESS['index'][ic1,ic,ic+1]]]['eq']['vN1'], ss[ic+1]['EQs'][ic1, ic, ESS['esr'][ESS['index'][ic1,ic,ic+1]]]['eq']['vI1'])+(1-p)*max(z,vI1)) - z

            vN1 = optimize.fsolve(vN1search,0)
            P1 = vI1 > vN1

            # 
            vN2 = (self.r2(c1,c) + self.beta*(P1*H2(ic,ic,ic)+(1-P1)*(p*self.Phi(ss[ic+1]['EQs'][ic1, ic,ESS['esr'][ESS['index'][ic1,ic,ic+1]]+1]['eq']['vN2'],ss[ic+1]['EQs'][ic1, ic,ESS['esr'][ESS['index'][ic1,ic,ic+1]]+1]['eq']['vI2']) + (1-p)*self.Phi(0,-self.K(c)))))/(1-self.beta*(1-P1)*(1-p))
            # removed +1
            # vN2 = (self.r2(c1,c) + self.beta*(P1*H2(ic,ic,ic)+(1-P1)*(p*self.Phi(ss[ic+1]['EQs'][ic1, ic,ESS['esr'][ESS['index'][ic1,ic,ic+1]]]['eq']['vN2'],ss[ic+1]['EQs'][ic1, ic,ESS['esr'][ESS['index'][ic1,ic,ic+1]]]['eq']['vI2']) + (1-p)*self.Phi(0,-self.K(c)))))/(1-self.beta*(1-P1)*(1-p))


            vI2 = vN2 - self.K(c)
            P2 = vI2 > vN2

            ss[ic]['EQs'][ic1,ic,0]['eq'] = self.EQ(P1, vN1, vI1, P2, vN2, vI2,ic1,ic,0) # changed to 0
            ss[ic]['nEQ'][ic1,ic] = 0 # xxx changed to 0 from 1
            ESS['bases'][ESS['index'][ic1,ic,ic]] = 0 # xxx changed to 0 from 1
            # end % Exit player 1 not at edge loop


        # % if at (c,x2,c) edge where x2<c - Player 1 is at the edge
        for ic2 in range(ic): # maybe start at 1 xxx
            c2 = self.C[ic2]

            # xxx vI2 is broken
            vI2 = self.r2(c,c2) - self.K(c) + self.beta*H2(ic,ic,ic)

            vN2search = lambda z: self.r2(c,c2) + self.beta*(p*max(ss[ic+1]['EQs'][ic, ic2,ESS['esr'][ESS['index'][ic,ic2,ic+1]]+1]['eq']['vN2'], ss[ic+1]['EQs'][ic, ic2, ESS['esr'][ESS['index'][ic,ic2,ic+1]]+1]['eq']['vI2']) + (1-p)*max(z,vI2)) - z
            #removed 1
            # vN2search = lambda z: self.r2(c,c2) + self.beta*(p*max(ss[ic+1]['EQs'][ic, ic2,ESS['esr'][ESS['index'][ic,ic2,ic+1]]]['eq']['vN2'], ss[ic+1]['EQs'][ic, ic2, ESS['esr'][ESS['index'][ic,ic2,ic+1]]]['eq']['vI2']) + (1-p)*max(z,vI2)) - z

            
            vN2 = optimize.fsolve(vN2search,0)
            P2 = vI2 > vN2

            vN1 = (self.r1(c,c2) + self.beta*(P2*H1(ic,ic,ic)+(1-P2)*(p*self.Phi(ss[ic+1]['EQs'][ic, ic2, ESS['esr'][ESS['index'][ic,ic2,ic+1]]+1]['eq']['vN1'],ss[ic+1]['EQs'][ic, ic2, ESS['esr'][ESS['index'][ic,ic2,ic+1]]+1]['eq']['vI1'])+(1-p)*self.Phi(0,-self.K(c)))))/(1-self.beta*(1-P2)*(1-p))
            # removed +1 from index
            # vN1 = (self.r1(c,c2) + self.beta*(P2*H1(ic,ic,ic)+(1-P2)*(p*self.Phi(ss[ic+1]['EQs'][ic, ic2, ESS['esr'][ESS['index'][ic,ic2,ic+1]]]['eq']['vN1'],ss[ic+1]['EQs'][ic, ic2, ESS['esr'][ESS['index'][ic,ic2,ic+1]]]['eq']['vI1'])+(1-p)*self.Phi(0,-self.K(c)))))/(1-self.beta*(1-P2)*(1-p))


            vI1 = vN1-self.K(c)
            P1 = vI1 > vN1


            ss[ic]['EQs'][ic,ic2,0]['eq'] = self.EQ(P1, vN1, vI1, P2, vN2, vI2,ic,ic2,0) # changed to 0
            ss[ic]['nEQ'][ic,ic2] = 0 # xxx changed to 0 from 1
            ESS['bases'][ESS['index'][ic,ic2,ic]] = 0 # xxx changed to 0 from 1


        # % No update of ESS.bases is necessary: "there can BE ONLY ONE
        # % equilibrium"  https://www.youtube.com/watch?v=sqcLjcSloXs
        # end % Exit player 2 not at edge loop
        # end % end solve_edge

        
        
        return ss, ESS


    def solve_interior(self,ss,ic,ESS):
        # % INPUT 
        # % ss is state space structure with solutions for final layer edge and
        # % corner
        # % ic is the level of technology for which to solve
        # % ESS is struc with information holding ESS.esr being equilibrium selection
        # % rule and ESS.bases being the bases of the ESS.esr's
        c = self.C[ic]
        for ic1 in range(ic):
            for ic2 in range(ic):
                ss,ESS = self.find_interior(ss,ic1,ic2,ic,c,ESS)
                
        return ss, ESS
    
    def find_interior(self,ss,ic1,ic2,ic,c,ESS):
        # % INPUT
        # % ss is state space
        # % mp and cost holds global parameters
        # % ic1 and ic2 are indexes for player 1 and player 2 such that mp.C[ic1]
        # % and mp.C[ic2] are marginal cost of the players
        # % k is the level of technology
        # % c=mp.C(k) state of the art marginal cost ... could be found inside
        # % function
        # % 

        # % get probability of technological development
        p = self.p[ic]
        q = 1-p

        # % h is used for selected equilibrium in state realized when technology
        # % develops hence ic+1 in ESS.index(ic1,ic2,ic+1)
        h = ESS['esr'][ESS['index'][ic1,ic2,ic+1]]+1 



        # % H1(iC1, iC2, iC) = (1-pi)*?(?[iC+1].EQs[iC1, iC2, ESS[iC1,iC2,iC+1]].vN1,?[iC+1].EQs[iC1, iC2, ESS[iC1,iC2,iC+1]].vI1) + pi*?(?[iC].EQs[iC1, iC2, ESS[iC1,iC2,iC]].vN1,?[iC].EQs[iC1, iC2, ESS[iC1,iC2,iC]].vI1)
        # % H2(iC1, iC2, iC) = (1-pi)*?(?[iC+1].EQs[iC1, iC2, ESS[iC1,iC2,iC+1]].vN2,?[iC+1].EQs[iC1, iC2, ESS[iC1,iC2,iC+1]].vI2) + pi*?(?[iC].EQs[iC1, iC2, ESS[iC1,iC2,iC]].vN2,?[iC].EQs[iC1, iC2, ESS[iC1,iC2,iC]].vI2)

        # H1 = lambda iC1, iC2, iC: p*self.Phi(  ss(iC+1).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC+1))+1).eq.vN1 , ss(iC+1).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC+1))+1).eq.vI1  ) + (1-p)*self.Phi(  ss(iC).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC))+1).eq.vN1 , ss(iC).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC))+1).eq.vI1  )
        # H2 = lambda iC1, iC2, iC: p*self.Phi(  ss(iC+1).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC+1))+1).eq.vN2 , ss(iC+1).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC+1))+1).eq.vI2  ) + (1-p) * self.Phi(  ss(iC).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC))+1).eq.vN2 , ss(iC).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC))+1).eq.vI2  )

        # removed +1 ESS...
        # H1 = lambda iC1, iC2, iC: p*self.Phi(  ss[iC+1]['EQs'][iC1, iC2, ESS['esr'][ESS['index'][iC1,iC2,iC+1]]]['eq']['vN1'], ss[iC+1]['EQs'][iC1, iC2, ESS['esr'][ESS['index'][iC1,iC2,iC+1]]]['eq']['vI1']  ) + (1-p)*self.Phi(  ss[iC]['EQs'][iC1, iC2, ESS['esr'][ESS['index'][iC1,iC2,iC]]]['eq']['vN1'], ss[iC]['EQs'][iC1, iC2, ESS['esr'][ESS['index'][iC1,iC2,iC]]]['eq']['vI1']  )
        # H2 = lambda iC1, iC2, iC: p*self.Phi(  ss[iC+1]['EQs'][iC1, iC2, ESS['esr'][ESS['index'][iC1,iC2,iC+1]]]['eq']['vN2'], ss[iC+1]['EQs'][iC1, iC2, ESS['esr'][ESS['index'][iC1,iC2,iC+1]]]['eq']['vI2']  ) + (1-p)*self.Phi(  ss[iC]['EQs'][iC1, iC2, ESS['esr'][ESS['index'][iC1,iC2,iC]]]['eq']['vN2'], ss[iC]['EQs'][iC1, iC2, ESS['esr'][ESS['index'][iC1,iC2,iC]]]['eq']['vI2']  )
        
        H1 = lambda iC1, iC2, iC: p*self.Phi(  ss[iC+1]['EQs'][iC1, iC2, 1+ ESS['esr'][ESS['index'][iC1,iC2,iC+1]]]['eq']['vN1'], ss[iC+1]['EQs'][iC1, iC2, 1+ ESS['esr'][ESS['index'][iC1,iC2,iC+1]]]['eq']['vI1']  ) + (1-p)*self.Phi(  ss[iC]['EQs'][iC1, iC2, 1+ ESS['esr'][ESS['index'][iC1,iC2,iC]]]['eq']['vN1'], ss[iC]['EQs'][iC1, iC2, 1+ ESS['esr'][ESS['index'][iC1,iC2,iC]]]['eq']['vI1']  )
        H2 = lambda iC1, iC2, iC: p*self.Phi(  ss[iC+1]['EQs'][iC1, iC2, 1+ ESS['esr'][ESS['index'][iC1,iC2,iC+1]]]['eq']['vN2'], ss[iC+1]['EQs'][iC1, iC2, 1+ ESS['esr'][ESS['index'][iC1,iC2,iC+1]]]['eq']['vI2']  ) + (1-p)*self.Phi(  ss[iC]['EQs'][iC1, iC2, 1+ ESS['esr'][ESS['index'][iC1,iC2,iC]]]['eq']['vN2'], ss[iC]['EQs'][iC1, iC2, 1+ ESS['esr'][ESS['index'][iC1,iC2,iC]]]['eq']['vI2']  )
        

        a = self.r1( self.C[ic1],self.C[ic2] ) - self.K(c) + self.beta * H1(ic,ic2,ic) #check
        b = self.beta * (   H1(ic,ic,ic) - H1(ic,ic2,ic)  ) #check
        d = self.r1( self.C[ic1],self.C[ic2] )   + self.beta * p * self.Phi( ss[ic+1]['EQs'][ic1,ic2,h]['eq']['vN1'] , ss[ic+1]['EQs'][ic1,ic2,h]['eq']['vI1']  )
        e = self.beta * H1(ic1,ic,ic)         - self.beta * p * self.Phi( ss[ic+1]['EQs'][ic1,ic2,h]['eq']['vN1'] , ss[ic+1]['EQs'][ic1,ic2,h]['eq']['vI1']  ) 

        pa = - self.beta * (1-p) * b
        pb = e + ( self.beta * (1-p) -1) * b - self.beta * (1-p) * a
        pc = d + ( self.beta * (1-p) -1 ) * a

        # Solve for p2 mixed strategy ... but also returns 1 and 0 for pure
        pstar2 = self.quad(pa,pb,pc)
        

        A = self.r2(self.C[ic1],self.C[ic2]) - self.K(c) + self.beta * H2(ic1,ic,ic)
        B = self.beta * ( H2(ic,ic,ic) - H2(ic1,ic,ic) )
        D = self.r2(self.C[ic1],self.C[ic2]) + self.beta * p * self.Phi( ss[ic+1]['EQs'][ic1,ic2,h]['eq']['vN2'] , ss[ic+1]['EQs'][ic1,ic2,h]['eq']['vI2'] )
        E = self.beta * H2(ic,ic2,ic) - self.beta * p * self.Phi( ss[ic+1]['EQs'][ic1,ic2,h]['eq']['vN2'] , ss[ic+1]['EQs'][ic1,ic2,h]['eq']['vI2'] )


        qa = - self.beta * (1-p) * B
        qb = E + ( self.beta * (1-p) - 1 ) * B - self.beta * (1-p) * A
        qc = D + ( self.beta * (1-p) - 1 ) * A


        pstar1 = self.quad(qa, qb, qc)



        count = 0 # maybe change to -1 from 0 (added -1 where it's used instead)
        for i in range(len(pstar1)):
            for j in range(len(pstar2)):
                if i in [0,1] and j in [0,1]: # set([i,j]).issuperset(set([1,2])): # matlab code: all(ismember([i,j],[1,2]))
                    
                    if  pc + pb * pstar2[j] + pa * pstar2[j]**2 < 0:
                        exP1 = 1
                    else:
                        exP1 = 0
                    if qc + qb * pstar1[i] + qa * pstar1[i]**2 < 0:
                        exP2 = 1
                    else:
                        exP2 = 0

                    if abs(exP1 - pstar1[i]) < 1e-7 and abs(exP2-pstar2[j]) < 1e-7:

                        count += 1
                        vI1 = a + b*pstar2[j]
                        vN1 = (d + e*pstar2[j] + self.beta*q*(1-pstar2[j])*(a+b*pstar2[j]))*pstar1[i]+(1-pstar1[i])*(d+e*pstar2[j])/(1-self.beta*q*(1-pstar2[j]))
                        vI2 = A + B*pstar1[i]
                        # xxx vN2 is wrong
                        vN2 = (D + E*pstar1[i] + self.beta*q*(1-pstar1[i])*(A+B*pstar1[i]))*pstar2[j]+(1-pstar2[j])*(D+E*pstar1[i])/(1-self.beta*q*(1-pstar1[i]))

                        ss[ic]['EQs'][ic1, ic2, count-1]['eq'] = self.EQ(pstar1[i],vN1,vI1,pstar2[j],vN2,vI2,ic1, ic2, count-1) # xx wut?
                    

                elif i > 1 and j > 1 and pstar1[i] >= 0 and pstar2[j] >= 0 and pstar1[i] <= 1 and pstar2[j] <= 1: # maybe i and j bigger than 2 (changed to 1)
                    count += 1
                    v1 = a + b * pstar2[j]
                    v2 = A + B * pstar1[i]
                    ss[ic]['EQs'][ic1, ic2, count-1]['eq'] = self.EQ(pstar1[i],v1,v1,pstar2[j],v2,v2,ic1, ic2, count-1)
            # end j loop
        # end i loop

        ss[ic]['nEQ'][ic1,ic2] = count -1 # xxx wut? 
        ESS['bases'][ESS['index'][ic1,ic2,ic]] = count -1 # added -1

        return ss, ESS # end find_interior

    def vscatter(self,V,jitter,sigma,adjust,VM):
        # % INPUT: N x 2+ matrix with value function values in first two columns
        # % jitter is indicator if 1 jitter is added
        # % sigma is std. of jitter noise
        # % adjust is indicator if 1 adjustment using weights is used
        # % VM is used to draw triangle ... should be social planners value for
        # % similar model as the one used to find V
        # % a is size increase to smallest all points (use to increase small bubbles)
        # % d is factor to decrease max(weights) because some equilibria are just
        # % very large in number
        V = V[:,0:1]
        a = 5
        d = 0.005
        Vu,Vx,Vz = np.unique(round(V+0.000001,3),'rows') 
        N = len(Vu[:,1]) 

        if adjust==1:
            weights = np.ones(N,1)
            for i in range(1,N):
                weights[i,1] = sum( Vz == i) # xxx fix here 
            # weights = a+weights./(d*max(weights)) # dot operation
        else:
            weights = np.ones(len(Vu),1)

        if jitter==1:
            e1=np.random.normal(sigma,N,1) # matlab normrnd(0,sigma,N,1)
            e2=np.random.normal(sigma,N,1)
        else:
            e1 = np.zeros(N,1)
            e2 = e1

        """
        Haven't touched any of this, it just draws some stuff
        """
        # 	scatter(Vu(:,1)+e1,Vu(:,2)+e2,weights,'filled')
        # 	grid on

        # 	if nargin>4
        #    line([0,VM],[VM,0]) 
        #    line([0,0],[0,VM])
        #    line([0,VM],[0,0])

        return Vu,Vx,Vz

    def rlsindex(self, nC):
        Ns = 0
        for i in range(0,nC+1):
            Ns = Ns + i**2
        T = 1 + 3 * (nC-1)
        Ns_in_stages = np.ones((1,T))
        j = 0
        l = nC
        while l>1:
            Ns_in_stages[0,j] = 1
            j += 1
            Ns_in_stages[0,j] = 2*(l-1)
            j += 1
            Ns_in_stages[0,j] = (l-1)**2
            j += 1
            l -= 1
        return np.cumsum(Ns_in_stages)

    def populate_p(self):
        """
        Simple list with 0 as last value
        """

        for a in range(0,self.nC):
            if a < self.nC - 1:
                self.p[a] = self.p[a]*self.pf
            else:
                self.p[a] = 0

    def populate_firms(self):
        for i in range(0,self.nC):
            self.firm2[i,:i+1,:i+1] = np.matlib.repmat(self.C[:i+1],i+1,1)
            self.firm1[i] = self.firm2[i].transpose().copy()
        