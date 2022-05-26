# Define the leapfrogging model

"""
This is not even close to done 😎
"""
import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
# %matplotlib inline
import quantecon as qe
import scipy.optimize as optimize
import scipy.sparse as sparse
from quantecon import compute_fixed_point
from quantecon.markov import DiscreteDP

class leapfrogging:

    def __init__(self, Cmax = 5, Cmin = 0, nC = 4, pf = 1, k0 = 0, k1 = 8.3, k2 = 1, R = 0.05, Dt = 1):
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
        self.r1 = lambda x1,x2: np.maximum(x2-x1,0)
        self.r2 = lambda x1,x2: np.maximum(x1-x2,0)
        self.Phi = lambda vN,vI: np.maximum(vN, vI) # xxx this should maybe be made to accept lists.

    """
    xxx this is a very long if function :(
    to do:
    fix the functions dependent on mp (Think it's the same as self)
    fix all the for loops, might have to start on 0
    define all the self.solve functions
        
    """
    def state_recursion(self,ss,ESS,tau,mp): 
        if tau == self.T:
            ss,Ess = self.solve_last_corner(ss,ESS,mp)
            tau -= 1
        if tau == self.T -1:
            ss,Ess = self.solve_last_edge(ss,ESS,mp)
            tau -= 1
        if tau == self.T -2:
            ss,Ess = self.solve_last_interior(ss,ESS,mp)
            tau -= 1
        
        dothis = 1
        while dothis == 1: # break when tau=0 
            if np.remainder(tau,3)==1: 
                ic = np.ceil((tau+2)/3)
                ss,ESS = self.solve_corner(ss,ic,ESS,mp)
                tau -= 1
                if tau == 0:
                    break
            if np.remainder(tau,3)==0:
                ic = np.ceil((tau+2)/3)
                ss, ESS = self.solve_edge(ss,ic,ESS,mp)
                tau -= 1
            if np.remainder(tau,3) == 2:
                ic = ceil((tau+2)/3)
                ss, ESS = self.solve_interior(ss,ic,ESS,mp)
                tau -= 1
        # end of the scary while loop
        return ss, ESS
    
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
        eq = {'P1':[], 'vN1':[], 'vI1':[], 'P2':[], 'vN2':[], 'vI2':[]}
        # arrays which will consist of dictionaries
        EQs = np.empty(shape=(5,5,5),dtype='object')
        tau = np.empty(shape=(5),dtype='object')
        for i in range(N):
            EQs[i,i,4] = {'eq':eq} # 4 is because max 5 eqs ... consult litterature
            tau[i] = {}
            tau[i]['EQs'] = EQs  # container for identified equilibriums
            tau[i]['nEQ'] = np.zeros((i+1,i+1)) # container for number of eqs in (x1,x2,c) point
                
			   

			# %  #  ##  ###  ####     State space with 4 stages.
			# %     ##  ###  ####     4x4 Hashtag field is reached with complete
			# %         ###  ####     technological development.
			# %              ####     for each hashtag (x1,x2,c) - point in state space - max 5 eq's
        return tau

    # define all the solve functions used by state_recursion
    def solve_last_corner(self,ss,ESS):
        h = self.nC # Number of technological levels
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
        # ss(h).EQs(h,h,1).eq = self.EQ(P1,vN1,vI1,P2,vN2,vI2)
        # # Only one equilibrium is possible:
        # ss(h).nEQ(h,h) = 1
        # ESS.bases(ESS.index(h,h,h)) = 1
        return ss, ESS

    """
    xxx
    To do:
    Define all of the functions used :(
    """
    def solve_last_edge(self,ss,ESS,mp):
        # % INPUT:
        # % cost and mp are global parameters
        # % ss state space structure (has info on eq's in corner of last layer)
        # % OUTPUT:
        # % Equilibria lf.EQ(P1, vN1, vI1, P2, vN2, vI2) for edge state space points
        # % of the final layer:
        # % Final layer <=> s=(x1,x2,c) with c = min(mp.C) 
        # % Edge <=> s=(x1,x2,c) with x2 = c = min(mp.C) and x1 > c or
        # % s=(x1,x2,c) with x1 = c = min(mp.C) and x2 > c

        ic = self.nC # Get the level of technology final layer
        c = self.Cmin # Get state of the art marginal cost for tech. of final layer

        h = 1
        # % h is used to select equilibria in the corner of the final layer but there
        # % is only ever 1 equilibria in the corner
        # % If we did not apply this apriori knowledge we would have to use ESS
        # % max(vN,vI | at the corner final layer)= mp.Phi(ss(ic).EQs(ic,ic,h).eq.vN1,ss(ic).EQs(ic,ic,h).eq.vI1)

        # Get the value of max choice in the corner of final layer s = (c,c,c)
        # xxx lots of nested functions here that are not defined
        g1_ccc = np.maximum(ss(ic).EQs(ic,ic,h).eq.vN1,ss(ic).EQs(ic,ic,h).eq.vI1)
        g2_ccc = np.maximum(ss(ic).EQs(ic,ic,h).eq.vN2,ss(ic).EQs(ic,ic,h).eq.vI2)

        # Player 2 is at the edge s=(x1,x2,c) with x2=c=min(mp.C) and x1>c
        # xxx start on 0?
        for ic1 in range(1,ic-1):
            x1 = self.C(ic1)
            vI1 = self.r1(x1,c) - self.K(c)  + self.beta * g1_ccc
            vN1search = lambda z: self.r1(x1,c) + self.beta * self.Phi(z,vI1) - z
            vN1 = optimize.fsolve(vN1search,0)
            P1 = vI1 > vN1


            vN2 = ( self.r2(x1,c) + self.beta * (P1*g2_ccc+(1-P1)*self.Phi(0,-mp.K(c))) )  /  ( 1-self.beta*(1-P1) )
            vI2 = vN2 - self.K(c)
            P2 = vI2 > vN2

            # xxx wtf is happening here

            # ss(ic).EQs(ic1,ic,h).eq = lf.EQ(P1, vN1, vI1, P2, vN2 , vI2)
            # ss(ic).nEQ(ic1,ic) = 1
            # ESS.bases(ESS.index(ic1,ic,ic)) = 1
        
        # xxx maybe start at 0?
        # Player 1 is at the edge s=(x1,x2,c) with x1=c=min(mp.C) and x2>c
        for ic2 in range(1,ic-1):
            x2 = self.C(ic2)
            vI2 = self.r2(c,x2) - self.K(c) + self.beta * g2_ccc
            vN2search = lambda x: self.r2(c, x2) + self.beta*self.Phi(x,vI2)-x
            vN2 = optimize.fsolve(vN2search,0)
            P2 = vI2 > vN2


            vN1 = (self.r1(c, x2) + self.beta*(P2*g1_ccc+(1-P2)*self.Phi(0, -self.K(c))))  /  ( 1-self.beta*(1-P2) )
            vI1 = vN1-self.K(c)
            P1 = vI1 > vN1

            # wtf is happening here xx
            # ss(ic).EQs(ic, ic2, 1).eq = self.EQ(P1, vN1, vI1, P2, vN2, vI2)
            # ss(ic).nEQ(ic, ic2) = 1
            # ESS.bases(ESS.index(ic,ic2,ic)) = 1

        return ss,ESS
    """
    To do: xxx
    define quad
    Define the functions used :(
    """
    def solve_last_interior(self,ss,ESS,mp):
        # outside loop
        ic = self.nC
        c = self.C(ic)

        """
        These very long lambda functions have a lot of stuff that's not defined
        I think they just find whether investing or not investing gives highest utility
        """
        g1 = lambda iC1, iC2, iC: np.maximum(ss(iC).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC))+1).eq.vN1,ss(iC).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC))+1).eq.vI1)
        g2 = lambda iC1, iC2, iC: np.maximum(ss(iC).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC))+1).eq.vN2,ss(iC).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC))+1).eq.vI2)

    # xxx start on 0?
        for ic1 in range(1,ic-1): #Player 1 loop begin
            for ic2 in range(1,ic-1): #Player 2 loop begin                
                # Player 1 -> leads to P2 candidates
                # what is g1 xxx
                a = self.r1(self.C(ic1), self.C(ic2)) - self.K(c) + self.beta*g1(ic, ic2, ic) #check
                b = self.beta*(g1(ic, ic, ic)-g1(ic, ic2, ic)) # check
                d = self.r1(self.C(ic1),self.C(ic2))
                e = self.beta*g1(ic1, ic, ic)


                b_0 = - self.beta * b # check 
                b_1 = self.beta * g1(ic1, ic, ic) + (self.beta-1)*b - self.beta*a # check
                b_2 = self.r1(self.C(ic1),self.C(ic2)) + (self.beta-1) * a # check 


                pstar2 = self.quad(b_0, b_1, b_2)
                # always return 1 and 0 for the pure strategies


                # Player 2 -> leads to P1 candidates
                A = self.r2(self.C(ic1), self.C(ic2)) - self.K(c) + self.beta*g2(ic1, ic, ic)
                B = self.beta*(g2(ic, ic, ic)-g2(ic1, ic, ic))
                D = self.r2(self.C(ic1),self.C(ic2))
                E = self.beta*g2(ic, ic2, ic)

                d_0 = - self.beta * B
                d_1 = self.beta*g2(ic, ic2, ic) + (self.beta-1) * B - self.beta*A
                d_2 = self.r2(self.C(ic1),self.C(ic2)) + (self.beta-1) * A
                
                pstar1 = self.quad(d_0, d_1, d_2)
                    
                    # % Find equilibria based on candidates
                    # % Number of equilibria found are 0 to begin with
                count = 0
                for i in range(len(pstar1)):
                        for j in range(len(pstar2)):
                            if set([i,j]).issuperset(set([1,2])): # matlab code: all(ismember([i,j],[1,2])) # these are pure strategies
                                # % If the polynomial is negative vI > vN
                                # % hence player invests set exPj=1 else 0
                                # % exP1 is best response to pstar2(j)
                                exP1 = b_2 + b_1 * pstar2(j) + b_0 * pstar2(j)^2 < 0
                                exP2 = d_2 + d_1 * pstar1(i) + d_0 * pstar1(i)^2 < 0

                                # % check if both are playing best response
                                # % in pure strategies. Players best response
                                # % should be equal to the candidate to which
                                # % the other player is best responding.
                                if abs(exP1 - pstar1(i)) < 1e-8 and abs(exP2-pstar2(j)) < 1e-8:
                                    # % if exP1=0 and pstar_i=0 true
                                    # % if exP1=1 and pstar_i=1 true
                                    # % Testing whether best response exP1 is
                                    # % equal to pstar1(i) to which Player 2
                                    # % is best responding ...
                                    count += 1
                                    vI1 = a + b*pstar2(j) 
                                    vN1 = (d + e*pstar2(j) + self.beta*(1-pstar2(j))*(a+b*pstar2(j)))*pstar1(i) + (1-pstar1(i))*(d+e*pstar2(j))/(1-self.beta*(1-pstar2(j)))
                                    vI2 = A + B*pstar1(i); 
                                    vN2 = (D + E*pstar1(i) + self.beta*(1-pstar1(i))*(A+B*pstar1(i)))*pstar2(j) + (1-pstar2(j))*(D+E*pstar1(i))/(1-self.beta*(1-pstar1(i)))

                                    ss(ic).EQs(ic1, ic2, count).eq = self.EQ(pstar1(i),vN1,vI1,pstar2(j),vN2,vI2)
                        
                            elif i > 2 and j > 2 and pstar1(i) >= 0 and pstar2(j) >= 0 and pstar1(i) <= 1 and pstar2(j) <= 1:
                                count += 1
                                v1 = a + b * pstar2(j)
                                v2 = A + B * pstar1(i)
                                ss(ic).EQs(ic1, ic2, count).eq = self.EQ(pstar1(i),v1,v1,pstar2(j),v2,v2)
                        # no more j loop
                # no more i loop    
                # wtf is happening here xx    
                # ss(ic).nEQ(ic1, ic2) = count
                # ESS.bases(ESS.index(ic1,ic2,ic)) = count
        return ss, ESS # end of solve_last_interior


    def solve_interior(self,ss,ic,ESS,mp):
        # % INPUT 
        # % ss is state space structure with solutions for final layer edge and
        # % corner
        # % ic is the level of technology for which to solve
        # % ESS is struc with information holding ESS.esr being equilibrium selection
        # % rule and ESS.bases being the bases of the ESS.esr's
        
        c = self.C(ic)
        for ic1 in range(1,ic-1):
            for ic2 in range(1,ic-1):
                ss,ESS = self.find_interior(ss,mp,ic1,ic2,ic,c,ESS)
        return ss, ESS
    
    def find_interior(self,ss,mp,ic1,ic2,ic,c,ESS):
        # % INPUT
        # % ss is state space
        # % mp and cost holds global parameters
        # % ic1 and ic2 are indexes for player 1 and player 2 such that mp.C(ic1)
        # % and mp.C(ic2) are marginal cost of the players
        # % k is the level of technology
        # % c=mp.C(k) state of the art marginal cost ... could be found inside
        # % function
        # % 

        # % get probability of technological development
        p = self.p(ic)
        q = 1-p

        # % h is used for selected equilibrium in state realized when technology
        # % develops hence ic+1 in ESS.index(ic1,ic2,ic+1)
        h = ESS.esr(ESS.index(ic1,ic2,ic+1))+1



        # % H1(iC1, iC2, iC) = (1-pi)*?(?[iC+1].EQs[iC1, iC2, ESS[iC1,iC2,iC+1]].vN1,?[iC+1].EQs[iC1, iC2, ESS[iC1,iC2,iC+1]].vI1) + pi*?(?[iC].EQs[iC1, iC2, ESS[iC1,iC2,iC]].vN1,?[iC].EQs[iC1, iC2, ESS[iC1,iC2,iC]].vI1)
        # % H2(iC1, iC2, iC) = (1-pi)*?(?[iC+1].EQs[iC1, iC2, ESS[iC1,iC2,iC+1]].vN2,?[iC+1].EQs[iC1, iC2, ESS[iC1,iC2,iC+1]].vI2) + pi*?(?[iC].EQs[iC1, iC2, ESS[iC1,iC2,iC]].vN2,?[iC].EQs[iC1, iC2, ESS[iC1,iC2,iC]].vI2)

        """
        These probably need to be fixed...
        """
        H1 = lambda iC1, iC2, iC: p*self.Phi(  ss(iC+1).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC+1))+1).eq.vN1 , ss(iC+1).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC+1))+1).eq.vI1  ) + (1-p)*self.Phi(  ss(iC).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC))+1).eq.vN1 , ss(iC).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC))+1).eq.vI1  )
        H2 = lambda iC1, iC2, iC: p*self.Phi(  ss(iC+1).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC+1))+1).eq.vN2 , ss(iC+1).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC+1))+1).eq.vI2  ) + (1-p) * self.Phi(  ss(iC).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC))+1).eq.vN2 , ss(iC).EQs(iC1, iC2, ESS.esr(ESS.index(iC1,iC2,iC))+1).eq.vI2  )

        a = self.r1( self.C(ic1),self.C(ic2) ) - self.K(c) + self.beta * H1(ic,ic2,ic) #check
        b = self.beta * (   H1(ic,ic,ic) - H1(ic,ic2,ic)  ) #check
        d = self.r1( self.C(ic1),self.C(ic2) )   + self.beta * p * self.Phi( ss(ic+1).EQs(ic1,ic2,h).eq.vN1 , ss(ic+1).EQs(ic1,ic2,h).eq.vI1  )
        e = self.beta * H1(ic1,ic,ic)         - self.beta * p * self.Phi( ss(ic+1).EQs(ic1,ic2,h).eq.vN1 , ss(ic+1).EQs(ic1,ic2,h).eq.vI1  ) 

        pa = - self.beta * (1-p) * b
        pb = e + ( self.beta * (1-p) -1) * b - self.beta * (1-p) * a
        pc = d + ( self.beta * (1-p) -1 ) * a

        # Solve for p2 mixed strategy ... but also returns 1 and 0 for pure
        pstar2 = lf.quad(pa,pb,pc)

        A = self.r2(self.C(ic1),self.C(ic2)) - self.K(c) + self.beta * H2(ic1,ic,ic)
        B = self.beta * ( H2(ic,ic,ic) - H2(ic1,ic,ic) )
        D = self.r2(self.C(ic1),self.C(ic2)) + self.beta * p * self.Phi( ss(ic+1).EQs(ic1,ic2,h).eq.vN2 , ss(ic+1).EQs(ic1,ic2,h).eq.vI2 )
        E = self.beta * H2(ic,ic2,ic) - self.beta * p * self.Phi( ss(ic+1).EQs(ic1,ic2,h).eq.vN2 , ss(ic+1).EQs(ic1,ic2,h).eq.vI2 )

        qa = - self.beta * (1-p) * B
        qb = E + ( self.beta * (1-p) - 1 ) * B - self.beta * (1-p) * A
        qc = D + ( self.beta * (1-p) - 1 ) * A

        pstar1 = lf.quad(qa, qb, qc)


        count = 0
        for i in range(1,len(pstar1)):
            for j in range(1,len(pstar2)):
                if set([i,j]).issuperset(set([1,2])): # matlab code: all(ismember([i,j],[1,2]))
                    exP1 = pc + pb * pstar2(j) + pa * pstar2(j)^2 < 0 # xxx self.pstar?
                    exP2 = qc + qb * pstar1(i) + qa * pstar1(i)^2 < 0

                if abs(exP1 - pstar1(i)) < 1e-7 and abs(exP2-pstar2(j)) < 1e-7:
                    count += 1
                    vI1 = a + b*pstar2(j)
                    vN1 = (d + e*pstar2(j) + self.beta*q*(1-pstar2(j))*(a+b*pstar2(j)))*pstar1(i)+(1-pstar1(i))*(d+e*pstar2(j))/(1-self.beta*q*(1-pstar2(j)))
                    vI2 = A + B*pstar1(i)
                    vN2 = (D + E*pstar1(i) + self.beta*q*(1-pstar1(i))*(A+B*pstar1(i)))*pstar2(j)+(1-pstar2(j))*(D+E*pstar1(i))/(1-self.beta*q*(1-pstar1(i)))

                    ss(ic).EQs(ic1, ic2, count).eq = self.EQ(pstar1(i),vN1,vI1,pstar2(j),vN2,vI2) # xx wut?
                
                
                elif i > 2 and j > 2 and pstar1(i) >= 0 and pstar2(j) >= 0 and pstar1(i) <= 1 and pstar2(j) <= 1:
                    count += 1
                    v1 = a + b * pstar2(j)
                    v2 = A + B * pstar1(i)
                    ss(ic).EQs(ic1, ic2, count).eq = lf.EQ(pstar1(i),v1,v1,pstar2(j),v2,v2)
                    
            # end j loop
        # end i loop

        # ss(ic).nEQ(ic1,ic2) = count # xxx wut? 
        # ESS.bases(ESS.index(ic1,ic2,ic)) = count

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
        V = V[:,1:2]
        a = 5
        d = 0.005
        Vu,Vx,Vz = unique(round(V+0.000001,3),'rows') # xxx no idea what this does
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
        