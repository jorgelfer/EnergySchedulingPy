"""
by Jorge

scheduling code
"""

# required for processing
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from scipy import sparse

class LP_dispatch:

    def __init__(self, pf, PTDF, batt, Pjk_lim, Gmax, cgn, clin, cdr, v_base, dvdp, storage, vmin, vmax):
        # constructor
        ###########
        
        # preprocess
        self.ipf = 1 / pf.values

        # correct the PTDF by the penalty factor
        self.PTDF = PTDF 
        PTDF_pf = self.ipf.T * PTDF.values 
        self.PTDF_pf = pd.DataFrame(PTDF_pf, index=PTDF.index, columns=PTDF.columns)

        # attributes
        self.n = len(PTDF.columns)  #number of nodes
        self.l = len(PTDF)          #number of lines
        self.Pjk_lim = Pjk_lim           #thermal limit of the line
        self.Gmax = Gmax                 #Max generation
        self.cgn = cgn                   #Cost of generation
        self.clin = clin                 #"cost of lines"
        self.cdr = cdr                   # cost of demand response
        self.v_base = v_base             #Base voltage of the system in volts
        self.vmin = vmin * 1000             # voltage mi
        self.vmax = vmax * 1000             # voltage max
        self.storage = storage 
        self.DR = True

        # correct the voltage sensitivity by the penalty factor
        self.dvdp = dvdp
        dvdp_pf = self.ipf.T * dvdp.values 
        self.dvdp_pf = pd.DataFrame(dvdp_pf, index=dvdp.index, columns=dvdp.columns)
        
        # battery attributes
        self.numBatteries = batt['numBatteries']
        self.batIncidence = batt['BatIncidence']
        self.batSizes     = batt['BatSizes']
        self.batChargingLimits   = batt['BatChargingLimits']  
        self.batEfficiencies      = batt['BatEfficiencies']     
        self.batInitEnergy   = batt['BatInitEnergy']  
        self.batPenalty   = batt['BatPenalty']    
        self.ccharbat     = batt['ccharbat']    
        self.ccapacity    = batt['ccapacity']

    # Methods
    def PTDF_LP_OPF(self, demandProfile,Pjk_0, v_0, Pg_0, PDR_0):

        # define number of points
        self.pointsInTime = np.size(demandProfile, 1)

        # build equality constraints matrices
        Aeq, beq = self.__buildEquality(demandProfile)
        
        # build inequality constraints matrices
        A, b = self.__buildInequality(v_0, Pg_0, PDR_0)

        # build cost function and bounds
        ub, lb, f = self.__buildCostAndBounds(demandProfile)
        
        # add storage portion
        if self.storage:
            # modify A, Aeq
            Aeq, A = self.__addStorage_A(Aeq, A)
            # modify beq, lb, ub, f                beq, lb, ub, f
            beq, lb, ub, f= self.__addStorage_rest(beq, lb, ub, f)

        # pass as list for gurobi
        lb = lb[0].tolist()
        ub = ub[0].tolist()
            
        # compute linear program optimization
        x, m, LMP = self.__linprog(f, Aeq, beq, A, b, lb, ub)
            
        return x, m, LMP, A

    # helper methods
    def __incidenceMat(self):
        """This method computes a matrix that defines which lines are connected to each of the nodes"""
        PTDF = self.PTDF
        
        Node2Line = np.zeros((self.n,self.l))

        for i in range(self.n): # node loop
            for j in range(self.l): # lines loop
                if PTDF.columns[i] == PTDF.index[j].split("-")[0]:
                    Node2Line[i,j] = 1
                elif PTDF.columns[i] == PTDF.index[j].split("-")[1]:
                    Node2Line[i,j] = -1
        return -Node2Line #- due to power criteria

    def __buildCostAndBounds(self, demandProfile):
        
        
        # line limits
        Pjk_lim = self.Pjk_lim
        Pjk_lim = np.reshape(Pjk_lim.values.T, (1,np.size(Pjk_lim.values)), order="F") 

        if self.DR:
            # max demand response
            DRmax = np.reshape(demandProfile.values.T, (1, demandProfile.size), order="F")

            #  define upper and lower bounds
            ub = np.concatenate((self.Gmax, DRmax, Pjk_lim),1)
            lb = np.concatenate((np.zeros((1, 2*self.n*self.pointsInTime)), -Pjk_lim),1)
            ## define coeffs
            f = np.concatenate((self.cgn, self.cdr, self.clin),1)
        else:
            
            #  define upper and lower bounds
            ub = np.concatenate((self.Gmax, Pjk_lim),1)
            lb = np.concatenate((np.zeros((1, self.n * self.pointsInTime)), -Pjk_lim),1)
            ## define coeffs
            f = np.concatenate((self.cgn, self.clin),1)

        return ub, lb, f 

    def __buildInequality(self, v_0, Pg_0, PDR_0):
        """Build inequality constraints"""

        # initial power 
        self.Pg_0 = np.reshape(Pg_0.values.T, (1,np.size(Pg_0)), order="F")
        # initial demandResponse 
        self.PDR_0 = np.reshape(PDR_0.values.T, (1,np.size(PDR_0)), order="F")
        
        #### for voltage ###

        # define limits 
        v_base = np.reshape(self.v_base.values.T, (1, np.size(self.v_base.values)), order="F")
        v_lb = -(self.vmin * v_base)
        v_ub = (self.vmax * v_base)

        # compute matrices 
        A, b = self.__buildSensitivityInequality(self.dvdp_pf, v_0, v_lb, v_ub)

        return A, b 

    def __buildSensitivityInequality(self, dxdp, x_0, x_lb, x_ub):
        """for both voltage and flows inequalities the procedure is very similar
            This method seeks to standardize the procedure"""
            
        dxdp = dxdp.values # remove the dataframe

        # Define A
        if self.DR:
            # define A
            A = np.block([[-dxdp, -dxdp, np.zeros((self.n,self.l))],     # -d/dp * Pg - dv/dp * Pdr + 0*Pjk
                           [dxdp, dxdp, np.zeros((self.n,self.l))]])     # d/dp * Pg + dv/dp * Pdr + 0*Pjk
            A = sparse.kron(sparse.csr_matrix(A), sparse.csr_matrix(np.eye(self.pointsInTime)))
        else:
            # define A
            A = np.block([[-dxdp, np.zeros((self.n,self.l))],     # -dv/dp * Pg 
                          [dxdp, np.zeros((self.n,self.l))]])     # -dv/dp * Pg
            A = np.kron(sparse.csr_matrix(A), sparse.csr_matrix(np.eye(self.pointsInTime)))   

        # Define b 

        # reshape initial value
        x_0 = np.reshape(x_0.values.T, (1,np.size(x_0.values)), order="F")

        # dxdp @ P0:
        # expand dxdp
        dxdp_kron = np.kron(dxdp, np.eye(self.pointsInTime))
        aux_dxdp = dxdp_kron @ (self.Pg_0 + self.PDR_0).T
        
        b = np.concatenate((x_lb.T + x_0.T - aux_dxdp, 
                            x_ub.T - x_0.T + aux_dxdp), axis=0)

        return A, b

    def __buildEquality(self, demandProfile):
        """Build equality constraints"""

        # Compute the demand portion of the PTDF-OPF definition:
        # i.e. Pjk*(-PTDF) = -PTDF * Pd (the right hand side is the demand portion)
        DPTDF = - self.PTDF @ demandProfile
        DPTDF = np.reshape(DPTDF.values.T, (1,DPTDF.size), order="F")

        # Demand vector for each hour
        D = np.reshape(demandProfile.values.T,(1,demandProfile.size), order="F")

        # compute the incidence matrix
        Imat = self.__incidenceMat()

        # Define Aeq:
        if self.DR:
            # Aeq1 (Nodal Balance, Demand Response, Line Change) 
            AeqPrim = np.block([[self.ipf.T * np.identity(self.n), self.ipf.T * np.identity(self.n), Imat], #% Nodal Balance Equations
                            [-self.PTDF_pf.values, -self.PTDF_pf.values, np.identity(self.l)]])               #% Change in Flows Equations
            Aeq = sparse.kron(sparse.csr_matrix(AeqPrim), sparse.csr_matrix(np.eye(self.pointsInTime)))                         #% Expand temporal equations
                  
        else:
            # Aeq1 (Nodal Balance, Line Change) 
            AeqPrim = np.block([[self.ipf.T * np.identity(self.n), Imat],   #% Nodal Balance Equations
                            [-self.PTDF_pf.values, np.identity(self.l)]])    #% Change in Flows Equations
            Aeq = sparse.kron(sparse.csr_matrix(AeqPrim), sparse.csr_matrix(np.eye(self.pointsInTime)))      #% Expand temporal equations

        # Define beq:

        # total demand for each hour
        beq = np.concatenate((D, DPTDF),1).T

        return Aeq, beq
    
    def __addStorage_A(self, Aeq, A):
        """Compute the battery portion for A's"""
                
        # Aeq:
        # columns: Pnt1,Pntf 
        # rows:    (n+l)*PointsInTime + numBatteries*(PointsInTime+2)
        if self.DR:
            Aeq1 = sparse.vstack( (Aeq, sparse.csr_matrix(np.zeros((self.numBatteries*(self.pointsInTime + 2), (2*self.n+self.l)*self.pointsInTime)) )) ) #% Adding part of batteries eff. equations
        else:
            Aeq1 = sparse.vstack( (Aeq, sparse.csr_matrix(np.zeros((self.numBatteries*(self.pointsInTime + 2), (self.n+self.l)*self.pointsInTime)) )) ) #% Adding part of batteries eff. equations

        ############
        # Equalities
        ############
        
        # build Aeq2:
        # columns: PscBt1,PscBtf,...,PsdBt1,PsdBtf,...,EBt1,EBtf 
        # rows:    (n)*PointsInTime 
        Aeq2 = self.__storageAeq2()
        
        # build Aeq2_auxP and Aeq2_auxE:
        # Aeq2_auxP:
        # columns: PscBt1,PscBtf,...,PsdBt1,PsdBtf 
        # rows:    n*PointsInTime (only nodes with storage connected)
        # Aeq2_auxE:
        # columns: EBt1,EBtf 
        # rows:    (PointsInTime + 1)*numBatteries  
        Aeq2_auxP, Aeq2_auxE, Aeq2_auxE0 = self.__storageAeq2_aux()
        
        # Adding Energy Balance and Initial Conditions
        # Aeq2 at this point:
        # columns: PscBt1,PscBtf,...,PsdBt1,PsdBtf,...,EBt1,EBtf 
        # rows:    n*PointsInTime + (1 + PointsInTime)*numBatteries
        Aeq2_aux = sparse.hstack((Aeq2_auxP, sparse.csr_matrix(Aeq2_auxE)))
        Aeq2 = sparse.vstack( (Aeq2, Aeq2_aux) ) 
        
        #Energy Storage final conditions
        # Aeq2 finally:
        # columns: PscBt1,PscBtf,...,PsdBt1,PsdBtf,...,EBt1,EBtf 
        # rows:    (n+l)*PointsInTime + (2 + PointsInTime)*numBatteries
        Aeq2_aux2 = np.concatenate((np.zeros((self.numBatteries, Aeq2_auxP.get_shape()[1] )), np.flip(Aeq2_auxE0.T, 1), np.flip(np.eye(self.numBatteries), 0)), axis=1)
        Aeq2 = sparse.vstack( (Aeq2, sparse.csr_matrix(Aeq2_aux2)) )
        
        # Build Aeq matrix
        # Aeq:
        # columns: Pnt1,Pntf-Pjkt1,Pjktf-PscBt1,PscBtf-PsdBt1,PsdBtf-EBt1,EBtf 
        # rows:    (n+l)*PointsInTime + (2 + PointsInTime)*numBatteries
        Aeq = sparse.hstack((Aeq1,Aeq2))
        
        ############
        # Inequalities
        ############
        
        # build Ain:
        Ain = self.__storageAin(A)
        
        return Aeq, Ain
    
    def __addStorage_rest(self, beq, lb, ub, f):
        """add storage portion to beq, lb, ub, f"""
        
        # add storage portion to beq
        beq = np.concatenate((beq, np.zeros((self.pointsInTime * self.numBatteries,1)), 
                              self.batInitEnergy.T, np.zeros((self.numBatteries,1))),0) # Add Balance Energy, Init & Final Conditions


        # add storage portion to lower bounds
        lb = np.concatenate(( lb.T,         # Generation limits
              np.kron(self.batChargingLimits,np.zeros((1,self.pointsInTime))).T,             # Charging limits
              np.kron(self.batChargingLimits,np.zeros((1,self.pointsInTime))).T,             # Discharging limits
              np.kron(self.batSizes,np.zeros((1,self.pointsInTime))).T,               # Battery capacity limits
              np.zeros((self.numBatteries,1)) ),0)                            # Initial capacity limits

        # add storage portion to upper bounds
        ub = np.concatenate(( ub.T,                         # Generation & Line limits
              np.kron(self.batChargingLimits,np.ones((1,self.pointsInTime))).T,    # Charging limits
              np.kron(self.batChargingLimits,np.ones((1,self.pointsInTime))).T,    # Discharging limits
              np.kron(self.batSizes,np.ones((1,self.pointsInTime))).T,             # Battery capacity limits
              self.batSizes.T), 0)                                            # Initial capacity limits

        f = np.concatenate((f, self.ccharbat, self.ccapacity),1) # % x = Pg Pdr Plin Psc Psd E E0

        return beq, lb, ub, f
    
    def __storageAin(self, A1):
        """include storage in inequality constraints"""
        
        # preprocess
        row, _ = np.where(self.batIncidence==1) # get nodes with storage
                
        # for the voltage
        A2 = self.__storageSensitivityA(row, self.dvdp)

        # finally concatenate with the original matrix
        Ain = sparse.hstack((A1,A2))
        
        return Ain

    def __storageSensitivityA(self, row, dxdp):
        
        dxdp = dxdp.values

        # portion related to lower bound 

        A_b1_aux = np.concatenate( (dxdp[:,row], - 1./(self.batPenalty)*dxdp[:,row]), 1)
        A_b1_aux1 = sparse.kron( A_b1_aux, sparse.csr_matrix(np.eye(self.pointsInTime)) )
        A_b1 = sparse.hstack(( A_b1_aux1, #-dx/dp *(-Psd+Psc)
                             sparse.csr_matrix(np.zeros((dxdp.shape[0]*self.pointsInTime, self.numBatteries*(self.pointsInTime + 1)) )) )) # -dx/dp *(0*E)
        
        # portion related to upper bound 
        A_b2_aux = np.concatenate( (-dxdp[:,row], 1./(self.batPenalty)*dxdp[:,row]), 1)
        A_b2_aux1 = sparse.kron( A_b2_aux, sparse.csr_matrix(np.eye(self.pointsInTime)) )
        A_b2 = sparse.hstack(( A_b2_aux1, # dx/dp *(-Psd+Psc)
                             sparse.csr_matrix(np.zeros((dxdp.shape[0]*self.pointsInTime,self.numBatteries*(self.pointsInTime + 1)) )) ))# -dx/dp *(0*E)

        # concatenate both blocks
        A2 = sparse.vstack((A_b1, A_b2))

        return A2 
    
    def __storageAeq2(self):
        """create auxiliary matrices to include storage in equality constraints"""
        
        ipf = np.expand_dims(self.ipf, axis=1)
        # preprocessing
        batIncid_Psc = sparse.kron(sparse.csr_matrix(self.batIncidence), sparse.csr_matrix(np.eye(self.pointsInTime)))
        batIncid_Psd = sparse.kron(sparse.csr_matrix(ipf * self.batIncidence), sparse.csr_matrix(np.eye(self.pointsInTime)))

        PTDFV = self.PTDF.values
        row, _ = np.where(self.batIncidence==1) # get nodes with storage
        
        #% Aeq2 (Energy Storage impact on Nodal & Line Equations, Energy Balance, Energy Storage Initial and Final Conditions)
        #% Impact on Nodal Equations
        
        #Batt penalty
        Aeq2 = sparse.hstack( (-batIncid_Psc, # * np.kron(BatPenalty, np.ones((1, PointsInTime)))
                                batIncid_Psd, 
                                sparse.csr_matrix(np.zeros((self.n*self.pointsInTime,self.numBatteries*(self.pointsInTime+1)))) ))
        
        # Impact on Line Equations
        aux_Aeq2lin = np.concatenate( (PTDFV[:,row], -1./(self.batPenalty)*PTDFV[:,row]), 1)
        
        Aeq2lin = sparse.hstack(( sparse.kron(sparse.csr_matrix(aux_Aeq2lin), sparse.csr_matrix(np.eye(self.pointsInTime)) ),
                                  sparse.csr_matrix(np.zeros((self.l*self.pointsInTime,self.numBatteries*(self.pointsInTime + 1)) )) ))
        

        # Aeq2 at this point:
        # columns: PscBt1,PscBtf,...,PsdBt1,PsdBtf,...,EBt1,EBtf 
        # rows:    (n+l)*PointsInTime        
        Aeq2 = sparse.vstack((Aeq2,Aeq2lin))
        
        return Aeq2
    
    def __storageAeq2_aux(self):
        """create auxiliary matrices to include storage in equality constraints"""
        # preprocess
        batIncid_Psc = sparse.kron(sparse.csr_matrix(np.eye(len(self.batEfficiencies.T)) * self.batEfficiencies), sparse.csr_matrix(np.eye(self.pointsInTime)))
        batIncid_Psd = sparse.kron(sparse.csr_matrix(np.eye(len(self.batEfficiencies.T)) * 1/self.batEfficiencies), sparse.csr_matrix(np.eye(self.pointsInTime)))

        # Energy Balance Equations
        # Aeq2_auxP:
        # columns: PscBt1,PscBtf,...,PsdBt1,PsdBtf 
        # rows:    n*PointsInTime (only nodes with storage connected)   
        Aeq2_auxP = sparse.hstack( (-batIncid_Psc,
                                    batIncid_Psd) )
        Aeq2_auxP = sparse.vstack(( Aeq2_auxP, sparse.csr_matrix(np.zeros((self.numBatteries, Aeq2_auxP.get_shape()[1] ))) ))
        
        # Aeq2_auxE:
        # columns: EBt1,EBtf 
        # rows:    (PointsInTime + 1)*numBatteries  
        Aeq2_auxE = np.eye( (self.pointsInTime + 1) * self.numBatteries)
    
        for i in range(self.numBatteries):
            init = i*self.pointsInTime
            endit = i*self.pointsInTime + self.pointsInTime
            Aeq2_auxE[init:endit,init:endit] = Aeq2_auxE[init:endit,init:endit] - np.eye(self.pointsInTime,k=-1)
        
        idx_E = [self.pointsInTime*self.numBatteries, self.pointsInTime*self.numBatteries]
        idx_E0 = [self.pointsInTime*self.numBatteries, (self.pointsInTime + 1)*self.numBatteries]
        Aeq2_auxE0 = np.zeros( (self.pointsInTime * self.numBatteries, self.numBatteries) )
        c = 0
        s = np.sum(Aeq2_auxE[:idx_E[0],:idx_E[1]], 1);
    
        for i in range(self.pointsInTime * self.numBatteries):
            if s[i]==1:
                Aeq2_auxE0[i,c] = -1
                c += 1
    
        Aeq2_auxE[:idx_E0[0], idx_E0[0]:idx_E0[1]+1] = Aeq2_auxE0

        return Aeq2_auxP, Aeq2_auxE, Aeq2_auxE0
    
    def __linprog(self, f, Aeq, beq, A, b, lb, ub):
        """compute LP optmization using gurobi"""
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            with gp.Model(env=env) as m:

                # create a new model
                m = gp.Model("LP1")
                m.Params.OutputFlag = 0
                
                # create variables
                x = m.addMVar(shape=Aeq.get_shape()[1], lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name="x")
                
                # multipy by the coefficients
                m.setObjective(f @ x, GRB.MINIMIZE)
                
                # add equality constraints
                m.addConstr(Aeq @ x == np.squeeze(beq), name="eq")

                # add inequality constraints
                m.addConstr(A @ x <= np.squeeze(b), name="ineq")
                ALMP = sparse.hstack( (Aeq.transpose(),A.transpose()) )
                                     
                # Optimize model
                m.optimize()
                
                # Compute LMP
                LMP = ALMP @ np.expand_dims(m.Pi,1)

        return x, m, LMP 

# +=================================================================================================
           
def main():

    print('please run the driver first')    
# +=================================================================================================
if __name__ == "__main__":
    main()
    

# with open('correctMat.npy', 'rb') as k:
#     fc = np.load(k)
#     Aeqc = np.load(k)
#     beqc = np.load(k)
#     Ac = np.load(k)
#     bc = np.load(k)
#     lbc = np.load(k)
#     ubc = np.load(k)

# with open('errorMat2.npy', 'rb') as k:
#     fe = np.load(k)
#     Aeqe = np.load(k)
#     beqe = np.load(k)
#     Ae = np.load(k)
#     be = np.load(k)
#     lbe = np.load(k)
#     ube = np.load(k)
