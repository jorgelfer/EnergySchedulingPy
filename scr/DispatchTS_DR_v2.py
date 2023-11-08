"""
by Jorge
"""

# required for processing
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import pathlib
import os

class Dispatch:
    
    def __init__(self, PTDF, PointsInTime, batt, Lmax, Gmax, cgn, clin, cdr, vBase=None, vSensi=None, initVolts=None):
        # data frame containing the PTDF of the system to analyze     
        self.PTDF = PTDF
        self.PointsInTime = PointsInTime
        self.batt = batt
        self.n = len(self.PTDF.columns) #number of nodes
        self.l = len(self.PTDF)         #number of lines
        self.Lmax = Lmax
        self.Gmax = Gmax
        self.cgn = cgn
        self.clin = clin
        self.cdr = cdr
        self.vBase = vBase
        self.vSensi = vSensi
        self.initVolts = initVolts


    def PTDF_OPF(self, DemandProfile, voltage=False, storage=False, DR = False, PV=False):
        #--------------------------------------------------------
        # Build matrices
        #--------------------------------------------------------

        # preprocessing
        PTDFV = self.PTDF.values
        PointsInTime = self.PointsInTime
        vSensiVal = self.vSensi.values
        
        # compute the incidence matrix
        Imat = self.__incidenceMat()

        # Compute the demand portion of the PTDF-OPF definition:
        # i.e. Pjk*(-PTDF) = -PTDF * Pd (the right hand side is the demand portion)
        DPTDF = - PTDFV @ DemandProfile
        DPTDF = np.reshape(DPTDF.T, (1,DPTDF.size), order="F")
        # Demand vector for each hour
        D = np.reshape(DemandProfile.T,(1,np.size(DemandProfile)), order="F")
        # max demand response
        DRmax = D
        
        if DR:
            # Equalities
            # define Aeq
            # Aeq1 (Nodal Balance, Demand Response, Line Change) 
            AeqPrim = np.block([[np.identity(self.n), np.identity(self.n), Imat], #% Nodal Balance Equations
                            [-PTDFV, -PTDFV, np.identity(self.l)]])               #% Change in Flows Equations
            Aeq1 = np.kron(AeqPrim, np.eye(PointsInTime)) #% Expand temporal equations
            
            # inequalities
            # define A
            # A1 (Nodal Balance, Demand Response, Line Change) 
            APrim = np.block([[np.zeros((self.n,self.n)), -vSensiVal, np.zeros((self.n,self.l))], #% Nodal Balance Equations
                            [np.zeros((self.n,self.n)), vSensiVal, np.zeros((self.n,self.l))]])               #% Change in Flows Equations
            A1 = np.kron(APrim, np.eye(PointsInTime))
                  
        else:
            # define Aeq
            # Aeq1 (Nodal Balance, Line Change) 
            AeqPrim = np.block([[np.identity(self.n), Imat],   #% Nodal Balance Equations
                            [-PTDFV, np.identity(self.l)]])    #% Change in Flows Equations
            Aeq1 = np.kron(AeqPrim, np.eye(PointsInTime))      #% Expand temporal equations
            
            A1 = None
                   
        # addition to current Aeq1 if PV is available
        # if PV: 
        #     Aeq1 = np.block([[Aeq1], 
        #                     [self.prop_renewables, np.zeros((np.size(self.prop_renewables,0),(self.n+self.l)*PointsInTime))]])  #% Adding Source Renewable proportion

        # addition to current Aeq1 if storage is available
        if storage:
            # compute batteries affected matrices
            f, Aeq, beq, Ain, b, lb, ub = self.__calc_batteries(Aeq1, A1, D, DPTDF, DRmax, DR, PV)
        
        else:
            # compute normal case
            Aeq = Aeq1
            
            Ain = A1
                 
    		# define Beq
            # if PV:
            #     beq = np.concatenate((D, np.zeros((1, np.size(self.prop_renewables,0))), DPTDF),1).T
            # else:
            #     beq = np.concatenate((D, DPTDF),1).T
                
            beq = np.concatenate((D, DPTDF),1).T
                
            
            if DR:
            
                # define B (inequality)
                initVolts = np.reshape(self.initVolts.values.T, (1,np.size(self.initVolts.values)), order="F")
                vBase = np.kron(np.expand_dims(self.vBase['baseVoltage'].values,1), np.ones((self.PointsInTime,1)))
                b = np.concatenate((initVolts - (950 * vBase).T, (10500 * vBase).T - initVolts),1).T
                
                #  define upper and lower bounds
                ub = np.concatenate((self.Gmax, DRmax, self.Lmax),1)
                lb = np.concatenate((np.zeros((1, 2*self.n * PointsInTime)), -self.Lmax),1)
                ## define coeffs
                f = np.concatenate((self.cgn, self.cdr, self.clin),1)
            else:
                
                b = None
                #  define upper and lower bounds
                ub = np.concatenate((self.Gmax, self.Lmax),1)
                lb = np.concatenate((np.zeros((1, self.n * PointsInTime)), -self.Lmax),1)
                ## define coeffs
                f = np.concatenate((self.cgn, self.clin),1)
            
        x, m = self.__linprog(f, Aeq, beq, lb.tolist(), ub.tolist(), D, Ain, b, voltage)
            
        return x, m
    
    def __calc_batteries(self, Aeq1, A1, D, DPTDF, DRmax, DR, PV):
        """Compute the battery portion for Aeq"""
        
        # preprocessing
        PTDFV = self.PTDF.values
        PointsInTime = self.PointsInTime
        vSensiVal = self.vSensi.values

        # get batteries
        numBatteries, BatIncidence, BatSizes, BatChargingLimits, BatEfficiencies, BatInitEnergy, Pbatcost, ccharbat, ccapacity, BatPenalty = self.__getbatteries()

        # with storage
        # Aeq1:
        # columns: Pnt1,Pntf,Pjkt1,Pjktf 
        # rows:    (n+l)*PointsInTime + numBatteries*(PointsInTime+2)
        if DR:
            Aeq1 = np.concatenate( (Aeq1, np.zeros((numBatteries*(PointsInTime + 2),(2*self.n+self.l)*PointsInTime)) ), 0) #% Adding part of batteries eff. equations
        else:
            Aeq1 = np.concatenate( (Aeq1, np.zeros((numBatteries*(PointsInTime + 2),(self.n+self.l)*PointsInTime)) ), 0) #% Adding part of batteries eff. equations

        
        #% Aeq2 (Energy Storage impact on Nodal & Line Equations, Energy Balance, Energy Storage Initial and Final Conditions)
        #% Impact on Nodal Equations
        bat_incid = np.kron(BatIncidence, np.eye(PointsInTime))
        #Batt penalty
        Aeq2 = np.concatenate( (-bat_incid* np.kron(BatPenalty, np.ones((1, PointsInTime))), # * np.kron(BatPenalty, np.ones((1, PointsInTime)))
                                bat_incid * np.kron(1./BatPenalty, np.ones((1, PointsInTime))), 
                                np.zeros((self.n*PointsInTime,numBatteries*(PointsInTime+1)))), 1)
        
        # Impact on Line Equations
        row, col = np.where(BatIncidence==1)
        Aeq2lin = np.concatenate(( np.kron(np.concatenate( (BatPenalty*PTDFV[:,row],#
                                                            -1./BatPenalty*PTDFV[:,row]), 1), np.eye(PointsInTime)),
                                  np.zeros((self.l*PointsInTime,numBatteries*(PointsInTime + 1)) )), 1)
        
        # portion greater than 0.95
        A2_b1 = np.concatenate(( np.kron(np.concatenate( (BatPenalty*vSensiVal[:,row],#
                                                       - 1./BatPenalty*vSensiVal[:,row]), 1), np.eye(PointsInTime)),#
                             np.zeros((self.n*PointsInTime,numBatteries*(PointsInTime + 1)) )), 1)
        
        # portion less than 1.05
        A2_b2 = np.concatenate(( np.kron(np.concatenate( (- BatPenalty*vSensiVal[:,row],#
                                                       1./BatPenalty*vSensiVal[:,row]), 1), np.eye(PointsInTime)),#
                             np.zeros((self.n*PointsInTime,numBatteries*(PointsInTime + 1)) )), 1)
        A2 = np.concatenate((A2_b1, A2_b2),0)
        # Aeq2 at this point:
        # columns: PscBt1,PscBtf,...,PsdBt1,PsdBtf,...,EBt1,EBtf 
        # rows:    (n+l)*PointsInTime        
        Aeq2 = np.concatenate((Aeq2,Aeq2lin), 0)
        # if PV:
        #     Aeq2 = np.block([[Aeq2],
        #                      [np.zeros((np.size(self.prop_renewables,0), np.size(Aeq2,1)))]]);

        
        # Energy Balance Equations
        # Aeq2_auxP:
        # columns: PscBt1,PscBtf,...,PsdBt1,PsdBtf 
        # rows:    n*PointsInTime (only nodes with storage connected)   
        Aeq2_auxP = np.concatenate((-bat_incid * np.kron(BatEfficiencies, np.ones((1, PointsInTime))),
                                    bat_incid * np.kron(1./BatEfficiencies, np.ones((1, PointsInTime))) ), 1)
        Aeq2_auxP = np.concatenate(( Aeq2_auxP[Aeq2_auxP.any(1),:], np.zeros((numBatteries, np.size(Aeq2_auxP,1))) ), 0)
        
        # Aeq2_auxE:
        # columns: EBt1,EBtf 
        # rows:    (PointsInTime + 1)*numBatteries  
        Aeq2_auxE = np.eye( (PointsInTime + 1) * numBatteries)
    
        for i in range(numBatteries):
            init = i*PointsInTime
            endit = i*PointsInTime + PointsInTime
            Aeq2_auxE[init:endit,init:endit] = Aeq2_auxE[init:endit,init:endit] - np.eye(PointsInTime,k=-1)
        
        idx_E = [PointsInTime*numBatteries, PointsInTime*numBatteries]
        idx_E0 = [PointsInTime*numBatteries, (PointsInTime + 1)*numBatteries]
        Aeq2_auxE0 = np.zeros( (PointsInTime * numBatteries, numBatteries) )
        c = 0
        s = np.sum(Aeq2_auxE[:idx_E[0],:idx_E[1]], 1);
    
        for i in range(PointsInTime * numBatteries):
            if s[i]==1:
                Aeq2_auxE0[i,c] = -1
                c += 1
    
        Aeq2_auxE[:idx_E0[0], idx_E0[0]:idx_E0[1]+1] = Aeq2_auxE0
        
        # Adding Energy Balance and Initial Conditions
        # Aeq2 at this point:
        # columns: PscBt1,PscBtf,...,PsdBt1,PsdBtf,...,EBt1,EBtf 
        # rows:    (n+l)*PointsInTime + (1 + PointsInTime)*numBatteries
        Aeq2 = np.block([[Aeq2],
                        [Aeq2_auxP, Aeq2_auxE]])
        
        #Energy Storage final conditions
        # Aeq2 finally:
        # columns: PscBt1,PscBtf,...,PsdBt1,PsdBtf,...,EBt1,EBtf 
        # rows:    (n+l)*PointsInTime + (2 + PointsInTime)*numBatteries
        Aeq2 = np.block([[Aeq2],
                        [np.zeros((numBatteries, np.size(Aeq2_auxP,1))), np.flip(Aeq2_auxE0.T, 1), np.flip(np.eye(numBatteries), 0)]])
        
        # Build Aeq matrix
        # Aeq:
        # columns: Pnt1,Pntf-Pjkt1,Pjktf-PscBt1,PscBtf-PsdBt1,PsdBtf-EBt1,EBtf 
        # rows:    (n+l)*PointsInTime + (2 + PointsInTime)*numBatteries
        Aeq = np.concatenate((Aeq1,Aeq2), 1)
        Ain = np.concatenate((A1,A2), 1)
        
        # Equalities RHS
        # if PV:
        #     beq = np.concatenate((D.T, DPTDF.T, np.zeros((np.size(self.prop_renewables,0),1)), np.zeros((np.size(Aeq2_auxE0,0),1)), BatInitEnergy.T, np.zeros((numBatteries,1))),0) # Add Balance Energy, Init & Final Conditions
        # else:
        #     beq = np.concatenate((D.T, DPTDF.T, np.zeros((np.size(Aeq2_auxE0,0),1)), BatInitEnergy.T, np.zeros((numBatteries,1))),0) # Add Balance Energy, Init & Final Conditions
        
        beq = np.concatenate((D.T, DPTDF.T, np.zeros((np.size(Aeq2_auxE0,0),1)), BatInitEnergy.T, np.zeros((numBatteries,1))),0) # Add Balance Energy, Init & Final Conditions

        
        if DR:
            # define B (inequality)
            initVolts = np.reshape(self.initVolts.values.T, (1,np.size(self.initVolts.values)), order="F")
            vBase = np.kron(np.expand_dims(self.vBase['baseVoltage'].values,1), np.ones((self.PointsInTime,1)))
            b = np.concatenate((initVolts - (950 * vBase).T, (10500 * vBase).T - initVolts),1).T
            
            # Lower bounds
            lb = np.concatenate(( np.zeros((2*self.n*PointsInTime,1)),         # Generation limits
                  -self.Lmax.T,                                              # Line limits
                  np.kron(BatChargingLimits,np.zeros((1,PointsInTime))).T,             # Charging limits
                  np.kron(BatChargingLimits,np.zeros((1,PointsInTime))).T,             # Discharging limits
                  np.kron(BatSizes,np.zeros((1,PointsInTime))).T,               # Battery capacity limits
                  np.zeros((numBatteries,1)) ),0)                            # Initial capacity limits

            # Upper bounds
            ub = np.concatenate( (self.Gmax.T, DRmax.T, self.Lmax.T,                         # Generation & Line limits
                  np.kron(BatChargingLimits,np.ones((1,PointsInTime))).T,    # Charging limits
                  np.kron(BatChargingLimits,np.ones((1,PointsInTime))).T,    # Discharging limits
                  np.kron(BatSizes,np.ones((1,PointsInTime))).T,             # Battery capacity limits
                  BatSizes.T), 0)                                            # Initial capacity limits

            #% Objective function
            f = np.concatenate((self.cgn, self.cdr, self.clin, ccharbat, ccapacity),1) # % x = Pg Pdr Plin Psc Psd E E0
        else:
            
            b = None
            
            # Lower bounds
            lb = np.concatenate(( np.zeros((self.n*PointsInTime,1)),         # Generation limits
                  -self.Lmax.T,                                              # Line limits
                  np.kron(BatChargingLimits,np.zeros((1,PointsInTime))).T,             # Charging limits
                  np.kron(BatChargingLimits,np.zeros((1,PointsInTime))).T,             # Discharging limits
                  np.kron(BatSizes,np.zeros((1,PointsInTime))).T,               # Battery capacity limits
                  np.zeros((numBatteries,1)) ),0)                            # Initial capacity limits

            # Upper bounds
            ub = np.concatenate( (self.Gmax.T, self.Lmax.T,                         # Generation & Line limits
                  np.kron(BatChargingLimits,np.ones((1,PointsInTime))).T,    # Charging limits
                  np.kron(BatChargingLimits,np.ones((1,PointsInTime))).T,    # Discharging limits
                  np.kron(BatSizes,np.ones((1,PointsInTime))).T,             # Battery capacity limits
                  BatSizes.T), 0)                                            # Initial capacity limits

            #% Objective function
            f = np.concatenate((self.cgn, self.clin, ccharbat, ccapacity),1) # % x = Pg Pdr Plin Psc Psd E E0

        return f, Aeq, beq, Ain, b, lb, ub
    
    def __getbatteries(self):
        """Method to extract all batteries attibutes"""

        numBatteries       = self.batt['numBatteries']
        BatIncidence = self.batt['BatIncidence']
        BatSizes     = self.batt['BatSizes']
        BatChargingLimits   = self.batt['BatChargingLimits']  
        BatEfficiencies      = self.batt['BatEfficiencies']     
        BatInitEnergy   = self.batt['BatInitEnergy']  
        #Jorge
        BatPenalty   = self.batt['BatPenalty']  
        Pbatcost     = self.batt['Pbatcost']    
        ccharbat     = self.batt['ccharbat']    
        ccapacity    = self.batt['ccapacity']   
        return numBatteries, BatIncidence, BatSizes, BatChargingLimits, BatEfficiencies, BatInitEnergy, Pbatcost, ccharbat, ccapacity, BatPenalty

    def __incidenceMat(self):
        """This method computes a matrix that defines which lines are connected to each of the nodes"""
        PTDF = self.PTDF
        l = self.l 
        n = self.n 
        
        Node2Line = np.zeros((n,l))

        for i in range(n): # node loop
            for j in range(l): # lines loop
                if PTDF.columns[i] in PTDF.index[j].split("-")[0]:
                    Node2Line[i,j] = 1
                elif PTDF.columns[i] in PTDF.index[j].split("-")[1]:
                    Node2Line[i,j] = -1
        return -Node2Line #- due to power criteria
                
    def __linprog(self, f, Aeq, beq, lb, ub, D, A, b, voltage=False):
        """Model related procesing"""
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            with gp.Model(env=env) as m:

                # create a new model
                m = gp.Model("LP1")
                
                # create variables
                x = m.addMVar(shape=np.size(Aeq,1), lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name="x")
                
                # multipy by the coefficients
                m.setObjective(f @ x, GRB.MINIMIZE)
                
                # add equality constraints
                m.addConstr(Aeq @ x == np.squeeze(beq), name="eq")

                # add inequality constraints
                if voltage:
                    m.addConstr(A @ x <= np.squeeze(b), name="ineq")

                     
                # Optimize model
                m.optimize()
        
        return x,m

# +=================================================================================================
def create_battery(PTDF, PointsInTime):
    """Method to define battery parameters"""

    batt = dict()
    numBatteries= 3
    batt['numBatteries'] = numBatteries
    BatIncidence = np.zeros((len(PTDF.columns),numBatteries))
    BatIncidence[PTDF.columns == '675.1', 0] = 1
    BatIncidence[PTDF.columns == '675.2', 1] = 1
    BatIncidence[PTDF.columns == '675.3', 2] = 1
    batt['BatIncidence'] = BatIncidence
    BatSizes = 300 * np.ones((1,numBatteries))
    batt['BatSizes'] = BatSizes
    BatChargingLimits = 100*np.ones((1,numBatteries))
    batt['BatChargingLimits'] = BatChargingLimits
    BatEfficiencies = 0.97*np.ones((1,numBatteries))
    batt['BatEfficiencies'] = BatEfficiencies
    BatInitEnergy = BatSizes * 0.4 
    batt['BatInitEnergy'] = BatInitEnergy
    Pbatcost = 0.01
    batt['Pbatcost'] = Pbatcost
    ccharbat = Pbatcost * np.ones((1,2*numBatteries*PointsInTime))
    batt['ccharbat'] = ccharbat
    ccapacity = Pbatcost * np.ones((1, numBatteries*(PointsInTime + 1)))
    batt['ccapacity'] = ccapacity
    batt['BatPenalty'] = np.ones((1,numBatteries)) 

    return batt

def load_PTDF(script_path):

    PTDF_file = pathlib.Path(script_path).joinpath("..", "inputs", "PTDF_jk_Nov-11-2021_1627.pkl")
    PTDF = pd.read_pickle(PTDF_file)
    
    # adjust lossless PTDF
    PTDF = PTDF / 10 # divide by perturbation injection value
    
    return PTDF

def load_GenerationMix(script_path):

    GenMix_file = pathlib.Path(script_path).joinpath("..", "inputs", "GeorgiaGenerationMix2.xlsx")
    t = pd.read_excel(GenMix_file)
    GenAlpha = t["Alpha"]
    GenBeta = np.expand_dims(t["Beta"], axis=1)
    PointsInTime = len(GenAlpha) 
    return GenAlpha, GenBeta, PointsInTime

def load_demandProfile(script_path, PointsInTime, GenBeta, Pf):

    Demand_file = pathlib.Path(script_path).joinpath("..", "inputs", "DemandProfile_Nov-11-2021_1627.pkl")
    DemandProfilei = pd.read_pickle(Demand_file)[0]
    # test Jorge
    # DemandProfilei = (1/Pf) * DemandProfilei
    ##
    DemandProfile = np.expand_dims(DemandProfilei, axis=1)
    # Expand feeder demand for time series analysis
    DemandProfile =  DemandProfile @ GenBeta.T
    # for debug
    # DemandProfile =  np.kron(DemandProfile, np.ones((1,PointsInTime)))
    DemandProfile = DemandProfile[:,:PointsInTime]
    return DemandProfile, DemandProfilei

def load_generationCosts(script_path, n, PointsInTime):
    
    GenPrice_file = pathlib.Path(script_path).joinpath("..", "inputs", "HourlyMarginalPrice.xlsx")
    tcost = pd.read_excel(GenPrice_file)
    Gcost = 10000*np.ones((n,PointsInTime))
    cost_wednesday = tcost.values[156,1:-1]
    Gcost[0,:] = cost_wednesday[:PointsInTime]
    Gcost[1,:] = cost_wednesday[:PointsInTime]
    Gcost[2,:] = cost_wednesday[:PointsInTime]
    return Gcost, cost_wednesday

def PV_system(Gmax, DemandProfile, DemandProfilei): 
    
    # Compute the energy consumption of the demand in KWh 
    EnergyConsumptionDay = np.sum(DemandProfile,1) / 24
    
    # Estimate the energy from PV to be 60% from the energy consumption of the house
    EnergyFromPV = 0.6 * EnergyConsumptionDay 
    
    # Estimate the PV size as Energy / 4
    PVsize = np.floor(EnergyFromPV / 4)
    PVsize = np.expand_dims(PVsize,1)
       
    # Estimate a PV Profile   
    np.random.seed(2021) # Set random seed so results are repeatable
    a = np.sin(np.linspace(-3,20,24)*np.pi/15) - 0.5 + np.random.rand(24)*0.2
    a[a<0] = 0
    a = a/max(a)
    PVProfile = np.expand_dims(a,1)
    
    # Compute the Ppv at each node
    Ppv = PVsize.T * np.kron(PVProfile, np.ones((1,len(PVsize))))
    Ppv = Ppv.T
    
    # Compute the new demand including Ppv
    DemandProfilePV = DemandProfile - Ppv

    return DemandProfilePV

def EV_profile(script_path, DemandProfile, DemandProfilei): # old Kartik version: PV_system(Gmax, PTDF, GenAlpha, Gcost, cost_wednesday, PointsInTime)
    
    EV_file = pathlib.Path(script_path).joinpath("..", "inputs", "DumbChargingProfiles.xlsx")
    tEV = pd.read_excel(EV_file, 'Profiles')
    EV_profile = tEV.values[1:,21:]

    # accomodate the interval from 8 to 8 -> 1 to 24
    EV_profile = np.concatenate((EV_profile[33:,:], EV_profile[0:33,:]), 0)
    
    # The initial file is in intervals of 30 min. To match price,
    # sum each consecutive row and turn it into 1h intervals.
    EV_profile = (EV_profile[0::2,:] + EV_profile[1::2,:])
    
    # nodes to connect the EV
    DemandProfilei['645.3'] = 1
    index = np.where(DemandProfilei)
    
    # create an instance of a demand profile
    PDev = np.zeros((np.shape(DemandProfile)))
    
    # define where the EV profiles will be connected to:
    PDev[index[0],:] = EV_profile.T
    
    # Compute the new demand including PDev (EV demand)
    DemandProfileEV = DemandProfile + PDev 

    return DemandProfileEV

def Loss_Penalties(batt, PTDF): 
        
    # compute dPgref
    dPgref = np.min(PTDF[:3])
    
    # dPl/dPgi = 1 - (- dPgref/dPgi) -> eq. L9_25
    
    # ITLi = dPL/dPGi 
    ITL = 1 + dPgref # Considering a PTDF with transfer from bus i to the slack. If PTDF is calculated in the converse, then it will be 1 - dPgref
    
    Pf = 1 / (1- ITL)

    # batt incidence
    BatIncidence = batt['BatIncidence'] 
    
    # nodes with batt 
    nodes = np.where(np.any(BatIncidence,1))
    
    # assign penalty factors 
    batt['BatPenalty'] = min([Pf.values[n] for n in nodes[0]])*np.ones((1,3)) 

    return batt, Pf

def load_voltageSensitivity(script_path):
    
    # voltage sensitivity
    voltageSensi_file = pathlib.Path(script_path).joinpath("..", "inputs", "VoltageSensitivity_Dec-24-2021_1143.pkl")
    voltageSensi = pd.read_pickle(voltageSensi_file)

    # adjust voltage sensi matrix
    voltageSensi = voltageSensi / 10 # divide by perturbation injection value
    
    # base voltage
    voltageBase_file = pathlib.Path(script_path).joinpath("..", "inputs", "BaseVoltage__Dec-24-2021_1143.pkl")
    voltageBase = pd.read_pickle(voltageBase_file)
    
    # initial voltages
    voltage_file = pathlib.Path(script_path).joinpath("..", "inputs", "initVoltage_Dec-27-2021_1916.pkl")
    voltage = pd.read_pickle(voltage_file)
    
    return voltageSensi, voltageBase, voltage


def load_lineLimits(script_path, PointsInTime):

    # debug:
    # Lmaxi = 2000 * np.ones((len(PTDF),1))
    Lmax_file = pathlib.Path(script_path).joinpath("..", "inputs", "Pjk_ratings_Nov-20-2021_1154.pkl") #high limits: Pjk_ratings_Nov-11-2021_1745 ## low limits:Pjk_ratings_Nov-20-2021_1154
    Lmaxi = pd.read_pickle(Lmax_file)[0]
    Lmax = np.kron(Lmaxi, np.ones((1,PointsInTime)))
    Lmax = np.reshape(Lmax.T, (1,np.size(Lmax)), order="F")
    # Line Info
    Linfo_file = pathlib.Path(script_path).joinpath("..", "inputs", "LineInfo__Nov-20-2021_1154.pkl")
    Linfo = pd.read_pickle(Linfo_file)
    return Lmax, Linfo
           
def main():

    # define the type of analysis;
    
    #include storage
    storage = True
    #include Penalty factors
    PF = True
    #include Demand Response
    DR = True
    #include PV systems
    PV = False
    #include EVs
    EV = False
    #include voltage constraints
    voltage = True
    
    script_path = os.path.dirname(os.path.abspath(__file__))
    
    # load PTDF results
    PTDF = load_PTDF(script_path)
    n = len(PTDF.columns)
    l = len(PTDF)

    # generation mix
    GenAlpha, GenBeta, PointsInTime = load_GenerationMix(script_path)
    # PointsInTime = 4
    
    # Storage
    batt = create_battery(PTDF, PointsInTime)
    
    #Penalty factors
    if PF:
        batt, Pf = Loss_Penalties(batt, PTDF)
        
    # round the PTDF to make the optimization work
    PTDF = PTDF.round()
    
    # Load demand profile
    DemandProfile, DemandProfilei = load_demandProfile(script_path, PointsInTime, GenBeta, Pf)

    # Load generation costs
    Gcost, cost_wednesday = load_generationCosts(script_path, n, PointsInTime)
    
    # Overall Generation costs:
    cgn = np.reshape(Gcost.T, (1,Gcost.size), order="F")
    
    
    # Line costs
    Pijcost = 0.01*np.ones((l,PointsInTime))
    clin = np.reshape(Pijcost.T, (1,Pijcost.size), order="F")
    
    # Define generation limits
    Gmax = np.zeros((n,1))
    Gmax[0,0] = 2000 # asume the slack conventional phase is here
    Gmax[1,0] = 2000 # asume the slack conventional phase is here
    Gmax[2,0] = 2000 # asume the slack conventional phase is here
    
    # Overall Generation limits:
    max_profile = np.kron(Gmax, np.ones((1,PointsInTime)))
    Gmax = np.reshape(max_profile.T, (1,np.size(max_profile)), order='F')

    # Line limits and info          
    Lmax, Linfo = load_lineLimits(script_path, PointsInTime) 
    
    #Demand Response (cost of shedding load)
    DRcost = 50
    cdr = DRcost*np.ones((1,n*PointsInTime)) 
    
    #PV system
    if PV:
        DemandProfilePV = PV_system(Gmax, DemandProfile, DemandProfilei)
        DemandProfile = DemandProfilePV
        
    #EV profiles
    if EV:
        DemandProfileEV = EV_profile(script_path, DemandProfile, DemandProfilei)
        DemandProfile = DemandProfileEV
        
    # load voltage base at each node
    vSensi, vBase, initVolts = load_voltageSensitivity(script_path)
    
    # create an instance of the dispatch class
    Dispatch_obj = Dispatch(PTDF, PointsInTime, batt, Lmax, Gmax, cgn, clin, cdr, vBase, vSensi, initVolts)

    # call the OPF method
    x, m = Dispatch_obj.PTDF_OPF(DemandProfile, voltage=voltage, storage=storage, DR = DR, PV=PV)

    print(x.X)
    print('Obj: %g' % np.round(m.objVal,5))
    print(np.round(m.objVal,5))
    
    #plot results
    Plot_obj = PlottingDispatch(x, DR, storage, PTDF, PointsInTime, batt, script_path)

    # Plot extract results
    Pg, Pdr, Pij, Pchar, Pdis, E = Plot_obj.__extract_results(x, DR, storage, batt)
    
    vSensi2 = np.kron(vSensi, np.eye(PointsInTime))
    D = np.reshape(DemandProfile.T,(np.size(DemandProfile),1), order="F")
    
    # dV_G = vSensi2 @ x.X[:n*PointsInTime]
    # dV_G = -vSensi2 @ D
    # dV_Pchar = -vSensi2 @ x.X[(n+l)*PointsInTime:(n+l)*PointsInTime + numBatteries*PointsInTime]
    # dV_Pdis = vSensi2 @ x.X[(n+l+numBatteries)*PointsInTime:(n+l+2*numBatteries)*PointsInTime]
    
    
    # Lmaxi = 2000 * np.ones((len(PTDF),1))
    # Lmax_file = pathlib.Path(script_path).joinpath("..", "inputs", "Pjk_ratings_Nov-20-2021_1154.pkl")
    # Lmaxi = pd.read_pickle(Lmax_file)[0]
    # Lmax = np.kron(Lmaxi, np.ones((1,PointsInTime)))
    # Lmax = np.reshape(Lmax.T, (1,np.size(Lmax)), order="F")
    
    #plot results
    # Plot_obj = PlottingDispatch(x, DR, storage, PTDF, PointsInTime, batt, script_path)
    
    # Plot demand response
    # Plot_obj.Plot_DemandResponse(niter="Low_Pjk_limits")
    
    # Plot Line Limits
    # Plot_obj.Plot_Pjk(Linfo, Lmax, niter="Low_Pjk_limits")
    
    # Plot Energy storage
    # Plot_obj.Plot_storage(batt, Gcost[0,:], niter="Low_Pjk_limits") 
    
    # PTDF
    # Plot_obj.Plot_PTDF()
    
    # Plot Demand
    # Plot_obj.Plot_Demand(DemandProfilei, DemandProfile)
    
    # Plot Dispatch
    # Plot_obj.Plot_Dispatch(niter="Low_Pjk_limits")
    
# +=================================================================================================
if __name__ == "__main__":
    main()


