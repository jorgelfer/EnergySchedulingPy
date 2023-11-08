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
#required for plotting
import matplotlib.pyplot as plt
# from matplotlib.patches import StepPatch
import seaborn as sns
from tqdm.auto import tqdm
import time

class Dispatch:
    
    def __init__(self, PTDF, PointsInTime, batt, Lmax, Gmax, cgn, clin):
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


    def PTDF_OPF(self, DemandProfile, storage=False, PV=False):
        #--------------------------------------------------------
        # Build matrices
        #--------------------------------------------------------

        # preprocessing
        PTDFV = self.PTDF.values
        PointsInTime = self.PointsInTime
        
        # compute the incidence matrix
        Imat = self.__incidenceMat()

        # Compute the demand portion of the PTDF-OPF definition:
        # i.e. Pjk*(-PTDF) = -PTDF * Pd (the right hand side is the demand portion)
        DPTDF = - PTDFV @ DemandProfile
        DPTDF = np.reshape(DPTDF.T, (1,DPTDF.size), order="F")
        # Demand vector for each hour
        D = np.reshape(DemandProfile.T,(1,np.size(DemandProfile)), order="F")

        # define Aeq
        # Aeq1 (Nodal Balance, Line Change) DemandResponse missing
        AeqPrim = np.block([[np.identity(self.n), Imat],                 #% Nodal Balance Equations
                        [-PTDFV, np.identity(self.l)]])                  #% Change in Flows Equations
        Aeq1 = np.kron(AeqPrim, np.eye(PointsInTime))               #% Expand temporal equations
        
        if storage:
            # compute batteries affected matrices
            f, Aeq, beq, lb, ub = self.__calc_batteries(Aeq1, D, DPTDF)
        
        elif PV:
            # compute batteries affected matrices
            f, Aeq, beq, lb, ub = self.__calc_batteries(Aeq1, D, DPTDF)
            
        else:
            # compute normal case
            Aeq = Aeq1
                 
    		# define Beq
            beq = np.concatenate((D, DPTDF),1).T
            
            # define the upper and lower bounds
            ub = np.concatenate((self.Gmax, self.Lmax),1)
            print(ub)
            lb = np.concatenate((np.zeros((1, self.n * PointsInTime)), -self.Lmax),1)
            print(lb)
            
            ## define coeffs
            f = np.concatenate((self.cgn, self.clin),1)
        
        x, m = self.__linprog(f, Aeq, beq, lb[0].tolist(), ub[0].tolist())
            
        return x, m
    
    def __calc_batteries(self, Aeq1, D, DPTDF):
        """Compute the battery portion for Aeq"""
        
        # preprocessing
        PTDFV = self.PTDF.values
        PointsInTime = self.PointsInTime

        # get batteries
        numBatteries, BatIncidence, BatSizes, BatChargingLimits, BatEfficiencies, BatInitEnergy, Pbatcost, ccharbat, ccapacity, BatPenalty = self.__getbatteries()

        # with storage
        # Aeq1:
        # columns: Pnt1,Pntf,Pjkt1,Pjktf 
        # rows:    (n+l)*PointsInTime + numBatteries*(PointsInTime+2)
        Aeq1 = np.concatenate( (Aeq1, np.zeros((numBatteries*(PointsInTime + 2),(self.n+self.l)*PointsInTime)) ), 0) #% Adding part of batteries eff. equations
        
        #% Aeq2 (Energy Storage impact on Nodal & Line Equations, Energy Balance, Energy Storage Initial and Final Conditions)
        #% Impact on Nodal Equations
        bat_incid = np.kron(BatIncidence, np.eye(PointsInTime))
        # pf = np.array([[0.954, 0.996, 0.936]])
        # pf = np.ones((1,numBatteries))
        #Batt penalty
        Aeq2 = np.concatenate( (-bat_incid * np.kron(BatPenalty, np.ones((1, PointsInTime))), 
                                bat_incid * np.kron(1./BatPenalty, np.ones((1, PointsInTime))), 
                                np.zeros((self.n*PointsInTime,numBatteries*(PointsInTime+1)))), 1)
        
        # Impact on Line Equations
        row, col = np.where(BatIncidence==1)
        Aeq2lin = np.concatenate(( np.kron(np.concatenate( (BatPenalty*PTDFV[:,row],
                                                            -1./BatPenalty*PTDFV[:,row]), 1), np.eye(PointsInTime)),
                                  np.zeros((self.l*PointsInTime,numBatteries*(PointsInTime + 1)) )), 1)
        # Aeq2 at this point:
        # columns: PscBt1,PscBtf,...,PsdBt1,PsdBtf,...,EBt1,EBtf 
        # rows:    (n+l)*PointsInTime        
        Aeq2 = np.concatenate((Aeq2,Aeq2lin), 0)
        
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
        
        # Equalities RHS
        beq = np.concatenate((D.T, DPTDF.T, np.zeros((np.size(Aeq2_auxE0,0),1)), BatInitEnergy.T, np.zeros((numBatteries,1))),0) # Add Balance Energy, Init & Final Conditions

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

        return f, Aeq, beq, lb, ub
    
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
                
    def __linprog(self, f, Aeq, beq, lb, ub):
        """Model related procesing"""
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            with gp.Model(env=env) as m:

                # create a new model
                m = gp.Model("LP1")

                print(ub)
                print(lb)

                # create variables
                x = m.addMVar(shape=np.size(Aeq,1), ub=ub, lb=lb, vtype=GRB.CONTINUOUS, name="x")
                
                # multipy by the coefficients
                m.setObjective(f @ x, GRB.MINIMIZE)
                
                # add equality constraints
                m.addConstr(Aeq @ x == np.squeeze(beq), name="eq")
        
                # Optimize model
                m.optimize()
        
        return x,m

    def Plot_results(self, x, m, cgn, lmax, lInfo, niter):
        """Method employed to plot results"""
    
        # preprocessing
        script_path = os.path.dirname(os.path.abspath(__file__))
        n = self.n         #number of nodes
        m = self.l         #number of lines
        PTDF = self.PTDF
        PointsInTime = self.PointsInTime
        numBatteries = self.batt['numBatteries']
        # get the nodes with batteries
        row, _ = np.where(self.batt['BatIncidence']==1)
        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y_%H%M', t)           
        # directory = f"Results_{timestamp}"
        # output_dir = pathlib.Path(script_path).joinpath("..","outputs", directory)
        # os.mkdir(output_dir)
        
        # Extract solution
        Pg    = np.reshape(x.X[:n*PointsInTime], (PointsInTime,n), order='F').T
        Pij   = np.reshape(x.X[n*PointsInTime:(n+m)*PointsInTime], (PointsInTime, m), order='F').T
        Pchar = np.reshape(x.X[(n+m)*PointsInTime:(n+m)*PointsInTime + numBatteries*PointsInTime] , (PointsInTime,numBatteries), order='F').T
        Pdis  = np.reshape(x.X[(n+m+numBatteries)*PointsInTime:(n + m + 2*numBatteries)*PointsInTime], (PointsInTime,numBatteries), order='F').T
        E     = np.reshape(x.X[(n+m + 2*numBatteries)*PointsInTime:-numBatteries], (PointsInTime,numBatteries), order='F').T
        
        # plotting the results
        ###########
        ## PTDF  ##
        ###########
        # plt.clf()
        # fig, ax = plt.subplots(figsize=(15, 9))
        # # plt.figure(figsize=(15,8))                  
        # ax = sns.heatmap(PTDF,annot=False)
        # output_img = pathlib.Path(script_path).joinpath("..","outputs", f"PTDF_{timestamp}.png")
        # plt.savefig(output_img)
        
        #######################
        ## Power in lines    ##
        #######################
        
        #nodes in lines
        # nil = [3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 3, 2, 2, 3]
        # namel = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 'L11', 'L12', 'L13', 'L14']
        nil = lInfo['NumNodes'].values
        namel = lInfo['Line_Name'].values
        lmax   = np.reshape(lmax, (PointsInTime, m), order='F').T
        cont = 0
        for ni, na in zip(nil, namel):
            
            plt.clf()
            fig, ax = plt.subplots(figsize=(15, 8))                
            leg = [node for node in PTDF.index[cont:cont + ni]]
            xrange = np.arange(1,PointsInTime+1,1)
            ax.plot(xrange,Pij[cont:cont + ni,:].T)
            plt.legend(leg)
            ax.set_title(f'Line power flow_{na}', fontsize=25)
            plt.ylabel('Power (kW)', fontsize=16)
            plt.xlabel('Time (hrs)', fontsize=16)
            ax.plot(xrange,lmax[cont:cont + ni,:].T, 'r')
            
            output_img = pathlib.Path(script_path).joinpath("..","outputs", f"Power_{na}_{timestamp}.png")
            plt.savefig(output_img)
            cont += ni
    
        #######################
        ## Power Dispatch  ##
        #######################
    
        plt.clf()
        fig, ax = plt.subplots(figsize=(15, 8))
        # plt.figure(figsize=(15,8))                  
        leg = [node for node in PTDF.columns[:3]]
        xrange = np.arange(1,PointsInTime+1,1)
        ax.plot(xrange,Pg[:3,:].T)
        ax.set_title(f'Power from Substation_{niter}', fontsize=25)
        plt.ylabel('Power (kW)', fontsize=16)
        plt.xlabel('Time (hrs)', fontsize=16)
        plt.legend(leg) 
        output_img = pathlib.Path(script_path).joinpath("..","outputs", f"Power_{niter}_{timestamp}.png")
        plt.savefig(output_img)
    
        ######################
        ## Storage storage  ##
        ######################
    
        plt.clf()
        fig, ax1 = plt.subplots(figsize=(15, 8))
        leg = [PTDF.columns[node] for node in row]
        ax1.step(xrange,E.T)
        ax1.set_title(f'Prices vs static battery charging_{niter}', fontsize=25)
        ax1.set_ylabel('Energy Storage (kWh)', fontsize=16)
        ax1.set_xlabel('Time (hrs)', fontsize=16)
        plt.legend(leg) 
    
        ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
        color = 'tab:red'
        ax2.set_ylabel('HourlyMarginalPrice ($/kWh)', color=color, fontsize=16)  # we already handled the x-label with ax1
        ax2.step(xrange,cgn.T, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout() 
    
        output_img = pathlib.Path(script_path).joinpath("..","outputs", f"EnergyStorage_{niter}_{timestamp}.png")
        plt.savefig(output_img)

# +=================================================================================================
def create_battery(PTDF, PointsInTime):
    """Method to define battery parameters"""

    batt = dict()
    numBatteries= 3
    batt['numBatteries'] = numBatteries
    BatIncidence = np.zeros((len(PTDF.columns),numBatteries))
    BatIncidence[PTDF.columns == '634.1', 0] = 1
    BatIncidence[PTDF.columns == '634.2', 1] = 1
    BatIncidence[PTDF.columns == '634.3', 2] = 1
    batt['BatIncidence'] = BatIncidence
    BatSizes = 400 * np.ones((1,numBatteries))
    batt['BatSizes'] = BatSizes
    BatChargingLimits = 100*np.ones((1,numBatteries))
    batt['BatChargingLimits'] = BatChargingLimits
    BatEfficiencies = 0.95*np.ones((1,numBatteries))
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

def main():
    
    script_path = os.path.dirname(os.path.abspath(__file__))
    
    # load PTDF results
    PTDF_file = pathlib.Path(script_path).joinpath("..", "inputs", "PTDF_jk_Nov-11-2021_1627.pkl")
    PTDF = pd.read_pickle(PTDF_file)
    
    # adjust lossless PTDF
    PTDF = PTDF / 10 # divide by perturbation injection value
    
    dPgref = PTDF[:3]
    Pf = -1 / dPgref
    # Pf.loc[(Pf.values >= 1)] = 1
    
    # PTDF[]= PTDF.round()
    PTDF.loc[:, 'sourcebus.1'] = 0
    PTDF.loc[:, 'sourcebus.2'] = 0
    PTDF.loc[:, 'sourcebus.3'] = 0
    PTDF = PTDF.round()
    
    # generation mix
    GenMix_file = pathlib.Path(script_path).joinpath("..", "inputs", "GeorgiaGenerationMix2.xlsx")
    t = pd.read_excel(GenMix_file)
    GenAlpha = t["Alpha"]
    GenBeta = np.expand_dims(t["Beta"], axis=1)
    PointsInTime = len(GenAlpha) 
    # PointsInTime = 4
    
    # Load demand profile
    Demand_file = pathlib.Path(script_path).joinpath("..", "inputs", "DemandProfile_Nov-11-2021_1627.pkl")
    DemandProfile = pd.read_pickle(Demand_file)[0]
    DemandProfile = np.expand_dims(DemandProfile, axis=1)
    # Expand feeder demand for time series analysis
    # DemandProfile =  DemandProfile @ GenBeta.T
    DemandProfile =  np.kron(DemandProfile, np.ones((1,PointsInTime)))
    DemandProfile = DemandProfile[:,:PointsInTime]
    # 
    # Load generation costs
    GenPrice_file = pathlib.Path(script_path).joinpath("..", "inputs", "HourlyMarginalPrice.xlsx")
    tcost = pd.read_excel(GenPrice_file)
    Gcost = 10000*np.ones((len(DemandProfile),PointsInTime))
    cost_wednesday = tcost.values[156,1:-1]
    Gcost[0,:] = cost_wednesday[:PointsInTime]
    Gcost[1,:] = cost_wednesday[:PointsInTime]
    Gcost[2,:] = cost_wednesday[:PointsInTime]
    cgn = np.reshape(Gcost.T, (1,Gcost.size), order="F")
    
    # Storage
    batt = create_battery(PTDF, PointsInTime)
    
    # Line costs
    Pijcost = np.zeros((len(PTDF),PointsInTime))
    clin = np.reshape(Pijcost.T, (1,Pijcost.size), order="F")
    
    # Define generation limits
    Gmax = np.zeros((len(PTDF.columns),1))
    Gmax[0,0] = 2000 # asume the slack conventional phase is here
    Gmax[1,0] = 2000 # asume the slack conventional phase is here
    Gmax[2,0] = 2000 # asume the slack conventional phase is here
    max_profile = np.kron(Gmax, np.ones((1,PointsInTime)))
    Gmax = np.reshape(max_profile.T, (1,np.size(max_profile)), order='F')
    
    # Define line limits
    # Lmaxi = 2000 * np.ones((len(PTDF),1))
    Lmax_file = pathlib.Path(script_path).joinpath("..", "inputs", "Pjk_ratings_Nov-11-2021_1745.pkl")
    Lmaxi = pd.read_pickle(Lmax_file)[0]
    Lmax = np.kron(Lmaxi, np.ones((1,PointsInTime)))
    Lmax = np.reshape(Lmax.T, (1,np.size(Lmax)), order="F")
    
    # Define line Info
    Linfo_file = pathlib.Path(script_path).joinpath("..", "inputs", "LineInfo__Nov-20-2021_1154.pkl")
    Linfo = pd.read_pickle(Linfo_file)

    # Storage
    batt = create_battery(PTDF, PointsInTime)

    # create an instance of the dispatch class
    Obj = Dispatch(PTDF, PointsInTime, batt, Lmax, Gmax, cgn, clin)

    # call the OPF method
    x, m = Obj.PTDF_OPF(DemandProfile, storage=False)

    print(x.X)
    print('Obj: %g' % m.objVal)
    
    # # Lmaxi = 2000 * np.ones((len(PTDF),1))
    # Lmax_file = pathlib.Path(script_path).joinpath("..", "outputs", "Pjk_ratings_Nov-20-2021_1154.pkl")
    # Lmaxi = pd.read_pickle(Lmax_file)[0]
    # Lmax = np.kron(Lmaxi, np.ones((1,PointsInTime)))
    # Lmax = np.reshape(Lmax.T, (1,np.size(Lmax)), order="F")
    
    # # plot results
    # Obj.Plot_results(x, m, Gcost[0,:], Lmax, Linfo,  1)
    
# +=================================================================================================
if __name__ == "__main__":
    main()


