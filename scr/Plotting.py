import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
# from matplotlib.patches import StepPatch
import seaborn as sns
import os
import time
import pathlib
import numpy as np
import pandas as pd
h = 6 
w = 4 
ext = '.pdf'

class plottingDispatch:

    def __init__(self, output_dir,  niter, PointsInTime, script_path, vmin=0.95, vmax=1.05, PTDF=None, Ain=None, title=False, dispatchType='LP'):
        
        if Ain is not None:
            self.Ain = Ain
            
        # preprocessing
        if PTDF is not None:
            self.PTDF = PTDF
            self.n = len(self.PTDF.columns) #number of nodes
            if dispatchType == 'LP':
                self.l = len(self.PTDF)         #number of lines
                self.jk = True
            else:
                self.jk = False
                self.l=0

        self.title = title

        self.niter = niter 
        
        self.PointsInTime = PointsInTime
        self.vmin = vmin
        self.vmax = vmax

        # time stamp 
        t = time.localtime()
        self.timestamp = time.strftime('%b-%d-%Y_%H%M%S', t)           
        
        # create directory to store results
        self.output_dir = output_dir
        
    def plot_voltage(self, vBase, initVolts, indexDemand, comp=False, x=None):
        
        if len(vBase.shape) == 1:
            vBase = np.kron(np.expand_dims(vBase,1), np.ones((1, self.PointsInTime)))
        
        if comp:
            dvoltages = self.Ain @ x.X
            
            dvolts = dvoltages[:np.size(dvoltages) // 2]
            
            dv    = np.reshape(dvolts[:self.n*self.PointsInTime], (self.PointsInTime,self.n), order='F').T
            
            v = initVolts - dv
            
        else:
            v = initVolts
        
        # compute the per unit value of the voltage
        vpu = v / (1000*vBase)
        
        # create a dataframe for limits
        limits = np.concatenate([self.vmin*np.ones((1, self.PointsInTime)), self.vmax*np.ones((1, self.PointsInTime))], axis = 0)
        dfLimits = pd.DataFrame(limits, index = ['lower limit', 'upper limit'], columns = vpu.columns)
        
        plt.clf()
        fig, ax = plt.subplots(figsize=(h,w)) 
        
        # concatenate dataframes
        concatenated = pd.concat([vpu, dfLimits])
        concatenated.T.plot(legend=False)

        if self.title:
            ax.set_title('Volts')
            plt.ylabel('Volts[pu]')
            plt.xlabel('Time (hrs)')
                
        fig.tight_layout()

        # create voltage directory to store results
        output_dirV = pathlib.Path(self.output_dir).joinpath("voltage")
        if not os.path.isdir(output_dirV):
            os.mkdir(output_dirV)

        output_img = pathlib.Path(self.output_dir).joinpath(f"voltage_{self.niter}_{self.timestamp}" + ext)
        plt.savefig(output_img)
        plt.close('all')

        # save as pickle as well
        output_pkl = pathlib.Path(output_dirV).joinpath(f"voltage_{self.niter}_{self.timestamp}.pkl")
        concatenated.to_pickle(output_pkl)
        
    def plot_PTDF(self):

        ###########
        ## PTDF  ##
        ###########
        plt.clf()
        fig, ax = plt.subplots(figsize=(h,w))
        sns.heatmap(self.PTDF,annot=False)
        fig.tight_layout()
        output_img = pathlib.Path(self.output_dir).joinpath(f"PTDF_{self.timestamp}" + ext)
        plt.savefig(output_img)
        plt.close('all')

    def plot_Pjk(self, Pij, Linfo, lmax):

        #######################
        ## Power in lines    ##
        #######################
        
        nil = Linfo['NumNodes'].values # nodes in lines (nil)
        namel = Linfo['Line_Name'].values # name of lines (namel)
        
        if not isinstance(lmax, pd.DataFrame):
            lmax   = np.reshape(lmax, (len(Pij.columns), len(Pij)), order='F').T
            lmax = pd.DataFrame(lmax, Pij.index, Pij.columns)
        
        # for each line - Main cycle
        cont = 0
        for ni, na in zip(nil, namel):
            
            # if np.any(Pij.values[cont:cont + ni,:] > lmax.values[cont:cont + ni,:]):
            if na == "Line.l3" or na == "Line.l115" or na == "Line.sw1":
                plt.clf()
                fig, ax = plt.subplots(figsize=(h,w))                
                # leg = [node for node in Pij.index[cont:cont + ni]]
                lim = lmax[cont:cont + ni].mean(axis = 0).to_frame()
                lim.columns = ['Thermal limit']
                concatenated = pd.concat([Pij[cont:cont + ni], lim.T])
                concatenated.T.plot()

                if self.title:
                    ax.set_title(f'Line power flow_{na}_{self.niter}')
                    plt.ylabel('Power (kW)')
                    plt.xlabel('Time (hrs)')
                    
                fig.tight_layout()
                # create flows directory to store results
                output_dirPjk = pathlib.Path(self.output_dir).joinpath("Flows")
                if not os.path.isdir(output_dirPjk):
                    os.mkdir(output_dirPjk)
                output_img = pathlib.Path(self.output_dir).joinpath(f"Power_{na}_{self.niter}_{self.timestamp}" + ext)
                plt.savefig(output_img)
                
                # save as pkl as well
                output_pkl = pathlib.Path(output_dirPjk).joinpath(f"Power_{na}_{self.niter}_{self.timestamp}.pkl")
                concatenated.to_pickle(output_pkl)
                
            cont += ni
            plt.close('all')

    def plot_Demand(self, DemandProfile):

        # ####################
        # ##   Demand  ## 
        # ####################
        totalDemand =  DemandProfile.sum(axis = 0).to_frame()
        
        # index = np.where(DemandProfilei) 
        plt.clf()
        fig, ax = plt.subplots(figsize=(h,w))
        
        totalDemand.T.plot(style='o--', color = 'grey', alpha=0.3)
        totalDemand.T.step()
        
        if self.title:
            ax.set_title('Demand profile')
            plt.ylabel('Power (kW)')
            plt.xlabel('Time (hrs)')
        
        fig.tight_layout()
        output_img = pathlib.Path(self.output_dir).joinpath("Demand_profile_{self.timestamp}"+ ext)
        plt.savefig(output_img)
        plt.close('all')

    def plot_Dispatch(self, Pg):

        #######################
        ## Power Dispatch  ##
        #######################
    
        plt.clf()
        fig, ax = plt.subplots(figsize=(h,w))
        Pg = Pg[Pg.any(axis=1)]
        Pg.T.plot(legend=False) # use any to plot dispatched nodes
        if self.title:
            ax.set_title(f'Power substation - peak = {np.sum(np.max(Pg,0))}', fontsize=15)
            plt.ylabel('Power (kW)', fontsize=12)
            plt.xlabel('Time (hrs)', fontsize=12)
        fig.tight_layout()

        output_dirDispatch = pathlib.Path(self.output_dir).joinpath("Dispatch")
        if not os.path.isdir(output_dirDispatch):
            os.mkdir(output_dirDispatch)

        output_img = pathlib.Path(self.output_dir).joinpath(f"Power_Dispatch_{self.niter}_{self.timestamp}"+ ext)
        plt.savefig(output_img)
        plt.close('all')

        # save as pkl as well
        output_pkl = pathlib.Path(output_dirDispatch).joinpath(f"Power_Dispatch_{self.niter}_{self.timestamp}.pkl")
        Pg.to_pickle(output_pkl)

    def plot_DemandResponse(self, Pdr):

        #######################
        ## Demand Response  ##
        #######################
    
        plt.clf()
        fig, ax = plt.subplots(figsize=(h,w))
        legendi = Pdr[Pdr.any(axis=1)].index.tolist() # use any to plot dispatched nodes
        Pdr.T.plot(legend=False)
        ax.legend(legendi, prop={'size': 10})
        
        if self.title:
            ax.set_title(f'Demand Response - total {round(np.sum(Pdr),3)}', fontsize=15)
            plt.ylabel('Power (kW)', fontsize=15)
            plt.xlabel('Time (hrs)', fontsize=15)

        output_dirDR = pathlib.Path(self.output_dir).joinpath("DR")
        if not os.path.isdir(output_dirDR):
            os.mkdir(output_dirDR)

        output_img = pathlib.Path(self.output_dir).joinpath(f"DemandResponse_{self.niter}_{self.timestamp}"+ ext)
        plt.savefig(output_img)
        plt.close('all')

        # save as pkl as well
        output_pkl = pathlib.Path(output_dirDR).joinpath(f"DemandResponse_{self.niter}_{self.timestamp}.pkl")
        Pdr.to_pickle(output_pkl)

    def plot_storage(self, E, batt, cgn):

        ######################
        ## Storage storage  ##
        ######################

        # get the nodes with batteries
        row, _ = np.where(batt['BatIncidence']==1)
    
        plt.clf()
        fig, ax1 = plt.subplots(figsize=(h,w))
        leg = [self.PTDF.columns[node] for node in row]
        xrange = np.arange(1,self.PointsInTime+1,1)
        ax1.step(xrange,(E).T)
        if self.title:
            ax1.set_title(f'Prices vs static battery charging_{self.niter}', fontsize=15)
            ax1.set_ylabel('Energy Storage (kWh)', fontsize=15)
            ax1.set_xlabel('Time (hrs)', fontsize=15)
        plt.legend(leg) 
    
        ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
        color = 'tab:purple'
        # ax2.set_ylabel('HourlyMarginalPrice ($/kWh)', color=color, fontsize=16)  # we already handled the x-label with ax1
        ax2.step(xrange, cgn.T, color=color)
        ax2.step(xrange, cgn.T, 'm*', markersize=3)
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout() 
    
        output_img = pathlib.Path(self.output_dir).joinpath(f"EnergyStorage_{self.niter}_{self.timestamp}"+ ext)
        plt.savefig(output_img)
        plt.close('all')
        
    def plot_LMP(self, LMP, LMPvar, demandProfilei=None):
        
        if demandProfilei is not None:
            LMPdemand = LMP[demandProfilei.values]
        else:
            LMPdemand = LMP
            
        LMPdir = pathlib.Path(self.output_dir).joinpath(f"LMP_{LMPvar}")

        if not os.path.isdir(LMPdir):
            os.mkdir(LMPdir)
                
        if LMPvar == 'Scharge' or LMPvar == 'Sdischarge' or LMPvar == 'E':
            k = 3
            cont = 3
        else:
            k = 3 
            cont = 3 
            
        nplots = int(len(LMPdemand)/cont)
         
        for i in range(nplots):
            plt.clf()
            fig, ax = plt.subplots(figsize=(h,w))
            try:
                LMPdemand[i*k:cont].T.plot()
            except:
                breakpoint()
            cont += k
            
            output_img = pathlib.Path(LMPdir).joinpath(f"LMP_{i}_{LMPvar}_{self.niter}_{self.timestamp}"+ ext)
            plt.savefig(output_img)
            plt.close('all')
    

    def extractLMP(self, LMP, DR, Storage, batt):
        
        n = self.n
        m = self.l 
        PointsInTime = self.PointsInTime
        numBatteries = batt['numBatteries']
        LMP = np.squeeze(LMP)
        
        # Extract solution
        if DR and not Storage:
            LMP_Pg    = np.reshape(LMP[:n*PointsInTime], (PointsInTime,n), order='F').T
            LMP_Pdr   = np.reshape(LMP[n*PointsInTime:2*n*PointsInTime], (PointsInTime,n), order='F').T;
            if self.jk:
                LMP_Pij   = np.reshape(LMP[2*n*PointsInTime:(2*n+m)*PointsInTime], (PointsInTime, m), order='F').T
            else:
                LMP_Pij = None
                                
            LMP_Pchar = None
            LMP_Pdis  = None
            LMP_E     = None

        elif DR and Storage:
            LMP_Pg    = np.reshape(LMP[:n*PointsInTime], (PointsInTime,n), order='F').T
            LMP_Pdr   = np.reshape(LMP[n*PointsInTime:2*n*PointsInTime], (PointsInTime,n), order='F').T;
            if self.jk:
                LMP_Pij   = np.reshape(LMP[2*n*PointsInTime:(2*n+m)*PointsInTime], (PointsInTime, m), order='F').T
            else:
                LMP_Pij = None
            LMP_Pchar = np.reshape(LMP[(2*n+m)*PointsInTime:(2*n+m)*PointsInTime + numBatteries*PointsInTime] , (PointsInTime,numBatteries), order='F').T
            LMP_Pdis  = np.reshape(LMP[(2*n+m+numBatteries)*PointsInTime:(2*n+m+2*numBatteries)*PointsInTime], (PointsInTime,numBatteries), order='F').T
            LMP_E     = np.reshape(LMP[(2*n+m+2*numBatteries)*PointsInTime:-numBatteries], (PointsInTime,numBatteries), order='F').T
                
        elif Storage and not DR:
            LMP_Pg    = np.reshape(LMP[:n*PointsInTime], (PointsInTime,n), order='F').T
            LMP_Pdr   = None
            if self.jk:
                LMP_Pij   = np.reshape(LMP[n*PointsInTime:(n+m)*PointsInTime], (PointsInTime, m), order='F').T
            else:
                LMP_Pij = None
            LMP_Pchar = np.reshape(LMP[(n+m)*PointsInTime:(n+m)*PointsInTime + numBatteries*PointsInTime] , (PointsInTime,numBatteries), order='F').T
            LMP_Pdis  = np.reshape(LMP[(n+m+numBatteries)*PointsInTime:(n+m+2*numBatteries)*PointsInTime], (PointsInTime,numBatteries), order='F').T
            LMP_E     = np.reshape(LMP[(n+m+2*numBatteries)*PointsInTime:-numBatteries], (PointsInTime,numBatteries), order='F').T
        else:
            LMP_Pg    = np.reshape(LMP[:n*PointsInTime], (PointsInTime,n), order='F').T
            LMP_Pdr   = None
            if self.jk:
                LMP_Pij   = np.reshape(LMP[n*PointsInTime:(n+m)*PointsInTime], (PointsInTime, m), order='F').T
            else:
                LMP_Pij = None
            LMP_Pchar = None
            LMP_Pdis  = None
            LMP_E     = None

        return LMP_Pg, LMP_Pdr, LMP_Pij, LMP_Pchar, LMP_Pdis, LMP_E

    def extractResults(self, x, DR, Storage, batt):
        
        n = self.n
        m = self.l 
        PointsInTime = self.PointsInTime
        numBatteries = batt['numBatteries']
        
        # Extract solution
        if DR and not Storage:
            Pg    = np.reshape(x.X[:n*PointsInTime], (PointsInTime,n), order='F').T
            Pdr   = np.reshape(x.X[n*PointsInTime:2*n*PointsInTime], (PointsInTime,n), order='F').T;
            if self.jk:
                Pij   = np.reshape(x.X[2*n*PointsInTime:(2*n+m)*PointsInTime], (PointsInTime, m), order='F').T
            else:
                Pij = 0
            Pchar = 0
            Pdis  = 0
            E     = 0
            
        elif DR and Storage:
            Pg    = np.reshape(x.X[:n*PointsInTime], (PointsInTime,n), order='F').T
            Pdr   = np.reshape(x.X[n*PointsInTime:2*n*PointsInTime], (PointsInTime,n), order='F').T;
            if self.jk:
                Pij   = np.reshape(x.X[2*n*PointsInTime:(2*n+m)*PointsInTime], (PointsInTime, m), order='F').T
            else:
                Pij = 0
            Pchar = np.reshape(x.X[(2*n+m)*PointsInTime:(2*n+m)*PointsInTime + numBatteries*PointsInTime] , (PointsInTime,numBatteries), order='F').T
            Pdis  = np.reshape(x.X[(2*n+m+numBatteries)*PointsInTime:(2*n+m+2*numBatteries)*PointsInTime], (PointsInTime,numBatteries), order='F').T
            E     = np.reshape(x.X[(2*n+m+2*numBatteries)*PointsInTime:-numBatteries], (PointsInTime,numBatteries), order='F').T
                
        elif Storage and not DR:
            Pg    = np.reshape(x.X[:n*PointsInTime], (PointsInTime,n), order='F').T
            Pdr   = 0
            if self.jk:
                Pij   = np.reshape(x.X[n*PointsInTime:(n+m)*PointsInTime], (PointsInTime, m), order='F').T
            else:
                Pij = 0
            Pchar = np.reshape(x.X[(n+m)*PointsInTime:(n+m)*PointsInTime + numBatteries*PointsInTime] , (PointsInTime,numBatteries), order='F').T
            Pdis  = np.reshape(x.X[(n+m+numBatteries)*PointsInTime:(n+m+2*numBatteries)*PointsInTime], (PointsInTime,numBatteries), order='F').T
            E     = np.reshape(x.X[(n+m+2*numBatteries)*PointsInTime:-numBatteries], (PointsInTime,numBatteries), order='F').T
        else:
            Pg    = np.reshape(x.X[:n*PointsInTime], (PointsInTime,n), order='F').T
            Pdr   = 0
            if self.jk:
                Pij   = np.reshape(x.X[n*PointsInTime:(n+m)*PointsInTime], (PointsInTime, m), order='F').T
            else:
                Pij = 0
            Pchar = 0
            Pdis  = 0
            E     = 0

        return Pg, Pdr, Pij, Pchar, Pdis, E 
