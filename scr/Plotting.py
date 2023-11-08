
#required for plotting
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9})
# from matplotlib.patches import StepPatch
import seaborn as sns
import os
import time
import pathlib
import numpy as np
import random
h = 4 
w = 2
ext = '.pdf'

class PlottingDispatch:

    def __init__(self, x, DR, Storage, PTDF, PointsInTime, batt, script_path, Ain, title=False):
        
        self.Ain = Ain
        self.title = title
        # preprocessing
        self.PTDF = PTDF
        self.n = len(self.PTDF.columns) #number of nodes
        self.l = len(self.PTDF)         #number of lines
        self.x = x
        
        self.PointsInTime = PointsInTime

        # extract the solution

        self.Pg, self.Pdr, self.Pij, self.Pchar, self.Pdis, self.E = self.__extract_results(x, DR, Storage, batt)

        # time stamp 
        t = time.localtime()
        self.timestamp = time.strftime('%b-%d-%Y_%H%M', t)           
        
        # create directory to store results
        today = time.strftime('%b-%d-%Y', t)
        directory = "Results_" + today
        self.output_dir = pathlib.Path(script_path).joinpath("..","outputs", directory)

        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

    def Plot_voltage(self, vBase, initVolts, demandProfilei):
        
        vbase = np.kron(np.expand_dims(vBase['baseVoltage'].values,1), np.ones((1, self.PointsInTime)))
        res_low = initVolts.values <= 950 * vbase
        res_high = initVolts.values >= 1050 * vbase
        
        dvoltages = self.Ain @ self.x.X
        dvolts = dvoltages[:np.size(dvoltages) // 2]
        
        dv    = np.reshape(dvolts[:self.n*self.PointsInTime], (self.PointsInTime,self.n), order='F').T
        
        v = initVolts - dv
        
        vpu = v / (1000*vbase)
        
        indexDemand = np.where(demandProfilei != 0)[0]
        
        leg = [node for node in vpu.index[indexDemand]]
        xrange = np.arange(1,self.PointsInTime+1,1)
        
        plt.clf()
        fig, ax = plt.subplots(figsize=(h,w))                
        
        # for each node
        for node in indexDemand:
            # hexadecimal = "#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])
            if vpu.index[node] == '675.2' or vpu.index[node] == '692.3':
                ax.plot(xrange,vpu.values[node,:], marker = '*', markersize=2, label=vpu.index[node])#, color=hexadecimal)
            else:
                ax.plot(xrange,vpu.values[node,:])

            if self.title:
                ax.set_title('Volts')
                plt.ylabel('Volts[pu]')
                plt.xlabel('Time (hrs)')
                
        plt.legend()
        ax.plot(xrange,0.95*np.ones((self.PointsInTime,1)), 'dr', markersize=3)
        ax.plot(xrange,1.05*np.ones((self.PointsInTime,1)), 'dr', markersize=3)
        fig.tight_layout()
        # if na == "Line.650632":
        output_img = pathlib.Path(self.output_dir).joinpath(f"voltage_{self.timestamp}" + ext)
        plt.savefig(output_img)
        plt.close('all')
        
        
        
    def Plot_PTDF(self):

        ###########
        ## PTDF  ##
        ###########
        plt.clf()
        fig, ax = plt.subplots(figsize=(h,w))
        ax = sns.heatmap(self.PTDF,annot=False)
        if self.title:
            ax.set_title('PTDF', fontsize=15)
        fig.tight_layout()
        output_img = pathlib.Path(self.output_dir).joinpath(f"PTDF_{self.timestamp}" + ext)
        plt.savefig(output_img)
        plt.close('all')

    def Plot_Pjk(self, Linfo, Lmax, niter):

        #######################
        ## Power in lines    ##
        #######################
        
        nil = Linfo['NumNodes'].values # nodes in lines (nil)
        namel = Linfo['Line_Name'].values # name of lines (namel)
        lmax   = np.reshape(Lmax, (self.PointsInTime, self.l), order='F').T
        xrange = np.arange(1,self.PointsInTime+1,1)
        
        # for each line - Main cycle
        cont = 0
        for ni, na in zip(nil, namel):
            plt.clf()
            fig, ax = plt.subplots(figsize=(h,w))                
            leg = [node for node in self.PTDF.index[cont:cont + ni]]
            ax.plot(xrange,self.Pij[cont:cont + ni,:].T)
            plt.legend(leg, loc="lower right")
            if self.title:
                ax.set_title(f'Line power flow_{na}_{niter}')
                plt.ylabel('Power (kW)')
                plt.xlabel('Time (hrs)')
            ax.plot(xrange,lmax[cont:cont + ni,:].T, 'dr', markersize=3)
            fig.tight_layout()
            if na == "Line.650632":
                output_img = pathlib.Path(self.output_dir).joinpath(f"Power_{na}_{niter}_{self.timestamp}" + ext)
                plt.savefig(output_img)
            cont += ni
        plt.close('all')

    def Plot_Demand(self, DemandProfilei, Beta):

        # ####################
        # ##   Demand  ## 
        # ####################
        totalDemand =  DemandProfilei.sum()
        
        DemandProfile = totalDemand * Beta
        
        # index = np.where(DemandProfilei) 
        plt.clf()
        fig, ax = plt.subplots(figsize=(h,w))

        # leg = [*self.PTDF.columns[index[0]].values]
        xrange = np.arange(1,self.PointsInTime+1,1)
        ax.step(xrange,DemandProfile, where='mid', label = 'Demand Profile')
        ax.plot(xrange,DemandProfile, 'o--', color = 'grey', alpha=0.3)
        
        if self.title:
            ax.set_title('Demand profile IEEE13')
            plt.ylabel('Power (kW)')
            plt.xlabel('Time (hrs)')
        plt.legend() 
        fig.tight_layout()
        output_img = pathlib.Path(self.output_dir).joinpath("Demand_profile_{self.timestamp}"+ ext)
        plt.savefig(output_img)
        plt.close('all')

    def Plot_Dispatch(self, niter):

        #######################
        ## Power Dispatch  ##
        #######################
    
        plt.clf()
        fig, ax = plt.subplots(figsize=(h,w))
        # plt.figure(figsize=(15,8))                  
        leg = [node for node in self.PTDF.columns[:3]]
        xrange = np.arange(1,self.PointsInTime+1,1)
        ax.plot(xrange,self.Pg[np.any(self.Pg,1)].T) # use any to plot dispatched nodes
        if self.title:
            ax.set_title(f'Power from Substation_{niter}', fontsize=15)
            plt.ylabel('Power (kW)', fontsize=12)
            plt.xlabel('Time (hrs)', fontsize=12)
        plt.legend(leg) 
        fig.tight_layout()
        output_img = pathlib.Path(self.output_dir).joinpath(f"Power_Dispatch_{niter}_{self.timestamp}"+ ext)
        plt.savefig(output_img)
        plt.close('all')

    def Plot_DemandResponse(self, niter):

        #######################
        ## Demand Response  ##
        #######################
    
        plt.clf()
        fig, ax = plt.subplots(figsize=(h,w))
        # plt.figure(figsize=(15,8))                  
        leg = [node for node in self.PTDF.columns[np.any(self.Pdr,1)]] # select the nodes that actually responded to the demand response
        xrange = np.arange(1,self.PointsInTime+1,1)
        ax.plot(xrange,self.Pdr[np.any(self.Pdr,1)].T) # use any to plot dispatched nodes
        if self.title:
            ax.set_title(f'Demand Response - total {round(np.sum(self.Pdr),3)}', fontsize=15)
            plt.ylabel('Power (kW)', fontsize=12)
            plt.xlabel('Time (hrs)', fontsize=12)
        plt.legend(leg)
        fig.tight_layout()
        output_img = pathlib.Path(self.output_dir).joinpath(f"DemandResponse_{niter}_{self.timestamp}"+ ext)
        plt.savefig(output_img)
        plt.close('all')

    def Plot_storage(self, batt, cgn, niter):

        ######################
        ## Storage storage  ##
        ######################

        # get the nodes with batteries
        row, _ = np.where(batt['BatIncidence']==1)
    
        plt.clf()
        fig, ax1 = plt.subplots(figsize=(h,w))
        leg = [self.PTDF.columns[node] for node in row]
        xrange = np.arange(1,self.PointsInTime+1,1)
        ax1.step(xrange,(self.E).T)
        if self.title:
            ax1.set_title(f'Prices vs static battery charging_{niter}', fontsize=15)
            ax1.set_ylabel('Energy Storage (kWh)', fontsize=12)
            ax1.set_xlabel('Time (hrs)', fontsize=12)
        plt.legend(leg) 
    
        ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
        color = 'tab:purple'
        # ax2.set_ylabel('HourlyMarginalPrice ($/kWh)', color=color, fontsize=16)  # we already handled the x-label with ax1
        ax2.step(xrange, cgn.T, color=color)
        ax2.step(xrange, cgn.T, 'm*', markersize=3)
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout() 
    
        output_img = pathlib.Path(self.output_dir).joinpath(f"EnergyStorage_{niter}_{self.timestamp}"+ ext)
        plt.savefig(output_img)
        plt.close('all')
    

    def __extract_results(self, x, DR, Storage, batt):
        
        
        n = self.n
        m = self.l
        PointsInTime = self.PointsInTime
        numBatteries = batt['numBatteries']
        
        # Extract solution
        if DR and not Storage:
            Pg    = np.reshape(x.X[:n*PointsInTime], (PointsInTime,n), order='F').T
            Pdr   = np.reshape(x.X[n*PointsInTime:2*n*PointsInTime], (PointsInTime,n), order='F').T;
            Pij   = np.reshape(x.X[2*n*PointsInTime:(2*n+m)*PointsInTime], (PointsInTime, m), order='F').T
            Pchar = 0
            Pdis  = 0
            E     = 0
            
        elif DR and Storage:
            Pg    = np.reshape(x.X[:n*PointsInTime], (PointsInTime,n), order='F').T
            Pdr   = np.reshape(x.X[n*PointsInTime:2*n*PointsInTime], (PointsInTime,n), order='F').T;
            Pij   = np.reshape(x.X[2*n*PointsInTime:(2*n+m)*PointsInTime], (PointsInTime, m), order='F').T
            Pchar = np.reshape(x.X[(2*n+m)*PointsInTime:(2*n+m)*PointsInTime + numBatteries*PointsInTime] , (PointsInTime,numBatteries), order='F').T
            Pdis  = np.reshape(x.X[(2*n+m+numBatteries)*PointsInTime:(2*n+m+2*numBatteries)*PointsInTime], (PointsInTime,numBatteries), order='F').T
            E     = np.reshape(x.X[(2*n+m+2*numBatteries)*PointsInTime:-numBatteries], (PointsInTime,numBatteries), order='F').T
        elif Storage and not DR:
            Pg    = np.reshape(x.X[:n*PointsInTime], (PointsInTime,n), order='F').T
            Pdr   = 0
            Pij   = np.reshape(x.X[n*PointsInTime:(n+m)*PointsInTime], (PointsInTime, m), order='F').T
            Pchar = np.reshape(x.X[(n+m)*PointsInTime:(n+m)*PointsInTime + numBatteries*PointsInTime] , (PointsInTime,numBatteries), order='F').T
            Pdis  = np.reshape(x.X[(n+m+numBatteries)*PointsInTime:(n+m+2*numBatteries)*PointsInTime], (PointsInTime,numBatteries), order='F').T
            E     = np.reshape(x.X[(n+m+2*numBatteries)*PointsInTime:-numBatteries], (PointsInTime,numBatteries), order='F').T
        else:
            Pg    = np.reshape(x.X[:n*PointsInTime], (PointsInTime,n), order='F').T
            Pdr   = 0
            Pij   = np.reshape(x.X[n*PointsInTime:(n+m)*PointsInTime], (PointsInTime, m), order='F').T
            Pchar = 0
            Pdis  = 0
            E     = 0
        return Pg, Pdr, Pij, Pchar, Pdis, E 
