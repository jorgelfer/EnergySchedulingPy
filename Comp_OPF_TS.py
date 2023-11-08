"""
by Jorge
"""

# required for processing
from scr.DispatchTS import Dispatch
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import pathlib
import os
#required for plotting
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
plt.rcParams.update({'font.size': 12})
# from matplotlib.patches import StepPatch

def create_battery(PTDF, PointsInTime):
    """Method to define battery parameters"""

    batt = dict()
    numBatteries= 3
    batt['numBatteries'] = numBatteries
    BatIncidence = np.zeros((len(PTDF.columns),numBatteries))
    BatIncidence[PTDF.columns == '692.1', 0] = 1
    BatIncidence[PTDF.columns == '692.2', 1] = 1
    BatIncidence[PTDF.columns == '692.3', 2] = 1
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

    
script_path = os.path.dirname(os.path.abspath(__file__))

# load PTDF results
PTDF_file = pathlib.Path(script_path).joinpath("inputs", "PTDF_jk_Nov-11-2021_1627.pkl")
PTDFi = pd.read_pickle(PTDF_file)

# adjust lossless PTDF
PTDF = PTDFi / 10 # divide by perturbation injection value
PTDF= PTDF.round()
# PTDF.replace(to_replace= np.nan, value = 0, inplace=True)

# generation mix
GenMix_file = pathlib.Path(script_path).joinpath("inputs", "GeorgiaGenerationMix2.xlsx")
t = pd.read_excel(GenMix_file)
GenAlpha = t["Alpha"]
GenBeta = np.expand_dims(t["Beta"], axis=1)
PointsInTime = len(GenAlpha) 
# PointsInTime = 4

# Load demand profile
Demand_file = pathlib.Path(script_path).joinpath("inputs", "DemandProfile_Nov-11-2021_1627.pkl")
DemandProfilei = 1* pd.read_pickle(Demand_file)[0]
DemandProfile = np.expand_dims(DemandProfilei, axis=1)
# Expand feeder demand for time series analysis
DemandProfile =  DemandProfile @ GenBeta.T
DemandProfile = DemandProfile[:,:PointsInTime]


# Load generation costs
GenPrice_file = pathlib.Path(script_path).joinpath("inputs", "HourlyMarginalPrice.xlsx")
tcost = pd.read_excel(GenPrice_file)
Gcost = 10000*np.ones((len(DemandProfile),PointsInTime))
cost_wednesday = tcost.values[156,1:-1]
Gcost[0,:] = cost_wednesday[:PointsInTime]
Gcost[1,:] = cost_wednesday[:PointsInTime]
Gcost[2,:] = cost_wednesday[:PointsInTime]
cgn = np.reshape(Gcost.T, (1,Gcost.size), order="F")


# Line costs
Pijcost = 0.01*np.ones((len(PTDF),PointsInTime))
clin = np.reshape(Pijcost.T, (1,Pijcost.size), order="F")

# Define generation limits
Gmax = np.zeros((len(PTDF.columns),1))
Gmax[0,0] = 2000 # asume the slack conventional phase is here
Gmax[1,0] = 2000 # asume the slack conventional phase is here
Gmax[2,0] = 2000 # asume the slack conventional phase is here
max_profile = np.kron(Gmax, np.ones((1,PointsInTime)))
Gmax = np.reshape(max_profile.T, (1,np.size(max_profile)), order='F')

# Define line limits
Lmax_file = pathlib.Path(script_path).joinpath("inputs", "Pjk_ratings_Nov-11-2021_1745.pkl")
Lmaxi = 1*pd.read_pickle(Lmax_file)[0]
Lmax = np.kron(Lmaxi, np.ones((1,PointsInTime)))
Lmax = np.reshape(Lmax.T, (1,np.size(Lmax)), order="F")

# Storage
batt = create_battery(PTDF, PointsInTime)
numBatteries = 3

##############################################
### Addition in this file: EV profile ########
##############################################

# EV load profile
EV_file = pathlib.Path(script_path).joinpath("inputs", "DumbChargingProfiles.xlsx")
tEV = pd.read_excel(EV_file, 'Profiles')
EV_profile = tEV.values[1:,21:]
# EV_profile = tEV.values[1:,:] # debug
# accomodate the interval from 8 to 8 -> 1 to 24
EV_profile = np.concatenate((EV_profile[33:,:], EV_profile[0:33,:]), 0)

# The initial file is in intervals of 30 min. To match price,
# sum each consecutive row and turn it into 1h intervals.
EV_profile = 2*(EV_profile[0::2,:] + EV_profile[1::2,:])

# nodes to connect the EV
DemandProfilei['645.3'] = 1
index = np.where(DemandProfilei)

####################################
########## optimization
####################################

nodes = [*PTDF.columns[index[0]].values]
n = len(nodes) - 1

# create an instance of the dispatch class
Obj = Dispatch(PTDF, PointsInTime, batt, Lmax, Gmax, cgn, clin)

# call the OPF method
x, m = Obj.PTDF_OPF(DemandProfile, storage=False)
# Extract solution
Pg = np.reshape(x.X[:n*PointsInTime], (PointsInTime,n), order='F').T
Dispatch_NoEv = Pg[:3,:].T


# Obj.Plot_results(x, m, Gcost[0,:], 'No_EV')

obj_mat = list()

# store initial result
obj_mat.append(m.objVal)

for i, ind in enumerate(index[0]):
    DemandProfile[ind,:] = DemandProfile[ind,:] + EV_profile[:,i].T
    
    # call the OPF method
    x, m = Obj.PTDF_OPF(DemandProfile, storage=False)
            
    #plot_results
    # Obj.Plot_results(x, m, Gcost[0,:], nodes[i])
    
    # store result
    obj_mat.append(m.objVal)
    

Pg = np.reshape(x.X[:n*PointsInTime], (PointsInTime,n), order='F').T
DispatchFullEVs = Pg[:3,:].T


###
## Dispatch difference
###

bs = mlines.Line2D([], [], color='black', marker='s', linestyle='',
                          markersize=10, label='sourcebus.1')
bo = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                          markersize=10, label='sourcebus.2')
bd = mlines.Line2D([], [], color='black', marker='d', linestyle='None',
                          markersize=10, label='sourcebus.2')

ne = mlines.Line2D([], [], color='red', marker='_', linestyle='None',
                          markersize=10, label='No_EV')
we = mlines.Line2D([], [], color='green', marker='_', linestyle='None',
                          markersize=10, label='20_EVs')

plt.clf()
fig, ax = plt.subplots(figsize=(15, 8))
# plt.figure(figsize=(15,8))                  
xrange = np.arange(1,PointsInTime+1,1)

markers = ["rs","ro", "rd"]
for yp, m in zip(Dispatch_NoEv.T, markers):
    ax.plot(xrange, yp, 'r')
    ax.plot(xrange, yp, m, markersize=10)

plt.legend(handles=[bs, bo, bd, ne, we])


markers = ["gs","go", "gd"]

for yp, m in zip(DispatchFullEVs.T, markers):
    ax.plot(xrange, yp, 'g')
    ax.plot(xrange, yp, m, markersize=10)

# ax.plot(xrange,DispatchFullEVs, 'g')
# ax.plot(xrange,DispatchFullEVs, 'gd')
ax.set_title('Power Dispatch impact with EVs', fontsize=25)
plt.ylabel('Power (kW)', fontsize=16)
plt.xlabel('Time (hrs)', fontsize=16)


output_img = pathlib.Path(script_path).joinpath("outputs", "Power_Comparison_EV_NS.png")
plt.savefig(output_img)


###
## EV cost function
###
nodes = [*PTDF.columns[index[0]].values]
nodes.insert(0,'No_EV')
plt.clf()
fig, ax = plt.subplots(figsize=(15, 8))
p = sns.lineplot(x=nodes, y=obj_mat)
p.set_xlabel("Incremental EV connection", fontsize = 20)
p.set_ylabel("Objective function cost [$]", fontsize = 20)
p.set_title("Cost function analysis including EVs", fontsize = 20)
# ax.set(title = "Cost function analysis including EVs", xlabel = "Incremental EV connection", ylabel = "Objective function cost [$]")
output_img = pathlib.Path(script_path).joinpath("outputs", "Cost_function.png")
plt.savefig(output_img)


###
## EV profile
###
# plt.clf()
# fig, ax = plt.subplots(figsize=(15, 8))
# # plt.figure(figsize=(15,8))                  
# leg = [f'EV_{node}' for node in range(1,21,1)]
# xrange = np.arange(1,PointsInTime+1,1)
# ax.step(xrange,EV_profile)
# ax.set_title('EV charging profile', fontsize=25)
# plt.ylabel('Power (kW)', fontsize=16)
# plt.xlabel('Time (hrs)', fontsize=16)
# plt.legend(leg) 
# output_img = pathlib.Path(script_path).joinpath("outputs", "EV_profile.png")
# plt.savefig(output_img)


# ###
# ## Demand
# ###
# DemandProfilei['645.3'] = 0
# index = np.where(DemandProfilei) 
# plt.clf()
# fig, ax = plt.subplots(figsize=(15, 8))
# # plt.figure(figsize=(15,8))                  
# leg = [*PTDF.columns[index[0]].values]
# xrange = np.arange(1,PointsInTime+1,1)
# ax.step(xrange,DemandProfile.T)
# ax.set_title('Demand profile IEEE13', fontsize=25)
# plt.ylabel('Power (kW)', fontsize=16)
# plt.xlabel('Time (hrs)', fontsize=16)
# plt.legend(leg) 
# output_img = pathlib.Path(script_path).joinpath("outputs", "Demand_profile.png")
# plt.savefig(output_img)


