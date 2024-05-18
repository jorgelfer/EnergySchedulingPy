# required for processing
from scr.LP_dispatch_modified import LP_dispatch
from scr.Plotting import plottingDispatch
from scr.SLP_dispatch_modified import SLP_dispatch
import matplotlib.pyplot as plt
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import os
import json

# define DSS path
dataset = "IEEETestCases"
NetworkModel = "123Bus_wye" # "SecondaryTestCircuit_modified", "13Bus", "123Bus", "case3", "4Bus-DY-Bal"
InFile1 = "IEEE123Master.dss" # "Master.DSS", "IEEE13Nodeckt.dss", "IEEE123Master.dss", "case3_unbalanced.dss", "4Bus-DY-Bal.dss"

# define optimization model
dispatch = 'LP'
ext = '.png'
plot = True 
h = 6 
w = 4 

####
# load qsts 
###
DIR = os.getcwd()
json_path = os.path.join(DIR, "..", dataset, NetworkModel, "qsts.json")

# Opening JSON file
f = open(json_path)
 
# returns JSON object as 
# a dictionary
qsts = json.load(f)

####
# preprocess PTDF
###
# row length
rl = len(qsts["dpdp"]["bpns"])

# column length
cl = len(qsts["dpdp"]["nodes"])

# reshape flatten array
PTDF = np.reshape(qsts["dpdp"]["matrix_ij"], (rl, cl), order='F')
PTDF = pd.DataFrame(PTDF, columns=qsts["dpdp"]["nodes"], index=qsts["dpdp"]["lns"])


####
# preprocess penalty factors
###
pf = pd.Series(np.ones(len(qsts["dpdp"]["nodes"])), index=qsts["dpdp"]["nodes"])

####
# adjust lossless PTDF
####
PTDF= PTDF.round()

# points in time
PointsInTime = len(qsts["time"])

####
# preprocess load
###
nodes = qsts["dpdp"]["nodes"]
ddict = {key: np.zeros(PointsInTime) for key in nodes}
for load in qsts["load"]:

    # get load bus uid
    bus = load["bus"] 

    # get load phases
    phases = load["phases"]

    # load power 
    for ph in phases:
        ddict[bus + f".{ph}"] = np.asarray(load["p"][f"{ph}"]) 

DemandProfile = pd.DataFrame(np.stack([ddict[n] for n in nodes]), index = np.asarray(nodes)) # in kW

####
# preprocess generation
####

# costs
Gcost = 10000 * np.ones(DemandProfile.shape)
cost = np.asarray([18.78, 19.35, 19.6, 20.28, 27.69, 34.93, 36.33, 31.69, 29.4, 28.7, 28.49, 25.97, 19.61, 18.65, 17.95, 17.65, 17.8, 18.26, 19.43, 25.88, 19.35, 19.81, 21.46, 19.16])  # in $/MWh
Gcost[0,:] = cost * 1e-3 # in $/kWh
Gcost[1,:] = cost * 1e-3 # in $/kWh
Gcost[2,:] = cost * 1e-3 # in $/kWh
cgn = np.reshape(Gcost.T, (1, Gcost.size), order="F")

# limits
Gmax = np.zeros((len(PTDF.columns), 1))
Gmax[0,0] = 2000 # [kW] asume the slack conventional phase is here
Gmax[1,0] = 2000 # [kW] asume the slack conventional phase is here
Gmax[2,0] = 2000 # [kW] asume the slack conventional phase is here
max_profile = np.kron(Gmax, np.ones((1,PointsInTime)))
Gmax = np.reshape(max_profile.T, (1,np.size(max_profile)), order='F')

####
# preprocess branches
####

# line costs
Pijcost = 0.0 * np.zeros((len(PTDF), PointsInTime))
clin = np.reshape(Pijcost.T, (1,Pijcost.size), order="F")

# define line limits
bpns = qsts["dpdp"]["bpns"]
ldict = {key: 0.0 for key in bpns}

for br in qsts["branch"]:
    # get branch uid
    uid = br["uid"].split(".")[1]

    # assign normal flow limit (already comes by phase in kW)
    normal_flow_limit = br["normal_flow_limit"] # in kVA

    for ph in br["phases"]:
        ldict[uid + f".{ph}"] = normal_flow_limit

# Lmaxi = pd.DataFrame(np.asarray([ldict[n] for n in bpns]), np.asarray(bpns))
Lmaxi = 2000 * np.ones((len(PTDF),1))
Lmax = np.kron(Lmaxi, np.ones((1,PointsInTime)))
Lmax = pd.DataFrame(Lmax, index=np.asarray(bpns))

####
# preprocess storage
####

batt = dict()
numBatteries= 6
batt['numBatteries'] = numBatteries
BatIncidence = np.zeros((len(PTDF.columns),numBatteries))
BatIncidence[PTDF.columns == '18.1', 0] = 1  # same nodes as Julia for 123Bus
BatIncidence[PTDF.columns == '18.2', 1] = 1
BatIncidence[PTDF.columns == '18.3', 2] = 1
BatIncidence[PTDF.columns == '97.1', 3] = 1
BatIncidence[PTDF.columns == '97.2', 4] = 1
BatIncidence[PTDF.columns == '97.3', 5] = 1
batt['BatIncidence'] = BatIncidence
BatSizes = 20 * np.ones((1, numBatteries))  # [kWh] same as capacity in the julia code
batt['BatSizes'] = BatSizes
BatChargingLimits = 7.6*np.ones((1,numBatteries)) # [kWh]
batt['BatChargingLimits'] = BatChargingLimits
BatEfficiencies = 0.90*np.ones((1,numBatteries))
batt['BatEfficiencies'] = BatEfficiencies
BatInitEnergy = BatSizes * 0.4 
batt['BatInitEnergy'] = BatInitEnergy
Pbatcost = 0                                # This parameter is importat for cost analysis puposes $/kWh
batt['Pbatcost'] = Pbatcost
ccharbat = Pbatcost * np.ones((1,2*numBatteries*PointsInTime))
batt['ccharbat'] = ccharbat
ccapacity = Pbatcost * np.ones((1, numBatteries*(PointsInTime + 1)))
batt['ccapacity'] = ccapacity
batt['BatPenalty'] = np.ones((1,numBatteries)) 


####
# preprocess demand response
####
DRcost = 1.0 # in $/kWh
n = len(PTDF.columns)
c = len(PTDF.index)
cdr = DRcost*np.ones((1, n * PointsInTime)) 

####
# preprocess voltage base for each node
####
nodes = qsts["dpdp"]["nodes"]
vdict = {key: 0.0 for key in nodes}

for bus in qsts["bus"]:
    # get bus uid
    uid = bus["uid"]

    # get load phases
    phases = bus["phases"]

    # load power 
    for ph in phases:
        vdict[uid + f".{ph}"] = bus["kV_base"]

v_basei = pd.DataFrame(np.asarray([vdict[n] for n in nodes]), index = np.asarray(nodes))
v_base = np.kron(v_basei, np.ones((1, PointsInTime)))
v_base = pd.DataFrame(v_base, index=v_basei.index)


####
# preprocess voltage sensitivity
####
# row and columng length
rcl = len(qsts["dvdp"]["nodes"])

# reshape flatten array
dvdp = np.reshape(qsts["dvdp"]["matrix"], (rcl, rcl), order='F')
dvdp = pd.DataFrame(dvdp, columns=qsts["dvdp"]["nodes"], index=qsts["dvdp"]["nodes"])


####
# preprocess initial flows 
####
bpns = qsts["dpdp"]["bpns"]
fdict = {key: np.zeros(PointsInTime) for key in bpns}
for br in qsts["branch"]:
    # get branch uid
    uid = br["uid"].split(".")[1]

    # for each flow
    for ph in br["phases"]:
        lenp = len(br["p_nm"][f"{ph}"])
        fdict[uid + f".{ph}"] = np.asarray(br["p_nm"][f"{ph}"])
Pjk_0 = pd.DataFrame(np.stack([fdict[n] for n in bpns]), index = np.asarray(bpns))

####
# preprocess initial voltages magnitutes
####
nodes = qsts["dpdp"]["nodes"]
vdict = {key: np.zeros(PointsInTime) for key in nodes}
for bus in qsts["bus"]:

    # get load bus uid
    uid = bus["uid"] 

    # get load phases
    phases = bus["phases"]

    # load power 
    for ph in phases:
        vdict[uid + f".{ph}"] = np.asarray(bus["vm"][f"{ph}"])
Vm_0 = pd.DataFrame(np.stack([vdict[n] for n in nodes]), index = np.asarray(nodes))

####
# initial generation
####
nodes = qsts["dpdp"]["nodes"]
gdict = {key: np.zeros(PointsInTime) for key in nodes}
for vs in qsts["vsource"]:

    # get load bus uid
    uid = vs["bus"] 

    # get load phases
    phases = vs["phases"]

    # load power 
    for ph in phases:
        gdict[uid + f".{ph}"] = np.asarray(vs["p"][f"{ph}"])
Pg_0 = pd.DataFrame(np.stack([gdict[n] for n in nodes]), index = np.asarray(nodes))

####
# initial demand response 
####
Pdr_0 = pd.DataFrame(0.0, index = np.asarray(nodes), columns = np.arange(PointsInTime))

####################################
########## optimization
####################################
storage = False # if storage is considered

# create an instance of the dispatch class
if dispatch == 'SLP':
    Obj = SLP_dispatch(pf, PTDF, batt, Lmax, Gmax, cgn, clin, cdr, v_base, dvdp, storage=storage)
    x, m, LMP = Obj.PTDF_SLP_OPF(DemandProfile, Pjk_0, Vm_0, Pg_0, Pdr_0)
else:
    #   = LP_dispatch(pf, PTDF, batt, Pjk_lim, Gmax, cgn, clin, cdr, v_base, dvdp, storage, vmin, vmax)
    Obj = LP_dispatch(pf, PTDF, batt, Lmax, Gmax, cgn, clin, cdr, v_base, dvdp, storage=storage)
    x, m, LMP, Ain = Obj.PTDF_LP_OPF(DemandProfile, Pjk_0, Vm_0, Pg_0, Pdr_0)

# call the OPF method
# x, m, LMP, Ain = dispatch_obj.PTDF_LP_OPF(demandProfile, Pjk_0, v_0, Pg_0, PDR_0)
print(x.X)
print('Obj: %g' % m.objVal)

#plot results
plot_obj = plottingDispatch(None, None, PointsInTime, DIR, vmin=0.95, vmax=1.05, PTDF=PTDF, dispatchType=dispatch)

# extract dispatch results
Pg, Pdr, Pij, Pchar, Pdis, E = plot_obj.extractResults(x=x, 
                                                       DR=True, 
                                                       Storage=storage, 
                                                       batt=batt
                                                       )
demandProfilei = DemandProfile.any(axis=1)
lnodes = np.where(demandProfilei)[0]    

# extract LMP results
LMP_Pg, LMP_Pdr, LMP_Pij, LMP_Pchar, LMP_Pdis, LMP_E = plot_obj.extractLMP(LMP, True, storage, batt)
outLMP = pd.DataFrame(LMP_Pg[lnodes,:], np.asarray(PTDF.columns[lnodes]), Vm_0.columns) 
subCost = pd.DataFrame(Gcost[0:3,:], index=PTDF.columns[0:3], columns=Vm_0.columns)
outLMP = pd.concat([subCost, outLMP], axis=0)
print(outLMP.head())

# save initial LMP
plt.clf()
fig, ax = plt.subplots(figsize=(h,w))
outLMP.T.plot(legend=False)
title = f"init_LMP_dispatch_{dispatch}"
ax.set_title(title)
fig.tight_layout()
output_img = os.path.join(DIR, title + ext)
plt.savefig(output_img)
plt.close('all')
