# required for processing
from scr.DispatchTS_DR_PV_S_DR_voltage import Dispatch
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import os
import json

# define DSS path
dataset = "IEEETestCases"
NetworkModel = "SecondaryTestCircuit_modified"    # "13Bus" 
InFile1 = "Master.DSS"                            # "IEEE13Nodeckt.dss"

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

# get dpdp
PTDF = qsts["dpdp"]["matrix"]

# reshape flatten array
PTDF = np.reshape(qsts["dpdp"]["matrix"], (rl, cl), order='F')
PTDF = pd.DataFrame(PTDF, columns=qsts["dpdp"]["nodes"], index=qsts["dpdp"]["nnns"])

# adjust lossless PTDF
PTDF= PTDF.round()

# points in time
PointsInTime = 24

####
# preprocess load
###
nodes = qsts["dpdp"]["nodes"]
ddict = {key: np.zeros(24) for key in nodes}
for load in qsts["load"]:

    # get load bus
    bus = load["bus"] 

    # get load phases
    phases = load["phases"]

    # load power 
    for ph in phases:
        ddict[bus + f".{ph}"] = np.asarray(load["p"][f"{ph}"])

DemandProfile = pd.DataFrame(np.stack([ddict[n] for n in nodes]), index = np.asarray(nodes))

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
Gmax[0,0] = 2000 # asume the slack conventional phase is here
Gmax[1,0] = 2000 # asume the slack conventional phase is here
Gmax[2,0] = 2000 # asume the slack conventional phase is here
max_profile = np.kron(Gmax, np.ones((1,PointsInTime)))
Gmax = np.reshape(max_profile.T, (1,np.size(max_profile)), order='F')

####
# preprocess branches
####

# line costs
Pijcost = np.zeros((len(PTDF), PointsInTime))
clin = np.reshape(Pijcost.T, (1,Pijcost.size), order="F")

# define line limits
bpns = qsts["dpdp"]["bpns"]
ldict = {key: 0.0 for key in bpns}

for br in qsts["branch"]:

    # get uid
    uid = br["uid"]

    # get line or transformer
    if br["transformer"]:
        # assign normal flow limit
        if len(br["phases"]) > 1:
            normal_flow_limit = br["s_base"] / len(br["phases"]) # in kW
        else:
            normal_flow_limit = br["s_base"] # in kW
    else:
        # assign normal flow limit
        normal_flow_limit = br["normal_flow_limit"] # in kW

    for ph in br["phases"]:
        ldict[uid + f".{ph}"] = normal_flow_limit

Lmaxi = pd.DataFrame(np.asarray([ldict[n] for n in bpns]), np.asarray(bpns))

Lmax = np.kron(Lmaxi, np.ones((1,PointsInTime)))
Lmax = np.reshape(Lmax.T, (1,np.size(Lmax)), order="F")

####
# preprocess storage
####

batt = dict()
numBatteries= 3
batt['numBatteries'] = numBatteries
BatIncidence = np.zeros((len(PTDF.columns),numBatteries))
BatIncidence[PTDF.columns == 'bussec4_1.1', 0] = 1
BatIncidence[PTDF.columns == 'bussec1_3.3', 1] = 1
BatIncidence[PTDF.columns == 'bussec5_1.1', 2] = 1
batt['BatIncidence'] = BatIncidence
BatSizes = 50 * np.ones((1, numBatteries))
batt['BatSizes'] = BatSizes
BatChargingLimits = 7.6*np.ones((1,numBatteries))
batt['BatChargingLimits'] = BatChargingLimits
BatEfficiencies = 0.90*np.ones((1,numBatteries))
batt['BatEfficiencies'] = BatEfficiencies
BatInitEnergy = BatSizes * 0.4 
batt['BatInitEnergy'] = BatInitEnergy
Pbatcost = 0.1
batt['Pbatcost'] = Pbatcost
ccharbat = Pbatcost * np.ones((1,2*numBatteries*PointsInTime))
batt['ccharbat'] = ccharbat
ccapacity = np.zeros((1, numBatteries*(PointsInTime + 1)))
batt['ccapacity'] = ccapacity
batt['BatPenalty'] = np.ones((1,numBatteries)) 

####
# preprocess demand response
####
DRcost = 0.5
cdr = DRcost*np.ones((1,len(PTDF.columns) * PointsInTime)) 

####################################
########## optimization
####################################

# create an instance of the dispatch class
# Obj = Dispatch(PTDF, PointsInTime, batt, Lmax, Gmax, cgn, clin)
# Obj = Dispatch(PTDF, PointsInTime, batt, Lmax, Gmax, cgn, clin, cdr, vBase, vSensi, initVolts, PVnodes)
Obj = Dispatch(PTDF, PointsInTime, batt, Lmax, Gmax, cgn, clin, cdr)

# call the OPF method
# x, m = Obj.PTDF_OPF(DemandProfile, storage=False)
# x, m, Ain = Obj.PTDF_OPF(DemandProfile, voltage=voltage, storage=storage, DR = DR, PV=PV)
x, m = Obj.PTDF_OPF(DemandProfile, storage=True, DR=True)




    # call the OPF method

print(x.X)
print('Obj: %g' % m.objVal)