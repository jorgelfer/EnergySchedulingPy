"""
by Jorge
"""

# required for processing
import numpy as np
import pandas as pd
import pathlib
#required for plotting
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
from Methods.SLP_dispatch import SLP_dispatch
from Methods.LP_dispatch import LP_dispatch
from Methods.plotting import plottingDispatch
from Methods.loadHelper import loadHelper

# Methods
def create_output(Pg, Pdr, Pchar, Pdis, LMP, load, PTDF, lnodes, Snodes, PVnodes, timeVec, PV, storage):
    """function to create outputs"""
    outLMP = pd.DataFrame(LMP[lnodes,:], np.asarray(PTDF.columns[lnodes]), timeVec) 
    lnames = [load.loc[n] for n in PTDF.columns[lnodes]]
    outDR = pd.DataFrame(Pdr[lnodes,:], np.asarray(lnames), timeVec)
    
    # check PV
    if PV:
        outGen = pd.DataFrame(Pg[PVnodes,:], np.asarray(PTDF.columns[PVnodes]), timeVec)
    else:
        outGen = None
    # check storage
    if storage:
        outPchar = pd.DataFrame(Pchar, np.asarray(PTDF.columns[Snodes]), timeVec) 
        outPdis = pd.DataFrame(Pdis, np.asarray(PTDF.columns[Snodes]), timeVec) 
    else:
        outPchar = None
        outPdis = None

    return outGen, outDR, outPchar, outPdis, outLMP

def create_battery(PTDF, pointsInTime, sbus='83', batSize=300):
    """function to define battery parameters"""

    batt = dict()
    numBatteries= 3
    batt['numBatteries'] = numBatteries
    BatIncidence = np.zeros((len(PTDF.columns),numBatteries))
    BatIncidence[PTDF.columns == sbus +'.1', 0] = 1
    BatIncidence[PTDF.columns == sbus +'.2', 1] = 1
    BatIncidence[PTDF.columns == sbus +'.3', 2] = 1
    batt['BatIncidence'] = BatIncidence
    
    BatSizes = batSize * np.ones((1,numBatteries))
    batt['BatSizes'] = BatSizes
    
    BatChargingLimits = (24/pointsInTime)*100*np.ones((1,numBatteries))
    batt['BatChargingLimits'] = BatChargingLimits
    BatEfficiencies = 0.97*np.ones((1,numBatteries))
    batt['BatEfficiencies'] = BatEfficiencies
    
    np.random.seed(2022) # Set random seed so results are repeatable
    BatInitEnergy = BatSizes * np.random.uniform(0.5, 0.8, size=(1,numBatteries)) 
    
    batt['BatInitEnergy'] = BatInitEnergy
    Pbatcost = 0.01
    batt['Pbatcost'] = Pbatcost
    ccharbat = Pbatcost * np.ones((1,2*numBatteries*pointsInTime))
    batt['ccharbat'] = ccharbat
    ccapacity = Pbatcost * np.ones((1, numBatteries*(pointsInTime + 1)))
    batt['ccapacity'] = ccapacity
    batt['BatPenalty'] = np.ones((1,numBatteries)) 

    return batt

def load_PTDF(script_path, case):
    '''function to load PTDF'''

    PTDF_file = pathlib.Path(script_path).joinpath("inputs", case,"PTDF_jk.pkl")
    PTDF = pd.read_pickle(PTDF_file)
    
    # adjust lossless PTDF
    PTDF = PTDF / 10 # divide by perturbation injection value
    
    return PTDF

def load_generationCosts(script_path, n, pointsInTime, freq):
    '''function to load generations costs and perform interpolation'''
    
    GenPrice_file = pathlib.Path(script_path).joinpath("inputs", "HourlyMarginalPrice.xlsx")
    tcost = pd.read_excel(GenPrice_file)
    gCost = 10000*np.ones((n,pointsInTime))
    
    # create load helper method
    help_obj = loadHelper(initfreq = 'H', finalFreq = freq, price=True)
    
    cost_wednesday = pd.Series(tcost.values[225,1:-1]) # 2018-08-14
    # call method for processing series
    cost_wednesday = help_obj.process_pdSeries(cost_wednesday)
    cost_wednesday = np.squeeze(cost_wednesday.values)
    
    gCost[0,:] = cost_wednesday
    gCost[1,:] = cost_wednesday
    gCost[2,:] = cost_wednesday
    
    return gCost, cost_wednesday

def create_PVsystems(freq, Gmax, PTDF, gCost, cost_wednesday, pointsInTime, pv1bus='634', pv2bus='680', pvSize=100):
    '''function to Define the utility scale PVs'''    

    nodesPV1 = [pv1bus +'.1',pv1bus +'.2',pv1bus +'.3']
    nodesPV2 = [pv2bus +'.1',pv2bus +'.2',pv2bus +'.3']

    # define the PV location
    PV1 = np.stack([PTDF.columns == nodesPV1[0], PTDF.columns ==nodesPV1[1], PTDF.columns ==nodesPV1[2]], axis=1)
    PV2 = np.stack([PTDF.columns ==nodesPV2[0],PTDF.columns ==nodesPV2[1],PTDF.columns ==nodesPV2[2]], axis=1)
    PVnodes = np.any(np.concatenate([PV1,PV2],axis=1), axis=1)
    
    # define the maximum output
    Gmax[np.where(np.any(PV1,axis=1))[0]] =  pvSize        #% Utility scale Solar PV    
    Gmax[np.where(np.any(PV2,axis=1))[0]] =  pvSize        #% Utility scale Solar PV

    if pv1bus == pv2bus:
        Gmax[np.where(np.any(PV1,axis=1))[0]] =  300        #% Utility scale Solar PV
    
    # define the cost
    gCost[PVnodes] = 0.1*cost_wednesday#*np.mean(cost_wednesday)
    
    # create load helper method
    help_obj = loadHelper(initfreq = 'H', finalFreq = freq)
    
    # Estimate a PV Profile
    np.random.seed(2022)
    a = np.sin(np.linspace(-4,19,24)*np.pi/15) - 0.5 + np.random.rand(24)*0.2
    a[a<0] = 0
    a = a/max(a)
    
    # call method for processing series
    PVProfile = help_obj.process_pdSeries(pd.Series(a))
    PVProfile[PVProfile<0.01] = 0
    PVProfile = np.squeeze(PVProfile.values)

    return Gmax, gCost, PVnodes, PVProfile

def compute_penaltyFactors(batt, PTDF): 
    '''function to Compute penalty factors'''
        
    # compute dPgref
    dPgref = np.min(PTDF[:3])
    
    # dPl/dPgi = 1 - (- dPgref/dPgi) -> eq. L9_25
    
    # ITLi = dPL/dPGi 
    ITL = 1 + dPgref # Considering a PTDF with transfer from bus i to the slack. If PTDF is calculated in the converse, then it will be 1 - dPgref
    
    Pf = 1 / (1- ITL)
    
    # substation correction
    Pf['150.1'] = 1
    Pf['150.2'] = 1
    Pf['150.3'] = 1
    # batt incidence
    BatIncidence = batt['BatIncidence'] 
    
    # nodes with batt 
    nodes = np.where(np.any(BatIncidence,1))[0]
    
    # assign penalty factors 
    batt['BatPenalty'] = np.asarray([Pf.values[n] for n in nodes])#min([Pf.values[n] for n in nodes[0]])*np.ones((1,3)) 

    return batt, Pf, nodes

def load_voltageSensitivity(script_path, case):
    '''funtion to load voltage sensitivity'''
    
    # voltage sensitivity
    dfVS_file = pathlib.Path(script_path).joinpath("inputs", case, "VoltageSensitivity.pkl")
    dfVS = pd.read_pickle(dfVS_file)

    # adjust voltage sensi matrix
    dfVS = dfVS / 10 # divide by perturbation injection value
    
    return dfVS


def load_lineLimits(script_path, case, PTDF, pointsInTime, DR, Pij):
    '''function to load line Limits'''

    if DR == True:
        Lmax_file = pathlib.Path(script_path).joinpath("inputs", case, "Pjk_ratings.pkl") #high limits: Pjk_ratings_Nov-11-2021_1745 ## low limits:Pjk_ratings_Nov-20-2021_1154
        Lmaxi = pd.read_pickle(Lmax_file)
    else:

        #debug:
        Lmaxi = 2000 * np.ones((len(PTDF),1))
        
    # expand tempolar equations
    Lmax = np.kron(Lmaxi, np.ones((1,pointsInTime)))
    
    # extract violating Lines
    Lmax = pd.DataFrame(Lmax, Pij.index, Pij.columns)
    compare = Pij > Lmax
    violatingLines = compare.any(axis=1)
    
    # Line Info
    Linfo_file = pathlib.Path(script_path).joinpath("inputs", case, "LineInfo.pkl")
    Linfo = pd.read_pickle(Linfo_file)

    return violatingLines, Lmax, Linfo

def compute_violatingVolts(v_0, v_base, vmin, vmax):

    # extract violating Lines
    v_lb = (vmin*1000)*v_base 
    v_ub = (vmax*1000)*v_base

    compare = (v_0 > v_ub) | (v_0 < v_lb) 
    violatingVolts = compare.any(axis=1)
    
    return violatingVolts

# define the type of analysis;

    
def schedulingDriver(batSize, pvSize, output_dir, iterName, freq, script_path, case, outDSS, dispatchType, vmin, vmax, PF=True, voltage=True, DR=True, plot=False):
    
    # define Storage and PV
    if batSize == 0:
        storage=False
    else:
        storage=True
    sbus='83'

    if pvSize == 0:
        PV=False
    else:
        PV=True
    pv1bus='66'
    pv2bus='80' 

    # extract DSS results
    loadNames = outDSS['loadNames'] 
    Pg_0 = outDSS['initPower']
    v_0 = outDSS['initVolts']
    v_base = outDSS['nodeBaseVolts']
    Pjk_0 = outDSS['initPjks']
    demandProfile = outDSS['initDemand']
    demandProfilei = demandProfile.any(axis=1)
    PDR_0 = pd.DataFrame(np.zeros(v_0.shape), index=Pg_0.index, columns=Pg_0.columns)
    pointsInTime = v_0.shape[1]
    
    # reshape base voltage:
    v_basei = v_base.to_frame()
    v_base = np.kron(v_basei, np.ones((1,pointsInTime)))
    v_base = pd.DataFrame(v_base, index=v_basei.index, columns=v_0.columns)

    violatingVolts = compute_violatingVolts(v_0, v_base, vmin, vmax)

    # load PTDF results
    PTDF = load_PTDF(script_path, case)
    n = len(PTDF.columns)
    l = len(PTDF)
        
    # Storage
    batt = create_battery(PTDF, pointsInTime, sbus, batSize)
    
    #Penalty factors
    if PF:
        batt, pf, Snodes = compute_penaltyFactors(batt, PTDF)
        
    # round the PTDF to make the optimization work
    PTDF = PTDF.round()
    
    # Line costs
    pijCost = 0.01*np.ones((l,pointsInTime))
    clin = np.reshape(pijCost.T, (1,pijCost.size), order="F")
    
    ## Generation settings
    # Load generation costs
    gCost, cost_wednesday = load_generationCosts(script_path, n, pointsInTime, freq)
       
    # Define generation limits
    Gmax = np.zeros((n,1))
    Gmax[0,0] = 2000 # asume the slack conventional phase is here
    Gmax[1,0] = 2000 # asume the slack conventional phase is here
    Gmax[2,0] = 2000 # asume the slack conventional phase is here
    ##
    
    # Line limits and info          
    violatingLines, Pjk_lim, Linfo = load_lineLimits(script_path, case, PTDF, pointsInTime, DR, Pjk_0) 
    
    #Demand Response (cost of shedding load)
    np.random.seed(2022) # Set random seed so results are repeatable
    DRcost = np.random.randint(100,300,size=(1,n)) 
    cdr = np.kron(DRcost, np.ones((1,pointsInTime))) 
    
    #PV system
    if PV:
        Gmax, gCost, PVnodes, PVProfile = create_PVsystems(freq, Gmax, PTDF, gCost, cost_wednesday, pointsInTime, pv1bus, pv2bus, pvSize)
        # Normal gen
        max_profile = np.kron(Gmax, np.ones((1,pointsInTime)))
        # PV nodes
        max_profile[PVnodes,:] = max_profile[PVnodes,:] * PVProfile
    else:
        PVnodes = None
        # Normal gen
        max_profile = np.kron(Gmax, np.ones((1,pointsInTime)))
        
    # Overall Generation limits:
    Gmax = np.reshape(max_profile.T, (1,np.size(max_profile)), order='F')
    
    # Overall Generation costs:
    cgn = np.reshape(gCost.T, (1,gCost.size), order="F")
    
    # load voltage base at each node
    dvdp = load_voltageSensitivity(script_path, case)

    # call the OPF method
    if dispatchType == 'SLP':
        # create an instance of the dispatch class
        dispatch_obj = SLP_dispatch(pf, PTDF, batt, Pjk_lim, Gmax, cgn, clin, cdr, v_base, dvdp, storage, vmin, vmax)
        x, m, LMP = dispatch_obj.PTDF_SLP_OPF(demandProfile, Pjk_0, v_0, Pg_0, PDR_0)
    else:
        dispatch_obj = LP_dispatch(pf, PTDF, batt, Pjk_lim, Gmax, cgn, clin, cdr, v_base, dvdp, storage, vmin, vmax)
        x, m, LMP, Ain = dispatch_obj.PTDF_LP_OPF(demandProfile, Pjk_0, v_0, Pg_0, PDR_0)
    
    #Create plot object
    plot_obj = plottingDispatch(output_dir, iterName, pointsInTime, script_path, vmin, vmin, PTDF=PTDF, dispatchType=dispatchType)
    
    # extract dispatch results
    Pg, Pdr, Pij, Pchar, Pdis, E = plot_obj.extractResults(x=x, DR=DR, Storage=storage, batt=batt)

    ################
    #operation costs
    ################
    Pdr_1d = np.reshape(Pdr, np.size(Pdr), order="F")
    costPdr = cdr @ Pdr_1d.T
    
    # extract LMP results
    LMP_Pg, LMP_Pdr, LMP_Pij, LMP_Pchar, LMP_Pdis, LMP_E = plot_obj.extractLMP(LMP, DR, storage, batt)

    lnodes = np.where(demandProfilei)[0]    
    # Define the output as Pandas DataFrame
    outGen, outDR, outPchar, outPdis, outLMP = create_output(Pg, Pdr, Pchar, Pdis, LMP_Pg, loadNames, PTDF, lnodes=lnodes, Snodes=Snodes, PVnodes=PVnodes, timeVec=v_0.columns, PV=PV, storage=storage)
    
    if plot:
        
        # plot demand response
        if DR:
            plot_obj.plot_DemandResponse(outDR)
        
        # plot Dispatch
        dfPg = pd.DataFrame(Pg, PTDF.columns, v_0.columns)
        plot_obj.plot_Dispatch(dfPg)
        
        # plot Storage
        if storage:
            plot_obj.plot_storage(E, batt, gCost[0,:])
        
        ## ploting LMPs
        #plot_obj.plot_LMP(outLMP, 'gen')
        
        # if LMP_Pdr is not None:
        #     LMP_Pdr = pd.DataFrame(LMP_Pdr, PTDF.columns, v_0.columns)
        #     plot_obj.plot_LMP(LMP_Pdr,'DR')
        
        # if LMP_Pchar is not None:
        #     LMP_Pchar = pd.DataFrame(LMP_Pchar, PTDF.columns[Snodes], v_0.columns)
        #     plot_obj.plot_LMP(LMP_Pchar,'Scharge')

        # if LMP_Pdis is not None:
        #     LMP_Pdis = pd.DataFrame(LMP_Pdis, PTDF.columns[Snodes], v_0.columns)
        #     plot_obj.plot_LMP(LMP_Pdis,'Sdischarge')

        # if LMP_E is not None:
        #     LMP_E = pd.DataFrame(LMP_E, PTDF.columns[Snodes], v_0.columns)
        #     plot_obj.plot_LMP(LMP_E,'E')

    return outGen, outDR, outPchar, outPdis, outLMP, costPdr, cgn, m.objVal
