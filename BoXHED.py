#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: BoXHED
"""

import numpy as np
import scipy as sp
import regionTree as rt
#import time

def mergeSorted(a,b):
    absort = np.concatenate((a,b), axis=0)
    return(np.sort(absort))


def prepData(traj, tpart):
    #Fill forward in time
    #Add mid point between last interp point and traj End
    lastPt = traj[-1,:]
    penultPt = traj[-2,:]
    newTraj = traj[0:-1,:]
    midPtTime = (lastPt[0]+penultPt[0])/2
    midPt = np.copy(lastPt)
    midPt[0] = midPtTime
    traj = np.concatenate((newTraj, midPt.reshape((1,midPt.size)), lastPt.reshape((1, lastPt.size))))
    #END Fix for VK and UK on 29 NOV 2017
    import bisect
    trajEnd = traj[-1,:]
    timeCol = traj[:,0]
    tpart = mergeSorted(timeCol, np.array(tpart))
    tpart = tpart[tpart <= timeCol[-1]]
    tpart = tpart[tpart >= timeCol[0]]
    interp = np.zeros((len(tpart),traj.shape[1]))
    interp[:,0] = tpart
    for t in timeCol:
        partIdx = bisect.bisect_left(tpart, t)
        idxRow = np.where(timeCol == t)[0][0]
        trajRow = traj[idxRow,1:traj.shape[1]]
        numRepeats = (len(tpart) - partIdx)
        interp[partIdx:interp.shape[0],1:interp.shape[1]] = np.repeat(np.array([trajRow]),numRepeats, axis=0)
    interp = interp[np.unique(interp[:,0], return_index=True)[1]]#Remove duplicates    
    dt = np.diff(interp[:,0])
    dt = dt.reshape((dt.size,1))
    interp = interp[0:-1,:]
    #interp = np.hstack((interp,dt))
    return (interp, dt, trajEnd)

def getData(lotraj, tpart):
    #TODO - Optimize list size
    trajEndPoint = np.zeros((len(lotraj), lotraj[0].shape[1]))
    #mergedData = np.array([]) 
    #mergedDT = np.array([])
    i = 0
    traj = lotraj[i]
    interp, dt, trajEnd = prepData(traj, tpart)
    #Fix for VK > 0 UK == 0 on 29 NOV 2017
    #Add mid point between last interp point and traj End
    #With value at Traj End
    #Take last DT split in half and double - 
    #This is now to modify the mergedDT - but does not affect TPart
    #lastdt = interp[-1]
    #dttemp = dt[0:-1]
    #dt = np.concatenate((dttemp, np.array(dt[-1]/2).reshape((1,1)),np.array(dt[-1]/2).reshape((1,1))), axis = 0)
    #midTrajEnd = np.copy(trajEnd)
    #midTrajEnd[0] = (lastdt[0] + trajEnd[0])/2
    #interp=np.concatenate((interp, midTrajEnd.reshape((1, midTrajEnd.size))))
    #END Fix for VK and UK on 29 NOV 2017
    trajEndPoint[i,:] = trajEnd
    mergedData = interp
    mergedDT = dt
    for i in range(1, len(lotraj)):
        traj = lotraj[i]
        interp, dt, trajEnd = prepData(traj, tpart)
        trajEndPoint[i,:] = trajEnd
        mergedData = np.vstack((mergedData,interp))
        mergedDT = np.vstack((mergedDT, dt))
    return (trajEndPoint, mergedData, mergedDT)


def getTimePartition(lotraj, numtimepartitions):
    mergedTimeCol = np.array([])
    for traj in lotraj:
        mergedTimeCol = np.append(mergedTimeCol, traj[:,0])
    percentiles = np.linspace(0, 100, numtimepartitions+1)#[10,20,30,40,50,60,70,80,90]    
    splitPoints = np.unique(np.percentile(mergedTimeCol, percentiles) )
    return splitPoints[1:-1]#Remove Time 0 and Max Time

def getVarPartitions(trajEndPoints, mergedData, numvarpartitions, cat):
    nCols = mergedData.shape[1]
    percentiles = np.linspace(0, 100, numvarpartitions+1)#[10,20,30,40,50,60,70,80,90]    
    percentiles = percentiles[1:]    
    splits = []
    for n in range(1, nCols): #Not considering Time
        colData = np.append(mergedData[:,n], trajEndPoints[:,n])        
        if((cat is not None) and (n in cat)):
            splits.append(np.unique(colData))
        else:
            splits.append(np.percentile(colData, percentiles))
    return splits

def buildTree(trajEndPoints, mergedData, mergedDT, runningF, tpart, globalSplits, perNode, delta, maxsplits, cat, varImp):
    numSplits = 0
    tree = rt.regionTree()
    trajEndPointsIDX = np.arange(0, trajEndPoints.shape[0])
    mergedDataIDX = np.arange(0, mergedData.shape[0])
    tree.setObs(trajEndPointsIDX, mergedDataIDX) #If we do row subsampling this will change
    rootVK = np.sum(delta) #If we do row subsampling this will change
    rootUK = np.inner(np.exp(runningF), mergedDT.flatten())
    rootScore = 0
    tree.root.setScore(rootUK, rootVK, rootScore)#Need to put actual uk, vk, score
    tree.root.setGamma(None)
    while(numSplits < maxsplits):
        #print numSplits
        numUncheckedNodes = tree.getNumUncheckedNodes()
        for n in range(0, numUncheckedNodes):
            node = tree.removeUncheckedNode()
            splitCand = findBestSplit(node, node.getObs(), trajEndPoints, mergedData, mergedDT,runningF, tpart, delta, cat, globalSplits = globalSplits, perNode=perNode)
            # splitCand = None if no good split point OR
            #(col, splitPoint, score, np.log(lUK/lVK), np.log(rUK/rVK),
            #                             childLTrajIDX, childRTrajIDX, childLMergedIDX, childRMergedIDX, cat) #True if col is categorical, #False otherwise            
            node.setSplitCand(splitCand)
        bestLeafNodeIdx = None
        bestLeafScore = 0
            #Go through all leaf nodes, make sure at least one leaf has
            #node.score  < 0
        for n in range(0, len(tree.leafs)):
            currNode = tree.leafs[n]
            if(currNode.score < bestLeafScore):
                bestLeafScore = currNode.score
                bestLeafNodeIdx = n
        if(bestLeafNodeIdx != None):
            splitNode, splitVar, splitScore = tree.splitLeafNode(bestLeafNodeIdx) 
            varImp[splitVar] -= splitScore
            numSplits += 1
            splitNode.deleteNode() #TODO : Make sure this really sets reference count to 0
            del splitNode
        else:
            break
    tree.cleanTree()
    return (tree, varImp)


            
def findBestSplit(parentNode, tupleOfObs, trajEndPoints, mergedData, mergedDT, runningF, tpart, delta, cat, globalSplits = None, perNode=True):
    nodeTrajEndPointIDX, nodeMergedIDX = tupleOfObs
    numCols = mergedData.shape[1]
    if(not(perNode)):
        splits = globalSplits        
        #splits = getVarPartitions(trajEndPoints, mergedData)
    else:
        splits = getVarPartitions(trajEndPoints[nodeTrajEndPointIDX,:], mergedData[nodeMergedIDX,:], numvarpartitions, cat)##BUGGY - NEED TO UPDATE
    splitCandidate = None
    splitCandScore = 0
    for col in range(0, numCols):
        #loop through each column/variable
        #Find Splits IN variable
        if(col == 0):
            curSplit = tpart
        else:
            curSplit = splits[col - 1]
        for splitPoint in curSplit:
            if((cat is not None) and (col in cat)):
                #Ideally - Binary or Categorical
                #likely - just categorical
                lTrajIDX = np.where(trajEndPoints[:,col] == splitPoint)[0] 
                rTrajIDX = np.setdiff1d(np.arange(0, trajEndPoints.shape[0]), lTrajIDX)
                childLTrajIDX = np.intersect1d(nodeTrajEndPointIDX, lTrajIDX, assume_unique=True)
                childRTrajIDX = np.intersect1d(nodeTrajEndPointIDX, rTrajIDX, assume_unique=True)
                lMergedDataIDX = np.where(mergedData[:,col] == splitPoint)[0] 
                rMergedDataIDX = np.setdiff1d(np.arange(0, mergedData.shape[0]), lMergedDataIDX)
                childLMergedIDX = np.intersect1d(nodeMergedIDX, lMergedDataIDX, assume_unique=True)
                childRMergedIDX = np.intersect1d(nodeMergedIDX, rMergedDataIDX, assume_unique=True)
            else:
                #CONTINUOUS VARIABLES
                #loop through the partitions
                lTrajIDX = np.where(trajEndPoints[:,col] <= splitPoint)[0] 
                rTrajIDX = np.setdiff1d(np.arange(0, trajEndPoints.shape[0]), lTrajIDX)
                childLTrajIDX = np.intersect1d(nodeTrajEndPointIDX, lTrajIDX, assume_unique=True)
                childRTrajIDX = np.intersect1d(nodeTrajEndPointIDX, rTrajIDX, assume_unique=True)
                #Because trajectory is left continuous, integration
                #uses the first to penultimate time-covariate points
                #and sum-products them with dt over the time interval.
                #Therefore we are strictly < in this, despite tree being built <=
                #By construction. so <= for traj end points, < for all intermediate points
                #Update 19 July 2017: This logic is only applies to time dimension
                if(col == 0):
                    lMergedDataIDX = np.where(mergedData[:,col] < splitPoint)[0] 
                else:
                    lMergedDataIDX = np.where(mergedData[:,col] <= splitPoint)[0]
                rMergedDataIDX = np.setdiff1d(np.arange(0, mergedData.shape[0]), lMergedDataIDX)
                childLMergedIDX = np.intersect1d(nodeMergedIDX, lMergedDataIDX, assume_unique=True)
                childRMergedIDX = np.intersect1d(nodeMergedIDX, rMergedDataIDX, assume_unique=True)
                #Now Calculate UK and VK for each child
            pUK, pVK, pScore = parentNode.getScore()
            if(pVK == 1):
                break
            lVK = int(np.sum(delta[childLTrajIDX]))
            rVK = pVK - lVK#np.sum(DELTA[childRTrajIDX])
            if(rVK <= 0):
                break #Since movingfromleft to right, if rVK is 0, it will
                    #Continue to be 0 with further split points to the right
                    #Because the left region is only getting bigger
            elif(lVK > 0):
                lUK = runningF[childLMergedIDX]#calcF(LOTREES, mergedData, childLMergedIDX)
                lUK = np.inner(np.exp(lUK), mergedDT[childLMergedIDX].flatten())
                rUK = pUK-lUK#np.inner(np.exp(rUK), mergedDT[childRMergedIDX].flatten())
                #calc SplitScore
                #FIX 11/30/2017
                if((lUK == 0) or (rUK == 0)):
                    break
                #End Fix 11/30/2017
                score = lVK*(1 + np.log(lUK/lVK)) + rVK*(1 + np.log(rUK/rVK)) - pVK*(1 + np.log(pUK/pVK) )
                if(score < splitCandScore):
                    splitCandScore = score
                    splitCandidate = (col, splitPoint, score, np.log(lUK/lVK), np.log(rUK/rVK),
                                         childLTrajIDX, childRTrajIDX, childLMergedIDX, childRMergedIDX,
                                         lUK, rUK, lVK, rVK, ((cat is not None) and (col in cat))) #SplitVar, SplitVal, SplitScore
    return splitCandidate

class boostedTrees():
    def __init__(self, LOTREES, F0, MAXSPLITS, NUMTIMEPARTITIONS, NUMVARPARTITIONS, SHRINK, NTREES, varImp):
        self.lotrees = LOTREES
        self.F0 = F0
        self.maxsplits = MAXSPLITS
        self.numtrees = NTREES
        self.shrink = SHRINK
        self.numtimepartitions = NUMTIMEPARTITIONS
        self.numvarpartitions = NUMVARPARTITIONS
        self.varImp = varImp


def dataPrep(delta, lotraj, numtimepartitions, numvarpartitions, cat):
    TOTALTIME = 0.0
    for traj in lotraj:
        TOTALTIME += traj[-1,0] - traj[0,0]
    tpart = getTimePartition(lotraj, numtimepartitions)
    F0 = np.log(np.sum(delta)/TOTALTIME)
    #start = time.time()
    trajEndPoints, mergedData, mergedDT = getData(lotraj, tpart)
    #end = time.time()
    #print end - start
    globalSplits = getVarPartitions(trajEndPoints, mergedData, numvarpartitions, cat)
    return (trajEndPoints, mergedData, mergedDT, F0, tpart, globalSplits)

def treeEnsemble(delta, F0, trajEndPoints, mergedData, mergedDT, tpart, globalSplits, cat, maxsplits=2, numtrees=100, numtimepartitions=20, numvarpartitions=20, shrink=0.1, GlobalPartition = True, verbose=0):
    varImp = {}
    for i in range(0, trajEndPoints.shape[1]):
        varImp[i] = 0
    perNode = not(GlobalPartition)
    lotrees = []
    runningF = F0*np.ones(mergedData.shape[0])
    runningF_endpoint = F0*np.ones(trajEndPoints.shape[0])
    while(len(lotrees) < numtrees):
        newTree, varImp = buildTree(trajEndPoints, mergedData, mergedDT, runningF, tpart, globalSplits, perNode, delta, maxsplits, cat, varImp)
        if(len(newTree.leafs) == 1):
        #Only a root node
        #it means no splits help any more
        #which means no new trees help
        #Only true if not subsampling rows
    #Tree should never split nodes if likelihood increases***
            break
        else:
            lotrees.append(newTree)
            values = newTree.getIntegrationValues(mergedData)
            endValues = newTree.getPredictedValues(trajEndPoints)
            runningF = runningF - shrink*values        
            runningF_endpoint = runningF_endpoint - shrink*endValues
    estimator = boostedTrees(lotrees, F0, maxsplits, numtimepartitions, numvarpartitions, shrink, len(lotrees), varImp)
    return estimator

def BoXHED(delta, lotraj, maxsplits=2, numtrees=100, numtimepartitions=10, numvarpartitions=10, shrink=0.1, GlobalPartition = True, verbose=0, cat=None):
    trajEndPoints, mergedData, mergedDT, F0, tpart, globalSplits = dataPrep(delta, lotraj, numtimepartitions, numvarpartitions, cat)
    return treeEnsemble(delta, F0, trajEndPoints, mergedData, mergedDT, tpart, globalSplits, cat, maxsplits, numtrees, numtimepartitions, numvarpartitions, shrink, GlobalPartition, verbose)

def predict(estimator, newdata, ntreelimit = np.Inf):
    predF = estimator.F0*np.ones(newdata.shape[0])
    if(ntreelimit > estimator.numtrees):
        ntreelimit = estimator.numtrees
    for tridx in range(0, ntreelimit):
        tr = estimator.lotrees[tridx]
        Fvalues = tr.getPredictedValues(newdata)
        predF = predF - estimator.shrink*Fvalues
    return predF

def logLik(estimator, delta, trajEndPoints, mergedData, mergedDT, candidatenumtrees, ntreelimit = np.Inf):
    predF = estimator.F0*np.ones(mergedData.shape[0])
    predF_endpoint = estimator.F0*np.ones(trajEndPoints.shape[0])
    logLiks = np.zeros(len(candidatenumtrees))
    if(ntreelimit > estimator.numtrees):
        ntreelimit = estimator.numtrees
    if(0 in candidatenumtrees):
        logLiks[candidatenumtrees.index(0)] = (np.inner(np.exp(predF), mergedDT.flatten()) - np.inner(predF_endpoint, delta))
    for tridx in range(0, ntreelimit):
        tr = estimator.lotrees[tridx]
        values = tr.getIntegrationValues(mergedData)        
        endvalues = tr.getPredictedValues(trajEndPoints)
        predF = predF - estimator.shrink*values
        predF_endpoint = predF_endpoint - estimator.shrink*endvalues
        if((tridx+1) in candidatenumtrees):
            logLiks[candidatenumtrees.index(tridx+1)] = (np.inner(np.exp(predF), mergedDT.flatten()) - np.inner(predF_endpoint, delta))
    return logLiks



#PLOT MIGHT BREAK IN Categorical     
def plot(estimator, varIndices, var1range, var2range, ntreelimit = np.Inf, plotPoints=500, clip=False, clipValue = None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    if(len(varIndices) > 2):
        #Throw Error
        print('Plot Error: varIndices length exceeds 2')
        return #for now just return
    var1 = np.linspace(var1range[0],var1range[1],plotPoints)
    var2 = np.linspace(var2range[0],var2range[1],plotPoints)
    var1v, var2v = np.meshgrid(var1,var2)
    newdata = np.hstack((var1v.reshape((var1v.size,1)), var2v.reshape((var2v.size, 1))))
    predF = predict(estimator, newdata, ntreelimit)    
    predLambda = np.exp(predF).reshape(var1v.shape)
    if(clip):
        predClipped = np.clip(predLambda, 0, clipValue)
    else:
        predClipped = predLambda
    fig = plt.figure()
    ax= fig.gca(projection='3d')
    surf = ax.plot_wireframe(var1v, var2v, predClipped, rstride=1, cstride=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Hazard')
    plt.show()
    return surf

def plotData(estimator, newdata, var1, var2, ntreelimit = np.Inf, rstride = 10, cstride = 10, clip=False, clipValue = None):
    import matplotlib.pyplot as plt
    #from mpl_toolkits.mplot3d import Axes3D
    #from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    predF = predict(estimator, newdata, ntreelimit)    
    predLambda = np.exp(predF).reshape(var1.shape)
    if(clip):
        predClipped = np.clip(predLambda, 0, clipValue)
    else:
        predClipped = predLambda
    fig = plt.figure()
    ax= fig.gca(projection='3d')
    #surf = ax.plot_wireframe(var1v, var2v, predClipped, rstride=rstride, cstride=cstride)
    surf = ax.plot_wireframe(newdata[:, var1], newdata[:, var2], predClipped, rstride=rstride, cstride=cstride)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Hazard')
    plt.show()
    return surf



def cv(delta, lotraj, nfolds = 5, maxsplits=[2,3,4], numtrees=[10,50,100,200], numtimepartitions=50, numvarpartitions=50, shrink=0.1, GlobalPartition = True, verbose=0, cat = None):
    maxTrees = np.max(numtrees)
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=nfolds)
    X = np.array(lotraj)
    #create 5 folds
    trainIdxFolds = []
    testIdxFolds = []
    for train_index, test_index in skf.split(X, delta):
        trainIdxFolds.append(train_index)
        testIdxFolds.append(test_index)     
    logLiksTable = np.zeros((len(maxsplits),len(numtrees)))
    for f in range(0, nfolds):
    #For each fold
        #For Future - If Cross-validation num time partitions
        #and num var partitions - the first two loops would have to be out here
        trainData = dataPrep(delta[trainIdxFolds[f]], X[trainIdxFolds[f]], numtimepartitions, numvarpartitions, cat)
        testData = dataPrep(delta[testIdxFolds[f]], X[testIdxFolds[f]], numtimepartitions, numvarpartitions, cat)
        #trainData = (trajEndPoints, mergedData, mergedDT, F0, tpart, globalSplits)    
        #grid search params
            #Call treeEnsemble
            #Fit estimator
            #Check likelihood
        for split in maxsplits:
            estimator = treeEnsemble(delta[trainIdxFolds[f]], trainData[3], trainData[0], trainData[1],trainData[2], trainData[4], trainData[5], cat, split, maxTrees, numtimepartitions, numvarpartitions, shrink, GlobalPartition, verbose)
            logLiksTable[maxsplits.index(split),:] += logLik(estimator, delta[testIdxFolds[f]], testData[0], testData[1], testData[2], numtrees, ntreelimit = maxTrees)
    logLiksTable = logLiksTable/float(nfolds)
    return(logLiksTable)
    #Return params that minimize likelihood
    #Also return the best fit estimator - CHECK THIS FOR MEMORY ISSUES
    #IF TOO MUCH MEMORY - SIMPLY RETURN THE PARAMETERS THAT RESULTED IN BEST
    
def variableImportance(estimator, colnames=None):
    #Colnames is either None, or an np.array of column names (including "time")
    varImp = estimator.varImp
    varImp = {k: v/max(varImp.values()) for k,v in varImp.items()}
    #This normalizes varImp
    orderedVars = sorted(varImp, key=lambda k: varImp[k], reverse = True)
    orderedVals = [varImp[k] for k in orderedVars]
    if(colnames is not None):
        varNames = colnames[orderedVars]
    else:
        varNames = orderedVars
    return{'varIndex':varNames, 'Importance':orderedVals}
