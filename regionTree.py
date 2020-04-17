#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 19:41:53 2017

@author: bobakm
"""
import numpy as np

class regionTree():
    def __init__(self):
        self.root = regionTreeNode([],None,None)
        self.uncheckedNodes = [self.root]
        self.leafs = [self.root]
    def setObs(self,trajEndPointsIDX, mergedDataIDX):
        self.root.setObs(trajEndPointsIDX, mergedDataIDX)
    def getNumUncheckedNodes(self):
        return len(self.uncheckedNodes)
    def removeUncheckedNode(self):
        return self.uncheckedNodes.pop(0)
    def splitLeafNode(self, leafIDX):
        parentNode = self.leafs.pop(leafIDX)
        #Make left node and make right node
        #put left and right node in unchecked nodes
        #and also in leafs
        parentPath = parentNode.getPath()
        #print parentPath
        leftAppend = (parentNode.bestSplitVar,parentNode.bestSplitVal,True, parentNode.cat)
        rightAppend = (parentNode.bestSplitVar,parentNode.bestSplitVal,False, parentNode.cat)
        leftPath = parentPath[:]
        rightPath = parentPath[:]
        leftPath.append(leftAppend)
        rightPath.append(rightAppend)
        #print leftPath
        #print rightPath
        leftChild = regionTreeNode(leftPath, self, parentNode)
        rightChild = regionTreeNode(rightPath, self, parentNode)
        childVals = parentNode.bestChildVals
        leftChild.setGamma(childVals[0])
        rightChild.setGamma(childVals[1])
        leftChild.setObs(parentNode.childLIDX[0],parentNode.childLIDX[1])
        rightChild.setObs(parentNode.childRIDX[0],parentNode.childRIDX[1]) 
        leftChild.setUkVk(parentNode.lUK, parentNode.lVK)
        rightChild.setUkVk(parentNode.rUK, parentNode.rVK)
        self.leafs.append(leftChild)
        self.leafs.append(rightChild)
        self.uncheckedNodes.append(leftChild)
        self.uncheckedNodes.append(rightChild)
        return (parentNode, parentNode.bestSplitVar, parentNode.score)
    def getPredictedValues(self, data):
        values = np.zeros(data.shape[0])
        for node in self.leafs:
            path = node.getPath()
            #split:(var, val, leftOrRight, cat) 
            #IF Cat == TRUE
            # left is == Cat, right is != cat
            # if CAT == False
            #left is <, right is >=
            lofCond = []
            minIDX = -1
            minLength = np.Inf
            for j in range(0, len(path)):
                split = path[j]
            #for split in path:
                var, val, left, cat = split
                if(left):
                    if(cat):
                        condIDX = np.where(data[:,var] == val)[0]
                    else:
                        condIDX = np.where(data[:,var] <= val)[0]
                else:
                    if(cat):
                        condIDX = np.where(data[:,var] != val)[0]
                    else:
                        condIDX = np.where(data[:,var] > val)[0]
                if(condIDX.size < minLength):
                    minLength = condIDX.size
                    minIDX = j
                lofCond.append(condIDX)
            minCond = lofCond[minIDX]
            for cond in lofCond:
                minCond = np.intersect1d(minCond, cond)
            #minCond is now "in region"
            values[minCond] = node.getGamma()
        return values
    def getIntegrationValues(self, mergedData):
        #For each row of merged Data 
        #determine which leaf node it is in
        #return value of that leaf node
        values = np.zeros(mergedData.shape[0])
        for node in self.leafs:
            path = node.getPath()
            #FOR INTEGRATION left is strictly < even if building the tree it was <=
            #split:(var, val, leftOrRight, cat) 
            #IF Cat == TRUE
            # left is == Cat, right is != cat
            # if CAT == False
            #left is <, right is >=
            #path = [Split_0, Split_1,..., Split_m]
            #Because the trajectory is left continuous
            #in Traj End it needs to be <= to include the final value at the split point
            #However, in merged Data it needs to be strict <
            #So that, in numeric integration, you are adding up Tothe split point not
            #from dt Past it.       
            lofCond = []
            minIDX = -1
            minLength = np.Inf
            for j in range(0, len(path)):
                split = path[j]
            #for split in path:
                var, val, left, cat = split
                    #Because trajectory is left continuous, integration
                #uses the first to penultimate time-covariate points
                #and sum-products them with dt over the time interval.
                #Therefore we are strictly < in this, despite tree being built <=
                #By construction.
                #Update 19 JULY 2017 - This logic is only applies to time dimension
                if(left):
                    if(cat):
                        condIDX = np.where(mergedData[:,var] == val)[0]
                    else:
                        if(var == 0):
                            condIDX = np.where(mergedData[:,var] < val)[0]
                        else:
                            condIDX = np.where(mergedData[:,var] <= val)[0]
                else:
                    if(cat):
                        condIDX = np.where(mergedData[:,var] != val)[0]
                    else:
                        if(var == 0):
                            condIDX = np.where(mergedData[:,var] >= val)[0]
                        else:
                            condIDX = np.where(mergedData[:,var] > val)[0]
                if(condIDX.size < minLength):
                    minLength = condIDX.size
                    minIDX = j
                lofCond.append(condIDX)
            minCond = lofCond[minIDX]
            for cond in lofCond:
                minCond = np.intersect1d(minCond, cond)
            #minCond is now "in region"
            values[minCond] = node.getGamma()
        return values
    def cleanTree(self):
        self.uncheckedNodes = None
        for node in self.leafs:
            node.cleanNode()

class regionTreeNode():
    def __init__(self, path, root, parent):
        self.path = path
        self.left = None
        self.right = None
        self.uk = None
        self.vk = None
        self.score = None
        if(root == None):
            self.root = self
        else:
            self.root = root
        self.parent = parent
        self.trajEndPointsIDX = None
        self.mergedDataIDX = None
        self.bestSplitVar = None
        self.bestSplitVal = None
        self.bestChildVals = None
        self.gamma = 0
        self.childLIDX = None
        self.childRIDX = None
        self.lUK = None
        self.lVK = None
        self.rUk = None
        self.rVK = None
        self.cat = None
    def setObs(self, trajEndPointsIDX, mergedDataIDX):
        self.trajEndPointsIDX = trajEndPointsIDX
        self.mergedDataIDX = mergedDataIDX
    def cleanNode(self):
        self.trajEndPointsIDX = None
        self.mergedDataIDX = None
        self.bestSplitVar = None
        self.bestSplitVal = None
        self.score = None
        self.bestChildVals = None
        self.childLIDX = None
        self.childRIDX = None
        self.lUK = None
        self.rUK = None
        self.lVK = None
        self.rVK = None
        self.cat = None
        self.uk = None
        self.vk = None
        self.parent = None
    def getObs(self):
        return (self.trajEndPointsIDX, self.mergedDataIDX)
    def getPath(self):
        return self.path
    def setSplitCand(self, splitCand):
        if(splitCand == None):
            self.score = 1
        else:
            self.bestSplitVar = splitCand[0]
            self.bestSplitVal = splitCand[1]
            self.score = splitCand[2]
            self.bestChildVals = (splitCand[3], splitCand[4])
            self.childLIDX = (splitCand[5], splitCand[7])
            self.childRIDX = (splitCand[6], splitCand[8])
            self.lUK = splitCand[9]
            self.rUK = splitCand[10]
            self.lVK = splitCand[11]
            self.rVK = splitCand[12]
            self.cat = splitCand[13]
#        (col, splitPoint, score, np.log(lUK/lVK), np.log(rUK/rVK),
#                                         childLTrajIDX, childRTrajIDX, childLMergedIDX, childRMergedIDX, cat)
    def setScore(self, uk, vk, score):
        self.uk = uk
        self.vk = vk
        self.score = score
    def setUkVk(self, uk, vk):
        self.uk = uk
        self.vk = vk
    def getScore(self):
        return (self.uk, self.vk, self.score)
    def setGamma(self, gamma):
        self.gamma = gamma
    def getGamma(self):
        return self.gamma
    def deleteNode(self):
        self.path = None
        self.left = None
        self.right = None
        self.uk = None
        self.vk = None
        self.score = None
        self.root = None
        self.parent = None
        self.trajEndPointsIDX = None
        self.mergedDataIDX = None
        self.bestSplitVar = None
        self.bestSplitVal = None
        self.bestChildVals = None
        self.gamma = None
        self.childLIDX = None
        self.childRIDX = None
        self.lUK = None
        self.lVK = None
        self.rUk = None
        self.rVK = None
        self.cat = None
        
        