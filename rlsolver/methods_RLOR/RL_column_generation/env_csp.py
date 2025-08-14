"""
Environments are needed to be registered in 'gym.env' according to the specification
"""

import gym
import dgl
import torch
import random
import numpy as np
from gurobipy import *
from scipy.sparse import coo_matrix
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity

class CuttingStockProblem(gym.Env):

    def __init__(self,
                 instance_attrs='Easy',
                 PoolSearchMode=2,
                 PoolSolutions=10,
                 numSelections=5,
                 numTestInstance=None,
                 rootPath=None
                 ):

        self.episode = 0
        # instance attributes
        self.instance_attrs = instance_attrs
        if self.instance_attrs == 'Easy':
            self.n_list = [50, 75, 100, 120]
            self.c_list = [50]
            self.w_min_list = [0.1, 0.2]
            self.w_max_list = [0.7, 0.8]
        elif self.instance_attrs == 'Normal':
            self.n_list = [75, 100, 120, 150]
            self.c_list = [100]
            self.w_min_list = [0.1, 0.2]
            self.w_max_list = [0.7, 0.8]
        elif self.instance_attrs == 'Hard':
            self.n_list = [125, 150]
            self.c_list = [200]
            self.w_min_list = [0.1, 0.2]
            self.w_max_list = [0.7, 0.8]

        self.PoolSearchMode = PoolSearchMode  # SolutionPool mode
        self.PoolSolutions = PoolSolutions  # size of PoolSolutions
        self.numSelections = numSelections  # num of selections
        self.indexSet = list(combinations(
            [i for i in range(self.PoolSolutions)], self.numSelections))  # action index
        self.numTestInstance = numTestInstance
        self.rootPath = rootPath
        self.readIndex = 0

    def step(self, action):

        self.num_action += 1
        if not isinstance(action, list):  # e.g., action = 2
            _ = [0 for i in range(self.PoolSolutions)]
            action = self.indexSet[action]
            for i in action:
                _[i] = 1
            action = _
        self.columnPool = [self.columnPool[i] for i, y in enumerate(action) if y == 1]
        self.columnIndex = [i for i, y in enumerate(action) if y == 1]

        # compute similarity
        self.distReward = [self.cosineDist[i[0], i[1]].item() for i in combinations(self.columnIndex, 2)]
        self.distReward = sum(self.distReward)

        # add columns
        for columnCoeff in self.columnPool:
            column = Column(columnCoeff, self.mainModel.getConstrs())
            self.mainModel.addVar(obj=1.0, vtype='C', column=column)

        # solve updated RMP
        self.mainModel.optimize()
        # get object and duals
        self.Dual = self.mainModel.getAttr(GRB.Attr.Pi, self.mainModel.getConstrs())
        self.objValue.append(self.mainModel.objVal)
        self.objReward = (self.objValue[-2] - self.objValue[-1]) / self.objValue[0]

        # update constrCoeff
        self.newCoeff = torch.tensor(self.columnPool).T
        self.constrCoeff = torch.column_stack((self.constrCoeff, self.newCoeff))
        # update constraint features
        self.Dual = self.Dual
        self.rowConnectivity = torch.sum(self.constrCoeff > 0, dim=1)
        self.Slack = -torch.tensor(self.mainModel.getAttr('Slack', self.mainModel.getConstrs()))
        self.featRow[:, 0] = torch.tensor(self.Dual)
        self.featRow[:, 1] = self.rowConnectivity
        self.featRow[:, 2] = self.Slack
        # update column features
        self.numNewCol = self.newCoeff.shape[1]
        self.reducedCost = torch.tensor(self.mainModel.getAttr('RC', self.mainModel.getVars()))
        self.colConnectivity = torch.cat((self.colConnectivity, torch.sum(self.constrCoeff[:, -self.numNewCol:] > 0, dim=0)))
        self.solutionValue = torch.tensor(self.mainModel.getAttr('X', self.mainModel.getVars()))
        self.Waste = torch.cat((self.Waste, 1 - torch.mm(self.lengthRatio.unsqueeze(0), self.constrCoeff[:, -self.numNewCol:]).reshape([-1])))
        self.lastInBasis = torch.cat((self.InBasis, torch.zeros(self.numNewCol)))
        self.lastOutBasis = torch.cat((self.OutBasis, torch.ones(self.numNewCol)))
        self.InBasis = (torch.tensor(self.mainModel.getAttr('VBasis', self.mainModel.getVars())) == 0).float()
        self.OutBasis = (torch.tensor(self.mainModel.getAttr('VBasis', self.mainModel.getVars())) != 0).float()
        self.numInBasis = torch.cat((self.numInBasis, torch.zeros(self.numNewCol))) + self.InBasis
        self.numOutBasis = torch.cat((self.numOutBasis, torch.zeros(self.numNewCol))) + self.OutBasis
        self.leftBasis = self.lastInBasis * self.OutBasis
        self.enterBasis = self.lastOutBasis * self.InBasis
        self.isCandidate = torch.zeros_like(self.reducedCost)
        self.featCol = torch.cat((self.reducedCost.unsqueeze(-1),
                                  self.colConnectivity.unsqueeze(-1),
                                  self.solutionValue.unsqueeze(-1),
                                  self.Waste.unsqueeze(-1),
                                  self.InBasis.unsqueeze(-1),
                                  self.OutBasis.unsqueeze(-1),
                                  self.numInBasis.unsqueeze(-1),
                                  self.numOutBasis.unsqueeze(-1),
                                  self.leftBasis.unsqueeze(-1),
                                  self.enterBasis.unsqueeze(-1),
                                  self.isCandidate.unsqueeze(-1)), dim=-1)

        # update PP
        for i in range(len(self.typesDemand)):
            self.c[i].obj = self.Dual[i]
        self.subModel.Params.PoolSearchMode = self.PoolSearchMode
        self.subModel.Params.PoolSolutions = self.PoolSolutions
        # solve PP
        self.subModel.setAttr(GRB.Attr.ModelSense, -1)
        self.subModel.optimize()

        # reward
        self.reward = 0.02 * self.distReward + 300 * self.objReward - 1
        self.iterNum += 1

        # candidate column pool
        self.columnPool = []
        self.reducedCost_candidate = []
        for i in range(self.subModel.SolCount):
            self.subModel.Params.SolutionNumber = i
            self.columnCoeff = [round(x) for x in self.subModel.getAttr('Xn', self.subModel.getVars())]
            self.columnPool.append(self.columnCoeff)
            self.reducedCost_candidate.append(self.subModel.PoolObjVal)

        self.candidateCoeff = torch.tensor(self.columnPool).T
        self.numRow = self.constrCoeff.shape[0]
        self.numCol = self.constrCoeff.shape[1]
        self.numCandidate = self.candidateCoeff.shape[1]
        # adj matrix
        self.adjMatrix = torch.cat(
            [torch.cat([torch.zeros(self.numRow, self.numRow), self.constrCoeff, self.candidateCoeff], dim=1),
             torch.cat([self.constrCoeff.T, torch.zeros(self.numCol, self.numCol),
                        torch.zeros(self.numCol, self.numCandidate)], dim=1),
             torch.cat([self.candidateCoeff.T, torch.zeros(self.numCandidate, self.numCol),
                        torch.zeros(self.numCandidate, self.numCandidate)],
                       dim=1)], dim=0)
        self.adjMatrix_coo = coo_matrix(self.adjMatrix)
        # draw_adjMatrix(self.adjMatrix_coo, self.numRow, self.numCol, self.numCandidate)
        # update candidate features
        self.reducedCost_candidate = 1 - torch.tensor(self.reducedCost_candidate)
        self.colConnectivity_candidate = torch.sum(self.candidateCoeff > 0, dim=0)
        self.solutionValue_candidate = torch.zeros_like(self.reducedCost_candidate)
        self.Waste_candidate = 1 - torch.mm(self.lengthRatio.unsqueeze(0), self.candidateCoeff.float()).squeeze()
        self.InBasis_candidate = torch.zeros_like(self.reducedCost_candidate)
        self.OutBasis_candidate = torch.zeros_like(self.reducedCost_candidate)
        self.numInBasis_candidate = torch.zeros_like(self.reducedCost_candidate)
        self.numOutBasis_candidate = torch.zeros_like(self.reducedCost_candidate)
        self.leftBasis_candidate = torch.zeros_like(self.reducedCost_candidate)
        self.enterBasis_candidate = torch.zeros_like(self.reducedCost_candidate)
        self.isCandidate_candidate = torch.ones_like(self.reducedCost_candidate)

        self.featCandidate = torch.cat((self.reducedCost_candidate.unsqueeze(-1),
                                        self.colConnectivity_candidate.unsqueeze(-1),
                                        self.solutionValue_candidate.unsqueeze(-1),
                                        self.Waste_candidate.unsqueeze(-1),
                                        self.InBasis_candidate.unsqueeze(-1),
                                        self.OutBasis_candidate.unsqueeze(-1),
                                        self.numInBasis_candidate.unsqueeze(-1),
                                        self.numOutBasis_candidate.unsqueeze(-1),
                                        self.leftBasis_candidate.unsqueeze(-1),
                                        self.enterBasis_candidate.unsqueeze(-1),
                                        self.isCandidate_candidate.unsqueeze(-1)), dim=-1)

        # bipartite graph (heterogeneous)
        self.edge = coo_matrix(torch.cat([self.constrCoeff, self.candidateCoeff], dim=1))
        self.graph_data = {
            ('col', 'edge', 'row'): (self.edge.col, self.edge.row),
            ('row', 'edge', 'col'): (self.edge.row, self.edge.col)
        }
        self.num_nodes_dict = {
            'row': self.numRow,
            'col': self.numCol + self.numCandidate,
        }
        # heterogeneous graph
        self.biGraph = dgl.heterograph(self.graph_data, self.num_nodes_dict)
        # attach features
        self.biGraph.nodes['row'].data['feat'] = self.featRow
        self.biGraph.nodes['col'].data['feat'] = torch.cat([self.featCol, self.featCandidate], dim=0)
        self.biGraph.edges[('col', 'edge', 'row')].data['weight'] = torch.tensor(self.edge.data).unsqueeze(-1)
        self.biGraph.edges[('row', 'edge', 'col')].data['weight'] = torch.tensor(self.edge.data).unsqueeze(-1)
        # draw_biGraph(self.biGraph, self.numRow, self.numCol, self.numCandidate)

        # similarity
        self.cosineDist = 1 - cosine_similarity(self.columnPool, self.columnPool)
        self.columnPool_Jaccard = [[index for index, row in enumerate(candidate) if row != 0] for candidate in
                                   self.columnPool]
        self.jaccardDist = torch.ones(self.PoolSolutions, self.PoolSolutions)
        for a, A in enumerate(self.columnPool_Jaccard):
            for b, B in enumerate(self.columnPool_Jaccard):
                lenA = len(A)
                lenB = len(B)
                lenAB = len([i for i in A if i in B])
                self.jaccardDist[a][b] = 1 - lenAB / (lenA + lenB - lenAB)
        self.cosineDist = torch.tensor(self.cosineDist).float()
        self.jaccardDist = torch.tensor(self.jaccardDist).float()

        # complete graph
        self.fcMatrix = coo_matrix(torch.ones(self.PoolSolutions, self.PoolSolutions))
        self.fcGraph = dgl.graph((self.fcMatrix.row, self.fcMatrix.col))
        self.fcGraph.edata['feat'] = torch.cat((self.cosineDist.view(-1, 1), self.jaccardDist.view(-1, 1)), dim=-1)

        if self.subModel.objVal > 1.0 + 1e-7:
            self.done = False
        else:
            self.done = True
            self.episode += 1
            print('Episode {} done, len = {}'.format(str(self.episode), str(self.num_action)))

        return {'biGraph': self.biGraph, 'fcGraph': self.fcGraph, 'globalFeat': self.globalFeat}, self.reward, self.done, {}

    def reset(self):
        self.num_action = 0
        self.fileAttrs = self.instance_attrs

        if self.numTestInstance is None:
            self.n = random.choice(self.n_list)
            self.c = random.choice(self.c_list)
            self.w_min = random.choice(self.w_min_list)
            self.w_max = random.choice(self.w_max_list)

            self.l_min = round(self.w_min * self.c)
            self.l_max = round(self.w_max * self.c)
            self.demand = sorted(np.random.randint(self.l_min, self.l_max, self.n))

            self.demandList, self.counts = np.unique(self.demand, return_counts=True)

            self.typeNum = len(self.demandList)
            self.length = self.c
            self.typesDemand, self.quantityDemand = list(self.demandList), list(self.counts)
        else:
            if self.rootPath is None:
                self.filePath = os.path.join('generated_data', self.fileAttrs,  self.fileAttrs + '_{:d}.txt'.format(self.readIndex))
            else:
                self.filePath = os.path.join(self.rootPath, 'generated_data', self.fileAttrs,  self.fileAttrs + '_{:d}.txt'.format(self.readIndex))
            self.typesDemand = []
            self.quantityDemand = []
            f = open(self.filePath, 'r')
            f.readline()
            self.typeNum = int(f.readline())
            self.length = int(f.readline())
            for b in range(self.typeNum):
                wj_dj = list(map(int, f.readline().split()))
                self.typesDemand.append(wj_dj[0])
                self.quantityDemand.append(wj_dj[1])
            f.close()
            self.readIndex += 1
            if self.readIndex == self.numTestInstance:
                self.readIndex = 0

        # initialize model
        self.mainModel = Model('Main Model')
        self.subModel = Model('SubModel')
        self.mainModel.Params.OutputFlag = 0
        self.subModel.Params.OutputFlag = 0

        # variables
        self.y = self.mainModel.addVars(len(self.typesDemand), obj=1, vtype='C', name='y')
        # constraints
        self.mainModel.addConstrs(((self.y[i] * (self.length // self.typesDemand[i])) >= self.quantityDemand[i]
                                   for i in range(len(self.typesDemand))), name='mainCon')
        # solve RMP
        self.mainModel.optimize()
        # get object and duals
        self.Dual = self.mainModel.getAttr(GRB.Attr.Pi, self.mainModel.getConstrs())
        self.objValue = []
        self.objValue.append(self.mainModel.objVal)

        # initialize constraint features
        self.diag = torch.tensor([self.mainModel.getCoeff(self.mainModel.getConstrs()[i], self.mainModel.getVars()[i]) for i in
                                  range(len(self.mainModel.getConstrs()))])
        self.constrCoeff = torch.diag(self.diag)
        self.Dual = self.Dual
        self.rowConnectivity = torch.sum(self.constrCoeff > 0, dim=1)
        self.Slack = -torch.tensor(self.mainModel.getAttr('Slack', self.mainModel.getConstrs()))
        self.RHS = torch.tensor(self.mainModel.getAttr('RHS', self.mainModel.getConstrs()))
        self.lengthRatio = torch.tensor(self.typesDemand) / self.length
        self.featRow = torch.cat((torch.tensor(self.Dual).unsqueeze(-1),
                                  self.rowConnectivity.unsqueeze(-1),
                                  self.Slack.unsqueeze(-1),
                                  self.RHS.unsqueeze(-1),
                                  self.lengthRatio.unsqueeze(-1)), dim=-1)
        # initialize column features
        self.reducedCost = torch.tensor(self.mainModel.getAttr('RC', self.mainModel.getVars()))
        self.colConnectivity = torch.sum(self.constrCoeff > 0, dim=0)
        self.solutionValue = torch.tensor(self.mainModel.getAttr('X', self.mainModel.getVars()))
        self.Waste = 1 - torch.mm(self.lengthRatio.unsqueeze(0), self.constrCoeff).squeeze()
        self.InBasis = (torch.tensor(self.mainModel.getAttr('VBasis', self.mainModel.getVars())) == 0).float()
        self.OutBasis = (torch.tensor(self.mainModel.getAttr('VBasis', self.mainModel.getVars())) != 0).float()
        self.numInBasis = self.InBasis
        self.numOutBasis = self.OutBasis
        self.leftBasis = self.OutBasis
        self.enterBasis = self.InBasis
        self.isCandidate = torch.zeros_like(self.reducedCost)
        self.featCol = torch.cat((self.reducedCost.unsqueeze(-1),
                                  self.colConnectivity.unsqueeze(-1),
                                  self.solutionValue.unsqueeze(-1),
                                  self.Waste.unsqueeze(-1),
                                  self.InBasis.unsqueeze(-1),
                                  self.OutBasis.unsqueeze(-1),
                                  self.numInBasis.unsqueeze(-1),
                                  self.numOutBasis.unsqueeze(-1),
                                  self.leftBasis.unsqueeze(-1),
                                  self.enterBasis.unsqueeze(-1),
                                  self.isCandidate.unsqueeze(-1)), dim=-1)

        # initialize PP
        self.c = self.subModel.addVars(len(self.typesDemand), obj=self.Dual, vtype='I', name='c')
        self.subModel.addConstr(self.c.prod(self.typesDemand) <= self.length, name='subCon')
        self.subModel.Params.PoolSearchMode = self.PoolSearchMode
        self.subModel.Params.PoolSolutions = self.PoolSolutions
        # solve PP
        self.subModel.setAttr(GRB.Attr.ModelSense, -1)
        self.subModel.optimize()
        self.iterNum = 1

        # candidate column pool
        self.columnPool = []
        self.reducedCost_candidate = []
        for a in range(self.subModel.SolCount):
            self.subModel.Params.SolutionNumber = a
            self.columnCoeff = [round(x) for x in self.subModel.getAttr('Xn', self.subModel.getVars())]
            self.columnPool.append(self.columnCoeff)
            self.reducedCost_candidate.append(self.subModel.PoolObjVal)

        self.candidateCoeff = torch.tensor(self.columnPool).T
        self.numRow = self.constrCoeff.shape[0]
        self.numCol = self.constrCoeff.shape[1]
        self.numCandidate = self.candidateCoeff.shape[1]
        # adj matrix
        self.adjMatrix = torch.cat([torch.cat([torch.zeros(self.numRow, self.numRow), self.constrCoeff, self.candidateCoeff], dim=1),
                                    torch.cat([self.constrCoeff.T, torch.zeros(self.numCol, self.numCol), torch.zeros(self.numCol, self.numCandidate)], dim=1),
                                    torch.cat([self.candidateCoeff.T, torch.zeros(self.numCandidate, self.numCol), torch.zeros(self.numCandidate, self.numCandidate)],
                                   dim=1)], dim=0)
        self.adjMatrix_coo = coo_matrix(self.adjMatrix)
        # draw_adjMatrix(self.adjMatrix_coo, self.numRow, self.numCol, self.numCandidate)
        # update candidate features
        self.reducedCost_candidate = 1 - torch.tensor(self.reducedCost_candidate)
        self.colConnectivity_candidate = torch.sum(self.candidateCoeff > 0, dim=0)
        self.solutionValue_candidate = torch.zeros_like(self.reducedCost_candidate)
        self.Waste_candidate = 1 - torch.mm(self.lengthRatio.unsqueeze(0), self.candidateCoeff.float()).squeeze()
        self.InBasis_candidate = torch.zeros_like(self.reducedCost_candidate)
        self.OutBasis_candidate = torch.zeros_like(self.reducedCost_candidate)
        self.numInBasis_candidate = torch.zeros_like(self.reducedCost_candidate)
        self.numOutBasis_candidate = torch.zeros_like(self.reducedCost_candidate)
        self.leftBasis_candidate = torch.zeros_like(self.reducedCost_candidate)
        self.enterBasis_candidate = torch.zeros_like(self.reducedCost_candidate)
        self.isCandidate_candidate = torch.ones_like(self.reducedCost_candidate)

        self.featCandidate = torch.cat((self.reducedCost_candidate.unsqueeze(-1),
                                        self.colConnectivity_candidate.unsqueeze(-1),
                                        self.solutionValue_candidate.unsqueeze(-1),
                                        self.Waste_candidate.unsqueeze(-1),
                                        self.InBasis_candidate.unsqueeze(-1),
                                        self.OutBasis_candidate.unsqueeze(-1),
                                        self.numInBasis_candidate.unsqueeze(-1),
                                        self.numOutBasis_candidate.unsqueeze(-1),
                                        self.leftBasis_candidate.unsqueeze(-1),
                                        self.enterBasis_candidate.unsqueeze(-1),
                                        self.isCandidate_candidate.unsqueeze(-1)), dim=-1)

        # bipartite graph (heterogeneous)
        self.edge = coo_matrix(torch.cat([self.constrCoeff, self.candidateCoeff], dim=1))
        self.graph_data = {
            ('col', 'edge', 'row'): (self.edge.col, self.edge.row),
            ('row', 'edge', 'col'): (self.edge.row, self.edge.col)
        }
        self.num_nodes_dict = {
            'row': self.numRow,
            'col': self.numCol + self.numCandidate,
        }
        # heterogeneous graph
        self.biGraph = dgl.heterograph(self.graph_data, self.num_nodes_dict)
        # attach features
        self.biGraph.nodes['row'].data['feat'] = self.featRow
        self.biGraph.nodes['col'].data['feat'] = torch.cat([self.featCol, self.featCandidate], dim=0)
        self.biGraph.edges[('col', 'edge', 'row')].data['weight'] = torch.tensor(self.edge.data).unsqueeze(-1)
        self.biGraph.edges[('row', 'edge', 'col')].data['weight'] = torch.tensor(self.edge.data).unsqueeze(-1)
        # draw_biGraph(self.biGraph, self.numRow, self.numCol, self.numCandidate)

        # similarity
        self.cosineDist = 1 - cosine_similarity(self.columnPool, self.columnPool)
        self.columnPool_Jaccard = [[index for index, row in enumerate(candidate) if row != 0] for candidate in self.columnPool]
        self.jaccardDist = torch.ones(self.PoolSolutions, self.PoolSolutions)
        for a, A in enumerate(self.columnPool_Jaccard):
            for b, B in enumerate(self.columnPool_Jaccard):
                lenA = len(A)
                lenB = len(B)
                lenAB = len([i for i in A if i in B])
                self.jaccardDist[a][b] = 1 - lenAB / (lenA + lenB - lenAB)
        self.cosineDist = torch.tensor(self.cosineDist).float()
        self.jaccardDist = torch.tensor(self.jaccardDist).float()

        # complete graph
        self.fcMatrix = coo_matrix(torch.ones(self.PoolSolutions, self.PoolSolutions))
        self.fcGraph = dgl.graph((self.fcMatrix.row, self.fcMatrix.col))
        self.fcGraph.edata['feat'] = torch.cat((self.cosineDist.view(-1, 1), self.jaccardDist.view(-1, 1)), dim=-1)

        # global features
        self.globalFeat = torch.tensor([sum(self.quantityDemand) / 100,
                                        self.length / 100,
                                        min(self.typesDemand) / self.length,
                                        max(self.typesDemand) / self.length]).float()
        return {'biGraph': self.biGraph, 'fcGraph': self.fcGraph, 'globalFeat': self.globalFeat}

    def close(self):
        return None



