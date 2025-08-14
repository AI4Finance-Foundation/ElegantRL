import numpy as np
from gurobipy import *


def GurobiIntSolve(A,b,c):
	c = -c # Gurobi default is maximization
	varrange = range(c.size)
	crange = range(b.size)
	m = Model('LP')
	m.params.OutputFlag = 0 #suppres output
	X = m.addVars(varrange, lb=0.0, ub=GRB.INFINITY, vtype=GRB.INTEGER,
                 obj=c,
                 name="X")
	C = m.addConstrs((sum(A[i,j]*X[j] for j in varrange)==b[i] for i in crange),'C')
	m.params.Method = -1 # primal simplex Method = 0
	m.optimize()
	# obtain results
	solution = []; 
	for i in X:
		solution.append(X[i].X);
	solution = np.asarray(solution)
	return m.ObjVal,solution


def GurobiSolve(A,b,c,Method=0):
	#print('solving starts')
	c = -c # Gurobi default is maximization
	varrange = range(c.size)
	crange = range(b.size)
	m = Model('LP')
	m.params.OutputFlag = 0 #suppress output
	X = m.addVars(varrange, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS,
                 obj=c,
                 name="X")
	C = m.addConstrs((sum(A[i,j]*X[j] for j in varrange)==b[i] for i in crange),'C')
	m.params.Method = Method # primal simplex Method = 0
	#print('start optimizing...')
	m.optimize()
	# obtain results
	solution = []; basis_index = []; RC = []
	for i in X:
		solution.append(X[i].X);
		RC.append(X[i].getAttr('RC'))
		if X[i].getAttr('VBasis') == 0:
			basis_index.append(i)
	solution = np.asarray(solution)
	RC = np.asarray(RC)
	basis_index = np.asarray(basis_index)
	#print('solving completes')
	return m.ObjVal,solution,basis_index,RC


def computeoptimaltab(A,b,RC,obj,basis_index):
	'''
	A - A matrix, b - constraint, RC - reduced cost, basis_index - basis 
	'''
	m,n = A.shape
	assert m == b.size; assert n == RC.size
	B = A[:,basis_index]
	try:
		INV = np.linalg.inv(B)
	except:
		print('basisindex length:', basis_index.size)
		print('Ashape:', A.shape)
		raise ValueError
	x = np.dot(INV,b)
	A_ = np.dot(INV,A)
	firstrow = np.append(-obj,RC)
	secondrow = np.column_stack((x,A_))
	tab = np.vstack((firstrow,secondrow))
	return tab


def generatecutzeroth(row):
	###
	# generate cut that includes cost/obj row as well
	###
	n = row.size
	a = row[1:n]
	b = row[0]
	cut_a = a - np.floor(a)
	cut_b = b - np.floor(b)
	return cut_a,cut_b


def updatetab(tab,cut_a,cut_b,basis_index):
	cut_a = -cut_a
	cut_b = -cut_b
	m,n = tab.shape
	A_ = tab[1:m,1:n]; b_ = tab[1:m,0]; c_ = tab[0,1:n]; obj = tab[0,0]
	Anew1 = np.column_stack((A_,np.zeros(m-1)))
	Anew2 = np.append(cut_a,1)
	Anew = np.vstack((Anew1,Anew2))
	bnew = np.append(b_,cut_b)
	cnew = np.append(c_,0)
	M1 = np.append(obj,cnew)
	M2 = np.column_stack((bnew,Anew))
	newtab = np.vstack((M1,M2))
	basis_index = np.append(basis_index,n-1)
	return newtab,basis_index,Anew,bnew
