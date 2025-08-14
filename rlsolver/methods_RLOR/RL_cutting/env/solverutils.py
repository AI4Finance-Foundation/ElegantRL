import numpy as np


def phase_two(x,basis_index,A,b,c):
	'''
	executes phase_two of generic simplex algorithm
	x: starting BFS
	basis_index: index of variables that are basic feasible in x
	A: constraint matrix
	b: constraint
	c: cost
	'''
	# check consistency of basis_index and x
	for i in range(len(x)):
		if i not in basis_index:
			if abs(x[i])>1e-8:
				sys.exit('inconsistent input')
	# compute initial tab
	(m,n) = np.shape(A)
	B = A[:,basis_index]
	cB = c[basis_index]
	invB = np.linalg.inv(B)
	M1 = np.dot(invB,A)
	M2 = np.dot(invB,b)
	M3 = c - np.dot(cB,M1)
	M4 = -np.dot(cB,M2)
	A_b = np.column_stack((M1,M2))
	c_o = np.append(M3,M4)
	tab = np.vstack((A_b,c_o))
	counter = 1
	var_selection_order = []
	# iteration
	while np.sum(tab[m,0:n]<0)>1e-8:
		# find column with smallest index
		jset = np.where(tab[m,0:n]<-0)
		j = jset[0][0]
		# print out tab for error checking
		# print 'iteration',counter 
		# print 'tab',tab
		# check u<0
		if np.sum(tab[0:m,j]>1e-8)<1:
			bounded = 0
			return
		# find u>0 and compute ratio for comparison
		# bland's rule
		u = tab[0:m,j]
		xB = x[basis_index]
		ratio = np.zeros_like(xB)
		for i in range(len(xB)):
			if u[i]>1e-10:
				ratio[i] = xB[i]/(u[i]+0.0)
			else:
				ratio[i] = -10
		lset = np.where(ratio==np.min(ratio[u>1e-10]))
		l = lset[0][0]
		# change basis
		for i in range(m+1):
			if i!=l:
				tab[i,:] = tab[i,:] + (-tab[i,j]/(tab[l,j]+0.0)) * tab[l,:]
				tab[i,j] = 0 # for numerical stability #
		tab[l,:] = tab[l,:]/(tab[l,j]+0.0)
		# update basis index
		basis_index[l] = j
		# update solution
		x = np.zeros(n)
		X = tab[0:m,n]
		for i in range(len(basis_index)):
			idxset = np.where(tab[0:m,basis_index[i]]==np.max(tab[0:m,basis_index[i]]))
			idx = idxset[0]
			if len(idx)>1:
				print('tab')
				print(tab)
				print(tab[0:m,basis_index[i]])
				print('basis',basis_index[i])
				print('idx',idx)
				print('error of multiple indices')
				sys.exit()
			idx = idx[0]
			x[basis_index[i]] = X[idx]
		counter += 1
		var_selection_order.append(l)
		if counter > 10000:
			print(counter)
			print(var_selection_order)
			sys.exit('iteration exceeds maxiter')
	bounded = 1
	## change the layout
	m,n = tab.shape
	A_ = tab[0:m-1,0:n-1]
	b_ = tab[0:m-1,n-1]
	cbar = tab[m-1,0:n-1]
	obj = tab[m-1,n-1]
	M1 = np.append(obj,cbar)
	M2 = np.column_stack((b_,A_))
	tab = np.vstack((M1,M2))
	return tab,bounded,basis_index


def phase_one(A,b,c):
	'''
	executes phase one of generic simplex algorithm
	A: constraint matrix
	b: constraint
	c: cost
	'''
	# modify the input
	m,n = np.shape(A)
	for i in range(m):
		if b[i]<-1e-10:
			A[i,:] = -A[i,:]
			b[i] = -b[i]
	# solve an aux problem
	A_aux = np.concatenate((A,np.eye(m)),axis=1)
	b_aux = b
	c_aux = np.append(np.zeros(n),np.ones(m))
	x = np.append(np.zeros(n),b)
	basis_index = np.arange(n,n+m)
	tab,bounded,basis_index = phase_two(x,basis_index,A_aux,b_aux,c_aux)
	## change the layout
	m1,n1 = tab.shape
	A_ = tab[1:m1,1:n1]
	b_ = tab[1:m1,0]
	cbar = tab[0,1:n1]
	obj = tab[0,0]
	M1 = np.append(cbar,obj)
	M2 = np.column_stack((A_,b_))
	tab = np.vstack((M2,M1))
	# check feasibility
	if abs(tab[m,n+m]) < 1e-8:
		feasible = 1
	else:
		feasible = 0
		basis_index = []
		tab = []
		return x,basis_index,feasible,tab,A,b
 	# if feasible, drive artificial var out of the set
	while np.sum(basis_index>n-1)>0:
		lset = np.where(basis_index>n-1)
		l = lset[0][0]
		# if all are zeros, eliminte the redundant row
		if np.sum(abs(tab[l,0:n])) < 1e-5:
			index = []
			for ii in range(m):
				if ii!=l:
					index.append(ii)
			basis_index = basis_index[np.asarray(index)]
			A = A[np.asarray(index),:]
			b = b[np.asarray(index)]
			index = np.append(index,m)
			tab = tab[np.asarray(index),:]
		else: # can find a nonzero element
			jset = np.where(abs(tab[l,0:n])>1e-5)
			j = jset[0][0]
			m,n = A.shape
			for k in range(m):
				if k!=l:
					tab[k,:] = tab[k,:] + (-tab[k,j]/(0.0+tab[l,j])) * tab[l,:]
					tab[k,j] = 0 # for numerical stability #
			tab[l,:] = tab[l,:]/(0.0+tab[l,j])
			basis_index[l] = j
	m = len(basis_index)
	x = np.zeros(n)
	X = tab[0:m,n+m]
	for i in range(m):
		idxset = np.where(tab[0:m,basis_index[i]]>0.5)
		idxset = idxset[0]
		if len(idxset)>1:
			print('idx set size exceeds 1')
			sys.exit()
		else:
			x[basis_index[i]] = X[idxset[0]]
	## change the layout
	m,n = tab.shape
	A_ = tab[0:m-1,0:n-m]
	b_ = tab[0:m-1,n-1]
	cbar = tab[m-1,0:n-m]
	obj = tab[m-1,n-1]
	M1 = np.append(obj,cbar)
	M2 = np.column_stack((b_,A_))
	tab = np.vstack((M1,M2))
	A = A_
	b = b_
	return x,basis_index,feasible,tab,A,b


def simplexalgo(A,b,c):
	x,basis_index,feasible,tab,A,b = phase_one(A,b,c)
	if feasible:
		tab,bounded,basis_index = phase_two(x,basis_index,A,b,c)
	else:
		bounded = 0
	return tab,bounded,feasible,basis_index


def lexcolumn(l,tab):
	m,n = tab.shape
	tab = tab[:,1:n]
	indexset = np.where(tab[l+1,:]<0)
	indexset = indexset[0]
	if len(indexset)==0:
		j = []
		return j
	columns = tab[:,indexset]
	for i in range(len(indexset)):
		columns[:,i] = columns[:,i]/columns[l+1,i]
	findlargestset = np.where(columns[0,:]==np.max(columns[0,:]))
	findlargest = findlargestset[0]
	counter = 1
	while len(findlargest)>=2:
		ntem,mtem= columns.shape
		columns = columns[1:ntem,findlargest]
		indexset = indexset[findlargest]
		findlargestset = np.where(columns[0,:]==np.max(columns[0,:]))
		findlargest = findlargestset[0]
		counter += 1
		if counter > m - 1:
			print('counter exceeds max num')
			sys.exit()
	j = indexset[findlargest]
	return j
	

def dualsimplexalgo(tab,basis_index):
	###
	# follow standard layout [obj,cbar;b,A]
	###
	m,n = tab.shape
	cc = 0
	var_selection_order = []
	# modfy the tab to be column lex positive
	for j in range(1,n):
		if (j-1) not in basis_index:
			columnfirstnonzeroset = np.where(abs(tab[0:m,j])>0)
			columnfirstnonzero = columnfirstnonzeroset[0][0]
			if tab[columnfirstnonzero,j]<0:
				if columnfirstnonzero==0:
					raise ValueError('reduced cost is negative for non bfs variable')
					#exit.sys()
				tab[1:m,j] = -tab[1:m,j]
	# start iteration
	while np.sum(tab[1:m,0]<-1e-10)>=1: # or <0
		#print 'iteration...'
		# choose the lth row with x_b[l]<0
		lset = np.where(tab[1:m,0]<0)
		l = lset[0][0]
		# check if lth rows are all positive
		if np.sum(tab[l+1,1:n]>=0)==n-1:
			print('primal nonfeasible due to dual unbounded...')
			# dual unbounded, primal non feasible
			print(tab[l+1,:])
			bounded = 1
			feasible = 0
			tab = []; basis_index = []
			return tab,bounded,feasible,basis_index
		cbar = tab[0,1:n]
		if np.sum(cbar<0)>=1:
			raise ValueError('error due to negative cbar in dual simplex')
			#sys.exit()
		# choose the pivot column
		j = lexcolumn(l,tab)
		# update basis index
		basis_index[l] = j
		# update tab
		for i in range(m):
			if i!=l+1:
				tab[i,:] = tab[i,:] + (-tab[i,j+1]/(0.0+tab[l+1,j+1])) * tab[l+1,:]
				tab[i,j+1] = 0
		tab[l+1,:] = tab[l+1,:]/(0.0+tab[l+1,j+1])
		cc += 1
		var_selection_order.append(l)
		if cc>1000:
			print('dual simplex iteration exceeds max iter')
			print(var_selection_order)
			sys.exit()
	bounded = 1
	feasible = 1
	return tab,bounded,feasible,basis_index


def compute_solution_from_tab(tab,basis_index):
	m,n = tab.shape
	x = np.zeros(n-1)
	basis_index = list(basis_index)
	for i in range(len(basis_index)):
		x[basis_index[i]] = tab[i+1,0]
	return x


def SolveLP(A,b,c):
	c=-c
	tab,bounded,feasible,basis_index = simplexalgo(A,b,c)
	obj = -tab[0,0]
	solution = compute_solution_from_tab(tab,basis_index)
	RC = tab[0,1:]
	return obj,solution,basis_index,RC


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


def SolveLPtab(tab,c):
	# extract data from the tab
	m,n = tab.shape
	#print c.size,n
	try:
		assert c.size == n-1
	except:
		print(c.size,tab.shape)
		raise ValueError
	A = tab[1:m,1:n]; b = tab[1:n,0]
	obj,sol,basis,rc = SolveLP(A,b,c) # dual simplex 1
	return obj,sol,basis,rc	


def SolveLPtabDual(tab,c,basis_index):
	# construct the entire tab
	c = -c
	#x = tab[:,0]
	#obj = -np.sum(c[basis_index] * x)
	#bigtab = np.append(obj,c)
	#bigtab = np.vstack((bigtab,tab))
	bigtab = tab
	tab,bounded,feasible,basis_index = dualsimplexalgo(bigtab,basis_index)
	return tab,basis_index


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


def generatecut_MIP(row,I,basis_index):
	'''
	generate cut for MIP
	I: set of vars required to be integers
	'''
	n = row.size
	b = row[0]
	a = row[1:n]
	f = a - np.floor(a)
	f0 = b - np.floor(b)
	cut_a = np.zeros(n-1)
	cut_b = 0
	for i in range(n-1):
		if i not in basis_index:
			if i in I:
				if f[i]<=f0:
					cut_a[i] = f[i]/(f0+0.0)
				else:
					cut_a[i] = (1-f[i])/(1+0.0-f0)
			else:
				if a[i]>=0:
					cut_a[i] = a[i]/(f0+0.0)
				else:
					cut_a[i] = -a[i]/(1+0.0-f0)
	cut_b = 1
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


def PRUNEtab(tab,basis_index,numvar):
	'''
	prune and return a basis_index cleared of redundant slacks
	'''
	aa = np.asarray(basis_index)
	while np.sum(aa>=numvar)>=1:
		tab,basis_index = prunetab(tab,basis_index,numvar)
		aa = np.asarray(basis_index)
	return tab,basis_index


def prunetab(tab,basis_index,numvar):
	'''
	m,n original size of the tab, m: original num of constraints, n: original num of vars (not including slack vars)
	drop the slack variables that enter basis
	'''
	M,N = tab.shape 
	for i in basis_index:
		if i>=numvar:
            # found a slack variable that enters the basis
            # drop the column
			lset = np.where(abs(tab[1:M,i+1]-1)<1e-8)
			l = lset[0][0]
			tab = np.delete(tab,i+1,1)
			tab = np.delete(tab,l+1,0)
			basis_index = list(basis_index)
			basis_index.remove(i)
			for j in range(len(basis_index)):
				if basis_index[j]>i:
					basis_index[j] -= 1
			basis_index = np.asarray(basis_index)
           # print 'pruning...'
			return tab,basis_index
	return tab,basis_index
