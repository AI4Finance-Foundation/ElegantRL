import numpy as np
# from env import gurobiutils
from solverutils import computeoptimaltab, generatecutzeroth

SOLVER = 'GUROBI'
if SOLVER == 'GUROBI':
	from gurobiutils import GurobiSolve
if SOLVER == 'SCIPY':
	from scipyutils import ScipyLinProgSolve


def computeoptimaltab(A, b, RC, obj, basis_index):
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


def compute_state(A,b,c):
	m,n = A.shape
	assert m == b.size and n == c.size
	A_tilde = np.column_stack((A,np.eye(m)))
	b_tilde = b
	c_tilde = np.append(c,np.zeros(m))
	if SOLVER == 'GUROBI':
		obj,sol,basis_index,rc = GurobiSolve(A_tilde,b_tilde,c_tilde)
	elif SOLVER == 'SCIPY':
		obj,sol,basis_index,rc = ScipyLinProgSolve(A_tilde,b_tilde,c_tilde)
	tab = computeoptimaltab(A_tilde,b_tilde,rc,obj,basis_index)
	tab = roundmarrays(tab)
	x = tab[:,0]
	#print tab
	done = True
	if np.sum(abs(np.round(x)-x)>1e-2) >= 1:
		done = False
	cuts_a = []
	cuts_b = []
	for i in range(x.size):
		if abs(round(x[i])-x[i])>1e-2:
			# fractional rows used to compute cut
			cut_a,cut_b = generatecutzeroth(tab[i,:])
			# a^T x + e^T y >= d
			assert cut_a.size == m+n
			a = cut_a[0:n]
			e = cut_a[n:]
			newA = np.dot(A.T,e) - a
			newb = np.dot(e,b) - cut_b
			cuts_a.append(newA)
			cuts_b.append(newb)
	cuts_a,cuts_b = np.array(cuts_a),np.array(cuts_b)
	return A,b,cuts_a,cuts_b,done,obj,x,tab


def roundmarrays(x,delta=1e-7):
	'''
	if certain components of x are very close to integers, round them
	'''
	index = np.where(abs(np.round(x)-x)<delta)
	x[index] = np.round(x)[index]
	return x


class GurobiOriginalEnv(object):
	def __init__(self, A, b, c, solution=None, reward_type='simple'):
		'''
		min c^T x, Ax <= b, x>=0
		'''
		self.A0 = A.copy()
		self.A = A.copy()
		self.b0 = b.copy()
		self.b = b.copy()
		self.c0 = c.copy()
		self.c = c.copy()
		self.x = None
		self.reward_type = reward_type
		assert reward_type in ['simple', 'obj']

		# upon init, check if the ip problem can be solved by lp
		#try:
		#	_, done = self._reset()
		#	assert done is False
		#except NotImplementedError:
		#	print('the env needs to be initialized with nontrivial ip')

	def check_init(self):
		_, done = self._reset()
		return done

	def _reset(self):
		self.A,self.b,self.cuts_a,self.cuts_b,self.done,self.oldobj,self.x,self.tab = compute_state(self.A0,self.b0,self.c0)
		return (self.A,self.b,self.c0,self.cuts_a,self.cuts_b), self.done

	def reset(self):
		s, d = self._reset()
		return s

	def step(self, action):
		cut_a,cut_b = self.cuts_a[action,:],self.cuts_b[action]
		self.A = np.vstack((self.A,cut_a))
		self.b = np.append(self.b,cut_b)
		try:
			self.A,self.b,self.cuts_a,self.cuts_b,self.done,self.newobj,self.x,self.tab = compute_state(self.A,self.b,self.c0)
			if self.reward_type == 'simple':
				reward = -1.0
			elif self.reward_type == 'obj':
				reward = np.abs(self.oldobj - self.newobj)
		except:
			print('error in lp iteration')
			self.done = True
			reward = 0.0
		self.oldobj = self.newobj
		self.A,self.b,self.cuts_a,self.cuts_b = map(roundmarrays,[self.A,self.b,self.cuts_a,self.cuts_b])
		return 	(self.A,self.b,self.c0,self.cuts_a,self.cuts_b), reward, self.done, {}

	def max_gap(self):
		"""
		this method computes the max achivable gap
		"""
		# preprocessing
		A, b, c = self.A0.copy(), self.b0.copy(), self.c0.copy()
		m,n = A.shape
		assert m == b.size and n == c.size
		A_tilde = np.column_stack((A,np.eye(m)))
		b_tilde = b
		c_tilde = np.append(c,np.zeros(m))
		A, b, c = A_tilde, b_tilde, c_tilde
		# compute gaps
		objint, solution_int = gurobiutils.GurobiIntSolve(A, b, c)
		objlp, solution_lp, _, _ = gurobiutils.GurobiSolve(A, b, c)
		return np.abs(objint - objlp), solution_int, solution_lp


class MultipleEnvs(object):
	def __init__(self, envs):
		self.envs = envs
		self.all_indices = list(range(len(self.envs)))
		self.available_indices = list(range(len(self.envs)))
		self.env_index = None
		self.env_now = None

	def reset(self):
		self.env_index = np.random.choice(self.available_indices)
		self.available_indices.remove(self.env_index)
		if len(self.available_indices) == 0:
			self.available_indices = self.all_indices[:]

		self.env_now = self.envs[self.env_index]
		return self.env_now.reset()

	def step(self, action):
		assert self.env_now is not None
		return self.env_now.step(action)


class timelimit_wrapper(object):
	def __init__(self, env, timelimit):
		self.env = env
		self.timelimit = timelimit
		self.counter = 0

	def reset(self):
		self.counter = 0
		return self.env.reset()

	def step(self, action):
		self.counter += 1
		obs, reward, done, info = self.env.step(action)
		if self.counter >= self.timelimit:
			done = True
		return obs, reward, done, info


# some functions for loading instances
def make_multiple_env(load_dir, idx_list, timelimit, reward_type):
	envs = []
	for idx in idx_list:
		print('loading training instances, dir {} idx {}'.format(load_dir, idx))
		A = np.load('{}/A_{}.npy'.format(load_dir, idx))
		b = np.load('{}/b_{}.npy'.format(load_dir, idx))
		c = np.load('{}/c_{}.npy'.format(load_dir, idx))
		env = timelimit_wrapper(GurobiOriginalEnv(A, b, c, solution=None, reward_type=reward_type), timelimit)
		envs.append(env)
	env_final = MultipleEnvs(envs)
	return env_final

