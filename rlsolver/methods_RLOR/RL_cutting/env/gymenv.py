import numpy as np
import cwrapping
GurobiEnv = cwrapping.gurobicpy.GurobiEnv


def make_float64(lists):
	newlists = []
	for e in lists:
		newlists.append(np.float64(e))
	return newlists


def check_feasibility(A, b, solution):
	RHS = np.dot(A, solution)
	if np.sum(RHS - (1.0 - 1e-10) * b > 1e-5) >= 1:
		return False
	else:
		return True


class GurobiOriginalEnv(object):
	def __init__(self, A, b, c, solution=None, reward_type='obj'):
		A, b, c = make_float64([A, b, c])
		self.baseenv = GurobiEnv()
		self.baseenv.reset(A, b, c)
		self.A0 = A.copy()
		self.b0 = b.copy()
		self.c0 = c.copy()
		self.IPsolution = solution
		self.reward_type = reward_type
		assert reward_type in ['simple', 'obj']

		# upon init, check if the ip problem can be solved by lp
		try:
			_, done = self._reset()
			assert done is False
		except NotImplementedError:
			print('the env needs to be initialized with nontrivial ip')

	def _reset(self):
		A,b,cutsa,cutsb,done,objval,xfull,tab = self.baseenv.reset(self.A0, self.b0, self.c0)
		self.cutsa = cutsa
		self.cutsb = cutsb
		self.objval = objval
		self.x = xfull
		self.tab = tab
		return (A, b, self.c0, cutsa, cutsb), done

	def reset(self):
		s, d = self._reset()
		return s

	def step(self, action):
		if isinstance(action, list):
			if len(action) >= 1:
				for a in action:
					cuta = self.cutsa[a,:]
					cutb = self.cutsb[a]
					A,b,cutsa,cutsb,done,objval,xfull,tab = self.baseenv.step(cuta, cutb)
			if len(action) == 0:
				return (self.A0,self.b0,self.c0,[],[]), 0.0, True
		elif isinstance(action, int):
			cuta = self.cutsa[action,:]
			cutb = self.cutsb[action]
			A,b,cutsa,cutsb,done,objval,xfull,tab = self.baseenv.step(cuta, cutb)
		else:
			raise NotImplementedError			
		# compute reward
		if self.reward_type == 'obj':
			reward = abs(objval - self.objval)
		elif self.reward_type == 'simple':
			reward = -1
		else:
			raise NotImplementedError

		self.cutsa = cutsa
		self.cutsb = cutsb
		self.objval = objval
		self.done = done
			
		self.x = xfull
		self.tab = tab

		return (A, b, self.c0, cutsa, cutsb), reward, done, {}


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
