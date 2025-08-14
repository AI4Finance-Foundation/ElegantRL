import numpy as np
from env.gymenv_v2 import GurobiOriginalEnv, timelimit_wrapper


def generate_randomip(n, m):
	# n variables
	# m constraints
	Adict = []
	bdict = []
	for i in range(m):
		a = np.random.randint(0, 5, size=n)
		b = np.random.randint(9 * n, 10 * n, size=1)[0] # 9n - 10n
		Adict.append(a)
		bdict.append(b)
	c = np.random.randint(1, 10, size=n)
	return np.array(Adict), np.array(bdict), c


def randomgraph(n, timelimit):
	A,b,c = generate_randomip(n)
	return timelimit_wrapper(GurobiOriginalEnv(A, b, c, solution=None, reward_type='obj'), timelimit=timelimit)


def hardrandomgraph_randomip(n, m):
	done = False
	counts = 0
	#while counts <= 20 or counts >= 95:
	while True:
		A, b, c = generate_randomip(n,m)
		env = GurobiOriginalEnv(A, b, c, solution=None, reward_type='obj')
		done = env.check_init()
		if not done:
			break
	return A, b, c


if __name__ == '__main__':

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--num-c', type=int, default=60)
	parser.add_argument('--num-v', type=int, default=60)
	parser.add_argument('--num-instances', type=int, default=10)
	args = parser.parse_args()

	n = args.num_v
	m = args.num_c
	num_instances = args.num_instances

	import os

	logdir = 'instances/randomip_n{}_m{}'.format(n, m)
	if not os.path.exists(logdir):
		os.makedirs(logdir)
	print('start generating instances')

	def create_new_env(seed, n, m):
		np.random.seed(seed)
		shared_count = 0
		while shared_count < num_instances:
			A, b, c = hardrandomgraph_randomip(n,m)
			np.save(logdir + '/A_{}'.format(shared_count), A)
			np.save(logdir + '/b_{}'.format(shared_count), b)
			np.save(logdir + '/c_{}'.format(shared_count), c)
			shared_count+= 1
			print('num of instances generated:', shared_count)

	create_new_env(0, n, m)
