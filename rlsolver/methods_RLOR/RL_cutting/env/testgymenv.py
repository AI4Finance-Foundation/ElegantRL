from env.gymenv_v2 import timelimit_wrapper, GurobiOriginalEnv
import numpy as np


def make_gurobi_env(load_dir, idx, timelimit):
	print('loading training instances, dir {} idx {}'.format(load_dir, idx))
	A = np.load('{}/A_{}.npy'.format(load_dir, idx))
	b = np.load('{}/b_{}.npy'.format(load_dir, idx))
	c = np.load('{}/c_{}.npy'.format(load_dir, idx))
	env = timelimit_wrapper(GurobiOriginalEnv(A, b, c, solution=None, reward_type='obj'), timelimit)
	return env

if __name__ == '__main__':

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--instance-idx', type=int, default=0)
	parser.add_argument('--instance-name', type=str, default='randomip_n60_m60')
	parser.add_argument('--timelimit', type=int, default=100)
	args = parser.parse_args()

	instance_idx = args.instance_idx
	instance_name = args.instance_name
	timelimit = args.timelimit

	# create an environment
	env = make_gurobi_env('instances/{}'.format(instance_name), instance_idx, timelimit)

	# gym loop
	s = env.reset()
	d = False
	t = 00
	while not d:
		a = np.random.randint(0, s[-1].size, 1)
		s, r, d, _ = env.step(list(a))
		print('step', t, 'reward', r, 'action space size', s[-1].size)
		t += 1

