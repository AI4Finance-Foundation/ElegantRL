from isaacgym import gymapi

gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()

# set extra parameters here
sim_params.dt = 1/60

# setting the graphics device id to -1 to avoid rendering
sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, sim_params)

# set up the grid of environments
num_envs = 64
envs_per_row = 8
env_spacing = 2.0
env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

# cache some common handles for later use
envs = []

# create and populate the environments
for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
    envs.append(env)

# run the simulation
while True:
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    break  # remove this line for infinite looping
    
gym.destroy_sim(sim)