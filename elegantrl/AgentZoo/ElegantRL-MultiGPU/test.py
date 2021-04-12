from elegantrl.env import *


def get_video_to_watch_gym_render():
    import cv2  # pip3 install opencv-python
    import gym  # pip3 install gym==0.17 pyglet==1.5.0  # env.render() bug in gym==0.18, pyglet==1.6
    import torch

    '''choose env'''
    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)

    # from elegantrl.env import PreprocessEnv
    env_name = ['BipedalWalker-v3', 'AntBulletEnv-v0',
                'KukaBulletEnv-v0',
                'ReacherBulletEnv-v0', 'PusherBulletEnv-v0',
                "ThrowerBulletEnv-v0", "StrikerBulletEnv-v0"
                ][1]
    env = PreprocessEnv(env=gym.make(env_name))

    '''initialize agent'''
    agent = None

    from elegantrl.agent import AgentPPO
    agent = AgentPPO()
    agent.if_use_dn = True
    net_dim = 2 ** 8
    cwd = f'./{env_name}_4/'

    # from elegantrl.agent import AgentModSAC
    # agent = AgentModSAC()
    # agent.if_use_dn = True
    # net_dim = 2 ** 8
    # cwd = f'./{env_name}_2/'

    device = None
    if agent is not None:
        state_dim = env.state_dim
        action_dim = env.action_dim
        agent.init(net_dim, state_dim, action_dim)
        agent.save_load_model(cwd=cwd, if_save=False)
        device = agent.device
        rd.seed(194686)
        torch.manual_seed(1942876)

    '''initialize evaluete and env.render()'''
    save_frame_dir = 'frames'

    if save_frame_dir:
        os.makedirs(save_frame_dir, exist_ok=True)

    state = env.reset()
    episode_return = 0
    step = 0
    for i in range(2 ** 9):
        print(i) if i % 128 == 0 else None
        for j in range(1):
            if agent is not None:
                s_tensor = torch.as_tensor((state,), dtype=torch.float32, device=device)
                a_tensor = agent.act(s_tensor)
                action = a_tensor.detach().cpu().numpy()[0]  # if use 'with torch.no_grad()', then '.detach()' not need.
            else:
                action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)

            episode_return += reward
            step += 1

            if done:
                print(f'{i:>6}, {step:6.0f}, {episode_return:8.3f}, {reward:8.3f}')
                state = env.reset()
                episode_return = 0
                step = 0
            else:
                state = next_state

        frame = env.render('rgb_array')
        frame = frame[50:210, 50:270]  # (240, 320) AntPyBulletEnv-v0
        # frame = cv2.resize(frame[:, :500], (500//2, 720//2))
        cv2.imwrite(f'{save_frame_dir}/{i:06}.png', frame)
        cv2.imshow('', frame)
        cv2.waitKey(1)
    env.close()
    # exit()

    '''convert frames png/jpg to video mp4/avi using ffmpeg'''
    if save_frame_dir:
        frame_shape = cv2.imread(f'{save_frame_dir}/{3:06}.png').shape
        print(f"frame_shape: {frame_shape}")

        save_video = 'gym_render.mp4'
        os.system(f"| Convert frames to video using ffmpeg. Save in {save_video}")
        os.system(f'ffmpeg -r 60 -f image2 -s {frame_shape[0]}x{frame_shape[1]} '
                  f'-i ./{save_frame_dir}/%06d.png '
                  f'-crf 25 -vb 20M -pix_fmt yuv420p {save_video}')


def test__show_available_env():
    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)

    env_names = list(gym.envs.registry.env_specs.keys())
    env_names.sort()
    for env_name in env_names:
        if env_name.find('Bullet') == -1:
            continue
        print(env_name)


# get_video_to_watch_gym_render()
# test__show_available_env()

class A:
    def __init__(self):
        self.m = 0

    def init(self):
        self.select = self.select1

    def select(self):
        print(self.m)

    def select1(self):
        print(self.m + 1)


a = A()
a.select()
a.select1()
