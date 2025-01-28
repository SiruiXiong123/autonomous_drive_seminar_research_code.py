from metadrive.obs.state_obs import StateObservation, LidarStateObservation
from metadrive.obs.observation_base import BaseObservation
import os
import gymnasium as gym
from metadrive.envs.metadrive_env import MetaDriveEnv
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from functools import partial
import numpy as np
from EgostateAndNavigation_obs import EgoStateNavigationobservation
import random
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps
from win32ui import ID_FILE_LOCATE
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

random.seed(123)
cfg=dict(
        num_scenarios=500,
        start_seed=123,
        random_lane_width=True,
        random_lane_num=False,
        use_render=False,
        traffic_density=0.1,
        traffic_mode="hybrid",
    )

base_path = r'C:\Users\xsr\Desktop\ego_state'
log_path = os.path.join(base_path, 'Training', 'Logs')


def create_env(need_monitor=False):
    env = MetaDriveEnv(cfg)
    if need_monitor:
        env = Monitor(env)
    return env


#防止子进程递归调用主程序脚本，需要if name

if __name__ == '__main__':
    env = SubprocVecEnv([partial(create_env, True) for _ in range(4)])

    obs = env.reset()

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    checkpoint_on_event = CheckpointCallback(save_freq=1, save_path="./logs/") #save_freq=1,每次触发回调保存模型
    event_callback = EveryNTimesteps(n_steps=409600, callback=checkpoint_on_event)

    model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1,tensorboard_log=log_path)
    # model = PPO('MlpPolicy', env, verbose=1, n_steps=4096, tensorboard_log=log_path)
    model.learn(total_timesteps=3000000, log_interval=100,callback=event_callback)
    PPO_Path = os.path.join(base_path, 'Training', 'Saved Models', 'overtake v0.5.zip')
    model.save(PPO_Path)

    # tensorboard --logdir "C:\Users\xsr\Desktop\ego_state\Training\Logs\TD3_1"

    # env = MetaDriveEnv(cfg)
    # episodes = 2
    # for episode in range(1, episodes + 1):
    #     state = env.reset()
    #     done = False
    #     score = 0
    #
    #     while not done:
    #         env.render(mode="topdown")
    #         action = env.action_space.sample()
    #         n_state, reward, done, truncated, info = env.step(action)
    #         score += reward
    #     print('Episode:{} Score:{}'.format(episode, score))