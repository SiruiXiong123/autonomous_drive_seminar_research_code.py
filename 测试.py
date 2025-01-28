from metadrive.obs.state_obs import StateObservation, LidarStateObservation
from metadrive.obs.observation_base import BaseObservation
import os
import gymnasium as gym
from metadrive.envs.metadrive_env import MetaDriveEnv
from EgostateAndNavigation_obs import EgoStateNavigationobservation
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from functools import partial
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import random

random.seed(123)

base_path = r'C:\Users\xsr\Desktop\ego_state'
log_path = os.path.join(base_path, 'Training', 'Logs')

def create_env(need_monitor=False):
    env = MetaDriveEnv(cfg)
    if need_monitor:
        env = Monitor(env)
    return env

if __name__ == '__main__':
    cfg = dict(
        num_scenarios=100,
        start_seed=0,
        random_lane_width=True,
        random_lane_num=False,
        use_render=True,
        traffic_density=0.1,
        traffic_mode="hybrid",
    )

    def create_env_for_testing():
        def _env_fn():
            return MetaDriveEnv(cfg)
        return DummyVecEnv([_env_fn])

    env = create_env_for_testing()
    TD3_Path = os.path.join(base_path, 'Training', 'Saved Models', 'overtake v0.5.zip')

    model = TD3.load(TD3_Path, env=env)
    # episode_rewards, episode_infos = evaluate_policy(
    #     model,
    #     env,
    #     n_eval_episodes=100,  # 评估 100 次
    #     deterministic=True,
    #     render=False,
    #     return_episode_rewards=True  # 返回每个 episode 的奖励和信息
    # )




    episodes = 5
    for episode in range(1, episodes + 1):
        obs = env.reset()  # VecEnv 的 reset 返回直接是 obs
        done = False
        score = 0

        while not done:
            env.render(mode="topdown")
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)  # VecEnv 的 step 返回 dones
            score += reward
            if done:
                break
    env.close()
