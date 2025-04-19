from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
import os

def train_ppo(env_id, timesteps=100000, save_path="saved_models/ppo_model.zip"):
    """
    آموزش مدل PPO
    """
    # ایجاد محیط
    env = make_vec_env(env_id, n_envs=1)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save(save_path)
    print(f"مدل PPO در مسیر {save_path} ذخیره شد.")
    return model

def train_dqn(env_id, timesteps=100000, save_path="saved_models/dqn_model.zip"):
    """
    آموزش مدل DQN
    """
    # ایجاد محیط
    env = make_vec_env(env_id, n_envs=1)
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save(save_path)
    print(f"مدل DQN در مسیر {save_path} ذخیره شد.")
    return model