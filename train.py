from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
import os
import glob
import torch

LOG_DIR = './logs/'




def train_model(env, total_timesteps=20_000_000, callbacks= []):
    # model_params = {'n_steps': 8960, 'gamma': 0.906, 'learning_rate': 2e-07, 'clip_range': 0.369, 'gae_lambda': 0.891}
    # model_params = {'n_steps': 1024, 'gamma': 0.906, 'learning_rate': 2e-07, 'clip_range': 0.369, 'gae_lambda': 0.891}
    model_params = {'learning_rate': 2e-07, 'batch_size': 128, 'n_steps': 512, 'n_epochs': 10, 'policy_kwargs': {'normalize_images': True}}
    model = PPO('MultiInputPolicy', env, tensorboard_log=LOG_DIR, verbose=1, **model_params)
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)
    model.save(os.path.join(callback.save_path, 'checkpointcall_final'))
    env.close()
    return model
