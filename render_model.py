from environment import  StreetFighterRenderEnv
from stable_baselines3 import PPO
import time


def display_game(model_path=None, n_episodes=2, obs_mode=2, skip=4, stack=40):
    env = StreetFighterRenderEnv(obs_mode, skip, stack)
    if model_path is not None:
        model = PPO.load(model_path)
    else:
        model = None
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            if model is not None:
                action, _ = model.predict(obs)
            else:
                action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            env.render()
            total_reward += reward
            time.sleep(0.01)
        print('Total Reward for episode {} is {}'.format(total_reward, episode))
        time.sleep(2)
    env.close()
