from environment import create_env
from train import train_model
from callbacks import create_callbacks
# parse the args
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--obs_mode', default=2, type=int, help='Training observation mode, 0 = diff gray obs, 1 = gray obs, 2 = concate of 1 and 2')
    parser.add_argument('--stack', default=10, type=int, help='Number of frame to stack for the model')
    parser.add_argument('--skip', default=0, type=int, help='Number of frame to skip')
    parser.add_argument('--n_envs', default=6, type=int, help='Number of env to train on')
    parser.add_argument('--total_timesteps', default=20_000_000, type=int, help='Number of frame to skip')
    parser.add_argument('--check_freq', default=5_000, type=int, help='Frequency of callback (eg. saving, eval model) to use.')
    parser.add_argument('--checkpoint_dir', default='./checkpoints/', type=str, help='What directory to save the checkpoints to')
    parser.add_argument('--scenario', default='scenario', type=str, help='What scenario to start with')
    parser.add_argument('--state', default='Champion.Level1.RyuVsGuile', type=str, help='What state  to start with')
    parser.add_argument('--hit_reward_strategy', default=2, type=int, help='What reward to give on hit. 0 1 or 2 todo describe what')
    args = parser.parse_args()
    env = create_env(**vars(args))
    callbacks = create_callbacks(**vars(args))
    model = train_model(env=env, total_timesteps=args.total_timesteps, callbacks=callbacks)
    # env = create_env(obs_mode=2, stack=20, skip=4, n_envs=15)
    # model = train_model(env=env,total_timesteps=20000)
    print("FINISH")
