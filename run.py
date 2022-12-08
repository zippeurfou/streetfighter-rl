from environment import create_env
from train import train_model
# parse the args
import argparse





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--obs_mode', default=2, type=int, help='Training observation mode, 0 = gray obs, 1 = diff gray obs, 2 = concate of 1 and 2')
    parser.add_argument('--stack', default=20, type=int, help='Number of frame to stack for the model')
    parser.add_argument('--skip', default=4, type=int, help='Number of frame to skip')
    parser.add_argument('--n_envs', default=15, type=int, help='Number of frame to skip')
    parser.add_argument('--total_timesteps', default=20_000_000, type=int, help='Number of frame to skip')
    args = parser.parse_args()
    env = create_env(obs_mode=args.obs_mode, stack=args.stack, skip=args.skip, n_envs=args.n_envs)
    model = train_model(env=env,total_timesteps=args.total_timesteps)
    # env = create_env(obs_mode=2, stack=20, skip=4, n_envs=15)
    # model = train_model(env=env,total_timesteps=20000)
    print("FINISH")
