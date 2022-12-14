# get the rom
import retro
import os
# Wrapp it so I can monitor and stack frames
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
# vec in parallel
from stable_baselines3.common.env_util import make_vec_env
from constant import GameInfo
from frameskip_wrapper import Frameskip, WaitForActionableState
from retro_renderer import RetroHumanRendering
from action_wrapper import StreetFighter2Discretizer
from reward_wrapper import SFRewardWrapper
from observation_wrapper import SFObservationWrapper


def StreetFighterEnv(obs_mode=2, skip=0, scenario='scenario', state='Champion.Level1.RyuVsGuile', hit_reward_strategy=2, mode='rgb_array', save_frames=True):
    # init custom env
    retro.data.Integrations.add_custom_path(
        os.path.join(os.getcwd(), "custom_integrations")
    )
    # make game
    env = retro.make(game="sf2", inttype=retro.data.Integrations.ALL, scenario=scenario, state=state)
    env = RetroHumanRendering(env, mode=mode, save_frames=save_frames)
    env = WaitForActionableState(env)
    # it is important to skip frame at the game level as we don't want to apply the delta on it
    # not sure it makes sense to combine waiting for action and skipping
    if skip > 0:
        env = Frameskip(env, skip)
    # provide a smaller discrete spce
    env = StreetFighter2Discretizer(env)
    env = SFRewardWrapper(env, hit_reward_strategy)
    env = SFObservationWrapper(env, obs_mode)
    env = Monitor(env)
    return env


def StreetFighterRenderEnv(obs_mode=2, skip=0, stack=10, allow_parallel=1, scenario='scenario', state='Champion.Level1.RyuVsGuile', hit_reward_strategy=2, mode='rgb_array', save_frames=True):
    if mode == 'human' and allow_parallel:
        print("You can't have mode human and in parallel sadly. Changing to rgb_array and saving frame so you can use the frame later")
        mode = 'rgb_array'
        save_frames = True
    def wrap_env():
        return StreetFighterEnv(obs_mode, skip, scenario, state, hit_reward_strategy, mode=mode, save_frames=save_frames)
    if allow_parallel:
        env = make_vec_env(wrap_env, n_envs=1, vec_env_cls=SubprocVecEnv)
    else:
        env = wrap_env()
        env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, stack, channels_order='last')
    return env


def create_env(obs_mode=2, skip=0, stack=10, scenario='scenario', state='Champion.Level1.RyuVsGuile', hit_reward_strategy=2, mode='rgb_array', save_frames=False, n_envs=6, **kwargs):
    """
    Create the environmnet for training.

    """
    def wrap_env():
        return StreetFighterEnv(obs_mode=obs_mode, skip=skip, scenario=scenario, state=state, hit_reward_strategy=hit_reward_strategy, mode=mode, save_frames=save_frames)
    print(f'Training on {n_envs} environments')
    if n_envs > 1:
        env = make_vec_env(wrap_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    else:
        env = wrap_env()
        env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, stack, channels_order='last')
    return env
