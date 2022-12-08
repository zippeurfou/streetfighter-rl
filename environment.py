# get the rom
import retro
# for delta and other help function
import numpy as np
# cv2 for image manipulation
import cv2
# Import environment base class for a wrapper
from gym import Env, Wrapper
# Import the space shapes for the environment
from gym.spaces import MultiBinary, Box
# Wrapp it so I can monitor and stack frames
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
# env in parallel
# from retrowrapper import RetroWrapper # No need
# vec in parallel
from stable_baselines3.common.env_util import make_vec_env
# Constant used for the game
MAX_HEALTH = 176
FPS = 65
MAX_ROUND_SEC = 99


class StreetFighter(Env):
    def __init__(self, obs_mode=0):
        super().__init__()
        # Specify action space and observation space
        # if obs_mode = 0 then difference
        # if 1 then original
        # if 2 then concatenation of both
        self.obs_mode = obs_mode
        if self.obs_mode < 2:
            self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        else:
            self.observation_space = Box(low=0, high=255, shape=(84, 84, 2), dtype=np.uint8)
        self.action_space = MultiBinary(12)
        # Startup and instance of the game
        self.game = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', use_restricted_actions=retro.Actions.FILTERED)
        # self.game = RetroWrapper(game='StreetFighterIISpecialChampionEdition-Genesis', use_restricted_actions=retro.Actions.FILTERED) # No need

    def reset(self):
        # Return the first frame
        obs = self.game.reset()
        obs = self.preprocess(obs)
        self.previous_frame = obs
        # if mode 2 then dstack them
        if self.obs_mode == 2:
            obs = np.dstack((obs, obs))
        self.n_steps = 0
        self.round_steps = 0
        self.round_fight = 0
        self.my_health = self.enemy_health = MAX_HEALTH
        self.enemy_matches_won = self.matches_won = 0
        self.fight_won = 0

        # Create a attribute to hold the score delta
        self.score = 0
        return obs

    def preprocess(self, observation):
        # Grayscaling
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        # Resize
        resize = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)
        # Add the channels value
        channels = np.reshape(resize, (84, 84, 1))
        return channels

    def step(self, action):
        # Take a step
        # print('ACTION')
        # print(action)
        obs, reward, done, info = self.game.step(action)
        # there are lot of things that we noticed
        # 1. score keep improving between round as bonus damage
        # 2. When someone die their life goes to -1, the frame take a bit of time to report it
        # In addition, there are a bunch of frame between round where your action is useless (we should aim to skip them)
        # Once it is back to be playable life of enemy and user go back to 0
        # round won get cancelled at every "match"
        # 3. We assume 99 seconds per round and 65 fps
        # let's add the logic manually because of that
        won_round = 0
        won_fight = 0
        lost_round = 0
        lost_fight = 0
        if not (info['enemy_health'] == -1 and info['health'] == -1):
            if info['enemy_health'] == -1:
                if self.matches_won == 0:
                    won_round = 1
                else:
                    self.fight_won += 1
                    won_round = 1
                    won_fight = 1

            if info['health'] == -1:
                if self.enemy_matches_won == 0:
                    lost_round = 1
                else:
                    lost_round = 1
                    lost_fight = 1
                    done = True

        # skip until next round
        if (won_round or lost_round) and done == False:
            sinfo = info
            while sinfo['enemy_health'] != 0 and sinfo['health'] != 0:
                _, _, done, sinfo = self.game.step(action)
                # should never happen but just in case
                if done:
                    print('should never happen, terminated between round')
                    break
            _, _, done, sinfo = self.game.step(action)

        obs = self.preprocess(obs)

        # Frame delta
        if self.obs_mode == 0:
            out_obs = obs - self.previous_frame
        elif self.obs_mode == 1:
            out_obs = obs
        elif self.obs_mode == 2:
            out_obs = obs - self.previous_frame
            out_obs = np.dstack((obs, out_obs))
        self.previous_frame = obs

        # Reshape the reward function
        # score_delta = info['score'] - self.score
        my_health_delta = min(max(info['health'], 0) - self.my_health, 0)
        enemy_health_delta = max(info['enemy_health'], 0) - self.enemy_health
        if info['enemy_health'] == MAX_HEALTH:
            enemy_health_delta = 0
        matches_won_delta = won_round
        enemy_matches_won_delta = lost_round
        self.n_steps += 1
        self.round_steps += 1
        self.round_fight += 1

        self.score = info['score']
        self.my_health = max(info['health'], 0)
        self.enemy_health = max(info['enemy_health'], 0)
        self.matches_won += won_round
        self.enemy_matches_won += lost_round
        self.fight_won += won_fight

        info['r_my_health'] = my_health_delta / MAX_HEALTH
        info['r_enemy_health'] = - enemy_health_delta / MAX_HEALTH
        info['r_match_won'] = matches_won_delta
        info['r_enemy_won'] = - enemy_matches_won_delta
        info['r_enemy_won_fight'] = - lost_fight
        info['r_won_fight'] = won_fight
        info['r_time_decay'] = - (-(self.n_steps * 0.00000000001) +
                                  (self.round_steps / (MAX_ROUND_SEC * FPS)) +
                                  (self.round_fight / (MAX_ROUND_SEC * FPS * 3))
                                  ) / 300
        reward = info['r_my_health'] + info['r_enemy_health'] + info['r_match_won'] + info['r_enemy_won'] + info['r_enemy_won_fight'] + info['r_won_fight']  # + info['r_time_decay']
        # if lost_round or won_round:
        #     print(f'At steps {self.n_steps}: won {won_round}, lost {lost_round}, reward {reward}')

        if won_round == 1 or lost_round == 1:
            self.round_steps = 0
        if won_fight == 1:
            self.round_fight = 0
            self.matches_won = 0
            self.enemy_matches_won = 0

        return out_obs, reward, done, info

    def render(self, *args, **kwargs):
        return self.game.render(*args, **kwargs)

    def close(self):
        self.game.close()


class Frameskip(Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def reset(self):
        return self.env.reset()

    def step(self, act):
        total_rew = 0.0
        done = None
        for i in range(self._skip):
            obs, rew, done, info = self.env.step(act)
            total_rew += rew
            if done:
                break
        return obs, total_rew, done, info


def StreetFighterEnv(obs_mode=2, skip=4):
    env = StreetFighter(obs_mode)
    env = Frameskip(env, skip)
    env = Monitor(env)
    return env


def StreetFighterRenderEnv(obs_mode=2, skip=4, stack=10):
    env = StreetFighterEnv(obs_mode, skip)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, stack, channels_order='last')
    return env


def create_env(obs_mode=2, skip=4, stack=10, n_envs=15):
    def wrap_env():
        return StreetFighterEnv(obs_mode, skip)
    env = make_vec_env(wrap_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    env = VecFrameStack(env, stack, channels_order='last')
    return env
