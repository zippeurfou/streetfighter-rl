# get the rom
import retro
# for delta and other help function
import numpy as np
# cv2 for image manipulation
import cv2
# Import environment base class for a wrapper
from gym import Env, Wrapper
# Import the space shapes for the environment
from gym.spaces import MultiBinary, Box, Dict
# Wrapp it so I can monitor and stack frames
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
# env in parallel
from retrowrapper import RetroWrapper  # No need
# for the custom game
import os
# vec in parallel
from stable_baselines3.common.env_util import make_vec_env
# for game reset
from collections import deque
# Constant used for the game
MAX_HEALTH = 176
FPS = 65
MAX_ROUND_SEC = 99
XPOS1_START = 205
XPOS2_START = 307
XPOS_MIN = 55
XPOS_MAX = 457
TIMER_START = 9923
YDISTANCE_MIN = 0
YDISTANCE_MAX = 67
XDISTANCE_MIN = 0
XDISTANCE_MAX = 187
XDISTANCE_START = 79
REWARDS_COEF = {'my_health': 1,
                'enemy_health': 2,  # Gives more weight to attacking the enemy vs losing health -> I want to be aggressive
                'round_won': 3,
                'round_lost': 3,
                'game_lost': 4,
                'fight_won': 4,
                'time_decay': 0,  # Ignore this, need to think about it a bit more, the formula ain't good
                'timer_decay': 0.001  # I don't want to be penalizing too much here. Still If I don't do it he like to just defend lol
                }


class StreetFighter(Env):
    def __init__(self, obs_mode=0, use_retro_wrapper=0, scenario='scenario', state='1stars', rewards_coef=REWARDS_COEF, hit_strategy=2):
        """
        Street Fighter environment

        :param obs_mode int: 0 -> difference between current frame and previous, 1 -> current frame, 2 -> dstack both 0 and 1
        :param use_retro_wrapper use_retro_wrapper int: 0 = No, 1 = Yes. Used to test performance improvement during training otherwise I use SubprocVecEnv
        :param scenario str: what scenario to use. This can help for later playing manually vs the AI for example
        :param state str: what state to use. List is in the custom rom file. This helps to fight in different setting (eg. different enemies)
        :param rewards_coef dict: check REWARD_COEF. Gives more weight to different kind of rewards
        :param hit_strategy int: 0 -> health delta by punch/kick, 1 -> when hit, delta of life remaining, 3 -> avg 0 and 1
        """

        super().__init__()
        # Specify action space and observation space
        # if obs_mode = 0 then difference
        # if 1 then original
        # if 2 then concatenation of both
        # the other part is:
        # Vector is:
        # 1. my current health
        # 2. enemy health
        # 3. difference my health - enemy health
        # 4. my position x
        # 5. enemy position x
        # 6. distance x between me and the enemy
        # 7. distance y between me and enemy
        # I have to be carefull to keep this as is
        self.vector_length = 7
        self.obs_mode = obs_mode
        self.coef = rewards_coef
        self.hit_strategy = hit_strategy
        # health go from -1 to 1, -1 if he is full life and I am at 0, 1 otherwise
        self.observation_space = Dict({'game_info': Box(-1, 1, (self.vector_length,), dtype=np.float64)})
        self.default_info = np.array([1, 1, 0, (XPOS1_START - XPOS_MIN) / (XPOS_MAX - XPOS_MIN), (XPOS2_START - XPOS_MIN) / (XPOS_MAX - XPOS_MIN), XDISTANCE_START / XDISTANCE_MAX, 0], dtype=np.float64)

        if self.obs_mode < 2:
            self.observation_space['img'] = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        else:
            self.observation_space['img'] = Box(low=0, high=255, shape=(84, 84, 2), dtype=np.uint8)
        self.action_space = MultiBinary(12)
        # Startup and instance of the game
        if use_retro_wrapper:
            self.game = RetroWrapper(game="sf2", inttype=retro.data.Integrations.ALL, use_restricted_actions=retro.Actions.FILTERED, scenario=scenario, state=state)
        else:
            self.game = retro.make(game="sf2", inttype=retro.data.Integrations.ALL, use_restricted_actions=retro.Actions.FILTERED, scenario=scenario, state=state)

    def reset(self):
        # Return the first frame
        img = self.game.reset()
        img = self.preprocess_img_obs(img)
        self.previous_frame = img
        # if mode 2 then dstack them
        if self.obs_mode == 2:
            img = np.dstack((img, img))
        self.n_steps = self.round_steps = self.round_fight = 0
        self.enemy_matches_won = self.matches_won = self.fight_won = 0
        self.my_health = self.enemy_health = MAX_HEALTH
        # Create a attribute to hold the score delta
        self.score = 0
        # nice thing is that the timer go down at each round so we can use it to skip frame that we don't care
        self.previous_timer = TIMER_START
        obs = {'img': img, 'game_info': self.default_info}
        return obs

    def preprocess_img_obs(self, observation):
        # Grayscaling
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        # Resize
        resize = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)
        # Add the channels value
        channels = np.reshape(resize, (84, 84, 1))
        # normalize
        # get it between 0 and 1
        # edit no need this get normalized for me.
        # channels = channels / 255.0
        return channels

    def preprocess_game_info(self, info):
        # Vector is:
        # 1. my current health
        # 2. enemy health
        # 3. difference my health - enemy health
        # 4. my position x
        # 5. enemy position x
        # 6. distance x between me and the enemy
        # 7. distance y between me and enemy
        return np.array([np.clip(info['health'], 0, MAX_HEALTH) / MAX_HEALTH,  # my health
                         np.clip(info['enemy_health'], 0, MAX_HEALTH) / MAX_HEALTH,  # enemy health
                         (np.clip(info['health'], 0, MAX_HEALTH) - np.clip(info['enemy_health'], 0, MAX_HEALTH)) / MAX_HEALTH,  # health diff
                         (np.clip(info['xpos1'], XPOS_MIN, XPOS_MAX) - XPOS_MIN) / (XPOS_MAX - XPOS_MIN),  # p1 position
                         (np.clip(info['xpos2'], XPOS_MIN, XPOS_MAX) - XPOS_MIN) / (XPOS_MAX - XPOS_MIN),  # p2 position
                         np.clip(info['xdist'], 0, XDISTANCE_MAX) / XDISTANCE_MAX,  # players x distances
                         np.clip(info['ydist'], 0, YDISTANCE_MAX) / YDISTANCE_MAX  # players y distances
                         ], dtype=np.float64)

    def preprocess_img(self, observation):
        # Grayscaling
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        # Resize
        resize = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)
        # Add the channels value
        channels = np.reshape(resize, (84, 84, 1))
        # normalize
        # get it between 0 and 1
        # edit no need this get normalized for me.
        # channels = channels / 255.0
        return channels

    def health_reward_calc(self, new_health, old_health):
        # doing some cleaning just in case..
        new_health = np.clip(new_health, 0, MAX_HEALTH)
        old_health = np.clip(old_health, 0, MAX_HEALTH)
        # if nothing happen, no reward
        # also if reset no reward
        # if I somehow have more health now, it's a bug so no reward
        if new_health == old_health or old_health == 0 or new_health > old_health or new_health == MAX_HEALTH:
            return 0
        out = []
        out.append((new_health - old_health) / old_health)
        out.append(-1 + new_health / MAX_HEALTH)
        out.append((out[0] + out[1]) / 2)
        return out[self.hit_strategy]

    def step(self, action):
        # Take a step
        # print('ACTION')
        # print(action)
        img, reward, done, info = self.game.step(action)
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
        # TODO check when -1 happen for both
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
            # finished because no more time
            # since we skip / stack and have some error from time to time with timer just use self.previous timer for safety
            if info['timer'] == 0 and self.previous_timer < 100:
                if info['enemy_health'] < info['health']:
                    if self.matches_won == 0:
                        won_round = 1
                    else:
                        self.fight_won += 1
                        won_round = 1
                        won_fight = 1
                else:
                    if self.enemy_matches_won == 0:
                        lost_round = 1
                    else:
                        lost_round = 1
                        lost_fight = 1
                        done = True

        # skip until next round
        # pretty much a hack, I was not able to figure out a variable is_playing in sf2
        if (won_round or lost_round or won_fight or self.previous_timer == info['timer']) and done is False:
            # want to skip and ignore
            sinfo = info
            # have a patience of 3 because somehow timer sometimes goes back to 0 as follow:  end timer -> 0 -> new timer -> start timer
            # didn't really need dequeue here..
            timer_buffer = deque([self.previous_timer] * 3, maxlen=3)
            unique_val = 1
            while sinfo['health'] < 1 or sinfo['enemy_health'] < 1 or unique_val < 4:
                if sinfo['timer'] not in timer_buffer:
                    timer_buffer.append(sinfo['timer'])
                    unique_val += 1
                simg, sreward, done, sinfo = self.game.step(action)
                # print(f'info: previous {self.previous_timer}, timer {sinfo["timer"]},  enemy_health {sinfo["enemy_health"]}, health {sinfo["health"]}, buffer {timer_buffer}, unique_val {unique_val},  done: {done}')
                # should never happen but just in case
                if done:
                    print('should never happen, terminated between round')
                    break

        img = self.preprocess_img_obs(img)
        game_info = self.preprocess_game_info(info)

        # Frame delta
        if self.obs_mode == 0:
            out_img = img - self.previous_frame
        elif self.obs_mode == 1:
            out_img = img
        elif self.obs_mode == 2:
            out_img = img - self.previous_frame
            out_img = np.dstack((img, out_img))

        # doing the preprocessing after so we don't do difference of difference..
        # Might be worth a try at some point maybe another mode
        self.previous_frame = img
        self.previous_timer = info['timer']
        # format out obs for the model
        out_obs = {'img': out_img, 'game_info': game_info}

        ## REWARDS ##
        # Reshape the reward function
        # score_delta = info['score'] - self.score
        # not using the score as:
        # 1. It increase even between round which makes the network thinks it can help
        # 2. It has a bias toward longer matches as you get point even if you lose round

        matches_won_delta = won_round
        enemy_matches_won_delta = lost_round
        # Was toying around with decreasing reward as the round goes up as well as the fight
        # and increasing as you play longer which mean you're still alive
        # The challenge here is to normalize it correctly to provide the right reward prioritization
        # Additionally because you don't hit or get hit all frames it can provide information for the model hard to understand
        # total amount of steps in the episode
        self.n_steps += 1
        # total amount of steps in the current round
        self.round_steps += 1
        # total amount of steps in the current fight
        self.round_fight += 1

        # collect the different subreward in the info so it is easier to debut
        # losing health is negative (delta is always <= 0 )
        # % change delta/old_health. So the closer it get to 0 the highest the reward will be
        info['r_my_health'] = self.health_reward_calc(new_health=info['health'], old_health=self.my_health)
        # enemy losing health is positive
        info['r_enemy_health'] = - self.health_reward_calc(new_health=info['enemy_health'], old_health=self.enemy_health)
        # if I win the round I add +1
        info['r_match_won'] = matches_won_delta
        # If I win the fight I add +1
        info['r_won_fight'] = won_fight
        # if I lose the round -1
        info['r_enemy_won'] = - enemy_matches_won_delta
        # if I lose the the fight -1
        info['r_enemy_won_fight'] = - lost_fight
        # Tried to add some rule there but ignoring it for now
        info['r_time_decay'] = - (-((self.n_steps) * 0.00000000001) +
                                  ((self.round_steps) / (MAX_ROUND_SEC * FPS)) +
                                  ((self.round_fight) / (MAX_ROUND_SEC * FPS * 3))
                                  ) / 300
        info['r_timer_decay'] = -1 + (TIMER_START - np.clip(info['timer'], 0, TIMER_START)) / TIMER_START
        reward = self.coef['my_health'] * info['r_my_health'] + \
            self.coef['enemy_health'] * info['r_enemy_health'] + \
            self.coef['round_won'] * info['r_match_won'] + \
            self.coef['round_lost'] * info['r_enemy_won'] + \
            self.coef['game_lost'] * info['r_enemy_won_fight'] + \
            self.coef['fight_won'] * info['r_won_fight'] + \
            self.coef['time_decay'] * info['r_time_decay'] + \
            self.coef['timer_decay'] * info['r_timer_decay']
        # the range should be between -3 and +3 ignoring the time component and the coeficients
        # If I win a fight (+1) I win a round (+1) and I give the hitting blow (+1) -> 3
        # So doing a small normalization to have the total reward beeing between -1 and 1
        reward = reward / (
            np.max((self.coef['my_health'], self.coef['enemy_health'])) +
            np.max((self.coef['round_won'], self.coef['round_lost'])) +
            np.max((self.coef['fight_won'], self.coef['game_lost'])) +
            self.coef['timer_decay']
        )
        # if reward != 0:
        # print(f'in game reward norm: {reward}, original: {reward*3}')

        # info for current round could be used if we wanted different kind of rewards added
        self.score = info['score']
        self.my_health = max(info['health'], 0)
        self.enemy_health = max(info['enemy_health'], 0)
        self.matches_won += won_round
        self.enemy_matches_won += lost_round
        self.fight_won += won_fight

        # updating info for next round
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
            # if rew != 0:
            # print(f'got a reward {rew}, total {total_rew}')
            if done:
                break
        return obs, total_rew, done, info


def StreetFighterEnv(obs_mode=2, skip=4, use_retro_wrapper=0):
    init_game()
    env = StreetFighter(obs_mode, use_retro_wrapper)
    env = Frameskip(env, skip)
    env = Monitor(env)
    return env


def init_game():
    retro.data.Integrations.add_custom_path(
        os.path.join(os.getcwd(), "custom_integrations")
    )


def StreetFighterRenderEnv(obs_mode=2, skip=4, stack=10):
    env = StreetFighterEnv(obs_mode, skip)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, stack, channels_order='last')
    return env


def create_env(obs_mode=2, skip=4, stack=10, n_envs=15, use_retro_wrapper=0, **kwargs):
    def wrap_env():
        return StreetFighterEnv(obs_mode, skip, use_retro_wrapper)
    print(f'Training on {n_envs} environments')
    if n_envs > 1:
        env = make_vec_env(wrap_env, n_envs=n_envs, vec_env_cls=DummyVecEnv if use_retro_wrapper else SubprocVecEnv)
    else:
        env = wrap_env()
        env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, stack, channels_order='last')
    return env
