from gym import Env, Wrapper, ActionWrapper
import numpy as np
from constant import GameInfo
import cv2
from gym.spaces import Dict, Box

# Had to use a wrapper and not obs wrapper because I need access to the step info


class SFObservationWrapper(Wrapper):
    def __init__(self, env, obs_mode=0):
        """
        Street Fighter observation wrapper.
        It return a Dict of img observation (resize and grayscaled) depending of the mode and also a Box with the different game info
        :param obs_mode int: 0 -> difference between current frame and previous, 1 -> current frame, 2 -> dstack both 0 and 1
        """

        super().__init__(env)
        self.env = env
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
        # 5. my_position y
        # 6. enemy position x
        # 7. enemy_position y
        # 8. distance x between me and the enemy
        # 9. distance y between me and enemy
        # I have to be carefull to keep this as is
        self.vector_length = 9
        self.obs_mode = obs_mode
        self.default_game_info = np.array([1, 1, 0, (GameInfo.XPOS1_START.value - GameInfo.XPOS_MIN.value) / (GameInfo.XPOS_MAX.value - GameInfo.XPOS_MIN.value),
                                           0, (GameInfo.XPOS2_START.value - GameInfo.XPOS_MIN.value) / (GameInfo.XPOS_MAX.value - GameInfo.XPOS_MIN.value), 0,
                                           GameInfo.XDISTANCE_START.value / GameInfo.XDISTANCE_MAX.value, 0],
                                          dtype=np.float64)
        # space info
        # health go from -1 to 1, -1 if he is full life and I am at 0, 1 otherwise
        self.observation_space = Dict({'game_info': Box(-1, 1, (self.vector_length,), dtype=np.float64)})
        img_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        # need python 3.8 so no switch case
        # return 2 img if mode = 2 so we can apply 2 different CNN.
        if self.obs_mode == 0 or self.obs_mode == 2:
            self.observation_space['img0'] = img_space
        if self.obs_mode == 1 or self.obs_mode == 2:
            self.observation_space['img1'] = img_space

    def reset(self):
        # Return the first frame
        game_img = self.env.reset()
        self.previous_frame = self.rescale_gray(game_img)
        out_obs = self.preprocess_img_obs(game_img)
        out_obs['game_info'] = self.default_game_info
        return out_obs

    def rescale_gray(self,game_img):
        # Grayscaling
        gray = cv2.cvtColor(game_img, cv2.COLOR_BGR2GRAY)
        # Resize
        resize = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)
        # Add the channels value
        channels = np.reshape(resize, (84, 84, 1))
        return channels

    def preprocess_img_obs(self, game_img):
        channels = self.rescale_gray(game_img)
        out_img = {}
        if self.obs_mode == 0 or self.obs_mode == 2:
            out_img['img0'] = channels - self.previous_frame
        if self.obs_mode == 1 or self.obs_mode == 2:
            out_img['img1'] = channels
        # update previous img
        self.previous_frame = channels
        return out_img

    def preprocess_game_info(self, info):
        # Vector is:
        # 1. my current health
        # 2. enemy health
        # 3. difference my health - enemy health
        # 4. my position x
        # 5. my_position y
        # 6. enemy position x
        # 7. enemy_position y
        # 8. distance x between me and the enemy
        # 9. distance y between me and enemy
        return np.array([np.clip(info['health'], 0, GameInfo.MAX_HEALTH.value) / GameInfo.MAX_HEALTH.value,  # 1. my health
                         np.clip(info['enemy_health'], 0, GameInfo.MAX_HEALTH.value) / GameInfo.MAX_HEALTH.value,  # 2.  enemy health
                         (np.clip(info['health'], 0, GameInfo.MAX_HEALTH.value) - np.clip(info['enemy_health'], 0, GameInfo.MAX_HEALTH.value)) / GameInfo.MAX_HEALTH.value,  # 3. health diff
                         (np.clip(info['xpos1'], GameInfo.XPOS_MIN.value, GameInfo.XPOS_MAX.value) - GameInfo.XPOS_MIN.value) / (GameInfo.XPOS_MAX.value - GameInfo.XPOS_MIN.value),  # 4. p1 position x
                         np.clip(info['ypos1'], 0, GameInfo.YPOS_MAX.value) / GameInfo.YPOS_MAX.value,  # 5. p1 position y
                         (np.clip(info['xpos2'], GameInfo.XPOS_MIN.value, GameInfo.XPOS_MAX.value) - GameInfo.XPOS_MIN.value) / (GameInfo.XPOS_MAX.value - GameInfo.XPOS_MIN.value),  # 6.  p2 position x
                         np.clip(info['ypos2'], 0, GameInfo.YPOS_MAX.value) / GameInfo.YPOS_MAX.value,  # 6.  p2 position y
                         np.clip(info['xdist'], 0, GameInfo.XDISTANCE_MAX.value) / GameInfo.XDISTANCE_MAX.value,  # 7. players x distances
                         np.clip(info['ydist'], 0, GameInfo.YDISTANCE_MAX.value) / GameInfo.YDISTANCE_MAX.value  # 8. players y distances
                         ], dtype=np.float64)

    def step(self, action):
        game_img, reward, done, info = self.env.step(action)
        out_obs = self.preprocess_img_obs(game_img)
        out_obs['game_info'] = self.preprocess_game_info(info)
        return out_obs, reward, done, info
