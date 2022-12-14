from gym import Wrapper
from constant import GameInfo

# skip bunch of frame
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

# skip frame until you can do something
class WaitForActionableState(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        self.currentJumpFrame = 0
        return self.env.reset()

    def should_skip(self, act=GameInfo.NO_ACTION.value, info={}):
        action = self.env.get_action_meaning(act)
        # import pdb; pdb.set_trace()
        if len(info) == 0:
            # should not happend but just safety
            return True
        if info['timer'] == GameInfo.TIMER_START.value:
            return True
        elif info['status1'] == GameInfo.JUMPING_STATUS.value and self.currentJumpFrame <= GameInfo.JUMP_LAG.value:
            self.currentJumpFrame += 1
            return True
        elif info['status1'] == GameInfo.JUMPING_STATUS.value and any([button in action for button in GameInfo.ACTION_BUTTONS.value]):
            return True
        elif info['status1'] not in GameInfo.ACTIONABLE_STATUSES.value:
            return True
        else:
            if info['status1'] != GameInfo.JUMPING_STATUS.value and self.currentJumpFrame > 0:
                self.currentJumpFrame = 0
            return False

    def step(self, act):
        done = None
        total_rew = 0
        obs, rew, done, info = self.env.step(act)
        if done:
            return obs, total_rew, done, info
        else:
            while self.should_skip(act, info):
                # print('debugging')
                # print(info)
                # self.env.render()
                # import time
                # time.sleep(0.02)
                # if  I should skip then do no action in the meantime
                obs, rew, done, info = self.env.step([GameInfo.NO_ACTION.value])
                total_rew += rew
                if done:
                    break
        # self.env.render()
        # import time
        # time.sleep(0.02)
        # print('freee')
        return obs, total_rew, done, info
