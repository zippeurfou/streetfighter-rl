from gym.spaces import MultiBinary,  Discrete
from gym import ActionWrapper
from constant import GameInfo
import numpy as np


# Stole this from https://github.com/corbosiny/AIVO-StreetFigherReinforcementLearning/blob/master/src/Discretizer.py
class Discretizer(ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.
    Args:
        combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        self._combos = combos
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = Discrete(len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()

    def get_action_meaning(self, act):
        return self._combos[act]


class StreetFighter2Discretizer(Discretizer):
    """
    Use Street Fighter 2
    based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    """

    def __init__(self, env):
        super().__init__(env=env, combos=GameInfo.COMBOS.value)


