from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video, HParam
from stable_baselines3.common.utils import get_latest_run_id
from utils import flatten_dict
from environment import StreetFighterRenderEnv
from typing import Any, Dict
import torch as th
import gym
import numbers
import os
import glob



class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env: gym.Env, render_freq: int, n_eval_episodes: int = 1, deterministic: bool = True):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            screens = []

            def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, recording the screen in the captured `screens` list

                :param _locals: A dictionary containing all local variables of the callback's scope
                :param _globals: A dictionary containing all global variables of the callback's scope
                """
                screen = self._eval_env.render(mode="rgb_array")
                # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                screens.append(screen.transpose(2, 0, 1))

            evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )
            self.logger.record(
                "trajectory/video",
                Video(th.ByteTensor([screens]), fps=40),
                exclude=("stdout", "log", "json", "csv"),
            )
        return True


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = os.path.join(save_path, f'checkpoints_{get_latest_run_id(save_path)+1}')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'checkpointcall_{}'.format(self.n_calls))
            self.model.save(model_path)
        return True




class HParamCallback(BaseCallback):
    def __init__(self, args):
        """
        Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
        """
        super().__init__()
        self.args = args

    def _on_training_start(self) -> None:
        info = self.model.__dict__
        model_params = flatten_dict(dict(filter(lambda elem: isinstance(elem[1], numbers.Number) or isinstance(elem[1], str) or isinstance(elem[1], dict), info.items())))

        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            **model_params, **info
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True



def create_callbacks(check_freq=10000, checkpoint_dir='./checkpoints/', obs_mode=2, skip=4, stack=20, **kwargs):
    args = flatten_dict(locals())
    callbacks = []
    callbacks.append(TrainAndLoggingCallback(check_freq=check_freq, save_path=checkpoint_dir))
    eval_env = StreetFighterRenderEnv(obs_mode, skip, stack)
    callbacks.append(VideoRecorderCallback(eval_env, check_freq))
    callbacks.append(HParamCallback(args=args))
    return callbacks
