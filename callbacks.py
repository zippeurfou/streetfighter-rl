from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video, HParam
from stable_baselines3.common.utils import get_latest_run_id
from stable_baselines3.common.logger import TensorBoardOutputFormat
from utils import flatten_dict
from environment import StreetFighterRenderEnv
from typing import Any, Dict
import torch as th
import numpy as np
import gym
import numbers
import os
import glob


class EvalEnvCallback(BaseCallback):
    def __init__(self, eval_env: gym.Env, render_freq: int, n_eval_episodes: int = 10, deterministic: bool = False):
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

    def _on_training_start(self):
        self._log_freq = 1000  # log every 1000 calls
        self.setup_writer()

    def setup_writer(self):
        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

        # get all the metadata that I want to track for my evaluation
        # do one step so I got the info filled I will reset it after
        self.eval_tracking = ['reward', 'episode_rewards', 'episode_lengths']
        _ = self._eval_env.reset()
        action = [self._eval_env.action_space.sample()]
        self._eval_env.step(action)
        self.eval_tracking.extend(["info_" + k for k in self._eval_env.get_attr('info')[0].keys()])
        self._eval_env.reset()
        layout_content = {}
        for m_name in self.eval_tracking:
            layout_content[m_name] = ['Margin', [f'eval/{m_name}/mean', f'eval/{m_name}/min', f'eval/{m_name}/max']]

        self.layout = {'eval': layout_content}
        self.tb_formatter.writer.add_custom_scalars(self.layout)
        self.tb_formatter.writer.flush()

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            tracked_info = {}

            for m_name in self.eval_tracking:
                tracked_info[m_name] = []

            def grab_metrics(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, recording the screen in the captured `screens` list

                :param _locals: A dictionary containing all local variables of the callback's scope
                :param _globals: A dictionary containing all global variables of the callback's scope
                """
                info_data = {}
                if 'info' in _locals.keys():
                    info_data = _locals['info']
                for m_name in self.eval_tracking:
                    if m_name.startswith('info_') and len(info_data) > 0:
                        tracked_info[m_name].append(info_data[m_name.replace('info_', '')])
                    else:
                        if m_name in _locals.keys():
                            tracked_info[m_name].append(_locals[m_name])

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_metrics,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
                return_episode_rewards=True,
            )

            tracked_info['episode_rewards'] = episode_rewards
            tracked_info['episode_lengths'] = episode_lengths
            for k, v in tracked_info.items():
                if len(v) == 0:
                    print(f'nothing for {k}')
                # import pdb;pdb.set_trace()
                self.tb_formatter.writer.add_scalar(f'eval/{k}/mean', np.mean(v) if len(v) > 0 else 0, self.num_timesteps)
                self.tb_formatter.writer.add_scalar(f'eval/{k}/min', np.min(v) if len(v) > 0 else 0, self.num_timesteps)
                self.tb_formatter.writer.add_scalar(f'eval/{k}/max', np.max(v) if len(v) > 0 else 0, self.num_timesteps)
            self.tb_formatter.writer.flush()
            # self.logger.record(
            #     "trajectory/video",
            #     Video(th.ByteTensor([screens]), fps=40),
            #     exclude=("stdout", "log", "json", "csv"),
            # )
        return True


class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env: gym.Env, render_freq: int, n_eval_episodes: int = 1, deterministic: bool = False):
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
            self.screens = []

            def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, recording the screen in the captured `screens` list

                :param _locals: A dictionary containing all local variables of the callback's scope
                :param _globals: A dictionary containing all global variables of the callback's scope
                """
                # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                s1 = self._eval_env.get_attr('frames_history')[0].copy()
                if len(s1) > len(self.screens):
                    self.screens = s1

            evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )
            vid_frames = [screen.transpose(2, 0, 1) for screen in self.screens]
            self.logger.record(
                "trajectory/video",
                Video(th.ByteTensor([vid_frames]), fps=40),
                exclude=("stdout", "log", "json", "csv"),
            )
        return True


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = os.path.join(save_path, f'checkpoints_{get_latest_run_id(save_path,"checkpoints")+1}')
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
        model_params = flatten_dict(dict(filter(lambda elem: isinstance(elem[1], numbers.Number) or isinstance(elem[1], str) or isinstance(elem[1], dict) and elem[0] not in ['_last_obs', 'observation_space'] and not elem[0].startswith("_"), info.items())))

        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            **model_params, **self.args
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
    eval_vid = StreetFighterRenderEnv(obs_mode=obs_mode, skip=skip, stack=stack, allow_parallel=0) # trick, lot of communication so not parallel.
    callbacks.append(VideoRecorderCallback(eval_vid, check_freq))
    eval_tb = StreetFighterRenderEnv(obs_mode=obs_mode, skip=skip, stack=stack, allow_parallel=1)
    callbacks.append(EvalEnvCallback(eval_tb, check_freq))
    callbacks.append(HParamCallback(args=args))
    return callbacks
