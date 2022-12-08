from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
import os
import glob

LOG_DIR = './logs/'
CHECKPOINT_DIR = './checkpoints/'


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = os.path.join(save_path, f'checkpoints_{get_latest_run_id()+1}')
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


def get_latest_run_id(log_path: str = CHECKPOINT_DIR, log_name: str = "checkpoints") -> int:
    """
    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.
    :param log_path: Path to the log folder containing several runs.
    :param log_name: Name of the experiment. Each run is stored
        in a folder named ``log_name_1``, ``log_name_2``, ...
    :return: latest run number
    """
    max_run_id = 0
    for path in glob.glob(os.path.join(log_path, f"{glob.escape(log_name)}_[0-9]*")):
        file_name = path.split(os.sep)[-1]
        ext = file_name.split("_")[-1]
        if log_name == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id


def train_model(env, total_timesteps=20000000):
    callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
    # model_params = {'n_steps': 8960, 'gamma': 0.906, 'learning_rate': 2e-07, 'clip_range': 0.369, 'gae_lambda': 0.891}
    # model_params = {'n_steps': 1024, 'gamma': 0.906, 'learning_rate': 2e-07, 'clip_range': 0.369, 'gae_lambda': 0.891}
    model_params = {'learning_rate': 2e-07}
    model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, **model_params)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(os.path.join(callback.save_path, 'checkpointcall_final'))
    env.close()
    return model
