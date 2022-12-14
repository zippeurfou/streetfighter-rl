import gym
import pygame
import numpy as np


# slightly quick edit from gym human randering to make it work with my sf2 environment.
class RetroHumanRendering(gym.Wrapper):

    def __init__(self, env, mode='human', save_frames=True):
        """Initialize a :class:`HumanRendering` instance.
        Args:
            env: The environment that is being wrapped
        """
        super().__init__(env)
        # ignoring for now
        # assert env.render_mode in [
        #     "rgb_array",
        #     "rgb_array_list",
        # ], f"Expected env.render_mode to be one of 'rgb_array' or 'rgb_array_list' but got '{env.render_mode}'"
        # assert (
        #     "render_fps" in env.metadata
        # ), "The base environment must specify 'render_fps' to be used with the HumanRendering wrapper"

        self.screen_size = None
        self.window = None
        self.clock = None
        self.save_frames = save_frames
        self.mode = mode

    @property
    def render_mode(self):
        """Always returns ``'human'``."""
        return "human"

    def step(self, *args, **kwargs):
        """Perform a step in the base environment and render a frame to the screen."""
        result = self.env.step(*args, **kwargs)
        if self.mode == 'human':
            self._render_frame()
        # elif self.mode == 'rgb_array':
        #     frame = self.env.render(mode='rgb_array')
        if self.save_frames:
            self.frames_history.append(self.env.render(mode='rgb_array'))
        return result

    def reset(self, *args, **kwargs):
        """Reset the base environment and render a frame to the screen."""
        result = self.env.reset(*args, **kwargs)
        self.frames_history = []
        if self.mode == 'human':
            self._render_frame()
        # elif self.mode == 'rgb_array':
        #     frame = self.env.render(mode='rgb_array')
        if self.save_frames:
            self.frames_history.append(self.env.render(mode='rgb_array'))

        return result

    def render(
        self, *args, **kwargs
    ):
        """Renders the environment."""
        if 'human' in args or 'human' in kwargs.values():
            print("can't render human mode here")
            return None
        elif 'rgb_array' in args or 'rgb_array' in kwargs.values():
            return self.env.render(*args, **kwargs)
        else:
            print(f'unknown type {args} , {kwargs}')
            return None

    def _render_frame(self):
        """Fetch the last frame from the base environment and render it to the screen."""
        # try:
        #     import pygame
        # except ImportError:
        #     raise DependencyNotInstalled(
        #         "pygame is not installed, run `pip install gym[box2d]`"
        #     )
        # if self.env.render_mode == "rgb_array_list":
        #     last_rgb_array = self.env.render()
        #     assert isinstance(last_rgb_array, list)
        #     last_rgb_array = last_rgb_array[-1]
        # elif self.env.render_mode == "rgb_array":
        #     last_rgb_array = self.env.render()
        # else:
        #     raise Exception(
        #         f"Wrapped environment must have mode 'rgb_array' or 'rgb_array_list', actual render mode: {self.env.render_mode}"
        #     )
        # ignoring the rest for now
        last_rgb_array = self.env.render(mode='rgb_array')
        assert isinstance(last_rgb_array, np.ndarray)

        rgb_array = np.transpose(last_rgb_array, axes=(1, 0, 2))
        if self.save_frames:
            self.frames_history.append(rgb_array)

        if self.screen_size is None:
            self.screen_size = rgb_array.shape[:2]

        assert (
            self.screen_size == rgb_array.shape[:2]
        ), f"The shape of the rgb array has changed from {self.screen_size} to {rgb_array.shape[:2]}"

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.screen_size)

        if self.clock is None:
            self.clock = pygame.time.Clock()

        surf = pygame.surfarray.make_surface(rgb_array)
        self.window.blit(surf, (0, 0))
        pygame.event.pump()
        # self.clock.tick(self.metadata["render_fps"])
        # metadata has something else for me.
        self.clock.tick(self.metadata['video.frames_per_second'])
        pygame.display.flip()

    def close(self):
        """Close the rendering window."""
        super().close()
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
