import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces


class FrozenLake(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.size = 6
        self.window_size = 256

        self.observation_space = spaces.Discrete(36)
        self.action_space = spaces.Discrete(4)

        self.Ns = 36
        self.Na = 4

        self.states = range(36)
        self.actions = range(4)

        self.initial_state = 0
        self.goal_state = 33
        self.terminal_states = [4, 6, 14, 25, 27, 28, 33]

        self.reward = [-1 if i not in self.terminal_states else -100 for i in self.states]
        self.reward[self.goal_state] = 0

        # given state and action returns the new state or -1 if state was a terminal state
        self.move = np.zeros((36, 4))

        self.move[:, 0] = [i + 1 if i % 6 != 5 else i for i in self.states] # go right
        self.move[:, 1] = [i + 6 if i < 30 else i for i in self.states]     # go down
        self.move[:, 2] = [i - 1 if i % 6 != 0 else i for i in self.states] # go left
        self.move[:, 3] = [i - 6 if i > 5 else i for i in self.states]      # go up

        for s in self.terminal_states:
            self.move[s] = np.full(4, -1)

        self.move = self.move.astype(int)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        return self._location
    
    def _get_info(self):
        return None
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.window = None

        self._location = self.initial_state

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def get_deterministic_state(self, state, action):
        new_state = self.move[state, action]
        return new_state
    
    def step(self, action):

        new_location = self.get_deterministic_state(self._location, action)

        if new_location != -1:
            self._location = new_location
        else:
            return self._get_obs(), 0, True, False, None
        
        terminated = True if new_location in self.terminal_states else False
        reward = self.reward[new_location]
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
    
    def step_no_actions(self, new_state):

        # check if it is a valid movement
        if new_state not in self.move[self._location]:
            return None, 0, True, False, None
        
        actions = np.where(self.move[self._location] == new_state)
        action = actions[0][0]

        self._location = new_state
        
        terminated = True if new_state in self.terminal_states else False
        reward = self.reward[new_state]
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return action, reward, terminated, False, info
    
    def _state_to_position(self, s):
        x = s % self.size
        y = s // self.size
        return np.array([x, y], dtype=int)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((245, 255, 254))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the goal
        pygame.draw.rect(
            canvas,
            (250, 250, 120),
            pygame.Rect(
                pix_square_size * self._state_to_position(self.goal_state),
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 0),
            (self._state_to_position(self._location) + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        for h in self.terminal_states:
            if h != self.goal_state:
                pygame.draw.rect(
                    canvas,
                    (151, 207, 205),
                    pygame.Rect(
                        pix_square_size * self._state_to_position(h),
                        (pix_square_size, pix_square_size),
                    ),
                )


        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=2,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=2,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()