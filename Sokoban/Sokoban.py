import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces


class Sokoban(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, render_fps=4, level=1):
        self.size = 5
        self.window_size = 256
        self.metadata["render_fps"] = render_fps

        self.observation_space = spaces.Dict({'agent': spaces.Discrete(20), 
                                             'box1': spaces.Discrete(20), 
                                             'box2': spaces.Discrete(20)})
        self.action_space = spaces.Discrete(4)

        self.Ns = 20 ** 3
        self.Na = 4

        self.states = [(a, b1, b2) for a in range(20) for b1 in range(20) for b2 in range(20)]
        self.actions = range(4)

        self.initial_state = (2, 5, 9)
        
        if level == 2:
            self.goals = (10, 12)
            self.terminal = (0, 1, 2, 6, 11, 15, 16, 17, 18, 19)
        elif level == 3:
            self.goals = (4, 12)
            self.terminal = (0, 1, 2, 6, 10, 11, 14, 15, 16, 17, 18, 19)
        else:
            self.goals = (10, 18)
            self.terminal = (0, 1, 2, 6, 11, 15, 19)

        self.walls = [np.array(p) for p in ((0, 0), (1, 0), (4, 0), (4, 1), (1, 3))]
        self.state_to_position = [np.array(p) for p in [(2, 0), (3, 0), 
                                                        (0, 1), (1, 1), (2, 1), (3, 1),
                                                        (0, 2), (1, 2), (2, 2), (3, 2), (4, 2), 
                                                        (0, 3), (2, 3), (3, 3), (4, 3),
                                                        (0, 4), (1, 4), (2, 4), (3, 4), (4, 4)]]
        self.position_to_state = [[None, 2, 6, 11, 15], 
                                  [None, 3, 7, None, 16], 
                                  [0, 4, 8, 12, 17], 
                                  [1, 5, 9, 13, 18],
                                  [None, None, 10, 14, 19]]
        
        self.action_to_direction = [np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1])]

        
        self.reward_matrix = np.full((20, 20, 20), -2)
        for state in self.states:
            self.reward_matrix[state] = self.reward(*state)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        return self._agent, self._box1, self._box2
    
    def _get_info(self):
        return None
    
    def reward(self, a, b1, b2):
        # box in terminal position
        if b1 in self.terminal or b2 in self.terminal:
            return -500
        # no boxes in place
        elif b1 not in self.goals and b2 not in self.goals:
            return -2
        # both boxes
        elif b1 in self.goals and b2 in self.goals:
            return 0
        # one box in place
        else:
            return -1
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.window = None

        self._agent, self._box1, self._box2 = self.initial_state

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def move(self, state, action):

        agent, box1, box2 = state
        agent_pos = self.state_to_position[agent]
        box1_pos = self.state_to_position[box1]
        box2_pos = self.state_to_position[box2]

        # check it doesnt get of limits
        new_agent_pos = np.clip(agent_pos + self.action_to_direction[action], 0, self.size - 1)

        # check if it is in a corner
        is_corner = np.array_equal(new_agent_pos, agent_pos)
        # check it doesnt hit a wall
        is_wall = any(np.array_equal(new_agent_pos, arr) for arr in self.walls)

        if is_wall or is_corner:
            return state
        
        # check if there is a box
        if np.array_equal(box1_pos, new_agent_pos):
            # box 1 in the way
            new_box1_pos = np.clip(box1_pos + self.action_to_direction[action], 0, self.size - 1)

            # check if it is in a corner
            is_corner = np.array_equal(new_box1_pos, box1_pos)
            # check it doesnt hit a wall
            is_wall = any(np.array_equal(new_box1_pos, arr) for arr in self.walls)

            if is_wall or is_corner:
                return state
            
            # check if there is the other box in there
            if np.array_equal(new_box1_pos, box2_pos):
                # box 2 in the way
                new_box2_pos = np.clip(box2_pos + self.action_to_direction[action], 0, self.size - 1)

                # check if it is in a corner
                is_corner = np.array_equal(new_box2_pos, box2_pos)
                # check it doesnt hit a wall
                is_wall = any(np.array_equal(new_box2_pos, arr) for arr in self.walls)

                if is_wall or is_corner:
                    return state
                
                # box 2 can move
                a, b = new_box2_pos
                box2 = self.position_to_state[a][b]

            # box 1 can move
            a, b = new_box1_pos
            box1 = self.position_to_state[a][b]

        elif np.array_equal(box2_pos, new_agent_pos):
            # box 2 in the way
            new_box2_pos = np.clip(box2_pos + self.action_to_direction[action], 0, self.size - 1)

            # check if it is in a corner
            is_corner = np.array_equal(new_box2_pos, box2_pos)
            # check it doesnt hit a wall
            is_wall = any(np.array_equal(new_box2_pos, arr) for arr in self.walls)

            if is_wall or is_corner:
                return state
            
            # check if there is the other box in there
            if np.array_equal(new_box2_pos, box1_pos):
                # box 1 in the way
                new_box1_pos = np.clip(box1_pos + self.action_to_direction[action], 0, self.size - 1)

                # check if it is in a corner
                is_corner = np.array_equal(new_box1_pos, box1_pos)
                # check it doesnt hit a wall
                is_wall = any(np.array_equal(new_box1_pos, arr) for arr in self.walls)

                if is_wall or is_corner:
                    return state
                
                # box 1 can move
                a, b = new_box1_pos
                box1 = self.position_to_state[a][b]

            # box 2 can move
            a, b = new_box2_pos
            box2 = self.position_to_state[a][b]
        
        # just move agent
        a, b = new_agent_pos
        agent = self.position_to_state[a][b]

        if box1 < box2:
            return (agent, box1, box2)
        else:
            return (agent, box2, box1)

    
    def step(self, action):

        new_state = self.move(self._get_obs(), action)
        self._agent, self._box1, self._box2 = new_state

        # wins
        if self.goals[0] == self._box1 and self.goals[1] == self._box2:
            terminated = True
            won = True
        # loses
        elif self._box1 in self.terminal or self._box2 in self.terminal:
            terminated = True
            won = False
        # continues
        else:
            terminated = False
            won = False
        
        reward = self.reward(*new_state)
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, won, info
    
    def step_no_actions(self, new_state):

        # check if it is a valid movement
        valid = False
        for a in self.actions:
            ns = self.move(self._get_obs(), a)
            if np.array_equal(np.array(ns), np.array(new_state)):
                action = a
                valid = True

        if not valid:
            return None, 0, True, False, None

        self._agent, self._box1, self._box2 = new_state

        # wins
        if self.goals[0] == self._box1 and self.goals[1] == self._box2:
            terminated = True
            won = True
        # loses
        elif self._box1 in self.terminal or self._box2 in self.terminal:
            terminated = True
            won = False
        # continues
        else:
            terminated = False
            won = False
        
        reward = self.reward(*new_state)
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return action, reward, terminated, won, info

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

        # First we draw the goals
        for goal in self.goals:
            pygame.draw.rect(
                canvas,
                (250, 250, 120),
                pygame.Rect(
                    pix_square_size * self.state_to_position[goal],
                    (pix_square_size, pix_square_size),
                ),
            )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 0),
            (self.state_to_position[self._agent] + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        for w in self.walls:
            pygame.draw.rect(
                canvas,
                (76, 103, 102),
                pygame.Rect(
                    pix_square_size * w,
                    (pix_square_size, pix_square_size),
                ),
            )

        pygame.draw.rect(
            canvas,
            (200, 120, 100),
            pygame.Rect(
                pix_square_size * self.state_to_position[self._box1] + 6,
                (pix_square_size - 10, pix_square_size - 10),
            ),
        )

        pygame.draw.rect(
            canvas,
            (200, 120, 100),
            pygame.Rect(
                pix_square_size * self.state_to_position[self._box2]  + 6,
                (pix_square_size - 10, pix_square_size - 10),
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