from enum import Enum
import numpy as np


class Direction(Enum):
    UP = 1
    DOWN = -1
    LEFT = 2
    RIGHT = -2


class Environment():

    def __init__(self, grid, move_prob=0.8):
        self.grid = grid
        self.agent_position = []
        
        # Default reward is minus like poison swamp.
        # It means agent have to reach the goal fast!
        self.default_reward = -0.04

        # Agent can move to decided direction in move_prob.
        # It means agent will move different direction in (1 - move_prob).
        self.move_prob = move_prob
        self.reset()
    
    def reset(self):
        # Locate agent at lower left corner
        self.agent_position = [len(self.grid) - 1, 0]
        return self.agent_position

    def action_space(self):
        return [Direction.UP, Direction.DOWN, 
                Direction.LEFT, Direction.RIGHT]

    def step(self, action):
        previous = list(self.agent_position)
        reward = self.default_reward
        done = False

        # Calculate action probability
        actions = self.action_space()
        opposite_direction = Direction(action.value * -1)
        action_probs = []
        for a in actions:
            prob = 0
            if a == action:
                prob = self.move_prob
            elif a != opposite_direction:
                prob = (1 - self.move_prob) / 2
            action_probs.append(prob)
        
        # Select action
        real_action = np.random.choice(actions, p=action_probs)

        # Move the agent
        if real_action == Direction.UP:
            self.agent_position[0] -= 1
        if real_action == Direction.DOWN:
            self.agent_position[0] += 1
        if real_action == Direction.LEFT:
            self.agent_position[1] -= 1
        if real_action == Direction.RIGHT:
            self.agent_position[1] += 1

        # Check out of grid
        if not (0 <= self.agent_position[0] < len(self.grid)):
            self.agent_position = previous
        if not (0 <= self.agent_position[1] < len(self.grid[0])):
            self.agent_position = previous
        
        # Check the agent bump the block
        cell_state = self.get_state()["attribute"]
        if cell_state == 1:
            # Get treasure! and game ends.
            reward = 1
            done = True
        elif cell_state == -1:
            # Go to hell! and the game ends.
            reward = -1
            done = True
        elif cell_state == 9:
            # Agent bumped the block
            self.agent_position = previous
        
        return self.agent_position, reward, done

    def get_state(self):
        return self._get_cell(*self.agent_position)

    def _get_cell(self, row, column):
        if 0 <= row < len(self.grid) and \
           0 <= column < len(self.grid[0]):
            return self.grid[row][column]
        else:
            print([row, column], [len(self.grid), len(self.grid[0])])
            return None
