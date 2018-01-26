from enum import Enum


class GridMove(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class Environment():

    def __init__(self, grid):
        self.grid = grid
        self.agent_position = []
        # Default reward is minus like poison swamp.
        # It means agent have to reach the goal fast!
        self.default_reward = -0.04
        self.reset()
    
    def reset(self):
        # Locate agent at lower left corner
        self.agent_position = [len(self.grid) - 1, 0]
        return self.agent_position

    def action_space(self):
        return [GridMove.UP, GridMove.DOWN, 
                GridMove.LEFT, GridMove.RIGHT]

    def step(self, action):
        previous = list(self.agent_position)
        reward = self.default_reward
        done = False

        # Move the agent
        if action == GridMove.UP:
            self.agent_position[0] -= 1
        if action == GridMove.DOWN:
            self.agent_position[0] += 1
        if action == GridMove.LEFT:
            self.agent_position[1] -= 1
        if action == GridMove.RIGHT:
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
