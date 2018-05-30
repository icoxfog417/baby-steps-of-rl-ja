from enum import Enum
import numpy as np


class Direction(Enum):
    UP = 1
    DOWN = -1
    LEFT = 2
    RIGHT = -2


class State():

    def __init__(self, row=-1, column=-1):
        self.row = row
        self.column = column

    def index(self, n_row):
        return self.row * n_row + self.column

    def __repr__(self):
        return "<State: [{}, {}]>".format(self.row, self.column)

    def clone(self):
        return State(self.row, self.column)

    def __hash__(self):
        return hash((self.row, self.column))

    def __eq__(self, other):
        return self.row == other.row and self.column == other.column


class Environment():

    def __init__(self, grid, move_prob=0.8):
        # Grid is 2d-array, and each value treated as attribute.
        # attribute is
        #  0: ordinary cell
        #  -1: damage cell (game end)
        #  1: reward cell (game end)
        #  9: block cell (can't locate agent)
        self.grid = grid
        self.agent_state = State()

        # Default reward is minus like poison swamp.
        # It means agent have to reach the goal fast!
        self.default_reward = -0.04

        # Agent can move to decided direction in move_prob.
        # It means agent will move different direction in (1 - move_prob).
        self.move_prob = move_prob
        self.reset()

    @property
    def row_length(self):
        return len(self.grid)

    @property
    def column_length(self):
        return len(self.grid[0])

    @property
    def action_space(self):
        return [Direction.UP, Direction.DOWN,
                Direction.LEFT, Direction.RIGHT]

    @property
    def states(self):
        states = []
        for row in range(self.row_length):
            for column in range(self.column_length):
                # Avoid the Block Cell
                if self.grid[row][column] != 9:
                    states.append(State(row, column))
        return states

    def reset(self):
        # Locate agent at lower left corner
        self.agent_state = State(self.row_length - 1, 0)
        return self.agent_state

    def step(self, action):
        next_state, reward, done = self.transit(self.agent_state, action)
        if next_state is not None:
            self.agent_state = next_state

        return next_state, reward, done

    def transit(self, state, action):
        transition_probs = self.transit_func(state, action)
        if len(transition_probs) == 0:
            return None, None, True

        next_states = []
        probs = []
        for s in transition_probs:
            next_states.append(s)
            probs.append(transition_probs[s])

        next_state = np.random.choice(next_states, p=probs)
        reward, done = self.reward_func(next_state)
        return next_state, reward, done

    def transit_func(self, state, action):
        transition_probs = {}
        if not self.can_action_at(state):
            # Already on the terminal cell
            return transition_probs

        actions = self.action_space
        opposite_direction = Direction(action.value * -1)

        for a in actions:
            prob = 0
            if a == action:
                prob = self.move_prob
            elif a != opposite_direction:
                prob = (1 - self.move_prob) / 2

            next_state = self._move(state, a)
            if next_state not in transition_probs:
                transition_probs[next_state] = prob
            else:
                transition_probs[next_state] += prob

        return transition_probs

    def can_action_at(self, state):
        if self.grid[state.row][state.column] == 0:
            return True
        else:
            return False

    def _move(self, state, action):
        if not self.can_action_at(state):
            raise Exception("Can't move from here!")

        next_state = state.clone()

        # Move state by action
        if action == Direction.UP:
            next_state.row -= 1
        elif action == Direction.DOWN:
            next_state.row += 1
        elif action == Direction.LEFT:
            next_state.column -= 1
        elif action == Direction.RIGHT:
            next_state.column += 1

        # Check the out of grid
        if not (0 <= next_state.row < self.row_length):
            next_state = state
        if not (0 <= next_state.column < self.column_length):
            next_state = state

        # Check the Agent bumped the block
        if self.grid[next_state.row][next_state.column] == 9:
            next_state = state

        return next_state

    def reward_func(self, state):
        reward = self.default_reward
        done = False

        # Check the attribute of next state
        attribute = self.grid[state.row][state.column]
        if attribute == 1:
            # Get treasure! and game ends.
            reward = 1
            done = True
        elif attribute == -1:
            # Go to hell! and the game ends.
            reward = -1
            done = True

        return reward, done
