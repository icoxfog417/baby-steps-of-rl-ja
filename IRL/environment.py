import numpy as np
from gym.envs.toy_text import discrete
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class GridWorldEnv(discrete.DiscreteEnv):

    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, grid, move_prob=0.8, default_reward=0.0):
        # grid is 2d-array, and each value treated as attribute.
        # attribute is
        #  0: ordinary cell
        #  -1: damage cell (game end)
        #  1: reward cell (game end)
        self.grid = grid
        if isinstance(grid, (list, tuple)):
            self.grid = np.array(grid)
        self._actions = {
            "LEFT": 0,
            "DOWN": 1,
            "RIGHT": 2,
            "UP": 3,
        }
        self.default_reward = default_reward
        self.move_prob = move_prob

        num_states = self.nrow * self.ncol
        num_actions = len(self._actions)

        # start from left down
        initial_state_prob = np.zeros(num_states)
        initial_state_prob[self.coordinate_to_state(self.nrow - 1, 0)] = 1.0

        # Make transitions
        P = {}

        for s in range(num_states):
            if s not in P:
                P[s] = {}

            reward = self.reward_func(s)
            done = self.has_done(s)
            if done:
                # Terminal state
                for a in range(num_actions):
                    P[s][a] = []
                    P[s][a].append([1.0, s, reward, done])
            else:
                for a in range(num_actions):
                    P[s][a] = []
                    transition_probs = self.transit_func(s, a)
                    for n_s in transition_probs:
                        reward = self.reward_func(n_s)
                        done = self.has_done(s)
                        P[s][a].append([transition_probs[n_s], n_s,
                                        reward, done])
        self.P = P
        super().__init__(num_states, num_actions, P, initial_state_prob)

    @property
    def nrow(self):
        return self.grid.shape[0]

    @property
    def ncol(self):
        return self.grid.shape[1]

    @property
    def shape(self):
        return self.grid.shape

    @property
    def actions(self):
        return list(range(self.action_space.n))

    @property
    def states(self):
        return list(range(self.observation_space.n))

    def state_to_coordinate(self, s):
        row, col = divmod(s, self.nrow)
        return row, col

    def coordinate_to_state(self, row, col):
        index = row * self.nrow + col
        return index

    def state_to_feature(self, s):
        feature = np.zeros(self.observation_space.n)
        feature[s] = 1.0
        return feature

    def transit_func(self, state, action):
        transition_probs = {}
        opposite_direction = (action + 2) % 4
        candidates = [a for a in range(len(self._actions))
                      if a != opposite_direction]

        for a in candidates:
            prob = 0
            if a == action:
                prob = self.move_prob
            else:
                prob = (1 - self.move_prob) / 2

            next_state = self._move(state, a)
            if next_state not in transition_probs:
                transition_probs[next_state] = prob
            else:
                transition_probs[next_state] += prob

        return transition_probs

    def reward_func(self, state):
        row, col = self.state_to_coordinate(state)
        reward = self.grid[row][col]
        return reward

    def has_done(self, state):
        row, col = self.state_to_coordinate(state)
        reward = self.grid[row][col]
        if np.abs(reward) == 1:
            return True
        else:
            return False

    def _move(self, state, action):
        next_state = state
        row, col = self.state_to_coordinate(state)
        next_row, next_col = row, col

        # Move state by action
        if action == self._actions["LEFT"]:
            next_col -= 1
        elif action == self._actions["DOWN"]:
            next_row += 1
        elif action == self._actions["RIGHT"]:
            next_col += 1
        elif action == self._actions["UP"]:
            next_row -= 1

        # Check the out of grid
        if not (0 <= next_row < self.nrow):
            next_row, next_col = row, col
        if not (0 <= next_col < self.ncol):
            next_row, next_col = row, col

        next_state = self.coordinate_to_state(next_row, next_col)

        return next_state

    def plot_on_grid(self, values):
        if len(values.shape) < 2:
            values = values.reshape(self.shape)
        fig, ax = plt.subplots()
        ax.imshow(values, cmap=cm.RdYlGn)
        ax.set_xticks(np.arange(self.ncol))
        ax.set_yticks(np.arange(self.nrow))
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    def test_grid():
        env = GridWorldEnv(grid=[
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], move_prob=1.0)
        s = env.reset()
        assert s == 12, "Start position is not left down"
        s, r, d, _ = env.step(0)  # Move to left wall
        assert s == 12, "Agent should be bumped to left wall"
        s, r, d, _ = env.step(1)  # Move to bottom wall
        assert s == 12, "Agent should be bumped to bottom wall"
        s, r, d, _ = env.step(2)  # Move to right
        assert s == 13, "Agent should go to right"
        s, r, d, _ = env.step(3)  # Move to up
        assert s == 9, "Agent should go to up"
        env.step(3)  # UP
        env.step(3)  # UP
        s, r, d, _ = env.step(0)  # LEFT
        assert d, "Agent should reache the goal"
        assert r == 1, "Agent should get reward"

    test_grid()
