import random
import unittest
from DP.environment import Environment


class TestEnvironment(unittest.TestCase):

    def test_run_environment(self):
        grid = self.get_sample_grid()
        env = Environment(grid)
        for i in range(100):
            state = env.reset()  # initialize agent position
            self.assertEqual(state.row, len(env.grid) - 1)
            self.assertEqual(state.column, 0)
            goal = False
            for t in range(10):
                action = random.choice(env.actions)
                state, reward, done = env.step(action)
                self.assertTrue(0 <= state.row < len(env.grid))
                self.assertTrue(0 <= state.column < len(env.grid[0]))
                if done:
                    print("Episode {}: get reward {}, {} timesteps".format(
                        i, reward, t + 1))
                    goal = True
                    break
            if not goal:
                print("Episode {}: no reward".format(i))

    def get_sample_grid(self):
        # 3 x 4 grid
        grid = [
            [0, 0, 0, 1],
            [0, 9, 0, -1],
            [0, 0, 0, 0]
            ]
        return grid
