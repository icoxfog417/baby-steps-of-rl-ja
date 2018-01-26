import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import random
import unittest
from DP.model import Environment


class TestEnvironment(unittest.TestCase):

    def test_run_environment(self):
        grid = self._get_sample_grid()
        env = Environment(grid)
        for i in range(100):
            position = env.reset()  # initialize agent position
            self.assertEqual(position[0], len(env.grid) - 1)
            self.assertEqual(position[1], 0)
            goal = False
            for t in range(10):
                action = random.choice(env.action_space())
                position, reward, done = env.step(action)
                self.assertTrue(0 <= position[0] < len(env.grid))
                self.assertTrue(0 <= position[1] < len(env.grid[0]))
                if done:
                    print("Episode {}: get reward {}, {} timesteps".format(i, reward, t + 1))
                    goal = True
                    break
            if not goal:
                print("Episode {}: no reward".format(i))

    def _get_sample_grid(self):
        # 3 x 4 grid
        grid =  [
            [{'index': [0, 0], 'attribute': 0, 'rewards': [0, 0, 0, 0]}, {'index': [0, 1], 'attribute': 0, 'rewards': [0, 0, 0, 0]}, {'index': [0, 2], 'attribute': 0, 'rewards': [0, 0, 0, 0]}, {'index': [0, 3], 'attribute': 1, 'rewards': [0, 0, 0, 0]}],
            [{'index': [1, 0], 'attribute': 0, 'rewards': [0, 0, 0, 0]}, {'index': [1, 1], 'attribute': 9, 'rewards': [0, 0, 0, 0]}, {'index': [1, 2], 'attribute': 0, 'rewards': [0, 0, 0, 0]}, {'index': [1, 3], 'attribute': -1, 'rewards': [0, 0, 0, 0]}],
            [{'index': [2, 0], 'attribute': 0, 'rewards': [0, 0, 0, 0]}, {'index': [2, 1], 'attribute': 0, 'rewards': [0, 0, 0, 0]}, {'index': [2, 2], 'attribute': 0, 'rewards': [0, 0, 0, 0]}, {'index': [2, 3], 'attribute': 0, 'rewards': [0, 0, 0, 0]}]]
        return grid


if __name__ == "__main__":
    unittest.main()
