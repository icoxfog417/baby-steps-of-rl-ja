import random
import unittest
from DP.environment import Environment
from DP.planner import ValueIterationPlanner, PolicyIterationPlanner


class TestPlanner(unittest.TestCase):

    def test_value_iteration(self):
        grid = self.get_sample_grid()
        env = Environment(grid)
        planner = ValueIterationPlanner(env)
        result = planner.plan()
        print("Value Iteration")
        for r in result:
            print(r)

    def test_policy_iteration(self):
        grid = self.get_sample_grid()
        env = Environment(grid)
        planner = PolicyIterationPlanner(env)
        result = planner.plan()
        print("Policy Iteration")
        for r in result:
            print(r)

    def get_sample_grid(self):
        # 3 x 4 grid
        grid = [
            [0, 0, 0, 1],
            [0, 9, 0, -1],
            [0, 0, 0, 0]
            ]
        return grid
