import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import random
from DP.environment import Environment


class Agent():

    def __init__(self, env):
        self.actions = env.action_space

    def action(self, state):
        # This is Agent's Policy!
        return random.choice(self.actions)


def main():
    # Make grid maze environment
    grid = [
        [0, 0, 0, 1],
        [0, 9, 0, -1],
        [0, 0, 0, 0]
    ]
    env = Environment(grid)
    agent = Agent(env)

    # Try 100 game
    for i in range(100):
        # Initialize agent position
        state = env.reset()
        total_reward = 0

        # Agent can move up to 10
        for t in range(10):
            action = agent.action(state)
            state, reward, done = env.step(action)
            total_reward += reward
            if done:
                break

        print("Episode {}: Agent gets {} reward.".format(i, total_reward))


if __name__ == "__main__":
    main()
