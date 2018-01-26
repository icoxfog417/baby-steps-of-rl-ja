import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import random
from DP.model import Environment


def main():
    # Make grid maze environment
    grid = get_sample_grid()
    env = Environment(grid)

    # Try 100 game
    for i in range(100):
        # Initialize agent position
        position = env.reset()
        goal = False
        total_reward = 0

        # Agent can move up to 10
        for t in range(10):
            action = random.choice(env.action_space())
            position, reward, done = env.step(action)
            total_reward += reward
            if done:
                goal = True
                break
        
        print("Episode {}: Agent {}, get total reward {}.".format(i,
            "reached goal" if goal else "timed out",
            total_reward
            ))


def get_sample_grid():
    # 3 x 4 grid
    grid =  [
        [{'index': [0, 0], 'attribute': 0, 'rewards': [0, 0, 0, 0]}, {'index': [0, 1], 'attribute': 0, 'rewards': [0, 0, 0, 0]}, {'index': [0, 2], 'attribute': 0, 'rewards': [0, 0, 0, 0]}, {'index': [0, 3], 'attribute': 1, 'rewards': [0, 0, 0, 0]}],
        [{'index': [1, 0], 'attribute': 0, 'rewards': [0, 0, 0, 0]}, {'index': [1, 1], 'attribute': 9, 'rewards': [0, 0, 0, 0]}, {'index': [1, 2], 'attribute': 0, 'rewards': [0, 0, 0, 0]}, {'index': [1, 3], 'attribute': -1, 'rewards': [0, 0, 0, 0]}],
        [{'index': [2, 0], 'attribute': 0, 'rewards': [0, 0, 0, 0]}, {'index': [2, 1], 'attribute': 0, 'rewards': [0, 0, 0, 0]}, {'index': [2, 2], 'attribute': 0, 'rewards': [0, 0, 0, 0]}, {'index': [2, 3], 'attribute': 0, 'rewards': [0, 0, 0, 0]}]]
    return grid


if __name__ == "__main__":
    main()
