import random
import math
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import gym


class MonteCarloAgent():

    def __init__(self, epsilon=0.1):
        self.Q = {}
        self.epsilon = epsilon

    def policy(self, s, actions):
        if random.random() < self.epsilon:
            return random.choice(actions)
        else:
            if s in self.Q:
                return np.argmax(self.Q[s])
            else:
                return random.choice(actions)

    def learn(self, env, episode_count=100000, gamma=0.99,
              render=False, report_interval=500):
        self.Q = defaultdict(lambda: [0] * len(actions))
        N = defaultdict(lambda: [0] * len(actions))
        actions = list(range(env.action_space.n))
        report_count = episode_count // report_interval

        rewards = []
        for e in range(episode_count):
            s = env.reset()
            # Gain experience
            experience = []
            while True:
                if render:
                    env.render()
                a = self.policy(s, actions)
                n_state, reward, done, info = env.step(a)
                experience.append(
                    {"state": s, "action": a, "reward": reward}
                )
                rewards.append(reward)
                s = n_state
                if done:
                    break

            for i, x in enumerate(experience):
                s = x["state"]
                a = x["action"]

                # Calculate discounted future reward
                G = 0
                t = 0
                for j in range(i, len(experience)):
                    G += math.pow(gamma, t) * experience[j]["reward"]
                    t += 1

                # Count visits
                N[s][a] += 1
                alpha = 1 / N[s][a]
                # Update
                self.Q[s][a] += alpha * (G - self.Q[s][a])

            if e != 0 and e % report_interval == 0:
                mean = np.round(np.mean(rewards), 3)
                std = np.round(np.std(rewards), 3)
                report_no = e // report_interval
                print("[{}/{}] Now Reward avg is {} (+/-{}).".format(
                    report_no, report_count, mean, std))
                rewards = []

    def play(self, env, episode_count=3, render=False):
        actions = list(range(env.action_space.n))
        self.show_q()
        for e in range(episode_count):
            s = env.reset()
            acquired = 0

            while True:
                if render:
                    env.render()
                a = self.policy(s, actions)
                n_state, reward, done, info = env.step(a)
                acquired += reward
                if done:
                    break

            print("Episode {}: get reward {}.".format(e, acquired))

    def show_q(self):
        for s in self.Q:
            print("At {}, expecteds are {}.".format(s, self.Q[s]))


def main():
    agent = MonteCarloAgent(epsilon=0.1)
    env = gym.make("FrozenLake-v0")
    agent.learn(env, gamma=1, episode_count=500000)

    # Show
    nrow = env.unwrapped.nrow
    ncol = env.unwrapped.ncol
    scale = 3
    map_row = nrow * scale
    map_col = ncol * scale
    reward_map = np.zeros((map_row, map_col))
    for r in range(nrow):
        for c in range(ncol):
            s = r * nrow + c
            if s in agent.Q:
                # In the display map, vertical index reverse.
                _r = 1 + (nrow - 1 - r) * scale
                _c = 1 + c * scale
                # LEFT = 0
                reward_map[_r][_c - 1] = agent.Q[s][0]
                # DOWN = 1
                reward_map[_r - 1][_c] = agent.Q[s][1]
                # RIGHT = 2
                reward_map[_r][_c + 1] = agent.Q[s][2]
                # UP = 3
                reward_map[_r + 1][_c] = agent.Q[s][3]
                # Center
                reward_map[_r][_c] = np.mean(agent.Q[s])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(reward_map, cmap="Greens", interpolation="nearest")
    ax.set_xlim(-0.5, map_col - 0.5)
    ax.set_ylim(-0.5, map_row - 0.5)
    ax.set_xticks(np.arange(-0.5, map_col, scale))
    ax.set_yticks(np.arange(-0.5, map_row, scale))
    ax.set_xticklabels(range(ncol + 1))
    ax.set_yticklabels(range(nrow + 1))
    ax.grid(which="both")
    plt.show()


if __name__ == "__main__":
    main()
