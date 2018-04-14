from collections import namedtuple
import random
import numpy as np
import matplotlib.pyplot as plt


Experience = namedtuple("Experience",
                        ["s", "a", "r", "n_s", "d"])


class FNAgent():

    def __init__(self, epsilon, estimator=None,
                 under_policy=False):
        self.epsilon = epsilon
        self.estimator = estimator
        self.under_policy = under_policy
        self.buffer_size = 0
        self.batch_size = 0
        self.experience = []
        self.reward_log = []

    def reset(self):
        self.estimator.reset()
        self.experience = []
        self.reward_log = []

    def policy(self, s):
        actions = self.estimator.actions
        if np.random.random() < self.epsilon or \
           not self.estimator.initialized:
            return np.random.randint(len(actions))
        else:
            estimates = self.estimator.estimate(s)
            if self.under_policy:
                action = np.random.choice(actions, size=1, p=estimates)[0]
                return action
            else:
                return np.argmax(estimates)

    def feedback(self, s, a, r, n_s, d):
        e = Experience(s, a, r, n_s, d)
        self.experience.append(e)
        if self.estimator.initialized:
            self.experience.pop(0)  # Delete old experience
            batch = random.sample(self.experience, self.batch_size)
            self.estimator.update(batch)
        else:
            if len(self.experience) == self.buffer_size:
                self.estimator.initialize(self.experience)

    def log(self, reward):
        self.reward_log.append(reward)

    def show_reward_log(self, interval=10, episode=-1):
        if episode > 0:
            rewards = self.reward_log[-interval:]
            mean = np.round(np.mean(rewards), 3)
            std = np.round(np.std(rewards), 3)
            print("At Episode {} average reward is {} (+/-{}).".format(
                   episode, mean, std))
        else:
            indices = list(range(0, len(self.reward_log), interval))
            means = []
            stds = []
            for i in indices:
                rewards = self.reward_log[i:(i + interval)]
                means.append(np.mean(rewards))
                stds.append(np.std(rewards))
            means = np.array(means)
            stds = np.array(stds)
            plt.figure()
            plt.title("Reward History")
            plt.grid()
            plt.fill_between(indices, means - stds, means + stds,
                             alpha=0.1, color="g")
            plt.plot(indices, means, "o-", color="g",
                     label="Rewards for each {} episode".format(interval))
            plt.legend(loc="best")
            plt.show()
