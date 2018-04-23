from collections import namedtuple
import os
import random
import numpy as np
from tensorflow.python import keras as K
import matplotlib.pyplot as plt


Experience = namedtuple("Experience",
                        ["s", "a", "r", "n_s", "d"])


class FNAgent():

    def __init__(self, epsilon, actions):
        self.epsilon = epsilon
        self.actions = actions
        self.model = None
        self.estimate_probs = False
        self.initialized = False

    @classmethod
    def load(cls, env, model_path, epsilon=0.0001):
        actions = list(range(env.action_space.n))
        agent = cls(epsilon, actions)
        agent.model = K.models.load_model(model_path)
        agent.initialized = True
        return agent

    def save(self, model_path):
        self.model.save(model_path, overwrite=True, include_optimizer=False)

    def initialize(self, experiences):
        raise Exception("You have to implements estimate method")

    def estimate(self, s):
        raise Exception("You have to implements estimate method")

    def update(self, experiences, gamma):
        raise Exception("You have to implements update method")

    def policy(self, s):
        if np.random.random() < self.epsilon or not self.initialized:
            return np.random.randint(len(self.actions))
        else:
            estimates = self.estimate(s)
            if self.estimate_probs:
                action = np.random.choice(self.actions,
                                          size=1, p=estimates)[0]
                return action
            else:
                return np.argmax(estimates)

    def play(self, env, episode_count=5, render=True):
        for e in range(episode_count):
            s = env.reset()
            done = False
            episode_reward = 0
            while not done:
                if render:
                    env.render()
                a = self.policy(s)
                n_state, reward, done, info = env.step(a)
                episode_reward += reward
                s = n_state
            else:
                print("Get reward {}".format(episode_reward))


class Trainer():

    def __init__(self, log_dir=""):
        self.buffer_size = 0
        self.batch_size = 0
        self.gamma = 0.99
        self.experiences = []
        self.reward_log = []
        self.report_interval = 1
        self.log_dir = log_dir
        if not self.log_dir:
            self.log_dir = os.path.join(os.path.dirname(__file__), "logs")

    def make_path(self, file_name):
        return os.path.join(self.log_dir, file_name)

    def train_loop(self, env, agent, episode_count=200, render=False):
        _env = self.make_train_env(env)
        self.experiences = []
        self.reward_log = []

        for i in range(episode_count):
            s = _env.reset()
            done = False
            step_count = 0
            self.episode_begin(i, agent)
            while not done:
                if render:
                    _env.render()
                a = agent.policy(s)
                n_state, reward, done, info = env.step(a)

                e = Experience(s, a, reward, n_state, done)
                self.experiences.append(e)
                if len(self.experiences) == self.buffer_size:
                    self.buffer_full(i, agent)
                elif len(self.experiences) > self.buffer_size:
                    self.experiences.pop(0)

                self.step(i, step_count, agent, e)

                s = n_state
                step_count += 1
            else:
                self.episode_end(i, step_count, agent)

    def make_train_env(self, env):
        return env

    def episode_begin(self, episode_count, agent):
        pass

    def buffer_full(self, episode_count, agent):
        pass

    def step(self, episode_count, step_count, agent, experience):
        pass

    def episode_end(self, episode_count, step_count, agent):
        pass

    def is_event(self, episode_count, interval):
        if episode_count != 0 and episode_count % interval == 0:
            return True
        else:
            return False

    def make_desc(self, name, values):
        mean = np.round(np.mean(values), 3)
        std = np.round(np.std(values), 3)
        desc = "{} is {} (+/-{})".format(name, mean, std)
        return desc

    def plot_logs(self, name, values, interval=10):
        indices = list(range(0, len(values), interval))
        means = []
        stds = []
        for i in indices:
            rewards = self.reward_log[i:(i + interval)]
            means.append(np.mean(rewards))
            stds.append(np.std(rewards))
        means = np.array(means)
        stds = np.array(stds)
        plt.figure()
        plt.title("{} History".format(name))
        plt.grid()
        plt.fill_between(indices, means - stds, means + stds,
                         alpha=0.1, color="g")
        plt.plot(indices, means, "o-", color="g",
                 label="Rewards for each {} episode".format(interval))
        plt.legend(loc="best")
        plt.show()
