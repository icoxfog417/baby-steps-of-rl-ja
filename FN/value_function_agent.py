import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import gym
from estimator import Estimator
from fn_agent import FNAgent
gym.envs.register(
    id="MountainCarForever-v0",
    entry_point="gym.envs.classic_control:MountainCarEnv",
    max_episode_steps=4000
)
"""
https://github.com/dennybritz/reinforcement-learning/pull/107
"""


class ValueFunction(Estimator):

    def __init__(self, actions, gamma):
        super().__init__(actions)
        self.gamma = gamma
        self.reset()

    def initialize(self, experiences, batch_size):
        scaler = StandardScaler()
        estimator = MultiOutputRegressor(
                        SGDRegressor(max_iter=1000))
        self.model = Pipeline([("scaler", scaler), ("estimator", estimator)])

        features = np.vstack([self.to_feature(e.s) for e in experiences])
        self.model.named_steps["scaler"].fit(features)

        self.update(experiences[:batch_size])
        self.initialized = True
        self.update(experiences[batch_size:])
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

    def estimate(self, s):
        feature = self.to_feature(s)
        estimated = self.model.predict(feature)[0]
        return estimated

    def to_feature(self, s):
        position, velocity = s
        p2 = position ** 2
        v2 = velocity ** 2
        pv = position * velocity
        pv2 = p2 * v2
        f = np.array((position, velocity, p2, v2, pv, pv2)).reshape((1, -1))
        return f

    def update(self, experiences):
        states = None
        estimateds = None

        for e in experiences:
            s, es = self._make_x_y(e)
            if states is None:
                states = s
                estimateds = es
            else:
                states = np.vstack((states, s))
                estimateds = np.vstack((estimateds, es))

        states = self.model.named_steps["scaler"].transform(states)
        """
        print("State:")
        print(states)
        print("Estimated")
        print(estimateds)
        """
        self.model.named_steps["estimator"].partial_fit(states, estimateds)

    def _make_x_y(self, e):
        # Calculate Reward
        reward = e.r
        if not e.d:
            if self.initialized:
                future = self.estimate(e.s_n)
            else:
                future = np.random.uniform(size=len(self.actions))
            reward = e.r + self.gamma * np.max(future)

        # Fix Estimation
        if self.initialized:
            estimated = self.estimate(e.s)
        else:
            estimated = np.random.uniform(size=len(self.actions))

        estimated[e.a] = reward

        # Update Model
        state = self.to_feature(e.s)
        return state, estimated


class ValueFunctionAgent(FNAgent):

    def __init__(self, epsilon=0.0):
        super().__init__(epsilon)

    def learn(self, env, episode_count=100, gamma=0.9,
              buffer_size=12800, batch_size=16,
              render=False, report_interval=10):
        actions = list(range(env.action_space.n))
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.estimator = ValueFunction(actions, gamma)

        initialized = False
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
                self.feedback(s, a, reward, n_state, done)
                s = n_state
            else:
                self.log(episode_reward)

            if not initialized and self.estimator.initialized:
                print("Done initialize the model. Now begin training.")
                initialized = True

            if e != 0 and e % report_interval == 0:
                self.show_reward_log(interval=report_interval, episode=e)


def train():
    agent = ValueFunctionAgent()
    interval = 1
    env = gym.make("MountainCarForever-v0")
    agent.learn(env, report_interval=interval, render=False)
    agent.show_reward_log(interval)


if __name__ == "__main__":
    train()
