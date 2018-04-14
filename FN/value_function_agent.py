import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import gym
from estimator import Estimator
from fn_agent import FNAgent


class ValueFunction(Estimator):

    def __init__(self, actions, gamma):
        super().__init__(actions)
        self.gamma = gamma
        self.reset()

    def initialize(self, experiences):
        scaler = StandardScaler()
        estimator = MLPRegressor(
                        hidden_layer_sizes=(10, 10),
                        max_iter=1)
        self.model = Pipeline([("scaler", scaler), ("estimator", estimator)])

        features = np.vstack([self.to_feature(e.s) for e in experiences])
        self.model.named_steps["scaler"].fit(features)

        # Avoid the predict before fit. Use a little sample to fit.
        self.update(experiences[:2])
        self.initialized = True
        print("Done initialize. From now, begin training!")

    def estimate(self, s):
        feature = self.to_feature(s)
        estimated = self.model.predict(feature)[0]
        return estimated

    def to_feature(self, s):
        # CartPole state is ...
        # position, speed, angle, angle_speed = s
        feature = np.array(s).reshape((1, -1))
        return feature

    def update(self, experiences):
        states = []
        estimateds = []

        for e in experiences:
            s, es = self._make_label_data(e)
            states.append(s)
            estimateds.append(es)

        states = np.vstack(states)
        estimateds = np.array(estimateds)
        states = self.model.named_steps["scaler"].transform(states)
        self.model.named_steps["estimator"].partial_fit(states, estimateds)

    def _make_label_data(self, e):
        # Calculate Reward
        reward = e.r
        if not e.d:
            if self.initialized:
                future = self.estimate(e.n_s)
            else:
                future = np.random.uniform(size=len(self.actions))
            reward = e.r + self.gamma * np.max(future)

        # Correct model estimation by gained reward
        if self.initialized:
            estimated = self.estimate(e.s)
        else:
            estimated = np.random.uniform(size=len(self.actions))

        estimated[e.a] = reward

        # Update Model
        state = self.to_feature(e.s)
        return state, estimated


class ValueFunctionAgent(FNAgent):

    def __init__(self, epsilon=0.1):
        super().__init__(epsilon)

    def learn(self, env, episode_count=200, gamma=0.9,
              buffer_size=1024, batch_size=32,
              render=False, report_interval=10):
        actions = list(range(env.action_space.n))
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.estimator = ValueFunction(actions, gamma)

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

            if e != 0 and e % report_interval == 0:
                self.show_reward_log(interval=report_interval, episode=e)


def train():
    agent = ValueFunctionAgent()
    env = gym.make("CartPole-v1")
    agent.learn(env, render=False)
    agent.show_reward_log()


if __name__ == "__main__":
    train()
