import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import gym
from estimator import Estimator
from fn_agent import FNAgent


class PolicyEstimator(Estimator):

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

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def estimate(self, s):
        evaluation = self.evaluate(s)
        action_probs = self.softmax(evaluation)
        return action_probs

    def evaluate(self, s):
        feature = self.to_feature(s)
        evaluation = self.model.predict(feature)[0]
        return evaluation

    def to_feature(self, s):
        # CartPole state is ...
        # position, speed, angle, angle_speed = s
        feature = np.array(s).reshape((1, -1))
        return feature

    def update(self, experiences):
        states = None
        evaluateds = None

        for e in experiences:
            s, ev = self._make_label_data(e)
            if states is None:
                states = s
                evaluateds = ev
            else:
                states = np.vstack((states, s))
                evaluateds = np.vstack((evaluateds, ev))

        states = self.model.named_steps["scaler"].transform(states)
        self.model.named_steps["estimator"].partial_fit(states, evaluateds)

    def _make_label_data(self, e):
        # Calculate Reward
        reward = e.r
        if not e.d:
            if self.initialized:
                future = self.evaluate(e.n_s)
            else:
                future = np.random.uniform(size=len(self.actions))
            reward = e.r + self.gamma * future[e.n_a]

        # Correct model estimation by gained reward
        if self.initialized:
            evaluation = self.evaluate(e.s)
        else:
            evaluation = np.random.uniform(size=len(self.actions))

        evaluation[e.a] = reward

        # Update Model
        state = self.to_feature(e.s)
        return state, evaluation


class PolicyGradientAgent(FNAgent):

    def __init__(self, epsilon=0.1):
        super().__init__(epsilon, under_policy=True)

    def learn(self, env, episode_count=200, gamma=0.9,
              buffer_size=1024, batch_size=32,
              render=False, report_interval=10):
        actions = list(range(env.action_space.n))
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.estimator = PolicyEstimator(actions, gamma)

        for e in range(episode_count):
            s = env.reset()
            done = False
            episode_reward = 0
            a = self.policy(s)
            while not done:
                if render:
                    env.render()
                n_state, reward, done, info = env.step(a)
                episode_reward += reward
                n_action = self.policy(n_state)
                self.feedback(s, a, reward, n_state, done, n_action)
                s = n_state
                a = n_action
            else:
                self.log(episode_reward)

            if e != 0 and e % report_interval == 0:
                self.show_reward_log(interval=report_interval, episode=e)


def train():
    agent = PolicyGradientAgent()
    env = gym.make("CartPole-v1")
    agent.learn(env, render=False)
    agent.show_reward_log()


if __name__ == "__main__":
    train()
