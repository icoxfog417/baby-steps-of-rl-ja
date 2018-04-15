import numpy as np
import tensorflow as tf
from tensorflow.python import keras as K
from PIL import Image
import gym
from estimator import Estimator
from fn_agent import FNAgent


class Preprocessor():

    def __init__(self, width, height, frame_count):
        self.width = width
        self.height = height
        self.frame_count = frame_count
        self._frames = []

    def transform(self, state):
        grayed = Image.fromarray(state).convert("L")
        resized = grayed.resize((self.width, self.height))
        resized = np.array(resized).astype("float")
        normalized = resized / 255  # scale to 0~1
        if len(self._frames) == 0:
            self._frames = [normalized] * self.frame_count
        else:
            self._frames.append(normalized)
            self._frames.pop(0)

        feature = np.array(self._frames)
        # Convert the feature shape (f, w, h) => (w, h, f)
        feature = np.transpose(feature, (1, 2, 0))
        return feature


class DeepQNetwork(Estimator):

    def __init__(self, actions, gamma):
        super().__init__(actions)
        self.gamma = gamma
        self._fixed_model = None
        self.reset()

    def initialize(self, experiences):
        features = np.vstack([self.to_feature(e.s) for e in experiences])

        feature_shape = features[1].shape
        self.set_estimator(feature_shape)
        self.set_trainer()
        self.initialized = True
        print("Done initialize. From now, begin training!")

    def set_estimator(self, feature_shape):
        model = K.Sequential()
        model.add(K.layers.Conv2D(
            32, kernel_size=8, strides=4, padding="same",
            input_shape=feature_shape, kernel_initializer="normal",
            activation="relu"))
        model.add(K.layers.Conv2D(
            64, kernel_size=4, strides=2, padding="same",
            kernel_initializer="normal",
            activation="relu"))
        model.add(K.layers.Conv2D(
            64, kernel_size=3, strides=1, padding="same",
            kernel_initializer="normal",
            activation="relu"))
        model.add(K.layers.Flatten())
        model.add(K.layers.Dense(512, kernel_initializer="normal",
                                 activation="relu"))
        model.add(K.layers.Dense(len(self.actions),
                                 kernel_initializer="normal"))
        self.model = model

    def estimate(self, s, from_fixed=False):
        model = self.model if not from_fixed else self._fixed_model
        feature = self.to_feature(s)
        estimated = model.predict(feature)[0]
        return estimated

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
