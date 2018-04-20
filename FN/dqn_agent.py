import os
import numpy as np
from tensorflow.python import keras as K
from tensorflow.python.keras._impl.keras.models import clone_model
from PIL import Image
import gym
import gym_ple
from estimator import Estimator
from fn_agent import FNAgent


class DeepQNetwork(Estimator):

    def __init__(self, actions, gamma):
        super().__init__(actions)
        self.gamma = gamma
        self._teacher_model = None
        self.reset()

    def initialize(self, experiences):
        feature_shape = experiences[0].s.shape
        self.set_estimator(feature_shape)
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
        self._teacher_model = clone_model(self.model)

    def estimate(self, state):
        return self.model.predict(np.array([state]))[0]

    def update(self, experiences):
        states = np.array([e.s for e in experiences])
        estimateds = self.model.predict(states)

        next_states = np.array([e.n_s for e in experiences])
        future_estimates = self._teacher_model.predict(next_states)

        for i, e in enumerate(experiences):
            reward = e.r
            if not e.d:
                reward += self.gamma * np.max(future_estimates[i])
            estimateds[i][e.a] = reward

        self.model.train_on_batch(states, estimateds)

    def update_teacher(self):
        self._teacher_model = clone_model(self.model)


class Observer():

    def __init__(self, env, width, height, frame_count):
        self._env = env
        self.width = width
        self.height = height
        self.frame_count = frame_count
        self._frames = []

    def reset(self):
        return self.transform(self._env.reset())

    def render(self):
        self._env.render()

    def step(self, action, skip=0):
        for i in range(skip + 1):
            n_state, reward, done, info = self._env.step(action)
            n_state = self.transform(n_state)
        return n_state, reward, done, info

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


class DeepQNetworkAgent(FNAgent):

    def __init__(self, epsilon=0.0001, model_path=""):
        super().__init__(epsilon)
        self.model_path = model_path
        if not model_path:
            path = os.path.join(os.path.dirname(__file__), "logs")
            if not os.path.exists(path):
                os.mkdir(path)
            self.model_path = os.path.join(path, "dqn_model.h5")

    def learn(self, env, episode_count=800, gamma=0.99, epsilon_decay=0.995,
              buffer_size=65536, batch_size=32, teacher_update_freq=10,
              render=False, report_interval=10):
        actions = list(range(env.action_space.n))
        obs = Observer(env, 80, 80, 4)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.estimator = DeepQNetwork(actions, gamma)
        final_epsilon = self.epsilon
        self.epsilon = 1.0
        training_step = -1

        for e in range(episode_count):
            s = obs.reset()
            done = False
            episode_reward = 0
            skip = 0 if training_step < 0 else 4
            while not done:
                if render:
                    obs.render()
                a = self.policy(s)
                n_state, reward, done, info = obs.step(a, skip=skip)
                episode_reward += reward
                switched = self.feedback(s, a, reward, n_state, done)
                if switched:
                    adam = K.optimizers.Adam(lr=1e-6, clipvalue=1.0)
                    self.estimator.model.compile(optimizer=adam, loss="mse")
                    training_step = 0

                s = n_state
            else:
                self.log(episode_reward)

            if training_step >= 0:
                self.epsilon = max(self.epsilon * epsilon_decay, final_epsilon)
                if training_step % teacher_update_freq == 0:
                    self.estimator.update_teacher()
                if training_step % report_interval == 0:
                    self.estimator.model.save(self.model_path, overwrite=True)
                training_step += 1

            if e != 0 and e % report_interval == 0:
                self.show_reward_log(interval=report_interval, episode=e)


def train():
    agent = DeepQNetworkAgent()
    env = gym.make("Catcher-v0")
    agent.learn(env, render=False)
    agent.show_reward_log()


if __name__ == "__main__":
    train()
