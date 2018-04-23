import os
import random
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python import keras as K
from tensorflow.python.keras._impl.keras.models import clone_model
from PIL import Image
import gym
import gym_ple
from fn_framework import FNAgent, Trainer, Experience


class DeepQNetworkAgent(FNAgent):

    def __init__(self, epsilon, actions):
        super().__init__(epsilon, actions)
        self._updater = None
        self._teacher_model = None

    def initialize(self, experiences, optimizer):
        feature_shape = experiences[0].s.shape
        self.make_model(feature_shape)
        self.model.compile(optimizer, loss="mse")
        self.initialized = True
        print("Done initialize. From now, begin training!")

    def make_model(self, feature_shape):
        model = K.Sequential()
        model.add(self._conv(filters=32, kernel_size=8, strides=4,
                             input_shape=feature_shape))
        model.add(self._conv(filters=64, kernel_size=4, strides=2))
        model.add(self._conv(filters=64, kernel_size=3, strides=1))
        model.add(K.layers.Flatten())
        model.add(self._liner(512, activation="relu"))
        model.add(self._liner(len(self.actions)))
        self.model = model
        self._teacher_model = clone_model(self.model)

    def _conv(self, filters, kernel_size, strides, input_shape=(None,)):
        return K.layers.Conv2D(
            filters, kernel_size=kernel_size, strides=strides,
            input_shape=input_shape, padding="same",
            kernel_initializer="normal", activation="relu")

    def _liner(self, size, activation=None):
        return K.layers.Dense(size, kernel_initializer="normal",
                              activation=activation)

    def estimate(self, state):
        return self.model.predict(np.array([state]))[0]

    def update(self, experiences, gamma):
        states = np.array([e.s for e in experiences])
        estimateds = self.model.predict(states)

        next_states = np.array([e.n_s for e in experiences])
        future_estimates = self._teacher_model.predict(next_states)

        for i, e in enumerate(experiences):
            reward = e.r
            if not e.d:
                reward += gamma * np.max(future_estimates[i])
            estimateds[i][e.a] = reward

        loss = self.model.train_on_batch(states, estimateds)
        return loss

    def update_teacher(self):
        self._teacher_model = clone_model(self.model)


class Observer():

    def __init__(self, env, width, height, frame_count):
        self._env = env
        self.width = width
        self.height = height
        self.frame_count = frame_count
        self._frames = []

    @property
    def action_space(self):
        return self._env.action_space

    def reset(self):
        return self.transform(self._env.reset())

    def render(self):
        self._env.render()

    def step(self, action):
        n_state, reward, done, info = self._env.step(action)
        return self.transform(n_state), reward, done, info

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


class DeepQNetworkTrainer(Trainer):

    def __init__(self, log_dir="", file_name=""):
        super().__init__(log_dir)
        self.file_name = file_name if file_name else "dqn_agent.h5"
        self.final_epsilon = 0.0001
        self.epsilon_decay = 1e-6
        self.training_count = 0
        self.loss = []
        self.callback = K.callbacks.TensorBoard(self.log_dir)

    def train(self, env, episode_count=2000, gamma=0.99,
              epsilon=0.0001, epsilon_decay=1e-6,
              buffer_size=50000, batch_size=32, teacher_update_freq=10,
              render=False, report_interval=10):
        if not isinstance(env, Observer):
            raise Exception("Environment have to be wrapped by Observer")

        actions = list(range(env.action_space.n))
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.final_epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.teacher_update_freq = teacher_update_freq
        self.training_count = 0
        self.report_interval = report_interval
        agent = DeepQNetworkAgent(1.0, actions)

        self.train_loop(env, agent, episode_count, render)
        return agent

    def episode_begin(self, episode_count, agent):
        self.loss = []

    def buffer_full(self, episode_count, agent):
        optimizer = K.optimizers.Adam(lr=1e-6)
        agent.initialize(self.experiences, optimizer)
        self.callback.set_model(agent.model)

    def step(self, episode_count, step_count, agent, experience):
        if agent.initialized:
            batch = random.sample(self.experiences, self.batch_size)
            loss = agent.update(batch, self.gamma)
            self.loss.append(loss)
            agent.epsilon = max(agent.epsilon - self.epsilon_decay,
                                self.final_epsilon)

    def episode_end(self, episode_count, step_count, agent):
        reward = sum([e.r for e in self.experiences[-step_count:]])
        self.reward_log.append(reward)
        if agent.initialized:
            self.write_log(self.training_count, np.mean(self.loss), reward)
            if self.is_event(self.training_count, self.report_interval):
                agent.save(os.path.join(self.log_dir, self.file_name))
            if self.is_event(self.training_count, self.teacher_update_freq):
                agent.update_teacher()
            self.training_count += 1

        if self.is_event(episode_count, self.report_interval):
            recent_rewards = self.reward_log[-self.report_interval:]
            desc = self.make_desc("reward", recent_rewards)
            print("At episode {}, {}".format(episode_count, desc))

    def write_log(self, index, loss, score):
        for name, value in zip(("loss", "score"), (loss, score)):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.callback.writer.add_summary(summary, index)
            self.callback.writer.flush()


def main(play):
    env = gym.make("Catcher-v0")
    obs = Observer(env, 80, 80, 4)
    trainer = DeepQNetworkTrainer(file_name="dqn_agent.h5")
    path = os.path.join(trainer.log_dir, trainer.file_name)

    if play:
        agent = DeepQNetworkAgent.load(env, path)
        agent.play(obs, render=True)
    else:
        trainer.train(obs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN Agent")
    parser.add_argument("--play", action="store_true",
                        help="play with trained model")

    args = parser.parse_args()
    main(args.play)
