import os
import argparse
import random
import numpy as np
import tensorflow as tf
from tensorflow.python import keras as K
from sklearn.preprocessing import StandardScaler
from PIL import Image
import gym
import gym_ple
from fn_framework import FNAgent, Trainer, Experience


class ActorCriticAgent(FNAgent):

    def __init__(self, epsilon, actions):
        super().__init__(epsilon, actions)
        self._updater = None

    def initialize(self, experiences, optimizer):
        feature_shape = experiences[0].s.shape
        self.make_model(feature_shape)
        self.set_updater(optimizer)
        self.initialized = True
        print("Done initialize. From now, begin training!")

    def _liner(self, size, activation=None):
        return K.layers.Dense(size, kernel_initializer="normal",
                              activation=activation)

    def make_model(self, feature_shape):
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
        model.add(self._liner(512))

        actor_layer = self._liner(len(self.actions), "softmax")
        action_probs = actor_layer(model.output)

        critic_layer = self._liner(1)
        values = critic_layer(model.output)

        self.model = K.Model(inputs=model.input,
                             outputs=[action_probs, values])

    def set_updater(self, optimizer, value_loss_weight=0.5):
        actions = tf.placeholder(shape=(None), dtype="int32")
        rewards = tf.placeholder(shape=(None), dtype="float32")

        action_probs, values = self.model.output
        values = tf.reshape(values, (-1,))
        advantages = rewards - values
        one_hot_actions = tf.one_hot(actions, len(self.actions), axis=1)
        selected_action_probs = tf.reduce_sum(one_hot_actions * action_probs,
                                              axis=1)
        clipped = tf.clip_by_value(selected_action_probs, 1e-10, 1.0)
        policy_loss = - tf.log(clipped) * advantages
        policy_loss = tf.reduce_mean(policy_loss)
        value_loss = tf.losses.mean_squared_error(rewards, values)

        loss = policy_loss + value_loss_weight * value_loss

        updates = optimizer.get_updates(loss=loss,
                                        params=self.model.trainable_weights)
        self._updater = K.backend.function(
                                        inputs=[self.model.input,
                                                actions, rewards],
                                        outputs=[loss],
                                        updates=updates)

    def estimate(self, state):
        action_probs, values = self.model.predict(np.array([state]))
        return action_probs[0]

    def update(self, states, actions, rewards):
        return self._updater([states, actions, rewards])


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


class ActorCriticTrainer(Trainer):

    def __init__(self, log_dir="", file_name=""):
        super().__init__(log_dir)
        self.file_name = file_name if file_name else "a2c_agent.h5"
        self._reward_scaler = None
        self.d_experiences = []
        self.final_epsilon = 0.0001
        self.epsilon_decay = 1e-6
        self.training_count = 0
        self.loss = []
        self.callback = K.callbacks.TensorBoard(self.log_dir)

    def train(self, env, episode_count=2000, gamma=0.99, epsilon=0.0001,
              epsilon_decay=1e-6, buffer_size=50000, batch_size=32,
              render=False, report_interval=10):
        if not isinstance(env, Observer):
            raise Exception("Environment have to be wrapped by Observer")

        actions = list(range(env.action_space.n))
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.final_epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.training_count = 0
        self.report_interval = report_interval
        agent = ActorCriticAgent(1.0, actions)

        self.train_loop(env, agent, episode_count, render)
        return agent

    def episode_begin(self, episode_count, agent):
        self.loss = []

    def step(self, episode_count, step_count, agent, experience):
        if agent.initialized:
            loss = agent.update(*self.make_batch())
            self.loss.append(loss)
            agent.epsilon = max(agent.epsilon - self.epsilon_decay,
                                self.final_epsilon)

    def make_batch(self):
        batch = random.sample(self.d_experiences, self.batch_size)
        states = [e.s for e in batch]
        actions = [e.a for e in batch]
        rewards = [e.r for e in batch]
        rewards = np.array(rewards).reshape((-1, 1))
        rewards = self._reward_scaler.transform(rewards).flatten()
        return states, actions, rewards

    def episode_end(self, episode_count, step_count, agent):
        rewards = [e.r for e in self.experiences]
        self.reward_log.append(sum(rewards))

        discounteds = []
        for t, r in enumerate(rewards):
            future_r = [_r * (self.gamma ** i) for i, _r in
                        enumerate(rewards[t:])]
            _r = sum(future_r)
            discounteds.append(_r)

        for i, e in enumerate(self.experiences):
            s, a, r, n_s, d = e
            d_r = discounteds[i]
            d_e = Experience(s, a, d_r, n_s, d)
            self.d_experiences.append(d_e)

        self.experiences = []

        if len(self.d_experiences) > self.buffer_size:
            self.d_experiences = self.d_experiences[-self.buffer_size:]
            if not agent.initialized:
                optimizer = K.optimizers.Adam(lr=1e-6)
                agent.initialize(self.d_experiences, optimizer)
                self.callback.set_model(agent.model)
                self._reward_scaler = StandardScaler()
                rewards = np.array([[e.r] for e in self.d_experiences])
                self._reward_scaler.fit(rewards)
            else:
                self.write_log(self.training_count,
                               np.mean(self.loss), sum(rewards))
                if self.is_event(self.training_count, self.report_interval):
                    agent.save(os.path.join(self.log_dir, self.file_name))
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
    trainer = ActorCriticTrainer(file_name="a2c_agent.h5")
    path = os.path.join(trainer.log_dir, trainer.file_name)

    if play:
        agent = ActorCriticAgent.load(env, path)
        agent.play(obs, render=True)
    else:
        trainer.train(obs, report_interval=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A2C Agent")
    parser.add_argument("--play", action="store_true",
                        help="play with trained model")

    args = parser.parse_args()
    main(args.play)
