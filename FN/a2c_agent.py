import random
import argparse
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.python import keras as K
from sklearn.preprocessing import StandardScaler
from PIL import Image
import gym
import gym_ple
from fn_framework import FNAgent, Trainer, Observer, Experience


class ActorCriticAgent(FNAgent):

    def __init__(self, epsilon, actions, test_mode=False):
        super().__init__(epsilon, actions)
        self.test_mode = test_mode
        self._updater = None
        self.estimate_probs = True

    def initialize(self, experiences, optimizer):
        feature_shape = experiences[0].s.shape
        if self.test_mode:
            self.make_test_model(feature_shape)
        else:
            self.make_model(feature_shape)
        self.set_updater(optimizer)
        self.initialized = True
        print("Done initialize. From now, begin training!")

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
        model.add(K.layers.Dense(256, activation="relu"))
        model.add(K.layers.BatchNormalization())

        actor_layer = K.layers.Dense(len(self.actions), activation="softmax")
        action_probs = actor_layer(model.output)

        critic_layer = K.layers.Dense(1)
        values = critic_layer(model.output)

        self.model = K.Model(inputs=model.input,
                             outputs=[action_probs, values])

    def make_test_model(self, feature_shape):
        model = K.Sequential()
        model.add(K.layers.Dense(64, input_shape=feature_shape,
                                 activation="relu"))
        model.add(K.layers.Dense(64, activation="relu"))

        actor_layer = K.layers.Dense(len(self.actions), activation="softmax")
        action_probs = actor_layer(model.output)
        critic_layer = K.layers.Dense(1, activation="tanh")
        values = critic_layer(model.output)

        self.model = K.Model(inputs=model.input,
                             outputs=[action_probs, values])

    def set_updater(self, optimizer,
                    value_loss_weight=0.5, entropy_weight=0.01):
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
        prob_entropy = tf.reduce_mean(self.categorical_entropy(action_probs))

        loss = policy_loss + value_loss_weight * value_loss
        loss -= entropy_weight * prob_entropy

        updates = optimizer.get_updates(loss=loss,
                                        params=self.model.trainable_weights)
        self._updater = K.backend.function(
                                        inputs=[self.model.input,
                                                actions, rewards],
                                        outputs=[loss],
                                        updates=updates)

    def categorical_entropy(self, logits):
        """
        From OpenAI A2C implementation
        https://github.com/openai/baselines/blob/master/baselines/a2c/utils.py
        """
        a0 = logits - tf.reduce_max(logits, 1, keep_dims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, 1, keep_dims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)

    def estimate(self, state):
        action_probs, values = self.model.predict(np.array([state]))
        return action_probs[0]

    def update(self, states, actions, rewards):
        return self._updater([states, actions, rewards])


class CatcherObserver(Observer):

    def __init__(self, env, width, height, frame_count):
        super().__init__(env)
        self.width = width
        self.height = height
        self.frame_count = frame_count
        self._frames = deque(maxlen=frame_count)

    def transform(self, state):
        grayed = Image.fromarray(state).convert("L")
        resized = grayed.resize((self.width, self.height))
        resized = np.array(resized).astype("float")
        normalized = resized / 255.0  # scale to 0~1
        if len(self._frames) == 0:
            for i in range(self.frame_count):
                self._frames.append(normalized)
        else:
            self._frames.append(normalized)
        feature = np.array(self._frames)
        # Convert the feature shape (f, w, h) => (w, h, f)
        feature = np.transpose(feature, (1, 2, 0))

        return feature


class ActorCriticTrainer(Trainer):

    def __init__(self, buffer_size=50000, batch_size=32,
                 gamma=0.99, initial_epsilon=0.1, final_epsilon=1e-3,
                 learning_rate=1e-3, report_interval=10,
                 log_dir="", file_name=""):
        super().__init__(buffer_size, batch_size, gamma,
                         report_interval, log_dir)
        self.file_name = file_name if file_name else "a2c_agent.h5"
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.learning_rate = learning_rate
        self.training_count = 0
        self.training_episode = 0
        self.d_experiences = deque(maxlen=self.buffer_size)
        self.loss = 0
        self.callback = K.callbacks.TensorBoard(self.log_dir)
        self.optimizer = K.optimizers.Adam(lr=learning_rate, clipvalue=1.0)

    def train(self, env, episode_count=3000, render=False, test_mode=False):
        actions = list(range(env.action_space.n))
        self.training_count = 0
        self.training_episode = episode_count
        agent = ActorCriticAgent(1.0, actions, test_mode)
        if not test_mode:
            self.optimizer = K.optimizers.RMSprop(lr=self.learning_rate,
                                                  decay=0.99, epsilon=1e-5)
        self.train_loop(env, agent, episode_count, render)
        agent.save(self.make_path(self.file_name))
        return agent

    def episode_begin(self, episode, agent):
        self.loss = 0
        self.experiences = []

    def step(self, episode, step_count, agent, experience):
        if agent.initialized:
            self.loss += agent.update(*self.make_batch())[0]

    def make_batch(self):
        batch = random.sample(self.d_experiences, self.batch_size)
        states = [e.s for e in batch]
        actions = [e.a for e in batch]
        rewards = [e.r for e in batch]
        rewards = np.array(rewards).reshape((-1, 1))
        rewards = self._reward_scaler.transform(rewards).flatten()
        return states, actions, rewards

    def episode_end(self, episode, step_count, agent):
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

        if self.storing and len(self.d_experiences) >= self.buffer_size:
            agent.initialize(self.d_experiences, self.optimizer)
            self.callback.set_model(agent.model)
            self._reward_scaler = StandardScaler()
            d_rewards = np.array([[e.r] for e in self.d_experiences])
            self._reward_scaler.fit(d_rewards)
            agent.epsilon = self.initial_epsilon
            self.training_episode -= episode
            self.storing = False

        if not self.storing and agent.initialized:
            loss = self.loss / step_count
            self.write_log(self.training_count, loss, sum(rewards))
            if self.is_event(self.training_count, self.report_interval):
                agent.save(self.make_path(self.file_name))

            diff = (self.initial_epsilon - self.final_epsilon)
            decay = diff / self.training_episode
            agent.epsilon = max(agent.epsilon - decay, self.final_epsilon)
            self.training_count += 1

        if self.is_event(episode, self.report_interval):
            recent_rewards = self.reward_log[-self.report_interval:]
            desc = self.make_desc("reward", recent_rewards)
            print("At episode {}, {}".format(episode, desc))

    def write_log(self, index, loss, score):
        for name, value in zip(("loss", "score"), (loss, score)):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.callback.writer.add_summary(summary, index)
            self.callback.writer.flush()


def main(play, is_test):
    trainer = ActorCriticTrainer(file_name="a2c_agent.h5")
    path = trainer.make_path(trainer.file_name)

    if is_test:
        print("Train on test mode")
        obs = gym.make("CartPole-v0")
    else:
        env = gym.make("Catcher-v0")
        obs = CatcherObserver(env, 80, 80, 4)
        trainer.learning_rate = 7e-4
        trainer.epsilon = 0.75

    if play:
        agent = ActorCriticAgent.load(env, path)
        agent.play(obs, render=True)
    else:
        trainer.train(obs, test_mode=is_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A2C Agent")
    parser.add_argument("--play", action="store_true",
                        help="play with trained model")
    parser.add_argument("--test", action="store_true",
                        help="train by test mode")

    args = parser.parse_args()
    main(args.play, args.test)
