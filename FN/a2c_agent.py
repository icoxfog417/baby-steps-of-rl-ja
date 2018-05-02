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

    def __init__(self, epsilon, actions):
        super().__init__(epsilon, actions)
        self._updater = None
        self.estimate_probs = True

    def initialize(self, experiences, optimizer):
        feature_shape = experiences[0].s.shape
        self.make_model(feature_shape)
        self.set_updater(optimizer)
        self.initialized = True
        print("Done initialize. From now, begin training!")

    def make_model(self, feature_shape):
        normal = K.initializers.glorot_normal()
        model = K.Sequential()
        model.add(K.layers.Conv2D(
            32, kernel_size=8, strides=4, padding="same",
            input_shape=feature_shape, kernel_initializer=normal,
            activation="relu"))
        model.add(K.layers.Conv2D(
            64, kernel_size=4, strides=2, padding="same",
            kernel_initializer=normal,
            activation="relu"))
        model.add(K.layers.Conv2D(
            64, kernel_size=3, strides=1, padding="same",
            kernel_initializer=normal,
            activation="relu"))
        model.add(K.layers.Flatten())
        model.add(K.layers.Dense(256, kernel_initializer=normal,
                                 activation="relu"))

        actor_layer = K.layers.Dense(len(self.actions), activation="softmax",
                                     kernel_initializer=normal)
        action_probs = actor_layer(model.output)

        critic_layer = K.layers.Dense(1, kernel_initializer=normal,
                                      activation="tanh")
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
                                        outputs=[loss,
                                                 policy_loss, value_loss,
                                                 prob_entropy],
                                        updates=updates)

    def categorical_entropy(self, logits):
        """
        From OpenAI A2C implementation
        https://github.com/openai/baselines/blob/master/baselines/a2c/utils.py
        """
        a0 = logits - tf.reduce_max(logits, 1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, 1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)

    def estimate(self, state):
        action_probs, values = self.model.predict(np.array([state]))
        return action_probs[0]

    def update(self, states, actions, rewards):
        return self._updater([states, actions, rewards])


class ActorCriticAgentTest(ActorCriticAgent):

    def make_model(self, feature_shape):
        normal = K.initializers.glorot_normal()
        model = K.Sequential()
        model.add(K.layers.Dense(64, input_shape=feature_shape,
                                 kernel_initializer=normal, activation="relu"))
        model.add(K.layers.Dense(64, kernel_initializer=normal,
                                 activation="relu"))

        actor_layer = K.layers.Dense(len(self.actions),
                                     kernel_initializer=normal,
                                     activation="softmax")
        action_probs = actor_layer(model.output)
        critic_layer = K.layers.Dense(1, kernel_initializer=normal,
                                      activation="tanh")
        values = critic_layer(model.output)

        self.model = K.Model(inputs=model.input,
                             outputs=[action_probs, values])


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
                 gamma=0.99, initial_epsilon=0.5, final_epsilon=1e-3,
                 learning_rate=1e-4, report_interval=10,
                 log_dir="", file_name=""):
        super().__init__(buffer_size, batch_size, gamma,
                         report_interval, log_dir)
        self.file_name = file_name if file_name else "a2c_agent.h5"
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.learning_rate = learning_rate
        self.d_experiences = deque(maxlen=self.buffer_size)
        self._reward_scaler = None
        self.training_episode = 0
        self.losses = {}

    def train(self, env, episode_count=1200, initial_count=200,
              test_mode=False, render=False):
        actions = list(range(env.action_space.n))
        if not test_mode:
            agent = ActorCriticAgent(1.0, actions)
        else:
            agent = ActorCriticAgentTest(1.0, actions)
        self.training_episode = episode_count

        self.train_loop(env, agent, episode_count, initial_count, render)
        agent.save(self.logger.path_of(self.file_name))
        return agent

    def episode_begin(self, episode, agent):
        self.losses = {"loss": [], "loss_policy": [], "loss_value": [],
                       "entropy": []}
        self.experiences = []

    def step(self, episode, step_count, agent, experience):
        if self.training:
            loss, pl, vl, en = agent.update(*self.make_batch())
            self.losses["loss"].append(loss)
            self.losses["loss_policy"].append(pl)
            self.losses["loss_value"].append(vl)
            self.losses["entropy"].append(en)

    def make_batch(self):
        batch = random.sample(self.d_experiences, self.batch_size)
        states = [e.s for e in batch]
        actions = [e.a for e in batch]
        rewards = [e.r for e in batch]
        rewards = np.array(rewards).reshape((-1, 1))
        rewards = self._reward_scaler.transform(rewards).flatten()
        return states, actions, rewards

    def begin_train(self, episode, agent):
        optimizer = K.optimizers.Adam(lr=self.learning_rate, clipvalue=10.0)
        agent.initialize(self.experiences, optimizer)
        self.logger.set_model(agent.model)
        agent.epsilon = self.initial_epsilon
        self.training_episode -= episode
        self._reward_scaler = StandardScaler()
        rewards = np.array([[e.r] for e in self.d_experiences])
        self._reward_scaler.fit(rewards)

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

        if not self.training and len(self.d_experiences) == self.buffer_size:
            self.begin_train(i, agent)
            self.training = True

        if self.training:
            self.logger.write(self.training_count, "reward", sum(rewards))
            self.logger.write(self.training_count, "epsilon", agent.epsilon)
            for k in self.losses:
                loss = sum(self.losses[k]) / step_count
                self.logger.write(self.training_count, "loss/" + k, loss)
            if self.is_event(self.training_count, self.report_interval):
                agent.save(self.logger.path_of(self.file_name))

            diff = (self.initial_epsilon - self.final_epsilon)
            decay = diff / self.training_episode
            agent.epsilon = max(agent.epsilon - decay, self.final_epsilon)

        if self.is_event(episode, self.report_interval):
            recent_rewards = self.reward_log[-self.report_interval:]
            self.logger.describe("reward", recent_rewards, episode=episode)


def main(play, is_test):
    trainer = ActorCriticTrainer(file_name="a2c_agent.h5")
    path = trainer.logger.path_of(trainer.file_name)
    agent_class = ActorCriticAgent

    if is_test:
        print("Train on test mode")
        obs = gym.make("CartPole-v0")
        agent_class = ActorCriticAgentTest
    else:
        env = gym.make("Catcher-v0")
        obs = CatcherObserver(env, 80, 80, 4)

    if play:
        agent = agent_class.load(obs, path)
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
