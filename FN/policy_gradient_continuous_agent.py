import os
import argparse
import random
from collections import deque
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import tensorflow as tf
from tensorflow.python import keras as K
import gym
from fn_framework import FNAgent, Trainer, Observer, Experience


class PolicyGradientContinuousAgent(FNAgent):

    def __init__(self, epsilon, low, high):
        super().__init__(epsilon, [low, high])
        self.scaler = None
        self._updater = None

    def save(self, model_path):
        super().save(model_path)
        joblib.dump(self.scaler, self.scaler_path(model_path))

    @classmethod
    def load(cls, env, model_path, epsilon=0.0001):
        low, high = [env.action_space.low[0], env.action_space.high[0]]
        agent = cls(epsilon, low, high)
        agent.model = K.models.load_model(model_path, custom_objects={
                        "SampleLayer": SampleLayer})
        agent.scaler = joblib.load(agent.scaler_path(model_path))
        return agent

    def scaler_path(self, model_path):
        fname, _ = os.path.splitext(model_path)
        fname += "_scaler.pkl"
        return fname

    def initialize(self, experiences, optimizer):
        self.scaler = StandardScaler()
        states = np.vstack([e.s for e in experiences])
        self.scaler.fit(states)
        feature_size = states.shape[1]

        normal = K.initializers.glorot_normal()
        base = K.models.Sequential([
            K.layers.Dense(10, activation="relu", input_shape=(feature_size,),
                           kernel_initializer=normal)
        ])
        mu = K.layers.Dense(1, activation="linear",
                            kernel_initializer=normal)(base.output)
        sigma = K.layers.Dense(1, activation="softplus",
                               kernel_initializer=normal)(base.output)

        self.dist_model = K.Model(inputs=base.input, outputs=[mu, sigma])

        low, high = self.actions
        action = SampleLayer(low, high)((mu, sigma))
        self.model = K.Model(inputs=base.input, outputs=[action])
        self.set_updater(optimizer)
        self.initialized = True
        print("Done initialize. From now, begin training!")

    def set_updater(self, optimizer):
        actions = tf.placeholder(shape=(None), dtype="float32")
        rewards = tf.placeholder(shape=(None), dtype="float32")

        mu, sigma = self.dist_model.output
        action_dist = tf.distributions.Normal(loc=tf.squeeze(mu),
                                              scale=tf.squeeze(sigma))
        action_probs = action_dist.prob(actions)
        clipped = tf.clip_by_value(action_probs, 1e-10, 1.0)
        loss = - tf.log(clipped) * tf.exp(rewards)
        loss = tf.reduce_mean(loss)

        updates = optimizer.get_updates(loss=loss,
                                        params=self.model.trainable_weights)
        self._updater = K.backend.function(
                                        inputs=[self.model.input,
                                                actions, rewards],
                                        outputs=[loss, action_probs, mu, sigma],
                                        updates=updates)

    def policy(self, s):
        if np.random.random() < self.epsilon or not self.initialized:
            low, high = self.actions
            return np.random.uniform(low, high)
        else:
            normalized = self.scaler.transform(s)
            action = self.model.predict(normalized)[0]
            return action[0]

    def update(self, states, actions, rewards):
        normalizeds = self.scaler.transform(states)
        actions = np.array(actions)
        rewards = np.array(rewards)

        loss, probs, mu, sigma = self._updater([normalizeds, actions, rewards])
        #print(loss)
        """
        for x in zip(actions, mu, sigma, probs):
            print(x)
        """


class SampleLayer(K.layers.Layer):

    def __init__(self, low, high, **kwargs):
        self.low = low
        self.high = high
        super(SampleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SampleLayer, self).build(input_shape)

    def call(self, x):
        mu, sigma = x
        epsilon_dist = tf.distributions.Normal(loc=0., scale=1.0)
        _sigma = K.layers.Lambda(lambda x: epsilon_dist.sample(1) * x)(sigma)
        action = mu + _sigma
        action = tf.clip_by_value(action, self.low, self.high)
        return action

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)

    def get_config(self):
        config = super().get_config()
        config["low"] = self.low
        config["high"] = self.high
        return config


class PendulumObserver(Observer):

    def step(self, action):
        n_state, reward, done, info = self._env.step([action])
        return self.transform(n_state), reward, done, info

    def transform(self, state):
        return np.array(state).reshape((1, -1))


class PolicyGradientContinuousTrainer(Trainer):

    def __init__(self, buffer_size=1024, batch_size=32,
                 gamma=0.9, report_interval=10, log_dir=""):
        super().__init__(buffer_size, batch_size, gamma,
                         report_interval, log_dir)
        self._reward_scaler = None
        self.d_experiences = deque(maxlen=buffer_size)

    def train(self, env, episode_count=220, epsilon=0.1, initial_count=-1,
              render=False):
        low, high = [env.action_space.low[0], env.action_space.high[0]]
        agent = PolicyGradientContinuousAgent(epsilon, low, high)

        self.train_loop(env, agent, episode_count, initial_count, render)
        return agent

    def episode_begin(self, episode, agent):
        self.experiences = []

    def step(self, episode, step_count, agent, experience):
        if agent.initialized:
            agent.update(*self.make_batch())

    def make_batch(self):
        batch = random.sample(self.d_experiences, self.batch_size)
        states = np.vstack([e.s for e in batch])
        actions = [e.a for e in batch]
        rewards = [e.r for e in batch]
        rewards = np.array(rewards).reshape((-1, 1))
        rewards = self._reward_scaler.transform(rewards).flatten()
        return states, actions, rewards

    def begin_train(self, episode, agent):
        optimizer = K.optimizers.Adam(clipnorm=1.0)
        agent.initialize(self.d_experiences, optimizer)
        self._reward_scaler = StandardScaler(with_mean=False)
        rewards = np.array([[e.r] for e in self.d_experiences])
        self._reward_scaler.fit(rewards)

    def episode_end(self, episode, step_count, agent):
        rewards = [e.r for e in self.experiences]
        self.reward_log.append(sum(rewards))
        discounteds = []
        for t, r in enumerate(rewards):
            d_r = [_r * (self.gamma ** i) for i, _r in
                   enumerate(rewards[t:])]
            d_r = sum(d_r)
            discounteds.append(d_r)

        for i, e in enumerate(self.experiences):
            s, a, r, n_s, d = e
            d_r = discounteds[i]
            d_e = Experience(s, a, d_r, n_s, d)
            self.d_experiences.append(d_e)

        if not self.training and len(self.d_experiences) == self.buffer_size:
            self.begin_train(i, agent)
            self.training = True

        if self.is_event(episode, self.report_interval):
            recent_rewards = self.reward_log[-self.report_interval:]
            self.logger.describe("reward", recent_rewards, episode=episode)


def main(play):
    env = PendulumObserver(gym.make("Pendulum-v0"))
    trainer = PolicyGradientContinuousTrainer()
    path = trainer.logger.path_of("policy_gradient_continuous_agent.h5")

    if play:
        agent = PolicyGradientContinuousAgent.load(env, path)
        agent.play(env)
    else:
        trained = trainer.train(env, episode_count=500)
        trainer.logger.plot("Rewards", trainer.reward_log,
                            trainer.report_interval)
        trained.save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PG Agent Pendulum-v0")
    parser.add_argument("--play", action="store_true",
                        help="play with trained model")

    args = parser.parse_args()
    main(args.play)
