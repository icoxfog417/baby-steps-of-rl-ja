import os
import argparse
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import tensorflow as tf
from tensorflow.python import keras as K
import gym
from fn_framework import FNAgent, Trainer, Experience


class PolicyGradientAgent(FNAgent):

    def __init__(self, epsilon, actions):
        super().__init__(epsilon, actions)
        self.estimate_probs = True
        self.scaler = None
        self._updater = None

    @classmethod
    def load(cls, env, model_path, epsilon=0.0001):
        agent = super().load(env, model_path, epsilon)
        agent.scaler = joblib.load(agent.scaler_path(model_path))
        return agent

    def save(self, model_path):
        super().save(model_path)
        joblib.dump(self.scaler, self.scaler_path(model_path))

    def scaler_path(self, model_path):
        fname, _ = os.path.splitext(model_path)
        fname += "_scaler.pkl"
        return fname

    def initialize(self, experiences, optimizer):
        self.scaler = StandardScaler()
        features = np.vstack([self.to_feature(e.s) for e in experiences])
        self.scaler.fit(features)

        feature_size = features.shape[1]
        self.model = K.models.Sequential([
            K.layers.Dense(10, activation="relu", input_shape=(feature_size,)),
            K.layers.Dense(10, activation="relu"),
            K.layers.Dense(len(self.actions), activation="softmax")
        ])
        self.set_updater(optimizer)
        self.initialized = True
        print("Done initialize. From now, begin training!")

    def set_updater(self, optimizer):
        actions = tf.placeholder(shape=(None), dtype="int32")
        rewards = tf.placeholder(shape=(None), dtype="float32")
        one_hot_actions = tf.one_hot(actions, len(self.actions), axis=1)
        action_probs = self.model.output
        selected_action_probs = tf.reduce_sum(one_hot_actions * action_probs,
                                              axis=1)
        clipped = tf.clip_by_value(selected_action_probs, 1e-10, 1.0)
        loss = - tf.log(clipped) * rewards
        loss = tf.reduce_mean(loss)

        updates = optimizer.get_updates(loss=loss,
                                        params=self.model.trainable_weights)
        self._updater = K.backend.function(
                                        inputs=[self.model.input,
                                                actions, rewards],
                                        outputs=[loss],
                                        updates=updates)

    def estimate(self, s):
        feature = self.to_feature(s)
        feature = self.scaler.transform(feature)
        action_probs = self.model.predict(feature)[0]
        return action_probs

    def to_feature(self, s):
        feature = np.array(s).reshape((1, -1))
        return feature

    def update(self, states, actions, rewards):
        _states = np.vstack([self.to_feature(s) for s in states])
        _states = self.scaler.transform(_states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        self._updater([_states, actions, rewards])


class PolicyGradientTrainer(Trainer):

    def __init__(self, buffer_size=1024, batch_size=32,
                 gamma=0.9, report_interval=10, log_dir=""):
        super().__init__(buffer_size, batch_size, gamma,
                         report_interval, log_dir)
        self._reward_scaler = None
        self.d_experiences = []

    def train(self, env, episode_count=220, epsilon=0.1, render=False):
        actions = list(range(env.action_space.n))
        agent = PolicyGradientAgent(epsilon, actions)

        self.train_loop(env, agent, episode_count, render)
        return agent

    def step(self, episode, step_count, agent, experience):
        if agent.initialized:
            agent.update(*self.make_batch())

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

        self.experiences = []

        if len(self.d_experiences) > self.buffer_size:
            self.d_experiences = self.d_experiences[-self.buffer_size:]
            if not agent.initialized:
                optimizer = K.optimizers.Adam()
                agent.initialize(self.d_experiences, optimizer)
                self._reward_scaler = StandardScaler()
                rewards = np.array([[e.r] for e in self.d_experiences])
                self._reward_scaler.fit(rewards)

        if self.is_event(episode, self.report_interval):
            recent_rewards = self.reward_log[-self.report_interval:]
            desc = self.make_desc("reward", recent_rewards)
            print("At episode {}, {}".format(episode, desc))


def main(play):
    env = gym.make("CartPole-v0")
    trainer = PolicyGradientTrainer()
    path = trainer.make_path("policy_gradient_agent.h5")

    if play:
        agent = PolicyGradientAgent.load(env, path)
        agent.play(env)
    else:
        trained = trainer.train(env)
        trainer.plot_logs("Rewards", trainer.reward_log,
                          trainer.report_interval)
        trained.save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PG Agent")
    parser.add_argument("--play", action="store_true",
                        help="play with trained model")

    args = parser.parse_args()
    main(args.play)
