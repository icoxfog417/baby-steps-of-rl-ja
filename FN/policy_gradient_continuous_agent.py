import os
import argparse
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import tensorflow as tf
from tensorflow.python import keras as K
import gym
from fn_framework import FNAgent, Trainer, Observer


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

    def initialize(self, experiences, actor_optimizer, critic_optimizer):
        self.scaler = StandardScaler()
        states = np.vstack([e.s for e in experiences])
        self.scaler.fit(states)
        feature_size = states.shape[1]

        base = K.models.Sequential()
        base.add(K.layers.Dense(16, activation="relu",
                                input_shape=(feature_size,)))
        base.add(K.layers.Dense(16, activation="relu"))
        base.add(K.layers.Dense(16, activation="relu"))

        # Actor
        #  define action distribution
        mu = K.layers.Dense(1, activation="tanh")(base.output)
        mu = K.layers.Lambda(lambda m: m * 2)(mu)
        #sigma = K.layers.Dense(1, activation="softplus")(base.output)
        #self.dist_model = K.Model(inputs=base.input, outputs=[mu, sigma])
        self.dist_model = K.Model(inputs=base.input, outputs=[mu])

        #  sample action from distribution
        low, high = self.actions
        action = SampleLayer(low, high)((mu))
        self.model = K.Model(inputs=base.input, outputs=[action])

        # Critic
        self.critic = K.models.Sequential([
            K.layers.Dense(32, activation="relu", input_shape=(feature_size + 1,)),
            K.layers.Dense(32, activation="relu"),
            K.layers.Dense(32, activation="relu"),
            K.layers.Dense(1, activation="linear")
        ])
        self.set_updater(actor_optimizer)
        self.critic.compile(loss="mse", optimizer=critic_optimizer)
        self.initialized = True
        print("Done initialize. From now, begin training!")

    def set_updater(self, optimizer):
        actions = tf.compat.v1.placeholder(shape=(None), dtype="float32")
        td_error = tf.compat.v1.placeholder(shape=(None), dtype="float32")

        # Actor loss
        mu = self.dist_model.output
        action_dist = tf.distributions.Normal(loc=tf.squeeze(mu),
                                              scale=0.1)
        action_probs = action_dist.prob(tf.squeeze(actions))
        clipped = tf.clip_by_value(action_probs, 1e-10, 1.0)
        loss = - tf.math.log(clipped) * td_error
        loss = tf.reduce_mean(loss)

        updates = optimizer.get_updates(loss=loss,
                                        params=self.model.trainable_weights)
        self._updater = K.backend.function(
                                        inputs=[self.model.input,
                                                actions, td_error],
                                        outputs=[loss, action_probs, mu],
                                        updates=updates)

    def policy(self, s):
        if np.random.random() < self.epsilon or not self.initialized:
            low, high = self.actions
            return np.random.uniform(low, high)
        else:
            normalized_s = self.scaler.transform(s)
            action = self.model.predict(normalized_s)[0]
            return action[0]

    def update(self, batch, gamma):
        states = np.vstack([e.s for e in batch])
        normalized_s = self.scaler.transform(states)
        actions = np.vstack([e.a for e in batch])

        # Calculate value
        next_states = np.vstack([e.n_s for e in batch])
        normalized_n_s = self.scaler.transform(next_states)
        n_s_actions = self.model.predict(normalized_n_s)
        feature_n = np.concatenate([normalized_n_s, n_s_actions], axis=1)
        n_s_values = self.critic.predict(feature_n)
        values = [b.r + gamma * (0 if b.d else 1) * n_s_values
                  for b, n_s_values in zip(batch, n_s_values)]
        values = np.array(values)

        feature = np.concatenate([normalized_s, actions], axis=1)
        td_error = values - self.critic.predict(feature)
        a_loss, probs, mu = self._updater([normalized_s, actions, td_error])
        c_loss = self.critic.train_on_batch(feature, values)

        """
        print([a_loss, c_loss])
        for x in zip(actions, mu, probs):
            print("Took action {}. (mu={}, its prob={})".format(*x))
        """


class SampleLayer(K.layers.Layer):

    def __init__(self, low, high, **kwargs):
        self.low = low
        self.high = high
        super(SampleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SampleLayer, self).build(input_shape)

    def call(self, x):
        mu = x
        actions = tf.distributions.Normal(loc=tf.squeeze(mu),
                                          scale=0.1).sample([1])
        actions = tf.clip_by_value(actions, self.low, self.high)
        return tf.reshape(actions, (-1, 1))

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
        return np.reshape(state, (1, -1))


class PolicyGradientContinuousTrainer(Trainer):

    def __init__(self, buffer_size=100000, batch_size=32,
                 gamma=0.99, report_interval=10, log_dir=""):
        super().__init__(buffer_size, batch_size, gamma,
                         report_interval, log_dir)

    def train(self, env, episode_count=220, epsilon=1.0, initial_count=-1,
              render=False):
        low, high = [env.action_space.low[0], env.action_space.high[0]]
        agent = PolicyGradientContinuousAgent(epsilon, low, high)

        self.train_loop(env, agent, episode_count, initial_count, render)
        return agent

    def begin_train(self, episode, agent):
        actor_optimizer = K.optimizers.Adam(lr=0.001, clipnorm=1.0)
        critic_optimizer = K.optimizers.Adam(lr=0.001, clipnorm=1.0)
        agent.initialize(self.experiences, actor_optimizer, critic_optimizer)
        agent.epsilon = 0.01

    def step(self, episode, step_count, agent, experience):
        if self.training:
            batch = random.sample(self.experiences, self.batch_size)
            agent.update(batch, self.gamma)

    def episode_end(self, episode, step_count, agent):
        reward = sum([e.r for e in self.get_recent(step_count)])
        self.reward_log.append(reward)

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
        trained = trainer.train(env, episode_count=1500, render=True)
        trainer.logger.plot("Rewards", trainer.reward_log,
                            trainer.report_interval)
        trained.save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PG Agent Pendulum-v0")
    parser.add_argument("--play", action="store_true",
                        help="play with trained model")

    args = parser.parse_args()
    main(args.play)
