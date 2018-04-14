import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.python import keras as K
import gym
from estimator import Estimator
from fn_agent import FNAgent


class PolicyEstimator(Estimator):

    def __init__(self, actions, gamma):
        super().__init__(actions)
        self.gamma = gamma
        self.reset()
        self.scaler = None
        self.reward_scaler = None
        self._trainer = None

    def initialize(self, experiences):
        self.scaler = StandardScaler()
        features = np.vstack([self.to_feature(e.s) for e in experiences])
        self.scaler.fit(features)
        self.reward_scaler = StandardScaler()
        rewards = np.array([[e.r] for e in experiences])
        self.reward_scaler.fit(rewards)

        feature_size = features.shape[1]
        self.set_estimator(feature_size)
        self.set_trainer()
        self.initialized = True
        print("Done initialize. From now, begin training!")

    def set_estimator(self, feature_size):
        features = K.Input(shape=(feature_size,))
        layer = None
        for size in [10, 10]:
            if layer is None:
                layer = features
            hidden = K.layers.Dense(size, activation=tf.nn.relu)(layer)
            layer = hidden
        outputs = K.layers.Dense(len(self.actions),
                                 activation=tf.nn.softmax)(layer)
        model = K.Model(inputs=features, outputs=outputs)
        self.model = model

    def set_trainer(self):
        actions = tf.placeholder(shape=(None), dtype="int32")
        rewards = tf.placeholder(shape=(None), dtype="float32")
        one_hot_actions = tf.one_hot(actions, len(self.actions), axis=1)
        action_probs = self.model.output
        selected_action_probs = tf.reduce_sum(one_hot_actions * action_probs,
                                              axis=1)
        clipped = tf.clip_by_value(selected_action_probs, 1e-10, 1.0)
        loss = - tf.log(clipped) * rewards
        loss = tf.reduce_mean(loss)

        optimizer = K.optimizers.Adam()
        updates = optimizer.get_updates(loss=loss,
                                        params=self.model.trainable_weights)

        self._trainer = K.backend.function(
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
        # CartPole state is ...
        # position, speed, angle, angle_speed = s
        feature = np.array(s).reshape((1, -1))
        return feature

    def update(self, experiences):
        states = []
        actions = []
        rewards = []

        for e in experiences:
            states.append(self.to_feature(e.s))
            actions.append(e.a)
            rewards.append(e.r)

        states = np.vstack(states)
        states = self.scaler.transform(states)
        actions = np.array(actions)
        rewards = np.array(rewards).reshape((-1, 1))
        rewards = self.reward_scaler.transform(rewards).flatten()
        self._trainer([states, actions, rewards])


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
            experiences = []
            rewards = []
            while not done:
                if render:
                    env.render()
                a = self.policy(s)
                n_state, reward, done, info = env.step(a)
                experiences.append([s, a, reward, n_state, done])
                rewards.append(reward)
                s = n_state
            else:
                # Calculate discounted reward on each time step
                discounteds = []
                for t, r in enumerate(rewards):
                    future_r = [_r * (gamma ** i) for i, _r in
                                enumerate(rewards[t:])]
                    _r = sum(future_r)
                    discounteds.append(_r)

                for r, ex in zip(discounteds, experiences):
                    s, a, _, n_s, done = ex
                    self.feedback(s, a, r, n_s, done)
                self.log(sum(rewards))

            if e != 0 and e % report_interval == 0:
                self.show_reward_log(interval=report_interval, episode=e)


def train():
    agent = PolicyGradientAgent()
    env = gym.make("CartPole-v1")
    agent.learn(env, render=False)
    agent.show_reward_log()


if __name__ == "__main__":
    train()
