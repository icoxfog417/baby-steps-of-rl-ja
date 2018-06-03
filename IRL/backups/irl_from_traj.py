import os
import argparse
import numpy as np
from collections import defaultdict
from sklearn.externals import joblib
from sklearn.neural_network import MLPRegressor
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.python import keras as K
import gym
from gym.envs.registration import register
register(id="FrozenLakeEasy-v0", entry_point="gym.envs.toy_text:FrozenLakeEnv",
         kwargs={"is_slippery": False})


tfe.enable_eager_execution()


class TeacherAgent():

    def __init__(self, env, epsilon=0.1):
        self.actions = list(range(env.action_space.n))
        self.num_states = env.observation_space.n
        self.epsilon = epsilon
        self.model = None

    def save(self, model_path):
        joblib.dump(self.model, model_path)

    @classmethod
    def load(cls, env, model_path, epsilon=0.1):
        agent = cls(env, epsilon)
        agent.model = joblib.load(model_path)
        return agent

    def initialize(self, state):
        # Only state => action projection is needed
        self.model = MLPRegressor(hidden_layer_sizes=(), max_iter=1)
        # Warmup to use predict method
        dummy_label = [np.random.uniform(size=len(self.actions))]
        self.model.partial_fit(np.array([self.transform(state)]),
                               np.array(dummy_label))
        return self

    def estimate(self, state):
        feature = self.transform(state)
        q = self.model.predict([feature])[0]
        return q

    def policy(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(len(self.actions))
        else:
            return np.argmax(self.estimate(state))

    def transform(self, state):
        feature = np.zeros(self.num_states)
        feature[state] = 1.0
        return feature

    @classmethod
    def train(cls, env, episode_count=3000, gamma=0.9,
              initial_epsilon=1.0, final_epsilon=0.1, report_interval=100):
        agent = cls(env, initial_epsilon).initialize(env.reset())
        rewards = []
        decay = (initial_epsilon - final_epsilon) / episode_count
        for e in range(episode_count):
            s = env.reset()
            done = False
            goal_reward = 0
            while not done:
                a = agent.policy(s)
                estimated = agent.estimate(s)

                n_state, reward, done, info = env.step(a)

                gain = reward + gamma * max(agent.estimate(n_state))
                estimated[a] = gain
                agent.model.partial_fit([agent.transform(s)], [estimated])
                s = n_state
            else:
                goal_reward = reward

            rewards.append(goal_reward)
            if e != 0 and e % report_interval == 0:
                recent = np.array(rewards[-report_interval:])
                print("At episode {}, reward is {}".format(
                        e, recent.mean()))
            agent.epsilon -= decay

        return agent


class IRL():

    def __init__(self, env):
        self.actions = list(range(env.action_space.n))
        self.num_states = env.observation_space.n
        self.rewards = tfe.Variable(tf.random_uniform(
                                        [env.observation_space.n]),
                                    name="rewards")
        """
        self.rewards = tfe.Variable(initial_value=[0.0, 0.0, 0.0, 0.0,
                                                   0.0, 0.0, 0.0, 0.0,
                                                   0.0, 0.0, 0.0, 0.0,
                                                   0.0, 0.0, 0.0, 1.0,],
                                    name="rewards")
        """
        self._updater = tfe.implicit_gradients(self.loss)

    """
    def value_estimate(self, steps, gamma):
        values = {}
        counts = {}
        for i, t in enumerate(steps):
            rewards = [self.rewards[s] for s in t]
            for j, s in enumerate(t):
                discounteds = [r * (gamma ** k)
                               for k, r in enumerate(rewards[j:])]
                discounted = tf.reduce_sum(discounteds)
                if s not in values:
                    values[s] = discounted
                    counts[s] = 0.0

                counts[s] += 1
                values[s] = tf.add(values[s], tf.divide(
                                    tf.subtract(discounted, values[s]),
                                    counts[s]))

        value_tensors = []
        total_count = sum([counts[s] for s in counts])
        for i in range(self.rewards.shape[0].value):
            if i in values:
                visit = counts[i] / total_count
                value = tf.multiply(values[i], visit)
            else:
                value = tf.constant(0.0)
            value_tensors.append(value)
        values = tf.stack(value_tensors)
        return values
    """

    def value_estimate(self, trajectory, gamma):
        values = {}
        one_host_trajectory = tf.one_hot(trajectory, self.num_states)
        rewards = tf.reduce_sum(one_host_trajectory * self.rewards, axis=1)
        for i, r in enumerate(rewards):
            future = [_r * (gamma ** (k + 1))
                      for k, _r in enumerate(rewards[(i + 1):])]
            reward = r + tf.reduce_sum(future)
            s = trajectory[i]
            values[s] = reward

        value_tensors = []
        for i in range(self.num_states):
            if i in values:
                value = values[i]
            else:
                value = tf.constant(0.0)
            value_tensors.append(value)
        values = tf.stack(value_tensors)
        return values

    def get_rewards(self):
        return self.rewards.numpy()

    def loss(self, teacher_steps, steps, gamma):
        teacher_values = tf.stack([self.value_estimate(t, gamma) for t in teacher_steps])
        values = tf.stack([self.value_estimate(t, gamma) for t in steps])
        best = tf.reduce_mean(teacher_values, axis=0)
        diff = tf.reduce_min(best - values, axis=0)
        #print(">>>>>>>>")
        #print(tf.reshape(best, (4, 4)))
        #print(tf.reshape(tf.reduce_mean(values, axis=0), (4, 4)))

        loss = tf.reduce_sum(tf.boolean_mask(diff, diff > 0))
        penalty = -2 * tf.reduce_sum(tf.boolean_mask(diff, diff < 0))
        loss += penalty

        #_loss = _loss + 1.5 * tf.reduce_sum(tf.abs(self.rewards))
        return loss

    def update(self, optimizer, teacher_steps, steps, gamma):
        loss = self.loss(teacher_steps, steps, gamma)
        optimizer.apply_gradients(self._updater(teacher_steps, steps, gamma))
        return loss, self.get_rewards()

    def take_action(self, Q, state, actions, epsilon=0.1):
        rand_action = np.random.randint(len(actions))
        if np.random.random() < epsilon:
            return rand_action
        elif state in Q and sum(Q[state]) != 0:
            return np.argmax(Q[state])
        else:
            return rand_action

    def estimate(self, env, teacher, episode_count=3000,
                 teacher_demo_size=256, batch_size=32,
                 learning_rate=1e-3, max_step=10,
                 gamma=0.9, report_interval=10):

        # Accumulate teacher's demonstration
        demos = []
        for e in range(teacher_demo_size):
            s = env.reset()
            done = False
            trajectory = [s]
            while not done:
                a = teacher.policy(s)
                n_state, reward, done, info = env.step(a)
                s = n_state
                trajectory.append(s)
            demos.append(trajectory)

        print("Start reward estimation.")
        actions = list(range(env.action_space.n))
        rewards = np.zeros((env.observation_space.n))
        Q = defaultdict(lambda: [0] * len(actions))
        optimizer = tf.train.AdamOptimizer(learning_rate)

        for e in range(episode_count):
            batch = []
            total_reward = 0
            for b in range(batch_size):
                s = env.reset()
                done = False
                trajectory = [s]
                step = 0
                epsilon = 1.0
                while not done and step < max_step:
                    a = self.take_action(Q, s, actions, epsilon)
                    n_state, reward, done, info = env.step(a)

                    estimated = Q[s][a]
                    gain = rewards[n_state] + gamma * max(Q[n_state])
                    Q[s][a] += learning_rate * (gain - estimated)
                    s = n_state
                    trajectory.append(s)
                    step += 1
                    epsilon = epsilon * ((batch_size - b) / batch_size)
                else:
                    total_reward += reward
                batch.append(trajectory)

            teacher_batch = np.random.choice(demos, size=batch_size)
            loss, new_rewards = self.update(optimizer,
                                            teacher_batch, batch, gamma)

            rewards = new_rewards

            if e % 10 == 0:
                print("At episode {}, reward={}, loss={}".format(
                        e, total_reward, loss))
                print("Reward")
                print(new_rewards.reshape(4, 4))


def main(train):
    env = gym.make("FrozenLakeEasy-v0")
    path = os.path.join(os.path.dirname(__file__), "irl_teacher.pkl")

    if train:
        agent = TeacherAgent.train(env)
        agent.save(path)
    else:
        teacher = TeacherAgent.load(env, path)
        irl = IRL(env)
        irl.estimate(env, teacher)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Imitation Learning")
    parser.add_argument("--train", action="store_true",
                        help="train teacher model")

    args = parser.parse_args()
    main(args.train)
