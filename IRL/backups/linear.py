import os
import numpy as np
from tensorflow.python import keras as K
import tensorflow as tf
from environment import Environment
from planner import PolicyIterationPlanner
import visualizer as viz


class LinerIRL():

    def __init__(self):
        self._updater = None
        self.rewards = None

    def initialize(self, num_states, num_actions, optimizer, C=1.0, r_max=2):
        # Variables
        best_trans_probs = tf.placeholder(tf.float32,
                                          shape=(num_states, num_states))
        other_trans_probss = tf.placeholder(tf.float32,
                                            shape=(num_states,
                                                   num_actions - 1,
                                                   num_states))
        gamma = tf.placeholder(tf.float32, shape=())
        rewards = tf.Variable(tf.random_normal([num_states], mean=r_max/2),
                              name="rewards")

        _indices = tf.constant([0] * num_states)
        _min_losses = tf.constant([1e+10] * num_states)
        eye = tf.eye(num_states)

        condition = lambda s, i, loss: tf.less(i, other_trans_probss.shape[1])  # noqa

        def process(s, i, loss):
            best_trans_prob = best_trans_probs[s]
            other_trans_prob = other_trans_probss[s][i]

            f_left = tf.reshape((best_trans_prob - other_trans_prob), (1, -1))
            f_right = tf.matrix_inverse(eye - gamma * best_trans_prob)

            # Limit the rewards of other actions smaller than best's one.
            R = tf.reshape(tf.clip_by_value(rewards, -r_max, r_max), (-1, 1))

            formula = K.backend.dot(K.backend.dot(f_left, f_right), R)

            # Formula should be positive
            _loss = tf.abs(tf.squeeze(tf.nn.leaky_relu(formula)))
            loss = tf.reduce_min([loss, _loss])
            i = tf.add(i, 1)
            return s, i, loss

        total_loss = tf.constant(0.0)
        for s in range(num_states):
            _, _, min_loss = tf.while_loop(condition, process,
                                           [s, _indices[s], _min_losses[s]])
            total_loss = tf.add(total_loss, min_loss)

        total_loss -= C * tf.reduce_sum(tf.abs(rewards))  # L1 regularization
        total_loss = -total_loss  # Maximize to Minimize

        # Get gradients
        updates = optimizer.get_updates(loss=total_loss, params=[rewards])
        self._updater = K.backend.function(
                                        inputs=[best_trans_probs,
                                                other_trans_probss,
                                                gamma],
                                        outputs=[total_loss, rewards],
                                        updates=updates)

    def to_trans_prob(self, env, probs):
        states = env.states
        mx = np.zeros(len(states))
        for s in states:
            if s in probs:
                mx[s.index(env.row_length)] = probs[s]
        return mx

    def estimate(self, env, teacher, episode_count=6000, learning_rate=1e-3,
                 gamma=0.9, report_interval=100):
        optimizer = K.optimizers.Adam(learning_rate)
        num_actions = len(env.action_space)
        num_states = len(env.states)
        self.initialize(num_states, num_actions, optimizer)
        loss_history = []
        for e in range(episode_count):
            best_trans_probs = []
            other_trans_probss = []
            for s in env.states:
                actions = teacher.policy[s]
                best_action = max(actions, key=actions.get)
                best_trans_prob = np.zeros(num_states)
                other_trans_probs = []
                for a in env.action_space:
                    probs = env.transit_func(s, a)
                    if len(probs) == 0:
                        continue
                    if a == best_action:
                        best_trans_prob = self.to_trans_prob(env, probs)
                    else:
                        other_trans_probs.append(
                            self.to_trans_prob(env, probs)
                        )
                if len(other_trans_probs) == 0:
                    other_trans_probs = [np.zeros(num_states)] * (num_actions - 1)

                other_trans_probs = np.array(other_trans_probs)

                best_trans_probs.append(best_trans_prob)
                other_trans_probss.append(other_trans_probs)

            best_trans_probs = np.array(best_trans_probs)
            other_trans_probss = np.array(other_trans_probss)

            loss, self.rewards = self._updater([best_trans_probs,
                                                other_trans_probss,
                                                gamma])
            loss_history.append(loss)
            if e != 0 and e % report_interval == 0:
                viz.describe(e, "loss", loss_history, report_interval)

        return loss_history


def main():
    grid = [
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]
    # Prepare Teacher
    env = Environment(grid)
    planner = PolicyIterationPlanner(env)
    planner.plan()

    # Execute IRL
    irl = LinerIRL()
    irl.estimate(env, planner)
    print(irl.rewards)

    # Plot Reward Map
    ncol = env.column_length
    nrow = env.row_length
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    fig, ax = plt.subplots()
    reward_map = irl.rewards.reshape((nrow, ncol))
    ax.imshow(reward_map, cmap=cm.RdYlGn)
    ax.set_xticks(np.arange(ncol))
    ax.set_yticks(np.arange(nrow))
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
