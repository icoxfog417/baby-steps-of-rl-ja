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

    def initialize(self, num_states, num_actions, optimizer, r_limit=2):
        # Variables
        best_trans_prob = tf.placeholder(tf.float32, shape=(num_states,))
        other_trans_probs = tf.placeholder(tf.float32, shape=(num_actions - 1, num_states))
        gamma = tf.placeholder(tf.float32, shape=())

        rewards = tf.Variable(tf.random_normal([num_states],
                              mean=r_limit / 2), name="rewards")

        # Calculate loss
        index = tf.constant(0)
        final_loss = tf.constant(1e+10)
        eye = tf.eye(num_states)

        condition = lambda i, loss: tf.less(i, other_trans_probs.shape[0])

        def process(i, loss):
            other_trans_prob = other_trans_probs[i]
            formula = K.backend.dot(tf.reshape((best_trans_prob - other_trans_prob), (1, -1)),
                                    tf.matrix_inverse(eye - gamma * best_trans_prob))
            formula = K.backend.dot(formula, tf.reshape(rewards, (-1, 1)))
            f_bound = tf.exp(tf.reduce_sum(K.activations.relu(-formula)))
            _loss = tf.squeeze(formula) #- f_bound
            loss = tf.reduce_min([loss, _loss])
            i = tf.add(i, 1)
            return i, loss

        final_index, final_loss = tf.while_loop(condition, process, [index, final_loss])
        l1 = tf.reduce_sum(tf.abs(rewards))  # L1 regularization
        r_bound = tf.exp(tf.reduce_sum(K.activations.relu(
                                    tf.abs(rewards) - r_limit)))
        final_loss -= (l1 )
        final_loss = -final_loss

        # Get gradients
        updates = optimizer.get_updates(loss=final_loss, params=[rewards])
        self._updater = K.backend.function(
                                        inputs=[best_trans_prob,
                                                other_trans_probs,
                                                gamma],
                                        outputs=[final_loss, rewards],
                                        updates=updates)

    def to_trans_prob(self, env, probs):
        states = env.states
        mx = np.zeros(len(states))
        for s in states:
            if s in probs:
                mx[s.index(env.row_length)] = probs[s]
        return mx

    def estimate(self, env, teacher, episode_count=3000, learning_rate=1e-3,
                 gamma=0.9, report_interval=100):
        optimizer = K.optimizers.Adam(learning_rate)
        num_actions = len(env.action_space)
        num_states = len(env.states)
        self.initialize(num_states, num_actions, optimizer)
        loss_history = []
        for e in range(episode_count):
            losses = []
            for s in env.states:
                actions = teacher.policy[s]
                best_action = max(actions, key=actions.get)
                best_action_trans_prob = None
                other_action_trans_probs = []
                for a in env.action_space:
                    probs = env.transit_func(s, a)
                    if len(probs) == 0:
                        continue
                    if a == best_action:
                        best_action_trans_prob = self.to_trans_prob(env, probs)
                    else:
                        other_action_trans_probs.append(
                            self.to_trans_prob(env, probs)
                        )
                if best_action_trans_prob is None:
                    continue

                other_action_trans_probs = np.array(other_action_trans_probs)
                loss, self.rewards = self._updater([best_action_trans_prob,
                                                    other_action_trans_probs,
                                                    gamma])
                losses.append(loss)

            else:
                loss_history.append(np.mean(losses))

            if e != 0 and e % report_interval == 0:
                viz.describe(e, "loss", loss_history, report_interval)

        return loss_history


def main():
    grid = [
        [0, 0, 0, 1],
        [0, -1, 0, -1],
        [0, 0, 0, 0],
    ]
    env = Environment(grid)
    planner = PolicyIterationPlanner(env)
    V_grid = planner.plan()

    irl = LinerIRL()
    loss_history = irl.estimate(env, planner)
    print(irl.rewards)

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
