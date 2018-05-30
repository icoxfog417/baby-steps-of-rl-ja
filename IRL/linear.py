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
        state = tf.placeholder(tf.int32, shape=())
        best_prob = tf.placeholder(tf.float32, shape=())
        other_prob = tf.placeholder(tf.float32, shape=(None,))
        next_state_mask = tf.placeholder(tf.int32, shape=(num_states,))
        gamma = tf.placeholder(tf.float32, shape=())

        rewards = tf.Variable(tf.random_normal([num_states],
                              mean=r_limit / 2), name="rewards")

        # Calculate loss
        eye = tf.eye(tf.size(other_prob))
        p_best = eye * best_prob
        p_others = eye * other_prob
        other_rewards = tf.boolean_mask(rewards, next_state_mask)
        formula = K.backend.dot((p_best - p_others),
                                tf.matrix_inverse(eye - gamma * p_others))

        formula = K.backend.dot(formula, tf.reshape(other_rewards, (-1, 1)))
        loss = tf.reduce_min(formula)

        l1 = tf.reduce_sum(tf.abs(rewards))  # L1 regularization
        f_bound = tf.exp(tf.reduce_sum(K.activations.relu(-formula)))
        r_bound = tf.exp(tf.reduce_sum(K.activations.relu(
                                        tf.abs(rewards) - r_limit)))
        loss -= (l1 + f_bound + r_bound)
        loss = -loss

        # Get gradients
        updates = optimizer.get_updates(loss=loss, params=[rewards])
        self._updater = K.backend.function(
                                        inputs=[state, best_prob,
                                                other_prob, next_state_mask,
                                                gamma],
                                        outputs=[loss, rewards],
                                        updates=updates)

    def _softmax(self, estimates):
        return np.exp(estimates) / np.sum(np.exp(estimates), axis=0)

    def estimate(self, env, teacher, episode_count=3000, learning_rate=1e-4,
                 gamma=0.2, report_interval=100):
        optimizer = K.optimizers.SGD(learning_rate)
        num_actions = len(env.action_space)
        num_states = len(env.states)
        self.initialize(num_states, num_actions, optimizer)
        loss_history = []
        for e in range(episode_count):
            losses = []
            for s in env.states:
                actions = teacher.policy[s]
                a = max(actions, key=actions.get)
                probs = env.transit_func(s, a)

                if len(probs) == 0:
                    continue

                max_state = max(probs, key=probs.get)
                other_states = [_s for _s in probs if _s != max_state]
                best_prob = probs[max_state]
                other_probs = [probs[_s] for _s in other_states]
                state_index = s.index(env.row_length)
                next_state_mask = [True if _s in other_states else False
                                   for _s in env.states]
                loss, self.rewards = self._updater([state_index, best_prob,
                                                    other_probs, next_state_mask,
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
