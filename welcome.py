import numpy as np
from tensorflow.python import keras as K
import gym
import gym_ple


def welcome():
    """
    Code to check installation of basic libraries
    """

    env = gym.make("Catcher-v0")
    num_action = env.action_space.n
    episode_count = 10

    s = env.reset()
    brain = K.Sequential()
    brain.add(K.layers.Dense(num_action, input_shape=[np.prod(s.shape)],
                             activation="softmax"))

    def policy(s):
        evaluation = brain.predict(np.array([s.flatten()]))
        return np.argmax(evaluation)

    for e in range(episode_count):
        s = env.reset()
        done = False
        while not done:
            env.render(mode="human")
            a = policy(s)
            n_state, reward, done, info = env.step(a)
            s = n_state


if __name__ == "__main__":
    welcome()
