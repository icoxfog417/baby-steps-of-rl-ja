import os
import sys
from collections import deque
import tensorflow as tf
from tensorflow.python import keras as K
from tensorflow.python.keras._impl.keras.models import clone_model
from PIL import Image
import numpy as np
import gym
import gym_ple  # noqa


class Agent(object):
    INPUT_SHAPE = (80, 80, 4)

    def __init__(self, num_actions):
        self.num_actions = num_actions
        normal = K.initializers.RandomNormal(stddev=0.05)
        model = K.Sequential()
        model.add(K.layers.Conv2D(
            32, kernel_size=8, strides=4, padding="same",
            input_shape=self.INPUT_SHAPE, kernel_initializer=normal,
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
        model.add(K.layers.Dense(512, kernel_initializer=normal, activation="relu"))
        model.add(K.layers.Dense(num_actions, kernel_initializer=normal))
        self.model = model

    def evaluate(self, state, model=None):
        _model = model if model else self.model
        _state = np.expand_dims(state, axis=0)  # add batch size dimension
        return _model.predict(_state)[0]

    def act(self, state, epsilon=0):
        if np.random.rand() <= epsilon:
            a = np.random.randint(low=0, high=self.num_actions, size=1)[0]
        else:
            print(np.mean(state))
            q = self.evaluate(state)
            a = np.argmax(q)
        return a


class Observer(object):

    def __init__(self, input_shape):
        self.size = input_shape[:2]  # width x height
        self.num_frames = input_shape[2]  # number of frames
        self._frames = []

    def observe(self, state):
        g_state = Image.fromarray(state).convert("L")  # to gray scale
        g_state = g_state.resize(self.size)  # resize game screen to input size
        g_state = np.array(g_state).astype("float")
        g_state /= 255  # scale to 0~1
        if len(self._frames) == 0:
            # full fill the frame cache
            self._frames = [g_state] * self.num_frames
        else:
            self._frames.append(g_state)
            self._frames.pop(0)  # remove most old state

        input_state = np.array(self._frames)
        # change frame_num x width x height => width x height x frame_num
        input_state = np.transpose(input_state, (1, 2, 0))
        return input_state


class Trainer(object):

    def __init__(self, env, agent, optimizer, model_dir=""):
        self.env = env
        self.agent = agent
        self.experience = []
        self._target_model = clone_model(self.agent.model)
        self.observer = Observer(agent.INPUT_SHAPE)
        self.model_dir = model_dir
        if not self.model_dir:
            self.model_dir = os.path.join(os.path.dirname(__file__), "model")
            if not os.path.isdir(self.model_dir):
                os.mkdir(self.model_dir)

        self.agent.model.compile(optimizer=optimizer, loss="mse")
        self.callback = K.callbacks.TensorBoard(self.model_dir)
        self.callback.set_model(self.agent.model)

    def get_batch(self, batch_size, gamma):
        batch_indices = np.random.randint(
            low=0, high=len(self.experience), size=batch_size)
        X = np.zeros((batch_size,) + self.agent.INPUT_SHAPE)
        y = np.zeros((batch_size, self.agent.num_actions))
        for i, b_i in enumerate(batch_indices):
            s, a, r, next_s, game_over = self.experience[b_i]
            X[i] = s
            y[i] = self.agent.evaluate(s)
            # future reward
            Q_sa = np.max(self.agent.evaluate(next_s,
                                              model=self._target_model))
            if game_over:
                y[i, a] = r
            else:
                y[i, a] = r + gamma * Q_sa
        return X, y

    def write_log(self, index, loss, score):
        for name, value in zip(("loss", "score"), (loss, score)):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.callback.writer.add_summary(summary, index)
            self.callback.writer.flush()

    def train(self,
              gamma=0.99,
              initial_epsilon=0.1, final_epsilon=0.0001,
              memory_size=50000,
              observation_epochs=100, training_epochs=2000,
              batch_size=32, render=True):

        self.experience = deque(maxlen=memory_size)
        epochs = observation_epochs + training_epochs
        epsilon = initial_epsilon
        model_path = os.path.join(self.model_dir, "agent_network.h5")
        fmt = "Epoch {:04d}/{:d} | Loss {:.5f} | Score: {} | e={:.4f} train={}"

        for e in range(epochs):
            loss = 0.0
            rewards = []
            initial_state = self.env.reset()
            state = self.observer.observe(initial_state)
            game_over = False
            is_training = True if e > observation_epochs else False

            # let's play the game
            while not game_over:
                if render:
                    self.env.render()

                if not is_training:
                    action = self.agent.act(state, epsilon=1)
                else:
                    action = self.agent.act(state, epsilon)

                next_state, reward, game_over, info = self.env.step(action)
                next_state = self.observer.observe(next_state)
                self.experience.append(
                    (state, action, reward, next_state, game_over)
                    )

                rewards.append(reward)

                if is_training:
                    X, y = self.get_batch(batch_size, gamma)
                    loss += self.agent.model.train_on_batch(X, y)

                state = next_state
            loss = loss / len(rewards)
            score = sum(rewards)

            if is_training:
                self.write_log(e - observation_epochs, loss, score)
                self._target_model.set_weights(self.agent.model.get_weights())

            if epsilon > final_epsilon:
                epsilon -= (initial_epsilon - final_epsilon) / epochs

            print(fmt.format(e + 1, epochs, loss, score, epsilon, is_training))

            if e % 100 == 0:
                self.agent.model.save(model_path, overwrite=True)

        self.agent.model.save(model_path, overwrite=True)


def main(render):
    env = gym.make("Catcher-v0")
    num_actions = env.action_space.n
    agent = Agent(num_actions)
    trainer = Trainer(env, agent, K.optimizers.Adam(lr=1e-6))
    trainer.train(render=render)


if __name__ == "__main__":
    render = False if len(sys.argv) < 2 else True
    main(render)
