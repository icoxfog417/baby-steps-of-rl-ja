import random
import argparse
from collections import deque
import numpy as np
from tensorflow.python import keras as K
from PIL import Image
import gym
import gym_ple
from fn_framework import FNAgent, Trainer, Observer


class DeepQNetworkAgent(FNAgent):

    def __init__(self, epsilon, actions):
        super().__init__(epsilon, actions)
        self._scaler = None
        self._teacher_model = None

    def initialize(self, experiences, optimizer):
        feature_shape = experiences[0].s.shape
        self.make_model(feature_shape)
        self.model.compile(optimizer, loss="mse")
        self.initialized = True
        print("Done initialization. From now, begin training!")

    def make_model(self, feature_shape):
        normal = K.initializers.glorot_normal()
        model = K.Sequential()
        model.add(K.layers.Conv2D(
            32, kernel_size=8, strides=4, padding="same",
            input_shape=feature_shape, kernel_initializer=normal,
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
        model.add(K.layers.Dense(256, kernel_initializer=normal,
                                 activation="relu"))
        model.add(K.layers.Dense(len(self.actions),
                                 kernel_initializer=normal))
        self.model = model
        self._teacher_model = K.models.clone_model(self.model)

    def estimate(self, state):
        return self.model.predict(np.array([state]))[0]

    def update(self, experiences, gamma):
        states = np.array([e.s for e in experiences])
        n_states = np.array([e.n_s for e in experiences])

        estimateds = self.model.predict(states)
        future = self._teacher_model.predict(n_states)

        for i, e in enumerate(experiences):
            reward = e.r
            if not e.d:
                reward += gamma * np.max(future[i])
            estimateds[i][e.a] = reward

        loss = self.model.train_on_batch(states, estimateds)
        return loss

    def update_teacher(self):
        self._teacher_model.set_weights(self.model.get_weights())


class DeepQNetworkAgentTest(DeepQNetworkAgent):

    def __init__(self, epsilon, actions):
        super().__init__(epsilon, actions)

    def make_model(self, feature_shape):
        normal = K.initializers.glorot_normal()
        model = K.Sequential()
        model.add(K.layers.Dense(64, input_shape=feature_shape,
                                 kernel_initializer=normal, activation="relu"))
        model.add(K.layers.Dense(len(self.actions), kernel_initializer=normal,
                                 activation="relu"))
        self.model = model
        self._teacher_model = K.models.clone_model(self.model)


class CatcherObserver(Observer):

    def __init__(self, env, width, height, frame_count):
        super().__init__(env)
        self.width = width
        self.height = height
        self.frame_count = frame_count
        self._frames = deque(maxlen=frame_count)

    def transform(self, state):
        grayed = Image.fromarray(state).convert("L")
        resized = grayed.resize((self.width, self.height))
        resized = np.array(resized).astype("float")
        normalized = resized / 255.0  # scale to 0~1
        if len(self._frames) == 0:
            for i in range(self.frame_count):
                self._frames.append(normalized)
        else:
            self._frames.append(normalized)
        feature = np.array(self._frames)
        # Convert the feature shape (f, w, h) => (h, w, f).
        feature = np.transpose(feature, (1, 2, 0))

        return feature


class DeepQNetworkTrainer(Trainer):

    def __init__(self, buffer_size=50000, batch_size=32,
                 gamma=0.99, initial_epsilon=0.5, final_epsilon=1e-3,
                 learning_rate=1e-3, teacher_update_freq=3, report_interval=10,
                 log_dir="", file_name=""):
        super().__init__(buffer_size, batch_size, gamma,
                         report_interval, log_dir)
        self.file_name = file_name if file_name else "dqn_agent.h5"
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.learning_rate = learning_rate
        self.teacher_update_freq = teacher_update_freq
        self.loss = 0
        self.training_episode = 0
        self._max_reward = -10

    def train(self, env, episode_count=1200, initial_count=200,
              test_mode=False, render=False, observe_interval=100):
        actions = list(range(env.action_space.n))
        if not test_mode:
            agent = DeepQNetworkAgent(1.0, actions)
        else:
            agent = DeepQNetworkAgentTest(1.0, actions)
            observe_interval = 0
        self.training_episode = episode_count

        self.train_loop(env, agent, episode_count, initial_count, render,
                        observe_interval)
        return agent

    def episode_begin(self, episode, agent):
        self.loss = 0

    def begin_train(self, episode, agent):
        optimizer = K.optimizers.Adam(lr=self.learning_rate, clipvalue=1.0)
        agent.initialize(self.experiences, optimizer)
        self.logger.set_model(agent.model)
        agent.epsilon = self.initial_epsilon
        self.training_episode -= episode

    def step(self, episode, step_count, agent, experience):
        if self.training:
            batch = random.sample(self.experiences, self.batch_size)
            self.loss += agent.update(batch, self.gamma)

    def episode_end(self, episode, step_count, agent):
        reward = sum([e.r for e in self.get_recent(step_count)])
        self.loss = self.loss / step_count
        self.reward_log.append(reward)
        if self.training:
            self.logger.write(self.training_count, "loss", self.loss)
            self.logger.write(self.training_count, "reward", reward)
            self.logger.write(self.training_count, "epsilon", agent.epsilon)
            if reward > self._max_reward:
                agent.save(self.logger.path_of(self.file_name))
                self._max_reward = reward
            if self.is_event(self.training_count, self.teacher_update_freq):
                agent.update_teacher()

            diff = (self.initial_epsilon - self.final_epsilon)
            decay = diff / self.training_episode
            agent.epsilon = max(agent.epsilon - decay, self.final_epsilon)

        if self.is_event(episode, self.report_interval):
            recent_rewards = self.reward_log[-self.report_interval:]
            self.logger.describe("reward", recent_rewards, episode=episode)


def main(play, is_test):
    file_name = "dqn_agent.h5" if not is_test else "dqn_agent_test.h5"
    trainer = DeepQNetworkTrainer(file_name=file_name)
    path = trainer.logger.path_of(trainer.file_name)
    agent_class = DeepQNetworkAgent

    if is_test:
        print("Train on test mode")
        obs = gym.make("CartPole-v0")
        agent_class = DeepQNetworkAgentTest
    else:
        env = gym.make("Catcher-v0")
        obs = CatcherObserver(env, 80, 80, 4)
        trainer.learning_rate = 1e-4

    if play:
        agent = agent_class.load(obs, path)
        agent.play(obs, render=True)
    else:
        trainer.train(obs, test_mode=is_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN Agent")
    parser.add_argument("--play", action="store_true",
                        help="play with trained model")
    parser.add_argument("--test", action="store_true",
                        help="train by test mode")

    args = parser.parse_args()
    main(args.play, args.test)
