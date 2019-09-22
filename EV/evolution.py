import os
import argparse
import numpy as np
from sklearn.externals.joblib import Parallel, delayed
from PIL import Image
import matplotlib.pyplot as plt
import gym

# Disable TensorFlow GPU for parallel execution
if os.name == "nt":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.python import keras as K


class EvolutionalAgent():

    def __init__(self, actions):
        self.actions = actions
        self.model = None

    def save(self, model_path):
        self.model.save(model_path, overwrite=True, include_optimizer=False)

    @classmethod
    def load(cls, env, model_path):
        actions = list(range(env.action_space.n))
        agent = cls(actions)
        agent.model = K.models.load_model(model_path)
        return agent

    def initialize(self, state, weights=()):
        normal = K.initializers.glorot_normal()
        model = K.Sequential()
        model.add(K.layers.Conv2D(
            3, kernel_size=5, strides=3,
            input_shape=state.shape, kernel_initializer=normal,
            activation="relu"))
        model.add(K.layers.Flatten())
        model.add(K.layers.Dense(len(self.actions), activation="softmax"))
        self.model = model
        if len(weights) > 0:
            self.model.set_weights(weights)

    def policy(self, state):
        action_probs = self.model.predict(np.array([state]))[0]
        action = np.random.choice(self.actions,
                                  size=1, p=action_probs)[0]
        return action

    def play(self, env, episode_count=5, render=True):
        for e in range(episode_count):
            s = env.reset()
            done = False
            episode_reward = 0
            while not done:
                if render:
                    env.render()
                a = self.policy(s)
                n_state, reward, done, info = env.step(a)
                episode_reward += reward
                s = n_state
            else:
                print("Get reward {}".format(episode_reward))


class CatcherObserver():

    def __init__(self, width, height, frame_count):
        import gym_ple
        self._env = gym.make("Catcher-v0")
        self.width = width
        self.height = height

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    def reset(self):
        return self.transform(self._env.reset())

    def render(self):
        self._env.render(mode="human")

    def step(self, action):
        n_state, reward, done, info = self._env.step(action)
        return self.transform(n_state), reward, done, info

    def transform(self, state):
        grayed = Image.fromarray(state).convert("L")
        resized = grayed.resize((self.width, self.height))
        resized = np.array(resized).astype("float")
        normalized = resized / 255.0  # scale to 0~1
        normalized = np.expand_dims(normalized, axis=2)  # H x W => W x W x C
        return normalized


class EvolutionalTrainer():

    def __init__(self, population_size=20, sigma=0.5, learning_rate=0.1,
                 report_interval=10):
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.weights = ()
        self.reward_log = []

    def train(self, epoch=100, episode_per_agent=1, render=False):
        env = self.make_env()
        actions = list(range(env.action_space.n))
        s = env.reset()
        agent = EvolutionalAgent(actions)
        agent.initialize(s)
        self.weights = agent.model.get_weights()

        with Parallel(n_jobs=-1) as parallel:
            for e in range(epoch):
                experiment = delayed(EvolutionalTrainer.run_agent)
                results = parallel(experiment(
                                episode_per_agent, self.weights, self.sigma)
                                for p in range(self.population_size))
                self.update(results)
                self.log()

        agent.model.set_weights(self.weights)
        return agent

    @classmethod
    def make_env(cls):
        return CatcherObserver(width=50, height=50, frame_count=5)

    @classmethod
    def run_agent(cls, episode_per_agent, base_weights, sigma,
                  max_step=1000):
        env = cls.make_env()
        actions = list(range(env.action_space.n))
        agent = EvolutionalAgent(actions)

        noises = []
        new_weights = []

        # Make weight
        for w in base_weights:
            noise = np.random.randn(*w.shape)
            new_weights.append(w + sigma * noise)
            noises.append(noise)

        # Test Play
        total_reward = 0
        for e in range(episode_per_agent):
            s = env.reset()
            if agent.model is None:
                agent.initialize(s, new_weights)
            done = False
            step = 0
            while not done and step < max_step:
                a = agent.policy(s)
                n_state, reward, done, info = env.step(a)
                total_reward += reward
                s = n_state
                step += 1

        reward = total_reward / episode_per_agent
        return reward, noises

    def update(self, agent_results):
        rewards = np.array([r[0] for r in agent_results])
        noises = np.array([r[1] for r in agent_results])
        normalized_rs = (rewards - rewards.mean()) / rewards.std()

        # Update base weights
        new_weights = []
        for i, w in enumerate(self.weights):
            noise_at_i = np.array([n[i] for n in noises])
            rate = self.learning_rate / (self.population_size * self.sigma)
            w = w + rate * np.dot(noise_at_i.T, normalized_rs).T
            new_weights.append(w)

        self.weights = new_weights
        self.reward_log.append(rewards)

    def log(self):
        rewards = self.reward_log[-1]
        print("Epoch {}: reward {:.3}(max:{}, min:{})".format(
            len(self.reward_log), rewards.mean(),
            rewards.max(), rewards.min()))

    def plot_rewards(self):
        indices = range(len(self.reward_log))
        means = np.array([rs.mean() for rs in self.reward_log])
        stds = np.array([rs.std() for rs in self.reward_log])
        plt.figure()
        plt.title("Reward History")
        plt.grid()
        plt.fill_between(indices, means - stds, means + stds,
                         alpha=0.1, color="g")
        plt.plot(indices, means, "o-", color="g",
                 label="reward")
        plt.legend(loc="best")
        plt.show()


def main(play):
    model_path = os.path.join(os.path.dirname(__file__), "ev_agent.h5")

    if play:
        env = EvolutionalTrainer.make_env()
        agent = EvolutionalAgent.load(env, model_path)
        agent.play(env, episode_count=5, render=True)
    else:
        trainer = EvolutionalTrainer()
        trained = trainer.train()
        trained.save(model_path)
        trainer.plot_rewards()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evolutional Agent")
    parser.add_argument("--play", action="store_true",
                        help="play with trained model")

    args = parser.parse_args()
    main(args.play)
