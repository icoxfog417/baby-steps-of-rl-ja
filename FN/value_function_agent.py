import random
import argparse
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
import gym
from fn_framework import FNAgent, Trainer


class ValueFunctionAgent(FNAgent):

    @classmethod
    def load(cls, env, model_path, epsilon=0.0001):
        actions = list(range(env.action_space.n))
        agent = cls(epsilon, actions)
        agent.model = joblib.load(model_path)
        agent.initialized = True
        return agent

    def save(self, model_path):
        joblib.dump(self.model, model_path)

    def initialize(self, experiences):
        scaler = StandardScaler()
        estimator = MLPRegressor(
                        hidden_layer_sizes=(10, 10),
                        max_iter=1)
        self.model = Pipeline([("scaler", scaler), ("estimator", estimator)])

        features = np.vstack([self.to_feature(e.s) for e in experiences])
        self.model.named_steps["scaler"].fit(features)

        # Avoid the predict before fit. Use a little sample to fit.
        self.update(experiences[:2], gamma=0)
        self.initialized = True
        print("Done initialize. From now, begin training!")

    def estimate(self, s):
        feature = self.to_feature(s)
        estimated = self.model.predict(feature)[0]
        return estimated

    def to_feature(self, s):
        # CartPole state is ...
        # position, speed, angle, angle_speed = s
        feature = np.array(s).reshape((1, -1))
        return feature

    def _predict(self, states):
        if self.initialized:
            predicteds = self.model.predict(states)
        else:
            size = len(self.actions) * len(states)
            predicteds = np.random.uniform(size=size)
            predicteds = predicteds.reshape((-1, len(self.actions)))
        return predicteds

    def update(self, experiences, gamma):
        states = np.vstack([self.to_feature(e.s) for e in experiences])
        n_states = np.vstack([self.to_feature(e.n_s) for e in experiences])

        estimateds = self._predict(states)
        future = self._predict(n_states)

        for i, e in enumerate(experiences):
            reward = e.r
            if not e.d:
                reward += gamma * np.max(future[i])
            estimateds[i][e.a] = reward

        estimateds = np.array(estimateds)
        states = self.model.named_steps["scaler"].transform(states)
        self.model.named_steps["estimator"].partial_fit(states, estimateds)


class ValueFunctionTrainer(Trainer):

    def __init__(self, log_dir=""):
        super().__init__(log_dir)

    def train(self, env, episode_count=220, gamma=0.9, epsilon=0.1,
              buffer_size=1024, batch_size=32,
              render=False, report_interval=10):
        actions = list(range(env.action_space.n))
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.report_interval = report_interval
        agent = ValueFunctionAgent(epsilon, actions)

        self.train_loop(env, agent, episode_count, render)
        return agent

    def buffer_full(self, agent):
        agent.initialize(self.experiences)

    def step(self, episode_count, step_count, experience, agent):
        if agent.initialized:
            batch = random.sample(self.experiences, self.batch_size)
            agent.update(batch, self.gamma)

    def episode_end(self, episode_count, step_count, agent):
        reward_in_episode = ([e.r for e in self.experiences[-step_count:]])
        self.reward_log.append(sum(reward_in_episode))

        if self.is_event(episode_count, self.report_interval):
            recent_rewards = self.reward_log[-self.report_interval:]
            desc = self.make_desc("reward", recent_rewards)
            print("At episode {}, {}".format(episode_count, desc))


def main(play):
    env = gym.make("CartPole-v1")
    trainer = ValueFunctionTrainer()
    path = trainer.make_path("value_function_agent.pkl")

    if play:
        agent = ValueFunctionAgent.load(env, path)
        agent.play(env)
    else:
        trained = trainer.train(env)
        trainer.plot_logs("Rewards", trainer.reward_log,
                          trainer.report_interval)
        trained.save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VF Agent")
    parser.add_argument("--play", action="store_true",
                        help="play with trained model")

    args = parser.parse_args()
    main(args.play)
