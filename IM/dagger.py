import os
import argparse
import warnings
import numpy as np
from sklearn.externals import joblib
from sklearn.neural_network import MLPRegressor, MLPClassifier
import gym
from gym.envs.registration import register
register(id="FrozenLakeEasy-v0", entry_point="gym.envs.toy_text:FrozenLakeEnv",
         kwargs={"is_slippery": False})


class TeacherAgent():

    def __init__(self, env, epsilon=0.1):
        self.actions = list(range(env.action_space.n))
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
        # Only state => action projection is needed.
        self.model = MLPRegressor(hidden_layer_sizes=(), max_iter=1)
        # Warmup to use predict method.
        dummy_label = [np.random.uniform(size=len(self.actions))]
        self.model.partial_fit([state], dummy_label)
        return self

    def estimate(self, state):
        q = self.model.predict([state])[0]
        return q

    def policy(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(len(self.actions))
        else:
            return np.argmax(self.estimate(state))

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
                agent.model.partial_fit([s], [estimated])
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


class FrozenLakeObserver():

    def __init__(self):
        self._env = gym.make("FrozenLakeEasy-v0")

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    def reset(self):
        return self.transform(self._env.reset())

    def render(self):
        self._env.render()

    def step(self, action):
        n_state, reward, done, info = self._env.step(action)
        return self.transform(n_state), reward, done, info

    def transform(self, state):
        feature = np.zeros(self.observation_space.n)
        feature[state] = 1.0
        return feature


class Student():

    def __init__(self, env):
        self.actions = list(range(env.action_space.n))
        self.model = None

    def initialize(self, state):
        self.model = MLPClassifier(hidden_layer_sizes=(), max_iter=1)
        dummy_action = 0
        self.model.partial_fit([state], [dummy_action],
                               classes=self.actions)
        return self

    def policy(self, state):
        return self.model.predict([state])[0]

    def imitate(self, env, teacher, initial_step=100, train_step=200,
                report_interval=10):
        states = []
        actions = []

        # Collect teacher's demonstrations.
        for e in range(initial_step):
            s = env.reset()
            done = False
            while not done:
                a = teacher.policy(s)
                n_state, reward, done, info = env.step(a)
                states.append(s)
                actions.append(a)
                s = n_state

        self.initialize(states[0])
        self.model.partial_fit(states, actions)

        print("Start imitation.")
        # Student tries to learn teacher's actions.
        step_limit = 20
        for e in range(train_step):
            s = env.reset()
            done = False
            rewards = []
            step = 0
            while not done and step < step_limit:
                a = self.policy(s)
                n_state, reward, done, info = env.step(a)
                states.append(s)
                actions.append(teacher.policy(s))
                s = n_state
                step += 1
            else:
                goal_reward = reward

            rewards.append(goal_reward)
            if e != 0 and e % report_interval == 0:
                recent = np.array(rewards[-report_interval:])
                print("At episode {}, reward is {}".format(
                        e, recent.mean()))

            with warnings.catch_warnings():
                # It will be fixed in latest scikit-learn.
                # https://github.com/scikit-learn/scikit-learn/issues/10449
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                self.model.partial_fit(states, actions)


def main(teacher):
    env = FrozenLakeObserver()
    path = os.path.join(os.path.dirname(__file__), "imitation_teacher.pkl")

    if teacher:
        agent = TeacherAgent.train(env)
        agent.save(path)
    else:
        teacher_agent = TeacherAgent.load(env, path)
        student = Student(env)
        student.imitate(env, teacher_agent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Imitation Learning")
    parser.add_argument("--teacher", action="store_true",
                        help="train teacher model")

    args = parser.parse_args()
    main(args.teacher)
