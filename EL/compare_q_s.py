from multiprocessing import Pool
from collections import defaultdict
import gym
from el_agent import ELAgent
from frozen_lake_util import show_q_value


class CompareAgent(ELAgent):

    def __init__(self, q_learning=True, epsilon=0.33):
        self.q_learning = q_learning
        super().__init__(epsilon)

    def learn(self, env, episode_count=1000, gamma=0.9,
              learning_rate=0.1, render=False, report_interval=50):
        self.init_log()
        self.Q = defaultdict(lambda: [0] * len(actions))
        actions = list(range(env.action_space.n))
        for e in range(episode_count):
            s = env.reset()
            done = False
            a = self.policy(s, actions)
            while not done:
                if render:
                    env.render()

                n_state, reward, done, info = env.step(a)

                if done and reward == 0:
                    reward = -0.5  # Reward as penalty

                n_action = self.policy(n_state, actions)

                if self.q_learning:
                    gain = reward + gamma * max(self.Q[n_state])
                else:
                    gain = reward + gamma * self.Q[n_state][n_action]

                estimated = self.Q[s][a]
                self.Q[s][a] += learning_rate * (gain - estimated)
                s = n_state

                if self.q_learning:
                    a = self.policy(s, actions)
                else:
                    a = n_action
            else:
                self.log(reward)

            if e != 0 and e % report_interval == 0:
                self.show_reward_log(episode=e)


def train(q_learning):
    env = gym.make("FrozenLakeEasy-v0")
    agent = CompareAgent(q_learning=q_learning)
    agent.learn(env, episode_count=3000)
    return dict(agent.Q)


if __name__ == "__main__":
    with Pool() as pool:
        results = pool.map(train, ([True, False]))
        for r in results:
            show_q_value(r)
