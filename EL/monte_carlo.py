import math
from collections import defaultdict
import gym
from el_agent import ELAgent
from frozen_lake_util import show_q_value


class MonteCarloAgent(ELAgent):

    def __init__(self, epsilon=0.1):
        super().__init__(epsilon)

    def learn(self, env, episode_count=100000, gamma=0.9,
              render=False, report_interval=100):
        self.init_log()
        self.Q = defaultdict(lambda: [0] * len(actions))
        N = defaultdict(lambda: [0] * len(actions))
        actions = list(range(env.action_space.n))

        for e in range(episode_count):
            s = env.reset()
            done = False
            # Gain experience
            experience = []
            while not done:
                if render:
                    env.render()
                a = self.policy(s, actions)
                n_state, reward, done, info = env.step(a)
                experience.append(
                    {"state": s, "action": a, "reward": reward}
                )
                s = n_state
            else:
                self.log(reward)

            for i, x in enumerate(experience):
                s = x["state"]
                a = x["action"]

                # Calculate discounted future reward
                G = 0
                t = 0
                for j in range(i, len(experience)):
                    G += math.pow(gamma, t) * experience[j]["reward"]
                    t += 1

                # Count visits
                N[s][a] += 1
                alpha = 1 / N[s][a]
                # Update
                self.Q[s][a] += alpha * (G - self.Q[s][a])

            if e != 0 and e % report_interval == 0:
                self.show_reward_log(episode=e)


def train():
    agent = MonteCarloAgent(epsilon=0.1)
    env = gym.make("FrozenLakeEasy-v0")
    agent.learn(env, episode_count=3000)
    show_q_value(agent.Q)
    agent.show_reward_log()


if __name__ == "__main__":
    train()
