import argparse
import numpy as np
from collections import defaultdict, Counter
import gym
from gym.envs.registration import register
register(id="FrozenLakeEasy-v0", entry_point="gym.envs.toy_text:FrozenLakeEnv",
         kwargs={"is_slippery": False})


class DynaAgent():

    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.actions = []
        self.value = None

    def policy(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(len(self.actions))
        else:
            if sum(self.value[state]) == 0:
                return np.random.randint(len(self.actions))
            else:            
                return np.argmax(self.value[state])

    def learn(self, env, episode_count=3000, gamma=0.9, learning_rate=0.1,
              steps_in_model=-1, report_interval=100):
        self.actions = list(range(env.action_space.n))
        self.value = defaultdict(lambda: [0] * len(self.actions))
        model = Model(self.actions)

        rewards = []
        for e in range(episode_count):
            s = env.reset()
            done = False
            goal_reward = 0
            while not done:
                a = self.policy(s)
                n_state, reward, done, info = env.step(a)

                # Update from experiments in the real environment.
                gain = reward + gamma * max(self.value[n_state])
                estimated = self.value[s][a]
                self.value[s][a] += learning_rate * (gain - estimated)

                if steps_in_model > 0:
                    model.update(s, a, reward, n_state)
                    for s, a, r, n_s in model.simulate(steps_in_model):
                        gain = r + gamma * max(self.value[n_s])
                        estimated = self.value[s][a]
                        self.value[s][a] += learning_rate * (gain - estimated)

                s = n_state
            else:
                goal_reward = reward

            rewards.append(goal_reward)
            if e != 0 and e % report_interval == 0:
                recent = np.array(rewards[-report_interval:])
                print("At episode {}, reward is {}".format(
                        e, recent.mean()))


class Model():

    def __init__(self, actions):
        self.num_actions = len(actions)
        self.transit_count = defaultdict(lambda: [Counter() for a in actions])
        self.total_reward = defaultdict(lambda: [0] *
                                        self.num_actions)
        self.history = defaultdict(Counter)

    def update(self, state, action, reward, next_state):
        self.transit_count[state][action][next_state] += 1
        self.total_reward[state][action] += reward
        self.history[state][action] += 1

    def transit(self, state, action):
        counter = self.transit_count[state][action]
        states = []
        counts = []
        for s, c in counter.most_common():
            states.append(s)
            counts.append(c)
        probs = np.array(counts) / sum(counts)
        return np.random.choice(states, p=probs)

    def reward(self, state, action):
        total_reward = self.total_reward[state][action]
        total_count = self.history[state][action]
        return total_reward / total_count

    def simulate(self, count):
        states = list(self.transit_count.keys())
        actions = lambda s: [a for a, c in self.history[s].most_common()
                             if c > 0]

        for i in range(count):
            state = np.random.choice(states)
            action = np.random.choice(actions(state))

            next_state = self.transit(state, action)
            reward = self.reward(state, action)

            yield state, action, reward, next_state


def main(steps_in_model):
    env = gym.make("FrozenLakeEasy-v0")
    agent = DynaAgent()
    agent.learn(env, steps_in_model=steps_in_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dyna Agent")
    parser.add_argument("--modelstep", type=int, default=-1,
                        help="step count in the model")

    args = parser.parse_args()
    main(args.modelstep)
