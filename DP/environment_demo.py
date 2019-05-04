import random
from environment import Environment


class Agent():

    def __init__(self, env):
        self.actions = env.actions

    def policy(self, state):
        return random.choice(self.actions)# randomに選んだ方向を返す。


def main():
    # Make grid environment.
    grid = [
        [0, 0, 0, 1],
        [0, 9, 0, -1],
        [0, 0, 0, 0]
    ]
    env = Environment(grid)
    agent = Agent(env)

    # Try 10 game.
    for i in range(10):
        # Initialize position of agent.
        state = env.reset()
        total_reward = 0
        done = False

        # ここが本ではfor文に入ってないように見えて混乱した。
        # doneがTrueになるのは、rewardを得た場合。詳細はreward_funcの中。
        # つまりrewardを得るために、永遠に動き続ける。
        while not done:
            action = agent.policy(state)# randomな方向を得る
            next_state, reward, done = env.step(action)#移動する。
            total_reward += reward
            state = next_state

        print("Episode {}: Agent gets {} reward.".format(i, total_reward))


if __name__ == "__main__":
    main()
