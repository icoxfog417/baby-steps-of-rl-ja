import gym


def run():
    """
    state: [position, velocity]
    action: 0, 1, 2 => 0: left 1: stay 2: right
    reward: always -1 (so the agent have to go goal ASAP)
    """
    env = gym.make("MountainCar-v0")
    for _ in range(5):
        env.reset()
        rewards = 0
        done = False
        for i in range(100):
            env.render()
            a = env.action_space.sample()
            state, reward, done, info = env.step(a)
            rewards += reward
            print("step{}: state={}, action={}".format(i, state, a))
            if done:
                break
        print("Done episode reward={}.".format(rewards))


if __name__ == "__main__":
    run()
