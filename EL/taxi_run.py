import numpy as np
import gym


if __name__ == "__main__":
    env = gym.make("Taxi-v2")
    for episode in range(10):    
        env.reset()
        for step in range(50):
            env.render()
            action = np.random.randint(env.action_space.n)
            state, reward, done, info = env.step(action)
            if done:
                break
