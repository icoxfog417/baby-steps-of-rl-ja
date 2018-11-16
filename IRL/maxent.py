import numpy as np
from planner import PolicyIterationPlanner
from tqdm import tqdm


class MaxEntIRL():

    def __init__(self, env):
        self.env = env
        self.planner = PolicyIterationPlanner(env)

    def estimate(self, trajectories, epoch=20, learning_rate=0.01, gamma=0.9):
        state_features = np.vstack([self.env.state_to_feature(s)
                                   for s in self.env.states])
        theta = np.random.uniform(size=state_features.shape[1])
        teacher_features = self.calculate_expected_feature(trajectories)

        for e in tqdm(range(epoch)):
            # Estimate reward.
            rewards = state_features.dot(theta.T)

            # Optimize policy under estimated reward.
            self.planner.reward_func = lambda s: rewards[s]
            self.planner.plan(gamma=gamma)

            # Estimate feature under policy.
            features = self.expected_features_under_policy(
                                self.planner.policy, trajectories)

            # Update to close to teacher.
            update = teacher_features - features.dot(state_features)
            theta += learning_rate * update

        estimated = state_features.dot(theta.T)
        estimated = estimated.reshape(self.env.shape)
        return estimated

    def calculate_expected_feature(self, trajectories):
        features = np.zeros(self.env.observation_space.n)
        for t in trajectories:
            for s in t:
                features[s] += 1

        features /= len(trajectories)
        return features

    def expected_features_under_policy(self, policy, trajectories):
        t_size = len(trajectories)
        states = self.env.states
        transition_probs = np.zeros((t_size, len(states)))

        initial_state_probs = np.zeros(len(states))
        for t in trajectories:
            initial_state_probs[t[0]] += 1
        initial_state_probs /= t_size
        transition_probs[0] = initial_state_probs

        for t in range(1, t_size):
            for prev_s in states:
                prev_prob = transition_probs[t - 1][prev_s]
                a = self.planner.act(prev_s)
                probs = self.env.transit_func(prev_s, a)
                for s in probs:
                    transition_probs[t][s] += prev_prob * probs[s]

        total = np.mean(transition_probs, axis=0)
        return total


if __name__ == "__main__":
    def test_estimate():
        from environment import GridWorldEnv
        env = GridWorldEnv(grid=[
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 0],
        ])
        # Train Teacher
        teacher = PolicyIterationPlanner(env)
        teacher.plan()
        trajectories = []
        print("Gather demonstrations of teacher.")
        for i in range(20):
            s = env.reset()
            done = False
            steps = [s]
            while not done:
                a = teacher.act(s)
                n_s, r, done, _ = env.step(a)
                steps.append(n_s)
                s = n_s
            trajectories.append(steps)

        print("Estimate reward.")
        irl = MaxEntIRL(env)
        rewards = irl.estimate(trajectories, epoch=100)
        print(rewards)
        env.plot_on_grid(rewards)

    test_estimate()
