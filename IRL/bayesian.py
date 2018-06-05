import numpy as np
import scipy.stats
from scipy.misc import logsumexp
import tensorflow.contrib.eager as tfe
from planner import PolicyIterationPlanner
from tqdm import tqdm


tfe.enable_eager_execution()


class BayesianIRL():

    def __init__(self, env, eta=0.8, prior_mean=0.0, prior_scale=0.5):
        self.env = env
        self.planner = PolicyIterationPlanner(env)
        self.eta = eta
        self.prior_dist = scipy.stats.norm(loc=prior_mean,
                                           scale=prior_scale)

    def estimate(self, trajectories, epoch=30, gamma=0.9,
                 learning_rate=0.1, sigma=0.05, sample_size=20):
        num_states = len(self.env.states)
        reward = np.random.normal(size=num_states, scale=1.0)

        def get_q(r, g):
            self.planner.reward_func = lambda s: r[s]
            V = self.planner.plan(g)
            Q = self.planner.policy_to_q(V, gamma)
            return Q

        for i in range(epoch):
            noises = np.random.randn(sample_size, num_states)
            scores = []
            for n in tqdm(noises):
                _reward = reward + sigma * n
                Q = get_q(_reward, gamma)

                # Calculate prior (sum of log prob)
                reward_prior = np.sum(self.prior_dist.logpdf(_r)
                                      for _r in _reward)

                # Calculate likelihood
                likelihood = self.calculate_likelihood(trajectories, Q)

                # Calculate posterior
                posterior = likelihood + reward_prior
                scores.append(posterior)

            rate = learning_rate / (sample_size * sigma)
            scores = np.array(scores)
            normalized_scores = (scores - scores.mean()) / scores.std()
            noise = np.mean(noises * normalized_scores.reshape((-1, 1)),
                            axis=0)
            reward = reward + rate * noise
            print("At iteration {} posterior={}".format(i, scores.mean()))

        reward = reward.reshape(self.env.shape)
        return reward

    def calculate_likelihood(self, trajectories, Q):
        advantage = 0.0
        for t in trajectories:
            t_advantage = 0.0
            for s, a in t:
                best_value = self.eta * Q[s][a]
                other_values = [self.eta * Q[s][_a] for _a in self.env.actions
                                if _a != a]
                other_total = logsumexp(other_values)
                t_advantage += (best_value - other_total)
            t_advantage /= len(t)
            advantage += t_advantage
        advantage /= len(trajectories)
        return advantage


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
        print("Gather the demonstration")
        for i in range(20):
            s = env.reset()
            done = False
            steps = []
            while not done:
                a = teacher.act(s)
                steps.append((s, a))
                n_s, r, done, _ = env.step(a)
                s = n_s
            trajectories.append(steps)

        print("Estimate reward")
        irl = BayesianIRL(env)
        rewards = irl.estimate(trajectories)
        print(rewards)
        env.plot_on_grid(rewards)

    test_estimate()
