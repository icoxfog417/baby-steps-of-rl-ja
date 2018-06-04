import numpy as np
from scipy.misc import logsumexp
import tensorflow.contrib.distributions as tfd
import tensorflow.contrib.eager as tfe
from planner import ValuteIterationPlanner, PolicyIterationPlanner


tfe.enable_eager_execution()


class BayesianIRL():

    def __init__(self, env, prior_dist=None, eta=0.7):
        self.env = env
        self.planner = ValuteIterationPlanner(env)

        self.prior_dist = prior_dist
        self.eta = eta
        if self.prior_dist is None:
            self.prior_dist = tfd.Normal(loc=0.0, scale=0.5)

    def estimate(self, trajectories, max_iter=1000, reward_range=1.0,
                 gamma=0.9, delta=0.2):
        num_states = len(self.env.states)
        rewards = np.random.uniform(low=-reward_range, high=reward_range,
                                    size=num_states).astype("f")

        for i in range(max_iter):
            rewards = self.edit_rewards(rewards, delta, reward_range)
            # Make evaluation under estimated rewards
            self.planner.reward_func = lambda s: rewards[s]
            V = self.planner.plan(gamma)
            Q = self.planner.v_to_q(V, gamma)

            reward_probs = self.prior_dist.prob(rewards)
            reward_prior = np.sum(np.log(reward_probs.numpy()))
            likelihood = self.calculate_likelihood(trajectories, Q)

            posterior = likelihood + reward_prior
            posterior_prob = np.exp(posterior)

            if i % 100 == 0:
                print("At iteration {} prob={}".format(i, posterior_prob))

        return rewards

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

    def edit_rewards(self, rewards, delta, reward_range):
        noise = np.random.uniform(low=-delta, high=delta, size=1)[0]
        target = np.random.choice(range(len(rewards)), size=1)[0]
        if -reward_range <= (rewards[target] + noise) <= reward_range:
            rewards[target] += noise
        return rewards


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
