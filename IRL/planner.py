import numpy as np


class Planner():

    def __init__(self, env, reward_func=None):
        self.env = env
        self.reward_func = reward_func
        if self.reward_func is None:
            self.reward_func = self.env.reward_func

    def initialize(self):
        self.env.reset()

    def transitions_at(self, state, action):
        reward = self.reward_func(state)
        done = self.env.has_done(state)
        transition = []
        if not done:
            transition_probs = self.env.transit_func(state, action)
            for next_state in transition_probs:
                prob = transition_probs[next_state]
                reward = self.reward_func(next_state)
                done = self.env.has_done(state)
                transition.append((prob, next_state, reward, done))
        else:
            transition.append((1.0, None, reward, done))
        for p, n_s, r, d in transition:
            yield p, n_s, r, d

    def plan(self, gamma=0.9, threshold=0.0001):
        raise Exception("Planner have to implements plan method.")


class ValuteIterationPlanner(Planner):

    def __init__(self, env):
        super().__init__(env)

    def plan(self, gamma=0.9, threshold=0.0001):
        self.initialize()
        V = np.zeros(len(self.env.states))
        while True:
            delta = 0
            for s in self.env.states:
                expected_rewards = []
                for a in self.env.actions:
                    reward = 0
                    for p, n_s, r, done in self.transitions_at(s, a):
                        if n_s is None:
                            reward = r
                            continue
                        reward += p * (r + gamma * V[n_s] * (not done))
                    expected_rewards.append(reward)
                max_reward = max(expected_rewards)
                delta = max(delta, abs(max_reward - V[s]))
                V[s] = max_reward

            if delta < threshold:
                break

        return V


class PolicyIterationPlanner(Planner):

    def __init__(self, env):
        super().__init__(env)
        self.policy = None
        self._limit_count = 1000

    def initialize(self):
        super().initialize()
        self.policy = np.ones((self.env.observation_space.n,
                               self.env.action_space.n))
        # First, take each action uniformly.
        self.polidy = self.policy / self.env.action_space.n

    def policy_to_q(self, V, gamma):
        Q = np.zeros((self.env.observation_space.n,
                      self.env.action_space.n))

        for s in self.env.states:
            for a in self.env.actions:
                a_p = self.policy[s][a]
                for p, n_s, r, done in self.transitions_at(s, a):
                    if done:
                        Q[s][a] += p * a_p * r
                    else:
                        Q[s][a] += p * a_p * (r + gamma * V[n_s])
        return Q

    def estimate_by_policy(self, gamma, threshold):
        V = np.zeros(self.env.observation_space.n)

        count = 0
        while True:
            delta = 0
            for s in self.env.states:
                expected_rewards = []
                for a in self.env.actions:
                    action_prob = self.policy[s][a]
                    reward = 0
                    for p, n_s, r, done in self.transitions_at(s, a):
                        if n_s is None:
                            reward = r
                            continue
                        reward += action_prob * p * \
                                  (r + gamma * V[n_s] * (not done))
                    expected_rewards.append(reward)
                max_reward = max(expected_rewards)
                delta = max(delta, abs(max_reward - V[s]))
                V[s] = max_reward

            if delta < threshold or count > self._limit_count:
                break
            count += 1

        return V

    def act(self, s):
        return np.argmax(self.policy[s])

    def plan(self, gamma=0.9, threshold=0.0001, keep_policy=False):
        if not keep_policy:
            self.initialize()

        count = 0
        while True:
            update_stable = True
            # Estimate expected reward under current policy.
            V = self.estimate_by_policy(gamma, threshold)

            for s in self.env.states:
                # Get action following to the policy (choose max prob's action).
                policy_action = self.act(s)

                # Compare with other actions.
                action_rewards = np.zeros(len(self.env.actions))
                for a in self.env.actions:
                    reward = 0
                    for p, n_s, r, done in self.transitions_at(s, a):
                        if n_s is None:
                            reward = r
                            continue
                        reward += p * (r + gamma * V[n_s] * (not done))
                    action_rewards[a] = reward
                best_action = np.argmax(action_rewards)
                if policy_action != best_action:
                    update_stable = False

                # Update policy (set best_action prob=1, otherwise=0 (greedy)).
                self.policy[s] = np.zeros(len(self.env.actions))
                self.policy[s][best_action] = 1.0

            if update_stable or count > self._limit_count:
                # If policy isn't updated, stop iteration.
                break
            count += 1

        return V


if __name__ == "__main__":
    def test_plan():
        from environment import GridWorldEnv
        env = GridWorldEnv(grid=[
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 0],
        ])
        print("Value Iteration.")
        vp = ValuteIterationPlanner(env)
        v = vp.plan()
        print(v.reshape(env.shape))

        print("Policy Iteration.")
        pp = PolicyIterationPlanner(env)
        v = pp.plan()
        print(v.reshape(env.shape))
        q = pp.policy_to_q(v, 0.9)
        print(np.sum(q, axis=1).reshape(env.shape))

    test_plan()
