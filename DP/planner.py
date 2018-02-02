class Planner():

    def __init__(self, env):
        self.env = env

    def transitions(self, state, action):
        for a, prob in zip(self.env.action_space,
                           self.env.get_action_probs(action)):
            next_state, reward, done = self.env.transit(state, a)
            if next_state is None:
                continue
            yield prob, next_state, reward

    def value_iteration(self, gamma=0.9, threshold=0.0001):
        actions = self.env.action_space
        V = {}
        for s in self.env.states:
            # Initialize each state's reward
            V[s] = 0

        while True:
            delta = 0
            for s in V:
                expected_rewards = []
                for a in actions:
                    r = 0
                    for prob, next_state, reward in self.transitions(s, a):
                        r += prob * (reward + gamma * V[next_state])
                    expected_rewards.append(r)
                best_action_reward = max(expected_rewards)
                delta = max(delta, abs(best_action_reward - V[s]))
                V[s] = best_action_reward
            if delta < threshold:
                break

        # Turn dictionary to grid
        V_grid = self.dict_to_grid(V)
        return V_grid

    def dict_to_grid(self, state_reward_dict):
        grid = []
        for i in range(self.env.row_length):
            row = [0] * self.env.column_length
            grid.append(row)
        for s in state_reward_dict:
            grid[s.row][s.column] = state_reward_dict[s]
        
        return grid
