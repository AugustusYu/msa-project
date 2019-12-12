import numpy as np


class CongestionGame():
    def __init__(self, alpha):
        self.alpha = float(alpha)

    def __call__(self, actions):
        n0 = np.sum(actions == 0)
        n1 = np.sum(actions == 1)
        c0 = n0 / self.alpha + 1
        c1 = n1 / self.alpha + 1
        return np.array([c0, c1])


class CongestionGameHighway():
    def __init__(self, alpha):
        self.alpha = float(alpha)

    def __call__(self, actions):
        n0 = np.sum(actions == 0)
        n1 = np.sum(actions == 1)
        n2 = np.sum(actions == 2)
        n3 = np.sum(actions == 3)
        c0 = (n0 + n2) / self.alpha + 1
        c1 = (n1 + n2) / self.alpha + 1
        c2 = (n0 + n1 + n2 + n2) / self.alpha
        c3 = 2
        return np.array([c0, c1, c2, c3])


class AgentEpsilonGreedy():
    def __init__(self, n_agents, n_actions, beta, epsilon):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.cost_matrix = np.random.rand(n_agents, n_actions)
        self.beta = beta
        self.epsilon = epsilon

    def play(self, game):
        actions = self.get_actions()
        costs = game(actions)
        self.update(actions, costs)
        # get current best actions and costs
        current_actions = self.cost_matrix.argmin(-1)
        current_costs = game(current_actions)
        # compute total cost
        average_cost = 0
        for i in range(self.n_actions):
            average_cost += current_costs[i] * np.sum(current_actions==i)
        average_cost /= self.n_agents
        # compute the count for each action
        counts = []
        for i in range(self.n_actions):
            counts.append(np.sum(current_actions==i))
        return average_cost, counts

    def get_actions(self):
        actions = self.cost_matrix.argmin(-1)
        random_actions = np.random.randint(0, self.n_actions, size=(self.n_agents,))
        random_idx = np.random.rand(self.n_agents) < self.epsilon
        actions[random_idx] = random_actions[random_idx]
        return actions

    def update(self, actions, costs):
        self.cost_matrix[np.arange(self.n_agents), actions] = self.beta * self.cost_matrix[np.arange(self.n_agents), actions] + (1-self.beta) * costs[actions]


class AgentUCB1():
    def __init__(self, n_agents, n_actions, beta):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.cost_matrix = np.random.rand(n_agents, n_actions)
        self.action_counts = np.ones([n_agents, n_actions])
        self.current_step = 1
        self.beta = beta

    def play(self, game):
        actions = self.get_actions()
        costs = game(actions)
        self.update(actions, costs)
        # get current best actions and costs
        current_actions = self.cost_matrix.argmin(-1)
        current_costs = game(current_actions)
        # compute total cost
        average_cost = 0
        for i in range(self.n_actions):
            average_cost += current_costs[i] * np.sum(current_actions==i)
        average_cost /= self.n_agents
        # compute the count for each action
        counts = []
        for i in range(self.n_actions):
            counts.append(np.sum(current_actions==i))
        return average_cost, counts

    def get_actions(self):
        actions = (self.cost_matrix-np.sqrt(2*np.log(self.current_step)/self.action_counts)).argmin(-1)
        self.current_step += 1
        self.action_counts[np.arange(self.n_agents), actions] += 1
        return actions

    def update(self, actions, costs):
        self.cost_matrix[np.arange(self.n_agents), actions] = self.beta * self.cost_matrix[np.arange(self.n_agents), actions] + (1-self.beta) * costs[actions]


class AgentThompson():
    # Reference: https://towardsdatascience.com/solving-multiarmed-bandits-a-comparison-of-epsilon-greedy-and-thompson-sampling-d97167ca9a50

    def __init__(self, n_agents, n_actions):
        # assume cost is in [0, 2]
        self.cost_bound = 2
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.a = np.ones([n_agents, n_actions])
        self.b = np.ones([n_agents, n_actions])

    def play(self, game):
        actions = self.get_actions()
        costs = game(actions)
        self.update(actions, costs)
        # get current best actions and costs
        # take the mean of beta distributions
        cost_means = self.a / (self.a + self.b)
        current_actions = cost_means.argmin(-1)
        current_costs = game(current_actions)
        # compute total cost
        average_cost = 0
        for i in range(self.n_actions):
            average_cost += current_costs[i] * np.sum(current_actions==i)
        average_cost /= self.n_agents
        # compute the count for each action
        counts = []
        for i in range(self.n_actions):
            counts.append(np.sum(current_actions==i))
        return average_cost, counts

    def get_actions(self):
        sampled_costs = [np.random.beta(a, b) for a, b in zip(self.a.reshape(-1), self.b.reshape(-2))]
        sampled_costs = np.array(sampled_costs).reshape(self.n_agents, self.n_actions)
        actions = sampled_costs.argmin(-1)
        return actions

    def update(self, actions, costs):
        costs = costs / self.cost_bound # normalize to [0, 1]
        costs = costs[actions]
        self.a[np.arange(self.n_agents), actions] += costs
        self.b[np.arange(self.n_agents), actions] += 1 - costs


if __name__ == '__main__':
    game = CongestionGameHighway(alpha=100)
    # agent = AgentEpsilonGreedy(100, 4, beta=0.8, epsilon=0.1)
    # agent = AgentUCB1(100, 4, beta=0.8)
    agent = AgentThompson(100, 4)
    for i in range(100):
        print(agent.play(game))