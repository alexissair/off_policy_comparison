import numpy as np


class Policy():
    def __init__(self, state_space=['start', 'end'], state_terminal = [False, True], action_space=['a', 'b'], alpha=0.1, epsilon=0.1, gamma=0.1, n=5, algorithm_used='tb', verbose=False):
        if isinstance(state_space, list):
            state_space = np.array(state_space)
        self.state_space = state_space
        self.state_count = self.state_space.shape[0]
        if isinstance(state_terminal, list):
            state_terminal = np.array(state_terminal)
        assert state_terminal.shape[0] == self.state_count
        self.state_terminal = state_terminal
        if isinstance(action_space, list):
            action_space = np.array(action_space)
        self.action_space = action_space
        self.action_count = self.action_space.shape[0]
        assert alpha > 0 and alpha <= 1
        self.alpha = alpha
        assert epsilon > 0
        self.epsilon = epsilon
        assert gamma > 0 and gamma <= 1
        self.gamma = gamma
        self.n = n
        self.algorithm_used = algorithm_used
        self.q = np.random.random((self.state_count, self.action_count))
        self.verbose = verbose
        self.operations = {
            'state': -1 * np.ones(shape=(n,), dtype='int32'),
            'action': -1 * np.ones(shape=(n,), dtype='int32'),
            'q': np.nan * np.ones(shape=(n,)),
            'delta': np.nan * np.ones(shape=(n,)),
            'c': np.nan * np.ones(shape=(n,)),
        }

    def get_operation(self, operation_string, index):
        res = self.operations[operation_string][index % self.n]
        if np.isnan(res) or (res.dtype == 'int32' and res == -1):
            raise Exception("{} at index {} is trying to be get but isn't set".format(operation_string, index))
        return res
    def set_operation(self, operation_string, index, value):
        self.operations[operation_string][index % self.n] = value

    def get_action_index_eps_greedy(self, state_index):
        temp = np.random.rand()
        if temp >= self.epsilon:
            action_index = np.argmax(self.q[state_index, :])
            return action_index
        return np.random.randint(0, self.action_count)

    def run_multiple_episode(self, number=100):
        for _ in range(number):
            self.run_episode(state_index=np.random.randint(0, self.state_count))

    def run_episode(self, state_index=None):
        self.init_episode(state_index=state_index)
        while self.tho < self.T - 1:
            if self.t < self.T:
                self.act_and_store()
            self.tho = self.t - self.n + 1
            if self.tho >= 0:
                self.update_q()
            self.t += 1

    def init_episode(self, state_index=None):
        if state_index is None:
            state_index = np.random.randint(0, self.state_count)
        self.set_operation('state', 0, state_index)
        action_index = self.get_action_index_eps_greedy(state_index)
        self.set_operation('action', 0, action_index)
        self.set_operation('q', 0, self.q[state_index, action_index])
        self.T = np.inf
        self.tho = -1
        self.t = 0

    def act_and_store(self):
        action_index = self.get_operation('action', self.t)
        reward, next_state_index = self.take_action(action_index)
        self.set_operation('state', self.t + 1, next_state_index)
        if self.is_terminal_state(next_state_index):
            self.handle_terminal_state(reward)
        else:
            self.handle_non_terminal_state(reward, next_state_index)

    def take_action(self, action_index):
        # TODO : implement this with respect to the environement
        next_state_index = action_index
        reward = -1 * action_index
        return reward, next_state_index

    def is_terminal_state(self, state_index):
        return self.state_terminal[state_index]

    def handle_terminal_state(self, reward):
        self.T = self.t + 1
        self.set_operation('delta', self.t, reward - self.get_operation('q', self.t))

    def handle_non_terminal_state(self, reward, next_state_index):
        delta = reward + self.gamma * self.get_q_expectation(next_state_index) - self.get_operation('q', self.t)
        self.set_operation('delta', self.t, delta)
        action_index = self.select_arbitrarly_action()
        self.set_operation('action', self.t + 1, action_index)
        self.set_operation('q', self.t + 1, self.q[next_state_index, action_index])
        c = self.get_c(next_state_index, action_index)
        self.set_operation('c', self.t + 1, c)

    def get_q_expectation(self, state_index):
        return self.epsilon * np.sum(self.q[state_index, :]) + (1 - self.epsilon) * np.max(self.q[state_index, :])

    def get_c(self, state_index, action_index):
        c = 1
        # TODO : c needs to be adapted to each algorithm (retrace, tree-backup...)
        if self.algorithm_used == 'tb':
            if np.max(self.q[state_index, :]) == self.q[state_index, action_index]:
                c = 1 - self.epsilon + self.epsilon / self.action_count
            else:
                c = self.epsilon / self.action_count
        return c

    def select_arbitrarly_action(self):
        return np.random.randint(0, self.action_count)

    def update_q(self):
        g = self.get_g()
        s_tho = self.get_operation('state', self.tho)
        a_tho = self.get_operation('action', self.tho)
        self.q[s_tho, a_tho] += self.alpha * (g - self.q[s_tho, a_tho])

    def get_g(self):
        z = 1
        g = self.get_operation('q', self.tho)
        temp = np.minimum(self.tho + self.n, self.T)
        for k in range(self.tho, int(temp)):
            g += z * self.get_operation('delta', k)
            try:
                z *= self.gamma * self.get_operation('c', k + 1)
            except Exception:
                break
        return g




    