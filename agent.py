import numpy as np
from gym.spaces.discrete import Discrete

class Agent():
    def __init__(self, env=None, state_space=['start', 'end'], state_terminal = [False, True], action_space=['a', 'b'], alpha=0.01, epsilon=0.5, gamma=0.8, n=100, algorithm_used='tb', verbose=False):
        if env is not None:
            self.init_with_env(env)
        else:
            self.init_with_spaces(state_space, state_terminal, action_space)
        assert alpha > 0 and alpha <= 1
        self.alpha = alpha
        assert epsilon > 0
        self.epsilon = epsilon
        assert gamma > 0 and gamma <= 1
        self.gamma = gamma
        self.n = n
        self.algorithm_used = algorithm_used
        self.q = 10e-3 * np.random.rand(self.state_count, self.action_count)
        self.verbose = verbose
        self.operations = {
            'state': -1 * np.ones(shape=(n,), dtype='int64'),
            'action': -1 * np.ones(shape=(n,), dtype='int64'),
            'q': np.nan * np.ones(shape=(n,)),
            'delta': np.nan * np.ones(shape=(n,)),
            'c': np.nan * np.ones(shape=(n,)),
        }

    def init_with_env(self, env):
        if type(env.action_space) != Discrete or type(env.observation_space) != Discrete:
            raise Exception("Non discrete spaces aren't implemented yet! Choose a discrete environement please.")
        self.env = env
        self.state_count = env.observation_space.n
        self.state_space = np.array([i for i in range(self.state_count)])
        self.state_terminal = np.array([False for _ in range(env.observation_space.n)])
        self.action_count = env.action_space.n
        self.action_space = np.array([i for i in range(self.action_count)])
        self.is_init_with_env = True

    def init_with_spaces(self, state_space, state_terminal, action_space):
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
        self.is_init_with_env = False
        
    def get_operation(self, operation_string, index):
        res = self.operations[operation_string][index % self.n]
        if np.isnan(res) or (res.dtype == 'int64' and res == -1):
            raise Exception("{} at index {} is trying to be get but isn't set".format(operation_string, index))
        return res
    def set_operation(self, operation_string, index, value):
        self.operations[operation_string][index % self.n] = value

    def get_action_index_with_q_policy(self, state_index, eps_greedy=True):
        if eps_greedy:
            temp = np.random.rand()
            if temp < self.epsilon :
                return np.random.randint(0, self.action_count)
        return np.argmax(self.q[state_index, :])

    def run_multiple_episode(self, number=100):
        self.n_episode = 0
        while self.n_episode <= number:
            self.n_episode += 1
            self.epsilon = np.maximum(0.01, self.epsilon / (self.n_episode)**(0.1))
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
        self.operations = {
            'state': -1 * np.ones(shape=(self.n,), dtype='int64'),
            'action': -1 * np.ones(shape=(self.n,), dtype='int64'),
            'q': np.nan * np.ones(shape=(self.n,)),
            'delta': np.nan * np.ones(shape=(self.n,)),
            'c': np.nan * np.ones(shape=(self.n,)),
        }
        state_index = self.init_state_index(state_index=state_index)
        self.set_operation('state', 0, state_index)
        action_index = self.get_action_index_with_q_policy(state_index,eps_greedy=True)
        self.set_operation('action', 0, action_index)
        self.set_operation('q', 0, self.q[state_index, action_index])
        self.T = np.inf
        self.tho = -1
        self.t = 0

    def init_state_index(self, state_index=None):
        if self.is_init_with_env:
            state_index = self.env.reset()
        if state_index is None:
            state_index = np.random.randint(0, self.state_count)
        return state_index

    def act_and_store(self):
        action_index = self.get_operation('action', self.t)
        reward, next_state_index, is_terminal = self.take_action(action_index)
        self.set_operation('state', self.t + 1, next_state_index)
        if is_terminal or self.is_terminal_state(next_state_index):
            self.handle_terminal_state(reward)
        else:
            self.handle_non_terminal_state(reward, next_state_index)

    def take_action(self, action_index):
        if self.is_init_with_env:
            next_state_index, reward, is_terminal, _ = self.env.step(self.action_space[action_index])
            # next_state_index = np.argwhere(self.state_space==next_state)[0]
        else:
            next_state_index = action_index
            reward = -1 * action_index
            is_terminal = None
        return reward, next_state_index, is_terminal

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

    def play_episode(self, state_index=None, verbose=False, render=False):
        state_index = self.init_state_index(state_index=state_index)
        is_terminal_state = False
        total_reward = 0
        while not is_terminal_state:
            self.render(render)
            action_index = self.get_action_index_with_q_policy(state_index=state_index, eps_greedy=False)
            if verbose:
                print("Current state index: {}".format(state_index))
                print("Action index: {}".format(action_index))
            reward, state_index, is_terminal_state = self.take_action(action_index)
            if verbose:
                print("Reward: {}".format(reward))
                print("Next state index: {}".format(state_index))
                print("Is terminal state: {}".format(is_terminal_state))
            total_reward += reward
        self.render(render)
        return total_reward

    def render(self, render):
        if render:
            try:
                self.env.render()
            except Exception:
                pass
    