import numpy as np


class Policy():
    def __init__(self, state_space=[0, 1], action_space=[0, 1], alpha=0.1, epsilon=0.1, verbose=False):
        if isinstance(state_space, list):
            state_space = np.array(state_space)
        self.state_space = state_space
        self.state_count = self.state_space.shape[0]
        if isinstance(action_space, list):
            action_space = np.array(action_space)
        self.action_space = action_space
        self.action_count = self.action_space.shape[0]
        assert alpha > 0 and alpha <= 1
        self.alpha = alpha
        assert epsilon > 0
        self.epsilon = epsilon
        self.Q = np.random.random((self.state_count, self.action_count))
        self.verbose = verbose

    def get_action_eps_greedy(self, state_index):
        temp = np.random.rand()
        if temp >= self.epsilon:
            action_index = np.argmax(self.Q[state_index, :])
            if self.verbose:
                print('max Q')
            return self.action_space[action_index]
        if self.verbose:
            print('random')
        return self.action_space[np.random.randint(0, self.action_count)]

    