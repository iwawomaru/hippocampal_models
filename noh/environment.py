from abc import ABCMeta, abstractmethod

class Environment(object):

    __metaclass__ = ABCMeta

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def train(self):
        raise NotImplementedError("`train` must be explicitly overridden")

class ReinforcementEnvironment(Environment):

    __metaclass__ = ABCMeta

    def __init__(self, model):
        if not model.RL_trainable:
            raise ValueError("model should be RL trainable")
        super(ReinforcementEnvironment, self).__init__(model)

    def train(self):
        stat = self.get_stat()
        act = self.model(stat)
        self.set_act(act)
        reward = self.get_reward()
        self.model.set_reward()

    @abstractmethod
    def get_stat(self):
        """ Return a vector """
        raise NotImplementedError("`get_stat` must be explicitly overridden")

    @abstractmethod
    def set_act(self, act):
        """ set a vector """
        raise NotImplementedError("`set_act` must be explicitly overridden")

    @abstractmethod
    def get_reward(self):
        """ Return some scholar value """
        raise NotImplementedError("`get_reward` must be explicitly overridden")

class SupervisedEnvironment(Environment):

    dataset = None
    test_dataset = None

    def __init__(self, model):
        super(SupervisedEnvironment, self).__init__(model)

    def train(self, epochs):
        self.model.train(data=self.dataset[0], label=self.dataset[1], epochs=epochs)

    @classmethod
    def get_dataset(cls):
        return cls.dataset

    @classmethod
    def get_test_dataset(cls):
        return cls.test_dataset

class UnsupervisedEnvironment(Environment):

    dataset = None
    test_dataset = None

    def __init__(self, model):
        super(UnsupervisedEnvironment, self).__init__(model)

    def train(self, epochs):
        self.model.train(data=self.dataset, label=None, epochs=epochs)

    @classmethod
    def get_dataset(cls):
        return cls.dataset

    @classmethod
    def get_test_dataset(cls):
        return cls.test_dataset
