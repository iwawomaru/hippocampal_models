from noh import TrainRule, PropRule, Planner, Circuit
import numpy as np


class HipMemProp(PropRule):
    def __init__(self, components):
        super(HipMemProp, self).__init__(components)

    def __call__(self, data=None):
        if data is None:
            data = np.zeros(self.DG.n_hidden)
        else:
            data = self.DG(data)
        data = self.CA3(data)
        return data


class HipTrain(TrainRule):
    def __init__(self, components):
        super(HipTrain, self).__init__(components)

    def __call__(self, data, label=None, epoch=1):
        data = self.DG(data)
        data = self.DG.rng.binomial(size=data.shape, n=1, p=data)
        print data
        self.CA3.train(data, epochs=epoch)


class HipPlanner(Planner):
    def __init__(self, components):
        super(HipPlanner, self).__init__(prop=HipMemProp, train=HipTrain, components=components)

    def prop_up(self, data):
        return self(data)


class HipModel(Circuit):
    def __init__(self, components):
        super(HipModel, self).__init__(planner=HipPlanner, components=components)

    def prop_down(self, data=None):
        if data is None:
            data = np.atleast_2d(self.CA3.prev_hidden)
        data = self.CA3.prop_down(data)[0][0:self.CA3.n_visible - self.CA3.n_hidden]
        data = self.DG.prop_down(data)
        return data