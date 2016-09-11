import numpy as np

from noh.components import RBM

class SimpleILRBM(RBM):
    def __init__(self, n_visible, n_hidden, lr_type="hinton_r_div", r_div=None, lr=None):
        super(SimpleILRBM, self).__init__(n_visible, n_hidden, lr_type, r_div, lr)

    def add_hidden_units(self, n_add_hidden):
        self.n_hidden += n_add_hidden
        self.b_hidden = np.r_[self.b_hidden, np.zeros(n_add_hidden)]
        print self.W.shape, (np.random.rand(self.n_visible, n_add_hidden)/self.n_visible).shape
        self.W = np.c_[self.W, np.random.rand(self.n_visible, n_add_hidden)/self.n_visible]
