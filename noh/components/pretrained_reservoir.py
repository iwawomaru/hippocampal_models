import numpy as np

from noh.components import Layer
from noh.utils import get_lr_func
from noh.activate_functions import sigmoid, p_sig

class PtReservoir(Layer):
    def __init__(self, n_visible, n_hidden, lr_type="hinton_r_div", r_div=None, lr=None):
        super(PtReservoir, self).__init__(n_visible+n_hidden, n_hidden)
        self.lr_type=lr_type
        self.get_lr = get_lr_func(lr_type=lr_type, r_div=r_div, lr=lr)
        self.prev_hidden = np.zeros(shape=n_hidden)

    def train(self, data, label=None, lr=0.01, k=1, epochs=1):
        self.unsupervised_train(data,  k=k, epochs=epochs)

    def supervised_train(self, data, label=None, lr=0.01, k=1, epochs=1000):
        super(PtReservoir, self).train(data, label=None, lr=lr, k=k, epochs=epochs)

    def unsupervised_train(self, data, epochs=1000, k=1):
        for i in xrange(epochs):
            for d in data:
                self.CD(data=d)
                self.prop_up(d)
            print "epoch: ", i, self.get_rec_cross_entropy(data)

    def prop_up(self, v):
        input_data = np.atleast_2d(np.r_[v, self.prev_hidden])
        res = self.activate(np.dot(input_data, self.W) + self.b_hidden)
        self.prev_hidden = self.rng.binomial(size=res.shape, n=1, p=res)[0]
        return res

    def rec(self, v):
        input_data = np.atleast_2d(np.r_[v, self.prev_hidden])
        hid = self.activate(np.dot(input_data, self.W) + self.b_hidden)
        rec = self.prop_down(hid)
        return rec[0][0:self.n_visible-self.n_hidden]

    def CD(self, data):
        data = np.atleast_2d(np.r_[data, self.prev_hidden])

        h_mean = sigmoid(np.dot(data, self.W) + self.b_hidden)
        h_sample = self.rng.binomial(size=h_mean.shape, n=1, p=h_mean)

        nv_mean = sigmoid(np.dot(h_sample, self.W.T) + self.b_visible)
        nv_sample = self.rng.binomial(size=nv_mean.shape, n=1, p=nv_mean)

        nh_mean = sigmoid(np.dot(nv_sample, self.W) + self.b_hidden)
        nh_sample = self.rng.binomial(size=nh_mean.shape, n=1, p=nh_mean)

        dW = (np.dot(data.T, h_sample) - np.dot(nv_sample.T, nh_mean)) / data.shape[0]
        lr = self.get_lr(weight=self.W, d_weight=dW)
        # print "lr = ", lr

        self.W += lr * dW
        self.b_visible += lr * np.mean(data - nv_sample, axis=0) / data.shape[0]
        self.b_hidden += lr * np.mean(h_sample - nh_mean, axis=0) / data.shape[0]

    def get_rec_cross_entropy(self, v):
        error = 0
        for d in v:
            rec_v = np.atleast_2d(self.rec(d))
            error -= np.mean(np.sum(d * np.log(rec_v) + (1 - d) * np.log(1 - rec_v), axis=1))
        return error

    def get_energy(self, data):
        """ Return Scala Value """
        hid = self.prop_up(data)
        eng = - np.dot(self.b_visible, data.T).sum(axis=0) - \
            np.dot(np.dot(data, self.W).T, hid).sum(axis=0) - \
            np.dot(hid, self.b_hidden).sum(axis=0)

        return eng.mean()
