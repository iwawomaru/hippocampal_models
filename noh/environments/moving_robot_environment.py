from noh.environment import UnsupervisedEnvironment
import numpy as np
from PIL import Image, ImageOps
import os


class MovingRobot(UnsupervisedEnvironment):

    n_dataset = 26
    img_shape = (64, 64, 3)

    def __init__(self, model):
        super(MovingRobot, self).__init__(model)
        self.dataset = np.load("moving_robot_dataset.npy")

    @classmethod
    def pic2dataset(cls):

        dataset = []
        for i in xrange(cls.n_dataset):
            filename = "./moving_robot_dataset/"+str(i+1)+".bmp"
            img = np.array((Image.open(filename))).flatten()
            dataset.append(img)
        dataset = np.array(dataset) / 255.
        np.save("moving_robot_dataset.npy", dataset)

    def rec_test(self, dir_name=""):

        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        for i, data in enumerate(self.dataset):
            img = self.model.rec(data).reshape(self.img_shape) * 255
            img = Image.fromarray(np.uint8(img))
            img.save(dir_name+"/rec_image"+str(i)+".bmp")
            self.model.prop_up(data)

    def rec_test_without_input(self, dir_name=""):

        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        for i in xrange(self.n_dataset):
            img = self.model.prop_down().reshape(self.img_shape) * 255
            img = Image.fromarray(np.uint8(img))
            img.save(dir_name + "/rec_image" + str(i) + ".bmp")
            self.model(None)
