from noh.environment import UnsupervisedEnvironment
import numpy as np
from PIL import Image, ImageOps
import os


class MovingBall(UnsupervisedEnvironment):

    n_dataset = 17
    img_shape = (32, 32, 3)

    def __init__(self, model):
        super(MovingBall, self).__init__(model)
        self.dataset = np.load("moving_ball_dataset.npy")

    @classmethod
    def pic2dataset(cls):

        dataset = []
        for i in xrange(cls.n_dataset):
            filename = "./moving_ball_dataset/image"+str(i)+".png"
            img = np.array(ImageOps.grayscale(Image.open(filename))).flatten()
            dataset.append(img)
        dataset = np.array(dataset) / 255.
        np.save("moving_ball_dataset.npy", dataset)

    def rec_test(self, dir_name=""):

        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        for i, data in enumerate(self.dataset):
            img = self.model.rec(data) * 255
            img = np.array([[d, d, d] for d in img]).reshape(self.img_shape)
            img = Image.fromarray(np.uint8(img))
            img.save(dir_name+"/rec_image"+str(i)+".png")
            self.model.prop_up(data)

    def rec_test_without_input(self, dir_name=""):

        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        for i in xrange(self.n_dataset):
            img = self.model.prop_down() * 255
            img = np.array([[d, d, d] for d in img]).reshape(self.img_shape)
            img = Image.fromarray(np.uint8(img))
            img.save(dir_name + "/rec_image" + str(i) + ".png")
            self.model(None)
