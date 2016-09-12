import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

from noh.environments.moving_robot_environment import MovingRobot
from noh.components import RBM, PtReservoir, SimpleILRBM, HipModel

if __name__ == "__main__":

    MovingRobot.pic2dataset()

    """ using RBM """
    """
    rbm = RBM(n_visible=64*64*3, n_hidden=300, lr_type="hinton_r_div", r_div=100)
    env = MovingRobot(rbm)
    env.train(10)
    env.rec_test(dir_name="rbm_res/")
    """

    """ using pretrained reservoir """
    """
    pt_resv = PtReservoir(n_visible=64*64*3, n_hidden=300, lr_type="hinton_r_div", r_div=100)
    env = MovingRobot(pt_resv)
    env.train(2)
    env.rec_test(dir_name="pt_resv_res/")
    """

    """ using simple incremental learning RBM """
    ilrbm = SimpleILRBM(n_visible=64*64*3, n_hidden=20, lr_type="hinton_r_div", r_div=100)
    env = MovingRobot(ilrbm)
    env.train(1000)
    env.rec_test(dir_name="ilrbm_res0/")

    ilrbm.add_hidden_units(50)
    env.train(10000)
    env.rec_test(dir_name="ilrbm_res1/")


    """ using RBM and pretrained ESN"""
    """
    dg_model = rbm
    ca3_model = PtReservoir(n_visible=dg_model.n_hidden, n_hidden=30, lr_type="hinton_r_div", r_div=1000)
    hip_components = {"DG":dg_model, "CA3":ca3_model}
    rbm_ptresv = HipModel(components=hip_components)
    env = MovingRobot(rbm_ptresv)
    env.train(45)
    env.rec_test_without_input("rbm_ptresv/")
    """

    """ using Simple IL-RBM and pretrained ESN"""
    dg_model = ilrbm
    ca3_model = PtReservoir(n_visible=dg_model.n_hidden, n_hidden=500, lr_type="hinton_r_div", r_div=1000)
    hip_components = {"DG": dg_model, "CA3": ca3_model}
    ilrbm_ptresv = HipModel(components=hip_components)
    env = MovingRobot(ilrbm_ptresv)
    env.train(1000)
    env.rec_test_without_input("ilrbm_ptresv/")
