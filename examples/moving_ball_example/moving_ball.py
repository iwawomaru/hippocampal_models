import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

from noh.environments.moving_ball_environment import MovingBall
from noh.components import RBM, PtReservoir, SimpleILRBM, HipModel

if __name__ == "__main__":

    print os.getcwd()
    MovingBall.pic2dataset()

    """ using RBM """

    rbm = RBM(n_visible=32*32, n_hidden=100, lr_type="hinton_r_div", r_div=100)
    env = MovingBall(rbm)
    env.train(1000)
    env.rec_test(dir_name="rbm_res/")


    """ using pretrained reservoir """
    pt_resv = PtReservoir(n_visible=32*32, n_hidden=30, lr_type="hinton_r_div", r_div=100)
    env = MovingBall(pt_resv)
    env.train(150)
    env.rec_test(dir_name="pt_resv_res/")

    """ using simple incremental learning RBM """

    ilrbm = SimpleILRBM(n_visible=32*32, n_hidden=5, lr_type="hinton_r_div", r_div=100)
    env = MovingBall(ilrbm)
    env.train(500)
    env.rec_test(dir_name="ilrbm_res0/")

    ilrbm.add_hidden_units(10)
    env.train(500)
    env.rec_test(dir_name="ilrbm_res1/")

    ilrbm.add_hidden_units(20)
    env.train(500)
    env.rec_test(dir_name="ilrbm_res2/")


    """ using RBM and pretrained ESN"""
    dg_model = rbm
    ca3_model = PtReservoir(n_visible=dg_model.n_hidden, n_hidden=30, lr_type="hinton_r_div", r_div=1000)
    hip_components = {"DG":dg_model, "CA3":ca3_model}
    rbm_ptresv = HipModel(components=hip_components)
    env = MovingBall(rbm_ptresv)
    env.train(4500)
    env.rec_test_without_input("rbm_ptresv/")

    """ using Simple IL-RBM and pretrained ESN"""
    dg_model = ilrbm
    ca3_model = PtReservoir(n_visible=dg_model.n_hidden, n_hidden=20, lr_type="hinton_r_div", r_div=1000)
    hip_components = {"DG": dg_model, "CA3": ca3_model}

    ilrbm_ptresv = HipModel(components=hip_components)
    env = MovingBall(ilrbm_ptresv)
    env.train(6000)
    env.rec_test_without_input("ilrbm_ptresv/")
