import unittest
import numpy as np

from rl.tools.function_approximators.normalizers import tf2NormalizerClip
from rl.tools.function_approximators.supervised_learners import SuperRobustKerasMLP

def assert_array(a,b):
    assert np.all(np.isclose(a-b,0.0, atol=1e-8))


class Tests(unittest.TestCase):

    def test_tf2_supervised_learner(self):

        cls = SuperRobustKerasMLP
        units = (1000, 400, 200, 100, 50)

        x_shape = (2,)
        y_shape = (1,)
        build_y_nor = lambda : tf2NormalizerClip(shape=y_shape, clip_thre=3.)
        fun = cls(x_shape, y_shape, units=units, build_y_nor=build_y_nor)

        xs = np.random.random([100]+list(x_shape))
        ys = np.sum(xs**2,axis=1, keepdims=True)+1
       
        ys_bad = ys.copy()
        ys_bad[0]+=1e10
        results = fun.update(xs, ys_bad, clip_y=True, epochs=10)
        fs = fun.predict(xs)
        print('mae', np.abs(fs-ys).mean())
        print('mse', np.square(fs-ys).mean())






if __name__=='__main__':
    unittest.main()



