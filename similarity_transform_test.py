import unittest
import similarity_transform as sim
import numpy as np
from scipy.spatial.transform import Rotation as scipy_rot


class TestSimilarityTransform(unittest.TestCase):
    def test_random_values(self):
        X = np.random.randn(3, 8)
        R, c, t = sim.similarity_transform(X, X)

        T = sim.construct_transformation_matrix(R, c, t)
        X_prime = sim.apply_transformation(X, T)
        np.testing.assert_allclose(X, X_prime, atol=1e-6)

    def test_reference_transformation(self):
        R_ref = scipy_rot.from_euler("z", 90, degrees=True).as_matrix()
        c_ref = 30
        t_ref = np.array([10, 10, 10])
        # print(
        #     "Reference: Rotation vector\n",
        #     R_ref,
        #     "\nscale\n",
        #     c_ref,
        #     "\ntranslation vector\n",
        #     t_ref,
        # )

        X = np.random.randn(3, 8)
        Y = R_ref @ (c_ref * X) + t_ref[:, np.newaxis].repeat(8, axis=1)
        R, c, t = sim.similarity_transform(X, Y)
        # print(
        #     "Estimate: Rotation vector\n",
        #     R,
        #     "\nscale\n",
        #     c,
        #     "\ntranslation vector\n",
        #     t,
        # )

        T = sim.construct_transformation_matrix(R, c, t)
        Y_prime = sim.apply_transformation(X, T)
        np.testing.assert_allclose(R_ref, R, atol=1e-6)
        np.testing.assert_allclose(c_ref, c, atol=1e-6)
        np.testing.assert_allclose(t_ref, t, atol=1e-6)
        np.testing.assert_allclose(Y, Y_prime, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
