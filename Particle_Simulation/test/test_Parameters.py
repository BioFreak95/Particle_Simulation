import numpy as np
import unittest
import numpy.testing as npt

from Particle_Simulation.Parameters import Parameters


class test_Parameters(unittest.TestCase):

    def test_3Dkvector(self):

        # setting up test parameters
        charges = np.ones(10).astype(np.float32)
        lj_sigmas = np.ones(10).astype(np.float32)
        lj_epsilons = np.ones(10).astype(np.float32)
        parameters = Parameters(temperature=1, box=np.array([1, 1, 1]), es_sigma=2, update_radius=0.5,
                                accuracy=0.5, charges=charges, lj_sigmas=lj_sigmas,
                                lj_epsilons=lj_epsilons, update_probability=0.5)

        reference = [[-2, 0, 0], [-1, -1, -1], [-1, -1, 0], [-1, -1, 1], [-1, 0, -1], [-1, 0, 0], [-1, 0, 1],
                     [-1, 1, -1], [-1, 1, 0], [-1, 1, 1], [0, -2, 0], [0, -1, -1], [0, -1, 0], [0, -1, 1], [0, 0, -2],
                     [0, 0, -1]]
        npt.assert_array_equal(parameters.k_vector, reference)

    def test_2Dkvector(self):

        # setting up test parameters
        charges = np.ones(10).astype(np.float32)
        lj_sigmas = np.ones(10).astype(np.float32)
        lj_epsilons = np.ones(10).astype(np.float32)
        parameters = Parameters(temperature=1, box=np.array([1, 1]), es_sigma=2, update_radius=0.5,
                                accuracy=0.5, charges=charges, lj_sigmas=lj_sigmas,
                                lj_epsilons=lj_epsilons, update_probability=0.5)

        reference = [[-2, 0], [-1, -1], [-1, 0], [-1, 1], [0, -2], [0, -1]]
        npt.assert_array_equal(parameters.k_vector, reference)

    def test_1Dkvector(self):

        # setting up test parameters
        charges = np.ones(10).astype(np.float32)
        lj_sigmas = np.ones(10).astype(np.float32)
        lj_epsilons = np.ones(10).astype(np.float32)
        parameters = Parameters(temperature=1, box=np.array([1]), es_sigma=2, update_radius=0.5,
                                accuracy=0.5, charges=charges, lj_sigmas=lj_sigmas,
                                lj_epsilons=lj_epsilons, update_probability=0.5)

        reference = [[-2], [-1]]
        npt.assert_array_equal(parameters.k_vector, reference)

    def test_negative_temperature(self):

        # setting up mock object
        mock = np.ones(10).astype(np.float32)
        self.assertRaises(ValueError, Parameters, -2, [1, 2, 3], 2, 2, np.array([0]), mock, mock, mock)

    def test_negative_box(self):

        # setting up mock object
        mock = np.ones(10).astype(np.float32)
        self.assertRaises(ValueError, Parameters, 2, [1, 2, -2], 2, 2, np.array([0]), mock, mock, mock)

    def test_negative_es_sigma(self):

        # setting up mock object
        mock = np.ones(10).astype(np.float32)
        self.assertRaises(ValueError, Parameters, 2, [1, 2, 3], -2, 2, np.array([0]), mock, mock, mock)

    def test_negative_update_radius(self):

        # setting up mock object
        mock = np.ones(10).astype(np.float32)
        self.assertRaises(ValueError, Parameters, 2, [1, 2, 3], 2, -2, np.array([0]), mock, mock, mock)
