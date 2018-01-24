from Particle_Simulation.Parameters import Parameters
import numpy as np
import unittest


class test_Parameters(unittest.TestCase):
    def test_3Dkvector(self):
        Parameter = Parameters(temperature=1, box=[1, 1, 1], es_sigma=2, update_radius=0.5, cutoff_radius=1, K_cutoff=2,
                               particle_types=np.array([0]))
        reference = [[-2, 0, 0], [-1, -1, -1], [-1, -1, 0], [-1, -1, 1], [-1, 0, -1], [-1, 0, 0], [-1, 0, 1],
                     [-1, 1, -1], [-1, 1, 0], [-1, 1, 1], [0, -2, 0], [0, -1, -1], [0, -1, 0], [0, -1, 1], [0, 0, -2],
                     [0, 0, -1]]
        self.assertEqual(Parameter.k_vector, reference)

    def test_2Dkvector(self):
        Parameter = Parameters(temperature=1, box=[1, 1, 1], es_sigma=2, update_radius=0.5, cutoff_radius=1, K_cutoff=2,
                               particle_types=np.array([0]))
        reference = [[-2, 0], [-1, -1], [-1, 0], [-1, 1], [0, -2], [0, -1]]
        self.assertEqual(Parameter.k_vector, reference)

    def test_1Dkvector(self):
        Parameter = Parameters(temperature=1, box=[1], es_sigma=2, update_radius=0.5, cutoff_radius=1, K_cutoff=2,
                               particle_types=np.array([0]))
        reference = [[-2], [-1]]
        self.assertEqual(Parameter.k_vector, reference)

    def test_negative_temperature(self):
        self.assertRaises(ValueError, Parameters, -2, [1, 2, 3], 2, 2, np.array([0]), 2, 2)

    def test_negative_box(self):
        self.assertRaises(ValueError, Parameters, 2, [1, 2, -2], 2, 2, np.array([0]), 2, 2)

    def test_negative_es_sigma(self):
        self.assertRaises(ValueError, Parameters, 2, [1, 2, 3], -2, 2, np.array([0]), 2, 2)

    def test_negative_update_radius(self):
        self.assertRaises(ValueError, Parameters, 2, [1, 2, 3], 2, -2, np.array([0]), 2, 2)

    def test_negative_cutoff_radius(self):
        self.assertRaises(ValueError, Parameters, 2, [1, 2, 3], 2, 2, np.array([0]), -2, 2)

    def test_negative_K_cutoff(self):
        self.assertRaises(ValueError, Parameters, 2, [1, 2, 3], 2, 2, np.array([0]), 2, -2)

    def test_wrong_particle_type(self):
        self.assertRaises(TypeError, Parameters, 2, [1, 2, 3], 2, 2, [0], 2, 2)
