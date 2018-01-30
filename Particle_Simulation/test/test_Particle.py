from Particle_Simulation.Particle import Particle
import numpy as np
import unittest


class test_Particle(unittest.TestCase):
    def test_wrong_typeIndex(self):
        self.assertRaises(TypeError, Particle, 2.3, np.array([1, 1, 1]))

    def test_wrong_position(self):
        self.assertRaises(TypeError, Particle, 1, 23)
